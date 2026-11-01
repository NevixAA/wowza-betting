"""Player-prop settlement grades what happened, never a guess (Data Truth audit 16, 19, 23, 25).

    python -m pytest tests/test_player_settlement_truth.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import update_results as U


@pytest.mark.parametrize("market,stats,want", [
    ("sot", {"shots_on_target": 1}, "WIN"),
    ("sot2", {"shots_on_target": 1}, "LOSS"),
    ("sot2", {"shots_on_target": 2}, "WIN"),
    ("sot3", {"shots_on_target": 2}, "LOSS"),
    ("goals2", {"goals": 2}, "WIN"),
    ("cards", {"yellow_card": 0}, "LOSS"),
    ("assists", {"assists": 1}, "WIN"),
    ("corners", {"corners": 3}, ""),
    # A stat the provider did not publish is unknown, never 0 -> LOSS.
    ("sot", {"shots_on_target": None}, ""),
    ("goals", {}, ""),
])
def test_threshold_markets_settle(market, stats, want):
    assert U._resolve_player_market(market, stats) == want


def _ledger(tmp_path, monkeypatch, days_ago, league="Premier League", player="Bukayo Saka",
            player_id="7", fixture_id="", market="sot2"):
    d = (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()
    p = tmp_path / "player_ledger.csv"
    pd.DataFrame([{"match_date": d, "league": league, "home_team": "Arsenal", "away_team": "Spurs",
                   "player_name": player, "market": market, "market_odds": "2.5", "result": "",
                   "pnl": "", "resolved_date": "", "player_id": player_id,
                   "fixture_id": fixture_id}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "PLAYER_LEDGER_FILE", p)
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    return p


def _pl(pid, name="P", team=1, minutes=90, sub=False, sot=0, goals=0, assists=0, yellow=0):
    """One /fixtures/players entry, exactly as API-Football shapes it (None stays None)."""
    return {"_team": team, "player": {"id": pid, "name": name},
            "statistics": [{"games": {"minutes": minutes, "substitute": sub},
                            "goals": {"total": goals, "assists": assists},
                            "shots": {"on": sot}, "cards": {"yellow": yellow}}]}


FIX = {"fixture": {"id": 1, "status": {"short": "FT"}},
       "teams": {"home": {"id": 1, "name": "Arsenal"}, "away": {"id": 2, "name": "Spurs"}},
       "goals": {"home": 2, "away": 1}}


def _api(monkeypatch, players, seen=None, calls=None, day=None, status="FT", events=None,
         team_stats=None):
    """Fake API-Football behind update_results._af_get. `players` is a list of _pl() entries;
    the two clubs are published only when at least one player of that club is listed."""
    fx = {**FIX, "fixture": {"id": 1, "status": {"short": status}}}

    def fake(endpoint, params, cache_hours):
        if calls is not None:
            calls.append((endpoint, dict(params)))
        if endpoint == "/fixtures" and "date" in params:
            if seen is not None:
                seen.append(params["season"])
            return {"response": [fx] if day is None else day}
        if endpoint == "/fixtures" and "id" in params:
            return {"response": [fx] if params["id"] == 1 else []}
        if endpoint == "/fixtures/players":
            blocks = {}
            for p in players:
                blocks.setdefault(p["_team"], []).append(
                    {k: v for k, v in p.items() if k != "_team"})
            return {"response": [{"team": {"id": t}, "players": ps} for t, ps in blocks.items()]}
        if endpoint == "/fixtures/events":
            return {"response": events or []}
        if endpoint == "/fixtures/statistics":
            return {"response": [{"team": {"id": t}, "statistics": [{"type": "Shots on Goal", "value": v}]}
                                 for t, v in (team_stats or {}).items()]}
        raise AssertionError(f"unexpected endpoint {endpoint}")
    monkeypatch.setattr(U, "_af_get", fake)


def _row(p):
    return pd.read_csv(p, dtype=str).iloc[0]


def test_no_stats_yet_stays_pending(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].fillna("").iloc[0] == ""


def test_no_stats_after_a_week_is_void_with_reason(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=8)
    _api(monkeypatch, [])
    U.update_player_results()
    row = _row(p)
    assert row["result"] == "VOID" and row["notes"] == "NO_STATS_7D"


def test_absent_from_both_published_squads_is_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(41, "Declan Rice", team=1), _pl(51, "Son", team=2)])
    U.update_player_results()
    row = _row(p)
    assert row["result"] == "VOID" and row["notes"] == "DNP (not in matchday squad)"


def test_absent_when_only_one_club_published_stays_pending(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(41, "Declan Rice", team=1)])
    U.update_player_results()
    assert pd.isna(_row(p)["result"])


def test_player_id_wins_over_names(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, player="Armand Laurienté", player_id="777")
    _api(monkeypatch, [_pl(777, "Someone Else", sot=3), _pl(51, "Son", team=2)])
    U.update_player_results()
    assert _row(p)["result"] == "WIN"


def test_no_player_id_is_never_matched_by_name(tmp_path, monkeypatch):
    # The old fallbacks (exact name, then first initial + surname) are gone: only the id identifies.
    p = _ledger(tmp_path, monkeypatch, days_ago=2, player_id="")
    _api(monkeypatch, [_pl(99, "Bukayo Saka", sot=3), _pl(51, "Son", team=2)])
    U.update_player_results()
    assert pd.isna(_row(p)["result"])


def test_threshold_prop_settles(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, "Bukayo Saka", sot=3), _pl(51, "Son", team=2)])
    U.update_player_results()
    assert _row(p)["result"] == "WIN"


def test_zero_minutes_is_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, minutes=0, sub=True), _pl(51, team=2)])
    U.update_player_results()
    assert _row(p)["notes"] == "DNP (0 min)"


def test_unused_substitute_is_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, minutes=None, sub=True, sot=None), _pl(51, team=2)])
    U.update_player_results()
    row = _row(p)
    assert row["result"] == "VOID" and row["notes"] == "DNP (unused substitute)"


def test_null_minutes_for_a_starter_is_not_dnp(tmp_path, monkeypatch):
    # minutes null was read as 0 and voided as 'DNP (0 min)'; a null is not evidence.
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, minutes=None, sub=False), _pl(51, team=2)])
    U.update_player_results()
    assert pd.isna(_row(p)["result"])


def test_null_shots_are_zero_only_when_the_team_total_proves_it(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, sot=None), _pl(8, sot=4), _pl(51, team=2)], team_stats={1: 4})
    U.update_player_results()
    assert _row(p)["result"] == "LOSS"


def test_null_shots_stay_pending_when_the_team_total_leaves_room(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, sot=None), _pl(8, sot=4), _pl(51, team=2)], team_stats={1: 6})
    U.update_player_results()
    assert pd.isna(_row(p)["result"])


def test_null_goals_come_from_complete_goal_events(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, market="goals")
    ev = [{"type": "Goal", "detail": "Normal Goal", "player": {"id": 7}, "assist": {"id": 8}},
          {"type": "Goal", "detail": "Penalty", "player": {"id": 8}, "assist": {"id": None}},
          {"type": "Goal", "detail": "Normal Goal", "player": {"id": 51}, "assist": {"id": None}}]
    _api(monkeypatch, [_pl(7, goals=None), _pl(51, team=2)], events=ev)
    U.update_player_results()
    assert _row(p)["result"] == "WIN"


def test_null_goals_stay_pending_when_events_are_incomplete(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, market="goals")
    ev = [{"type": "Goal", "detail": "Normal Goal", "player": {"id": 51}, "assist": {"id": None}}]
    _api(monkeypatch, [_pl(7, goals=None), _pl(51, team=2)], events=ev)      # 1 event, 3 goals
    U.update_player_results()
    assert pd.isna(_row(p)["result"])


def test_stored_fixture_id_is_used_without_a_name_lookup(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, fixture_id="1.0")
    calls = []
    _api(monkeypatch, [_pl(7, sot=2), _pl(51, team=2)], calls=calls)
    U.update_player_results()
    assert _row(p)["result"] == "WIN"
    assert not [c for c in calls if c[0] == "/fixtures" and "date" in c[1]]


def test_missing_fixture_id_is_found_in_the_day_pool_and_written_back(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [_pl(7, sot=2), _pl(51, team=2)])
    U.update_player_results()
    row = _row(p)
    assert row["result"] == "WIN" and row["fixture_id"] == "1"


def test_fixture_id_is_written_back_even_before_the_result(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [], status="NS")
    U.update_player_results()
    row = _row(p)
    assert pd.isna(row["result"]) and row["fixture_id"] == "1"


def test_two_fixtures_matching_is_not_guessed(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    twin = {**FIX, "fixture": {"id": 2, "status": {"short": "FT"}}}
    _api(monkeypatch, [_pl(7, sot=2), _pl(51, team=2)], day=[FIX, twin])
    U.update_player_results()
    row = _row(p)
    assert pd.isna(row["result"]) and pd.isna(row["fixture_id"])


def test_postponed_fixture_is_void(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [], status="PST")
    U.update_player_results()
    row = _row(p)
    assert row["result"] == "VOID" and row["notes"] == "POSTPONED"


def test_league_without_props_stays_pending(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, league="Not A League")
    calls = []
    _api(monkeypatch, [], calls=calls)
    U.update_player_results()
    assert pd.isna(_row(p)["result"]) and not calls


@pytest.mark.parametrize("today,want", [("2026-11-01", ["2627", "2526"]),
                                        ("2027-03-01", ["2627", "2526"]),
                                        ("2027-08-15", ["2728", "2627"])])
def test_fd_std_settlement_reads_the_live_season(today, want):
    assert U._fd_std_season_codes(today) == want


def test_af_fallback_uses_the_matchs_season_not_todays(monkeypatch):
    import src.api_football_ou as AFO
    seen = []
    monkeypatch.setattr(AFO, "fetch_league_fixture_results",
                        lambda lid, season, league: seen.append(season) or pd.DataFrame())
    monkeypatch.setattr(U, "_AF_RESULTS_CACHE", {})
    U.fetch_scores_apifootball("Championship", "2026-05-02")      # last campaign's fixture
    U.fetch_scores_apifootball("Championship", "2026-10-25")      # this campaign's
    assert seen == ["2025", "2026"]


def test_ht_tip_settles_from_api_football_when_stores_miss_it(tmp_path, monkeypatch):
    d = (datetime.utcnow().date() - timedelta(days=2)).isoformat()
    monkeypatch.setattr(U.config, "OUTPUT_DIR", tmp_path)
    led = tmp_path / "ht_ledger.csv"
    pd.DataFrame([{"match_date": d, "league": "League One", "home_team": "Plymouth Argyle",
                   "away_team": "Wycombe Wanderers", "market": "ht_over05", "fair_odds": "1.4",
                   "result": "", "pnl": "", "notes": ""}]).to_csv(led, index=False)
    pd.DataFrame(columns=["league", "date", "home_team", "away_team", "ht_home_goals",
                          "ht_away_goals"]).to_parquet(tmp_path / "fd_history.parquet")
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [
        {"home_team": "Plymouth", "away_team": "Wycombe", "home_score": 2, "away_score": 1,
         "date_str": d, "ht_home": 1, "ht_away": 0}])
    fn = getattr(U, "update_ht_results", None) or getattr(U, "update_ht_ledger", None)
    if fn is None:
        pytest.skip("HT settler entry point not found")
    if hasattr(U, "HT_LEDGER_FILE"):
        monkeypatch.setattr(U, "HT_LEDGER_FILE", led)
    fn()
    assert pd.read_csv(led, dtype=str)["result"].iloc[0] == "WIN"


def test_season_comes_from_the_registry(tmp_path, monkeypatch):
    from src.season_resolver import resolve_api_season
    _ledger(tmp_path, monkeypatch, days_ago=2, league="Japan J-League")
    seen = []
    _api(monkeypatch, [], seen)
    U.update_player_results()
    d = (datetime.utcnow().date() - timedelta(days=2)).isoformat()
    assert seen == [resolve_api_season("Japan J-League", d)]
