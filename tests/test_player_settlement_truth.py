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
])
def test_threshold_markets_settle(market, stats, want):
    assert U._resolve_player_market(market, stats) == want


def _ledger(tmp_path, monkeypatch, days_ago, league="Premier League", player="Bukayo Saka"):
    d = (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()
    p = tmp_path / "player_ledger.csv"
    pd.DataFrame([{"match_date": d, "league": league, "home_team": "Arsenal", "away_team": "Spurs",
                   "player_name": player, "market": "sot2", "market_odds": "2.5", "result": "",
                   "pnl": "", "resolved_date": ""}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "PLAYER_LEDGER_FILE", p)
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    return p


def _api(monkeypatch, stats, seen=None):
    import player_model.api_football as AF

    def fid(league_id, season, date_str, home, away):
        if seen is not None:
            seen.append(season)
        return 1
    monkeypatch.setattr(AF, "find_fixture_id", fid)
    monkeypatch.setattr(AF, "get_fixture_player_stats", lambda _id: stats)


def test_no_stats_yet_stays_pending(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].fillna("").iloc[0] == ""


def test_no_stats_after_a_week_is_void_with_reason(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=8)
    _api(monkeypatch, [])
    U.update_player_results()
    row = pd.read_csv(p, dtype=str).iloc[0]
    assert row["result"] == "VOID" and row["notes"] == "NO_STATS_7D"


def test_player_absent_from_published_stats_is_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [{"player_name": "Declan Rice", "minutes_played": 90, "shots_on_target": 1}])
    U.update_player_results()
    row = pd.read_csv(p, dtype=str).iloc[0]
    assert row["result"] == "VOID" and row["notes"] == "DNP"


def test_abbreviated_stats_name_is_not_a_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, player="Bukayo Saka")
    _api(monkeypatch, [{"player_name": "B. Saka", "player_id": 99, "minutes_played": 90, "shots_on_target": 2},
                       {"player_name": "Declan Rice", "player_id": 98, "minutes_played": 90, "shots_on_target": 0}])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].iloc[0] == "WIN"


def test_player_id_wins_over_names(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, player="Armand Laurienté")
    led = pd.read_csv(p, dtype=str); led["player_id"] = "777"; led.to_csv(p, index=False)
    _api(monkeypatch, [{"player_name": "Someone Else", "player_id": 777, "minutes_played": 90, "shots_on_target": 3}])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].iloc[0] == "WIN"


def test_two_same_initial_surname_candidates_are_not_guessed(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2, player="Alex Silva")
    _api(monkeypatch, [{"player_name": "A. Silva", "player_id": 1, "minutes_played": 90, "shots_on_target": 3},
                       {"player_name": "Andre Silva", "player_id": 2, "minutes_played": 90, "shots_on_target": 0}])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["notes"].iloc[0] == "DNP"


def test_threshold_prop_settles(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [{"player_name": "Bukayo Saka", "minutes_played": 90, "shots_on_target": 3}])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].iloc[0] == "WIN"


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
