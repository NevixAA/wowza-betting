"""Round 5 settlement fixes (settle5), 2026-11-02.

* P1-2e  team ledgers settle on 90 minutes: AET / PEN fixtures come from API-Football's
         score.fulltime, and a football-data / OddsAPI score is not used for a fixture API-Football
         reports as AET / PEN (knockout guard);
* a fixture API-Football moved more than 48h away voids 'RESCHEDULED' on a same-pair season lookup;
* the restored player rows are graded IN their sidecar, never appended to player_ledger.csv;
* clv_tracker grades 'cards' with update_results.card_market_result (one card rule).

    python -m pytest tests/test_settlement_round5.py -q -p no:cacheprovider
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import update_results as U
from src import api_football_ou as AFO
from src import shadow_settlement as ss

LEAGUE = "USA MLS"


def _d(days_ago: int) -> str:
    return (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()


def _plus(d: str, k: int) -> str:
    return str((pd.Timestamp(d) + pd.Timedelta(days=k)).date())


def _ev(home, away, hs, as_, date, **kw):
    return {"home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
            "date_str": date, **kw}


def _af_fixture(fid, home, away, date, status, goals, ft, ht=(0, 0)):
    return {"fixture": {"id": fid, "date": f"{date}T23:30:00+00:00", "status": {"short": status}},
            "teams": {"home": {"name": home}, "away": {"name": away}},
            "goals": {"home": goals[0], "away": goals[1]},
            "score": {"halftime": {"home": ht[0], "away": ht[1]},
                      "fulltime": {"home": ft[0], "away": ft[1]}}}


# ── API-Football: AET / PEN carry the 90-minute score ─────────────────────────────────────────────

def test_af_results_extra_time_rows_carry_the_90_minute_score(monkeypatch):
    d = "2026-10-29"
    resp = {"response": [
        _af_fixture(1, "Inter Miami", "Nashville", d, "AET", goals=(2, 1), ft=(1, 1)),
        _af_fixture(2, "LAFC", "Austin", d, "PEN", goals=(0, 0), ft=(0, 0)),
        _af_fixture(3, "Seattle", "Portland", d, "FT", goals=(3, 2), ft=(3, 2))]}
    seen = []
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: seen.append(params) or resp)
    df = AFO.fetch_league_fixture_results(253, "2026", "", statuses="FT-AET-PEN")
    assert seen[-1]["status"] == "FT-AET-PEN"
    r = df.set_index("home_team")
    assert (r.loc["Inter Miami", "home_goals"], r.loc["Inter Miami", "away_goals"]) == (1, 1)
    assert r.loc["Inter Miami", "status"] == "AET" and r.loc["LAFC", "status"] == "PEN"
    assert (r.loc["Seattle", "home_goals"], r.loc["Seattle", "away_goals"]) == (3, 2)
    # the training loader's call (default 'FT') keeps its frame exactly as before
    assert "status" not in AFO.fetch_league_fixture_results(253, "2026", "").columns


def test_af_extra_time_without_a_90_minute_score_is_skipped(monkeypatch):
    fx = _af_fixture(1, "Inter Miami", "Nashville", "2026-10-29", "AET", goals=(2, 1), ft=(1, 1))
    fx["score"]["fulltime"] = {"home": None, "away": None}
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: {"response": [fx]})
    assert AFO.fetch_league_fixture_results(253, "2026", "", statuses="FT-AET-PEN").empty


def test_af_extra_time_list_is_none_when_not_fetched(monkeypatch):
    monkeypatch.setattr(AFO, "_KEY", "x")
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: None)
    assert AFO.fetch_league_extra_time_results(253, "2026", "") is None
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: {"errors": {"requests": "quota"},
                                                                   "response": []})
    assert AFO.fetch_league_extra_time_results(253, "2026", "") is None
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: {"response": []})
    assert AFO.fetch_league_extra_time_results(253, "2026", "").empty


def test_fallback_settles_an_aet_fixture_on_90_minutes(monkeypatch):
    d = _d(4)
    monkeypatch.setattr(AFO, "_get", lambda ep, params, ttl_h=0: {"response": [
        _af_fixture(1, "Inter Miami", "Nashville", d, "AET", goals=(2, 1), ft=(1, 1))]})
    monkeypatch.setattr(U, "_AF_RESULTS_CACHE", {})
    evs = U.fetch_scores_apifootball(LEAGUE, d)
    assert len(evs) == 1 and (evs[0]["home_score"], evs[0]["away_score"]) == (1, 1)
    assert evs[0]["status"] == "AET"


# ── knockout guard on football-data / OddsAPI scores ─────────────────────────────────────────────

@pytest.fixture
def et(monkeypatch):
    """API-Football's AET / PEN list for the league-season (None = not fetched)."""
    state = {"rows": [], "calls": 0}

    def fake(lid, season, league=""):
        state["calls"] += 1
        if state["rows"] is None:
            return None
        return pd.DataFrame(state["rows"], columns=["date", "home_team", "away_team", "home_goals",
                                                     "away_goals", "ht_home_goals", "ht_away_goals",
                                                     "status", "fixture_id"])
    monkeypatch.setattr(AFO, "fetch_league_extra_time_results", fake)
    monkeypatch.setattr(AFO, "_KEY", "x")
    monkeypatch.setattr(U, "_AF_ET_CACHE", {})
    return state


SHARP_COLS = ["signal_date", "match_date", "league", "home_team", "away_team", "market_label", "side",
              "signal", "opening_odds", "current_odds", "drift_pct", "consensus_pct", "n_books",
              "result", "pnl", "resolved_date"]


def _sharp(tmp_path, monkeypatch, d, primary):
    p = tmp_path / "sharp_ledger.csv"
    pd.DataFrame([[d, d, LEAGUE, "Inter Miami", "Nashville", "Over 2.5", "OVER", "STRONG", "2.0",
                   "2.0", "", "", "", "", "", ""]], columns=SHARP_COLS).to_csv(p, index=False)
    monkeypatch.setattr(U, "SHARP_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: primary)
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [])
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    return p


def test_sharp_120_minute_score_is_replaced_by_the_90_minute_one(et, tmp_path, monkeypatch):
    # OddsAPI: 2-1 (after extra time) -> Over 2.5 WIN. On 90 minutes it was 1-1 -> LOSS.
    d = _d(2)
    p = _sharp(tmp_path, monkeypatch, d, [_ev("Inter Miami", "Nashville", 2, 1, d)])
    et["rows"] = [[d, "Inter Miami", "Nashville", 1, 1, 0, 1, "AET", 77]]
    U.update_sharp_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]
    assert out["result"] == "LOSS" and out["pnl"] == "-1.0"


def test_sharp_guard_unavailable_stays_pending(et, tmp_path, monkeypatch):
    d = _d(2)
    p = _sharp(tmp_path, monkeypatch, d, [_ev("Inter Miami", "Nashville", 2, 1, d)])
    et["rows"] = None
    U.update_sharp_results()
    assert pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]["result"] == ""


def test_sharp_regular_time_match_settles_on_the_primary_score(et, tmp_path, monkeypatch):
    d = _d(2)
    p = _sharp(tmp_path, monkeypatch, d, [_ev("Inter Miami", "Nashville", 2, 1, d)])
    et["rows"] = [[d, "LAFC", "Austin", 0, 0, 0, 0, "PEN", 78]]          # another tie went to pens
    U.update_sharp_results()
    assert pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]["result"] == "WIN"
    assert et["calls"] == 1


def test_side_market_btts_on_90_minutes(et, tmp_path, monkeypatch):
    d = _d(2)
    p = tmp_path / "side_bets_ledger.csv"
    pd.DataFrame([["live", d, d, LEAGUE, "Inter Miami", "Nashville", "btts", "1.8", "SNIPER",
                   "", "", "", ""]],
                 columns=["source", "generated_at", "match_date", "league", "home_team", "away_team",
                          "market", "odds", "signal_tier", "result", "pnl", "closing_odds",
                          "clv_pct"]).to_csv(p, index=False)
    monkeypatch.setattr(U, "SIDE_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {LEAGUE: "USA"})
    monkeypatch.setattr(U, "fetch_scores_fd", lambda league: [_ev("Inter Miami", "Nashville", 1, 1, d)])
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [])
    monkeypatch.setattr(U, "_closing_for_market", lambda *a, **k: float("nan"))
    et["rows"] = [[d, "Inter Miami", "Nashville", 1, 0, 0, 0, "AET", 77]]   # 1-0 at 90, 1-1 a.e.t.
    U.update_side_market_results()
    assert pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]["result"] == "LOSS"


def test_ou_team_event_goes_through_the_guard(et):
    d = _d(3)
    et["rows"] = [[d, "Inter Miami", "Nashville", 1, 1, 1, 0, "AET", 77]]
    ev, held = U._team_event(LEAGUE, "Inter Miami", "Nashville", d,
                             [[_ev("Inter Miami", "Nashville", 3, 1, d)]], fallback=False)
    assert held == "" and U._grade_ou_event(ev, "OVER")[0] == "LOSS"   # 1-1, not 3-1


def test_shadow_settles_on_90_minutes_through_the_guard(et):
    ko = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d 20:00:00")
    d = ko[:10]
    df = pd.DataFrame([{"fixture_id": "e1", "market": "ou25", "logged_at": d + "T08:00:00+00:00",
                        "match_date": d, "kickoff_utc": ko, "league": LEAGUE, "v9_side": "OVER",
                        "entry_odds": "1.9", "home_team": "Inter Miami", "away_team": "Nashville",
                        "result": "", "settle_status": ""}]).astype(str)
    et["rows"] = [[d, "Inter Miami", "Nashville", 1, 1, 1, 0, "AET", 77]]
    prov = [("oddsapi", lambda league: [_ev("Inter Miami", "Nashville", 2, 1, d)])]
    updates, _ = ss.settle_frame(df, providers=prov, extra_time_guard=U._knockout_guard)
    f = updates[0]["fields"]
    assert (f["home_goals"], f["away_goals"], f["result"]) == ("1", "1", "LOSS")
    assert f["result_source"] == "api-football (90 min)"
    et["rows"] = None
    U._AF_ET_CACHE.clear()
    updates, pending = ss.settle_frame(df, providers=prov, extra_time_guard=U._knockout_guard)
    assert not updates and pending == {"KNOCKOUT_GUARD_UNAVAILABLE": 1}


def test_already_settled_aet_rows_are_restated_beside_the_ledger():
    d = "2026-10-25"
    led = pd.DataFrame([
        {"match_date": d, "league": LEAGUE, "home_team": "Inter Miami", "away_team": "Nashville",
         "side": "OVER", "odds": "1.9", "result": "WIN", "pnl": "0.9"},       # 2-1 a.e.t., 1-1 at 90
        {"match_date": d, "league": LEAGUE, "home_team": "Seattle", "away_team": "Portland",
         "side": "OVER", "odds": "1.9", "result": "WIN", "pnl": "0.9"}])      # regular time
    et = [_ev("Inter Miami", "Nashville", 1, 1, d, status="AET")]
    before = led.copy()
    out = U.extra_time_restatement({"bets_ledger": led}, lambda league, date: et)
    assert len(out) == 1
    r = out.iloc[0]
    assert (r["original_result"], r["restated_result"], r["restated_pnl"], r["score_90"]) == \
        ("WIN", "LOSS", -1.0, "1-1")
    assert bool(r["changed"]) and led.equals(before)


# ── rescheduled > 48h, then played ────────────────────────────────────────────────────────────────

@pytest.fixture
def af(monkeypatch):
    """API-Football via update_results._af_get: the day pool and the season pool."""
    state = {"day": [], "season": [], "calls": []}

    def fake(endpoint, params, cache_hours):
        state["calls"].append(dict(params))
        return {"response": state["day"] if "date" in params else state["season"]}
    monkeypatch.setattr(U, "_af_get", fake)
    monkeypatch.setattr(U, "_SEASON_FIXTURES", {})
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    monkeypatch.setattr(U, "_STATUS_ALLOWED", None)
    return state


def _sfx(home, away, date, status="FT"):
    return {"fixture": {"id": 5, "date": f"{date}T19:00:00+00:00", "status": {"short": status}},
            "teams": {"home": {"name": home}, "away": {"name": away}}}


def test_moved_more_than_48h_voids_rescheduled(af):
    d = _d(10)
    af["day"] = [_sfx("LA Galaxy", "LAFC", d)]                       # the round went ahead
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 6)),  # the match was moved +6 days
                    _sfx("Orlando City", "Inter Miami", _plus(d, -60)),
                    _sfx("LA Galaxy", "LAFC", d)]
    assert U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City") == ("RESCHEDULED", "")


def test_moved_and_not_yet_played_beyond_48h_also_voids(af):
    d = _d(5)
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 9), "NS")]
    assert U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")[0] == "RESCHEDULED"


def test_two_same_pair_fixtures_are_ambiguous_and_stay_pending(af):
    d = _d(20)
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 5)),
                    _sfx("Inter Miami", "Orlando City", _plus(d, 15), "NS")]
    label, why = U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")
    assert label is None and why.startswith("reschedule_ambiguous")


def test_same_pair_inside_the_window_is_not_voided(af):
    d = _d(10)
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 1))]   # UTC next day, played
    label, why = U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")
    assert label is None and why.startswith("no_result_yet")
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 1), "PST")]
    U._SEASON_FIXTURES.clear()                                          # next run
    assert U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")[0] == "POSTPONED"


def test_only_an_earlier_meeting_is_not_this_match(af):
    d = _d(10)
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, -40))]
    label, why = U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")
    assert label is None and why.startswith("reschedule_not_this_match")


def test_window_still_open_makes_no_season_lookup(af):
    d = _d(1)
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 6))]
    label, why = U._postponement_evidence(LEAGUE, d, "Inter Miami", "Orlando City")
    assert label is None and why.startswith("fixture_not_found_on_date")
    assert all("date" in c for c in af["calls"])


def test_side_ledger_row_voids_rescheduled_with_a_note(af, tmp_path, monkeypatch):
    d = _d(10)
    p = tmp_path / "side_bets_ledger.csv"
    pd.DataFrame([["live", d, d, LEAGUE, "Inter Miami", "Orlando City", "btts", "1.8", "SNIPER",
                   "", "", "", ""]],
                 columns=["source", "generated_at", "match_date", "league", "home_team", "away_team",
                          "market", "odds", "signal_tier", "result", "pnl", "closing_odds",
                          "clv_pct"]).to_csv(p, index=False)
    monkeypatch.setattr(U, "SIDE_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: [])
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [])
    af["season"] = [_sfx("Inter Miami", "Orlando City", _plus(d, 7))]
    U.update_side_market_results()
    r = pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]
    assert (r["result"], r["pnl"], r["notes"]) == ("VOID", "0.0", "RESCHEDULED")


# ── restored player rows: graded in the sidecar ───────────────────────────────────────────────────

def _papi(monkeypatch):
    fx = {"fixture": {"id": 1, "status": {"short": "FT"}, "date": ""},
          "teams": {"home": {"id": 1, "name": "Arsenal"}, "away": {"id": 2, "name": "Spurs"}},
          "goals": {"home": 1, "away": 1}, "score": {"fulltime": {"home": 1, "away": 1}}}
    pls = [{"team": {"id": 1}, "players": [{"player": {"id": 7, "name": "P"}, "statistics": [{
        "games": {"minutes": 90, "substitute": False}, "goals": {"total": 1, "assists": 0},
        "shots": {"on": 2}, "cards": {"yellow": 0, "red": 0}}]}]},
           {"team": {"id": 2}, "players": [{"player": {"id": 51, "name": "S"}, "statistics": [{
               "games": {"minutes": 90, "substitute": False}, "goals": {"total": 1, "assists": 0},
               "shots": {"on": 1}, "cards": {"yellow": 1, "red": 0}}]}]}]

    def fake(endpoint, params, cache_hours):
        if endpoint == "/fixtures":
            return {"response": [fx]}
        if endpoint == "/fixtures/players":
            return {"response": pls}
        return {"response": []}
    monkeypatch.setattr(U, "_af_get", fake)
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")


def test_restored_rows_are_graded_in_their_sidecar_never_in_the_ledger(tmp_path, monkeypatch):
    d = _d(30)
    cols = ["signal_date", "match_date", "league", "home_team", "away_team", "player_name", "team",
            "position", "market", "tier", "model_prob", "market_odds", "ev", "edge_rel", "n_games",
            "is_played", "result", "pnl", "notes", "resolved_date", "player_id", "fixture_id",
            "restore_category", "first_seen_commit"]
    base = dict.fromkeys(cols, "")
    base.update(signal_date=d, match_date=d, league="Premier League", home_team="Arsenal",
                away_team="Spurs", tier="AVOID", player_id="7", fixture_id="1.0",
                restore_category="lost_in_bot_commit", first_seen_commit="da399394d9")
    rows = [{**base, "player_name": "P", "market": "goals"},
            {**base, "player_name": "P", "market": "sot3", "market_odds": "4.5", "tier": "VALUABLE"},
            {**base, "player_name": "Q", "market": "goals", "player_id": ""}]
    side = tmp_path / "player_ledger_restored_2026-11.csv"
    pd.DataFrame(rows, columns=cols).to_csv(side, index=False, lineterminator="\n")
    led = tmp_path / "player_ledger.csv"
    led.write_text("signal_date,match_date,result\n", encoding="utf-8")
    monkeypatch.setattr(U, "PLAYER_LEDGER_FILE", led)
    _papi(monkeypatch)

    out = U.grade_restored_player_sidecar(side)
    assert out["graded"] == 2
    r = pd.read_csv(side, dtype=str, keep_default_na=False)
    assert list(r["graded_result"]) == ["WIN", "LOSS", ""]          # no player_id -> pending
    assert list(r["graded_pnl"]) == ["", "-1.0", ""]                 # unpriced AVOID: no P/L
    assert list(r["graded_notes"])[0].startswith("unpriced")
    assert list(r["result"]) == ["", "", ""] and list(r["pnl"]) == ["", "", ""]   # first sight kept
    assert list(r.columns[:len(cols)]) == cols
    assert led.read_text(encoding="utf-8") == "signal_date,match_date,result\n"   # ledger untouched
    # idempotent: graded rows are not graded again
    assert U.grade_restored_player_sidecar(side)["graded"] == 0


# ── one card rule in clv_records ──────────────────────────────────────────────────────────────────

def test_clv_tracker_cards_use_the_shared_rule(tmp_path, monkeypatch):
    from player_model import clv_tracker, config as pconfig
    monkeypatch.setattr(pconfig, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"bet_id": "p|2026-10-10|A Player|cards", "odds_bet": 3.0, "result": "", "pnl": ""},
                  {"bet_id": "p|2026-10-10|B Player|cards", "odds_bet": 3.0, "result": "", "pnl": ""}]
                 ).to_csv(tmp_path / "clv_records.csv", index=False)
    pd.DataFrame([{"player_name": n, "market": "cards", "match_date": "2026-10-10", "player_id": pid,
                   "fixture_id": "900", "home_team": "Arsenal", "away_team": "Spurs"}
                  for n, pid in (("A Player", "1"), ("B Player", "2"))]
                 ).to_csv(tmp_path / "player_ledger.csv", index=False)
    hist = pd.DataFrame([{"player_name": "A Player", "date": pd.Timestamp("2026-10-10"), "goals": 0,
                          "assists": 0, "shots_on_target": 0, "yellow_cards": 1, "red_cards": None,
                          "home_team": "Arsenal", "away_team": "Spurs", "player_id": 1,
                          "fixture_id": 900},
                         {"player_name": "B Player", "date": pd.Timestamp("2026-10-10"), "goals": 0,
                          "assists": 0, "shots_on_target": 0, "yellow_cards": 0, "red_cards": None,
                          "home_team": "Arsenal", "away_team": "Spurs", "player_id": 2,
                          "fixture_id": 900}])
    pq = tmp_path / "ph.parquet"
    hist.to_parquet(pq)
    assert clv_tracker.settle_results(parquet_path=pq) == 1
    out = pd.read_csv(tmp_path / "clv_records.csv", keep_default_na=False)
    # yellow known, red unknown: WIN under the shared rule (was pending); no card + red unknown: pending
    assert list(out["result"]) == ["WIN", ""]
    assert U.card_market_result(1, None) == "WIN" and U.card_market_result(0, None) == ""
