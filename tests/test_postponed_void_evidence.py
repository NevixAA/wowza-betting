"""Postponed fixtures are voided on EVIDENCE only (estate scan P2, 2026-11-02).

45 main, 164 sharp and 743 prop rows sat pending > 7 days because only the props settler could ever
void. Now the O/U, side-market, sharp and HT settlers void a row when API-Football's own fixture
(the league's fixtures on that date, both clubs resolved in that pool) says PST / CANC / ABD - and a
row with no such evidence stays pending, whatever its age.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import update_results as U

LEAGUE = "USA MLS"


def _d(days_ago):
    return (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()


def _fx(status, home="Inter Miami", away="Orlando City"):
    return {"fixture": {"id": 9, "status": {"short": status}},
            "teams": {"home": {"name": home}, "away": {"name": away}}}


@pytest.fixture
def af(monkeypatch):
    """Fake the league's fixtures-on-a-date pool behind update_results._af_get."""
    state = {"day": [], "season": [], "calls": []}

    def fake(endpoint, params, cache_hours):
        state["calls"].append((endpoint, dict(params)))
        assert endpoint == "/fixtures", endpoint
        # the season pool is the round-5 same-pair reschedule lookup (no 'date' param)
        return {"response": state["day"] if "date" in params else state["season"]}
    monkeypatch.setattr(U, "_af_get", fake)
    monkeypatch.setattr(U, "_SEASON_FIXTURES", {})
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    monkeypatch.setattr(U, "_STATUS_ALLOWED", None)
    return state


@pytest.mark.parametrize("status,label", [("PST", "POSTPONED"), ("CANC", "CANCELLED"),
                                          ("ABD", "ABANDONED")])
def test_evidence_statuses_void(af, status, label):
    af["day"] = [_fx(status), _fx("FT", "LA Galaxy", "LAFC")]
    assert U._postponement_evidence(LEAGUE, _d(9), "Inter Miami", "Orlando City") == (label, "")


@pytest.mark.parametrize("status", ["FT", "NS", "TBD", "1H"])
def test_other_statuses_stay_pending(af, status):
    af["day"] = [_fx(status)]
    label, why = U._postponement_evidence(LEAGUE, _d(30), "Inter Miami", "Orlando City")
    assert label is None and why.startswith("no_result_yet")


def test_match_missing_from_the_day_stays_pending(af):
    af["day"] = [_fx("PST", "LA Galaxy", "LAFC")]          # another fixture was postponed
    label, why = U._postponement_evidence(LEAGUE, _d(30), "Inter Miami", "Orlando City")
    assert label is None and why.startswith("fixture_not_found_on_date")


def test_no_key_no_lookup(af, monkeypatch):
    monkeypatch.delenv("APIFOOTBALL_KEY")
    assert U._postponement_evidence(LEAGUE, _d(30), "Inter Miami", "Orlando City")[0] is None
    assert not af["calls"]


def test_unknown_league_no_lookup(af):
    assert U._postponement_evidence("Not A League", _d(30), "A", "B")[0] is None
    assert not af["calls"]


def test_budget_plan_limits_and_rotates():
    rows = pd.DataFrame({"league": [LEAGUE] * 6, "match_date": [_d(i) for i in range(3, 9)]})
    today = datetime.utcnow().date().isoformat()
    a = U._plan_status_checks([rows], today, budget=2, now=datetime(2026, 11, 2, 7))
    assert len(a) == 2
    seen = set()
    for h in range(24):
        seen |= U._plan_status_checks([rows], today, budget=2, now=datetime(2026, 11, 2, h))
    assert len(seen) == 6                                     # every league-day gets its turn


# ── the settlers ──────────────────────────────────────────────────────────────

def test_sharp_postponed_is_void_and_unknown_stays_pending(af, monkeypatch, tmp_path):
    p = tmp_path / "sharp_ledger.csv"
    rows = [[_d(10), _d(10), LEAGUE, "Inter Miami", "Orlando City", "Over 2.5", "OVER", "STRONG",
             "1.9", "", "", "", "", "", "", ""],
            [_d(10), _d(10), LEAGUE, "LA Galaxy", "LAFC", "Over 2.5", "OVER", "STRONG",
             "1.9", "", "", "", "", "", "", ""]]
    cols = ["signal_date", "match_date", "league", "home_team", "away_team", "market_label", "side",
            "signal", "opening_odds", "current_odds", "drift_pct", "consensus_pct", "n_books",
            "result", "pnl", "resolved_date"]
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    monkeypatch.setattr(U, "SHARP_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: [])
    af_calls = []
    monkeypatch.setattr(U, "fetch_scores_apifootball",
                        lambda league, match_date=None: af_calls.append(league) or [])
    af["day"] = [_fx("PST"), _fx("NS", "LA Galaxy", "LAFC")]
    U.update_sharp_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert af_calls, "sharp now falls back to API-Football results like O/U and side markets"
    assert out["result"].tolist() == ["VOID", ""]
    assert out["pnl"].tolist() == ["0.0", ""]


def test_sharp_settles_from_api_football_results(af, monkeypatch, tmp_path):
    p = tmp_path / "sharp_ledger.csv"
    pd.DataFrame([{"signal_date": _d(10), "match_date": _d(10), "league": LEAGUE,
                   "home_team": "Inter Miami", "away_team": "Orlando City", "market_label": "Over 2.5",
                   "side": "OVER", "signal": "STRONG", "opening_odds": "2.1", "current_odds": "1.9",
                   "result": "", "pnl": "", "resolved_date": ""}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "SHARP_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: [])
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [
        {"home_team": "Inter Miami", "away_team": "Orlando City", "home_score": 2, "away_score": 2,
         "date_str": _d(10)}])
    U.update_sharp_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert out["result"].tolist() == ["WIN"] and out["pnl"].tolist() == ["0.9"]
    assert not af["calls"]                                    # a played match needs no status lookup


def test_side_market_postponed_is_void_with_note(af, monkeypatch, tmp_path):
    p = tmp_path / "side_bets_ledger.csv"
    pd.DataFrame([{"source": "live", "match_date": _d(9), "league": LEAGUE, "home_team": "Inter Miami",
                   "away_team": "Orlando City", "market": "btts", "odds": "1.8", "result": "",
                   "pnl": "", "closing_odds": "", "clv_pct": "", "notes": ""},
                  {"source": "live", "match_date": _d(9), "league": LEAGUE, "home_team": "LA Galaxy",
                   "away_team": "LAFC", "market": "btts", "odds": "1.8", "result": "",
                   "pnl": "", "closing_odds": "", "clv_pct": "", "notes": ""}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "SIDE_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: [])
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [])
    af["day"] = [_fx("CANC"), _fx("FT", "LA Galaxy", "LAFC")]
    U.update_side_market_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert out["result"].tolist() == ["VOID", ""]
    assert out["notes"].tolist() == ["CANCELLED", ""]
    assert out["pnl"].tolist() == ["0.0", ""]


def test_ht_postponed_is_void(af, monkeypatch, tmp_path):
    monkeypatch.setattr(U.config, "OUTPUT_DIR", tmp_path)
    led = tmp_path / "ht_ledger.csv"
    pd.DataFrame([{"source": "live", "match_date": _d(9), "league": LEAGUE, "home_team": "Inter Miami",
                   "away_team": "Orlando City", "market": "ht_over05", "result": "", "pnl": "",
                   "notes": ""}]).to_csv(led, index=False)
    pd.DataFrame(columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals", "ht_home_goals",
                          "ht_away_goals"]).to_parquet(tmp_path / "fd_history.parquet")
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league, match_date=None: [])
    af["day"] = [_fx("ABD")]
    U.update_ht_results()
    out = pd.read_csv(led, dtype=str, keep_default_na=False)
    assert out["result"].tolist() == ["VOID"] and out["notes"].tolist() == ["ABANDONED"]
