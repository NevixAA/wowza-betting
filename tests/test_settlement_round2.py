"""Round 2 settlement fixes (CLAUDE_PROMPT_2.md V1, P1-2, P1-8, P2 red card), 2026-11-02.

    python -m pytest tests/test_settlement_round2.py -q -p no:cacheprovider
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import update_results as U
from src import shadow_settlement as ss


def _d(days_ago: int) -> str:
    return (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()


def _plus(d: str, k: int) -> str:
    return str((pd.Timestamp(d) + pd.Timedelta(days=k)).date())


SHARP_COLS = ["signal_date", "match_date", "league", "home_team", "away_team", "market_label", "side",
              "signal", "opening_odds", "current_odds", "drift_pct", "consensus_pct", "n_books",
              "result", "pnl", "resolved_date"]


def _sharp(tmp_path, monkeypatch, rows):
    p = tmp_path / "sharp_ledger.csv"
    pd.DataFrame(rows, columns=SHARP_COLS).to_csv(p, index=False)
    monkeypatch.setattr(U, "SHARP_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {})
    monkeypatch.setattr(U, "fetch_scores", lambda *a, **k: [])
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    return p


def _ev(home, away, hs, as_, date):
    return {"home_team": home, "away_team": away, "home_score": hs, "away_score": as_, "date_str": date}


# ── V1: sharp P/L at the signal-time price ────────────────────────────────────────────────────────

def test_sharp_settles_at_signal_time_price_not_opening(tmp_path, monkeypatch):
    d = _d(5)
    p = _sharp(tmp_path, monkeypatch, [[d, d, "USA MLS", "Inter Miami", "Orlando City", "Over 2.5",
                                        "OVER", "STEAM_SHARP", "2.20", "1.85", "-15.9", "", "",
                                        "", "", ""]])
    monkeypatch.setattr(U, "fetch_scores_apifootball",
                        lambda league, match_date=None: [_ev("Inter Miami", "Orlando City", 2, 1, d)])
    U.update_sharp_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert out["result"].tolist() == ["WIN"]
    assert out["pnl"].tolist() == ["0.85"]                     # 1.85 - 1, not 2.20 - 1
    assert out["pnl_basis"].tolist() == ["signal_price"]
    assert out["opening_odds"].tolist() == ["2.20"]            # first-sight values untouched


def test_sharp_without_signal_price_grades_result_only(tmp_path, monkeypatch):
    d = _d(5)
    p = _sharp(tmp_path, monkeypatch, [[d, d, "USA MLS", "Inter Miami", "Orlando City", "Over 2.5",
                                        "OVER", "STRONG", "2.20", "", "", "", "", "", "", ""]])
    monkeypatch.setattr(U, "fetch_scores_apifootball",
                        lambda league, match_date=None: [_ev("Inter Miami", "Orlando City", 2, 1, d)])
    U.update_sharp_results()
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert out["result"].tolist() == ["WIN"] and out["pnl"].tolist() == [""]


def test_sharp_restatement_sidecar_keeps_original_and_skips_new_basis(tmp_path):
    led = pd.DataFrame([
        ["2026-10-01", "2026-10-02", "L", "A", "B", "x", "OVER", "S", "2.2", "1.85", "", "", "", "WIN", "1.2", "d"],
        ["2026-10-01", "2026-10-02", "L", "C", "D", "x", "HOME", "S", "3.0", "2.5", "", "", "", "LOSS", "-1.0", "d"],
        ["2026-10-01", "2026-10-02", "L", "E", "F", "x", "AWAY", "S", "3.0", "2.5", "", "", "", "", "", ""],
    ], columns=SHARP_COLS)
    led["pnl_basis"] = ["", "", ""]
    src, dst = tmp_path / "sharp_ledger.csv", tmp_path / "corrections" / "sharp_ledger_restated.csv"
    led.to_csv(src, index=False)
    before = src.read_bytes()
    out = U.write_sharp_restatement(src, dst)
    assert src.read_bytes() == before                          # the ledger is never rewritten
    assert len(out) == 2                                       # settled rows only
    r = out.set_index("home_team")
    assert r.loc["A", "original_pnl"] == "1.2" and float(r.loc["A", "restated_pnl"]) == pytest.approx(0.85)
    assert float(r.loc["C", "restated_pnl"]) == -1.0
    led.loc[0, "pnl_basis"] = "signal_price"                   # already on the new basis
    assert len(U.restate_sharp_ledger(led)) == 1


# ── P1-2: rows that could never settle ────────────────────────────────────────────────────────────

def test_result_one_day_later_in_the_source_settles():
    # MLS evening kick-off: the source dates it D+1 (UTC); the exact-date match never found it.
    d = "2026-10-20"
    found = U._find_result("Inter Miami", "Orlando City", d, "OVER",
                           [_ev("Inter Miami", "Orlando City", 3, 1, _plus(d, 1))])
    assert found is not None and found[0] == "WIN"


def test_rescheduled_within_48h_settles_beyond_does_not():
    d = "2026-10-20"
    assert U._find_result("Inter Miami", "Orlando City", d, "OVER",
                          [_ev("Inter Miami", "Orlando City", 0, 1, _plus(d, 2))])[0] == "LOSS"
    assert U._find_result("Inter Miami", "Orlando City", d, "OVER",
                          [_ev("Inter Miami", "Orlando City", 0, 1, _plus(d, 3))]) is None


def test_window_never_picks_between_two_scores():
    d = "2026-10-20"
    evs = [_ev("Inter Miami", "Orlando City", 0, 1, _plus(d, 1)),
           _ev("Inter Miami", "Orlando City", 2, 2, _plus(d, 2))]
    assert U._find_result("Inter Miami", "Orlando City", d, "OVER", evs) is None


def test_two_spelling_club_in_one_pool_settles():
    # fd_history holds the same Austrian fixture under 'Tirol' and 'WSG Wattens': the resolver
    # refused the pool as ambiguous, so the row never settled.
    d = "2026-10-18"
    evs = [_ev("WSG Tirol", "Rapid Vienna", 1, 1, d), _ev("WSG Wattens", "Rapid Vienna", 1, 1, d)]
    assert U._match_event("Tirol", "Rapid Vienna", evs, {d}) is not None
    # ... but never when the two reports disagree on the score
    evs[1] = _ev("WSG Wattens", "Rapid Vienna", 2, 1, d)
    assert U._match_event("Tirol", "Rapid Vienna", evs, {d}) is None


def test_two_different_clubs_are_still_refused():
    d = "2026-10-18"
    evs = [_ev("Inter", "Milan", 1, 1, d), _ev("Inter Miami", "Milan", 2, 1, d)]
    assert U._match_event("Inter Mia", "Milan", evs, {d}) is None


@pytest.fixture
def af(monkeypatch):
    state = {"day": []}
    monkeypatch.setattr(U, "_af_get", lambda e, p, c: {"response": state["day"]})
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    monkeypatch.setattr(U, "_STATUS_ALLOWED", None)
    return state


def _fx(status):
    return {"fixture": {"id": 9, "status": {"short": status}},
            "teams": {"home": {"name": "Inter Miami"}, "away": {"name": "Orlando City"}}}


@pytest.mark.parametrize("status", ["PST", "ABD"])
def test_postponed_inside_48h_window_stays_pending(af, status):
    af["day"] = [_fx(status)]
    label, why = U._postponement_evidence("USA MLS", _d(1), "Inter Miami", "Orlando City")
    assert label is None and why.startswith("postponed_within_48h_window")
    assert U._postponement_evidence("USA MLS", _d(3), "Inter Miami", "Orlando City")[0] is not None


def test_cancelled_voids_at_once(af):
    af["day"] = [_fx("CANC")]
    assert U._postponement_evidence("USA MLS", _d(1), "Inter Miami", "Orlando City")[0] == "CANCELLED"


# ── player grader: 90 minutes, red card, reschedule ───────────────────────────────────────────────

def _pledger(tmp_path, monkeypatch, market, days_ago=5, fixture_id="1"):
    p = tmp_path / "player_ledger.csv"
    pd.DataFrame([{"match_date": _d(days_ago), "league": "Premier League", "home_team": "Arsenal",
                   "away_team": "Spurs", "player_name": "P", "market": market, "market_odds": "3.0",
                   "result": "", "pnl": "", "resolved_date": "", "player_id": "7",
                   "fixture_id": fixture_id}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "PLAYER_LEDGER_FILE", p)
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    return p


def _papi(monkeypatch, status="FT", players=None, events=None, date=None, goals=(1, 1), ft=(1, 1)):
    fx = {"fixture": {"id": 1, "status": {"short": status}, "date": date or ""},
          "teams": {"home": {"id": 1, "name": "Arsenal"}, "away": {"id": 2, "name": "Spurs"}},
          "goals": {"home": goals[0], "away": goals[1]},
          "score": {"fulltime": {"home": ft[0], "away": ft[1]}}}
    pls = players or [(1, {"player": {"id": 7, "name": "P"}, "statistics": [{
        "games": {"minutes": 120, "substitute": False}, "goals": {"total": 1, "assists": 0},
        "shots": {"on": 1}, "cards": {"yellow": 0, "red": 0}}]}),
        (2, {"player": {"id": 51, "name": "S"}, "statistics": [{
            "games": {"minutes": 120, "substitute": False}, "goals": {"total": 1, "assists": 0},
            "shots": {"on": 1}, "cards": {"yellow": 0, "red": 0}}]})]

    def fake(endpoint, params, cache_hours):
        if endpoint == "/fixtures":
            return {"response": [fx]}
        if endpoint == "/fixtures/players":
            blocks = {}
            for t, p in pls:
                blocks.setdefault(t, []).append(p)
            return {"response": [{"team": {"id": t}, "players": ps} for t, ps in blocks.items()]}
        if endpoint == "/fixtures/events":
            return {"response": events or []}
        return {"response": []}
    monkeypatch.setattr(U, "_af_get", fake)


def _res(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).iloc[0]


def _goal(pid, minute, extra=None, detail="Normal Goal", comments=None):
    return {"type": "Goal", "detail": detail, "time": {"elapsed": minute, "extra": extra},
            "player": {"id": pid}, "assist": {"id": None}, "comments": comments}


def test_extra_time_goal_does_not_settle_a_90_minute_prop(tmp_path, monkeypatch):
    # 1-1 after 90, player 7 scores in the 105th minute: whole-match sheet says goals=1 -> WIN was
    # extra-time leakage. On 90 minutes he did not score.
    p = _pledger(tmp_path, monkeypatch, "goals")
    _papi(monkeypatch, status="AET", goals=(2, 1), ft=(1, 1),
          events=[_goal(9, 30), _goal(51, 90, extra=3), _goal(7, 105)])
    U.update_player_results()
    assert _res(p)["result"] == "LOSS"


def test_extra_time_sot_bound_and_pending(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "sot2")                  # whole-match SOT 1 < 2 -> LOSS
    _papi(monkeypatch, status="AET", goals=(2, 1), ft=(1, 1),
          events=[_goal(9, 30), _goal(51, 90, extra=3), _goal(7, 105)])
    U.update_player_results()
    assert _res(p)["result"] == "LOSS"
    p = _pledger(tmp_path, monkeypatch, "sot")                   # 1 >= 1 but maybe in extra time
    U.update_player_results()
    assert _res(p)["result"] == ""


def test_extra_time_incomplete_events_stay_pending(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "goals")
    _papi(monkeypatch, status="PEN", goals=(1, 1), ft=(1, 1), events=[_goal(7, 20)])
    U.update_player_results()
    assert _res(p)["result"] == ""


def test_straight_red_card_wins_the_cards_prop(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "cards")
    pls = [(1, {"player": {"id": 7, "name": "P"}, "statistics": [{
        "games": {"minutes": 40, "substitute": False}, "goals": {"total": 0, "assists": 0},
        "shots": {"on": 0}, "cards": {"yellow": 0, "red": 1}}]}),
        (2, {"player": {"id": 51, "name": "S"}, "statistics": [{
            "games": {"minutes": 90, "substitute": False}, "goals": {"total": 0, "assists": 0},
            "shots": {"on": 0}, "cards": {"yellow": 0, "red": 0}}]})]
    _papi(monkeypatch, players=pls)
    U.update_player_results()
    assert _res(p)["result"] == "WIN"


@pytest.mark.parametrize("yc,rc", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)])
def test_one_card_rule_in_player_ledger_and_clv_records(yc, rc):
    from player_model.clv_tracker import _MARKET_HIT
    want = "WIN" if _MARKET_HIT["cards"]({"yc": yc, "rc": rc}) else "LOSS"
    assert U.card_market_result(yc, rc) == want
    assert U._resolve_player_market("cards", {"yellow_card": int(yc > 0), "red_card": rc}) == want


def test_fixture_played_more_than_48h_later_is_void(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "goals", days_ago=10)
    _papi(monkeypatch, status="FT", date=_plus(_d(10), 4) + "T19:00:00+00:00")
    U.update_player_results()
    r = _res(p)
    assert r["result"] == "VOID" and r["notes"].startswith("RESCHEDULED")


def test_fixture_played_within_48h_settles_on_the_actual_match(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "goals", days_ago=10)
    _papi(monkeypatch, status="FT", date=_plus(_d(10), 1) + "T19:00:00+00:00")
    U.update_player_results()
    assert _res(p)["result"] == "WIN"


def test_player_fixture_found_on_the_next_utc_day(tmp_path, monkeypatch):
    p = _pledger(tmp_path, monkeypatch, "goals", fixture_id="")
    d = _d(5)
    seen = []
    fx = {"fixture": {"id": 1, "status": {"short": "FT"}, "date": _plus(d, 1) + "T01:00:00+00:00"},
          "teams": {"home": {"id": 1, "name": "Arsenal"}, "away": {"id": 2, "name": "Spurs"}},
          "goals": {"home": 1, "away": 1}}
    pl = lambda pid: {"player": {"id": pid, "name": "x"}, "statistics": [{    # noqa: E731
        "games": {"minutes": 90, "substitute": False}, "goals": {"total": 1, "assists": 0},
        "shots": {"on": 1}, "cards": {"yellow": 0, "red": 0}}]}

    def fake(endpoint, params, cache_hours):
        if endpoint == "/fixtures" and "date" in params:
            seen.append(params["date"])
            return {"response": [fx] if params["date"] == _plus(d, 1) else []}
        if endpoint == "/fixtures/players":
            return {"response": [{"team": {"id": 1}, "players": [pl(7)]},
                                 {"team": {"id": 2}, "players": [pl(51)]}]}
        return {"response": []}
    monkeypatch.setattr(U, "_af_get", fake)
    U.update_player_results()
    r = _res(p)
    assert r["result"] == "WIN" and r["fixture_id"] == "1" and seen[:2] == [d, _plus(d, 1)]


# ── P1-8: shadow settlement looks results up in the MATCH's season ───────────────────────────────

def test_shadow_provider_gets_the_match_date_and_caches_per_season():
    rows = []
    for fid, ko in (("e1", "2026-12-05 20:00:00"), ("e2", "2027-03-07 20:00:00")):
        rows.append({"fixture_id": fid, "market": "ou25", "logged_at": ko[:10] + "T08:00:00+00:00",
                     "match_date": ko[:10], "kickoff_utc": ko, "league": "USA MLS",
                     "v9_side": "OVER", "entry_odds": "1.9", "home_team": "Inter Miami",
                     "away_team": "Orlando City", "result": "", "settle_status": ""})
    df = pd.DataFrame(rows).astype(str)
    calls = []

    def af(league, match_date=None):
        calls.append(match_date)
        return [_ev("Inter Miami", "Orlando City", 3, 0, match_date)] if match_date else []
    updates, pending = ss.settle_frame(df, providers=[("api-football", af)],
                                       now=datetime(2027, 4, 1, tzinfo=timezone.utc))
    assert calls == ["2026-12-05", "2027-03-07"]               # one call per season, with the date
    assert len(updates) == 2
