"""TOP-3 per league per gameweek (src/top3_league.py, scripts/top3_league_replay.py) - synthetic data only.

Owner rules 2026-11-08: 3 best value tips per league per gameweek across all team markets; only on the
league's OWN bars (never a global bar); fewer than 3 when fewer clear the bar (never padded); the direction
rule; first sight / week freeze; behind a flag that is OFF by default.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import config
from src import top3_league as T
from src.research import league_weekly as LW

ROOT = Path(__file__).resolve().parents[1]
LEAGUE = "La Liga 2"
WEEK = "La Liga 2|2026-W47"
NOW = pd.Timestamp("2026-11-20T17:45:00Z")           # after the freeze, before the lock
LOCK = pd.Timestamp("2026-11-20T19:00:00Z")          # earliest kickoff of the week
LEAD = 120.0                                          # freeze 17:00Z


def _bars(v=0.02, m=0.04, s=0.06, approved=False):
    return {"valuable_th": v, "marksman_th": m, "sniper_th": s, "approved": approved, "status": "UNPROVEN",
            "evidence": "WALK_FORWARD_HOLDOUT"}


def _ctx(markets=("ou25", "btts", "over15", "over35"), applied=True, league=LEAGUE, **kw):
    std = {league: _bars(**kw)} if "ou25" in markets else {}
    side = {m: {league: _bars(**kw)} for m in markets if m != "ou25"}
    return {"standard": std, "side": side,
            "applied": {f"{m}|{league}" for m in markets} if applied else set()}


def _cand(i, market="OU25", side="OVER", pm=0.60, pf=0.50, ev_lb=0.05, home=None, away=None,
          ko="2026-11-21T15:00:00Z", fs="2026-11-19T10:00:00Z", league=LEAGUE, pq="TWO_SIDED_MULTI_BOOK"):
    r = {c: "" for c in LW.CAND_COLS}
    r.update({"selection_engine_version": "market_edge_v2", "fixture_id": f"fx{i}", "market": market,
              "first_seen_at": fs, "league_week_id": f"{league}|2026-W47", "league": league,
              "model_type": "standard", "season": "2026/27", "match_date": ko[:10], "home_team": home or f"H{i}",
              "away_team": away or f"A{i}", "kickoff_utc": ko, "side": side, "p_model_raw": pm,
              "p_market_fair": pf, "ev_lower_bound": ev_lb, "ev_point": ev_lb + 0.01, "p_lower_bound": pm - 0.02,
              "odds": 2.0, "n_books": 5, "price_quality_state": pq, "market_source": "MEDIAN_TWO_SIDED_BOOKS"})
    return r


def _frame(rows):
    first = pd.DataFrame(rows)
    # the week's lock: the earliest kickoff among its candidates
    lock_row = _cand(99, ko=LOCK.strftime("%Y-%m-%dT%H:%M:%SZ"), pm=0.5, pf=0.5, ev_lb=-0.1)
    return pd.concat([first, pd.DataFrame([lock_row])], ignore_index=True).astype(object)


def _freeze(cands, ctx, now=NOW, committed=None, done=None):
    return T.freeze_due(cands, done or set(), ctx, now, LEAD, committed or {}, {}, None)


# ── the rules ───────────────────────────────────────────────────────────────────────────────────
def test_at_most_three_per_league_week_ranked_by_value_across_markets():
    rows = [_cand(1, "OU25", "OVER", ev_lb=0.01), _cand(2, "BTTS", "YES", ev_lb=0.09),
            _cand(3, "OVER15", "OVER", ev_lb=0.05), _cand(4, "OVER35", "OVER", ev_lb=0.07),
            _cand(5, "OU25", "UNDER", pm=0.65, ev_lb=0.03)]
    picks, weeks = _freeze(_frame(rows), _ctx())
    assert len(picks) == 3 and len(weeks) == 1
    assert list(picks["fixture_id"]) == ["fx2", "fx4", "fx3"]          # RS-1: ev_lower_bound first
    assert list(picks["top3_rank"]) == [1, 2, 3]
    assert set(picks["bar_source"]) == {"LEAGUE"} and set(picks["selection_mode"]) == {"TOP3_LEAGUE_GW"}
    assert json.loads(weeks["excluded"].iloc[0])["RANKED_BELOW_TOP3"] == 2


def test_never_padded_when_fewer_clear_the_bar():
    rows = [_cand(1, pm=0.60, pf=0.50), _cand(2, "BTTS", "YES", pm=0.53, pf=0.52),   # edge 0.01 < 0.02
            _cand(3, "OVER15", "OVER", pm=0.51, pf=0.50)]
    picks, weeks = _freeze(_frame(rows), _ctx())
    assert list(picks["fixture_id"]) == ["fx1"]
    assert weeks["n_picks"].iloc[0] == 1
    assert json.loads(weeks["excluded"].iloc[0])["BELOW_LEAGUE_BAR"] >= 2


def test_no_global_bar_ever(monkeypatch):
    monkeypatch.setattr(config, "SIDE_GLOBAL_BARS", {"sniper": 0.0, "marksman": 0.0, "valuable": -1.0})
    rows = [_cand(1, pm=0.95, pf=0.50, ev_lb=0.5), _cand(2, "BTTS", "YES", pm=0.95, pf=0.40)]
    picks, weeks = _freeze(_frame(rows), {"standard": {}, "side": {}, "applied": set()})
    assert picks.empty
    assert json.loads(weeks["excluded"].iloc[0])["NO_LEAGUE_BAR"] == 3
    src = inspect.getsource(T)
    for banned in ("SIDE_GLOBAL_BARS", "global_bars", "side_bars(", "VALUABLE_THRESHOLD"):
        assert banned not in src.split('"""', 2)[2], banned


def test_pre_repair_params_without_valuable_bar_are_not_the_league_bars():
    ctx = _ctx()
    ctx["standard"][LEAGUE] = {"sniper_th": 0.07, "marksman_th": 0.05, "approved": True}   # old shape
    b, why = T.league_bars(ctx, LEAGUE, "OU25")
    assert b is None and why == T.R_NO_BAR


def test_bars_must_be_applied_by_the_owner():
    b, why = T.league_bars(_ctx(applied=False), LEAGUE, "BTTS")
    assert b is None and why == T.R_NOT_APPLIED
    b, why = T.league_bars(_ctx(v=0.05, m=0.04, s=0.06), LEAGUE, "BTTS")
    assert b is None and why == T.R_BAD_BAR


def test_no_two_sided_price_no_tip():
    rows = [_cand(1, pm=0.9, pf=0.5, pq="INSUFFICIENT")]
    picks, weeks = _freeze(_frame(rows), _ctx())
    assert picks.empty and json.loads(weeks["excluded"].iloc[0])["NO_TWO_SIDED_PRICE"] == 1


def test_direction_rule_within_the_week_and_against_production():
    rows = [_cand(1, "OU25", "UNDER", pm=0.7, ev_lb=0.09, home="X", away="Y"),
            _cand(2, "BTTS", "YES", pm=0.7, ev_lb=0.08, home="X", away="Y"),       # BTTS Yes = OVER
            _cand(3, "OVER15", "OVER", pm=0.7, ev_lb=0.07, home="P", away="Q")]
    picks, weeks = _freeze(_frame(rows), _ctx())
    assert list(picks["fixture_id"]) == ["fx1", "fx3"]
    assert json.loads(weeks["excluded"].iloc[0])["DIRECTION_CONFLICT"] == 1
    committed = {("2026-11-21", "P", "Q"): {"direction": "UNDER"}}             # earlier live production tip
    picks2, _ = _freeze(_frame(rows), _ctx(), committed=committed)
    assert "fx3" not in set(picks2["fixture_id"])


def test_week_freezes_once_before_the_lock_from_first_sight_only():
    rows = [_cand(1, ev_lb=0.05), _cand(2, "BTTS", "YES", ev_lb=0.09, fs="2026-11-20T17:30:00Z")]  # late
    early, weeks0 = _freeze(_frame(rows), _ctx(), now=pd.Timestamp("2026-11-20T16:59:00Z"))
    assert early.empty and weeks0.empty                                          # before FREEZE: nothing
    picks, weeks = _freeze(_frame(rows), _ctx())
    assert list(picks["fixture_id"]) == ["fx1"]
    assert json.loads(weeks["excluded"].iloc[0])["FIRST_SEEN_AFTER_FREEZE"] == 1
    assert weeks["freeze_ts"].iloc[0] == "2026-11-20T17:00:00Z"
    again, w2 = _freeze(_frame(rows), _ctx(), done={WEEK})
    assert again.empty and w2.empty                                              # decided once


def test_a_started_fixture_is_skipped_and_the_next_qualifier_moves_up():
    rows = [_cand(1, ev_lb=0.09, ko="2026-11-20T19:00:00Z"), _cand(2, ev_lb=0.05)]
    picks, weeks = _freeze(_frame(rows), _ctx(), now=pd.Timestamp("2026-11-20T19:05:00Z"))
    assert list(picks["fixture_id"]) == ["fx2"]
    assert json.loads(weeks["excluded"].iloc[0])["STARTED_BEFORE_ISSUE"] >= 1


def test_policy_switch_defaults_to_current_and_never_stakes():
    assert T.policy_switch({}, LEAGUE) == "CURRENT"
    assert T.policy_switch({"top3_league": {"leagues": {LEAGUE: "paper"}}}, LEAGUE) == "PAPER"
    assert T.policy_switch({"top3_league": {"leagues": {LEAGUE: "bogus"}}}, LEAGUE) == "CURRENT"
    picks, _ = _freeze(_frame([_cand(1)]), _ctx())
    assert picks["stake_effect"].str.startswith("NONE").all()
    assert "stake_amount" not in picks.columns


# ── flag, lead and production isolation ────────────────────────────────────────────────────────
PROD = ["output/stake_plan.csv", "output/bets_ledger.csv", "output/side_bets_ledger.csv", "output/predictions.csv",
        "output/bets.csv", "config/bankroll_policy.json"]


def _hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in PROD if (ROOT / f).exists()}


def test_flag_off_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(T.FLAG_ENV, raising=False)
    before = _hashes()
    for fn in (T.run, T.settle, T.notify):
        assert fn(out_dir=tmp_path) == {"enabled": False}
    assert list(tmp_path.iterdir()) == []
    assert _hashes() == before


def test_flag_on_without_lead_freezes_nothing_and_touches_only_its_own_files(tmp_path):
    shadow = pd.DataFrame([{"market": "OU25", "selection_engine_version": "market_edge_v2"}])
    shadow.to_csv(tmp_path / "edge_engine_shadow.csv", index=False)
    before = _hashes()
    r = T.run(now=NOW, out_dir=tmp_path, env={T.FLAG_ENV: "true"}, committed={})
    assert r["enabled"] and r.get("status") == "FREEZE_LEAD_NOT_SET", r
    written = {p.name for p in tmp_path.iterdir()} - {"edge_engine_shadow.csv"}
    assert written <= {T.CANDIDATES_NAME, T.REPORT_NAME, T.PICKS_NAME, T.WEEKS_NAME}
    assert not (tmp_path / T.PICKS_NAME).exists()
    assert _hashes() == before


def test_lead_has_no_default():
    assert T.freeze_lead_min({}) is None
    assert T.freeze_lead_min({T.LEAD_ENV: "0"}) is None and T.freeze_lead_min({T.LEAD_ENV: "x"}) is None
    assert T.freeze_lead_min({T.LEAD_ENV: "90"}) == 90.0


def test_store_settle_and_notify_round_trip(tmp_path):
    picks, weeks = _freeze(_frame([_cand(1, home="H1", away="A1"), _cand(2, "BTTS", "YES", ev_lb=0.01)]), _ctx())
    assert T.picks_store(tmp_path).append(picks) == 2
    v91 = pd.DataFrame([{"fixture_id": "fx1", "match_date": "2026-11-21", "home_team": "H1", "away_team": "A1",
                         "home_goals": "2", "away_goals": "1", "result_source": "test"}])
    out, n = T.settle_picks(T.picks_store(tmp_path).read(), v91, pd.Timestamp("2026-11-22T12:00:00Z"))
    assert n == 1
    r = out[out["fixture_id"] == "fx1"].iloc[0]
    assert r["result"] == "WIN" and float(r["flat_pnl"]) == pytest.approx(1.0)
    sent = []
    env = {T.FLAG_ENV: "1"}
    assert T.notify(out_dir=tmp_path, env=env, sender=lambda t: sent.append(t) or True)["sent"] == 1
    assert T.notify(out_dir=tmp_path, env=env, sender=lambda t: sent.append(t) or True)["sent"] == 0
    assert "TOP-3 La Liga 2" in sent[0] and "own, owner-approved" in sent[0]


def test_workflows_gate_the_steps_on_the_repository_variable():
    for wf in ("predict.yml", "update_results.yml"):
        t = (ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
        assert "TOP3_LEAGUE_ENABLED: ${{ vars.TOP3_LEAGUE_ENABLED }}" in t
        assert "python -m src.top3_league" in t
    assert "TOP3_" not in (ROOT / "config.py").read_text(encoding="utf-8")   # research lock env untouched


# ── replay ──────────────────────────────────────────────────────────────────────────────────────
def _replay():
    spec = importlib.util.spec_from_file_location("top3_replay", ROOT / "scripts" / "top3_league_replay.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bet(i, market="over25", side="yes", edge=0.05, p=0.6, odds=2.0, pnl=1.0, date="2026-05-02",
         stage="holdout", league=LEAGUE, home=None, away=None):
    return {"market": market, "league": league, "date": date, "stage": stage, "home": home or f"H{i}",
            "away": away or f"A{i}", "side": side, "p_side": p, "odds_taken": odds, "edge": edge, "pnl": pnl}


def test_replay_uses_only_the_league_bars_and_never_pads():
    R = _replay()
    bars = {"ou25": {LEAGUE: _bars()}, "btts": {LEAGUE: _bars()}}
    bets = pd.DataFrame([_bet(1, edge=0.05, p=0.7), _bet(2, "btts", edge=0.03, p=0.6),
                         _bet(3, "over15", edge=0.20, p=0.9),               # no over15 bars -> no tip
                         _bet(4, edge=0.01), _bet(5, "btts", edge=0.10, p=0.55), _bet(6, edge=0.04, p=0.58),
                         _bet(7, edge=0.06, p=0.5, league="Serie B")])    # league without bars
    picks, excl = R.select(bets, bars)
    assert len(picks) == 3 and picks["league"].eq(LEAGUE).all()
    assert excl["NO_LEAGUE_BAR"] == 2 and excl["BELOW_LEAGUE_BAR"] == 1 and excl["RANKED_BELOW_TOP3"] == 1
    assert picks["out_of_sample"].all()
    few, _ = R.select(pd.DataFrame([_bet(1)]), bars)
    assert len(few) == 1


def test_replay_marks_in_period_split_rows_before_the_check_window_in_sample():
    R = _replay()
    b = _bars()
    b.update(evidence="IN_PERIOD_SPLIT", split={"check_from": "2026-09-01"})
    picks, _ = R.select(pd.DataFrame([_bet(1, "over15", date="2026-08-15"), _bet(2, "over15", date="2026-09-12")]),
                        {"over15": {LEAGUE: b}})
    assert dict(zip(picks["date"], picks["out_of_sample"])) == {"2026-08-15": False, "2026-09-12": True}


def test_replay_refuses_to_run_without_real_bars(tmp_path):
    R = _replay()
    with pytest.raises(SystemExit, match="no bars proposals"):
        R.main(["--track", "standard", "--predictions", str(tmp_path / "p.parquet"), "--bars-dir", str(tmp_path)])
