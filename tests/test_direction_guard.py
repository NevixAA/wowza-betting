"""One goal DIRECTION per fixture (owner rule 2026-10-25).

"what i meant to deny its over 2.5 and under 2.5 for the same match, or over 1.5 and under 2.5, or
under 3.5 and over 2.5". Same-direction tips on one match are fine; BTTS never conflicts. The FIRST
tip stands (across runs: the live ledgers; within one run: the O/U 2.5 tip).

Real case pinned below: 2026-10-25 Viking FK v Tromso, O/U 2.5 UNDER (2026-10-18) + Over 3.5 (10-20).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src import direction_guard as DG
from src import ledger as L
from src.bankroll_policy import goal_direction as bp_direction
from src.bankroll_policy import size_day

DAY = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
HOME, AWAY, LG = "Viking FK", "Tromso", "Testland League"


def _ou(side, tier="SNIPER", edge=0.12, home=HOME, away=AWAY):
    return {"date": DAY, "league": LG, "home_team": home, "away_team": away, "bet": side,
            "best_side": side, "odds_over25": 1.95, "odds_under25": 2.6, "best_edge": edge,
            "signal_tier": tier, "drift_signal": "New", "model_type": "new_format"}


def _side(market, tier="VALUABLE", edge=0.05, home=HOME, away=AWAY):
    side = {"btts": "YES", "over15": "OVER", "over35": "OVER"}[market]
    return {"date": DAY, "league": LG, "home_team": home, "away_team": away, "model_type": "new_format",
            "market": market, "side": side, "signal_tier": tier, "odds": 2.06, "model_prob": 0.55,
            "fair_prob": 0.45, "edge": edge, "ev": 0.1, "market_odds": 2.06}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    return tmp_path


def _run(bets_rows, side_rows):
    """One predict run's write sequence exactly as pipeline.mode_predict orders it."""
    bets = pd.DataFrame(bets_rows) if bets_rows else pd.DataFrame(columns=list(_ou("OVER")))
    side = pd.DataFrame(side_rows) if side_rows else pd.DataFrame(columns=list(_side("btts")))
    b, s, c = DG.apply(bets, side)
    DG.record(c)
    if len(s):
        L.append_side_market_tips(s)
    if len(b):
        L.append_tips(b)
    return b, s, c


def _ledgered(out):
    b = pd.read_csv(out / "bets_ledger.csv") if (out / "bets_ledger.csv").exists() else pd.DataFrame()
    s = pd.read_csv(out / "side_bets_ledger.csv") if (out / "side_bets_ledger.csv").exists() else pd.DataFrame()
    return ([f"ou25 {x}" for x in b.get("side", [])] + [str(m) for m in s.get("market", [])])


# ── direction mapping ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("market,side,expected", [
    ("ou25", "OVER", "OVER"), ("ou25", "UNDER", "UNDER"), ("over15", "OVER", "OVER"),
    ("over35", "", "OVER"), ("under35", "", "UNDER"), ("btts", "YES", None), ("btts", "NO", None),
    ("ou25", "AVOID", None)])
def test_goal_direction(market, side, expected):
    assert DG.goal_direction(market, side) == expected
    assert bp_direction(market.upper() if market != "ou25" else "OU25", side) == expected


# ── the three pairs the owner named ──────────────────────────────────────────────────────────────
def test_over25_then_under25_across_runs_is_the_existing_flip_guard(out):
    _run([_ou("OVER", "MARKSMAN")], [])
    b, _, c = _run([_ou("UNDER", "VALUABLE")], [])
    # passed through to the EXISTING flip guard, which keeps it out of the send source and ledger
    assert len(b) == 1 and c.empty
    assert len(L.drop_flipped(b)) == 0
    led = pd.read_csv(out / "bets_ledger.csv")
    assert list(led["side"]) == ["OVER"] and "FLIP_SEEN UNDER" in str(led.iloc[0]["notes"])


def test_over15_then_under25_across_runs(out):
    _run([], [_side("over15")])
    b, s, c = _run([_ou("UNDER")], [_side("over15")])
    assert b.empty and len(s) == 1
    assert _ledgered(out) == ["over15"]
    assert c.iloc[0]["suppressed_market"] == "ou25" and c.iloc[0]["stood_market"] == "over15"
    assert c.iloc[0]["reason"] == "OPPOSITE_TO_EARLIER_TIP"


def test_under25_then_over35_across_runs_viking_v_tromso(out):
    _run([_ou("UNDER", "SNIPER")], [])
    b, s, c = _run([_ou("UNDER", "SNIPER")], [_side("over35", "VALUABLE")])
    assert len(b) == 1 and s.empty
    assert _ledgered(out) == ["ou25 UNDER"]
    row = c.iloc[0]
    assert (row["suppressed_market"], row["suppressed_direction"], row["stood_market"],
            row["stood_direction"], row["stood_source"]) == ("over35", "OVER", "ou25", "UNDER", "bets_ledger")


def test_within_run_ou25_wins(out):
    b, s, c = _run([_ou("UNDER")], [_side("over35", "SNIPER", edge=0.20), _side("btts")])
    assert len(b) == 1 and list(s["market"]) == ["btts"]
    assert c.iloc[0]["reason"] == "OPPOSITE_TO_OU25_SAME_RUN"
    assert sorted(_ledgered(out)) == ["btts", "ou25 UNDER"]


def test_first_tip_stands_across_runs_even_when_it_is_the_side_market(out):
    _run([], [_side("over35")])                          # run 1: Over 3.5 only
    b, s, _ = _run([_ou("UNDER", "SNIPER", edge=0.30)], [_side("over35")])   # run 2: strong Under
    assert b.empty and len(s) == 1
    # and run 3 again: still the first one
    b, s, c = _run([_ou("UNDER", "SNIPER", edge=0.30)], [_side("over35")])
    assert b.empty and len(s) == 1 and len(c) == 1


# ── what is allowed ─────────────────────────────────────────────────────────────────────────────
def test_same_direction_pairs_are_allowed(out):
    b, s, c = _run([_ou("OVER")], [_side("over15"), _side("over35"), _side("btts")])
    assert len(b) == 1 and len(s) == 3 and c.empty
    b, s, c = _run([_ou("OVER")], [_side("over15"), _side("over35"), _side("btts")])
    assert c.empty


def test_btts_never_conflicts(out):
    _run([_ou("UNDER")], [])
    b, s, c = _run([_ou("UNDER")], [_side("btts")])
    assert list(s["market"]) == ["btts"] and c.empty
    _run([], [_side("btts", home="A", away="B")])
    b, s, c = _run([_ou("UNDER", home="A", away="B")], [])
    assert len(b) == 1 and c.empty


def test_other_fixtures_are_untouched(out):
    _run([_ou("UNDER")], [])
    b, s, c = _run([_ou("OVER", home="X", away="Y")], [_side("over35", home="Z", away="W")])
    assert len(b) == 1 and len(s) == 1 and c.empty


# ── evidence ────────────────────────────────────────────────────────────────────────────────────
def test_conflicts_are_first_sight(out):
    _run([_ou("UNDER")], [])
    for _ in range(3):
        _run([_ou("UNDER")], [_side("over35")])
    c = pd.read_csv(out / "direction_conflicts.csv")
    assert len(c) == 1 and list(c.columns) == DG.CONFLICT_COLS


def test_guard_fails_open_and_never_raises(out, monkeypatch):
    monkeypatch.setattr(DG, "committed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    bets, side = pd.DataFrame([_ou("UNDER")]), pd.DataFrame([_side("over35")])
    b, s, c = DG.apply(bets, side)
    assert len(b) == 1 and len(s) == 1 and c.empty


# ── end to end through pipeline.mode_predict ────────────────────────────────────────────────────
def _preds():
    """Two fixtures scored by the 'model': Viking v Tromso (O/U UNDER + Over 3.5 + BTTS) and a clean
    one (O/U OVER + Over 1.5). Edges are far above the global fallback bars."""
    rows = []
    for home, away, side in ((HOME, AWAY, "UNDER"), ("Clean FC", "Other FC", "OVER")):
        rows.append({"date": DAY, "kickoff_utc": f"{DAY} 15:00:00", "league": LG, "home_team": home,
                     "away_team": away, "best_side": side, "bet": side, "bet_stake": 1.0,
                     "odds_over25": 1.95, "odds_under25": 2.40,
                     "edge_over": 0.15 if side == "OVER" else -0.2,
                     "edge_under": 0.15 if side == "UNDER" else -0.2, "best_edge": 0.15,
                     "signal_tier": "SNIPER", "drift_signal": "New", "model_type": "new_format",
                     "p_over35": 0.70 if side == "UNDER" else 0.10, "odds_over35": 2.20,
                     "p_btts": 0.75 if side == "UNDER" else 0.10, "odds_btts": 1.90,
                     "p_over15": 0.10 if side == "UNDER" else 0.95, "odds_over15": 1.40})
    return pd.DataFrame(rows)


@pytest.fixture
def predict_env(out, monkeypatch):
    import pipeline
    import src.predict as P
    import src.stake_plan as SP
    from src import edge_engine_shadow, shadow_experiments
    monkeypatch.setattr(P, "predict_upcoming", lambda *a, **k: (_preds(), []))
    monkeypatch.setattr(pipeline, "load_models", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_log_shadow", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_log_h2h", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "append_ht_tips", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "append_ht_observations", lambda *a, **k: None)
    monkeypatch.setattr(shadow_experiments, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    seen = {}
    monkeypatch.setattr(edge_engine_shadow, "run",
                        lambda preds, bets, side, **k: seen.update(bets=bets, side=side))
    monkeypatch.setattr(SP, "PLAN", out / "stake_plan.csv")
    monkeypatch.setattr(SP, "AUDIT", out / "stake_audit.csv")
    monkeypatch.setattr(SP, "load_league_ratings", lambda *a, **k: {})
    return pipeline, seen


def test_mode_predict_suppressed_tip_reaches_nothing_but_the_conflicts_log(out, predict_env):
    pipeline, seen = predict_env
    pipeline.mode_predict(historical=pd.DataFrame({"league": [LG]}))

    viking = lambda d: d[(d["home_team"] == HOME) & (d["away_team"] == AWAY)]
    bets = pd.read_csv(out / "bets.csv")
    side = pd.read_csv(out / "side_bets.csv")
    bl, sl = pd.read_csv(out / "bets_ledger.csv"), pd.read_csv(out / "side_bets_ledger.csv")
    plan = pd.read_csv(out / "stake_plan.csv")
    conf = pd.read_csv(out / "direction_conflicts.csv")

    # the O/U 2.5 UNDER stands everywhere
    assert list(viking(bets)["bet"]) == ["UNDER"] and list(viking(bl)["side"]) == ["UNDER"]
    # Over 3.5 on Viking: not in the send source (Telegram), ledger (album, top picks) or stake plan
    assert "over35" not in set(viking(side)["market"])
    assert "over35" not in set(viking(sl)["market"])
    vplan = plan[plan["fixture"] == f"{DAY}|{HOME}|{AWAY}"]
    assert set(vplan["market"]) == {"OU25", "BTTS"}
    # BTTS on Viking is not a goal-total line and stays
    assert "btts" in set(viking(side)["market"]) and "btts" in set(viking(sl)["market"])
    # the clean fixture is untouched: O/U OVER + Over 1.5
    clean = side[side["home_team"] == "Clean FC"]
    assert set(clean["market"]) == {"over15"}
    # the evidence row
    assert len(conf) == 1
    r = conf.iloc[0]
    assert (r["home_team"], r["suppressed_market"], r["stood_market"], r["reason"]) == \
        (HOME, "over35", "ou25", "OPPOSITE_TO_OU25_SAME_RUN")
    # the research shadow still saw every tip
    assert "over35" in set(viking(seen["side"])["market"])

    # second run: still suppressed, conflicts log stays first-sight
    pipeline.mode_predict(historical=pd.DataFrame({"league": [LG]}))
    assert "over35" not in set(viking(pd.read_csv(out / "side_bets_ledger.csv"))["market"])
    assert len(pd.read_csv(out / "direction_conflicts.csv")) == 1


def test_album_table_does_not_show_the_suppressed_tip(out, predict_env, monkeypatch):
    pipeline, _ = predict_env
    pipeline.mode_predict(historical=pd.DataFrame({"league": [LG]}))
    from telegram_bot import tips_table as TT
    lo = (pd.Timestamp(DAY) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(DAY) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = pd.DataFrame(TT.tips_rows(lo, hi))
    v = rows[rows["home_team"] == HOME]
    assert len(v) == 2                                                   # O/U 2.5 UNDER + BTTS
    assert not v["market"].astype(str).str.contains("3.5").any()


# ── stake plan defence in depth ─────────────────────────────────────────────────────────────────
def _sel(market, side, fixture="F1", odds=2.0):
    return {"date": DAY, "fixture": fixture, "league": "Argentina Primera Division",
            "model_type": "new_format", "market": market, "side": side, "signal_tier": "SNIPER", "odds": odds}


def _policy():
    """A minimal policy with every track staked, one_position_per_fixture OFF so the direction guard
    is the only thing that can zero a stake."""
    t = {"state": "ACTIVE", "base_fraction": 0.01, "max_odds": 2.75, "evidence_score": 1.0}
    return {"policy_version": "test", "portfolio": {"one_position_per_fixture": False},
            "tier_multipliers": {"SNIPER": 1.0},
            "tracks": {f"new_format|{m}": dict(t) for m in ("OU25", "OVER15", "OVER35", "BTTS")}}


def test_size_day_zeroes_the_opposite_direction_within_a_day():
    sel = pd.DataFrame([_sel("OVER35", "OVER"), _sel("OU25", "UNDER"), _sel("BTTS", "YES"),
                        _sel("OVER15", "OVER", fixture="F2"), _sel("OU25", "OVER", fixture="F2")])
    out = size_day(sel, _policy(), 100.0, 100.0)
    r = dict(zip(out["fixture"] + "|" + out["market"], out["reason"]))
    assert r["F1|OVER35"] == "OPPOSITE_DIRECTION_ON_FIXTURE"        # OU25 UNDER wins the day
    assert r["F1|OU25"] == "OK" and r["F1|BTTS"] == "OK"
    assert r["F2|OVER15"] == "OK" and r["F2|OU25"] == "OK"          # same direction
    assert float(out.loc[out["reason"] == "OPPOSITE_DIRECTION_ON_FIXTURE", "final_fraction"].sum()) == 0.0


def test_size_day_respects_an_earlier_staked_direction():
    sel = pd.DataFrame([_sel("OU25", "UNDER"), _sel("OVER15", "OVER", fixture="F2")])
    out = size_day(sel, _policy(), 100.0, 100.0, prior_directions={"F1": "OVER", "F2": "OVER"})
    r = dict(zip(out["fixture"], out["reason"]))
    assert r == {"F1": "OPPOSITE_DIRECTION_ON_FIXTURE", "F2": "OK"}


def test_size_day_unchanged_without_conflicts():
    sel = pd.DataFrame([_sel("OU25", "UNDER"), _sel("BTTS", "YES"), _sel("OVER15", "OVER", fixture="F2")])
    pol = _policy()
    a = size_day(sel, pol, 100.0, 100.0)
    b = size_day(sel, pol, 100.0, 100.0, prior_directions={"F1": "UNDER"})
    pd.testing.assert_frame_equal(a, b)
    assert "OPPOSITE_DIRECTION_ON_FIXTURE" not in set(a["reason"])


def test_stake_plan_reads_prior_directions_from_the_audit(out, monkeypatch):
    import src.stake_plan as SP
    monkeypatch.setattr(SP, "AUDIT", out / "stake_audit.csv")
    pd.DataFrame([
        {"fixture": "F1", "market": "OVER35", "side": "YES/OVER", "final_fraction": 0.004, "generated_at": "2026-10-20 10:00:00"},
        {"fixture": "F1", "market": "OU25", "side": "UNDER", "final_fraction": 0.004, "generated_at": "2026-10-21 10:00:00"},
        {"fixture": "F2", "market": "OU25", "side": "UNDER", "final_fraction": 0.0, "generated_at": "2026-10-20 10:00:00"},
        {"fixture": "F3", "market": "BTTS", "side": "YES", "final_fraction": 0.004, "generated_at": "2026-10-20 10:00:00"},
    ]).to_csv(out / "stake_audit.csv", index=False)
    assert SP.prior_directions() == {"F1": "OVER"}                   # zero stakes and BTTS fix nothing
    assert SP.prior_directions(["F2"]) == {}


def test_one_position_per_fixture_rule_unchanged():
    """With the policy's one_position_per_fixture ON, same-direction duplicates still collapse to one
    position exactly as before (reason ONE_POSITION_PER_FIXTURE)."""
    pol = _policy()
    pol["portfolio"]["one_position_per_fixture"] = True
    sel = pd.DataFrame([_sel("OU25", "OVER"), _sel("OVER15", "OVER")])
    out = size_day(sel, pol, 100.0, 100.0)
    assert sorted(out["reason"]) == ["OK", "ONE_POSITION_PER_FIXTURE"]
