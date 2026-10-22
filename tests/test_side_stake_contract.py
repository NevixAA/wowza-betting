"""P0 2026-10-22: side-market tips must reach the bankroll policy with their REAL model_type and odds.

pipeline._generate_side_bets dropped model_type/side and named the price `market_odds`, while
src/stake_plan.selections read getattr(r, "model_type", "unknown") / getattr(r, "odds", None). Every
BTTS / OVER15 tip landed on `unknown|BTTS` -> UNSUPPORTED -> zero stake, including the ACTIVE
new_format|BTTS track. test_stake_plan.py never caught it because it built consumer inputs by hand.

Every test here feeds the REAL producer output into the REAL consumer. Nothing is hand-shaped.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

import config
import pipeline
from src import stake_plan as sp
from src.bankroll_policy import load_policy, size_day

NF = "Argentina Primera Division"
NF2 = "Brazil Serie A"
STD = "Championship"
PAPER = "USA MLS"
DAY = "2026-10-23"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # global side bars (SNIPER 0.10 / MARKSMAN 0.08 / VALUABLE 0.04), no committed per-league file
    monkeypatch.setattr(pipeline, "_load_side_market_thresholds", lambda: {})
    # every league rated 10 -> rating scale 1.0; no network
    monkeypatch.setattr(sp, "load_league_ratings", lambda *a, **k: {
        lg: 10 for lg in (NF, NF2, STD, PAPER)})
    monkeypatch.setattr(sp, "PLAN", tmp_path / "stake_plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "stake_audit.csv")


def _pred(home, league, model_type=None, **mk):
    """One predictions.csv-shaped row. mk: market -> (p, odds)."""
    r = {"date": DAY, "league": league, "home_team": home, "away_team": home + " B",
         "model_type": model_type or config.model_type_for_league(league)}
    for m in ("btts", "over15", "over35"):
        p, o = mk.get(m, (None, None))
        r[f"p_{m}"], r[f"odds_{m}"] = p, o
    return r


def _run(rows, policy=None):
    preds = pd.DataFrame(rows)
    side = pipeline._generate_side_bets(preds, config.SIDE_MARKETS)
    sel = sp.selections(None, side)
    plan = size_day(sel, policy or load_policy(), bankroll=100.0, peak=100.0,
                    league_ratings={lg: 10 for lg in (NF, NF2, STD, PAPER)})
    return side, sel, plan


def _row(plan, home, market):
    g = plan[(plan["fixture"] == f"{DAY}|{home}|{home} B") & (plan["market"] == market)]
    assert len(g) == 1, (home, market, plan[["fixture", "market"]])
    return g.iloc[0]


def test_producer_emits_the_contract():
    side = pipeline._generate_side_bets(pd.DataFrame([_pred("A", NF, btts=(0.62, 2.10))]),
                                        config.SIDE_MARKETS)
    assert list(side.columns) == list(sp.SIDE_SELECTION_COLUMNS) + ["market_odds"]
    sp.check_side_selections(side)                                  # does not raise


def test_nf_btts_active_gets_a_positive_stake():                     # (1)
    _, _, plan = _run([_pred("A", NF, btts=(0.62, 2.10))])
    r = _row(plan, "A", "BTTS")
    assert r["track"] == "new_format|BTTS" and r["deployment_state"] == "ACTIVE"
    assert r["reason"] == "OK" and r["final_fraction"] > 0


def test_nf_over15_maps_to_its_track():                              # (2)
    _, _, plan = _run([_pred("B", NF, over15=(0.75, 1.60))])
    r = _row(plan, "B", "OVER15")
    assert r["track"] == "new_format|OVER15" and r["deployment_state"] == "ACTIVE_LOW"
    assert r["reason"] == "OK" and r["final_fraction"] > 0


def test_standard_btts_stays_paper():                                # (3)
    _, _, plan = _run([_pred("C", STD, btts=(0.62, 2.10))])
    r = _row(plan, "C", "BTTS")
    assert r["track"] == "standard|BTTS" and r["deployment_state"] == "PAPER"
    assert r["final_fraction"] == 0 and r["reason"] == "NO_REAL_STAKE_PAPER"


def test_over35_stays_blocked():                                     # (4)
    _, _, plan = _run([_pred("D", NF, over35=(0.55, 2.50)), _pred("E", STD, over35=(0.55, 2.50))])
    for home, mt in (("D", "new_format"), ("E", "standard")):
        r = _row(plan, home, "OVER35")
        assert r["track"] == f"{mt}|OVER35" and r["deployment_state"] == "BLOCKED"
        assert r["final_fraction"] == 0


def test_odds_and_model_type_survive_exactly():                      # (5) (6)
    rows = [_pred("F", NF, btts=(0.62, 2.07)), _pred("G", STD, over15=(0.80, 1.53)),
            _pred("H", NF2, over35=(0.55, 2.43))]
    side, sel, plan = _run(rows)
    want = {(r["home_team"], m): (r["model_type"], r[f"odds_{m}"])
            for r in rows for m in ("btts", "over15", "over35") if r[f"p_{m}"] is not None}
    for r in side.itertuples(index=False):
        assert (r.model_type, r.odds) == want[(r.home_team, r.market)]
    for r in plan.itertuples(index=False):
        home = r.fixture.split("|")[1]
        mt, odds = want[(home, r.market.lower())]
        assert r.model_type == mt and r.odds == odds and r.track == f"{mt}|{r.market}"


def test_no_unknown_track_unless_source_is_unknown():                # (7)
    _, _, plan = _run([_pred("I", NF, btts=(0.62, 2.10), over15=(0.75, 1.60)),
                       _pred("J", STD, btts=(0.62, 2.10), over15=(0.80, 1.53))])
    assert not plan["track"].str.startswith("unknown|").any()
    # a genuinely untagged league (neither list) is still reported as unknown, not forced
    _, _, plan = _run([_pred("K", "Nowhere League", btts=(0.62, 2.10))])
    assert _row(plan, "K", "BTTS")["track"] == "unknown|BTTS"


def test_predictions_tag_wins_and_blank_tag_falls_back():
    side = pipeline._generate_side_bets(pd.DataFrame([
        _pred("L", NF, model_type="standard", btts=(0.62, 2.10)),          # row tag is the source
        {**_pred("M", NF, btts=(0.62, 2.10)), "model_type": None}]),       # no tag -> config
        config.SIDE_MARKETS)
    got = dict(zip(side["home_team"], side["model_type"]))
    assert got == {"L": "standard", "M": "new_format"}


def test_one_position_per_fixture_still_holds():                     # (8)
    _, _, plan = _run([_pred("N", NF, btts=(0.62, 2.10), over15=(0.75, 1.60))])
    live = plan[plan["final_fraction"] > 0]
    assert len(live) == 1 and live.iloc[0]["market"] == "BTTS"           # higher evidence_score
    assert _row(plan, "N", "OVER15")["reason"] == "ONE_POSITION_PER_FIXTURE"


def test_paper_league_stays_paper():                                 # (9)
    _, _, plan = _run([_pred("O", PAPER, btts=(0.62, 2.10))])
    r = _row(plan, "O", "BTTS")
    assert r["track"] == "new_format|BTTS" and r["reason"] == "PAPER_LEAGUE" and r["final_fraction"] == 0


def test_odds_caps_unchanged():                                      # (10)
    _, _, plan = _run([_pred("P", NF, btts=(0.70, 2.65)), _pred("Q", NF, btts=(0.70, 3.05))])
    assert _row(plan, "P", "BTTS")["reason"] == "ODDS_ABOVE_TRACK_CAP"   # track max 2.60
    assert _row(plan, "Q", "BTTS")["reason"] == "ODDS_BLOCKED"           # hard 3.00
    # a track that asks for more than the 2.75 hard track cap still gets 2.75; 3.00 still blocks
    pol = copy.deepcopy(load_policy())
    pol["tracks"]["new_format|BTTS"]["max_odds"] = 5.0
    _, _, plan = _run([_pred("R", NF, btts=(0.70, 2.80)), _pred("S", NF, btts=(0.70, 3.00)),
                       _pred("T", NF, btts=(0.70, 2.70))], policy=pol)
    assert _row(plan, "R", "BTTS")["reason"] == "ODDS_ABOVE_TRACK_CAP"
    assert _row(plan, "S", "BTTS")["reason"] == "ODDS_BLOCKED"
    assert _row(plan, "T", "BTTS")["reason"] == "OK"


def test_full_write_path_stakes_nf_btts():
    preds = pd.DataFrame([_pred("U", NF, btts=(0.62, 2.10)), _pred("V", NF2, over15=(0.75, 1.60))])
    plan = sp.write(None, pipeline._generate_side_bets(preds, config.SIDE_MARKETS))
    assert set(plan["track"]) == {"new_format|BTTS", "new_format|OVER15"}
    assert (plan["final_fraction"] > 0).all()
    assert sp.health_check(plan, load_policy()) == []


# ── guards: the 2026-10-22 failure shape must be loud ────────────────────────────────────────────

def test_guard_rejects_side_bets_without_model_type_or_odds():
    side = pipeline._generate_side_bets(pd.DataFrame([_pred("W", NF, btts=(0.62, 2.10))]),
                                        config.SIDE_MARKETS)
    old_shape = side.drop(columns=["model_type", "odds", "side"])        # what the producer used to emit
    with pytest.raises(sp.SelectionSchemaError):
        sp.selections(None, old_shape)
    lost = side.assign(model_type=None)
    with pytest.raises(sp.SelectionSchemaError):
        sp.selections(None, lost)
    with pytest.raises(sp.SelectionSchemaError):
        sp.selections(None, side.assign(odds=float("nan")))
    assert sp.write(None, old_shape).empty                               # never a silent plan


def test_health_check_flags_active_market_all_unsupported():
    side = pipeline._generate_side_bets(pd.DataFrame([_pred("X", NF, btts=(0.62, 2.10))]),
                                        config.SIDE_MARKETS)
    pol = load_policy()
    bad = size_day(sp.selections(None, side.assign(model_type="unknown")), pol, bankroll=100.0, peak=100.0)
    probs = sp.health_check(bad, pol)
    assert len(probs) == 1 and probs[0].startswith("BTTS:")
    good = size_day(sp.selections(None, side), pol, bankroll=100.0, peak=100.0)
    assert sp.health_check(good, pol) == []
