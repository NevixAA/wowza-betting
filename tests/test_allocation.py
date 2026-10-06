"""Bankroll allocation per cell. The job of these tests is to keep it from sizing noise."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.allocation import (ACCUMULATING, ALLOCATABLE, CAUSE_CALIBRATION, CAUSE_MARKET_BETTER,
                            CAUSE_ODDS_BAND, CAUSE_SIDE_BIAS, INCONCLUSIVE, MAX_CELL_FRACTION,
                            MAX_TOTAL_FRACTION, NEGATIVE, allocate, diagnose)


def bets(n=200, win_rate=0.55, odds=2.0, league="L", market="ou25", track="new_format",
         side_mix=True, seed=0, days=60):
    rng = np.random.default_rng(seed)
    won = rng.random(n) < win_rate
    return pd.DataFrame({
        "league": league, "market": market, "model_type": track,
        "match_date": pd.to_datetime("2026-09-25") + pd.to_timedelta(
            rng.integers(0, days, n), unit="D"),
        "odds": odds, "result": np.where(won, "WIN", "LOSS"),
        "pnl": np.where(won, odds - 1, -1.0),
        "signal_tier": "SNIPER",
        "side": rng.choice(["OVER", "UNDER"], n) if side_mix else "UNDER",
    })


# ── sizing ────────────────────────────────────────────────────────────────────────────────────
def test_a_clearly_profitable_cell_gets_a_stake():
    cells, _ = allocate(bets(n=300, win_rate=0.62, odds=2.0))
    c = cells[0]
    assert c.state == ALLOCATABLE and 0 < c.fraction <= MAX_CELL_FRACTION


def test_a_cell_whose_interval_spans_zero_gets_exactly_zero():
    """Not a small number — zero. The honest reading of 'could be negative' is 'do not bet'."""
    cells, _ = allocate(bets(n=200, win_rate=0.505, odds=2.0))
    assert cells[0].state == INCONCLUSIVE and cells[0].fraction == 0.0


def test_a_losing_cell_is_marked_negative_not_merely_unfunded():
    cells, _ = allocate(bets(n=250, win_rate=0.35, odds=2.0))
    assert cells[0].state == NEGATIVE and cells[0].fraction == 0.0


def test_a_thin_cell_is_accumulating_and_reports_how_long():
    cells, _ = allocate(bets(n=20, win_rate=0.70))
    c = cells[0]
    assert c.state == ACCUMULATING and c.fraction == 0.0
    assert c.n_to_target > 0 and c.weeks_to_target is not None


def test_many_bets_on_few_matchdays_do_not_count_as_many_observations():
    """n=200 across 3 Saturdays is nearer 3 observations than 200."""
    d = bets(n=200, win_rate=0.70, days=3)
    cells, _ = allocate(d)
    assert cells[0].state == ACCUMULATING and "matchdays" in cells[0].reason


def test_the_stake_uses_the_CI_FLOOR_not_the_point_estimate():
    """Kelly is linear in the edge with zero intercept, so sizing on a mean sizes a lucky cell
    like a skilled one. Full Kelly on this estate's numbers ruins 98.9% of paths."""
    cells, _ = allocate(bets(n=400, win_rate=0.62, odds=2.0))
    c = cells[0]
    implied_from_mean = (c.roi / (c.mean_odds - 1)) * 0.25
    assert c.fraction < implied_from_mean, "sized on the mean rather than the lower bound"


def test_no_single_cell_can_exceed_its_cap():
    cells, _ = allocate(bets(n=500, win_rate=0.95, odds=3.0))
    assert cells[0].fraction <= MAX_CELL_FRACTION


def test_total_exposure_is_capped_across_cells():
    """Slates correlate; a day where every cell fires must not be a bankroll event."""
    d = pd.concat([bets(n=300, win_rate=0.70, odds=3.0, league=f"L{i}", seed=i)
                   for i in range(20)], ignore_index=True)
    cells, summary = allocate(d)
    assert summary["total_fraction"] <= MAX_TOTAL_FRACTION + 1e-9


def test_the_epoch_excludes_evidence_from_a_broken_pipeline():
    """Performance before the data fixes is not evidence about the current system."""
    old = bets(n=300, win_rate=0.70)
    old["match_date"] = pd.to_datetime("2026-08-01")
    cells, summary = allocate(old, reliable_from="2026-09-24")
    assert summary["rows_used"] == 0 and not cells


def test_valuable_tips_never_size_a_stake():
    """VALUABLE is a collection layer. Sizing from it would allocate real bankroll on bets
    nobody intended to place."""
    d = bets(n=300, win_rate=0.70)
    d["signal_tier"] = "VALUABLE"
    _, summary = allocate(d)
    assert summary["rows_used"] == 0


# ── diagnosis ─────────────────────────────────────────────────────────────────────────────────
def test_a_one_sided_cell_is_identified():
    d = bets(n=80, win_rate=0.35, side_mix=False)
    assert any(r["cause"] == CAUSE_SIDE_BIAS for r in diagnose(d))


def test_a_longshot_drag_is_identified_and_separated_from_the_model():
    """The cheapest defect to fix: it needs a price bound, not a model change."""
    good = bets(n=60, win_rate=0.52, odds=2.0, seed=1)
    bad = bets(n=20, win_rate=0.05, odds=3.2, seed=2)
    r = diagnose(pd.concat([good, bad], ignore_index=True))
    hit = [x for x in r if x["cause"] == CAUSE_ODDS_BAND]
    assert hit and "price bound" in hit[0]["fix"]


def test_overconfidence_is_identified_when_probabilities_are_present():
    d = bets(n=120, win_rate=0.40, odds=2.0)
    d["model_prob"] = 0.60
    hit = [x for x in diagnose(d) if x["cause"] == CAUSE_CALIBRATION]
    assert hit and "recalibrate" in hit[0]["fix"]


def test_a_cell_the_market_prices_better_is_identified():
    d = bets(n=120, win_rate=0.35, odds=2.0)       # break-even 50%, we hit 35%
    assert any(r["cause"] == CAUSE_MARKET_BETTER for r in diagnose(d))


def test_several_causes_are_reported_not_one_verdict():
    """A league can be one-sided AND dragged by longshots; fixing one leaves the other."""
    d = pd.concat([bets(n=60, win_rate=0.40, odds=2.0, side_mix=False, seed=3),
                   bets(n=20, win_rate=0.05, odds=3.2, side_mix=False, seed=4)],
                  ignore_index=True)
    assert len({r["cause"] for r in diagnose(d)}) >= 2


def test_a_thin_cell_is_not_diagnosed():
    assert diagnose(bets(n=10))[0]["cause"] == "UNDIAGNOSED"
