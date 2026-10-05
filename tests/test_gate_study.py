"""§21 — the guarantees that make the gate study trustworthy rather than merely large."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.gate_study as gs

REG = Path(__file__).resolve().parents[1] / "output" / "league_market_gate_registry.json"

#: Columns that are only knowable AFTER the bet. They may grade a signal; they may never select
#: one. A gate fitted on a closing price is a gate that cannot be run.
FUTURE_ONLY = ("closing_odds", "closing_fair", "clv", "clv_pct", "result", "pnl",
               "open_to_close", "current_to_close", "market_moved_toward_model")


# ── 10. retrospective code can never award CONFIRMED ──────────────────────────────────────────
def test_no_confirmed_grade_exists_in_the_retrospective_study():
    src = inspect.getsource(gs)
    assert '"CONFIRMED"' not in src.replace('"CONFIRMED cells: 0', ""), \
        "a retrospective script must not be able to assign CONFIRMED"
    assert gs.FORWARD == "FORWARD_TEST"


@pytest.mark.skipif(not REG.exists(), reason="registry not generated in this checkout")
def test_the_published_registry_contains_no_confirmed_cell():
    cells = json.loads(REG.read_text(encoding="utf-8"))["cells"]
    assert not [c for c in cells if c.get("evidence_grade") == "CONFIRMED"]


# ── 5. the ledger cannot answer "what if I lowered the bar?" ──────────────────────────────────
def test_a_candidate_below_the_production_gate_is_marked_censored():
    """Fixtures under the live floor never entered the ledger, so a lower threshold is
    unobservable there — reporting a number for it would be inventing one."""
    src = inspect.getsource(gs.study_cell)
    assert "LEDGER_CENSORED" in src or "CENSORED" in src
    assert "below the production gate" in src.lower() or "BELOW the production gate" in src


@pytest.mark.skipif(not REG.exists(), reason="registry not generated")
def test_censored_cells_carry_no_live_roi():
    cells = json.loads(REG.read_text(encoding="utf-8"))["cells"]
    for c in cells:
        if c.get("live_status") == gs.CENSORED:
            assert c.get("roi_live") is None, f"{c['league']}|{c['market']} reported a censored ROI"


# ── 6. no closing-line leakage into selection ─────────────────────────────────────────────────
def test_selection_never_filters_on_a_future_column():
    """The only column the threshold sweep may filter on is the edge known at prediction time."""
    src = inspect.getsource(gs.study_cell) + inspect.getsource(gs.main)
    for col in FUTURE_ONLY:
        assert f'best_edge >= ' in src  # the real selector
        assert f'[{col} >=' not in src and f'["{col}"] >=' not in src, \
            f"selection filtered on the future column {col}"


def test_clv_and_result_are_used_only_for_grading():
    """_metrics grades; study_cell selects. CLV must appear in the former, not the latter."""
    assert "clv_pct" in inspect.getsource(gs._metrics)
    sel = inspect.getsource(gs.study_cell)
    assert "clv" not in sel.split("# GRADE")[0].lower().replace("live_clv", "")


# ── 7. block bootstrap groups by matchday ─────────────────────────────────────────────────────
def test_bootstrap_resamples_matchdays_not_individual_bets():
    """Twenty correlated bets on one Saturday are not twenty observations. An i.i.d. bootstrap
    would make every interval too tight, which is how noise becomes a finding."""
    rng = np.random.default_rng(0)
    # 20 matchdays, 10 perfectly-correlated bets each
    day = np.repeat(np.arange(20), 10)
    pnl = np.repeat(rng.normal(0, 1, 20), 10)
    lo_b, hi_b = gs.block_ci(pnl, day, seed=1)
    lo_i, hi_i = gs.block_ci(pnl, np.arange(len(pnl)), seed=1)   # each bet its own "day"
    assert (hi_b - lo_b) > (hi_i - lo_i) * 1.5, \
        "matchday blocking did not widen the interval on correlated data"


def test_bootstrap_refuses_too_few_matchdays():
    assert gs.block_ci(np.array([1.0, -1.0]), np.array([1, 1])) == (None, None)


# ── 8. market / model / league isolation ──────────────────────────────────────────────────────
@pytest.mark.skipif(not REG.exists(), reason="registry not generated")
def test_every_cell_is_keyed_by_all_three_dimensions():
    cells = json.loads(REG.read_text(encoding="utf-8"))["cells"]
    keys = [(c["league"], c["market"], c["model_type"]) for c in cells]
    assert len(keys) == len(set(keys)), "a cell was emitted twice — dimensions are being pooled"
    assert len({m for _, m, _ in keys}) > 1, "only one market present — markets are pooled"


@pytest.mark.skipif(not REG.exists(), reason="registry not generated")
def test_one_league_can_hold_different_gates_per_market():
    cells = json.loads(REG.read_text(encoding="utf-8"))["cells"]
    by = {}
    for c in cells:
        by.setdefault(c["league"], set()).add(c["production_sniper"])
    assert any(len(v) > 1 for v in by.values()), \
        "every market in every league resolved to one gate — the market dimension is inert"


# ── 9. a declining edge curve is never 'optimised' ────────────────────────────────────────────
def test_a_declining_curve_returns_not_ranking_rather_than_an_argmax():
    assert gs.NOT_RANKING == "EDGE_NOT_RANKING_OUTCOMES"
    src = inspect.getsource(gs.study_cell)
    i_mono, i_best = src.index("mono < 0"), src.index("cdf.roi.idxmax")
    assert i_mono < i_best, "the argmax is taken before the monotonicity check"


# ── 13. regime ids ────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not REG.exists(), reason="registry not generated")
def test_every_cell_records_the_regime_that_produced_it():
    cells = json.loads(REG.read_text(encoding="utf-8"))["cells"]
    assert all(c.get("regime_id") for c in cells)
    assert len({c["regime_id"] for c in cells}) == 1, "cells mixed across threshold regimes"


# ── break-even arithmetic ─────────────────────────────────────────────────────────────────────
def test_break_even_is_mean_of_inverse_odds():
    """mean(1/odds), never 1/mean(odds) — they differ by Jensen's inequality and the wrong one
    flattered the SNIPER tier by 1.5pp on this estate."""
    d = pd.DataFrame({"pnl": [1.0, -1.0], "odds": [1.5, 4.5],
                      "result": ["WIN", "LOSS"],
                      "date": pd.to_datetime(["2026-09-01", "2026-09-02"])})
    m = gs._metrics(d)
    assert m["break_even"] == pytest.approx(np.mean([1 / 1.5, 1 / 4.5]), abs=1e-4)  # 4dp
    assert m["break_even"] != pytest.approx(1 / np.mean([1.5, 4.5]), abs=1e-4)
