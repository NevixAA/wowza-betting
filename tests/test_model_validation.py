"""Tests for the leak-free validation harness and the promotion gate.

    python -m pytest tests/test_model_validation.py -q

These are the tests the estate did not have. The meta-stack leak survived months of green runs
because nothing ever asserted that blocks do not overlap or that a leaked model scores
suspiciously well. A harness that cannot detect leakage is not evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_validation import (BLOCKS, MIN_BLOCK, calibration_line,
                                  chronological_split, dataset_id, ece, score)


def _frame(n=4000, seed=0):
    """Synthetic fixtures with a real but modest signal, in date order."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.6 * x)))
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="6h"),
        "feat": x,
        "over25": (rng.random(n) < p).astype(float),
    })


# ── the split itself ──────────────────────────────────────────────────────────────────────────
def test_blocks_do_not_overlap_and_cover_everything():
    sp = chronological_split(_frame(), "over25", ["feat"])
    rngs = [sp.fit, sp.cal, sp.meta, sp.holdout]
    for (a1, b1), (a2, b2) in zip(rngs, rngs[1:]):
        assert b1 == a2, "blocks must be contiguous with no gap and no overlap"
    assert rngs[0][0] == 0 and rngs[-1][1] == sp.n, "blocks must cover the whole frame"


def test_split_is_chronological_not_random():
    """Every block must end before the next begins, IN TIME. A random split would fail this."""
    d = _frame().sort_values("date").reset_index(drop=True)
    sp = chronological_split(d, "over25", ["feat"])
    for (_, b1), (a2, _) in zip([sp.fit, sp.cal, sp.meta], [sp.cal, sp.meta, sp.holdout]):
        assert d["date"].iloc[b1 - 1] <= d["date"].iloc[a2], "a later block starts before an earlier one ends"


def test_split_is_deterministic():
    """Same input twice -> identical split. A non-stable sort would break this on date ties."""
    d = _frame()
    assert chronological_split(d, "over25", ["feat"]) == chronological_split(d, "over25", ["feat"])


def test_tied_dates_cannot_permute_the_split():
    """All-identical dates: the split must still be reproducible, not shuffled by the sort."""
    d = _frame(1000)
    d["date"] = pd.Timestamp("2024-01-01")
    a = chronological_split(d, "over25", ["feat"])
    b = chronological_split(d.copy(), "over25", ["feat"])
    assert a == b


def test_small_frame_is_refused_not_silently_scored():
    sp = chronological_split(_frame(100), "over25", ["feat"])
    assert not sp.ok(), "a frame too small for MIN_BLOCK must report not-ok rather than a number"


# ── dataset identity ──────────────────────────────────────────────────────────────────────────
def test_dataset_id_changes_when_the_data_changes():
    """Two results are only comparable if this matches — the production gate's missing guard."""
    d = _frame()
    base = dataset_id(d, "over25", ["feat"])
    assert dataset_id(d.iloc[:-50], "over25", ["feat"]) != base, "fewer rows must change the id"
    flipped = d.copy()
    flipped.loc[flipped.index[:200], "over25"] = 1 - flipped["over25"].iloc[:200]
    assert dataset_id(flipped, "over25", ["feat"]) != base, "different labels must change the id"
    assert dataset_id(d, "over25", ["feat"]) == base, "same data must give the same id"


# ── metrics ───────────────────────────────────────────────────────────────────────────────────
def test_ece_zero_for_perfect_calibration():
    y = np.array([0.0] * 500 + [1.0] * 500)
    p = np.array([0.0] * 500 + [1.0] * 500)
    assert ece(y, p) < 1e-9


def test_ece_detects_overconfidence():
    """Claiming 0.9 and realising 0.5 must register, since this is the estate's real defect."""
    y = np.array([1.0] * 500 + [0.0] * 500)
    p = np.full(1000, 0.9)
    assert ece(y, p) > 0.35


def test_calibration_slope_below_one_means_overconfident():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.2, 0.8, 8000)
    y = (rng.random(8000) < true).astype(float)
    over = np.clip(0.5 + (true - 0.5) * 2.2, 0.01, 0.99)   # spread wider than reality
    slope, _ = calibration_line(y, over)
    assert slope < 1.0, "an over-dispersed model must show a calibration slope below 1"


def test_score_reports_sample_size_with_every_metric():
    y = np.array([0.0, 1.0] * 500)
    s = score(y, np.full(1000, 0.5))
    assert s["n"] == 1000
    for k in ("log_loss", "brier", "ece", "cal_slope", "claimed", "realised"):
        assert k in s


# ── THE LEAKAGE TEST ──────────────────────────────────────────────────────────────────────────
def test_a_model_fitted_on_its_own_scoring_block_looks_better_than_it_is():
    """This is the bug in src/model.py, reduced to its essence.

    A stacker fitted on the labels it is then scored against must beat one fitted on a separate
    block. If this assertion ever fails, the harness has stopped being able to see leakage and
    no result it produces can be trusted.
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(7)
    n = 2000
    # Three base predictions that are pure noise: an honest stacker can do nothing with them.
    P_meta, P_hold = rng.random((n, 3)), rng.random((n, 3))
    y_meta, y_hold = (rng.random(n) < 0.5).astype(float), (rng.random(n) < 0.5).astype(float)

    honest = LogisticRegression(max_iter=500).fit(P_meta, y_meta)
    leaky = LogisticRegression(max_iter=500).fit(P_hold, y_hold)      # fitted on the test labels

    ll_honest = score(y_hold, honest.predict_proba(P_hold)[:, 1])["log_loss"]
    ll_leaky = score(y_hold, leaky.predict_proba(P_hold)[:, 1])["log_loss"]
    assert ll_leaky < ll_honest, (
        "a leaked fit must score better on pure noise — if it does not, this harness can no "
        "longer detect the defect it was built for")


def test_future_rows_never_appear_in_an_earlier_block():
    """Guards the one thing a chronological split exists to guarantee."""
    d = _frame(5000)
    sp = chronological_split(d, "over25", ["feat"])
    d = d.sort_values("date", kind="mergesort").reset_index(drop=True)
    holdout_start = d["date"].iloc[sp.holdout[0]]
    for name, rng in (("fit", sp.fit), ("cal", sp.cal), ("meta", sp.meta)):
        assert d["date"].iloc[rng[0]:rng[1]].max() <= holdout_start, \
            f"{name} block contains a fixture at or after the holdout began"
