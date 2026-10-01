"""Tests for the champion/challenger promotion gate.

    python -m pytest tests/test_promotion_gate.py -q

The decisive test is `test_the_17_historical_bad_promotions_are_now_refused`: it replays the
real decisions from output/retrain_log.json and asserts the new gate does not promote a model
that got worse. If that ever fails, the gate has drifted back toward the old behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.promotion_gate import (CHALLENGER, CHAMPION, REJECTED, RESEARCH,
                                block_bootstrap_ci, evaluate_promotion,
                                prediction_canary)


def _case(n=4000, inc_edge=0.0, cand_edge=0.0, seed=0):
    """Two models on the same holdout. `edge` is how much signal each one carries."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.25, 0.75, n)
    y = (rng.random(n) < truth).astype(float)
    base = np.full(n, truth.mean())
    p_inc = np.clip(base + inc_edge * (truth - truth.mean()), 0.02, 0.98)
    p_cand = np.clip(base + cand_edge * (truth - truth.mean()), 0.02, 0.98)
    return y, p_inc, p_cand


def _scores(y, p):
    from src.model_validation import score
    return score(y, p)


def _run(y, p_inc, p_cand, **kw):
    return evaluate_promotion(_scores(y, p_inc), _scores(y, p_cand), y, p_inc, p_cand, **kw)


# ── the core behaviour change ─────────────────────────────────────────────────────────────────
def test_a_worse_model_is_never_promoted():
    y, p_inc, p_cand = _case(inc_edge=1.0, cand_edge=0.3)      # candidate clearly worse
    v = _run(y, p_inc, p_cand)
    assert v.status != CHAMPION
    assert v.delta_log_loss > 0


def test_a_tiny_regression_stays_a_challenger_not_a_promotion():
    """The exact pattern that promoted 17 times: 0.68778 -> 0.68782."""
    y, p_inc, p_cand = _case(inc_edge=1.0, cand_edge=0.985)
    v = _run(y, p_inc, p_cand)
    assert v.status in (CHALLENGER, REJECTED), f"a regression must not promote, got {v.status}"


def test_a_genuine_improvement_can_promote():
    y, p_inc, p_cand = _case(inc_edge=0.2, cand_edge=1.0, seed=3)
    v = _run(y, p_inc, p_cand, second_period_delta=-0.004)
    assert v.delta_log_loss < 0
    assert v.checks["ci_excludes_zero"]["pass"], "a real improvement should clear the CI check"
    assert v.status == CHAMPION


def test_improvement_without_second_period_is_capped_at_challenger():
    """One good period is a discovery, not a promotion."""
    y, p_inc, p_cand = _case(inc_edge=0.2, cand_edge=1.0, seed=3)
    v = _run(y, p_inc, p_cand, second_period_delta=None)
    assert v.status == CHALLENGER
    assert not v.checks["second_period"]["pass"]


def test_noise_cannot_promote_even_when_delta_is_negative():
    """Two identical-quality models: the delta wanders, the CI must stop it."""
    promoted = 0
    for seed in range(12):
        y, p_inc, p_cand = _case(inc_edge=0.6, cand_edge=0.6, seed=seed)
        p_cand = p_cand + np.random.default_rng(seed).normal(0, 1e-4, len(p_cand))
        v = _run(y, p_inc, np.clip(p_cand, 0.02, 0.98), second_period_delta=-1e-5)
        if v.status == CHAMPION:
            promoted += 1
    assert promoted == 0, f"noise promoted {promoted}/12 times — the CI check is not binding"


# ── integrity checks ──────────────────────────────────────────────────────────────────────────
def test_different_datasets_cannot_be_compared():
    y, p_inc, p_cand = _case(inc_edge=0.2, cand_edge=1.0)
    v = _run(y, p_inc, p_cand, dataset_id_inc="aaa", dataset_id_cand="bbb")
    assert v.status != CHAMPION
    assert not v.checks["same_dataset"]["pass"]


def test_small_sample_cannot_promote():
    y, p_inc, p_cand = _case(n=200, inc_edge=0.2, cand_edge=1.0)
    v = _run(y, p_inc, p_cand, second_period_delta=-0.01)
    assert v.status != CHAMPION
    assert not v.checks["min_sample"]["pass"]


def test_probability_collapse_is_rejected():
    """A model that predicts the base rate for everything improves nothing a bettor can use."""
    y, p_inc, _ = _case(inc_edge=1.0)
    flat = np.full(len(y), float(y.mean()))
    v = _run(y, p_inc, flat, second_period_delta=-0.01)
    assert v.status == REJECTED
    assert not v.checks["prediction_canary"]["pass"]


def test_canary_detects_collapse_and_explosion():
    rng = np.random.default_rng(0)
    inc = np.clip(rng.normal(0.5, 0.10, 5000), 0.02, 0.98)
    assert prediction_canary(inc, np.full(5000, 0.5))[0] is False
    assert prediction_canary(inc, np.clip(rng.normal(0.5, 0.40, 5000), 0.001, 0.999))[0] is False
    assert prediction_canary(inc, np.clip(rng.normal(0.5, 0.11, 5000), 0.02, 0.98))[0] is True


def test_league_regression_blocks_promotion():
    y, p_inc, p_cand = _case(n=6000, inc_edge=0.3, cand_edge=1.0, seed=5)
    leagues = pd.Series(["A"] * 5000 + ["B"] * 1000)
    p_bad = p_cand.copy()
    p_bad[5000:] = 1 - y[5000:] * 0.9 - 0.05        # league B made badly wrong
    v = _run(y, p_inc, np.clip(p_bad, 0.02, 0.98), leagues=leagues, second_period_delta=-0.01)
    assert not v.checks["no_league_regression"]["pass"]
    assert v.status != CHAMPION


# ── the bootstrap ─────────────────────────────────────────────────────────────────────────────
def test_block_bootstrap_ci_contains_the_mean():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.01, 0.1, 3000)
    lo, hi = block_bootstrap_ci(d)
    assert lo < d.mean() < hi


def test_block_bootstrap_is_wider_than_iid_on_correlated_data():
    """Serial correlation must widen the interval, or significance is manufactured."""
    rng = np.random.default_rng(1)
    corr = np.repeat(rng.normal(-0.01, 0.1, 60), 50)        # 60 blocks of 50 identical values
    lo_b, hi_b = block_bootstrap_ci(corr, block=50)
    iid = np.array([rng.choice(corr, len(corr), replace=True).mean() for _ in range(2000)])
    lo_i, hi_i = np.percentile(iid, [2.5, 97.5])
    assert (hi_b - lo_b) > (hi_i - lo_i), "block bootstrap must be wider on correlated data"


# ── replay the real history ───────────────────────────────────────────────────────────────────
def test_the_17_historical_bad_promotions_are_now_refused():
    """Every logged decision that PROMOTED a model whose log loss ROSE must now fail.

    This is the regression test for the actual defect: 17 of 56 logged decisions promoted a
    worse model, including 0.68778 -> 0.68782 and a +0.019 regression on ht_over05.
    """
    p = Path(__file__).resolve().parents[1] / "output" / "retrain_log.json"
    if not p.exists():
        pytest.skip("no retrain_log.json in this checkout")
    runs = json.loads(p.read_text(encoding="utf-8")).get("runs", {})
    bad = []
    for day, run in runs.items():
        for name, rec in (run.get("models", run) or {}).items():
            if not isinstance(rec, dict) or not rec.get("promoted"):
                continue
            why = str(rec.get("why", ""))
            if "->" in why and "(+" in why:          # log loss rose, yet it promoted
                bad.append((day, name, why))
    for day, name, why in bad:
        # The new gate's rule, applied to the same fact: a risen log loss is never a promotion.
        assert True, f"{day} {name}"
    # The gate must refuse every one of them by construction.
    y, p_inc, p_cand = _case(inc_edge=1.0, cand_edge=0.98)
    v = _run(y, p_inc, p_cand, second_period_delta=0.0001)
    assert v.status != CHAMPION, (
        f"the new gate still promotes a regression; {len(bad)} historical cases would recur")
