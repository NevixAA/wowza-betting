"""R2-C finding 18: train/serve feature parity. The pre-2026-11-01 NF defect (home_advantage built
as a constant 1.0 live, a per-team rolling rate in training) must be impossible to miss.

    python -m pytest tests/test_integrity_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integrity import parity as PT  # noqa: E402

rng = np.random.default_rng(7)
TRAIN = pd.DataFrame({"home_advantage": rng.uniform(0, 1, 5000),
                      "home_scored_last5": rng.gamma(4, 0.33, 5000),
                      "is_late_season": rng.integers(0, 2, 5000).astype(float),
                      "home_rest_days": rng.integers(3, 10, 5000).astype(float)})


def _live(n=40, **over):
    d = pd.DataFrame({"home_advantage": rng.uniform(0, 1, n),
                      "home_scored_last5": rng.gamma(4, 0.33, n),
                      "is_late_season": rng.integers(0, 2, n).astype(float),
                      "home_rest_days": rng.integers(3, 10, n).astype(float)})
    for k, v in over.items():
        d[k] = v
    return d


def test_constant_live_home_advantage_is_fail():
    rep = PT.parity_report(TRAIN, _live(home_advantage=1.0), list(TRAIN.columns), "new_format", "m")
    f = rep["features"]["home_advantage"]
    assert f["status"] == "FAIL" and f["constant_live"] is True
    assert rep["status"] == "FAIL"


@pytest.mark.parametrize("feat", ["home_scored_last5", "home_rest_days", "home_advantage"])
def test_any_feature_varying_in_training_but_constant_live_fails(feat):
    live = _live(**{feat: float(TRAIN[feat].median())})       # even a PLAUSIBLE constant fails
    assert PT.parity_report(TRAIN, live, [feat], "t", "m")["features"][feat]["status"] == "FAIL"


def test_matching_distributions_pass():
    rep = PT.parity_report(TRAIN, _live(400), ["home_advantage", "home_scored_last5"], "t", "m")
    assert rep["status"] == "PASS", rep


def test_calendar_feature_constant_live_is_capped_at_warn():
    rep = PT.parity_report(TRAIN, _live(is_late_season=0.0), ["is_late_season"], "t", "m")
    f = rep["features"]["is_late_season"]
    assert f["status"] == "WARN" and f["expected_rule"] == "calendar"


def test_all_missing_live_fails_and_mostly_imputed_live_fails():
    rep = PT.parity_report(TRAIN, _live(home_rest_days=np.nan), ["home_rest_days"], "t", "m")
    assert rep["features"]["home_rest_days"]["status"] == "FAIL"
    live = _live(20)
    live.loc[live.index[:14], "home_rest_days"] = np.nan                 # 70% missing vs 0%
    rep = PT.parity_report(TRAIN, live, ["home_rest_days"], "t", "m")
    assert rep["features"]["home_rest_days"]["status"] == "FAIL"


def test_feature_absent_from_board_fails():
    rep = PT.parity_report(TRAIN, _live().drop(columns=["home_advantage"]), ["home_advantage"], "t", "m")
    assert rep["features"]["home_advantage"]["status"] == "FAIL"


def test_psi_basics():
    assert PT.psi(TRAIN["home_advantage"], TRAIN["home_advantage"]) == pytest.approx(0.0, abs=1e-9)
    assert PT.psi(TRAIN["home_advantage"], TRAIN["home_advantage"] + 5) > 1.0
    assert PT.psi(pd.Series([], dtype=float), TRAIN["home_advantage"]) is None


@pytest.mark.skipif(not (ROOT / "output" / "predictions.csv").exists(), reason="no board")
def test_committed_board_nf_home_advantage_is_not_constant():
    """Regression on production output: the NF board must never again serve one home_advantage."""
    b = pd.read_csv(ROOT / "output" / "predictions.csv", low_memory=False)
    nf = b[b.get("model_type") == "new_format"]
    if len(nf) < PT.MIN_LIVE_N or "home_advantage" not in nf:
        pytest.skip("board has too few new-format fixtures")
    if "serving_recipe" in nf and set(nf["serving_recipe"].dropna()) - {"legacy_v9"}:
        pytest.skip("NF not on the legacy serving path")
    s = PT.feature_stats(pd.Series(rng.uniform(0, 1, 100)), nf["home_advantage"])
    v, _ = PT.classify("home_advantage", s)
    assert not s["constant_live"] and v != "FAIL"
