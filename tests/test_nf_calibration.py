"""NF O/U 2.5 recalibration: fitted on post-fix outcomes, applied only when proven (owner 2026-11-01).

    python -m pytest tests/test_nf_calibration.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import nf_calibration as C


def _log(tmp_path, n, p_model, true_rate, when="2026-11-05T12:00:00Z", seed=1):
    rng = np.random.default_rng(seed)
    over = rng.uniform(size=n) < true_rate
    pd.DataFrame({"model_type": "new_format", "market": "ou25", "logged_at": when, "p_v9": p_model,
                  "home_goals": np.where(over, 2, 1), "away_goals": np.where(over, 1, 0)}
                 ).to_csv(tmp_path / "s.csv", index=False)
    return tmp_path / "s.csv"


def test_applies_when_enough_evidence_and_moves_toward_outcomes(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58))
    assert cal["active"] and cal["shift"] > 0
    out = C.apply(pd.Series([0.48]), cal).iloc[0]
    assert 0.55 < out < 0.61


def test_not_applied_below_min_n(tmp_path):
    cal = C.fit(_log(tmp_path, 60, 0.48, 0.58))
    assert not cal["active"]
    assert C.apply(pd.Series([0.48]), cal).iloc[0] == 0.48


def test_not_applied_when_already_calibrated(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.58, 0.58))
    assert not cal["active"]


def test_rows_before_the_fixed_pipeline_are_ignored(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58, when="2026-10-15T12:00:00Z"))
    assert cal["n"] == 0 and not cal["active"]


def test_rollback_switch(tmp_path, monkeypatch):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58))
    monkeypatch.setenv("NF_CALIBRATION", "0")
    assert C.apply(pd.Series([0.48]), cal).iloc[0] == 0.48
