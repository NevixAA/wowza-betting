"""NF O/U 2.5 recalibration: fitted on post-fix outcomes, applied only when proven (owner 2026-11-01).

    python -m pytest tests/test_nf_calibration.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import nf_calibration as C


def _reg(tmp_path, since="2026-10-25T08:46:05Z"):
    """A champion registry whose serving NF champion was installed at `since` (None: no key)."""
    import json
    body = {"newformat": {"champion_since": since}} if since is not None else {"standard": {}}
    (tmp_path / "champ.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path / "champ.json"


def _log(tmp_path, n, p_model, true_rate, when="2026-11-05T12:00:00Z", seed=1):
    rng = np.random.default_rng(seed)
    over = rng.uniform(size=n) < true_rate
    pd.DataFrame({"model_type": "new_format", "market": "ou25", "logged_at": when, "p_v9": p_model,
                  "home_goals": np.where(over, 2, 1), "away_goals": np.where(over, 1, 0)}
                 ).to_csv(tmp_path / "s.csv", index=False)
    return tmp_path / "s.csv"


def test_applies_when_enough_evidence_and_moves_toward_outcomes(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58), champions=_reg(tmp_path))
    assert cal["active"] and cal["shift"] > 0
    out = C.apply(pd.Series([0.48]), cal).iloc[0]
    assert 0.55 < out < 0.61


def test_not_applied_below_min_n(tmp_path):
    cal = C.fit(_log(tmp_path, 60, 0.48, 0.58), champions=_reg(tmp_path))
    assert not cal["active"]
    assert C.apply(pd.Series([0.48]), cal).iloc[0] == 0.48


def test_not_applied_when_already_calibrated(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.58, 0.58), champions=_reg(tmp_path))
    assert not cal["active"]


def test_rows_before_the_fixed_pipeline_are_ignored(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58, when="2026-10-15T12:00:00Z"), champions=_reg(tmp_path))
    assert cal["n"] == 0 and not cal["active"]


def test_rollback_switch(tmp_path, monkeypatch):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58), champions=_reg(tmp_path))
    monkeypatch.setenv("NF_CALIBRATION", "0")
    assert C.apply(pd.Series([0.48]), cal).iloc[0] == 0.48


# ── the shift belongs to the model it was fitted against (owner 2026-11-09, clean promotion) ──

def test_rows_from_the_previous_champion_are_ignored_after_a_promotion(tmp_path):
    """A new NF champion installed on 2026-11-20: rows the OLD model predicted (11-05) never fit
    the new model's shift."""
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58), champions=_reg(tmp_path, "2026-11-20T00:00:00Z"))
    assert cal["n"] == 0 and not cal["active"]
    assert cal["since"] == "2026-11-20T00:00:00Z" and cal["since_basis"] == "serving champion installed"


def test_rows_from_the_new_champion_are_used(tmp_path):
    cal = C.fit(_log(tmp_path, 400, 0.48, 0.58, when="2026-11-25T12:00:00Z"),
                champions=_reg(tmp_path, "2026-11-20T00:00:00Z"))
    assert cal["n"] == 400 and cal["active"]


def test_unreadable_champion_registry_fits_nothing(tmp_path):
    for reg in (_reg(tmp_path, None), tmp_path / "missing.json"):
        cal = C.fit(_log(tmp_path, 400, 0.48, 0.58), champions=reg)
        assert not cal["active"] and cal["n"] == 0 and cal["since"] is None
        assert C.apply(pd.Series([0.48]), cal).iloc[0] == 0.48


def test_the_committed_registry_window_is_readable():
    window, basis = C.fit_window()
    assert window is not None, basis
