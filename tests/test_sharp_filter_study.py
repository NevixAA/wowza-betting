"""Unit tests of the sharp-filter study's timing and signal rules (synthetic rows, no licensed data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sfs", ROOT / "scripts" / "sharp_filter_study.py")
sfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfs)

P = {"signal": {"quantile": 0.5, "min_history": 3, "sharp_book": "pinnacle", "ou_reference_line": 2.5,
                "consensus_min_books": 2}}
T0 = pd.Timestamp("2025-01-10 12:00", tz="UTC")


def test_threshold_uses_strictly_earlier_dates_only():
    m = pd.DataFrame({"league": "L", "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                                           "2025-01-04", "2025-01-04", "2025-01-05"]),
                      "pin": [0.01, 0.02, 0.03, 0.50, 0.60, 0.0]})
    thr = sfs.thresholds(m, "pin", [], P)
    assert thr.iloc[:3].isna().all()                       # WARMUP: fewer than 3 earlier fixtures
    assert thr.iloc[3] == thr.iloc[4] == pytest.approx(0.02)   # same-day rows never see each other
    assert thr.iloc[5] == pytest.approx(0.03)              # median of 0.01, 0.02, 0.03, 0.50, 0.60


def test_threshold_is_per_league():
    m = pd.DataFrame({"league": ["A"] * 4 + ["B"],
                      "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]),
                      "pin": [0.01, 0.01, 0.01, 0.0, 0.0]})
    thr = sfs.thresholds(m, "pin", [], P)
    assert thr.iloc[3] == pytest.approx(0.01) and np.isnan(thr.iloc[4])


def test_direction_and_line_moves():
    v = pd.Series([0.05, -0.05, 0.01, np.inf, -np.inf, np.nan, 0.05])
    thr = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, np.nan])
    assert sfs.direction(v, thr).tolist()[:5] == [1.0, -1.0, 0.0, 1.0, -1.0]
    assert np.isnan(sfs.direction(v, thr).iloc[5]) and np.isnan(sfs.direction(v, thr).iloc[6])


def test_classify_both_requires_agreement():
    pin = pd.Series([1.0, 1.0, 0.0, np.nan])
    cons = pd.Series([1.0, -1.0, 0.0, 1.0])
    assert sfs.classify_signal(pin, cons, "BOTH").tolist() == ["AGREE", "MIXED", "NONE", "NO_DATA"]
    assert sfs.classify_signal(pin, cons, "PIN").tolist() == ["AGREE", "AGREE", "NONE", "NO_DATA"]


def test_window_bet_is_never_before_signal_and_next_is_strictly_later():
    leads = [73, 49, 25, 13, 6.5, 2.2, 1.1]
    snaps = pd.DataFrame({"fixture_id": "f", "ts": [T0 - pd.Timedelta(hours=h) for h in leads], "lead_h": leads})
    W = {"open_lead_h": [60, 84], "open_target_h": 72, "signal_lead_h": [4, 10], "signal_target_h": 6,
         "bet": "signal_snapshot"}
    t = sfs.window_times(snaps, W)
    assert t["lead_open"].iloc[0] == 73 and t["lead_sig"].iloc[0] == 6.5 and t["ts_bet"].iloc[0] == t["ts_sig"].iloc[0]
    t2 = sfs.window_times(snaps, W | {"bet": "next_snapshot"})
    assert t2["ts_bet"].iloc[0] > t2["ts_sig"].iloc[0] and t2["lead_bet"].iloc[0] == 2.2


def test_assert_outside_repo_refuses_repo_paths(tmp_path):
    with pytest.raises(SystemExit):
        sfs.assert_outside_repo(ROOT / "output" / "x")
    assert sfs.assert_outside_repo(tmp_path / "w").exists()
