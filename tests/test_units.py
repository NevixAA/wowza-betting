"""src/units.py: exact P/L totals and half-up display (issue #19, -8.445u shown as -8.44 or -8.45)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import units as U  # noqa: E402


def test_total_is_order_independent_and_exact():
    vals = [0.1] * 10 + [-1.0]                       # float sum = -1.1102e-16, exact = 0
    assert U.total(vals) == 0.0 and U.total(list(reversed(vals))) == 0.0
    s = pd.Series([1.075, -0.95, -1.0, 2.1, -9.67])   # exact -8.445
    for order in (s, s[::-1], s.sample(frac=1, random_state=3)):
        assert U.total(order) == -8.445
        assert U.fmt(U.total(order)) == "-8.45u"


def test_float_sum_of_the_same_values_can_round_the_other_way():
    # the defect being fixed: a float total of an exact half lands either side of it
    vals = [0.1] * 3 + [-0.315]                      # exact -0.015
    assert f"{sum(vals):+.2f}u" == "-0.01u"          # float: -0.014999999999999958
    assert U.fmt(U.total(vals)) == "-0.02u"


def test_total_skips_missing_like_pandas():
    assert U.total([1.0, None, np.nan, "x", -0.5]) == 0.5
    assert U.total([]) == 0.0 and U.total(pd.Series([], dtype=float)) == 0.0


def test_home_performance_totals_are_exact():
    """dashboard_data.today_view.performance (Home + Tips staked P/L) sums exactly: the float sum
    of these rows is -0.01499999..., which displayed as -0.01u; the stored values add to -0.015."""
    from dashboard_data import today_view
    s = pd.DataFrame({"signal_tier": ["SNIPER", "MARKSMAN", "SNIPER", "SNIPER", "VALUABLE"],
                      "_mkt": ["O/U 2.5"] * 5, "_pnl": [0.1, 0.1, 0.1, -0.315, 0.1],
                      "_dt": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-04"]),
                      # THE staked definition (owner 2026-11-04): a recorded policy stake, not the tier
                      "_staked": [True, True, True, True, False], "_stake_units": [1.0, 1.0, 1.0, 1.0, None]})
    perf = today_view.performance(s, today="2026-10-05")
    assert perf["pnl"] == -0.015 and U.fmt(perf["pnl"]) == "-0.02u"
    assert perf["last7_pnl"] == -0.015 and perf["drawdown"] == -0.315
    assert perf["by_market"]["P/L"].tolist() == [-0.015]
    assert perf["curve"]["cum"].tolist()[-1] == -0.015


def test_fmt_half_up_sign_suffix_places():
    assert U.fmt(-8.445) == "-8.45u"
    assert U.fmt(8.445) == "+8.45u"
    assert U.fmt(0.005) == "+0.01u"
    assert U.fmt(-0.004) == "+0.00u"                 # never "-0.00u"
    assert U.fmt(-0.0105, places=3) == "-0.011u"
    assert U.fmt(2.5, places=1, sign=False, suffix="") == "2.5"
    assert U.fmt(np.float64(-1.235)) == "-1.24u"
    assert U.fmt(None) == U.MISSING and U.fmt(np.nan) == U.MISSING
