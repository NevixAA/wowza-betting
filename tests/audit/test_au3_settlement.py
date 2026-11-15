"""Adversarial tests for the AU3 independent settlement / ROI helpers (scripts/audit/au3_*.py).

Each test FAILS if the audit code would: invert an UNDER, mis-map a line, count a missing outcome as a
loss, accept an ambiguous club match, swap home/away, or let a NaN P&L silently become 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "audit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from au3_common import roi_ci  # noqa: E402
from au3_settle import Resolver, outcome  # noqa: E402


@pytest.mark.parametrize("hg,ag,side,want", [
    (1, 1, "UNDER", "WIN"), (2, 0, "UNDER", "WIN"), (2, 1, "UNDER", "LOSS"), (0, 0, "UNDER", "WIN"),
    (2, 1, "OVER", "WIN"), (1, 1, "OVER", "LOSS"), (3, 3, "OVER", "WIN"),
])
def test_ou25_line_and_side(hg, ag, side, want):
    assert outcome("ou25", side, hg, ag) == want


def test_under_is_not_one_minus_over_inverted():
    # every score: exactly one of OVER/UNDER wins (no push on a .5 line)
    for hg in range(6):
        for ag in range(6):
            assert {outcome("ou25", "OVER", hg, ag), outcome("ou25", "UNDER", hg, ag)} == {"WIN", "LOSS"}


@pytest.mark.parametrize("m,hg,ag,want", [
    ("btts", 1, 1, "WIN"), ("btts", 3, 0, "LOSS"), ("btts", 0, 0, "LOSS"),
    ("over15", 1, 0, "LOSS"), ("over15", 1, 1, "WIN"),
    ("over35", 2, 1, "LOSS"), ("over35", 2, 2, "WIN"),
])
def test_side_markets(m, hg, ag, want):
    assert outcome(m, "YES", hg, ag) == want


def test_unknown_side_is_not_settled():
    assert outcome("ou25", "AVOID", 1, 1) == "NO_SIDE"


def _res(rows):
    d = pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "hg", "ag", "src"])
    d["date"] = pd.to_datetime(d["date"])
    return Resolver(d)


def test_swapped_teams_do_not_match():
    R = _res([("L", "2026-10-01", "Alpha FC", "Beta United", 2, 0, "af")])
    sc, why = R.find("L", "2026-10-01", "Beta United", "Alpha FC")
    assert sc is None


def test_ambiguous_fuzzy_match_is_refused():
    R = _res([("L", "2026-10-01", "Gamba Osaka", "Kashima Antlers", 1, 0, "af"),
              ("L", "2026-10-01", "Cerezo Osaka", "Kashima Antlers", 0, 0, "af")])
    sc, why = R.find("L", "2026-10-01", "Osaka", "Kashima")
    assert sc is None and why in ("AMBIGUOUS", "NAME_UNRESOLVED", "NO_SCORE_ROW")


def test_missing_result_is_unresolved_not_loss():
    R = _res([("L", "2026-10-01", "Alpha FC", "Beta United", 2, 0, "af")])
    sc, why = R.find("L", "2026-10-09", "Alpha FC", "Beta United")
    assert sc is None and why != "OK"


def test_other_league_never_used():
    R = _res([("L1", "2026-10-01", "Alpha FC", "Beta United", 2, 0, "af")])
    sc, why = R.find("L2", "2026-10-01", "Alpha FC", "Beta United")
    assert sc is None and why == "NO_LEAGUE_IN_RESULTS"


def test_roi_counts_only_given_rows_and_rejects_nan_silently_as_zero():
    pnl = np.array([1.0, -1.0, np.nan])
    r = roi_ci(pnl[~np.isnan(pnl)])
    assert r["n"] == 2 and r["roi"] == 0.0
    r2 = roi_ci(pnl)          # NaN must propagate, never become 0
    assert np.isnan(r2["pnl"]) or np.isnan(r2["roi"])


def test_roi_formula():
    r = roi_ci([1.1, -1.0, -1.0, 0.9])
    assert r["n"] == 4 and abs(r["roi"] - 0.0) < 1e-12
