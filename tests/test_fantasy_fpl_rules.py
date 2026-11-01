"""Fantasy scoring uses FPL's real rules (owner fix 2026-11-01: DC thresholds, GKP goal points).

    python -m pytest tests/test_fantasy_fpl_rules.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from player_model import fantasy as F
from player_model import fantasy_distribution as FD

BOOT = Path(__file__).resolve().parents[1] / "output" / "fpl_bootstrap.json"


def test_goal_points_match_fpl_game_config():
    if not BOOT.exists():
        pytest.skip("no bootstrap")
    goals = json.loads(BOOT.read_text(encoding="utf-8"))["game_config"]["scoring"]["goals_scored"]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert F.FPL_GOAL_PTS[pos] == goals[pos] == FD.GOAL_PTS[pos], pos


def test_dc_thresholds_are_fpls():
    assert F._DC_THRESH["DEF"] == 10 and F._DC_THRESH["MID"] == 12 and F._DC_THRESH["FWD"] == 12


@pytest.mark.parametrize("mean,thr,lo,hi", [
    (12.0, 10.0, 1.2, 2.0),     # a DEF averaging 12 actions clears 10 most weeks
    (5.0, 12.0, 0.0, 0.05),     # a FWD averaging 5 almost never reaches 12
])
def test_dc_points_follow_the_real_bar(mean, thr, lo, hi):
    assert lo <= F._poisson_dc_pts(mean, thr) <= hi


def test_p_start_constant_is_still_what_the_data_says():
    import pandas as pd
    from player_model.fantasy_residual import P_START_GIVEN_1_59
    p = Path(__file__).resolve().parents[1] / "player_history.parquet"
    if not p.exists():
        pytest.skip("no player history")
    h = pd.read_parquet(p, columns=["league", "date", "minutes", "started"])
    h = h[(h.league == "Premier League") & h.minutes.between(1, 59)
          & (pd.to_datetime(h.date, errors="coerce") >= "2025-07-01")].dropna(subset=["started"])
    share = h["started"].astype(bool).mean()
    assert abs(share - P_START_GIVEN_1_59) < 0.05, f"measured {share:.3f}; refit P_START_GIVEN_1_59"


def test_forwards_no_longer_out_score_defenders_on_dc():
    # typical FPL dc_per90 (2026-11-01 bootstrap medians): DEF 7.4, FWD 3.7
    assert F._poisson_dc_pts(7.4, F._DC_THRESH["DEF"]) > F._poisson_dc_pts(3.7, F._DC_THRESH["FWD"])
