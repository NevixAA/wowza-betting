"""Upcoming features belong to THEIR fixture, whatever order the board arrives in (2026-11-01).

Rest days and HT features were computed on a sorted frame and copied back by position, so on an
unsorted board most fixtures carried another fixture's values.

    python -m pytest tests/test_upcoming_row_alignment.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_upcoming_features
from src.ht_features import HT_FEATURE_COLS, attach_ht_features
from tests.test_standard_forensic_repair import STD_A, STD_B, _frame

KEY = ["date", "league", "home_team", "away_team"]


@pytest.fixture(scope="module")
def raw():
    return _frame()


@pytest.fixture(scope="module")
def board(raw):
    last = raw["date"].max()
    ta = sorted(raw.loc[raw.league == STD_A, "home_team"].unique())
    tb = sorted(raw.loc[raw.league == STD_B, "home_team"].unique())
    rows = []
    # Two upcoming dates, and the same club twice, so a positional mix-up is visible.
    for k, d in enumerate((last + pd.Timedelta(days=3), last + pd.Timedelta(days=10))):
        rows += [{"date": d, "league": STD_A, "home_team": ta[i + k], "away_team": ta[i + 5]}
                 for i in range(4)]
        rows += [{"date": d, "league": STD_B, "home_team": tb[i], "away_team": tb[i + 5 + k]}
                 for i in range(4)]
    up = pd.DataFrame(rows)
    up["odds_over25"], up["odds_under25"] = 2.0, 1.8
    # Deliberately NOT in date order: later fixtures first.
    return up.iloc[::-1].reset_index(drop=True)


def _by_key(df, cols):
    return df.set_index(KEY)[cols].sort_index()


def test_rest_days_do_not_depend_on_board_order(raw, board):
    cols = ["home_rest_days", "away_rest_days"]
    a = _by_key(build_upcoming_features(board, raw), cols)
    b = _by_key(build_upcoming_features(board.sample(frac=1, random_state=7).reset_index(drop=True), raw), cols)
    pd.testing.assert_frame_equal(a, b)


def test_rest_days_are_the_gap_to_the_teams_previous_match(raw, board):
    f = build_upcoming_features(board, raw)
    for r in f.itertuples():
        team = r.home_team
        prev = pd.concat([raw.loc[(raw.home_team == team) | (raw.away_team == team), "date"],
                          board.loc[((board.home_team == team) | (board.away_team == team))
                                    & (board.date < r.date), "date"]])
        prev = prev[prev < r.date]
        want = (r.date - prev.max()).days if len(prev) else np.nan
        assert r.home_rest_days == pytest.approx(want, nan_ok=True), (team, r.date)


def test_ht_features_come_back_in_the_callers_order(raw, board):
    shuffled = board.sample(frac=1, random_state=3).reset_index(drop=True)
    out = attach_ht_features(raw, shuffled)
    assert list(out["home_team"]) == list(shuffled["home_team"])
    assert list(out["away_team"]) == list(shuffled["away_team"])
    cols = [c for c in HT_FEATURE_COLS if c in out.columns]
    a = _by_key(attach_ht_features(raw, board), cols)
    b = _by_key(out, cols)
    pd.testing.assert_frame_equal(a, b)
