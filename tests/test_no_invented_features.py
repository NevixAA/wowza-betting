"""No invented feature values; league averages per league in row order (Data Truth 7, 12).

    python -m pytest tests/test_no_invented_features.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import _add_home_advantage


def _games():
    rows = []
    d = pd.Timestamp("2025-08-01")
    # League A scores a lot away, league B almost never; interleaved by date.
    for k in range(6):
        rows.append({"date": d + pd.Timedelta(days=2 * k), "league": "A", "home_team": "A1",
                     "away_team": f"A{k + 2}", "home_goals": 2, "away_goals": 3})
        rows.append({"date": d + pd.Timedelta(days=2 * k + 1), "league": "B", "home_team": "B1",
                     "away_team": f"B{k + 2}", "home_goals": 1, "away_goals": 0})
    return pd.DataFrame(rows)


def test_first_home_game_has_no_invented_home_advantage():
    out = _add_home_advantage(_games())
    first = out.sort_values("date").groupby("league").head(1)
    assert first["home_advantage"].isna().all()
    assert first["away_home_adv_factor"].isna().all()


def test_league_average_is_the_rows_own_league():
    out = _add_home_advantage(_games()).sort_values("date")
    b = out[out["league"] == "B"]
    # B1 scores 1 at home every time, above league B's away average (0): later B rows must show
    # it dominating. With league A's average (3) pasted in, it never would.
    assert b["home_advantage"].dropna().eq(1.0).all() and b["home_advantage"].notna().sum() >= 3
