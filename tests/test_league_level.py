"""As-of league-level features (src/league_level.py): no row may see its own day or the future.

    python -m pytest tests/test_league_level.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import league_level as LL  # noqa: E402


def _frame(seed=0, days=120, per_day=4, leagues=("A", "B")):
    rng = np.random.default_rng(seed)
    rows = []
    for lg, lam in zip(leagues, (1.0, 1.8)):
        for k in range(days):
            for j in range(per_day):
                rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=k), "league": lg,
                             "season": "2025", "home_team": f"{lg}{j}", "away_team": f"{lg}{j + 10}",
                             "home_goals": rng.poisson(lam), "away_goals": rng.poisson(lam)})
    return pd.DataFrame(rows)


def test_future_and_same_day_results_never_change_a_row():
    d = _frame()
    base = LL.add_league_level(d)
    cut = pd.Timestamp("2025-03-01")
    e = d.copy()
    later = e["date"] >= cut                        # rewrite the cut day itself and everything after
    e.loc[later, "home_goals"] = 9
    e.loc[later, "away_goals"] = 9
    f = LL.add_league_level(e)
    cols = LL.feature_names()
    upto = d["date"] <= cut                         # rows ON the cut day must not see the cut day
    pd.testing.assert_frame_equal(base.loc[upto, cols], f.loc[upto, cols])
    assert not base.loc[d["date"] > cut + pd.Timedelta(days=1), cols].equals(
        f.loc[d["date"] > cut + pd.Timedelta(days=1), cols])


def test_value_equals_manual_prior_mean_and_leagues_differ():
    d = _frame()
    f = LL.add_league_level(d)
    row = f[(f["league"] == "A") & (f["date"] == pd.Timestamp("2025-04-10"))].iloc[0]
    prior = d[(d["league"] == "A") & (d["date"] < pd.Timestamp("2025-04-10"))].sort_values("date")
    tg = prior["home_goals"] + prior["away_goals"]
    # whole-date window: the last >= 100 prior matches, here exactly 100 (4 per day)
    assert row["lg_goals_pg_100"] == np.float64(tg.tail(100).mean())
    assert row["lg_o25_rate_100"] == (tg.tail(100) > 2.5).mean()
    lv = f[f["date"] == pd.Timestamp("2025-04-10")].groupby("league")["lg_goals_pg_300"].mean()
    assert lv["B"] > lv["A"] + 1.0, "a high-scoring league must read as high-scoring"


def test_too_little_history_is_nan_and_unplayed_rows_contribute_nothing():
    d = _frame(days=5)
    f = LL.add_league_level(d)
    assert f["lg_goals_pg_100"].isna().all(), "fewer than MIN_PRIOR prior matches -> NaN, never a guess"
    d2 = _frame()
    extra = d2.iloc[[0]].assign(date=pd.Timestamp("2025-03-02"), home_goals=np.nan, away_goals=np.nan)
    a = LL.add_league_level(d2)
    b = LL.add_league_level(pd.concat([d2, extra], ignore_index=True)).iloc[:len(d2)]
    pd.testing.assert_frame_equal(a[LL.feature_names()], b[LL.feature_names()])
