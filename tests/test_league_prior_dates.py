"""League averages use strictly earlier DATES only (owner 2026-11-09: fix the same-day leak).

    python -m pytest tests/test_league_prior_dates.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import feature_engineering as FE


def _df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03",
                                "2026-01-01", "2026-01-03"]),
        "league": ["A", "A", "A", "A", "A", "B", "B"],
        "total_goals": [1.0, 3.0, 5.0, np.nan, 2.0, 4.0, 0.0],
    })


def test_same_day_matches_never_see_each_other():
    m = FE._league_prior_dates_mean(_df(), "total_goals")
    assert np.isnan(m.iat[0]) and np.isnan(m.iat[1])          # first day: nothing earlier
    assert m.iat[2] == m.iat[3] == 2.0                          # day 2 sees only day 1 (1, 3)
    assert m.iat[4] == 3.0                                      # day 3: 1, 3, 5 (the NaN skipped)


def test_leagues_never_mix():
    m = FE._league_prior_dates_mean(_df(), "total_goals")
    assert np.isnan(m.iat[5]) and m.iat[6] == 4.0


def test_row_order_does_not_matter():
    d = _df()
    a = FE._league_prior_dates_mean(d, "total_goals")
    s = d.sample(frac=1.0, random_state=3)
    b = FE._league_prior_dates_mean(s, "total_goals").reindex(d.index)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_no_league_gives_nan():
    d = _df()
    d.loc[0, "league"] = None
    assert np.isnan(FE._league_prior_dates_mean(d, "total_goals").iat[0])


def test_training_average_equals_the_serving_average_on_every_date():
    """The training value for a match equals the plain mean over the league's history strictly
    before its date - the number build_upcoming_features computes for that fixture."""
    d = _df()
    m = FE._league_expanding_avg(d)
    for i, r in d.iterrows():
        h = d[(d["league"] == r["league"]) & (d["date"] < r["date"])]["total_goals"]
        want = h.mean() if h.notna().any() else np.nan
        assert (np.isnan(want) and np.isnan(m.iat[i])) or abs(m.iat[i] - want) < 1e-12
