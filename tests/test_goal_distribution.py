"""Tests for the coherent goal-distribution challenger.

The point of a distribution is that incoherent output is impossible. These assert that, because
it is the one property separate binary classifiers cannot offer and the main reason to prefer it.
"""
from __future__ import annotations
import numpy as np, pandas as pd, pytest
from src.goal_distribution import DixonColes, coherence_violations, rps


def _matches(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(20)]
    h = rng.choice(teams, n); a = rng.choice(teams, n)
    ok = h != a
    h, a, n = h[ok], a[ok], int(ok.sum())
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="8h"),
        "league": "L1", "home_team": h, "away_team": a,
        "home_goals": rng.poisson(1.5, n).astype(float),
        "away_goals": rng.poisson(1.1, n).astype(float)})


def test_probabilities_are_coherent_by_construction():
    d = _matches()
    p = DixonColes().fit(d.iloc[:2000]).predict(d.iloc[2000:])
    v = coherence_violations(p)
    for k in ("over15_below_over25", "over25_below_over35",
              "btts_above_over15", "1x2_not_summing_to_1"):
        assert v[k] == 0, f"{k}: a score distribution cannot produce this"


def test_matrix_sums_to_one():
    m = DixonColes().fit(_matches()).matrix(1.5, 1.1)
    assert abs(m.sum() - 1.0) < 1e-9


def test_every_probability_is_in_range():
    d = _matches()
    p = DixonColes().fit(d.iloc[:2000]).predict(d.iloc[2000:])
    for c in [c for c in p.columns if c.startswith("p_")]:
        assert p[c].between(0, 1).all(), f"{c} left [0,1]"


def test_higher_lambda_means_more_goals():
    """Sanity: a stronger attack must raise the over markets, or the parameterisation is wrong."""
    m = DixonColes()
    lo = m.predict(pd.DataFrame([{"league": "X", "home_team": "a", "away_team": "b"}]))
    m.league_mu["X"] = (2.5, 2.0)
    hi = m.predict(pd.DataFrame([{"league": "X", "home_team": "a", "away_team": "b"}]))
    assert hi["p_over25"].iloc[0] > lo["p_over25"].iloc[0]


def test_unknown_team_falls_back_without_crashing():
    d = _matches()
    gd = DixonColes().fit(d)
    p = gd.predict(pd.DataFrame([{"league": "L1", "home_team": "NEVER_SEEN",
                                  "away_team": "ALSO_NEW"}]))
    assert p["p_over25"].between(0, 1).all()


def test_rps_rewards_the_correct_ordered_outcome():
    """RPS must punish a confident wrong end more than a confident adjacent miss."""
    oc = np.array([0])                                   # home win
    good = rps(np.array([0.8]), np.array([0.15]), np.array([0.05]), oc)
    near = rps(np.array([0.15]), np.array([0.8]), np.array([0.05]), oc)   # said draw
    far = rps(np.array([0.05]), np.array([0.15]), np.array([0.8]), oc)    # said away
    assert good < near < far


def test_fit_uses_only_the_frame_it_is_given():
    """No leakage: a model fitted on the first half must not know the second half's teams."""
    d = _matches()
    gd = DixonColes().fit(d.iloc[:1500])
    later_only = set(zip(d.iloc[1500:]["league"], d.iloc[1500:]["home_team"])) - \
                 set(zip(d.iloc[:1500]["league"], d.iloc[:1500]["home_team"]))
    for key in later_only:
        assert key not in gd.attack, "a team seen only after the fit window has a rating"
