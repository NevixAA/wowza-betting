"""
Poisson-based expected goals and over/under probability.

Uses each team's rolling attack/defense ratings (already computed by
feature_engineering) to estimate expected goals per team via the
multiplicative Poisson model — the core of Dixon-Coles.

Expected home goals: lambda_h = home_attack_str * away_defense_str * half_avg
Expected away goals: lambda_a = away_attack_str * home_defense_str * half_avg

Then P(total goals > 2.5) is computed analytically from the independent
Poisson distributions, without any MLE fitting. This gives a principled
probabilistic feature that complements the ML ensemble.

Feature added:  poisson_prob_over25  (float in [0, 1])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.stats import poisson as _sp_poisson
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Core probability ──────────────────────────────────────────────────────────

def _pmf(k: int, lam: float) -> float:
    if _HAS_SCIPY:
        return float(_sp_poisson.pmf(k, lam))
    return float(np.exp(-lam) * (lam ** k) / np.math.factorial(k))


def poisson_over25(lambda_h: float, lambda_a: float) -> float:
    """P(home_goals + away_goals > 2.5) from independent Poisson distributions."""
    if (
        lambda_h is None or lambda_a is None
        or np.isnan(lambda_h) or np.isnan(lambda_a)
        or lambda_h <= 0 or lambda_a <= 0
    ):
        return np.nan
    # P(total <= 2) = sum over (h, a) where h + a <= 2
    p_under = sum(
        _pmf(h, lambda_h) * _pmf(a, lambda_a)
        for h in range(3)
        for a in range(3 - h)
    )
    return float(1.0 - p_under)


# ── DataFrame-level feature addition ─────────────────────────────────────────

def add_poisson_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lambda_home, lambda_away, and poisson_prob_over25 to the DataFrame.

    Requires these columns to already exist (computed by feature_engineering):
        home_attack_str, away_attack_str, home_defense_str, away_defense_str,
        league_avg_goals
    """
    df = df.copy()
    half_avg = (df["league_avg_goals"] / 2).replace(0, np.nan)

    # Multiplicative model: attack_str * opp_defense_str * base_rate
    df["lambda_home"] = (df["home_attack_str"] * df["away_defense_str"] * half_avg).clip(0.1, 8.0)
    df["lambda_away"] = (df["away_attack_str"] * df["home_defense_str"] * half_avg).clip(0.1, 8.0)

    df["poisson_prob_over25"] = df.apply(
        lambda row: poisson_over25(row["lambda_home"], row["lambda_away"]),
        axis=1,
    )

    return df
