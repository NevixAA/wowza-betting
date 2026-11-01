"""
league_quality.py — Opposition quality and league tier enrichment
=================================================================
Adds per-row columns that allow the model to distinguish a player scoring
0.3 goals/game in Panama liga vs the same rate in the Premier League,
and to discount that form when facing elite opposition (England, France, etc.).

Call enrich_league_quality(history) BEFORE build_features().

DATA TRUTH (owner rule, 2026-11-01): NO HAND TIERS. The typed LEAGUE_TIERS / NATIONAL_TEAM_TIERS
tables (tier 1-4 per league name, quality 1.0/0.75/0.5/0.3, unknown league -> tier 3) were opinions,
not measurements, and are deleted. No source in the repos measures league strength on one scale
across leagues (the history holds no cross-league fixtures to link them), so:
  * league_tier, league_quality and player_career_avg_quality are NaN (explicitly missing);
  * the opponent-defence rating features are MEASURED from the fixture player ratings by
    feature_engineering (team_match_table), with a minimum sample, and normalised by the league's own
    measured average there.
NEEDS RESOURCE (owner approval): a measured, cross-league strength per league or club, e.g. club Elo
(clubelo.com), UEFA association / club coefficients, or Opta Power Rankings, stored per league in
config/leagues.json and read through src/league_registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_COLUMNS = ("league_tier", "league_quality", "player_career_avg_quality")


def league_tier(league) -> float:
    """Measured tier of `league`: none exists in the repos (see module docstring) -> NaN for every
    league. Kept as a function so callers have one place to read it from when a measured source
    is approved."""
    return float("nan")


def enrich_league_quality(history: list[dict]) -> list[dict]:
    """
    Add league quality columns to history rows: league_tier, league_quality and
    player_career_avg_quality, all NaN until a measured league strength exists.

    The opponent-defence rating columns (opp_def_player_rating_pg, opp_top_def_rating) and the
    composites built on them (opp_def_player_quality, context_quality_discount,
    quality_mismatch_*) are computed in feature_engineering.build_features from the fixture player
    ratings, with ONE definition shared by training and serving. Computing them here as well (with
    6.8 / 7.0 / 0.50 fills and min_periods=1) gave two definitions of the same feature.
    """
    if not history:
        return history

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    for c in QUALITY_COLUMNS:
        df[c] = np.nan
    return df.to_dict("records")
