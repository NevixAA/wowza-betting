"""
league_quality.py — Opposition quality and league tier enrichment
=================================================================
Adds per-row columns that allow the model to distinguish a player scoring
0.3 goals/game in Panama liga vs the same rate in the Premier League,
and to discount that form when facing elite opposition (England, France, etc.).

Call enrich_league_quality(history) BEFORE build_features().
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# ── League tier table ──────────────────────────────────────────────────────────
# Tier 1 = elite (PL, UCL, La Liga, Serie A)
# Tier 2 = strong second tier (Championship, Bundesliga 2, Eredivisie...)
# Tier 3 = mid (Brazil, Argentina, Saudi, K-League, MLS, Scandinavian...)
# Tier 4 = lower (Ireland, Finland, China, Romania...)
# Tier 5 = very weak / untracked
LEAGUE_TIERS: dict[str, int] = {
    # Tier 1 — elite European
    "Premier League":          1,
    "La Liga":                 1,
    "Bundesliga 1":            1,
    "Serie A":                 1,
    "Ligue 1":                 1,
    "Bundesliga":              1,
    "Champions League":        1,
    "UEFA Champions League":   1,
    "Europa League":           1,
    # Tier 2 — strong second-tier European
    "Championship":            2,
    "Bundesliga 2":            2,
    "La Liga 2":               2,
    "Serie B":                 2,
    "Ligue 2":                 2,
    "Dutch Eredivisie":        2,
    "Portuguese Primeira Liga": 2,
    "Scottish Premiership":    2,
    "Belgian First Division A": 2,
    "Turkish Super Lig":       2,
    "Austrian Bundesliga":     2,
    "Greek Super League":      2,
    "Polish Ekstraklasa":      2,
    "Denmark Superliga":       2,
    "Norway Eliteserien":      2,
    "Sweden Allsvenskan":      2,
    # Tier 3 — major non-European / lower European
    "League One":              3,
    "League Two":              3,
    "National League":         3,
    "Saudi Pro League":        3,
    "K-League 1":              3,
    "Brazil Serie A":          3,
    "Argentina Primera Division": 3,
    "Mexico Liga MX":          3,
    "Japan J-League":          3,
    "USA MLS":                 3,
    "MLS":                     3,
    "Scotland Championship":   3,
    "Scotland League One":     3,
    "Scotland League Two":     3,
    "Scottish Championship":   3,
    "Scottish League One":     3,
    "Scottish League Two":     3,
    "Romania Superliga":       3,
    "Romanian Superliga":      3,
    "Finland Veikkausliiga":   3,
    # Tier 4 — lower leagues
    "Ireland Premier Division": 4,
    "China Super League":      4,
    # World Cup & international tournaments (treated as tier 1 context)
    "World Cup":               1,
    "WC 2026":                 1,
    "UEFA Euro":               1,
    "Copa America":            2,
    "CONCACAF Gold Cup":       3,
    "CONCACAF Nations League": 3,
    "Africa Cup of Nations":   3,
}

# National team FIFA tier (rough, based on ranking bands)
NATIONAL_TEAM_TIERS: dict[str, int] = {
    # Tier 1 elite
    "France": 1, "England": 1, "Spain": 1, "Germany": 1, "Brazil": 1,
    "Argentina": 1, "Portugal": 1, "Netherlands": 1, "Belgium": 1,
    "Italy": 1, "Croatia": 1, "Uruguay": 1,
    # Tier 2 strong
    "USA": 2, "Mexico": 2, "Colombia": 2, "Senegal": 2, "Morocco": 2,
    "Japan": 2, "South Korea": 2, "Australia": 2, "Switzerland": 2,
    "Denmark": 2, "Austria": 2, "Turkey": 2, "Poland": 2, "Serbia": 2,
    "Ecuador": 2, "Canada": 2, "Chile": 2, "Peru": 2,
    # Tier 3 mid
    "Saudi Arabia": 3, "Iran": 3, "South Africa": 3, "Nigeria": 3,
    "Ghana": 3, "Costa Rica": 3, "Honduras": 3, "Jamaica": 3,
    "Venezuela": 3, "Bolivia": 3, "Paraguay": 3,
    # Tier 4 weaker
    "Panama": 4, "El Salvador": 4, "Cuba": 4, "Haiti": 4,
    "New Zealand": 4, "Indonesia": 4, "Vietnam": 4,
}

# Convert tier to quality score: tier 1 = 1.0, tier 2 = 0.75, tier 3 = 0.5, tier 4 = 0.30
def _tier_to_quality(tier: int) -> float:
    return {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.30}.get(tier, 0.25)


def enrich_league_quality(history: list[dict]) -> list[dict]:
    """
    Add league quality and player vs player strength features to history rows.

    New columns added per row:
      league_tier              — tier 1-4 of the league this match was played in
      league_quality           — 1.0/0.75/0.5/0.3 based on tier
      player_career_avg_quality— expanding-window career average league quality
      opp_def_player_rating_pg — rolling mean rating of opponent defenders
      opp_top_def_rating       — rolling mean of TOP defender rating (max per fixture)
      opp_def_player_quality   — composite: aerial × rating of opponent CB group
      quality_mismatch_goals   — goals_pg × (career_quality / opp_defence_quality)
      context_quality_discount — how hard today's opponent is vs player's career average

    All zero-safe; missing values gracefully default.
    """
    if not history:
        return history

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    n = 5  # rolling window — keep consistent with feature_engineering ROLLING_N

    # ── League tier per row ───────────────────────────────────────────────────
    def _get_tier(league: str) -> int:
        if not isinstance(league, str):
            return 3
        for key, tier in LEAGUE_TIERS.items():
            if key.lower() in league.lower() or league.lower() in key.lower():
                return tier
        return 3  # default: mid-tier

    df["league_tier"]    = df["league"].apply(_get_tier) if "league" in df.columns else 3
    df["league_quality"] = df["league_tier"].apply(_tier_to_quality)

    # ── Career average league quality (expanding, shift=1) ────────────────────
    grp = df.groupby("player_id", group_keys=False)
    df["player_career_avg_quality"] = grp["league_quality"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0.65))

    # ── Opponent defensive player individual quality ──────────────────────────
    # Aggregate rating of defensive players per fixture/team
    # (rating is the API match rating 0-10, already in history rows)
    _pos = df["position"].str.upper().fillna("") if "position" in df.columns         else pd.Series([""] * len(df), index=df.index)
    _def_mask = _pos.str.startswith("D")

    if _def_mask.any() and "rating" in df.columns:
        _defs = df[_def_mask].copy()
        _defs["_rating_clean"] = pd.to_numeric(_defs["rating"], errors="coerce").fillna(6.8)

        _def_agg = (
            _defs.groupby(["fixture_id", "team"])
            .agg(
                _date=("date", "first"),
                _avg_rating=("_rating_clean", "mean"),
                _top_rating=("_rating_clean", "max"),
                _n=("player_id", "count"),
            )
            .reset_index()
        )

        _def_agg = _def_agg.sort_values(["team", "_date"])
        _dgrp = _def_agg.groupby("team", group_keys=False)
        _def_agg["opp_def_player_rating_pg"] = _dgrp["_avg_rating"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _def_agg["opp_top_def_rating"] = _dgrp["_top_rating"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        if "opponent" not in df.columns:
            df["opponent"] = df.apply(
                lambda r: r["away_team"] if r.get("is_home") else r.get("home_team", ""), axis=1)

        _def_feats = _def_agg[["fixture_id","team",
                                "opp_def_player_rating_pg",
                                "opp_top_def_rating"]].rename(columns={"team":"opponent"})
        df = df.merge(_def_feats, on=["fixture_id","opponent"], how="left")
    else:
        df["opp_def_player_rating_pg"] = 6.8
        df["opp_top_def_rating"]       = 7.0

    df["opp_def_player_rating_pg"] = df["opp_def_player_rating_pg"].fillna(6.8)
    df["opp_top_def_rating"]       = df["opp_top_def_rating"].fillna(7.0)

    # ── Opponent defensive quality composite (aerial × rating) ───────────────
    # Normalized so 6.8 rating × 0.50 aerial win = 1.0 baseline
    _aerial = df.get("opp_def_aerial_win_rate", pd.Series(0.50, index=df.index))
    df["opp_def_player_quality"] = (
        (df["opp_def_player_rating_pg"] / 6.8)   # normalized rating (1.0 = league avg)
        * (df["opp_top_def_rating"]     / 7.0)   # best CB quality
        * (_aerial.fillna(0.50)         / 0.50)  # aerial dominance
    ).clip(0.2, 4.0)

    # ── Quality mismatch — player career quality vs today opponent quality ─────
    # If player scored in tier-3 leagues but faces tier-1 defence → heavy discount
    # context_quality_discount: 1.0 = same quality, <1.0 = facing harder opposition
    df["context_quality_discount"] = (
        df["player_career_avg_quality"] / df["opp_def_player_quality"].replace(0, 0.5)
    ).clip(0.1, 3.0)

    # Quality-adjusted goals per game
    if "goals_pg" in df.columns:
        df["quality_mismatch_goals"] = (
            df.get("goals_pg", 0.0) * df["context_quality_discount"]
        ).clip(0.0, 5.0)
    else:
        df["quality_mismatch_goals"] = 0.0

    if "sot_pg" in df.columns:
        df["quality_mismatch_sot"] = (
            df.get("sot_pg", 0.0) * df["context_quality_discount"]
        ).clip(0.0, 10.0)
    else:
        df["quality_mismatch_sot"] = 0.0

    # Drop helper cols
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    return df.to_dict("records")
