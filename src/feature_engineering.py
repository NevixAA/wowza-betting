"""
Feature Engineering — ZERO DATA LEAKAGE.

All rolling/form features use only past data:
  - Each team's stats are shifted by 1 before rolling, so the current match
    is never included in its own features.
  - League averages are expanding (up to but not including the current match).
  - Rest days use only the previous match date.

Features produced per match row
────────────────────────────────
Rolling form (last N matches before this one):
  home_scored_last5      away_scored_last5
  home_conceded_last5    away_conceded_last5
  home_over25_last5      away_over25_last5    (team's over-2.5 rate last N)
  home_shots_last5       away_shots_last5     (avg shots per game last N)
  home_sot_last5         away_sot_last5       (avg shots on target per game last N)
  home_sot_ratio_last5   away_sot_ratio_last5 (SOT/shots rate last N)

Strength vs league (lagged league average):
  home_attack_str        away_attack_str      (= team_avg_scored / half_league_avg)
  home_defense_str       away_defense_str     (lower = better defense)

Match context:
  home_advantage         (always 1.0 — explicit for model interpretability)
  home_rest_days         away_rest_days

Set piece proxies (from CSV columns):
  combined_corners_pg    combined_fouls_pg    combined_sot_ratio

Actual set piece goals (Sofascore cache):
  home_sp_goals_pg       away_sp_goals_pg
  home_pen_goals_pg      away_pen_goals_pg
  home_fk_goals_pg       away_fk_goals_pg

Odds-derived:
  implied_prob_over      implied_prob_under
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

_N = config.ROLLING_N


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_team_centric(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode match rows into one row per team per match. Preserves row index."""
    home = matches[["date", "league", "home_team", "away_team",
                    "home_goals", "away_goals", "over25",
                    "home_shots", "home_sot"]].copy()
    home["_src_idx"] = matches.index
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "scored", "away_goals": "conceded",
        "home_shots": "shots", "home_sot": "sot",
    })
    home["is_home"] = 1

    away = matches[["date", "league", "home_team", "away_team",
                    "home_goals", "away_goals", "over25",
                    "away_shots", "away_sot"]].copy()
    away["_src_idx"] = matches.index
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "scored", "home_goals": "conceded",
        "away_shots": "shots", "away_sot": "sot",
    })
    away["is_home"] = 0

    tc = pd.concat([home, away], ignore_index=True)
    return tc.sort_values(["team", "league", "date"]).reset_index(drop=True)


def _rolling(tc: pd.DataFrame, stat: str, n: int) -> pd.Series:
    """Leakage-free rolling mean: shift(1) then rolling(n)."""
    return tc.groupby("team")[stat].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )


def _league_expanding_avg(df: pd.DataFrame) -> pd.Series:
    """Expanding league-wide goal average up to (not including) each match."""
    df = df.copy()
    result = pd.Series(np.nan, index=df.index)
    for league, grp in df.groupby("league"):
        exp = grp["total_goals"].expanding().mean().shift(1)
        result.loc[grp.index] = exp.values
    return result


# ── Rest days ────────────────────────────────────────────────────────────────

def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        hd = last_match.get(h)
        ad = last_match.get(a)
        home_rest.append((row["date"] - hd).days if hd is not None else np.nan)
        away_rest.append((row["date"] - ad).days if ad is not None else np.nan)
        last_match[h] = row["date"]
        last_match[a] = row["date"]

    # Restore original order
    idx_map = dict(zip(df.sort_values("date").index, zip(home_rest, away_rest)))
    df["home_rest_days"] = [idx_map[i][0] for i in df.index]
    df["away_rest_days"] = [idx_map[i][1] for i in df.index]
    return df


# ── Referee foul avg ─────────────────────────────────────────────────────────

def _compute_referee_stats(matches: pd.DataFrame) -> tuple[dict, float]:
    """
    Compute per-referee average total fouls from historical data.
    Returns (ref_dict, global_default) where ref_dict maps name → avg_fouls.
    Minimum 3 matches to be included.
    """
    df = matches.copy()
    if "referee" not in df.columns or df["referee"].isna().all():
        return {}, 22.0

    df = df[df["referee"].notna()].copy()
    df["referee"] = df["referee"].astype(str).str.strip()
    df["total_fouls"] = (
        pd.to_numeric(df.get("home_fouls", 0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("away_fouls", 0), errors="coerce").fillna(0)
    )
    global_default = float(df["total_fouls"].mean()) if len(df) > 0 else 22.0

    grp = df.groupby("referee").agg(
        ref_foul_avg=("total_fouls", "mean"),
        ref_matches=("total_fouls",  "count"),
    ).reset_index()

    result = {
        row["referee"]: round(float(row["ref_foul_avg"]), 2)
        for _, row in grp.iterrows()
        if row["ref_matches"] >= 3
    }
    return result, global_default


# ── Sofascore SP goals ───────────────────────────────────────────────────────

def _load_sofascore_cache() -> dict:
    if not config.SOFASCORE_CACHE.exists():
        return {}
    try:
        return json.loads(config.SOFASCORE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


_SS_DEFAULTS = {
    "sp_goals_pg": 0.30, "pen_goals_pg": 0.07,
    "fk_goals_pg": 0.03, "headed_goals_pg": 0.20,
}

_ss_cache_loaded: Optional[dict] = None


def _ss_lookup(league: str, team: str, stat: str) -> float:
    global _ss_cache_loaded
    if _ss_cache_loaded is None:
        _ss_cache_loaded = _load_sofascore_cache()

    teams_dict = _ss_cache_loaded.get(league, {}).get("teams", {})
    if not teams_dict:
        return _SS_DEFAULTS[stat]

    if team in teams_dict:
        return teams_dict[team].get(stat, _SS_DEFAULTS[stat])

    # Fuzzy match
    try:
        from rapidfuzz import process as fp
        hit = fp.extractOne(team, list(teams_dict.keys()), score_cutoff=75)
        if hit:
            return teams_dict[hit[0]].get(stat, _SS_DEFAULTS[stat])
    except ImportError:
        # Simple substring fallback
        tl = team.lower()
        for name in teams_dict:
            if tl in name.lower() or name.lower() in tl:
                return teams_dict[name].get(stat, _SS_DEFAULTS[stat])

    return _SS_DEFAULTS[stat]


def _merge_sofascore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    records = df[["league", "home_team", "away_team"]].to_dict("records")

    cols = {
        "home_sp_goals_pg":  ("home_team", "sp_goals_pg"),
        "away_sp_goals_pg":  ("away_team", "sp_goals_pg"),
        "home_pen_goals_pg": ("home_team", "pen_goals_pg"),
        "away_pen_goals_pg": ("away_team", "pen_goals_pg"),
        "home_fk_goals_pg":  ("home_team", "fk_goals_pg"),
        "away_fk_goals_pg":  ("away_team", "fk_goals_pg"),
    }
    for out_col, (team_col, stat) in cols.items():
        df[out_col] = [
            _ss_lookup(r["league"], r[team_col], stat) for r in records
        ]
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def build_features(matches: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """
    Add all engineered features to the matches DataFrame.
    Input must come from data_loader.load_all_matches().
    Returns a new DataFrame with feature columns appended.
    No data leakage: all features use only information available before each match.
    """
    if n is None:
        n = _N

    df = matches.copy().sort_values("date").reset_index(drop=True)

    # ── Rolling form (team-centric, shift(1)) ─────────────────────────────
    tc = _build_team_centric(df)
    tc["roll_scored"]   = _rolling(tc, "scored",  n)
    tc["roll_conceded"] = _rolling(tc, "conceded", n)
    tc["roll_over25"]   = _rolling(tc, "over25",  n)
    tc["roll_shots"]    = _rolling(tc, "shots",   n)
    tc["roll_sot"]      = _rolling(tc, "sot",     n)

    # Map back by original row index to avoid any cross-league duplicates
    home_tc = tc[tc["is_home"] == 1][
        ["_src_idx", "roll_scored", "roll_conceded", "roll_over25", "roll_shots", "roll_sot"]
    ].copy()
    home_tc = home_tc.rename(columns={
        "roll_scored":   "home_scored_last5",
        "roll_conceded": "home_conceded_last5",
        "roll_over25":   "home_over25_last5",
        "roll_shots":    "home_shots_last5",
        "roll_sot":      "home_sot_last5",
    })
    away_tc = tc[tc["is_home"] == 0][
        ["_src_idx", "roll_scored", "roll_conceded", "roll_over25", "roll_shots", "roll_sot"]
    ].copy()
    away_tc = away_tc.rename(columns={
        "roll_scored":   "away_scored_last5",
        "roll_conceded": "away_conceded_last5",
        "roll_over25":   "away_over25_last5",
        "roll_shots":    "away_shots_last5",
        "roll_sot":      "away_sot_last5",
    })

    df["_src_idx"] = df.index
    df = df.merge(home_tc, on="_src_idx", how="left")
    df = df.merge(away_tc, on="_src_idx", how="left")
    df = df.drop(columns=["_src_idx"])

    # ── League-wide expanding average ──────────────────────────────────────
    df["league_avg_goals"] = _league_expanding_avg(df).values

    # ── Strength metrics ────────────────────────────────────────────────────
    half_avg = (df["league_avg_goals"] / 2).replace(0, np.nan)
    df["home_attack_str"]  = df["home_scored_last5"]   / half_avg
    df["away_attack_str"]  = df["away_scored_last5"]   / half_avg
    df["home_defense_str"] = df["home_conceded_last5"] / half_avg
    df["away_defense_str"] = df["away_conceded_last5"] / half_avg

    # ── Match context ───────────────────────────────────────────────────────
    df["home_advantage"] = 1.0
    df = _add_rest_days(df)

    # ── Set piece proxies from CSV ──────────────────────────────────────────
    df["combined_corners_pg"] = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)
    df["combined_fouls_pg"]   = df["home_fouls"].fillna(0)   + df["away_fouls"].fillna(0)
    # Rolling team SOT ratio (leak-free: each team's avg over their last N games)
    df["home_sot_ratio_last5"] = df["home_sot_last5"] / df["home_shots_last5"].replace(0, np.nan)
    df["away_sot_ratio_last5"] = df["away_sot_last5"] / df["away_shots_last5"].replace(0, np.nan)
    df["combined_sot_ratio"] = (
        df["home_sot_ratio_last5"].fillna(0.35) + df["away_sot_ratio_last5"].fillna(0.33)
    ) / 2

    # ── Actual SP goals from Sofascore cache ───────────────────────────────
    df = _merge_sofascore(df)

    # ── Referee foul avg ────────────────────────────────────────────────────
    ref_stats, ref_default = _compute_referee_stats(df)
    df["referee_foul_avg"] = df["referee"].map(
        lambda r: ref_stats.get(str(r).strip(), ref_default) if pd.notna(r) else ref_default
    )

    # ── Implied probability from historical odds ────────────────────────────
    df["implied_prob_over"]  = 1.0 / df["odds_over25"].replace(0, np.nan)
    df["implied_prob_under"] = 1.0 / df["odds_under25"].replace(0, np.nan)

    return df


# ── Upcoming match features (no target col) ───────────────────────────────────

def build_upcoming_features(
    upcoming: pd.DataFrame,
    historical: pd.DataFrame,
    n: int = None,
) -> pd.DataFrame:
    """
    Compute features for upcoming (unplayed) fixtures using only historical data.
    `upcoming` must have: date, league, home_team, away_team, odds_over25, odds_under25
    `historical` is the full historical DataFrame from load_all_matches().
    """
    if n is None:
        n = _N

    upcoming = upcoming.copy()
    upcoming["home_advantage"] = 1.0

    hist = historical.sort_values("date").copy()

    def _team_recent(team: str, stat_home: str, stat_away: str, nn: int) -> float:
        hm = hist[hist["home_team"] == team][stat_home].dropna().tail(nn)
        aw = hist[hist["away_team"] == team][stat_away].dropna().tail(nn)
        vals = list(hm) + list(aw)
        return float(np.mean(vals[-nn:])) if vals else np.nan

    def _team_over25_rate(team: str, nn: int) -> float:
        hm = hist[hist["home_team"] == team]["over25"].dropna().tail(nn)
        aw = hist[hist["away_team"] == team]["over25"].dropna().tail(nn)
        vals = list(hm) + list(aw)
        return float(np.mean(vals[-nn:])) if vals else np.nan

    rows = upcoming.to_dict("records")
    feat_records = []
    for row in rows:
        ht, at = row["home_team"], row["away_team"]
        feat = dict(row)
        feat["home_scored_last5"]   = _team_recent(ht, "home_goals",  "away_goals",  n)
        feat["home_conceded_last5"] = _team_recent(ht, "away_goals",  "home_goals",  n)
        feat["away_scored_last5"]   = _team_recent(at, "home_goals",  "away_goals",  n)
        feat["away_conceded_last5"] = _team_recent(at, "away_goals",  "home_goals",  n)
        feat["home_over25_last5"]   = _team_over25_rate(ht, n)
        feat["away_over25_last5"]   = _team_over25_rate(at, n)
        feat["home_shots_last5"]    = _team_recent(ht, "home_shots",  "away_shots",  n)
        feat["home_sot_last5"]      = _team_recent(ht, "home_sot",    "away_sot",    n)
        feat["away_shots_last5"]    = _team_recent(at, "home_shots",  "away_shots",  n)
        feat["away_sot_last5"]      = _team_recent(at, "home_sot",    "away_sot",    n)
        feat_records.append(feat)

    df = pd.DataFrame(feat_records)

    # League avg from history
    league_avgs = hist.groupby("league")["total_goals"].mean()
    df["league_avg_goals"] = df["league"].map(league_avgs)

    half_avg = (df["league_avg_goals"] / 2).replace(0, np.nan)
    df["home_attack_str"]  = df["home_scored_last5"]   / half_avg
    df["away_attack_str"]  = df["away_scored_last5"]   / half_avg
    df["home_defense_str"] = df["home_conceded_last5"] / half_avg
    df["away_defense_str"] = df["away_conceded_last5"] / half_avg

    # Rest days using historical
    combined = pd.concat([
        hist[["date", "home_team", "away_team"]].tail(1000),
        df[["date", "home_team", "away_team"]],
    ]).sort_values("date").reset_index(drop=True)
    combined_rd = _add_rest_days(combined)
    tail = combined_rd.tail(len(df)).reset_index(drop=True)
    df["home_rest_days"] = tail["home_rest_days"].values
    df["away_rest_days"] = tail["away_rest_days"].values

    # Set piece proxies — use league averages from history
    lg_corners = hist.groupby("league").apply(
        lambda g: (g["home_corners"].mean() + g["away_corners"].mean())
    )
    lg_fouls = hist.groupby("league").apply(
        lambda g: (g["home_fouls"].mean() + g["away_fouls"].mean())
    )
    df["combined_corners_pg"] = df["league"].map(lg_corners).fillna(9.5)
    df["combined_fouls_pg"]   = df["league"].map(lg_fouls).fillna(22.0)
    # Per-team rolling SOT ratios (replaces the old hardcoded 0.34 global fallback)
    df["home_sot_ratio_last5"] = df["home_sot_last5"] / df["home_shots_last5"].replace(0, np.nan)
    df["away_sot_ratio_last5"] = df["away_sot_last5"] / df["away_shots_last5"].replace(0, np.nan)
    df["combined_sot_ratio"] = (
        df["home_sot_ratio_last5"].fillna(0.35) + df["away_sot_ratio_last5"].fillna(0.33)
    ) / 2

    # Sofascore
    df = _merge_sofascore(df)

    # Referee foul avg — use historical ref stats; fallback to global default
    ref_stats, ref_default = _compute_referee_stats(hist)
    ref_col = df.get("referee", pd.Series(dtype=object))
    df["referee_foul_avg"] = ref_col.map(
        lambda r: ref_stats.get(str(r).strip(), ref_default) if pd.notna(r) else ref_default
    )

    # Implied probs
    df["implied_prob_over"]  = 1.0 / df["odds_over25"].replace(0, np.nan)
    df["implied_prob_under"] = 1.0 / df["odds_under25"].replace(0, np.nan)

    return df
