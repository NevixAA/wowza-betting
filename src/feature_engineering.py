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
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.team_names import resolve as resolve_team

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

try:
    from src.poisson import dixon_coles_p_over25 as _dc_p_over25
except ImportError:
    try:
        from poisson import dixon_coles_p_over25 as _dc_p_over25
    except ImportError:
        _dc_p_over25 = None

_N = config.ROLLING_N


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_team_centric(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode match rows into one row per team per match. Preserves row index."""
    has_ht = "ht_home_goals" in matches.columns and matches["ht_home_goals"].notna().any()

    # Include corners, fouls, xG, and inside-box shots for rolling averages
    _corner_cols   = [c for c in ["home_corners",   "away_corners"]   if c in matches.columns]
    _foul_cols     = [c for c in ["home_fouls",     "away_fouls"]     if c in matches.columns]
    _xg_cols       = [c for c in ["home_xg",        "away_xg"]        if c in matches.columns]
    _insidebox_cols = [c for c in ["home_insidebox", "away_insidebox"] if c in matches.columns]

    home_cols = ["date", "league", "home_team", "away_team",
                 "home_goals", "away_goals", "over25", "home_shots", "home_sot"]
    home_cols += [c for c in _corner_cols + _foul_cols + _xg_cols + _insidebox_cols
                  if c in matches.columns]
    if has_ht:
        home_cols += ["ht_home_goals", "ht_away_goals"]

    home = matches[home_cols].copy()
    home["_src_idx"] = matches.index
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "scored", "away_goals": "conceded",
        "home_shots": "shots", "home_sot": "sot",
        "home_corners": "corners", "home_fouls": "fouls",
        "home_xg": "xg", "home_insidebox": "insidebox",
    })
    if has_ht:
        home = home.rename(columns={"ht_home_goals": "ht_scored", "ht_away_goals": "ht_conceded"})
    home["is_home"] = 1

    away_cols = ["date", "league", "home_team", "away_team",
                 "home_goals", "away_goals", "over25", "away_shots", "away_sot"]
    away_cols += [c for c in _corner_cols + _foul_cols + _xg_cols + _insidebox_cols
                  if c in matches.columns]
    if has_ht:
        away_cols += ["ht_home_goals", "ht_away_goals"]

    away = matches[away_cols].copy()
    away["_src_idx"] = matches.index
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "scored", "home_goals": "conceded",
        "away_corners": "corners", "away_fouls": "fouls",
        "away_shots": "shots", "away_sot": "sot",
        "away_xg": "xg", "away_insidebox": "insidebox",
    })
    if has_ht:
        away = away.rename(columns={"ht_away_goals": "ht_scored", "ht_home_goals": "ht_conceded"})
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

def _add_home_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team rolling home advantage index.
    HA = fraction of team's last N home games where they scored > league-avg goals conceded.
    A team that consistently scores well at home gets HA > 0.5; weak home teams < 0.5.
    Replaces the constant 1.0 which provided zero discriminative signal.
    """
    df = df.copy()
    n = _N
    ha_home, ha_away = [], []
    df_sorted = df.sort_values("date").reset_index(drop=True)
    league_avg = df_sorted.groupby("league")["away_goals"].expanding().mean().shift(1)
    df_sorted["league_avg_away_goals"] = league_avg.values

    home_ha: dict[str, float] = {}
    away_ha: dict[str, float] = {}
    home_results: dict[str, list] = {}

    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        lg_avg = row.get("league_avg_away_goals", 1.1) or 1.1

        # Lookup existing window
        h_wins = home_results.get(h, [])
        ha_home.append(float(np.mean(h_wins[-n:])) if h_wins else 0.55)

        a_wins = home_results.get(a, [])
        ha_away.append(float(np.mean(a_wins[-n:])) if a_wins else 0.45)

        # Update after this match (shift(1) — use result for next match)
        if not pd.isna(row.get("home_goals")):
            scored = float(row["home_goals"])
            home_results.setdefault(h, []).append(1.0 if scored > lg_avg else 0.0)

    df_sorted["home_advantage"]      = ha_home
    df_sorted["away_home_adv_factor"] = ha_away
    # Restore original order
    df_sorted = df_sorted.set_index(df.sort_values("date").index)
    df["home_advantage"]       = df_sorted["home_advantage"].values
    df["away_home_adv_factor"] = df_sorted["away_home_adv_factor"].values
    return df


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


# ── Season-to-date venue stats ───────────────────────────────────────────────

def _add_season_venue_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding season-to-date home/away goal averages and clean sheet rates.
    Leakage-free: expanding mean shifted by 1 row within each (league, season, team, venue).
    These complement the last-5 rolling features by capturing full-season tendencies.
    """
    out = df.copy()
    for col in ["home_season_goals_h", "away_season_goals_a",
                "home_season_conceded_h", "away_season_conceded_a",
                "home_cs_rate_h", "away_cs_rate_a"]:
        out[col] = np.nan

    if "season" not in out.columns:
        return out

    s = out.sort_values("date")

    for _, grp in s.groupby(["league", "season", "home_team"], sort=False):
        idx = grp.index
        out.loc[idx, "home_season_goals_h"]    = grp["home_goals"].expanding().mean().shift(1).values
        out.loc[idx, "home_season_conceded_h"] = grp["away_goals"].expanding().mean().shift(1).values
        out.loc[idx, "home_cs_rate_h"]         = (grp["away_goals"] == 0).astype(float).expanding().mean().shift(1).values

    for _, grp in s.groupby(["league", "season", "away_team"], sort=False):
        idx = grp.index
        out.loc[idx, "away_season_goals_a"]    = grp["away_goals"].expanding().mean().shift(1).values
        out.loc[idx, "away_season_conceded_a"] = grp["home_goals"].expanding().mean().shift(1).values
        out.loc[idx, "away_cs_rate_a"]         = (grp["home_goals"] == 0).astype(float).expanding().mean().shift(1).values

    return out


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
    # Rolling corners, fouls, xG, and inside-box — leak-free, replaces match-day actuals
    if "corners" in tc.columns:
        tc["roll_corners"] = _rolling(tc, "corners", n)
    if "fouls" in tc.columns:
        tc["roll_fouls"] = _rolling(tc, "fouls", n)
    if "xg" in tc.columns:
        tc["roll_xg"] = _rolling(tc, "xg", n)
    if "insidebox" in tc.columns:
        tc["roll_insidebox"] = _rolling(tc, "insidebox", n)

    has_ht = "ht_scored" in tc.columns and tc["ht_scored"].notna().any()
    if has_ht:
        tc["roll_ht_scored"]   = _rolling(tc, "ht_scored",   n)
        tc["roll_ht_conceded"] = _rolling(tc, "ht_conceded", n)
        # Team HT tendency: % of last N games with >= 1 / >= 2 HT goals (combined)
        tc["ht_total"] = tc["ht_scored"] + tc["ht_conceded"]
        tc["ht_over05_flag"] = (tc["ht_total"] >= 1).astype(float).where(tc["ht_total"].notna())
        tc["ht_over15_flag"] = (tc["ht_total"] >= 2).astype(float).where(tc["ht_total"].notna())
        tc["roll_ht_over05"] = _rolling(tc, "ht_over05_flag", n)
        tc["roll_ht_over15"] = _rolling(tc, "ht_over15_flag", n)

    # Map back by original row index to avoid any cross-league duplicates
    home_cols = ["_src_idx", "roll_scored", "roll_conceded", "roll_over25", "roll_shots", "roll_sot"]
    away_cols = ["_src_idx", "roll_scored", "roll_conceded", "roll_over25", "roll_shots", "roll_sot"]
    has_corners   = "roll_corners"   in tc.columns
    has_fouls     = "roll_fouls"     in tc.columns
    has_xg        = "roll_xg"        in tc.columns
    has_insidebox = "roll_insidebox" in tc.columns
    if has_corners:
        home_cols.append("roll_corners")
        away_cols.append("roll_corners")
    if has_fouls:
        home_cols.append("roll_fouls")
        away_cols.append("roll_fouls")
    if has_xg:
        home_cols.append("roll_xg")
        away_cols.append("roll_xg")
    if has_insidebox:
        home_cols.append("roll_insidebox")
        away_cols.append("roll_insidebox")
    if has_ht:
        home_cols += ["roll_ht_scored", "roll_ht_conceded", "roll_ht_over05", "roll_ht_over15"]
        away_cols += ["roll_ht_scored", "roll_ht_conceded", "roll_ht_over05", "roll_ht_over15"]

    home_tc = tc[tc["is_home"] == 1][home_cols].copy()
    home_tc = home_tc.rename(columns={
        "roll_scored":       "home_scored_last5",
        "roll_conceded":     "home_conceded_last5",
        "roll_over25":       "home_over25_last5",
        "roll_shots":        "home_shots_last5",
        "roll_sot":          "home_sot_last5",
        "roll_corners":      "home_corners_pg_roll",
        "roll_fouls":        "home_fouls_pg_roll",
        "roll_xg":           "home_xg_last5",
        "roll_insidebox":    "home_insidebox_last5",
        "roll_ht_scored":    "home_ht_scored_last5",
        "roll_ht_conceded":  "home_ht_conceded_last5",
        "roll_ht_over05":    "home_ht_over05_rate",
        "roll_ht_over15":    "home_ht_over15_rate",
    })
    away_tc = tc[tc["is_home"] == 0][away_cols].copy()
    away_tc = away_tc.rename(columns={
        "roll_scored":       "away_scored_last5",
        "roll_conceded":     "away_conceded_last5",
        "roll_over25":       "away_over25_last5",
        "roll_shots":        "away_shots_last5",
        "roll_sot":          "away_sot_last5",
        "roll_corners":      "away_corners_pg_roll",
        "roll_fouls":        "away_fouls_pg_roll",
        "roll_xg":           "away_xg_last5",
        "roll_insidebox":    "away_insidebox_last5",
        "roll_ht_scored":    "away_ht_scored_last5",
        "roll_ht_conceded":  "away_ht_conceded_last5",
        "roll_ht_over05":    "away_ht_over05_rate",
        "roll_ht_over15":    "away_ht_over15_rate",
    })

    df["_src_idx"] = df.index
    df = df.merge(home_tc, on="_src_idx", how="left")
    df = df.merge(away_tc, on="_src_idx", how="left")
    df = df.drop(columns=["_src_idx"])

    # ── HT-derived features (when available) ──────────────────────────────
    if has_ht and "home_ht_scored_last5" in df.columns:
        df["combined_ht_goals_avg"] = (
            df["home_ht_scored_last5"].fillna(0) + df["away_ht_scored_last5"].fillna(0)
        )
        if "league_avg_goals" in df.columns:
            half_avg = (df["league_avg_goals"] / 2).replace(0, np.nan)
        else:
            half_avg = pd.Series(0.65, index=df.index)
        df["home_ht_attack_str"]  = df["home_ht_scored_last5"]   / half_avg
        df["away_ht_attack_str"]  = df["away_ht_scored_last5"]   / half_avg
        df["home_ht_defense_str"] = df["home_ht_conceded_last5"] / half_avg
        df["away_ht_defense_str"] = df["away_ht_conceded_last5"] / half_avg

    # ── League-wide expanding average ──────────────────────────────────────
    df["league_avg_goals"] = _league_expanding_avg(df).values

    # ── Strength metrics ────────────────────────────────────────────────────
    half_avg = (df["league_avg_goals"] / 2).replace(0, np.nan)
    df["home_attack_str"]  = df["home_scored_last5"]   / half_avg
    df["away_attack_str"]  = df["away_scored_last5"]   / half_avg
    df["home_defense_str"] = df["home_conceded_last5"] / half_avg
    df["away_defense_str"] = df["away_conceded_last5"] / half_avg

    # ── Match context ───────────────────────────────────────────────────────
    # Per-team rolling home advantage (replaces constant 1.0 which has no discriminative power)
    # = fraction of home games where team scored more than away average goals conceded
    df = _add_rest_days(df)
    df = _add_home_advantage(df)
    df = _add_season_venue_stats(df)

    # ── Set piece proxies — rolling historical averages (no match-day leakage) ──
    # combined_corners_pg_roll and combined_fouls_pg_roll come from the team-centric
    # rolling computation above — these are pre-match historical averages, not actuals
    df["home_corners_pg_roll"]  = df.get("home_corners_pg_roll",  pd.Series(np.nan, index=df.index))
    df["away_corners_pg_roll"]  = df.get("away_corners_pg_roll",  pd.Series(np.nan, index=df.index))
    df["home_fouls_pg_roll"]    = df.get("home_fouls_pg_roll",    pd.Series(np.nan, index=df.index))
    df["away_fouls_pg_roll"]    = df.get("away_fouls_pg_roll",    pd.Series(np.nan, index=df.index))
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
    if "referee" in df.columns:
        df["referee_foul_avg"] = df["referee"].map(
            lambda r: ref_stats.get(str(r).strip(), ref_default) if pd.notna(r) else ref_default
        )
    else:
        df["referee_foul_avg"] = ref_default

    # ── Implied probability from historical odds ────────────────────────────
    df["implied_prob_over"]  = 1.0 / df["odds_over25"].replace(0, np.nan)
    df["implied_prob_under"] = 1.0 / df["odds_under25"].replace(0, np.nan)
    df["bookmaker_overround"] = (
        df["implied_prob_over"] + df["implied_prob_under"] - 1.0
    ).clip(lower=0.0)

    # Dixon-Coles corrected P(over 2.5) using per-team rolling goal rates as lambdas
    if _dc_p_over25 is not None and "home_scored_last5" in df.columns:
        _lh = df["home_scored_last5"].clip(lower=0.3, upper=4.0)
        _la = df["away_scored_last5"].clip(lower=0.3, upper=4.0)
        df["p_over25_poisson_dc"] = [_dc_p_over25(h, a) for h, a in zip(_lh, _la)]
    else:
        df["p_over25_poisson_dc"] = df["implied_prob_over"].fillna(0.5)

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

    _league_clubs: dict = {}
    _resolved: dict = {}

    def _clubs_in(league: str) -> list:
        """Every club name football-data uses in this league (cached)."""
        if league not in _league_clubs:
            h = hist[hist["league"] == league] if league else hist
            _league_clubs[league] = sorted(
                set(h["home_team"].dropna().astype(str)) |
                set(h["away_team"].dropna().astype(str))
            )
        return _league_clubs[league]

    def _find_team_rows(team: str, col: str, league: str = "") -> pd.DataFrame:
        """Rows for a team, reconciling OddsAPI spelling against football-data spelling.

        The old fallback was `hist[col].str.lower().str.startswith(team.split()[0].lower())`,
        which BOTH missed most real differences and was unsafe. It failed on "1. FC
        Kaiserslautern" (first word "1."), "Cadiz CF" (accent), "QPR" (initialism) and
        "Karlsruher SC" (stem) — 46% of standard fixtures resolved to nothing and got a full
        set of median-imputed form features — while "Real Valladolid CF" would have matched
        ANY club starting "Real". src.team_names.resolve is league-scoped and refuses an
        ambiguous match instead of guessing.
        """
        mask = hist[col] == team
        rows = hist[mask & (hist["league"] == league)] if league else hist[mask]
        if not rows.empty:
            return rows

        key = (str(team), str(league))
        if key not in _resolved:
            hit = resolve_team(team, _clubs_in(league))
            _resolved[key] = hit
            if hit:
                log.info(f"[names] {league}: '{team}' -> '{hit}'")
            else:
                log.warning(f"[names] {league}: no history club matches '{team}' — "
                            f"its form features will be imputed")
        hit = _resolved[key]
        if not hit:
            return hist.iloc[0:0]          # explicit miss, never a loose guess
        m = hist[col] == hit
        return hist[m & (hist["league"] == league)] if league else hist[m]

    def _team_recent(team: str, stat_home: str, stat_away: str, nn: int,
                     league: str = "") -> float:
        """
        Returns blended home+away average — consistent with build_features() training.
        stat_home = column when team plays AT HOME (e.g. "home_goals" for scored)
        stat_away = column when team plays AWAY    (e.g. "away_goals" for scored)
        Both represent the SAME semantic quantity (scored/conceded) from different venues.
        Returns NaN when the column is absent (e.g. xG not available for standard-format leagues).
        """
        home_rows = _find_team_rows(team, "home_team", league)
        away_rows = _find_team_rows(team, "away_team", league)
        if stat_home not in home_rows.columns or stat_away not in away_rows.columns:
            return np.nan
        h = home_rows[stat_home].dropna().tail(nn)
        a = away_rows[stat_away].dropna().tail(nn)
        vals = list(h) + list(a)
        return float(np.mean(vals[-nn:])) if vals else np.nan

    def _team_over25_rate(team: str, nn: int, league: str = "") -> float:
        h = _find_team_rows(team, "home_team", league)["over25"].dropna().tail(nn)
        a = _find_team_rows(team, "away_team", league)["over25"].dropna().tail(nn)
        vals = list(h) + list(a)
        return float(np.mean(vals[-nn:])) if vals else np.nan

    rows = upcoming.to_dict("records")
    feat_records = []
    for row in rows:
        ht, at, lg = row["home_team"], row["away_team"], row.get("league", "")
        feat = dict(row)
        feat["home_scored_last5"]   = _team_recent(ht, "home_goals",  "away_goals",  n, lg)
        feat["home_conceded_last5"] = _team_recent(ht, "away_goals",  "home_goals",  n, lg)
        feat["away_scored_last5"]   = _team_recent(at, "home_goals",  "away_goals",  n, lg)
        feat["away_conceded_last5"] = _team_recent(at, "away_goals",  "home_goals",  n, lg)
        feat["home_over25_last5"]   = _team_over25_rate(ht, n, lg)
        feat["away_over25_last5"]   = _team_over25_rate(at, n, lg)
        feat["home_shots_last5"]    = _team_recent(ht, "home_shots",  "away_shots",  n, lg)
        feat["home_sot_last5"]      = _team_recent(ht, "home_sot",    "away_sot",    n, lg)
        feat["away_shots_last5"]    = _team_recent(at, "home_shots",  "away_shots",  n, lg)
        feat["away_sot_last5"]      = _team_recent(at, "home_sot",    "away_sot",    n, lg)
        # HT rolling features (NaN for leagues without HTHG/HTAG data)
        feat["home_ht_scored_last5"]    = _team_recent(ht, "ht_home_goals", "ht_away_goals", n, lg)
        feat["home_ht_conceded_last5"]  = _team_recent(ht, "ht_away_goals", "ht_home_goals", n, lg)
        feat["away_ht_scored_last5"]    = _team_recent(at, "ht_home_goals", "ht_away_goals", n)
        feat["away_ht_conceded_last5"]  = _team_recent(at, "ht_away_goals", "ht_home_goals", n)
        def _ht_rate(team, threshold, nn, league=""):
            # League-isolated: never mix HT stats across different leagues
            h = hist[hist["home_team"] == team]
            a = hist[hist["away_team"] == team]
            if league:
                h = h[h["league"] == league]
                a = a[a["league"] == league]
            hm   = h["ht_home_goals"].dropna()
            hm_t = h["ht_away_goals"].dropna()
            aw   = a["ht_away_goals"].dropna()
            aw_t = a["ht_home_goals"].dropna()
            total = [(a_ + b) for a_, b in zip(list(hm.tail(nn)), list(hm_t.tail(nn)))] + \
                    [(a_ + b) for a_, b in zip(list(aw.tail(nn)), list(aw_t.tail(nn)))]
            if not total:
                return float('nan')
            return sum(1 for x in total if x >= threshold) / len(total)

        feat["home_ht_over05_rate"] = _ht_rate(ht, 1, n, lg)
        feat["away_ht_over05_rate"] = _ht_rate(at, 1, n, lg)
        feat["home_ht_over15_rate"] = _ht_rate(ht, 2, n, lg)
        feat["away_ht_over15_rate"] = _ht_rate(at, 2, n, lg)

        # Rolling corners / fouls per team (ML audit features — must match build_features())
        feat["home_corners_pg_roll"] = _team_recent(ht, "home_corners", "away_corners", n, lg)
        feat["away_corners_pg_roll"] = _team_recent(at, "home_corners", "away_corners", n, lg)
        feat["home_fouls_pg_roll"]   = _team_recent(ht, "home_fouls",   "away_fouls",   n, lg)
        feat["away_fouls_pg_roll"]   = _team_recent(at, "home_fouls",   "away_fouls",   n, lg)
        # xG and inside-box shots (from API-Football cache — new-format leagues only)
        feat["home_xg_last5"]        = _team_recent(ht, "home_xg",        "away_xg",        n, lg)
        feat["away_xg_last5"]        = _team_recent(at, "home_xg",        "away_xg",        n, lg)
        feat["home_insidebox_last5"] = _team_recent(ht, "home_insidebox", "away_insidebox", n, lg)
        feat["away_insidebox_last5"] = _team_recent(at, "home_insidebox", "away_insidebox", n, lg)
        feat["home_possession_last5"] = _team_recent(ht, "home_possession", "away_possession", n, lg)
        feat["away_possession_last5"] = _team_recent(at, "home_possession", "away_possession", n, lg)
        feat["home_blocked_last5"]    = _team_recent(ht, "home_blocked",    "away_blocked",    n, lg)
        feat["away_blocked_last5"]    = _team_recent(at, "home_blocked",    "away_blocked",    n, lg)

        # Season-to-date venue stats (current season from historical data — no shift needed)
        def _venue_season_stats(team, team_col, scored_col, conceded_col, league=""):
            rows = hist[hist[team_col] == team]
            if league:
                rows = rows[rows["league"] == league]
            if rows.empty or "season" not in rows.columns:
                return np.nan, np.nan, np.nan
            latest_szn = rows.sort_values("date")["season"].iloc[-1]
            rows = rows[rows["season"] == latest_szn]
            if rows.empty:
                return np.nan, np.nan, np.nan
            return (
                float(rows[scored_col].dropna().mean()),
                float(rows[conceded_col].dropna().mean()),
                float((rows[conceded_col] == 0).mean()),
            )

        (feat["home_season_goals_h"], feat["home_season_conceded_h"], feat["home_cs_rate_h"]) = \
            _venue_season_stats(ht, "home_team", "home_goals", "away_goals", lg)
        (feat["away_season_goals_a"], feat["away_season_conceded_a"], feat["away_cs_rate_a"]) = \
            _venue_season_stats(at, "away_team", "away_goals", "home_goals", lg)

        # Away team's rolling home advantage (fraction of their own home games they dominate)
        at_home_rows = _find_team_rows(at, "home_team", lg).tail(n)
        if at_home_rows.empty:
            feat["away_home_adv_factor"] = 0.45
        else:
            lg_avg_away = hist[hist["league"] == lg]["away_goals"].mean() if lg else hist["away_goals"].mean()
            lg_avg_away = lg_avg_away if pd.notna(lg_avg_away) and lg_avg_away > 0 else 1.1
            feat["away_home_adv_factor"] = float(
                (at_home_rows["home_goals"] > lg_avg_away).sum() / len(at_home_rows)
            )

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

    # HT strength features
    df["combined_ht_goals_avg"] = (
        df["home_ht_scored_last5"].fillna(0) + df["away_ht_scored_last5"].fillna(0)
    )
    df["home_ht_attack_str"]  = df["home_ht_scored_last5"]   / half_avg
    df["away_ht_attack_str"]  = df["away_ht_scored_last5"]   / half_avg
    df["home_ht_defense_str"] = df["home_ht_conceded_last5"] / half_avg
    df["away_ht_defense_str"] = df["away_ht_conceded_last5"] / half_avg

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

    # ── Phase 2: Injury features — key_attacker_absent_home / _away ─────────────
    # Pre-fetch per league (one batch per league, not per row) then apply per row.
    # Falls back gracefully to 0 when API_KEY is absent or API returns nothing.
    df["key_attacker_absent_home"] = 0.0
    df["key_attacker_absent_away"] = 0.0
    try:
        from src.api_football_ou import get_league_injury_context, injury_features_from_context
        _inj_ctx: dict = {}
        for _lg in df["league"].unique():
            _lid = config.API_FOOTBALL_IDS.get(_lg)
            if not _lid:
                continue
            _szn = config.API_FOOTBALL_SEASONS.get(_lg, "2025")
            try:
                _inj_ctx[_lg] = get_league_injury_context(_lid, _szn)
            except Exception:
                pass

        if _inj_ctx:
            _inj_rows = []
            for _, _r in df.iterrows():
                _ctx = _inj_ctx.get(_r.get("league", ""), {})
                _inj_rows.append(injury_features_from_context(
                    _r["home_team"], _r["away_team"], _r["date"], _ctx
                ))
            _inj_df = pd.DataFrame(_inj_rows, index=df.index)
            df["key_attacker_absent_home"] = _inj_df["key_attacker_absent_home"].fillna(0.0)
            df["key_attacker_absent_away"] = _inj_df["key_attacker_absent_away"].fillna(0.0)
    except Exception as e:
        # Falling back to 0 is safe, but SAY SO. A missing APIFOOTBALL_KEY (which is exactly
        # what happened in predict.yml for months) looks identical to a quiet run when this
        # is a bare `pass`. output/feature_health.json records the resulting degeneracy.
        log.warning(f"[enrich] injury features unavailable, using defaults: {e}")

    # ── Phase 3: /teams/statistics — real-time season-to-date venue averages ────
    # Overrides _venue_season_stats() values computed from FD historical data.
    # Useful at season start (few FD rows) and for current-form accuracy.
    # Skips silently when API key absent or league not in API_FOOTBALL_IDS.
    try:
        from src.api_football_ou import fetch_upcoming_team_stats
        _ts = fetch_upcoming_team_stats(df)
        if _ts:
            for _idx, _r in df.iterrows():
                _ht = str(_r.get("home_team", ""))
                _at = str(_r.get("away_team", ""))
                _hs = _ts.get(_ht, {})
                _as = _ts.get(_at, {})
                if _hs:
                    if _hs.get("goals_for_h") is not None:
                        df.at[_idx, "home_season_goals_h"]    = _hs["goals_for_h"]
                    if _hs.get("goals_against_h") is not None:
                        df.at[_idx, "home_season_conceded_h"] = _hs["goals_against_h"]
                    if _hs.get("cs_rate_h") is not None:
                        df.at[_idx, "home_cs_rate_h"]         = _hs["cs_rate_h"]
                if _as:
                    if _as.get("goals_for_a") is not None:
                        df.at[_idx, "away_season_goals_a"]    = _as["goals_for_a"]
                    if _as.get("goals_against_a") is not None:
                        df.at[_idx, "away_season_conceded_a"] = _as["goals_against_a"]
                    if _as.get("cs_rate_a") is not None:
                        df.at[_idx, "away_cs_rate_a"]         = _as["cs_rate_a"]
    except Exception as e:
        # These override REAL model inputs (season venue goals/conceded/clean-sheet rates),
        # so a silent failure means predict quietly scores on stale football-data values.
        log.warning(f"[enrich] /teams/statistics unavailable, keeping FD historical: {e}")

    # ── Phases 4-7: Lineup, H2H, API odds ────────────────────────────────────
    # Defaults applied first so the model always has valid values even when
    # API is unavailable or lineup not yet released.
    # Phase 7 extensions (BTTS, O/U 3.5, draw odds)
    df["api_implied_btts"]   = 0.50
    df["api_implied_over35"] = 0.30
    df["api_implied_over15"] = 0.80
    df["api_implied_draw"]   = 0.27

    df["home_attack_formation"]  = 0.65
    df["away_attack_formation"]  = 0.65
    df["combined_attack_intent"] = 1.30
    df["home_forward_count"]     = 1.0
    df["away_forward_count"]     = 1.0
    df["h2h_over25_rate"]        = 0.50
    df["h2h_avg_goals"]          = 2.60
    df["h2h_home_win_rate"]      = 0.43
    df["h2h_n"]                  = 0.0
    df["api_implied_over25"]     = df["implied_prob_over"].fillna(0.5) if "implied_prob_over" in df.columns else 0.5
    df["api_overround"]          = 0.05

    try:
        from src.api_football_ou import (
            resolve_upcoming_fixture_ids,
            fetch_lineup_features,
            fetch_h2h_features,
            fetch_prematch_odds_features,
        )
        _fids = resolve_upcoming_fixture_ids(df)
        for _feat_dict in [
            fetch_lineup_features(df, fixture_ids=_fids),
            fetch_h2h_features(df),
            fetch_prematch_odds_features(df, fixture_ids=_fids),
        ]:
            for _idx, _feats in _feat_dict.items():
                for _k, _v in _feats.items():
                    if _k in df.columns:
                        df.at[_idx, _k] = _v
    except Exception as e:
        # Lineup / H2H / Bet365 pre-match odds all share this handler, so ONE failure here
        # pinned all 14 of those columns to their hardcoded defaults at once.
        log.warning(f"[enrich] lineup/H2H/odds features unavailable, using defaults: {e}")

    # ── Phases 9-10: Season round + coach features ────────────────────────────
    df["season_stage_ratio"]      = 0.50
    df["is_late_season"]          = 0.0
    df["home_coach_tenure_days"]  = 180.0
    df["home_coach_is_caretaker"] = 0.0
    df["away_coach_tenure_days"]  = 180.0
    df["away_coach_is_caretaker"] = 0.0
    try:
        from src.api_football_ou import fetch_season_round_features, fetch_coach_features
        for _feat_dict in [
            fetch_season_round_features(df),
            fetch_coach_features(df),
        ]:
            for _idx, _feats in _feat_dict.items():
                for _k, _v in _feats.items():
                    if _k in df.columns:
                        df.at[_idx, _k] = _v
    except Exception as e:
        log.warning(f"[enrich] season-round/coach features unavailable, using defaults: {e}")

    # Artificial pitch — static per-league flag (Finland/Sweden/Norway)
    _art = getattr(config, "ARTIFICIAL_PITCH_LEAGUES", set())
    df["home_pitch_artificial"] = df["league"].isin(_art).astype(float)

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
    df["bookmaker_overround"] = (
        df["implied_prob_over"] + df["implied_prob_under"] - 1.0
    ).clip(lower=0.0)

    # Dixon-Coles corrected P(over 2.5)
    if _dc_p_over25 is not None and "home_scored_last5" in df.columns:
        _lh = df["home_scored_last5"].clip(lower=0.3, upper=4.0)
        _la = df["away_scored_last5"].clip(lower=0.3, upper=4.0)
        df["p_over25_poisson_dc"] = [_dc_p_over25(h, a) for h, a in zip(_lh, _la)]
    else:
        df["p_over25_poisson_dc"] = df["implied_prob_over"].fillna(0.5)

    # ── Optional: API-Football standings metadata (not model features) ────────
    # Adds table position, GD/game, form pts for display in predictions.csv.
    # Falls back silently if API key unavailable or request fails.
    for col in ["home_table_pos", "away_table_pos", "home_gd_pg", "away_gd_pg",
                "home_form_pts", "away_form_pts", "home_win_rate_h", "away_win_rate_a",
                "table_pos_gap"]:
        df[col] = np.nan

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.api_football import build_standings_map, lookup_team_standing
        import config as _cfg

        _standings_cache: dict = {}

        for i, row in df.iterrows():
            league  = row.get("league", "")
            lg_id   = _cfg.API_FOOTBALL_IDS.get(league)
            if not lg_id:
                continue

            season = _cfg.API_FOOTBALL_SEASONS.get(league, _cfg.API_SEASON)

            if (lg_id, season) not in _standings_cache:
                _standings_cache[(lg_id, season)] = build_standings_map(lg_id, season)
            smap   = _standings_cache[(lg_id, season)]
            n_teams = max(len(smap), 1)

            home_s = lookup_team_standing(smap, str(row.get("home_team", "")))
            away_s = lookup_team_standing(smap, str(row.get("away_team", "")))

            if home_s:
                df.at[i, "home_table_pos"]  = round(home_s["rank"] / n_teams, 3)
                df.at[i, "home_gd_pg"]      = home_s["gd_per_game"]
                df.at[i, "home_form_pts"]   = home_s["form_pts"]
                df.at[i, "home_win_rate_h"] = home_s["home_win_rate"]
            if away_s:
                df.at[i, "away_table_pos"]  = round(away_s["rank"] / n_teams, 3)
                df.at[i, "away_gd_pg"]      = away_s["gd_per_game"]
                df.at[i, "away_form_pts"]   = away_s["form_pts"]
                df.at[i, "away_win_rate_a"] = away_s["away_win_rate"]
            if home_s and away_s:
                df.at[i, "table_pos_gap"] = round(
                    (away_s["rank"] - home_s["rank"]) / n_teams, 3
                )
    except Exception:
        pass  # standings are metadata only — never break predictions

    return df
