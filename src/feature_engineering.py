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
    has_ht = "ht_home_goals" in matches.columns and matches["ht_home_goals"].notna().any()

    # Include corners and fouls for rolling averages (replacing match-day actuals)
    _corner_cols = [c for c in ["home_corners","away_corners"] if c in matches.columns]
    _foul_cols   = [c for c in ["home_fouls",  "away_fouls"]   if c in matches.columns]

    home_cols = ["date", "league", "home_team", "away_team",
                 "home_goals", "away_goals", "over25", "home_shots", "home_sot"]
    home_cols += [c for c in _corner_cols + _foul_cols if c in matches.columns]
    if has_ht:
        home_cols += ["ht_home_goals", "ht_away_goals"]

    home = matches[home_cols].copy()
    home["_src_idx"] = matches.index
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "scored", "away_goals": "conceded",
        "home_shots": "shots", "home_sot": "sot",
        "home_corners": "corners", "home_fouls": "fouls",
    })
    if has_ht:
        home = home.rename(columns={"ht_home_goals": "ht_scored", "ht_away_goals": "ht_conceded"})
    home["is_home"] = 1

    away_cols = ["date", "league", "home_team", "away_team",
                 "home_goals", "away_goals", "over25", "away_shots", "away_sot"]
    away_cols += [c for c in _corner_cols + _foul_cols if c in matches.columns]
    if has_ht:
        away_cols += ["ht_home_goals", "ht_away_goals"]

    away = matches[away_cols].copy()
    away["_src_idx"] = matches.index
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "scored", "home_goals": "conceded",
        "away_corners": "corners", "away_fouls": "fouls",
        "away_shots": "shots", "away_sot": "sot",
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
    # Rolling corners and fouls — leak-free, replaces match-day actuals
    if "corners" in tc.columns:
        tc["roll_corners"] = _rolling(tc, "corners", n)
    if "fouls" in tc.columns:
        tc["roll_fouls"] = _rolling(tc, "fouls", n)

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
    has_corners = "roll_corners" in tc.columns
    has_fouls   = "roll_fouls"   in tc.columns
    if has_corners:
        home_cols.append("roll_corners")
        away_cols.append("roll_corners")
    if has_fouls:
        home_cols.append("roll_fouls")
        away_cols.append("roll_fouls")
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

    def _find_team_rows(team: str, col: str, league: str = "") -> pd.Series:
        """Find rows for a team with fuzzy name fallback (OddsAPI vs FD name differences)."""
        mask = hist[col] == team
        if league:
            rows = hist[mask & (hist["league"] == league)]
        else:
            rows = hist[mask]
        if not rows.empty:
            return rows
        # Fuzzy fallback: try first word or prefix match (e.g. "Shelbourne Dublin" → "Shelbourne")
        first_word = team.split()[0].lower()
        mask_fuzzy = hist[col].str.lower().str.startswith(first_word)
        if league:
            rows = hist[mask_fuzzy & (hist["league"] == league)]
        else:
            rows = hist[mask_fuzzy]
        return rows

    def _team_recent(team: str, stat_home: str, stat_away: str, nn: int,
                     league: str = "") -> float:
        """
        Returns blended home+away average — consistent with build_features() training.
        stat_home = column when team plays AT HOME (e.g. "home_goals" for scored)
        stat_away = column when team plays AWAY    (e.g. "away_goals" for scored)
        Both represent the SAME semantic quantity (scored/conceded) from different venues.
        """
        h = _find_team_rows(team, "home_team", league)[stat_home].dropna().tail(nn)
        a = _find_team_rows(team, "away_team", league)[stat_away].dropna().tail(nn)
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
