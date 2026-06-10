"""
Player Feature Engineering v3 — match-level rolling features
=============================================================
build_features(match_rows)
  Input:  flat list of player-match dicts from collect_match_history()
          Each dict = one player in one specific completed match.
  Output: DataFrame with proper rolling features (shift=1, no leakage)
          and per-match binary targets.

build_upcoming_features(players, history_df)
  Uses actual rolling history from build_features() output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(match_rows: list[dict], n: int = None) -> pd.DataFrame:
    """
    Build ML training data from per-match player stats.

    Features: rolling averages of player's PREVIOUS N matches (shift=1, zero leakage).
    Targets:  actual outcome in THIS match (not the same as features).

    Also computes opponent defensive rolling features per fixture.
    """
    if not match_rows:
        return pd.DataFrame()

    n = n or config.ROLLING_N

    df = pd.DataFrame(match_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # Drop bench/DNP rows (< 10 min)
    df = df[df["minutes"] >= 10].copy()

    # ── Per-player rolling features — shift(1) prevents any leakage ──────────
    grp = df.groupby("player_id", group_keys=False)

    df["goals_pg"]    = grp["goals"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["assists_pg"]  = grp["assists"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["shots_pg"]    = grp["shots_total"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["sot_pg"]      = grp["shots_on_target"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["cards_pg"]    = grp["yellow_cards"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["minutes_pg"]  = grp["minutes"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    df["key_passes_pg"] = grp["key_passes"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    df["sot_rate"] = (
        df["sot_pg"] / df["shots_pg"].replace(0, np.nan)
    ).fillna(0.0)

    # Count of previous games this player has in the dataset
    df["n_prev_games"] = grp["date"].transform("cumcount")
    df["starter_rate"] = grp["started"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    # ── Opponent defensive rolling features ───────────────────────────────────
    # For each (fixture, team): aggregate goals scored/conceded from player rows.
    match_agg = (
        df.groupby(["fixture_id", "team"])
        .agg(
            date=("date", "first"),
            goals_scored=("goals", "sum"),
            sot_scored=("shots_on_target", "sum"),
        )
        .reset_index()
    )

    # Self-join to find opponent stats for each (fixture, team)
    opp = match_agg[["fixture_id", "team", "goals_scored", "sot_scored"]].rename(columns={
        "team":         "opponent_team",
        "goals_scored": "goals_conceded_match",
        "sot_scored":   "sot_conceded_match",
    })
    match_def = match_agg.merge(opp, on="fixture_id")
    match_def = match_def[match_def["team"] != match_def["opponent_team"]].copy()

    # Rolling defensive avg per team — shift(1) so it's pre-match knowledge
    match_def = match_def.sort_values(["team", "date"])
    tgrp = match_def.groupby("team", group_keys=False)
    match_def["opp_goals_conceded_pg"] = tgrp["goals_conceded_match"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    match_def["opp_sot_conceded_pg"] = tgrp["sot_conceded_match"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    match_def["team_goals_pg_roll"] = tgrp["goals_scored"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    # Join back to player rows via (fixture_id, opponent)
    df["opponent"] = df.apply(
        lambda r: r["away_team"] if r["is_home"] else r["home_team"], axis=1
    )
    opp_feats = match_def[["fixture_id", "team",
                            "opp_goals_conceded_pg",
                            "opp_sot_conceded_pg",
                            "team_goals_pg_roll"]].rename(columns={"team": "opponent"})
    df = df.merge(opp_feats, on=["fixture_id", "opponent"], how="left")

    df["opp_goals_conceded_pg"] = df["opp_goals_conceded_pg"].fillna(1.3)
    df["opp_sot_conceded_pg"]   = df["opp_sot_conceded_pg"].fillna(4.5)
    df["team_goals_pg_roll"]    = df["team_goals_pg_roll"].fillna(1.3)

    # ── Position encoding ─────────────────────────────────────────────────────
    pos = df["position"].str.upper().fillna("")
    df["pos_forward"]    = pos.str.startswith("F").astype(int)
    df["pos_midfielder"] = pos.str.startswith("M").astype(int)
    df["pos_defender"]   = pos.str.startswith("D").astype(int)

    # ── Target variables — actual outcome in THIS match ───────────────────────
    df["target_goals"]   = (df["goals"]             >= 1).astype(int)
    df["target_sot"]     = (df["shots_on_target"]   >= 1).astype(int)
    df["target_cards"]   = (df["yellow_cards"]       >= 1).astype(int)
    df["target_assists"] = (df["assists"]             >= 1).astype(int)

    # Drop rows with no prior history (rolling features would all be NaN)
    df = df[df["n_prev_games"] >= 1].copy()

    # Ensure all feature cols numeric
    for col in config.PLAYER_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.reset_index(drop=True)


def build_upcoming_features(
    upcoming: list[dict],
    history: pd.DataFrame,
    referee_profile: dict | None = None,
    match_context:   dict | None = None,
) -> pd.DataFrame:
    """
    Build feature rows for upcoming player predictions.
    Uses the player's last N matches from match-level history as rolling features.

    upcoming: list of player dicts (player_id, player_name, team, opponent, is_home, ...)
    history:  match-level DataFrame from build_features()
    """
    if not upcoming or history.empty:
        return pd.DataFrame()

    n   = config.ROLLING_N
    ref = referee_profile or {}
    ctx = match_context   or {}
    rows = []

    for p in upcoming:
        pid  = p.get("player_id")
        name = p.get("player_name", "")

        # Find player's match history
        if pid:
            phist = history[history["player_id"] == pid].sort_values("date").tail(n)
        else:
            phist = history[history["player_name"].str.lower() == name.lower()].sort_values("date").tail(n)

        if phist.empty:
            continue

        n_games    = len(phist)
        goals_pg   = float(phist["goals"].mean())
        assists_pg = float(phist["assists"].mean())
        shots_pg   = float(phist["shots_total"].mean())
        sot_pg     = float(phist["shots_on_target"].mean())
        cards_pg   = float(phist["yellow_cards"].mean())
        minutes_pg = float(phist["minutes"].mean())
        kp_pg      = float(phist["key_passes"].mean()) if "key_passes" in phist.columns else 0.0
        starter_rate = float(phist["started"].mean()) if "started" in phist.columns else 0.8
        sot_rate   = sot_pg / shots_pg if shots_pg > 0 else 0.0

        pos = str(p.get("position",
                   phist["position"].iloc[-1] if "position" in phist.columns else "")).upper()

        rows.append({
            "player_id":   pid,
            "player_name": name,
            "team":        p.get("team", ""),
            "opponent":    p.get("opponent", ""),
            "position":    p.get("position", ""),
            "minutes_est": p.get("minutes", int(minutes_pg)),
            "n_games":     n_games,
            "data_source": "match_level",
            # Rolling form
            "goals_pg":            round(goals_pg,   4),
            "assists_pg":          round(assists_pg,  4),
            "shots_pg":            round(shots_pg,    4),
            "sot_pg":              round(sot_pg,      4),
            "cards_pg":            round(cards_pg,    4),
            "minutes_pg":          round(minutes_pg,  1),
            "key_passes_pg":       round(kp_pg,       4),
            "sot_rate":            round(sot_rate,    4),
            "starter_rate":        round(starter_rate, 3),
            # Match context
            "is_home":              float(p.get("is_home", 0.5)),
            "opp_goals_conceded_pg": ctx.get("opp_goals_conceded_pg", 1.3),
            "opp_sot_conceded_pg":   ctx.get("opp_sot_conceded_pg",   4.5),
            "team_goals_pg_roll":    ctx.get("team_goals_pg_roll",    1.3),
            # Position
            "pos_forward":    int(pos.startswith("F")),
            "pos_midfielder": int(pos.startswith("M")),
            "pos_defender":   int(pos.startswith("D")),
            "rest_days":      p.get("rest_days", 6.0),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_rolling_features(match_rows: list[dict], n: int = None) -> dict:
    """Compute rolling feature dict from a player's recent match rows (predict path)."""
    if n is None:
        n = config.ROLLING_N
    if not match_rows:
        return {}

    rows  = match_rows[:n]
    df    = pd.DataFrame(rows)
    apps  = len(rows)

    def _pg(col):
        return float(df[col].sum() / apps) if col in df.columns else 0.0

    goals_pg   = _pg("goals")
    assists_pg = _pg("assists")
    shots_pg   = _pg("shots_total")
    sot_pg     = _pg("shots_on_target")
    cards_pg   = _pg("yellow_cards")
    minutes_pg = _pg("minutes")
    kp_pg      = _pg("key_passes")
    sot_rate   = sot_pg / shots_pg if shots_pg > 0 else 0.0

    return {
        "goals_pg":      round(goals_pg,   4),
        "assists_pg":    round(assists_pg,  4),
        "shots_pg":      round(shots_pg,    4),
        "sot_pg":        round(sot_pg,      4),
        "cards_pg":      round(cards_pg,    4),
        "minutes_pg":    round(minutes_pg,  1),
        "key_passes_pg": round(kp_pg,       4),
        "sot_rate":      round(sot_rate,    4),
        "n_games":       apps,
        "data_source":   "match_level",
    }


def compute_ges(row: dict, opp_weakness: float = 1.0, penalty_duty: bool = False) -> float:
    """Goal Edge Score (0.0-1.0). Gates goals/SOT signals."""
    xg_form  = min(row.get("goals_pg", 0) / 0.50, 1.0)
    shot_vol = min(row.get("shots_pg", 0) / 3.5, 1.0)
    pen      = 1.0 if penalty_duty else 0.0
    opp      = min(opp_weakness / 1.5, 1.0)
    min_sec  = min(row.get("minutes_pg", 0) / 85.0, 1.0)
    return round(0.40*xg_form + 0.20*shot_vol + 0.15*pen + 0.15*opp + 0.10*min_sec, 3)
