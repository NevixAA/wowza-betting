"""
Player Feature Engineering v2
==============================
Two data modes:
  1. MATCH-LEVEL (preferred): rolling stats from API-Football match history
  2. SEASON-LEVEL (fallback): FBref season aggregates

Key improvements over v1:
  - Referee factor (strictness z-score)
  - Set-piece features (set_piece_shot_rate, aerial_won_rate)
  - Proper sot_rate column
  - team_corners_per90 feature
  - GES (Goal Edge Score) computation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_rolling_features(match_rows: list[dict], n: int = None) -> dict:
    """
    Build rolling feature dict from a player's last N match stats.
    match_rows: list of per-match dicts from api_football.get_player_recent_stats()
    """
    if n is None:
        n = config.ROLLING_N
    if not match_rows:
        return {}

    rows = match_rows[:n]
    df = pd.DataFrame(rows)
    apps = len(rows)
    total_min = df["minutes_played"].sum() if "minutes_played" in df.columns else apps * 75

    def pg(col):
        return float(df[col].sum() / apps) if col in df.columns else 0.0

    goals_pg   = pg("goals")
    assists_pg = pg("assists")
    shots_pg   = pg("shots_total")
    sot_pg     = pg("shots_on_target")
    cards_pg   = pg("yellow_card")
    kp_pg      = pg("key_passes")
    minutes_pg = total_min / apps
    sot_rate   = sot_pg / shots_pg if shots_pg > 0 else 0.0

    dw = df["duels_won"].sum()  if "duels_won"   in df.columns else 0
    dt = df["duels_total"].sum() if "duels_total" in df.columns else 1
    aerial_won_rate = dw / max(dt, 1)

    return {
        "goals_pg":            round(goals_pg, 4),
        "assists_pg":          round(assists_pg, 4),
        "shots_pg":            round(shots_pg, 4),
        "sot_pg":              round(sot_pg, 4),
        "cards_pg":            round(cards_pg, 4),
        "minutes_pg":          round(minutes_pg, 1),
        "key_passes_pg":       round(kp_pg, 4),
        "sot_rate":            round(sot_rate, 4),
        "set_piece_shot_rate": 0.10,   # requires Sofascore — placeholder
        "touches_box_per90":   0.0,
        "aerial_won_rate":     round(aerial_won_rate, 4),
        "n_games":             apps,
        "data_source":         "match_level",
    }


def build_features(rows: list[dict]) -> pd.DataFrame:
    """Convert FBref season-level rows into ML feature DataFrame (training)."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["appearances"] >= config.MIN_APPEARANCES].copy()

    apps = df["appearances"].clip(lower=1)

    df["goals_pg"]      = df["goals"] / apps
    df["assists_pg"]    = df["assists"] / apps
    df["shots_pg"]      = df.get("shots_total",     pd.Series(0, index=df.index)) / apps
    df["sot_pg"]        = df.get("shots_on_target", pd.Series(0, index=df.index)) / apps
    df["cards_pg"]      = df.get("yellow_cards",    pd.Series(0, index=df.index)) / apps
    df["minutes_pg"]    = df.get("minutes",         pd.Series(apps * 75, index=df.index)) / apps
    df["key_passes_pg"] = df.get("key_passes",      pd.Series(0, index=df.index)) / apps
    df["sot_rate"]      = (df["sot_pg"] / df["shots_pg"].replace(0, np.nan)).fillna(0.0)

    df["set_piece_shot_rate"] = 0.10
    df["touches_box_per90"]   = 0.0
    df["aerial_won_rate"]     = 0.45

    pos = df["position"].str.upper().fillna("")
    df["pos_forward"]    = pos.str.startswith("F").astype(int)
    df["pos_midfielder"] = pos.str.startswith("M").astype(int)
    df["pos_defender"]   = pos.str.startswith("D").astype(int)

    lg_avg_g = df.groupby("league")["goals_pg"].transform("mean").clip(lower=0.01)
    df["opp_goals_conceded_pg"] = lg_avg_g
    df["opp_shots_conceded_pg"] = df.groupby("league")["shots_pg"].transform("mean").clip(lower=1)
    df["opp_sot_conceded_pg"]   = df.groupby("league")["sot_pg"].transform("mean").clip(lower=0.5)

    team_g = df.groupby(["league", "team"])["goals"].transform("sum")
    lg_avg  = df.groupby("league")["goals"].transform("sum") / df.groupby("league")["team"].transform("nunique")
    df["team_attack_str"]      = (team_g / lg_avg.clip(lower=1)).clip(0.1, 5.0)
    df["team_goals_scored_pg"] = df["goals_pg"]
    df["team_corners_per90"]   = 5.0

    df["referee_strictness"] = 0.0
    df["ref_cards_per_game"] = 3.5
    df["rest_days"]          = 6.0
    df["is_home"]            = 0.5

    df["target_goals"]   = (df["goals_pg"]   > 0).astype(int)
    df["target_assists"] = (df["assists_pg"]  > 0).astype(int)
    df["target_sot"]     = (df["sot_pg"]      > 0).astype(int)
    df["target_cards"]   = (df["cards_pg"]    > 0).astype(int)

    for col in config.PLAYER_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.reset_index(drop=True)


def build_upcoming_features(
    upcoming: list[dict],
    history: pd.DataFrame,
    referee_profile: dict | None = None,
    match_context: dict | None = None,
) -> pd.DataFrame:
    """
    Build feature rows for upcoming player predictions.
    upcoming: list of player dicts with optional rolling_features key
    history:  season-level training DataFrame (fallback)
    """
    if not upcoming or history.empty:
        return pd.DataFrame()

    ref = referee_profile or {"yellows_per_game": 3.5, "strictness_score": 0.0}
    ctx = match_context or {}
    rows = []

    for p in upcoming:
        pid     = p.get("player_id")
        name    = p.get("player_name", "")
        rolling = p.get("rolling_features", {})

        if not rolling:
            hist = history[history["player_id"] == pid] if pid else pd.DataFrame()
            if hist.empty:
                hist = history[history["player_name"].str.lower() == name.lower()]
            if hist.empty:
                continue
            rec     = hist.iloc[-1].to_dict()
            rolling = {k: rec.get(k, 0) for k in config.PLAYER_FEATURE_COLS}
            rolling["data_source"] = "season_fallback"
            rolling["n_games"]     = int(rec.get("appearances", config.MIN_GAMES_SIGNAL))

        pos    = p.get("position", rolling.get("position", ""))
        pos_up = str(pos).upper()
        n_games = int(rolling.get("n_games", config.MIN_GAMES_SIGNAL))

        rows.append({
            "player_id":   pid,
            "player_name": name,
            "team":        p.get("team", ""),
            "opponent":    p.get("opponent", ""),
            "position":    pos,
            "minutes_est": p.get("minutes", int(rolling.get("minutes_pg", 75))),
            "n_games":     n_games,
            "data_quality": min(1.0, n_games / 15.0),
            "data_source": rolling.get("data_source", "unknown"),
            # Rolling form
            "goals_pg":           rolling.get("goals_pg", 0),
            "assists_pg":         rolling.get("assists_pg", 0),
            "shots_pg":           rolling.get("shots_pg", 0),
            "sot_pg":             rolling.get("sot_pg", 0),
            "cards_pg":           rolling.get("cards_pg", 0),
            "minutes_pg":         rolling.get("minutes_pg", 75),
            "key_passes_pg":      rolling.get("key_passes_pg", 0),
            "sot_rate":           rolling.get("sot_rate", 0),
            "set_piece_shot_rate": rolling.get("set_piece_shot_rate", 0.10),
            "touches_box_per90":   rolling.get("touches_box_per90", 0),
            "aerial_won_rate":     rolling.get("aerial_won_rate", 0.45),
            # Match context
            "is_home":              float(p.get("is_home", 0.5)),
            "opp_goals_conceded_pg": ctx.get("opp_goals_conceded_pg", 1.3),
            "opp_shots_conceded_pg": ctx.get("opp_shots_conceded_pg", 12.0),
            "opp_sot_conceded_pg":   ctx.get("opp_sot_conceded_pg", 4.5),
            "team_attack_str":       ctx.get("team_attack_str", 1.0),
            "team_corners_per90":    ctx.get("team_corners_per90", 5.0),
            "team_goals_scored_pg":  rolling.get("goals_pg", 0),
            # Referee
            "referee_strictness":  ref.get("strictness_score", 0.0),
            "ref_cards_per_game":  ref.get("yellows_per_game", 3.5),
            # Position
            "pos_forward":    int(pos_up.startswith("F")),
            "pos_midfielder": int(pos_up.startswith("M")),
            "pos_defender":   int(pos_up.startswith("D")),
            "rest_days": p.get("rest_days", 6.0),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_ges(row: dict, opp_weakness: float = 1.0, penalty_duty: bool = False) -> float:
    """Goal Edge Score (0.0-1.0). Gate goals/SOT signals on this."""
    xg_form  = min(row.get("goals_pg", 0) / 0.50, 1.0)
    shot_vol = min(row.get("shots_pg", 0) / 3.5, 1.0)
    pen      = 1.0 if penalty_duty else 0.0
    opp      = min(opp_weakness / 1.5, 1.0)
    min_sec  = min(row.get("minutes_pg", 0) / 85.0, 1.0)
    return round(0.40*xg_form + 0.20*shot_vol + 0.15*pen + 0.15*opp + 0.10*min_sec, 3)
