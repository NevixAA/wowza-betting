"""
Player feature engineering.
Input: season-level player stats from Sofascore (goals, assists, shots, cards, appearances).
Output: per-player feature rows ready for model training/prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(rows: list[dict]) -> pd.DataFrame:
    """
    Convert raw Sofascore player rows into ML feature DataFrame.
    Computes per-game rates and normalises by league average.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["appearances"] >= config.MIN_APPEARANCES].copy()

    # Per-game rates
    apps = df["appearances"].clip(lower=1)
    mins = df["minutes"].clip(lower=1)

    df["goals_pg"]      = df["goals"]           / apps
    df["assists_pg"]    = df["assists"]          / apps
    df["shots_pg"]      = df["shots_total"]      / apps
    df["sot_pg"]        = df["shots_on_target"]  / apps
    df["cards_pg"]      = df["yellow_cards"]     / apps
    df["minutes_pg"]    = df["minutes"]          / apps
    df["key_passes_pg"] = df.get("key_passes", pd.Series(0, index=df.index)) / apps

    # Position encoding
    pos = df["position"].str.upper().fillna("")
    df["pos_forward"]    = pos.str.startswith("F").astype(int)
    df["pos_midfielder"] = pos.str.startswith("M").astype(int)
    df["pos_defender"]   = pos.str.startswith("D").astype(int)

    # Opponent defensive weakness — use league-level team stats as proxy
    # (opponent = unknown for season stats; use league average)
    league_avg_goals = df.groupby("league")["goals_pg"].transform("mean").clip(lower=0.01)
    df["opp_goals_conceded_pg"] = league_avg_goals  # proxy: league avg
    df["opp_shots_conceded_pg"] = df.groupby("league")["shots_pg"].transform("mean").clip(lower=1)

    # Team attack strength (team's total goals vs league average)
    team_goals = df.groupby(["league", "team"])["goals"].transform("sum")
    league_avg = df.groupby("league")["goals"].transform("sum") / df.groupby("league")["team"].transform("nunique")
    df["team_attack_str"]      = (team_goals / league_avg.clip(lower=1)).clip(0.1, 5.0)
    df["team_goals_scored_pg"] = df["goals_pg"]

    # Rest days (unknown for season data — use league average ~6 days)
    df["rest_days"] = 6.0

    # Home/away (unknown for season data — neutral 0.5)
    df["is_home"] = 0.5

    # Target variables (binary: did it happen at least once per X games?)
    # Use probability as training target: rate per game
    df["target_goals"]   = (df["goals_pg"]   > 0).astype(int)
    df["target_assists"] = (df["assists_pg"]  > 0).astype(int)
    df["target_sot"]     = (df["sot_pg"]      > 0).astype(int)
    df["target_cards"]   = (df["cards_pg"]    > 0).astype(int)

    # Fill remaining NaNs
    for col in config.PLAYER_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.reset_index(drop=True)


def build_upcoming_features(
    upcoming: list[dict],
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature rows for upcoming player predictions.
    upcoming: list of player dicts from the fixture lineup.
    history:  the full training DataFrame (output of build_features).
    """
    if not upcoming or history.empty:
        return pd.DataFrame()

    rows = []
    for p in upcoming:
        pid  = p.get("player_id")
        name = p.get("player_name", "")

        # Find this player in history
        hist = history[history["player_id"] == pid] if pid else pd.DataFrame()

        if hist.empty:
            # Try name match as fallback
            hist = history[history["player_name"].str.lower() == name.lower()]

        if hist.empty or len(hist) < 1:
            continue

        rec = hist.iloc[-1].to_dict()

        row = {
            "player_id":   pid or rec.get("player_id"),
            "player_name": name or rec.get("player_name"),
            "team":        p.get("team", rec.get("team", "")),
            "opponent":    p.get("opponent", ""),
            "position":    p.get("position", rec.get("position", "")),
            "minutes_est": p.get("minutes", int(rec.get("minutes_pg", 75))),

            "goals_pg":      rec.get("goals_pg", 0),
            "assists_pg":    rec.get("assists_pg", 0),
            "shots_pg":      rec.get("shots_pg", 0),
            "sot_pg":        rec.get("sot_pg", 0),
            "cards_pg":      rec.get("cards_pg", 0),
            "minutes_pg":    rec.get("minutes_pg", 75),
            "key_passes_pg": rec.get("key_passes_pg", 0),

            "is_home":     float(p.get("is_home", 0.5)),
            "rest_days":   7.0,

            "opp_goals_conceded_pg": rec.get("opp_goals_conceded_pg", 1.3),
            "opp_shots_conceded_pg": rec.get("opp_shots_conceded_pg", 12.0),
            "team_goals_scored_pg":  rec.get("team_goals_scored_pg", rec.get("goals_pg", 0)),
            "team_attack_str":       rec.get("team_attack_str", 1.0),

            "pos_forward":    int(str(p.get("position", "")).upper().startswith("F")),
            "pos_midfielder": int(str(p.get("position", "")).upper().startswith("M")),
            "pos_defender":   int(str(p.get("position", "")).upper().startswith("D")),
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
