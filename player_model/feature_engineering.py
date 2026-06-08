"""Player feature engineering — builds rolling per-player stats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  flat player rows (from data_fetcher.parse_players)
    Output: same rows with PLAYER_FEATURE_COLS added, targets added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    n = config.ROLLING_N

    def _roll(series: pd.Series) -> pd.Series:
        """Leak-free rolling mean: shift(1) then rolling(n)."""
        return series.shift(1).rolling(n, min_periods=1).mean()

    grp = df.groupby("player_id")

    df["goals_pg"]      = grp["goals"].transform(_roll)
    df["assists_pg"]    = grp["assists"].transform(_roll)
    df["shots_pg"]      = grp["shots_total"].transform(_roll)
    df["sot_pg"]        = grp["shots_on"].transform(_roll)
    df["cards_pg"]      = grp["yellow_card"].transform(_roll)
    df["minutes_pg"]    = grp["minutes"].transform(_roll)
    df["key_passes_pg"] = grp["key_passes"].transform(_roll)

    # Appearance count — filter players with too few games
    df["appearances"] = grp["date"].transform("cumcount")

    # Position encoding
    df["pos_forward"]   = df["position"].str.startswith("F").astype(int)
    df["pos_midfielder"]= df["position"].str.startswith("M").astype(int)
    df["pos_defender"]  = df["position"].str.startswith("D").astype(int)

    # Opponent defensive weakness (rolling goals conceded per game by opponent)
    opp_stats = (
        df.groupby(["opponent", "date"])
        .agg(goals_conceded=("goals", "sum"), shots_conceded=("shots_total", "sum"))
        .reset_index()
        .sort_values(["opponent", "date"])
    )
    opp_stats["opp_goals_conceded_pg"] = (
        opp_stats.groupby("opponent")["goals_conceded"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    opp_stats["opp_shots_conceded_pg"] = (
        opp_stats.groupby("opponent")["shots_conceded"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.merge(
        opp_stats[["opponent", "date", "opp_goals_conceded_pg", "opp_shots_conceded_pg"]],
        on=["opponent", "date"], how="left",
    )

    # Team attack strength (team's rolling goals scored per game)
    team_goals = (
        df.groupby(["team", "date"])["goals"].sum()
        .reset_index()
        .sort_values(["team", "date"])
    )
    team_goals["team_goals_scored_pg"] = (
        team_goals.groupby("team")["goals"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.merge(
        team_goals[["team", "date", "team_goals_scored_pg"]],
        on=["team", "date"], how="left",
    )
    # Normalise to a strength ratio
    league_avg = df["team_goals_scored_pg"].median()
    df["team_attack_str"] = df["team_goals_scored_pg"] / league_avg.clip(0.1)

    # Rest days
    df["rest_days"] = (
        df.groupby("player_id")["date"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(7)
        .clip(1, 21)
    )

    # Fill remaining NaNs
    for col in config.PLAYER_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def build_upcoming_features(
    upcoming: list[dict],
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features for a list of upcoming player dicts.
    upcoming: list of dicts with keys: player_id, player_name, team, opponent, is_home, position, minutes (est.)
    history:  full historical DataFrame (output of build_features on training data)
    """
    if not upcoming:
        return pd.DataFrame()

    rows = []
    for p in upcoming:
        pid = p["player_id"]
        hist = history[history["player_id"] == pid].sort_values("date")

        if len(hist) < config.MIN_APPEARANCES:
            continue

        last = hist.tail(config.ROLLING_N)
        opp  = p["opponent"]
        opp_hist = history[history["opponent"] == opp].sort_values("date").tail(10)

        row = {
            "player_id":   pid,
            "player_name": p["player_name"],
            "team":        p["team"],
            "opponent":    opp,
            "is_home":     p.get("is_home", 0),
            "position":    p.get("position", ""),
            "minutes_est": p.get("minutes", 90),

            "goals_pg":      last["goals"].mean(),
            "assists_pg":    last["assists"].mean(),
            "shots_pg":      last["shots_total"].mean(),
            "sot_pg":        last["shots_on"].mean(),
            "cards_pg":      last["yellow_card"].mean(),
            "minutes_pg":    last["minutes"].mean(),
            "key_passes_pg": last["key_passes"].mean(),

            "opp_goals_conceded_pg": opp_hist["goals"].mean() if not opp_hist.empty else 1.3,
            "opp_shots_conceded_pg": opp_hist["shots_total"].mean() if not opp_hist.empty else 12.0,
            "team_goals_scored_pg":  last["goals"].mean(),
            "team_attack_str":       last["goals"].mean() / max(history["goals"].median(), 0.1),

            "pos_forward":    int(str(p.get("position", "")).upper().startswith("F")),
            "pos_midfielder": int(str(p.get("position", "")).upper().startswith("M")),
            "pos_defender":   int(str(p.get("position", "")).upper().startswith("D")),

            "rest_days": 7,
        }
        rows.append(row)

    return pd.DataFrame(rows)
