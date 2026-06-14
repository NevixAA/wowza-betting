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

    if "started" in df.columns:
        df["starter_rate"] = grp["started"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["starter_rate"] = 0.8

    # ── Phase 1: ratio and per-90 rolling features (zero leakage) ────────────
    # Build intermediate per-game raw ratios, then roll them.
    mins_safe = df["minutes"].replace(0, np.nan)

    df["_shot_acc"]  = (df["shots_on_target"] / df["shots_total"].replace(0, np.nan)).fillna(0.0)
    df["shot_accuracy_rate"] = grp["_shot_acc"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    df["_kp90"]  = (df["key_passes"] / mins_safe * 90).fillna(0.0)
    df["kp_per90"] = grp["_kp90"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    df["_gi90"]  = ((df["goals"] + df["assists"]) / mins_safe * 90).fillna(0.0)
    df["goal_involvement_rate"] = grp["_gi90"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    if "duels_won" in df.columns and "duels_total" in df.columns:
        df["_box90"]  = ((df["shots_total"] + df["duels_won"]) / mins_safe * 90).fillna(0.0)
        df["box_actions_per90"] = grp["_box90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        df["_aerial"] = (df["duels_won"] / df["duels_total"].replace(0, np.nan)).fillna(0.0)
        df["aerial_won_rate"] = grp["_aerial"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        df["_duel90"] = (df["duels_total"] / mins_safe * 90).fillna(0.0)
        df["duel_intensity_per90"] = grp["_duel90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["box_actions_per90"]    = 0.0
        df["aerial_won_rate"]      = 0.0
        df["duel_intensity_per90"] = 0.0

    if "fouls_committed" in df.columns and "fouls_drawn" in df.columns:
        df["_fd90"]  = (df["fouls_drawn"]     / mins_safe * 90).fillna(0.0)
        df["fouls_drawn_per90"] = grp["_fd90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        df["_fp90"]  = (df["fouls_committed"] / mins_safe * 90).fillna(0.0)
        df["fouls_per90"] = grp["_fp90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        df["_fcr"]   = df["fouls_committed"] / (df["fouls_committed"] + df["fouls_drawn"] + 0.01)
        df["foul_committer_ratio"] = grp["_fcr"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["fouls_drawn_per90"]   = 0.0
        df["fouls_per90"]         = 0.0
        df["foul_committer_ratio"] = 0.0

    # team_corners_pg: rolling corners earned per match
    if "team_corners" in df.columns:
        df["team_corners_pg"] = grp["team_corners"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["team_corners_pg"] = 5.0  # league average fallback

    # Drop intermediate columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

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

    # ── Opponent defender-level matchup features ──────────────────────────────
    # Aggregate stats from defensive players (pos starts with 'D') per fixture/team,
    # then roll shift(1) and join on opponent — same pattern as opp_goals_conceded_pg.
    _pos = df["position"].str.upper().fillna("") if "position" in df.columns else pd.Series([""] * len(df), index=df.index)
    _def_mask = _pos.str.startswith("D")

    if _def_mask.any() and "duels_won" in df.columns:
        _defs = df[_def_mask].copy()
        _def_agg = _defs.groupby(["fixture_id", "team"]).agg(
            _date=("date", "first"),
            _aerial_won=("duels_won", "sum"),
            _aerial_total=("duels_total", "sum"),
            _fouls=("fouls_committed", "sum"),
            _cards=("yellow_cards", "sum"),
            _n=("player_id", "count"),
        ).reset_index()

        _def_agg["_aerial"] = (
            _def_agg["_aerial_won"] / _def_agg["_aerial_total"].replace(0, np.nan)
        ).fillna(0.50)
        _def_agg["_fouls_pg"] = _def_agg["_fouls"] / _def_agg["_n"].replace(0, 1)
        _def_agg["_cards_pg"] = _def_agg["_cards"] / _def_agg["_n"].replace(0, 1)

        _def_agg = _def_agg.sort_values(["team", "_date"])
        _dgrp = _def_agg.groupby("team", group_keys=False)
        _def_agg["opp_def_aerial_win_rate"] = _dgrp["_aerial"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _def_agg["opp_def_fouls_pg"] = _dgrp["_fouls_pg"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _def_agg["opp_def_cards_pg"] = _dgrp["_cards_pg"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        _def_feats = _def_agg[["fixture_id", "team",
                                "opp_def_aerial_win_rate",
                                "opp_def_fouls_pg",
                                "opp_def_cards_pg"]].rename(columns={"team": "opponent"})
        df = df.merge(_def_feats, on=["fixture_id", "opponent"], how="left")
    else:
        df["opp_def_aerial_win_rate"] = np.nan
        df["opp_def_fouls_pg"]        = np.nan
        df["opp_def_cards_pg"]        = np.nan

    df["opp_def_aerial_win_rate"] = df["opp_def_aerial_win_rate"].fillna(0.50)
    df["opp_def_fouls_pg"]        = df["opp_def_fouls_pg"].fillna(1.50)
    df["opp_def_cards_pg"]        = df["opp_def_cards_pg"].fillna(0.25)

    # Clean up temp columns from defender aggregation
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    # ── Position encoding ─────────────────────────────────────────────────────
    pos = df["position"].str.upper().fillna("")
    df["pos_forward"]    = pos.str.startswith("F").astype(int)
    df["pos_midfielder"] = pos.str.startswith("M").astype(int)
    df["pos_defender"]   = pos.str.startswith("D").astype(int)

    # ── Composite features (depend on rolling stats + position) ──────────────
    df["shooting_efficiency_index"] = (
        df["goals_pg"] / df["sot_pg"].replace(0, np.nan)
    ).fillna(0.0).clip(upper=1.0)

    df["card_exposure_index"] = (
        df["cards_pg"] * (df["minutes_pg"] / 90.0) * (1 - df["pos_forward"])
    )

    df["sot_quality_score"] = df["shot_accuracy_rate"] * df["sot_pg"]

    df["opp_adjusted_shot_threat"] = df["shots_pg"] * df["opp_sot_conceded_pg"]

    df["creative_playmaker_score"] = (
        df["kp_per90"] * (df["pos_midfielder"] + 0.5 * df["pos_forward"])
    )

    # set_piece_threat_score: aerial × corner volume × 0.30 sp rate × position weight
    df["set_piece_threat_score"] = (
        df["aerial_won_rate"]
        * (df["team_corners_pg"] / 6.0).clip(upper=2.0)
        * 0.30
        * (df["pos_defender"] + 0.7 * df["pos_midfielder"])
    )

    # ── Matchup composites (player strength vs specific defender weakness) ─────
    # aerial_matchup_score: player wins aerials AND opponent CBs lose them
    # High → aerial striker edge vs physically weak defense
    df["aerial_matchup_score"] = (
        df["aerial_won_rate"] * (1.0 - df["opp_def_aerial_win_rate"])
    ).clip(lower=0.0)

    # foul_draw_matchup_score: foul-drawing player vs foul-prone defense
    # High → more free-kick and card market opportunities
    df["foul_draw_matchup_score"] = (
        df["fouls_drawn_per90"] * df["opp_def_fouls_pg"]
    )

    # opp_def_aggression: opponent defensive aggression index
    # High → physical, card-prone defense = more disruption to creative players
    df["opp_def_aggression"] = (
        df["opp_def_fouls_pg"] * (1.0 + df["opp_def_cards_pg"])
    )

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
        pos_forward    = int(pos.startswith("F"))
        pos_midfielder = int(pos.startswith("M"))
        pos_defender   = int(pos.startswith("D"))

        mins_s = phist["minutes"].replace(0, np.nan)
        opp_sot_c = ctx.get("opp_sot_conceded_pg", 4.5)

        # Phase 1 rolling features
        shot_accuracy_rate = float(
            (phist["shots_on_target"] / phist["shots_total"].replace(0, np.nan))
            .fillna(0.0).mean()
        )
        kp_per90 = float((phist["key_passes"] / mins_s * 90).fillna(0.0).mean()) \
            if "key_passes" in phist.columns else 0.0
        goal_involvement_rate = float(
            ((phist["goals"] + phist["assists"]) / mins_s * 90).fillna(0.0).mean()
        )
        shooting_efficiency_index = float(min(goals_pg / sot_pg if sot_pg > 0 else 0.0, 1.0))

        if "duels_won" in phist.columns and "duels_total" in phist.columns:
            box_actions_per90 = float(
                ((phist["shots_total"] + phist["duels_won"]) / mins_s * 90).fillna(0.0).mean()
            )
            aerial_won_rate = float(
                (phist["duels_won"] / phist["duels_total"].replace(0, np.nan)).fillna(0.0).mean()
            )
            duel_intensity_per90 = float((phist["duels_total"] / mins_s * 90).fillna(0.0).mean())
        else:
            box_actions_per90    = shots_pg * 0.6
            aerial_won_rate      = 0.0
            duel_intensity_per90 = 0.0

        if "fouls_committed" in phist.columns and "fouls_drawn" in phist.columns:
            fouls_drawn_per90 = float((phist["fouls_drawn"] / mins_s * 90).fillna(0.0).mean())
            fouls_per90       = float((phist["fouls_committed"] / mins_s * 90).fillna(0.0).mean())
            foul_committer_ratio = float(
                (phist["fouls_committed"] / (phist["fouls_committed"] + phist["fouls_drawn"] + 0.01))
                .mean()
            )
        else:
            fouls_drawn_per90    = 0.0
            fouls_per90          = 0.0
            foul_committer_ratio = 0.0

        team_corners_pg = float(phist["team_corners_pg"].mean()) \
            if "team_corners_pg" in phist.columns else \
            float((phist["team_corners"] / phist["minutes"].replace(0, np.nan) * 90).fillna(0).mean()) \
            if "team_corners" in phist.columns else 5.0

        sot_quality_score        = round(shot_accuracy_rate * sot_pg, 4)
        opp_adjusted_shot_threat = round(shots_pg * opp_sot_c, 4)
        creative_playmaker_score = round(kp_per90 * (pos_midfielder + 0.5 * pos_forward), 4)
        card_exposure_index      = round(cards_pg * (minutes_pg / 90.0) * (1 - pos_forward), 4)
        # SET_PIECE_THREAT_SCORE: aerial ability × corner volume × 0.30 sp concession rate (league avg)
        set_piece_threat_score   = round(aerial_won_rate * (team_corners_pg / 6.0) * 0.30
                                         * (pos_defender + 0.7 * pos_midfielder), 4)

        # ── Opponent defender matchup features ──────────────────────────────
        # Look up opponent team's defenders from history directly
        opp_team = p.get("opponent", "")
        opp_def_aerial_win_rate = ctx.get("opp_def_aerial_win_rate", 0.50)
        opp_def_fouls_pg        = ctx.get("opp_def_fouls_pg",        1.50)
        opp_def_cards_pg        = ctx.get("opp_def_cards_pg",        0.25)

        if opp_team and not history.empty and "position" in history.columns:
            _opp_defs = history[
                history["team"].str.lower().str.contains(opp_team.lower()[:6], na=False) &
                history["position"].str.upper().str.startswith("D", na=False)
            ].sort_values("date").tail(n * 5)

            if not _opp_defs.empty:
                if "duels_won" in _opp_defs.columns and "duels_total" in _opp_defs.columns:
                    _awr = (_opp_defs["duels_won"] / _opp_defs["duels_total"].replace(0, np.nan)).fillna(0.5)
                    opp_def_aerial_win_rate = float(_awr.mean())
                if "fouls_committed" in _opp_defs.columns:
                    opp_def_fouls_pg = float(_opp_defs["fouls_committed"].mean())
                if "yellow_cards" in _opp_defs.columns:
                    opp_def_cards_pg = float(_opp_defs["yellow_cards"].mean())

        aerial_matchup_score    = round(max(aerial_won_rate * (1.0 - opp_def_aerial_win_rate), 0.0), 4)
        foul_draw_matchup_score = round(fouls_drawn_per90 * opp_def_fouls_pg, 4)
        opp_def_aggression      = round(opp_def_fouls_pg * (1.0 + opp_def_cards_pg), 4)

        rows.append({
            "player_id":   pid,
            "player_name": name,
            "team":        p.get("team", ""),
            "opponent":    p.get("opponent", ""),
            "position":    p.get("position", ""),
            "minutes_est": p.get("minutes", int(minutes_pg)),
            "n_games":     n_games,
            "n_prev_games": n_games,
            "data_source": "match_level",
            # Base rolling form
            "goals_pg":            round(goals_pg,   4),
            "assists_pg":          round(assists_pg,  4),
            "shots_pg":            round(shots_pg,    4),
            "sot_pg":              round(sot_pg,      4),
            "cards_pg":            round(cards_pg,    4),
            "minutes_pg":          round(minutes_pg,  1),
            "key_passes_pg":       round(kp_pg,       4),
            "sot_rate":            round(sot_rate,    4),
            "starter_rate":        round(starter_rate, 3),
            # Phase 1 features
            "shot_accuracy_rate":        round(shot_accuracy_rate,        4),
            "kp_per90":                  round(kp_per90,                   4),
            "goal_involvement_rate":     round(goal_involvement_rate,      4),
            "shooting_efficiency_index": round(shooting_efficiency_index,  4),
            "box_actions_per90":         round(box_actions_per90,          4),
            "aerial_won_rate":           round(aerial_won_rate,            4),
            "duel_intensity_per90":      round(duel_intensity_per90,       4),
            "fouls_drawn_per90":         round(fouls_drawn_per90,          4),
            "fouls_per90":               round(fouls_per90,                4),
            "foul_committer_ratio":      round(foul_committer_ratio,       4),
            "sot_quality_score":         sot_quality_score,
            "opp_adjusted_shot_threat":  opp_adjusted_shot_threat,
            "creative_playmaker_score":  creative_playmaker_score,
            "card_exposure_index":       card_exposure_index,
            "team_corners_pg":           round(team_corners_pg, 3),
            "set_piece_threat_score":    set_piece_threat_score,
            # Match context
            "is_home":               float(p.get("is_home", 0.5)),
            "opp_goals_conceded_pg": ctx.get("opp_goals_conceded_pg", 1.3),
            "opp_sot_conceded_pg":   opp_sot_c,
            "team_goals_pg_roll":    ctx.get("team_goals_pg_roll",    1.3),
            # Opponent defender matchup features
            "opp_def_aerial_win_rate": round(opp_def_aerial_win_rate, 4),
            "opp_def_fouls_pg":        round(opp_def_fouls_pg,        4),
            "opp_def_cards_pg":        round(opp_def_cards_pg,        4),
            "aerial_matchup_score":    aerial_matchup_score,
            "foul_draw_matchup_score": foul_draw_matchup_score,
            "opp_def_aggression":      opp_def_aggression,
            # Position
            "pos_forward":    pos_forward,
            "pos_midfielder": pos_midfielder,
            "pos_defender":   pos_defender,
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
    """Goal Edge Score (0.0-1.0). Gates goals/SOT signals.

    For forwards: replaces xg_form with clinical efficiency (sot_rate * opp_weakness).
    For non-forwards: uses legacy goals_pg form (header/set-piece scorers).
    """
    xg_form  = min(row.get("goals_pg", 0) / 0.50, 1.0)
    shot_vol = min(row.get("shots_pg", 0) / 3.5, 1.0)
    pen      = 1.0 if penalty_duty else 0.0
    opp      = min(opp_weakness / 1.5, 1.0)
    min_sec  = min(row.get("minutes_pg", 0) / 85.0, 1.0)

    is_fwd = float(row.get("pos_forward", 0))
    sot_rate = row.get("sot_rate", 0.0)
    # clinical: forward accuracy × defensive leakiness, normalised to [0,1]
    clinical = min(sot_rate * opp_weakness / (0.35 * 1.5), 1.0)
    form = clinical if is_fwd else xg_form

    return round(0.40 * form + 0.20 * shot_vol + 0.15 * pen + 0.15 * opp + 0.10 * min_sec, 3)
