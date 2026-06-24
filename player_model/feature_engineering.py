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

    if "rating" in df.columns:
        df["rating_pg"] = grp["rating"].transform(
            lambda x: x.replace(0, np.nan).shift(1).rolling(n, min_periods=1).mean()
        ).fillna(7.0)
    else:
        df["rating_pg"] = 7.0

    df["sot_rate"] = (
        df["sot_pg"] / df["shots_pg"].replace(0, np.nan)
    ).fillna(0.0)

    # ── League quality columns (added by enrich_league_quality preprocessing) ─
    for _lq_col in ["league_tier", "league_quality", "player_career_avg_quality",
                     "opp_def_player_rating_pg", "opp_top_def_rating",
                     "opp_def_player_quality", "context_quality_discount",
                     "quality_mismatch_goals", "quality_mismatch_sot"]:
        if _lq_col not in df.columns:
            _defaults = {
                "league_tier": 3, "league_quality": 0.5,
                "player_career_avg_quality": 0.65,
                "opp_def_player_rating_pg": 6.8, "opp_top_def_rating": 7.0,
                "opp_def_player_quality": 1.0,
                "context_quality_discount": 1.0,
                "quality_mismatch_goals": 0.0, "quality_mismatch_sot": 0.0,
            }
            df[_lq_col] = _defaults.get(_lq_col, 0.0)
        else:
            df[_lq_col] = pd.to_numeric(df[_lq_col], errors="coerce").fillna(
                {"league_tier": 3, "league_quality": 0.5,
                 "player_career_avg_quality": 0.65,
                 "opp_def_player_rating_pg": 6.8, "opp_top_def_rating": 7.0,
                 "opp_def_player_quality": 1.0, "context_quality_discount": 1.0,
                 "quality_mismatch_goals": 0.0, "quality_mismatch_sot": 0.0,
                }.get(_lq_col, 0.0))

    # Count of previous games this player has in the dataset
    df["n_prev_games"] = grp["date"].transform("cumcount")

    # ── Venue-split rolling rates (home vs away performances differ) ──────────
    if "is_home" in df.columns:
        df["_idx"] = np.arange(len(df))
        for _raw, _pref in [
            ("goals",            "goals"),
            ("shots_total",      "shots"),
            ("shots_on_target",  "sot"),
        ]:
            if _raw not in df.columns:
                continue
            for _venue, _is_h in [("home", True), ("away", False)]:
                _col      = f"{_pref}_{_venue}_pg"
                _fallback = f"{_pref}_pg"
                _mask     = df["is_home"].astype(bool) == _is_h
                _vsub     = df[_mask][["_idx", "player_id", "date", _raw]].copy()
                if not _vsub.empty:
                    _vsub = _vsub.sort_values(["player_id", "date"])
                    _vsub[_col] = _vsub.groupby("player_id")[_raw].transform(
                        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
                    )
                    df = df.merge(_vsub[["_idx", _col]], on="_idx", how="left")
                else:
                    df[_col] = np.nan
                df[_col] = df[_col].fillna(df.get(_fallback, pd.Series(0.0, index=df.index)))
        df.drop(columns=["_idx"], inplace=True, errors="ignore")
    else:
        for _pref, _base in [("goals", "goals_pg"), ("shots", "shots_pg"), ("sot", "sot_pg")]:
            df[f"{_pref}_home_pg"] = df.get(_base, 0.0)
            df[f"{_pref}_away_pg"] = df.get(_base, 0.0)

    if "started" in df.columns:
        df["starter_rate"] = grp["started"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["starter_rate"] = 0.8

    # ── Phase 1: ratio and per-90 rolling features (zero leakage) ────────────
    # Refresh groupby — new columns added above (venue splits, starter_rate) aren't
    # visible in the original grp object under pandas copy-on-write semantics.
    grp = df.groupby("player_id", group_keys=False)

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

    # Career-to-date averages — stable priors complementing the volatile last-5 rolling stats
    # Expanding (not rolling) so they capture the full career baseline rather than just last-N.
    df["career_goals_pg"]   = grp["goals"].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.0)
    df["career_sot_pg"]     = grp["shots_on_target"].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.0)
    df["career_shots_pg"]   = grp["shots_total"].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.0)
    df["career_assists_pg"] = grp["assists"].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.0)

    # ── Age and physical profile features ────────────────────────────────────
    if "age" in df.columns and df["age"].gt(0).any():
        df["age_peak_delta"] = (df["age"] - 27).abs()
    else:
        df["age_peak_delta"] = 0.0
        df["age"] = 25.0  # fallback default

    if "height_cm" not in df.columns:
        df["height_cm"] = 180.0
    if "weight_kg" not in df.columns:
        df["weight_kg"] = 75.0

    # Aerial interaction — physical build × aerial success rate
    if "aerial_won_rate" in df.columns:
        df["height_aerial_interaction"] = (
            (df["height_cm"].clip(160, 200) - 160) / 40  # normalize to 0-1
        ) * df["aerial_won_rate"]
    else:
        df["height_aerial_interaction"] = 0.0

    # ── Season-level new fields (from extended fetch_player_season_stats) ────
    for col, default in [
        ("season_start_rate",     0.7),
        ("season_pass_accuracy",  0.75),
        ("season_dribble_pg",     0.3),
        ("season_fouls_pg",       1.0),
        ("season_fouls_drawn_pg", 0.8),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # ── Injury/sidelined features ────────────────────────────────────────────
    # These require the sidelined data to have been pre-merged into df.
    # If not present, default to safe neutral values.
    for col, default in [
        ("chronic_injury_risk",      0.0),
        ("days_since_last_injury",   365.0),
        ("return_from_injury_flag",  0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── Referee strictness ───────────────────────────────────────────────────
    # referee column added by _fetch_league_fixtures(); compute per-referee avg
    if "referee" in df.columns and df["referee"].notna().any():
        ref_grp = df.groupby("referee")["yellow_cards"].mean()
        df["referee_yellows_pg"] = df["referee"].map(ref_grp).fillna(df["yellow_cards"].mean())
        ref_cards_mean = df["yellow_cards"].mean() or 0.3
        ref_max = df.groupby("referee")["yellow_cards"].mean().max() or 1.0
        df["referee_strictness"] = df["referee_yellows_pg"] / (ref_max or 1.0)
    else:
        df["referee_yellows_pg"]  = df["yellow_cards"].mean() if "yellow_cards" in df.columns else 0.3
        df["referee_strictness"]  = 0.5

    # ── New raw rolling features from extended data fields ───────────────────
    # Defensive actions
    if "tackles_total" in df.columns:
        df["tackles_pg"] = grp["tackles_total"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        df["_tack90"] = (df["tackles_total"] / mins_safe * 90).fillna(0.0)
        df["tackles_per90"] = grp["_tack90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["tackles_pg"] = 0.0;  df["tackles_per90"] = 0.0

    if "interceptions" in df.columns:
        df["interceptions_pg"] = grp["interceptions"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["interceptions_pg"] = 0.0

    if "tackles_total" in df.columns and "interceptions" in df.columns:
        df["_def90"] = (
            (df["tackles_total"] + df["interceptions"]) / mins_safe * 90
        ).fillna(0.0)
        df["defensive_actions_per90"] = grp["_def90"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["defensive_actions_per90"] = 0.0

    # Dribbles
    if "dribbles_success" in df.columns and "dribbles_attempted" in df.columns:
        df["dribbles_pg"] = grp["dribbles_success"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        df["_drib_acc"] = (
            df["dribbles_success"] / df["dribbles_attempted"].replace(0, np.nan)
        ).fillna(0.0)
        df["dribble_success_rate"] = grp["_drib_acc"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["dribbles_pg"] = 0.0;  df["dribble_success_rate"] = 0.0

    if "dribbles_past" in df.columns:
        df["dribbled_past_pg"] = grp["dribbles_past"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["dribbled_past_pg"] = 0.0

    # Red cards, pass involvement, offsides
    if "red_cards" in df.columns:
        df["red_cards_pg"] = grp["red_cards"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["red_cards_pg"] = 0.0

    if "passes_total" in df.columns:
        df["passes_pg"] = grp["passes_total"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["passes_pg"] = 0.0

    if "offsides" in df.columns:
        df["offsides_pg"] = grp["offsides"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["offsides_pg"] = 0.0

    # Penalty stats
    if "penalty_won" in df.columns:
        df["penalties_won_pg"] = grp["penalty_won"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["penalties_won_pg"] = 0.0

    if "penalty_scored" in df.columns and "penalty_missed" in df.columns:
        _pen_total = (df["penalty_scored"] + df["penalty_missed"]).replace(0, np.nan)
        df["_pen_conv"] = (df["penalty_scored"] / _pen_total).fillna(0.75)
        df["penalty_conversion_rate"] = grp["_pen_conv"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["penalty_conversion_rate"] = 0.75

    # Goalkeeper own rolling stats (meaningful for GKs; zero for others)
    if "saves" in df.columns:
        df["saves_pg"] = grp["saves"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        if "goals_conceded" in df.columns:
            df["goals_conceded_gk_pg"] = grp["goals_conceded"].transform(
                lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            df["_gk_sr"] = df["saves"] / (df["saves"] + df["goals_conceded"] + 0.001)
        else:
            df["goals_conceded_gk_pg"] = 0.0
            df["_gk_sr"] = df["saves"] / (df["saves"] + 0.001)
        df["gk_save_rate"] = grp["_gk_sr"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    else:
        df["saves_pg"] = 0.0;  df["goals_conceded_gk_pg"] = 0.0;  df["gk_save_rate"] = 0.72

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

    # ── Opponent mean defender height ─────────────────────────────────────────
    # Aggregate height of defensive players per fixture/team, join as opponent feature
    if _def_mask.any() and "height_cm" in df.columns:
        _height_agg = (
            df[_def_mask]
            .groupby(["fixture_id", "team"])["height_cm"]
            .mean()
            .reset_index()
            .rename(columns={"team": "opponent", "height_cm": "opp_def_mean_height"})
        )
        df = df.merge(_height_agg, on=["fixture_id", "opponent"], how="left")
    else:
        df["opp_def_mean_height"] = np.nan
    df["opp_def_mean_height"]   = df["opp_def_mean_height"].fillna(181.0)
    df["height_diff_vs_opp_def"] = df["height_cm"].fillna(180.0) - df["opp_def_mean_height"]
    df["height_advantage_score"] = (df["height_diff_vs_opp_def"] / 10.0).clip(-2.0, 2.0)

    # ── Opponent set piece goals conceded ─────────────────────────────────────
    # Only available when enrich_sp_events() was called before build_features()
    if "sp_goal" in df.columns:
        _sp_match = (
            df.groupby(["fixture_id", "team"])
            .agg(date=("date", "first"), _sp_scored=("sp_goal", "sum"))
            .reset_index()
        )
        _sp_opp = _sp_match.rename(columns={
            "team": "opponent", "_sp_scored": "_sp_against"}).drop(columns=["date"])
        _sp_def = _sp_match.merge(_sp_opp, on="fixture_id")
        _sp_def = _sp_def[_sp_def["team"] != _sp_def["opponent"]].copy()
        _sp_def = _sp_def.sort_values(["team", "date"])
        _spgrp  = _sp_def.groupby("team", group_keys=False)
        _sp_def["opp_sp_goals_conceded_pg"] = _spgrp["_sp_against"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _sp_opp_feats = _sp_def[["fixture_id","team","opp_sp_goals_conceded_pg"]].rename(
            columns={"team": "opponent"})
        df = df.merge(_sp_opp_feats, on=["fixture_id","opponent"], how="left")
    else:
        df["opp_sp_goals_conceded_pg"] = np.nan
    df["opp_sp_goals_conceded_pg"] = df["opp_sp_goals_conceded_pg"].fillna(0.12)

    # ── Opposition strength index (quality adjustment) ────────────────────────
    # Each team's rolling goals conceded relative to the league average.
    # opp_strength_index > 1 = weak defence (easy), < 1 = strong defence (hard).
    _league_avg = (
        match_agg.groupby("league")["goals_scored"].transform("mean")
        if "league" in match_agg.columns
        else match_agg["goals_scored"].mean()
    )
    match_agg["_league_avg"] = (
        df.groupby(["fixture_id","team"])["league"].transform("first").map(
            df.groupby("league")["goals"].mean()
        ).fillna(1.3)
        if "league" in df.columns
        else 1.3
    )
    _conc_agg = match_agg.copy()
    _conc_agg["_opp_concede"] = _conc_agg["goals_scored"]
    _conc_agg = _conc_agg.sort_values(["team", "date"])
    _cgrp = _conc_agg.groupby("team", group_keys=False)
    _conc_agg["_opp_conc_roll"] = _cgrp["_opp_concede"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    _conc_feats = _conc_agg[["fixture_id","team","_opp_conc_roll"]].rename(
        columns={"team": "opponent"})
    df = df.merge(_conc_feats, on=["fixture_id","opponent"], how="left")
    _lg_mean = df.groupby("league")["goals"].transform("mean").fillna(1.3) if "league" in df.columns else 1.3
    df["opp_strength_index"] = (
        df["_opp_conc_roll"].fillna(1.3) / _lg_mean.replace(0, 1.3)
    ).clip(0.3, 3.0)
    df["quality_adj_goals_pg"] = (df["goals_pg"] * df["opp_strength_index"]).clip(0.0, 5.0)
    df["quality_adj_sot_pg"]   = (df["sot_pg"]   * df["opp_strength_index"]).clip(0.0, 10.0)
    df.drop(columns=[c for c in df.columns if c.startswith("_opp_conc")], inplace=True, errors="ignore")

    # Clean up temp columns from defender aggregation
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    # ── Opponent goalkeeper matchup features ──────────────────────────────────
    # Aggregate GK saves + conceded per team per fixture → roll shift(1) → join on opponent.
    # "opponent" column already computed above.
    _gk_pos = df["position"].str.upper().fillna("") if "position" in df.columns \
        else pd.Series([""] * len(df), index=df.index)
    _gk_mask = _gk_pos.str.startswith("G")

    if _gk_mask.any() and "saves" in df.columns and "goals_conceded" in df.columns:
        _gks = df[_gk_mask].copy()
        _gk_agg = _gks.groupby(["fixture_id", "team"]).agg(
            _date=("date", "first"),
            _saves=("saves", "sum"),
            _conceded=("goals_conceded", "sum"),
        ).reset_index()

        _gk_agg["_save_rate"] = (
            _gk_agg["_saves"] / (_gk_agg["_saves"] + _gk_agg["_conceded"] + 0.001)
        )
        _gk_agg = _gk_agg.sort_values(["team", "_date"])
        _gkgrp  = _gk_agg.groupby("team", group_keys=False)
        _gk_agg["opp_gk_save_rate"] = _gkgrp["_save_rate"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _gk_agg["opp_gk_saves_pg"]  = _gkgrp["_saves"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())

        _gk_feats = _gk_agg[["fixture_id", "team",
                              "opp_gk_save_rate", "opp_gk_saves_pg"]].rename(
                                  columns={"team": "opponent"})
        df = df.merge(_gk_feats, on=["fixture_id", "opponent"], how="left")
    else:
        df["opp_gk_save_rate"] = np.nan
        df["opp_gk_saves_pg"]  = np.nan

    df["opp_gk_save_rate"] = df["opp_gk_save_rate"].fillna(0.72)
    df["opp_gk_saves_pg"]  = df["opp_gk_saves_pg"].fillna(3.0)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    # ── Position-split opponent concede stats ─────────────────────────────────
    # How many goals/SOT do FORWARDS vs MIDFIELDERS score against each team?
    # More precise matchup signal than the team-level opp_goals_conceded_pg.
    _pos_upper = df["position"].str.upper().fillna("") if "position" in df.columns \
        else pd.Series([""] * len(df), index=df.index)
    for _pp, _sfx, _g_def, _s_def in [("F", "fwd", 1.0, 3.5), ("M", "mid", 0.4, 2.0)]:
        _pmask = _pos_upper.str.startswith(_pp)
        if not _pmask.any():
            df[f"opp_goals_conceded_vs_{_sfx}_pg"] = _g_def
            df[f"opp_sot_conceded_vs_{_sfx}_pg"]   = _s_def
            continue
        _pagg = (
            df[_pmask]
            .groupby(["fixture_id", "team"])
            .agg(_date=("date", "first"), _goals=("goals", "sum"), _sot=("shots_on_target", "sum"))
            .reset_index()
        )
        _popp = _pagg[["fixture_id", "team", "_goals", "_sot"]].rename(columns={
            "team": "opponent_team", "_goals": f"_gvs{_sfx}", "_sot": f"_svs{_sfx}",
        })
        _pm = _pagg[["fixture_id", "team", "_date"]].merge(_popp, on="fixture_id")
        _pm = _pm[_pm["team"] != _pm["opponent_team"]].copy()
        _pm = _pm.sort_values(["team", "_date"])
        _ptg = _pm.groupby("team", group_keys=False)
        _pm[f"opp_goals_conceded_vs_{_sfx}_pg"] = _ptg[f"_gvs{_sfx}"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _pm[f"opp_sot_conceded_vs_{_sfx}_pg"] = _ptg[f"_svs{_sfx}"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        _pf = _pm[["fixture_id", "team",
                    f"opp_goals_conceded_vs_{_sfx}_pg",
                    f"opp_sot_conceded_vs_{_sfx}_pg"]].rename(columns={"team": "opponent"})
        df = df.merge(_pf, on=["fixture_id", "opponent"], how="left")
        df[f"opp_goals_conceded_vs_{_sfx}_pg"] = df[f"opp_goals_conceded_vs_{_sfx}_pg"].fillna(_g_def)
        df[f"opp_sot_conceded_vs_{_sfx}_pg"]   = df[f"opp_sot_conceded_vs_{_sfx}_pg"].fillna(_s_def)
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    # ── Opponent midfielder discipline (cards market) ─────────────────────────
    # How many fouls/cards do opponent MIDFIELDERS commit?
    # Midfield battles cause the most yellows — this is the primary card matchup signal.
    if "position" in df.columns:
        _pos_upper2 = df["position"].str.upper().fillna("")
        _mmask = _pos_upper2.str.startswith("M")
        if _mmask.any():
            _magg = (
                df[_mmask]
                .groupby(["fixture_id", "team"])
                .agg(_date=("date", "first"),
                     _fouls=("fouls_committed", "sum"),
                     _cards=("yellow_cards", "sum"))
                .reset_index()
            )
            _mopp = _magg[["fixture_id", "team", "_fouls", "_cards"]].rename(columns={
                "team": "opponent_team",
            })
            _mm = _magg[["fixture_id", "team", "_date"]].merge(_mopp, on="fixture_id")
            _mm = _mm[_mm["team"] != _mm["opponent_team"]].copy()
            _mm = _mm.sort_values(["team", "_date"])
            _mmg = _mm.groupby("team", group_keys=False)
            _mm["opp_mid_fouls_pg"] = _mmg["_fouls"].transform(
                lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            _mm["opp_mid_cards_pg"] = _mmg["_cards"].transform(
                lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            _mf = _mm[["fixture_id", "team", "opp_mid_fouls_pg", "opp_mid_cards_pg"]].rename(
                columns={"team": "opponent"})
            df = df.merge(_mf, on=["fixture_id", "opponent"], how="left")
        else:
            df["opp_mid_fouls_pg"] = 2.0
            df["opp_mid_cards_pg"] = 0.4
    else:
        df["opp_mid_fouls_pg"] = 2.0
        df["opp_mid_cards_pg"] = 0.4
    df["opp_mid_fouls_pg"] = df.get("opp_mid_fouls_pg", 2.0)
    df["opp_mid_fouls_pg"] = pd.to_numeric(df["opp_mid_fouls_pg"], errors="coerce").fillna(2.0)
    df["opp_mid_cards_pg"] = pd.to_numeric(df.get("opp_mid_cards_pg", 0.4), errors="coerce").fillna(0.4)
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

    # ── Set piece event rolling features (requires enrich_sp_events preprocessing) ──
    for _raw_col, _feat_name in [
        ("sp_goal",    "sp_goals_pg"),
        ("headed_goal","headed_goals_pg"),
        ("fk_goal",    "fk_goals_pg"),
        ("sp_assist",  "sp_assist_pg"),
    ]:
        if _raw_col in df.columns:
            df[_feat_name] = grp[_raw_col].transform(
                lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        else:
            df[_feat_name] = 0.0

    for _raw_col, _feat_name in [
        ("sp_goal",   "career_sp_goals_rate"),
        ("sp_assist", "career_sp_assist_rate"),
    ]:
        if _raw_col in df.columns:
            df[_feat_name] = grp[_raw_col].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0.0))
        else:
            df[_feat_name] = 0.0

    _goal_denom = df["goals_pg"].replace(0, np.nan)
    df["sp_goals_share"]     = (df["sp_goals_pg"]     / _goal_denom).clip(0, 1).fillna(0.0)
    df["headed_goals_share"] = (df["headed_goals_pg"] / _goal_denom).clip(0, 1).fillna(0.0)

    # sp_taker_score: probability 0-1 that this player DELIVERS set pieces
    # Signals: career SP assist rate (they created SP goals), FK goals (direct taker),
    # position prior (wide/AM > CM > CB > GK)
    _pos_taker = (
        df["pos_midfielder"] * 0.45
        + df["pos_forward"]  * 0.30
        + df["pos_defender"] * 0.05
    ).clip(0, 1)
    df["sp_taker_score"] = (
        (df["career_sp_assist_rate"].clip(0, 0.15) / 0.15) * 0.60
        + (df["fk_goals_pg"].clip(0, 0.05)         / 0.05) * 0.25
        + _pos_taker                                        * 0.15
    ).clip(0, 1)

    # sp_receiver_score: high when NOT a taker AND high aerial ability
    # This is the feature for predicting defenders scoring from corners
    df["sp_receiver_score"] = (
        (1.0 - df["sp_taker_score"])
        * df["aerial_won_rate"]
        * (df["pos_defender"] + 0.6 * df["pos_midfielder"])
    ).clip(0, 1)

    # Upgraded set_piece_threat_score using real SP data where available
    df["set_piece_threat_score"] = (
        df["aerial_won_rate"]
        * (df["team_corners_pg"] / 6.0).clip(upper=2.0)
        * df["opp_sp_goals_conceded_pg"].fillna(0.12) / 0.12
        * (df["pos_defender"] + 0.7 * df["pos_midfielder"])
    ).clip(0, 3.0)

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

    # ── New composite features using extended stats ───────────────────────────
    # Attacker vs GK matchup — how many SOTs actually beat this GK
    df["att_vs_gk_threat"] = (
        df["sot_pg"] * (1.0 - df["opp_gk_save_rate"])
    ).clip(lower=0.0)

    # Clinical advantage over GK — shot accuracy vs GK save rate
    df["clinical_vs_gk"] = (
        df["shot_accuracy_rate"] * (1.0 - df["opp_gk_save_rate"])
    ).fillna(0.0).clip(lower=0.0)

    # Dribble creativity — volume × quality
    df["dribble_creativity_score"] = (
        df["dribbles_pg"] * df["dribble_success_rate"]
    )

    # Defensive solidity — high defensive actions, rarely dribbled past
    df["defensive_solidity"] = (
        df["defensive_actions_per90"]
        * (1.0 - df["dribbled_past_pg"] / (df["dribbled_past_pg"] + 1.0))
    ).clip(lower=0.0)

    # Penalty threat — direct penalty wins + foul drawing (scaled)
    df["penalty_threat_score"] = (
        df["penalties_won_pg"] * 3.0 + df["fouls_drawn_per90"] * 0.08
    ).clip(upper=1.0)

    # Card risk index — yellow + red cards + foul tendency (capped at 1.0 = normalized)
    df["card_risk_index"] = (
        df["cards_pg"] * 2.0 + df["red_cards_pg"] * 3.0 + df["foul_committer_ratio"] * 0.5
    ).clip(upper=1.0)

    # Midfield engine — pass volume + key passes + tackling, midfielder/defender weighted
    df["midfield_engine_score"] = (
        (df["passes_pg"] / 50.0).clip(upper=1.0) * 0.4
        + (df["kp_per90"] / 3.0).clip(upper=1.0) * 0.4
        + (df["tackles_per90"] / 5.0).clip(upper=1.0) * 0.2
    ) * (df["pos_midfielder"] + 0.3 * df["pos_defender"])

    # Offside aggressiveness — forwards pushing the line
    df["offside_aggressiveness"] = (
        df["offsides_pg"] * df["pos_forward"]
    ).clip(upper=2.0)

    # ── Agent-designed composite features ────────────────────────────────────
    # Attacker vs GK — full conversion chain (shots × accuracy × (1 - save_rate))
    df["volume_shot_penetration"] = (
        df["shots_pg"] * df["shot_accuracy_rate"] * (1.0 - df["opp_gk_save_rate"])
    ).clip(lower=0.0)

    # Attacker vs GK — adds conversion quality (goals/SOT finisher rating)
    df["finishing_threat_index"] = (
        df["sot_pg"] * df["shooting_efficiency_index"] * (1.0 - df["opp_gk_save_rate"])
    ).clip(lower=0.0)

    # Physical box presence vs GK — dangerous actions in the area vs GK stopping power
    df["box_dominance_vs_gk"] = (
        (df["box_actions_per90"] + df["dribbles_pg"]) * (1.0 - df["opp_gk_save_rate"])
    ).clip(lower=0.0)

    # Penalizes wasteful aggressors — offside-discounted threat
    df["aggression_adjusted_threat"] = (
        df["sot_pg"] * (1.0 - df["opp_gk_save_rate"])
        / (1.0 + df["offsides_pg"] * 0.3)
    ).clip(lower=0.0)

    # How easily a player loses the ball — positional vulnerability signal
    df["defensive_vulnerability_index"] = (
        (
            df["dribbled_past_pg"] * 1.5
            + (1.0 - df["dribble_success_rate"]) * 0.5
            + (1.0 - df["aerial_won_rate"]) * 0.5
        ).clip(upper=2.0)
        * (df["pos_forward"] * 1.0 + df["pos_midfielder"] * 0.7 + df["pos_defender"] * 0.3)
    )

    # Ball-carrying and progression ability — combines dribbles + passing volume + quality
    df["progressive_carrier_score"] = (
        (
            (df["dribbles_pg"] / 3.0).clip(upper=1.0) * 0.45
            + (df["passes_pg"] / 60.0).clip(upper=1.0) * 0.35
            + df["dribble_success_rate"] * 0.20
        ).clip(upper=1.0)
        * (df["pos_midfielder"] * 1.0 + df["pos_forward"] * 0.8 + df["pos_defender"] * 0.5)
    )

    # Bilateral foul involvement — player in constant friction (cards market primary target)
    df["disciplinary_pressure_index"] = (
        df["fouls_drawn_per90"] * 0.20
        + df["fouls_per90"] * 0.25
        + df["cards_pg"] * 1.5
        + df["duel_intensity_per90"] * 0.10
    ).clip(upper=2.0)

    # Foul-drawing market composite (fouls drawn + penalty area threat + carrying)
    df["foul_magnet_score"] = (
        df["fouls_drawn_per90"] * 0.30
        + df["penalty_threat_score"] * 0.40
        + df["progressive_carrier_score"] * 0.30
    ).clip(upper=1.0)

    # ── Position-specific matchup composites ─────────────────────────────────
    df["forward_matchup_score"] = (
        df["sot_pg"] * df["opp_goals_conceded_vs_fwd_pg"] * df["pos_forward"]
    ).clip(lower=0.0)

    df["mid_threat_vs_defense"] = (
        df["kp_per90"] * df["opp_goals_conceded_vs_mid_pg"] * df["pos_midfielder"]
    ).clip(lower=0.0)

    # ── Additional player-vs-player matchup formulas ─────────────────────────
    # Box-active player vs goal-conceding defense
    df["box_threat_vs_leaky_defense"] = (
        df["box_actions_per90"] * df["opp_goals_conceded_pg"]
    ).clip(lower=0.0)

    # Efficient finisher vs shot-allowing GK
    df["efficiency_vs_leaky_keeper"] = (
        df["shooting_efficiency_index"] * df["opp_sot_conceded_pg"]
    ).clip(lower=0.0)

    # Playmaker vs card-prone defense — key passes exploit disciplinary-fragile CBs
    df["kp_vs_aggressive_defense"] = (
        df["kp_per90"] * df["opp_def_cards_pg"]
    ).clip(lower=0.0)

    # Team momentum × defensive weakness × forward position — three-way interaction
    df["team_momentum_forward_matchup"] = (
        df["team_goals_pg_roll"] * df["opp_goals_conceded_pg"] * df["pos_forward"]
    ).clip(lower=0.0)

    # Set piece aerial threat × corner delivery volume
    df["set_piece_corner_matchup"] = (
        df["set_piece_threat_score"] * (df["team_corners_pg"] / 6.0).clip(upper=2.0)
    ).clip(lower=0.0)

    # ── Full set piece composites ─────────────────────────────────────────────
    # defender_sp_edge: the core formula the user designed
    #   aerial ability × height advantage × opp SP weakness × NOT taker × corner volume
    _h_adj = (df["height_advantage_score"] + 2.0) / 4.0   # remap [-2,+2] → [0,1]
    df["defender_sp_edge"] = (
        df["sp_receiver_score"]
        * _h_adj
        * (df["opp_sp_goals_conceded_pg"] / 0.12)
        * (df["team_corners_pg"] / 6.0).clip(0.3, 2.0)
    ).clip(0, 4.0)

    # ── Defender SOT edge (biggest edge in player props market) ──────────────
    # Defenders at corners/FKs put ~5x more shots ON TARGET than they score.
    # Bookmakers price defenders on SOT at 4.0-8.0 because "defenders don't shoot"
    # but tall aerial CBs at high-corner teams have real 20-40% SOT probability.
    # This feature is the primary signal for the sot/sot2/sot3 defender edge.
    _sp_sot_proxy = (df["sp_goals_pg"] * 5.0 + df["headed_goals_pg"] * 4.0).clip(0, 0.6)
    df["defender_sot_edge"] = (
        df["sp_receiver_score"]                              # aerial ability × not taker
        * _h_adj                                             # height advantage vs opp CBs
        * (df["team_corners_pg"] / 4.0).clip(0.2, 3.0)      # corner volume (more corners = more chances)
        * (df["opp_sp_goals_conceded_pg"] / 0.12)           # opp SP vulnerability
        * (1.0 + _sp_sot_proxy * 4.0)                       # amplify by historical SP productivity
    ).clip(0, 5.0)

    # defender_sot_role_index: how central this player is to team's SP aerial attack
    # Primary driver of sot2/sot3 for defenders — they need to be THE aerial target
    df["defender_sot_role_index"] = (
        df["pos_defender"]
        * df["aerial_won_rate"]
        * df["sp_receiver_score"]
        * (df["team_corners_pg"] / 5.0).clip(0.2, 2.0)
    ).clip(0, 2.0)

    # sp_threat_vs_weak_sp_defense: historical SP rate vs weak SP defence
    df["sp_threat_vs_weak_sp_defense"] = (
        df["sp_goals_pg"]
        * (df["opp_sp_goals_conceded_pg"] / 0.12)
        * (1.0 - df["sp_taker_score"])
    ).clip(0, 1.0)

    # aerial_height_sp_composite: aerial × physical height edge × opp SP weakness
    df["aerial_height_sp_composite"] = (
        df["aerial_won_rate"]
        * _h_adj.clip(0.3, 1.0)
        * (df["opp_sp_goals_conceded_pg"] / 0.12)
        * (df["pos_defender"] + 0.5 * df["pos_midfielder"])
    ).clip(0, 2.0)

    # sp_goal_probability_composite: all signals combined
    df["sp_goal_probability_composite"] = (
        df["career_sp_goals_rate"]  * 0.30
        + df["sp_goals_pg"]         * 0.30
        + df["aerial_won_rate"] * _h_adj * 0.20
        + (df["opp_sp_goals_conceded_pg"] / 0.12 - 1.0).clip(-0.5, 1.0) * 0.20
    ).clip(0, 1.0)

    # sp_taker_assist_edge: probability of getting an assist via set piece
    df["sp_taker_assist_edge"] = (
        df["sp_taker_score"]
        * (df["career_sp_assist_rate"].clip(0, 0.10) / 0.10)
        * (df["team_corners_pg"] / 6.0).clip(0.3, 2.0)
    ).clip(0, 1.0)

    # Creative playmaker vs foul-prone defense
    df["creative_pressure_matchup"] = (
        df["creative_playmaker_score"] * df["opp_def_fouls_pg"]
    ).clip(lower=0.0)

    # Dribbler vs physically dominant defensive line
    df["dribbler_vs_defensive_line"] = (
        df["dribble_creativity_score"] * (1.0 - df["opp_def_aerial_win_rate"])
    ).clip(lower=0.0)

    # Progressive carrier vs pressing/aggressive defense
    df["carrier_vs_press"] = (
        df["progressive_carrier_score"] * df["opp_def_aggression"]
    ).clip(lower=0.0)

    # ── Card-market matchup features (Phase 5) ────────────────────────────────
    # Physical confrontations from failed dribbles — core card risk signal
    df["dribble_contact_rate"] = (
        df["dribbles_pg"] * (1.0 - df["dribble_success_rate"])
    ).clip(lower=0.0)

    # Dribbler vs aggressive midfield — failed dribbles into a foul-prone midfield
    df["tackle_dribble_clash"] = (
        df["dribble_contact_rate"] * df["opp_mid_fouls_pg"]
    ).clip(lower=0.0)

    # Full card pressure index — own discipline × opponent midfield aggression × referee
    df["card_clash_index"] = (
        (df["fouls_per90"] + df["dribble_contact_rate"])
        * df["opp_mid_cards_pg"]
        * df["referee_strictness"].clip(lower=0.1)
    ).clip(lower=0.0)

    # Opponent midfield discipline — direct opponent card signal for the match
    df["opp_mid_discipline"] = (
        df["opp_mid_fouls_pg"] * 0.6 + df["opp_mid_cards_pg"] * 2.0
    ).clip(lower=0.0)

    # ── Target variables — actual outcome in THIS match ───────────────────────
    df["target_goals"]   = (df["goals"]             >= 1).astype(int)
    df["target_sot"]     = (df["shots_on_target"]   >= 1).astype(int)
    df["target_cards"]   = (df["yellow_cards"]       >= 1).astype(int)
    df["target_assists"] = (df["assists"]             >= 1).astype(int)
    df["target_goals2"]  = (df["goals"]             >= 2).astype(int)
    df["target_sot2"]    = (df["shots_on_target"]   >= 2).astype(int)
    df["target_sot3"]    = (df["shots_on_target"]   >= 3).astype(int)

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

        # Find player's match history — full career for priors, last-n for rolling
        if pid:
            phist_all = history[history["player_id"] == pid].sort_values("date")
        else:
            phist_all = history[history["player_name"].str.lower() == name.lower()].sort_values("date")

        phist = phist_all.tail(n)

        if phist.empty:
            continue

        n_games    = len(phist)
        # Raw rolling means — clipped to training-observed 95th-percentile maxima to prevent
        # out-of-distribution extrapolation when tournament players have inflated recent stats.
        goals_pg   = min(float(phist["goals"].mean()),   0.60)
        assists_pg = min(float(phist["assists"].mean()),  0.50)
        shots_pg   = min(float(phist["shots_total"].mean()), 5.0)
        sot_pg     = min(float(phist["shots_on_target"].mean()), 3.0)
        cards_pg   = min(float(phist["yellow_cards"].mean()), 0.60)
        minutes_pg = min(float(phist["minutes"].mean()), 90.0)
        kp_pg      = min(float(phist["key_passes"].mean()) if "key_passes" in phist.columns else 0.0, 4.0)
        starter_rate = float(phist["started"].mean()) if "started" in phist.columns else 0.8
        sot_rate   = sot_pg / shots_pg if shots_pg > 0 else 0.0

        # Venue-split rolling rates
        if "is_home" in phist.columns:
            _h = phist[phist["is_home"].astype(bool)]
            _a = phist[~phist["is_home"].astype(bool)]
            goals_home_pg = float(_h["goals"].mean())                              if len(_h) else goals_pg * 1.05
            goals_away_pg = float(_a["goals"].mean())                              if len(_a) else goals_pg * 0.95
            shots_home_pg = float(_h["shots_total"].mean())      if len(_h) and "shots_total"      in _h.columns else shots_pg * 1.05
            shots_away_pg = float(_a["shots_total"].mean())      if len(_a) and "shots_total"      in _a.columns else shots_pg * 0.95
            sot_home_pg   = float(_h["shots_on_target"].mean())  if len(_h) and "shots_on_target"  in _h.columns else sot_pg   * 1.05
            sot_away_pg   = float(_a["shots_on_target"].mean())  if len(_a) and "shots_on_target"  in _a.columns else sot_pg   * 0.95
        else:
            goals_home_pg = goals_pg * 1.05;  goals_away_pg = goals_pg * 0.95
            shots_home_pg = shots_pg * 1.05;  shots_away_pg = shots_pg * 0.95
            sot_home_pg   = sot_pg   * 1.05;  sot_away_pg   = sot_pg   * 0.95

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

        # ── New extended rolling stats ────────────────────────────────────────
        tackles_pg = float(phist["tackles_total"].mean()) \
            if "tackles_total" in phist.columns else 0.0
        tackles_per90 = float(
            (phist["tackles_total"] / mins_s * 90).fillna(0.0).mean()
        ) if "tackles_total" in phist.columns else 0.0
        interceptions_pg = float(phist["interceptions"].mean()) \
            if "interceptions" in phist.columns else 0.0
        defensive_actions_per90 = float(
            ((phist.get("tackles_total", pd.Series(0, index=phist.index)) +
              phist.get("interceptions", pd.Series(0, index=phist.index))) / mins_s * 90
             ).fillna(0.0).mean()
        ) if "tackles_total" in phist.columns else 0.0

        if "dribbles_success" in phist.columns and "dribbles_attempted" in phist.columns:
            dribbles_pg = float(phist["dribbles_success"].mean())
            dribble_success_rate = float(
                (phist["dribbles_success"] /
                 phist["dribbles_attempted"].replace(0, np.nan)).fillna(0.0).mean()
            )
        else:
            dribbles_pg = 0.0;  dribble_success_rate = 0.0

        dribbled_past_pg  = float(phist["dribbles_past"].mean())  if "dribbles_past"  in phist.columns else 0.0
        red_cards_pg      = float(phist["red_cards"].mean())       if "red_cards"      in phist.columns else 0.0
        passes_pg         = float(phist["passes_total"].mean())    if "passes_total"   in phist.columns else 0.0
        offsides_pg       = float(phist["offsides"].mean())        if "offsides"       in phist.columns else 0.0
        penalties_won_pg  = float(phist["penalty_won"].mean())     if "penalty_won"    in phist.columns else 0.0

        if "penalty_scored" in phist.columns and "penalty_missed" in phist.columns:
            _pen_tot = (phist["penalty_scored"] + phist["penalty_missed"]).replace(0, np.nan)
            penalty_conversion_rate = float((phist["penalty_scored"] / _pen_tot).fillna(0.75).mean())
        else:
            penalty_conversion_rate = 0.75

        saves_pg              = float(phist["saves"].mean())         if "saves"          in phist.columns else 0.0
        goals_conceded_gk_pg  = float(phist["goals_conceded"].mean()) if "goals_conceded" in phist.columns else 0.0
        if "saves" in phist.columns and "goals_conceded" in phist.columns:
            _gk_sr = phist["saves"] / (phist["saves"] + phist["goals_conceded"] + 0.001)
            gk_save_rate = float(_gk_sr.mean())
        else:
            gk_save_rate = 0.72

        # Pre-compute intermediate composites needed by downstream features
        _progressive_carrier_score = min(
            min(dribbles_pg / 3.0, 1.0) * 0.45
            + min(passes_pg / 60.0, 1.0) * 0.35
            + dribble_success_rate * 0.20,
            1.0,
        ) * (pos_midfielder * 1.0 + pos_forward * 0.8 + pos_defender * 0.5)

        _penalty_threat_score = min(penalties_won_pg * 3.0 + fouls_drawn_per90 * 0.08, 1.0)

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

        # Position-split opp concede stats — look up from history aggregates
        opp_goals_conceded_vs_fwd_pg = 1.0
        opp_sot_conceded_vs_fwd_pg   = 3.5
        opp_goals_conceded_vs_mid_pg = 0.4
        opp_sot_conceded_vs_mid_pg   = 2.0
        if opp_team and not history.empty and "position" in history.columns:
            _opp_hist = history[
                history["team"].str.lower().str.contains(opp_team.lower()[:6], na=False)
            ].sort_values("date").tail(n * 5)
            if not _opp_hist.empty:
                _opp_pos = _opp_hist["position"].str.upper().fillna("")
                _fwd = _opp_hist[_opp_pos.str.startswith("F")]
                _mid = _opp_hist[_opp_pos.str.startswith("M")]
                if len(_fwd) >= 3:
                    opp_goals_conceded_vs_fwd_pg = float(_fwd["goals"].mean())
                    opp_sot_conceded_vs_fwd_pg   = float(_fwd["shots_on_target"].mean())
                if len(_mid) >= 3:
                    opp_goals_conceded_vs_mid_pg = float(_mid["goals"].mean())
                    opp_sot_conceded_vs_mid_pg   = float(_mid["shots_on_target"].mean())

        # ── Opponent GK lookup ────────────────────────────────────────────────
        opp_gk_save_rate = ctx.get("opp_gk_save_rate", 0.72)
        opp_gk_saves_pg  = ctx.get("opp_gk_saves_pg",  3.0)

        if opp_team and not history.empty and "position" in history.columns \
                and "saves" in history.columns:
            _opp_gks = history[
                history["team"].str.lower().str.contains(opp_team.lower()[:6], na=False) &
                history["position"].str.upper().str.startswith("G", na=False)
            ].sort_values("date").tail(n * 3)

            if not _opp_gks.empty:
                if "goals_conceded" in _opp_gks.columns:
                    _gksr = _opp_gks["saves"] / (
                        _opp_gks["saves"] + _opp_gks["goals_conceded"] + 0.001)
                    opp_gk_save_rate = float(_gksr.mean())
                opp_gk_saves_pg = float(_opp_gks["saves"].mean())

        # ── Opponent midfielder discipline lookup (card market) ──────────────
        opp_mid_fouls_pg = ctx.get("opp_mid_fouls_pg", 2.0)
        opp_mid_cards_pg = ctx.get("opp_mid_cards_pg", 0.4)

        if opp_team and not history.empty and "position" in history.columns:
            _opp_mids = history[
                history["team"].str.lower().str.contains(opp_team.lower()[:6], na=False) &
                history["position"].str.upper().str.startswith("M", na=False)
            ].sort_values("date").tail(n * 5)
            if not _opp_mids.empty:
                if "fouls_committed" in _opp_mids.columns:
                    opp_mid_fouls_pg = float(_opp_mids["fouls_committed"].mean())
                if "yellow_cards" in _opp_mids.columns:
                    opp_mid_cards_pg = float(_opp_mids["yellow_cards"].mean())

        # Career-to-date averages (full history, not capped at last-n)
        career_goals_pg   = float(phist_all["goals"].mean())             if len(phist_all) else 0.0
        career_sot_pg     = float(phist_all["shots_on_target"].mean())   if len(phist_all) else 0.0
        career_shots_pg   = float(phist_all["shots_total"].mean())       if len(phist_all) else 0.0
        career_assists_pg = float(phist_all["assists"].mean())           if len(phist_all) else 0.0

        # Rating (API-Football match rating — 0 means missing, use 7.0 fallback)
        rating_pg = float(phist["rating"].replace(0, np.nan).mean()) \
            if "rating" in phist.columns else 7.0
        if np.isnan(rating_pg):
            rating_pg = 7.0

        # ── Season stats (from enriched parquet — most recent non-null row) ────
        def _latest(col, default=0.0):
            if col in phist_all.columns:
                vals = phist_all[col].dropna()
                return float(vals.iloc[-1]) if len(vals) else default
            return default

        season_goals_pg    = _latest("season_goals_pg",   goals_pg)
        season_assists_pg  = _latest("season_assists_pg",  assists_pg)
        season_shots_pg    = _latest("season_shots_pg",    shots_pg)
        season_sot_pg      = _latest("season_sot_pg",      sot_pg)
        season_cards_pg    = _latest("season_cards_pg",    cards_pg)
        season_minutes_pg  = _latest("season_minutes_pg",  minutes_pg)
        season_appearances = _latest("season_appearances", float(len(phist_all)))
        season_start_rate  = _latest("season_start_rate",  starter_rate)
        season_pass_accuracy = _latest("season_pass_accuracy", 0.75)
        season_dribble_pg  = _latest("season_dribble_pg",  dribbles_pg)
        season_fouls_pg    = _latest("season_fouls_pg",    fouls_per90)
        season_fouls_drawn_pg = _latest("season_fouls_drawn_pg", fouls_drawn_per90)

        # ── Profile data (age, height) ─────────────────────────────────────────
        age       = _latest("age",       25.0)
        height_cm = _latest("height_cm", 180.0)
        age_peak_delta         = abs(age - 27.0)
        height_aerial_interaction = height_cm * aerial_won_rate

        # ── Injury features ────────────────────────────────────────────────────
        chronic_injury_risk     = _latest("chronic_injury_risk",     0.0)
        days_since_last_injury  = _latest("days_since_last_injury",  365.0)
        return_from_injury_flag = _latest("return_from_injury_flag", 0.0)

        # ── Referee features ───────────────────────────────────────────────────
        _ref_prof = referee_profile or {}
        referee_yellows_pg  = float(_ref_prof.get("yellows_per_game", 4.0))
        referee_strictness  = float(_ref_prof.get("strictness_score",  1.0))

        # ── Missing composites (computed from vars already in scope) ───────────
        carrier_vs_press          = max(_progressive_carrier_score * opp_def_aggression, 0.0)
        box_threat_vs_leaky_defense = max(box_actions_per90 * ctx.get("opp_goals_conceded_pg", 1.3), 0.0)
        efficiency_vs_leaky_keeper  = max(shooting_efficiency_index * (1.0 - opp_gk_save_rate), 0.0)
        kp_vs_aggressive_defense    = max(kp_per90 * opp_def_fouls_pg, 0.0)
        team_momentum_forward_matchup = max(ctx.get("team_goals_pg_roll", 1.3) * pos_forward, 0.0)
        set_piece_corner_matchup    = max(set_piece_threat_score * team_corners_pg, 0.0)
        creative_pressure_matchup   = max(creative_playmaker_score * opp_def_aggression, 0.0)
        _dribble_creativity         = dribbles_pg * dribble_success_rate
        dribbler_vs_defensive_line  = max(_dribble_creativity * opp_def_fouls_pg, 0.0)

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
            "rating_pg":           round(rating_pg, 3),
            # Career-to-date priors (stable baseline; model learns to weight vs rolling)
            "career_goals_pg":     round(career_goals_pg,   4),
            "career_sot_pg":       round(career_sot_pg,     4),
            "career_shots_pg":     round(career_shots_pg,   4),
            "career_assists_pg":   round(career_assists_pg, 4),
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
            # Venue-split rates
            "goals_home_pg":  round(goals_home_pg, 4),
            "goals_away_pg":  round(goals_away_pg, 4),
            "shots_home_pg":  round(shots_home_pg, 4),
            "shots_away_pg":  round(shots_away_pg, 4),
            "sot_home_pg":    round(sot_home_pg,   4),
            "sot_away_pg":    round(sot_away_pg,   4),
            # Extended raw rolling stats
            "tackles_pg":               round(tackles_pg,              4),
            "interceptions_pg":         round(interceptions_pg,        4),
            "defensive_actions_per90":  round(defensive_actions_per90, 4),
            "dribbles_pg":              round(dribbles_pg,             4),
            "dribble_success_rate":     round(dribble_success_rate,    4),
            "dribbled_past_pg":         round(dribbled_past_pg,        4),
            "red_cards_pg":             round(red_cards_pg,            4),
            "passes_pg":                round(passes_pg,               4),
            "offsides_pg":              round(offsides_pg,             4),
            "penalties_won_pg":         round(penalties_won_pg,        4),
            "penalty_conversion_rate":  round(penalty_conversion_rate, 4),
            "saves_pg":                 round(saves_pg,                4),
            "goals_conceded_gk_pg":     round(goals_conceded_gk_pg,    4),
            "gk_save_rate":             round(gk_save_rate,            4),
            # Opponent GK matchup
            "opp_gk_save_rate":         round(opp_gk_save_rate,        4),
            "opp_gk_saves_pg":          round(opp_gk_saves_pg,         4),
            # New composite features
            "att_vs_gk_threat":         round(max(sot_pg * (1.0 - opp_gk_save_rate), 0.0), 4),
            "clinical_vs_gk":           round(max(shot_accuracy_rate * (1.0 - opp_gk_save_rate), 0.0), 4),
            "dribble_creativity_score": round(dribbles_pg * dribble_success_rate, 4),
            "defensive_solidity":       round(max(defensive_actions_per90 * (1.0 - dribbled_past_pg / (dribbled_past_pg + 1.0)), 0.0), 4),
            "penalty_threat_score":     round(min(penalties_won_pg * 3.0 + fouls_drawn_per90 * 0.08, 1.0), 4),
            "card_risk_index":          round(min(cards_pg * 2.0 + red_cards_pg * 3.0 + foul_committer_ratio * 0.5, 1.0), 4),
            "midfield_engine_score":    round((min(passes_pg / 50.0, 1.0) * 0.4 + min(kp_per90 / 3.0, 1.0) * 0.4 + min(tackles_per90 / 5.0, 1.0) * 0.2) * (pos_midfielder + 0.3 * pos_defender), 4),
            "offside_aggressiveness":   round(min(offsides_pg * pos_forward, 2.0), 4),
            # Agent-designed composites
            "volume_shot_penetration":  round(max(shots_pg * shot_accuracy_rate * (1.0 - opp_gk_save_rate), 0.0), 4),
            "finishing_threat_index":   round(max(sot_pg * shooting_efficiency_index * (1.0 - opp_gk_save_rate), 0.0), 4),
            "box_dominance_vs_gk":      round(max((box_actions_per90 + dribbles_pg) * (1.0 - opp_gk_save_rate), 0.0), 4),
            "aggression_adjusted_threat":    round(max(sot_pg * (1.0 - opp_gk_save_rate) / (1.0 + offsides_pg * 0.3), 0.0), 4),
            "defensive_vulnerability_index": round(min(dribbled_past_pg * 1.5 + (1.0 - dribble_success_rate) * 0.5 + (1.0 - aerial_won_rate) * 0.5, 2.0) * (pos_forward * 1.0 + pos_midfielder * 0.7 + pos_defender * 0.3), 4),
            "progressive_carrier_score":     round(_progressive_carrier_score, 4),
            "disciplinary_pressure_index":   round(min(fouls_drawn_per90 * 0.20 + fouls_per90 * 0.25 + cards_pg * 1.5 + duel_intensity_per90 * 0.10, 2.0), 4),
            "foul_magnet_score":             round(min(fouls_drawn_per90 * 0.30 + _penalty_threat_score * 0.40 + _progressive_carrier_score * 0.30, 1.0), 4),
            # Position-split opponent concede stats
            "opp_goals_conceded_vs_fwd_pg":  round(opp_goals_conceded_vs_fwd_pg, 4),
            "opp_sot_conceded_vs_fwd_pg":    round(opp_sot_conceded_vs_fwd_pg,   4),
            "opp_goals_conceded_vs_mid_pg":  round(opp_goals_conceded_vs_mid_pg, 4),
            "opp_sot_conceded_vs_mid_pg":    round(opp_sot_conceded_vs_mid_pg,   4),
            # Position-specific matchup composites
            "forward_matchup_score": round(max(sot_pg * opp_goals_conceded_vs_fwd_pg * pos_forward, 0.0), 4),
            "mid_threat_vs_defense": round(max(kp_per90 * opp_goals_conceded_vs_mid_pg * pos_midfielder, 0.0), 4),
            # Opponent midfielder discipline (card market features)
            "opp_mid_fouls_pg":      round(opp_mid_fouls_pg, 4),
            "opp_mid_cards_pg":      round(opp_mid_cards_pg, 4),
            # Card-market composites
            "dribble_contact_rate":  round(max(dribbles_pg * (1.0 - dribble_success_rate), 0.0), 4),
            "tackle_dribble_clash":  round(max(dribbles_pg * (1.0 - dribble_success_rate) * opp_mid_fouls_pg, 0.0), 4),
            "card_clash_index":      round(max((fouls_per90 + dribbles_pg * (1.0 - dribble_success_rate)) * opp_mid_cards_pg * max(ctx.get("referee_strictness", 1.0), 0.1), 0.0), 4),
            "opp_mid_discipline":    round(max(opp_mid_fouls_pg * 0.6 + opp_mid_cards_pg * 2.0, 0.0), 4),
            # Season stats (enriched — full-season aggregates)
            "season_goals_pg":       round(season_goals_pg,       4),
            "season_assists_pg":     round(season_assists_pg,     4),
            "season_shots_pg":       round(season_shots_pg,       4),
            "season_sot_pg":         round(season_sot_pg,         4),
            "season_cards_pg":       round(season_cards_pg,       4),
            "season_minutes_pg":     round(season_minutes_pg,     1),
            "season_appearances":    round(season_appearances,    0),
            "season_start_rate":     round(season_start_rate,     4),
            "season_pass_accuracy":  round(season_pass_accuracy,  4),
            "season_dribble_pg":     round(season_dribble_pg,     4),
            "season_fouls_pg":       round(season_fouls_pg,       4),
            "season_fouls_drawn_pg": round(season_fouls_drawn_pg, 4),
            # Profile features
            "age":                   round(age,       1),
            "height_cm":             round(height_cm, 1),
            "age_peak_delta":        round(age_peak_delta,             2),
            "height_aerial_interaction": round(height_aerial_interaction, 4),
            # Injury features
            "chronic_injury_risk":     round(chronic_injury_risk,     4),
            "days_since_last_injury":  round(days_since_last_injury,  1),
            "return_from_injury_flag": round(return_from_injury_flag, 0),
            # Referee features
            "referee_yellows_pg":    round(referee_yellows_pg,  4),
            "referee_strictness":    round(referee_strictness,  4),
            # Missing composites
            "carrier_vs_press":               round(carrier_vs_press,               4),
            "box_threat_vs_leaky_defense":    round(box_threat_vs_leaky_defense,    4),
            "efficiency_vs_leaky_keeper":     round(efficiency_vs_leaky_keeper,     4),
            "kp_vs_aggressive_defense":       round(kp_vs_aggressive_defense,       4),
            "team_momentum_forward_matchup":  round(team_momentum_forward_matchup,  4),
            "set_piece_corner_matchup":       round(set_piece_corner_matchup,       4),
            "creative_pressure_matchup":      round(creative_pressure_matchup,      4),
            "dribbler_vs_defensive_line":     round(dribbler_vs_defensive_line,     4),
            # ── Set piece event features ──────────────────────────────────────────
            "sp_goals_pg":               round(float(phist["sp_goals_pg"].iloc[-1])               if "sp_goals_pg"               in phist.columns and len(phist) > 0 else 0.0, 4),
            "headed_goals_pg":           round(float(phist["headed_goals_pg"].iloc[-1])           if "headed_goals_pg"           in phist.columns and len(phist) > 0 else 0.0, 4),
            "fk_goals_pg":               round(float(phist["fk_goals_pg"].iloc[-1])               if "fk_goals_pg"               in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_assist_pg":              round(float(phist["sp_assist_pg"].iloc[-1])              if "sp_assist_pg"              in phist.columns and len(phist) > 0 else 0.0, 4),
            "career_sp_goals_rate":      round(float(phist["career_sp_goals_rate"].iloc[-1])      if "career_sp_goals_rate"      in phist.columns and len(phist) > 0 else 0.0, 4),
            "career_sp_assist_rate":     round(float(phist["career_sp_assist_rate"].iloc[-1])     if "career_sp_assist_rate"     in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_goals_share":            round(float(phist["sp_goals_share"].iloc[-1])            if "sp_goals_share"            in phist.columns and len(phist) > 0 else 0.0, 4),
            "headed_goals_share":        round(float(phist["headed_goals_share"].iloc[-1])        if "headed_goals_share"        in phist.columns and len(phist) > 0 else 0.0, 4),
            # ── SP role ───────────────────────────────────────────────────────────
            "sp_taker_score":            round(float(phist["sp_taker_score"].iloc[-1])            if "sp_taker_score"            in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_receiver_score":         round(float(phist["sp_receiver_score"].iloc[-1])         if "sp_receiver_score"         in phist.columns and len(phist) > 0 else 0.0, 4),
            # ── Height matchup ────────────────────────────────────────────────────
            "opp_def_mean_height":       round(float(phist["opp_def_mean_height"].iloc[-1])       if "opp_def_mean_height"       in phist.columns and len(phist) > 0 else 181.0, 1),
            "height_diff_vs_opp_def":    round(float(phist["height_diff_vs_opp_def"].iloc[-1])    if "height_diff_vs_opp_def"    in phist.columns and len(phist) > 0 else 0.0, 1),
            "height_advantage_score":    round(float(phist["height_advantage_score"].iloc[-1])    if "height_advantage_score"    in phist.columns and len(phist) > 0 else 0.0, 2),
            # ── Opponent SP vulnerability ─────────────────────────────────────────
            "opp_sp_goals_conceded_pg":  round(float(phist["opp_sp_goals_conceded_pg"].iloc[-1])  if "opp_sp_goals_conceded_pg"  in phist.columns and len(phist) > 0 else 0.12, 4),
            # ── SP composites ─────────────────────────────────────────────────────
            "defender_sp_edge":              round(float(phist["defender_sp_edge"].iloc[-1])              if "defender_sp_edge"              in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_threat_vs_weak_sp_defense":  round(float(phist["sp_threat_vs_weak_sp_defense"].iloc[-1])  if "sp_threat_vs_weak_sp_defense"  in phist.columns and len(phist) > 0 else 0.0, 4),
            "aerial_height_sp_composite":    round(float(phist["aerial_height_sp_composite"].iloc[-1])    if "aerial_height_sp_composite"    in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_goal_probability_composite": round(float(phist["sp_goal_probability_composite"].iloc[-1]) if "sp_goal_probability_composite" in phist.columns and len(phist) > 0 else 0.0, 4),
            "sp_taker_assist_edge":          round(float(phist["sp_taker_assist_edge"].iloc[-1])          if "sp_taker_assist_edge"          in phist.columns and len(phist) > 0 else 0.0, 4),
            # ── Opposition quality adjustment ─────────────────────────────────────
            "opp_strength_index":        round(float(phist["opp_strength_index"].iloc[-1])        if "opp_strength_index"        in phist.columns and len(phist) > 0 else 1.0, 4),
            "quality_adj_goals_pg":      round(float(phist["quality_adj_goals_pg"].iloc[-1])      if "quality_adj_goals_pg"      in phist.columns and len(phist) > 0 else 0.0, 4),
            "quality_adj_sot_pg":        round(float(phist["quality_adj_sot_pg"].iloc[-1])        if "quality_adj_sot_pg"        in phist.columns and len(phist) > 0 else 0.0, 4),
            # ── League quality and player vs player ───────────────────────────────
            "league_tier":               float(phist["league_tier"].iloc[-1])               if "league_tier"               in phist.columns and len(phist) > 0 else 3,
            "league_quality":            round(float(phist["league_quality"].iloc[-1])            if "league_quality"            in phist.columns and len(phist) > 0 else 0.5,  4),
            "player_career_avg_quality": round(float(phist["player_career_avg_quality"].iloc[-1]) if "player_career_avg_quality" in phist.columns and len(phist) > 0 else 0.65, 4),
            "opp_def_player_rating_pg":  round(float(phist["opp_def_player_rating_pg"].iloc[-1])  if "opp_def_player_rating_pg"  in phist.columns and len(phist) > 0 else 6.8,  2),
            "opp_top_def_rating":        round(float(phist["opp_top_def_rating"].iloc[-1])        if "opp_top_def_rating"        in phist.columns and len(phist) > 0 else 7.0,  2),
            "opp_def_player_quality":    round(float(phist["opp_def_player_quality"].iloc[-1])    if "opp_def_player_quality"    in phist.columns and len(phist) > 0 else 1.0,  4),
            "context_quality_discount":  round(float(phist["context_quality_discount"].iloc[-1])  if "context_quality_discount"  in phist.columns and len(phist) > 0 else 1.0,  4),
            "quality_mismatch_goals":    round(float(phist["quality_mismatch_goals"].iloc[-1])    if "quality_mismatch_goals"    in phist.columns and len(phist) > 0 else 0.0,  4),
            "quality_mismatch_sot":      round(float(phist["quality_mismatch_sot"].iloc[-1])      if "quality_mismatch_sot"      in phist.columns and len(phist) > 0 else 0.0,  4),
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
