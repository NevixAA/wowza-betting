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

        # Find player's match history — full career for priors, last-n for rolling
        if pid:
            phist_all = history[history["player_id"] == pid].sort_values("date")
        else:
            phist_all = history[history["player_name"].str.lower() == name.lower()].sort_values("date")

        phist = phist_all.tail(n)

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
