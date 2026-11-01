"""
Fantasy defaults register (R2-D, finding 22, 2026-11-01).

Every hard-coded default / prior / fallback that can move a Fantasy number, classified as

  OBSERVED              computed from observed data, or a real FPL game rule
  TRAIN_FITTED_PRIOR    fitted on historical data (says whether the fit is reproducible)
  DOMAIN_PRIOR          football / FPL knowledge constant, not fitted
  MISSING_DATA_FALLBACK fires only when an input is missing

This module is DESCRIPTIVE: it changes no projection. Fantasy V2 (FPL-anchored mean + shadow
residual) behaves exactly as before. Suspected correctness bugs are listed in SUSPECTED_BUGS and
reported to the owner; none is silently fixed here. `summary()` adds runtime fire counts read
from the committed outputs, and is exposed in output/fantasy_health.json ("defaults_audit") and
output/player_props_integrity_health.json ("fantasy_defaults").

Reach: V2 = fantasy_v2.run chain; V1 = fantasy.generate_fantasy_tips (workflow step 1);
V1->V2 = through the projection log (wowza_uncond / wowza_gap, shadow residual + benchmark arm);
DASH = dashboard only; DEAD = no caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGISTER_VERSION = "FANTASY-DEFAULTS-1.0"
OUT = Path(__file__).resolve().parents[1] / "output"

# id, where, value, class, reach, importance, detectable at runtime, note
DEFAULTS: list[dict] = [dict(zip(
    ("id", "where", "value", "class", "reach", "importance", "detectable", "note"), r)) for r in [
    # ---- requested by the owner brief
    ("minutes_default_60", "fantasy.py:229", "minutes_pg NaN -> 60", "MISSING_DATA_FALLBACK", "V1",
     "LOW", False, "effectively dead: rows already filtered to minutes_pg >= 45"),
    ("p_start_default_0_6", "fantasy.py:354", "unknown start rate -> 0.60", "MISSING_DATA_FALLBACK",
     "V1->V2", "MED", False, "lands exactly on the 0.60 captain floor / rotation boundary, unflagged"),
    ("p_start_blend_0_6_0_4", "fantasy.py:354", "0.6 recent + 0.4 season start rate", "DOMAIN_PRIOR",
     "V1->V2", "MED", False, ""),
    ("position_default_MID", "fantasy.py:302-303", "unknown position -> MID", "MISSING_DATA_FALLBACK",
     "V1", "LOW", True, "fires only when FPL matching fails (fpl_matched=False)"),
    ("goal_points_fallback_5", "fantasy.py:306", "goal_pts NaN -> 5", "MISSING_DATA_FALLBACK", "V1",
     "LOW", False, "dead after the position alias"),
    ("goal_points_gkp_6", "fantasy.py:33,35; fantasy_panel.py:244; fantasy_distribution.py:37",
     "GKP goal = 6", "OBSERVED", "V1->V2, V2 feature", "LOW", False,
     "WRONG RULE: live FPL game_config.scoring.goals_scored.GKP = 10"),
    ("referee_cards_3_5", "feature_engineering.py:312-319,1423", "missing referee -> 3.5 yellows/match",
     "MISSING_DATA_FALLBACK", "V1 (props features, cards market)", "LOW", False, ""),
    ("fdr_heuristic_xg", "fantasy.py:323", "opp xG = 0.5 + 0.42*avg FDR (next 5)", "DOMAIN_PRIOR",
     "V1->V2", "MED", False, "unfitted magic numbers; averaged over 5 fixtures, not the next GW"),
    ("cs_lambda_blend", "fantasy.py:333", "0.6*conceded + 0.4*fdr_xg", "DOMAIN_PRIOR", "V1->V2", "MED",
     False, ""),
    # ---- V1 projection
    ("dc_thresholds", "fantasy.py:41", "DEF 5 / MID 4.5 / FWD 4 actions", "DOMAIN_PRIOR", "V1->V2", "HIGH",
     False, "SUSPECTED BUG: FPL rule is DEF>=10 CBIT, MID/FWD>=12 CBIRT; inflates dc_pts"),
    ("bonus_scale_0_66", "fantasy.py:172", "x0.66", "TRAIN_FITTED_PRIOR", "V1->V2", "MED", False,
     "fitted in scripts/fantasy_backtest.py with cs=0 and narrower defensive actions than production"),
    ("sot2_weight_0_45", "fantasy.py:309", "x0.45", "TRAIN_FITTED_PRIOR", "V1->V2", "MED", False,
     "lstsq in scripts/fantasy_backtest.py:108-117, hard-coded"),
    ("prop_model_missing_zero", "fantasy.py:227", "missing props model -> p = 0", "MISSING_DATA_FALLBACK",
     "V1->V2", "HIGH", False, "all attacking points vanish silently"),
    ("props_batch_median", "player_model/model.py _prep", "NaN -> median of the scoring batch",
     "MISSING_DATA_FALLBACK", "V1->V2", "MED", False,
     "R2-D finding 20: legacy serving payloads only; repaired challengers use a fit-block imputer"),
    ("fdr_default_3", "fantasy.py:413,415", "missing FDR -> 3", "MISSING_DATA_FALLBACK", "V1", "LOW", False, ""),
    ("fixture_mult_clip_v1", "fantasy.py:418", "0.60-1.40", "DOMAIN_PRIOR", "V1->V2", "MED", False, ""),
    ("status_default_available", "fpl_api.py:218,229", "unknown FPL status -> available",
     "MISSING_DATA_FALLBACK", "V1", "MED", False, "unknown data becomes MORE confident"),
    ("season_hardcoded", "fantasy.py:693,712", "seasons=('2026','2025')", "MISSING_DATA_FALLBACK", "V1",
     "MED", False, "hard-coded season; silently serves last season's squads from mid-2027"),
    # ---- V2 (production product)
    ("v2_p_start_from_mix", "fantasy_residual.py:708", "p_start = P60 + 0.35*P(1-59)", "DOMAIN_PRIOR",
     "V2", "HIGH", False, "unfitted; drives Confidence/Risk labels and captain pools"),
    ("v2_attack_defence_share", "fantasy_v2.py:67-68", "FWD .55/0, MID .45/.08, DEF .20/.40, GKP .05/.45",
     "DOMAIN_PRIOR", "V2", "MED", False, ""),
    ("v2_rate_fallback_2pts", "fantasy_v2.py:320", "no ppg -> 2.0 pts/fixture", "MISSING_DATA_FALLBACK",
     "V2", "MED", False, "phantom 3/5-GW points for non-playing players"),
    ("v2_long_play_rate", "fantasy_v2.py:315", "start_rate + 0.25 (0.2 if none)", "DOMAIN_PRIOR", "V2",
     "MED", False, "NaN start_rate propagates NaN into xpts_5gw"),
    ("v2_team_missing_league_avg", "fantasy_panel.py:143-144", "unknown team -> league-mean lambdas",
     "MISSING_DATA_FALLBACK", "V2", "MED", False, "unknown fixture looks average, not unknown"),
    ("v2_home_adv", "fantasy_panel.py:145", "1.10 / 0.91", "DOMAIN_PRIOR", "V2", "MED", False, ""),
    ("v2_pos_xg_xa_prior", "fantasy_panel.py:216-217", "per-position xG/xA per 90 priors", "DOMAIN_PRIOR",
     "V2", "MED", False, "not fitted in code"),
    ("v2_chance_fill", "fantasy_panel.py:229", "NaN chance -> 100 if status 'a' else 0",
     "MISSING_DATA_FALLBACK", "V2", "HIGH", False, "follows FPL convention (harsh, not optimistic)"),
    ("v2_matrix_nan0", "fantasy_residual.py:84", "every residual feature NaN -> 0", "MISSING_DATA_FALLBACK",
     "V2 (shadow residual)", "MED", False, ""),
    ("v2_cond_minutes", "fantasy_residual.py:701", "missing bucket -> 25 / 88 min", "MISSING_DATA_FALLBACK",
     "V2", "MED", False, ""),
    ("v2_confidence_thresholds", "fantasy_v2.py:256-262", "p_start 0.85 / 0.60", "DOMAIN_PRIOR", "V2",
     "MED", True, ""),
    ("v2_captain_pools", "fantasy_v2.py:404-425", "p_start 0.85 / 0.75; 0.7/0.8 x best", "DOMAIN_PRIOR",
     "V2", "HIGH", False, ""),
    ("v2_minutes_model_C", "fantasy_residual.py:557", "C=0.5", "DOMAIN_PRIOR", "V2", "MED", False, ""),
    ("v2_nb_dispersion", "fantasy_residual.py:572-596", "likelihood-chosen size", "TRAIN_FITTED_PRIOR",
     "V2", "MED", False, "reproducible"),
    ("v2_ridge_alpha", "fantasy_residual.py:131", "leave-one-GW-out alpha", "TRAIN_FITTED_PRIOR", "V2",
     "LOW", True, "reproducible"),
    ("v2_fpl_scoring_rules", "fantasy_panel.py:243; fantasy_residual.py:60",
     "CS GKP4 DEF4 MID1 FWD0; assist 3; appearance 1/2", "OBSERVED", "V2", "MED", False,
     "match live game_config"),
]]

SUSPECTED_BUGS = [
    {"id": "dc_thresholds", "severity": "HIGH", "reach": "V1->V2 (shadow residual + benchmark arm)",
     "evidence": "fantasy.py:41 uses 5/4.5/4 actions incl. duels_won; FPL rule DEF>=10 CBIT, MID/FWD>=12 "
                 "CBIRT. fantasy_tips.csv mean dc_pts FWD 1.29 > DEF 1.28; fpl_bootstrap shows 0% of FWD "
                 "reach 12. Plausibly a large part of the documented +1.0 Wowza bias."},
    {"id": "goal_points_gkp_6", "severity": "LOW", "reach": "V1, V2 feature",
     "evidence": "output/fpl_bootstrap.json game_config.scoring.goals_scored = GKP 10, DEF 6, MID 5, FWD 4"},
    {"id": "bonus_cs_points_as_probability", "severity": "MED", "reach": "V1->V2",
     "evidence": "fantasy.py passes cs_pts (0-4) into _expected_bonus(cs=...) which expects a probability"},
    {"id": "status_n_not_out_v1", "severity": "MED (latent)", "reach": "V1",
     "evidence": "fpl_api.py:232 injured only for i/s/u; V2 (fantasy_panel.py:230) includes 'n'"},
    {"id": "v1_next_gw_is_5_fixture_average", "severity": "MED", "reach": "V1->V2",
     "evidence": "fantasy.py _per_game/_cs_row average 5 fixtures; settled against one GW"},
    {"id": "v2_p_start_unfitted", "severity": "MED", "reach": "V2 labels and captain pools",
     "evidence": "fantasy_residual.py:708 p_start = P60 + 0.35*P(1-59), constant not fitted"},
    {"id": "fantasy_health_pinned_aging", "severity": "MED", "reach": "health only",
     "evidence": "fantasy_health.py hard-codes official_squads UNKNOWN_AGE so status can never be OK"},
    {"id": "season_hardcoded", "severity": "MED from mid-2027", "reach": "V1 squads",
     "evidence": "fantasy.py:693,712 seasons=('2026','2025')"},
    {"id": "ranking_gate_without_ci", "severity": "LOW", "reach": "V2 acceptance gate",
     "evidence": "fantasy_residual.py docstring promises a CI; code requires spearman > FPL each GW"},
]


def _read(name: str) -> pd.DataFrame | None:
    p = OUT / name
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def fire_counts() -> dict:
    """How often the detectable fallbacks fired in the latest committed outputs."""
    c: dict = {}
    t = _read("fantasy_tips.csv")
    if t is not None and len(t):
        c["v1_rows"] = int(len(t))
        if "p_start" in t:
            c["p_start_exactly_0_6"] = int(np.isclose(pd.to_numeric(t["p_start"], errors="coerce"), 0.6).sum())
        if "fpl_matched" in t:
            c["fpl_unmatched_rows"] = int((~t["fpl_matched"].astype(str).str.lower().eq("true")).sum())
        if "fixtures_available" in t:
            c["fixtures_unavailable_rows"] = int((~t["fixtures_available"].astype(str).str.lower().eq("true")).sum())
        for col in ("p_goal", "p_assist", "p_sot2"):
            if col in t:
                c[f"{col}_all_zero"] = bool((pd.to_numeric(t[col], errors="coerce").fillna(0) == 0).all())
        if {"position", "dc_pts"} <= set(t.columns):
            c["mean_dc_pts_by_position"] = {k: round(float(v), 3) for k, v in
                                            t.groupby("position")["dc_pts"].mean().items()}
    v = _read("fantasy_v2_projections.csv")
    if v is not None and len(v):
        c["v2_rows"] = int(len(v))
        if "n_fix" in v:
            c["v2_blank_gw_rows"] = int((pd.to_numeric(v["n_fix"], errors="coerce").fillna(1) == 0).sum())
        if "start_rate" in v:
            c["v2_start_rate_missing"] = int(pd.to_numeric(v["start_rate"], errors="coerce").isna().sum())
        if "minutes_source" in v:
            c["v2_minutes_source"] = {str(k): int(n) for k, n in v["minutes_source"].value_counts().items()}
    return c


def summary() -> dict:
    by_class: dict = {}
    for d in DEFAULTS:
        by_class[d["class"]] = by_class.get(d["class"], 0) + 1
    return {
        "register_version": REGISTER_VERSION,
        "counts_by_class": by_class,
        "important": [d for d in DEFAULTS if d["importance"] == "HIGH"],
        "suspected_bugs": [b["id"] for b in SUSPECTED_BUGS],
        "fire_counts": fire_counts(),
        "behaviour_changed": False,
    }
