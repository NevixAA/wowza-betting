"""
Value Betting Engine — v9
==========================
v8 core + two additions from v7:

1. Both-losing guard
   Reads the both_losing flag from v7's best_params.json.
   If a league is flagged as both_losing → all signals set to AVOID.

2. Drift tier filter (from v7 odds drift concept)
   After assigning the fixed 3-tier label, adjust using drift_signal:
     SNIPER + Conflicted  → downgrade to VALUE
     VALUE  + Confirmed   → upgrade to SNIPER  (if best_edge >= DRIFT_UPGRADE_EDGE)

Fixed 3-tier labels (league-independent):
  SNIPER  — edge >= 0.10   → bet full stake
  VALUE   — edge >= 0.04   → bet half stake / monitor
  AVOID   — edge <  0.04   → skip
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

# ── Per-league params (v9 optimizer output + v7 fallback) ────────────────────

_LEAGUE_PARAMS: Optional[dict] = None


def _load_league_params() -> dict:
    global _LEAGUE_PARAMS
    if _LEAGUE_PARAMS is not None:
        return _LEAGUE_PARAMS

    params: dict = {}

    # 1. Try v9 optimizer output first
    pf9 = config.MODELS_DIR / "best_params_v9.json"
    if pf9.exists():
        try:
            params = json.loads(pf9.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Merge v7 best_params.json — pull both_losing flags and min-odds if missing
    v7_file = config.V7_BEST_PARAMS
    if v7_file.exists():
        try:
            v7 = json.loads(v7_file.read_text(encoding="utf-8"))
            for league, lp in v7.items():
                if league not in params:
                    params[league] = {}
                # Only import the both_losing flag and min-odds from v7
                if "both_losing" not in params[league]:
                    params[league]["both_losing"] = lp.get("both_losing", False)
                if "min_over_odds" not in params[league]:
                    params[league]["min_over_odds"] = lp.get("min_over_odds", config.MIN_OVER_ODDS)
                if "min_under_odds" not in params[league]:
                    params[league]["min_under_odds"] = lp.get("min_under_odds", config.MIN_UNDER_ODDS)
        except Exception as e:
            log.debug(f"Could not load v7 best_params: {e}")

    _LEAGUE_PARAMS = params
    return _LEAGUE_PARAMS


def reload_league_params() -> None:
    global _LEAGUE_PARAMS
    _LEAGUE_PARAMS = None
    _load_league_params()


def _get_league_cfg(league: str) -> dict:
    params = _load_league_params()
    return params.get(str(league)) or params.get("_default") or {}


def _is_both_losing(league: str) -> bool:
    return bool(_get_league_cfg(league).get("both_losing", False))


def _get_min_odds(league: str) -> tuple[float, float]:
    lp = _get_league_cfg(league)
    mo = float(lp.get("min_over_odds",  config.MIN_OVER_ODDS))
    mu = float(lp.get("min_under_odds", config.MIN_UNDER_ODDS))
    return mo, mu


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kelly(p: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    return float(np.clip((b * p - (1 - p)) / b, 0.0, 1.0))


def _stake(edge: float, p: float, odds: float, threshold: float) -> float:
    if edge < threshold:
        return 0.0
    if config.USE_KELLY:
        return round(_kelly(p, odds) * config.KELLY_FRACTION * config.FLAT_STAKE, 4)
    return config.FLAT_STAKE


def _base_tier(edge: float, side: str = "", league: str = "") -> str:
    """
    Three-tier signal system:
      SNIPER   — meets per-league threshold (highest confidence, full stake)
      MARKSMAN — edge >= MARKSMAN_THRESHOLD (medium-high, 3/4 stake)
      VALUABLE — edge >= VALUABLE_THRESHOLD (moderate, half stake / monitor)
      AVOID    — below threshold
    """
    has_per_league = league and league in config.LEAGUE_SNIPER_THRESHOLDS
    if has_per_league:
        sniper_thresh = config.LEAGUE_SNIPER_THRESHOLDS[league]
    elif side == "OVER":
        sniper_thresh = config.SNIPER_THRESHOLD_OVER
    elif side == "UNDER":
        sniper_thresh = config.SNIPER_THRESHOLD_UNDER
    else:
        sniper_thresh = config.SNIPER_THRESHOLD

    if edge >= sniper_thresh:
        return "SNIPER"

    # Ceiling only applies to leagues using the global threshold (no per-league calibration).
    # Per-league thresholds are backtest-optimised — no ceiling needed there.
    # Per-league ceilings override global for new-format leagues with thinner data.
    if not has_per_league:
        ceiling = config.LEAGUE_EDGE_CEILING.get(league, config.EDGE_CEILING) \
                  if hasattr(config, "LEAGUE_EDGE_CEILING") else config.EDGE_CEILING
        if edge > ceiling:
            return "MARKSMAN"

    if edge >= config.MARKSMAN_THRESHOLD:
        return "MARKSMAN"
    if edge >= config.VALUABLE_THRESHOLD:
        return "VALUABLE"
    return "AVOID"


def _apply_drift_adjustment(tier: str, drift_signal: str, best_edge: float) -> str:
    """
    Adjust tier based on market drift signal (three-tier system).
      SNIPER   + Conflicted  → MARKSMAN  (market disagrees — one step down)
      MARKSMAN + Confirmed   → SNIPER    (market confirms, edge strong enough)
      VALUABLE + Confirmed   → MARKSMAN  (market confirms moderate edge)
    """
    if tier == "SNIPER" and drift_signal == "Conflicted":
        log.debug(f"Drift downgrade SNIPER→MARKSMAN (Conflicted, edge={best_edge:.3f})")
        return "MARKSMAN"
    if tier == "MARKSMAN" and drift_signal == "Confirmed" and best_edge >= config.DRIFT_UPGRADE_EDGE:
        log.debug(f"Drift upgrade MARKSMAN→SNIPER (Confirmed, edge={best_edge:.3f})")
        return "SNIPER"
    if tier == "VALUABLE" and drift_signal == "Confirmed":
        log.debug(f"Drift upgrade VALUABLE→MARKSMAN (Confirmed, edge={best_edge:.3f})")
        return "MARKSMAN"
    return tier


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_value(df: pd.DataFrame, p_col: str = "p_over25") -> pd.DataFrame:
    """
    Add value betting columns to df.

    Required input columns : p_over25, odds_over25, odds_under25, league
    Optional (drift)       : drift_signal

    Added columns:
        impl_prob_over, impl_prob_under
        edge_over, edge_under
        kelly_over, kelly_under
        ev_over, ev_under
        stake_over, stake_under
        tier_over, tier_under
        min_odds_blocked
        bet                 OVER | UNDER | AVOID
        bet_stake
        signal_tier         SNIPER | VALUE | AVOID  (drift-adjusted)
        both_losing         True/False — league flagged by v7 optimizer
    """
    df = df.copy()

    p_over  = pd.to_numeric(df[p_col], errors="coerce").clip(0.001, 0.999)
    p_under = 1.0 - p_over
    ov_odds = pd.to_numeric(df["odds_over25"],  errors="coerce").replace(0, np.nan)
    un_odds = pd.to_numeric(df["odds_under25"], errors="coerce").replace(0, np.nan)

    df["impl_prob_over"]  = 1.0 / ov_odds
    df["impl_prob_under"] = 1.0 / un_odds
    df["edge_over"]  = p_over  - df["impl_prob_over"]
    df["edge_under"] = p_under - df["impl_prob_under"]

    df["kelly_over"]  = [_kelly(p, o) for p, o in zip(p_over,  ov_odds.fillna(0))]
    df["kelly_under"] = [_kelly(p, o) for p, o in zip(p_under, un_odds.fillna(0))]

    df["ev_over"]  = df["edge_over"]  * ov_odds
    df["ev_under"] = df["edge_under"] * un_odds

    # Per-league min-odds
    n = len(df)
    mo_arr = np.full(n, config.MIN_OVER_ODDS)
    mu_arr = np.full(n, config.MIN_UNDER_ODDS)
    if "league" in df.columns:
        for lg in df["league"].unique():
            mask = (df["league"] == lg).values
            mo, mu = _get_min_odds(str(lg))
            mo_arr[mask] = mo
            mu_arr[mask] = mu

    mo_s = pd.Series(mo_arr, index=df.index)
    mu_s = pd.Series(mu_arr, index=df.index)

    # Both-losing guard (from v7 optimizer)
    if "league" in df.columns:
        df["both_losing"] = df["league"].map(lambda lg: _is_both_losing(str(lg)))
    else:
        df["both_losing"] = False

    # Stake (uses VALUE_THRESHOLD as minimum to award stake)
    df["stake_over"]  = [_stake(e, p, o, config.VALUE_THRESHOLD) for e, p, o in
                         zip(df["edge_over"],  p_over,  ov_odds.fillna(0))]
    df["stake_under"] = [_stake(e, p, o, config.VALUE_THRESHOLD) for e, p, o in
                         zip(df["edge_under"], p_under, un_odds.fillna(0))]

    # Base tier — per-league threshold if available, else side-specific global
    leagues = df.get("league", pd.Series([""] * len(df), index=df.index))
    df["tier_over"]  = [_base_tier(e, "OVER",  lg) for e, lg in zip(df["edge_over"],  leagues)]
    df["tier_under"] = [_base_tier(e, "UNDER", lg) for e, lg in zip(df["edge_under"], leagues)]

    # Min-odds filter
    over_odds_ok  = ov_odds >= mo_s
    under_odds_ok = un_odds >= mu_s

    df["min_odds_blocked"] = (
        ((df["edge_over"]  >= config.VALUE_THRESHOLD) & ~over_odds_ok.fillna(False)) |
        ((df["edge_under"] >= config.VALUE_THRESHOLD) & ~under_odds_ok.fillna(False))
    )

    # Bet direction (requires edge >= VALUE_THRESHOLD + min-odds + not both_losing)
    over_q  = (df["edge_over"]  >= config.VALUE_THRESHOLD) & over_odds_ok.fillna(False) & ~df["both_losing"]
    under_q = (df["edge_under"] >= config.VALUE_THRESHOLD) & under_odds_ok.fillna(False) & ~df["both_losing"]

    df["bet"] = "AVOID"
    df.loc[over_q,  "bet"] = "OVER"
    df.loc[under_q, "bet"] = "UNDER"
    both = over_q & under_q
    df.loc[both, "bet"] = np.where(
        df.loc[both, "edge_over"] >= df.loc[both, "edge_under"], "OVER", "UNDER"
    )

    df["bet_stake"] = np.where(
        df["bet"] == "OVER",  df["stake_over"],
        np.where(df["bet"] == "UNDER", df["stake_under"], 0.0)
    )

    # Best edge / side
    best_is_over = df["edge_over"] >= df["edge_under"]
    df["best_edge"] = np.where(best_is_over, df["edge_over"], df["edge_under"])
    df["best_side"] = np.where(best_is_over, "OVER", "UNDER")

    # signal_tier: base tier on best edge side
    raw_tier = np.where(best_is_over, df["tier_over"], df["tier_under"])
    df["signal_tier"] = raw_tier

    # Force AVOID when below VALUE_THRESHOLD or both_losing
    df.loc[df["best_edge"] < config.VALUE_THRESHOLD, "signal_tier"] = "AVOID"
    df.loc[df["both_losing"], "signal_tier"] = "AVOID"

    # Drift adjustment (if drift_signal column exists)
    if "drift_signal" in df.columns:
        adjusted = []
        for _, row in df.iterrows():
            tier = str(row["signal_tier"])
            if tier == "AVOID":
                adjusted.append("AVOID")
                continue
            drift = str(row.get("drift_signal", "New"))
            new_tier = _apply_drift_adjustment(tier, drift, float(row["best_edge"]))
            adjusted.append(new_tier)
        df["signal_tier"] = adjusted

    return df


def generate_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return SNIPER + VALUE rows sorted by best_edge descending.
    Both confirmed bets (bet=OVER/UNDER) and VALUE-tier fixtures below per-league
    threshold are included — giving full visibility into all 3 tiers.
    """
    df = df.copy()
    visible = df[df["signal_tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"])].copy()
    visible = visible.sort_values("best_edge", ascending=False)

    output_cols = [
        "date", "league", "home_team", "away_team",
        "best_side", "odds_over25", "odds_under25",
        "edge_over", "edge_under", "best_edge",
        "bet", "bet_stake", "signal_tier",
        "drift_signal", "over_drift", "under_drift", "odds_snapshots",
        "both_losing", "min_odds_blocked",
    ]
    output_cols = [c for c in output_cols if c in visible.columns]
    return visible[output_cols].reset_index(drop=True)
