"""
Portfolio-level stake management.

allocate_daily_stakes()   — cap correlated same-fixture O/U exposure
apply_drawdown_multiplier() — scale stakes during losing runs
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

STAKE_STATE_FILE   = config.OUTPUT_DIR / "stake_state.json"
BET_LOG_FILE       = config.OUTPUT_DIR / "bet_log.csv"

MAX_DAILY_SNIPERS  = 6     # never send more than 6 SNIPER tips per day
DRAWDOWN_WINDOW    = 20    # rolling P&L window (bets)
DRAWDOWN_HALT      = -0.15 # -15% rolling → 40% stake
DRAWDOWN_CAUTION   = -0.08 # -8%  rolling → 70% stake


def allocate_daily_stakes(bets: pd.DataFrame) -> pd.DataFrame:
    """
    Cap same-day correlated exposure.

    Rules applied in order:
    1. One O/U bet per fixture — keep highest edge.
    2. Cap SNIPER count at MAX_DAILY_SNIPERS — trim lowest-edge excess.

    bets must have: home_team, away_team, date, signal_tier, best_edge
    """
    if bets.empty:
        return bets

    bets = bets.copy()

    # Rule 1: one bet per fixture
    bets["_fix"] = (
        bets["home_team"].astype(str) + "|"
        + bets["away_team"].astype(str) + "|"
        + bets["date"].astype(str)
    )
    n_before = len(bets)
    bets = (
        bets.sort_values("best_edge", ascending=False)
        .drop_duplicates(subset="_fix", keep="first")
        .drop(columns="_fix")
        .reset_index(drop=True)
    )
    n_dropped = n_before - len(bets)
    if n_dropped:
        log.info(f"Portfolio: removed {n_dropped} duplicate-fixture bets")

    # Rule 2: cap SNIPER count
    sniper_mask = bets["signal_tier"] == "SNIPER"
    n_snipers   = sniper_mask.sum()
    if n_snipers > MAX_DAILY_SNIPERS:
        keep_idx    = bets[sniper_mask].nlargest(MAX_DAILY_SNIPERS, "best_edge").index
        drop_idx    = bets[sniper_mask].index.difference(keep_idx)
        bets        = bets.drop(index=drop_idx).reset_index(drop=True)
        log.info(
            f"Portfolio: capped SNIPER bets {n_snipers} → {MAX_DAILY_SNIPERS} "
            f"(removed {len(drop_idx)} lowest-edge)"
        )

    return bets


# ── Drawdown stake multiplier ──────────────────────────────────────────────────

def _load_stake_state() -> dict:
    if STAKE_STATE_FILE.exists():
        try:
            return json.loads(STAKE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"multiplier": 1.0, "updated_at": ""}


def _save_stake_state(state: dict) -> None:
    STAKE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAKE_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _compute_drawdown_multiplier() -> float:
    """0.40 / 0.70 / 1.00 based on last DRAWDOWN_WINDOW bets' P&L."""
    if not BET_LOG_FILE.exists():
        return 1.0
    try:
        df = pd.read_csv(BET_LOG_FILE)
    except Exception:
        return 1.0

    if "pnl" not in df.columns or len(df) < 5:
        return 1.0

    recent = df["pnl"].dropna().tail(DRAWDOWN_WINDOW)
    if recent.empty:
        return 1.0

    total    = recent.sum()
    turnover = max(recent.abs().sum(), 1.0)
    roi      = total / turnover

    if roi <= DRAWDOWN_HALT:
        return 0.40
    if roi <= DRAWDOWN_CAUTION:
        return 0.70
    return 1.0


def apply_drawdown_multiplier(stake: float) -> float:
    """Apply drawdown-based stake multiplier; persist state to stake_state.json."""
    multiplier = _compute_drawdown_multiplier()
    state      = _load_stake_state()
    if multiplier != state.get("multiplier"):
        state["multiplier"] = multiplier
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_stake_state(state)
        if multiplier < 1.0:
            log.warning(f"Drawdown protection active: stake ×{multiplier:.2f}")
    return round(stake * multiplier, 4)


def get_stake_state() -> dict:
    """Return current stake multiplier and trigger thresholds."""
    return {
        "multiplier":       _compute_drawdown_multiplier(),
        "drawdown_window":  DRAWDOWN_WINDOW,
        "halt_threshold":   DRAWDOWN_HALT,
        "caution_threshold": DRAWDOWN_CAUTION,
    }
