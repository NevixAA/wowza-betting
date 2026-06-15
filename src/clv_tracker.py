"""
CLV (Closing Line Value) tracker.

CLV = (close_odds - bet_odds) / bet_odds
Positive CLV = we beat the closing line, validating that our edge was real.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

CLV_FILE    = config.OUTPUT_DIR / "clv_tracker.csv"
_FIELDNAMES = ["recorded_at", "bet_id", "match", "market", "bet_odds", "close_odds", "clv"]


def _ensure_file() -> None:
    CLV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CLV_FILE.exists():
        with open(CLV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()


def record_bet_and_clv(
    bet_id: str,
    match: str,
    market: str,
    bet_odds: float,
    close_odds: float,
) -> float:
    """
    Record a bet and its CLV.  Returns the CLV value.

    Call this once the market closes (typically 1-2 min before kick-off).
    A mean CLV > 0 across many bets is the strongest long-run edge signal.
    """
    clv = (close_odds - bet_odds) / bet_odds if bet_odds > 0 else 0.0
    _ensure_file()
    with open(CLV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "bet_id":      bet_id,
            "match":       match,
            "market":      market,
            "bet_odds":    round(bet_odds,   4),
            "close_odds":  round(close_odds, 4),
            "clv":         round(clv,        4),
        })
    if abs(clv) > 0.05:
        direction = "+" if clv > 0 else ""
        log.info(f"CLV  {match} | {market} | bet={bet_odds} close={close_odds} CLV={direction}{clv:.3f}")
    return clv


def get_clv_summary(last_n: int = 0) -> dict:
    """Return aggregate CLV metrics.  last_n=0 means all bets."""
    _ensure_file()
    with open(CLV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if last_n:
        rows = rows[-last_n:]

    clv_vals = [float(r["clv"]) for r in rows if r.get("clv")]
    n = len(clv_vals)
    if not n:
        return {"n_bets": 0, "mean_clv": None, "pct_positive": None}

    return {
        "n_bets":       n,
        "mean_clv":     round(sum(clv_vals) / n, 4),
        "pct_positive": round(sum(1 for v in clv_vals if v > 0) / n * 100, 1),
    }
