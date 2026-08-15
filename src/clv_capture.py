"""
CLV capture — Closing-Line Value infrastructure
================================================
CLV is the single metric that predicts long-run profit, and we have NEVER measured it
(historical odds were a single midnight snapshot — no open→close line). This module logs,
for every selection: the odds we'd take NOW (timestamped) and later the CLOSING odds, then
computes whether we beat the close.

CLV_RECORD is the identification key: without odds-at-bet + odds-at-close (no-vig), you
cannot tell a real early-value pick from noise the market never confirmed.

Two CLV measures per record:
  clv_pct   = odds_bet / odds_close − 1                (got better odds than close → +)
  clv_prob  = p_close_novig − p_bet_novig              (market moved toward your side → +)

Positive, persistent CLV (at cell and book level) is the go/no-go gate before any real stake.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.edge_engine import power_devig, proportional_devig

CLV_FILE = Path(__file__).resolve().parents[1] / "output" / "clv_records.csv"
# NOTE: "pnl" must stay in this list. clv_tracker.settle_results() writes a pnl column into
# clv_records.csv; _save() below uses csv.DictWriter(fieldnames=_FIELDS), which RAISES
# ValueError("dict contains fields not in fieldnames") on any key it doesn't know. So the
# first time grading actually filled pnl, every later log_bet()/capture_close() would have
# crashed. Dormant until now only because grading has never succeeded (2026-08-15).
_FIELDS = ["bet_id", "ts_bet", "market", "player", "match", "side",
           "odds_bet", "under_odds_bet", "odds_close", "under_odds_close",
           "p_bet_novig", "p_close_novig", "clv_pct", "clv_prob", "result", "pnl", "notes"]


def _novig(over_odds, under_odds):
    return power_devig(over_odds, under_odds) or proportional_devig(over_odds, under_odds)


def _load() -> list[dict]:
    if not CLV_FILE.exists():
        return []
    with open(CLV_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(rows: list[dict]) -> None:
    CLV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def log_bet(bet_id: str, market: str, player: str, match: str, side: str,
            odds_bet: float, under_odds_bet: float | None, ts_bet: str | None = None,
            notes: str = "") -> None:
    """Record the price we'd take now + a timestamp. Call at decision time."""
    rows = [r for r in _load() if r["bet_id"] != bet_id]   # upsert
    p_bet = _novig(odds_bet, under_odds_bet)
    rows.append({
        "bet_id": bet_id, "ts_bet": ts_bet or datetime.now(timezone.utc).isoformat(),
        "market": market, "player": player, "match": match, "side": side,
        "odds_bet": odds_bet, "under_odds_bet": under_odds_bet or "",
        "odds_close": "", "under_odds_close": "",
        "p_bet_novig": round(p_bet, 5) if p_bet else "", "p_close_novig": "",
        "clv_pct": "", "clv_prob": "", "result": "", "notes": notes,
    })
    _save(rows)


def capture_close(bet_id: str, odds_close: float, under_odds_close: float | None) -> dict | None:
    """Fill in the closing line (run just before kickoff) and compute CLV."""
    rows = _load()
    hit = None
    for r in rows:
        if r["bet_id"] == bet_id:
            r["odds_close"] = odds_close
            r["under_odds_close"] = under_odds_close or ""
            p_close = _novig(odds_close, under_odds_close)
            r["p_close_novig"] = round(p_close, 5) if p_close else ""
            try:
                r["clv_pct"] = round(float(r["odds_bet"]) / odds_close - 1.0, 4)
            except Exception:
                r["clv_pct"] = ""
            if p_close and r.get("p_bet_novig") not in ("", None):
                r["clv_prob"] = round(p_close - float(r["p_bet_novig"]), 5)
            hit = r
            break
    if hit:
        _save(rows)
    return hit


def clv_report() -> dict:
    """Aggregate CLV over closed records. Primary metric is clv_pct (odds_bet/odds_close−1),
    which works with one-sided (over-only) prices; clv_prob (no-vig) is reported only when
    two-sided closing odds were available. Positive mean => beat the close => genuine signal."""
    all_closed = [r for r in _load() if r.get("clv_pct") not in ("", None)]
    if not all_closed:
        return {"n": 0, "note": "no closed records yet — collection is calendar-time"}
    pct = [float(r["clv_pct"]) for r in all_closed]
    cp = [float(r["clv_prob"]) for r in all_closed if r.get("clv_prob") not in ("", None)]
    n = len(pct)
    beat = sum(1 for x in pct if x > 0)
    mean_pct = sum(pct) / n
    return {
        "n": n,
        "mean_clv_pct": round(mean_pct, 4),
        "mean_clv_prob": round(sum(cp) / len(cp), 5) if cp else None,  # None => over-only feed
        "beat_close_rate": round(beat / n, 3),
        "verdict": "POSITIVE CLV — genuine edge signal" if mean_pct > 0
                   else "non-positive CLV — no edge; do not stake real money",
    }
