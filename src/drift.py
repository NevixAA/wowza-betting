"""
Odds Drift Tracker (v9)
========================
Snapshots Over/Under 2.5 odds every time the pipeline runs and tracks
how the market moves between first-seen and current odds.

Key insight
-----------
When odds DROP on a side, sharp money came in on that side — the bookmaker
shortens to re-balance. So:
  model says UNDER + Under odds dropped  → market CONFIRMS  → keep/upgrade tier
  model says UNDER + Under odds rose     → market CONFLICTS → downgrade tier
  (same logic for OVER)

Storage
-------
  v9/odds_history_v9.json
  {
    "Home vs Away | YYYY-MM-DD": [
      {"ts": "2026-04-12T14:00:00", "over": 1.85, "under": 1.92},
      {"ts": "2026-04-13T09:00:00", "over": 1.80, "under": 1.97},
      ...
    ]
  }

Drift columns added to predictions DataFrame
--------------------------------------------
  first_over_odds   — odds when first seen
  first_under_odds
  over_drift        — first_over  - current_over  (positive = shortened = money in)
  under_drift       — first_under - current_under
  drift_signal      — Confirmed | Conflicted | Neutral | New
  odds_snapshots    — number of times this fixture has been seen
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HISTORY_FILE = config.ODDS_HISTORY_JSON

DRIFT_COLS = [
    "first_over_odds", "first_under_odds",
    "over_drift", "under_drift",
    "drift_signal", "odds_snapshots",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _key(home: str, away: str, match_date) -> str:
    """Stable fixture key: 'Home vs Away | YYYY-MM-DD'."""
    d = str(match_date)[:10]
    return f"{str(home).strip()} vs {str(away).strip()} | {d}"


def _empty() -> dict:
    return {
        "first_over_odds":  np.nan,
        "first_under_odds": np.nan,
        "over_drift":       np.nan,
        "under_drift":      np.nan,
        "drift_signal":     "New",
        "odds_snapshots":   1,
    }


def _load() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict) -> None:
    HISTORY_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _classify(side: str, over_drift: float, under_drift: float, n: int) -> str:
    """
    Classify whether market movement confirms or conflicts with the model signal.
    side: 'OVER' | 'UNDER' | anything else → Neutral
    """
    if n <= 1:
        return "New"

    ct = config.DRIFT_CONFIRM_THRESHOLD
    cf = config.DRIFT_CONFLICT_THRESHOLD
    s  = str(side).upper()

    if s == "OVER" and not np.isnan(over_drift):
        if over_drift   >  ct:  return "Confirmed"
        if over_drift   < -cf:  return "Conflicted"

    if s == "UNDER" and not np.isnan(under_drift):
        if under_drift  >  ct:  return "Confirmed"
        if under_drift  < -cf:  return "Conflicted"

    return "Neutral"


# ── public API ────────────────────────────────────────────────────────────────

def snapshot(df: pd.DataFrame) -> None:
    """
    Persist today's Over/Under odds for every row in df.
    df must have columns: home_team, away_team, date, odds_over25, odds_under25.
    Call this BEFORE enrich() so the current run is already stored.
    """
    history = _load()
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for _, row in df.iterrows():
        ov  = row.get("odds_over25")
        un  = row.get("odds_under25")
        try:
            ov_f = float(ov)
            un_f = float(un)
        except (TypeError, ValueError):
            continue
        if np.isnan(ov_f) and np.isnan(un_f):
            continue

        k    = _key(row.get("home_team", ""), row.get("away_team", ""), row.get("date", ""))
        snap = {"ts": ts}
        if not np.isnan(ov_f):
            snap["over"]  = round(ov_f, 3)
        if not np.isnan(un_f):
            snap["under"] = round(un_f, 3)

        history.setdefault(k, []).append(snap)

    _save(history)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drift columns to df in-place.
    Requires: home_team, away_team, date, best_side (or signal_tier).
    Returns df with DRIFT_COLS added.
    """
    history = _load()
    df = df.copy()

    rows_drift = []
    for _, row in df.iterrows():
        k    = _key(row.get("home_team", ""), row.get("away_team", ""), row.get("date", ""))
        snaps = history.get(k, [])
        n     = len(snaps)

        if n == 0:
            rows_drift.append(_empty())
            continue

        first = snaps[0]
        last  = snaps[-1]

        first_ov = float(first.get("over",  np.nan))
        first_un = float(first.get("under", np.nan))
        curr_ov  = float(last.get("over",   np.nan))
        curr_un  = float(last.get("under",  np.nan))

        od = round(first_ov - curr_ov, 3) if not (np.isnan(first_ov) or np.isnan(curr_ov)) else np.nan
        ud = round(first_un - curr_un, 3) if not (np.isnan(first_un) or np.isnan(curr_un)) else np.nan

        # Use best_side from betting output, or fall back to signal_tier
        side = str(row.get("best_side", row.get("signal_tier", ""))).upper()
        if side in ("SNIPER", "VALUE", "AVOID"):
            side = ""   # tier labels, not sides — need actual side

        # Infer side from bet column if available
        bet = str(row.get("bet", "")).upper()
        if bet in ("OVER", "UNDER"):
            side = bet

        sig = _classify(side, od, ud, n)

        rows_drift.append({
            "first_over_odds":  round(first_ov, 3) if not np.isnan(first_ov) else np.nan,
            "first_under_odds": round(first_un, 3) if not np.isnan(first_un) else np.nan,
            "over_drift":       od,
            "under_drift":      ud,
            "drift_signal":     sig,
            "odds_snapshots":   n,
        })

    drift_df = pd.DataFrame(rows_drift, index=df.index)
    for col in DRIFT_COLS:
        df[col] = drift_df[col]

    return df


def purge_old(cutoff_days: int = 10) -> int:
    """Remove fixture history older than cutoff_days. Returns count removed."""
    history  = _load()
    cutoff   = date.today() - timedelta(days=cutoff_days)
    to_del   = []
    for k in history:
        try:
            d = datetime.strptime(k.split(" | ")[-1][:10], "%Y-%m-%d").date()
            if d < cutoff:
                to_del.append(k)
        except Exception:
            pass
    for k in to_del:
        del history[k]
    if to_del:
        _save(history)
    return len(to_del)
