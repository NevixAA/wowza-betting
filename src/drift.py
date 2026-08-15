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
  v9/odds_history_v9.json   -- MUST be committed by predict.yml, see below
  {
    "Home vs Away | YYYY-MM-DD": [
      {"ts": "2026-04-12T14:00:00", "over": 1.85, "under": 1.92},
      {"ts": "2026-04-13T09:00:00", "over": 1.80, "under": 1.97, "n": 4,
       "ts_last": "2026-04-13T11:00:00"},
      ...
    ]
  }

Only PRICE CHANGES are appended. Re-seeing a fixture at an unchanged price bumps `n` and
`ts_last` on the last entry instead of adding a row, so `snapshots[0]` is still the opening
price, `snapshots[-1]` is still the latest/closing price, and the file stays small enough to
commit every predict run.

WHY THAT MATTERS (fixed 2026-08-15): this file used to be gitignored and untracked, so in CI
every predict run started with NO history, wrote exactly one snapshot, and read back n=1.
Result: drift_signal was "New" and over/under_drift were 0.0 for 100% of rows, forever — the
entire drift feature, including the tier upgrade/downgrade in betting.py, was inert, and
opening_odds in the ledger always equalled the current price. Drift only works if the history
survives between runs, which means it has to be committed.

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

# Persistent, NEVER-purged open->close archive for STANDARD-format O/U 2.5 (mirrors
# newformat_odds_history.csv). The JSON above is purged after 10 days, so closing-odds
# lookups race the purge; this CSV is the durable record update_results falls back to.
STD_ARCHIVE_FILE = config.OUTPUT_DIR / "standard_odds_history.csv"

# New-format dense archive. nf_odds_capture.yml already writes newformat_odds_history.csv
# 3x/day, which yields only ~1.2 snapshots per fixture — an entry price, NOT an
# open->moving->close curve. Predict runs every 5 minutes and ALREADY downloads new-format
# odds in the same OddsAPI payload as standard; we were simply discarding them. Archiving
# them here gives new-format the same density standard enjoys (~4.9 price changes/fixture)
# at ZERO extra API cost, which matters because new-format carries most live signals and its
# CLV was unreliable precisely because its "closing" price was often a lone early snapshot.
#
# Deliberately a SEPARATE file from newformat_odds_history.csv: that one is owned by
# nf_odds_capture.yml, and having two workflows write the same CSV would race on the git
# push. update_results reads both.
NF_ARCHIVE_FILE = config.OUTPUT_DIR / "newformat_odds_dense.csv"

_ARCHIVE_COLS = ["snapshot_date", "snapshot_ts", "match_date", "league", "match", "market", "odds"]

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


def _append_persistent_archive(df: pd.DataFrame, ts: str) -> None:
    """Append O/U snapshots to the per-model dense archives (distinct-price append).

    Standard rows -> standard_odds_history.csv, new-format rows -> newformat_odds_dense.csv.
    Both are written on EVERY predict run, which is what turns a lone entry price into a
    real open->moving->close curve. Props are untouched (own capture).
    """
    buckets: dict = {"standard": [], "new_format": []}
    today = ts[:10]
    for _, row in df.iterrows():
        lg = row.get("league", "")
        mt = config.model_type_for_league(lg)
        if mt not in buckets:
            continue                      # unknown league -> not archived
        match = f"{str(row.get('home_team', '')).strip()} vs {str(row.get('away_team', '')).strip()}"
        md = str(row.get("date", ""))[:10]
        # O/U 2.5 (over+under) + O/U 1.5/3.5 — all already fetched by predict's `totals`
        # call, so capturing them here is FREE + dense (every predict run). BTTS is NOT in the
        # totals response (needs the per-event endpoint) -> it stays on the 3x/day captures.
        for mkt, col in (("over25", "odds_over25"), ("under25", "odds_under25"),
                         ("over15", "odds_over15"), ("over35", "odds_over35")):
            try:
                o = float(row.get(col))
            except (TypeError, ValueError):
                continue
            if np.isnan(o) or o <= 1.0:
                continue
            buckets[mt].append({"snapshot_date": today, "snapshot_ts": ts, "match_date": md,
                                "league": str(lg), "match": match, "market": mkt,
                                "odds": round(o, 3)})

    for mt, path in (("standard", STD_ARCHIVE_FILE), ("new_format", NF_ARCHIVE_FILE)):
        rows = buckets[mt]
        if not rows:
            continue
        _write_archive(pd.DataFrame(rows, columns=_ARCHIVE_COLS), path)


def _write_archive(new: pd.DataFrame, path) -> None:
    if path.exists():
        try:
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        except Exception:
            pass
    # keep price CHANGES (consecutive-distinct) per fixture+market -> open..close curve that
    # preserves the true last snapshot even when the line reverts (plain distinct-dedup drops it).
    new = new.sort_values(["match_date", "match", "market", "snapshot_ts"])
    _prev = new.groupby(["match_date", "match", "market"])["odds"].shift()
    new = new[new["odds"].ne(_prev)].sort_values("snapshot_ts")
    new.to_csv(path, index=False)


def _classify(side: str, over_drift: float, under_drift: float,
              n_distinct: int, n_seen: int = 1) -> str:
    """
    Classify whether market movement confirms or conflicts with the model signal.
    side: 'OVER' | 'UNDER' | anything else → Neutral

    n_distinct = how many different PRICES we've stored; n_seen = how many times we've
    observed the fixture. With only one price there is no movement to classify — but that
    means two different things: seen once = genuinely "New", seen many times at a flat price
    = "Neutral" (the market has had chances to move and didn't).
    """
    if n_distinct <= 1:
        return "New" if n_seen <= 1 else "Neutral"

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

        prev = history.setdefault(k, [])
        if prev:
            last = prev[-1]
            if last.get("over") == snap.get("over") and last.get("under") == snap.get("under"):
                # Unchanged price: record that we saw it again rather than appending a
                # duplicate. predict runs every 5 min, so appending blindly would add ~30k
                # entries/day and make this file far too big to commit — and it MUST be
                # committed or drift dies (see module docstring).
                last["n"] = int(last.get("n", 1)) + 1
                last["ts_last"] = ts
                continue
        prev.append(snap)

    _save(history)

    # Durable open->close archive for standard (survives the 10-day JSON purge). Never let a
    # failure here break the JSON snapshot, which drives the live drift signal.
    try:
        _append_persistent_archive(df, ts)
    except Exception as e:
        print(f"[drift] persistent standard archive skipped: {e}")


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
        n     = len(snaps)                                        # distinct PRICES stored
        n_seen = sum(int(s.get("n", 1)) for s in snaps)            # times the fixture was seen

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

        sig = _classify(side, od, ud, n, n_seen)

        rows_drift.append({
            "first_over_odds":  round(first_ov, 3) if not np.isnan(first_ov) else np.nan,
            "first_under_odds": round(first_un, 3) if not np.isnan(first_un) else np.nan,
            "over_drift":       od,
            "under_drift":      ud,
            "drift_signal":     sig,
            "odds_snapshots":   n_seen,
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
