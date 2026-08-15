"""
One-time (re-runnable) CLV backfill for ALREADY-SETTLED ledger rows.
=====================================================================
update_results only ever visits rows whose `result` is still empty, so any row that settled
while the closing-price lookup was broken keeps a blank closing_odds/clv_pct forever. Two
fixes landed on 2026-08-15 that make those rows resolvable after the fact:

  * _closing_from_csv now falls back to _names_match, so OddsAPI ledger names reconcile with
    API-Football archive names ("Viborg FF"/"Viborg", "Tijuana"/"Club Tijuana");
  * side markets read closing prices from the archives at all.

This script applies both retroactively. It only ever FILLS BLANKS — an existing
closing_odds/clv_pct is never overwritten — so it is safe to re-run.

    python scripts/backfill_clv.py --dry-run
    python scripts/backfill_clv.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import update_results as ur


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))


def _pct(entry: float, close: float) -> float:
    """Same convention as bets_ledger / ht_ledger: percent, positive = beat the close."""
    return round((entry - close) / close * 100.0, 2)


def backfill_main(dry: bool) -> int:
    f = config.OUTPUT_DIR / "bets_ledger.csv"
    if not f.exists():
        print("bets_ledger.csv not found")
        return 0
    led = pd.read_csv(f, dtype=str)
    for c in ("closing_odds", "clv_pct"):
        if c not in led.columns:
            led[c] = ""

    settled = led["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
    target = settled & _blank(led["closing_odds"])
    idxs = list(led[target].index)
    print(f"bets_ledger      : {len(led)} rows | {int(settled.sum())} settled | "
          f"{len(idxs)} settled-without-CLV")

    filled = 0
    for i in idxs:
        r = led.loc[i]
        close = ur._closing_odds(r["home_team"], r["away_team"], r["match_date"], r["side"])
        if close is None or (isinstance(close, float) and np.isnan(close)) or close <= 1.0:
            continue
        try:
            entry = float(r["odds"])
        except (TypeError, ValueError):
            continue
        if entry <= 1.0:
            continue
        led.at[i, "closing_odds"] = str(round(close, 3))
        led.at[i, "clv_pct"] = str(_pct(entry, close))
        filled += 1

    print(f"                 -> filled {filled}")
    if filled and not dry:
        led.to_csv(f, index=False)
    return filled


def backfill_side(dry: bool) -> int:
    f = config.OUTPUT_DIR / "side_bets_ledger.csv"
    if not f.exists():
        print("side_bets_ledger.csv not found")
        return 0
    led = pd.read_csv(f, dtype=str)
    for c in ("closing_odds", "clv_pct"):
        if c not in led.columns:
            led[c] = ""

    settled = led["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
    target = settled & _blank(led["closing_odds"])
    idxs = list(led[target].index)
    print(f"side_bets_ledger : {len(led)} rows | {int(settled.sum())} settled | "
          f"{len(idxs)} settled-without-CLV")

    filled = 0
    for i in idxs:
        r = led.loc[i]
        mkt = ur._SIDE_ARCHIVE_MARKET.get(str(r["market"]), str(r["market"]))
        close = ur._closing_for_market(r["home_team"], r["away_team"], r["match_date"], mkt)
        if np.isnan(close) or close <= 1.0:
            continue
        try:
            entry = float(r["odds"])
        except (TypeError, ValueError):
            continue
        if entry <= 1.0:
            continue
        led.at[i, "closing_odds"] = str(round(close, 3))
        led.at[i, "clv_pct"] = str(_pct(entry, close))
        filled += 1

    print(f"                 -> filled {filled}")
    if filled and not dry:
        led.to_csv(f, index=False)
    return filled


def report():
    for name in ("bets_ledger.csv", "side_bets_ledger.csv"):
        p = config.OUTPUT_DIR / name
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str)
        s = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
        if s.empty:
            continue
        have = int((~_blank(s["clv_pct"])).sum())
        vals = pd.to_numeric(s.loc[~_blank(s["clv_pct"]), "clv_pct"], errors="coerce").dropna()
        mean = f"{vals.mean():+.2f}%" if len(vals) else "n/a"
        beat = f"{(vals > 0).mean() * 100:.0f}%" if len(vals) else "n/a"
        print(f"  {name:<22} CLV coverage {have}/{len(s)} ({have/len(s)*100:.1f}%) | "
              f"mean CLV {mean} | beat-close {beat}")


def main():
    ap = argparse.ArgumentParser(description="Backfill closing_odds/clv_pct on settled rows")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    n = backfill_main(a.dry_run) + backfill_side(a.dry_run)
    print(f"\n{'[DRY] would fill' if a.dry_run else 'filled'} {n} row(s) total\n")
    print("CLV coverage after:" if not a.dry_run else "CLV coverage now (unchanged, dry run):")
    report()


if __name__ == "__main__":
    main()
