"""
Retire ledger rows that were never bettable.
============================================
signal_tier and `bet` used to be derived independently: `bet` enforces MIN_OVER_ODDS /
MIN_UNDER_ODDS while signal_tier came from best_edge alone, so a fixture priced below the
minimum kept a live tier with bet="AVOID" and was published as a tip anyway.

Those rows carry side="AVOID". They can NEVER settle, because _find_result only grades
OVER or UNDER, so they sit in the pending column forever and inflate tip counts.

VOID is the correct classification: no stake was ever possible, so there is no result and
no P/L. _settled_only() already excludes VOID from every win-rate and P/L figure, so this
changes no money number — it just stops 122 rows pretending to be live bets.

The code fix (src/betting.py, 2026-08-16) prevents new ones; this retires the backlog.

    python scripts/void_unbettable_rows.py --dry-run
    python scripts/void_unbettable_rows.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

NOTE = "unbettable: no side cleared MIN_ODDS (retired 2026-08-16)"
TARGETS = [
    ("bets_ledger.csv", "side"),
    ("side_bets_ledger.csv", None),   # side markets have no side column; skip unless present
]


def run(dry: bool) -> int:
    total = 0
    for name, sidecol in TARGETS:
        p = config.OUTPUT_DIR / name
        if not p.exists() or sidecol is None:
            continue
        d = pd.read_csv(p, dtype=str)
        if sidecol not in d.columns:
            continue
        for c in ("result", "pnl", "notes"):
            if c not in d.columns:
                d[c] = ""

        unbettable = d[sidecol].astype(str).str.upper() == "AVOID"
        unsettled = ~d["result"].astype(str).str.upper().isin(["WIN", "LOSS", "VOID"])
        mask = unbettable & unsettled
        n = int(mask.sum())
        print(f"{name}: {n} unbettable row(s) to retire "
              f"(of {int(unbettable.sum())} with side=AVOID)")
        if n:
            by_tier = d.loc[mask, "signal_tier"].value_counts().to_dict()
            print(f"   by tier: {by_tier}")
            print(f"   date range: {d.loc[mask,'match_date'].min()} .. {d.loc[mask,'match_date'].max()}")
        if n and not dry:
            d.loc[mask, "result"] = "VOID"
            d.loc[mask, "pnl"] = "0"
            d.loc[mask, "notes"] = d.loc[mask, "notes"].fillna("").replace("nan", "")
            d.loc[mask, "notes"] = NOTE
            d.to_csv(p, index=False)
            print(f"   -> wrote {p.name}")
        total += n
    return total


def main():
    ap = argparse.ArgumentParser(description="VOID ledger rows that had no bettable side")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = run(a.dry_run)
    print(f"\n{'[DRY] would retire' if a.dry_run else 'retired'} {n} row(s)")


if __name__ == "__main__":
    main()
