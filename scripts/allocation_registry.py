"""Bankroll allocation per `league × market × model`, and why the losing cells lose.

    python scripts/allocation_registry.py
    python scripts/allocation_registry.py --bankroll 2000 --diagnose

Writes `output/allocation_registry.json`, which is the file that accumulates. Run it weekly; the
point is the TREND, not today's snapshot.

WHAT THIS IS NOT. It is not a switch. Nothing here enables or disables a league, changes a
threshold, or sizes a live bet. The system has run without a known pipeline defect for about two
weeks, and two league approvals flipped in a single threshold refit on 2026-10-06. Any cell this
tool sizes today is sized on evidence that is weeks old.

WHAT IT IS FOR. Six months from now the question "which cells have earned a stake, and how big?"
becomes answerable. This builds the record that answers it, and makes the waiting legible: every
cell reports how many more settled bets it needs and roughly how long that takes at its own rate.

READ `weeks_to_target` AS THE HEADLINE. A cell at 0.0% with 11 weeks to go is not a failure, it
is a measurement in progress. A cell at 0.0% whose CI is entirely negative is a different thing
and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.allocation import (ALLOCATABLE, NEGATIVE, RELIABLE_FROM, TARGET_N,  # noqa: E402
                            allocate, diagnose)

OUT = config.OUTPUT_DIR / "allocation_registry.json"


def _load() -> pd.DataFrame:
    """Main O/U and side markets, tagged by market, from the live ledgers."""
    frames = []
    for f, mkt in (("bets_ledger.csv", "ou25"), ("side_bets_ledger.csv", None)):
        p = config.OUTPUT_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "source" in d.columns:
            d = d[d["source"].astype(str) == "live"]
        d["market"] = mkt if mkt else d["market"]
        if "model_type" not in d.columns:
            d["model_type"] = d["league"].map(config.model_type_for_league)
        blank = d["model_type"].isna() | d["model_type"].astype(str).str.strip().isin(["", "nan"])
        d.loc[blank, "model_type"] = d.loc[blank, "league"].map(config.model_type_for_league)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=2000.0)
    ap.add_argument("--reliable-from", default=RELIABLE_FROM)
    ap.add_argument("--diagnose", action="store_true",
                    help="explain the losing cells as well as sizing them")
    # SIZING AND DIAGNOSIS NEED DIFFERENT WINDOWS, and conflating them makes the diagnostic
    # useless. Sizing must use only data from a period when the pipeline was known-good, which
    # is currently ~2 weeks and ~70 bets. But a side bias, a longshot drag or a calibration gap
    # is a STRUCTURAL property of a cell — it does not appear or vanish because a data bug was
    # fixed — so diagnosing it on 12 bets answers "too few to diagnose" for every cell while
    # the same question is perfectly answerable over the full season.
    ap.add_argument("--diagnose-from", default=str(getattr(config, "PERFORMANCE_CUTOFF_DATE",
                                                           "2026-08-10")),
                    help="window for DIAGNOSIS (default: the performance cutoff, i.e. more "
                         "history than sizing uses)")
    a = ap.parse_args()

    bets = _load()
    if bets.empty:
        print("no ledger data")
        return 1

    cells, summary = allocate(bets, reliable_from=a.reliable_from)
    print(f"evidence epoch: {a.reliable_from}  "
          f"({summary['rows_used']:,} of {summary['rows_before_epoch_filter']:,} settled staked "
          f"bets are recent enough to count)")
    print(f"{summary['cells']} cells | allocatable {summary['allocatable']} | "
          f"total bankroll committed {summary['total_fraction']:.2%}\n")

    hdr = (f"{'league':<26}{'market':<8}{'track':<11}{'n':>5}{'days':>6}{'roi':>8}"
           f"{'CI lo':>8}{'stake':>8}{'$':>8}  state / wait")
    print(hdr); print("-" * len(hdr))
    for c in cells:
        wait = (f"{c.weeks_to_target:.0f}w to {TARGET_N}"
                if c.weeks_to_target else ("" if c.n_to_target == 0 else "rate unknown"))
        print(f"{c.league[:25]:<26}{c.market:<8}{c.model_type:<11}{c.n:>5}{c.matchdays:>6}"
              f"{(c.roi if c.roi is not None else float('nan')):>8.3f}"
              f"{(c.ci_lo if c.ci_lo is not None else float('nan')):>8.3f}"
              f"{c.fraction:>8.2%}{c.fraction * a.bankroll:>8.0f}  {c.state} {wait}")

    print("\nstates:", {k: v for k, v in summary["states"].items() if v})
    alloc = [c for c in cells if c.state == ALLOCATABLE]
    if alloc:
        print(f"\nCells with a stake today ({len(alloc)}):")
        for c in alloc:
            print(f"  {c.league} · {c.market}: {c.fraction:.2%} "
                  f"(${c.fraction * a.bankroll:.0f}/bet) — {c.reason}")
    else:
        print("\nNo cell has earned a stake yet. That is the expected answer at this stage and "
              "is not a failure of the tool — the evidence epoch opened "
              f"{a.reliable_from} and cells need {TARGET_N} settled bets over "
              "enough separate matchdays to be sized at all.")

    if a.diagnose:
        print("\n" + "=" * 78)
        print("WHY THE LOSING CELLS LOSE — ranked causes, each with the fix it implies")
        print(f"diagnosis window: {a.diagnose_from} onward (wider than the {a.reliable_from} "
              f"sizing epoch — a structural defect does not vanish because a bug was fixed)")
        print("=" * 78)
        d = bets.copy()
        d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce", utc=True)
        d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
        d = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
              & d["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))
              & (d["match_date"] >= pd.Timestamp(a.diagnose_from, tz="UTC"))]
        # Rank losers by the WIDER window. A cell with 1 bet inside the sizing epoch may
        # have 60 in the diagnosis window and be the most informative row in the table.
        agg = (d.groupby(["league", "market", "model_type"])["pnl"]
               .agg(["sum", "size"]).reset_index().sort_values("sum"))
        agg = agg[(agg["sum"] < 0) & (agg["size"] >= 15)]
        if agg.empty:
            print("\n  no losing cell has 15+ bets in the diagnosis window yet")
        for _, row in agg.head(10).iterrows():
            g = d[(d.league == row.league) & (d.market == row.market)
                  & (d.model_type == row.model_type)]
            print(f"\n{row.league} | {row.market} | {row.model_type}   "
                  f"{row['sum']:+.2f}u on {int(row['size'])} bets")
            for r in diagnose(g):
                print(f"    [{r['cause']}] {r['evidence']}")
                print(f"      -> {r['fix']}")

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": ("Accumulating record of what bankroll fraction each league x market x model "
                    "cell justifies. NOT a switch: nothing here enables, disables or sizes a "
                    "live bet. Read weeks_to_target as the headline."),
        "method": ("Quarter-Kelly on the LOWER bound of a matchday-block bootstrap CI, not on "
                   "the point estimate. A cell whose interval includes zero gets exactly 0.0%. "
                   "Capped per cell and in total."),
        **summary,
        "target_n": TARGET_N,
        "cells": [c.as_dict() for c in cells],
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
