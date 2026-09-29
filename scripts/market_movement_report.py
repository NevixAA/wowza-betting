"""Does market movement carry information? Per model x market x league. Read-only.

    python scripts/market_movement_report.py
    python scripts/market_movement_report.py --min-n 20

Reads the `mv_*` columns written by scripts/market_movement_backfill.py and asks one question
in three cuts: when the price moved TOWARD our side between open and close, did we do better?

IT REPORTS. IT DOES NOT RECOMMEND. Every cell here is in-sample on a single season, and the
per-league threshold work is the standing warning: 10 of 10 genuinely-tested cells came back
UNCHANGED and `holds_oos` was False on all 23. A cell that looks strong here is a hypothesis for
a walk-forward in Pro, not a reason to change a tier.

WHY THE CUTS ARE SEPARATE. Invariant 1 keeps standard and new-format apart everywhere except
the one drift rule in production, which applies identical global constants to both. Measured,
they behave differently — new-format's NEUTRAL bucket runs +16.6% against standard's -7.3% —
so pooling them would hide exactly the thing worth finding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

LEDGERS = [("main O/U", "bets_ledger.csv"), ("side markets", "side_bets_ledger.csv"),
           ("half-time", "ht_ledger.csv"), ("player props", "player_ledger.csv")]
NON_BET = {"AVOID", "WATCH"}


def _settled(d):
    if "result" not in d.columns:
        return d.iloc[0:0]
    s = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    # Props track every tier including AVOID; an AVOID is the model declining to bet and
    # counting it is a category error. The team ledgers are tier-filtered at write time.
    if "tier" in s.columns:
        s = s[~s["tier"].astype(str).str.upper().isin(NON_BET)]
    s["pnl"] = pd.to_numeric(s["pnl"], errors="coerce")
    return s.dropna(subset=["pnl"])


def _agg(g):
    n = len(g)
    return pd.Series({
        "n": n,
        "hit": round(float((g["result"].astype(str).str.upper() == "WIN").mean()), 3),
        "pnl": round(float(g["pnl"].sum()), 2),
        "roi": round(float(g["pnl"].sum() / n), 4),
        "mean_clv": round(float(g["mv_clv_pct"].mean()), 2) if g["mv_clv_pct"].notna().any() else None,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=15,
                    help="cells below this are printed but flagged as noise")
    a = ap.parse_args()

    for label, fname in LEDGERS:
        p = config.OUTPUT_DIR / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "mv_signal" not in d.columns:
            print(f"\n{label}: no mv_* columns — run market_movement_backfill.py first")
            continue
        s = _settled(d)
        s = s[s["mv_signal"].astype(str) != "NO_CURVE"]
        if s.empty:
            print(f"\n{'=' * 78}\n{label.upper()}: no settled rows with a captured curve")
            continue

        print(f"\n{'=' * 78}\n{label.upper()}  —  {len(s):,} settled bets with a real price curve")
        print("=" * 78)

        print("\nBy movement signal (CONFIRMED = our price shortened between open and close):")
        g = s.groupby("mv_signal", observed=True).apply(_agg, include_groups=False)
        print(g.to_string())

        mt = "model_type" if "model_type" in s.columns else None
        if mt and s[mt].nunique() > 1:
            print("\nBy model track x signal  — the tracks the one production rule pools:")
            g2 = s.groupby([mt, "mv_signal"], observed=True).apply(_agg, include_groups=False)
            print(g2.to_string())

        if "market" in s.columns and s["market"].nunique() > 1:
            print("\nBy market x signal:")
            g3 = s.groupby(["market", "mv_signal"], observed=True).apply(_agg, include_groups=False)
            print(g3[g3["n"] >= 5].to_string())

        if "league" in s.columns:
            print(f"\nTop league x signal cells with n >= {a.min_n}:")
            g4 = (s.groupby(["league", "mv_signal"], observed=True)
                   .apply(_agg, include_groups=False).reset_index())
            g4 = g4[g4["n"] >= a.min_n].sort_values("roi", ascending=False)
            print(g4.head(12).to_string(index=False) if len(g4)
                  else f"   none reach n={a.min_n}")

        # The honest headline: does CLV predict the outcome at all?
        q = s.dropna(subset=["mv_clv_pct"])
        if len(q) >= 30:
            won = (q["result"].astype(str).str.upper() == "WIN").astype(float)
            r = float(np.corrcoef(q["mv_clv_pct"], won)[0, 1])
            hi = q[q["mv_clv_pct"] > 0]; lo = q[q["mv_clv_pct"] <= 0]
            print(f"\nDoes beating the close predict winning?  corr(CLV, win) = {r:+.3f}  (n={len(q):,})")
            if len(hi) and len(lo):
                print(f"   positive CLV: n={len(hi):>5} ROI {hi['pnl'].sum() / len(hi) * 100:+6.1f}%")
                print(f"   negative CLV: n={len(lo):>5} ROI {lo['pnl'].sum() / len(lo) * 100:+6.1f}%")
    print("\nEvery cell above is in-sample on one season. Treat a strong cell as a hypothesis")
    print("for a Pro walk-forward, never as a reason to move a threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
