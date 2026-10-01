"""§9 — market movement per SEGMENT, not globally. Research table. Changes no production tier.

    python scripts/movement_segments.py

The production drift rule treats CONFIRMED as universally good and applies one set of global
constants to both model tracks. Full-curve evidence says that is wrong: on the main O/U track
new-format CONFIRMED runs +6.8% while standard CONFIRMED runs -12.7%, and on side markets BTTS
CONFIRMED runs +32.4%. One rule cannot be right for all of them.

This writes the per-cell table the brief specifies. It does NOT change a tier. Promising cells
are preregistered and tested on future data only.

NO_CURVE IS NOT NEUTRAL and is never merged with it. "We never saw a price" and "we watched the
price and it did not move" are different facts that call for opposite reactions, and collapsing
them is how drift_signal sat on "New" for 100% of rows while looking like a working signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.promotion_gate import block_bootstrap_ci

LEDGERS = [("main_ou", "bets_ledger.csv"), ("side", "side_bets_ledger.csv"),
           ("half_time", "ht_ledger.csv"), ("props", "player_ledger.csv")]
MIN_CELL = 15


def _settled(d):
    s = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if "tier" in s.columns:                      # props track every tier incl. AVOID
        s = s[~s["tier"].astype(str).str.upper().isin({"AVOID", "WATCH"})]
    s["pnl"] = pd.to_numeric(s["pnl"], errors="coerce")
    return s.dropna(subset=["pnl"])


def _col(s, *names):
    """First present column as a 1-D Series, else all-NaN.

    `DataFrame.get(name)` returns None when the column is absent, and pd.cut on that raises
    "Input array must be 1 dimensional" — which looks like a shape bug and is really a missing
    column. The ledgers genuinely differ: props store `market_odds`, not `odds`.
    """
    for n in names:
        if n in s.columns:
            c = s[n]
            return c.iloc[:, 0] if getattr(c, "ndim", 1) > 1 else c
    return pd.Series(np.nan, index=s.index)


def _bands(s):
    o = pd.to_numeric(_col(s, "odds", "market_odds", "entry_odds"), errors="coerce")
    s["odds_band"] = pd.cut(o, [0, 1.8, 2.2, 3.0, 99],
                            labels=["<1.8", "1.8-2.2", "2.2-3.0", ">3.0"]).astype(str)
    # Entry timing: hours between the tip being generated and kickoff, where both are recorded.
    gen = pd.to_datetime(_col(s, "generated_at", "signal_date"), errors="coerce")
    ko = pd.to_datetime(_col(s, "kickoff_utc", "match_date"), errors="coerce")
    hrs = (ko - gen).dt.total_seconds() / 3600
    s["entry_band"] = pd.cut(hrs, [-1e9, 3, 12, 48, 1e9],
                             labels=["<3h", "3-12h", "12-48h", ">48h"]).astype(str)
    return s


def main() -> int:
    rows = []
    for track, fname in LEDGERS:
        p = config.OUTPUT_DIR / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "mv_signal" not in d.columns:
            print(f"{track}: no mv_* columns — run scripts/market_movement_backfill.py first")
            continue
        s = _bands(_settled(d))
        s["won"] = (s["result"].astype(str).str.upper() == "WIN").astype(float)
        s["model_type"] = _col(s, "model_type").fillna("unknown").astype(str)
        s["market"] = _col(s, "market").fillna(track).astype(str)
        s["league"] = _col(s, "league").fillna("unknown").astype(str)
        for keys, g in s.groupby(["model_type", "market", "league", "odds_band",
                                  "entry_band", "mv_signal"], observed=True):
            n = len(g)
            if n < MIN_CELL:
                continue
            pnl = g["pnl"].to_numpy()
            lo, hi = block_bootstrap_ci(pnl, block=max(2, n // 10))
            clv = pd.to_numeric(g.get("mv_clv_pct"), errors="coerce")
            rows.append({
                "track": track, "model_type": keys[0], "market": keys[1], "league": keys[2],
                "odds_band": keys[3], "entry_time_band": keys[4], "movement_signal": keys[5],
                "n": n, "hit_rate": round(g["won"].mean(), 4),
                "pnl_per_bet": round(float(pnl.mean()), 4), "pnl": round(float(pnl.sum()), 2),
                "mean_clv": round(float(clv.mean()), 3) if clv.notna().any() else None,
                "median_clv": round(float(clv.median()), 3) if clv.notna().any() else None,
                "positive_clv_rate": round(float((clv > 0).mean()), 3) if clv.notna().any() else None,
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "excludes_zero": bool(lo > 0 or hi < 0),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        print("no cells reached the minimum size"); return 0
    out = out.sort_values("pnl_per_bet", ascending=False)
    out.to_csv(config.OUTPUT_DIR / "movement_segments.csv", index=False)

    print(f"{len(out)} cells with n >= {MIN_CELL}\n")
    print("=== movement signal, pooled per track (the production rule's own assumption) ===")
    for track, g in out.groupby("track"):
        agg = (g.groupby("movement_signal")
                .apply(lambda x: pd.Series({
                    "cells": len(x), "n": int(x["n"].sum()),
                    "pnl": round(float(x["pnl"].sum()), 2),
                    "roi": round(float((x["pnl_per_bet"] * x["n"]).sum() / x["n"].sum()), 4)}),
                       include_groups=False))
        print(f"\n{track}:"); print(agg.to_string())
    print(f"\n=== cells whose CI EXCLUDES zero ({int(out['excludes_zero'].sum())} of {len(out)}) ===")
    e = out[out["excludes_zero"]]
    if len(e):
        print(e[["track","model_type","market","league","odds_band","movement_signal",
                 "n","pnl_per_bet","ci_lo","ci_hi"]].head(14).to_string(index=False))
    print(f"\nAt a 5% threshold, {0.05*len(out):.1f} of {len(out)} cells would clear by chance alone.")
    print("These are DISCOVERIES. Preregister before acting — see registry/preregistered_hypotheses.json")
    print(f"\nwrote {config.OUTPUT_DIR / 'movement_segments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
