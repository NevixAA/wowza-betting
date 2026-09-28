"""Try to break the HT 0.5 blend edge before anyone acts on it. READ-ONLY.

    python scripts/ht_robustness.py

WHY. `scripts/ht_recalibrate.py` reports that a 50/50 blend of the recalibrated model with the
de-vigged market price beats the market on Brier (0.2063 vs 0.2148) and that selecting on
>=5% edge would have returned +2.0% over 457 bets, +8.8% over 226 at >=8%.

That is exactly the kind of result this estate has been burned by before, so the job here is to
ATTACK it, not to celebrate it. Four independent ways it could be fake:

  1. ONE REGIME. The priced window is about five weeks (2026-08-22 -> 2026-09-27). If half-time
     scoring simply ran below its long-run rate that month, an all-UNDER selection prints a
     profit that has nothing to do with the model.
  2. ONE SIDE. Every selected bet is UNDER 0.5. A rule that only ever bets one way is a
     directional bet on the base rate wearing a model's clothes.
  3. NOISE. A few hundred bets at odds near 3.2 is a wide distribution. A bootstrap says whether
     the interval clears zero.
  4. A PLACEBO WOULD DO AS WELL. If shuffling the model's probabilities keeps the profit, the
     profit came from the selection's SIZE and the odds band, not from the model.

Test 4 is the one that matters most and the one usually skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from scripts.ht_recalibrate import walk_forward_isotonic

LINE = "05"
TGT, DV = f"ht_over{LINE}", f"o{LINE}_p_devig"
OC, UC = f"o{LINE}_over_odds", f"o{LINE}_under_odds"
RNG = np.random.default_rng(20260928)


def _pnl(q: pd.DataFrame, pcol: str, thr: float):
    """(pnl array, n_over, n_under) for edge-selected flat 1u bets at the closing price."""
    bo = q[(q[pcol] - q[DV]) >= thr]
    bu = q[((1 - q[pcol]) - (1 - q[DV])) >= thr]
    pn = np.concatenate([
        np.where(bo[TGT] == 1, bo[OC] - 1, -1.0) if len(bo) else np.array([]),
        np.where(bu[TGT] == 0, bu[UC] - 1, -1.0) if len(bu) else np.array([]),
    ])
    return pn, len(bo), len(bu)


def main() -> int:
    f = config.OUTPUT_DIR / f"ht_backtest_over{LINE}.csv"
    if not f.exists():
        print(f"{f.name} missing — run scripts/ht_backtest.py --line 05 first")
        return 1
    d = pd.read_csv(f)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["p", TGT, "date"]).sort_values("date").reset_index(drop=True)
    d["p_cal"] = walk_forward_isotonic(d, "p", TGT)
    q = d.dropna(subset=["p_cal", DV]).copy()
    q["p_blend"] = 0.5 * q[DV] + 0.5 * q["p_cal"]

    print(f"priced & calibrated fixtures: {len(q):,}   "
          f"{q['date'].min().date()} -> {q['date'].max().date()}   "
          f"({(q['date'].max() - q['date'].min()).days} days, {q['league'].nunique()} leagues)")
    print(f"half-time OVER 0.5 base rate in this window: {q[TGT].mean():.4f}")

    # ── 1. is the window itself unusual? ────────────────────────────────────────────────
    print("\n1. IS THIS WINDOW A NORMAL ONE?")
    print(f"   over0.5 rate, priced window   : {q[TGT].mean():.4f}  (n={len(q):,})")
    print(f"   over0.5 rate, whole backtest  : {d[TGT].mean():.4f}  (n={len(d):,})")
    diff = q[TGT].mean() - d[TGT].mean()
    verdict = ("the window IS unusual — treat every ROI below with suspicion"
               if abs(diff) > 0.03 else "the window is not unusual on its own")
    print(f"   difference {diff * 100:+.2f}pp — {verdict}")

    # ── 2 & 3. split-half and bootstrap, per threshold ──────────────────────────────────
    print("\n2. DOES IT SURVIVE A CHRONOLOGICAL SPLIT-HALF, AND A BOOTSTRAP?")
    mid = q["date"].quantile(0.5)
    h1, h2 = q[q["date"] <= mid], q[q["date"] > mid]
    rows = []
    for thr in (0.03, 0.05, 0.08, 0.10):
        pn, no, nu = _pnl(q, "p_blend", thr)
        if len(pn) == 0:
            continue
        boot = np.array([RNG.choice(pn, len(pn), replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p1, _, _ = _pnl(h1, "p_blend", thr)
        p2, _, _ = _pnl(h2, "p_blend", thr)
        rows.append({
            "edge>=": f"{thr:.0%}", "bets": len(pn), "over": no, "under": nu,
            "roi": round(pn.mean(), 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "excl_0": "YES" if lo > 0 else "no",
            "h1_n": len(p1), "h1_roi": round(p1.mean(), 4) if len(p1) else np.nan,
            "h2_n": len(p2), "h2_roi": round(p2.mean(), 4) if len(p2) else np.nan,
            "both_pos": "YES" if (len(p1) and len(p2)
                                  and p1.mean() > 0 and p2.mean() > 0) else "no",
        })
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))

    # ── 4. placebo ──────────────────────────────────────────────────────────────────────
    print("\n3. PLACEBO — shuffle the model's probabilities, keep everything else identical.")
    print("   If a shuffled model earns the same, the profit is the odds band, not the model.")
    out = []
    for thr in (0.05, 0.08):
        real, _, _ = _pnl(q, "p_blend", thr)
        sims = []
        for _ in range(400):
            s = q.copy()
            s["p_cal"] = RNG.permutation(s["p_cal"].to_numpy())
            s["p_blend"] = 0.5 * s[DV] + 0.5 * s["p_cal"]
            pn, _, _ = _pnl(s, "p_blend", thr)
            if len(pn):
                sims.append(pn.mean())
        sims = np.array(sims)
        pct = float((sims >= real.mean()).mean()) if len(sims) else np.nan
        out.append({"edge>=": f"{thr:.0%}", "real_roi": round(real.mean(), 4),
                    "placebo_mean": round(float(sims.mean()), 4) if len(sims) else np.nan,
                    "placebo_p95": round(float(np.percentile(sims, 95)), 4) if len(sims) else np.nan,
                    "p_value": round(pct, 4),
                    "verdict": "real beats placebo" if pct < 0.05 else "INDISTINGUISHABLE"})
    print(pd.DataFrame(out).to_string(index=False))

    # ── 5. is it just "bet UNDER on everything"? ────────────────────────────────────────
    print("\n4. IS IT JUST A STANDING UNDER BET?")
    blind = np.where(q[TGT] == 0, q[UC] - 1, -1.0)
    print(f"   blind UNDER on all {len(q):,} priced fixtures: ROI {blind.mean() * 100:+.2f}%")
    for thr in (0.05, 0.08):
        pn, _, _ = _pnl(q, "p_blend", thr)
        print(f"   selected at edge>={thr:.0%} ({len(pn)} bets)       : "
              f"ROI {pn.mean() * 100:+.2f}%   -> selection adds "
              f"{(pn.mean() - blind.mean()) * 100:+.2f}pp over betting every one")
    print("\n   If selection adds nothing over the blind bet, the model is not doing the work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
