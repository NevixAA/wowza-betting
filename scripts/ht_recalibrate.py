"""Fix the HT models' tail calibration, and measure whether it actually helps. READ-ONLY.

    python scripts/ht_recalibrate.py                # both lines
    python scripts/ht_recalibrate.py --line 05

THE DEFECT THIS TARGETS. `scripts/ht_backtest.py` showed the HT models are GOOD at ranking
(ht_over05 AUC 0.663 out-of-sample on 25,457 fixtures, +5.8% Brier skill) but badly
OVER-DISPERSED: they push probabilities toward the extremes far harder than reality supports.

    claimed 0.75-0.80 -> happened 0.588   (+18.5pp)
    claimed 0.80-0.90 -> happened 0.561   (+27.9pp)
    claimed 0.90-1.00 -> happened 0.653   (+27.4pp)
    ...and the mirror image at the bottom: claimed 0.20-0.30 -> happened 0.386 (-13.8pp)

Overall the model looks almost unbiased (-2.03pp) because those two errors cancel. That average
is why the defect survived: the training metric reports the average, while tips are sent ONLY
from the tail, where the model is 22.9pp overconfident.

THE FIX. Isotonic regression — a monotone map from claimed probability to observed frequency.
It cannot change the model's RANKING (so the AUC is untouched by construction) and can only
restate its confidence. That is exactly the shape of this defect.

LEAK-FREE, WHICH IS THE WHOLE POINT. The map is fitted walk-forward: for each chronological
chunk it uses ONLY earlier out-of-sample predictions. Fitting isotonic on the same rows it is
scored against would produce a beautiful calibration curve that means nothing, which is the
standard way this analysis goes wrong.

It also reports the MARKET BLEND separately, because "the model is now honest" and "the model
beats the price" are different claims and only the second one pays.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

#: Where the live notifier sends from. Calibration is judged HERE, not on the average.
LIVE_BANDS = {"05": [("OVER  p>=0.75", "over", 0.75), ("UNDER p<=0.30", "under", 0.30)],
              "15": [("OVER  p>=0.60", "over", 0.60), ("UNDER p<=0.25", "under", 0.25)]}

#: Chronological chunks for the walk-forward calibration fit. The first chunk cannot be
#: calibrated (nothing precedes it) and is reported as such rather than silently passed through.
N_CHUNKS = 8


def walk_forward_isotonic(d: pd.DataFrame, pcol: str, ycol: str,
                          n_chunks: int = N_CHUNKS) -> pd.Series:
    """Out-of-sample recalibrated probability. NaN where no prior data existed to fit on.

    `d` MUST ALREADY BE SORTED BY DATE, AND THIS FUNCTION MUST NOT RE-SORT IT.

    The first version did `d = d.sort_values("date").reset_index(drop=True)` defensively here,
    and that silently destroyed the result. pandas' default sort is NOT stable, so re-sorting a
    frame with thousands of same-date fixtures permutes rows within each day; the returned
    Series then carried a fresh 0..n-1 index in the PERMUTED order, and assigning it back onto
    the caller's frame aligned by label — landing every calibrated value on a different fixture.

    It announced itself loudly, which is the only reason it was caught: AUC fell from 0.667 to
    0.475. Isotonic regression is monotone, so it CANNOT change ranking — an AUC that moves at
    all, let alone below 0.5, is proof of a row-alignment bug and not of a bad calibrator.

    This is the same trap CLAUDE.md records against `pd.merge_asof` in v11's momentum work,
    where it went unnoticed for weeks and invalidated every published number. There the symptom
    was plausible; here it was absurd. Index alignment after any reorder is the hazard, not the
    particular function that reorders.
    """
    from sklearn.isotonic import IsotonicRegression
    out = pd.Series(np.nan, index=d.index, dtype=float)
    edges = np.linspace(0, len(d), n_chunks + 1).astype(int)
    for k in range(1, n_chunks):
        tr = d.iloc[:edges[k]]
        te = d.iloc[edges[k]:edges[k + 1]]
        if te.empty or tr[ycol].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(tr[pcol].to_numpy(float), tr[ycol].to_numpy(float))
        out.iloc[edges[k]:edges[k + 1]] = iso.predict(te[pcol].to_numpy(float))
    return out


def band_table(p: pd.Series, y: pd.Series, line: str) -> pd.DataFrame:
    rows = []
    for name, side, thr in LIVE_BANDS[line]:
        m = (p >= thr) if side == "over" else (p <= thr)
        n = int(m.sum())
        if n == 0:
            rows.append({"band": name, "n": 0, "claimed": np.nan,
                         "realised": np.nan, "gap_pp": np.nan})
            continue
        claimed = float(p[m].mean()) if side == "over" else float(1 - p[m].mean())
        realised = float(y[m].mean()) if side == "over" else float(1 - y[m].mean())
        rows.append({"band": name, "n": n, "claimed": round(claimed, 4),
                     "realised": round(realised, 4),
                     "gap_pp": round((claimed - realised) * 100, 1)})
    return pd.DataFrame(rows)


def edge_roi(q: pd.DataFrame, pcol: str, line: str, thresholds=(0.02, 0.03, 0.04, 0.05, 0.08)):
    """Flat 1u, settled at the CLOSING price, against the de-vigged market probability."""
    tgt, dv = f"ht_over{line}", f"o{line}_p_devig"
    oc, uc = f"o{line}_over_odds", f"o{line}_under_odds"
    rows = []
    for thr in thresholds:
        bo = q[(q[pcol] - q[dv]) >= thr]
        bu = q[((1 - q[pcol]) - (1 - q[dv])) >= thr]
        n = len(bo) + len(bu)
        if n == 0:
            rows.append({"edge>=": f"{thr:.0%}", "bets": 0, "roi": np.nan, "hit": np.nan})
            continue
        pnl = (np.where(bo[tgt] == 1, bo[oc] - 1, -1).sum()
               + np.where(bu[tgt] == 0, bu[uc] - 1, -1).sum())
        wins = int((bo[tgt] == 1).sum() + (bu[tgt] == 0).sum())
        rows.append({"edge>=": f"{thr:.0%}", "bets": n, "roi": round(float(pnl / n), 4),
                     "hit": round(wins / n, 3), "over": len(bo), "under": len(bu)})
    return pd.DataFrame(rows)


def run(line: str) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    f = config.OUTPUT_DIR / f"ht_backtest_over{line}.csv"
    if not f.exists():
        print(f"[ht_over{line}] {f.name} not found — run scripts/ht_backtest.py first")
        return {}
    d = pd.read_csv(f)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    tgt = f"ht_over{line}"
    d = d.dropna(subset=["p", tgt, "date"]).sort_values("date").reset_index(drop=True)

    d["p_cal"] = walk_forward_isotonic(d, "p", tgt)
    ok = d.dropna(subset=["p_cal"]).copy()
    skipped = len(d) - len(ok)

    print(f"\n{'=' * 78}\nHT OVER {line[0]}.{line[1]} — RECALIBRATION  "
          f"(n={len(ok):,} scored; first {skipped:,} rows had no prior data to fit on)")
    print("=" * 78)

    y = ok[tgt].astype(float)
    res = {"line": line, "n": int(len(ok)), "n_uncalibratable": int(skipped)}
    print(f"\n{'metric':<22}{'raw':>12}{'recalibrated':>16}   verdict")
    for name, fn, better in (("AUC (ranking)", roc_auc_score, "higher"),
                             ("Brier", brier_score_loss, "lower"),
                             ("log loss", lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6)),
                              "lower")):
        r = float(fn(y, ok["p"])); c = float(fn(y, ok["p_cal"]))
        win = (c > r) if better == "higher" else (c < r)
        print(f"{name:<22}{r:>12.4f}{c:>16.4f}   {'BETTER' if win else 'worse '}")
        res[f"raw_{name.split()[0].lower()}"] = round(r, 5)
        res[f"cal_{name.split()[0].lower()}"] = round(c, 5)

    print("\nCalibration by band — raw vs recalibrated:")
    for lbl, col in (("RAW        ", "p"), ("RECALIBRATED", "p_cal")):
        b = ok.copy()
        b["bin"] = pd.cut(b[col], [0, .3, .5, .7, .75, .8, .9, 1.0], include_lowest=True)
        g = (b.groupby("bin", observed=True)
               .agg(n=(tgt, "size"), claimed=(col, "mean"), realised=(tgt, "mean")).dropna())
        g["gap_pp"] = ((g["claimed"] - g["realised"]) * 100).round(1)
        print(f"\n  {lbl}")
        print("  " + g.round(3).to_string().replace("\n", "\n  "))

    print("\nTHE BAND THAT SHIPS — this is the number that matters:")
    braw = band_table(ok["p"], y, line).rename(columns={"gap_pp": "gap_pp_RAW",
                                                        "n": "n_RAW"})
    bcal = band_table(ok["p_cal"], y, line).rename(columns={"gap_pp": "gap_pp_CAL",
                                                            "n": "n_CAL"})
    m = braw[["band", "n_RAW", "gap_pp_RAW"]].merge(bcal[["band", "n_CAL", "gap_pp_CAL"]],
                                                    on="band")
    print(m.to_string(index=False))
    res["bands"] = m.to_dict("records")

    # ── against the market ───────────────────────────────────────────────────────────────
    dv = f"o{line}_p_devig"
    if dv in ok.columns and ok[dv].notna().sum() >= 50:
        q = ok.dropna(subset=[dv]).copy()
        mb = brier_score_loss(q[tgt], q[dv])
        wr = brier_score_loss(q[tgt], q["p"])
        wc = brier_score_loss(q[tgt], q["p_cal"])
        q["p_blend"] = 0.5 * q[dv] + 0.5 * q["p_cal"]
        bb = brier_score_loss(q[tgt], q["p_blend"])
        print(f"\nAGAINST THE MARKET — priced rows only, n={len(q):,}")
        print(f"  market alone            {mb:.4f}")
        print(f"  model raw               {wr:.4f}")
        print(f"  model recalibrated      {wc:.4f}")
        print(f"  50/50 blend (calibrated){bb:>9.4f}   "
              f"-> {'BEATS' if bb < mb else 'loses to'} the market by {mb - bb:+.5f}")
        res.update({"priced_n": int(len(q)), "market_brier": round(float(mb), 5),
                    "model_raw_brier": round(float(wr), 5),
                    "model_cal_brier": round(float(wc), 5),
                    "blend_brier": round(float(bb), 5)})

        print("\n  Edge selection vs the de-vigged close (flat 1u, settled at the close):")
        for lbl, col in (("RAW model  ", "p"), ("RECALIBRATED", "p_cal"),
                         ("BLEND 50/50", "p_blend")):
            e = edge_roi(q, col, line)
            print(f"\n  {lbl}")
            print("  " + e.to_string(index=False).replace("\n", "\n  "))
            res[f"edge_{col}"] = e.to_dict("records")
        print("\n  Read the bets column first. A high ROI on a handful of bets is noise.")
    else:
        print("\nAGAINST THE MARKET — too few priced rows to compare.")
        res["priced_n"] = 0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["05", "15", "both"], default="both")
    a = ap.parse_args()
    out = [r for r in (run(l) for l in (["05", "15"] if a.line == "both" else [a.line])) if r]
    if out:
        p = config.OUTPUT_DIR / "ht_recalibration_summary.json"
        p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
