"""§2 + §3 — rank every candidate architecture on ONE untouched chronological holdout.

    python scripts/model_ranking.py --target over25

Answers two questions with one evaluation, because both need the same split:

  §2  how big is the production meta-stack's validation leak, really?
      `v9_meta_leaky` reproduces production EXACTLY, including fitting the stacker on the block
      it is then scored on. `meta_corrected` is the same architecture with the stacker fitted on
      its own block. The gap between them is the size of the illusion -- nothing more. It is an
      EVALUATION-INTEGRITY number and does not explain betting losses.

  §3  is Pro's HGB actually the strongest challenger?
      Every candidate sees the same FIT and CAL blocks and is judged on the same HOLDOUT, so the
      differences are architecture and nothing else. A candidate scored on its own split would
      be comparing test sets, not models.

The market is included where a fair comparison exists. If the de-vigged market price beats every
model on the same rows, that is the finding, and a model that improves log loss while losing to
the market is a forecasting improvement rather than a betting edge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

OUT = config.OUTPUT_DIR / "model_ranking.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="over25")
    ap.add_argument("--format", dest="fmt", default="standard",
                    choices=["standard", "newformat", "all"],
                    help="tracks are NEVER pooled (invariant 1); 'all' runs each separately")
    a = ap.parse_args()

    from src.data_loader import load_all_matches
    from src.feature_engineering import build_features
    from src.model_validation import evaluate_candidates, score, ece

    df = build_features(load_all_matches())
    tracks = ["standard", "newformat"] if a.fmt == "all" else [a.fmt]
    report = {}

    for track in tracks:
        # TAG THE TRACK FROM THE CANONICAL FUNCTION, never from a column that may not exist.
        # The first version tested `if "model_type" in df.columns` and fell through to the whole
        # frame when it was absent -- which it is, since build_features does not emit it. Both
        # "tracks" then returned byte-identical numbers on 84,550 pooled rows, i.e. a silent
        # invariant-1 violation that LOOKED like two separate results.
        if "model_type" not in df.columns:
            df = df.assign(model_type=df["league"].map(config.model_type_for_league))
        want = "new_format" if track == "newformat" else track
        d = df[df["model_type"].astype(str) == want]
        d = d[d[a.target].notna()]
        if len(d) < 2000:
            print(f"[{track}] only {len(d)} rows — skipping")
            continue
        print(f"\n=== {track} / {a.target} — {len(d):,} rows ===")
        table, split = evaluate_candidates(d, target=a.target)

        # The market, on the SAME holdout rows. Only where a two-sided price exists: a one-sided
        # quote cannot be de-vigged, and 1/odds with the vig left in is biased high every time.
        #
        # `split.holdout` is an INDEX RANGE, not a frame. The candidates are scored on the rows
        # that range selects out of the chronologically sorted frame, so the market must be
        # scored on exactly those rows or it is being compared on a different test set — the
        # very error this whole module exists to prevent.
        d_sorted = d.sort_values("date", kind="mergesort").reset_index(drop=True)
        lo, hi = split.holdout
        hold = d_sorted.iloc[lo:hi]
        over_c = {"over25": ("odds_over25", "odds_under25")}.get(a.target)
        mkt_row = None
        if over_c and all(c in hold.columns for c in over_c):
            o, u = pd.to_numeric(hold[over_c[0]], errors="coerce"), pd.to_numeric(
                hold[over_c[1]], errors="coerce")
            ok = o.notna() & u.notna() & (o > 1) & (u > 1)
            if ok.sum() >= 200:
                po, pu = 1 / o[ok], 1 / u[ok]
                p_mkt = (po / (po + pu)).to_numpy()        # proportional de-vig, 2-outcome
                y = pd.to_numeric(hold.loc[ok, a.target], errors="coerce").to_numpy()
                m = score(y, p_mkt)
                mkt_row = {"candidate": "market_devigged", **m,
                           "ece": round(ece(y, p_mkt), 5), "n": int(ok.sum()),
                           "note": "SUBSET of the holdout — only rows with a two-sided price"}

        # SECOND PASS, on identical rows. The market can only be scored where a two-sided price
        # exists, so comparing it against models scored on the full holdout compares test sets.
        if mkt_row is not None:
            mask = np.zeros(len(hold), dtype=bool)
            mask[np.flatnonzero(ok.to_numpy())] = True
            table, split = evaluate_candidates(d, target=a.target, score_mask=mask)
            print(f"  (all candidates rescored on the {int(mask.sum())} market-priced holdout "
                  f"rows, so model and market are compared on identical fixtures)")

        print(table.to_string(index=False))
        if mkt_row:
            print(f"\n  market_devigged   log_loss={mkt_row['log_loss']:.5f} "
                  f"brier={mkt_row['brier']:.5f} auc={mkt_row.get('auc')} "
                  f"(n={mkt_row['n']} of {len(hold)})")

        rows = table.to_dict("records")
        by = {r["candidate"]: r for r in rows}
        leak = None
        if "v9_meta_leaky" in by and "meta_corrected" in by:
            leak = round(by["meta_corrected"]["log_loss"] - by["v9_meta_leaky"]["log_loss"], 6)
            print(f"\n  §2 LEAK SIZE: the leaky stack reports {leak:+.6f} log loss better than "
                  f"the same architecture validated honestly.")
        report[track] = {"n": int(len(d)), "target": a.target, "candidates": rows,
                         "market": mkt_row, "leak_log_loss": leak,
                         "holdout_rows": int(len(hold)),
                         "holdout_span": [str(hold["date"].min())[:10],
                                          str(hold["date"].max())[:10]]
                         if "date" in hold.columns else None}

    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
