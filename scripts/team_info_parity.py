"""Row-level train/serve parity of the team-information block over the WHOLE history (offline).

For N sampled match dates (every track, seeded) plus the last K dates of the frame, every row is
computed twice with src.team_info_features.info_features:

    training = the FULL collected store (what the Phase 2 harness fits on)
    serving  = the store TRUNCATED at that row's tip time (kickoff - 153 h): only fixtures that had
               kicked off, careers and transfers dated before it - what the live collection holds
               when predict runs

and every cell is compared (NaN == NaN; otherwise |a - b| <= 1e-9). Expected: 0 differences.

    python scripts/team_info_parity.py --out DIR [--frame FEATURE_FRAME.parquet] [--sample 60] [--last 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import team_info_features as TI  # noqa: E402

TOL = 1e-9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--last", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args(argv)
    if a.frame:
        feat = pd.read_parquet(a.frame)
    else:
        from src.retrain_harness import protocol as P
        feat, _ = P.load_committed_frame(P.load_config())
    feat = feat.reset_index(drop=True)
    store = TI.load_store(feat)
    days = np.array(sorted(pd.to_datetime(feat["date"]).dt.normalize().unique()))
    rng = np.random.default_rng(a.seed)
    pick = set(rng.choice(days, size=min(a.sample, len(days)), replace=False).tolist()) | set(days[-a.last:].tolist())
    rows = feat[pd.to_datetime(feat["date"]).dt.normalize().isin(pick)]
    kick = TI.kickoff_of(rows, store)
    full = TI.info_features(rows, store, kickoffs=kick)
    T = kick - np.timedelta64(TI.EARLY_TIP_MAX_HOURS, "h")
    serve = pd.DataFrame(index=rows.index, columns=TI.INFO_FEATURE_COLS, dtype=float)
    for t in np.unique(T):
        m = T == t
        serve.loc[rows.index[m]] = TI.info_features(rows[m], store.truncated(pd.Timestamp(t, tz="UTC")),
                                                    kickoffs=kick[m]).to_numpy()
    A, B = full.to_numpy(float), serve.to_numpy(float)
    bad = ~((np.isnan(A) & np.isnan(B)) | (np.abs(A - B) <= TOL))
    cells = [{"date": str(rows["date"].iat[i])[:10], "league": rows["league"].iat[i], "home": rows["home_team"].iat[i],
              "away": rows["away_team"].iat[i], "column": TI.INFO_FEATURE_COLS[j], "training": A[i, j], "serving": B[i, j]}
             for i, j in zip(*np.nonzero(bad))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).to_csv(out / "team_info_parity_cells.csv", index=False)
    meta = {"version": TI.INFO_VERSION, "dates": len(pick), "rows": int(len(rows)),
            "tip_times": int(len(np.unique(T))), "cells": int(A.size), "non_missing_cells": int(np.isfinite(A).sum()),
            "mismatched_cells": int(bad.sum()),
            "by_column": pd.DataFrame(cells)["column"].value_counts().to_dict() if cells else {}}
    (out / "team_info_parity_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if not bad.any() else 1


if __name__ == "__main__":
    sys.exit(main())
