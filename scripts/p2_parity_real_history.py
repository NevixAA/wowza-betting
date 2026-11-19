"""Phase 2 / P0 train-serve parity on REAL history (offline; committed caches only).

1. Real-data parity: for every match played on each of the last K match dates in the merged history
   (output/fd_history.parquet cache via data_loader.load_all_matches, download skipped, network
   blocked) whose two clubs play only once that day, compare
     training = build_features(history up to and including that date) row for the match, with
     serving  = build_upcoming_features(the match as a fixture, history strictly before that date)
   on every _FORM_FEATURES column and has_ht_history: equal within TOL, NaN where NaN.
2. Training unchanged by the fix: the same training frame built by the pre-fix code (1ea46192^,
   read with `git show`) must give the identical 26 form columns.

    python scripts/p2_parity_real_history.py <out_dir> [K]
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GIT = r"C:\Program Files\Git\cmd\git.exe"
TOL = 1e-9


def _no_net(*a, **k):
    raise OSError("network blocked (offline parity check)")


socket.socket.connect = _no_net
socket.create_connection = _no_net

from src import feature_engineering as FE  # noqa: E402
from src import data_loader as DL          # noqa: E402

FORM = [c for pair in FE._FORM_FEATURES.values() for c in pair]
COLS = FORM + list(FE.HT_HISTORY_FEATURES)


def _old_fe(out: Path):
    src = subprocess.run([GIT, "-C", str(ROOT), "show", "1ea46192^:src/feature_engineering.py"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    p = out / "fe_pre_fix.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fe_pre_fix_rh", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(out_dir: str, k: int = 6) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    DL._ci_download_all = lambda have=None: []
    hist = DL.load_all_matches(force=True).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist[hist["home_goals"].notna() & hist["away_goals"].notna()]
    dates = sorted(hist["date"].dt.normalize().unique())[-k:]
    upto = hist[hist["date"] <= dates[-1]]

    train = FE.build_features(upto)
    train["date"] = pd.to_datetime(train["date"])

    rows, per_date = [], {}
    for d in dates:
        day = hist[hist["date"].dt.normalize() == d]
        side = pd.concat([day[["league", "home_team"]].rename(columns={"home_team": "t"}),
                          day[["league", "away_team"]].rename(columns={"away_team": "t"})])
        twice = set(map(tuple, side[side.duplicated(keep=False)].to_numpy()))
        fx = day[[(lg, h) not in twice and (lg, a) not in twice
                  for lg, h, a in zip(day["league"], day["home_team"], day["away_team"])]]
        before = hist[hist["date"] < d]
        up = fx[["date", "league", "home_team", "away_team", "odds_over25", "odds_under25"]].reset_index(drop=True)
        sv = FE.build_upcoming_features(up, before)
        tr = up[["date", "league", "home_team", "away_team"]].merge(
            train, on=["date", "league", "home_team", "away_team"], how="left", validate="one_to_one")
        n_bad = 0
        for c in COLS:
            a = tr[c] if c in tr.columns else pd.Series(np.nan, index=tr.index)
            b = sv[c]
            bad = ~((a.isna() & b.isna()) | ((a - b).abs() <= TOL))
            for i in np.flatnonzero(bad.to_numpy()):
                rows.append({"date": str(d)[:10], "league": up.at[i, "league"], "home": up.at[i, "home_team"],
                             "away": up.at[i, "away_team"], "column": c, "training": a.iat[i], "serving": b.iat[i]})
            n_bad += int(bad.sum())
        per_date[str(d)[:10]] = {"fixtures": int(len(up)), "excluded_twice_same_day": int(len(day) - len(fx)),
                                 "cells": int(len(up) * len(COLS)), "mismatched_cells": n_bad}
        print(str(d)[:10], per_date[str(d)[:10]], flush=True)
    pd.DataFrame(rows).to_csv(out / "real_history_mismatches.csv", index=False)

    old = _old_fe(out).build_features(upto)
    same = {}
    for c in FORM:
        a, b = old[c], train[c]
        same[c] = int((~((a.isna() & b.isna()) | ((a - b).abs() <= 0))).sum())
    meta = {"history_rows_played": int(len(hist)), "dates": [str(d)[:10] for d in dates],
            "training_rows": int(len(train)), "per_date": per_date,
            "total_mismatched_cells": int(sum(v["mismatched_cells"] for v in per_date.values())),
            "training_form_cells_differing_pre_fix_vs_now": same}
    (out / "real_history_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
