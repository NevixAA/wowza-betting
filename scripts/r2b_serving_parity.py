"""Serving-parity replay on REAL data (R2-B). Run once in a main checkout and once in the branch
checkout on the same runner; the two output files must be byte-identical.

    python scripts/r2b_serving_parity.py --out parity.csv

Scores, read-only:
  * Standard O/U 2.5 repaired serving (src.standard_serving: frozen alias map -> serving history
    -> build_upcoming_features_repaired -> predict_proba with the persisted imputer) for
      - every Standard fixture on the committed board (output/predictions.csv), and
      - a replay of the last 30 days of played Standard fixtures (history cut at their first date);
  * every committed LEGACY serving payload (Standard, NF, BTTS, O1.5, O3.5, HT 0.5, HT 1.5) on the
    newest 600 rows of the legacy training feature frame (batch exactly as passed).
Probabilities are written with repr() precision. Nothing is written outside --out.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config  # noqa: E402

LEGACY = ["MODEL_FILE_STANDARD", "MODEL_FILE_NEWFORMAT", "MODEL_FILE_BTTS", "MODEL_FILE_OVER15",
          "MODEL_FILE_OVER35", "HT_MODEL_FILE_05", "HT_MODEL_FILE_15"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from src.data_loader import load_all_matches
    from src.feature_engineering import build_features
    from src.model import predict_proba
    from src import standard_serving as SS
    rows = []
    hist_all = load_all_matches()
    pay = SS.load_payload()
    hist = SS.serving_history(hist_all, pay)
    board = pd.read_csv(ROOT / "output" / "predictions.csv")
    b = board[board["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    b = b.drop_duplicates(["date", "league", "home_team", "away_team"]).reset_index(drop=True)
    if len(b):
        sc = SS.score_fixtures(b[["date", "league", "home_team", "away_team"]], hist, pay)
        for (_, r), p in zip(b.iterrows(), sc["p_over25_repaired"]):
            rows.append(("std_repaired_board", r["date"], r["league"], r["home_team"], r["away_team"], repr(float(p))))
    d = pd.to_datetime(hist["date"])
    rec = hist[d >= d.max() - pd.Timedelta(days=30)]
    cut = pd.to_datetime(rec["date"]).min()
    from src.features_repaired import build_upcoming_features_repaired
    f = build_upcoming_features_repaired(rec[["date", "league", "home_team", "away_team"]], hist,
                                         price_mode="none", team_scope="league", cutoff=cut)
    for (_, r), p in zip(rec.iterrows(), predict_proba(f, payload=pay)):
        rows.append(("std_repaired_replay", r["date"], r["league"], r["home_team"], r["away_team"], repr(float(p))))
    feat = build_features(hist_all).dropna(subset=["over25", "home_scored_last5"]).tail(600)
    for name in LEGACY:
        path = Path(getattr(config, name))
        if not path.exists():
            continue
        with open(path, "rb") as fh:
            lp = pickle.load(fh)
        # The training frame lacks some live-only enrichment columns (e.g. api_implied_*); add
        # every payload feature it is missing as NaN, identically in the main and branch runs,
        # so sklearn sees the payload's exact feature list.
        fl = feat.copy()
        for c in lp["feature_cols"]:
            if c not in fl.columns:
                fl[c] = float("nan")
        for (_, r), p in zip(fl.iterrows(), predict_proba(fl, payload=lp)):
            rows.append((name, r["date"], r["league"], r["home_team"], r["away_team"], repr(float(p))))
    out = pd.DataFrame(rows, columns=["set", "date", "league", "home_team", "away_team", "p"])
    out.to_csv(a.out, index=False)
    print(out.groupby("set").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
