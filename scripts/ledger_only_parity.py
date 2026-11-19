"""Train/serve parity of the LEDGER-ONLY path (src/ledger_all_leagues.py) on the TRAINING_ONLY leagues.

Offline (network blocked, download skipped). For each of the last K match dates of the ledger-only
leagues, every finished API-Football fixture of those leagues that day is served EXACTLY the way the
ledger-only writer serves an upcoming one:

    API-Football fixture (league id, team ids)  -> identity_map (results-proven club names)
    -> board_frame -> src.predict.serve_fixtures (history strictly before that date)

and compared with the training frame (src.feature_engineering.build_features on the COVID-filtered
history, the frame every trainer fits on) for the same match, on the raw feature columns of every
served payload (O/U 2.5 Standard, O/U 1.5 / 3.5, BTTS, HT 0.5 / 1.5). Each payload then scores both
frames and the probability difference is reported.

    python scripts/ledger_only_parity.py <out_dir> [K]

PASS = 0 mismatched cells, 0 fixtures unresolved, and every probability equal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import team_train_serve_parity as TP          # noqa: E402  (blocks the network on import)
from src import data_loader as DL             # noqa: E402
from src import feature_engineering as FE     # noqa: E402
from src import ledger_all_leagues as LA      # noqa: E402
from src import prediction_ledger as PL       # noqa: E402
from src.model import predict_proba           # noqa: E402


def main(out_dir: str, k: int = 6) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import config
    std, nf, side = LA.load_payloads()
    payloads = {"standard_ou25": std, **{f"side_{t}": p for t, p in side.items()}}
    from src.model import load_models
    for name, f in (("ht_over05", config.HT_MODEL_FILE_05), ("ht_over15", config.HT_MODEL_FILE_15)):
        if f.exists():
            payloads[name] = load_models(model_file=f)
    cols = sorted({c for p in payloads.values() for c in TP.raw_columns(p)})

    leagues = LA.ledger_only_leagues(PL.LeagueSlugs())
    names = {int(e["api_football_id"]): e["name"] for e in leagues}
    DL._ci_download_all = lambda have=None: []
    hist = DL.load_all_matches(force=True).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist[hist["home_goals"].notna() & hist["away_goals"].notna()]
    afx = pd.read_parquet(LA.AF_FIXTURES)
    afx = afx[afx["league_id"].isin(list(names)) & afx["status"].isin(["FT"])].copy()
    afx["d"] = pd.to_datetime(afx["date"]).dt.normalize()
    lo = hist[hist["league"].isin(list(names.values()))]
    dates = sorted(set(lo["date"].dt.normalize().unique()) & set(afx["d"].unique()))[-k:]
    train = FE.build_features(TP.training_rows(hist[hist["date"] <= dates[-1]]))
    train["date"] = pd.to_datetime(train["date"])
    ident, why = LA.identity_map(hist, afx, leagues)
    af_names = LA.af_names(afx)

    key = ["date", "league", "home_team", "away_team"]
    cell_rows, prob_rows, per_date, unresolved = [], [], {}, []
    for d in dates:
        day = afx[afx["d"] == d]
        fx = pd.DataFrame({"fixture_id": day["fixture_id"].astype(int), "league": day["league_id"].map(names),
                           "league_id": day["league_id"].astype(int), "league_state": "TRAINING_ONLY",
                           "kickoff_utc": pd.to_datetime(day["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=15),
                           "date": day["d"].dt.strftime("%Y-%m-%d"), "home_id": day["home_id"].astype(int),
                           "away_id": day["away_id"].astype(int)}).reset_index(drop=True)
        board = LA.board_frame(fx, ident, why, af_names)
        bad_id = board["not_scored_reason"] != ""
        unresolved += board.loc[bad_id, ["league", "home_team", "away_team", "not_scored_reason"]].to_dict("records")
        board = board[~bad_id].reset_index(drop=True)
        before = hist[hist["date"] < d]
        sv = LA.score(board, before, (std, nf, side))
        sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
        # the training row of the same match (the AF fixture's day; a provider date one day apart is
        # not this parity's question - such a fixture is reported as having no training row)
        tr = sv[key].merge(train, on=key, how="left", validate="one_to_one", indicator=True)
        no_train = tr["_merge"] != "both"
        n_bad = 0
        for c in cols:
            a = pd.to_numeric(tr[c], errors="coerce") if c in tr.columns else pd.Series(np.nan, index=tr.index)
            b = pd.to_numeric(sv[c], errors="coerce") if c in sv.columns else pd.Series(np.nan, index=sv.index)
            bad = ~((a.isna() & b.isna()) | ((a - b).abs() <= TP.TOL)) & ~no_train
            for i in np.flatnonzero(bad.to_numpy()):
                cell_rows.append({"date": str(d)[:10], "league": sv.at[i, "league"], "home": sv.at[i, "home_team"],
                                  "away": sv.at[i, "away_team"], "column": c, "training": a.iat[i],
                                  "serving": b.iat[i]})
            n_bad += int(bad.sum())
        trm = tr[~no_train].reset_index(drop=True)
        svm = sv[~no_train.to_numpy()].reset_index(drop=True)
        for name, p in payloads.items():
            pa = np.asarray(predict_proba(trm, payload=p), dtype=float)
            pb = np.asarray(predict_proba(svm, payload=p), dtype=float)
            for i in range(len(svm)):
                prob_rows.append({"date": str(d)[:10], "payload": name, "league": svm.at[i, "league"],
                                  "home": svm.at[i, "home_team"], "away": svm.at[i, "away_team"],
                                  "p_training": pa[i], "p_serving": pb[i]})
        per_date[str(d)[:10]] = {"fixtures": int(len(fx)), "identity_unresolved": int(bad_id.sum()),
                                 "no_training_row": int(no_train.sum()), "compared": int((~no_train).sum()),
                                 "cells": int((~no_train).sum() * len(cols)), "mismatched_cells": n_bad}
        print(str(d)[:10], per_date[str(d)[:10]], flush=True)

    cells, probs = pd.DataFrame(cell_rows), pd.DataFrame(prob_rows)
    cells.to_csv(out / "ledger_only_parity_cells.csv", index=False)
    probs.to_csv(out / "ledger_only_parity_probs.csv", index=False)
    prob_summary = {}
    if len(probs):
        probs["absdiff"] = (probs["p_training"] - probs["p_serving"]).abs()
        for name, g in probs.groupby("payload"):
            fin = g["absdiff"].dropna()
            prob_summary[name] = {"rows": int(len(g)), "nan_rows": int(g["absdiff"].isna().sum()),
                                  "max_abs_pp": round(float(fin.max()) * 100, 6) if len(fin) else None,
                                  "rows_exact": int((fin <= 1e-12).sum())}
    meta = {"dates": [str(d)[:10] for d in dates], "leagues": sorted(names.values()), "raw_columns": len(cols),
            "per_date": per_date,
            "fixtures": int(sum(v["fixtures"] for v in per_date.values())),
            "compared": int(sum(v["compared"] for v in per_date.values())),
            "cells": int(sum(v["cells"] for v in per_date.values())),
            "total_mismatched_cells": int(sum(v["mismatched_cells"] for v in per_date.values())),
            "identity_unresolved": unresolved,
            "mismatched_cells_by_column": cells.groupby("column").size().to_dict() if len(cells) else {},
            "by_league": {lg: int(n) for lg, n in probs.drop_duplicates(["date", "home", "away"])
                          .groupby("league").size().items()} if len(probs) else {},
            "probability": prob_summary}
    (out / "ledger_only_parity_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_date"}, indent=2, default=str))
    return meta


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
