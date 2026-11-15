"""Train/serve parity of EVERY raw input of the clean-retrain team payloads (offline, committed data).

For each of the last K match dates in the committed history (network blocked, download skipped), every
match whose two clubs play only once that day is built twice:

    training = src.feature_engineering.build_features(history up to and including that date, with the
               COVID seasons dropped first exactly as the trainers drop them - training_rows)
               (the frame the Phase 2 harness fits on: retrain_harness.protocol.load_committed_frame)
    serving  = build_upcoming_features(the match as a fixture, history strictly before that date)
               + src.ht_features.attach_ht_features (the HT block predict.py attaches), the
               two builders src/predict.py runs before scoring

and compared on the raw feature columns the payloads read (their feature_cols minus the
`__missing` indicators, which _prep derives from the same columns). Then each payload scores both
frames and the probability difference is reported, so a skew is measured in what it does to the
served number, not only in cells.

    python scripts/team_train_serve_parity.py <out_dir> <payload.pkl>[,<payload.pkl>...] [K]

When a payload reads the team-information block (src/team_info_features.INFO_FEATURE_COLS), the
training frame gets it from the FULL collected store (what the harness fits on) and the serving
frame from the store TRUNCATED to what had kicked off before each match's tip time
(InfoStore.truncated), i.e. what the live collection holds when predict runs; kickoffs come from the
provider fixture list, as serving receives them.

A payload file may be a serving payload (dict with feature_cols) or a Phase 2 harness candidate
pickle ({'candidate': ChampionRecipe, ...}); the harness's payload is read from the candidate.
"""
from __future__ import annotations

import json
import pickle
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOL = 1e-9


def _no_net(*a, **k):
    raise OSError("network blocked (offline parity check)")


socket.socket.connect = _no_net
socket.create_connection = _no_net

from src import data_loader as DL            # noqa: E402
from src import feature_engineering as FE    # noqa: E402
from src.model import predict_proba          # noqa: E402


def _payload(path: str) -> dict:
    with open(path, "rb") as f:
        o = pickle.load(f)
    if isinstance(o, dict) and "candidate" in o:
        return o["candidate"].payload
    return o


def training_rows(hist: pd.DataFrame) -> pd.DataFrame:
    """The rows the trainers build features from: pipeline.mode_train and the Phase 2 harness
    (retrain_harness.protocol.load_committed_frame) both drop the COVID seasons BEFORE
    build_features. The training side must use that frame, or a serving aggregate over the full
    history looks identical to training here while the served model was trained without those
    seasons (the league_avg_goals skew this check missed until 2026-11-15)."""
    import config
    from src.retrain_harness.protocol import load_config
    if bool(load_config()["data"].get("exclude_covid_seasons")) != bool(config.EXCLUDE_COVID_SEASONS):
        raise ValueError("config.EXCLUDE_COVID_SEASONS and the Phase 2 harness disagree on COVID seasons")
    if not config.EXCLUDE_COVID_SEASONS:
        return hist
    return hist[~config.covid_season_mask(hist).to_numpy()]


def raw_columns(payload: dict) -> list[str]:
    return [c for c in payload["feature_cols"] if not c.endswith("__missing")]


def _serving_info(sv: pd.DataFrame, up: pd.DataFrame, store, TI) -> pd.DataFrame:
    """The info block as serving computes it: per tip time, from the store truncated there."""
    kick = TI.kickoff_of(up, store)
    T = kick - np.timedelta64(TI.EARLY_TIP_MAX_HOURS, "h")
    blk = pd.DataFrame(index=up.index, columns=TI.INFO_FEATURE_COLS, dtype=float)
    for t in np.unique(T):
        m = T == t
        part = TI.info_features(up[m], store.truncated(pd.Timestamp(t, tz="UTC")), kickoffs=kick[m])
        blk.loc[up.index[m], TI.INFO_FEATURE_COLS] = part.to_numpy()
    sv = sv.drop(columns=[c for c in TI.INFO_FEATURE_COLS if c in sv.columns])
    for c in TI.INFO_FEATURE_COLS:
        sv[c] = blk[c].to_numpy(dtype=float)
    return sv


def main(out_dir: str, payload_files: str, k: int = 6) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {"/".join(Path(p).parts[-3:]): _payload(p) for p in payload_files.split(",")}
    cols = sorted({c for p in payloads.values() for c in raw_columns(p)})

    DL._ci_download_all = lambda have=None: []
    hist = DL.load_all_matches(force=True).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist[hist["home_goals"].notna() & hist["away_goals"].notna()]
    dates = sorted(hist["date"].dt.normalize().unique())[-k:]
    train = FE.build_features(training_rows(hist[hist["date"] <= dates[-1]]))
    train["date"] = pd.to_datetime(train["date"])
    from src import team_info_features as TI
    info_on = bool(set(cols) & set(TI.INFO_FEATURE_COLS))
    if info_on:
        store = TI.load_store(train)
        train = train.join(TI.info_features(train, store))

    from src.ht_features import HT_FEATURE_COLS, attach_ht_features
    key = ["date", "league", "home_team", "away_team"]
    cell_rows, prob_rows, per_date = [], [], {}
    for d in dates:
        day = hist[hist["date"].dt.normalize() == d]
        side = pd.concat([day[["league", "home_team"]].rename(columns={"home_team": "t"}),
                          day[["league", "away_team"]].rename(columns={"away_team": "t"})])
        twice = set(map(tuple, side[side.duplicated(keep=False)].to_numpy()))
        fx = day[[(lg, h) not in twice and (lg, a) not in twice
                  for lg, h, a in zip(day["league"], day["home_team"], day["away_team"])]]
        before = hist[hist["date"] < d]
        keep = [c for c in key + ["odds_over25", "odds_under25"] if c in fx.columns]
        up = fx[keep].reset_index(drop=True)
        sv = FE.build_upcoming_features(up, before)
        ht = attach_ht_features(before, sv, identity_aliases=getattr(DL, "IDENTITY_ALIASES", None))
        if len(ht) == len(sv):
            for c in HT_FEATURE_COLS:
                if c in ht.columns:
                    sv[c] = ht[c].to_numpy()
        if info_on:
            sv = _serving_info(sv, up, store, TI)
        tr = up[key].merge(train, on=key, how="left", validate="one_to_one")
        n_bad = 0
        for c in cols:
            a = tr[c] if c in tr.columns else pd.Series(np.nan, index=tr.index)
            b = sv[c] if c in sv.columns else pd.Series(np.nan, index=sv.index)
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            bad = ~((a.isna() & b.isna()) | ((a - b).abs() <= TOL))
            for i in np.flatnonzero(bad.to_numpy()):
                cell_rows.append({"date": str(d)[:10], "league": up.at[i, "league"],
                                  "home": up.at[i, "home_team"], "away": up.at[i, "away_team"],
                                  "column": c, "training": a.iat[i], "serving": b.iat[i]})
            n_bad += int(bad.sum())
        for name, p in payloads.items():
            pa = np.asarray(predict_proba(tr, payload=p), dtype=float)
            pb = np.asarray(predict_proba(sv, payload=p), dtype=float)
            for i in range(len(up)):
                prob_rows.append({"date": str(d)[:10], "payload": name, "league": up.at[i, "league"],
                                  "home": up.at[i, "home_team"], "away": up.at[i, "away_team"],
                                  "p_training": pa[i], "p_serving": pb[i]})
        per_date[str(d)[:10]] = {"fixtures": int(len(up)), "cells": int(len(up) * len(cols)),
                                 "mismatched_cells": n_bad}
        print(str(d)[:10], per_date[str(d)[:10]], flush=True)

    cells = pd.DataFrame(cell_rows)
    probs = pd.DataFrame(prob_rows)
    cells.to_csv(out / "team_parity_cells.csv", index=False)
    probs.to_csv(out / "team_parity_probs.csv", index=False)
    by_col = cells.groupby("column").size().sort_values(ascending=False).to_dict() if len(cells) else {}
    prob_summary = {}
    if len(probs):
        probs["absdiff"] = (probs["p_training"] - probs["p_serving"]).abs()
        for name, g in probs.groupby("payload"):
            fin = g["absdiff"].dropna()
            prob_summary[name] = {"rows": int(len(g)), "nan_rows": int(g["absdiff"].isna().sum()),
                                  "mean_abs_pp": round(float(fin.mean()) * 100, 4) if len(fin) else None,
                                  "max_abs_pp": round(float(fin.max()) * 100, 4) if len(fin) else None,
                                  "rows_over_1pp": int((fin > 0.01).sum()),
                                  "rows_exact": int((fin <= 1e-12).sum())}
    meta = {"dates": [str(d)[:10] for d in dates], "raw_columns": cols, "per_date": per_date,
            "total_mismatched_cells": int(sum(v["mismatched_cells"] for v in per_date.values())),
            "mismatched_cells_by_column": by_col, "probability": prob_summary}
    (out / "team_parity_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "raw_columns"}, indent=2, default=str))
    return meta


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
