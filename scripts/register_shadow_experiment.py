"""Train, FREEZE and register the H-MODEL-01 HGB Over 2.5 challenger(s). Run once.

    python scripts/register_shadow_experiment.py            # both tracks
    python scripts/register_shadow_experiment.py --dry-run  # train + report, write nothing

WHY. H-MODEL-01 ("HGB beats the corrected v9 stack on Over 2.5, prospectively") was registered
with a confirmation window starting 2026-10-02, but the HGB model it names never existed as an
artifact: src/model_validation.evaluate_candidates fitted it in memory and discarded it. A
prospective test needs a FIXED model scored on fixtures it could not have seen, so this script
fits it once and freezes the bytes.

THE RECIPE IS THE ONE THAT PRODUCED THE HISTORICAL RESULT (model_validation._hgb + a sigmoid
calibration on a later block through FrozenEstimator), with only these choices made here:

* DEVELOPMENT CUTOFF = the hypothesis's own `development_period` ("to 2026-09-28"), read from
  registry/preregistered_hypotheses.json, never typed. Every training row is on/before it, so
  every fixture the challenger is later scored on is out-of-sample by construction.
* FIT : CAL = 5 : 1 chronologically — the same proportion as the harness's FIT (60%) : CAL (12%).
* Feature columns = the CURRENT CHAMPION's own `feature_cols` for that track, so at predict time
  champion and challenger read the identical feature row (same `_prep` median imputation).
* Tracks are never pooled (invariant 1): one challenger per track, its own rows only.
* Training frame = exactly mode_train's (COVID seasons excluded, rows with over25 and form).

Nothing here touches a serving model, the champion pointer or the retrain challenger slot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

log = logging.getLogger("register_shadow_experiment")
HYP_FILE = config.BASE_DIR / "registry" / "preregistered_hypotheses.json"
HYP_ID = "H-MODEL-01"
CAL_SHARE = 1 / 6


def _hypothesis() -> dict:
    d = json.loads(HYP_FILE.read_text(encoding="utf-8"))
    hs = d if isinstance(d, list) else d.get("hypotheses", d)
    hs = hs if isinstance(hs, list) else list(hs.values())
    for h in hs:
        if h.get("id") == HYP_ID:
            return h
    raise SystemExit(f"{HYP_ID} not in {HYP_FILE}")


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="standard,newformat")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from src.data_loader import load_all_matches
    from src.feature_engineering import build_features
    from src.model import _prep, load_models
    from src.model_validation import _hgb
    from src import model_registry as reg
    from src import shadow_experiments as SE

    hyp = _hypothesis()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(hyp.get("development_period", "")))
    if not m:
        raise SystemExit(f"{HYP_ID}: development_period unparseable: {hyp.get('development_period')}")
    cutoff = pd.Timestamp(m.group(1))
    log.info(f"{HYP_ID}: development cutoff {cutoff.date()} (from the hypothesis)")

    raw = load_all_matches()
    if config.EXCLUDE_COVID_SEASONS:
        raw = raw[~raw["season"].isin(config.COVID_SEASONS)]
    feat = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"]).copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce")

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for track in [t.strip() for t in a.tracks.split(",") if t.strip()]:
        leagues = (config.STANDARD_FORMAT_LEAGUES if track == "standard"
                   else config.NEW_FORMAT_LEAGUES)
        champ = load_models(model_file=reg.MODEL_FILES[track])
        cols = list(champ["feature_cols"])
        d = (valid[valid["league"].isin(leagues) & (valid["date"] <= cutoff)]
             .sort_values("date", kind="mergesort").reset_index(drop=True))
        k = int(round(len(d) * (1 - CAL_SHARE)))
        X, used = _prep(d, cols)
        if used != cols:
            raise SystemExit(f"{track}: training frame lacks champion features "
                             f"{sorted(set(cols) - set(used))}")
        y = d["over25"].astype(int).to_numpy()
        h = _hgb().fit(X.iloc[:k], y[:k])
        cc = CalibratedClassifierCV(FrozenEstimator(h), method="sigmoid").fit(X.iloc[k:], y[k:])
        p_cal = cc.predict_proba(X.iloc[k:])[:, 1]
        info = {
            "experiment_id": f"{HYP_ID}__{track}", "hypothesis_id": HYP_ID,
            "status": "ACTIVE", "model_type": track, "market": "over25",
            "pred_column": "p_over25", "v91_shadow_column": "p_pro_hgb",
            "challenger_family": "pro_hgb", "trained_at": trained_at,
            "registered_at": trained_at, "training_cutoff": str(cutoff.date()),
            "training_start": str(d["date"].min().date()),
            "fit_rows": k, "cal_rows": int(len(d) - k),
            "cal_from": str(d["date"].iloc[k].date()),
            "training_data_version": hashlib.sha256(
                f"{len(d)}|{d['date'].min()}|{d['date'].max()}|{int(y.sum())}".encode()
            ).hexdigest()[:12],
            "feature_schema": hashlib.sha256("|".join(sorted(cols)).encode()).hexdigest()[:12],
            "n_features": len(cols),
            "champion_at_registration": reg.stamp_for(track).get("model_version", ""),
            "recipe": "model_validation._hgb() fit on first 5/6 chronologically; sigmoid "
                      "CalibratedClassifierCV(FrozenEstimator) on last 1/6; no sample weights; "
                      "features = champion feature_cols; _prep median imputation",
            "cal_block_log_loss": _ll(y[k:], p_cal),
            "code_commit": os.getenv("GITHUB_SHA", "")[:12] or "local",
            "sklearn": __import__("sklearn").__version__,
            "note": "SHADOW ONLY. Never served, never tiered, never staked.",
        }
        log.info(f"[{track}] rows={len(d):,} fit={k:,} cal={len(d) - k:,} "
                 f"{info['training_start']}..{cutoff.date()} cal-block LL={info['cal_block_log_loss']:.5f}")
        if a.dry_run:
            continue
        payload = {"target": "over25", "models": {"pro_hgb": cc}, "feature_cols": cols,
                   "metrics": {"pro_hgb": {"cal_block_log_loss": info["cal_block_log_loss"]}}}
        exp = SE.register(info, payload)
        log.info(f"[{track}] registered {exp['challenger_version']} -> {exp['artifact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
