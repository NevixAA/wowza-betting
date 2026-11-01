"""
PROPS-R2 challenger replay (R2-D, finding 20). RETROSPECTIVE. Runs on GitHub Actions only.

For every props market, on the SAME player_history snapshot and the SAME chronological split:
  A  legacy   - train(recipe="legacy"): batch-median _prep over the whole frame (holdout stats
                leak into training). Exactly what the weekly retrain produces today.
  B  repaired - train(recipe="props_r2_repaired"): persisted fit-block imputer + feature contract.

Holdout metrics (LL, Brier, ECE, calibration intercept/slope, AUC) on the rows BOTH arms score,
paired delta LL / Brier with a week-block bootstrap CI. Nothing is promoted, no serving file is
touched: challenger payloads are written to models/challengers/props_r2/ only.

Usage:  python scripts/props_r2_challengers.py [--markets goals,sot] [--max-rows N]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import model as M  # noqa: E402
from player_model import props_contract as PC  # noqa: E402

OUT_DIR = ROOT / "output" / "props_r2"
MODEL_DIR = ROOT / "models" / "challengers" / "props_r2"
EXP_DIR = ROOT / "config" / "experiments" / "props_r2"
EPS = 1e-6


def _ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p), bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": q})
    g = df.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / len(df))


def _calib(y, p):
    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None, None
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def _metrics(y, p):
    a, b = _calib(y, p)
    return {
        "n": int(len(y)), "events": int(y.sum()), "base_rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "log_loss": float(_ll(y, p).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "ece10": _ece(y, p),
        "calib_intercept": a, "calib_slope": b,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
    }


def _block_ci(delta: np.ndarray, blocks: np.ndarray, reps=1000, seed=7):
    """Week-block bootstrap of mean(delta). Rows of one week move together (players in the same
    matches are correlated), so the CI is honest about dependence."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(ub)
    bs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, k, k)
        bs[i] = sums[idx].sum() / cnts[idx].sum()
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], int(k)


def _ens(results):
    return np.mean([v["holdout_p"] for v in results.values()], axis=0)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def run(markets, max_rows=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(ROOT / "player_history.parquet")
    if max_rows:
        df = df.sort_values("date").tail(max_rows)
    data_info = {
        "rows": int(len(df)),
        "newest_match": str(pd.to_datetime(df["date"], errors="coerce").max())[:10],
        "oldest_match": str(pd.to_datetime(df["date"], errors="coerce").min())[:10],
        "feature_list_sha256": hashlib.sha256(
            json.dumps(list(config.PLAYER_FEATURE_COLS)).encode()).hexdigest(),
        "note": "player_history injury columns are point-in-time since 2026-10-26 "
                "(fb43970c); serving champions were trained on the pre-repair leaked values. "
                "BOTH arms here train on the repaired data, so the comparison isolates "
                "preprocessing.",
    }
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_class": "RETROSPECTIVE",
        "lifecycle_stage": "RETROSPECTIVE_REPLAY",
        "code_sha": _git_sha(),
        "contract_version": PC.CONTRACT_VERSION,
        "imputer_version": PC.IMPUTER_VERSION,
        "data": data_info,
        "promotion": "NONE - no serving file written; owner review required for any cutover",
        "markets": {},
    }
    scored_parts = []
    for market in markets:
        tm = time.time()
        print(f"\n=== {market} ===", flush=True)
        try:
            res_a, hold_a, cut_a = M.train(df, market, return_holdout=True, recipe="legacy")
            res_b, hold_b, cut_b = M.train(df, market, return_holdout=True, recipe=PC.RECIPE_REPAIRED)
        except ValueError as e:
            report["markets"][market] = {"status": "SKIPPED", "why": str(e)}
            continue
        assert len(hold_a) == len(hold_b) and cut_a == cut_b
        assert (hold_a["date"].to_numpy() == hold_b["date"].to_numpy()).all()
        y = hold_a[config.MARKET_TARGETS[market]].astype(int).to_numpy()
        pa, pb = _ens(res_a), _ens(res_b)
        both = ~np.isnan(pb)
        pay_b = M.build_payload(res_b, market)
        diag = M.predict_with_diagnostics(hold_b, pay_b)
        np.testing.assert_allclose(diag["p"].to_numpy()[both], pb[both], rtol=0, atol=1e-9)
        wk = pd.to_datetime(hold_a["date"], errors="coerce").dt.strftime("%G-%V").to_numpy()[both]
        d_ll = _ll(y[both], pb[both]) - _ll(y[both], pa[both])
        d_br = (pb[both] - y[both]) ** 2 - (pa[both] - y[both]) ** 2
        ci_ll, k = _block_ci(d_ll, wk)
        ci_br, _ = _block_ci(d_br, wk)
        # AS SERVED: production scores one fixture's players per call, so the legacy batch
        # median is that fixture's median. Re-score the holdout the same way with the legacy
        # payload to measure how much the batch moves a served probability.
        pay_a = M.build_payload(res_a, market)
        p_served = np.full(len(hold_a), np.nan)
        if "fixture_id" in hold_a.columns:
            h = hold_a.reset_index(drop=True)
            for _, g in h.groupby("fixture_id", sort=False):
                p_served[g.index.to_numpy()] = M.predict_proba(g, pay_a).to_numpy()
        served_ok = ~np.isnan(p_served)
        batch_shift = np.abs(p_served[served_ok] - pa[served_ok])
        raw = pickle.dumps(pay_b)
        sha = hashlib.sha256(raw).hexdigest()
        (MODEL_DIR / f"props_r2_{market}.pkl").write_bytes(raw)
        rec = {
            "experiment_id": f"PROPS-R2-{market}",
            "status": "OK",
            "training_cutoff": cut_a,
            "blocks": pay_b["blocks"],
            "required_fail_excluded": pay_b["required_fail_excluded"],
            "imputer_dropped_unfittable": pay_b["imputer"]["dropped_unfittable"],
            "n_features": {"legacy": len(res_a[next(iter(res_a))]["feature_cols"]),
                           "repaired": len(pay_b["feature_cols"])},
            "legacy": _metrics(y[both], pa[both]),
            "repaired": _metrics(y[both], pb[both]),
            "legacy_all_holdout_rows": _metrics(y, pa),
            "legacy_as_served_per_fixture": (_metrics(y[both & served_ok], p_served[both & served_ok])
                                             if (both & served_ok).any() else None),
            "legacy_batch_sensitivity": ({
                "rows": int(served_ok.sum()),
                "rows_shifted_gt_1e-9": int((batch_shift > 1e-9).sum()),
                "mean_abs_shift": float(batch_shift.mean()),
                "p99_abs_shift": float(np.quantile(batch_shift, 0.99)),
                "max_abs_shift": float(batch_shift.max()),
                "meaning": "|legacy p scored per fixture (as served) - legacy p scored in the "
                           "whole-frame batch (as trained)|; 0 for a batch-invariant model",
            } if served_ok.any() else None),
            "delta_repaired_minus_legacy": {
                "log_loss": float(d_ll.mean()), "log_loss_ci95": ci_ll,
                "brier": float(d_br.mean()), "brier_ci95": ci_br, "week_blocks": k},
            "holdout_feature_health": {
                "rows": int(len(diag)),
                "required_fail_rows": int(diag["required_fail"].sum()),
                "rows_with_any_missing": int((diag["n_missing"] > 0).sum()),
                "rows_with_degraded": int((diag["n_degraded"] > 0).sum()),
                "mean_n_missing": float(diag["n_missing"].mean()),
                "mean_n_degraded": float(diag["n_degraded"].mean()),
            },
            "challenger_artifact": f"models/challengers/props_r2/props_r2_{market}.pkl",
            "challenger_sha256": sha,
            "minutes": round((time.time() - tm) / 60, 1),
        }
        d = rec["delta_repaired_minus_legacy"]
        print(f"  legacy LL {rec['legacy']['log_loss']:.5f}  repaired LL {rec['repaired']['log_loss']:.5f}"
              f"  dLL {d['log_loss']:+.5f} CI {d['log_loss_ci95']}", flush=True)
        report["markets"][market] = rec
        part = diag[["p", "n_missing", "n_degraded", "n_required_missing", "required_fail",
                     "imputer_version", "contract_version", "recipe"]].copy()
        part.insert(0, "p_legacy_served", p_served)
        part.insert(0, "p_legacy", pa)
        part.insert(0, "y", y)
        for c in ("date", "player_id", "fixture_id", "league"):
            if c in hold_b.columns:
                part.insert(0, c, hold_b[c].to_numpy())
        part.insert(0, "market", market)
        scored_parts.append(part)
        del res_a, res_b, hold_a, hold_b, pay_a, pay_b, diag
        gc.collect()
    report["minutes_total"] = round((time.time() - t0) / 60, 1)
    (OUT_DIR / "props_r2_challengers.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if scored_parts:
        pd.concat(scored_parts).to_csv(OUT_DIR / "props_r2_holdout_scored.csv.gz", index=False)
    print(f"\nwrote {OUT_DIR / 'props_r2_challengers.json'} in {report['minutes_total']} min")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(config.MARKETS))
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    run([m for m in a.markets.split(",") if m], a.max_rows)
