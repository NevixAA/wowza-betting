"""Promote the props clean-retrain models the owner approved into serving (props_gate.promote_clean_retrain).

    python scripts/promote_props_clean.py --artifact-dir <downloaded props-clean-retrain artifact> \
        --run-id 42513445187 --markets sot,sot2,goals,assists,cards [--dry-run]

Owner 2026-11-08: "u can upload the new props ofcourse, and still collect the opening moving closing".

WHICH CANDIDATE PER MARKET (a rule, computed here from the run's own holdout predictions):
hgb_nan_native serves when it beats recipe_retrained on the holdout - the paired per-row log-loss
difference (hgb - recipe), resampled by ISO week (BOOT resamples, fixed seed), has its 95% upper bound
below 0. Otherwise recipe_retrained serves (the current design; no proven gain from the alternative).
Both must beat player_base_rate (the report's own week-block interval upper bound < 0), or the market is
not promoted at all.

Every number recorded in the promotion comes from the artifact: the report (metrics, hashes, code sha)
and props_harness_holdout_scored.csv.gz (the head-to-head). Markets hidden pending their own clean
retrain (config.HIDDEN_PENDING_CLEAN_RETRAIN) are refused by the gate.
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

from player_model import props_gate as PPG  # noqa: E402

#: the owner's words recorded with each promotion (--approval / --approved-on override them)
APPROVAL = "u can upload the new props ofcourse, and still collect the opening moving closing"
APPROVED_ON = "2026-11-08"
BOOT = 4000
SEED = 1
EPS = 1e-6


def _ll(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def head_to_head(scored: pd.DataFrame, market: str) -> dict:
    """hgb_nan_native - recipe_retrained, per-row log loss, ISO-week block bootstrap."""
    g = scored[scored["market"] == market].copy()
    g["d"] = _ll(g["y"], g["p_hgb_nan_native"]) - _ll(g["y"], g["p_recipe_retrained"])
    g["wk"] = pd.to_datetime(g["date"]).dt.strftime("%G-%V")
    wk = g.groupby("wk")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    w = len(wk)
    s, c = wk["sum"].to_numpy(), wk["count"].to_numpy()
    bs = np.array([s[i].sum() / c[i].sum() for i in (rng.integers(0, w, w) for _ in range(BOOT))])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(len(g)), "weeks": int(w), "ll_delta_hgb_minus_recipe": float(g["d"].mean()),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "p_one_sided": float((np.sum(bs >= 0) + 1) / (len(bs) + 1)), "boot": BOOT, "seed": SEED}


def choose(market_rep: dict, h2h: dict) -> tuple[str | None, str]:
    vs = market_rep["holdout"]["log_loss_minus_base_rate"]
    beats_base = {c: (vs.get(c) or {}).get("ci95", [None, None])[1] for c in ("hgb_nan_native", "recipe_retrained")}
    if h2h["ci95_hi"] < 0 and beats_base["hgb_nan_native"] is not None and beats_base["hgb_nan_native"] < 0:
        return "hgb_nan_native", "beats recipe_retrained (CI95 upper < 0) and player_base_rate"
    if beats_base["recipe_retrained"] is not None and beats_base["recipe_retrained"] < 0:
        return "recipe_retrained", "hgb_nan_native not shown better than the recipe; recipe beats player_base_rate"
    return None, "no candidate beats player_base_rate"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--markets", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--approval", default=APPROVAL)
    ap.add_argument("--approved-on", default=APPROVED_ON)
    a = ap.parse_args(argv)
    d = Path(a.artifact_dir)
    rep = json.loads((d / "output" / "props_harness" / "props_harness_report.json").read_text(encoding="utf-8"))
    if rep.get("status") != "OK":
        raise SystemExit(f"run status is {rep.get('status')!r}, not OK - nothing promoted")
    scored = pd.read_csv(d / "output" / "props_harness" / "props_harness_holdout_scored.csv.gz")
    out = []
    for m in [x.strip() for x in a.markets.split(",") if x.strip()]:
        mr = rep["markets"][m]
        h2h = head_to_head(scored, m)
        cand, why = choose(mr, h2h)
        row = {"market": m, "candidate": cand, "why": why, "head_to_head": h2h}
        if cand is None:
            out.append(row)
            continue
        art = mr["artifacts"][cand]
        raw = (d / art["path"]).read_bytes()
        hm = mr["holdout"]["metrics"][cand]
        metrics = {"n": hm["n"], "events": hm["events"], "log_loss": hm["log_loss"], "brier": hm["brier"],
                   "ece10": hm["ece10"], "calib_slope": hm["calib_slope"],
                   "ll_minus_player_base_rate": mr["holdout"]["log_loss_minus_base_rate"][cand],
                   "ll_delta_vs_recipe": h2h["ll_delta_hgb_minus_recipe"] if cand == "hgb_nan_native" else 0.0,
                   "ci95_lo": h2h["ci95_lo"], "ci95_hi": h2h["ci95_hi"], "head_to_head": h2h,
                   "holdout_dates": mr["holdout"]["dates"], "rule": why}
        if a.dry_run:
            PPG.load_clean_artifact(raw, art["sha256"])
            row["artifact_sha256"] = art["sha256"]
        else:
            rec = PPG.promote_clean_retrain(m, raw, expected_sha256=art["sha256"], run_id=a.run_id,
                                            code_sha=rep["code_sha"], holdout_metrics=metrics,
                                            approval=a.approval, approved_on=a.approved_on)
            row.update(artifact_sha256=art["sha256"], serving_sha256=rec["serving_sha256"],
                       model_version=rec["model_version"], prior=rec["champion_version"])
        out.append(row)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
