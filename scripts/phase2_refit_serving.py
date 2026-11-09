"""Refit the Phase 2 WINNING recipe on ALL clean data for serving (owner 2026-11-09, decision B).

The Phase 2 harness (src/retrain_harness) validated `champion_recipe` per track x market on the
approved holdout 2026-04-01..2026-11-08, fitting it only on data before 2026-04-01. This script fits
the SAME candidate code (retrain_harness.candidates.build('champion_recipe', ...).fit) on the same
committed frame (retrain_harness.protocol.load_committed_frame), the same eligible rows and the same
feature sets, on every eligible row dated up to the holdout end. Nothing about the recipe changes;
only the training window grows.

THE RECIPE'S OWN TAIL. src.model.train_repaired always keeps an untouched holdout block (>= 20 rows;
metrics only, nothing is fitted on it). Declared rule: that tail is the SHORTEST run of whole ISO
weeks ending on the holdout end date that holds at least 20 eligible rows. Every earlier row is
split base / calibration / meta (80/10/10 by whole dates), exactly as the harness's holdout stage.

Owner gate: refuses unless PHASE2_RETRAIN_ENABLED=true and the holdout is owner-approved
(retrain_harness.protocol.assert_fit_allowed), as the harness does. Writes ONLY to --out (outside
models/ and output/): one payload per track x market plus refit_manifest.json. Installing them is
src/promotion_gate's job.

    PHASE2_RETRAIN_ENABLED=true python scripts/phase2_refit_serving.py --out <dir> [--tracks standard,new_format]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrain_harness import candidates as C   # noqa: E402
from src.retrain_harness import protocol as P     # noqa: E402

MIN_TAIL_ROWS = 20          # src.model.train_repaired's own minimum block size


def tail_start(rows: pd.DataFrame, end: pd.Timestamp, min_rows: int = MIN_TAIL_ROWS) -> pd.Timestamp:
    """Monday of the earliest ISO week such that rows dated [that Monday, end] number >= min_rows."""
    d = pd.to_datetime(rows["date"]).dt.normalize()
    d = d[d <= end]
    monday = (end - pd.Timedelta(days=end.weekday())).normalize()
    first = d.min()
    while monday >= first - pd.Timedelta(days=7):
        if int((d >= monday).sum()) >= min_rows:
            return monday
        monday -= pd.Timedelta(days=7)
    raise ValueError(f"fewer than {min_rows} eligible rows in total")


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def refit(cfg: dict, feat: pd.DataFrame, out: Path, tracks: list[str], markets: list[str] | None = None) -> dict:
    P.assert_fit_allowed(cfg)
    out.mkdir(parents=True, exist_ok=True)
    _, end = P.holdout_window(cfg)
    rec = {"refit_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_sha": _code_sha(),
           "config_sha256": P.config_sha(cfg), "data_end": str(end.date()),
           "candidate": "champion_recipe", "tail_rule": __doc__.split("THE RECIPE'S OWN TAIL.")[1].split("\n\n")[0].strip(),
           "models": {}}
    for track in tracks:
        feats = P.feature_sets(cfg, track, feat.columns)[""][0]
        for market in markets or list(cfg["markets"]):
            rows = P.eligible_rows(feat, cfg, track, market)
            rows = rows[pd.to_datetime(rows["date"]).dt.normalize() <= end]
            if rows.empty:
                rec["models"][f"{track}/{market}"] = {"status": "no eligible rows"}
                continue
            hs = tail_start(rows, end)
            d = pd.to_datetime(rows["date"]).dt.normalize()
            frames = {"train": rows[d < hs], "test": rows[d >= hs]}
            cand = C.build("champion_recipe", cfg, track, market, feats)
            cand.fit(frames)
            payload = cand.payload
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            f = out / track / f"{market}.pkl"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(raw)
            rec["models"][f"{track}/{market}"] = {
                "status": "fitted", "file": str(f.relative_to(out)), "sha256": hashlib.sha256(raw).hexdigest(),
                "recipe": payload.get("recipe"), "rows": int(len(rows)),
                "train_dates": [str(d[d < hs].min().date()), str(d[d < hs].max().date())],
                "tail": [str(hs.date()), str(end.date())], "tail_rows": int((d >= hs).sum()),
                "blocks": payload.get("blocks"), "n_features": len(payload.get("feature_cols") or []),
                "dropped_all_missing_features": payload.get("dropped_all_missing_features"),
                "served_tail_metrics": payload.get("served_holdout_metrics"),
                "notes": list(cand.notes)}
            print(track, market, rec["models"][f"{track}/{market}"]["train_dates"],
                  rec["models"][f"{track}/{market}"]["tail"], flush=True)
    (out / "refit_manifest.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--markets", default="")
    a = ap.parse_args(argv)
    cfg = P.load_config()
    tracks = [t for t in a.tracks.split(",") if t] or list(cfg["tracks"])
    markets = [m for m in a.markets.split(",") if m] or None
    P.assert_fit_allowed(cfg)
    out = P.assert_run_folder(Path(a.out))
    feat, info = P.load_committed_frame(cfg)
    rec = refit(cfg, feat, out, tracks, markets)
    rec["data"] = {k: v for k, v in (info or {}).items() if k in ("loaded_rows", "frame_hash")}
    (out / "refit_manifest.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
