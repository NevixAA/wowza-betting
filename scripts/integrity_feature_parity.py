"""Train/serve feature parity health for every serving model (R2-C finding 18). READ-ONLY inputs.

    python scripts/integrity_feature_parity.py            # writes the two health files below

    output/feature_parity_health.json      all tracks / all serving models
    output/nf_feature_parity_health.json   the new-format slice (NF OU2.5 + side models on NF rows)

TRAIN side: EXACTLY the trainers' rows (D3, owner 2026-11-23 "Yes, do all"). Every serving payload is
fitted by scripts/phase2_refit_serving.refit on src.retrain_harness.protocol.load_committed_frame
(load_all_matches on the committed banks, offline, every loader correction incl. the football-data
season files and the half-time fills, COVID seasons excluded) restricted to
protocol.eligible_rows(track, market) between the payload part's training_start and training_end.
This monitor builds the same frame through the same function and takes the same rows per payload
part (track x market), so a FAIL is a real train/serve difference, never a reference built another
way. (Until 2026-11-23 the reference was integrity_af_join_replay.base_history + the legacy
enrichment joins: no football-data season files, no HT fills, the retired v9_repaired recipe for
Standard O/U 2.5 - it reported FAIL on tracks the cell-level check passes, 0 of 2,666 cells.)
Statistics are unweighted (time-decay weights are not applied).

LIVE side: output/predictions.csv as committed (the latest predict board), split by model_type.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)


#: parity key -> (payload file, track part). Track-split payloads (BTTS / O1.5 / O3.5) hold the
#: Standard part at the top level and the New-Format part under tracks["new_format"].
def _payload_specs() -> dict:
    import config
    from src.integrity.model_health import serving_file
    m = {"standard|OU25": (config.MODELS_DIR / serving_file("standard|OU25", "model_v9_standard.pkl"), "standard"),
         "new_format|OU25": (Path(config.MODEL_FILE_NEWFORMAT), "new_format")}
    for t, f in config.SIDE_MARKETS.items():
        m[f"standard|{t}"] = (Path(f), "standard")
        m[f"new_format|{t}"] = (Path(f), "new_format")
    for nm in ("ht_over05", "ht_over15"):
        m[f"standard|{nm}"] = (config.MODELS_DIR / f"model_{nm}.pkl", "standard")
        m[f"new_format|{nm}"] = (config.MODELS_DIR / f"model_v9_{nm}_nf.pkl", "new_format")
    return m


def payload_part(pay: dict, track: str) -> dict:
    """The part of a payload that serves `track` (top level unless it is under tracks[track])."""
    tracks = pay.get("tracks") if isinstance(pay, dict) else None
    if isinstance(tracks, dict) and track in tracks:
        return tracks[track]
    return pay


def _models() -> dict:
    """parity key -> (payload file, part) for every serving payload part that exists."""
    import joblib
    out = {}
    for k, (f, track) in _payload_specs().items():
        if not f.exists():
            continue
        try:
            out[k] = (f, payload_part(joblib.load(f), track))
        except Exception as e:                                    # noqa: BLE001
            out[k] = (f, None)
            print(f"[parity] could not load {f.name}: {e}")
    return out


def train_frame(frame: str | None = None):
    """The trainers' feature frame: protocol.load_committed_frame (or a saved copy of it, --frame)."""
    import pandas as pd
    from src.retrain_harness import protocol as P
    cfg = P.load_config()
    if frame:
        feat = pd.read_parquet(frame)
        feat["date"] = pd.to_datetime(feat["date"])
        from src.standard_repair import frame_hash
        info = {"frame_file": frame, "frame_hash": frame_hash(feat)}
    else:
        feat, info = P.load_committed_frame(cfg)
    return feat, info, cfg


def training_rows(feat, cfg: dict, track: str, part: dict):
    """The rows the payload part was fitted on: protocol.eligible_rows for its market, dated
    training_start..training_end (the recipe's own tail after training_end is never fitted)."""
    import pandas as pd
    from src.retrain_harness import protocol as P
    target = part.get("target")
    market = next((m for m, spec in (cfg.get("markets") or {}).items() if spec.get("target") == target), None)
    if market is None:
        raise ValueError(f"no harness market for target {target!r}")
    rows = P.eligible_rows(feat, cfg, track, market)
    d = pd.to_datetime(rows["date"]).dt.normalize()
    if part.get("training_start"):
        rows = rows[d >= pd.Timestamp(part["training_start"])]
        d = d.loc[rows.index]
    if part.get("training_end"):
        rows = rows[d <= pd.Timestamp(part["training_end"])]
    return rows, market


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="output/predictions.csv")
    ap.add_argument("--out-all", default="output/feature_parity_health.json")
    ap.add_argument("--out-nf", default="output/nf_feature_parity_health.json")
    ap.add_argument("--frame", default=None, help="a saved copy of protocol.load_committed_frame (local "
                    "reproduction); default: build it, as the trainers do")
    a = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING)
    import pandas as pd
    from src.integrity.parity import parity_report, PARITY_RULE_VERSION

    board = pd.read_csv(ROOT / a.board, low_memory=False)
    live = {"standard": board[board["model_type"] == "standard"],
            "new_format": board[board["model_type"] == "new_format"]}
    feat, frame_info, cfg = train_frame(a.frame)
    models = _models()

    reports = {}
    for key, (f, part) in models.items():
        feats = (part or {}).get("feature_cols")
        if not feats:
            reports[key] = {"status": "UNKNOWN", "reason": f"payload {Path(f).name} unreadable"}
            continue
        track = key.split("|")[0]
        rows, market = training_rows(feat, cfg, track, part)
        served = [c for c in feats if not str(c).endswith("__missing")]
        r = parity_report(rows, live[track], served, track, Path(f).name)
        r["train_reference"] = {"market": market, "training_start": part.get("training_start"),
                                "training_end": part.get("training_end"),
                                "payload_training_rows": part.get("training_rows"),
                                "payload_frame_hash": part.get("frame_hash"),
                                "missing_indicators_not_compared": len(feats) - len(served)}
        reports[key] = r

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    board_meta = {"path": a.board, "rows": int(len(board)),
                  "generated_at": str(board["generated_at"].iloc[0]) if "generated_at" in board and len(board) else None,
                  "git_sha": str(board["git_sha"].iloc[0]) if "git_sha" in board and len(board) else None,
                  "rows_standard": int(len(live["standard"])), "rows_new_format": int(len(live["new_format"]))}

    def _overall(rs):
        st = [r.get("status") for r in rs.values()]
        return "FAIL" if "FAIL" in st else "WARN" if "WARN" in st else "PASS" if st else "UNKNOWN"

    common = {"generated_at": gen, "rule_version": PARITY_RULE_VERSION, "live_board": board_meta,
              "train_source": "the trainers' frame: src.retrain_harness.protocol.load_committed_frame "
                              "(offline, committed data, COVID seasons excluded), rows = "
                              "protocol.eligible_rows per payload part between its training_start and "
                              "training_end; unweighted",
              "train_frame": frame_info,
              "nf_feature_parity_flag": os.getenv("NF_FEATURE_PARITY", "1")}
    all_doc = {"artifact": "feature_parity_health", **common, "status": _overall(reports),
               "models": reports}
    nf = {k: v for k, v in reports.items() if k.startswith("new_format|")}
    nf_doc = {"artifact": "nf_feature_parity_health", **common, "status": _overall(nf), "models": nf}
    for path, doc in ((a.out_all, all_doc), (a.out_nf, nf_doc)):
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    for k, r in reports.items():
        bad = [f"{f}:{s['status']}" for f, s in r.get("features", {}).items() if s["status"] != "PASS"]
        print(f"{k:24s} {r.get('status'):5s} {r.get('counts')} {bad[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
