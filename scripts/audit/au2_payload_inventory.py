"""AU2 forensic audit - payload inventory (Phase 2B/2C). READ-ONLY.

For every serving model artifact committed on the audited commit, record: path, sha256, recipe,
feature list (count, order hash), persisted imputer (yes/no), base models, meta blender, the
StandardScaler training means/sds inside the LR pipeline (a training-distribution fingerprint that
needs no retraining), and every provenance key the payload carries.

    python scripts/audit/au2_payload_inventory.py  ->  reports/au2_data/payload_inventory.json
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import config  # noqa: E402

OUT = ROOT / "reports" / "au2_data" / "payload_inventory.json"

TEAM = {
    "standard_legacy": config.MODEL_FILE_STANDARD,
    "standard_repaired": config.MODEL_FILE_STANDARD_REPAIRED,
    "newformat": config.MODEL_FILE_NEWFORMAT,
    "btts": config.MODEL_FILE_BTTS,
    "over15": config.MODEL_FILE_OVER15,
    "over35": config.MODEL_FILE_OVER35,
    "ht_over05": config.HT_MODEL_FILE_05,
    "ht_over15": config.HT_MODEL_FILE_15,
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _scaler(model):
    """Return (mean_, scale_) of the first StandardScaler found inside a calibrated wrapper."""
    pipe = getattr(model, "_pipe", None) or getattr(model, "pipe", None) or model
    steps = getattr(pipe, "named_steps", None)
    if not steps:
        return None
    for st in steps.values():
        if hasattr(st, "mean_") and hasattr(st, "scale_"):
            return np.asarray(st.mean_, float), np.asarray(st.scale_, float)
    return None


def describe(name: str, path: Path) -> dict:
    rec = {"name": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
           "exists": path.exists()}
    if not path.exists():
        return rec
    rec["sha256"] = _sha(path)
    rec["bytes"] = path.stat().st_size
    with open(path, "rb") as f:
        p = pickle.load(f)
    if not isinstance(p, dict):
        rec["payload_type"] = type(p).__name__
        return rec
    fc = list(p.get("feature_cols") or [])
    rec["keys"] = sorted(map(str, p.keys()))
    rec["recipe"] = p.get("recipe", "<absent = legacy>")
    rec["n_features"] = len(fc)
    rec["feature_cols"] = fc
    rec["feature_order_sha"] = hashlib.sha256("|".join(fc).encode()).hexdigest()[:16]
    rec["has_persisted_imputer"] = bool(p.get("imputer"))
    models = p.get("models") or {}
    rec["base_models"] = [k for k in models if k != "__meta__"]
    rec["has_meta"] = "__meta__" in models
    rec["model_classes"] = {k: type(v).__name__ for k, v in models.items()}
    for k in ("trained_at", "training_cutoff", "train_rows", "model_version", "retrain_batch_id",
              "target", "metrics", "sklearn_version", "data_fingerprint", "schema_hash",
              "holdout_start", "provenance", "research"):
        if k in p:
            v = p[k]
            try:
                json.dumps(v)
                rec[k] = v
            except TypeError:
                rec[k] = repr(v)[:400]
    sc = None
    for k, m in models.items():
        if k == "__meta__":
            continue
        sc = _scaler(m)
        if sc is not None:
            rec["scaler_from"] = k
            break
    if sc is not None and len(sc[0]) == len(fc):
        rec["train_scaler"] = {c: {"mean": round(float(mu), 6), "sd": round(float(sd), 6)}
                               for c, mu, sd in zip(fc, sc[0], sc[1])}
        rec["constant_in_training"] = [c for c, sd in zip(fc, sc[1]) if sd <= 1e-12 or sd == 1.0 and False]
    elif sc is not None:
        rec["scaler_len_mismatch"] = [len(sc[0]), len(fc)]
    return rec


def describe_player(market: str, path: Path) -> dict:
    rec = {"name": f"player_{market}", "path": str(path.relative_to(ROOT)).replace("\\", "/"),
           "exists": path.exists()}
    if not path.exists():
        return rec
    rec["sha256"] = _sha(path)
    with open(path, "rb") as f:
        p = pickle.load(f)
    if not isinstance(p, dict):
        rec["payload_type"] = type(p).__name__
        return rec
    rec["keys"] = sorted(map(str, p.keys()))
    fc = list(p.get("feature_cols") or p.get("features") or [])
    rec["n_features"] = len(fc)
    rec["feature_cols"] = fc
    rec["feature_order_sha"] = hashlib.sha256("|".join(fc).encode()).hexdigest()[:16]
    rec["has_persisted_imputer"] = bool(p.get("imputer"))
    rec["recipe"] = p.get("recipe", "<absent = legacy>")
    for k in ("trained_at", "training_cutoff", "train_rows", "model_version", "target", "market",
              "threshold", "metrics", "props_contract", "feature_contract_sha"):
        if k in p:
            v = p[k]
            try:
                json.dumps(v)
                rec[k] = v
            except TypeError:
                rec[k] = repr(v)[:400]
    models = p.get("models") or {}
    rec["model_classes"] = {k: type(v).__name__ for k, v in models.items()} if isinstance(models, dict) else type(models).__name__
    return rec


def main() -> int:
    from player_model import config as pcfg
    out = {"audited_commit": "fb0e9697ec8556d4dd3df57f31f8ed5415a5012d", "team": {}, "player": {}}
    for n, p in TEAM.items():
        out["team"][n] = describe(n, Path(p))
    files = getattr(pcfg, "MODEL_FILES", None) or {}
    for m, p in files.items():
        out["player"][m] = describe_player(m, Path(p))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for n, r in out["team"].items():
        print(n, r.get("recipe"), r.get("n_features"), "imputer", r.get("has_persisted_imputer"),
              "meta", r.get("has_meta"), r.get("feature_order_sha"), (r.get("sha256") or "")[:10])
    for n, r in out["player"].items():
        print("player", n, r.get("n_features"), "imputer", r.get("has_persisted_imputer"),
              r.get("feature_order_sha"), r.get("keys"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
