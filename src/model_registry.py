"""Model identity, champion pointer and lineage records. CHAMPION HIGH-WATER MARK (owner 2026-10-18).

    from src import model_registry as reg
    reg.stamp_for("standard")          # -> {model_version, retrain_batch_id, training_cutoff, ...}
    reg.record_decision(...)           # every retrain verdict, PROMOTE / KEEP_CHAMPION / SHADOW_MORE
    # champion changes (promotion, rollback, bootstrap) live ONLY in src/promotion_gate.py

────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
────────────────────────────────────────────────────────────────────────────────────────────
Until now a retrain overwrote `models/model_v9_<x>.pkl` in place and nothing recorded which
artifact produced which prediction. `provenance.model_sha` hashed size+mtime over ALL .pkl files
together, and `git checkout` resets mtime on every CI run — so it changed every run (95 distinct
values against 4 real model changes) and could not name a model. The question "the week-33 model
produced these 84 bets; how did they do?" had no answer.

IDENTITY = THE BYTES. A model version is `<model_type>_<trained_at>_<sha256[:8]>` and is looked up
from the SHA-256 of the file actually on disk. Predict never trusts the pointer alone: if the file
it loaded is not the one the pointer names, the row is stamped `UNREGISTERED_<type>_<sha12>` — a
real identity, never an invented one.

LAYOUT (everything except the serving file is append-only):

    models/model_v9_<x>.pkl                       the serving file predict loads. Unchanged path.
    models/registry/current_champions.json        champion pointer per model type
    models/registry/<type>/versions/<v>.json      immutable manifest, one per trained version
    models/registry/<type>/champions/<v>.pkl.gz   immutable bytes of every version that served
    models/registry/<type>/challenger.pkl.gz      the one SHADOW_MORE challenger being shadowed
    output/model_lineage/champion_history.csv     append-only, one row per champion era
    output/model_lineage/retrain_batches.csv      append-only, one row per retrain decision
    output/model_lineage/high_water_mark.json     the bar each model type must beat
    output/model_lineage/shadow_h2h.csv           champion vs challenger on the same future fixtures

Nothing here may break predict. Every predict-side function returns a usable value on any error.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

log = logging.getLogger(__name__)

#: Every independent model family. One champion each — improvement in one never moves another's
#: bar. Labels are the ones `pipeline._train_one` already uses in retrain_log.json.
MODEL_FILES: dict[str, Path] = {
    "standard": config.MODEL_FILE_STANDARD,
    "newformat": config.MODEL_FILE_NEWFORMAT,
    "btts": config.MODEL_FILE_BTTS,
    "over15": config.MODEL_FILE_OVER15,
    "over35": config.MODEL_FILE_OVER35,
    "ht_over05": config.HT_MODEL_FILE_05,
    "ht_over15": config.HT_MODEL_FILE_15,
    # Standard O/U 2.5 on the repaired recipe (owner 2026-10-28). Its own model type, champion and
    # serving file, so the legacy "standard" champion stays intact as the rollback. Installed only
    # by promotion_gate.owner_install; FROZEN (see FROZEN_MODEL_TYPES) — no retrain submits to it.
    "standard_repaired": config.MODEL_FILE_STANDARD_REPAIRED,
}
#: What each model's probability column is called in predictions.csv.
PRED_COLUMN = {"standard": "p_over25", "newformat": "p_over25", "btts": "p_btts",
               "over15": "p_over15", "over35": "p_over35",
               "ht_over05": "p_ht_over05", "ht_over15": "p_ht_over15",
               "standard_repaired": "p_over25"}
TARGET = {"standard": "over25", "newformat": "over25", "btts": "btts", "over15": "over15",
          "over35": "over35", "ht_over05": "ht_over05", "ht_over15": "ht_over15",
          "standard_repaired": "over25"}
#: Model types no scheduled retrain may submit a candidate to (promotion_gate refuses them). The
#: repaired Standard model changes only by an explicit owner_install (owner decision 2026-10-28:
#: frozen until the owner decides how it is retrained).
FROZEN_MODEL_TYPES = frozenset({"standard_repaired"})

REGISTRY_DIR = config.MODELS_DIR / "registry"
LINEAGE_DIR = config.OUTPUT_DIR / "model_lineage"
POINTER_FILE = REGISTRY_DIR / "current_champions.json"
CHAMPION_HISTORY = LINEAGE_DIR / "champion_history.csv"
RETRAIN_BATCHES = LINEAGE_DIR / "retrain_batches.csv"
HIGH_WATER = LINEAGE_DIR / "high_water_mark.json"

UNKNOWN_LEGACY = "UNKNOWN_LEGACY"
ROLES = ("CHAMPION", "CHALLENGER", "REJECTED", "SHADOW", "ARCHIVED")

CHAMPION_HISTORY_COLS = [
    "model_type", "champion_version", "start_at", "end_at", "promotion_retrain_batch",
    "prior_champion", "holdout_n", "holdout_logloss", "holdout_brier", "holdout_ece",
    "fresh_oos_logloss", "promotion_source", "promotion_reason", "artifact_sha256",
    "git_commit", "reconstructed",
]
RETRAIN_BATCH_COLS = [
    "retrain_batch_id", "decided_at", "model_type", "challenger_version",
    "champion_at_training_time", "verdict", "promoted", "champion_after", "training_rows",
    "training_start", "training_end", "training_cutoff", "holdout_n", "ll_champion",
    "ll_challenger", "ll_delta", "ll_ci_lo", "ll_ci_hi", "brier_delta", "ece_delta",
    "later_half_ll_delta", "league_regressions", "safe_to_run", "better_than_champion",
    "comparison_basis", "reason", "legacy_rule_would_promote", "source",
]


# ── small utilities ───────────────────────────────────────────────────────────────────────────
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_version(model_type: str, trained_at: datetime, artifact_sha: str) -> str:
    """`standard_20261018T080423Z_1a2b3c4d`. Deterministic in its inputs, unique per artifact."""
    return f"{model_type}_{trained_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}_{artifact_sha[:8]}"


def batch_id(started_at: datetime) -> str:
    """One id per retrain RUN, shared by every model trained in it."""
    return f"rb_{started_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write-then-rename, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write_bytes(path, json.dumps(obj, indent=2, default=str).encode("utf-8"))


def _append_csv(path: Path, cols: list[str], row: dict) -> None:
    """APPEND-ONLY. Never rewrites an existing line; a new column set would be a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in cols})


# ── manifests ─────────────────────────────────────────────────────────────────────────────────
def manifest_path(model_type: str, version: str) -> Path:
    return REGISTRY_DIR / model_type / "versions" / f"{version}.json"


def write_manifest(manifest: dict) -> Path:
    """Write a version's manifest. IMMUTABLE: an existing manifest is never changed.

    Re-writing identical identity fields is a no-op (a re-run of the same step); anything else
    raises, because two different artifacts claiming one version would make every attribution
    built on it meaningless.
    """
    p = manifest_path(manifest["model_type"], manifest["model_version"])
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("artifact_sha256") != manifest.get("artifact_sha256"):
            raise ValueError(f"manifest {p.name} exists with a different artifact — refusing "
                             f"to overwrite an immutable version")
        return p
    _atomic_write_json(p, manifest)
    return p


def read_manifest(model_type: str, version: str) -> dict | None:
    p = manifest_path(model_type, version)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def all_manifests() -> list[dict]:
    out = []
    for p in sorted(REGISTRY_DIR.glob("*/versions/*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


# ── champion pointer ──────────────────────────────────────────────────────────────────────────
def read_pointer() -> dict:
    try:
        return json.loads(POINTER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def champion_artifact_path(model_type: str, version: str) -> Path:
    return REGISTRY_DIR / model_type / "champions" / f"{version}.pkl.gz"


def store_champion_artifact(model_type: str, version: str, raw: bytes) -> Path:
    """Keep the exact bytes of every version that ever served. Never overwritten."""
    p = champion_artifact_path(model_type, version)
    if p.exists():
        if sha256_bytes(gzip.decompress(p.read_bytes())) != sha256_bytes(raw):
            raise ValueError(f"{p.name} exists with different bytes — immutable, refusing")
        return p
    _atomic_write_bytes(p, gzip.compress(raw, 6))
    return p


# set_champion / rollback LIVE IN src/promotion_gate.py (owner 2026-10-25): the canonical gate is
# the ONLY writer of the serving file and of current_champions.json. This module keeps identity,
# manifests, immutable artifacts and the challenger slot; it never changes which model serves.


# ── challenger slot (true head-to-head shadow) ────────────────────────────────────────────────
def challenger_path(model_type: str) -> Path:
    return REGISTRY_DIR / model_type / "challenger.pkl.gz"


def challenger_meta_path(model_type: str) -> Path:
    return REGISTRY_DIR / model_type / "challenger.json"


def set_challenger(model_type: str, manifest: dict, raw: bytes) -> None:
    """Park a SHADOW_MORE challenger where predict can score it without serving it.

    The slot never touches the serving file. One slot per model type: the latest SHADOW_MORE
    challenger replaces the previous one (its manifest and metrics stay in the lineage).
    """
    _atomic_write_bytes(challenger_path(model_type), gzip.compress(raw, 6))
    _atomic_write_json(challenger_meta_path(model_type),
                       {**manifest, "model_role": "SHADOW", "shadow_since": iso(utcnow())})


def clear_challenger(model_type: str) -> None:
    for p in (challenger_path(model_type), challenger_meta_path(model_type)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def load_challenger(model_type: str) -> tuple[dict, dict] | None:
    """(payload, meta) of the shadowed challenger, or None. Verifies bytes against the meta."""
    p, m = challenger_path(model_type), challenger_meta_path(model_type)
    if not (p.exists() and m.exists()):
        return None
    meta = json.loads(m.read_text(encoding="utf-8"))
    raw = gzip.decompress(p.read_bytes())
    if sha256_bytes(raw) != meta.get("artifact_sha256"):
        log.warning(f"[registry] {model_type} challenger bytes do not match its manifest — ignored")
        return None
    return pickle.loads(raw), meta


# ── predict-side attribution ──────────────────────────────────────────────────────────────────
_version_cache: dict[str, dict] = {}


def _index_by_sha() -> dict[str, dict]:
    return {m.get("artifact_sha256"): m for m in all_manifests() if m.get("artifact_sha256")}


def stamp_for(model_type: str, path: Path | None = None) -> dict:
    """The identity of the model file predict actually loaded for `model_type`.

    Cached per process (predict loads each model once). NEVER RAISES: any failure returns an
    honest UNKNOWN rather than a guess, because a crashed predict sends no tips at all.
    """
    if model_type in _version_cache:
        return _version_cache[model_type]
    out = {"model_version": UNKNOWN_LEGACY, "retrain_batch_id": "", "training_cutoff": "",
           "prediction_role": "CHAMPION"}
    try:
        f = Path(path) if path else MODEL_FILES[model_type]
        if f.exists():
            sha = sha256_file(f)
            man = _index_by_sha().get(sha)
            if man:
                out.update(model_version=man["model_version"],
                           retrain_batch_id=man.get("retrain_batch_id", ""),
                           training_cutoff=man.get("training_cutoff", ""))
            else:
                out["model_version"] = f"UNREGISTERED_{model_type}_{sha[:12]}"
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[registry] could not identify {model_type} ({e}) — stamped UNKNOWN")
    _version_cache[model_type] = out
    return out


def ou_model_for(model_type_tag: str) -> str:
    """Which O/U model scored a row. Mirrors src/predict.py:332-348 exactly, INCLUDING its
    fallbacks: a new-format row is scored by the standard model when the new-format file is
    missing, and an `unknown` league row is always scored by the standard model."""
    if str(model_type_tag) == "new_format" and MODEL_FILES["newformat"].exists():
        return "newformat"
    return "standard"


def ou_model_for_row(model_type_tag, serving_recipe=None) -> str:
    """`ou_model_for` that also honours the Standard O/U 2.5 serving switch: a row predict stamped
    serving_recipe == "v9_repaired" was scored by the "standard_repaired" model."""
    if str(serving_recipe) == "v9_repaired" and str(model_type_tag) == "standard":
        return "standard_repaired"
    return ou_model_for(model_type_tag)


def market_model(market: str) -> str | None:
    m = str(market).strip().lower()
    return m if m in MODEL_FILES else None


# ── retrain decisions and the high-water mark ─────────────────────────────────────────────────
def record_decision(row: dict) -> None:
    _append_csv(RETRAIN_BATCHES, RETRAIN_BATCH_COLS, row)


def update_high_water(model_type: str, entry: dict) -> None:
    """The bar a challenger must beat, per model type.

    Two benchmarks, deliberately (owner brief §34/§55):
      historical — the champion's validated numbers on the holdout that promoted it. Fixed for
                   the champion's life; a promotion is what raises it.
      current    — the same champion re-scored on the LATEST holdout at the latest retrain.
    The gate always decides on the CURRENT comparison (same rows, same labels), because two log
    losses on different holdouts are not comparable — on 2026-09-23 a larger holdout alone moved
    standard 0.68885 -> 0.64066 with no change in the model.
    """
    try:
        hw = json.loads(HIGH_WATER.read_text(encoding="utf-8"))
    except Exception:
        hw = {}
    hw[model_type] = {**(hw.get(model_type) or {}), **entry, "updated_at": iso(utcnow())}
    _atomic_write_json(HIGH_WATER, hw)


def payload_bytes(results: dict, target: str = "over25", extra: dict | None = None) -> bytes:
    """Byte-identical to what `src.model.save_models` writes, so the serving file's format and
    every reader of it are unchanged. (save_models stores target='over25' for side models too —
    kept as-is; changing a payload field is not this task.)

    FIXED 2026-10-28: this used to rebuild the dict from four keys and so DROPPED everything a
    v9_repaired result carries (`imputer`, `recipe`, block/version metadata) — a repaired model
    installed through it would have served with batch-median imputation, the very defect the
    recipe removes. It now goes through `src.model.build_payload`, which is the legacy dict
    exactly for legacy results (same keys, same order -> same pickle bytes) and adds the recipe
    metadata for repaired ones. `extra` adds payload-level provenance (e.g. `research`, the frozen
    `identity_alias_map`)."""
    from src.model import build_payload
    payload = build_payload(results, target)
    if extra:
        payload.update(extra)
    return pickle.dumps(payload)


def write_metrics_file(model_type: str, results: dict) -> None:
    """Same file save_models writes beside the serving model; only on promotion."""
    stem = MODEL_FILES[model_type].stem
    _atomic_write_bytes(config.MODELS_DIR / f"metrics_{stem}.json",
                        json.dumps({k: v["metrics"] for k, v in results.items()},
                                   indent=2).encode("utf-8"))
