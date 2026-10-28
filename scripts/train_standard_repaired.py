"""Train (and, with --install, owner-install) the REPAIRED Standard O/U 2.5 serving model.

    python scripts/train_standard_repaired.py --out _work                      # train only
    python scripts/train_standard_repaired.py --out _work --install \\
        --approved-by "Nevo (owner)" --approval "<verbatim owner approval>"     # + install

Owner decision 2026-10-28 ("switch Standard now"). Run by .github/workflows/
train_standard_repaired.yml — NOT on the laptop (training is CI-only).

RECIPE = the arm the Pro benchmark measured (V9_REPAIRED, price_mode "none", league-scoped team
stats, derived date weights, four-block chronological stacking, persisted train-only imputer):

  1. v9's own loader (src.data_loader.load_all_matches via src.standard_repair.load_history, COVID
     seasons excluded as mode_train does), rows dated STRICTLY BEFORE the cutoff (default: today
     UTC) — the cutoff is applied BEFORE identity derivation, so no post-cutoff row informs even
     the alias map;
  2. identity canonicalisation: the explicit alias map committed at
     config/standard_ou25_identity_aliases.json (copied byte-for-byte from Pro
     output/standard_repair/identity_aliases.json, identity/1.0) + evidence-derived aliases; the
     MERGED map is frozen into the payload (`identity_alias_map`) and serving applies exactly that
     map with derivation off;
  3. src.features_repaired as-of features -> Standard track rows -> src.model.train(recipe=
     "v9_repaired");
  4. payload bytes via src.model_registry.payload_bytes (which now keeps `imputer` + recipe
     metadata) plus `research` provenance and the frozen alias map;
  5. --install: src.promotion_gate.owner_install("standard_repaired", ...) — the ONLY writer of
     models/model_v9_standard_repaired.pkl and of the pointer entry. models/model_v9_standard.pkl
     (the legacy champion, the rollback) is never opened for writing.

Nothing here reads or writes predict.yml, thresholds, tiers or any other model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402

log = logging.getLogger("train_standard_repaired")
ALIAS_MAP = ROOT / "config" / "standard_ou25_identity_aliases.json"
MODEL_TYPE = "standard_repaired"


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _commit() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def train(cutoff: pd.Timestamp, alias_map: Path, raw: pd.DataFrame | None = None) -> tuple[bytes, dict]:
    """Return (payload bytes, training record). `raw` = injected history (tests); default loads
    v9's own history."""
    import sklearn

    from src import model_registry as reg
    from src.features_repaired import FEATURE_RECIPE_VERSION
    from src.identity_repair import IDENTITY_VERSION
    from src.imputer import IMPUTER_VERSION
    from src.model import MODEL_RECIPE_VERSION, train as model_train
    from src.standard_repair import (derived_weights, feature_cols, feature_frame, frame_hash,
                                     load_history, prepare_history, track_rows)

    raw = load_history() if raw is None else raw
    raw = raw[pd.to_datetime(raw["date"], errors="coerce").dt.normalize() < cutoff]
    hist, idrep = prepare_history(raw, "v9_repaired", str(alias_map), True)
    frozen_map = {lg: dict(m) for lg, m in (idrep.get("aliases") or {}).items()}
    feat = feature_frame(hist, "v9_repaired", "none", None, "league")
    if pd.to_datetime(feat["date"]).max() >= cutoff:
        raise SystemExit("integrity: a training row is on/after the cutoff")
    rows = track_rows(feat, "standard", "over25")
    cols = feature_cols("v9_repaired", "none", "standard")
    w = derived_weights(rows)
    results = model_train(rows, target="over25", sample_weight=w, feature_cols=cols,
                          recipe="v9_repaired")
    d = pd.to_datetime(rows["date"])
    research = {
        "recipe": "v9_repaired", "arm": "V9_REPAIRED", "price_mode": "none", "track": "standard",
        "target": "over25", "team_scope": "league", "weights": "derived",
        "rows": int(len(rows)), "frame_hash": frame_hash(rows),
        "training_cutoff": str(cutoff.date()),
        "training_start": str(d.min().date()), "training_end": str(d.max().date()),
        "identity": f"{IDENTITY_VERSION}:explicit+derived", "identity_version": IDENTITY_VERSION,
        "alias_map_file": str(alias_map.relative_to(ROOT)) if alias_map.is_relative_to(ROOT) else str(alias_map),
        "alias_map_sha256": _sha256(alias_map),
        "identity_report": {k: v for k, v in idrep.items() if k not in ("aliases", "quarantine_rows")},
        "feature_recipe_version": FEATURE_RECIPE_VERSION, "imputer_version": IMPUTER_VERSION,
        "model_recipe_version": MODEL_RECIPE_VERSION, "v9_commit": _commit(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "python": platform.python_version(),
        "data_env": {k: os.getenv(k, "") for k in ("WOWZA_FULL_ENRICH", "WOWZA_USE_BACKTEST_ARCHIVE")},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw_bytes = reg.payload_bytes(results, "over25",
                                  extra={"research": research, "identity_alias_map": frozen_map})
    pay = pickle.loads(raw_bytes)
    if not pay.get("imputer") or pay.get("recipe") != "v9_repaired":
        raise SystemExit("integrity: payload bytes lost the imputer/recipe")
    record = {"trained_at": research["generated_at"], "training_cutoff": research["training_cutoff"],
              "training_start": research["training_start"], "training_end": research["training_end"],
              "training_rows": research["rows"], "code_commit_sha": research["v9_commit"][:12],
              "retrain_batch_id": f"owner_{research['generated_at'].replace('-', '').replace(':', '')}",
              "served_holdout_metrics": pay.get("served_holdout_metrics"),
              "blocks": pay.get("blocks"), "n_aliases": sum(len(m) for m in frozen_map.values())}
    return raw_bytes, record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="work dir for the payload copy + training record")
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD; rows strictly before (default today UTC)")
    ap.add_argument("--alias-map", default=str(ALIAS_MAP))
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--approved-by", default="")
    ap.add_argument("--approval", default="")
    ap.add_argument("--supersede", default="", help="exact current standard_repaired version to replace")
    ap.add_argument("--install-from", default="",
                    help="dir holding standard_repaired_over25.pkl + training_record.json of an EARLIER "
                         "(previewed) run: install exactly those bytes, no retraining")
    ap.add_argument("--expected-sha256", default="", help="with --install-from: refuse unless it matches")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from src import model_registry as reg
    if a.install_from:
        src_dir = Path(a.install_from)
        raw_bytes = (src_dir / "standard_repaired_over25.pkl").read_bytes()
        record = json.loads((src_dir / "training_record.json").read_text(encoding="utf-8"))
        sha = reg.sha256_bytes(raw_bytes)
        if sha != record.get("artifact_sha256") or (a.expected_sha256 and sha != a.expected_sha256.strip()):
            raise SystemExit(f"integrity: payload sha256 {sha} does not match the training record "
                             f"({record.get('artifact_sha256')}) / --expected-sha256 ({a.expected_sha256})")
        a.install = True
    else:
        cutoff = pd.Timestamp(a.cutoff) if a.cutoff else pd.Timestamp(datetime.now(timezone.utc).date())
        raw_bytes, record = train(cutoff, Path(a.alias_map))
        record["artifact_sha256"] = reg.sha256_bytes(raw_bytes)
    (out / "standard_repaired_over25.pkl").write_bytes(raw_bytes)
    if a.install:
        from src import promotion_gate as PG
        man = PG.owner_install(MODEL_TYPE, raw_bytes, approval=a.approval, approved_by=a.approved_by,
                               training=record, supersede=a.supersede)
        record["installed_version"] = man["model_version"]
        if reg.sha256_file(config.MODEL_FILE_STANDARD_REPAIRED) != record["artifact_sha256"]:
            raise SystemExit("integrity: installed serving file does not match the trained bytes")
    (out / "training_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
