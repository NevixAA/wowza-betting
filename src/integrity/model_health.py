"""output/model_integrity_health.json — one record per serving track (R2 shared artifact).

SKELETON BY DESIGN (R2-C). Fields this workstream can know from committed artifacts are filled;
fields other R2 workstreams own (identity / season-resolver / league-registry versions, batch
invariance, weight alignment, holdout independence for the legacy recipes) are "UNKNOWN" until
their health files exist, and are picked up automatically when they do:
    output/identity_health.json, output/season_resolution_health.json,
    output/league_registry_health.json, output/player_props_integrity_health.json
An UNKNOWN field never counts as PASS: a track's status is the worst of its known checks, and a
track with any UNKNOWN required check is at best WARN.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HEALTH_VERSION = "model-integrity-health-0.1"
UNKNOWN = "UNKNOWN"
RANK = {"PASS": 0, "WARN": 1, "UNKNOWN": 1, "FAIL": 2}

#: track -> (model_type, market, payload file, predictions version column, p column)
TRACKS = {
    "standard|OU25": ("standard", "OU25", "model_v9_standard_repaired.pkl", "model_version", "p_over25"),
    "new_format|OU25": ("new_format", "OU25", "model_v9_newformat.pkl", "model_version", "p_over25"),
    "standard|BTTS": ("standard", "BTTS", "model_v9_btts.pkl", "model_version_btts", "p_btts"),
    "new_format|BTTS": ("new_format", "BTTS", "model_v9_btts.pkl", "model_version_btts", "p_btts"),
    "standard|OU15": ("standard", "OU15", "model_v9_over15.pkl", "model_version_over15", "p_over15"),
    "new_format|OU15": ("new_format", "OU15", "model_v9_over15.pkl", "model_version_over15", "p_over15"),
    "standard|OU35": ("standard", "OU35", "model_v9_over35.pkl", "model_version_over35", "p_over35"),
    "new_format|OU35": ("new_format", "OU35", "model_v9_over35.pkl", "model_version_over35", "p_over35"),
    "standard|HT_O05": ("standard", "HT_O05", "model_ht_over05.pkl", "model_version_ht_over05", "p_ht_over05"),
    "standard|HT_O15": ("standard", "HT_O15", "model_ht_over15.pkl", "model_version_ht_over15", "p_ht_over15"),
}
PARITY_KEY = {"standard|OU25": "standard|OU25", "new_format|OU25": "new_format|OU25",
              "standard|BTTS": "standard|btts", "new_format|BTTS": "new_format|btts",
              "standard|OU15": "standard|over15", "new_format|OU15": "new_format|over15",
              "standard|OU35": "standard|over35", "new_format|OU35": "new_format|over35",
              "standard|HT_O05": "standard|ht_over05", "standard|HT_O15": "standard|ht_over15"}


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


def _sha(p: Path) -> str:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except OSError:
        return UNKNOWN


def _price_provenance(track: str, payload_feats, prov: dict | None, side: dict | None) -> dict:
    from src.integrity.price_provenance import price_features_in
    pf = price_features_in(payload_feats or [])
    mt, mk = track.split("|")
    out = {"price_features_in_model": pf}
    if mk == "OU25" and mt == "standard":
        out.update(status="PASS" if not pf else "FAIL",
                   note="v9_repaired serves price_mode=none: football-only model, no market input")
    elif mk == "OU25":
        out.update(status="WARN", note="NF model carries bookmaker_overround: historical value is a "
                   "last-snapshot (near-closing) best-of-books price while live is the current board; "
                   "83% of NF training rows have it missing (median-imputed). Market info inside a "
                   "pre-match model (finding 15).")
    elif mk in ("BTTS", "OU15", "OU35"):
        syn = (prov or {}).get("synthetic_side_prices_in_backtests", {}).get(
            {"BTTS": "btts", "OU15": "over15", "OU35": "over35"}[mk], {})
        placed_syn = syn.get("placed_rows_synthetic")
        out.update(synthetic_share_historical=syn.get("synthetic_share"),
                   placed_rows_synthetic=placed_syn,
                   production_edge_rule="p - (1/odds)/1.08 (fixed 8% margin assumption)")
        if side:
            out["devig_shadow_ledger"] = side.get("ledger_retrospective", {}).get("by_market", {}).get(
                {"BTTS": "btts", "OU15": "over15", "OU35": "over35"}[mk])
        out["status"] = ("FAIL" if placed_syn else "WARN") if placed_syn is not None else UNKNOWN
        out["note"] = ("historical side prices mostly SYNTHETIC (diagnostic only, excluded from ROI); "
                       "live edge uses a fixed 1.08 margin, not a two-sided de-vig (shadow logged)")
    else:
        out.update(status=UNKNOWN, note="HT prices not audited by R2-C")
    if pf and mk != "OU25":
        out["status"] = "WARN" if out.get("status") == "PASS" else out.get("status")
    return out


def build(root: Path) -> dict:
    import pandas as pd
    out_dir = root / "output"
    models = root / "models"
    parity = _read(out_dir / "feature_parity_health.json") or {}
    prov = _read(out_dir / "price_provenance_health.json")
    side = _read(out_dir / "side_edge_devig_health.json")
    joins = _read(out_dir / "join_integrity_health.json") or {}
    ext = {n: _read(out_dir / f"{n}.json") for n in ("identity_health", "season_resolution_health",
                                                       "league_registry_health",
                                                       "player_props_integrity_health")}
    try:
        board = pd.read_csv(out_dir / "predictions.csv", low_memory=False)
    except Exception:                                              # noqa: BLE001
        board = pd.DataFrame()
    tracks = {}
    for key, (mt, mk, pfile, vcol, pcol) in TRACKS.items():
        pay, feats = None, None
        try:
            import joblib
            pay = joblib.load(models / pfile)
            feats = pay.get("feature_cols")
        except Exception:                                          # noqa: BLE001
            pass
        rows = board[board.get("model_type", pd.Series(dtype=str)) == mt] if len(board) else board
        ver = (rows[vcol].dropna().astype(str).value_counts().index[0]
               if len(rows) and vcol in rows and rows[vcol].notna().any() else UNKNOWN)
        repaired = bool(pay and pay.get("recipe") == "v9_repaired")
        research = (pay or {}).get("research", {}) if pay else {}
        par = (parity.get("models") or {}).get(PARITY_KEY[key], {})
        rec = {
            "model_type": mt, "market": mk, "payload": pfile, "payload_sha256_16": _sha(models / pfile),
            "recipe": (pay or {}).get("recipe", "legacy_v9") if pay else UNKNOWN,
            "model_recipe_version": (pay or {}).get("model_recipe_version", "legacy_v9 (unversioned)") if pay else UNKNOWN,
            "serving_version": ver,
            "training_cutoff": research.get("training_cutoff", UNKNOWN) if repaired else
                               (str(rows["training_cutoff"].dropna().iloc[0]) if len(rows) and "training_cutoff" in rows
                                and rows["training_cutoff"].notna().any() else UNKNOWN),
            "code_sha": (str(board["git_sha"].iloc[0]) if len(board) and "git_sha" in board else UNKNOWN),
            "data_hash": research.get("frame_hash", UNKNOWN) if repaired else UNKNOWN,
            "identity_version": research.get("identity_version", UNKNOWN) if repaired else UNKNOWN,
            "feature_recipe_version": research.get("feature_recipe_version", UNKNOWN) if repaired else "legacy_v9 (unversioned)",
            "imputer_version": (pay or {}).get("imputer_version", UNKNOWN) if repaired else
                               "batch-median at predict (src/model._prep: X.median())",
            "season_resolver_version": ((ext["season_resolution_health"] or {}).get("version", UNKNOWN)),
            "league_registry_version": ((ext["league_registry_health"] or {}).get("version", UNKNOWN)),
            "n_features": len(feats) if feats else UNKNOWN,
            "chronology": ({"status": "PASS", "blocks": pay.get("blocks")} if repaired and pay.get("blocks")
                           else {"status": UNKNOWN, "note": "legacy recipe: no persisted block boundaries"}),
            "parity": {"status": par.get("status", UNKNOWN), "counts": par.get("counts"),
                       "fail_features": [f for f, s in (par.get("features") or {}).items() if s.get("status") == "FAIL"]},
            "batch_invariance": {"status": "PASS" if repaired else "FAIL" if pay else UNKNOWN,
                                 "note": "train-only persisted imputer" if repaired else
                                 "legacy _prep fills NaN with the scoring batch's median"},
            "weight_alignment": {"status": UNKNOWN, "owner": "R2 model-recipe workstream"},
            "holdout_independence": ({"status": "PASS", "holdout": pay["blocks"].get("holdout")}
                                     if repaired and pay.get("blocks") else {"status": UNKNOWN}),
            "price_provenance": _price_provenance(key, feats, prov, side),
            "missing_feature_policy": "persisted train-only imputer" if repaired else "batch median (legacy)",
            "serving_fallback_policy": (
                "row-level fallback to legacy_v9 when repaired features are unavailable; invalid "
                "STANDARD_OU25_SERVING_RECIPE silently becomes legacy_v9 (finding 17: fails OPEN)"
                if key == "standard|OU25" else
                "unknown leagues are scored by the STANDARD payload (finding 8: fails OPEN)"
                if mt == "standard" else
                "missing NF payload falls back to the standard payload (src/predict.py:332)"),
            "join_integrity": {"status": joins.get("status", UNKNOWN),
                               "af_history_join_mode": joins.get("serving_join_mode", UNKNOWN)},
        }
        checks = [rec["chronology"]["status"], rec["parity"]["status"], rec["batch_invariance"]["status"],
                  rec["weight_alignment"]["status"], rec["holdout_independence"]["status"],
                  rec["price_provenance"].get("status", UNKNOWN), rec["join_integrity"]["status"]]
        if "fails OPEN" in rec["serving_fallback_policy"] or "falls back" in rec["serving_fallback_policy"]:
            checks.append("WARN")
        worst = max(checks, key=lambda s: RANK.get(s, 1))
        rec["status"] = "FAIL" if worst == "FAIL" else "WARN" if worst in ("WARN", UNKNOWN) else "PASS"
        rec["unknown_checks"] = [n for n, s in zip(
            ["chronology", "parity", "batch_invariance", "weight_alignment", "holdout_independence",
             "price_provenance", "join_integrity"], checks) if s == UNKNOWN]
        tracks[key] = rec
    props = ext["player_props_integrity_health"]
    tracks["props|ALL"] = {"model_type": "props", "market": "ALL",
                           "status": (props or {}).get("status", UNKNOWN),
                           "note": "owned by the props workstream (player_props_integrity_health.json)"}
    statuses = [t["status"] for t in tracks.values()]
    return {"artifact": "model_integrity_health", "version": HEALTH_VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "FAIL" if "FAIL" in statuses else "WARN" if any(s != "PASS" for s in statuses) else "PASS",
            "inputs_present": {"feature_parity_health": bool(parity), "price_provenance_health": prov is not None,
                               "side_edge_devig_health": side is not None, "join_integrity_health": bool(joins),
                               **{k: v is not None for k, v in ext.items()}},
            "tracks": tracks}


def write(root: Path, path: Path | None = None) -> dict:
    doc = build(root)
    p = path or (root / "output" / "model_integrity_health.json")
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return doc
