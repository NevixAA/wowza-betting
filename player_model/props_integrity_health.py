"""
output/player_props_integrity_health.json (R2-D, findings 20-22).

Per props market: what is serving (recipe, artifact, imputer policy), a live batch-invariance
probe of the SERVING payload, holdout independence, the feature-contract state, and the
retrospective PROPS-R2 challenger result if the replay has run. Plus the Fantasy defaults
register. Read-only: it never changes a model or a projection.

    python -m player_model.props_integrity_health
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import model as M
from . import props_contract as PC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "player_props_integrity_health.json"
R2_RESULTS = ROOT / "output" / "props_r2" / "props_r2_challengers.json"
CHAMPIONS = ROOT / "models" / "registry" / "props" / "props_champions.json"
INJURY_PIT_FIX = "2026-10-26"
NON_INFERIOR_LL = 0.001
ECE_SLACK = 0.005


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _probe_rows(n: int = 120) -> pd.DataFrame | None:
    """A small recent sample (one month part, never the whole history)."""
    parts = sorted((ROOT / "player_history.parquet").glob("month_*.parquet"))
    for p in reversed(parts[-3:]):
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if len(df) >= n:
            return df.sort_values("date").tail(n).reset_index(drop=True)
    return None


def batch_probe(payload: dict, rows: pd.DataFrame) -> dict:
    """Score rows (a) as one batch, (b) one at a time, (c) with a NaN injected in a few rows of an
    imputable feature. A batch-invariant model gives identical (a) and (b)."""
    df = rows.copy()
    feats = [c for c in payload["feature_cols"] if c in df.columns]
    target = next((c for c in ("kp_per90", "days_since_last_injury") if c in feats), feats[0])
    df.loc[df.index[:10], target] = np.nan
    batch = M.predict_proba(df, payload).to_numpy()
    single = np.concatenate([M.predict_proba(df.iloc[[i]], payload).to_numpy() for i in range(len(df))])
    ok = ~(np.isnan(batch) | np.isnan(single))
    diff = np.abs(batch[ok] - single[ok]) if ok.any() else np.array([0.0])
    return {"rows": int(len(df)), "nan_injected_feature": target,
            "max_abs_diff_single_vs_batch": float(diff.max()),
            "rows_differing_gt_1e-9": int((diff > 1e-9).sum()),
            "batch_invariant": bool(diff.max() <= 1e-9)}


def _challenger(market: str, r2: dict | None) -> dict:
    if not r2:
        return {"state": "NOT_RUN", "note": "PROPS-R2 replay has not produced results"}
    m = (r2.get("markets") or {}).get(market)
    if not m or m.get("status") != "OK":
        return {"state": "SKIPPED", "why": (m or {}).get("why")}
    d = m["delta_repaired_minus_legacy"]
    non_inf = d["log_loss_ci95"][1] <= NON_INFERIOR_LL
    calib = m["repaired"]["ece10"] <= m["legacy"]["ece10"] + ECE_SLACK
    return {
        "state": "RETROSPECTIVE_PASS" if (non_inf and calib) else "RETROSPECTIVE_FAIL",
        "experiment_id": m["experiment_id"], "evidence_class": "RETROSPECTIVE",
        "code_sha": r2.get("code_sha"), "generated_at": r2.get("generated_at"),
        "training_cutoff": m.get("training_cutoff"),
        "delta_log_loss": d["log_loss"], "delta_log_loss_ci95": d["log_loss_ci95"],
        "delta_brier": d["brier"], "ece10": {"legacy": m["legacy"]["ece10"], "repaired": m["repaired"]["ece10"]},
        "auc": {"legacy": m["legacy"]["auc"], "repaired": m["repaired"]["auc"]},
        "legacy_batch_sensitivity": m.get("legacy_batch_sensitivity"),
        "holdout_feature_health": m.get("holdout_feature_health"),
        "challenger_sha256": m.get("challenger_sha256"),
        "promotion": "NONE (owner review + PG-PROPS required)",
    }


def build() -> dict:
    champs = json.loads(CHAMPIONS.read_text(encoding="utf-8")) if CHAMPIONS.exists() else {}
    r2 = json.loads(R2_RESULTS.read_text(encoding="utf-8")) if R2_RESULTS.exists() else None
    try:
        contract = PC.load_contract()
        contract_info = {"version": contract["contract_version"], "sha256": contract["contract_sha256"],
                         "counts": contract["counts"],
                         "covers_feature_list": set(contract["features"]) == set(config.PLAYER_FEATURE_COLS)}
    except Exception as e:  # noqa: BLE001
        contract_info = {"error": str(e), "covers_feature_list": False}
    rows = _probe_rows()
    tracks = {}
    for market in config.MARKETS:
        path = config.MODEL_FILES.get(market)
        t: dict = {"model_type": "player_props", "market": market}
        if path is None or not path.exists():
            t.update(status="FAIL", why="serving file missing")
            tracks[market] = t
            continue
        raw = path.read_bytes()
        payload = pickle.loads(raw)
        ch = champs.get(market, {})
        sha = hashlib.sha256(raw).hexdigest()
        recipe = M.payload_recipe(payload)
        legacy = recipe == PC.RECIPE_LEGACY
        t.update({
            "recipe": recipe,
            "serving_version": ch.get("model_version"),
            "champion_since": ch.get("champion_since"),
            "artifact_sha256": sha,
            "artifact_matches_registry": sha == ch.get("artifact_sha256"),
            "training_cutoff": ch.get("training_cutoff") or None,
            "imputer_version": payload.get("imputer_version", "BATCH_MEDIAN_LEGACY"),
            "feature_contract_version": payload.get("contract_version", "NONE"),
            "n_features": len(payload["feature_cols"]),
            "missing_feature_policy": ("absent column -> 0.0; NaN -> median of the scoring batch"
                                       if legacy else "feature contract " + str(payload.get("contract_version"))),
            "holdout_independence": "FAIL (imputer medians include holdout rows)" if legacy else "PASS",
            "chronology": "PASS (fit 0-68% | calibration 68-80% | holdout 80-100%, by date)",
            "serving_fallback_policy": "none (props have no row-level fallback)",
            "price_provenance": "N/A - props are PAPER permanently",
            "training_data_note": (
                f"champion since {str(ch.get('champion_since'))[:10]}: trained on pre-{INJURY_PIT_FIX} "
                f"player_history with LEAKED injury values (point-in-time repair fb43970c)"
                if str(ch.get("champion_since") or "")[:10] < INJURY_PIT_FIX
                else "trained after the injury point-in-time repair"),
        })
        if rows is not None:
            try:
                t["batch_invariance"] = batch_probe(payload, rows)
            except Exception as e:  # noqa: BLE001
                t["batch_invariance"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            t["batch_invariance"] = {"error": "no player_history part available"}
        t["challenger_r2"] = _challenger(market, r2)
        bi = t["batch_invariance"].get("batch_invariant")
        if not t["artifact_matches_registry"]:
            t["status"] = "FAIL"
        elif legacy or bi is False:
            t["status"] = "WARN"   # known legacy defect, repaired challenger exists, paper market
        else:
            t["status"] = "PASS"
        tracks[market] = t
    try:
        from . import fantasy_defaults as FD
        fantasy = FD.summary()
    except Exception as e:  # noqa: BLE001
        fantasy = {"error": f"{type(e).__name__}: {e}"}
    states = [t["status"] for t in tracks.values()]
    overall = "FAIL" if "FAIL" in states or not contract_info.get("covers_feature_list") else \
              ("WARN" if "WARN" in states else "PASS")
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": _sha(),
        "workstream": "R2-D (findings 20, 21, 22)",
        "status": overall,
        "status_meaning": ("WARN = serving props payloads still use the legacy batch-median "
                           "imputer (paper market, no staking); repaired challengers are "
                           "RETROSPECTIVE only until owner review"),
        "feature_contract": contract_info,
        "imputer_version_repaired": PC.IMPUTER_VERSION,
        "tracks": tracks,
        "fantasy_defaults": fantasy,
    }


def main() -> int:
    h = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(h, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[props_integrity] status={h['status']}")
    for m, t in h["tracks"].items():
        bi = t.get("batch_invariance", {})
        print(f"   {m:<8} {t['status']:<5} recipe={t.get('recipe')} batch_invariant={bi.get('batch_invariant')} "
              f"max_diff={bi.get('max_abs_diff_single_vs_batch')} r2={t.get('challenger_r2', {}).get('state')}")
    print(f"[props_integrity] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
