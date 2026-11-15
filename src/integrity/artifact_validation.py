"""Production artifact validation - team payloads and the fantasy path (REAL DATA ONLY, 2026-11-15).

    python -m src.integrity.artifact_validation            # print
    python -m src.integrity.artifact_validation --write    # output/production_artifact_validation.json

For every team payload src/predict.py can load: path, sha256, version, training dates, feature schema
(count + hash), preprocessing recipe, imputer version, missing-data behaviour, the unsupported
features it references (from src.integrity.real_data_provenance), its status (active / standby), and
a PRODUCTION-PATH INFERENCE PROBE: the payload is loaded with src.model.load_models (the loader
predict uses) and scored with predict_proba(serving=True) (the call predict makes) on rows built from
its own training distribution:

    complete      every input at its training median                 -> a finite probability
    optional_gap  every non-required input missing                   -> finite (explicit imputation)
    core_gap      a REQUIRED (core form) input missing               -> NO probability (NaN)
    batch         the three rows scored together == each scored alone -> batch invariance

SAFE only when all four hold. A payload the serving guard refuses is BLOCKED (fails closed: it serves
nothing); one that cannot be loaded or probed is UNVERIFIED. The fantasy path has no serialized model
(fitted at run time); its row lists the files it depends on, with their sha256.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "production_artifact_validation.json"
#: a league routed to each track (config registry), for the probe rows
PROBE_LEAGUE = {"standard": "Championship", "new_format": "Brazil Serie A"}


def _sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _probe_rows(part: dict, league: str) -> pd.DataFrame:
    feats = (part.get("imputer") or {}).get("features") or {}
    base = [c for c in part["feature_cols"] if not c.endswith("__missing")]
    med = {c: (feats.get(c) or {}).get("median") for c in base}
    req = [c for c in base if (feats.get(c) or {}).get("required")]
    complete = {c: (np.nan if v is None else float(v)) for c, v in med.items()}
    optional_gap = {c: (complete[c] if c in req else np.nan) for c in base}
    core_gap = dict(complete)
    if req:
        core_gap[req[0]] = np.nan
    df = pd.DataFrame([complete, optional_gap, core_gap], index=["complete", "optional_gap", "core_gap"])
    df["league"] = league
    return df


def inference_probe(payload: dict, tracks=("standard",)) -> dict:
    """{track: {"verdict": SAFE|BLOCKED|UNSAFE|UNVERIFIED, ...}} through the production scoring call."""
    from src.model import predict_proba, serving_refusal, UnservablePayload
    from src.integrity.real_data_provenance import _parts
    out = {}
    for track, part in _parts(payload, tuple(tracks)).items():
        why = serving_refusal(part)
        if why:
            out[track] = {"verdict": "BLOCKED", "reason": why}
            continue
        try:
            rows = _probe_rows(part, PROBE_LEAGUE.get(track, "Championship"))
            # the whole payload is scored, so a track-split payload routes the rows itself
            p = predict_proba(rows, payload=payload, serving=True)
            alone = pd.concat([predict_proba(rows.loc[[i]], payload=payload, serving=True) for i in rows.index])
        except UnservablePayload as e:
            out[track] = {"verdict": "BLOCKED", "reason": str(e)}
            continue
        except Exception as e:                                         # noqa: BLE001
            out[track] = {"verdict": "UNVERIFIED", "reason": f"probe failed: {type(e).__name__}: {e}"[:200]}
            continue
        res = {k: (None if not np.isfinite(v) else round(float(v), 6)) for k, v in p.items()}
        ok = (res["complete"] is not None and 0 < res["complete"] < 1
              and res["optional_gap"] is not None and 0 < res["optional_gap"] < 1
              and res["core_gap"] is None
              and np.allclose(p.to_numpy(float), alone.to_numpy(float), equal_nan=True, rtol=0, atol=1e-12))
        out[track] = {"verdict": "SAFE" if ok else "UNSAFE", "probabilities": res,
                      "batch_invariant": bool(np.allclose(p.to_numpy(float), alone.to_numpy(float),
                                                          equal_nan=True, rtol=0, atol=1e-12))}
    return out


def _schema_hash(cols) -> str:
    return hashlib.sha256("\n".join(cols).encode("utf-8")).hexdigest()[:16]


def team_artifacts(registry_rows: list[dict] | None = None) -> list[dict]:
    import config
    from src.model import load_models, serving_refusal
    from src.integrity import real_data_provenance as RP
    rows = registry_rows if registry_rows is not None else RP.team_registry()
    out = []
    for label, attr, tracks in RP.TEAM_ARTIFACTS:
        path = Path(getattr(config, attr))
        rec = {"model": label, "config": attr, "path": path.relative_to(ROOT).as_posix()
               if path.is_relative_to(ROOT) else str(path), "sha256": _sha(path),
               "status": RP.artifact_status(attr)}
        try:
            pl = load_models(model_file=path)
        except Exception as e:                                         # noqa: BLE001
            out.append({**rec, "verdict": "UNVERIFIED", "reason": f"cannot load: {type(e).__name__}: {e}"[:200]})
            continue
        imp = pl.get("imputer") or {}
        fc = list(pl.get("feature_cols") or [])
        bad = sorted({r["feature"] for r in rows if r["artifact"] == label
                      and r["classification"] not in RP.ELIGIBLE})
        probe = inference_probe(pl, tracks)
        verdicts = {v["verdict"] for v in probe.values()}
        verdict = ("UNVERIFIED" if "UNVERIFIED" in verdicts else "UNSAFE" if "UNSAFE" in verdicts
                   else "BLOCKED" if "BLOCKED" in verdicts else "SAFE")
        out.append({**rec,
                    "version": pl.get("model_recipe_version") or pl.get("recipe") or "legacy",
                    "recipe": pl.get("recipe"), "target": pl.get("target"),
                    "training_start": pl.get("training_start") or (pl.get("research") or {}).get("training_start"),
                    "training_end": pl.get("training_end") or (pl.get("research") or {}).get("training_end"),
                    "training_cutoff": pl.get("training_cutoff") or (pl.get("research") or {}).get("training_cutoff"),
                    "code_sha": pl.get("code_sha"), "frame_hash": pl.get("frame_hash"),
                    "n_features": len(fc), "feature_schema_hash": _schema_hash(fc),
                    "tracks": {t: (None if s is None else {"n_features": len(s.get("feature_cols") or []),
                                                           "imputer_version": (s.get("imputer") or {}).get("version")})
                               for t, s in (pl.get("tracks") or {}).items()},
                    "imputer_version": imp.get("version"),
                    "imputer_fitted_on": imp.get("fitted_on_dates"),
                    "missing_data_behaviour": (
                        "no imputer: batch median (refused at serving)" if not imp else
                        f"train-only medians ({imp.get('version')}), `__missing` indicators, clip to the "
                        "training range; 1.1 refuses a no-training-values feature; serving: a fixture missing "
                        "its core form gets no probability"),
                    "unsupported_features_referenced": bad,
                    "serving_refusal": serving_refusal(pl), "inference_probe": probe, "verdict": verdict})
    return out


FANTASY_DEPENDENCIES = ("models/model_player_goals.pkl", "models/model_player_assists.pkl",
                        "models/model_player_sot2.pkl", "output/fpl_af_crosswalk.csv",
                        "output/fantasy_fpl_snapshots.csv", "output/fantasy_projection_log.csv")


def fantasy_artifacts() -> list[dict]:
    deps = [{"path": d, "sha256": _sha(ROOT / d), "present": (ROOT / d).exists()} for d in FANTASY_DEPENDENCIES]
    return [{"model": "Fantasy (V1 board + V2 projections)", "status": "active",
             "serialized_model": None,
             "fitted_at_run_time": "V2 LinearResidual + MinutesModel on FINAL pre-deadline FPL snapshot "
                                   "gameweeks (player_model/fantasy_v2.project); V1 is formula-based on the "
                                   "props payloads",
             "dependencies": deps,
             "props_dependency": "props payload provenance belongs to fix/props-data-audit: PENDING",
             "verdict": "UNVERIFIED" if not all(d["present"] for d in deps[:3]) else "PENDING_PROPS"}]


def validate() -> dict:
    return {"artifact": "production_artifact_validation",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "team": team_artifacts(), "fantasy": fantasy_artifacts()}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    doc = validate()
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for a in doc["team"]:
        print(f"{a['verdict']:10} {a['status']:20} {a['model']}: {a.get('n_features')} features, "
              f"imputer {a.get('imputer_version')}, sha {str(a['sha256'])[:12]}"
              + (f"  [{a['serving_refusal']}]" if a.get("serving_refusal") else ""))
    for a in doc["fantasy"]:
        print(f"{a['verdict']:10} {a['status']:20} {a['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
