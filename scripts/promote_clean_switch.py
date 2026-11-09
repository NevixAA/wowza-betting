"""The clean-data switch (owner 2026-11-09): assemble the serving payloads from a refit and install
them through the promotion gate's one-time owner override.

    python scripts/promote_clean_switch.py assemble --refit <refit_dir> --out <dir>
        writes <dir>/<model_type>.pkl + assembly.json (sha256 per model type); installs nothing
    python scripts/promote_clean_switch.py install --assembled <dir> --override <override_id>
        src.promotion_gate.owner_clean_switch for every assembled model type (the override record in
        config/promotion_overrides.json must list each artifact's exact sha256), then records the
        archive of the frozen 2026-10-28 Standard model (standard_repaired)

SERVING MAP (src/predict.py, src/model.py):
    standard   <- refit standard/over25     model_v9_standard.pkl  (Standard rows; serving switch legacy_v9)
    newformat  <- refit new_format/over25   model_v9_newformat.pkl (New-Format rows)
    btts / over15 / over35 <- refit standard/<m> + new_format/<m> as ONE track-split payload: the
               Standard payload at the top level and the New-Format payload under tracks['new_format']
               - the format src.model.train_by_track writes and _predict_by_track routes per row
    ht_over05 / ht_over15 <- refit standard/<m> (predict.py scores HT on Standard rows only; the
               New-Format HT refits have no serving path and are not installed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SINGLE = {"standard": ("standard", "over25"), "newformat": ("new_format", "over25"),
          "ht_over05": ("standard", "ht_over05"), "ht_over15": ("standard", "ht_over15")}
SPLIT = ("btts", "over15", "over35")


def _load(refit: Path, track: str, market: str) -> dict:
    return pickle.loads((refit / track / f"{market}.pkl").read_bytes())


def assemble(refit: Path, out: Path) -> dict:
    from src import model as M
    man = json.loads((refit / "refit_manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rec = {"refit": str(refit), "refit_code_sha": man.get("code_sha"), "data": man.get("data"),
           "models": {}}

    def _write(mt, payload, sources):
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        (out / f"{mt}.pkl").write_bytes(raw)
        rec["models"][mt] = {"sha256": hashlib.sha256(raw).hexdigest(), "sources": sources,
                             "recipe": payload.get("recipe"), "target": payload.get("target"),
                             "n_features": len(payload.get("feature_cols") or []),
                             "track_split": payload.get("track_split")}

    for mt, (track, market) in SINGLE.items():
        _write(mt, _load(refit, track, market), [f"{track}/{market}"])
    for market in SPLIT:
        std = dict(_load(refit, "standard", market))
        nf = _load(refit, "new_format", market)
        rows = {t: (man["models"].get(f"{t}/{market}") or {}).get("rows") for t in ("standard", "new_format")}
        std["tracks"] = {"new_format": nf}
        std["track_split"] = {"primary": M.PRIMARY_TRACK, "routed": list(M.ROUTED_TRACKS),
                              "router": "src.model.track_of (config.model_type_for_league)",
                              "rows": rows, "rows_in_no_track": 0,
                              "status": {"standard": "trained", "new_format": "trained"},
                              "assembled_from": "scripts/promote_clean_switch.py (Phase 2 refit, per track)"}
        if tuple(std["tracks"]) != tuple(M.ROUTED_TRACKS):
            raise ValueError("routed tracks changed in src.model; re-check the assembly")
        _write(market, std, [f"standard/{market}", f"new_format/{market}"])
    (out / "assembly.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def install(assembled: Path, override_id: str) -> list[dict]:
    from src import model_registry as reg
    from src import promotion_gate as PG
    rec = json.loads((assembled / "assembly.json").read_text(encoding="utf-8"))
    done = []
    for mt, info in rec["models"].items():
        raw = (assembled / f"{mt}.pkl").read_bytes()
        prior = (reg.read_pointer().get(mt) or {}).get("model_version", "")
        payload = pickle.loads(raw)
        m = PG.owner_clean_switch(
            mt, raw, override_id=override_id, supersede=prior,
            training={"training_start": payload.get("training_start"),
                      "training_end": payload.get("training_end"),
                      "training_cutoff": payload.get("training_cutoff"),
                      "training_rows": payload.get("training_rows"),
                      "code_commit_sha": (rec.get("refit_code_sha") or "")[:12],
                      "retrain_batch_id": f"phase2_refit_{override_id}",
                      "evidence": {"phase2_runs": "reports/phase2_runs/ (leak-fixed re-run)",
                                   "sources": info["sources"]}})
        done.append({"model_type": mt, "version": m["model_version"], "sha256": m["artifact_sha256"],
                     "prior": prior})
        print(mt, m["model_version"], "replaces", prior or "-", flush=True)
    o = [x for x in PG.read_overrides().get("overrides", []) if x.get("id") == override_id][0]
    PG.record_archive("standard_repaired",
                      f"owner {o['approved_on']}: \"the models that worked before we reach 100% clean data are "
                      f"irrelevant\" - the 2026-10-28 v9_repaired Standard O/U 2.5 model stops serving "
                      f"(predict.yml STANDARD_OU25_SERVING_RECIPE=legacy_v9 serves the clean-retrain "
                      f"standard champion); pointer, artifact and file kept as the rollback [{override_id}]")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("assemble")
    a1.add_argument("--refit", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("install")
    a2.add_argument("--assembled", required=True)
    a2.add_argument("--override", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        print(json.dumps({k: v["sha256"] for k, v in assemble(Path(a.refit), Path(a.out))["models"].items()},
                         indent=2))
    else:
        print(json.dumps(install(Path(a.assembled), a.override), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
