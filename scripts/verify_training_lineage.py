"""Write reports/lineage/training_lineage_evidence.json: the join lineage of every ACTIVE payload
(owner P0 deep-repair item 3, 2026-11-22). READ-ONLY with respect to models, data and serving.

    python scripts/verify_training_lineage.py \
        --std-frame <saved Standard training frame .parquet>      (frame ebe47c041a7f595f, 2026-11-19 refit)
        --std-assembly <assembly.json of that install>
        --nf-keys <training keys .parquet of the 2026-11-18 refit>  (frame 9ed237ebc04aee09)
        --nf-manifest <refit_manifest.json of that refit>

Per payload part (a track-split payload has a Standard and a New-Format part):
  code    git show <code_sha>:<file> for the training-code files (src/data_loader.py, src/integrity/joins.py,
          src/feature_engineering.py, src/retrain_harness/protocol.py) -> training_lineage.code_audit
  frame   the training rows of that track x market -> training_lineage.frame_audit (duplicate fixture keys,
          a club twice on one day, unregistered / other-track leagues, row count vs the payload)
  link    payload sha256 == the install record's sha256; the frame / keys hash == the refit's recorded hash
  pit     the Standard frame's form feature recomputed from earlier matches only (training_lineage.pit_audit)
Props payloads: code audit of player_model/props_harness.py + player_model/feature_engineering.py at their
code_sha (fixture-id keyed joins; the team enrichment joins are not imported). No price is read or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.integrity import training_lineage as TL  # noqa: E402

#: payload -> [(track, harness market)] parts; the track-split side payloads carry both tracks
PARTS = {"model_v9_standard.pkl": [("standard", "over25")],
         "model_v9_newformat.pkl": [("new_format", "over25")],
         "model_v9_btts.pkl": [("standard", "btts"), ("new_format", "btts")],
         "model_v9_over15.pkl": [("standard", "over15"), ("new_format", "over15")],
         "model_v9_over35.pkl": [("standard", "over35"), ("new_format", "over35")],
         "model_ht_over05.pkl": [("standard", "ht_over05")],
         "model_ht_over15.pkl": [("standard", "ht_over15")],
         "model_v9_ht_over05_nf.pkl": [("new_format", "ht_over05")],
         "model_v9_ht_over15_nf.pkl": [("new_format", "ht_over15")]}
PROPS_CODE = ("player_model/props_harness.py", "player_model/feature_engineering.py")


def _git_show(sha: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def _on_remote(sha: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "branch", "-r", "--contains", sha], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _part(payload: dict, track: str) -> dict:
    """The fitted part that serves `track`: tracks[track] of a split payload, else the payload."""
    tr = payload.get("tracks") if isinstance(payload.get("tracks"), dict) else None
    if tr and track != (payload.get("track_split") or {}).get("primary", "standard"):
        return tr.get(track) or {}
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-frame", required=True)
    ap.add_argument("--std-assembly", required=True)
    ap.add_argument("--nf-keys", required=True)
    ap.add_argument("--nf-manifest", required=True)
    ap.add_argument("--out", default=str(TL.EVIDENCE))
    a = ap.parse_args(argv)
    import joblib
    import config
    from src.retrain_harness import protocol as P
    from src.standard_repair import frame_hash
    from src import league_registry as LR
    registry = {e["name"] for e in LR.leagues()}            # canonical names only, never aliases
    track_of = config.model_type_for_league

    asm = json.loads(Path(a.std_assembly).read_text(encoding="utf-8"))
    nfm = json.loads(Path(a.nf_manifest).read_text(encoding="utf-8"))
    feat = pd.read_parquet(a.std_frame)
    std_hash = frame_hash(feat)
    nf_keys = pd.read_parquet(a.nf_keys)
    nf_info_p = Path(a.nf_keys).parent / "info.json"
    nf_info = json.loads(nf_info_p.read_text(encoding="utf-8")) if nf_info_p.exists() else {}
    nf_hash = (nf_info.get("info") or {}).get("frame_hash")
    cfg = P.load_config()

    code_cache: dict = {}

    def code(sha):
        if sha not in code_cache:
            src = {f: _git_show(sha, f) for f in TL.TEAM_CODE}
            code_cache[sha] = {**TL.code_audit(src), "code_sha": sha, "on_remote": _on_remote(sha)}
        return code_cache[sha]

    payloads = {}
    for pfile, parts in PARTS.items():
        p = ROOT / "models" / pfile
        pay = joblib.load(p)
        sha = _sha(p)
        recs = []
        for track, market in parts:
            part = _part(pay, track)
            csha = part.get("code_sha") or pay.get("code_sha")
            rows = part.get("training_rows")
            end = part.get("training_end")
            if track == "standard":
                keys = P.eligible_rows(feat, cfg, "standard", market)[["league", "date", "home_team", "away_team"]]
                src = {"frame": "saved Standard training frame", "frame_hash": std_hash,
                       "recorded_frame_hash": (asm.get("data") or {}).get("frame_hash"),
                       "refit_code_sha": asm.get("refit_code_sha")}
            else:
                keys = nf_keys[(nf_keys["track"] == "new_format") & (nf_keys["market"] == market)]
                src = {"frame": "training keys saved by the 2026-11-18 refit", "frame_hash": nf_hash,
                       "recorded_frame_hash": (nfm.get("data") or {}).get("frame_hash"),
                       "refit_code_sha": nfm.get("code_sha")}
            fa = TL.frame_audit(keys, track, track_of, registry, train_end=end, expected_rows=rows)
            ca = code(csha)
            link_ok = (src["frame_hash"] == src["recorded_frame_hash"] and src["refit_code_sha"] == csha)
            st = TL.worst([fa["status"], ca["status"], TL.PASS if link_ok else TL.FAIL])
            recs.append({"track": track, "market": market, "code_sha": csha, "payload_frame_hash": part.get("frame_hash"),
                         "training_rows": rows, "training_end": end, "source": src, "link_ok": link_ok,
                         "frame_audit": fa, "code_audit_status": ca["status"], "status": st})
        install_sha = ((asm.get("models") or {}).get(pfile.replace("model_v9_", "").replace("model_", "")
                                                       .replace(".pkl", "")) or {}).get("sha256")
        st = TL.worst(r["status"] for r in recs)
        payloads[pfile] = {"artifact_sha256": sha, "install_record_sha256": install_sha, "parts": recs,
                           "status": st,
                           "summary": "; ".join(f"{r['track']}/{r['market']}: {r['frame_audit']['rows']:,} rows "
                                                f"({r['frame_audit'].get('rows_to_train_end')} to train end = payload "
                                                f"{r['training_rows']}), dup keys {r['frame_audit']['duplicate_fixture_keys']}, "
                                                f"club twice/day {r['frame_audit']['club_twice_same_day']}, other-track/"
                                                f"unregistered leagues {len(r['frame_audit']['leagues_on_other_track'])}/"
                                                f"{len(r['frame_audit']['leagues_not_in_registry'])}, code {r['code_audit_status']}"
                                                for r in recs)}
    props = {}
    for p in sorted((ROOT / "models").glob("model_player_*.pkl")):
        pay = joblib.load(p)
        prov = pay.get("provenance") or {}
        csha = prov.get("code_sha")
        src = {f: _git_show(csha, f) for f in PROPS_CODE} if csha else {}
        ca = TL.code_audit(src) if src else {"status": TL.UNVERIFIED, "findings": []}
        imports_team_joins = any(t and ("data_loader" in t or "safe_left_enrich" in t) for t in src.values())
        st = TL.worst([ca["status"], TL.FAIL if imports_team_joins else TL.PASS])
        props[p.name] = {"artifact_sha256": _sha(p), "code_sha": csha, "run_id": prov.get("run_id"),
                         "code_on_remote": _on_remote(csha) if csha else False,
                         "code_audit": ca, "imports_team_enrichment_joins": imports_team_joins, "status": st,
                         "summary": f"code {ca['status']} at {str(csha)[:9]}; joins fixture_id / league keyed; team "
                                    f"enrichment joins {'IMPORTED' if imports_team_joins else 'not imported'}; "
                                    "row-level props frame not re-run here"}
    pit = TL.pit_audit(feat)
    doc = {"artifact": "training_lineage_evidence", "version": TL.LINEAGE_VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generated_by": "scripts/verify_training_lineage.py",
           "inputs": {"std_frame_hash": std_hash, "std_frame_rows": int(len(feat)),
                      "nf_keys_frame_hash": nf_hash, "nf_keys_rows": int(len(nf_keys))},
           "code": code_cache, "point_in_time": pit, "payloads": payloads, "props": props,
           "status": TL.worst([*(v["status"] for v in payloads.values()), *(v["status"] for v in props.values()),
                               pit["status"]])}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"lineage {doc['status']} -> {out}")
    for k, v in payloads.items():
        print(f"  {v['status']:5s} {k}: {v['summary']}")
    for k, v in props.items():
        print(f"  {v['status']:5s} {k}: {v['summary']}")
    print(f"  PIT {pit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
