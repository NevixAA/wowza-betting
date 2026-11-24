"""Model release registry - readable names for prediction-ledger model versions (owner 2026-11-23).

    python scripts/build_model_release_registry.py --pro-repo <wowzaV9-Pro clone>            # rewrite
    python scripts/build_model_release_registry.py --pro-repo <clone> --check                # exit 1 if stale

OWNER: never make the owner read "btts_20261005T101905Z_23...". Every ledger model version gets
"<release name> · <install date> · <status>", built from ONE committed file, config/model_release_registry.json,
which this script DERIVES from the install history - nothing is typed by hand except the release names
(config/model_release_names.json: release key -> name; a key with no name stays "unknown release").

INSTALL HISTORY (first-parent history of main = what production actually served, and when):
  v9 team        models/registry/current_champions.json  slot -> model_version (+ champion_since, promotion
                 source, retrain batch, stage). served_from = the first main commit carrying it as champion,
                 served_to = the first main commit replacing it.
  v9 props       models/registry/props/props_champions.json  market -> model_version + artifact_sha256. The
                 ledger names a props version '<recipe>_<sha8>' with its own recipe word, so a ledger version is
                 matched on market + the 8-hex ARTIFACT digest (the same bytes), never on the name.
  v9 corners/cards  models/side_models/side_models.json model_version.
  Pro dc / combo    the ledger version is '<model>_<calc>_<sha8 of src/combo/score_model.py | match_picture.py>'
                 (wowzaV9-Pro src/prediction_ledger_pro.py): each main commit of that file gives one version.
install_date = champion_since (team / props) or the first serving commit (side, Pro).
Deterministic: commit times only, no wall clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "model_release_registry.json"
NAMES = ROOT / "config" / "model_release_names.json"
SCHEMA = "mrr/1.0"

TEAM = "models/registry/current_champions.json"
PROPS = "models/registry/props/props_champions.json"
SIDE = "models/side_models/side_models.json"
PRO_FILES = {"pro_bb_dc": "src/combo/score_model.py", "pro_bb_combo": "src/combo/match_picture.py"}
PRO_CALC = "src/combo/score_model.py"


def _git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8")


def _utc(iso: str) -> str:
    import pandas as pd
    return pd.Timestamp(iso).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def history(repo: Path, ref: str, path: str) -> list[tuple[str, str, bytes]]:
    """(commit, commit time UTC, file bytes) of every first-parent commit of `ref` touching `path`, oldest first."""
    out = []
    for line in _git(repo, "log", "--first-parent", "--reverse", "--format=%H %cI", ref, "--", path).splitlines():
        h, t = line.split()
        try:
            out.append((h, _utc(t), _git(repo, "show", f"{h}:{path}", binary=True)))
        except subprocess.CalledProcessError:                         # deleted in that commit
            out.append((h, _utc(t), b""))
    return out


def _windows(events: list[tuple[str, str, str, dict]]) -> list[dict]:
    """events: (time, slot, version, record) per commit snapshot -> one window per (slot, version) run."""
    cur: dict = {}
    rows: list[dict] = []
    for t, slot, ver, rec in events:
        if cur.get(slot) and cur[slot]["version_id"] == ver:
            continue
        if cur.get(slot):
            cur[slot]["served_to"] = t
        if ver:
            cur[slot] = {"slot": slot, "version_id": ver, "served_from": t, "served_to": None, **rec}
            rows.append(cur[slot])
        else:
            cur[slot] = None
    return rows


def team_versions(repo: Path, ref: str) -> list[dict]:
    ev = []
    for _, t, raw in history(repo, ref, TEAM):
        d = json.loads(raw.decode("utf-8")) if raw else {}
        for slot, v in sorted(d.items()):
            if not isinstance(v, dict) or not v.get("model_version"):
                continue
            stage = re.search(r"\bstage (\S+)", str(v.get("promotion_reason") or ""))
            ev.append((t, f"team:{slot}", v["model_version"], {
                "installed_at": v.get("champion_since") or t,
                "release_key": f"{v.get('retrain_batch_id') or ''}:{stage.group(1) if stage else ''}",
                "promotion_source": v.get("promotion_source") or "", "match": "exact"}))
    return _windows(ev)


def props_versions(repo: Path, ref: str) -> list[dict]:
    ev = []
    for _, t, raw in history(repo, ref, PROPS):
        d = json.loads(raw.decode("utf-8")) if raw else {}
        for mkt, v in sorted(d.items()):
            if not isinstance(v, dict) or not v.get("model_version"):
                continue
            src = v.get("promotion_source") or ""
            recipe = re.sub(rf"^props_{re.escape(mkt)}_|_[0-9a-f]{{8}}$", "", v["model_version"])
            ev.append((t, f"props:{mkt}", v["model_version"], {
                "installed_at": v.get("champion_since") or t, "release_key": f"props:{src}:{recipe}",
                "promotion_source": src, "artifact8": str(v.get("artifact_sha256") or "")[:8],
                "match": "market+artifact"}))
    return _windows(ev)


def props_challengers(repo: Path, ref: str) -> list[dict]:
    """Props challengers (models/registry/props/<market>/challenger.json): predict beside the champion, never serve."""
    rows = []
    paths = _git(repo, "ls-tree", "-r", "--name-only", ref, "models/registry/props/").splitlines()
    for path in sorted(p for p in paths if p.endswith("/challenger.json")):
        mkt = path.split("/")[-2]
        ev = []
        for _, t, raw in history(repo, ref, path):
            d = json.loads(raw.decode("utf-8")) if raw else {}
            ver = d.get("model_version") or ""
            ev.append((t, f"props_challenger:{mkt}", ver, {
                "installed_at": d.get("shadow_since") or d.get("registered_at") or t,
                "release_key": f"props:CHALLENGER:{d.get('candidate') or ''}", "promotion_source": "SHADOW",
                "artifact8": str(d.get("artifact_sha256") or "")[:8], "match": "market+artifact",
                "role": "challenger"}))
        rows += _windows(ev)
    return rows


def side_versions(repo: Path, ref: str) -> list[dict]:
    ev = []
    for _, t, raw in history(repo, ref, SIDE):
        d = json.loads(raw.decode("utf-8")) if raw else {}
        ver = d.get("model_version") or ""
        ev.append((t, "side:corners_cards", ver, {"installed_at": t, "release_key": f"side:{ver.rsplit('_', 1)[0]}",
                                                 "promotion_source": "SIDE_MODELS_FILE", "match": "exact"}))
    return _windows(ev)


def pro_versions(repo: Path, ref: str) -> list[dict]:
    rows = []
    for model_id, path in PRO_FILES.items():
        ev = []
        for h, t, raw in history(repo, ref, path):
            if not raw:
                ev.append((t, f"pro:{model_id}", "", {}))
                continue
            calc_src = _git(repo, "show", f"{h}:{PRO_CALC}")
            m = re.search(r'^CALC_VERSION\s*=\s*["\']([^"\']+)["\']', calc_src, re.M)
            calc = m.group(1) if m else ""
            sha8 = hashlib.sha256(raw).hexdigest()[:8]
            ver = f"{model_id}_{calc}_{sha8}"
            ev.append((t, f"pro:{model_id}", ver, {"installed_at": t, "release_key": f"pro:{model_id}:{calc}",
                                                   "promotion_source": "PRO_MAIN_COMMIT", "match": "exact"}))
        rows += _windows(ev)
    return rows


def build(v9_repo: Path, pro_repo: Path | None, v9_ref: str = "origin/main", pro_ref: str = "origin/main") -> dict:
    names = json.loads(NAMES.read_text(encoding="utf-8"))["names"]
    rows = (team_versions(v9_repo, v9_ref) + props_versions(v9_repo, v9_ref) + props_challengers(v9_repo, v9_ref)
            + side_versions(v9_repo, v9_ref))
    if pro_repo:
        rows += pro_versions(pro_repo, pro_ref)
    for r in rows:
        r.setdefault("role", "champion")
        r.setdefault("artifact8", "")
        r["release_name"] = names.get(r["release_key"], "")
    rows.sort(key=lambda r: (r["slot"], r["served_from"], r["version_id"]))
    return {"artifact": "model_release_registry", "schema_version": SCHEMA,
            "about": "GENERATED by scripts/build_model_release_registry.py from the first-parent install history of "
                     "main (v9 champion files, side_models.json, Pro src/combo files). Names: "
                     "config/model_release_names.json. Do not hand-edit.",
            "sources": [TEAM, PROPS, SIDE] + ([f"wowzaV9-Pro:{p}" for p in PRO_FILES.values()] if pro_repo else []),
            "versions": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pro-repo", type=Path, required=True, help="a clone of NevixAA/wowzaV9-Pro (its origin/main)")
    ap.add_argument("--pro-ref", default="origin/main")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    doc = build(ROOT, a.pro_repo, pro_ref=a.pro_ref)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print("config/model_release_registry.json is stale - rerun scripts/build_model_release_registry.py")
            return 1
        print(f"model release registry in sync ({len(doc['versions'])} versions)")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(doc['versions'])} versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
