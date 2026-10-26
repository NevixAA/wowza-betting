"""Build the SLIM dashboard repo tree (NevixAA/wowza-dashboard) from this v9 checkout.

    python scripts/build_dashboard_repo.py OUT_DIR [--force]

Copies exactly the CODE files listed in deploy/dashboard_manifest.json, plus the deploy/SLIM
marker, the manifest itself, a slim requirements.txt (deploy/requirements-dashboard.txt), a
README saying the tree is generated, and a .gitignore for the data the runtime mirror downloads.
NO data files and NO models are copied: dashboard_data/remote_mirror.py downloads the manifest's
DATA files from v9's public raw URLs at runtime.

OUT_DIR may be an existing clone of the slim repo: everything in it except .git/ is replaced, so
files dropped from the manifest disappear from the slim repo on the next push. It refuses to build
inside this v9 checkout.

Regenerate the manifest first when pages change: python scripts/dashboard_manifest.py --write
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deploy" / "dashboard_manifest.json"
REQS = ROOT / "deploy" / "requirements-dashboard.txt"

README = """# wowza-dashboard (GENERATED - do not edit by hand)

This repository is the SLIM deployment of the Wowza Streamlit dashboard. It is **generated** from
the v9 production repo (NevixAA/wowza-betting) by `scripts/build_dashboard_repo.py` and pushed as
a whole. Any hand edit here is overwritten by the next build: change the code in v9, then rebuild.

* Code: the dashboard's pages and the modules they import, listed in `deploy/dashboard_manifest.json`.
* Data: NOT in this repo. `dashboard_data/remote_mirror.py` downloads the manifest's data files from
  `https://raw.githubusercontent.com/NevixAA/wowza-betting/main/<path>` into the same relative paths
  at startup and then at most every 10 minutes (the `deploy/SLIM` marker switches it on).
* Built from v9 commit: `{sha}` at {built}.

Full instructions: `deploy/README_DASHBOARD_DEPLOY.md`.
"""


def _git_sha() -> str:
    import os
    try:
        return subprocess.run([os.getenv("WOWZA_GIT", "git"), "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:                                                # noqa: BLE001
        return "unknown"


def _gitignore(data_paths: list[str], code_paths: list[str]) -> str:
    lines = ["# GENERATED. Data below is downloaded at runtime by dashboard_data/remote_mirror.py;",
             "# it must never be committed to the slim repo.",
             "deploy/.mirror_sync.json", "**/.*.tmp", "dashboard_data/_remote_cache/",
             "__pycache__/", "*.pyc", ".env", ".api_keys", ".streamlit/secrets.toml", ""]
    code_tops = {p.split("/")[0] for p in code_paths if "/" in p}
    whole = sorted({p.split("/")[0] for p in data_paths if "/" in p} - code_tops)
    lines.append("# whole data directories (no code file lives in them)")
    lines += [f"/{t}/" for t in whole]
    lines.append("# individual data files")
    lines += [f"/{p}" for p in data_paths if p.split("/")[0] not in whole]
    return "\n".join(lines) + "\n"


def build(out: Path, force: bool = False) -> dict:
    out = out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit(f"refusing to build inside the v9 checkout: {out}")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    code = [e["path"] if isinstance(e, dict) else e for e in m["code"]]
    data = [e["path"] if isinstance(e, dict) else e for e in m["data"]]
    clash = sorted(set(code) & set(data))
    if clash:
        raise SystemExit(f"manifest lists files as both code and data: {clash}")

    if out.exists():
        existing = [p for p in out.iterdir() if p.name != ".git"]
        if existing and not force and not (out / ".git").exists() and not (out / "deploy" / "SLIM").exists():
            raise SystemExit(f"{out} is not empty and is not a slim build; pass --force")
        for p in existing:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    for rel in code:
        src = ROOT / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if missing:
        raise SystemExit(f"manifest code files missing from the checkout: {missing}")

    sha = _git_sha()
    if sha == "unknown":
        sha = f"unknown (manifest traced at {m.get('v9_commit')})"
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "deploy").mkdir(exist_ok=True)
    shutil.copy2(MANIFEST, out / "deploy" / "dashboard_manifest.json")
    (out / "deploy" / "SLIM").write_text(
        "This tree is the SLIM dashboard deployment. Its presence switches on\n"
        "dashboard_data/remote_mirror.py (data downloaded from v9's public raw URLs).\n", encoding="utf-8")
    shutil.copy2(REQS, out / "requirements.txt")
    (out / "README.md").write_text(README.format(sha=sha, built=built), encoding="utf-8")
    (out / ".gitignore").write_text(_gitignore(data, code),encoding="utf-8")
    (out / "deploy" / "BUILD_INFO.json").write_text(json.dumps(
        {"built_at_utc": built, "v9_commit_of_manifest": m.get("v9_commit"), "v9_head_at_build": _git_sha(),
         "code_files": len(code), "data_files_mirrored": len(data)}, indent=1), encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    n = sum(1 for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    return {"out": str(out), "files": n, "bytes": total, "mb": round(total / 1e6, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--force", action="store_true", help="replace a non-empty directory that is not a slim build")
    a = ap.parse_args(argv)
    r = build(Path(a.out_dir), a.force)
    print(f"built {r['files']} files, {r['mb']} MB -> {r['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
