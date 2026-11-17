#!/usr/bin/env python3
"""The ONE price-store resolver: place the private store's class A / B / O files at their public
paths in a workspace, so every reader keeps reading the path it reads today.

    PRICE_STORE_DIR=<checkout of wowza-exec af-prices-live> \\
        python scripts/price_store_hydrate.py --repo wowza-betting --branch main [--prune] [--verify]

* PRICE_STORE_DIR unset or empty  ->  no-op, exit 0: readers use the public files exactly as today.
  That is the feature gate and the one-line rollback (unset the variable).
* Only paths selected by config/price_store_paths.json are touched (class C / F never).
* --prune also deletes selected workspace files the store does not hold, so the run reads the
  store's set and nothing else (used by the zero-difference gate).
* --verify reports how many hydrated files differ from the workspace's git HEAD blob.

Counts only in the log: never a file's contents.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import price_store_sync as pss  # noqa: E402


def git_head_blobs(ws: Path) -> dict:
    try:
        return {p: oid for p, (_, oid) in pss.ls_tree(ws, "HEAD").items()}
    except pss.StoreError:
        return {}


def hash_object(path: Path) -> str:
    return subprocess.run(["git", "hash-object", "--no-filters", str(path)], capture_output=True,
                          check=True).stdout.decode().strip()


def hydrate(store_dir: Path, repo: str, branch: str, ws: Path, cfg: dict, prune: bool, verify: bool) -> dict:
    if not (store_dir / ".price_store").is_file():
        raise SystemExit(f"::error title=price store::{store_dir} is not a price store checkout (.price_store marker missing)")
    src = store_dir / pss.prefix_of(repo, branch)
    if not src.is_dir():
        raise SystemExit(f"::error title=price store::the store holds no {repo}@{branch} files")
    files = sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file())
    files = [p for p in files if pss.selected(cfg, repo, p)]
    res = {"repo": repo, "branch": branch, "store_files": len(files), "copied": 0, "pruned": 0}
    for rel in files:
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dst)
        res["copied"] += 1
    if prune:
        keep = set(files)
        for p in ws.rglob("*"):
            if ".git" in p.parts or not p.is_file():
                continue
            rel = p.relative_to(ws).as_posix()
            if pss.selected(cfg, repo, rel) and rel not in keep:
                p.unlink(); res["pruned"] += 1
    if verify:
        head = git_head_blobs(ws)
        res["differs_from_head"] = sum(1 for rel in files if rel in head and hash_object(ws / rel) != head[rel])
        res["not_in_head"] = sum(1 for rel in files if rel not in head)
        res["head_selected_missing"] = sum(1 for rel in head if pss.selected(cfg, repo, rel) and not (ws / rel).is_file())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="place private-store price files at their public paths")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--store-dir", default=os.environ.get("PRICE_STORE_DIR", ""))
    ap.add_argument("--config", default=str(pss.CONFIG))
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    if not a.store_dir:
        print("[price_store] PRICE_STORE_DIR not set: public paths (today's behaviour)")
        return 0
    res = hydrate(Path(a.store_dir), a.repo, a.branch, Path(a.workspace).resolve(), pss.load_config(Path(a.config)),
                  a.prune, a.verify)
    print("[price_store hydrate] " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
