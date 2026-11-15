"""AU1 Phase 0 - workflow execution map across the estate (READ-ONLY, reads git objects only).

For every .github/workflows/*.yml on <repo>@origin/main: triggers (cron list, dispatch), python entry
points actually executed, env keys set on steps (names only, never values of secrets), files the
workflow stages for commit, concurrency group, timeout.
Usage: python scripts/audit/au1_workflow_map.py  ->  reports/au1_data/workflow_map.csv
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WOWZA = ROOT.parent
REPOS = {"v9": WOWZA / "v9", "pro": WOWZA / "pro", "v11": WOWZA / "v11", "dashboard": WOWZA / "dashboard_slim_build"}


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def main() -> None:
    rows = []
    for name, repo in REPOS.items():
        if not repo.exists():
            continue
        sha = git(repo, "rev-parse", "origin/main").strip()
        files = [f for f in git(repo, "ls-tree", "--name-only", "origin/main", ".github/workflows/").split("\n") if f.endswith((".yml", ".yaml"))]
        for f in files:
            text = git(repo, "show", f"origin/main:{f}")
            try:
                y = yaml.safe_load(text) or {}
            except Exception as e:                       # noqa: BLE001
                y = {"_err": str(e)}
            on = y.get(True, y.get("on", {})) or {}
            crons = [c.get("cron") for c in (on.get("schedule") or [])] if isinstance(on, dict) else []
            triggers = sorted(on.keys()) if isinstance(on, dict) else [str(on)]
            pys, envs, timeouts = set(), set(), set()
            for job in (y.get("jobs") or {}).values():
                if job.get("timeout-minutes"):
                    timeouts.add(str(job["timeout-minutes"]))
                for st in job.get("steps", []) or []:
                    run = st.get("run") or ""
                    pys.update(m.strip() for m in re.findall(r"python3? (?:-m )?[\w./-]+(?: --mode [\w-]+)?", run))
                    for k, v in (st.get("env") or {}).items():
                        envs.add(k if "secrets." in str(v) else f"{k}={v}")
            staged = sorted(set(re.findall(r"(output/[\w./*-]+|models/[\w./*-]+|telegram_bot/[\w.]+|registry/[\w./*-]+|data/[\w./*-]+)", text)))
            rows.append({"repo": name, "repo_sha": sha[:10], "workflow": Path(f).name, "name": y.get("name", ""),
                         "triggers": ";".join(map(str, triggers)), "crons": " ; ".join(c for c in crons if c),
                         "python_entrypoints": " ; ".join(sorted(pys)), "env": " ; ".join(sorted(envs)),
                         "concurrency": str((y.get("concurrency") or {}).get("group", "") if isinstance(y.get("concurrency"), dict) else y.get("concurrency", "")),
                         "timeout_min": ",".join(sorted(timeouts)), "paths_referenced": " ; ".join(staged[:40])})
    out = ROOT / "reports" / "au1_data" / "workflow_map.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} workflows -> {out}")
    for r in rows:
        print(f"{r['repo']:9} {r['workflow']:38} {r['crons'][:60]:60} {r['python_entrypoints'][:110]}")


if __name__ == "__main__":
    main()
