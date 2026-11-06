r"""Record the go-live state of all five Wowza repos. READ-ONLY: it changes nothing on GitHub.

    python scripts/golive/snapshot_state.py --out C:\Users\nevo\dev\golive_state_<tag>.json

For each repo (gh api / gh CLI, needs `gh auth status` OK):
  * main and data-truth-100 head SHAs (the pre-merge SHA every rollback returns to),
  * every workflow with its id, path and state (active / disabled_manually / ...), so the freeze and
    the unfreeze restore EXACTLY the states recorded here - a workflow disabled on purpose
    (docs/STOPLIGHT.md) is never re-enabled by accident,
  * repository variables (names and values; none of these is a secret),
  * runs queued or in progress (a merge must not start while one can still push).
Prints a one-line summary per repo. Exit 1 if any repo has a queued / in-progress run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys

GH = shutil.which("gh") or "gh"
REPOS = ["wowza-betting", "wowzaV9-Pro", "wowza_v11", "wowza-dashboard", "wowza-exec"]


def gh(*args: str):
    r = subprocess.run([GH, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {r.stderr.strip()[:500]}")
    return json.loads(r.stdout) if r.stdout.strip() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    snap = {"taken_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "repos": {}}
    busy = 0
    for r in REPOS:
        full = f"NevixAA/{r}"
        main_sha = gh("api", f"repos/{full}/branches/main")["commit"]["sha"]
        try:
            dt_sha = gh("api", f"repos/{full}/branches/data-truth-100")["commit"]["sha"]
        except RuntimeError:
            dt_sha = None
        wfs = gh("api", f"repos/{full}/actions/workflows?per_page=100")["workflows"]
        variables = gh("api", f"repos/{full}/actions/variables?per_page=100")["variables"]
        running = []
        for status in ("queued", "in_progress"):
            for run in gh("api", f"repos/{full}/actions/runs?status={status}&per_page=100")["workflow_runs"]:
                running.append({"id": run["id"], "name": run["name"], "branch": run["head_branch"], "status": status})
        busy += len(running)
        snap["repos"][r] = {
            "main": main_sha, "data_truth_100": dt_sha,
            "workflows": [{"id": w["id"], "name": w["name"], "path": w["path"], "state": w["state"]} for w in wfs],
            "variables": {v["name"]: v["value"] for v in variables},
            "running": running,
        }
        n_active = sum(w["state"] == "active" for w in wfs)
        print(f"{r:16} main {main_sha[:10]}  dt {str(dt_sha)[:10]}  workflows {len(wfs)} ({n_active} active)  "
              f"vars {sorted(snap['repos'][r]['variables'])}  running {len(running)}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1)
    print("written", a.out)
    return 1 if busy else 0


if __name__ == "__main__":
    sys.exit(main())
