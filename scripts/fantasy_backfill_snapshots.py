"""Rebuild output/fantasy_fpl_snapshots.csv from the git history of the FPL bootstrap cache.

    python scripts/fantasy_backfill_snapshots.py [--git PATH_TO_GIT]

fantasy_refresh.yml has committed output/fpl_bootstrap.json (and its fetch-time sidecar
fpl_cache_meta.json) once a day since 2026-09-10. Each version is a full pre-deadline view of
every FPL player. This replays them oldest-first through fantasy_snapshots.update_archive, so the
archive ends up holding, per gameweek, the last view taken before that gameweek's deadline.

Needs FULL history (a CI checkout is depth-1), so it is a local one-off; the daily job keeps the
archive current from then on. Versions whose sidecar has no bootstrap fetch time are skipped —
an undatable snapshot cannot be placed on the right side of a deadline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import fantasy_snapshots as fs  # noqa: E402


def _git(git: str, *args: str) -> str:
    return subprocess.run([git, "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", default="git")
    ap.add_argument("--out", default=str(fs.ARCHIVE))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        out.unlink()
    shas = _git(a.git, "log", "--reverse", "--format=%H", "--",
                "output/fpl_bootstrap.json").split()
    print(f"{len(shas)} bootstrap versions in history")
    for sha in shas:
        try:
            boot = json.loads(_git(a.git, "show", f"{sha}:output/fpl_bootstrap.json"))
            meta = json.loads(_git(a.git, "show", f"{sha}:output/fpl_cache_meta.json"))
        except Exception as e:                                       # noqa: BLE001
            print(f"  {sha[:8]} skipped ({type(e).__name__})")
            continue
        ts = meta.get("fpl_bootstrap.json")
        n = fs.update_archive(boot, float(ts) if ts else None, path=out)
        ev = fs.target_event(boot)
        print(f"  {sha[:8]} fetched={ts} target_gw={ev and ev.get('id')} rows={n}")
    d = fs.load(out)
    if not d.empty:
        print(d.groupby("target_gw")["fetched_at"].max().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
