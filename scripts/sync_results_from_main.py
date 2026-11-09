"""
Bring main's newest results into data-truth-100 before go-live (P7, 2026-11-09).
================================================================================
data-truth-100 is frozen while main keeps receiving results (football-data history, the af_history
extend job, the player history extend job), so the repaired branch's history stopped about 2026-10-30
and every tool reading it (the specials tracker, the bars study, the retrains) misses the newest weeks.

THE RULE, per file, decided by git itself (merge-base of origin/main and HEAD):
  * data-truth-100 never changed the file since the branch point, main did  -> take main's file as is.
    That is exactly what the go-live merge will do; it only happens earlier. (Today: fd_history.parquet
    and player_history.parquet/month_2026-08..11.)
  * BOTH changed it (af_history.parquet: the repair removed proven duplicates and the collection
    appended gap rows) -> append ONLY main's rows dated AFTER the last date this branch holds for that
    league. Older rows are never re-added, so a duplicate the owner's rule removed cannot come back.
    Every appended row is listed in output/corrections/af_history_synced_from_main.csv with main's
    commit and the sync time.
  * this branch changed it and main did not -> nothing to do.
Nothing is ever rewritten in place; values are only added. Run again at go-live, right before the
merge, so the rows main gains until then are carried too.

    python scripts/sync_results_from_main.py [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WHOLE_FILE = ["output/fd_history.parquet"]
WHOLE_DIRS = ["player_history.parquet"]
ROW_LEVEL = {"output/af_history.parquet": ["league", "date", "home_team", "away_team"]}
SIDECAR = ROOT / "output" / "corrections" / "af_history_synced_from_main.csv"
MAIN = "origin/main"


def _git(*a, binary=False):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode().strip()


def _blob(ref, path):
    return _git("rev-parse", f"{ref}:{path}")


def plan() -> dict:
    mb = _git("merge-base", MAIN, "HEAD")
    if not mb:
        raise SystemExit(f"no merge-base with {MAIN}: fetch it first (git fetch origin main)")
    paths = list(WHOLE_FILE)
    for d in WHOLE_DIRS:
        names = (_git("ls-tree", "--name-only", f"{MAIN}:{d}") or "").split()
        paths += [f"{d}/{n}" for n in names]
    take, both, ours = [], [], []
    for p in paths + list(ROW_LEVEL):
        b, h, m = _blob(mb, p), _blob("HEAD", p), _blob(MAIN, p)
        if m is None or m == h:
            continue
        if h == b:                       # this branch never changed it (or never had it)
            take.append(p)
        elif m == b:
            ours.append(p)
        else:
            both.append(p)
    return {"merge_base": mb, "main": _git("rev-parse", MAIN), "take_main": take, "both_changed": both,
            "ours_only": ours}


def run(dry_run: bool = False) -> dict:
    pl = plan()
    out = {**pl, "rows_appended": {}}
    if dry_run:
        return out
    for p in pl["take_main"]:
        if p in ROW_LEVEL:
            continue
        data = _git("show", f"{MAIN}:{p}", binary=True)
        (ROOT / p).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / p).write_bytes(data)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for p in [x for x in pl["both_changed"] + pl["take_main"] if x in ROW_LEVEL]:
        key = ROW_LEVEL[p]
        ours = pd.read_parquet(ROOT / p)
        theirs = pd.read_parquet(io.BytesIO(_git("show", f"{MAIN}:{p}", binary=True)))
        base_raw = _git("show", f"{pl['merge_base']}:{p}", binary=True)
        base = pd.read_parquet(io.BytesIO(base_raw)) if base_raw else theirs.iloc[0:0]

        def keys(df):
            k = df[key].copy()
            k["date"] = pd.to_datetime(k["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            return list(map(tuple, k.astype(str).to_numpy()))
        # THREE-WAY: a row main ADDED since the branch point (not in the merge-base file) that this
        # branch does not hold. A row this branch removed (a proven duplicate) is in the merge-base
        # file, so it is never re-added, whatever its date.
        in_base, in_ours = set(keys(base)), set(keys(ours))
        tk = keys(theirs)
        newer = theirs[[k not in in_base and k not in in_ours for k in tk]]
        if newer.empty:
            out["rows_appended"][p] = 0
            continue
        merged = pd.concat([ours, newer[[c for c in newer.columns if c in ours.columns]]], ignore_index=True)
        merged.to_parquet(ROOT / p, index=False)
        side = newer[key].copy()
        side["source_commit"], side["synced_at"], side["file"] = pl["main"], now, p
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        side.to_csv(SIDECAR, mode="a", header=not SIDECAR.exists(), index=False)
        out["rows_appended"][p] = int(len(newer))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    r = run(dry_run=ap.parse_args(argv).dry_run)
    print(f"[syncmain] main {r['main'][:9]} merge-base {r['merge_base'][:9]}")
    print(f"[syncmain] take main's file ({len(r['take_main'])}): {r['take_main']}")
    print(f"[syncmain] both changed -> newer rows only: {r['both_changed']}  appended: {r['rows_appended']}")
    print(f"[syncmain] this branch only: {r['ours_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
