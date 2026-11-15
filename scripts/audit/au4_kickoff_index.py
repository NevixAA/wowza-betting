"""AU4 - kickoff index for player-prop fixtures, rebuilt from git history.

No prop price file stores a kickoff time (player_prop_odds_history / prop_quotes carry only a UTC
match_date), so "close strictly pre-kickoff" cannot be checked from the price files alone.
output/player_tips.csv carries kickoff_utc per (date, league, match), but it is a rolling file.
This script samples up to K commits per calendar day of its git history on origin/main and
unions the (match, date, league, kickoff_utc) tuples it has ever contained.

Output: reports/au4_data/prop_kickoff_index.csv  (match, league, date, kickoff_utc, n_seen,
n_distinct_kickoff, first_commit_ts, last_commit_ts). A fixture whose kickoff changed between
commits (postponement / reschedule) keeps every value in n_distinct_kickoff; the LAST observed
value is used as kickoff_utc.

Usage: python scripts/audit/au4_kickoff_index.py [--per-day 3] [--ref origin/main]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import ROOT, heavy_lock, save_csv  # noqa: E402


def _git(*args) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-day", type=int, default=3)
    ap.add_argument("--ref", default="origin/main")
    a = ap.parse_args()
    log = _git("log", "--format=%H %cI", a.ref, "--", "output/player_tips.csv").decode().split("\n")
    commits = pd.DataFrame([l.split(" ", 1) for l in log if l.strip()], columns=["sha", "ts"])
    commits["ts"] = pd.to_datetime(commits["ts"], utc=True)
    commits["day"] = commits["ts"].dt.strftime("%Y-%m-%d")
    pick = []
    for _, g in commits.sort_values("ts").groupby("day"):
        idx = sorted(set([0, len(g) // 2, len(g) - 1]))[: a.per_day]
        pick.append(g.iloc[idx])
    pick = pd.concat(pick)
    print(f"[kickoff_index] {len(commits)} commits, sampling {len(pick)}", flush=True)
    rows = []
    for sha, ts in zip(pick["sha"], pick["ts"]):
        try:
            raw = _git("show", f"{sha}:output/player_tips.csv")
            d = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in ("date", "kickoff_utc", "league", "match"))
        except Exception:
            continue
        if "kickoff_utc" not in d.columns:
            continue
        d = d.dropna(subset=["kickoff_utc"]).drop_duplicates(["date", "league", "match", "kickoff_utc"])
        d["commit_ts"] = ts
        rows.append(d)
    k = pd.concat(rows, ignore_index=True)
    k["kickoff_utc"] = pd.to_datetime(k["kickoff_utc"], utc=True, errors="coerce", format="mixed")
    k = k.dropna(subset=["kickoff_utc"])
    k = k.sort_values("commit_ts")
    g = k.groupby(["match", "league", "date"], dropna=False)
    out = g.agg(kickoff_utc=("kickoff_utc", "last"), n_seen=("kickoff_utc", "size"),
                n_distinct_kickoff=("kickoff_utc", "nunique"),
                first_commit_ts=("commit_ts", "min"), last_commit_ts=("commit_ts", "max")).reset_index()
    p = save_csv(out, "prop_kickoff_index.csv")
    print(f"[kickoff_index] {len(out)} fixtures -> {p}; rescheduled={int((out.n_distinct_kickoff>1).sum())}")


if __name__ == "__main__":
    with heavy_lock("AU4 kickoff_index"):
        main()
