"""
collect_coaches.py - API-Football /coachs?team for every team id in our fixture lists (owner decision
2026-11-04: coaches were fetched live only, never stored historically).

    python scripts/collect_coaches.py [--limit N] [--dry-run]

CALLS. ONE /coachs?team per provider team id in output/strength/competition_fixtures.parquet
(continental cup clubs included), most recently active team first. The endpoint is spelled 'coachs'
by the provider.

OUTPUT (output/collect/ only)
  coaches_teams.parquet   the journal: team_id, n_coaches (0 = the provider's real empty answer,
        never asked again), fetched_at.
  coach_careers.parquet   one row per career entry of every coach returned: coach_id, coach_name,
        team_id (the career entry's club - a coach's career covers other clubs too, all kept),
        start, end (ISO dates as the provider gives them; a current post has end None), fetched_at.
        De-duplicated on (coach_id, team_id, start): the same career is returned by every club the
        coach worked for. A career entry without a team id or a start cannot be placed: not stored,
        counted.
Collection only: no model feature reads these yet (they come with the Phase 2 retrain). A stored team
is not asked again; refreshing current posts later is an owner decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet)

JOURNAL = COLLECT_DIR / "coaches_teams.parquet"
CAREERS = COLLECT_DIR / "coach_careers.parquet"
J_COLS = ["team_id", "n_coaches", "fetched_at"]
C_COLS = ["coach_id", "coach_name", "team_id", "start", "end", "fetched_at"]


def parse(body: dict | None) -> tuple[int, list[dict], int]:
    """-> (coaches returned, career rows, career entries without team id or start)."""
    rows, unplaceable, ts = [], 0, AL.now_iso()
    resp = (body or {}).get("response") or []
    for c in resp:
        cid = AL.int_or_none(c.get("id"))
        for e in c.get("career") or []:
            tid = AL.int_or_none((e.get("team") or {}).get("id"))
            st, en = e.get("start"), e.get("end")
            if cid is None or tid is None or not st:
                unplaceable += 1
                continue
            rows.append({"coach_id": cid, "coach_name": c.get("name"), "team_id": tid,
                         "start": str(st)[:10], "end": str(en)[:10] if en else None, "fetched_at": ts})
    return len(resp), rows, unplaceable


def todo_teams(fx: pd.DataFrame | None = None) -> list[int]:
    done = AL.done_keys(JOURNAL, ["team_id"])
    return [t for t in AL.team_ids(fx) if (str(t),) not in done]


def _flush(cb, jb):
    if cb:                                     # journal last: it decides what is asked again
        append_parquet(CAREERS, cb, ["coach_id", "team_id", "start"], C_COLS)
    if jb:
        append_parquet(JOURNAL, jb, ["team_id"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None) -> dict:
    todo = todo_teams(fx)
    stats = {"teams_in_fixture_lists": len(AL.team_ids(fx)), "planned_calls": len(todo), "calls": 0,
             "career_rows": 0, "unplaceable_career_entries": 0}
    print(f"[coaches] {json.dumps(stats)}")
    if dry_run:
        return stats
    client = client or AFClient()
    cb, jb = [], []
    try:
        for tid in todo[: limit or None]:
            body = client.get("coachs", {"team": int(tid)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            n, rows, bad = parse(body)
            cb.extend(rows)
            stats["career_rows"] += len(rows)
            stats["unplaceable_career_entries"] += bad
            jb.append({"team_id": int(tid), "n_coaches": n, "fetched_at": AL.now_iso()})
            if len(jb) >= FLUSH_EVERY:
                _flush(cb, jb)
                cb, jb = [], []
    except BudgetStop as e:
        print(f"[coaches] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush(cb, jb)
        client.flush()
    print(f"[coaches] {json.dumps(stats)}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /coachs per team")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the planned call count; no call")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
