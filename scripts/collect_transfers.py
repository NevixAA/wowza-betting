"""
collect_transfers.py - API-Football /transfers?team for every team id in our fixture lists (owner
decision 2026-11-04: transfers were fetched live only, never stored historically).

    python scripts/collect_transfers.py [--limit N] [--dry-run]

CALLS. ONE /transfers?team per provider team id in output/strength/competition_fixtures.parquet
(continental cup clubs included), most recently active team first.

OUTPUT (output/collect/ only)
  transfers_teams.parquet  the journal: team_id, n_players (0 = the provider's real empty answer,
        never asked again), fetched_at.
  transfers.parquet        one row per transfer: player_id, player_name, date (ISO, as the provider
        gives it), type (the provider's own text - 'Loan', 'Free', a fee, 'N/A' - kept verbatim,
        never interpreted), team_in_id, team_out_id, fetched_at. De-duplicated on
        (player_id, date, team_in_id, team_out_id): one transfer is returned by both clubs. A
        transfer without a player id or a date cannot be placed: not stored, counted. A missing
        in / out club id stays None.
Collection only: no model feature reads these yet (they come with the Phase 2 retrain). A stored team
is not asked again; refreshing it after a transfer window is an owner decision.
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

JOURNAL = COLLECT_DIR / "transfers_teams.parquet"
ROWS = COLLECT_DIR / "transfers.parquet"
J_COLS = ["team_id", "n_players", "fetched_at"]
R_COLS = ["player_id", "player_name", "date", "type", "team_in_id", "team_out_id", "fetched_at"]
R_KEY = ["player_id", "date", "team_in_id", "team_out_id"]


def parse(body: dict | None) -> tuple[int, list[dict], int]:
    """-> (players returned, transfer rows, transfers without player id or date)."""
    rows, unplaceable, ts = [], 0, AL.now_iso()
    resp = (body or {}).get("response") or []
    for item in resp:
        p = item.get("player") or {}
        pid = AL.int_or_none(p.get("id"))
        for t in item.get("transfers") or []:
            d = t.get("date")
            if pid is None or not d:
                unplaceable += 1
                continue
            teams = t.get("teams") or {}
            rows.append({"player_id": pid, "player_name": p.get("name"), "date": str(d)[:10],
                         "type": t.get("type"),
                         "team_in_id": AL.int_or_none((teams.get("in") or {}).get("id")),
                         "team_out_id": AL.int_or_none((teams.get("out") or {}).get("id")),
                         "fetched_at": ts})
    return len(resp), rows, unplaceable


def todo_teams(fx: pd.DataFrame | None = None) -> list[int]:
    done = AL.done_keys(JOURNAL, ["team_id"])
    return [t for t in AL.team_ids(fx) if (str(t),) not in done]


def _flush(rb, jb):
    if rb:                                     # journal last: it decides what is asked again
        append_parquet(ROWS, rb, R_KEY, R_COLS)
    if jb:
        append_parquet(JOURNAL, jb, ["team_id"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None) -> dict:
    todo = todo_teams(fx)
    stats = {"teams_in_fixture_lists": len(AL.team_ids(fx)), "planned_calls": len(todo), "calls": 0,
             "transfer_rows": 0, "unplaceable_transfers": 0}
    print(f"[transfers] {json.dumps(stats)}")
    if dry_run:
        return stats
    client = client or AFClient()
    rb, jb = [], []
    try:
        for tid in todo[: limit or None]:
            body = client.get("transfers", {"team": int(tid)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            n, rows, bad = parse(body)
            rb.extend(rows)
            stats["transfer_rows"] += len(rows)
            stats["unplaceable_transfers"] += bad
            jb.append({"team_id": int(tid), "n_players": n, "fetched_at": AL.now_iso()})
            if len(jb) >= FLUSH_EVERY:
                _flush(rb, jb)
                rb, jb = [], []
    except BudgetStop as e:
        print(f"[transfers] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush(rb, jb)
        client.flush()
    print(f"[transfers] {json.dumps(stats)}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /transfers per team")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the planned call count; no call")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
