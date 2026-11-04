"""
collect_transfers.py - API-Football /transfers?team for every team id in our fixture lists (owner
decision 2026-11-04: transfers were fetched live only, never stored historically).

    python scripts/collect_transfers.py [--limit N] [--dry-run]

CALLS. ONE /transfers?team per provider team id in output/strength/competition_fixtures.parquet
(continental cup clubs included), most recently active team first.

OUTPUT (output/collect/ only)
  transfers_teams.parquet  the journal: team_id, n_players (0 = the provider's real empty answer),
        fetched_at, refresh_window ('initial', or the transfer-window close that made the team due).
        One row per (team_id, refresh_window); a row written before fix8/history2 has none.
  transfers.parquet        one row per transfer: player_id, player_name, date (ISO, as the provider
        gives it), type (the provider's own text - 'Loan', 'Free', a fee, 'N/A' - kept verbatim,
        never interpreted), team_in_id, team_out_id, fetched_at. De-duplicated on the WHOLE answer
        (R_KEY + type + player_name), first seen kept: one transfer is returned by both clubs, and a
        refresh that answers the same adds nothing. A changed answer (a type the provider has since
        filled in) is a NEW row beside the previous one, which is never deleted. A transfer without
        a player id or a date cannot be placed: not stored, counted. A missing in / out club id
        stays None.
REFRESH (owner-approved cadence 2026-11-04, collect_af_lists.refresh_todo): every team is asked once
more after each transfer-window close (2 September / 3 February, or the league's own closes in the
registry) - the window's signings only exist after it. About 1,645 calls per refresh, budget-guarded.
Collection only: no model feature reads these yet (they come with the Phase 2 retrain).
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
J_COLS = ["team_id", "n_players", "fetched_at", "refresh_window"]
R_COLS = ["player_id", "player_name", "date", "type", "team_in_id", "team_out_id", "fetched_at"]
R_KEY = ["player_id", "date", "team_in_id", "team_out_id"]
#: the whole answer: a refresh adds a row only when the provider's answer changed
R_ANSWER = R_KEY + ["type", "player_name"]


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


def todo_teams(fx: pd.DataFrame | None = None, today=None) -> list[tuple[int, str]]:
    """[(team_id, refresh_window)] due now (collect_af_lists.refresh_todo)."""
    return AL.refresh_todo(JOURNAL, fx, today)


def _flush(rb, jb):
    if rb:                                     # journal last: it decides what is asked again
        append_parquet(ROWS, rb, R_ANSWER, R_COLS, keep="first")
    if jb:
        append_parquet(JOURNAL, jb, ["team_id", "refresh_window"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None, today=None) -> dict:
    todo = todo_teams(fx, today)
    stats = {"teams_in_fixture_lists": len(AL.team_ids(fx)), "planned_calls": len(todo),
             "planned_initial": sum(w == AL.INITIAL for _, w in todo),
             "planned_refresh": sum(w != AL.INITIAL for _, w in todo), "calls": 0,
             "transfer_rows": 0, "unplaceable_transfers": 0}
    print(f"[transfers] {json.dumps(stats)}")
    if dry_run:
        return stats
    client = client or AFClient()
    rb, jb = [], []
    try:
        for tid, window in todo[: limit or None]:
            body = client.get("transfers", {"team": int(tid)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            n, rows, bad = parse(body)
            rb.extend(rows)
            stats["transfer_rows"] += len(rows)
            stats["unplaceable_transfers"] += bad
            jb.append({"team_id": int(tid), "n_players": n, "fetched_at": AL.now_iso(),
                       "refresh_window": window})
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
