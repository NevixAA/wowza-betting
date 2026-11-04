"""
collect_lineups.py - API-Football /fixtures/lineups history for every FINISHED fixture of a registry
league in our fixture lists (owner decision 2026-11-04: lineups were fetched live only, never stored).

    python scripts/collect_lineups.py [--limit N] [--dry-run]

CALLS. ONE /fixtures/lineups?fixture per finished fixture (the provider has no league-season form of
this endpoint), NEWEST FIRST so a budget stop leaves the oldest seasons for the next run-day; plus one
/leagues?id per league for the coverage flags (collect_af_lists, journaled). A league-season whose
provider flag says lineups=False is skipped without a call and counted ('no_provider_lineups'); a
season whose flag is unknown (coverage call failed / season absent) is skipped this run and counted
('coverage_unknown') - the same rule as the round-6 inside-box collector. This is the largest of the
2026-11-04 collectors, so the workflow runs it last of the API-Football steps.

OUTPUT (output/collect/ only)
  lineups_fixtures.parquet  the journal: fixture_id, league_id, season, n_teams (0 = the provider's
        real empty answer, never asked again), fetched_at.
  lineups_teams.parquet     one row per (fixture_id, team_id): formation, coach_id, coach_name,
        n_start, n_subs. A formation / coach the provider does not give stays None.
  lineups_players.parquet   one row per (fixture_id, team_id, role, slot): player_id, player_name,
        number, pos, grid, role ('start' | 'sub'), slot (order in the provider's list). A player
        without a provider id keeps player_id None (never matched by name).
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

JOURNAL = COLLECT_DIR / "lineups_fixtures.parquet"
TEAMS = COLLECT_DIR / "lineups_teams.parquet"
PLAYERS = COLLECT_DIR / "lineups_players.parquet"
J_COLS = ["fixture_id", "league_id", "season", "n_teams", "fetched_at"]
T_COLS = ["fixture_id", "team_id", "formation", "coach_id", "coach_name", "n_start", "n_subs"]
P_COLS = ["fixture_id", "team_id", "role", "slot", "player_id", "player_name", "number", "pos", "grid"]


def parse(body: dict | None, fixture_id: int) -> tuple[list[dict], list[dict]]:
    teams, players = [], []
    for t in (body or {}).get("response") or []:
        tid = AL.int_or_none((t.get("team") or {}).get("id"))
        coach = t.get("coach") or {}
        start, subs = t.get("startXI") or [], t.get("substitutes") or []
        teams.append({"fixture_id": int(fixture_id), "team_id": tid, "formation": t.get("formation"),
                      "coach_id": AL.int_or_none(coach.get("id")), "coach_name": coach.get("name"),
                      "n_start": len(start), "n_subs": len(subs)})
        for role, lst in (("start", start), ("sub", subs)):
            for i, e in enumerate(lst):
                p = (e or {}).get("player") or {}
                players.append({"fixture_id": int(fixture_id), "team_id": tid, "role": role, "slot": i,
                                "player_id": AL.int_or_none(p.get("id")), "player_name": p.get("name"),
                                "number": AL.int_or_none(p.get("number")), "pos": p.get("pos"),
                                "grid": p.get("grid")})
    return teams, players


def plan(fx: pd.DataFrame | None = None) -> dict:
    rf = AL.registry_fixtures(fx)
    done = AL.done_keys(JOURNAL, ["fixture_id"])
    todo = rf[~rf["fixture_id"].astype(str).map(lambda x: (x,) in done)] if len(rf) else rf
    todo = todo.sort_values(["date", "fixture_id"], ascending=[False, False], kind="mergesort")
    seasons = {(int(a), str(b)) for a, b in zip(rf["league_id"], rf["season"])}
    return {"registry_fixtures": int(len(rf)), "todo": todo.reset_index(drop=True),
            "coverage": AL.coverage_todo(seasons)}


def _flush(tb, pb, jb):
    if pb:                                     # journal last: it decides what is asked again
        append_parquet(PLAYERS, pb, ["fixture_id", "team_id", "role", "slot"], P_COLS)
    if tb:
        append_parquet(TEAMS, tb, ["fixture_id", "team_id"], T_COLS)
    if jb:
        append_parquet(JOURNAL, jb, ["fixture_id"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None) -> dict:
    p = plan(fx)
    cov_known = AL.coverage_map("lineups")
    todo = p["todo"]
    no_cov = sum(1 for a, b in zip(todo["league_id"], todo["season"])
                 if cov_known.get((int(a), str(b))) is False)
    stats = {"registry_finished_fixtures": p["registry_fixtures"],
             "planned_coverage_calls": len(p["coverage"]),
             "planned_fixture_calls_upper_bound": int(len(todo)) - no_cov,
             "skipped_no_provider_lineups_known": no_cov, "calls": 0, "fetched": 0,
             "skipped": {}}
    stats["planned_calls_total_upper_bound"] = (stats["planned_coverage_calls"]
                                                 + stats["planned_fixture_calls_upper_bound"])
    print(f"[lineups] {json.dumps({k: v for k, v in stats.items() if k != 'skipped'})}")
    if dry_run:
        return stats
    client = client or AFClient()
    tb, pb, jb = [], [], []
    try:
        stats["calls"] += AL.ensure_coverage(client, p["coverage"])
        cov = AL.coverage_map("lineups")
        n = 0
        for f in todo.itertuples(index=False):
            flag = cov.get((int(f.league_id), str(f.season)))
            if flag is not True:
                tag = "no_provider_lineups" if flag is False else "coverage_unknown"
                stats["skipped"][tag] = stats["skipped"].get(tag, 0) + 1
                continue
            if limit is not None and n >= limit:
                break
            n += 1
            body = client.get("fixtures/lineups", {"fixture": int(f.fixture_id)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            teams, players = parse(body, f.fixture_id)
            tb.extend(teams)
            pb.extend(players)
            jb.append({"fixture_id": int(f.fixture_id), "league_id": int(f.league_id),
                       "season": str(f.season), "n_teams": len(teams), "fetched_at": AL.now_iso()})
            stats["fetched"] += 1
            if len(jb) >= FLUSH_EVERY:
                _flush(tb, pb, jb)
                tb, pb, jb = [], [], []
    except BudgetStop as e:
        print(f"[lineups] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush(tb, pb, jb)
        client.flush()
    print(f"[lineups] calls {stats['calls']:,}; fixtures stored {stats['fetched']:,}; "
          f"skipped {stats['skipped']}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /fixtures/lineups per finished fixture")
    ap.add_argument("--limit", type=int, default=None, help="at most N fixture calls")
    ap.add_argument("--dry-run", action="store_true", help="print the planned call count; no call")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
