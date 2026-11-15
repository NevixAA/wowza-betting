"""
collect_injuries.py - API-Football /injuries history for every registry league-season in our fixture
lists (owner decision 2026-11-04: until now injuries were fetched live only and never stored).

    python scripts/collect_injuries.py [--limit N] [--dry-run]

CALLS. ONE /injuries?league&season per league-season (a season holds hundreds of fixtures, so this is
~300x cheaper than per fixture), newest season first, plus one /leagues?id per league for the
coverage flags (collect_af_lists, journaled). A league-season answer is checked for completeness:
  * paging.total > paging.current  -> the provider split it into pages this call did not return;
  * `results` != the number of entries returned.
Either one marks the league-season 'incomplete' and ONLY THEN its finished fixtures are asked one by
one (/injuries?fixture), each journaled. An entry for another league / season is not stored (counted).

RE-ASKING. A stored league-season answer (an empty one included) is never asked again, except an
OPEN season whose finished-fixture count in the fixture list has grown since the answer was stored
(step 1 refreshes the current season every run): one call then refreshes it. Rows are de-duplicated
on (fixture_id, date, team_id, player_id), the newest answer winning.

OUTPUT (output/collect/ only)
  injuries.parquet          one row per entry: league_id, season, fixture_id, date, team_id,
        player_id, player_name, type, reason, source ('league_season' | 'fixture'), fetched_at.
        A field the provider does not give stays None (never a placeholder).
  injuries_seasons.parquet  the journal: league_id, season, status ('ok' | 'empty' | 'incomplete'),
        results, n_entries, paging_current, paging_total, n_finished_at_fetch, coverage_injuries
        (the provider's flag, for the record: the call is made whatever it says), fetched_at.
  injuries_fixtures.parquet the per-fixture fallback journal: fixture_id, league_id, season,
        n_entries, fetched_at.
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
                            append_parquet, read_parquet)

ROWS = COLLECT_DIR / "injuries.parquet"
SEASON_JOURNAL = COLLECT_DIR / "injuries_seasons.parquet"
FIXTURE_JOURNAL = COLLECT_DIR / "injuries_fixtures.parquet"
ROW_COLS = ["league_id", "season", "fixture_id", "date", "team_id", "player_id", "player_name",
            "type", "reason", "source", "fetched_at"]
ROW_KEY = ["fixture_id", "date", "team_id", "player_id"]
SJ_COLS = ["league_id", "season", "status", "results", "n_entries", "paging_current", "paging_total",
           "n_finished_at_fetch", "coverage_injuries", "fetched_at"]
FJ_COLS = ["fixture_id", "league_id", "season", "n_entries", "fetched_at"]


def parse_entries(body: dict | None, league_id: int, season: str, source: str) -> tuple[list[dict], int]:
    """Entries of this league-season -> rows; returns (rows, entries of another league/season)."""
    rows, foreign, ts = [], 0, AL.now_iso()
    for e in (body or {}).get("response") or []:
        lg = e.get("league") or {}
        if (AL.int_or_none(lg.get("id")) not in (None, int(league_id))
                or (lg.get("season") is not None and str(lg.get("season")) != str(season))):
            foreign += 1
            continue
        p, t, f = e.get("player") or {}, e.get("team") or {}, e.get("fixture") or {}
        d = f.get("date")
        rows.append({"league_id": int(league_id), "season": str(season),
                     "fixture_id": AL.int_or_none(f.get("id")), "date": str(d)[:10] if d else None,
                     "team_id": AL.int_or_none(t.get("id")), "player_id": AL.int_or_none(p.get("id")),
                     "player_name": p.get("name"), "type": p.get("type"), "reason": p.get("reason"),
                     "source": source, "fetched_at": ts})
    return rows, foreign


def completeness(body: dict | None) -> dict:
    """The provider's own paging / results fields; complete only when both agree."""
    b = body or {}
    pg = b.get("paging") or {}
    cur, tot = AL.int_or_none(pg.get("current")), AL.int_or_none(pg.get("total"))
    res, n = AL.int_or_none(b.get("results")), len(b.get("response") or [])
    complete = not (cur is not None and tot is not None and tot > cur) and (res is None or res == n)
    return {"complete": complete, "results": res, "n_entries": n, "paging_current": cur,
            "paging_total": tot}


def league_seasons(fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """(league_id, season, n_finished, last_date) for every registry league-season, newest first."""
    rf = AL.registry_fixtures(fx)
    if rf.empty:
        return pd.DataFrame(columns=["league_id", "season", "n_finished", "last_date"])
    g = rf.groupby(["league_id", "season"]).agg(n_finished=("fixture_id", "size"),
                                                 last_date=("date", "max")).reset_index()
    return g.sort_values(["last_date", "league_id"], ascending=[False, True], kind="mergesort") \
            .reset_index(drop=True)


def plan(fx: pd.DataFrame | None = None) -> dict:
    fx = AL.fixture_list() if fx is None else fx
    ls = league_seasons(fx)
    sj = read_parquet(SEASON_JOURNAL)
    stored = {}
    if not sj.empty:
        for r in sj.itertuples(index=False):
            stored[(int(r.league_id), str(r.season))] = (str(r.status), AL.int_or_none(r.n_finished_at_fetch))
    seasons, incomplete = [], []
    for r in ls.itertuples(index=False):
        k = (int(r.league_id), str(r.season))
        if k not in stored:
            seasons.append(r)
        elif stored[k][0] == "incomplete":
            incomplete.append(k)
        elif stored[k][1] is None or int(r.n_finished) > stored[k][1]:
            seasons.append(r)                               # open season with newly finished games
    done_fx = AL.done_keys(FIXTURE_JOURNAL, ["fixture_id"])
    rf = AL.registry_fixtures(fx)
    inc = set(incomplete)
    fallback = rf[[(int(a), str(b)) in inc for a, b in zip(rf["league_id"], rf["season"])]] \
        if len(rf) else rf
    fallback = fallback[~fallback["fixture_id"].astype(str).map(lambda x: (x,) in done_fx)] \
        if len(fallback) else fallback
    fallback = fallback.sort_values("date", ascending=False, kind="mergesort")
    cov_todo = AL.coverage_todo({(int(a), str(b)) for a, b in zip(ls["league_id"], ls["season"])})
    return {"league_seasons": ls, "seasons": seasons, "fallback": fallback, "coverage": cov_todo}


def _flush(rows, sj, fj):
    if rows:                                   # rows first: the journals decide what is asked again
        append_parquet(ROWS, rows, ROW_KEY, ROW_COLS)
    if sj:
        append_parquet(SEASON_JOURNAL, sj, ["league_id", "season"], SJ_COLS)
    if fj:
        append_parquet(FIXTURE_JOURNAL, fj, ["fixture_id"], FJ_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None) -> dict:
    p = plan(fx)
    stats = {"league_seasons": int(len(p["league_seasons"])),
             "planned_coverage_calls": len(p["coverage"]),
             "planned_league_season_calls": len(p["seasons"]),
             "planned_fixture_fallback_calls": int(len(p["fallback"])),
             "fixture_fallback_note": "per-fixture calls are planned only for a league-season whose "
                                      "stored answer is incomplete; unknown until it is asked",
             "calls": 0, "entries_stored": 0, "foreign_entries_skipped": 0, "incomplete_seasons": 0}
    stats["planned_calls_total"] = (stats["planned_coverage_calls"] + stats["planned_league_season_calls"]
                                    + stats["planned_fixture_fallback_calls"])
    print(f"[injuries] {json.dumps({k: v for k, v in stats.items() if k != 'fixture_fallback_note'})}")
    if dry_run:
        return stats
    client = client or AFClient()
    rows, sj, fj = [], [], []
    try:
        stats["calls"] += AL.ensure_coverage(client, p["coverage"])
        cov = AL.coverage_map("injuries")
        for r in p["seasons"][: limit or None]:
            body = client.get("injuries", {"league": int(r.league_id), "season": str(r.season)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            c = completeness(body)
            got, foreign = parse_entries(body, r.league_id, r.season, "league_season")
            stats["foreign_entries_skipped"] += foreign
            status = "incomplete" if not c["complete"] else ("ok" if got else "empty")
            if status == "incomplete":
                stats["incomplete_seasons"] += 1
                print(f"  [incomplete] league {r.league_id} season {r.season}: {c}; per-fixture next")
            rows.extend(got)
            stats["entries_stored"] += len(got)
            sj.append({"league_id": int(r.league_id), "season": str(r.season), "status": status,
                       **{k: c[k] for k in ("results", "n_entries", "paging_current", "paging_total")},
                       "n_finished_at_fetch": int(r.n_finished),
                       "coverage_injuries": cov.get((int(r.league_id), str(r.season))),
                       "fetched_at": AL.now_iso()})
            if len(rows) >= FLUSH_EVERY * 10 or len(sj) >= 20:
                _flush(rows, sj, fj)
                rows, sj = [], []
        _flush(rows, sj, fj)
        rows, sj = [], []
        # fixtures of league-seasons found incomplete (this run's included)
        fb = plan(fx)["fallback"]
        for f in fb.itertuples(index=False):
            body = client.get("injuries", {"fixture": int(f.fixture_id)})
            if body is None:
                continue
            stats["calls"] += 1
            got, foreign = parse_entries(body, f.league_id, f.season, "fixture")
            stats["foreign_entries_skipped"] += foreign
            rows.extend(got)
            stats["entries_stored"] += len(got)
            fj.append({"fixture_id": int(f.fixture_id), "league_id": int(f.league_id),
                       "season": str(f.season), "n_entries": len(got), "fetched_at": AL.now_iso()})
            if len(fj) >= FLUSH_EVERY:
                _flush(rows, sj, fj)
                rows, fj = [], []
    except BudgetStop as e:
        print(f"[injuries] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush(rows, sj, fj)
        client.flush()
    print(f"[injuries] calls {stats['calls']:,}; entries stored {stats['entries_stored']:,}; "
          f"incomplete league-seasons {stats['incomplete_seasons']}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /injuries per league-season")
    ap.add_argument("--limit", type=int, default=None, help="at most N league-season calls")
    ap.add_argument("--dry-run", action="store_true", help="print the planned call count; no call")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
