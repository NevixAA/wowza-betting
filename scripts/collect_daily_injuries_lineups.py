"""
collect_daily_injuries_lineups.py - the DAILY incremental injuries + lineups collection (owner decision
2026-11-15, "Schedule daily"). Workflow: .github/workflows/daily_injuries_lineups.yml.

    python scripts/collect_daily_injuries_lineups.py --dry-run          # planned calls, no call
    python scripts/collect_daily_injuries_lineups.py                    # the daily run
    python scripts/collect_daily_injuries_lineups.py --limit-seasons 3 --limit-prematch 5 \
        --limit-lineups 10                                              # a capped test run

Until 2026-11-15 injuries.parquet / lineups were written only by the run-day steps 11 and 14 of
data_truth_collect.yml (last run 2026-11-08), so every newer match read injuries and `started` as
missing. Each daily run asks ONLY what is not stored yet or is due a refresh:

  1. FIXTURE LISTS  scripts/collect_competition_fixtures.py: /leagues?id per competition (52) + one
     /fixtures?league&season per OPEN season (ended seasons are never re-read). New finished fixtures
     enter output/strength/competition_fixtures.parquet; the not-yet-played ones are kept in
     output/collect/upcoming_fixtures.parquet (same answer, no extra call).
  2. COVERAGE       /leagues?id for a league whose coverage journal lacks a planned season (rare).
  3. INJURIES, SEASON  one /injuries?league&season per OPEN registry league-season (season_start <=
     today <= season_end + SEASON_GRACE_D in competition_seasons.parquet) not asked in the last
     REFRESH_H hours - the finished fixtures' injury lists, refreshed as the provider completes them.
     Journal: injuries_seasons.parquet (the existing one; status ok | empty | incomplete).
  4. INJURIES, PRE-KICKOFF  one /injuries?fixture per upcoming registry fixture kicking off within
     PREMATCH_WINDOW_H hours, in a league-season whose provider coverage flag says injuries=True
     (af_league_coverage.parquet), not asked in the last REFRESH_H hours. Rows source
     'fixture_prematch'; journal injuries_prematch.parquet (fixture_id, league_id, season, kickoff,
     n_entries, fetched_at). Every injuries row carries fetched_at AND first_seen_at (the earliest
     stored fetch, never moved by a re-fetch), so a feature can prove "known before tip time".
  5. LINEUPS        scripts/collect_lineups.py: one /fixtures/lineups?fixture per FINISHED registry
     fixture not in the journal (the backlog is 0 since 2026-11-09, so this is the newly finished
     fixtures only), newest first, seasons the provider flags without lineups skipped. Every team and
     player row carries fetched_at. ANNOUNCED pre-kickoff lineups are the prematch capture's slot
     lineups_45 (scripts/capture_prematch.py, 25..45 min before kickoff): a daily run cannot see them.
     RE-ASKS (2026-11-16): an EMPTY lineups answer is asked again, after the new fixtures, while its last
     ask was made before kickoff date + 7 days and >= 20 h ago (the provider fills lower leagues late;
     collect_lineups.retry_plan). Each re-ask is one call under the same cap and floor; the run record
     counts them (lineups.retry_due / retry_calls / retry_filled / retry_still_empty).

BUDGET (its own, like the prematch capture): at most DAILY_CAP calls per UTC day, journaled in
output/collect/af_budget_daily.json (NOT the run-day job's af_budget.json), and the shared account
floor collect_common.FLOOR (25,000 remaining of the 75,000/day plan) checked before EVERY call.

FAIL LOUD (owner 2026-11-15, same rule as the Pro injury fix): the run exits 1 - after its rows are
stored and committed - when work was DUE and nothing came back:
  * injury league-seasons with injuries coverage were due and 0 entries were stored from them;
  * pre-kickoff fixtures were due and no call succeeded, or >= PREMATCH_MIN_FOR_ZERO_CHECK were
    asked and all came back empty;
  * finished fixtures with provider lineups were due and 0 player rows were stored (FIRST asks only:
    a re-asked empty that is still empty is the provider's answer, not a failure);
  * a budget stop / failed requests left due work with 0 rows.
A partial budget stop with rows stored is a warning; the next run continues (every step is journaled).

REPORT: one JSON line per run in output/collect/daily_collect_runs.jsonl (calls, rows per league,
fetch times, failures) - committed with the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
import collect_common as CC  # noqa: E402
import collect_competition_fixtures as CF  # noqa: E402
import collect_injuries as INJ  # noqa: E402
import collect_lineups as LU  # noqa: E402
from collect_common import COLLECT_DIR, AFClient, BudgetStop, append_parquet, read_parquet  # noqa: E402

BUDGET_FILE = COLLECT_DIR / "af_budget_daily.json"
#: Calls per UTC day (hard). Estimate 2026-11-15 (--dry-run + the stored lists of 2026-10-01..11-09):
#: fixture lists 52 + ~45 open seasons, injuries ~42 open league-seasons + ~10-60 pre-kickoff fixtures
#: (14 open league-seasons have injuries coverage; 23 finished fixtures/day on average in them, 56 at most), lineups ~54 finished fixtures/day on average and
#: ~150-250 on a busy weekend day: ~200-450 calls. Lineup re-asks of empty answers (2026-11-16) add one
#: call per empty per daily run for about 7 days after kickoff: 26 empties stored for 2026 seasons on 2026-11-15
#: (leagues 43 / 180 / 183 / 184), i.e. 26 re-ask calls on a day they are all due. 1,500 is > 3x the busiest estimate.
DAILY_CAP = 1_500
FLOOR = CC.FLOOR
REFRESH_H = 20
PREMATCH_WINDOW_H = 26
SEASON_GRACE_D = 3
PREMATCH_MIN_FOR_ZERO_CHECK = 10
PREMATCH_JOURNAL = COLLECT_DIR / "injuries_prematch.parquet"
PJ_COLS = ["fixture_id", "league_id", "season", "kickoff", "n_entries", "fetched_at"]
RUN_LOG = COLLECT_DIR / "daily_collect_runs.jsonl"


def open_seasons_all(now: datetime) -> int:
    """Open competition-seasons of the fixture-list journal (every competition step 1 reads)."""
    j = read_parquet(CF.JOURNAL)
    if j.empty:
        return 0
    today = pd.Timestamp(now.date())
    st, en = pd.to_datetime(j["season_start"], errors="coerce"), pd.to_datetime(j["season_end"], errors="coerce")
    return int(((st <= today) & (en >= today)).sum())


def open_league_seasons(now: datetime, registry_ids: set[int] | None = None) -> list[tuple[int, str]]:
    """Registry league-seasons whose provider season span (competition_seasons.parquet) holds today."""
    j = read_parquet(CF.JOURNAL)
    if j.empty:
        return []
    ids = AL.registry_af_ids() if registry_ids is None else registry_ids
    today = pd.Timestamp(now.date())
    st = pd.to_datetime(j["season_start"], errors="coerce")
    en = pd.to_datetime(j["season_end"], errors="coerce")
    m = (st <= today) & (en + pd.Timedelta(days=SEASON_GRACE_D) >= today) & \
        pd.to_numeric(j["league_id"], errors="coerce").isin(ids)
    return sorted({(int(a), str(b)) for a, b in zip(j.loc[m, "league_id"], j.loc[m, "season"])})


def _recent(path: Path, key: list[str], now: datetime) -> set[tuple]:
    """Keys of `path` asked within the last REFRESH_H hours."""
    d = read_parquet(path)
    if d.empty:
        return set()
    t = pd.to_datetime(d["fetched_at"], errors="coerce", utc=True)
    d = d[t >= pd.Timestamp(now) - pd.Timedelta(hours=REFRESH_H)]
    return {tuple(str(x) for x in r) for r in d[key].itertuples(index=False)}


def plan_injury_seasons(now: datetime, registry_ids: set[int] | None = None) -> list[tuple[int, str]]:
    recent = _recent(INJ.SEASON_JOURNAL, ["league_id", "season"], now)
    return [k for k in open_league_seasons(now, registry_ids) if (str(k[0]), k[1]) not in recent]


def plan_prematch(now: datetime, registry_ids: set[int] | None = None) -> pd.DataFrame:
    """Upcoming registry fixtures kicking off within PREMATCH_WINDOW_H hours in an injuries-covered
    league-season, not asked within REFRESH_H hours; soonest first. A fixture without a provider
    kickoff timestamp is not planned (it cannot be placed in the window)."""
    up = read_parquet(CF.UPCOMING)
    if up.empty:
        return pd.DataFrame(columns=CF.UPCOMING_COLS)
    ids = AL.registry_af_ids() if registry_ids is None else registry_ids
    up = up[pd.to_numeric(up["league_id"], errors="coerce").isin(ids)].copy()
    ko = pd.to_numeric(up["kickoff_ts"], errors="coerce")
    t0 = pd.Timestamp(now).timestamp()
    up = up[(ko > t0) & (ko <= t0 + PREMATCH_WINDOW_H * 3600)]
    cov = AL.coverage_map("injuries")
    up = up[pd.Series([cov.get((int(a), str(b))) is True for a, b in zip(up["league_id"], up["season"])],
                      index=up.index, dtype=bool)]
    recent = _recent(PREMATCH_JOURNAL, ["fixture_id"], now)
    up = up[pd.Series([(str(int(f)),) not in recent for f in up["fixture_id"]], index=up.index, dtype=bool)]
    return up.sort_values(["kickoff_ts", "fixture_id"], kind="mergesort").reset_index(drop=True)


def _finished_counts() -> dict[tuple[int, str], int]:
    rf = AL.registry_fixtures()
    if rf.empty:
        return {}
    g = rf.groupby(["league_id", "season"]).size()
    return {(int(a), str(b)): int(n) for (a, b), n in g.items()}


def collect_injuries(client: AFClient, now: datetime, stats: dict, limit_seasons: int | None,
                     limit_prematch: int | None) -> None:
    cov = AL.coverage_map("injuries")
    seasons = plan_injury_seasons(now)[: limit_seasons if limit_seasons is not None else None]
    fin = _finished_counts()
    s = stats["injuries"]
    s["season_calls_due"] = len(seasons)
    s["season_calls_due_covered"] = sum(1 for k in seasons if cov.get(k) is True)
    pm = plan_prematch(now)
    if limit_prematch is not None:
        pm = pm.head(limit_prematch)
    s["prematch_due"] = int(len(pm))
    rows, sj, pj = [], [], []
    try:
        for lid, season in seasons:
            body = client.get("injuries", {"league": lid, "season": season})
            if body is None:
                s["failed_calls"] += 1
                continue
            s["calls"] += 1
            c = INJ.completeness(body)
            got, foreign = INJ.parse_entries(body, lid, season, "league_season")
            s["foreign_entries_skipped"] += foreign
            status = "incomplete" if not c["complete"] else ("ok" if got else "empty")
            if status == "incomplete":
                s["incomplete_seasons"].append(f"{lid}/{season}")
            rows.extend(got)
            lg = s["per_league"].setdefault(str(lid), {"season_entries": 0, "prematch_fixtures": 0,
                                                       "prematch_entries": 0, "covered": cov.get((lid, season))})
            lg["season_entries"] += len(got)
            if cov.get((lid, season)) is True:
                s["season_entries_covered"] += len(got)
            s["season_entries"] += len(got)
            sj.append({"league_id": lid, "season": season, "status": status,
                       **{k: c[k] for k in ("results", "n_entries", "paging_current", "paging_total")},
                       "n_finished_at_fetch": fin.get((lid, season)),
                       "coverage_injuries": cov.get((lid, season)), "fetched_at": AL.now_iso()})
        if rows:
            INJ.store_rows(rows)
        if sj:
            append_parquet(INJ.SEASON_JOURNAL, sj, ["league_id", "season"], INJ.SJ_COLS)
        rows, sj = [], []

        for f in pm.itertuples(index=False):
            body = client.get("injuries", {"fixture": int(f.fixture_id)})
            if body is None:
                s["failed_calls"] += 1
                continue
            s["calls"] += 1
            s["prematch_asked"] += 1
            got, foreign = INJ.parse_entries(body, f.league_id, f.season, "fixture_prematch")
            s["foreign_entries_skipped"] += foreign
            rows.extend(got)
            s["prematch_entries"] += len(got)
            lg = s["per_league"].setdefault(str(int(f.league_id)), {"season_entries": 0, "prematch_fixtures": 0,
                                                                    "prematch_entries": 0, "covered": True})
            lg["prematch_fixtures"] += 1
            lg["prematch_entries"] += len(got)
            pj.append({"fixture_id": int(f.fixture_id), "league_id": int(f.league_id), "season": str(f.season),
                       "kickoff": f.kickoff, "n_entries": len(got), "fetched_at": AL.now_iso()})
    except BudgetStop as e:
        s["budget_stop"] = str(e)
        stats["budget_stop"] = str(e)
    finally:
        if rows:
            INJ.store_rows(rows)
        if sj:
            append_parquet(INJ.SEASON_JOURNAL, sj, ["league_id", "season"], INJ.SJ_COLS)
        if pj:
            append_parquet(PREMATCH_JOURNAL, pj, ["fixture_id"], PJ_COLS)


def failures(stats: dict) -> list[str]:
    """The fail-loud rule: due work that returned nothing."""
    out = []
    i, lu = stats["injuries"], stats["lineups"]
    if i["season_calls_due_covered"] > 0 and i["season_entries_covered"] == 0:
        out.append(f"injuries: {i['season_calls_due_covered']} covered league-season(s) due, 0 entries stored "
                   f"({i['calls']} call(s) ok, {i['failed_calls']} failed, budget stop: {i.get('budget_stop')})")
    if i["prematch_due"] > 0 and i["prematch_asked"] == 0:
        out.append(f"injuries: {i['prematch_due']} pre-kickoff fixture(s) due, no call succeeded "
                   f"({i['failed_calls']} failed, budget stop: {i.get('budget_stop')})")
    elif i["prematch_asked"] >= PREMATCH_MIN_FOR_ZERO_CHECK and i["prematch_entries"] == 0:
        out.append(f"injuries: {i['prematch_asked']} covered pre-kickoff fixture(s) asked, all empty")
    if lu["due"] > 0 and lu["player_rows"] == 0:
        out.append(f"lineups: {lu['due']} finished fixture(s) with provider lineups due, 0 player rows stored "
                   f"({lu['calls']} call(s), budget stop: {lu.get('budget_stop')})")
    fx = stats["fixtures"]
    if fx.get("budget_stop") and fx["fetched"] == 0 and fx["due"] > 0:
        out.append(f"fixture lists: {stats['fixtures']['due']} open season(s) due, none read "
                   f"(budget stop: {stats['fixtures']['budget_stop']})")
    return out


def dry_run(now: datetime) -> dict:
    """Planned calls from the stored files, no call. The fixture-list step is estimated from the
    journal (it decides its seasons from the provider's /leagues answer during the run)."""
    ids = AL.registry_af_ids()
    open_all = open_seasons_all(now)
    lu = LU.plan(now=now)
    cov_l = AL.coverage_map("lineups")
    lu_due = sum(1 for a, b in zip(lu["todo"]["league_id"], lu["todo"]["season"]) if cov_l.get((int(a), str(b))) is True)
    lu_retry = sum(1 for a, b in zip(lu["retry"]["league_id"], lu["retry"]["season"])
                   if cov_l.get((int(a), str(b))) is True)
    out = {"fixture_lists_leagues_calls": len(CF.competitions()), "fixture_lists_open_season_calls": open_all,
           "injury_season_calls": len(plan_injury_seasons(now, ids)),
           "injury_prematch_calls_from_stored_upcoming": int(len(plan_prematch(now, ids))),
           "upcoming_list_stored": CF.UPCOMING.exists(),
           "lineup_calls_before_list_refresh": lu_due, "lineup_retry_calls": lu_retry,
           "lineup_retry_final_unknown_kickoff": len(lu["retry_final_unknown_kickoff"]),
           "lineup_note": "newly finished fixtures enter after step 1; ~54/day mean Oct-Nov 2026",
           "daily_cap": DAILY_CAP, "floor": FLOOR}
    print(f"[daily] dry-run: {json.dumps(out)}")
    return out


def run(limit_seasons: int | None = None, limit_prematch: int | None = None, limit_lineups: int | None = None,
        skip_fixtures: bool = False, client: AFClient | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    client = client or AFClient(budget_file=BUDGET_FILE, daily_cap=DAILY_CAP, floor=FLOOR)
    stats = {"started_at": now.isoformat(), "limits": {"seasons": limit_seasons, "prematch": limit_prematch,
                                                        "lineups": limit_lineups, "skip_fixtures": skip_fixtures},
             "fixtures": {}, "injuries": {"calls": 0, "failed_calls": 0, "season_calls_due": 0,
                                          "season_calls_due_covered": 0, "season_entries": 0,
                                          "season_entries_covered": 0, "prematch_due": 0, "prematch_asked": 0,
                                          "prematch_entries": 0, "foreign_entries_skipped": 0,
                                          "incomplete_seasons": [], "per_league": {}},
             "lineups": {}}
    # 1. fixture lists (finished + upcoming)
    if not skip_fixtures:
        due = open_seasons_all(now)
        st = CF.run(client=client)
        stats["fixtures"] = {"fetched": st["fetched"], "finished_rows": st.get("rows"),
                             "budget_stop": st.get("budget_stop"), "due": due}
    else:
        stats["fixtures"] = {"skipped": True, "fetched": 0, "due": 0}
    # 2. coverage for planned league-seasons the journal lacks
    try:
        keys = set(open_league_seasons(now))
        up = read_parquet(CF.UPCOMING)
        if not up.empty:
            keys |= {(int(a), str(b)) for a, b in zip(up["league_id"], up["season"])
                     if int(a) in AL.registry_af_ids()}
        stats["coverage_calls"] = AL.ensure_coverage(client, AL.coverage_todo(keys))
    except BudgetStop as e:
        stats["budget_stop"] = str(e)
    # 3 + 4. injuries
    collect_injuries(client, now, stats, limit_seasons, limit_prematch)
    # 5. lineups (finished fixtures not in the journal, then empty answers due a re-ask)
    p = LU.plan(now=now)
    cov_l = AL.coverage_map("lineups")
    due = sum(1 for a, b in zip(p["todo"]["league_id"], p["todo"]["season"]) if cov_l.get((int(a), str(b))) is True)
    retry_due = sum(1 for a, b in zip(p["retry"]["league_id"], p["retry"]["season"])
                    if cov_l.get((int(a), str(b))) is True)
    if limit_lineups is not None:
        due = min(due, limit_lineups)
        retry_due = min(retry_due, max(0, limit_lineups - due))
    lu = LU.run(limit=limit_lineups, client=client, now=now)
    stats["lineups"] = {"due": due, "calls": lu["calls"], "fixtures_stored": lu["fetched"],
                        "player_rows": lu["player_rows"], "per_league": lu["per_league"],
                        "retry_due": retry_due, "retry_calls": lu["retry_calls"],
                        "retry_filled": lu["retry_filled"], "retry_still_empty": lu["retry_still_empty"],
                        "retry_player_rows": lu["retry_player_rows"], "retry_filled_ids": lu["retry_filled_ids"],
                        "retry_final_unknown_kickoff": lu["retry_final_unknown_kickoff"],
                        "skipped": lu["skipped"], "budget_stop": lu.get("budget_stop")}
    client.flush()
    stats["calls_this_run"] = client.calls_run
    stats["account_remaining_after"] = client.remaining
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["failures"] = failures(stats)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stats, default=str) + "\n")
    print(f"[daily] {json.dumps(stats, default=str)}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="daily incremental injuries + lineups collection")
    ap.add_argument("--dry-run", action="store_true", help="planned calls from the stored files; no call")
    ap.add_argument("--limit-seasons", type=int, default=None, help="at most N injury league-season calls")
    ap.add_argument("--limit-prematch", type=int, default=None, help="at most N pre-kickoff injury calls")
    ap.add_argument("--limit-lineups", type=int, default=None, help="at most N lineup calls")
    ap.add_argument("--skip-fixtures", action="store_true", help="do not refresh the fixture lists")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    if a.dry_run:
        dry_run(now)
        return 0
    st = run(a.limit_seasons, a.limit_prematch, a.limit_lineups, a.skip_fixtures, now=now)
    for f in st["failures"]:
        print(f"::error title=Daily injuries/lineups collect::{f}")
    if st.get("budget_stop") and not st["failures"]:
        print(f"::warning title=Daily injuries/lineups collect::budget stop ({st['budget_stop']}); "
              "rows stored, the next run continues")
    return 1 if st["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
