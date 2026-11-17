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
        real empty answer), fetched_at. ONE row per fixture: the LATEST answer (a re-ask replaces it).
  lineups/audit/attempts.parquet  the audit log, append-only (2026-11-16): one row per /fixtures/lineups answer
        stored from that date on - fixture_id, league_id, season, n_teams, fetched_at, ask ('first' |
        'retry' | 'journal' = the answer stored before this log existed, copied from the journal when its
        fixture is first re-asked), kickoff_date. Key (fixture_id, fetched_at): nothing is overwritten.
  lineups/teams_<league_id>_<season>.parquet    one row per (fixture_id, team_id): formation,
        coach_id, coach_name, n_start, n_subs. A formation / coach the provider does not give stays None.
  lineups/players_<league_id>_<season>.parquet  one row per (fixture_id, team_id, role, slot):
        player_id, player_name, number, pos, grid, role ('start' | 'sub'), slot (order in the
        provider's list). A player without a provider id keeps player_id None (never matched by name).
  fetched_at on every team and player row (2026-11-15): the time of the /fixtures/lineups answer the
        row came from (rows stored before that date took it from their fixture's journal row,
        backfill_fetched_at(); a fixture without a journal row keeps None).
ONE FILE PER LEAGUE-SEASON (fix8/history2, 2026-11-04). The full plan is about 119k fixtures x ~40
player rows: as ONE parquet (lineups_players.parquet, the round-7 layout) it would grow towards
GitHub's 100 MB file limit and every flush rewrote the whole file. A league-season holds at most a
few hundred fixtures (a few MB), so no file can reach the limit, and a flush rewrites only the
league-seasons it touched. The journal keeps its single file (5 small columns per fixture) and its
semantics: written LAST, it alone decides what is asked again. read_lineups() is the reader: it
reads every partition, plus a round-7 single file if one exists (a partition row wins on the same
key); nothing is deleted or moved.
EMPTY ANSWERS ARE RE-ASKED WHILE RECENT (2026-11-16). The daily collector asks a fixture right after it
finishes, and lower leagues often get their lineups filled by the provider hours or days later (run
43773536934: 11 league-43 fixtures of 2026-11-14 came back empty). An empty answer (n_teams 0) is
asked again when its LAST ask was made before kickoff + RETRY_WINDOW_D days and at least
RETRY_SPACING_H hours ago. An empty whose last ask was made after that window is FINAL (the
2018-2022 empties were asked years after the match: final). KICKOFF is the provider's kickoff date
of the fixture list (competition_fixtures.parquet `date` = fixture.date[:10]; the list keeps no
time), read as 00:00 UTC of that date, so the window can only end EARLIER than kickoff + 7 days,
never later. A fixture without a known kickoff date has its empty treated as final and is logged
(stats 'retry_final_unknown_kickoff', empties asked in the last RETRY_WINDOW_D days only). Retries
come AFTER the never-asked fixtures (a budget stop leaves retries, not new fixtures, for the next
run) and pass through the same client budget (daily cap + account floor). A retry's answer replaces
the journal row (filled or still empty); every answer is kept in lineups/audit/attempts.parquet.
Collection only: no model feature reads these yet (they come with the Phase 2 retrain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, read_parquet)

JOURNAL = COLLECT_DIR / "lineups_fixtures.parquet"
LINEUPS_DIR = COLLECT_DIR / "lineups"
#: the round-7 single-file layout: read if present, never written
LEGACY = {"teams": COLLECT_DIR / "lineups_teams.parquet", "players": COLLECT_DIR / "lineups_players.parquet"}
J_COLS = ["fixture_id", "league_id", "season", "n_teams", "fetched_at"]
T_COLS = ["fixture_id", "team_id", "formation", "coach_id", "coach_name", "n_start", "n_subs", "fetched_at"]
P_COLS = ["fixture_id", "team_id", "role", "slot", "player_id", "player_name", "number", "pos", "grid",
          "fetched_at"]
COLS = {"teams": T_COLS, "players": P_COLS}
KEYS = {"teams": ["fixture_id", "team_id"], "players": ["fixture_id", "team_id", "role", "slot"]}
A_COLS = ["fixture_id", "league_id", "season", "n_teams", "fetched_at", "ask", "kickoff_date"]
A_KEY = ["fixture_id", "fetched_at"]
#: an empty answer is re-asked while its last ask was made before kickoff + RETRY_WINDOW_D days ...
RETRY_WINDOW_D = 7
#: ... and at least RETRY_SPACING_H hours after that last ask (the daily run is 24 h apart)
RETRY_SPACING_H = 20


def attempts_path(root: Path | None = None) -> Path:
    """lineups/audit/attempts.parquet: every stored answer, append-only. A sub-folder, so no reader or
    check that globs the teams_/players_ partitions (lineups/*.parquet) ever sees it."""
    return (root or LINEUPS_DIR) / "audit" / "attempts.parquet"


def _safe(season) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in str(season))


def partition_path(kind: str, league_id: int, season: str, root: Path | None = None) -> Path:
    """lineups/<kind>_<league_id>_<season>.parquet (kind 'teams' | 'players')."""
    return (root or LINEUPS_DIR) / f"{kind}_{int(league_id)}_{_safe(season)}.parquet"


def read_lineups(kind: str, league_id: int | None = None, season: str | None = None,
                 root: Path | None = None, legacy: Path | None = None) -> pd.DataFrame:
    """Every stored lineup row of `kind` ('teams' | 'players'), optionally one league / season, from
    the partitions (+ a round-7 single file if present; a partition row wins on the same key)."""
    root = root or LINEUPS_DIR
    pat = f"{kind}_{int(league_id) if league_id is not None else '*'}_" \
          f"{_safe(season) if season is not None else '*'}.parquet"
    parts = [pd.read_parquet(p) for p in sorted(root.glob(pat))] if root.exists() else []
    old = Path(legacy or LEGACY[kind])
    if old.exists():
        o = pd.read_parquet(old)
        if league_id is not None or season is not None:
            j = read_parquet(JOURNAL)
            sel = j
            if league_id is not None:
                sel = sel[sel["league_id"].astype(int) == int(league_id)]
            if season is not None:
                sel = sel[sel["season"].astype(str) == str(season)]
            o = o[o["fixture_id"].isin(set(sel["fixture_id"]))]
        parts.insert(0, o)
    if not parts:
        return pd.DataFrame(columns=COLS[kind])
    out = pd.concat([p.reindex(columns=COLS[kind]) for p in parts], ignore_index=True)
    return out.drop_duplicates(KEYS[kind], keep="last").reset_index(drop=True)


def parse(body: dict | None, fixture_id: int, fetched_at: str | None = None) -> tuple[list[dict], list[dict]]:
    teams, players = [], []
    ts = fetched_at or AL.now_iso()
    for t in (body or {}).get("response") or []:
        tid = AL.int_or_none((t.get("team") or {}).get("id"))
        coach = t.get("coach") or {}
        start, subs = t.get("startXI") or [], t.get("substitutes") or []
        teams.append({"fixture_id": int(fixture_id), "team_id": tid, "formation": t.get("formation"),
                      "coach_id": AL.int_or_none(coach.get("id")), "coach_name": coach.get("name"),
                      "n_start": len(start), "n_subs": len(subs), "fetched_at": ts})
        for role, lst in (("start", start), ("sub", subs)):
            for i, e in enumerate(lst):
                p = (e or {}).get("player") or {}
                players.append({"fixture_id": int(fixture_id), "team_id": tid, "role": role, "slot": i,
                                "player_id": AL.int_or_none(p.get("id")), "player_name": p.get("name"),
                                "number": AL.int_or_none(p.get("number")), "pos": p.get("pos"),
                                "grid": p.get("grid"), "fetched_at": ts})
    return teams, players


def backfill_fetched_at(root: Path | None = None, journal: Path | None = None) -> dict:
    """One-off (2026-11-15): give every stored partition row without fetched_at the fetched_at of its
    fixture's journal row (the time that fixture's answer was stored). No call. A row whose fixture
    has no journal row keeps None. Returns {files_rewritten, rows_filled, rows_left_none}."""
    root = root or LINEUPS_DIR
    j = read_parquet(journal or JOURNAL)
    when = dict(zip(pd.to_numeric(j["fixture_id"]).astype("int64"), j["fetched_at"])) if not j.empty else {}
    out = {"files_rewritten": 0, "rows_filled": 0, "rows_left_none": 0}
    for f in sorted(root.glob("*.parquet")) if root.exists() else []:
        kind = f.name.split("_", 1)[0]
        if kind not in COLS:
            continue
        d = pd.read_parquet(f)
        if "fetched_at" in d.columns and d["fetched_at"].notna().all():
            continue
        if "fetched_at" not in d.columns:
            d["fetched_at"] = None
        miss = d["fetched_at"].isna()
        d["fetched_at"] = d["fetched_at"].astype(object)
        d.loc[miss, "fetched_at"] = pd.to_numeric(d.loc[miss, "fixture_id"]).astype("int64").map(when)
        out["rows_filled"] += int((miss & d["fetched_at"].notna()).sum())
        out["rows_left_none"] += int(d["fetched_at"].isna().sum())
        tmp = f.with_suffix(".tmp.parquet")
        d.reindex(columns=COLS[kind]).to_parquet(tmp, index=False)
        os.replace(tmp, f)
        out["files_rewritten"] += 1
    return out


def retry_plan(journal: pd.DataFrame, rf: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, list[int]]:
    """Empty journal answers due a re-ask at `now` (see the module doc), newest kickoff first, and the
    ids of recent empties whose kickoff date is unknown (final, logged). Columns of the plan:
    fixture_id, league_id, season, date (the provider's kickoff date), prev_fetched_at."""
    cols = ["fixture_id", "league_id", "season", "date", "prev_fetched_at"]
    if journal.empty or "n_teams" not in journal.columns:
        return pd.DataFrame(columns=cols), []
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    e = journal[pd.to_numeric(journal["n_teams"], errors="coerce") == 0].copy()
    if e.empty:
        return pd.DataFrame(columns=cols), []
    e["fixture_id"] = pd.to_numeric(e["fixture_id"]).astype("int64")
    last = pd.to_datetime(e["fetched_at"], errors="coerce", utc=True, format="ISO8601")
    dates = {}
    if len(rf):
        dates = dict(zip(pd.to_numeric(rf["fixture_id"]).astype("int64"), rf["date"]))
    e["date"] = e["fixture_id"].map(dates)
    ko = pd.to_datetime(e["date"], errors="coerce", format="%Y-%m-%d", utc=True)
    unknown = ko.isna() & last.notna() & (last >= now - pd.Timedelta(days=RETRY_WINDOW_D))
    due = ko.notna() & last.notna() & (last < ko + pd.Timedelta(days=RETRY_WINDOW_D)) & \
        (last <= now - pd.Timedelta(hours=RETRY_SPACING_H))
    r = e[due].copy()
    r["prev_fetched_at"] = r["fetched_at"]
    r["season"] = r["season"].astype(str)
    r["league_id"] = pd.to_numeric(r["league_id"]).astype("int64")
    r = r.sort_values(["date", "fixture_id"], ascending=[False, False], kind="mergesort")
    return r[cols].reset_index(drop=True), sorted(int(x) for x in e.loc[unknown, "fixture_id"])


def plan(fx: pd.DataFrame | None = None, now: datetime | None = None) -> dict:
    """never-asked finished fixtures ('todo', newest first) + empty answers due a re-ask ('retry')."""
    now = now or datetime.now(timezone.utc)
    rf = AL.registry_fixtures(fx)
    j = read_parquet(JOURNAL)
    done = AL.done_keys(JOURNAL, ["fixture_id"])
    todo = rf[~rf["fixture_id"].astype(str).map(lambda x: (x,) in done)] if len(rf) else rf
    todo = todo.sort_values(["date", "fixture_id"], ascending=[False, False], kind="mergesort")
    retry, unknown = retry_plan(j, rf, now)
    seasons = {(int(a), str(b)) for a, b in zip(rf["league_id"], rf["season"])}
    return {"registry_fixtures": int(len(rf)), "todo": todo.reset_index(drop=True), "retry": retry,
            "retry_final_unknown_kickoff": unknown, "coverage": AL.coverage_todo(seasons)}


def _flush(tb, pb, jb, ab=None):
    """tb / pb: {(league_id, season): rows}. Partitions first, then the attempts log, the journal LAST:
    it decides what is asked again, so a crash between them re-asks a fixture rather than losing it."""
    for kind, buf in (("players", pb), ("teams", tb)):
        for (lid, season), rows in sorted(buf.items()):
            if rows:
                append_parquet(partition_path(kind, lid, season), rows, KEYS[kind], COLS[kind])
    if ab:
        append_parquet(attempts_path(), ab, A_KEY, A_COLS, keep="first")
    if jb:
        append_parquet(JOURNAL, jb, ["fixture_id"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        fx: pd.DataFrame | None = None, now: datetime | None = None) -> dict:
    """Ask the never-asked fixtures (newest first), then the empty answers due a re-ask; `limit` caps
    both together. 'player_rows' / 'fetched' / per_league 'fixtures' count FIRST asks only; the
    re-asks have their own 'retry_*' counters (the daily fail-loud rule reads first asks only)."""
    p = plan(fx, now)
    cov_known = AL.coverage_map("lineups")
    todo, retry = p["todo"], p["retry"]
    no_cov = sum(1 for a, b in zip(todo["league_id"], todo["season"])
                 if cov_known.get((int(a), str(b))) is False)
    r_no_cov = sum(1 for a, b in zip(retry["league_id"], retry["season"])
                   if cov_known.get((int(a), str(b))) is False)
    stats = {"registry_finished_fixtures": p["registry_fixtures"],
             "planned_coverage_calls": len(p["coverage"]),
             "planned_fixture_calls_upper_bound": int(len(todo)) - no_cov,
             "planned_retry_calls_upper_bound": int(len(retry)) - r_no_cov,
             "skipped_no_provider_lineups_known": no_cov, "calls": 0, "fetched": 0,
             "skipped": {}, "player_rows": 0, "per_league": {},
             "retry_calls": 0, "retry_filled": 0, "retry_still_empty": 0, "retry_player_rows": 0,
             "retry_filled_ids": [], "retry_final_unknown_kickoff": p["retry_final_unknown_kickoff"]}
    stats["planned_calls_total_upper_bound"] = (stats["planned_coverage_calls"]
                                                 + stats["planned_fixture_calls_upper_bound"]
                                                 + stats["planned_retry_calls_upper_bound"])
    print(f"[lineups] {json.dumps({k: v for k, v in stats.items() if k not in ('skipped', 'per_league')})}")
    if p["retry_final_unknown_kickoff"]:
        print(f"[lineups] {len(p['retry_final_unknown_kickoff'])} recent empty answer(s) without a known "
              f"kickoff date in the fixture list: final, not re-asked: {p['retry_final_unknown_kickoff'][:50]}")
    if dry_run:
        return stats
    client = client or AFClient()
    tb, pb, jb, ab = {}, {}, [], []
    work = [(f, False) for f in todo.itertuples(index=False)] + \
           [(f, True) for f in retry.itertuples(index=False)]
    try:
        stats["calls"] += AL.ensure_coverage(client, p["coverage"])
        cov = AL.coverage_map("lineups")
        n = 0
        for f, is_retry in work:
            flag = cov.get((int(f.league_id), str(f.season)))
            if flag is not True:
                tag = "no_provider_lineups" if flag is False else "coverage_unknown"
                tag = f"retry_{tag}" if is_retry else tag
                stats["skipped"][tag] = stats["skipped"].get(tag, 0) + 1
                continue
            if limit is not None and n >= limit:
                break
            n += 1
            body = client.get("fixtures/lineups", {"fixture": int(f.fixture_id)})
            if body is None:
                continue                                   # failed: nothing recorded, asked next run
            stats["calls"] += 1
            ts = AL.now_iso()
            teams, players = parse(body, f.fixture_id, ts)
            part = (int(f.league_id), str(f.season))
            tb.setdefault(part, []).extend(teams)
            pb.setdefault(part, []).extend(players)
            key = {"fixture_id": int(f.fixture_id), "league_id": int(f.league_id), "season": str(f.season)}
            jb.append({**key, "n_teams": len(teams), "fetched_at": ts})
            kd = f.date if isinstance(f.date, str) and f.date else None
            if is_retry:
                # the answer this re-ask replaces in the journal, kept in the log (idempotent on the key)
                ab.append({**key, "n_teams": 0, "fetched_at": f.prev_fetched_at, "ask": "journal",
                           "kickoff_date": kd})
            ab.append({**key, "n_teams": len(teams), "fetched_at": ts, "ask": "retry" if is_retry else "first",
                       "kickoff_date": kd})
            lg = stats["per_league"].setdefault(str(int(f.league_id)), {"fixtures": 0, "player_rows": 0,
                                                                         "retry_calls": 0, "retry_player_rows": 0})
            if is_retry:
                stats["retry_calls"] += 1
                stats["retry_player_rows"] += len(players)
                if teams:
                    stats["retry_filled"] += 1
                    stats["retry_filled_ids"].append(int(f.fixture_id))
                else:
                    stats["retry_still_empty"] += 1
                lg["retry_calls"] += 1
                lg["retry_player_rows"] += len(players)
            else:
                stats["player_rows"] += len(players)
                lg["fixtures"] += 1
                lg["player_rows"] += len(players)
                stats["fetched"] += 1
            if len(jb) >= FLUSH_EVERY:
                _flush(tb, pb, jb, ab)
                tb, pb, jb, ab = {}, {}, [], []
    except BudgetStop as e:
        print(f"[lineups] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush(tb, pb, jb, ab)
        client.flush()
    print(f"[lineups] calls {stats['calls']:,}; fixtures stored {stats['fetched']:,}; re-asked empties "
          f"{stats['retry_calls']:,} (filled {stats['retry_filled']:,}, still empty "
          f"{stats['retry_still_empty']:,}); skipped {stats['skipped']}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /fixtures/lineups per finished fixture")
    ap.add_argument("--limit", type=int, default=None, help="at most N fixture calls")
    ap.add_argument("--dry-run", action="store_true", help="print the planned call count; no call")
    ap.add_argument("--backfill-fetched-at", action="store_true",
                    help="no call: fill fetched_at of stored rows from the journal, then exit")
    a = ap.parse_args(argv)
    if a.backfill_fetched_at:
        print(f"[lineups] backfill fetched_at: {backfill_fetched_at()}")
        return 0
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
