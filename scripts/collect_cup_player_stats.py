"""
collect_cup_player_stats.py - API-Football /fixtures/players for every UEFA club cup fixture
(Champions League, Europa League, Conference League) in the span player_history covers
(owner question 2026-11-02: cup props tips exist, player_history held zero cup rows).

    python scripts/collect_cup_player_stats.py            # collect what is missing
    python scripts/collect_cup_player_stats.py --dry-run  # print the plan (no API call)

WHICH FIXTURES. The registry's club cups (competition_type 'club_cup' with an api_football_id:
2, 3, 848) as listed in output/strength/competition_fixtures.parquet (scripts/
collect_competition_fixtures.py, step 1 of the same workflow; finished fixtures only), dated on or
after the first match day of player_history. ONE call per fixture. Order: fixtures with a club
whose league player_history holds first (their players are the ones we tip), newest first, so a
budget stop leaves the most useful part done.

OUTPUT output/collect/cup_player_rows.parquet - one row per (fixture_id, player_id) with
player_history's raw columns, parsed by player_model.data_fetcher._parse_fixture_player (the parser
the league collect uses: >= 10 minutes, no API id -> dropped, a missing stat is not invented),
league = the registry cup name, season = the provider's season, team_id / opponent_id = the
API-Football team ids (side by team id), fixture_status = FT / AET / PEN. team_corners and referee
are None in these rows (never typed); they come from the sidecar below.
JOURNAL output/collect/cup_player_fixtures.parquet: one row per fetched fixture (n_rows; an EMPTY
answer is recorded and never asked again); a failed request records nothing and is retried.

CORNERS + REFEREE (owner-approved 2026-11-02, fix6/props6) -> SIDECAR
output/collect/cup_fixture_meta.parquet, one row per cup fixture that has player rows: referee
(/fixtures fixture.referee), home_id / away_id, home_corners / away_corners ("Corner Kicks" of
/fixtures/statistics; a null value stays NaN), corners_status ok / empty / pending. The player rows
are never rewritten; player_model.cup_history.cup_rows() joins the sidecar onto them (a value the
row already carries wins). Calls:
  * /fixtures?ids=<up to 20 ids>   ONE call per 20 fixtures: the referee, and the fixture's
    statistics when the answer embeds them (then no per-fixture call is made);
  * /fixtures/statistics?fixture=  one call per fixture whose /fixtures answer had no statistics
    (corners_status 'pending').
First run: 3,779 cup fixtures in the window (dry run 2026-11-02) -> 3,779 /fixtures/players + 189
/fixtures + at most 3,779 /fixtures/statistics = at most 7,747 calls; then only new fixtures.
Budget: scripts/collect_common.AFClient (checked before every call; a stop is a clean exit 0, the
next run resumes from the journals).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, history_frame, read_parquet)

FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
OUT = COLLECT_DIR / "cup_player_rows.parquet"
JOURNAL = COLLECT_DIR / "cup_player_fixtures.parquet"
FINISHED = {"FT", "AET", "PEN"}
COLS = ["player_id", "player_name", "fixture_id", "date", "league", "home_team", "away_team", "team",
        "is_home", "referee", "season", "position", "minutes", "started", "goals", "assists",
        "shots_total", "shots_on_target", "yellow_cards", "key_passes", "rating", "fouls_committed",
        "fouls_drawn", "duels_total", "duels_won", "team_corners", "red_cards", "passes_total",
        "offsides", "tackles_total", "interceptions", "dribbles_attempted", "dribbles_success",
        "dribbles_past", "goals_conceded", "saves", "penalty_scored", "penalty_missed",
        "penalty_won", "penalty_saved", "team_id", "opponent_id", "fixture_status"]
JOURNAL_COLS = ["fixture_id", "league", "season", "date", "n_rows", "fetched_at"]
META = COLLECT_DIR / "cup_fixture_meta.parquet"
META_COLS = ["fixture_id", "date", "referee", "home_id", "away_id", "home_corners", "away_corners",
             "corners_status", "fetched_at"]
IDS_PER_CALL = 20                     # API-Football /fixtures?ids= maximum


def _corner_value(stats_block) -> float:
    """'Corner Kicks' of one team's statistics list; NaN when absent or null (never 0)."""
    for s in stats_block or []:
        if str(s.get("type", "")).strip().lower() == "corner kicks":
            try:
                return float(s.get("value"))
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def parse_corners(stats: list | None, home_id: int, away_id: int) -> tuple[float, float, str]:
    """(home_corners, away_corners, status) from a /fixtures/statistics answer (or the statistics
    embedded in /fixtures?ids=). Sides by team id; an empty answer -> NaN, NaN, 'empty'."""
    if not stats:
        return float("nan"), float("nan"), "empty"
    by = {int((b.get("team") or {}).get("id") or -1): _corner_value(b.get("statistics")) for b in stats}
    return by.get(int(home_id), float("nan")), by.get(int(away_id), float("nan")), "ok"


def meta_plan(journal: pd.DataFrame, meta: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fixtures needing the /fixtures?ids= call, fixtures needing /fixtures/statistics): the
    journaled cup fixtures WITH player rows that the sidecar lacks / holds as corners 'pending'.
    Newest first."""
    if journal.empty:
        e = pd.DataFrame(columns=["fixture_id", "date", "home_id", "away_id"])
        return e, e
    j = journal[pd.to_numeric(journal["n_rows"], errors="coerce") > 0][["fixture_id", "date"]].copy()
    j["fixture_id"] = pd.to_numeric(j["fixture_id"], errors="coerce")
    sides = fx[["fixture_id", "home_id", "away_id"]].assign(
        fixture_id=pd.to_numeric(fx["fixture_id"], errors="coerce")).drop_duplicates("fixture_id")
    j = j.merge(sides, on="fixture_id", how="inner").sort_values(["date", "fixture_id"], ascending=False)
    held = set(pd.to_numeric(meta["fixture_id"], errors="coerce")) if not meta.empty else set()
    pend = (set(pd.to_numeric(meta.loc[meta["corners_status"] == "pending", "fixture_id"], errors="coerce"))
            if not meta.empty else set())
    return (j[~j["fixture_id"].isin(held)].reset_index(drop=True),
            j[j["fixture_id"].isin(pend)].reset_index(drop=True))


def run_meta(client: AFClient, fx: pd.DataFrame, dry_run: bool = False) -> dict:
    """Referee + corners for the cup fixtures with player rows (see the module docstring)."""
    todo, pending = meta_plan(read_parquet(JOURNAL), read_parquet(META), fx)
    st = {"meta_todo": int(len(todo)), "corners_pending": int(len(pending)), "calls_fixtures": 0,
          "calls_statistics": 0, "referees": 0, "corners_ok": 0}
    n_ids = -(-len(todo) // IDS_PER_CALL)
    print(f"[cups-meta] {len(todo):,} fixture(s) without referee/corners -> {n_ids:,} /fixtures call(s) "
          f"(<= {IDS_PER_CALL} ids each) + up to {len(todo) + len(pending):,} /fixtures/statistics call(s)")
    if dry_run or (todo.empty and pending.empty):
        return st
    buf: list[dict] = []

    def _flush():
        if buf:
            append_parquet(META, buf, ["fixture_id"], META_COLS)
            buf.clear()
    try:
        for i in range(0, len(todo), IDS_PER_CALL):
            chunk = todo.iloc[i:i + IDS_PER_CALL]
            body = client.get("fixtures", {"ids": "-".join(str(int(f)) for f in chunk["fixture_id"])})
            st["calls_fixtures"] += 1
            if body is None:
                continue                                   # failed: nothing recorded, retried next run
            got = {int((r.get("fixture") or {}).get("id") or -1): r for r in body.get("response") or []}
            now = datetime.now(timezone.utc).isoformat()
            for _, f in chunk.iterrows():
                r = got.get(int(f["fixture_id"]))
                if r is None:
                    continue                               # not answered: asked again next run
                ref = (r.get("fixture") or {}).get("referee")
                ref = ref.strip() if isinstance(ref, str) and ref.strip() else None
                if "statistics" in r:
                    hc, ac, cs = parse_corners(r.get("statistics"), f["home_id"], f["away_id"])
                else:
                    hc, ac, cs = float("nan"), float("nan"), "pending"
                buf.append({"fixture_id": int(f["fixture_id"]), "date": str(f["date"])[:10], "referee": ref,
                            "home_id": int(f["home_id"]), "away_id": int(f["away_id"]),
                            "home_corners": hc, "away_corners": ac, "corners_status": cs, "fetched_at": now})
                st["referees"] += ref is not None
                st["corners_ok"] += cs == "ok"
            if len(buf) >= FLUSH_EVERY:
                _flush()
        _flush()
        meta = read_parquet(META)
        pend = meta[meta["corners_status"] == "pending"] if not meta.empty else meta
        for _, m in pend.iterrows():
            body = client.get("fixtures/statistics", {"fixture": int(m["fixture_id"])})
            st["calls_statistics"] += 1
            if body is None:
                continue                                   # stays 'pending', retried next run
            hc, ac, cs = parse_corners(body.get("response"), m["home_id"], m["away_id"])
            buf.append({**{c: m[c] for c in META_COLS}, "home_corners": hc, "away_corners": ac,
                        "corners_status": cs, "fetched_at": datetime.now(timezone.utc).isoformat()})
            st["corners_ok"] += cs == "ok"
            if len(buf) >= FLUSH_EVERY:
                _flush()
    except BudgetStop as e:
        print(f"[cups-meta] budget stop: {e} (resumable)")
        st["budget_stop"] = str(e)
    finally:
        _flush()
    print(f"[cups-meta] {st}")
    return st


def cup_ids() -> dict[int, str]:
    """{API-Football league id: registry cup name} for the registry's club cups."""
    from src import league_registry as LR
    return {int(e["api_football_id"]): e["name"] for e in LR.leagues()
            if e.get("competition_type") == "club_cup" and e.get("api_football_id") is not None}


def plan(fx: pd.DataFrame, cups: dict, start, journal: pd.DataFrame, history_leagues=()) -> pd.DataFrame:
    """The cup fixtures to fetch: finished, dated >= start, not journaled, priority-ordered."""
    if fx.empty or start is None:
        return fx.iloc[0:0]
    f = fx[pd.to_numeric(fx["league_id"], errors="coerce").isin(list(cups))].copy()
    f = f[f["status"].isin(FINISHED)]
    f["_d"] = pd.to_datetime(f["date"], errors="coerce")
    f = f[f["_d"] >= pd.Timestamp(start)]
    if not journal.empty:
        f = f[~pd.to_numeric(f["fixture_id"], errors="coerce").isin(
            pd.to_numeric(journal["fixture_id"], errors="coerce"))]
    ours = set()
    if history_leagues:
        dom = fx[fx["league"].isin(set(history_leagues))]
        ours = set(pd.to_numeric(dom["home_id"], errors="coerce")) | set(pd.to_numeric(dom["away_id"], errors="coerce"))
    f["_ours"] = f["home_id"].isin(ours) | f["away_id"].isin(ours)
    return f.sort_values(["_ours", "_d", "fixture_id"], ascending=[False, False, False]).reset_index(drop=True)


def parse(body: dict | None, fix) -> list[dict]:
    """player_history rows of one /fixtures/players answer for the cup fixture `fix` (a row of
    competition_fixtures). A team block that is neither side by team id is skipped."""
    from player_model.data_fetcher import _parse_fixture_player, fixture_side
    rows = []
    hid, aid = int(fix["home_id"]), int(fix["away_id"])
    fixture = {"teams": {"home": {"id": hid}, "away": {"id": aid}}}
    for block in (body or {}).get("response") or []:
        is_home = fixture_side(block, fixture, fix["home_name"], fix["away_name"])
        if is_home is None:
            print(f"  [fix {fix['fixture_id']}] team block {(block.get('team') or {}).get('name')!r} is "
                  f"neither side - skipped")
            continue
        meta = {"fixture_id": int(fix["fixture_id"]), "date": str(fix["date"])[:10],
                "home_team": fix["home_name"], "away_team": fix["away_name"],
                "team": (block.get("team") or {}).get("name", ""), "is_home": is_home,
                "team_corners": None, "referee": None, "season": str(fix["season"])}
        for entry in block.get("players") or []:
            r = _parse_fixture_player(entry, meta, fix["league"])
            if r:
                r["team_id"], r["opponent_id"] = (hid, aid) if is_home else (aid, hid)
                r["fixture_status"] = fix["status"]
                rows.append(r)
    return rows


def _start_and_leagues():
    h = history_frame(["date", "league"])
    if h.empty:
        return None, ()
    d = pd.to_datetime(h["date"], errors="coerce")
    return (d.min().normalize() if d.notna().any() else None), tuple(h["league"].dropna().unique())


def run(dry_run: bool = False, limit: int | None = None, client: AFClient | None = None) -> dict:
    fx = read_parquet(FIXTURES)
    start, leagues = _start_and_leagues()
    todo = plan(fx, cup_ids(), start, read_parquet(JOURNAL), leagues)
    stats = {"todo": int(len(todo)), "priority": int(todo["_ours"].sum()) if len(todo) else 0,
             "fetched": 0, "rows": 0}
    print(f"[cups] {stats['todo']:,} cup fixture(s) to fetch ({stats['priority']:,} with a club of a "
          f"player_history league) - one API-Football call each; window from {start}")
    if dry_run:
        m = run_meta(None, fx, dry_run=True)
        n = stats["todo"] + m["meta_todo"]                 # every new fixture also needs its meta
        stats["max_calls"] = stats["todo"] + -(-n // IDS_PER_CALL) + n + m["corners_pending"]
        print(f"[cups] plan: at most {stats['max_calls']:,} API-Football call(s) "
              f"({stats['todo']:,} /fixtures/players + {-(-n // IDS_PER_CALL):,} /fixtures + "
              f"<= {n + m['corners_pending']:,} /fixtures/statistics)")
        return stats
    client = client or AFClient()
    calls0 = client.calls_run
    if todo.empty:
        stats.update(run_meta(client, fx))
        stats["api_calls"] = client.calls_run - calls0
        client.flush()
        return stats
    buf, jbuf = [], []

    def _flush():
        if buf:
            append_parquet(OUT, buf, ["fixture_id", "player_id"], COLS)
            buf.clear()
        if jbuf:
            append_parquet(JOURNAL, jbuf, ["fixture_id"], JOURNAL_COLS)
            jbuf.clear()
    try:
        for _, fix in todo.head(limit or len(todo)).iterrows():
            body = client.get("fixtures/players", {"fixture": int(fix["fixture_id"])})
            if body is None:
                continue                                  # failed: nothing recorded, retried next run
            rows = parse(body, fix)
            buf.extend(rows)
            jbuf.append({"fixture_id": int(fix["fixture_id"]), "league": fix["league"],
                         "season": str(fix["season"]), "date": str(fix["date"])[:10],
                         "n_rows": len(rows), "fetched_at": datetime.now(timezone.utc).isoformat()})
            stats["fetched"] += 1
            stats["rows"] += len(rows)
            if len(jbuf) >= FLUSH_EVERY:
                _flush()
    except BudgetStop as e:
        print(f"[cups] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        _flush()
        client.flush()
    if "budget_stop" not in stats:
        stats.update(run_meta(client, fx))
        client.flush()
    stats["api_calls"] = client.calls_run - calls0
    print(f"[cups] {stats}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect UEFA cup player stats per fixture")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
