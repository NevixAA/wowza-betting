"""Daily TEAM CARDS for the 17 Standard leagues (owner 2026-11-16, audit gap G6 "Collect team cards daily").

    python scripts/collect_standard_team_cards.py [--max-calls 80] [--dry-run]

WHY. No scheduled job collected yellow/red team cards for the Standard (football-data) leagues:
fd_history drops HY/AY/HR/AR, Pro team_stats dropped them, and af_history_extend covers only the
new-format and AF-only leagues (reports/R_SIDE_DATA_TRUTH_LIVE_2026-11-16.md, section B).

WHAT. For every finished (FT/AET/PEN) fixture of config.STANDARD_FORMAT_LEAGUES played on or after
COLLECT_FROM, one /fixtures/statistics call, read under the owner's null rule (src/af_team_stats.py,
owner 2026-11-04: in a team block the provider ANSWERED, a null count is 0; no answer -> missing; a
type the answered block does not list -> missing). Each row carries the provider fixture_id and
fetched_at (UTC).

COST. One /fixtures?date call per day in the window (3) + one /fixtures/statistics per new finished
fixture (~16/day across the 17 leagues). Stops at --max-calls and when the account has under
MIN_QUOTA requests left. FROM NOW ON ONLY: nothing before COLLECT_FROM is fetched (no backfill).

WHERE. output/af_team_cards_std.parquet, its own file. NOT af_history.parquet: data_loader merges
af_history into the Standard leagues' training rows, so adding rows there would change model inputs.
Append-only: a stored fixture is never re-fetched or rewritten. A fixture whose statistics were not
answered is not stored and is retried on later days while it is inside the window.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import af_team_stats as ATS  # noqa: E402

OUTPUT = ROOT / "output" / "af_team_cards_std.parquet"
_BASE = "https://v3.football.api-sports.io"
COLLECT_FROM = "2026-11-16"          # owner: collect from now on, no backfill
WINDOW_DAYS = 3                       # today and the 2 days before
FINISHED = {"FT", "AET", "PEN"}
MIN_QUOTA = int(os.getenv("MIN_QUOTA", "20000"))
SOURCE = "api-football /fixtures/statistics"

COLUMNS = ["fixture_id", "league", "league_id", "season", "match_date", "kickoff_utc", "status",
           "home_team", "away_team", "home_team_id", "away_team_id", "FTHG", "FTAG",
           "HY", "AY", "HR", "AR", "home_answered", "away_answered", "source", "fetched_at"]


def leagues() -> dict:
    """{af league id: league name} for the Standard leagues."""
    return {int(config.API_FOOTBALL_IDS[lg]): lg for lg in sorted(set(config.STANDARD_FORMAT_LEAGUES))
            if lg in config.API_FOOTBALL_IDS}


def window(today: str, start: str = COLLECT_FROM, days: int = WINDOW_DAYS) -> list[str]:
    t = pd.Timestamp(today)
    out = [(t - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days - 1, -1, -1)]
    return [d for d in out if d >= start]


def parse_row(fx: dict, stats_resp, league: str, fetched_at: str) -> dict | None:
    """One stored row, or None when the provider answered no statistics for either team."""
    f, lg, tm, g = fx.get("fixture") or {}, fx.get("league") or {}, fx.get("teams") or {}, fx.get("goals") or {}
    home, away = tm.get("home") or {}, tm.get("away") or {}
    blocks = {}
    for blk in stats_resp or []:
        tid = (blk.get("team") or {}).get("id")
        if tid is not None:
            blocks[tid] = blk.get("statistics") or []
    hs, as_ = blocks.get(home.get("id")), blocks.get(away.get("id"))
    h_ok, a_ok = ATS.answered(hs), ATS.answered(as_)
    if not (h_ok or a_ok):
        return None
    return {
        "fixture_id": int(f["id"]), "league": league, "league_id": lg.get("id"),
        "season": str(lg.get("season")) if lg.get("season") is not None else None,
        "match_date": str(f.get("date", ""))[:10], "kickoff_utc": f.get("date"),
        "status": (f.get("status") or {}).get("short"),
        "home_team": home.get("name"), "away_team": away.get("name"),
        "home_team_id": home.get("id"), "away_team_id": away.get("id"),
        "FTHG": g.get("home"), "FTAG": g.get("away"),
        "HY": ATS.team_count(hs, "Yellow Cards"), "AY": ATS.team_count(as_, "Yellow Cards"),
        "HR": ATS.team_count(hs, "Red Cards"), "AR": ATS.team_count(as_, "Red Cards"),
        "home_answered": h_ok, "away_answered": a_ok, "source": SOURCE, "fetched_at": fetched_at,
    }


class Client:
    def __init__(self, key: str, max_calls: int):
        self.key, self.max_calls, self.calls, self.stopped = key, max_calls, 0, ""

    def get(self, endpoint: str, params: dict):
        if self.calls >= self.max_calls:
            self.stopped = f"call cap {self.max_calls}"
            return None
        for attempt in range(3):
            self.calls += 1
            try:
                r = requests.get(f"{_BASE}/{endpoint}", headers={"x-apisports-key": self.key},
                                 params=params, timeout=25)
            except Exception as e:                                     # noqa: BLE001
                print(f"  [warn] {endpoint}: {type(e).__name__}")
                time.sleep(2 * (attempt + 1))
                continue
            try:
                left = int(r.headers.get("x-ratelimit-requests-remaining", "999999"))
            except ValueError:
                left = 999999
            if r.status_code == 429:
                time.sleep(20); continue
            if r.status_code != 200:
                print(f"  [warn] {endpoint} HTTP {r.status_code}")
                return None
            body = r.json()
            errs = body.get("errors")
            if errs:
                if isinstance(errs, dict) and "rateLimit" in errs:
                    time.sleep(20); continue
                print(f"  [warn] {endpoint} errors={errs}")
                return None
            if left < MIN_QUOTA:
                self.stopped = f"account quota floor ({left} < {MIN_QUOTA})"
                self.max_calls = self.calls
            time.sleep(0.3)
            return body
        return None


def stored_ids(path: Path | None = None) -> set:
    path = path or OUTPUT
    if not path.exists():
        return set()
    return {int(x) for x in pd.read_parquet(path, columns=["fixture_id"])["fixture_id"]}


def append(rows: list[dict], path: Path | None = None) -> int:
    path = path or OUTPUT
    if not rows:
        return 0
    new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if path.exists():
        old = pd.read_parquet(path)
        new = new[~new["fixture_id"].isin(set(old["fixture_id"].astype(int)))]
        if new.empty:
            return 0
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    for c in ("HY", "AY", "HR", "AR", "FTHG", "FTAG", "league_id", "home_team_id", "away_team_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["match_date", "league", "fixture_id"]).to_parquet(path, index=False)
    return int(len(new))


def run(max_calls: int = 80, dry_run: bool = False, today: str | None = None, client=None) -> dict:
    key = os.getenv("APIFOOTBALL_KEY", "")
    if client is None:
        if not key:
            print("[team_cards] APIFOOTBALL_KEY not set - nothing collected")
            return {"calls": 0, "rows": 0}
        client = Client(key, max_calls)
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lg = leagues()
    have = stored_ids()
    days = window(today)
    print(f"[team_cards] {len(lg)} Standard leagues; window {days or 'empty (before COLLECT_FROM)'}; "
          f"{len(have):,} fixture(s) already stored; call cap {max_calls}")
    todo = []
    for d in days:
        body = client.get("fixtures", {"date": d, "timezone": "UTC"})
        for fx in (body or {}).get("response") or []:
            lid = (fx.get("league") or {}).get("id")
            st = ((fx.get("fixture") or {}).get("status") or {}).get("short")
            fid = (fx.get("fixture") or {}).get("id")
            if lid in lg and st in FINISHED and fid is not None and int(fid) not in have:
                todo.append(fx)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, unanswered = [], 0
    for fx in todo:
        body = client.get("fixtures/statistics", {"fixture": fx["fixture"]["id"]})
        if body is None and client.stopped:
            break
        row = parse_row(fx, (body or {}).get("response"), lg[fx["league"]["id"]], fetched_at)
        if row is None:
            unanswered += 1
            continue
        rows.append(row)
    n = 0 if dry_run else append(rows)
    by_lg = pd.Series([r["league"] for r in rows]).value_counts().to_dict() if rows else {}
    print(f"[team_cards] {len(todo)} new finished fixture(s); {len(rows)} answered, {unanswered} not "
          f"answered (retried later); {'dry-run, nothing written' if dry_run else f'+{n} row(s) -> {OUTPUT.name}'}")
    for k, v in sorted(by_lg.items()):
        print(f"    {k:28} {v}")
    print(f"[team_cards] API calls: {client.calls}" + (f" (stopped: {client.stopped})" if client.stopped else ""))
    return {"calls": client.calls, "rows": n, "todo": len(todo), "unanswered": unanswered}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(max_calls=a.max_calls, dry_run=a.dry_run)
