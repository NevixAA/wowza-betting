"""
Shared plan inputs for the API-Football history collectors of owner decision 2026-11-04 (injuries,
lineups, coaches, transfers: data fetched LIVE only until now, never stored historically). Imported
by scripts/collect_injuries.py, collect_lineups.py, collect_coaches.py, collect_transfers.py; not a
collector itself.

WHERE THE PLAN COMES FROM (no typed league or season list):
  * leagues: src.league_registry entries with an api_football_id;
  * league-seasons and fixtures: the fixture lists step 1 already collected
    (output/strength/competition_fixtures.parquet: FINISHED fixtures only, the provider's own season
    label, restricted by collect_competition_fixtures to seasons overlapping the history we hold);
  * season end / open season: output/collect/competition_seasons.parquet (the provider's own dates);
  * team ids: every home / away id in the fixture lists (continental cup clubs included).
Identity is the provider's own league / fixture / team / player id throughout; no name is resolved.

COVERAGE (/leagues?id, ONE call per league, journaled in output/collect/af_league_coverage.parquet):
  the provider's per-season coverage flags for injuries and lineups (and fixture / player
  statistics, events, for the record). af_stats_coverage.parquet (round 6) stores only
  statistics_fixtures, so it cannot answer the injuries / lineups question; this file is separate so
  that the round-6 journal is never rewritten. A flag the provider does not give stays None.

TRANSFER-WINDOW REFRESH (coaches, transfers; owner-approved cadence, fix8/history2 2026-11-04).
  Until now a stored team was never asked again, so a coach sacked or a player signed after the first
  answer never reached the store. Now every team is asked ONCE MORE after each transfer window
  closes: the windows close about 2 September and 3 February for the European calendar; a
  calendar-year league uses its own closes when its registry entry holds them
  (`transfer_window_closes`: ["MM-DD", ...]), otherwise the same two dates (owner decision - not a
  guess). A team in several leagues takes the latest close that has passed. A team is due when it
  was never asked ('initial') or its latest answer predates that close (refresh_window = the close's
  ISO date). The journal gains one row per (team, refresh_window) and the answers are only ever
  ADDED (see collect_coaches / collect_transfers): about 1,645 calls per refresh, every one through
  AFClient (budget-guarded, resumable - a budget stop leaves the rest due).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import COLLECT_DIR, AFClient, append_parquet, read_parquet  # noqa: E402

FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
SEASONS = COLLECT_DIR / "competition_seasons.parquet"
COVERAGE = COLLECT_DIR / "af_league_coverage.parquet"
FINISHED = {"FT", "AET", "PEN"}
COVERAGE_FLAGS = {"injuries": ("injuries",), "lineups": ("fixtures", "lineups"),
                  "events": ("fixtures", "events"),
                  "statistics_fixtures": ("fixtures", "statistics_fixtures"),
                  "statistics_players": ("fixtures", "statistics_players")}
COVERAGE_COLS = ["league_id", "season", *COVERAGE_FLAGS, "fetched_at"]
#: owner-approved transfer-window closes (month, day) of the European calendar, 2026-11-04
WINDOW_CLOSES = ((9, 2), (2, 3))
INITIAL = "initial"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def int_or_none(v):
    try:
        return None if v is None or v != v else int(v)
    except (TypeError, ValueError):
        return None


def registry_af_ids() -> set[int]:
    """api_football_id of every registry league that has one (club leagues and club cups alike;
    owner 2026-11-04: cups are collected for player data even though they are not tipped)."""
    from src import league_registry as LR
    return {int(e["api_football_id"]) for e in LR.leagues() if e.get("api_football_id") is not None}


def fixture_list(path: Path | None = None) -> pd.DataFrame:
    """The collected fixture list, finished fixtures only, one row per fixture id."""
    fx = read_parquet(path or FIXTURES)
    if fx.empty:
        return pd.DataFrame(columns=["fixture_id", "league_id", "season", "date", "home_id", "away_id"])
    fx = fx[fx["status"].isin(FINISHED)] if "status" in fx.columns else fx
    fx = fx.dropna(subset=["fixture_id", "league_id"]).copy()
    fx["fixture_id"] = fx["fixture_id"].astype(int)
    fx["league_id"] = fx["league_id"].astype(int)
    fx["season"] = fx["season"].astype(str)
    return fx.drop_duplicates("fixture_id", keep="last").reset_index(drop=True)


def registry_fixtures(fx: pd.DataFrame | None = None, ids: set[int] | None = None) -> pd.DataFrame:
    fx = fixture_list() if fx is None else fx
    ids = registry_af_ids() if ids is None else ids
    return fx[fx["league_id"].isin(ids)].reset_index(drop=True)


def team_ids(fx: pd.DataFrame | None = None) -> list[int]:
    """Every provider team id in the fixture lists, most recently active first (they feed the live
    features, so a budget stop leaves the old, inactive clubs for the next run-day)."""
    fx = fixture_list() if fx is None else fx
    if fx.empty:
        return []
    t = pd.concat([fx[["home_id", "date"]].rename(columns={"home_id": "team_id"}),
                   fx[["away_id", "date"]].rename(columns={"away_id": "team_id"})], ignore_index=True)
    t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce")
    t = t.dropna(subset=["team_id"])
    last = t.groupby("team_id")["date"].max().reset_index()
    last = last.sort_values(["date", "team_id"], ascending=[False, True], kind="mergesort")
    return [int(x) for x in last["team_id"]]


def _league_closes() -> dict[int, tuple]:
    """{api_football_id: ((month, day), ...)} for registry entries that hold their own
    `transfer_window_closes`; every other league uses WINDOW_CLOSES."""
    from src import league_registry as LR
    out = {}
    for e in LR.leagues():
        raw, af = e.get("transfer_window_closes"), e.get("api_football_id")
        if not raw or af is None:
            continue
        try:
            out[int(af)] = tuple((int(s[:2]), int(s[3:5])) for s in raw)
        except (TypeError, ValueError):
            print(f"[refresh] {e.get('name')}: transfer_window_closes {raw!r} unreadable - the "
                  f"European closes are used")
    return out


def last_close(closes, today) -> pd.Timestamp:
    """The latest window close on or before `today` among (month, day) closes."""
    today = pd.Timestamp(today).normalize()
    cands = [pd.Timestamp(year=y, month=m, day=d) for y in (today.year, today.year - 1) for m, d in closes]
    return max(c for c in cands if c <= today)


def team_last_close(fx: pd.DataFrame | None = None, today=None,
                    league_closes: dict | None = None) -> dict[int, pd.Timestamp]:
    """{team_id: the latest passed window close of any league the team plays in}."""
    fx = fixture_list() if fx is None else fx
    today = pd.Timestamp.now(tz="UTC").tz_localize(None) if today is None else pd.Timestamp(today)
    lc = _league_closes() if league_closes is None else league_closes
    out: dict[int, pd.Timestamp] = {}
    if fx.empty:
        return out
    t = pd.concat([fx[["home_id", "league_id"]].rename(columns={"home_id": "team_id"}),
                   fx[["away_id", "league_id"]].rename(columns={"away_id": "team_id"})], ignore_index=True)
    t = t.dropna().drop_duplicates()
    for tid, lid in zip(t["team_id"].astype(int), t["league_id"].astype(int)):
        c = last_close(lc.get(lid, WINDOW_CLOSES), today)
        if tid not in out or c > out[tid]:
            out[tid] = c
    return out


def refresh_todo(journal: Path, fx: pd.DataFrame | None = None, today=None,
                 league_closes: dict | None = None) -> list[tuple[int, str]]:
    """[(team_id, refresh_window)] due now, never-asked teams first, each part most recently active
    first (team_ids order). See the module docstring."""
    fx = fixture_list() if fx is None else fx
    teams = team_ids(fx)
    closes = team_last_close(fx, today, league_closes)
    j = read_parquet(journal)
    last = {}
    if not j.empty:
        t = pd.to_datetime(j["fetched_at"], errors="coerce", utc=True).dt.tz_localize(None)
        last = pd.Series(t.to_numpy(), index=j["team_id"].astype(int)).groupby(level=0).max().to_dict()
    initial = [(tid, INITIAL) for tid in teams if tid not in last]
    due = [(tid, closes[tid].date().isoformat()) for tid in teams
           if tid in last and tid in closes and (pd.isna(last[tid]) or last[tid] < closes[tid])]
    return initial + due


def done_keys(path: Path, cols: list[str]) -> set[tuple]:
    """Keys already in a journal parquet (an empty answer included)."""
    j = read_parquet(path)
    if j.empty or not set(cols) <= set(j.columns):
        return set()
    return {tuple(r) for r in j[cols].astype(str).itertuples(index=False)}


# ── /leagues coverage ───────────────────────────────────────────────────────────────────────────

def parse_league_coverage(body: dict | None, league_id: int) -> list[dict]:
    """/leagues?id -> one row per season with the provider's coverage flags (None when absent)."""
    rows, ts = [], now_iso()
    for item in (body or {}).get("response") or []:
        for s in item.get("seasons") or []:
            if s.get("year") is None:
                continue
            cov = s.get("coverage") or {}
            row = {"league_id": int(league_id), "season": str(s["year"]), "fetched_at": ts}
            for col, path in COVERAGE_FLAGS.items():
                v = cov
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                row[col] = bool(v) if isinstance(v, bool) else None
            rows.append(row)
    return rows


def coverage_todo(league_seasons: set[tuple[int, str]], path: Path | None = None) -> list[int]:
    """League ids whose coverage journal lacks one of the planned seasons (one call each)."""
    have = done_keys(path or COVERAGE, ["league_id", "season"])
    return sorted({lid for lid, s in league_seasons if (str(lid), str(s)) not in have})


def ensure_coverage(client: AFClient, league_ids: list[int], path: Path | None = None) -> int:
    """Ask /leagues?id for each league id; journal the answer. Returns the calls made."""
    calls = 0
    for lid in league_ids:
        body = client.get("leagues", {"id": int(lid)})
        if body is None:
            continue                                      # failed: nothing recorded, asked next run
        calls += 1
        rows = parse_league_coverage(body, lid)
        if rows:
            append_parquet(path or COVERAGE, rows, ["league_id", "season"], COVERAGE_COLS)
    return calls


def coverage_map(flag: str, path: Path | None = None) -> dict[tuple[int, str], bool | None]:
    cov = read_parquet(path or COVERAGE)
    if cov.empty or flag not in cov.columns:
        return {}
    out = {}
    for lid, s, v in zip(cov["league_id"], cov["season"], cov[flag]):
        out[(int(lid), str(s))] = None if v is None or v != v else bool(v)
    return out
