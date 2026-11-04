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
