"""
Side-stats refresh: API-Football /fixtures/statistics for finished fixtures the stats stores lack.
=================================================================================================
Owner 2026-11-18: "keep the verified stats fresh". The side models (src/side_models/) read corners and
cards from the fullstats fill (output/collect/af_history_fullstats_fill.parquet, frozen since
2026-11-09) and, for fixtures it does not hold, from this job's sidecar
output/collect/af_side_stats.parquet. football-data's raw season files (the second source of the
Standard leagues) are refreshed by scripts/collect_fd_raw.py (free, no key) in the same workflow step.

TARGETS: every finished fixture (output/strength/competition_fixtures.parquet, FT/AET/PEN) of a TRACKED
registry league (production_state ENABLED / ENABLED_PROVIDER_UNSUPPORTED / PAPER) since
config/side_models.json single_source.window_from, that neither store answers yet. Newest first, so a
capped run always serves the freshest matches. A recorded EMPTY answer (the provider has no statistics
for the fixture) is asked again only after RETRY_AFTER_H and at most MAX_ATTEMPTS times in all.

PARSING: scripts/collect_af_history_gap.parse_full_stats - by the provider's TEAM IDS, under the owner's
team null rule (src/af_team_stats). Nothing is filled: no answer = missing.

BUDGET (hard, checked before EVERY call): --max-calls for this run (1..MAX_CALLS_CAP, required), the
daily collector's own journal output/collect/af_budget_daily.json with its cap of 1,500 calls per UTC
day (the two share it), and the account floor collect_common.FLOOR (25,000 remaining of 75,000/day).

    python scripts/collect_side_stats.py --dry-run                # targets + planned calls, no API call
    python scripts/collect_side_stats.py --max-calls 150          # the real run (APIFOOTBALL_KEY)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_common as CC  # noqa: E402
from collect_common import COLLECT_DIR, AFClient, BudgetStop, append_parquet, read_parquet  # noqa: E402

SIDECAR = COLLECT_DIR / "af_side_stats.parquet"
RUNS = COLLECT_DIR / "side_stats_runs.jsonl"
FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
FILL = COLLECT_DIR / "af_history_fullstats_fill.parquet"
BUDGET_FILE = COLLECT_DIR / "af_budget_daily.json"
DAILY_CAP = 1_500                 # shared with scripts/collect_daily_injuries_lineups.py (same journal)
MAX_CALLS_CAP = 500               # a single run may never ask for more
RETRY_AFTER_H = 48
MAX_ATTEMPTS = 3
TRACKED = ("ENABLED", "ENABLED_PROVIDER_UNSUPPORTED", "PAPER")
STEMS = ("corners", "yellow", "red")
COLS = (["fixture_id", "league", "date", "home_id", "away_id"]
        + [f"{side}_{s}" for s in STEMS for side in ("home", "away")]
        + ["answered", "attempts", "fetched_at", "source"])


def tracked_leagues() -> dict:
    reg = json.loads((ROOT / "config" / "leagues.json").read_text(encoding="utf-8"))["leagues"]
    return {e["name"]: e for e in reg if e.get("production_state") in TRACKED and e.get("api_football_id")}


def window_from() -> str:
    cfg = json.loads((ROOT / "config" / "side_models.json").read_text(encoding="utf-8"))
    return cfg["single_source"]["window_from"]


def targets(now: datetime | None = None, fixtures: pd.DataFrame | None = None,
            fill: pd.DataFrame | None = None, side: pd.DataFrame | None = None) -> pd.DataFrame:
    """Finished tracked-league fixtures without an answer in either store, newest first."""
    now = now or datetime.now(timezone.utc)
    fx = pd.read_parquet(FIXTURES) if fixtures is None else fixtures
    fx = fx[fx["league"].isin(tracked_leagues()) & fx["status"].isin(("FT", "AET", "PEN"))
            & (fx["date"] >= window_from())].drop_duplicates("fixture_id")
    fill = read_parquet(FILL, columns=["fixture_id"]) if fill is None else fill
    side = read_parquet(SIDECAR) if side is None else side
    have = set(fill["fixture_id"]) if not fill.empty else set()
    t = fx[~fx["fixture_id"].isin(have)].copy()
    if not side.empty:
        ans = set(side.loc[side["answered"].eq(True), "fixture_id"])
        t = t[~t["fixture_id"].isin(ans)]
        last = side.set_index("fixture_id")
        empty = side[side["answered"].eq(False)]
        age_h = (pd.Timestamp(now) - pd.to_datetime(empty["fetched_at"], utc=True)).dt.total_seconds() / 3600
        wait = set(empty.loc[(age_h < RETRY_AFTER_H) | (empty["attempts"] >= MAX_ATTEMPTS), "fixture_id"])
        t = t[~t["fixture_id"].isin(wait)]
        t["attempts_before"] = t["fixture_id"].map(last["attempts"]) if len(last) else None
    return t.sort_values(["date", "fixture_id"], ascending=[False, False]).reset_index(drop=True)


def parse(body: dict | None, home_id: int, away_id: int) -> dict | None:
    """{home_/away_ corners/yellow/red} or None when the answer does not hold both clubs."""
    import collect_af_history_gap as G
    p = G.parse_full_stats((body or {}).get("response") or [], int(home_id), int(away_id))
    if p is None:
        return None
    return {f"{side}_{s}": p.get(f"{side}_{s}") for s in STEMS for side in ("home", "away")}


def run(max_calls: int, *, dry_run: bool = False, client: AFClient | None = None,
        now: datetime | None = None) -> dict:
    if not 1 <= int(max_calls) <= MAX_CALLS_CAP:
        raise SystemExit(f"--max-calls {max_calls} outside [1, {MAX_CALLS_CAP}] - refusing to run")
    now = now or datetime.now(timezone.utc)
    t = targets(now)
    stats = {"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "targets": int(len(t)), "max_calls": int(max_calls),
             "calls": 0, "answered": 0, "empty": 0, "failed": 0, "budget_stop": "",
             "by_league": t["league"].value_counts().to_dict() if len(t) else {}}
    if dry_run:
        stats["planned_calls"] = int(min(len(t), max_calls))
        print(json.dumps(stats, indent=1))
        return stats
    client = client or AFClient(budget_file=BUDGET_FILE, daily_cap=DAILY_CAP, floor=CC.FLOOR)
    side_prev = read_parquet(SIDECAR)
    prev_att = (dict(zip(side_prev["fixture_id"], side_prev["attempts"])) if not side_prev.empty else {})
    rows = []
    try:
        for r in t.head(int(max_calls)).itertuples(index=False):
            body = client.get("fixtures/statistics", {"fixture": int(r.fixture_id)})
            stats["calls"] += 1
            if body is None:                                    # failed request: nothing recorded
                stats["failed"] += 1
                continue
            vals = parse(body, r.home_id, r.away_id)
            att = int(prev_att[r.fixture_id]) + 1 if r.fixture_id in prev_att else 1
            row = {"fixture_id": int(r.fixture_id), "league": r.league, "date": r.date,
                   "home_id": int(r.home_id), "away_id": int(r.away_id), "answered": vals is not None,
                   "attempts": att, "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "source": "api-football /fixtures/statistics"}
            row.update(vals or {})
            rows.append(row)
            stats["answered" if vals is not None else "empty"] += 1
    except BudgetStop as e:
        stats["budget_stop"] = str(e)
        print(f"[side_stats] budget stop: {e} (resumable)")
    finally:
        client.flush()
        if rows:
            append_parquet(SIDECAR, rows, key=["fixture_id"], columns=COLS, keep="last")
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        with RUNS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(stats) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "by_league"}))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="side-stats refresh (capped)")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.max_calls is None:
        if not a.dry_run:
            raise SystemExit("--max-calls is required for a real run (hard cap, no default)")
        a.max_calls = MAX_CALLS_CAP
    run(a.max_calls, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
