"""
backfill_af_history.py
======================
One-time historical backfill: fetch match statistics from API-Football
for all new-format leagues and save as output/af_history.parquet.

This promotes new-format leagues from "goals only" to full-feature
training (shots, corners, fouls, cards) — same richness as standard leagues.

Cost: ~16,000–20,000 API-Football credits (one-time).
Cache: scripts/af_history_cache/ — never re-fetches completed fixtures.
Output: output/af_history.parquet

Usage:
    python scripts/backfill_af_history.py            # full run
    python scripts/backfill_af_history.py --dry-run  # estimate call count only
    python scripts/backfill_af_history.py --league "Brazil Serie A"  # single league
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_DIR = Path(__file__).parent / "af_history_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT = ROOT / "output" / "af_history.parquet"
OUTPUT.parent.mkdir(exist_ok=True)

_BASE    = "https://v3.football.api-sports.io"
_KEY     = os.getenv("APIFOOTBALL_KEY", "")
_HEADERS = {"x-apisports-key": _KEY}

# ── League config ─────────────────────────────────────────────────────────────
# (league_name, af_league_id, [backfill_seasons])
# Calendar-year leagues use the year as season (2020 → 2020 season)
# Split-year leagues use start year (2020 → 2020/21)

LEAGUES = [
    # Summer / Americas / Asia — calendar year
    ("Brazil Serie A",           71,  ["2019","2020","2021","2022","2023","2024"]),
    ("Japan J-League",           98,  ["2019","2020","2021","2022","2023","2024"]),
    ("Mexico Liga MX",          262,  ["2019","2020","2021","2022","2023","2024"]),
    ("China Super League",      169,  ["2019","2020","2021","2022","2023","2024"]),
    ("USA MLS",                 253,  ["2019","2020","2021","2022","2023","2024"]),
    ("Argentina Primera Division",128,["2019","2020","2021","2022","2023","2024"]),
    ("Ireland Premier Division",357,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Finland Veikkausliiga",   244,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Sweden Allsvenskan",      113,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Norway Eliteserien",      103,  ["2019","2020","2021","2022","2023","2024","2025"]),
    # European winter/spring — split year (season = start year)
    ("Denmark Superliga",       119,  ["2019","2020","2021","2022","2023","2024"]),
    ("Austrian Bundesliga",     218,  ["2019","2020","2021","2022","2023","2024"]),
]


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, cache_key: str, dry_run: bool = False) -> dict | None:
    cache_f = CACHE_DIR / f"{cache_key}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text(encoding="utf-8"))

    if dry_run:
        return None

    if not _KEY:
        print("[backfill] APIFOOTBALL_KEY not set — set env var and retry")
        sys.exit(1)

    try:
        r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=20)
        remaining = r.headers.get("x-ratelimit-requests-remaining", "?")
        if r.status_code == 429:
            print(f"  [rate limit] sleeping 60s...")
            time.sleep(60)
            r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  [error] {endpoint} {params}: HTTP {r.status_code}")
            return None
        data = r.json()
        cache_f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.35)   # ~170 calls/min max — safe for pro tier
        return data
    except Exception as e:
        print(f"  [error] {endpoint}: {e}")
        return None


def _stat(stats_list: list, stat_type: str) -> int | None:
    for s in stats_list:
        if s.get("type") == stat_type:
            val = s.get("value")
            if val is None or val == "None":
                return None
            try:
                return int(str(val).replace("%", "").strip())
            except (ValueError, TypeError):
                return None
    return None


# ── Fetch season fixtures ─────────────────────────────────────────────────────

def fetch_season(league_name: str, league_id: int, season: str, dry_run: bool) -> list[dict]:
    ck = f"fixtures_{league_id}_{season}"
    data = _get("fixtures", {"league": league_id, "season": season}, ck, dry_run)

    if data is None:
        return []

    fixtures = data.get("response", [])
    completed = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "FT"]
    print(f"  {league_name} {season}: {len(completed)}/{len(fixtures)} completed fixtures")
    return completed


def fetch_stats(fixture_id: int, dry_run: bool) -> dict | None:
    ck = f"stats_{fixture_id}"
    data = _get("fixtures/statistics", {"fixture": fixture_id}, ck, dry_run)
    if data is None:
        return None
    return data.get("response")


# ── Parse one fixture into a row ──────────────────────────────────────────────

def parse_fixture(league_name: str, season: str, f: dict, stats_resp) -> dict | None:
    fix  = f.get("fixture", {})
    teams = f.get("teams", {})
    goals = f.get("goals", {})

    date_str = fix.get("date", "")[:10]
    home = teams.get("home", {}).get("name", "")
    away = teams.get("away", {}).get("name", "")
    fthg = goals.get("home")
    ftag = goals.get("away")

    if not home or not away or fthg is None or ftag is None:
        return None

    row = {
        "league":     league_name,
        "season":     season,
        "date":       date_str,
        "home_team":  home,
        "away_team":  away,
        "FTHG":       int(fthg),
        "FTAG":       int(ftag),
        "FTR":        "H" if fthg > ftag else ("A" if ftag > fthg else "D"),
        # Shot columns — None if API-Football doesn't have stats for this fixture
        "HS":  None, "AS":  None,
        "HST": None, "AST": None,
        "HC":  None, "AC":  None,
        "HF":  None, "AF":  None,
        "HY":  None, "AY":  None,
        "HR":  None, "AR":  None,
    }

    if stats_resp and len(stats_resp) >= 2:
        h_stats = stats_resp[0].get("statistics", [])
        a_stats = stats_resp[1].get("statistics", [])

        row["HS"]  = _stat(h_stats, "Total Shots")
        row["AS"]  = _stat(a_stats, "Total Shots")
        row["HST"] = _stat(h_stats, "Shots on Goal")
        row["AST"] = _stat(a_stats, "Shots on Goal")
        row["HC"]  = _stat(h_stats, "Corner Kicks")
        row["AC"]  = _stat(a_stats, "Corner Kicks")
        row["HF"]  = _stat(h_stats, "Fouls")
        row["AF"]  = _stat(a_stats, "Fouls")
        row["HY"]  = _stat(h_stats, "Yellow Cards")
        row["AY"]  = _stat(a_stats, "Yellow Cards")
        row["HR"]  = _stat(h_stats, "Red Cards")
        row["AR"]  = _stat(a_stats, "Red Cards")

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run(target_league: str | None = None, dry_run: bool = False) -> None:
    all_rows: list[dict] = []
    total_api_calls = 0
    total_fixtures  = 0

    leagues_to_run = [(n, i, s) for n, i, s in LEAGUES if target_league is None or n == target_league]

    for league_name, league_id, seasons in leagues_to_run:
        print(f"\n{'='*60}")
        print(f"  {league_name}  (AF id={league_id})")
        print(f"{'='*60}")

        for season in seasons:
            fixtures = fetch_season(league_name, league_id, season, dry_run)
            total_api_calls += 1  # fixtures list call

            for i, f in enumerate(fixtures):
                fix_id = f["fixture"]["id"]
                stats_resp = fetch_stats(fix_id, dry_run)
                total_api_calls += 1

                row = parse_fixture(league_name, season, f, stats_resp)
                if row:
                    all_rows.append(row)

                if (i + 1) % 50 == 0:
                    print(f"    ... {i+1}/{len(fixtures)} done  (total rows so far: {len(all_rows)})")

            total_fixtures += len(fixtures)

    print(f"\n{'='*60}")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Total fixtures:  {total_fixtures}")
    print(f"  Total rows:      {len(all_rows)}")

    if dry_run:
        print("  [dry-run] no data saved")
        return

    if not all_rows:
        print("  No rows collected — check API key and cache")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["league", "date"]).reset_index(drop=True)

    # Merge with existing af_history if it exists (incremental runs)
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["league", "date", "home_team", "away_team"], keep="last")
        df = df.sort_values(["league", "date"]).reset_index(drop=True)

    df.to_parquet(OUTPUT, index=False)
    pct_with_shots = df["HS"].notna().mean() * 100
    print(f"\n  Saved {len(df)} rows → {OUTPUT.name}")
    print(f"  Shot data coverage: {pct_with_shots:.1f}%")
    print(f"\n  Next step: run pipeline.py --mode train to retrain team model")
    print(f"  Note: leagues stay as NEW_FORMAT — API-Football and FD data are kept separate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill API-Football match stats for new-format leagues")
    parser.add_argument("--dry-run", action="store_true", help="Estimate call count without hitting API")
    parser.add_argument("--league", type=str, default=None, help="Run for a single league only")
    args = parser.parse_args()

    run(target_league=args.league, dry_run=args.dry_run)
