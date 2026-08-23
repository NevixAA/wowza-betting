"""
backfill_af_odds.py  (v2 — parallel, Ultra-plan speed)
=======================================================
Fetch BTTS / Over 1.5 / Over 3.5 historical odds from API-Football for all
new-format league fixtures. Saves output/af_odds_history.parquet.

Reuses fixture lists in af_history_cache/ (already downloaded — 0 extra calls).
Cache: af_odds_cache/ — skips any already-fetched fixture (fully resumable).

Speed: 20 parallel workers  ≈ 500 calls/min  → 20 K fixtures in ~40 min.
Daily limit: Ultra plan = 75 K / day → no quota worries today.

Usage:
    python scripts/backfill_af_odds.py               # full parallel run
    python scripts/backfill_af_odds.py --dry-run     # count uncached, then exit
    python scripts/backfill_af_odds.py --workers 30  # tune parallelism
    python scripts/backfill_af_odds.py --league "Brazil Serie A"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# FULL-MATCH BTTS ONLY. Duplicated verbatim from
# scripts/capture_std_sidemarket_odds_forward.py, which carries the full explanation; all four
# sites had the same substring-matching defect and must not diverge.
#
# The short version: `"Both Teams Score" in name` also matches bet 34, "Both Teams Score -
# First Half" (Yes=5.50 vs the real 1.91), and the parse loop assigns on every match so the
# last one wins. The corrupted pair looks perfect — overround 1.04, stable to kickoff — it
# just answers a different question.
_BTTS_BET_ID = 8
_BTTS_DISQUALIFY = ("Half", "1st", "2nd", "/", "Total Goals", "Corner", "Card",
                    "Player", "Shot", "Foul", "Handicap", "Minute")
_BTTS_EXACT = ("Both Teams Score", "Both Teams To Score", "BTTS")


def _is_fullmatch_btts(bet: dict) -> bool:
    """True only for the 90-minute Both-Teams-To-Score market."""
    if bet.get("id") == _BTTS_BET_ID:
        return True
    name = (bet.get("name") or "").strip()
    if any(x in name for x in _BTTS_DISQUALIFY):
        return False
    return name in _BTTS_EXACT



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORY_CACHE_DIR = Path(__file__).parent / "af_history_cache"
ODDS_CACHE_DIR    = Path(__file__).parent / "af_odds_cache"
ODDS_CACHE_DIR.mkdir(exist_ok=True)

OUTPUT = ROOT / "output" / "af_odds_history.parquet"
OUTPUT.parent.mkdir(exist_ok=True)

_BASE    = "https://v3.football.api-sports.io"
_KEY     = os.getenv("APIFOOTBALL_KEY", "")
_HEADERS = {"x-apisports-key": _KEY}
BET365   = 8

# Thread-safe counters
_lock         = threading.Lock()
_done_count   = 0
_quota_hit    = False

LEAGUES = [
    ("Brazil Serie A",              71,  ["2019","2020","2021","2022","2023","2024"]),
    ("Japan J-League",              98,  ["2019","2020","2021","2022","2023","2024"]),
    ("Mexico Liga MX",             262,  ["2019","2020","2021","2022","2023","2024"]),
    ("China Super League",         169,  ["2019","2020","2021","2022","2023","2024"]),
    ("USA MLS",                    253,  ["2019","2020","2021","2022","2023","2024"]),
    ("Argentina Primera Division", 128,  ["2019","2020","2021","2022","2023","2024"]),
    ("Ireland Premier Division",   357,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Finland Veikkausliiga",      244,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Sweden Allsvenskan",         113,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Norway Eliteserien",         103,  ["2019","2020","2021","2022","2023","2024","2025"]),
    ("Denmark Superliga",          119,  ["2019","2020","2021","2022","2023","2024"]),
    ("Austrian Bundesliga",        218,  ["2019","2020","2021","2022","2023","2024"]),
]


def _parse_odds(data: dict) -> dict:
    btts = over15 = over35 = None
    for entry in data.get("response", []):
        for bk in entry.get("bookmakers", []):
            if bk.get("id") != BET365:
                continue
            for bet in bk.get("bets", []):
                name = bet.get("name", "")
                if _is_fullmatch_btts(bet):
                    for v in bet.get("values", []):
                        try:
                            if v.get("value") == "Yes":
                                btts = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            pass
                elif "Over/Under" in name:
                    for v in bet.get("values", []):
                        label = v.get("value", "")
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        if "Over 1.5" in label:
                            over15 = odd
                        elif "Over 3.5" in label:
                            over35 = odd
    return {"odds_btts": btts, "odds_over15": over15, "odds_over35": over35}


def _fetch_one(task: dict) -> dict | None:
    """Fetch odds for a single fixture. Returns result row or None on skip/error."""
    global _quota_hit
    if _quota_hit:
        return None

    fix_id   = task["fixture_id"]
    cache_f  = ODDS_CACHE_DIR / f"odds_{fix_id}.json"

    if cache_f.exists():
        data = json.loads(cache_f.read_text(encoding="utf-8"))
    else:
        if not _KEY:
            print("[backfill_odds] APIFOOTBALL_KEY not set")
            sys.exit(1)
        for attempt in range(3):
            try:
                r = requests.get(f"{_BASE}/odds",
                                 headers=_HEADERS,
                                 params={"fixture": fix_id, "bookmaker": BET365},
                                 timeout=20)
                remaining = int(r.headers.get("x-ratelimit-requests-remaining", 9999))
                if r.status_code == 429:
                    time.sleep(15)
                    continue
                if remaining < 100:
                    with _lock:
                        _quota_hit = True
                    print(f"\n  [!] Quota low (remaining={remaining}) — stopping")
                    return None
                if r.status_code != 200:
                    return None
                data = r.json()
                # Detect rate-limit error in response body (API returns 200 with error dict)
                if data.get("errors"):
                    err = data["errors"]
                    if "rateLimit" in str(err) or "requests" in str(err).lower():
                        time.sleep(15)
                        continue  # retry without caching
                    return None  # other API error — skip fixture
                cache_f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(2)

    odds = _parse_odds(data)
    row = {**task, **odds}

    global _done_count
    with _lock:
        _done_count += 1
        cnt = _done_count
    if cnt % 500 == 0:
        print(f"  ... {cnt} fetched")

    return row


def _build_task_list(target_league: str | None) -> list[dict]:
    tasks = []
    leagues_to_run = [(n, i, s) for n, i, s in LEAGUES
                      if target_league is None or n == target_league]
    for league_name, league_id, seasons in leagues_to_run:
        for season in seasons:
            fix_cache = HISTORY_CACHE_DIR / f"fixtures_{league_id}_{season}.json"
            if not fix_cache.exists():
                print(f"  [skip] {league_name} {season}: fixture cache missing")
                continue
            data = json.loads(fix_cache.read_text(encoding="utf-8"))
            for f in data.get("response", []):
                if f.get("fixture", {}).get("status", {}).get("short") != "FT":
                    continue
                tasks.append({
                    "fixture_id": f["fixture"]["id"],
                    "league":     league_name,
                    "season":     season,
                    "date":       f["fixture"].get("date", "")[:10],
                    "home_team":  f.get("teams", {}).get("home", {}).get("name", ""),
                    "away_team":  f.get("teams", {}).get("away", {}).get("name", ""),
                })
    return tasks


def run(target_league: str | None = None, dry_run: bool = False,
        workers: int = 20) -> None:
    tasks = _build_task_list(target_league)

    # Split: already cached (instant load) vs needs API call
    cached_tasks  = [t for t in tasks if (ODDS_CACHE_DIR / f"odds_{t['fixture_id']}.json").exists()]
    missing_tasks = [t for t in tasks if t not in cached_tasks]

    print(f"\n  Total fixtures:  {len(tasks):,}")
    print(f"  Already cached:  {len(cached_tasks):,}")
    print(f"  Need API calls:  {len(missing_tasks):,}")

    if dry_run:
        print(f"\n  At ~500 calls/min with {workers} workers: "
              f"~{len(missing_tasks) // 500 + 1} min to complete")
        return

    if not missing_tasks and not cached_tasks:
        print("  Nothing to do.")
        return

    start = time.time()
    all_rows: list[dict] = []

    print(f"\n  Running {workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                all_rows.append(row)
            if _quota_hit:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    elapsed = time.time() - start
    print(f"\n  Done: {len(all_rows):,} rows in {elapsed:.0f}s  "
          f"({len(all_rows)/elapsed*60:.0f} rows/min)")

    if _quota_hit:
        print("  [!] Quota low — run again tomorrow (cache is safe)")

    if not all_rows:
        print("  No rows collected.")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Merge with any existing parquet (incremental run support)
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["fixture_id"], keep="last")

    df = df.sort_values(["league", "date"]).reset_index(drop=True)
    df.to_parquet(OUTPUT, index=False)

    btts_cov   = df["odds_btts"].notna().mean()
    over15_cov = df["odds_over15"].notna().mean()
    over35_cov = df["odds_over35"].notna().mean()

    print(f"\n  Saved {len(df):,} rows -> {OUTPUT.name}")
    print(f"  Coverage — BTTS {btts_cov:.0%}  O1.5 {over15_cov:.0%}  O3.5 {over35_cov:.0%}")
    print(f"\n  Next step:")
    print(f"    python pipeline.py --mode train")
    print(f"    python pipeline.py --mode backtest-side")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--workers",  type=int, default=20)
    parser.add_argument("--league",   type=str, default=None)
    args = parser.parse_args()
    run(target_league=args.league, dry_run=args.dry_run, workers=args.workers)
