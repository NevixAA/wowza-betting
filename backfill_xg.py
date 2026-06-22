#!/usr/bin/env python3
"""
Historical xG + possession + blocked shots backfill.

Iterates all leagues x historical seasons and calls get_fixture_statistics()
for every completed fixture. Results are permanently cached (TTL 1 year).
Next training run automatically picks up the richer data.

Usage:
    python backfill_xg.py                          # all leagues, 3 seasons back
    python backfill_xg.py --seasons 2023 2024 2025 # specific seasons
    python backfill_xg.py --league "Championship"  # single league
    python backfill_xg.py --start-from 3           # skip first N leagues (resume after quota reset)
    python backfill_xg.py --dry-run                # show what would run, no API calls

Daily quota: 7,500. Each fixture = 1 call. ~4,200 fixtures across all leagues/seasons.
Already-cached fixtures are skipped for free (1-year TTL).
Re-run daily with --start-from to resume where quota ran out.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.api_football_ou import (
    get_completed_fixtures,
    get_fixture_statistics,
    fetch_shots_for_league,
    _KEY,
)


# Historical seasons to backfill per league type
STANDARD_SEASONS = ["2022", "2023", "2024", "2025"]   # European format (Aug-May)
SUMMER_SEASONS   = ["2023", "2024", "2025", "2026"]   # Calendar-year leagues

# Leagues that use summer/calendar-year seasons
SUMMER_LEAGUES = {
    "Sweden Allsvenskan", "Norway Eliteserien", "Finland Veikkausliiga",
    "Ireland Premier Division", "Brazil Serie A", "Japan J-League",
    "Mexico Liga MX", "China Super League", "USA MLS",
}


def get_seasons_for_league(league: str) -> list[str]:
    """Return the historical seasons to backfill for a given league."""
    if league in SUMMER_LEAGUES:
        return SUMMER_SEASONS
    return STANDARD_SEASONS


def backfill_league_season(league: str, league_id: int, season: str, dry_run: bool = False) -> int:
    """
    Backfill one league/season. Returns number of fixtures processed.
    """
    fixtures = get_completed_fixtures(league_id, season)
    if not fixtures:
        print(f"  [{league} {season}] No completed fixtures found.")
        return 0

    print(f"  [{league} {season}] {len(fixtures)} fixtures to process...")

    if dry_run:
        print(f"  [{league} {season}] DRY RUN — skipping API calls.")
        return len(fixtures)

    cached = 0
    fetched = 0
    for i, fix in enumerate(fixtures):
        stats = get_fixture_statistics(fix["id"])
        if stats:
            if stats.get("home_xg") is not None or stats.get("home_shots") is not None:
                cached += 1
            fetched += 1
        # Small courtesy delay to avoid hammering the API
        if i % 10 == 9:
            time.sleep(0.2)

    print(f"  [{league} {season}] Done: {fetched}/{len(fixtures)} fetched, {cached} with xG data.")
    return fetched


def main():
    parser = argparse.ArgumentParser(description="Backfill historical xG and fixture stats cache.")
    parser.add_argument("--seasons", nargs="+", help="Override seasons to backfill (e.g. 2023 2024 2025)")
    parser.add_argument("--league", help="Backfill only this league name")
    parser.add_argument("--start-from", type=int, default=1, metavar="N",
                        help="Skip first N-1 leagues and start from league #N (1-based). Use to resume after a quota reset.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without making API calls")
    args = parser.parse_args()

    if not _KEY and not args.dry_run:
        print("ERROR: APIFOOTBALL_KEY env var not set. Set it or use --dry-run.")
        sys.exit(1)

    all_leagues = {}
    for lg, lid in config.API_FOOTBALL_IDS.items():
        if lg in ("World Cup 2026", "World Cup"):
            continue
        if args.league and lg != args.league:
            continue
        seasons = args.seasons if args.seasons else get_seasons_for_league(lg)
        all_leagues[lg] = (lid, seasons)

    # Apply --start-from: skip leagues before position N (1-based)
    league_items = list(all_leagues.items())
    if args.start_from > 1:
        skipped = args.start_from - 1
        print(f"Skipping first {skipped} league(s) (--start-from {args.start_from}):")
        for name, _ in league_items[:skipped]:
            print(f"  skip: {name}")
        league_items = league_items[skipped:]

    leagues_to_run = dict(league_items)
    total_fixtures = 0
    total_leagues  = len(leagues_to_run)
    print(f"Backfill plan: {total_leagues} leagues")
    print("=" * 60)

    for i, (league, (lid, seasons)) in enumerate(leagues_to_run.items(), args.start_from):
        print(f"\n[{i}/{len(all_leagues)}] {league} (ID={lid}) — seasons: {', '.join(seasons)}")
        for season in seasons:
            n = backfill_league_season(league, lid, season, dry_run=args.dry_run)
            total_fixtures += n
            if not args.dry_run:
                time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Backfill complete: {total_fixtures} fixtures processed across {total_leagues} leagues.")
    if args.dry_run:
        print(f"Estimated API calls: ~{total_fixtures} (1 per fixture)")
        quota = 7500
        days  = round(total_fixtures / quota, 2)
        print(f"At 7,500/day quota: {days} days to complete.")


if __name__ == "__main__":
    main()
