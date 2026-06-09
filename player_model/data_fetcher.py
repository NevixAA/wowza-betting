"""
Player Stats Data Fetcher — API-Football (Starter plan required).
Replaces FBref scraper. Uses /players endpoint for season stats.

Endpoints used:
  GET /players?league={id}&season={year}&page={n}  — season stats per player
  GET /fixtures/players?fixture={id}               — per-fixture stats (rolling features)

Cached for 7 days. ~25 API calls per league per collection run.
"""
from __future__ import annotations

import os
import time
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from . import config

BASE_URL      = "https://v3.football.api-sports.io"
CACHE_DIR     = config.BASE_DIR / "player_match_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DAYS    = 7
REQUEST_DELAY = 1.2   # ~50 req/min, well within Starter rate limit

# API-Football league IDs for player stats collection
# Only leagues with reliable player stat coverage
APIFOOTBALL_LEAGUES: dict[str, tuple[int, str]] = {
    "Premier League":   (39,  "2024"),
    "Bundesliga":       (78,  "2024"),
    "La Liga":          (140, "2024"),
    "Serie A":          (135, "2024"),
    "Ligue 1":          (61,  "2024"),
    "Championship":     (40,  "2024"),
    "League One":       (41,  "2024"),
    "Bundesliga 2":     (79,  "2024"),
    "Ireland Premier":  (357, "2025"),
    "Finland Veikk":    (244, "2025"),
    "Champions League": (2,   "2024"),
    "Europa League":    (3,   "2024"),
    "Conference League":(848, "2024"),
    "World Cup":        (1,   "2026"),
}

# Keep this alias so pipeline.py import doesn't break
FBREF_LEAGUES = APIFOOTBALL_LEAGUES


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict) -> dict:
    key = os.getenv("APIFOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("APIFOOTBALL_KEY not set in environment")
    time.sleep(REQUEST_DELAY)
    r = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers={"x-apisports-key": key},
        params=params,
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API-Football {r.status_code}: {r.text[:200]}")
    d = r.json()
    errors = d.get("errors", {})
    if errors:
        raise RuntimeError(f"API-Football errors: {errors}")
    return d


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"players_{h}.json"


def _load_cache(key: str) -> Optional[list]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched > timedelta(days=CACHE_DAYS):
            return None
        return data.get("players")
    except Exception:
        return None


def _save_cache(key: str, players: list) -> None:
    p = _cache_path(key)
    p.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "players": players},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Player stats parser ───────────────────────────────────────────────────────

def _parse_player(entry: dict, league: str) -> Optional[dict]:
    """Parse one player entry from /players response."""
    p    = entry.get("player", {})
    name = p.get("name", "").strip()
    if not name:
        return None

    stats = entry.get("statistics", [{}])[0]
    games  = stats.get("games", {})
    goals  = stats.get("goals", {})
    shots  = stats.get("shots", {})
    cards  = stats.get("cards", {})
    passes = stats.get("passes", {})
    duels  = stats.get("duels", {})
    team   = stats.get("team", {})

    appearances = int(games.get("appearences") or 0)
    minutes     = int(games.get("minutes")     or 0)
    if appearances < config.MIN_APPEARANCES or minutes < 1:
        return None

    goals_n  = int(goals.get("total")   or 0)
    assists_n = int(goals.get("assists") or 0)
    shots_n  = int(shots.get("total")   or 0)
    sot_n    = int(shots.get("on")      or 0)
    yellow_n = int(cards.get("yellow")  or 0)
    key_pass = int(passes.get("key")    or 0)
    position = str(games.get("position") or p.get("position") or "").strip()

    return {
        "player_id":       int(p.get("id", abs(hash(name)) % 10_000_000)),
        "player_name":     name,
        "team":            team.get("name", ""),
        "league":          league,
        "position":        position,
        "appearances":     appearances,
        "minutes":         minutes,
        "goals":           goals_n,
        "assists":         assists_n,
        "shots_total":     shots_n,
        "shots_on_target": sot_n,
        "yellow_cards":    yellow_n,
        "key_passes":      key_pass,
        "rating":          float(games.get("rating") or 0),
    }


# ── Public: fetch one league ──────────────────────────────────────────────────

def fetch_league_player_stats(
    league: str,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return flat list of player stat dicts for one league.
    Fetched from API-Football /players endpoint, paginated.
    Cached for CACHE_DAYS.
    """
    info = APIFOOTBALL_LEAGUES.get(league)
    if not info:
        print(f"  [fetch] Unknown league: {league}")
        return []

    lg_id, season = info
    cache_key = f"apifootball_players|{league}|{season}"

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            print(f"  [{league}] {len(cached)} players (cached)")
            return cached

    print(f"  [{league}] Fetching from API-Football (league={lg_id}, season={season})...")

    players: list[dict] = []
    page = 1

    while True:
        try:
            data = _api_get("players", {"league": lg_id, "season": season, "page": page})
        except RuntimeError as e:
            print(f"  [{league}] API error page {page}: {e}")
            break

        entries  = data.get("response", [])
        paging   = data.get("paging", {})
        total_pg = int(paging.get("total", 1))

        for entry in entries:
            parsed = _parse_player(entry, league)
            if parsed:
                players.append(parsed)

        print(f"    page {page}/{total_pg} — {len(entries)} entries")

        if page >= total_pg:
            break
        page += 1

    print(f"  [{league}] {len(players)} players with ≥{config.MIN_APPEARANCES} appearances")

    if players:
        _save_cache(cache_key, players)

    return players


# ── Bulk collection ───────────────────────────────────────────────────────────

def collect_history(
    max_fixtures: int = 800,
    leagues: dict | None = None,
    seasons: list[str] | None = None,
) -> list[dict]:
    """Collect player season stats from API-Football for all supported leagues."""
    if leagues is None:
        leagues = APIFOOTBALL_LEAGUES

    all_rows: list[dict] = []
    for league in leagues:
        rows = fetch_league_player_stats(league, force_refresh=False)
        all_rows.extend(rows)

    print(f"[DONE] {len(all_rows)} total player rows across {len(leagues)} leagues.")
    return all_rows
