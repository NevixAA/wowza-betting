"""
Player Stats Data Fetcher — API-Football (credit-efficient).

Credit budget:
  - collect_history(): ~50 calls per league/season (one-time)
  - fetch_fixture_players(): 1 call per fixture (only for SNIPER picks)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from . import config


def _headers() -> dict:
    return {
        "x-rapidapi-key":  config.API_KEY,
        "x-rapidapi-host": config.API_HOST,
    }


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, headers=_headers(), params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if config.CACHE_FILE.exists():
        try:
            return json.loads(config.CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    config.CACHE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Fixture list ──────────────────────────────────────────────────────────────

def fetch_finished_fixtures(league_id: int, season: str) -> list[dict]:
    """Return list of finished fixture dicts for a league/season. Cached."""
    cache = _load_cache()
    key = f"fixtures|{league_id}|{season}"
    if key in cache:
        return cache[key]

    resp = _get(
        f"https://{config.API_HOST}/v3/fixtures",
        {"league": league_id, "season": season, "status": "FT"},
    )
    fixtures = resp.get("response", [])
    cache[key] = fixtures
    _save_cache(cache)
    return fixtures


# ── Per-fixture player stats ──────────────────────────────────────────────────

def fetch_fixture_players(fixture_id: int) -> list[dict]:
    """
    Return per-player stats for one fixture.
    1 API call. Cached permanently.
    """
    cache = _load_cache()
    key = f"players|{fixture_id}"
    if key in cache:
        return cache[key]

    resp = _get(
        f"https://{config.API_HOST}/v3/fixtures/players",
        {"fixture": fixture_id},
    )
    players = resp.get("response", [])
    cache[key] = players
    _save_cache(cache)
    time.sleep(0.25)  # stay within rate limit
    return players


# ── Parse player rows from API response ──────────────────────────────────────

def parse_players(fixture_meta: dict, fixture_players: list[dict]) -> list[dict]:
    """
    Convert raw API response into flat player rows.
    fixture_meta: one entry from fetch_finished_fixtures()
    fixture_players: response from fetch_fixture_players()
    """
    fixture_id   = fixture_meta["fixture"]["id"]
    match_date   = fixture_meta["fixture"]["date"][:10]
    home_team    = fixture_meta["teams"]["home"]["name"]
    away_team    = fixture_meta["teams"]["away"]["name"]
    home_goals   = fixture_meta["goals"]["home"] or 0
    away_goals   = fixture_meta["goals"]["away"] or 0

    rows = []
    for team_block in fixture_players:
        team_name = team_block["team"]["name"]
        is_home   = int(team_name == home_team)

        for p in team_block["players"]:
            info  = p["player"]
            stats = p["statistics"][0] if p["statistics"] else {}
            games = stats.get("games", {})
            mins  = games.get("minutes") or 0

            if mins < 10:   # skip unused subs
                continue

            position = (games.get("position") or "").upper()
            rows.append({
                "fixture_id":  fixture_id,
                "date":        match_date,
                "player_id":   info["id"],
                "player_name": info["name"],
                "team":        team_name,
                "is_home":     is_home,
                "opponent":    away_team if is_home else home_team,
                "position":    position,
                "minutes":     mins,
                "goals":       (stats.get("goals", {}) or {}).get("total") or 0,
                "assists":     (stats.get("goals", {}) or {}).get("assists") or 0,
                "shots_total": (stats.get("shots", {}) or {}).get("total") or 0,
                "shots_on":    (stats.get("shots", {}) or {}).get("on") or 0,
                "yellow_card": int(((stats.get("cards", {}) or {}).get("yellow") or 0) > 0),
                "key_passes":  (stats.get("passes", {}) or {}).get("key") or 0,
                "rating":      float((games.get("rating") or 0) or 0),
                "home_goals":  home_goals,
                "away_goals":  away_goals,
            })
    return rows


# ── Bulk history collection (one-time, credit-capped) ─────────────────────────

def collect_history(
    max_fixtures: int = 800,
    leagues: dict | None = None,
    seasons: list[str] | None = None,
) -> list[dict]:
    """
    Collect historical player stats for training.
    max_fixtures caps total API calls to stay within credit budget.
    Returns flat list of player rows.
    """
    if leagues is None:
        leagues = config.TRAINING_LEAGUES
    if seasons is None:
        seasons = config.TRAINING_SEASONS

    all_rows: list[dict] = []
    calls = 0

    for season in seasons:
        for league_name, league_id in leagues.items():
            if calls >= max_fixtures:
                print(f"[STOP] Credit cap reached at {calls} calls.")
                return all_rows

            print(f"  Fetching fixtures: {league_name} {season}")
            try:
                fixtures = fetch_finished_fixtures(league_id, season)
            except Exception as e:
                print(f"  [WARN] {league_name} {season}: {e}")
                continue

            for fix in fixtures:
                if calls >= max_fixtures:
                    break
                fid = fix["fixture"]["id"]
                cache = _load_cache()
                if f"players|{fid}" in cache:
                    # Already collected — parse without API call
                    player_data = cache[f"players|{fid}"]
                    rows = parse_players(fix, player_data)
                    all_rows.extend(rows)
                    continue

                try:
                    player_data = fetch_fixture_players(fid)
                    rows = parse_players(fix, player_data)
                    all_rows.extend(rows)
                    calls += 1
                except Exception as e:
                    print(f"  [WARN] fixture {fid}: {e}")

            print(f"  [{league_name} {season}] {calls} API calls used so far")

    print(f"[DONE] Collected {len(all_rows)} player rows using {calls} API calls.")
    return all_rows
