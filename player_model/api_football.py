"""
API-Football Client — Player Stats, Lineups, Referees
======================================================
Uses direct API-Football.com (api-sports.io) — FREE tier: 100 calls/day.
Register at: https://dashboard.api-football.com/register
Add key to .env: APIFOOTBALL_KEY=your_key

Endpoints used:
  /fixtures           — fixture IDs for a league/season
  /fixtures/players   — per-player stats for a completed fixture
  /fixtures/lineups   — confirmed starting XI + bench
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from . import config

# Direct API-Football.com (api-sports.io) — different from RapidAPI
import os as _os
_APIFOOTBALL_KEY = _os.getenv("APIFOOTBALL_KEY", "")
HEADERS  = {"x-apisports-key": _APIFOOTBALL_KEY}
BASE_URL = "https://v3.football.api-sports.io"
CACHE_DIR = config.BASE_DIR / "player_match_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str, max_age_h: int = 24) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
        if datetime.now() - ts < timedelta(hours=max_age_h):
            return data
    except Exception:
        pass
    return None


def _save_cache(key: str, data: dict) -> None:
    data["_cached_at"] = datetime.now().isoformat()
    _cache_path(key).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _get(endpoint: str, params: dict, cache_hours: int = 24) -> Optional[dict]:
    """Make one API-Football call with caching."""
    if not _APIFOOTBALL_KEY:
        return None  # no key — fallback to FBref only
    cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
    cached = _load_cache(cache_key, max_age_h=cache_hours)
    if cached:
        return cached

    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        _save_cache(cache_key, data)
        time.sleep(0.5)  # polite delay
        return data
    except Exception as e:
        print(f"[api_football] Error {endpoint}: {e}")
        return None


# ── Fixtures ──────────────────────────────────────────────────────────────────

def get_recent_fixtures(league_id: int, season: str, last_n: int = 10) -> list[dict]:
    """Get last N completed fixtures for a league."""
    data = _get("/fixtures", {
        "league": league_id, "season": season,
        "last": last_n, "status": "FT"
    }, cache_hours=6)
    if not data:
        return []
    return data.get("response", [])


def get_upcoming_fixtures(league_id: int, season: str, next_n: int = 5) -> list[dict]:
    """Get next N upcoming fixtures for a league."""
    data = _get("/fixtures", {
        "league": league_id, "season": season, "next": next_n
    }, cache_hours=2)
    if not data:
        return []
    return data.get("response", [])


# ── Player match stats ────────────────────────────────────────────────────────

def get_fixture_player_stats(fixture_id: int) -> list[dict]:
    """
    Get per-player stats for one completed fixture.
    Returns flat list of player stat dicts.
    """
    data = _get("/fixtures/players", {"fixture": fixture_id}, cache_hours=168)  # 7 days
    if not data:
        return []

    players = []
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "")
        for player_entry in team_data.get("players", []):
            p   = player_entry.get("player", {})
            s   = (player_entry.get("statistics") or [{}])[0]
            games = s.get("games", {})
            shots = s.get("shots", {})
            goals = s.get("goals", {})
            cards = s.get("cards", {})
            passes = s.get("passes", {})
            duels  = s.get("duels", {})
            fouls  = s.get("fouls", {})

            players.append({
                "fixture_id":      fixture_id,
                "player_id":       p.get("id"),
                "player_name":     p.get("name", ""),
                "team":            team_name,
                "position":        games.get("position", ""),
                "minutes_played":  games.get("minutes") or 0,
                "started":         bool(games.get("minutes", 0) > 45),
                "goals":           goals.get("total") or 0,
                "assists":         goals.get("assists") or 0,
                "shots_total":     shots.get("total") or 0,
                "shots_on_target": shots.get("on") or 0,
                "yellow_card":     int((cards.get("yellow") or 0) > 0),
                "red_card":        int((cards.get("red") or 0) > 0),
                "key_passes":      passes.get("key") or 0,
                "fouls_committed": fouls.get("committed") or 0,
                "fouls_drawn":     fouls.get("drawn") or 0,
                "duels_total":     duels.get("total") or 0,
                "duels_won":       duels.get("won") or 0,
            })
    return players


# ── Lineups ───────────────────────────────────────────────────────────────────

def get_fixture_lineup(fixture_id: int) -> dict:
    """
    Get confirmed lineup for a fixture.
    Returns {home_team: [...players], away_team: [...players]}
    """
    data = _get("/fixtures/lineups", {"fixture": fixture_id}, cache_hours=2)
    if not data:
        return {}

    result = {}
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "")
        starters  = [
            {
                "player_id":   p.get("player", {}).get("id"),
                "player_name": p.get("player", {}).get("name", ""),
                "position":    p.get("pos", ""),
                "number":      p.get("number"),
                "started":     True,
            }
            for p in team_data.get("startXI", [])
        ]
        subs = [
            {
                "player_id":   p.get("player", {}).get("id"),
                "player_name": p.get("player", {}).get("name", ""),
                "position":    p.get("pos", ""),
                "number":      p.get("number"),
                "started":     False,
            }
            for p in team_data.get("substitutes", [])
        ]
        result[team_name] = starters + subs
    return result


# ── Referee ───────────────────────────────────────────────────────────────────

def get_fixture_referee(fixture_id: int) -> str:
    """Return referee name for a fixture."""
    data = _get("/fixtures", {"id": fixture_id}, cache_hours=24)
    if not data:
        return ""
    resp = data.get("response", [{}])
    if not resp:
        return ""
    return resp[0].get("fixture", {}).get("referee", "") or ""


# ── Rolling player stats ──────────────────────────────────────────────────────

def get_player_recent_stats(player_id: int, league_id: int, season: str,
                             last_n: int = 5) -> list[dict]:
    """
    Get last N match-level stats for a specific player.
    Returns list of per-match stat dicts sorted by date (most recent first).
    """
    # Get recent fixtures for this league
    fixtures = get_recent_fixtures(league_id, season, last_n=20)
    if not fixtures:
        return []

    player_matches = []
    for fix in fixtures[:20]:
        fix_id = fix.get("fixture", {}).get("id")
        if not fix_id:
            continue
        stats = get_fixture_player_stats(fix_id)
        for s in stats:
            if s.get("player_id") == player_id and s.get("minutes_played", 0) > 0:
                s["date"] = fix.get("fixture", {}).get("date", "")
                player_matches.append(s)
                break

    # Sort by date descending, return last N
    player_matches.sort(key=lambda x: x.get("date", ""), reverse=True)
    return player_matches[:last_n]


# ── Referee profiles ──────────────────────────────────────────────────────────

def build_referee_profile(referee_name: str, league_id: int, season: str) -> dict:
    """
    Compute referee profile from recent fixtures.
    Returns: {yellows_per_game, strictness_score, n_games}
    """
    if not referee_name:
        return {"yellows_per_game": 3.5, "strictness_score": 0.0, "n_games": 0}

    fixtures = get_recent_fixtures(league_id, season, last_n=30)
    ref_fixtures = [
        f for f in fixtures
        if referee_name.lower() in (f.get("fixture", {}).get("referee", "") or "").lower()
    ]

    if len(ref_fixtures) < 3:
        return {"yellows_per_game": 3.5, "strictness_score": 0.0, "n_games": 0}

    total_yellows = 0
    for fix in ref_fixtures:
        fix_id = fix.get("fixture", {}).get("id")
        if fix_id:
            stats = get_fixture_player_stats(fix_id)
            total_yellows += sum(s.get("yellow_card", 0) for s in stats)

    n = len(ref_fixtures)
    avg_yellows = total_yellows / n
    # League average is ~3.5 yellows/game; compute z-score
    league_avg = 3.5
    league_std = 1.0
    strictness = (avg_yellows - league_avg) / league_std

    return {
        "yellows_per_game": round(avg_yellows, 2),
        "strictness_score": round(strictness, 2),
        "n_games": n,
    }
