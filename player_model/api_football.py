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


def _load_dotenv_once() -> None:
    """Populate os.environ from the project .env for local dev, WITHOUT overriding vars
    already set (so CI / GitHub secrets always take precedence). Values are used silently
    and never logged."""
    envf = Path(__file__).resolve().parents[1] / ".env"
    if not envf.exists():
        return
    try:
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k and k not in _os.environ:
                _os.environ[k] = v.strip().strip('"').strip("'")
    except Exception:
        pass


_load_dotenv_once()
_APIFOOTBALL_KEY = _os.getenv("APIFOOTBALL_KEY", "").strip()
HEADERS  = {"x-apisports-key": _APIFOOTBALL_KEY}
BASE_URL = "https://v3.football.api-sports.io"
# Use WowzaApp junction (no Hebrew path) for cache
import sys as _sys
_WOWZA_DIR = Path("C:/WowzaApp") if Path("C:/WowzaApp").exists() else config.BASE_DIR
CACHE_DIR = _WOWZA_DIR / "player_match_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    import re, hashlib
    # Use hash for safety — removes all special chars, Hebrew path issues etc.
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


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
    """Get recently completed fixtures for a league (last 30 days)."""
    from datetime import date, timedelta
    today    = date.today()
    from_dt  = (today - timedelta(days=30)).isoformat()
    to_dt    = today.isoformat()
    data = _get("/fixtures", {
        "league": league_id, "season": season,
        "from": from_dt, "to": to_dt, "status": "FT"
    }, cache_hours=6)
    if not data:
        return []
    results = data.get("response", [])
    # Return last N sorted by date descending
    results.sort(key=lambda x: x.get("fixture", {}).get("date", ""), reverse=True)
    return results[:last_n]


def get_upcoming_fixtures(league_id: int, season: str, next_n: int = 5) -> list[dict]:
    """Get next N upcoming fixtures for a league."""
    data = _get("/fixtures", {
        "league": league_id, "season": season, "next": next_n
    }, cache_hours=2)
    if not data:
        return []
    return data.get("response", [])


_POS_MAP = {"Goalkeeper": "G", "Defender": "D", "Midfielder": "M", "Attacker": "F"}


def get_league_teams(league_id: int, season: str) -> list[dict]:
    """All teams in a league/season -> [{id, name}]. Cached 24h."""
    data = _get("/teams", {"league": league_id, "season": season}, cache_hours=24)
    if not data:
        return []
    return [{"id": t["team"]["id"], "name": t["team"]["name"]}
            for t in data.get("response", []) if t.get("team")]


def get_team_squad(team_id: int) -> list[dict]:
    """Current squad for a team -> [{player_id, player_name, position, number, age}]. Cached 24h.
    position mapped to our F/M/D/G codes."""
    data = _get("/players/squads", {"team": team_id}, cache_hours=24)
    if not data or not data.get("response"):
        return []
    players = data["response"][0].get("players", [])
    return [{"player_id": p.get("id"), "player_name": p.get("name"),
             "position": _POS_MAP.get(p.get("position"), p.get("position")),
             "number": p.get("number"), "age": p.get("age")}
            for p in players]


def get_pl_squads(league_id: int, season: str) -> list[dict]:
    """All squads in a league -> flat [{team, player_id, player_name, position, number, age}].
    ~1 call for teams + 1 per team (cached 24h). Used for the fantasy daily transfer-window
    squad overlay (output/pl_squads_official.csv)."""
    rows = []
    for t in get_league_teams(league_id, season):
        for pl in get_team_squad(t["id"]):
            pl["team"] = t["name"]
            rows.append(pl)
    return rows


def _norm_name(s: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


_GENERIC_TOKENS = {"fc", "cf", "sc", "afc", "cd", "ac", "ss", "as", "us",
                   "if", "fk", "sk", "bk", "club", "cp", "sd"}
_RESERVE_TOKENS = {"b", "c", "ii", "2", "reserve", "reserves", "u19", "u20",
                   "u21", "u23", "academy", "youth", "castilla"}
_TOKEN_ALIASES = {"utd": "united", "weds": "wednesday", "wed": "wednesday",
                  "wanderers": "wanderers", "spurs": "tottenham"}


def _team_match(a: str, b: str) -> bool:
    """True iff a and b name the SAME senior club.

    Handles all three cases the naive matchers got wrong:
      • abbreviations: "Man City" == "Manchester City", "Inter" == "Internazionale"
        (token-prefix match, min length 3);
      • distinct clubs sharing a word stay apart: "Man City" != "Man United",
        "Real Madrid" != "Real Sociedad", "West Ham" != "West Brom";
      • reserve/youth sides rejected: "Barcelona" != "Barcelona B".
    Generic suffixes (FC/CF/AC/…) are ignored. Fuzzy fallback only when no token relation.
    """
    import difflib
    na, nb = _norm_name(a), _norm_name(b)
    if na == nb:
        return True
    _alias = lambda toks: {_TOKEN_ALIASES.get(t, t) for t in toks}
    sa = _alias(set(na.split()) - _GENERIC_TOKENS)
    sb = _alias(set(nb.split()) - _GENERIC_TOKENS)
    if not sa or not sb:
        return False

    ra, rb = sa & _RESERVE_TOKENS, sb & _RESERVE_TOKENS
    ca, cb = sa - _RESERVE_TOKENS, sb - _RESERVE_TOKENS
    if bool(ra) != bool(rb) and ca and cb and (ca <= cb or cb <= ca):
        return False   # senior vs reserve of the same club -> different teams

    def _tok(t, others):   # exact, or one a prefix of the other (>=3 chars) -> abbreviation
        return any(t == o or (len(t) >= 3 and o.startswith(t)) or (len(o) >= 3 and t.startswith(o))
                   for o in others)

    if ca and all(_tok(t, cb) for t in ca):   # every identity token of A maps into B
        return True
    if cb and all(_tok(t, ca) for t in cb):   # …or vice-versa
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.90


def find_fixture_id(league_id: int, season: str, date_str: str, home: str, away: str) -> int | None:
    """Find fixture ID for a completed (FT) match by league, season, date, and team names."""
    data = _get("/fixtures", {
        "league": league_id, "season": season, "date": date_str, "status": "FT",
    }, cache_hours=168)  # completed fixtures cached 7 days
    if not data:
        return None

    for fix in data.get("response", []):
        fh = fix.get("teams", {}).get("home", {}).get("name", "")
        fa = fix.get("teams", {}).get("away", {}).get("name", "")
        if _team_match(home, fh) and _team_match(away, fa):
            return fix.get("fixture", {}).get("id")
    return None


def get_injured_players(league_id: int, season: str, date_str: str) -> set[str]:
    """
    Return normalized names of players currently injured/suspended for a league on a given date.
    Uses /injuries endpoint cached 4 hours (injury status is stable within a day).
    Returns empty set if no key or endpoint unavailable.
    """
    data = _get("/injuries", {
        "league": league_id, "season": season, "date": date_str,
    }, cache_hours=4)
    if not data:
        return set()
    injured: set[str] = set()
    for entry in data.get("response", []):
        name = entry.get("player", {}).get("name", "")
        if name:
            injured.add(_norm_name(name))
    return injured


def get_fixture_status(league_id: int, season: str, date_str: str, home: str, away: str) -> tuple[str, int | None]:
    """
    Check the status of any fixture (played, postponed, cancelled, etc.) regardless of completion.
    Returns (status_short, fixture_id).
      "FT"        — full time (played)
      "PST"       — postponed
      "CANC"      — cancelled
      "NS"        — not started yet
      "LIVE"      — in progress
      "NOT_FOUND" — no fixture found for this match

    Used to mark player ledger rows as is_played=False when a match was called off.
    Cached only 2 hours (status can change until the match is done).
    """
    data = _get("/fixtures", {
        "league": league_id, "season": season, "date": date_str,
    }, cache_hours=2)
    if not data:
        return "NOT_FOUND", None

    for fix in data.get("response", []):
        fh = fix.get("teams", {}).get("home", {}).get("name", "")
        fa = fix.get("teams", {}).get("away", {}).get("name", "")
        if _team_match(home, fh) and _team_match(away, fa):
            status = fix.get("fixture", {}).get("status", {}).get("short", "")
            fid    = fix.get("fixture", {}).get("id")
            return status, fid

    return "NOT_FOUND", None


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
                "minutes_played":  int(games.get("minutes") or 0),
                "started":         (not bool(games.get("substitute"))) if games.get("substitute") is not None else bool((games.get("minutes") or 0) > 45),
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
            if s.get("player_id") == player_id and (s.get("minutes_played") or 0) > 0:
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
