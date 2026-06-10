"""
Player Stats Data Fetcher — API-Football (Pro plan required).

Two collection modes:
  1. collect_match_history()   — per-fixture player stats (PROPER training data)
       GET /fixtures?league={id}&season={year}&last={n}  — fixture list
       GET /fixtures/players?fixture={id}               — per-match player stats
       ~1 + last_n requests per league. Cached permanently per fixture.

  2. collect_history() / fetch_league_player_stats()    — season aggregates (fallback)
       GET /players?league={id}&season={year}&page={n}
       Cached 7 days. Large page counts (50-250 pages per league).
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

BASE_URL          = "https://v3.football.api-sports.io"
CACHE_DIR         = config.BASE_DIR / "player_match_cache"
FIXTURE_CACHE_DIR = config.BASE_DIR / "fixture_player_cache"
CACHE_DIR.mkdir(exist_ok=True)
FIXTURE_CACHE_DIR.mkdir(exist_ok=True)

CACHE_DAYS    = 7
REQUEST_DELAY = 1.2
MAX_RETRIES   = 3

# Core betting leagues — default for collect_match_history()
APIFOOTBALL_LEAGUES: dict[str, tuple[int, str]] = {
    "Premier League":  (39,  "2024"),
    "Bundesliga":      (78,  "2024"),
    "La Liga":         (140, "2024"),
    "Serie A":         (135, "2024"),
    "Ligue 1":         (61,  "2024"),
    "Championship":    (40,  "2024"),
    "League One":      (41,  "2024"),
    "Bundesliga 2":    (79,  "2024"),
    "Ireland Premier": (357, "2025"),
    "Finland Veikk":   (244, "2025"),
    "World Cup":       (1,   "2026"),
}

EUROPEAN_CUPS: dict[str, tuple[int, str]] = {
    "Champions League": (2,   "2024"),
    "Europa League":    (3,   "2024"),
    "Conference League":(848, "2024"),
}

FBREF_LEAGUES = APIFOOTBALL_LEAGUES

WC26_LEAGUE_ID = 1
WC26_SEASON    = "2026"


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict) -> dict:
    key = os.getenv("APIFOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("APIFOOTBALL_KEY not set in environment")
    time.sleep(REQUEST_DELAY)

    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                f"{BASE_URL}/{endpoint}",
                headers={"x-apisports-key": key},
                params=params,
                timeout=20,
            )
            if r.status_code == 429:
                wait = 60 * attempt
                print(f"    [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"API-Football {r.status_code}: {r.text[:200]}")
            d = r.json()
            errors = d.get("errors", {})
            if errors:
                raise RuntimeError(f"API-Football errors: {errors}")
            return d
        except requests.exceptions.RequestException as e:
            last_exc = RuntimeError(f"network error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    raise last_exc


# ── Season-stats cache (7-day TTL) ────────────────────────────────────────────

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


# ── Fixture cache (permanent — fixture results never change) ──────────────────

def _fixture_cache_path(fixture_id: int) -> Path:
    return FIXTURE_CACHE_DIR / f"fix_{fixture_id}.json"


def _load_fixture_cache(fixture_id: int) -> Optional[list]:
    p = _fixture_cache_path(fixture_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_fixture_cache(fixture_id: int, data: list) -> None:
    p = _fixture_cache_path(fixture_id)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Per-fixture player stats (match-level data) ───────────────────────────────

def _fetch_league_fixtures(league_id: int, season: str, last_n: int = 99) -> list[dict]:
    """Fetch list of completed fixtures for a league. Single API call.
    Note: API caps 'last' at 99 — values >= 100 return an error."""
    last_n = min(last_n, 99)
    data = _api_get("fixtures", {
        "league": league_id,
        "season": season,
        "last":   last_n,
    })
    # Filter to only finished matches (FT, AET, PEN) in case any scheduled sneak in
    finished = {"FT", "AET", "PEN", "AWD", "WO"}
    return [f for f in data.get("response", [])
            if f.get("fixture", {}).get("status", {}).get("short", "") in finished]


def _fetch_fixture_player_stats(fixture_id: int) -> list[dict]:
    """
    Fetch per-player stats for one fixture.
    Cached permanently — fixture results never change.
    Returns list of team dicts: [{"team": {...}, "players": [...]}]
    """
    cached = _load_fixture_cache(fixture_id)
    if cached is not None:
        return cached

    data = _api_get("fixtures/players", {"fixture": fixture_id})
    response = data.get("response", [])

    if response:
        _save_fixture_cache(fixture_id, response)

    return response


def _parse_fixture_player(player_entry: dict, meta: dict, league: str) -> Optional[dict]:
    """Parse one player from a /fixtures/players response entry."""
    p = player_entry.get("player", {})
    name = p.get("name", "").strip()
    if not name:
        return None

    stats_list = player_entry.get("statistics", [{}])
    if not stats_list:
        return None
    stats = stats_list[0]

    games   = stats.get("games",   {})
    minutes = int(games.get("minutes") or 0)
    if minutes < 10:
        return None

    goals_d  = stats.get("goals",   {})
    shots_d  = stats.get("shots",   {})
    cards_d  = stats.get("cards",   {})
    passes_d = stats.get("passes",  {})

    return {
        "player_id":       int(p.get("id", abs(hash(name)) % 10_000_000)),
        "player_name":     name,
        "fixture_id":      meta["fixture_id"],
        "date":            meta["date"],
        "league":          league,
        "home_team":       meta["home_team"],
        "away_team":       meta["away_team"],
        "team":            meta["team"],
        "is_home":         int(meta["is_home"]),
        "position":        str(games.get("position") or p.get("position") or "").strip(),
        "minutes":         minutes,
        "goals":           int(goals_d.get("total")   or 0),
        "assists":         int(goals_d.get("assists")  or 0),
        "shots_total":     int(shots_d.get("total")   or 0),
        "shots_on_target": int(shots_d.get("on")      or 0),
        "yellow_cards":    int(cards_d.get("yellow")  or 0),
        "key_passes":      int(passes_d.get("key")    or 0),
        "rating":          float(games.get("rating")  or 0),
    }


def collect_match_history(
    leagues: dict | None = None,
    last_n:  int = 99,
) -> list[dict]:
    """
    Collect per-match player stats from API-Football.
    Returns flat list — one dict per player per match.

    API cost: 1 (fixture list) + last_n (player stats) per league.
    Fixture stats are cached permanently so re-runs are free.

    With last_n=100 and 8 leagues: ~808 API requests first run,
    ~8 requests on subsequent runs (only fixture list, rest cached).
    """
    if leagues is None:
        leagues = APIFOOTBALL_LEAGUES

    all_rows: list[dict] = []

    for league_name, (league_id, season) in leagues.items():
        print(f"  [{league_name}] Fetching fixture list (last {last_n})...")

        try:
            fixtures = _fetch_league_fixtures(league_id, season, last_n)
        except RuntimeError as e:
            print(f"  [{league_name}] fixture list error: {e}")
            continue

        if not fixtures:
            print(f"  [{league_name}] No completed fixtures found")
            continue

        cached_count = sum(
            1 for fix in fixtures
            if _fixture_cache_path(fix.get("fixture", {}).get("id", 0)).exists()
        )
        print(f"  [{league_name}] {len(fixtures)} fixtures ({cached_count} cached, "
              f"{len(fixtures) - cached_count} to fetch)...")

        league_rows = 0
        for fix in fixtures:
            fixture_id = fix.get("fixture", {}).get("id")
            if not fixture_id:
                continue

            fix_date  = fix.get("fixture", {}).get("date", "")[:10]
            home_team = fix.get("teams", {}).get("home", {}).get("name", "")
            away_team = fix.get("teams", {}).get("away", {}).get("name", "")

            try:
                team_data_list = _fetch_fixture_player_stats(fixture_id)
            except RuntimeError as e:
                print(f"    [fix {fixture_id}] error: {e}")
                continue

            for team_data in team_data_list:
                team_name = team_data.get("team", {}).get("name", "")
                is_home   = (team_name == home_team)
                meta = {
                    "fixture_id": fixture_id,
                    "date":       fix_date,
                    "home_team":  home_team,
                    "away_team":  away_team,
                    "team":       team_name,
                    "is_home":    is_home,
                }
                for player_entry in team_data.get("players", []):
                    parsed = _parse_fixture_player(player_entry, meta, league_name)
                    if parsed:
                        all_rows.append(parsed)
                        league_rows += 1

        print(f"  [{league_name}] {league_rows} player-match rows collected")

    print(f"[DONE] {len(all_rows)} total player-match rows across {len(leagues)} leagues.")
    return all_rows


# ── World Cup 2026 national team collection ───────────────────────────────────

def _fetch_wc26_teams() -> list[dict]:
    """Fetch all 48 WC2026 teams from API-Football. Returns list of team dicts."""
    data = _api_get("teams", {"league": WC26_LEAGUE_ID, "season": WC26_SEASON})
    return data.get("response", [])


def collect_national_team_history(last_n: int = 10) -> list[dict]:
    """
    Collect per-match player stats for all WC2026 national teams.

    For each of the 48 WC teams:
      1. GET /fixtures?team={id}&last={n}  — last N fixtures (qualifiers, NL, friendlies)
      2. GET /fixtures/players?fixture={id} — player stats (cached permanently)

    API cost: 1 (team list) + 48 (fixture lists) + up to 48*last_n (player stats, cached).
    On first run ~529 requests; subsequent runs ~49 (everything else cached).
    """
    last_n = min(last_n, 99)
    finished = {"FT", "AET", "PEN", "AWD", "WO"}

    print(f"[wc26] Fetching WC2026 team list...")
    try:
        teams = _fetch_wc26_teams()
    except RuntimeError as e:
        print(f"[wc26] Failed to fetch team list: {e}")
        return []

    if not teams:
        print("[wc26] No WC2026 teams found — tournament may not be registered yet.")
        return []

    print(f"[wc26] {len(teams)} teams found. Collecting last {last_n} fixtures each...")
    all_rows: list[dict] = []

    for entry in teams:
        team_info = entry.get("team", {})
        team_id   = team_info.get("id")
        team_name = team_info.get("name", f"team_{team_id}")

        try:
            data = _api_get("fixtures", {"team": team_id, "last": last_n})
            fixtures = [
                f for f in data.get("response", [])
                if f.get("fixture", {}).get("status", {}).get("short", "") in finished
            ]
        except RuntimeError as e:
            print(f"  [{team_name}] fixture list error: {e}")
            continue

        if not fixtures:
            print(f"  [{team_name}] No completed fixtures found")
            continue

        cached_count = sum(
            1 for f in fixtures
            if _fixture_cache_path(f.get("fixture", {}).get("id", 0)).exists()
        )
        team_rows = 0
        for fix in fixtures:
            fixture_id = fix.get("fixture", {}).get("id")
            if not fixture_id:
                continue

            fix_date  = fix.get("fixture", {}).get("date", "")[:10]
            home_team = fix.get("teams", {}).get("home", {}).get("name", "")
            away_team = fix.get("teams", {}).get("away", {}).get("name", "")
            league_name = fix.get("league", {}).get("name", "International")

            try:
                team_data_list = _fetch_fixture_player_stats(fixture_id)
            except RuntimeError as e:
                print(f"    [fix {fixture_id}] error: {e}")
                continue

            for team_data in team_data_list:
                t_name  = team_data.get("team", {}).get("name", "")
                is_home = (t_name == home_team)
                meta = {
                    "fixture_id": fixture_id,
                    "date":       fix_date,
                    "home_team":  home_team,
                    "away_team":  away_team,
                    "team":       t_name,
                    "is_home":    is_home,
                }
                for player_entry in team_data.get("players", []):
                    parsed = _parse_fixture_player(player_entry, meta, league_name)
                    if parsed:
                        all_rows.append(parsed)
                        team_rows += 1

        print(f"  [{team_name}] {len(fixtures)} fixtures ({cached_count} cached) -> {team_rows} rows")

    print(f"[wc26] Done. {len(all_rows)} player-match rows across {len(teams)} national teams.")
    return all_rows


# ── Season-level collection (fallback / supplementary) ────────────────────────

def _parse_player(entry: dict, league: str) -> Optional[dict]:
    p    = entry.get("player", {})
    name = p.get("name", "").strip()
    if not name:
        return None

    stats = entry.get("statistics", [{}])[0]
    games  = stats.get("games",  {})
    goals  = stats.get("goals",  {})
    shots  = stats.get("shots",  {})
    cards  = stats.get("cards",  {})
    passes = stats.get("passes", {})
    team   = stats.get("team",   {})

    appearances = int(games.get("appearences") or 0)
    minutes     = int(games.get("minutes")     or 0)
    if appearances < config.MIN_APPEARANCES or minutes < 1:
        return None

    return {
        "player_id":       int(p.get("id", abs(hash(name)) % 10_000_000)),
        "player_name":     name,
        "team":            team.get("name", ""),
        "league":          league,
        "position":        str(games.get("position") or p.get("position") or "").strip(),
        "appearances":     appearances,
        "minutes":         minutes,
        "goals":           int(goals.get("total")   or 0),
        "assists":         int(goals.get("assists")  or 0),
        "shots_total":     int(shots.get("total")   or 0),
        "shots_on_target": int(shots.get("on")      or 0),
        "yellow_cards":    int(cards.get("yellow")  or 0),
        "key_passes":      int(passes.get("key")    or 0),
        "rating":          float(games.get("rating") or 0),
    }


def fetch_league_player_stats(league: str, force_refresh: bool = False) -> list[dict]:
    """Season-aggregate player stats (fallback). Cached 7 days."""
    _all_leagues = {**APIFOOTBALL_LEAGUES, **EUROPEAN_CUPS}
    info = _all_leagues.get(league)
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

    print(f"  [{league}] Fetching season stats (league={lg_id}, season={season})...")

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

        print(f"    page {page}/{total_pg} -- {len(entries)} entries")

        if page >= total_pg:
            break
        page += 1

    if players:
        _save_cache(cache_key, players)

    print(f"  [{league}] {len(players)} players with >={config.MIN_APPEARANCES} appearances")
    return players


def collect_history(
    max_fixtures: int = 800,
    leagues: dict | None = None,
    seasons: list[str] | None = None,
) -> list[dict]:
    """Season-aggregate collection (fallback path). Prefer collect_match_history()."""
    if leagues is None:
        leagues = APIFOOTBALL_LEAGUES

    all_rows: list[dict] = []
    for league in leagues:
        rows = fetch_league_player_stats(league, force_refresh=False)
        all_rows.extend(rows)

    print(f"[DONE] {len(all_rows)} total player rows across {len(leagues)} leagues.")
    return all_rows
