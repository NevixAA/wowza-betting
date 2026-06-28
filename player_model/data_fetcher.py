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

import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config

BASE_URL          = "https://v3.football.api-sports.io"
CACHE_DIR         = config.BASE_DIR / "player_match_cache"
FIXTURE_CACHE_DIR = config.BASE_DIR / "fixture_player_cache"
CACHE_DIR.mkdir(exist_ok=True)
FIXTURE_CACHE_DIR.mkdir(exist_ok=True)
EVENTS_CACHE_DIR = config.BASE_DIR / "fixture_events_cache"
EVENTS_CACHE_DIR.mkdir(exist_ok=True)

CACHE_DAYS    = 7
REQUEST_DELAY = 0.15  # ~6.7 calls/sec = ~400/min — uses the Ultra plan's 450/min cap (was 0.5=120/min,
                      # leaving 73% of Ultra's rate unused). 429s self-heal via the rate-limit backoff.
MAX_RETRIES   = 3

# Global rate-limiter: enforces REQUEST_DELAY between API calls across ALL threads.
# Without this, each thread sleeps independently → no real throughput gain.
_rate_lock      = threading.Lock()
_last_call_time = 0.0


def _rate_limited_sleep() -> None:
    global _last_call_time
    with _rate_lock:
        now     = time.monotonic()
        elapsed = now - _last_call_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        _last_call_time = time.monotonic()

# Core betting leagues — default for collect_match_history()
# Current season per league — used for odds/predict pipeline (single season reference)
APIFOOTBALL_LEAGUES: dict[str, tuple[int, str]] = {
    "Premier League":  (39,  "2025"),
    "Bundesliga":      (78,  "2025"),
    "La Liga":         (140, "2025"),
    "Serie A":         (135, "2025"),
    "Ligue 1":         (61,  "2025"),
    "Championship":    (40,  "2025"),
    "League One":      (41,  "2025"),
    "Bundesliga 2":    (79,  "2025"),
    "Ireland Premier": (357, "2026"),
    "Finland Veikk":   (244, "2026"),
    "World Cup":       (1,   "2026"),
}

# Multi-season collect config: (league_id, season, last_n)
# Historical seasons: large last_n to capture full season (cached — free on re-run)
# Current season: last_n=150 so weekly runs pick up new matches automatically
COLLECT_SEASONS: dict[str, list[tuple[int, str, int]]] = {
    # (league_id, season, last_n)
    # last_n >= 100 → full-season fetch (all completed fixtures, for historical seasons)
    # last_n < 100  → rolling last-N via API `last=` param (current season updates)
    "Premier League":  [(39,  "2024", 500), (39,  "2025", 99)],
    "Bundesliga":      [(78,  "2024", 400), (78,  "2025", 99)],
    "La Liga":         [(140, "2024", 400), (140, "2025", 99)],
    "Serie A":         [(135, "2024", 400), (135, "2025", 99)],
    "Ligue 1":         [(61,  "2024", 400), (61,  "2025", 99)],
    "Championship":    [(40,  "2024", 600), (40,  "2025", 99)],
    "League One":      [(41,  "2024", 600), (41,  "2025", 99)],
    "Bundesliga 2":    [(79,  "2024", 400), (79,  "2025", 99)],
    "Ireland Premier": [(357, "2025", 300), (357, "2026", 99)],
    "Finland Veikk":   [(244, "2025", 300), (244, "2026", 99)],
    "World Cup":       [(1,   "2026", 99)],
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
    _rate_limited_sleep()

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

FIXTURE_STATS_CACHE_DIR = config.BASE_DIR / "fixture_stats_cache"
FIXTURE_STATS_CACHE_DIR.mkdir(exist_ok=True)

LINEUP_CACHE_DIR = config.BASE_DIR / "lineup_cache"
LINEUP_CACHE_DIR.mkdir(exist_ok=True)
LINEUP_CACHE_TTL_HOURS = 4  # re-fetch after 4h — lineups can change until confirmed


def _fixture_cache_path(fixture_id: int) -> Path:
    return FIXTURE_CACHE_DIR / f"fix_{fixture_id}.json"


def _fixture_stats_cache_path(fixture_id: int) -> Path:
    return FIXTURE_STATS_CACHE_DIR / f"stats_{fixture_id}.json"


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


def _fetch_fixture_statistics(fixture_id: int) -> dict:
    """
    Fetch team-level statistics for a completed fixture.
    Cached permanently. Returns {team_name: {corners: N, shots_total: N, possession: N}}.
    """
    cache_p = _fixture_stats_cache_path(fixture_id)
    if cache_p.exists():
        try:
            return json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        data = _api_get("fixtures/statistics", {"fixture": fixture_id})
    except RuntimeError:
        return {}

    result = {}
    for team_data in data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "")
        stats = {s.get("type", ""): s.get("value") for s in team_data.get("statistics", [])}
        result[team_name] = {
            "corners":     int(stats.get("Corner Kicks") or 0),
            "shots_total": int(stats.get("Total Shots") or 0),
            "possession":  float(str(stats.get("Ball Possession") or "0%").replace("%", "") or 0),
        }

    if result:
        cache_p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    return result


# ── Lineup fetch (pre-match starting XI) ─────────────────────────────────────

def fetch_lineup(fixture_id: int) -> dict:
    """
    Fetch confirmed starting XI for a fixture.
    Returns:
      confirmed  : bool — True if lineup has been announced
      starters   : set[int] — player_ids in starting XI (both teams)
      subs       : set[int] — player_ids on bench (both teams)
      home_starters / away_starters : set[int] — per-team splits
    Cached for LINEUP_CACHE_TTL_HOURS — re-fetched until confirmed.
    """
    cache_p = LINEUP_CACHE_DIR / f"lineup_{fixture_id}.json"
    if cache_p.exists():
        try:
            raw = json.loads(cache_p.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(raw.get("fetched_at", "2000-01-01"))
            if datetime.now() - fetched < timedelta(hours=LINEUP_CACHE_TTL_HOURS):
                raw["starters"]       = set(raw.get("starters", []))
                raw["subs"]           = set(raw.get("subs", []))
                raw["home_starters"]  = set(raw.get("home_starters", []))
                raw["away_starters"]  = set(raw.get("away_starters", []))
                return raw
        except Exception:
            pass

    empty = {"confirmed": False, "starters": set(), "subs": set(),
             "home_starters": set(), "away_starters": set(), "home_team": "", "away_team": ""}
    try:
        resp = _api_get("fixtures/lineups", {"fixture": fixture_id})
    except RuntimeError:
        return empty

    lineups = resp.get("response", [])
    if not lineups:
        return empty

    starters: set[int] = set()
    subs:     set[int] = set()
    home_starters: set[int] = set()
    away_starters: set[int] = set()
    home_team = ""
    away_team = ""

    for i, team_lu in enumerate(lineups):
        team_name = team_lu.get("team", {}).get("name", "")
        is_home = (i == 0)
        if is_home:
            home_team = team_name
        else:
            away_team = team_name
        for p in team_lu.get("startXI", []):
            pid = p.get("player", {}).get("id")
            if pid:
                pid = int(pid)
                starters.add(pid)
                (home_starters if is_home else away_starters).add(pid)
        for p in team_lu.get("substitutes", []):
            pid = p.get("player", {}).get("id")
            if pid:
                subs.add(int(pid))

    confirmed = bool(starters)
    result = {
        "confirmed":      confirmed,
        "home_team":      home_team,
        "away_team":      away_team,
        "starters":       list(starters),
        "subs":           list(subs),
        "home_starters":  list(home_starters),
        "away_starters":  list(away_starters),
        "fetched_at":     datetime.now().isoformat(),
    }
    if confirmed:
        cache_p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    result["starters"]       = starters
    result["subs"]           = subs
    result["home_starters"]  = home_starters
    result["away_starters"]  = away_starters
    return result


# ── Squad sync (transfer tracking) ───────────────────────────────────────────

def fetch_current_squad(team_id: int, season: str) -> list[dict]:
    """
    Fetch current squad members for a team.
    Returns list of {player_id, player_name, position}.
    Used by squad-sync mode to detect transfers.
    """
    try:
        data = _api_get("players/squads", {"team": team_id})
        players = []
        for squad in data.get("response", []):
            for p in squad.get("players", []):
                pid = p.get("id")
                if pid:
                    players.append({
                        "player_id":   int(pid),
                        "player_name": p.get("name", ""),
                        "position":    p.get("position", ""),
                    })
        return players
    except RuntimeError:
        return []


def fetch_league_teams(league_id: int, season: str) -> list[dict]:
    """
    Fetch all teams in a league season.
    Returns list of {team_id, team_name}.
    """
    try:
        data = _api_get("teams", {"league": league_id, "season": season})
        return [
            {
                "team_id":   int(t.get("team", {}).get("id", 0)),
                "team_name": t.get("team", {}).get("name", ""),
            }
            for t in data.get("response", [])
            if t.get("team", {}).get("id")
        ]
    except RuntimeError:
        return []


# ── Per-fixture player stats (match-level data) ───────────────────────────────

def _fetch_league_fixtures(league_id: int, season: str, last_n: int = 99) -> list[dict]:
    """
    Fetch completed fixtures for a league season.
    - last_n < 100: uses API `last=N` (rolling update, current season).
    - last_n >= 100: fetches all completed fixtures for the season (no `last` cap).
      Use this for historical seasons to get the full dataset in one call.
    """
    finished = {"FT", "AET", "PEN", "AWD", "WO"}
    if last_n >= 100:
        # Full-season fetch — no `last` cap
        data = _api_get("fixtures", {"league": league_id, "season": season})
    else:
        data = _api_get("fixtures", {
            "league": league_id,
            "season": season,
            "last":   last_n,
        })
    return [
        {
            "id":      fix.get("fixture", {}).get("id"),
            "date":    str(fix.get("fixture", {}).get("date", ""))[:10],
            "home":    fix.get("teams", {}).get("home", {}).get("name", ""),
            "away":    fix.get("teams", {}).get("away", {}).get("name", ""),
            "referee": fix.get("fixture", {}).get("referee", "") or "",
            "_raw":    fix,
        }
        for fix in data.get("response", [])
        if fix.get("fixture", {}).get("status", {}).get("short", "") in finished
    ]


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

    goals_d    = stats.get("goals",    {})
    shots_d    = stats.get("shots",    {})
    cards_d    = stats.get("cards",    {})
    passes_d   = stats.get("passes",   {})
    duels_d    = stats.get("duels",    {})
    fouls_d    = stats.get("fouls",    {})
    tackles_d  = stats.get("tackles",  {})
    dribbles_d = stats.get("dribbles", {})
    penalty_d  = stats.get("penalty",  {})

    substitute = games.get("substitute")
    started = (not bool(substitute)) if substitute is not None else bool(minutes > 45)

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
        "referee":         meta.get("referee", ""),
        "season":          meta.get("season", ""),
        "position":        str(games.get("position") or p.get("position") or "").strip(),
        "minutes":         minutes,
        "started":         int(started),
        "goals":           int(goals_d.get("total")   or 0),
        "assists":         int(goals_d.get("assists")  or 0),
        "shots_total":     int(shots_d.get("total")   or 0),
        "shots_on_target": int(shots_d.get("on")      or 0),
        "yellow_cards":    int(cards_d.get("yellow")  or 0),
        "key_passes":      int(passes_d.get("key")    or 0),
        "rating":          float(games.get("rating")  or 0),
        "fouls_committed": int(fouls_d.get("committed") or 0),
        "fouls_drawn":     int(fouls_d.get("drawn")     or 0),
        "duels_total":     int(duels_d.get("total")     or 0),
        "duels_won":       int(duels_d.get("won")       or 0),
        "team_corners":    int(meta.get("team_corners", 0)),
        # Defensive / technical
        "red_cards":           int(cards_d.get("red")                or 0),
        "passes_total":        int(passes_d.get("total")             or 0),
        "offsides":            int(stats.get("offsides")             or 0),
        "tackles_total":       int(tackles_d.get("total")            or 0),
        "interceptions":       int(tackles_d.get("interceptions")    or 0),
        "dribbles_attempted":  int(dribbles_d.get("attempts")        or 0),
        "dribbles_success":    int(dribbles_d.get("success")         or 0),
        "dribbles_past":       int(dribbles_d.get("past")            or 0),
        # Goalkeeper
        "goals_conceded":      int(goals_d.get("conceded")           or 0),
        "saves":               int(goals_d.get("saves")              or 0),
        # Penalty
        "penalty_scored":      int(penalty_d.get("scored")           or 0),
        "penalty_missed":      int(penalty_d.get("missed")           or 0),
        "penalty_won":         int(penalty_d.get("won")              or 0),
        "penalty_saved":       int(penalty_d.get("saved")            or 0),
    }


def _collect_one_season(
    league_name: str,
    league_id:   int,
    season:      str,
    last_n:      int,
) -> list[dict]:
    """Collect player-match rows for a single league+season. Fixture stats are permanently cached."""
    print(f"  [{league_name} {season}] Fetching fixture list (last {last_n})...")
    try:
        fixtures = _fetch_league_fixtures(league_id, season, last_n)
    except RuntimeError as e:
        print(f"  [{league_name} {season}] fixture list error: {e}")
        return []

    if not fixtures:
        print(f"  [{league_name} {season}] No completed fixtures found")
        return []

    cached_count = sum(
        1 for fix in fixtures
        if _fixture_cache_path(fix.get("id") or fix.get("fixture", {}).get("id", 0)).exists()
    )
    to_fetch = len(fixtures) - cached_count
    print(f"  [{league_name} {season}] {len(fixtures)} fixtures ({cached_count} cached, {to_fetch} to fetch)...")

    rows: list[dict] = []
    for fix in fixtures:
        if "_raw" in fix:
            fixture_id = fix["id"]
            fix_date   = fix["date"]
            home_team  = fix["home"]
            away_team  = fix["away"]
            referee    = fix.get("referee", "")
        else:
            fixture_id = fix.get("fixture", {}).get("id")
            fix_date   = fix.get("fixture", {}).get("date", "")[:10]
            home_team  = fix.get("teams", {}).get("home", {}).get("name", "")
            away_team  = fix.get("teams", {}).get("away", {}).get("name", "")
            referee    = fix.get("fixture", {}).get("referee", "") or ""
        if not fixture_id:
            continue

        try:
            team_data_list = _fetch_fixture_player_stats(fixture_id)
        except RuntimeError as e:
            print(f"    [fix {fixture_id}] error: {e}")
            continue

        fix_stats = _fetch_fixture_statistics(fixture_id)

        for team_data in team_data_list:
            team_name = team_data.get("team", {}).get("name", "")
            team_st   = fix_stats.get(team_name, {})
            meta = {
                "fixture_id":  fixture_id,
                "date":        fix_date,
                "home_team":   home_team,
                "away_team":   away_team,
                "team":        team_name,
                "is_home":     (team_name == home_team),
                "team_corners": team_st.get("corners", 0),
                "referee":     referee,
                "season":      season,
            }
            for player_entry in team_data.get("players", []):
                parsed = _parse_fixture_player(player_entry, meta, league_name)
                if parsed:
                    rows.append(parsed)

    print(f"  [{league_name} {season}] {len(rows)} player-match rows collected")
    return rows


def collect_match_history(
    leagues: dict | None = None,
    last_n:  int = 99,
) -> list[dict]:
    """
    Collect per-match player stats across ALL configured seasons per league.

    Uses COLLECT_SEASONS by default (multi-season: 2024 + 2025 per league).
    Fixture stats are cached permanently — re-runs only fetch NEW fixtures,
    making weekly updates very cheap (only current-season new games are fetched).

    Pass a custom `leagues` dict (name → (id, season)) to override with a
    single-season collection for a specific league.
    """
    # Multi-season path (default)
    if leagues is None:
        all_rows: list[dict] = []
        seen_fixtures: set = set()
        for league_name, season_list in COLLECT_SEASONS.items():
            for (league_id, season, season_last_n) in season_list:
                rows = _collect_one_season(league_name, league_id, season, season_last_n)
                for r in rows:
                    key = (r.get("fixture_id"), r.get("player_id"))
                    if key not in seen_fixtures:
                        seen_fixtures.add(key)
                        all_rows.append(r)
        n_leagues = len(COLLECT_SEASONS)
        n_seasons = sum(len(v) for v in COLLECT_SEASONS.values())
        print(f"[DONE] {len(all_rows)} total player-match rows across {n_leagues} leagues / {n_seasons} season-slots.")
        return all_rows

    # Single-season override path (used when caller passes explicit leagues dict)
    all_rows = []
    for league_name, (league_id, season) in leagues.items():
        all_rows.extend(_collect_one_season(league_name, league_id, season, last_n))
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

            fix_stats = _fetch_fixture_statistics(fixture_id)

            for team_data in team_data_list:
                t_name  = team_data.get("team", {}).get("name", "")
                is_home = (t_name == home_team)
                team_st = fix_stats.get(t_name, {})
                meta = {
                    "fixture_id":   fixture_id,
                    "date":         fix_date,
                    "home_team":    home_team,
                    "away_team":    away_team,
                    "team":         t_name,
                    "is_home":      is_home,
                    "team_corners": team_st.get("corners", 0),
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


# ─────────────────────────────────────────────────────────────────────────────
# Player season statistics (/players/statistics)
# Stable season-level priors: more data than last-5 rolling, more current than
# career averages. Used as supplementary features alongside rolling averages.
# ~1 call per unique player per season. Cached 24h.
# ─────────────────────────────────────────────────────────────────────────────

_SEASON_STATS_CACHE_DIR = config.BASE_DIR / "player_season_stats_cache"
_SEASON_STATS_CACHE_DIR.mkdir(exist_ok=True)


def _season_stats_cache_path(player_id: int, league_id: int, season: str) -> Path:
    return _SEASON_STATS_CACHE_DIR / f"ss_{player_id}_{league_id}_{season}.json"


def fetch_player_season_stats(player_id: int, league_id: int, season: str) -> dict:
    """
    Fetch season-aggregate stats for one player from /players/statistics.
    Returns per-game averages: season_goals_pg, season_sot_pg, season_shots_pg,
    season_assists_pg, season_cards_pg, season_minutes_pg, season_appearances.
    Cached 24h. Returns {} on any failure.
    """
    key = os.getenv("APIFOOTBALL_KEY", "")
    if not key:
        return {}

    cache_p = _season_stats_cache_path(player_id, league_id, season)
    if cache_p.exists():
        try:
            data = json.loads(cache_p.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - ts < timedelta(hours=24):
                return data.get("stats", {})
        except Exception:
            pass

    try:
        data = _api_get("players", {
            "id": player_id, "league": league_id, "season": season,
        })
    except RuntimeError:
        return {}

    resp = data.get("response", [])
    if not resp:
        return {}

    # Extract and cache profile data for free — same response, zero extra calls
    _player_obj = resp[0].get("player", {})
    if _player_obj:
        _profile_p = CACHE_DIR / f"profile_{player_id}.json"
        if not _profile_p.exists():
            try:
                _h = _player_obj.get("height") or ""
                _w = _player_obj.get("weight") or ""
                _profile_p.write_text(json.dumps({
                    "age":       int(_player_obj.get("age") or 0),
                    "height_cm": float(str(_h).replace("cm", "").strip() or 0),
                    "weight_kg": float(str(_w).replace("kg", "").strip() or 0),
                }, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    stats_list = resp[0].get("statistics", [])
    if not stats_list:
        return {}

    s       = stats_list[0]
    games   = s.get("games",  {})
    goals   = s.get("goals",  {})
    shots   = s.get("shots",  {})
    cards   = s.get("cards",  {})

    apps = max(int(games.get("appearences") or 0), 1)
    result = {
        "season_appearances":  apps,
        "season_goals_pg":     (int(goals.get("total")   or 0)) / apps,
        "season_assists_pg":   (int(goals.get("assists")  or 0)) / apps,
        "season_shots_pg":     (int(shots.get("total")   or 0)) / apps,
        "season_sot_pg":       (int(shots.get("on")      or 0)) / apps,
        "season_cards_pg":     (int(cards.get("yellow")  or 0) + int(cards.get("red") or 0)) / apps,
        "season_minutes_pg":   (int(games.get("minutes") or 0)) / apps,
    }

    # Additional season-level fields (from same API response, zero extra calls)
    lineups      = s.get("games", {}).get("lineups", 0) or 0
    passes_obj   = s.get("passes",    {}) or {}
    dribbles_obj = s.get("dribbles", {}) or {}
    fouls_obj    = s.get("fouls",     {}) or {}

    result["season_start_rate"]      = (int(lineups) / apps) if apps > 0 else 0.0
    result["season_pass_accuracy"]   = float(passes_obj.get("accuracy") or 0) / 100.0  # convert % to 0-1
    result["season_dribble_pg"]      = (int(dribbles_obj.get("success") or 0)) / apps
    result["season_fouls_pg"]        = (int(fouls_obj.get("committed") or 0)) / apps
    result["season_fouls_drawn_pg"]  = (int(fouls_obj.get("drawn") or 0)) / apps

    cache_p.write_text(
        json.dumps({"_cached_at": datetime.now().isoformat(), "stats": result},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def fetch_all_player_season_stats(
    df: "pd.DataFrame",
    leagues: dict | None = None,
    max_workers: int = 5,
) -> dict[int, dict]:
    """
    Batch-fetch season stats for all unique player_ids in df.
    Matches player → league via the 'league' column in df.
    Returns {player_id: stats_dict}. Used in --mode enrich-season.
    Also caches profile data (age/height) for free from the same response.
    """
    if leagues is None:
        leagues = APIFOOTBALL_LEAGUES

    # Build deduplicated task list first
    tasks: list[tuple[int, int, str]] = []
    seen: set[int] = set()
    for _, row in df.iterrows():
        pid = int(row.get("player_id", 0))
        if not pid or pid in seen:
            continue
        seen.add(pid)
        lg   = str(row.get("league", ""))
        info = leagues.get(lg)
        if not info:
            continue
        league_id, season = info
        tasks.append((pid, league_id, season))

    results: dict[int, dict] = {}

    def _fetch_one(args: tuple) -> tuple[int, dict]:
        pid, league_id, season = args
        try:
            return pid, fetch_player_season_stats(pid, league_id, season)
        except Exception:
            return pid, {}

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            pid, stats = future.result()
            if stats:
                results[pid] = stats
            done += 1
            if done % 200 == 0:
                print(f"[season-stats] {done}/{len(tasks)} players fetched...")

    print(f"[season-stats] Fetched stats for {len(results)}/{len(seen)} players.")
    return results


def fetch_player_profile(player_id: int) -> dict:
    """
    Fetch static player profile: age, height, weight, nationality.
    Cached indefinitely (profiles don't change meaningfully).
    Endpoint: /players?id={player_id}&season=2025
    """
    cache_p = CACHE_DIR / f"profile_{player_id}.json"
    if cache_p.exists():
        try:
            return json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    data = _api_get("players", {"id": player_id, "season": "2025"})
    resp = data.get("response", [])
    if not resp:
        return {}

    player = resp[0].get("player", {})
    age_val    = player.get("age") or 0
    height_str = player.get("height") or ""
    weight_str = player.get("weight") or ""

    def _parse_cm(s: str) -> float:
        # "185 cm" -> 185.0
        try:
            return float(str(s).replace("cm", "").strip())
        except Exception:
            return 0.0

    def _parse_kg(s: str) -> float:
        try:
            return float(str(s).replace("kg", "").strip())
        except Exception:
            return 0.0

    result = {
        "age":        int(age_val),
        "height_cm":  _parse_cm(height_str),
        "weight_kg":  _parse_kg(weight_str),
    }

    cache_p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def fetch_all_player_profiles(df: "pd.DataFrame", max_workers: int = 5) -> dict:
    """
    Batch fetch profiles for all unique player_ids in df.
    Returns {player_id: profile_dict}.
    Note: enrich-season already caches profiles as a side-effect, so most
    calls here will be instant cache hits after running enrich-season.
    """
    unique_pids = list(df["player_id"].dropna().astype(int).unique()) if "player_id" in df.columns else []
    total = len(unique_pids)
    results = {}

    def _fetch_one(pid: int) -> tuple[int, dict]:
        try:
            return pid, fetch_player_profile(pid)
        except Exception:
            return pid, {}

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, int(pid)): pid for pid in unique_pids}
        for future in as_completed(futures):
            pid, profile = future.result()
            results[pid] = profile
            done += 1
            if done % 500 == 0:
                print(f"[profiles] {done}/{total} profiles fetched...")

    print(f"[profiles] Done. {len(results)} profiles fetched.")
    return results


def fetch_player_sidelined(player_id: int) -> list[dict]:
    """
    Fetch a player's injury/sidelined history.
    Cached 30 days (past injuries don't change, new ones are added).
    Endpoint: /sidelined?player={player_id}
    Returns list of {reason, start, end} dicts.
    """
    cache_p = CACHE_DIR / f"sidelined_{player_id}.json"
    if cache_p.exists():
        try:
            cached = json.loads(cache_p.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
            if (datetime.now() - cached_at).days < 30:
                return cached.get("data", [])
        except Exception:
            pass

    data = _api_get("sidelined", {"player": player_id})
    resp = data.get("response", [])

    result = []
    for entry in resp:
        result.append({
            "reason": entry.get("type", ""),
            "start":  str(entry.get("start", ""))[:10],
            "end":    str(entry.get("end", ""))[:10] if entry.get("end") else "",
        })

    cache_p.write_text(
        json.dumps({"_cached_at": datetime.now().isoformat(), "data": result},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def fetch_all_player_sidelined(df: "pd.DataFrame", max_workers: int = 5) -> dict:
    """
    Batch fetch sidelined history for all unique player_ids in df.
    Returns {player_id: [sidelined_entries]}.
    """
    unique_pids = list(df["player_id"].dropna().astype(int).unique()) if "player_id" in df.columns else []
    total = len(unique_pids)
    results = {}

    def _fetch_one(pid: int) -> tuple[int, list]:
        try:
            return pid, fetch_player_sidelined(pid)
        except Exception:
            return pid, []

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, int(pid)): pid for pid in unique_pids}
        for future in as_completed(futures):
            pid, data = future.result()
            results[pid] = data
            done += 1
            if done % 500 == 0:
                print(f"[sidelined] {done}/{total} players fetched...")

    print(f"[sidelined] Done. {len(results)} players processed.")
    return results


# ── Fixture goal events (set-piece data enrichment) ──────────────────────────

def _fetch_fixture_events(fixture_id: int) -> list[dict]:
    """Fetch all goal events for one fixture. Cached permanently."""
    cache_f = EVENTS_CACHE_DIR / f"events_{fixture_id}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text(encoding="utf-8"))
    _rate_limited_sleep()
    api_key = os.getenv("APIFOOTBALL_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.get(
            f"{BASE_URL}/fixtures/events",
            headers={"x-apisports-key": api_key},
            params={"fixture": fixture_id, "type": "Goal"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json().get("response", [])
        cache_f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    except Exception:
        return []


def _parse_sp_events(events: list[dict], fixture_id: int) -> list[dict]:
    """Parse goal events into per-player set piece rows for one fixture."""
    rows = []
    for ev in events:
        detail     = (ev.get("detail") or "").lower()
        is_header  = "header"    in detail
        is_fk      = "free kick" in detail
        is_penalty = "penalty"   in detail
        is_sp      = (is_header or is_fk) and not is_penalty

        scorer = ev.get("player") or {}
        if scorer.get("id"):
            rows.append({
                "fixture_id":  fixture_id,
                "player_id":   scorer["id"],
                "sp_goal":     1 if is_sp     else 0,
                "headed_goal": 1 if is_header else 0,
                "fk_goal":     1 if is_fk     else 0,
                "sp_assist":   0,
                "fk_assist":   0,
            })

        assist = ev.get("assist") or {}
        if assist.get("id"):
            rows.append({
                "fixture_id":  fixture_id,
                "player_id":   assist["id"],
                "sp_goal":     0,
                "headed_goal": 0,
                "fk_goal":     0,
                "sp_assist":   1 if is_sp else 0,
                "fk_assist":   1 if is_fk else 0,
            })
    return rows


def enrich_sp_events(history: list[dict], max_workers: int = 20) -> list[dict]:
    """
    Enrich match history rows with set piece event data.
    Fetches /fixtures/events for each fixture in parallel. Adds columns:
      sp_goal, headed_goal, fk_goal, sp_assist, fk_assist  (all int, default 0)
    Fully cached — subsequent runs cost 0 API calls.
    """
    import pandas as _pd

    SP_COLS = ["sp_goal", "headed_goal", "fk_goal", "sp_assist", "fk_assist"]
    fixture_ids = list({int(r["fixture_id"]) for r in history if r.get("fixture_id")})
    uncached = sum(1 for fid in fixture_ids
                   if not (EVENTS_CACHE_DIR / f"events_{fid}.json").exists())
    print(f"  [sp_events] {len(fixture_ids):,} fixtures "
          f"({uncached:,} need API calls)...")

    events_by_fixture: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {pool.submit(_fetch_fixture_events, fid): fid for fid in fixture_ids}
        for fut in as_completed(fut_map):
            fid = fut_map[fut]
            events_by_fixture[fid] = fut.result() or []

    sp_rows: list[dict] = []
    for fid, events in events_by_fixture.items():
        sp_rows.extend(_parse_sp_events(events, fid))

    if not sp_rows:
        for row in history:
            for col in SP_COLS:
                row.setdefault(col, 0)
        return history

    sp_df = (
        _pd.DataFrame(sp_rows)
        .groupby(["fixture_id", "player_id"], as_index=False)
        .sum()
    )

    hist_df = _pd.DataFrame(history)
    if "player_id" not in hist_df.columns:
        for row in history:
            for col in SP_COLS:
                row.setdefault(col, 0)
        return history

    hist_df["player_id"] = hist_df["player_id"].astype(int)
    hist_df["fixture_id"] = hist_df["fixture_id"].astype(int)
    sp_df["player_id"]   = sp_df["player_id"].astype(int)
    sp_df["fixture_id"]  = sp_df["fixture_id"].astype(int)

    merged = hist_df.merge(sp_df, on=["fixture_id", "player_id"], how="left")
    for col in SP_COLS:
        if col not in merged.columns:
            merged[col] = 0
        else:
            merged[col] = merged[col].fillna(0).astype(int)

    n_sp     = int((merged["sp_goal"]     > 0).sum())
    n_header = int((merged["headed_goal"] > 0).sum())
    n_takers = int((merged["sp_assist"]   > 0).sum())
    print(f"  [sp_events] {n_sp} SP goals ({n_header} headers) | {n_takers} SP assist rows matched")

    return merged.to_dict("records")
