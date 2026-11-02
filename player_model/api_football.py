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

import atexit
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from . import config
from .text_fix import repair_text

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


# ── per-run call accounting (added 2026-08-19) ────────────────────────────────
# WHY: API-Football usage went 11.5k -> 42.6k/day of a 75,000 cap over 08-16..08-18 and NOTHING
# could say which workflow was responsible. output/api_usage_log.csv records only the account-wide
# cumulative counter with a single label ("poll"), so it shows the total moving and nothing else.
# Attributing it by regressing usage deltas on which workflows were running gave R^2=0.125 and
# assigned 217 calls/run to the usage monitor, which makes exactly ONE call — i.e. collinearity
# noise, not an answer.
#
# Every API-Football response already carries x-ratelimit-requests-remaining, so exact accounting
# is FREE — no extra request. Counted at this one choke point, which every endpoint helper below
# routes through.
#
# Deliberately PRINT-ONLY rather than writing a CSV: a committed file would need per-workflow
# staging plumbing, would grow the repo, and would add another concurrent writer to race on. The
# workflow log is already durable and readable via the Actions API, which is enough to attribute.
_STATS = {"calls": 0, "cache_hits": 0, "first_remaining": None, "last_remaining": None,
          "errors": 0}


def _report_usage() -> None:
    """One-line per-run summary, printed at process exit. Must never raise."""
    try:
        s = _STATS
        if not s["calls"] and not s["cache_hits"]:
            return
        label = _os.getenv("GITHUB_WORKFLOW") or "local"
        spent = ""
        if s["first_remaining"] is not None and s["last_remaining"] is not None:
            # Quota actually consumed account-wide during this run. Can exceed our own `calls`
            # when another workflow overlaps, so both numbers are reported rather than one.
            spent = (f"  quota_remaining {s['first_remaining']}->{s['last_remaining']} "
                     f"(account-wide delta {s['first_remaining'] - s['last_remaining']})")
        total = s["calls"] + s["cache_hits"]
        hit = 100.0 * s["cache_hits"] / total if total else 0.0
        print(f"[api_usage] workflow={label} af_calls={s['calls']} "
              f"cache_hits={s['cache_hits']} cache_hit_rate={hit:.0f}% "
              f"errors={s['errors']}{spent}")
    except Exception:
        pass


atexit.register(_report_usage)


def _get(endpoint: str, params: dict, cache_hours: int = 24) -> Optional[dict]:
    """Make one API-Football call with caching."""
    if not _APIFOOTBALL_KEY:
        return None  # no key — fallback to FBref only
    cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
    cached = _load_cache(cache_key, max_age_h=cache_hours)
    if cached:
        _STATS["cache_hits"] += 1
        return cached

    url = f"{BASE_URL}{endpoint}"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        except Exception as e:
            _STATS["errors"] += 1
            print(f"[api_football] Error {endpoint}: {e}")
            return None
        # Count every response that reached the API, including 429/non-200: they consume quota
        # too, so counting only successes would under-report exactly when we are in trouble.
        _STATS["calls"] += 1
        try:
            rem = r.headers.get("x-ratelimit-requests-remaining")
            if rem is not None:
                rem = int(rem)
                if _STATS["first_remaining"] is None:
                    _STATS["first_remaining"] = rem
                _STATS["last_remaining"] = rem
        except (TypeError, ValueError):
            pass
        if r.status_code == 429:            # rate-limited → back off and retry (self-heal)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        data = r.json()
        _save_cache(cache_key, data)
        # ~400/min (Ultra plan allows 450). Was 0.5s=120/min — the same 0.5→0.15 speed-up
        # already applied to data_fetcher.py in commit 16fb189; this path had been left behind.
        time.sleep(0.15)
        return data
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


def get_fixtures_between(league_id: int, season: str, date_from: str, date_to: str) -> list[dict]:
    """EVERY fixture of a league between two dates (YYYY-MM-DD, inclusive).

    Use this, not get_upcoming_fixtures, for a slate: `next=5` returned only the first five
    kickoffs, so a 10-match Premier League round or an 18-match Champions League matchday was
    cut to 5 and the rest never got props (2026-11-01: 13 of 26 leagues sat at exactly 5)."""
    data = _get("/fixtures", {
        "league": league_id, "season": season, "from": date_from, "to": date_to
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
    return [{"player_id": p.get("id"), "player_name": repair_text(p.get("name")),
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


def _fixture_teams(fix: dict) -> tuple[str | None, str | None]:
    """(home name, away name) of one /fixtures record; None for a side the provider left out."""
    t = fix.get("teams") or {}
    return (t.get("home") or {}).get("name") or None, (t.get("away") or {}).get("name") or None


def match_fixture(fixtures: list[dict], home: str, away: str) -> dict | None:
    """The ONE fixture in `fixtures` (a single league's /fixtures response for one date) whose
    clubs are `home` and `away`, or None.

    Both clubs are resolved by src.team_names.resolve against the POOL of that league-date's
    fixture clubs (home names for `home`, away names for `away`): ambiguity is refused by the
    resolver, and the resolved pair must then be one fixture. Data Truth (2026-11-02): this
    replaces a pairwise test of every fixture (`_team_match(home, fh) and _team_match(away, fa)`),
    which judged each candidate alone and so could not see that a name fitted two clubs."""
    named = [f for f in (fixtures or []) if all(_fixture_teams(f))]
    if not named or not home or not away:
        return None
    resolve = config._v9_src("team_names").resolve
    rh = resolve(str(home), sorted({_fixture_teams(f)[0] for f in named}))
    ra = resolve(str(away), sorted({_fixture_teams(f)[1] for f in named}))
    if not rh or not ra:
        return None
    hits = [f for f in named if _fixture_teams(f) == (rh, ra)]
    return hits[0] if len(hits) == 1 else None


# ── Public raw accessors (Data Truth, 2026-11-02) ─────────────────────────────────────────────
# Raw provider JSON (the `response` list), every null exactly as API-Football sent it. Settlement
# (update_results.py) needs the raw records because the flattened helpers below are shaped for
# feature building. These are the supported way in; nothing outside this module calls _get.
# An empty list means "no data" (no key, HTTP error, or the provider returned nothing).

def _response(endpoint: str, params: dict, cache_hours: int) -> list:
    data = _get(endpoint, params, cache_hours=cache_hours)
    return list((data or {}).get("response") or [])


def fixture_by_id_raw(fixture_id: int, cache_hours: int = 2) -> dict | None:
    """GET /fixtures?id=<fixture_id>: the one fixture record, or None (none, or more than one)."""
    hits = _response("/fixtures", {"id": int(fixture_id)}, cache_hours)
    return hits[0] if len(hits) == 1 else None


def fixtures_on_date_raw(league_id: int, season: str, date_str: str, cache_hours: int = 2,
                         status: str | None = None) -> list[dict]:
    """GET /fixtures?league&season&date[&status]: every fixture of one league on one date."""
    params = {"league": league_id, "season": season, "date": date_str}
    if status:
        params["status"] = status
    return _response("/fixtures", params, cache_hours)


def fixture_players_raw(fixture_id: int, cache_hours: int = 168) -> list[dict]:
    """GET /fixtures/players?fixture=<id>: per-team blocks of per-player statistics."""
    return _response("/fixtures/players", {"fixture": int(fixture_id)}, cache_hours)


def fixture_events_raw(fixture_id: int, cache_hours: int = 168) -> list[dict]:
    """GET /fixtures/events?fixture=<id>: goals, cards, substitutions, VAR events."""
    return _response("/fixtures/events", {"fixture": int(fixture_id)}, cache_hours)


def fixture_statistics_raw(fixture_id: int, cache_hours: int = 168) -> list[dict]:
    """GET /fixtures/statistics?fixture=<id>: per-team match statistics."""
    return _response("/fixtures/statistics", {"fixture": int(fixture_id)}, cache_hours)


def fixture_odds_raw(fixture_id: int, cache_hours: int = 1) -> list[dict]:
    """GET /odds?fixture=<id>: bookmaker blocks for one fixture (player props among them)."""
    return _response("/odds", {"fixture": int(fixture_id)}, cache_hours)


def find_fixture_id(league_id: int, season: str, date_str: str, home: str, away: str) -> int | None:
    """Find fixture ID for a completed (FT) match by league, season, date, and team names
    (clubs resolved within that league-date's fixtures, see match_fixture)."""
    fix = match_fixture(fixtures_on_date_raw(league_id, season, date_str, cache_hours=168,
                                             status="FT"), home, away)   # FT cached 7 days
    return (fix.get("fixture") or {}).get("id") if fix else None


def get_fixture_goals(league_id: int, season: str, date_str: str, home: str, away: str):
    """Final + half-time goals for a completed match, by league/season/date/team names.
    Returns (ft_home, ft_away, ht_home, ht_away), or None if the fixture isn't found / not FT.
    ht_home/ht_away are None when the API carries no half-time score for that fixture. Used to
    grade the live-signal history; reuses the same cached /fixtures call as find_fixture_id."""
    fix = match_fixture(fixtures_on_date_raw(league_id, season, date_str, cache_hours=168,
                                             status="FT"), home, away)
    if fix is None:
        return None
    g  = fix.get("goals") or {}
    ht = (fix.get("score") or {}).get("halftime") or {}
    fth, fta = g.get("home"), g.get("away")
    if fth is None or fta is None:
        return None
    hth, hta = ht.get("home"), ht.get("away")
    return (int(fth), int(fta),
            int(hth) if hth is not None else None,
            int(hta) if hta is not None else None)


def get_injured_players(league_id: int, season: str, date_str: str) -> set[int]:
    """
    Return the API-Football player IDS of players injured/suspended for a league on a given date.
    Uses /injuries endpoint cached 4 hours (injury status is stable within a day).
    Returns empty set if no key or endpoint unavailable.

    IDS, NOT NAMES (Data Truth, 2026-11-01): predict matched these on normalised names, so two
    players sharing a short name ("J. Silva") blocked each other. An entry without an id is
    skipped and counted in the log line.
    """
    data = _get("/injuries", {
        "league": league_id, "season": season, "date": date_str,
    }, cache_hours=4)
    if not data:
        return set()
    injured: set[int] = set()
    no_id = 0
    for entry in data.get("response", []):
        pid = (entry.get("player") or {}).get("id")
        try:
            injured.add(int(pid))
        except (TypeError, ValueError):
            no_id += 1
    if no_id:
        print(f"[injuries] league {league_id}: {no_id} entr(ies) without a player id skipped")
    return injured


def get_fixture_status(league_id: int, season: str, date_str: str, home: str, away: str) -> tuple[str | None, int | None]:
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
    fix = match_fixture(fixtures_on_date_raw(league_id, season, date_str, cache_hours=2), home, away)
    if fix is None:
        return "NOT_FOUND", None
    f = fix.get("fixture") or {}
    # A found fixture whose status the provider left out has status None (unknown), never "".
    return (f.get("status") or {}).get("short"), f.get("id")


# ── Player match stats ────────────────────────────────────────────────────────

def _int_or_none(v) -> int | None:
    """An integer the provider sent, or None for a null / unparseable value (never 0)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _flag_or_none(v) -> int | None:
    """1 if the count is > 0, 0 if it is 0, None if the provider sent no count."""
    n = _int_or_none(v)
    return None if n is None else int(n > 0)


def get_fixture_player_stats(fixture_id: int) -> list[dict]:
    """
    Get per-player stats for one completed fixture.
    Returns flat list of player stat dicts.

    PROVIDER NULLS STAY None (Data Truth, 2026-11-02). API-Football sends null for minutes and
    for counting stats it did not record (commonly goals.total / shots.on for a player with none,
    and everything for an unused substitute). This used to write 0 for every null, so "not
    recorded" and "recorded zero" were indistinguishable to every consumer. `started` is None when
    the provider sent neither `substitute` nor minutes. Settlement reads the raw records
    (fixture_players_raw) and decides itself what a null means.
    """
    players = []
    for team_data in fixture_players_raw(fixture_id, cache_hours=168):   # 7 days
        team_name = (team_data.get("team") or {}).get("name")
        for player_entry in team_data.get("players") or []:
            p   = player_entry.get("player") or {}
            s   = (player_entry.get("statistics") or [{}])[0] or {}
            games = s.get("games") or {}
            shots = s.get("shots") or {}
            goals = s.get("goals") or {}
            cards = s.get("cards") or {}
            passes = s.get("passes") or {}
            duels  = s.get("duels") or {}
            fouls  = s.get("fouls") or {}
            minutes = _int_or_none(games.get("minutes"))
            sub = games.get("substitute")
            if sub is not None:
                started = not bool(sub)
            elif minutes is not None:
                started = minutes > 45
            else:
                started = None

            players.append({
                "fixture_id":      fixture_id,
                "player_id":       p.get("id"),
                "player_name":     repair_text(p.get("name")) if p.get("name") else None,
                "team":            team_name,
                "position":        games.get("position"),
                "minutes_played":  minutes,
                "started":         started,
                "goals":           _int_or_none(goals.get("total")),
                "assists":         _int_or_none(goals.get("assists")),
                "shots_total":     _int_or_none(shots.get("total")),
                "shots_on_target": _int_or_none(shots.get("on")),
                "yellow_card":     _flag_or_none(cards.get("yellow")),
                "red_card":        _flag_or_none(cards.get("red")),
                "key_passes":      _int_or_none(passes.get("key")),
                "fouls_committed": _int_or_none(fouls.get("committed")),
                "fouls_drawn":     _int_or_none(fouls.get("drawn")),
                "duels_total":     _int_or_none(duels.get("total")),
                "duels_won":       _int_or_none(duels.get("won")),
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
                "player_name": repair_text(p.get("player", {}).get("name", "")),
                "position":    p.get("pos", ""),
                "number":      p.get("number"),
                "started":     True,
            }
            for p in team_data.get("startXI", [])
        ]
        subs = [
            {
                "player_id":   p.get("player", {}).get("id"),
                "player_name": repair_text(p.get("player", {}).get("name", "")),
                "position":    p.get("pos", ""),
                "number":      p.get("number"),
                "started":     False,
            }
            for p in team_data.get("substitutes", [])
        ]
        result[team_name] = starters + subs
    return result


# ── Referee ───────────────────────────────────────────────────────────────────

def get_fixture_referee(fixture_id: int) -> str | None:
    """Referee name for a fixture, or None when the provider has none (was "")."""
    fix = fixture_by_id_raw(fixture_id, cache_hours=24)
    return ((fix or {}).get("fixture") or {}).get("referee") or None


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
            mp = s.get("minutes_played")
            if s.get("player_id") == player_id and mp is not None and mp > 0:
                s["date"] = fix.get("fixture", {}).get("date", "")
                player_matches.append(s)
                break

    # Sort by date descending, return last N
    player_matches.sort(key=lambda x: x.get("date", ""), reverse=True)
    return player_matches[:last_n]


# ── Referee profiles ──────────────────────────────────────────────────────────

def _referee_key(name) -> str:
    """Normalised referee name for EXACT comparison. API-Football carries "M. Oliver" or
    "Michael Oliver, England"; the country suffix is dropped, nothing else is guessed."""
    return " ".join(_norm_name(str(name or "").split(",")[0]).replace(".", ". ").split())


def build_referee_profile(referee_name: str, league_id: int, season: str,
                          league_avg: float | None = None,
                          league_std: float | None = None) -> dict:
    """
    Compute referee profile from recent fixtures.
    Returns: {yellows_per_game, strictness_score, n_games}

    Data Truth (2026-11-01): no typed fallback. Unknown referee or fewer than
    config.MIN_REF_MATCHES matches -> yellows_per_game and strictness_score are NaN. The z-score
    uses the league's MEASURED yellows/match mean and sd (pass them in, e.g. from
    feature_engineering.league_yellow_stats over player_history); without them strictness is NaN.
    The name test is exact equality of the normalised name (was a substring test).
    NEEDS RESOURCE: API-Football gives the referee as a name string only (no referee id).
    """
    nan = float("nan")
    if not referee_name:
        return {"yellows_per_game": nan, "strictness_score": nan, "n_games": 0}

    want = _referee_key(referee_name)
    fixtures = get_recent_fixtures(league_id, season, last_n=30)
    ref_fixtures = [f for f in fixtures
                    if _referee_key(f.get("fixture", {}).get("referee")) == want]

    if len(ref_fixtures) < config.MIN_REF_MATCHES:
        return {"yellows_per_game": nan, "strictness_score": nan, "n_games": len(ref_fixtures)}

    total_yellows = 0
    n = 0
    n_unknown = 0
    for fix in ref_fixtures:
        fix_id = fix.get("fixture", {}).get("id")
        stats = get_fixture_player_stats(fix_id) if fix_id else []
        if not stats:
            continue                     # no player stats -> this match is not counted
        # A player who PLAYED with no card count makes the match total unknown: the match is
        # skipped (counted below), never summed as if he had no booking. An unused substitute
        # (no minutes) cannot be booked on the pitch, so his null is not evidence either way.
        played = [s for s in stats
                  if s.get("minutes_played") is not None and s["minutes_played"] > 0]
        if any(s.get("yellow_card") is None for s in played):
            n_unknown += 1
            continue
        total_yellows += sum(s["yellow_card"] for s in played)
        n += 1
    if n_unknown:
        print(f"[referee] {referee_name}: {n_unknown} match(es) skipped - a player's card count is null")
    if n < config.MIN_REF_MATCHES:
        return {"yellows_per_game": nan, "strictness_score": nan, "n_games": n}

    avg_yellows = total_yellows / n
    ok = (league_avg is not None and league_std is not None
          and league_avg == league_avg and league_std == league_std and league_std > 0)
    strictness = (avg_yellows - league_avg) / league_std if ok else nan

    return {
        "yellows_per_game": round(avg_yellows, 2),
        "strictness_score": round(strictness, 2) if ok else nan,
        "n_games": n,
    }
