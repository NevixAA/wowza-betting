"""
API-Football Client — Main Model (Standings, Statistics, Injuries, Odds, H2H, Live)
=====================================================================================
Uses direct API-Football.com (api-sports.io) Pro plan: 7,500 calls/day.
Same APIFOOTBALL_KEY environment variable as player_model/api_football.py.

Endpoints:
  /standings            — table position + form (24h cache, 11 calls/day)
  /fixtures/statistics  — shots, corners, fouls (permanent cache for FT)
  /injuries             — injured/suspended players (12h cache)
  /odds                 — bookmaker over/under odds (2h cache)
  /fixtures?h2h=...     — head-to-head history (30-day cache)
  /predictions          — API meta-model (6h cache)
  /fixtures/events      — goal events for live recency gate (2min cache)
  /fixtures?live=all    — live fixture status (90s cache)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

_APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "").strip()
HEADERS  = {"x-apisports-key": _APIFOOTBALL_KEY}
BASE_URL = "https://v3.football.api-sports.io"

# Use WowzaApp junction to avoid Hebrew path issues
_BASE     = Path("C:/WowzaApp") if Path("C:/WowzaApp").exists() else Path(__file__).resolve().parents[1]
CACHE_DIR = _BASE / "api_football_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    import hashlib
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def _load_cache(key: str, max_age_h: float) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts   = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
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


def _get(endpoint: str, params: dict, cache_hours: float = 24) -> Optional[dict]:
    if not _APIFOOTBALL_KEY:
        return None
    cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
    if cache_hours > 0:
        cached = _load_cache(cache_key, max_age_h=cache_hours)
        if cached:
            return cached

    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            # WARNING, not debug. Every caller turns a None from here into an empty list,
            # which downstream reads as "there is nothing" rather than "the API refused".
            # On 2026-08-16 the daily quota was exhausted by 06:00 and the whole
            # API-Football surface went dark — live scanner, props gate, prop odds,
            # collect, settlement fallback — and NOT ONE log line said so, because this
            # was log.debug. 429 is called out explicitly since quota is the usual cause.
            _why = "QUOTA/RATE LIMIT" if r.status_code == 429 else f"HTTP {r.status_code}"
            log.warning(f"[api_football] {_why} on {endpoint} — returning no data "
                        f"(caller will see an empty result). body={r.text[:160]}")
            return None
        data = r.json()
        # API-Football answers 200 with an `errors` payload for plan/quota/parameter
        # problems, so a 200 is NOT proof of success.
        errs = data.get("errors") if isinstance(data, dict) else None
        if errs:
            # A PARAMETER error is a permanent bug in our request — it will fail identically on
            # every future call and no amount of waiting fixes it. A quota/plan error is
            # transient. The old message called every error "an exhausted daily quota", which is
            # how a malformed `league` param masqueraded as a quiet day and killed the live
            # scanner for 14 days. Same return value (no data), very different severity.
            param_keys = {"league", "season", "team", "fixture", "date", "ids", "live",
                          "player", "bookmaker", "bet", "page", "timezone"}
            bad_params = sorted(param_keys & set(errs)) if isinstance(errs, dict) else []
            if bad_params:
                log.error(f"[api_football] {endpoint} REJECTED OUR PARAMETERS: {errs} — this is "
                          f"a permanent client bug in {bad_params}, not a quota problem. It "
                          f"will return no data on every call until the request is fixed.")
            else:
                log.warning(f"[api_football] {endpoint} returned errors={errs} — treating as "
                            f"no data (plan/quota shape: transient, retry later)")
            return None
        if cache_hours > 0:
            _save_cache(cache_key, data)
        time.sleep(0.3)
        return data
    except Exception as e:
        log.warning(f"[api_football] {endpoint} failed: {type(e).__name__}: {e}")
        return None


# ── Standings ─────────────────────────────────────────────────────────────────

def get_standings(league_id: int, season: str) -> list[dict]:
    """
    Returns list of standing dicts for a league/season. 24h cache.
    Each dict: team_id, team_name, rank, points, played, gd, gd_per_game,
               goals_for_pg, goals_against_pg, home_win_rate, away_win_rate,
               form_pts, form_str
    """
    data = _get("/standings", {"league": league_id, "season": season}, cache_hours=24)
    if not data:
        return []

    standings = []
    for group in data.get("response", []):
        league_obj = group.get("league", {})
        for group_standings in league_obj.get("standings", []):
            for entry in group_standings:
                team      = entry.get("team", {})
                all_s     = entry.get("all",  {})
                home_s    = entry.get("home", {})
                away_s    = entry.get("away", {})

                played  = all_s.get("played", 0) or 0
                gf      = (all_s.get("goals") or {}).get("for",     0) or 0
                ga      = (all_s.get("goals") or {}).get("against", 0) or 0

                h_played = home_s.get("played", 0) or 0
                h_win    = home_s.get("win",    0) or 0
                a_played = away_s.get("played", 0) or 0
                a_win    = away_s.get("win",    0) or 0

                form_str  = (entry.get("form") or "")[-5:]
                form_pts  = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form_str)

                standings.append({
                    "team_id":          team.get("id"),
                    "team_name":        team.get("name", ""),
                    "rank":             entry.get("rank", 99),
                    "points":           entry.get("points", 0),
                    "played":           played,
                    "gd":               gf - ga,
                    "gd_per_game":      round((gf - ga) / max(played, 1), 3),
                    "goals_for_pg":     round(gf / max(played, 1), 3),
                    "goals_against_pg": round(ga / max(played, 1), 3),
                    "home_win_rate":    round(h_win / max(h_played, 1), 3),
                    "away_win_rate":    round(a_win / max(a_played, 1), 3),
                    "form_pts":         form_pts,
                    "form_str":         form_str,
                })
    return standings


def build_standings_map(league_id: int, season: str) -> dict[str, dict]:
    """
    Returns {lowercase_team_name: standing_dict} for fuzzy lookup.
    Returns empty dict if API key unavailable or request fails.
    """
    standings = get_standings(league_id, season)
    return {s["team_name"].lower().strip(): s for s in standings}


def lookup_team_standing(standings_map: dict[str, dict], team_name: str) -> Optional[dict]:
    """
    Fuzzy-match team name against standings map.
    Tries exact → token overlap → prefix match.
    Returns standing dict or None.
    """
    if not standings_map or not team_name:
        return None
    key = team_name.lower().strip()
    if key in standings_map:
        return standings_map[key]

    # Token overlap (handles "Burnley FC" → "Burnley")
    words = set(key.split())
    best_score, best_hit = 0, None
    for name, standing in standings_map.items():
        overlap = len(words & set(name.split()))
        if overlap > best_score:
            best_score, best_hit = overlap, standing
    if best_score >= 1 and best_hit is not None:
        return best_hit
    return None


# ── Fixture statistics ────────────────────────────────────────────────────────

def get_fixture_statistics(fixture_id: int, cache_hours: float = 8760) -> dict:
    """
    Returns {home: {shots, shots_on_target, corners, fouls, possession, offsides},
              away: {...}}
    Use cache_hours=0.025 (90s) for live matches; default 8760h (permanent) for finished.
    """
    data = _get("/fixtures/statistics", {"fixture": fixture_id}, cache_hours=cache_hours)
    if not data:
        return {}

    result = {}
    for i, team_data in enumerate(data.get("response", [])):
        side = "home" if i == 0 else "away"
        team = team_data.get("team", {})

        parsed: dict = {"team_id": team.get("id"), "team_name": team.get("name", "")}
        for stat in team_data.get("statistics", []):
            t = stat.get("type", "")
            v = stat.get("value")
            if t == "Shots on Goal":
                parsed["shots_on_target"] = int(v or 0)
            elif t == "Shots off Goal":
                parsed["shots_off"] = int(v or 0)
            elif t == "Total Shots":
                parsed["shots"] = int(v or 0)
            elif t == "Corner Kicks":
                parsed["corners"] = int(v or 0)
            elif t == "Fouls":
                parsed["fouls"] = int(v or 0)
            elif t == "Ball Possession":
                try:
                    parsed["possession"] = float(str(v or "50%").replace("%", ""))
                except ValueError:
                    parsed["possession"] = 50.0
            elif t == "Offsides":
                parsed["offsides"] = int(v or 0)
        result[side] = parsed
    return result


# ── Injuries ──────────────────────────────────────────────────────────────────

def get_injuries(league_id: int, season: str, fixture_id: int) -> list[dict]:
    """
    Returns injured/suspended players for a specific fixture.
    12h cache (injuries can change on matchday).
    """
    data = _get("/injuries", {
        "league": league_id, "season": season, "fixture": fixture_id,
    }, cache_hours=12)
    if not data:
        return []

    injuries = []
    for entry in data.get("response", []):
        player = entry.get("player", {})
        team   = entry.get("team",   {})
        injuries.append({
            "team_id":     team.get("id"),
            "team_name":   team.get("name", ""),
            "player_id":   player.get("id"),
            "player_name": player.get("name", ""),
            "position":    player.get("type", ""),
            "reason":      entry.get("reason", ""),
        })
    return injuries


def count_attacking_injuries(injuries: list[dict], team_name: str) -> int:
    """Count attackers/midfielders missing for a team (fuzzy name match)."""
    key = team_name.lower()[:6]
    team_inj = [i for i in injuries if key in i["team_name"].lower()[:8]]
    attacking = {"f", "m", "attacker", "midfielder", "forward"}
    return sum(
        1 for i in team_inj
        if any(pos in i.get("position", "").lower() for pos in attacking)
    )


# ── Odds ──────────────────────────────────────────────────────────────────────

def get_odds(fixture_id: int, bookmaker_id: int = 6) -> dict:
    """
    Returns over/under 2.5 odds from a bookmaker.
    Bookmaker 6 = Bet365, 8 = Pinnacle.
    2h cache (odds move frequently).
    """
    data = _get("/odds", {"fixture": fixture_id, "bookmaker": bookmaker_id}, cache_hours=2)
    if not data:
        return {}

    over_odd = under_odd = 0.0
    for entry in data.get("response", []):
        for bm in entry.get("bookmakers", []):
            for bet in bm.get("bets", []):
                if bet.get("name") != "Goals Over/Under":
                    continue
                for val in bet.get("values", []):
                    label = str(val.get("value", ""))
                    odd   = float(val.get("odd") or 0)
                    if "Over 2.5" in label:
                        over_odd = odd
                    elif "Under 2.5" in label:
                        under_odd = odd
    if over_odd and under_odd:
        return {"odds_over25": over_odd, "odds_under25": under_odd,
                "bookmaker_id": bookmaker_id}
    return {}


# ── H2H ───────────────────────────────────────────────────────────────────────

def get_h2h(team_a_id: int, team_b_id: int, last_n: int = 10) -> list[dict]:
    """
    Returns last N H2H completed fixtures between two teams.
    30-day cache (H2H history changes slowly).
    """
    h2h_key = f"{min(team_a_id, team_b_id)}-{max(team_a_id, team_b_id)}"
    data = _get("/fixtures", {"h2h": h2h_key, "last": min(last_n, 20)}, cache_hours=720)
    if not data:
        return []

    matches = []
    for fix in data.get("response", []):
        goals = fix.get("goals", {})
        home_g = goals.get("home")
        away_g = goals.get("away")
        if home_g is None or away_g is None:
            continue
        total = home_g + away_g
        matches.append({
            "fixture_id":   fix.get("fixture", {}).get("id"),
            "date":         fix.get("fixture", {}).get("date", ""),
            "home_team_id": fix.get("teams", {}).get("home", {}).get("id"),
            "away_team_id": fix.get("teams", {}).get("away", {}).get("id"),
            "home_goals":   home_g,
            "away_goals":   away_g,
            "total_goals":  total,
            "over25":       int(total > 2),
            "btts":         int(home_g > 0 and away_g > 0),
        })
    return matches


def compute_h2h_features(matches: list[dict]) -> dict:
    """Aggregate H2H fixtures into summary features."""
    if not matches:
        return {}
    totals  = [m["total_goals"] for m in matches]
    over25s = [m["over25"]      for m in matches]
    btts    = [m["btts"]        for m in matches]
    return {
        "h2h_avg_goals":   round(sum(totals)  / len(totals),  3),
        "h2h_over25_rate": round(sum(over25s) / len(over25s), 3),
        "h2h_btts_rate":   round(sum(btts)    / len(btts),    3),
        "h2h_n":           len(matches),
    }


# ── Predictions ───────────────────────────────────────────────────────────────

def get_predictions(fixture_id: int) -> dict:
    """
    Returns API-Football's own prediction for a fixture.
    6h cache. Use as secondary confirmation signal only.
    """
    data = _get("/predictions", {"fixture": fixture_id}, cache_hours=6)
    if not data:
        return {}
    resp = data.get("response", [])
    if not resp:
        return {}
    pred  = resp[0].get("predictions", {})
    goals = pred.get("goals", {})
    return {
        "api_winner":     (pred.get("winner") or {}).get("name", ""),
        "api_under_over": pred.get("under_over", ""),
        "api_goals_home": goals.get("home"),
        "api_goals_away": goals.get("away"),
        "api_advice":     pred.get("advice", ""),
    }


# ── Fixture events ────────────────────────────────────────────────────────────

def get_fixture_events(fixture_id: int) -> list[dict]:
    """
    Returns list of events for a live fixture (goals, cards, subs).
    2-minute cache for live accuracy.
    """
    data = _get("/fixtures/events", {"fixture": fixture_id}, cache_hours=0.033)
    if not data:
        return []
    events = []
    for ev in data.get("response", []):
        events.append({
            "time":      (ev.get("time") or {}).get("elapsed", 0) or 0,
            "type":      ev.get("type",   ""),
            "detail":    ev.get("detail", ""),
            "team_id":   (ev.get("team") or {}).get("id"),
            "team_name": (ev.get("team") or {}).get("name", ""),
        })
    return events


def last_goal_elapsed(events: list[dict]) -> Optional[int]:
    """Returns elapsed minute of most recent goal, or None if no goals."""
    goal_events = [e for e in events if e.get("type") == "Goal"]
    if not goal_events:
        return None
    return max(e["time"] for e in goal_events)


# ── Live fixtures ─────────────────────────────────────────────────────────────

def get_live_fixtures(league_ids: list[int] = None) -> list[dict]:
    """
    Returns all currently live fixtures, filtered to given league IDs.
    90-second cache to avoid rate pressure.
    Returns list of dicts: fixture_id, league_id, league_name, home_team,
    away_team, home_goals, away_goals, status, elapsed_mins.
    """
    # LEAGUE FILTERING IS CLIENT-SIDE, DELIBERATELY (fixed 2026-08-23).
    #
    # This used to send `live=all&league=39-140-135-...`. API-Football rejects that with
    # HTTP 200 and errors={'league': 'The League field must contain an integer.'} — the
    # dash-separated list is valid for `ids`/`live`, but `league` takes ONE integer. `_get`
    # treats any `errors` payload as no-data, so the call returned [] on EVERY scan.
    #
    # It killed the live scanner silently for 14 days: last real output 2026-08-09 00:40 UTC,
    # live_tips.csv and live_games.csv left header-only, inplay_snapshots.csv frozen at 129
    # rows. The workflow went green every 5 minutes throughout, because "no live games" is a
    # completely normal answer and nothing distinguished it from "the request was malformed".
    # Measured the same day: this call returned 0 fixtures while the unscoped call returned 9.
    #
    # Filtering here rather than server-side because:
    #   * it costs exactly the same — one /fixtures call either way, `live=all` is not billed
    #     per league;
    #   * live_scanner ALREADY re-filters by `id_to_name.get(league_id)`, so the server-side
    #     filter was duplicating a client-side one that is authoritative anyway;
    #   * it makes a zero self-diagnosing. `live=<dash list>` (the documented alternative) is
    #     accepted but also returns 0 when our leagues are quiet, which is indistinguishable
    #     from the malformed case — and being unable to tell those apart is what hid this for
    #     two weeks. Now the log prints both counts, so "9 live worldwide, 0 in our leagues"
    #     reads very differently from "0 live worldwide".
    params: dict = {"live": "all"}

    data = _get("/fixtures", params, cache_hours=0.025)  # ~90s
    if not data:
        return []

    wanted = {int(l) for l in league_ids} if league_ids else None
    fixtures = []
    n_raw = 0
    for fix in data.get("response", []):
        n_raw += 1
        if wanted is not None and (fix.get("league") or {}).get("id") not in wanted:
            continue
        status_obj = fix.get("fixture", {}).get("status", {})
        short      = status_obj.get("short", "")
        elapsed    = status_obj.get("elapsed") or 0

        if short not in ("1H", "HT", "2H", "ET", "P"):
            continue

        league = fix.get("league", {})
        teams  = fix.get("teams",  {})
        goals  = fix.get("goals",  {})

        fixtures.append({
            "fixture_id":   fix.get("fixture", {}).get("id"),
            "league_id":    league.get("id"),
            "league_name":  league.get("name", ""),
            "home_team_id": (teams.get("home") or {}).get("id"),
            "home_team":    (teams.get("home") or {}).get("name", ""),
            "away_team_id": (teams.get("away") or {}).get("id"),
            "away_team":    (teams.get("away") or {}).get("name", ""),
            "home_goals":   goals.get("home") or 0,
            "away_goals":   goals.get("away") or 0,
            "status":       short,
            "elapsed_mins": elapsed,
        })
    # Both counts, always. A bare "0 live games" is the exact message that made a broken
    # request look like a quiet Sunday for two weeks.
    log.info(f"[api_football] live: {n_raw} in-play worldwide, {len(fixtures)} in our "
             f"{len(wanted) if wanted else 'all'} league(s)")
    return fixtures
