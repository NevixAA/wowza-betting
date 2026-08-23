"""
API-Football integration for O/U model features.

Phase 1 — Shot statistics for new-format leagues
  football-data.co.uk new-format CSVs contain goals only (no shots/SOT).
  This module backfills shot data from API-Football so combined_sot_ratio
  and related features are real rolling averages instead of the 0.35 constant.

Phase 2 — Injury/suspension features for upcoming matches
  key_attacker_absent_home / key_attacker_absent_away: 1 if any of the top-3
  scorers for that team is listed as injured/suspended for this fixture.

Uses RapidAPI gateway (API_KEY env var, same as config.py).
Cache: {project_root}/apifootball_ou_cache/ — TTL-based JSON files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

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



# ── API credentials ───────────────────────────────────────────────────────────
# Direct api-sports.io service — same key/endpoint as player_model/api_football.py
_KEY     = os.getenv("APIFOOTBALL_KEY", "")
_BASE    = "https://v3.football.api-sports.io"
_HEADERS = {"x-apisports-key": _KEY}

CACHE_DIR = config.BASE_DIR / "apifootball_ou_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(endpoint: str, params: dict) -> str:
    s = endpoint + json.dumps(sorted(params.items()))
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _load_cache(key: str, ttl_h: float) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(d.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at < timedelta(hours=ttl_h):
            return d
    except Exception:
        pass
    return None


def _save_cache(key: str, data: dict) -> None:
    data = dict(data)
    data["_cached_at"] = datetime.now().isoformat()
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


# ── API call ──────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, ttl_h: float = 24.0) -> Optional[dict]:
    """RapidAPI call with disk cache. Returns parsed JSON or None on error."""
    ck = _cache_key(endpoint, params)
    cached = _load_cache(ck, ttl_h)
    if cached is not None:
        return cached

    # Cache-only mode — set by the every-5-min predict job (env AF_OU_CACHE_ONLY=1).
    # A cold/incomplete cache must NOT block predict with thousands of slow shot fetches
    # (that overran the 15-min job timeout and silenced all tips since ~2026-06-30).
    # In this mode we never make a live call here; the daily full-enrichment run
    # (retrain / a dedicated warm-cache job, without this env) populates the cache.
    import os as _os
    if _os.getenv("AF_OU_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes"):
        return None

    if not _KEY:
        return None

    url = f"{_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=15)
        if r.status_code == 429:
            print(f"[api_football_ou] rate limit on {endpoint} — sleeping 60s")
            time.sleep(60)
            r = requests.get(url, headers=_HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[api_football_ou] {endpoint}: HTTP {r.status_code} {r.text[:120]}")
            return None
        data = r.json()
        # Do NOT cache error responses. API-Football returns HTTP-200 with a non-empty `errors`
        # dict + results:0 on quota-exhaustion / rate-limit / bad params. Caching those poisons
        # the 24h disk cache and silently breaks shot-enrichment for a full day even after the
        # quota resets (this was the NF "No completed fixtures" gap). Legit empties (no errors)
        # are still cached to avoid re-fetching. [v10 NF fix 2026-07-13]
        if not data.get("errors"):
            _save_cache(ck, data)
        time.sleep(0.4)   # courtesy delay — ~150 calls/min max
        return data
    except Exception as e:
        print(f"[api_football_ou] {endpoint}: {e}")
        return None


# ── Name normalisation ────────────────────────────────────────────────────────

_STRIP_SUFFIXES = re.compile(
    r"\b(FC|CF|SC|BK|IF|IFK|SK|FK|AC|AS|SS|SD|CD|UD|RJ|SP|MG|RS|PA|BA|CE|EC|"
    r"CA|CR|SE|AF|GD|BS|SL|RFC|AFC|FBC|JFC)\b",
    re.IGNORECASE,
)

# Known cross-source aliases: applied after suffix stripping and lowercasing.
# Keys and values are already normalised (lowercase, no accents).
_POST_NORM_ALIASES: dict[str, str] = {
    "vasco da gama": "vasco",
    "rb bragantino": "bragantino",
    "red bull bragantino": "bragantino",
    "atlanta united": "atlanta utd",
    "internazionale": "inter",
    "sporting cp": "sporting",
}


def _norm_name(name: str) -> str:
    """Lowercase, strip accents, remove common football suffixes."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = _STRIP_SUFFIXES.sub("", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    name = name.rstrip("-").strip()   # "Atletico-MG" → "atletico-" → "atletico"
    return _POST_NORM_ALIASES.get(name, name)


def _to_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _to_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Shot statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_completed_fixtures(league_id: int, season: str) -> list[dict]:
    """
    Return all completed fixtures for a league/season.
    Result: [{id, date, home, away}, ...]  — cached 24 h.
    """
    data = _get("fixtures", {"league": league_id, "season": season, "status": "FT"}, ttl_h=24)
    if not data:
        return []
    out = []
    for f in data.get("response", []):
        fix   = f.get("fixture", {})
        teams = f.get("teams", {})
        out.append({
            "id":   fix.get("id"),
            "date": str(fix.get("date", ""))[:10],
            "home": teams.get("home", {}).get("name", ""),
            "away": teams.get("away", {}).get("name", ""),
        })
    return out


def get_fixture_statistics(fixture_id: int) -> dict:
    """
    Return match stats for one completed fixture — shots, corners, fouls.
    Cached 1 year (completed match data never changes).
    API returns response[0]=home team, response[1]=away team.
    """
    data = _get("fixtures/statistics", {"fixture": fixture_id}, ttl_h=8760)
    if not data:
        return {}
    response = data.get("response", [])
    if len(response) < 2:
        return {}

    def _extract(team_block: dict) -> dict:
        return {s["type"]: s.get("value") for s in team_block.get("statistics", [])}

    home_s = _extract(response[0])
    away_s = _extract(response[1])
    return {
        "home_shots":      _to_int(home_s.get("Total Shots")),
        "away_shots":      _to_int(away_s.get("Total Shots")),
        "home_sot":        _to_int(home_s.get("Shots on Goal")),
        "away_sot":        _to_int(away_s.get("Shots on Goal")),
        "home_corners":    _to_int(home_s.get("Corner Kicks")),
        "away_corners":    _to_int(away_s.get("Corner Kicks")),
        "home_fouls":      _to_int(home_s.get("Fouls")),
        "away_fouls":      _to_int(away_s.get("Fouls")),
        "home_xg":         _to_float(home_s.get("expected_goals")),
        "away_xg":         _to_float(away_s.get("expected_goals")),
        "home_insidebox":  _to_int(home_s.get("Shots insidebox")),
        "away_insidebox":  _to_int(away_s.get("Shots insidebox")),
        "home_possession": _to_float(str(home_s.get("Ball Possession") or "0").replace("%", "") or None),
        "away_possession": _to_float(str(away_s.get("Ball Possession") or "0").replace("%", "") or None),
        "home_blocked":    _to_int(home_s.get("Blocked Shots")),
        "away_blocked":    _to_int(away_s.get("Blocked Shots")),
    }


def fetch_shots_for_league(league_id: int, season: str, league_name: str = "") -> pd.DataFrame:
    """
    Fetch match stats (shots, corners, fouls) for a league/season from cache.
    Returns DataFrame: date(datetime), home_team, away_team,
                       home_shots, away_shots, home_sot, away_sot,
                       home_corners, away_corners, home_fouls, away_fouls,
                       _home_norm, _away_norm  (for matching).
    """
    fixtures = get_completed_fixtures(league_id, season)
    if not fixtures:
        print(f"[api_football_ou] No completed fixtures: league={league_id} season={season}")
        return pd.DataFrame()

    label = league_name or str(league_id)
    print(f"[api_football_ou] Fetching shots for {label}: {len(fixtures)} fixtures")

    rows = []
    for i, fix in enumerate(fixtures):
        stats = get_fixture_statistics(fix["id"])
        if not stats or stats.get("home_shots") is None:
            continue
        rows.append({
            "date":          fix["date"],
            "home_team":     fix["home"],
            "away_team":     fix["away"],
            "home_shots":    stats["home_shots"],
            "away_shots":    stats["away_shots"],
            "home_sot":      stats["home_sot"],
            "away_sot":      stats["away_sot"],
            "home_corners":  stats.get("home_corners"),
            "away_corners":  stats.get("away_corners"),
            "home_fouls":    stats.get("home_fouls"),
            "away_fouls":    stats.get("away_fouls"),
            "home_xg":       stats.get("home_xg"),
            "away_xg":       stats.get("away_xg"),
            "home_insidebox": stats.get("home_insidebox"),
            "away_insidebox": stats.get("away_insidebox"),
            "home_possession": stats.get("home_possession"),
            "away_possession": stats.get("away_possession"),
            "home_blocked":    stats.get("home_blocked"),
            "away_blocked":    stats.get("away_blocked"),
        })
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(fixtures)}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners", "home_fouls", "away_fouls",
                "home_xg", "away_xg", "home_insidebox", "away_insidebox",
                "home_possession", "away_possession", "home_blocked", "away_blocked"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_home_norm"] = df["home_team"].apply(_norm_name)
    df["_away_norm"] = df["away_team"].apply(_norm_name)

    print(f"[api_football_ou] {len(df)} fixtures with shot data for {label}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1b — Fixture results for leagues not on football-data.co.uk
# ─────────────────────────────────────────────────────────────────────────────

def fetch_league_fixture_results(
    league_id: int, season: str, league_name: str = ""
) -> pd.DataFrame:
    """
    Fetch completed fixture results (goals + HT) for a league/season.
    Used for Saudi Pro League, K-League 1 and any other league not on FD.
    Permanent cache (TTL 1 year) — completed match results never change.

    Returns DataFrame: date, home_team, away_team,
                       home_goals, away_goals, ht_home_goals, ht_away_goals
    """
    data = _get("fixtures", {"league": league_id, "season": season, "status": "FT"}, ttl_h=8760)
    if not data:
        return pd.DataFrame()

    rows = []
    for f in data.get("response", []):
        fix   = f.get("fixture", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        score = f.get("score", {})
        ht    = score.get("halftime", {})

        home_g = goals.get("home")
        away_g = goals.get("away")
        if home_g is None or away_g is None:
            continue

        ht_home = ht.get("home")
        ht_away = ht.get("away")

        rows.append({
            "date":          str(fix.get("date", ""))[:10],
            "home_team":     teams.get("home", {}).get("name", ""),
            "away_team":     teams.get("away", {}).get("name", ""),
            "home_goals":    int(home_g),
            "away_goals":    int(away_g),
            "ht_home_goals": int(ht_home) if ht_home is not None else None,
            "ht_away_goals": int(ht_away) if ht_away is not None else None,
        })

    if not rows:
        return pd.DataFrame()

    label = league_name or str(league_id)
    print(f"[api_football_ou] {label} season {season}: {len(rows)} fixture results loaded")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna() & df["home_team"].notna() & df["away_team"].notna()]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Injury features
# ─────────────────────────────────────────────────────────────────────────────

def get_top_scorers(league_id: int, season: str) -> dict[str, list[str]]:
    """
    Top 3 goal scorers per team for the league/season.
    Returns {team_name_norm: [player_norm, ...]}  — cached 7 days.
    """
    data = _get("players/topscorers", {"league": league_id, "season": season}, ttl_h=168)
    if not data:
        return {}

    team_scorers: dict[str, list[str]] = {}
    for entry in data.get("response", []):
        player_norm = _norm_name(entry.get("player", {}).get("name", ""))
        stats_list  = entry.get("statistics", [])
        if not stats_list:
            continue
        team_norm = _norm_name(stats_list[0].get("team", {}).get("name", ""))
        if team_norm not in team_scorers:
            team_scorers[team_norm] = []
        if len(team_scorers[team_norm]) < 3:
            team_scorers[team_norm].append(player_norm)

    return team_scorers


def get_upcoming_fixture_ids(league_id: int, season: str, next_n: int = 20) -> dict[tuple, int]:
    """
    Upcoming fixture IDs keyed by (home_norm, away_norm, date_str).
    Cached 4 h.
    """
    data = _get("fixtures", {"league": league_id, "season": season, "next": next_n}, ttl_h=4)
    if not data:
        return {}

    result: dict[tuple, int] = {}
    for f in data.get("response", []):
        fix   = f.get("fixture", {})
        teams = f.get("teams", {})
        fix_id    = fix.get("id")
        date_str  = str(fix.get("date", ""))[:10]
        home_norm = _norm_name(teams.get("home", {}).get("name", ""))
        away_norm = _norm_name(teams.get("away", {}).get("name", ""))
        result[(home_norm, away_norm, date_str)] = fix_id

    return result


def get_injuries_for_fixture(fixture_id: int) -> dict[str, list[str]]:
    """
    Injured/suspended players for a specific fixture.
    Returns {team_norm: [player_norm, ...]}  — cached 2 h.
    """
    data = _get("injuries", {"fixture": fixture_id}, ttl_h=2)
    if not data:
        return {}

    injuries: dict[str, list[str]] = {}
    for entry in data.get("response", []):
        team_norm   = _norm_name(entry.get("team", {}).get("name", ""))
        player_norm = _norm_name(entry.get("player", {}).get("name", ""))
        injuries.setdefault(team_norm, []).append(player_norm)

    return injuries


def get_league_injury_context(league_id: int, season: str) -> dict:
    """
    Pre-fetch everything needed for injury features for all upcoming fixtures
    in one league.  Call once per league, reuse for every row.

    Returns:
        {
            "top_scorers":  {team_norm: [player_norm, ...]},
            "fixture_ids":  {(home_norm, away_norm, date_str): fixture_id},
            "injuries":     {fixture_id: {team_norm: [player_norm, ...]}},
        }
    """
    ctx: dict = {
        "top_scorers": get_top_scorers(league_id, season),
        "fixture_ids": get_upcoming_fixture_ids(league_id, season),
        "injuries":    {},
    }
    for fix_id in ctx["fixture_ids"].values():
        ctx["injuries"][fix_id] = get_injuries_for_fixture(fix_id)
    return ctx


def injury_features_from_context(
    home_team: str,
    away_team: str,
    match_date,
    ctx: dict,
) -> dict[str, float]:
    """
    Compute key_attacker_absent_home / _away from a pre-fetched league context.
    Returns {key_attacker_absent_home: 0/1, key_attacker_absent_away: 0/1}.
    """
    out = {"key_attacker_absent_home": 0.0, "key_attacker_absent_away": 0.0}
    if not ctx:
        return out

    home_norm = _norm_name(home_team)
    away_norm = _norm_name(away_team)
    date_str  = str(match_date)[:10]

    fix_id = ctx["fixture_ids"].get((home_norm, away_norm, date_str))
    if fix_id is None:
        return out

    injuries     = ctx["injuries"].get(fix_id, {})
    top_scorers  = ctx["top_scorers"]

    home_scorers = top_scorers.get(home_norm, [])
    away_scorers = top_scorers.get(away_norm, [])
    home_injured = injuries.get(home_norm, [])
    away_injured = injuries.get(away_norm, [])

    if home_scorers and any(p in home_injured for p in home_scorers):
        out["key_attacker_absent_home"] = 1.0
    if away_scorers and any(p in away_injured for p in away_scorers):
        out["key_attacker_absent_away"] = 1.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Team season statistics (/teams/statistics)
# Fills season-start gaps in rolling features and improves venue-split accuracy.
# Supplements _add_season_venue_stats() computed from FD historical data.
# ~40 calls/day (2 per upcoming match × ~20 matches/day). TTL: 24h.
# ─────────────────────────────────────────────────────────────────────────────

_TEAM_ID_CACHE_PATH = config.BASE_DIR / "team_id_cache.json"


def _load_team_id_cache() -> dict:
    try:
        return json.loads(_TEAM_ID_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_team_id_cache(cache: dict) -> None:
    try:
        _TEAM_ID_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _resolve_team_id(
    team_name: str, league_id: int, season: str
) -> Optional[int]:
    """
    Resolve API-Football team ID by name within a league/season.
    Checks persistent local cache first (zero API cost on cache hit).
    Falls back to GET /teams?league=&season= (1 call per league-season, not per team).
    Results cached permanently — team IDs don't change.
    """
    norm      = _norm_name(team_name)
    league_key = str(league_id)
    cache      = _load_team_id_cache()

    if league_key in cache and norm in cache[league_key]:
        return cache[league_key].get(norm)

    # Fetch all teams in this league/season at once (1 API call, populates many teams)
    data = _get("teams", {"league": league_id, "season": season}, ttl_h=8760)
    if not data:
        return None

    if league_key not in cache:
        cache[league_key] = {}

    team_id = None
    for entry in data.get("response", []):
        t    = entry.get("team", {})
        t_id = t.get("id")
        t_nm = _norm_name(t.get("name", ""))
        if t_id and t_nm:
            cache[league_key][t_nm] = t_id
            if t_nm == norm:
                team_id = t_id

    _save_team_id_cache(cache)
    return team_id


def fetch_team_season_stats(
    team_name: str,
    league_name: str,
    season: str | None = None,
) -> dict:
    """
    Fetch season-to-date team statistics from /teams/statistics.
    Returns a dict with keys matching the O/U model's season-venue feature names:
        goals_for_h, goals_for_a, goals_against_h, goals_against_a,
        cs_rate_h, cs_rate_a, wins_rate_h, wins_rate_a, form_win_rate,
        played_h, played_a
    Returns {} on any failure (API unavailable, unknown league, etc.).
    TTL: 24h — refreshes once per day as the season progresses.
    """
    league_id = config.API_FOOTBALL_IDS.get(league_name)
    if not league_id or not _KEY:
        return {}

    if season is None:
        season = config.API_SEASON

    team_id = _resolve_team_id(team_name, league_id, season)
    if not team_id:
        return {}

    data = _get(
        "teams/statistics",
        {"team": team_id, "league": league_id, "season": season},
        ttl_h=6,
    )
    if not data:
        return {}

    resp = data.get("response", {})
    if not resp:
        return {}

    fixtures = resp.get("fixtures",     {})
    goals    = resp.get("goals",        {})
    cs       = resp.get("clean_sheet",  {})
    form_str = resp.get("form", "")     or ""

    played_h = int(fixtures.get("played", {}).get("home", 0) or 0) or 1
    played_a = int(fixtures.get("played", {}).get("away", 0) or 0) or 1
    wins_h   = int(fixtures.get("wins",   {}).get("home", 0) or 0)
    wins_a   = int(fixtures.get("wins",   {}).get("away", 0) or 0)
    cs_h     = int(cs.get("home", 0) or 0)
    cs_a     = int(cs.get("away", 0) or 0)

    def _gavg(block: dict, venue: str) -> Optional[float]:
        try:
            v = block.get("average", {}).get(venue)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    goals_for_h = _gavg(goals.get("for",     {}), "home")
    goals_for_a = _gavg(goals.get("for",     {}), "away")
    goals_agn_h = _gavg(goals.get("against", {}), "home")
    goals_agn_a = _gavg(goals.get("against", {}), "away")

    recent  = form_str[-5:] if form_str else ""
    form_wr = recent.count("W") / max(len(recent), 1)

    return {
        "goals_for_h":     goals_for_h,
        "goals_for_a":     goals_for_a,
        "goals_against_h": goals_agn_h,
        "goals_against_a": goals_agn_a,
        "cs_rate_h":       cs_h / played_h,
        "cs_rate_a":       cs_a / played_a,
        "wins_rate_h":     wins_h / played_h,
        "wins_rate_a":     wins_a / played_a,
        "form_win_rate":   form_wr,
        "played_h":        played_h,
        "played_a":        played_a,
    }


def fetch_upcoming_team_stats(
    upcoming_df: "pd.DataFrame",  # type: ignore[type-arg]
    season: str | None = None,
) -> dict[str, dict]:
    """
    Batch-fetch /teams/statistics for every home + away team in upcoming_df.
    Returns {team_name: stats_dict}. Skips leagues not in API_FOOTBALL_IDS.
    Falls back to {} per team on any failure — never raises.
    """
    if not _KEY:
        return {}

    if season is None:
        season = config.API_SEASON

    results: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()

    for _, row in upcoming_df.iterrows():
        lg = str(row.get("league", ""))
        for team in [str(row.get("home_team", "")), str(row.get("away_team", ""))]:
            if team and (team, lg) not in seen:
                seen.add((team, lg))
                try:
                    s = fetch_team_season_stats(team, lg, season)
                    if s:
                        results[team] = s
                except Exception:
                    pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Upcoming fixture ID resolution (/fixtures)
# Maps (home_norm, away_norm) → API-Football fixture_id for upcoming matches.
# Required by Phase 5 (lineups) and Phase 7 (pre-match odds).
# ~1 call per league with upcoming fixtures. TTL 6h.
# ─────────────────────────────────────────────────────────────────────────────

def resolve_upcoming_fixture_ids(
    upcoming_df: "pd.DataFrame",
    season: str | None = None,
) -> dict[tuple[str, str], int]:
    """
    Fetch /fixtures?league=X&season=Y&next=10 per league in upcoming_df.
    Returns {(home_norm, away_norm): fixture_id}.
    """
    if not _KEY:
        return {}
    if season is None:
        season = config.API_SEASON

    result: dict[tuple[str, str], int] = {}
    for lg in upcoming_df["league"].unique():
        league_id = config.API_FOOTBALL_IDS.get(str(lg))
        if not league_id:
            continue
        data = _get("fixtures", {"league": league_id, "season": season, "next": 10}, ttl_h=6)
        if not data:
            continue
        for entry in data.get("response", []):
            teams = entry.get("teams", {})
            hn    = _norm_name(teams.get("home", {}).get("name", ""))
            an    = _norm_name(teams.get("away", {}).get("name", ""))
            fid   = entry.get("fixture", {}).get("id")
            if hn and an and fid:
                result[(hn, an)] = fid
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Lineup / formation features (/fixtures/lineups)
# Confirmed starting XIs appear ~1-2h before kickoff.
# Features: formation attacking intent score, forward count per side.
# ~40 calls/day (1 per upcoming fixture). TTL 2h.
# ─────────────────────────────────────────────────────────────────────────────

_FORMATION_ATTACK: dict[str, float] = {
    "3-4-3": 0.90, "4-3-3": 0.85, "4-4-2": 0.80, "4-2-3-1": 0.80,
    "3-4-1-2": 0.78, "3-4-2-1": 0.75, "4-1-4-1": 0.72, "4-3-2-1": 0.68,
    "4-4-1-1": 0.70, "3-5-2": 0.70, "5-3-2": 0.65, "4-5-1": 0.55,
    "5-3-1-1": 0.50, "5-4-1": 0.45,
}
_FWD_POS = {"F", "FW", "ST", "CF", "LW", "RW", "SS", "AM"}


def _formation_attack_score(formation: str) -> float:
    clean = (formation or "").strip()
    if clean in _FORMATION_ATTACK:
        return _FORMATION_ATTACK[clean]
    parts = clean.split("-")
    try:
        return min(0.40 + int(parts[-1]) * 0.12, 1.0)
    except (IndexError, ValueError):
        return 0.65


def fetch_lineup_features(
    upcoming_df: "pd.DataFrame",
    fixture_ids: "dict[tuple[str, str], int] | None" = None,
    season: str | None = None,
) -> dict[int, dict]:
    """
    Returns {row_index: {home_attack_formation, away_attack_formation,
                          combined_attack_intent, home_forward_count, away_forward_count}}.
    Skips rows where lineup not yet released (TTL 2h, so auto-refreshes closer to KO).
    """
    if not _KEY:
        return {}
    if fixture_ids is None:
        fixture_ids = resolve_upcoming_fixture_ids(upcoming_df, season)

    results: dict[int, dict] = {}

    for idx, row in upcoming_df.iterrows():
        hn  = _norm_name(str(row.get("home_team", "")))
        an  = _norm_name(str(row.get("away_team", "")))
        fid = fixture_ids.get((hn, an))
        if not fid:
            continue

        data = _get("fixtures/lineups", {"fixture": fid}, ttl_h=2)
        if not data or not data.get("response"):
            continue
        resp = data["response"]
        if len(resp) < 2:
            continue

        home_e = next((e for e in resp if _norm_name(e.get("team", {}).get("name", "")) == hn), resp[0])
        away_e = next((e for e in resp if _norm_name(e.get("team", {}).get("name", "")) == an), resp[-1])

        def _fwd_count(entry: dict) -> float:
            return float(sum(
                1 for p in entry.get("startXI", [])
                if (p.get("player", {}).get("pos") or "").upper() in _FWD_POS
            ))

        ha = _formation_attack_score(home_e.get("formation", ""))
        aa = _formation_attack_score(away_e.get("formation", ""))
        results[idx] = {
            "home_attack_formation":  ha,
            "away_attack_formation":  aa,
            "combined_attack_intent": ha + aa,
            "home_forward_count":     _fwd_count(home_e),
            "away_forward_count":     _fwd_count(away_e),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Head-to-head features (/fixtures/headtohead)
# Last 10 H2H matches between the two teams.
# Features: h2h_over25_rate, h2h_avg_goals, h2h_home_win_rate, h2h_n.
# ~40 calls/day (1 per upcoming match). TTL 168h (H2H history rarely changes).
# ─────────────────────────────────────────────────────────────────────────────

def fetch_h2h_features(
    upcoming_df: "pd.DataFrame",
    season: str | None = None,
) -> dict[int, dict]:
    """
    Returns {row_index: {h2h_over25_rate, h2h_avg_goals, h2h_home_win_rate, h2h_n}}.
    Requires team IDs from Phase 3 (_resolve_team_id). Falls back gracefully.
    """
    if not _KEY:
        return {}
    if season is None:
        season = config.API_SEASON

    results: dict[int, dict] = {}

    for idx, row in upcoming_df.iterrows():
        lg  = str(row.get("league", ""))
        ht  = str(row.get("home_team", ""))
        at  = str(row.get("away_team", ""))
        lid = config.API_FOOTBALL_IDS.get(lg)
        if not lid:
            continue
        try:
            home_id = _resolve_team_id(ht, lid, season)
            away_id = _resolve_team_id(at, lid, season)
        except Exception:
            continue
        if not home_id or not away_id:
            continue

        data = _get("fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 10}, ttl_h=168)
        if not data:
            continue
        matches = data.get("response", [])
        if not matches:
            continue

        over25, goals_list, home_wins = [], [], []
        for m in matches:
            g  = m.get("goals", {})
            gh = _to_int(g.get("home")) or 0
            ga = _to_int(g.get("away")) or 0
            t  = gh + ga
            goals_list.append(t)
            over25.append(1 if t > 2 else 0)
            m_home_id  = m.get("teams", {}).get("home", {}).get("id")
            m_home_won = m.get("teams", {}).get("home", {}).get("winner")
            home_wins.append(1 if (m_home_id == home_id and m_home_won) else 0)

        n = len(goals_list)
        results[idx] = {
            "h2h_over25_rate":   sum(over25)      / n,
            "h2h_avg_goals":     sum(goals_list)   / n,
            "h2h_home_win_rate": sum(home_wins)    / n,
            "h2h_n":             float(n),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Pre-match odds cross-validation (/odds)
# Fetches Bet365 O/U 2.5 odds from API-Football for upcoming fixtures.
# Cross-validates against our FD/OddsAPI odds; market agreement = confidence.
# ~40 calls/day (1 per upcoming fixture). TTL 6h.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prematch_odds_features(
    upcoming_df: "pd.DataFrame",
    fixture_ids: "dict[tuple[str, str], int] | None" = None,
    season: str | None = None,
) -> dict[int, dict]:
    """
    Returns {row_index: {api_odds_over25, api_odds_under25,
                          api_implied_over25, api_overround}}.
    TTL: 6h (odds drift closer to kickoff).
    """
    if not _KEY:
        return {}
    if fixture_ids is None:
        fixture_ids = resolve_upcoming_fixture_ids(upcoming_df, season)

    BET365_ID = 8
    results: dict[int, dict] = {}

    for idx, row in upcoming_df.iterrows():
        hn  = _norm_name(str(row.get("home_team", "")))
        an  = _norm_name(str(row.get("away_team", "")))
        fid = fixture_ids.get((hn, an))
        if not fid:
            continue

        data = _get("odds", {"fixture": fid, "bookmaker": BET365_ID}, ttl_h=6)
        if not data or not data.get("response"):
            continue

        over_odd = under_odd = None
        btts_yes = btts_no = None
        over35_odd = under35_odd = None
        over15_odd = under15_odd = None
        draw_odd = None

        for entry in data["response"]:
            for bk in entry.get("bookmakers", []):
                if bk.get("id") != BET365_ID:
                    continue
                for bet in bk.get("bets", []):
                    name = bet.get("name") or ""
                    if _is_fullmatch_btts(bet):
                        for v in bet.get("values", []):
                            odd = _to_float(v.get("odd"))
                            if v.get("value") == "Yes" and odd:
                                btts_yes = odd
                            elif v.get("value") == "No" and odd:
                                btts_no = odd
                    elif "Over/Under" in name:
                        for v in bet.get("values", []):
                            label = v.get("value", "")
                            odd   = _to_float(v.get("odd"))
                            if "Over 3.5" in label and odd:
                                over35_odd = odd
                            elif "Under 3.5" in label and odd:
                                under35_odd = odd
                            elif "Over 1.5" in label and odd:
                                over15_odd = odd
                            elif "Under 1.5" in label and odd:
                                under15_odd = odd
                            elif "Over 2.5" in label and odd:
                                over_odd = odd
                            elif "Under 2.5" in label and odd:
                                under_odd = odd
                    elif name in ("Match Winner", "1X2"):
                        for v in bet.get("values", []):
                            if v.get("value") == "Draw":
                                draw_odd = _to_float(v.get("odd"))

        results[idx] = {
            "api_odds_over25":    over_odd,
            "api_odds_under25":   under_odd,
            "api_implied_over25": round(1.0 / over_odd, 4) if over_odd else None,
            "api_overround":      round((1.0 / over_odd + 1.0 / under_odd) - 1.0, 4) if over_odd and under_odd else None,
            "api_implied_btts":   round(1.0 / btts_yes, 4) if btts_yes else None,
            "api_implied_over35": round(1.0 / over35_odd, 4) if over35_odd else None,
            "api_implied_over15": round(1.0 / over15_odd, 4) if over15_odd else None,
            "api_implied_draw":   round(1.0 / draw_odd, 4) if draw_odd else None,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — Season stage (current round / total rounds)
# Signals late-season pressure which affects team motivation and lineup rotation.
# ~25 calls/day (1 per league). TTL 24h.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_season_round_features(
    upcoming_df: "pd.DataFrame",
) -> dict[int, dict]:
    """
    Returns {row_index: {season_stage_ratio, is_late_season}}.
    season_stage_ratio: current_round / total_rounds (0=week1, 1=final week).
    is_late_season: 1 if stage_ratio >= 0.80.
    TTL: 24h (round advances weekly).
    """
    if not _KEY:
        return {}

    results: dict[int, dict] = {}

    # cache current round per league
    league_stage: dict[str, dict] = {}
    for lg in upcoming_df.get("league", pd.Series(dtype=str)).unique():
        lid = config.API_FOOTBALL_IDS.get(lg)
        szn = config.API_FOOTBALL_SEASONS.get(lg, "2025")
        if not lid:
            continue
        if lg in league_stage:
            continue

        # current round
        curr = _get("fixtures/rounds", {"league": lid, "season": szn, "current": "true"}, ttl_h=24)
        curr_rounds = curr.get("response", []) if curr else []

        # all rounds
        all_ = _get("fixtures/rounds", {"league": lid, "season": szn}, ttl_h=168)
        all_rounds = all_.get("response", []) if all_ else []

        if not curr_rounds or not all_rounds:
            continue

        def _parse_round_num(label: str) -> int:
            import re
            m = re.search(r"(\d+)", str(label))
            return int(m.group(1)) if m else 0

        current_n = _parse_round_num(curr_rounds[-1])
        total_n   = len(all_rounds)
        if total_n < 1:
            continue

        ratio = round(current_n / total_n, 4)
        league_stage[lg] = {
            "season_stage_ratio": ratio,
            "is_late_season":     float(ratio >= 0.80),
        }

    for idx, row in upcoming_df.iterrows():
        lg = str(row.get("league", ""))
        stage = league_stage.get(lg)
        if stage:
            results[idx] = stage

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10 — Coach tenure & caretaker detection
# Short tenures = instability = increased variance in team output.
# ~4 calls/day (1 per team per week, 7-day TTL).
# ─────────────────────────────────────────────────────────────────────────────

def fetch_coach_features(
    upcoming_df: "pd.DataFrame",
    fixture_ids: "dict[tuple[str, str], int] | None" = None,
    season: str | None = None,
) -> dict[int, dict]:
    """
    Returns {row_index: {home_coach_tenure_days, home_coach_is_caretaker,
                          away_coach_tenure_days, away_coach_is_caretaker}}.
    TTL: 7 days per team (coaches rarely change mid-week).
    """
    if not _KEY:
        return {}

    from datetime import date as date_cls

    team_coach: dict[str, dict] = {}

    def _get_coach(team_name: str, league: str) -> dict:
        if team_name in team_coach:
            return team_coach[team_name]
        lid = config.API_FOOTBALL_IDS.get(league)
        szn = config.API_FOOTBALL_SEASONS.get(league, "2025")
        if not lid:
            return {}
        team_id = _resolve_team_id(team_name, lid, szn)
        if not team_id:
            return {}
        data = _get("coachs", {"team": team_id}, ttl_h=168)
        if not data or not data.get("response"):
            team_coach[team_name] = {}
            return {}
        for coach in data["response"]:
            career = coach.get("career", [])
            # find current club (no end date)
            for c in reversed(career):
                if c.get("team", {}).get("id") == team_id and not c.get("end"):
                    start_str = c.get("start", "")
                    try:
                        start_date = date_cls.fromisoformat(str(start_str)[:10])
                        tenure_days = (date_cls.today() - start_date).days
                    except Exception:
                        tenure_days = 180
                    result = {
                        "coach_tenure_days":  tenure_days,
                        "coach_is_caretaker": float(tenure_days < 14),
                    }
                    team_coach[team_name] = result
                    return result
        team_coach[team_name] = {}
        return {}

    results: dict[int, dict] = {}
    for idx, row in upcoming_df.iterrows():
        ht = str(row.get("home_team", ""))
        at = str(row.get("away_team", ""))
        lg = str(row.get("league", ""))
        hc = _get_coach(ht, lg)
        ac = _get_coach(at, lg)
        results[idx] = {
            "home_coach_tenure_days":  float(hc.get("coach_tenure_days",  180)),
            "home_coach_is_caretaker": float(hc.get("coach_is_caretaker", 0.0)),
            "away_coach_tenure_days":  float(ac.get("coach_tenure_days",  180)),
            "away_coach_is_caretaker": float(ac.get("coach_is_caretaker", 0.0)),
        }
    return results
