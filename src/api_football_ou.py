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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.season_resolver import (SEASON_UNKNOWN, is_known as _season_known,  # noqa: E402
                                 record_degradation as _season_degraded,
                                 resolve_api_season)


def _season_for(league, kickoff_date=None, explicit: str | None = None, site: str = "") -> str:
    """The API-Football season for a league (R2 finding 6). `explicit` (a caller-supplied season)
    wins; otherwise src.season_resolver decides from the league's own calendar rule. Returns
    SEASON_UNKNOWN - and records the degradation - rather than a generic year: callers skip the
    enrichment on SEASON_UNKNOWN. This replaces the old generic config-level season ("2025")."""
    if explicit:
        return str(explicit)
    s = resolve_api_season(league, kickoff_date)
    if not _season_known(s):
        _season_degraded(f"api_football_ou.{site}", league)
    return s


def _row_date(row):
    return row.get("date") if hasattr(row, "get") else None

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


# ── Club identity ─────────────────────────────────────────────────────────────
#
# Board (OddsAPI) club names are matched to API-Football's names ONLY through the canonical,
# league-scoped resolver src.team_names.resolve (2026-11-02). The old _norm_name key stripped
# suffixes like "SP"/"MG"/"RJ" ("Atletico-MG" and "Atletico GO" both lost their state tag) and
# carried its own alias dict (_POST_NORM_ALIASES, now in the team_names alias store). Within one
# API-Football payload names are compared as API-Football wrote them; players by player id.

def _resolve_club(name: str, af_names) -> Optional[str]:
    """The ONE API-Football name in `af_names` (one league) that is provably `name`, else None."""
    from src.team_names import resolve
    return resolve(str(name or ""), sorted({str(n) for n in af_names if n}))


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


def _pct(val) -> Optional[float]:
    """'55%' -> 55.0; None/empty/unparseable -> None (never 0)."""
    if val is None:
        return None
    return _to_float(str(val).replace("%", "").strip() or None)


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
        # absent possession -> None (was "0" -> 0%, an invented value)
        "home_possession": _pct(home_s.get("Ball Possession")),
        "away_possession": _pct(away_s.get("Ball Possession")),
        "home_blocked":    _to_int(home_s.get("Blocked Shots")),
        "away_blocked":    _to_int(away_s.get("Blocked Shots")),
    }


def fetch_shots_for_league(league_id: int, season: str, league_name: str = "") -> pd.DataFrame:
    """
    Fetch match stats (shots, corners, fouls) for a league/season from cache.
    Returns DataFrame: date(datetime), home_team, away_team (API-Football spellings),
                       home_shots, away_shots, home_sot, away_sot,
                       home_corners, away_corners, home_fouls, away_fouls.
    Callers match the clubs with src.team_names.resolve within the league.
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

    print(f"[api_football_ou] {len(df)} fixtures with shot data for {label}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1b — Fixture results for leagues not on football-data.co.uk
# ─────────────────────────────────────────────────────────────────────────────

#: API-Football statuses of a match decided after extra time / on penalties. For these `goals` holds
#: the 120-minute score; `score.fulltime` holds the 90-minute one (regular time + stoppage).
EXTRA_TIME_STATUSES = ("AET", "PEN")


def _results_ttl(season, league_name: str) -> float:
    """A finished season keeps a year; the current one (or an unknown league) is refreshed every 6 h."""
    if league_name:
        from src.season_resolver import resolve_api_season, is_known
        cur = resolve_api_season(league_name)
        if is_known(cur) and str(season) != str(cur):
            return 8760
    return 6


def _result_rows(data: dict, with_status: bool) -> list[dict]:
    """One row per finished fixture of an API-Football /fixtures response.

    SETTLEMENT IS ON 90 MINUTES (owner rule): for an AET / PEN fixture the goals are taken from
    score.fulltime (the 90-minute score), never from `goals` (120 minutes). An extra-time fixture
    without a 90-minute score is skipped - never filled from the 120-minute total."""
    rows = []
    for f in data.get("response", []):
        fix   = f.get("fixture") or {}
        teams = f.get("teams") or {}
        goals = f.get("goals") or {}
        score = f.get("score") or {}
        ht    = score.get("halftime") or {}
        status = (fix.get("status") or {}).get("short") or "FT"
        src = (score.get("fulltime") or {}) if status in EXTRA_TIME_STATUSES else goals
        home_g, away_g = src.get("home"), src.get("away")
        if home_g is None or away_g is None:
            continue
        ht_home, ht_away = ht.get("home"), ht.get("away")
        row = {
            "date":          str(fix.get("date", ""))[:10],
            "home_team":     (teams.get("home") or {}).get("name", ""),
            "away_team":     (teams.get("away") or {}).get("name", ""),
            "home_goals":    int(home_g),
            "away_goals":    int(away_g),
            "ht_home_goals": int(ht_home) if ht_home is not None else None,
            "ht_away_goals": int(ht_away) if ht_away is not None else None,
        }
        if with_status:
            row["status"] = status
            row["fixture_id"] = fix.get("id")
        rows.append(row)
    return rows


def _results_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[df["date"].notna() & df["home_team"].notna() & df["away_team"].notna()]


def fetch_league_fixture_results(
    league_id: int, season: str, league_name: str = "", statuses: str = "FT"
) -> pd.DataFrame:
    """
    Fetch completed fixture results (goals + HT) for a league/season.
    Used for Saudi Pro League, K-League 1 and any other league not on FD.
    A completed result never changes, but the LIST of completed fixtures in a season still being
    played grows every week. The 1-year cache froze the current season's list (restored across CI
    runs by actions/cache), so its history stopped at the first fetch (Data Truth audit finding 35).
    A finished season keeps the year; the current one (or an unknown league) is refreshed every 6 h.

    `statuses` is API-Football's status filter. The default 'FT' keeps the training loader's
    frame exactly as it was. Settlement passes 'FT-AET-PEN': extra-time fixtures then carry their
    90-minute score (score.fulltime) plus `status` and `fixture_id` columns.

    Returns DataFrame: date, home_team, away_team,
                       home_goals, away_goals, ht_home_goals, ht_away_goals [, status, fixture_id]
    """
    ttl = _results_ttl(season, league_name)
    data = _get("fixtures", {"league": league_id, "season": season, "status": statuses}, ttl_h=ttl)
    if not data:
        return pd.DataFrame()

    rows = _result_rows(data, with_status=statuses != "FT")
    if not rows:
        return pd.DataFrame()

    label = league_name or str(league_id)
    print(f"[api_football_ou] {label} season {season}: {len(rows)} fixture results loaded")
    return _results_frame(rows)


def fetch_league_extra_time_results(
    league_id: int, season: str, league_name: str = ""
) -> Optional[pd.DataFrame]:
    """The league/season's fixtures decided after extra time or on penalties (status AET / PEN),
    each with its 90-minute score. The KNOCKOUT GUARD of update_results reads this before it lets
    a football-data / OddsAPI score settle a fixture: those sources do not say whether a score is
    90 or 120 minutes (MLS play-offs, Argentina knockouts).

    None when the list could not be fetched (no key, HTTP error) - the caller must not read that
    as "no extra-time fixtures". An empty frame means the provider answered with none."""
    if not _KEY:
        return None
    data = _get("fixtures", {"league": league_id, "season": season, "status": "AET-PEN"},
                ttl_h=_results_ttl(season, league_name))
    if data is None or data.get("errors"):           # quota / rate-limit answers come as HTTP 200
        return None
    rows = [r for r in _result_rows(data, with_status=True) if r["status"] in EXTRA_TIME_STATUSES]
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals",
                                     "ht_home_goals", "ht_away_goals", "status", "fixture_id"])
    return _results_frame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Injury features
# ─────────────────────────────────────────────────────────────────────────────

def get_top_scorers(league_id: int, season: str) -> dict[str, list[int]]:
    """
    Top 3 goal scorers per team for the league/season.
    Returns {API-Football team name: [player id, ...]}  — cached 7 days. A scorer without a
    player id or team is skipped (never matched by name).
    """
    data = _get("players/topscorers", {"league": league_id, "season": season}, ttl_h=168)
    if not data:
        return {}

    team_scorers: dict[str, list[int]] = {}
    for entry in data.get("response", []):
        pid        = (entry.get("player") or {}).get("id")
        stats_list = entry.get("statistics", [])
        if pid is None or not stats_list:
            continue
        team = (stats_list[0].get("team") or {}).get("name")
        if not team:
            continue
        if len(team_scorers.setdefault(team, [])) < 3:
            team_scorers[team].append(pid)

    return team_scorers


def get_upcoming_fixture_ids(league_id: int, season: str, next_n: int = 20) -> dict[tuple, int]:
    """
    Upcoming fixture IDs keyed by (home, away, date_str) as API-Football names the clubs.
    Cached 4 h.
    """
    data = _get("fixtures", {"league": league_id, "season": season, "next": next_n}, ttl_h=4)
    if not data:
        return {}

    result: dict[tuple, int] = {}
    for f in data.get("response", []):
        fix   = f.get("fixture", {})
        teams = f.get("teams", {})
        fix_id   = fix.get("id")
        date_str = str(fix.get("date", ""))[:10]
        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")
        if fix_id and home and away and len(date_str) == 10:
            result[(home, away, date_str)] = fix_id

    return result


def get_injuries_for_fixture(fixture_id: int) -> dict[str, list[int]]:
    """
    Injured/suspended players for a specific fixture.
    Returns {API-Football team name: [player id, ...]}  — cached 2 h.
    """
    data = _get("injuries", {"fixture": fixture_id}, ttl_h=2)
    if not data:
        return {}

    injuries: dict[str, list[int]] = {}
    for entry in data.get("response", []):
        team = (entry.get("team") or {}).get("name")
        pid  = (entry.get("player") or {}).get("id")
        if team and pid is not None:
            injuries.setdefault(team, []).append(pid)

    return injuries


def get_league_injury_context(league_id: int, season: str) -> dict:
    """
    Pre-fetch everything needed for injury features for all upcoming fixtures
    in one league.  Call once per league, reuse for every row.

    Returns (API-Football club names, player ids):
        {
            "top_scorers":  {team: [player_id, ...]},
            "fixture_ids":  {(home, away, date_str): fixture_id},
            "injuries":     {fixture_id: {team: [player_id, ...]}},
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
    Returns {key_attacker_absent_home: 0/1, key_attacker_absent_away: 0/1}, or NaN for both when
    the injuries were not fetched or the fixture id is unresolved. That used to be 0.0 ("we
    checked and nobody is out"), which is what all 154/154 board fixtures carried
    (Data Truth audit finding 47).
    """
    unknown = {"key_attacker_absent_home": float("nan"), "key_attacker_absent_away": float("nan")}
    if not ctx:
        return unknown

    date_str = str(match_date)[:10]
    # The board clubs resolved (src.team_names.resolve) among THAT date's API-Football fixtures
    # of the league; exactly one fixture or unknown.
    day = [(h, a, fid) for (h, a, d), fid in ctx.get("fixture_ids", {}).items() if d == date_str]
    rh = _resolve_club(home_team, [h for h, _, _ in day])
    ra = _resolve_club(away_team, [a for _, a, _ in day])
    hits = [(h, a, fid) for h, a, fid in day if h == rh and a == ra] if rh and ra else []
    if len(hits) != 1:
        return unknown
    af_home, af_away, fix_id = hits[0]
    out = {"key_attacker_absent_home": 0.0, "key_attacker_absent_away": 0.0}

    injuries     = ctx["injuries"].get(fix_id, {})
    top_scorers  = ctx["top_scorers"]

    home_scorers = top_scorers.get(af_home, [])
    away_scorers = top_scorers.get(af_away, [])
    home_injured = injuries.get(af_home, [])
    away_injured = injuries.get(af_away, [])

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

def _resolve_team_id(
    team_name: str, league_id: int, season: str
) -> Optional[int]:
    """
    Resolve API-Football team ID by name within a league/season: the league-season's team list
    (GET /teams?league=&season=, one call per league-season, disk-cached a year by _get) and the
    canonical resolver src.team_names.resolve over its names. None when no team is provably the
    club. (The old persistent team_id_cache.json was keyed by the _norm_name identity guess.)
    """
    data = _get("teams", {"league": league_id, "season": season}, ttl_h=8760)
    if not data:
        return None
    ids: dict[str, int] = {}
    for entry in data.get("response", []):
        t = entry.get("team", {}) or {}
        if t.get("id") and t.get("name"):
            ids[str(t["name"])] = t["id"]
    hit = _resolve_club(team_name, ids)
    return ids.get(hit) if hit else None


def fetch_team_season_stats(
    team_name: str,
    league_name: str,
    season: str | None = None,
    kickoff_date=None,
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

    season = _season_for(league_name, kickoff_date, season, "fetch_team_season_stats")
    if not _season_known(season):
        return {}

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

    played_h = _to_int((fixtures.get("played") or {}).get("home"))
    played_a = _to_int((fixtures.get("played") or {}).get("away"))
    wins_h   = _to_int((fixtures.get("wins")   or {}).get("home"))
    wins_a   = _to_int((fixtures.get("wins")   or {}).get("away"))
    cs_h     = _to_int(cs.get("home"))
    cs_a     = _to_int(cs.get("away"))

    def _rate(num, den):
        # 0 games played or a missing count -> None, so the caller's override is skipped (the
        # divisor used to be forced to 1, giving a 0.0 rate at every season start)
        return (num / den) if (num is not None and den) else None

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
    form_wr = recent.count("W") / len(recent) if recent else None

    return {
        "goals_for_h":     goals_for_h,
        "goals_for_a":     goals_for_a,
        "goals_against_h": goals_agn_h,
        "goals_against_a": goals_agn_a,
        "cs_rate_h":       _rate(cs_h, played_h),
        "cs_rate_a":       _rate(cs_a, played_a),
        "wins_rate_h":     _rate(wins_h, played_h),
        "wins_rate_a":     _rate(wins_a, played_a),
        "form_win_rate":   form_wr,
        "played_h":        played_h,
        "played_a":        played_a,
    }


def fetch_upcoming_team_stats(
    upcoming_df: "pd.DataFrame",  # type: ignore[type-arg]
    season: str | None = None,
) -> dict[tuple[str, str], dict]:
    """
    Batch-fetch /teams/statistics for every home + away team in upcoming_df.
    Returns {(league, team_name): stats_dict} - keyed by league so two board clubs with the same
    name in different leagues never overwrite each other. Skips leagues not in API_FOOTBALL_IDS.
    Falls back to {} per team on any failure — never raises.
    """
    if not _KEY:
        return {}

    results: dict[tuple[str, str], dict] = {}
    seen: set[tuple[str, str]] = set()

    for _, row in upcoming_df.iterrows():
        lg = str(row.get("league", ""))
        for team in [str(row.get("home_team", "")), str(row.get("away_team", ""))]:
            if team and (team, lg) not in seen:
                seen.add((team, lg))
                try:
                    s = fetch_team_season_stats(team, lg, season, kickoff_date=_row_date(row))
                    if s:
                        results[(lg, team)] = s
                except Exception as e:
                    print(f"[api_football_ou] team stats skipped for {team} ({lg}): {e}")

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
) -> dict[tuple[str, str, str, str], int]:
    """
    Fetch /fixtures?league=X&season=Y&next=10 per league in upcoming_df.
    Returns {(league, date_str, home, away): fixture_id} with API-Football's club names - see
    _fixture_key; _row_fixture_id resolves board names onto them. Keyed by
    league and date too: one name-only dict across all leagues let a same-named pairing in
    another competition (or a second leg) overwrite the id. Each fixture's API-Football team ids
    are kept in _FIXTURE_TEAM_IDS so lineups are matched by id, never by position.
    """
    if not _KEY:
        return {}
    result: dict[tuple[str, str], int] = {}
    for lg in upcoming_df["league"].unique():
        league_id = config.API_FOOTBALL_IDS.get(str(lg))
        if not league_id:
            continue
        _d = (upcoming_df.loc[upcoming_df["league"] == lg, "date"].min()
              if "date" in upcoming_df.columns else None)
        szn = _season_for(lg, _d, season, "resolve_upcoming_fixture_ids")
        if not _season_known(szn):
            continue
        data = _get("fixtures", {"league": league_id, "season": szn, "next": 10}, ttl_h=6)
        if not data:
            continue
        for entry in data.get("response", []):
            teams = entry.get("teams", {})
            fid   = entry.get("fixture", {}).get("id")
            key   = _fixture_key(lg, entry.get("fixture", {}).get("date"),
                                 teams.get("home", {}).get("name", ""),
                                 teams.get("away", {}).get("name", ""))
            if key and fid:
                result[key] = fid
                _FIXTURE_TEAM_IDS[fid] = (teams.get("home", {}).get("id"),
                                          teams.get("away", {}).get("id"))
    return result


#: fixture_id -> (home team id, away team id), filled by resolve_upcoming_fixture_ids
_FIXTURE_TEAM_IDS: dict[int, tuple] = {}


def _fixture_key(league, date, home, away):
    """(league, 'YYYY-MM-DD', home, away), or None when any part is missing. Names as given."""
    d = str(date)[:10] if date is not None else ""
    hn, an = str(home or "").strip(), str(away or "").strip()
    if not (league and len(d) == 10 and hn and an):
        return None
    return (str(league), d, hn, an)


def _row_fixture_id(row, fixture_ids: dict):
    """The API-Football fixture id of a board row: both clubs resolved (src.team_names.resolve)
    among the API-Football fixtures of the SAME league and date; exactly one fixture or None."""
    key = _fixture_key(row.get("league"), _row_date(row), row.get("home_team"), row.get("away_team"))
    if not key:
        return None
    lg, d, home, away = key
    day = [(k, fid) for k, fid in fixture_ids.items() if k[0] == lg and k[1] == d]
    rh = _resolve_club(home, [k[2] for k, _ in day])
    ra = _resolve_club(away, [k[3] for k, _ in day])
    hits = [fid for k, fid in day if rh and ra and k[2] == rh and k[3] == ra]
    return hits[0] if len(hits) == 1 else None


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
    # unlisted or missing formation -> NaN (was a typed formula / 0.65)
    return float("nan")


def fetch_lineup_features(
    upcoming_df: "pd.DataFrame",
    fixture_ids: "dict[tuple, int] | None" = None,
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
        fid = _row_fixture_id(row, fixture_ids)
        if not fid:
            continue

        data = _get("fixtures/lineups", {"fixture": fid}, ttl_h=2)
        if not data or not data.get("response"):
            continue
        resp = data["response"]
        if len(resp) < 2:
            continue

        # Match each block by the fixture's API-Football team id, never by position: an
        # unmatched block used to be assumed to be resp[0] / resp[-1].
        home_id, away_id = _FIXTURE_TEAM_IDS.get(fid, (None, None))
        home_e = next((e for e in resp if home_id and e.get("team", {}).get("id") == home_id), None)
        away_e = next((e for e in resp if away_id and e.get("team", {}).get("id") == away_id), None)
        if home_e is None or away_e is None:
            print(f"[api_football_ou] lineup skipped for fixture {fid}: team ids not matched")
            continue

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
    results: dict[int, dict] = {}

    for idx, row in upcoming_df.iterrows():
        lg  = str(row.get("league", ""))
        ht  = str(row.get("home_team", ""))
        at  = str(row.get("away_team", ""))
        lid = config.API_FOOTBALL_IDS.get(lg)
        if not lid:
            continue
        szn = _season_for(lg, _row_date(row), season, "fetch_h2h_features")
        if not _season_known(szn):
            continue
        try:
            home_id = _resolve_team_id(ht, lid, szn)
            away_id = _resolve_team_id(at, lid, szn)
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
            gh = _to_int(g.get("home"))
            ga = _to_int(g.get("away"))
            if gh is None or ga is None:     # postponed/abandoned: no score, not a 0-0
                continue
            t  = gh + ga
            goals_list.append(t)
            over25.append(1 if t > 2 else 0)
            m_home_id  = m.get("teams", {}).get("home", {}).get("id")
            m_home_won = m.get("teams", {}).get("home", {}).get("winner")
            home_wins.append(1 if (m_home_id == home_id and m_home_won) else 0)

        n = len(goals_list)
        if not n:
            continue
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
    fixture_ids: "dict[tuple, int] | None" = None,
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
        fid = _row_fixture_id(row, fixture_ids)
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
        if not lid:
            continue
        _d = upcoming_df.loc[upcoming_df["league"] == lg, "date"].min() if "date" in upcoming_df.columns else None
        szn = _season_for(lg, _d, None, "fetch_season_round_features")
        if not _season_known(szn):
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
    fixture_ids: "dict[tuple, int] | None" = None,
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

    # keyed (league, team): a same-named club in another league is a different coach
    def _get_coach(team_name: str, league: str) -> dict:
        key = (league, team_name)
        if key in team_coach:
            return team_coach[key]
        lid = config.API_FOOTBALL_IDS.get(league)
        if not lid:
            return {}
        szn = _season_for(league, None, None, "fetch_coach_features")
        if not _season_known(szn):
            return {}
        team_id = _resolve_team_id(team_name, lid, szn)
        if not team_id:
            return {}
        data = _get("coachs", {"team": team_id}, ttl_h=168)
        if not data or not data.get("response"):
            team_coach[key] = {}
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
                        # unparseable start -> tenure unknown (was an invented 180 days)
                        team_coach[key] = {}
                        return {}
                    result = {
                        "coach_tenure_days":  tenure_days,
                        "coach_is_caretaker": float(tenure_days < 14),
                    }
                    team_coach[key] = result
                    return result
        team_coach[key] = {}
        return {}

    results: dict[int, dict] = {}
    for idx, row in upcoming_df.iterrows():
        ht = str(row.get("home_team", ""))
        at = str(row.get("away_team", ""))
        lg = str(row.get("league", ""))
        hc = _get_coach(ht, lg)
        ac = _get_coach(at, lg)
        # unknown coach -> NaN, never 180 days / "not a caretaker"
        _nan = float("nan")
        results[idx] = {
            "home_coach_tenure_days":  float(hc["coach_tenure_days"]) if hc else _nan,
            "home_coach_is_caretaker": float(hc["coach_is_caretaker"]) if hc else _nan,
            "away_coach_tenure_days":  float(ac["coach_tenure_days"]) if ac else _nan,
            "away_coach_is_caretaker": float(ac["coach_is_caretaker"]) if ac else _nan,
        }
    return results
