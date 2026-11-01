"""
Update bets_ledger.csv with actual match results.

Fetches completed match scores from OddsAPI and fills in `result` (WIN/LOSS/VOID)
and `pnl` for any tips where the match has been played but the result is missing.

PnL is calculated on a flat 1-unit stake:
    WIN  → +(odds - 1)
    LOSS → -1.0
    VOID → 0.0

Usage
-----
    python update_results.py           # fetch last 3 days of completed matches
    python update_results.py --days 7  # look back further
    python update_results.py --dry-run # show what would change, no writes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

LEDGER_FILE        = config.OUTPUT_DIR / "bets_ledger.csv"
PLAYER_LEDGER_FILE = config.OUTPUT_DIR / "player_ledger.csv"
SHARP_LEDGER_FILE  = config.OUTPUT_DIR / "sharp_ledger.csv"
SIDE_LEDGER_FILE   = config.OUTPUT_DIR / "side_bets_ledger.csv"


# ── Team name normalisation ───────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase + strip accents + collapse whitespace for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_str.lower().split())


def _names_match(a: str, b: str) -> bool:
    """True if two team names refer to the same club, by the canonical resolver
    (src.team_names.resolve_result: normalised equality, aliases incl. result spellings, then
    identity tokens that must pair up, reserve sides only with reserve sides).

    This was substring-or-first-word, which called Real Madrid = Real Sociedad, Man City = Man
    United, Bristol City = Bristol Rovers and Sheffield United = Sheffield Wednesday, while missing
    LA Galaxy / Los Angeles Galaxy, Bodø/Glimt / Bodo/Glimt and Bragantino-SP / RB Bragantino, so
    8 live tips sat unsettled and 64 had no closing price (Data Truth audit findings 17, 18).

    PAIRWISE ONLY - NOT USED FOR SETTLEMENT HERE. With a one-name pool the resolver's ambiguity
    refusal can never fire, so any token-subset pair matches ('Inter' / 'Inter Miami'). Settlement
    in this file resolves inside the whole league+date pool with `_match_event`; this stays only
    for external callers (src/research/league_weekly.py) until they move to a pool."""
    from src.team_names import resolve_result
    if not a or not b:
        return False
    return resolve_result(str(a), [str(b)]) is not None or resolve_result(str(b), [str(a)]) is not None


def _canon_league(league) -> str | None:
    """Canonical league name through the registry (config/leagues.json), or None when unknown."""
    from src.league_registry import canonical_name
    return canonical_name(league)


def _match_event(home, away, events: list[dict], dates) -> dict | None:
    """THE completed event for a fixture among `events` played on one of `dates`, or None.

    Both clubs are resolved by src.team_names.resolve_result against EVERY home / away name in
    that date pool, so the resolver's ambiguity refusal works ('Inter' with both 'Inter' and
    'Inter Miami' in the pool resolves to neither). Several hits are accepted only when they are
    the same fixture reported twice with the same score; otherwise nothing is returned - a late
    settlement is better than a wrong one."""
    from src.team_names import resolve_result
    if not home or not away:
        return None
    dates = {str(d)[:10] for d in ([dates] if isinstance(dates, str) else dates)}
    cands = [e for e in events if str(e.get("date_str", ""))[:10] in dates
             and e.get("home_team") and e.get("away_team")]
    if not cands:
        return None
    rh = resolve_result(str(home), sorted({str(e["home_team"]) for e in cands}))
    ra = resolve_result(str(away), sorted({str(e["away_team"]) for e in cands}))
    if not rh or not ra:
        return None
    hits = [e for e in cands if str(e["home_team"]) == rh and str(e["away_team"]) == ra]
    scores = {(e["home_score"], e["away_score"], e.get("ht_home"), e.get("ht_away")) for e in hits}
    return hits[0] if hits and len(scores) == 1 else None


# ── football-data.co.uk scores fetch ─────────────────────────────────────────
# "new" format: football-data.co.uk/new/{code}.csv  (cols: Home/Away/HG/AG/Date, no HT)
# "std" format: football-data.co.uk/mmz4281/{season}/{code}.csv (cols: HomeTeam/AwayTeam/FTHG/FTAG/HTHG/HTAG/Date)
# Column layout of each football-data file format (a property of the provider, not of a league):
# (home_col, away_col, ft_hg, ft_ag, date_col, ht_hg, ht_ag)
_FD_LAYOUT = {
    "new": ("Home", "Away", "HG", "AG", "Date", None, None),
    "std": ("HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
}


def _fd_sources() -> dict:
    """{canonical league: (fmt, code, *layout)} for every registry league with a football_data_code.

    Was a hand-typed table of 14 leagues, so enabled leagues missing from it (Championship,
    Serie B, Romanian Superliga, ...) fell to OddsAPI's 3-day window and then API-Football. The
    file format follows the league's model_type: new_format leagues are football-data's /new/
    files, standard leagues its per-season mmz4281 files. A league with a code but no model_type
    has no known format and is left out (logged)."""
    from src import league_registry as R
    from src.league_registry import LeagueMap
    out = {}
    for e in R.leagues():
        code = e.get("football_data_code")
        if not code:
            continue
        fmt = {"standard": "std", "new_format": "new"}.get(e.get("model_type"))
        if fmt is None:
            log.debug(f"  FD {e['name']}: football_data_code {code} but no model_type - format unknown, not used")
            continue
        out[e["name"]] = (fmt, code, *_FD_LAYOUT[fmt])
    return LeagueMap(out, R.alias_map())


_FD_SOURCES = _fd_sources()

_FD_NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
_FD_STD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def _fd_std_season_codes(today=None) -> list[str]:
    """football-data season folders for the CURRENT and PREVIOUS split season ('2627', '2526').
    This was the literal '2526', so from August 2026 no 2026/27 fixture in these leagues could
    ever settle here (Data Truth audit findings 15, 29). Previous season too: a tip from May still
    needs grading in August."""
    d = pd.Timestamp(today or datetime.utcnow().date())
    start = d.year if d.month >= 7 else d.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in (start, start - 1)]


def fetch_scores_fd(league: str) -> list[dict]:
    """Fetch completed results from football-data.co.uk (free, no quota). Includes HT scores where available."""
    if league not in _FD_SOURCES:
        return []

    fmt, code, home_col, away_col, hg_col, ag_col, date_col, ht_hg_col, ht_ag_col = _FD_SOURCES[league]

    try:
        from io import StringIO

        urls = ([_FD_NEW_URL.format(code=code)] if fmt == "new" else
                [_FD_STD_URL.format(season=s, code=code) for s in _fd_std_season_codes()])
        parts = []
        for url in urls:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                log.warning(f"  FD {league}: HTTP {r.status_code} for {url}")
                continue
            parts.append(pd.read_csv(StringIO(r.text), encoding="utf-8-sig",
                                     on_bad_lines="skip", low_memory=False))
        if not parts:
            return []
        raw = pd.concat(parts, ignore_index=True)

        if fmt == "new" and "Season" in raw.columns:
            # This year and last year, never a literal list: '2025|2026' dropped every NF result
            # from the 2027 calendar seasons on (Data Truth audit finding 21).
            _y = datetime.utcnow().year
            raw = raw[raw["Season"].astype(str).str.contains(f"{_y - 1}|{_y}")].copy()

        results = []
        for _, row in raw.iterrows():
            try:
                hg = int(float(row[hg_col]))
                ag = int(float(row[ag_col]))
                dt = pd.to_datetime(row[date_col], dayfirst=True, errors="coerce")
                if pd.isna(dt):
                    continue

                entry = {
                    "home_team":  str(row[home_col]).strip(),
                    "away_team":  str(row[away_col]).strip(),
                    "home_score": hg,
                    "away_score": ag,
                    "date_str":   str(dt.date()),
                }

                # HT scores (standard format only)
                if ht_hg_col and ht_ag_col:
                    try:
                        entry["ht_home"] = int(float(row[ht_hg_col]))
                        entry["ht_away"] = int(float(row[ht_ag_col]))
                    except (ValueError, KeyError):
                        pass

                results.append(entry)
            except (ValueError, KeyError):
                continue

        log.info(f"  FD {league}: {len(results)} completed matches loaded")
        return results

    except Exception as e:
        log.warning(f"  FD {league} exception: {e}")
        return []


# ── OddsAPI scores fetch ──────────────────────────────────────────────────────

def fetch_scores(sport_key: str, days_from: int) -> list[dict]:
    """
    Fetch completed events from OddsAPI scores endpoint.
    Returns list of dicts: {home_team, away_team, home_score, away_score, date_str}.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    try:
        r = requests.get(
            url,
            params={
                "apiKey":    config.ODDS_API_KEY,
                "daysFrom":  min(days_from, 3),  # OddsAPI max is 3
            },
            timeout=15,
        )
        if r.status_code != 200:
            log.debug(f"  OddsAPI scores {sport_key}: HTTP {r.status_code} — {r.text[:200]}")
            return []

        results = []
        for event in r.json():
            if not event.get("completed"):
                continue
            scores = event.get("scores") or []
            if len(scores) < 2:
                continue

            score_map: dict[str, int] = {}
            for s in scores:
                try:
                    score_map[s["name"]] = int(s["score"])
                except (KeyError, ValueError):
                    pass

            home = event.get("home_team")
            away = event.get("away_team")
            if not home or not away or home not in score_map or away not in score_map:
                continue

            dt_raw = event.get("commence_time")
            try:
                dt = pd.to_datetime(dt_raw, utc=True).tz_localize(None)
            except Exception:
                continue
            if pd.isna(dt):
                continue

            results.append({
                "home_team":   home,
                "away_team":   away,
                "home_score":  score_map[home],
                "away_score":  score_map[away],
                "date_str":    str(dt.date()),
                # OddsAPI event id == the shadow log's fixture_id. Additive: no ledger reads it.
                "event_id":    event.get("id"),
            })
        return results

    except Exception as e:
        log.debug(f"  OddsAPI scores {sport_key} exception: {e}")
        return []


# ── CLV lookup ───────────────────────────────────────────────────────────────

_CSV_ARCHIVES = None


# Every persistent odds archive we can pull a closing price out of. All share the schema
# snapshot_date,snapshot_ts,match_date,league,match,market,odds.
_ARCHIVE_FILES = (
    "newformat_odds_dense.csv",               # new-format O/U, EVERY predict run (densest NF)
    "newformat_odds_history.csv",             # new-format: O/U 2.5 + BTTS + O/U 1.5/3.5, 3x/day
    "standard_odds_history.csv",              # standard O/U 2.5/1.5/3.5, every predict run
    "standard_sidemarket_odds_history.csv",   # standard BTTS + O/U 1.5/3.5 (std_odds_capture)
)
# Order matters: the dense archives come first so a fixture present in both resolves against
# the finer-grained curve, whose last snapshot is much closer to the true closing line.


def _load_archives() -> list:
    """Load + key the odds archives once per process."""
    global _CSV_ARCHIVES
    if _CSV_ARCHIVES is None:
        _CSV_ARCHIVES = []
        for name in _ARCHIVE_FILES:
            p = config.OUTPUT_DIR / name
            if not p.exists():
                continue
            try:
                d = pd.read_csv(p)
                mm = d["match"].astype(str)
                d["_h"] = mm.apply(lambda m: m.split(" vs ")[0].strip() if " vs " in m else None)
                d["_a"] = mm.apply(lambda m: m.split(" vs ")[1].strip() if " vs " in m else None)
                d["_dk"] = d["match_date"].astype(str).str[:10]
                # Canonical league per row, so a lookup never crosses competitions.
                canon = {lg: _canon_league(lg) for lg in d["league"].dropna().unique()} \
                    if "league" in d.columns else {}
                d["_lk"] = d["league"].map(canon) if "league" in d.columns else None
                _CSV_ARCHIVES.append(d)
            except Exception as e:                                    # noqa: BLE001
                log.warning(f"  odds archive {name} unreadable ({e}) - not used for closing prices")
    return _CSV_ARCHIVES


def _closing_for_market(home: str, away: str, match_date: str, mkt: str,
                        league: str | None = None) -> float:
    """Last archived price for an explicit archive market key
    (over25 / under25 / btts_yes / over15 / over35 / ...). np.nan when not found.

    LEAGUE-SCOPED, RESOLVED IN THE POOL. The ledger carries OddsAPI names and the archives
    API-Football ones ("Viborg FF"/"Viborg", "Tijuana"/"Club Tijuana"), so names are resolved by
    team_names.resolve_result against every club of THAT league on THAT date. This used to search
    every league on the date with lowercase equality and then the pairwise _names_match, so a
    same-named fixture in another competition could supply the close. No league -> no close."""
    lk, dk = _canon_league(league), str(match_date)[:10]
    if lk is None:
        log.debug(f"  closing {mkt}: league {league!r} not in the registry - no closing price")
        return np.nan
    from src.team_names import resolve_result
    for arch in _load_archives():
        same = arch[(arch["_dk"] == dk) & (arch["_lk"] == lk) & (arch["market"].astype(str) == mkt)
                    & arch["_h"].notna() & arch["_a"].notna()]
        if same.empty:
            continue
        rh = resolve_result(str(home), sorted(set(same["_h"])))
        ra = resolve_result(str(away), sorted(set(same["_a"])))
        if not rh or not ra:
            continue
        m = same[(same["_h"] == rh) & (same["_a"] == ra)]
        if not m.empty:
            try:
                return float(m.sort_values("snapshot_ts").iloc[-1]["odds"])
            except (TypeError, ValueError):
                continue
    return np.nan


def _closing_from_csv(home: str, away: str, match_date: str, side: str,
                      league: str | None = None) -> float:
    """O/U 2.5 closing price from the archives. Used when odds_history_v9.json has already
    rolled the fixture out (10-day purge) — the cause of new-format's 9% closing coverage."""
    return _closing_for_market(home, away, match_date,
                               "under25" if side == "UNDER" else "over25", league=league)


# side_bets_ledger market key -> the market key used inside the odds archives.
# The ledger stores the BTTS *Yes* price, which the archives call btts_yes.
_SIDE_ARCHIVE_MARKET = {"btts": "btts_yes", "over15": "over15", "over35": "over35"}


def _closing_odds(home: str, away: str, match_date: str, side: str,
                  league: str | None = None) -> float:
    """Closing (last pre-KO) odds: try the ephemeral JSON first, then fall back to the
    persistent CSV archives so a fixture purged from the JSON still gets a closing price."""
    val = _closing_odds_json(home, away, match_date, side)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = _closing_from_csv(home, away, match_date, side, league=league)
    return val


def _closing_odds_json(home: str, away: str, match_date: str, side: str) -> float:
    """
    Return the LAST recorded odds snapshot before kick-off from odds_history_v9.json.
    This approximates the closing line.

    EXACT KEY ONLY. The JSON is keyed "home vs away | date" in the same OddsAPI spelling the
    ledger stores, with no league. The old fallback compared lowercased names over the WHOLE file
    and never checked the date, so any fixture with the same names on any date could supply the
    close. A miss here falls through to the league-scoped CSV archives.
    """
    if not config.ODDS_HISTORY_JSON.exists():
        return np.nan
    try:
        history = json.loads(config.ODDS_HISTORY_JSON.read_text(encoding="utf-8"))
        key = f"{home} vs {away} | {match_date}"
        snapshots = history.get(key)
        if not snapshots:
            return np.nan
        last = snapshots[-1]
        field = "under" if side == "UNDER" else "over"
        val = last.get(field)
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan


# ── API-Football results (last-resort settlement source) ─────────────────────

_AF_RESULTS_CACHE: dict = {}


def fetch_scores_apifootball(league: str, match_date: str | None = None) -> list[dict]:
    """Completed fixtures for a league from API-Football, in fetch_scores() shape.

    WHY THIS EXISTS: leagues without a football-data source (Argentina, Brazil, Finland,
    Norway, Mexico) fall back to the OddsAPI scores endpoint, whose lookback is capped at
    THREE DAYS (`min(days_from, 3)` — an OddsAPI limit, not ours). A tip in one of those
    leagues that is not settled within 3 days can therefore NEVER be settled, and 89 such
    rows had been stranded since May 2026 with no result and no P/L. Another 60 sat in
    football-data leagues where the fixture or the team names did not reconcile.

    API-Football has all of them and `fetch_league_fixture_results` is cached for a YEAR
    (completed results never change), so this is one call per league per season.

    Called LAZILY — only for a row the primary source could not resolve — so a healthy run
    where everything settles normally makes no extra API calls at all. That matters: this
    job runs every 2 hours and has no actions/cache step.
    """
    out: list[dict] = []
    lid = config.API_FOOTBALL_IDS.get(league)
    # League-aware season (R2 finding 6): was API_FOOTBALL_SEASONS.get(league, API_SEASON="2025").
    # The MATCH's season, not today's: a tip from the previous campaign was looked up in the
    # current one and could never settle - 52 live O/U 2.5 tips from May-Oct 2026 were stranded
    # (Data Truth audit findings 22, 30). Cached per (league, season).
    from src.season_resolver import resolve_api_season, is_known, record_degradation
    season = resolve_api_season(league, match_date) if lid else None
    if (league, season) in _AF_RESULTS_CACHE:
        return _AF_RESULTS_CACHE[(league, season)]
    if lid and not is_known(season):
        record_degradation("update_results._af_completed", league)
        log.warning(f"  [api-football] {league}: season unresolvable - fallback settlement skipped")
        lid = None
    if lid:
        try:
            from src.api_football_ou import fetch_league_fixture_results
            df = fetch_league_fixture_results(lid, season, league)
            for _, r in df.iterrows():
                try:
                    hg, ag = int(r["home_goals"]), int(r["away_goals"])
                except (TypeError, ValueError, KeyError):
                    continue
                # A fixture without both clubs or a date cannot be identified - skip it, never
                # carry an empty name into matching.
                if any(pd.isna(r.get(c)) or not str(r.get(c)).strip()
                       for c in ("home_team", "away_team", "date")):
                    continue
                ev = {
                    "home_team":  str(r["home_team"]),
                    "away_team":  str(r["away_team"]),
                    "home_score": hg,
                    "away_score": ag,
                    "date_str":   str(r["date"])[:10],
                }
                hh, ha = r.get("ht_home_goals"), r.get("ht_away_goals")
                if pd.notna(hh) and pd.notna(ha):
                    ev["ht_home"], ev["ht_away"] = int(hh), int(ha)
                out.append(ev)
            log.info(f"  [api-football] {league}: {len(out)} completed fixtures "
                     f"(season {season}) for fallback settlement")
        except Exception as e:
            log.warning(f"  [api-football] {league}: results fetch failed ({e})")
    else:
        log.debug(f"  [api-football] {league}: no API_FOOTBALL_IDS entry")

    _AF_RESULTS_CACHE[(league, season)] = out
    return out


# ── Result lookup ─────────────────────────────────────────────────────────────

def _find_result(
    home: str, away: str, date_str: str, side: str,
    completed: list[dict],
) -> tuple[str, float, dict] | None:
    """
    Search `completed` for this fixture.
    Returns (result_str, total_goals, extras) or None if not found.
    extras contains ht_total if HT scores are available.
    """
    ev = _match_event(home, away, completed, {date_str})
    if ev is not None:
        total = ev["home_score"] + ev["away_score"]
        # Both half-time goals or no half-time total: one missing half was added as -1 before.
        ht_total = (ev["ht_home"] + ev["ht_away"]
                    if ev.get("ht_home") is not None and ev.get("ht_away") is not None else None)

        if side == "OVER":
            won = total > 2.5
        elif side == "UNDER":
            won = total <= 2.5
        elif side == "HT_OVER_0.5":
            won = ht_total is not None and ht_total >= 1
        elif side == "HT_UNDER_0.5":
            won = ht_total is not None and ht_total < 1
        elif side == "HT_OVER_1.5":
            won = ht_total is not None and ht_total >= 2
        elif side == "HT_UNDER_1.5":
            won = ht_total is not None and ht_total <= 1
        else:
            return None

        if side.startswith("HT_") and ht_total is None:
            return None  # no HT data available, can't grade

        result = "WIN" if won else "LOSS"
        extras = {}
        if ht_total is not None:
            extras["ht_score"] = f"{ev['ht_home']}-{ev['ht_away']}"
            extras["ht_total"] = ht_total
        return result, total, extras

    return None


# ── Player prop result resolver ───────────────────────────────────────────────
#
# SETTLEMENT NEVER GUESSES (Data Truth 2026-11-01, issue #1 steps 1-2):
#   * the fixture is the row's stored API-Football fixture_id; only a row without one is looked up,
#     by resolving BOTH clubs inside that league's fixtures of that date (src.team_names), and the
#     id found is written back to the row so the next run settles by id;
#   * the player is matched by player_id ONLY. The first-initial + surname guess is gone; a row
#     with no player_id stays pending (logged) - its producer has the id;
#   * "did not play" needs evidence from the match sheet: 0 minutes, an unused substitute, or
#     absence from both clubs' published match squads. A null minutes value is not evidence;
#   * a stat the provider left null is never read as 0. It is taken from the fixture's own
#     record only when that record is provably complete (goal events that add up to the final
#     score; shots on target that add up to the team total), otherwise the row stays pending.

_FT_STATUSES = ("FT", "AET", "PEN")
_VOID_STATUSES = {"PST": "POSTPONED", "CANC": "CANCELLED", "ABD": "ABANDONED"}

_PLAYER_MARKET_STAT = {"goals": "goals", "sot": "shots_on_target", "cards": "yellow_card",
                       "assists": "assists"}


def _resolve_player_market(market: str, stats: dict) -> str:
    """WIN/LOSS for a player prop: `<base>` means 1+, `<base><N>` means N+ ('sot2' = 2+ shots on
    target). The producer writes sot2/sot3/goals2; this used to know only the bare names, so every
    multi-threshold prop stayed pending for ever (Data Truth audit finding 19). Unknown market or a
    stat that is not known -> '' (the row stays pending; a missing stat was read as 0 = LOSS)."""
    import re
    m = re.fullmatch(r"([a-z]+?)(\d*)", str(market or "").strip().lower())
    if not m or m.group(1) not in _PLAYER_MARKET_STAT:
        return ""
    need = int(m.group(2) or 1)
    v = stats.get(_PLAYER_MARKET_STAT[m.group(1)])
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "WIN" if int(v) >= need else "LOSS"


def _af_get(endpoint: str, params: dict, cache_hours: int):
    """One cached API-Football call (player_model.api_football's client and cache). Raw JSON, so
    a null the provider sent stays null here (get_fixture_player_stats turns it into 0)."""
    from player_model.api_football import _get
    return _get(endpoint, params, cache_hours=cache_hours)


def _af_response(endpoint: str, params: dict, cache_hours: int) -> list:
    data = _af_get(endpoint, params, cache_hours)
    return list((data or {}).get("response") or [])


def _as_id(v) -> int | None:
    """An integer provider id from a ledger cell ('1635606', '1635606.0'), or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _fixture_teams(fx: dict) -> tuple[str | None, str | None]:
    t = fx.get("teams") or {}
    return (t.get("home") or {}).get("name"), (t.get("away") or {}).get("name")


def _find_prop_fixture(league_id: int, season: str, date_str: str, home: str, away: str):
    """(fixture, reason) for a row with no stored fixture_id: that league's fixtures on that date,
    both clubs resolved in that pool. Exactly one fixture or none."""
    from src.team_names import resolve_result
    day = _af_response("/fixtures", {"league": league_id, "season": season, "date": date_str}, 2)
    named = [f for f in day if all(_fixture_teams(f))]
    if not named:
        return None, "no API-Football fixtures for the league on that date"
    rh = resolve_result(str(home), sorted({_fixture_teams(f)[0] for f in named}))
    ra = resolve_result(str(away), sorted({_fixture_teams(f)[1] for f in named}))
    if not rh or not ra:
        return None, "clubs not resolved in that date's fixtures"
    hits = [f for f in named if _fixture_teams(f) == (rh, ra)]
    if len(hits) != 1:
        return None, f"{len(hits)} fixtures match"
    return hits[0], ""


def _fixture_by_id(fixture_id: int):
    hits = _af_response("/fixtures", {"id": fixture_id}, 2)
    return hits[0] if len(hits) == 1 else None


def _fixture_players(fixture_id: int) -> dict:
    """{'teams': {team_id: [player rec]}, 'by_id': {player_id: rec}} from /fixtures/players.
    Every value the provider left null stays None."""
    teams, by_id = {}, {}
    for block in _af_response("/fixtures/players", {"fixture": fixture_id}, 168):
        tid = (block.get("team") or {}).get("id")
        recs = []
        for pe in block.get("players") or []:
            p = pe.get("player") or {}
            s = (pe.get("statistics") or [{}])[0] or {}
            games, goals = s.get("games") or {}, s.get("goals") or {}
            shots, cards = s.get("shots") or {}, s.get("cards") or {}
            yel = cards.get("yellow")
            rec = {"player_id": p.get("id"), "player_name": p.get("name"), "team_id": tid,
                   "minutes_played": games.get("minutes"), "substitute": games.get("substitute"),
                   "goals": goals.get("total"), "assists": goals.get("assists"),
                   "shots_on_target": shots.get("on"),
                   "yellow_card": None if yel is None else int(int(yel) > 0)}
            recs.append(rec)
            if rec["player_id"] is not None:
                by_id[int(rec["player_id"])] = rec
        if tid is not None:
            teams[tid] = recs
    return {"teams": teams, "by_id": by_id}


def _goal_events_complete(events: list, fixture: dict) -> bool:
    """The event feed holds every goal: scored goal events (own goals included, missed penalties
    not) add up to the final score."""
    g = fixture.get("goals") or {}
    if g.get("home") is None or g.get("away") is None:
        return False
    n = sum(1 for e in events if e.get("type") == "Goal" and e.get("detail") != "Missed Penalty")
    return n == int(g["home"]) + int(g["away"])


def _fill_null_stats(rec: dict, fixture_id: int, fixture: dict, players: dict) -> dict:
    """Replace null stats with values PROVEN from the fixture's own record; leave them None
    otherwise. Lazy: the extra endpoints are called only when a needed stat is null."""
    out = dict(rec)
    pid = int(rec["player_id"])
    if any(out.get(k) is None for k in ("goals", "assists", "yellow_card")):
        ev = _af_response("/fixtures/events", {"fixture": fixture_id}, 168)
        if _goal_events_complete(ev, fixture):
            scored = [e for e in ev if e.get("type") == "Goal"
                      and e.get("detail") not in ("Own Goal", "Missed Penalty")]
            if out.get("goals") is None:
                out["goals"] = sum(1 for e in scored if (e.get("player") or {}).get("id") == pid)
            if out.get("assists") is None:
                out["assists"] = sum(1 for e in scored if (e.get("assist") or {}).get("id") == pid)
            if out.get("yellow_card") is None:
                out["yellow_card"] = int(any(e.get("type") == "Card"
                                             and str(e.get("detail", "")).startswith("Yellow")
                                             and (e.get("player") or {}).get("id") == pid for e in ev))
    if out.get("shots_on_target") is None:
        tid = rec.get("team_id")
        total = None
        for blk in _af_response("/fixtures/statistics", {"fixture": fixture_id}, 168):
            if (blk.get("team") or {}).get("id") == tid:
                total = next((s.get("value") for s in blk.get("statistics") or []
                              if s.get("type") == "Shots on Goal"), None)
        known = [r["shots_on_target"] for r in players["teams"].get(tid, [])
                 if r.get("shots_on_target") is not None]
        # The team total is fully accounted for by the players who have a value -> the nulls are 0.
        if total is not None and int(total) == sum(int(v) for v in known):
            out["shots_on_target"] = 0
    return out


def _appearance(rec: dict | None, players: dict, fixture_id: int, pid: int) -> tuple[str, str]:
    """('PLAYED' | 'DNP' | 'UNKNOWN', note). DNP only on evidence from the match sheet."""
    if rec is None:
        # Absent from BOTH clubs' published squads for the match -> not in the matchday squad.
        if len(players["teams"]) == 2 and all(players["teams"].values()):
            return "DNP", "DNP (not in matchday squad)"
        return "UNKNOWN", "player stats published for fewer than both clubs"
    mins, sub = rec.get("minutes_played"), rec.get("substitute")
    if mins is not None and int(mins) > 0:
        return "PLAYED", ""
    if mins is not None and int(mins) == 0:
        return "DNP", "DNP (0 min)"
    if sub is True:
        # Bench with no minutes: an unused substitute, unless a substitution event names him.
        ev = _af_response("/fixtures/events", {"fixture": fixture_id}, 168)
        if not any(e.get("type") == "subst" and pid in ((e.get("player") or {}).get("id"),
                                                         (e.get("assist") or {}).get("id"))
                   for e in ev):
            return "DNP", "DNP (unused substitute)"
    return "UNKNOWN", "minutes not published"


def update_player_results(days: int = 3, dry_run: bool = False) -> None:
    """Resolve player prop outcomes in player_ledger.csv using API-Football."""
    if not PLAYER_LEDGER_FILE.exists():
        log.info("player_ledger.csv not found — skipping player prop resolution")
        return

    ledger = pd.read_csv(PLAYER_LEDGER_FILE, dtype=str)

    # Add new columns if missing (backward compat with ledgers written before this version)
    for col in ("is_played", "notes", "fixture_id"):
        if col not in ledger.columns:
            ledger[col] = ""

    today_str = str(datetime.utcnow().date())
    pending_mask = (
        ledger["result"].isna() | (ledger["result"].str.strip() == "")
    ) & (ledger["match_date"] < today_str)
    pending = ledger[pending_mask].copy()

    if pending.empty:
        log.info("player_ledger: no pending player prop results")
        return

    log.info(f"player_ledger: {len(pending)} pending tip(s) to resolve")

    import os
    if not os.getenv("APIFOOTBALL_KEY", ""):
        log.info("APIFOOTBALL_KEY not set — skipping player prop resolution")
        return

    from src.league_registry import prop_leagues
    from src.season_resolver import resolve_api_season, is_known
    prop_ids = prop_leagues()

    fixture_cache: dict = {}       # row fixture key -> (fixture | None, reason)
    players_cache: dict = {}       # fixture_id -> _fixture_players()
    updates: list[dict] = []
    fixture_ids: dict = {}         # idx -> fixture_id found for a row that had none
    skipped: dict[str, int] = {}

    def _skip(reason: str, msg: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        log.debug(f"  pending ({reason}): {msg}")

    for idx, row in pending.iterrows():
        home, away = row["home_team"], row["away_team"]
        date_str, market, player = row["match_date"], row["market"], row["player_name"]
        what = f"{player} {market} {home} vs {away} ({date_str})"

        league = _canon_league(row["league"])
        league_id = prop_ids.get(league) if league else None
        if not league_id:
            _skip("league_not_a_prop_league", f"{what}: league {row['league']!r}")
            continue

        # The fixture's OWN season from the canonical resolver (config/leagues.json season rules).
        # The hand rule here ("Jan-Jun means yr-1") is European-only: calendar-year leagues (MLS,
        # Brazil, Nordics, China, Ireland) were looked up a year behind every spring, and J1 (season
        # 2027 since 2026-07) never settled (Data Truth audit findings 23, 25).
        season = resolve_api_season(league, date_str)
        if not is_known(season):
            _skip("season_unknown", f"{what}: no season for {date_str} in the registry")
            continue

        # ── Step 1: the fixture — stored id first, else resolved in that day's pool ──
        fid = _as_id(row.get("fixture_id"))
        ck = ("id", fid) if fid else ("day", league_id, season, date_str, home, away)
        if ck not in fixture_cache:
            if fid:
                fx = _fixture_by_id(fid)
                fixture_cache[ck] = (fx, "" if fx else f"fixture {fid} not returned by API-Football")
            else:
                fixture_cache[ck] = _find_prop_fixture(league_id, season, date_str, home, away)
        fixture, why = fixture_cache[ck]
        if fixture is None:
            _skip("fixture_not_found", f"{what}: {why}")
            continue
        fid = _as_id((fixture.get("fixture") or {}).get("id"))
        if fid is None:
            _skip("fixture_not_found", f"{what}: fixture carries no id")
            continue
        if _as_id(row.get("fixture_id")) is None:
            fixture_ids[idx] = fid

        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short")
        if status in _VOID_STATUSES:
            label = _VOID_STATUSES[status]
            log.info(f"  {home} vs {away} ({date_str}) — {label}, marking VOID")
            updates.append({"idx": idx, "is_played": "False", "result": "VOID",
                            "pnl": "0.0", "notes": label, "resolved_date": today_str})
            continue
        if status not in _FT_STATUSES:
            _skip("not_finished", f"{what}: fixture {fid} status {status!r}")
            continue

        # ── Step 2: the fixture's player sheet ───────────────────────────────
        if fid not in players_cache:
            players_cache[fid] = _fixture_players(fid)
        players = players_cache[fid]

        # NO STATS FOR THE WHOLE FIXTURE IS NOT "DID NOT PLAY". An API error, an empty body or stats
        # not yet published all look like this, and until 2026-11-01 every prop on such a fixture
        # was voided as DNP for good (Data Truth audit finding 16). Leave it pending and retry; only
        # after 7 days is it voided, and then with a note that says why.
        if not players["by_id"]:
            _age = (pd.Timestamp(today_str) - pd.Timestamp(date_str)).days
            if _age >= 7:
                log.warning(f"  {home} vs {away} ({date_str}): no player stats after {_age} days — VOID (NO_STATS)")
                updates.append({"idx": idx, "is_played": "True", "result": "VOID",
                                "pnl": "0.0", "notes": "NO_STATS_7D", "resolved_date": today_str})
            else:
                _skip("no_player_stats_yet", what)
            continue

        # ── Step 3: the player, by player_id only ────────────────────────────
        pid = _as_id(row.get("player_id"))
        if pid is None:
            _skip("no_player_id", f"{what}: ledger row has no player_id - not matched by name")
            continue
        rec = players["by_id"].get(pid)

        # ── Step 4: did the player play? Only the match sheet can say no ─────
        played, note = _appearance(rec, players, fid, pid)
        if played == "DNP":
            log.info(f"  {what}: {note} (VOID)")
            updates.append({"idx": idx, "is_played": "True", "result": "VOID",
                            "pnl": "0.0", "notes": note, "resolved_date": today_str})
            continue
        if played != "PLAYED":
            _skip("appearance_unknown", f"{what}: {note}")
            continue
        minutes = int(rec["minutes_played"])

        # ── Step 5: resolve WIN/LOSS ──────────────────────────────────────────
        result = _resolve_player_market(market, rec)
        if not result:
            result = _resolve_player_market(market, _fill_null_stats(rec, fid, fixture, players))
        if not result:
            _skip("stat_unknown", f"{what}: market not settleable or stat not published")
            continue

        # AN UNPRICED ROW HAS NO P&L, BECAUSE THERE WAS NO BET. This fell back to
        # `mkt_odds = 1.0`, which prices a win at 0.0 and a loss at -1.0 — so a prop nobody
        # could have backed lost a unit and could never gain one. Invariant 13: most AVOID rows
        # are "never priced", not "rejected", and `enrich_with_odds` skips them before an edge is
        # ever computed. Measured on player_ledger since 2026-08-10: 1,158 graded losses carried
        # a pnl but only 343 of them had a price, against 175 wins that all did — 815 rows
        # charged a unit on a bet that did not exist, and the file reported AVOID at -967u.
        #
        # It never touched real P&L (AVOID is not staked, and props are paper by invariant 2),
        # but it makes the ledger useless for the one question it could answer: how the model
        # did on the props it rejected.
        #
        # The RESULT is still recorded — whether the prop hit is real information and costs
        # nothing to keep. Only the money column is left empty, per invariant 9: write NaN,
        # never an invented number.
        try:
            mkt_odds = float(row["market_odds"])
        except (TypeError, ValueError):
            mkt_odds = None
        if mkt_odds is None or not mkt_odds > 1.0:
            pnl, note = "", "unpriced — result only, no stake"
        else:
            pnl = round(mkt_odds - 1.0, 4) if result == "WIN" else -1.0
            note = ""

        updates.append({"idx": idx, "is_played": "True", "result": result,
                        "pnl": pnl, "notes": note, "resolved_date": today_str})
        # `pnl` is DELIBERATELY "" for an unpriced row (see the note above — invariant 9, write
        # nothing rather than an invented number). This log line still formatted it with
        # `:+.3f`, so the first unpriced prop raised
        #     ValueError: Unknown format code 'f' for object of type 'str'
        # and killed the whole process. Per invariant 13 most props are never priced, so it hit
        # on essentially every run: update_results failed 10 times consecutively between
        # 2026-09-06 19:26 and 2026-09-08, and because the commit step had no `if: always()` the
        # O/U settlements that had ALREADY succeeded in the same run were discarded with the
        # runner each time.
        #
        # Formatted defensively rather than by reverting the "" — the data decision was right;
        # only its rendering was wrong.
        pnl_txt = f"{pnl:+.3f}u" if isinstance(pnl, (int, float)) else "n/a (unpriced)"
        log.info(
            f"  {'[DRY]' if dry_run else '[ OK]'} "
            f"{player} ({market}, {minutes}min) {home} vs {away} ({date_str}) → {result}  "
            f"PnL={pnl_txt}"
        )

    if skipped:
        log.info(f"player_ledger: left pending this run - {dict(sorted(skipped.items()))}")

    if not updates and not fixture_ids:
        log.info("  No player prop results resolved.")
        return

    if dry_run:
        voids   = sum(1 for u in updates if u["result"] == "VOID")
        wins    = sum(1 for u in updates if u["result"] == "WIN")
        losses  = sum(1 for u in updates if u["result"] == "LOSS")
        print(f"\n  [DRY RUN] Would update {len(updates)} player prop row(s): "
              f"{wins}W / {losses}L / {voids} VOID, and write {len(fixture_ids)} fixture_id(s). "
              f"No files written.")
        return

    # Issue #1 step 1: the fixture id found for a row that had none is written back, so the next
    # run (and every reader) identifies the fixture by id instead of by club names.
    ledger["fixture_id"] = ledger["fixture_id"].astype(object)
    for i, fid in fixture_ids.items():
        ledger.at[i, "fixture_id"] = str(fid)
    for u in updates:
        ledger.at[u["idx"], "is_played"]     = u["is_played"]
        ledger.at[u["idx"], "result"]        = u["result"]
        ledger.at[u["idx"], "pnl"]           = str(u["pnl"])
        ledger.at[u["idx"], "notes"]         = u.get("notes", "")
        ledger.at[u["idx"], "resolved_date"] = u["resolved_date"]

    ledger.to_csv(PLAYER_LEDGER_FILE, index=False)

    wins      = sum(1 for u in updates if u["result"] == "WIN")
    losses    = sum(1 for u in updates if u["result"] == "LOSS")
    voids     = sum(1 for u in updates if u["result"] == "VOID")
    # Unpriced rows carry pnl "" and are excluded from the total — summing them as 0.0 would
    # quietly restate "no bet existed" as "a bet that broke even".
    staked    = [u for u in updates if u["result"] != "VOID" and str(u["pnl"]) != ""]
    unpriced  = sum(1 for u in updates if str(u["pnl"]) == "")
    total_pnl = sum(float(u["pnl"]) for u in staked)
    log.info(f"player_ledger: {len(updates)} resolved — {wins}W / {losses}L / {voids} VOID — "
             f"PnL={total_pnl:+.3f}u over {len(staked)} priced row(s)"
             + (f"; {unpriced} unpriced row(s) graded with no P&L" if unpriced else "")
             + (f"; fixture_id written on {len(fixture_ids)} row(s)" if fixture_ids else ""))
    log.info(f"  Saved → {PLAYER_LEDGER_FILE}")


# ── Sharp money result resolver ───────────────────────────────────────────────

def _resolve_sharp_side(side: str, home_score: int, away_score: int) -> str:
    """Determine WIN/LOSS for a sharp money signal given the final score."""
    total = home_score + away_score
    if side == "OVER":
        return "WIN" if total > 2.5 else "LOSS"
    if side == "UNDER":
        return "WIN" if total <= 2.5 else "LOSS"
    if side == "HOME":
        return "WIN" if home_score > away_score else "LOSS"
    if side == "AWAY":
        return "WIN" if away_score > home_score else "LOSS"
    if side == "DRAW":
        return "WIN" if home_score == away_score else "LOSS"
    return ""


def update_sharp_results(days: int = 3, dry_run: bool = False) -> None:
    """Resolve sharp money tip outcomes in sharp_ledger.csv."""
    if not SHARP_LEDGER_FILE.exists():
        log.info("sharp_ledger.csv not found — skipping sharp money resolution")
        return

    ledger = pd.read_csv(SHARP_LEDGER_FILE, dtype=str)
    today_str = str(datetime.utcnow().date())
    pending_mask = (
        ledger["result"].isna() | (ledger["result"].str.strip() == "")
    ) & (ledger["match_date"] < today_str)
    pending = ledger[pending_mask].copy()

    if pending.empty:
        log.info("sharp_ledger: no pending sharp money results")
        return

    log.info(f"sharp_ledger: {len(pending)} pending tip(s) to resolve")

    scores_cache: dict[str, list[dict]] = {}
    updates: list[dict] = []

    for idx, row in pending.iterrows():
        league   = row["league"]
        home     = row["home_team"]
        away     = row["away_team"]
        date_str = row["match_date"]
        side     = row["side"]

        if not side:
            log.debug(f"  sharp: no side parsed for {row.get('market_label', '')} — skipping")
            continue

        if league not in scores_cache:
            if league in _FD_SOURCES:
                log.info(f"  Fetching scores for {league} from football-data.co.uk ...")
                scores_cache[league] = fetch_scores_fd(league)
            else:
                sk = config.ODDS_API_SPORT_KEYS.get(league)
                if sk:
                    log.info(f"  Fetching scores for {league} ({sk}) last {days} days ...")
                    scores_cache[league] = fetch_scores(sk, days)
                    log.info(f"    → {len(scores_cache[league])} completed events")
                else:
                    scores_cache[league] = []

        ev = _match_event(home, away, scores_cache.get(league, []), {date_str})
        match_result = _resolve_sharp_side(side, ev["home_score"], ev["away_score"]) if ev else None

        if not match_result:
            log.debug(f"  sharp: {home} vs {away} {date_str} not found in scores")
            continue

        # No usable entry price -> the RESULT is graded, the P/L is not. This defaulted to odds 1.0,
        # so a WIN was booked at 0.0 with no flag (Data Truth audit finding 27).
        try:
            opening_odds = float(row["opening_odds"])
        except (TypeError, ValueError):
            opening_odds = np.nan
        if not opening_odds > 1.0:
            pnl = np.nan
            log.warning(f"  sharp: {home} vs {away} {date_str} has no entry price — result kept, P/L left empty")
        else:
            pnl = round(opening_odds - 1.0, 4) if match_result == "WIN" else -1.0

        updates.append({"idx": idx, "result": match_result, "pnl": pnl})
        log.info(
            f"  {'[DRY]' if dry_run else '[ OK]'} "
            f"{home} vs {away} ({side}) {date_str} → {match_result}  PnL={pnl:+.3f}u"
        )

    if not updates:
        log.info("  No sharp money results found in any scores source.")
        return

    if dry_run:
        print(f"\n  [DRY RUN] Would update {len(updates)} sharp row(s). No files written.")
        return

    resolved_today = str(datetime.utcnow().date())
    for u in updates:
        ledger.at[u["idx"], "result"]        = u["result"]
        # An unpriced row's P/L stays EMPTY, not the text 'nan'.
        ledger.at[u["idx"], "pnl"]           = "" if np.isnan(u["pnl"]) else str(u["pnl"])
        ledger.at[u["idx"], "resolved_date"] = resolved_today

    ledger.to_csv(SHARP_LEDGER_FILE, index=False)

    wins      = sum(1 for u in updates if u["result"] == "WIN")
    losses    = sum(1 for u in updates if u["result"] == "LOSS")
    priced    = [u["pnl"] for u in updates if not np.isnan(u["pnl"])]
    log.info(f"sharp_ledger: {len(updates)} resolved — {wins}W/{losses}L — "
             f"PnL={sum(priced):+.3f}u over {len(priced)} priced row(s)")
    log.info(f"  Saved → {SHARP_LEDGER_FILE}")


# ── Side-market result resolver (BTTS / Over 1.5 / Over 3.5) ────────────────────

def _resolve_side_market(market: str, home_score: int, away_score: int) -> str:
    """WIN/LOSS for a side market given the final score."""
    total = home_score + away_score
    if market == "btts":
        return "WIN" if (home_score > 0 and away_score > 0) else "LOSS"
    if market == "over15":
        return "WIN" if total > 1.5 else "LOSS"
    if market == "over35":
        return "WIN" if total > 3.5 else "LOSS"
    return ""


def update_side_market_results(days: int = 3, dry_run: bool = False) -> None:
    """Resolve BTTS / Over 1.5 / Over 3.5 outcomes in side_bets_ledger.csv.

    Reuses the same score sources as the O/U ledger (football-data first, OddsAPI fallback).
    Fills result + pnl (flat 1u) AND closing_odds + clv_pct.

    CLV used to be left blank here because side-market closing lines weren't captured. They
    are now: std_odds_capture.yml writes standard BTTS/O1.5/O3.5 into
    standard_sidemarket_odds_history.csv 3x/day, and new-format side prices land in
    newformat_odds_history.csv — so the archives can be read the same way the O/U 2.5 path
    reads them (2026-08-15).
    """
    if not SIDE_LEDGER_FILE.exists():
        log.info("side_bets_ledger.csv not found — skipping side-market resolution")
        return

    ledger = pd.read_csv(SIDE_LEDGER_FILE, dtype=str)
    for col in ("result", "pnl", "closing_odds", "clv_pct"):
        if col not in ledger.columns:
            ledger[col] = ""

    today_str = str(datetime.utcnow().date())
    pending_mask = (
        ledger["result"].isna() | (ledger["result"].str.strip() == "")
    ) & (ledger["match_date"] < today_str)
    pending = ledger[pending_mask].copy()

    if pending.empty:
        log.info("side_bets_ledger: no pending side-market results")
        return

    log.info(f"side_bets_ledger: {len(pending)} pending tip(s) to resolve")

    scores_cache: dict[str, list[dict]] = {}
    updates: list[dict] = []

    for idx, row in pending.iterrows():
        league   = row["league"]
        home     = row["home_team"]
        away     = row["away_team"]
        date_str = row["match_date"]
        market   = str(row.get("market", ""))

        if league not in scores_cache:
            if league in _FD_SOURCES:
                log.info(f"  Fetching scores for {league} from football-data.co.uk ...")
                scores_cache[league] = fetch_scores_fd(league)
            else:
                sk = config.ODDS_API_SPORT_KEYS.get(league)
                scores_cache[league] = fetch_scores(sk, days) if sk else []

        # LAST RESORT: API-Football, exactly as the O/U 2.5 path does. Without it a side tip whose
        # fixture the primary source missed (football-data lag or spelling, OddsAPI's 3-day window)
        # stayed pending forever while the O/U tip on the same match settled: 38 BTTS/O1.5/O3.5
        # rows since 2026-10-10 (found 2026-10-25). Lazy, so a healthy run costs no calls.
        ev_found = (_match_event(home, away, scores_cache.get(league, []), {date_str})
                    or _match_event(home, away, fetch_scores_apifootball(league, date_str), {date_str}))
        if ev_found is None:
            log.debug(f"  side {market}: {home} vs {away} {date_str} not found in any scores source")
            continue

        res = _resolve_side_market(market, ev_found["home_score"], ev_found["away_score"])
        if not res:
            continue
        # No usable entry price -> result graded, P/L left empty (was odds 1.0: a WIN booked at 0.0
        # with no flag, Data Truth audit finding 27).
        try:
            odds = float(row["odds"])
        except (TypeError, ValueError):
            odds = np.nan
        if not odds > 1.0:
            pnl = np.nan
            log.warning(f"  side {market}: {home} vs {away} {date_str} has no entry price — P/L left empty")
        else:
            pnl = round(odds - 1.0, 4) if res == "WIN" else -1.0

        # Closing line -> CLV, same percent convention as bets_ledger/ht_ledger.
        # A missing closing price is normal (thin 2nd-div side markets); grade the result
        # anyway and leave CLV blank rather than dropping the row.
        close = _closing_for_market(home, away, date_str,
                                    _SIDE_ARCHIVE_MARKET.get(market, market), league=league)
        clv = np.nan
        if not np.isnan(close) and close > 0 and odds > 1.0:
            clv = round((odds - close) / close * 100.0, 2)

        updates.append({"idx": idx, "result": res, "pnl": pnl,
                        "closing_odds": close, "clv_pct": clv})
        clv_str = f"  CLV={clv:+.1f}%" if not np.isnan(clv) else ""
        log.info(
            f"  {'[DRY]' if dry_run else '[ OK]'} "
            f"{home} vs {away} ({market}) {date_str} → {res}  PnL={pnl:+.3f}u{clv_str}"
        )

    if not updates:
        log.info("  No side-market results found in any scores source.")
        return
    if dry_run:
        print(f"\n  [DRY RUN] Would update {len(updates)} side-market row(s). No files written.")
        return

    for u in updates:
        ledger.at[u["idx"], "result"] = u["result"]
        # An unpriced row's P/L stays EMPTY, not the text 'nan'.
        ledger.at[u["idx"], "pnl"]    = "" if np.isnan(u["pnl"]) else str(u["pnl"])
        if not np.isnan(u["closing_odds"]):
            ledger.at[u["idx"], "closing_odds"] = str(u["closing_odds"])
        if not np.isnan(u["clv_pct"]):
            ledger.at[u["idx"], "clv_pct"] = str(u["clv_pct"])
    ledger.to_csv(SIDE_LEDGER_FILE, index=False)

    wins   = sum(1 for u in updates if u["result"] == "WIN")
    losses = sum(1 for u in updates if u["result"] == "LOSS")
    priced = [u["pnl"] for u in updates if not np.isnan(u["pnl"])]
    log.info(f"side_bets_ledger: {len(updates)} resolved — {wins}W/{losses}L — "
             f"PnL={sum(priced):+.3f}u over {len(priced)} priced row(s)")
    log.info(f"  Saved → {SIDE_LEDGER_FILE}")


def _resolve_ht(market: str, ht_home: int, ht_away: int) -> str:
    """WIN/LOSS for a half-time O/U market given the HT score; '' when either half is unknown
    (a missing half was read as 0 goals)."""
    if ht_home is None or ht_away is None or pd.isna(ht_home) or pd.isna(ht_away):
        return ""
    tot = int(ht_home) + int(ht_away)
    line = 1 if "05" in market else 2          # over0.5 needs >=1 ; over1.5 needs >=2
    over_hit = tot >= line
    if market.startswith("ht_over"):
        return "WIN" if over_hit else "LOSS"
    if market.startswith("ht_under"):
        return "WIN" if not over_hit else "LOSS"
    return ""


def _group_by_day_league(df: pd.DataFrame, day_col: str) -> dict:
    """{(day, canonical league): rows}. Leagues go through the registry on both sides of every
    lookup, so an alias spelling can neither miss nor cross a competition; rows whose league the
    registry does not know are left out."""
    canon = {lg: _canon_league(lg) for lg in df["league"].dropna().unique()}
    d = df.assign(_lk=df["league"].map(canon)).dropna(subset=["_lk"])
    return {k: g for k, g in d.groupby([day_col, "_lk"], sort=False)}


def update_ht_observations() -> None:
    """Fill half-time scores into output/ht_observations.csv. Idempotent.

    The observation log is the forward evidence base for the half-time question: model belief +
    market price at scoring time, and what actually happened. It reuses the same score sources
    and the same league-scoped name resolution as update_ht_results, because the two must never
    disagree about what a fixture's half-time score was.
    """
    f = config.OUTPUT_DIR / "ht_observations.csv"
    if not f.exists():
        return
    obs = pd.read_csv(f)
    if obs.empty:
        return
    for c in ("ht_home_goals", "ht_away_goals", "ht_total_goals", "ht_over05", "ht_over15"):
        if c not in obs.columns:
            obs[c] = ""
        obs[c] = obs[c].astype(object)
    today = str(datetime.utcnow().date())
    pending = obs[(obs["ht_total_goals"].isna()
                   | (obs["ht_total_goals"].astype(str).str.strip() == ""))
                  & (obs["match_date"].astype(str) < today)]
    if pending.empty:
        log.info("ht_observations: nothing pending")
        return

    from src.team_names import resolve as _resolve_team
    frames = []
    for path, ren in ((config.OUTPUT_DIR / "fd_history.parquet",
                       {"ht_home_goals": "hh", "ht_away_goals": "ha"}),
                      (config.OUTPUT_DIR / "af_ht_history.parquet",
                       {"HTHG": "hh", "HTAG": "ha"})):
        if not path.exists():
            continue
        try:
            d = pd.read_parquet(path, columns=["league", "date", "home_team", "away_team",
                                               *ren.keys()])
            frames.append(d.rename(columns=ren))
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"ht_observations: could not read {path.name} ({e})")
    if not frames:
        log.info("ht_observations: no score source available")
        return
    ht = pd.concat(frames, ignore_index=True).dropna(subset=["hh", "ha"])
    ht["_d"] = ht["date"].astype(str).str[:10]
    by = _group_by_day_league(ht, "_d")

    filled = 0
    for i in pending.index:
        g = by.get((str(obs.at[i, "match_date"])[:10], _canon_league(obs.at[i, "league"])))
        if g is None or g.empty:
            continue
        h = _resolve_team(str(obs.at[i, "home_team"]), list(g["home_team"]))
        a = _resolve_team(str(obs.at[i, "away_team"]), list(g["away_team"]))
        if not h or not a:
            continue
        m = g[(g["home_team"] == h) & (g["away_team"] == a)]
        if len(m) != 1:
            continue          # ambiguous -> leave it, a wrong score is worse than a late one
        hh, ha = float(m.iloc[0]["hh"]), float(m.iloc[0]["ha"])
        obs.at[i, "ht_home_goals"] = hh
        obs.at[i, "ht_away_goals"] = ha
        obs.at[i, "ht_total_goals"] = hh + ha
        obs.at[i, "ht_over05"] = 1.0 if (hh + ha) >= 1 else 0.0
        obs.at[i, "ht_over15"] = 1.0 if (hh + ha) >= 2 else 0.0
        filled += 1

    if filled:
        obs.to_csv(f, index=False)
    log.info(f"ht_observations: filled {filled} of {len(pending)} pending -> {f.name}")


def update_ht_results(days: int = 3) -> None:
    """Grade ht_ledger.csv against actual HT scores and fill
    entry/closing/CLV from the captured HT odds (standard_sidemarket_odds_history ht_* markets).
    Completes the open->close->CLV->result loop for the HT model. HT closing odds may be absent
    (books don't always post 2nd-div HT O/U) -> CLV stays blank there, result still grades."""
    f = config.OUTPUT_DIR / "ht_ledger.csv"
    if not f.exists():
        log.info("ht_ledger.csv not found — skipping HT resolution"); return
    led = pd.read_csv(f)
    # `notes` belongs in this list too. An all-empty column reads back as float64, so writing a
    # string into it raises TypeError and the whole grading run dies after doing its work but
    # before saving -- which looks exactly like "nothing to grade".
    for c in ("entry_odds", "closing_odds", "clv_pct", "result", "pnl", "notes"):
        if c not in led.columns:
            led[c] = ""
        led[c] = led[c].astype(object)           # avoid pyarrow-string write crash
    today = str(datetime.utcnow().date())
    pending = led[(led["result"].isna() | (led["result"].astype(str).str.strip() == ""))
                  & (led["match_date"].astype(str) < today)]
    if pending.empty:
        log.info("ht_ledger: no pending HT results"); return

    # ── HT SCORES COME FROM BOTH STORES, AND THE STANDARD ONE IS THE IMPORTANT ONE ──────
    # This used to read af_ht_history.parquet ALONE and match on a bare alphanumeric key. Both
    # halves of that were wrong, and together they meant the HT model was NEVER GRADED ONCE:
    #
    #   * af_ht_history holds only NEW-FORMAT leagues — MLS, Argentina, Brazil, Japan, K-League
    #     and so on. HT tips are STANDARD-only (League One, League Two, La Liga 2). The league
    #     overlap between the two sets is EMPTY, so no row could ever match. It was also stale
    #     from 2026-06-23, but that was the lesser problem: fixing the date would have fixed
    #     nothing.
    #   * fd_history.parquet has carried `ht_home_goals` / `ht_away_goals` all along — 21,840
    #     rows, current to yesterday, covering exactly Championship / League One / League Two /
    #     La Liga 2. The data was sitting there the whole time.
    #   * And a bare `_norm` key cannot join these sources anyway, because club names differ
    #     between them (invariant 11): the ledger says `Plymouth Argyle`, `Wycombe Wanderers`,
    #     `Accrington Stanley`; football-data says `Plymouth`, `Wycombe`, `Accrington`. Swapping
    #     the source alone lifted matching only 0 -> 2 of 47. Resolving names LEAGUE-SCOPED with
    #     team_names.resolve lifts it to 38 of 47; the remaining 9 are fixtures football-data has
    #     not published yet and they grade on the next run.
    #
    # Both stores are read, so new-format HT coverage keeps working if it is ever tipped.
    from src.team_names import resolve as _resolve_team

    _frames = []
    _fdp = config.OUTPUT_DIR / "fd_history.parquet"
    if _fdp.exists():
        try:
            _fd = pd.read_parquet(_fdp, columns=["league", "date", "home_team", "away_team",
                                                 "ht_home_goals", "ht_away_goals"])
            _frames.append(_fd.rename(columns={"ht_home_goals": "hh", "ht_away_goals": "ha"}))
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"ht_ledger: could not read fd_history.parquet ({e})")
    _afp = config.OUTPUT_DIR / "af_ht_history.parquet"
    if _afp.exists():
        try:
            _af = pd.read_parquet(_afp, columns=["league", "date", "home_team", "away_team",
                                                 "HTHG", "HTAG"])
            _frames.append(_af.rename(columns={"HTHG": "hh", "HTAG": "ha"}))
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"ht_ledger: could not read af_ht_history.parquet ({e})")
    if not _frames:
        log.info("ht_ledger: no HT score source available — cannot grade yet"); return

    ht = pd.concat(_frames, ignore_index=True).dropna(subset=["hh", "ha"])
    ht["_d"] = ht["date"].astype(str).str[:10]
    # Group once, so the per-row lookup is a dict hit and not a scan of 21,000 rows.
    _by_day = _group_by_day_league(ht, "_d")

    def _ht_score(league: str, home: str, away: str, day: str):
        """HT goals for one fixture, or None. League-scoped so a name can't cross competitions."""
        g = _by_day.get((day, _canon_league(league)))
        if g is not None and not g.empty:
            h = _resolve_team(home, list(g["home_team"]))
            a = _resolve_team(away, list(g["away_team"]))
            if h and a:
                m = g[(g["home_team"] == h) & (g["away_team"] == a)]
                # Exactly one, or nothing. An ambiguous match is not graded — a wrong settlement
                # is far worse than a late one.
                if len(m) == 1:
                    return (m.iloc[0]["hh"], m.iloc[0]["ha"])
        # LAST RESORT: API-Football's completed fixtures for THIS match's season carry HT scores.
        # Without it a fixture missing from both committed stores stayed pending for ever
        # (Data Truth audit finding 24). Same rules: names resolved in that day's whole pool (a
        # one-name pool let any token-subset pair through), exactly one match.
        ev = _match_event(home, away, fetch_scores_apifootball(str(league), day), {day})
        if ev is None or ev.get("ht_home") is None or ev.get("ht_away") is None:
            return None
        return (ev["ht_home"], ev["ht_away"])

    # ── THE ODDS JOIN NEEDS THE SAME NAME RESOLUTION, for the same reason ───────────────
    # The captured prices are real and plentiful — 4,478 ht_* rows, all four lines, 250 on a
    # single match day — but they are written from the OddsAPI event name while the ledger
    # carries the predict-side name, and the two disagree exactly as invariant 11 says they
    # will: `Accrington ST` against `Accrington Stanley`, `Cheltenham` against `Cheltenham
    # Town`. Matching those with `_norm` found entry odds for 4 of 38 graded tips, so almost
    # every settlement fell back to the model's OWN fair odds — which makes the P/L circular
    # and close to meaningless. Resolve league-scoped, like the scores above.
    ohp = config.OUTPUT_DIR / "standard_sidemarket_odds_history.csv"
    oh_by_day: dict = {}
    if ohp.exists():
        try:
            oh = pd.read_csv(ohp).sort_values("snapshot_ts")
            oh = oh[oh["market"].astype(str).str.startswith("ht_")]
            mm = oh["match"].astype(str)
            oh["_h"] = mm.apply(lambda m: m.split(" vs ")[0].strip() if " vs " in m else "")
            oh["_a"] = mm.apply(lambda m: m.split(" vs ")[1].strip() if " vs " in m else "")
            oh["_dk"] = oh["match_date"].astype(str).str[:10]
            oh_by_day = _group_by_day_league(oh, "_dk")
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"ht_ledger: could not read side-market odds ({e})")
            oh_by_day = {}

    def _ht_odds(league: str, home: str, away: str, day: str, market: str):
        """(entry, close) for one fixture+line, or (None, None). Opening and closing price."""
        g = oh_by_day.get((day, _canon_league(league)))
        if g is None or g.empty:
            return None, None
        h = _resolve_team(home, list(g["_h"]))
        a = _resolve_team(away, list(g["_a"]))
        if not h or not a:
            return None, None
        m = g[(g["_h"] == h) & (g["_a"] == a) & (g["market"].astype(str) == market)]
        if m.empty:
            return None, None
        try:
            # Already sorted by snapshot_ts: first row is the open, last is the close.
            return float(m.iloc[0]["odds"]), float(m.iloc[-1]["odds"])
        except (TypeError, ValueError):
            return None, None

    graded = 0
    for i in pending.index:
        home = led.at[i, "home_team"]; away = led.at[i, "away_team"]
        d = str(led.at[i, "match_date"])[:10]; mkt = str(led.at[i, "market"])
        sc = _ht_score(led.at[i, "league"], home, away, d)
        if sc is None:
            continue
        res = _resolve_ht(mkt, sc[0], sc[1])
        if not res:
            continue
        entry, close = _ht_odds(led.at[i, "league"], home, away, d, mkt)
        # The price logged WITH the tip is the entry; the capture's first snapshot only when the
        # row has none. The archive open used to overwrite the logged price.
        try:
            logged = float(led.at[i, "entry_odds"])
        except (TypeError, ValueError):
            logged = np.nan
        if logged > 1.0:
            entry = logged
        led.at[i, "result"] = res
        # SETTLE AT ENTRY OR NOT AT ALL. This fell back to the closing price and then to the
        # model's own fair odds (1/p): a P/L with no market price behind it, which only restates
        # the model's confidence. No entry price -> result graded, P/L left empty, flagged.
        if entry and entry > 1.0:
            led.at[i, "pnl"] = round(entry - 1.0, 3) if res == "WIN" else -1.0
        else:
            led.at[i, "pnl"] = np.nan
            led.at[i, "notes"] = "no_entry_price_result_only"
            log.info(f"  ht {mkt}: {home} vs {away} {d} has no entry price — result kept, P/L left empty")
        if entry:
            led.at[i, "entry_odds"] = entry
        if close:
            led.at[i, "closing_odds"] = close
        if entry and close and close > 1.0:
            led.at[i, "clv_pct"] = round((entry / close - 1.0) * 100.0, 2)   # PERCENT, like bets_ledger
        graded += 1

    if graded:
        led.to_csv(f, index=False)
    log.info(f"ht_ledger: graded {graded} HT tip(s) → {f}")


# ── Live in-play signal grader ────────────────────────────────────────────────
# The Live Center "History" tab shows the scanner's in-play signals and whether they'd have
# won. Nothing was writing those results back — the page tried to settle on-the-fly against the
# match DB, which has no current-season new-format scores (af_history.parquet stops 2025). This
# grader fills result/final_total/ht_total into output/live_signals_history.csv. ISOLATION:
# touches only that file + its own score fetches; no model, no other ledger.

def _live_side_from_bet(bet: str):
    """Map a live-signal bet string -> settlement side (OVER/UNDER/HT_*), or None."""
    b = str(bet).upper()
    if "HT" in b:
        if "OVER 0.5"  in b: return "HT_OVER_0.5"
        if "UNDER 0.5" in b: return "HT_UNDER_0.5"
        if "OVER 1.5"  in b: return "HT_OVER_1.5"
        if "UNDER 1.5" in b: return "HT_UNDER_1.5"
        return None
    if "OVER 2.5"  in b: return "OVER"
    if "UNDER 2.5" in b: return "UNDER"
    return None


def _settle_live(side, ft_total, ht_total):
    """WIN/LOSS for a live signal given final + half-time totals (None if not gradable yet)."""
    if side is None:
        return None
    if side.startswith("HT_"):
        if ht_total is None or pd.isna(ht_total):
            return None
        if side == "HT_OVER_0.5":  return "WIN" if ht_total >= 1 else "LOSS"
        if side == "HT_UNDER_0.5": return "WIN" if ht_total < 1  else "LOSS"
        if side == "HT_OVER_1.5":  return "WIN" if ht_total >= 2 else "LOSS"
        if side == "HT_UNDER_1.5": return "WIN" if ht_total <= 1 else "LOSS"
        return None
    if ft_total is None or pd.isna(ft_total):
        return None
    if side == "OVER":  return "WIN" if ft_total > 2.5  else "LOSS"
    if side == "UNDER": return "WIN" if ft_total <= 2.5 else "LOSS"
    return None


def _date_variants(d: str) -> list:
    """[d, d-1, d+1] — kickoffs can land a day either side of the signal date in UTC."""
    try:
        base = datetime.strptime(str(d)[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return [str(d)[:10]]
    from datetime import timedelta
    return [(base + timedelta(days=off)).strftime("%Y-%m-%d") for off in (0, -1, 1)]


def update_live_signal_results(days: int = 3, dry_run: bool = False) -> None:
    """Grade output/live_signals_history.csv so the Live Center History tab shows win/loss.
    Sources, cheap-first: football-data.co.uk (free, FT + HT for standard) and af_ht_history
    (free HT) -> API-Football per fixture for whatever's still missing (the only current source
    for Argentina/Brazil FT+HT). Writes final_total/ht_total/result. Idempotent: only rows
    without a WIN/LOSS and with a past date are (re)graded."""
    f = config.OUTPUT_DIR / "live_signals_history.csv"
    if not f.exists():
        log.info("live_signals_history.csv not found — skipping live-signal resolution"); return
    hist = pd.read_csv(f)
    if hist.empty or "match" not in hist.columns:
        log.info("live_signals: nothing to grade"); return
    for c in ("final_total", "ht_total", "result"):
        if c not in hist.columns:
            hist[c] = ""
        hist[c] = hist[c].astype(object)          # avoid pyarrow-string write crash

    today_str = str(datetime.utcnow().date())
    def _pending(r):
        res = str(r.get("result", "")).strip().upper()
        return res not in ("WIN", "LOSS") and str(r.get("date", ""))[:10] < today_str
    pending_idx = [i for i in hist.index if _pending(hist.loc[i])]
    if not pending_idx:
        log.info("live_signals: no pending results"); return
    log.info(f"live_signals: {len(pending_idx)} pending signal(s) to resolve")

    # free source 1: football-data.co.uk (per league, cached in-memory this run)
    fd_cache: dict = {}
    def _fd_completed(league):
        if league not in _FD_SOURCES:
            return []
        if league not in fd_cache:
            fd_cache[league] = fetch_scores_fd(league)
        return fd_cache[league]

    # free source 2: af_ht_history.parquet (half-time totals, mostly older new-format matches),
    # grouped by (date, canonical league) and resolved in that pool. It was keyed by lowercased
    # names + date with no league, so a same-named club in another competition could grade it.
    ht_by_day: dict = {}
    htp = config.OUTPUT_DIR / "af_ht_history.parquet"
    if htp.exists():
        try:
            _ht = pd.read_parquet(htp, columns=["league", "home_team", "away_team", "date", "HTHG", "HTAG"])
            _ht = _ht.dropna(subset=["HTHG", "HTAG", "home_team", "away_team"])
            _ht = _ht.assign(date_str=_ht["date"].astype(str).str[:10])
            ht_by_day = _group_by_day_league(_ht, "date_str")
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"live_signals: HT parquet read failed ({e}) - not used")

    def _ht_events(league, dd):
        g = ht_by_day.get((dd, _canon_league(league)))
        if g is None:
            return []
        return [{"home_team": r.home_team, "away_team": r.away_team, "date_str": dd,
                 "home_score": int(r.HTHG), "away_score": int(r.HTAG)} for r in g.itertuples(index=False)]

    graded = 0
    for i in pending_idx:
        row = hist.loc[i]
        m = str(row.get("match", ""))
        if " vs " not in m:
            continue
        home, away = [s.strip() for s in m.split(" vs ", 1)]
        d = str(row.get("date", ""))[:10]
        league = str(row.get("league", ""))
        side = _live_side_from_bet(row.get("bet", ""))
        if side is None:
            continue
        # Signal date first, then a day either side (UTC kickoffs); each date is its own pool and
        # only the first date with exactly one fixture counts.
        variants = _date_variants(d)

        def _first(events):
            for dd in variants:
                ev = _match_event(home, away, events, {dd})
                if ev is not None:
                    return ev
            return None

        ft_total = ht_total = None
        # 1) football-data (free) — FT (+HT for standard leagues)
        ev = _first(_fd_completed(league))
        if ev is not None:
            ft_total = ev["home_score"] + ev["away_score"]
            if ev.get("ht_home") is not None and ev.get("ht_away") is not None:
                ht_total = ev["ht_home"] + ev["ht_away"]
        # 2) HT parquet (free) — new-format half-time fallback
        if ht_total is None:
            evh = next((e for e in (_match_event(home, away, _ht_events(league, dd), {dd})
                                    for dd in variants) if e is not None), None)
            if evh is not None:
                ht_total = evh["home_score"] + evh["away_score"]
        # 3) API-Football — the only current source for Argentina/Brazil FT+HT. The league's
        #    completed fixtures for the match's season, resolved in the date pool (this was a
        #    per-fixture call matching names pairwise).
        need_ft = ft_total is None and not side.startswith("HT_")
        need_ht = ht_total is None and side.startswith("HT_")
        if need_ft or need_ht:
            eva = _first(fetch_scores_apifootball(league, d))
            if eva is not None:
                if ft_total is None:
                    ft_total = eva["home_score"] + eva["away_score"]
                if ht_total is None and eva.get("ht_home") is not None and eva.get("ht_away") is not None:
                    ht_total = eva["ht_home"] + eva["ht_away"]

        res = _settle_live(side, ft_total, ht_total)
        if res is None:
            continue
        if ft_total is not None:
            hist.at[i, "final_total"] = int(ft_total)
        if ht_total is not None:
            hist.at[i, "ht_total"] = int(ht_total)
        hist.at[i, "result"] = res
        graded += 1
        log.info(f"  {'[DRY]' if dry_run else '[ OK]'} {home} vs {away} ({d}) "
                 f"{row.get('bet')}  FT={ft_total} HT={ht_total}  → {res}")

    if graded and not dry_run:
        hist.to_csv(f, index=False)
    log.info(f"live_signals: graded {graded} signal(s) → {f}")


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Prospective shadow log settlement (P0 2026-10-20) ────────────────────────────────────────
#
# output/v91_shadow.csv had 641 frozen predictions and ZERO settled rows: nothing in the estate
# ever wrote its result/closing_odds/clv columns, and its rows carried only an OddsAPI event id,
# so no results source could be joined to them. The grading logic lives in
# src/shadow_settlement.py (pure, tested); this function only supplies it with the SAME result
# hierarchy and odds archives the ledgers above use.

SHADOW_FILE  = config.OUTPUT_DIR / "v91_shadow.csv"
SHADOW_HEALTH_FILE = config.OUTPUT_DIR / "shadow_settlement_health.json"


def _read_shadow(path: Path) -> tuple[pd.DataFrame, str]:
    """dtype=str + no NA parsing, so every untouched cell is written back byte-for-byte."""
    raw = path.read_bytes()
    eol = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    return pd.read_csv(path, dtype=str, keep_default_na=False), eol


def _write_shadow(df: pd.DataFrame, path: Path, eol: str) -> None:
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, lineterminator=eol)
    tmp.replace(path)


def _shadow_identity() -> dict:
    """fixture_id -> (home, away) for rows written before the identity columns existed.

    Two sources, both already in the repo: the current predictions.csv (its `_oa_event_id` is
    exactly what `_log_shadow` stored as fixture_id), and the natural-key fallback
    `date|home|away` that `_log_shadow` uses when a fixture had no OddsAPI event. Anything else
    is identified by the OddsAPI scores feed's own event id inside the settlement.
    """
    out: dict = {}
    p = config.OUTPUT_DIR / "predictions.csv"
    if p.exists():
        try:
            pr = pd.read_csv(p, dtype=str, usecols=lambda c: c in (
                "_oa_event_id", "home_team", "away_team"))
            for fid, h, a in zip(pr.get("_oa_event_id", []), pr["home_team"], pr["away_team"]):
                if isinstance(fid, str) and fid.strip():
                    out.setdefault(fid.strip(), (h, a))
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"shadow identity: predictions.csv unreadable ({e})")
    return out


def _write_shadow_health(df: pd.DataFrame, pending: dict | None = None,
                         archives: list | None = None, now=None) -> dict:
    from src.shadow_settlement import health
    h = health(df, pending=pending)
    SHADOW_HEALTH_FILE.write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
    _write_clv_collection_health(df, archives, now)
    return h


def _archives_with_heartbeats() -> list:
    """The ledgers' archives with each capture's last-seen heartbeat folded into its change-log
    (src/odds_heartbeat.py). A heartbeat row is an ordinary pre-kickoff observation, so the close
    becomes the latest time the capture actually SAW the price, not the time it last moved."""
    from src import odds_heartbeat
    archives = list(zip(_ARCHIVE_FILES, _load_archives_named()))
    try:
        return odds_heartbeat.merge_into_archives(archives, odds_heartbeat.load())
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"shadow: heartbeat merge skipped ({e})")
        return archives


def _write_clv_collection_health(df: pd.DataFrame, archives: list | None = None,
                                 now=None) -> None:
    """output/clv_collection_health.json - certified-close coverage. Never breaks settlement."""
    try:
        from src import clv_collection_health as cch
        h = cch.build(df, archives if archives is not None else _archives_with_heartbeats(),
                      now=now)
        # Next to the shadow health file (so tests that redirect one redirect both).
        cch.write(h, SHADOW_HEALTH_FILE.parent / "clv_collection_health.json")
        log.info(f"clv health: {h['check']['status']} {h['check']['reasons']}")
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"clv health skipped ({e})")


def settle_shadow_predictions(days: int = 3, dry_run: bool = False,
                              journal: str | None = None,
                              providers: list | None = None,
                              archives: list | None = None,
                              now=None) -> dict:
    """Settle finished, unsettled, genuinely-prospective rows of output/v91_shadow.csv.

    Result hierarchy = the ledgers': football-data (where configured) -> OddsAPI scores (3-day
    window, only for rows inside it, so a stranded row cannot burn OddsAPI credits every run)
    -> API-Football. Closing price = last archived price strictly before kickoff, from the
    ledgers' own odds archives. Grading-only columns are written; first-sight columns are hashed
    and the write is refused if they would change.

    `journal`: also write the updates as JSON there (the workflow re-applies them onto the
    freshest origin copy of the file, so it never has to resolve a conflict with predict).
    `providers` / `archives` / `now` exist for tests.
    """
    from src.shadow_settlement import apply_updates, settle_frame

    if not SHADOW_FILE.exists():
        log.info("shadow: v91_shadow.csv not found — nothing to settle")
        return {"settled": 0}
    df, eol = _read_shadow(SHADOW_FILE)

    if providers is None:
        def _fd(league):
            return fetch_scores_fd(league) if league in _FD_SOURCES else []

        def _oa(league):
            sk = config.ODDS_API_SPORT_KEYS.get(league)
            return fetch_scores(sk, days) if (sk and config.ODDS_API_KEY) else []

        providers = [("football-data", _fd, None),
                     ("oddsapi", _oa, min(days, 3)),
                     ("api-football", fetch_scores_apifootball, None)]
    if archives is None:
        archives = _archives_with_heartbeats()

    # No fallback_matcher: the pairwise _names_match let any token-subset pair through once the
    # pool-scoped resolver had (rightly) refused. An unresolved fixture stays pending, counted.
    updates, pending = settle_frame(df, providers=providers, archives=archives,
                                    identity=_shadow_identity(), now=now)
    log.info(f"shadow: {len(updates)} row(s) settleable now; pending {pending}")
    if dry_run:
        return {"settled": 0, "would_settle": len(updates), "pending": pending}
    if journal:
        Path(journal).write_text(json.dumps(updates), encoding="utf-8")
    if updates:
        out, n = apply_updates(df, updates)
    else:
        out, n = df, 0
    from src.shadow_settlement import enforce_close_rule
    out, n_stale = enforce_close_rule(out)
    if n or n_stale:
        _write_shadow(out, SHADOW_FILE, eol)
        log.info(f"shadow: settled {n} row(s), {n_stale} close(s) >30 min re-labelled -> {SHADOW_FILE.name}")
    h = _write_shadow_health(out, pending, archives, now)
    log.info(f"shadow: {h.get('settled_rows')}/{h.get('shadow_rows')} settled · close "
             f"{h.get('valid_closing_price_pct')}% · CLV {h.get('clv_pct')}%")
    return {"settled": n, "pending": pending, "health": h}


def reapply_shadow_settlement(journal: str) -> int:
    """Re-apply a journal onto whatever v91_shadow.csv is on disk now (offline, idempotent).

    Used by the workflow after it resets to origin/main: predict may have appended rows since
    this run started, and re-applying onto THAT file keeps both writers' work without any
    side-picking merge.
    """
    from src.shadow_settlement import apply_updates
    p = Path(journal)
    if not p.exists() or not SHADOW_FILE.exists():
        return 0
    updates = json.loads(p.read_text(encoding="utf-8") or "[]")
    df, eol = _read_shadow(SHADOW_FILE)
    out, n = apply_updates(df, updates)
    from src.shadow_settlement import enforce_close_rule
    out, n_stale = enforce_close_rule(out)
    if n or n_stale:
        _write_shadow(out, SHADOW_FILE, eol)
    _write_shadow_health(out)
    log.info(f"shadow: re-applied {n} settlement(s) from journal")
    return n


def _load_archives_named() -> list:
    """`_load_archives()` returns frames without their names; keep them aligned by re-reading
    the same file list in the same order (missing files are skipped there, so skip here too)."""
    frames = []
    for name in _ARCHIVE_FILES:
        p = config.OUTPUT_DIR / name
        if not p.exists():
            frames.append(None)
            continue
        try:
            frames.append(pd.read_csv(p, dtype={"odds": float}, low_memory=False))
        except Exception:                                             # noqa: BLE001
            frames.append(None)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=3,
                        help="How many past days to fetch scores for (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing the ledger")
    parser.add_argument("--shadow-journal", default=os.getenv("SHADOW_SETTLEMENT_JOURNAL"),
                        help="Also write shadow settlements as JSON here (CI re-applies it)")
    parser.add_argument("--shadow-reapply", default=None, metavar="JOURNAL",
                        help="ONLY re-apply a shadow settlement journal to v91_shadow.csv")
    parser.add_argument("--shadow-only", action="store_true",
                        help="Settle only the shadow log (skip every ledger)")
    args = parser.parse_args()

    if args.shadow_reapply:
        reapply_shadow_settlement(args.shadow_reapply)
        return
    if args.shadow_only:
        settle_shadow_predictions(days=args.days, dry_run=args.dry_run,
                                  journal=args.shadow_journal)
        return

    if not LEDGER_FILE.exists():
        log.error(f"Ledger not found: {LEDGER_FILE}")
        sys.exit(1)

    ledger = pd.read_csv(LEDGER_FILE, dtype=str)

    # Only rows with no result yet and a past match date
    today_str = str(datetime.utcnow().date())
    pending_mask = (
        ledger["result"].isna() | (ledger["result"].str.strip() == "")
    ) & (ledger["match_date"] < today_str)

    pending = ledger[pending_mask].copy()
    if pending.empty:
        log.info("No pending bets_ledger results to fill in.")
    else:
        log.info(f"Found {len(pending)} tip(s) with missing results (match_date < {today_str})")

    # Fetch scores — FD first (free), OddsAPI as fallback when FD is stale. Each source stays its
    # OWN pool: a fixture both report must not count as two candidates (or mix two spellings).
    scores_cache: dict[str, list[list[dict]]] = {}
    for league in pending["league"].unique():
        sport_key = config.ODDS_API_SPORT_KEYS.get(league)
        if league in _FD_SOURCES:
            log.info(f"  Fetching scores for {league} from football-data.co.uk ...")
            fd_results = fetch_scores_fd(league)
            scores_cache[league] = [fd_results]

            # Fall back to OddsAPI only for RECENT missing dates (within args.days)
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d")
            pending_dates = set(pending[
                (pending["league"] == league) & (pending["match_date"] >= cutoff)
            ]["match_date"].tolist())
            fd_dates = {ev["date_str"] for ev in fd_results}
            missing_dates = {d for d in pending_dates if d not in fd_dates}
            if missing_dates and sport_key:
                log.info(f"  FD missing dates {missing_dates} — falling back to OddsAPI...")
                api_results = fetch_scores(sport_key, args.days)
                log.info(f"    → OddsAPI returned {len(api_results)} completed events")
                scores_cache[league].append(api_results)
            continue

        if not sport_key:
            log.info(f"  {league}: no football-data or OddsAPI scores source — API-Football only")
            scores_cache[league] = []
            continue
        log.info(f"  Fetching scores for {league} ({sport_key}) last {args.days} days ...")
        scores_cache[league] = [fetch_scores(sport_key, args.days)]
        log.info(f"    → {len(scores_cache[league][0])} completed events returned")

    # Match tips to results
    filled = 0
    updates: list[dict] = []

    for idx, row in pending.iterrows():
        league = row["league"]
        found = None
        for completed in scores_cache.get(league, []):
            found = _find_result(
                home=row["home_team"],
                away=row["away_team"],
                date_str=row["match_date"],
                side=row["side"],
                completed=completed,
            )
            if found is not None:
                break

        if found is None:
            # LAST RESORT: API-Football. Only reached when the primary source could not
            # resolve this fixture, so a healthy run costs nothing. This is what finally
            # settles leagues with no football-data feed once they age past OddsAPI's
            # 3-day scores window.
            found = _find_result(
                home=row["home_team"],
                away=row["away_team"],
                date_str=row["match_date"],
                side=row["side"],
                completed=fetch_scores_apifootball(league, str(row["match_date"])[:10]),
            )

        if found is None:
            log.debug(f"  Not found: {row['home_team']} vs {row['away_team']} ({row['match_date']})")
            continue

        result_str, total_goals, extras = found
        # No usable entry price -> the RESULT is graded, the P/L is left empty (one malformed odds
        # cell must also not abort settlement for every other bet).
        try:
            odds = float(row["odds"])
        except (TypeError, ValueError):
            odds = np.nan
        if not odds > 1.0:
            pnl = np.nan
            log.warning(f"  no entry price ({row.get('odds')!r}) for {row['home_team']} vs "
                        f"{row['away_team']} — result kept, P/L left empty")
        elif result_str == "WIN":
            pnl = round(odds - 1.0, 4)
        else:
            pnl = -1.0

        # Closing Line Value
        cl = _closing_odds(row["home_team"], row["away_team"], row["match_date"], row["side"],
                           league=league)
        clv = (round((odds - cl) / cl * 100, 2)
               if (not np.isnan(cl) and cl > 0 and odds > 1.0) else np.nan)

        update = {
            "idx":          idx,
            "result":       result_str,
            "pnl":          pnl,
            "total_goals":  total_goals,
            "closing_odds": cl,
            "clv_pct":      clv,
        }
        if "ht_score" in extras:
            update["ht_score"] = extras["ht_score"]
        updates.append(update)

        ht_str  = f"  HT={extras['ht_score']}" if "ht_score" in extras else ""
        clv_str = f"  CLV={clv:+.1f}%" if not np.isnan(clv) else ""
        log.info(
            f"  {'[DRY]' if args.dry_run else '[ OK]'} "
            f"{row['home_team']} vs {row['away_team']} "
            f"({row['match_date']})  "
            f"{row['side']} @ {odds}  "
            f"Goals={total_goals}{ht_str}  → {result_str}  PnL={pnl:+.3f}u{clv_str}"
        )
        filled += 1

    if not updates:
        log.info("No pending matches found in any scores source. If matches were recently played, try --days with a larger value or re-run after the local CSVs are refreshed.")
    elif args.dry_run:
        print(f"\n  [DRY RUN] Would update {filled} row(s). No files written.")

    # Write back to ledger
    if updates and not args.dry_run:
        if "ht_score" not in ledger.columns:
            ledger["ht_score"] = ""
        for u in updates:
            ledger.at[u["idx"], "result"] = u["result"]
            ledger.at[u["idx"], "pnl"]    = "" if np.isnan(u["pnl"]) else str(u["pnl"])
            if not np.isnan(u["closing_odds"]):
                ledger.at[u["idx"], "closing_odds"] = str(u["closing_odds"])
            if not np.isnan(u["clv_pct"]):
                ledger.at[u["idx"], "clv_pct"] = str(u["clv_pct"])
            if u.get("ht_score"):
                ledger.at[u["idx"], "ht_score"] = u["ht_score"]

        ledger.to_csv(LEDGER_FILE, index=False)

    if updates and not args.dry_run:
        total_pnl  = sum(u["pnl"] for u in updates if not np.isnan(u["pnl"]))
        wins       = sum(1 for u in updates if u["result"] == "WIN")
        losses     = sum(1 for u in updates if u["result"] == "LOSS")
        clv_vals   = [u["clv_pct"] for u in updates if not np.isnan(u["clv_pct"])]
        avg_clv    = np.mean(clv_vals) if clv_vals else np.nan

        print("\n" + "=" * 60)
        print(f"  RESULTS UPDATED — {filled} bet(s)")
        print("=" * 60)
        print(f"  Win  : {wins}")
        print(f"  Loss : {losses}")
        print(f"  PnL  : {total_pnl:+.3f} units (flat 1u stakes)")
        if not np.isnan(avg_clv):
            verdict = "SHARP (beat closing line)" if avg_clv > 0 else "SOFT (market moved against)"
            print(f"  CLV  : {avg_clv:+.2f}% avg  →  {verdict}")

        # Full ledger P&L if any resolved rows exist
        all_with_result = ledger[ledger["result"].isin(["WIN", "LOSS", "VOID"])].copy()
        if not all_with_result.empty:
            all_with_result["pnl"] = pd.to_numeric(all_with_result["pnl"], errors="coerce")
            total_all = all_with_result["pnl"].sum()
            w_all = (all_with_result["result"] == "WIN").sum()
            l_all = (all_with_result["result"] == "LOSS").sum()
            hit   = w_all / (w_all + l_all) if (w_all + l_all) > 0 else 0
            print()
            print(f"  LEDGER TOTALS ({len(all_with_result)} settled bets)")
            print(f"  Win rate : {hit:.0%}  ({w_all}W / {l_all}L)")
            print(f"  Total PnL: {total_all:+.3f} units")

        print()
        log.info(f"Ledger saved → {LEDGER_FILE}")

    # Also resolve player prop, sharp money, and side-market ledgers
    update_player_results(days=args.days, dry_run=args.dry_run)
    update_sharp_results(days=args.days, dry_run=args.dry_run)
    update_side_market_results(days=args.days, dry_run=args.dry_run)
    update_ht_results(days=args.days)
    # Guarded: this is an EVIDENCE log, not a tipping path. If it ever throws it must not take
    # the live-signal resolution down with it — the workflow comment above records that a crash
    # mid-sequence leaves later ledgers ungraded.
    try:
        update_ht_observations()
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"ht_observations update skipped: {e}")
    update_live_signal_results(days=args.days, dry_run=args.dry_run)
    # LAST and guarded, like ht_observations: an evidence log must never take a ledger down.
    try:
        settle_shadow_predictions(days=args.days, dry_run=args.dry_run,
                                  journal=args.shadow_journal)
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"shadow settlement skipped: {e}")


if __name__ == "__main__":
    main()
