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

# Club names are matched ONLY by src.team_names.resolve_result inside a league+date pool
# (_match_event, _closing_for_market, _find_prop_fixture). The pairwise _names_match and the
# lowercase _norm key were deleted (Data Truth round 2): a one-name pool can never be ambiguous,
# so any token-subset pair matched ('Inter' / 'Inter Miami').

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
    if rh and ra:
        hits = [e for e in cands if str(e["home_team"]) == rh and str(e["away_team"]) == ra]
    else:
        hits = _one_club_hits(home, away, cands)
    scores = {(e["home_score"], e["away_score"], e.get("ht_home"), e.get("ht_away")) for e in hits}
    return hits[0] if hits and len(scores) == 1 else None


def _same_club(a, b) -> bool:
    """Two spellings are provably one club: each resolves to the other (src.team_names tiers:
    normalised equality, the alias store, identical identity tokens)."""
    from src.team_names import resolve_result
    a, b = str(a), str(b)
    return resolve_result(a, [b]) == b and resolve_result(b, [a]) == a


def _one_club_hits(home, away, cands: list[dict]) -> list[dict]:
    """TWO-SPELLING CLUBS (round 2, P1-2). A source can carry one club under two spellings in the
    same league + date pool (fd_history: 'Tirol' and 'WSG Wattens'; one fixture reported twice), and
    resolve_result then refuses the pool as ambiguous, so the row could never settle. Fall back to
    the events whose clubs each resolve to the row's clubs one name at a time - accepted only when
    every matched spelling on a side is provably ONE club (the identity alias map), so two
    different clubs are still refused. The caller still requires a single score."""
    hits = [e for e in cands if _same_club(home, e["home_team"]) and _same_club(away, e["away_team"])]
    for side in ("home_team", "away_team"):
        names = sorted({str(e[side]) for e in hits})
        if any(not _same_club(names[0], n) for n in names[1:]):
            return []
    return hits


# ── RESCHEDULED MATCHES: THE 48-HOUR RULE (owner decision 2026-11-02, round 2 P1-2) ─────────────────
# Every grader matched a result on the row's EXACT date, so a fixture a source dates one day off
# (UTC vs local kick-off: MLS evening games are the next UTC day; fd_history holds MLS on two dates)
# or a match moved by a day never settled. The rule now: a rescheduled match settles on the ACTUAL
# match when it was played within 48 hours of the original kick-off, and its tips VOID only on status
# evidence once that window has closed. Ledgers carry a date, not a kick-off time, so the window is
# taken in whole days: the original date first, then D-1 .. D+2 - always with exactly one fixture
# and one score (_match_event), so a wider window can never pick between two matches.
_RESCHEDULE_DAYS = (-1, 2)


def _settle_dates(date_str) -> list[set]:
    """[{D}, {D-1 .. D+2}] for a row dated D: the exact date first, then the 48-hour window."""
    d = pd.Timestamp(str(date_str)[:10])
    lo, hi = _RESCHEDULE_DAYS
    return [{str(d.date())}, {str((d + pd.Timedelta(days=k)).date()) for k in range(lo, hi + 1)}]


def _find_event(home, away, events: list[dict], date_str) -> dict | None:
    """THE event for a row dated `date_str`: on that date, else inside the 48-hour window."""
    for dates in _settle_dates(date_str):
        ev = _match_event(home, away, events, dates)
        if ev is not None:
            if str(ev.get("date_str", ""))[:10] != str(date_str)[:10]:
                log.info(f"  {home} vs {away}: dated {str(date_str)[:10]}, result found on "
                         f"{ev.get('date_str')} (inside the 48-hour reschedule window)")
            return ev
    return None


def _window_closed(date_str, today=None) -> bool:
    """True once the 48-hour reschedule window of a fixture dated `date_str` has fully passed."""
    t = pd.Timestamp(today or datetime.utcnow().date())
    return t > pd.Timestamp(str(date_str)[:10]) + pd.Timedelta(days=_RESCHEDULE_DAYS[1])


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
            # FT + AET + PEN (round 5): extra-time fixtures used to be filtered out here, so an MLS
            # play-off or Argentina knockout tip could never settle from API-Football. They come
            # with their 90-MINUTE score (score.fulltime) - settlement ignores extra time.
            df = fetch_league_fixture_results(lid, season, league, statuses="FT-AET-PEN")
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
                st = r.get("status")
                if isinstance(st, str) and st:
                    ev["status"] = st
                out.append(ev)
            log.info(f"  [api-football] {league}: {len(out)} completed fixtures "
                     f"(season {season}) for fallback settlement")
        except Exception as e:
            log.warning(f"  [api-football] {league}: results fetch failed ({e})")
    else:
        log.debug(f"  [api-football] {league}: no API_FOOTBALL_IDS entry")

    _AF_RESULTS_CACHE[(league, season)] = out
    return out


# ── KNOCKOUT GUARD: TEAM LEDGERS SETTLE ON 90 MINUTES (round 5, P1-2e) ──────────────────────────────
# football-data and OddsAPI publish ONE final score and do not say whether it is the 90- or the
# 120-minute score. In a league with knock-out rounds (MLS play-offs, Argentina's knockout stage,
# Championship play-offs) a match can go to extra time, and a 120-minute score would settle an O/U,
# BTTS or sharp tip on goals scored after the 90 minutes the owner rule settles on.
#
# So before a football-data / OddsAPI score settles a row, the league's API-Football fixtures decided
# after extra time / on penalties (status AET / PEN) for the match's season are looked up: if the
# fixture is one of them, the row settles on API-Football's 90-minute score (score.fulltime) instead.
# One call per (league, season) per run, cached here. When that list cannot be fetched although a key
# is set, the row stays PENDING (logged) - a late settlement is better than a 120-minute one. With
# no key at all (local runs) or no API-Football id for the league there is no source to check
# against; the row settles as before and the run counts it.
_AF_ET_CACHE: dict = {}
_KNOCKOUT_UNCHECKED: dict[str, int] = {}


def _extra_time_events(league, match_date) -> list[dict] | None:
    """The league's AET / PEN fixtures of the match's season as events (90-minute score), [] when
    there is no source to check, None when the source exists but did not answer."""
    from src import api_football_ou as AFO
    from src.season_resolver import resolve_api_season, is_known
    lid = config.API_FOOTBALL_IDS.get(league)
    if not lid:
        _KNOCKOUT_UNCHECKED["no_api_football_id"] = _KNOCKOUT_UNCHECKED.get("no_api_football_id", 0) + 1
        return []
    season = resolve_api_season(league, str(match_date)[:10])
    if not is_known(season):
        _KNOCKOUT_UNCHECKED["season_unknown"] = _KNOCKOUT_UNCHECKED.get("season_unknown", 0) + 1
        return []
    if not AFO._KEY:
        _KNOCKOUT_UNCHECKED["no_api_key"] = _KNOCKOUT_UNCHECKED.get("no_api_key", 0) + 1
        return []
    k = (league, season)
    if k not in _AF_ET_CACHE:
        try:
            df = AFO.fetch_league_extra_time_results(lid, season, league)
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"  [knockout guard] {league} {season}: lookup failed ({e})")
            df = None
        if df is None:
            _AF_ET_CACHE[k] = None
        else:
            _AF_ET_CACHE[k] = [{"home_team": str(r["home_team"]), "away_team": str(r["away_team"]),
                                "home_score": int(r["home_goals"]), "away_score": int(r["away_goals"]),
                                "date_str": str(r["date"])[:10], "status": str(r["status"]),
                                **({"ht_home": int(r["ht_home_goals"]), "ht_away": int(r["ht_away_goals"])}
                                   if pd.notna(r.get("ht_home_goals")) and pd.notna(r.get("ht_away_goals"))
                                   else {})}
                               for _, r in df.iterrows()]
    return _AF_ET_CACHE[k]


def _knockout_guard(league, home, away, date_str, ev: dict) -> tuple[dict | None, str]:
    """(the event to settle on, '') for a football-data / OddsAPI event `ev`, or (None, why).
    An API-Football event (it carries `status`) is already on 90 minutes and passes unchanged."""
    if ev is None or ev.get("status"):
        return ev, ""
    et = _extra_time_events(league, date_str)
    if et is None:
        return None, "knockout_guard_unavailable (API-Football AET/PEN list not fetched)"
    hit = _find_event(home, away, et, date_str)
    if hit is None:
        return ev, ""
    log.info(f"  {home} vs {away} {str(date_str)[:10]}: API-Football reports {hit['status']} - "
             f"settled on the 90-minute score {hit['home_score']}-{hit['away_score']} "
             f"(source said {ev.get('home_score')}-{ev.get('away_score')})")
    return hit, ""


def _team_event(league, home, away, date_str, pools: list,
                fallback: bool = True) -> tuple[dict | None, str]:
    """THE 90-minute event for a team-ledger row: the first primary pool (football-data / OddsAPI)
    that holds the fixture, through the knockout guard; else (when `fallback`) API-Football - lazy,
    already on 90 minutes. (None, '') when no source has it, (None, why) when the guard holds it."""
    for pool in pools:
        ev = _find_event(home, away, pool or [], date_str)
        if ev is not None:
            return _knockout_guard(league, home, away, date_str, ev)
    if not fallback:
        return None, ""
    return _find_event(home, away, fetch_scores_apifootball(league, str(date_str)[:10]), date_str), ""


EXTRA_TIME_RESTATED_FILE = config.OUTPUT_DIR / "corrections" / "team_ledgers_extra_time_restated.csv"


def _restate_one(ledger: str, row, ev: dict) -> str:
    """The row's result re-graded on the 90-minute event `ev`, with the ledger's own grader."""
    hs, as_ = int(ev["home_score"]), int(ev["away_score"])
    if ledger == "bets_ledger":
        g = _grade_ou_event(ev, str(row.get("side", "")))
        return g[0] if g else ""
    if ledger == "side_bets_ledger":
        return _resolve_side_market(str(row.get("market", "")), hs, as_)
    if ledger == "sharp_ledger":
        return _resolve_sharp_side(str(row.get("side", "")), hs, as_) or ""
    return ""


def extra_time_restatement(frames: dict, et_lookup) -> pd.DataFrame:
    """Settled WIN / LOSS team-ledger rows whose fixture API-Football reports as AET / PEN, with the
    result re-graded on the 90-minute score BESIDE the committed one (the ledgers are not rewritten).
    `frames` = {ledger name: DataFrame}; `et_lookup(league, date)` -> AET/PEN events or None.
    The restated P/L uses the price the ledger settled on (sharp: the signal-time price)."""
    out = []
    for name, d in frames.items():
        if d is None or d.empty or "result" not in d.columns:
            continue
        for _, r in d[d["result"].isin(["WIN", "LOSS"])].iterrows():
            et = et_lookup(r["league"], str(r["match_date"])[:10])
            if not et:
                continue
            hit = _find_event(r["home_team"], r["away_team"], et, str(r["match_date"])[:10])
            if hit is None:
                continue
            new = _restate_one(name, r, hit)
            if not new:
                continue
            if name == "sharp_ledger":
                odds = _sharp_entry_odds(r)
            else:
                try:
                    odds = float(r.get("odds"))
                except (TypeError, ValueError):
                    odds = float("nan")
            priced = str(r.get("pnl", "")).strip() not in ("", "nan") and odds > 1.0
            out.append({"ledger": name, "match_date": r["match_date"], "league": r["league"],
                        "home_team": r["home_team"], "away_team": r["away_team"],
                        "market": r.get("market", "") if name == "side_bets_ledger" else r.get("side", ""),
                        "status": hit["status"], "score_90": f"{hit['home_score']}-{hit['away_score']}",
                        "original_result": r["result"], "original_pnl": r.get("pnl", ""),
                        "restated_result": new,
                        "restated_pnl": "" if not priced else (round(odds - 1.0, 4) if new == "WIN" else -1.0),
                        "changed": new != r["result"],
                        "note": "settled on 90 minutes (owner rule); source API-Football score.fulltime"})
    return pd.DataFrame(out)


def write_extra_time_restatement(dst: Path | None = None) -> pd.DataFrame:
    """output/corrections/team_ledgers_extra_time_restated.csv. Needs API-Football (CI); one call per
    (league, season) with settled rows. Ledgers untouched."""
    dst = dst or EXTRA_TIME_RESTATED_FILE
    frames = {}
    for name, p in (("bets_ledger", LEDGER_FILE), ("side_bets_ledger", SIDE_LEDGER_FILE),
                    ("sharp_ledger", SHARP_LEDGER_FILE)):
        try:
            frames[name] = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"extra-time restatement: {p.name} unreadable ({e})")
    out = extra_time_restatement(frames, _extra_time_events)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, lineterminator="\n")
    n = int(out["changed"].sum()) if not out.empty else 0
    log.info(f"extra-time restatement: {len(out)} settled row(s) on AET/PEN fixtures, {n} result(s) "
             f"change on 90 minutes -> {dst}")
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
    return _grade_ou_event(_find_event(home, away, completed, date_str), side)


def _grade_ou_event(ev: dict | None, side: str) -> tuple[str, float, dict] | None:
    """(result, total goals, extras) for an O/U 2.5 / HT side on one matched event, or None."""
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
#   * a null in the player sheet is read with THE shared null rule, player_model.player_stat_nulls
#     (owner decision 2026-11-04, fix round 10 'settle'): a null COUNT is 0 when the player played
#     (minutes > 0) AND the provider returned player statistics for the fixture; otherwise it stays
#     missing and the row stays pending. Until fix round 10 settlement had its own reading - a null
#     shots-on-target settled LOSS only when the team 'Shots on Goal' total reconciled with the
#     players who had a value, and otherwise stayed pending for ever. A row settled because the
#     rule read a null as 0 is recorded BESIDE the ledger (NULL_RULE_SETTLED_FILE).

_FT_STATUSES = ("FT", "AET", "PEN")
_VOID_STATUSES = {"PST": "POSTPONED", "CANC": "CANCELLED", "ABD": "ABANDONED"}


# ── POSTPONED / CANCELLED / ABANDONED: VOID ON EVIDENCE ONLY (estate scan P2, 2026-11-02) ──────────
# Only the props settler ever voided a postponed fixture. The O/U, side-market, sharp and HT ledgers
# had no path to VOID at all, so a postponed match sat pending for ever: 45 main, 10 side, 19 HT and
# 164 sharp rows were pending more than 7 days on the committed ledgers (2026-11-02).
#
# A row is voided ONLY when API-Football's own fixture says PST / CANC / ABD: the league's fixtures
# on the row's date (league id + season from the registry), both clubs resolved in that pool by
# src.team_names (exactly one fixture, the same lookup the props settler uses). Never on age: a row
# no source can settle stays pending, and the reason is logged and counted every run.
#
# BUDGET. Each distinct (league, season, date) costs one API-Football call per run (the client cache
# does not survive a CI runner). main() plans the days for the whole run: at most
# UPDATE_RESULTS_AF_STATUS_BUDGET (default 40) of them, chosen in an order that rotates every hour,
# so a backlog larger than the budget is still fully covered within a day. None = no plan (tests,
# ad-hoc calls) -> every lookup is allowed.
_STATUS_ALLOWED: set | None = None


def _status_day(league, date_str) -> tuple | None:
    canon = _canon_league(league)
    lid = config.API_FOOTBALL_IDS.get(canon) if canon else None
    if not lid:
        return None
    from src.season_resolver import resolve_api_season, is_known
    season = resolve_api_season(canon, str(date_str)[:10])
    return (lid, season, str(date_str)[:10]) if is_known(season) else None


def _plan_status_checks(frames: list, today_str: str, budget: int | None = None,
                        now: datetime | None = None) -> set:
    """Choose this run's (league id, season, date) status lookups across every pending ledger."""
    import hashlib
    budget = int(os.getenv("UPDATE_RESULTS_AF_STATUS_BUDGET", "40")) if budget is None else budget
    days = set()
    for df in frames:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            d = _status_day(r.get("league"), r.get("match_date"))
            if d and d[2] < today_str:
                days.add(d)
    salt = (now or datetime.utcnow()).strftime("%Y-%m-%d %H")
    order = sorted(days, key=lambda d: hashlib.sha1(f"{salt}|{d}".encode()).hexdigest())
    chosen = set(order[:max(budget, 0)])
    if len(days) > len(chosen):
        log.info(f"status checks: {len(chosen)} of {len(days)} league-days this run (budget {budget}); "
                 f"the rest rotate into later runs")
    return chosen


def _pending_frames_for_status(today_str: str) -> list:
    """Pending rows at least 2 days old from every ledger that can void on status evidence (the
    day after a match its result normally settles from the scores; no lookup is spent on it)."""
    cut = str((pd.Timestamp(today_str) - pd.Timedelta(days=1)).date())
    out = []
    for p in (LEDGER_FILE, SIDE_LEDGER_FILE, SHARP_LEDGER_FILE, config.OUTPUT_DIR / "ht_ledger.csv"):
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:                                             # noqa: BLE001
            continue
        if {"result", "match_date", "league"} <= set(d.columns):
            out.append(d[(d["result"].str.strip() == "") & (d["match_date"] < cut)])
    return out


# ── RESCHEDULED > 48 HOURS, THEN PLAYED (round 5) ──────────────────────────────────────────────────
# API-Football MOVES a rescheduled fixture to its new date, so the original-date lookup above finds
# nothing and the row stayed pending for ever. Once the 48-hour window has closed, the league's
# fixtures of the match's season are searched for the SAME PAIR (both clubs resolved in the season's
# pool by src.team_names, same home / away):
#   * exactly one such fixture, dated after the window (D+2)  -> VOID 'RESCHEDULED' (owner rule: a
#     match not played within 48h of the original kick-off voids, whatever happens to it later);
#   * exactly one, inside the window                          -> its own status decides: PST / CANC /
#     ABD voids, a finished match settles from the results (it is in the 48-hour result window);
#   * two or more (four-meeting leagues, best-of-3 play-offs) -> AMBIGUOUS, stays pending, logged;
#   * one dated before the window                             -> not this match, stays pending.
# Ledgers carry a date, not a kick-off time, so the window is taken in whole days (D-1 .. D+2), the
# same window the result lookup uses. One call per (league, season) per run, cached here.
_DAY_MISS = ("no API-Football fixtures for the league on that date",
             "clubs not resolved in that date's fixtures", "0 fixtures match")
_SEASON_FIXTURES: dict = {}


def _season_fixtures(league_id, season) -> list:
    k = (league_id, str(season))
    if k not in _SEASON_FIXTURES:
        _SEASON_FIXTURES[k] = _af_response("/fixtures", {"league": league_id, "season": season}, 6)
    return _SEASON_FIXTURES[k]


def _reschedule_evidence(day: tuple, date_str, home, away, day_why: str) -> tuple[str | None, str]:
    """('RESCHEDULED' | a _VOID_STATUSES label, '') or (None, why the row stays pending)."""
    from src.team_names import resolve_result
    lid, season, d = day
    try:
        fixtures = _season_fixtures(lid, season)
    except Exception as e:                                            # noqa: BLE001
        return None, f"fixture_not_found_on_date ({day_why}; season lookup failed: {type(e).__name__})"
    named = [f for f in fixtures if all(_fixture_teams(f))]
    if not named:
        return None, f"fixture_not_found_on_date ({day_why}; no season fixtures)"
    rh = resolve_result(str(home), sorted({_fixture_teams(f)[0] for f in named}))
    ra = resolve_result(str(away), sorted({_fixture_teams(f)[1] for f in named}))
    if not rh or not ra:
        return None, f"fixture_not_found_on_date ({day_why}; clubs not resolved in the season's fixtures)"
    pair = [f for f in named if _fixture_teams(f) == (rh, ra)]
    if not pair:
        return None, f"fixture_not_found_on_date ({day_why}; no same-pair fixture in the season)"
    if len(pair) > 1:
        dates = sorted(str((f.get("fixture") or {}).get("date") or "")[:10] for f in pair)
        return None, f"reschedule_ambiguous ({len(pair)} same-pair fixtures in season {season}: {dates})"
    fx = pair[0]
    on = str((fx.get("fixture") or {}).get("date") or "")[:10]
    status = ((fx.get("fixture") or {}).get("status") or {}).get("short")
    if not on:
        return None, "fixture_not_found_on_date (same-pair fixture has no date)"
    lo = str((pd.Timestamp(d) + pd.Timedelta(days=_RESCHEDULE_DAYS[0])).date())
    hi = str((pd.Timestamp(d) + pd.Timedelta(days=_RESCHEDULE_DAYS[1])).date())
    if on < lo:
        return None, f"reschedule_not_this_match (only same-pair fixture is on {on}, before {d})"
    if on > hi:
        log.info(f"  {home} vs {away} {d}: API-Football moved the fixture to {on} ({status}), "
                 f"more than 48h later -> RESCHEDULED")
        return "RESCHEDULED", ""
    if status in _VOID_STATUSES:                       # the window has closed (checked by the caller)
        return _VOID_STATUSES[status], ""
    return None, f"no_result_yet (same-pair fixture on {on} inside the 48h window, status {status!r})"


def _postponement_evidence(league, date_str, home, away) -> tuple[str | None, str]:
    """('POSTPONED' | 'CANCELLED' | 'ABANDONED', '') when API-Football reports that status for this
    fixture, else (None, why the row stays pending)."""
    if not os.getenv("APIFOOTBALL_KEY", ""):
        return None, "no_status_source (APIFOOTBALL_KEY not set)"
    day = _status_day(league, date_str)
    if day is None:
        return None, "no_status_source (league or season not in the registry)"
    if _STATUS_ALLOWED is not None and day not in _STATUS_ALLOWED:
        return None, "status_not_checked_this_run (budget)"
    try:
        fx, why = _find_prop_fixture(day[0], day[1], day[2], home, away)
    except Exception as e:                                            # noqa: BLE001
        return None, f"status_lookup_failed ({type(e).__name__})"
    if fx is None:
        if why in _DAY_MISS and _window_closed(date_str):
            return _reschedule_evidence(day, date_str, home, away, why)
        return None, f"fixture_not_found_on_date ({why})"
    status = ((fx.get("fixture") or {}).get("status") or {}).get("short")
    if status in _VOID_STATUSES:
        # 48-hour rule: a postponed / abandoned match replayed within 48 hours settles on the actual
        # match, so the status only voids once that window has closed (CANC is final at once).
        if status != "CANC" and not _window_closed(date_str):
            return None, f"postponed_within_48h_window ({status})"
        return _VOID_STATUSES[status], ""
    return None, f"no_result_yet (API-Football status {status!r})"


def _join_note(old, label: str) -> str:
    old = "" if old is None or (isinstance(old, float) and np.isnan(old)) else str(old).strip()
    return label if not old or old == "nan" else f"{old}; {label}"


def _log_pending(ledger: str, reasons: dict, today_str: str, dates: list) -> None:
    """One line per run: why rows stay pending (never voided on age), and how many are > 7 days."""
    if not reasons:
        return
    cut = str((pd.Timestamp(today_str) - pd.Timedelta(days=7)).date())
    old = sum(1 for d in dates if str(d)[:10] < cut)
    log.info(f"{ledger}: {len(dates)} left pending this run ({old} older than 7 days) - "
             f"{dict(sorted(reasons.items()))}")

_PLAYER_MARKET_STAT = {"goals": "goals", "sot": "shots_on_target", "cards": "yellow_card",
                       "assists": "assists"}


def _known(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def card_market_result(yellow, red) -> str:
    """THE 'cards' (player to be carded) rule: a yellow OR a red card wins - a straight red is a card.
    WIN on any known card, LOSS only when both are known to be 0, '' otherwise (pending).
    player_ledger graded on yellow only, so a straight red was a LOSS here and a WIN in
    clv_records (player_model.clv_tracker._MARKET_HIT['cards'] = yc>=1 or rc>=1); round 2 P2.
    tests/test_settlement_round2.py holds both graders to this one rule."""
    if (_known(yellow) and int(yellow) >= 1) or (_known(red) and int(red) >= 1):
        return "WIN"
    if _known(yellow) and _known(red):
        return "LOSS"
    return ""


CARD_REGRADE_FILE = config.OUTPUT_DIR / "corrections" / "player_ledger_card_regrade_2026-11.csv"


def card_regrade_sidecar(ledger: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Settled 'cards' rows whose committed result disagrees with card_market_result() on the
    player's recorded cards (player_history: player_id + match date). The ledger is not rewritten;
    original and restated values sit side by side. A row without a unique stats match is skipped."""
    h = history.assign(_pid=pd.to_numeric(history["player_id"], errors="coerce"),
                       _d=history["date"].astype(str).str[:10])
    h = h.dropna(subset=["_pid"]).drop_duplicates(["_pid", "_d"], keep=False)    # ambiguous: skip
    idx = {(p, d): (y, r) for p, d, y, r in zip(h["_pid"], h["_d"], h["yellow_cards"], h["red_cards"])}
    rows = []
    c = ledger[(ledger["market"] == "cards") & ledger["result"].isin(["WIN", "LOSS"])]
    for _, r in c.iterrows():
        pid = pd.to_numeric(r.get("player_id"), errors="coerce")
        hit = idx.get((pid, str(r["match_date"])[:10]))
        if hit is None:
            continue
        yc, rc = (None if pd.isna(v) else float(v) for v in hit)
        new = card_market_result(yc, rc)
        # Only rows the RULE changes: the old yellow-only rule on the same stats gives the committed
        # result. A row where the stats sources themselves disagree is a different finding (P1-5).
        old = "" if yc is None else ("WIN" if yc >= 1 else "LOSS")
        if not new or new == r["result"] or old != r["result"]:
            continue
        try:
            o = float(r.get("market_odds"))
        except (TypeError, ValueError):
            o = np.nan
        priced = str(r.get("pnl", "")).strip() != "" and o > 1.0
        rows.append({**{k: r.get(k, "") for k in ("signal_date", "match_date", "league", "home_team",
                                                   "away_team", "player_name", "player_id", "market",
                                                   "tier", "market_odds")},
                     "original_result": r["result"], "original_pnl": r.get("pnl", ""),
                     "restated_result": new,
                     "restated_pnl": (round(o - 1.0, 4) if new == "WIN" else -1.0) if priced else "",
                     "yellow_cards": hit[0], "red_cards": hit[1],
                     "note": "one card rule (yellow OR red wins), as clv_records grades it; "
                             "source player_history.parquet"})
    return pd.DataFrame(rows)


def write_card_regrade(dst: Path | None = None) -> pd.DataFrame:
    """Write output/corrections/player_ledger_card_regrade_2026-11.csv (offline; ledger untouched)."""
    dst = dst or CARD_REGRADE_FILE
    led = pd.read_csv(PLAYER_LEDGER_FILE, dtype=str, keep_default_na=False)
    hist = pd.read_parquet(config.OUTPUT_DIR.parent / "player_history.parquet",
                           columns=["player_id", "date", "yellow_cards", "red_cards"])
    out = card_regrade_sidecar(led, hist)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, lineterminator="\n")
    log.info(f"card regrade: {len(out)} row(s) -> {dst}")
    return out


def _resolve_player_market(market: str, stats: dict) -> str:
    """WIN/LOSS for a player prop: `<base>` means 1+, `<base><N>` means N+ ('sot2' = 2+ shots on
    target). The producer writes sot2/sot3/goals2; this used to know only the bare names, so every
    multi-threshold prop stayed pending for ever (Data Truth audit finding 19). Unknown market or a
    stat that is not known -> '' (the row stays pending; a missing stat was read as 0 = LOSS).
    A stat that is unknown but has a known UPPER bound (`<stat>_max`: the whole-match count of an
    extra-time match, which can only be >= the 90-minute count) settles LOSS when the bound is
    below the line, and stays pending otherwise."""
    import re
    m = re.fullmatch(r"([a-z]+?)(\d*)", str(market or "").strip().lower())
    if not m or m.group(1) not in _PLAYER_MARKET_STAT:
        return ""
    need = int(m.group(2) or 1)
    if m.group(1) == "cards":
        return card_market_result(stats.get("yellow_card"), stats.get("red_card")) if need == 1 else ""
    key = _PLAYER_MARKET_STAT[m.group(1)]
    v = stats.get(key)
    if not _known(v):
        cap = stats.get(f"{key}_max")
        return "LOSS" if _known(cap) and int(cap) < need else ""
    return "WIN" if int(v) >= need else "LOSS"


# ── PROPS SETTLE ON 90 MINUTES (owner decision 2026-11-02, round 2 P1-2) ─────────────────────────
# Regular time + stoppage time; extra time is ignored. The `status == 'FT'` filter that dropped
# every extra-time match is gone (AET / PEN are finished statuses), but API-Football's player sheet
# of an AET match counts all 120 minutes, so reading it as-is would leak extra time into a 90-minute
# market. For AET / PEN the count stats are rebuilt from the fixture's event feed, kept to events in
# regular time (elapsed <= 90; stoppage keeps elapsed 90 with an `extra`), and only when that feed is
# provably complete: its regular-time goals add up to the 90-minute score (score.fulltime). Shots on
# target have no event feed: the whole-match count is only an upper bound (it settles a LOSS when
# already below the line), anything else stays pending - never an invented number.
_EXTRA_TIME_STATUSES = ("AET", "PEN")


def _regulation(e: dict) -> bool:
    t = e.get("time") or {}
    try:
        return int(t.get("elapsed")) <= 90 and "shootout" not in str(e.get("comments") or "").lower()
    except (TypeError, ValueError):
        return False


def _regulation_stats(rec: dict, fixture_id: int, fixture: dict) -> tuple[dict | None, str]:
    """(90-minute stats, '') for one player of an AET/PEN fixture, or (None, why)."""
    ft = ((fixture.get("score") or {}).get("fulltime") or {})
    if ft.get("home") is None or ft.get("away") is None:
        return None, "no 90-minute score (score.fulltime) for an extra-time match"
    ev = _af_response("/fixtures/events", {"fixture": fixture_id}, 168)
    reg = [e for e in ev if _regulation(e)]
    goals = [e for e in reg if e.get("type") == "Goal" and e.get("detail") != "Missed Penalty"]
    if len(goals) != int(ft["home"]) + int(ft["away"]):
        return None, "event feed does not add up to the 90-minute score"
    pid = int(rec["player_id"])
    if any(e.get("type") == "subst" and not _regulation(e)
           and pid in ((e.get("player") or {}).get("id"), (e.get("assist") or {}).get("id")) for e in ev):
        return None, "substituted in extra time - 90-minute appearance not provable"
    mine = lambda e, k="player": (e.get(k) or {}).get("id") == pid        # noqa: E731
    scored = [e for e in goals if e.get("detail") != "Own Goal"]
    cards = [e for e in reg if e.get("type") == "Card" and mine(e)]
    out = dict(rec)
    out["goals"] = sum(1 for e in scored if mine(e))
    out["assists"] = sum(1 for e in scored if mine(e, "assist"))
    out["yellow_card"] = int(any(str(e.get("detail", "")).startswith("Yellow") for e in cards))
    out["red_card"] = int(any(str(e.get("detail", "")) in ("Red Card", "Second Yellow card") for e in cards))
    out["shots_on_target_max"] = rec.get("shots_on_target")
    out["shots_on_target"] = None
    return out, ""


def _find_prop_fixture_near(league: str, league_id: int, date_str: str, home: str, away: str):
    """_find_prop_fixture on the row's date, then the next and the previous day (API-Football
    dates fixtures in UTC: an evening kick-off in the Americas is the next UTC day). Exactly one
    fixture or none; each day's season from src.season_resolver."""
    from src.season_resolver import resolve_api_season, is_known
    why = "season unknown"
    for k in (0, 1, -1):
        d = str((pd.Timestamp(str(date_str)[:10]) + pd.Timedelta(days=k)).date())
        season = resolve_api_season(league, d)
        if not is_known(season):
            continue
        fx, why = _find_prop_fixture(league_id, season, d, home, away)
        if fx is not None:
            return fx, ""
    return None, why


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


#: the per-player count columns (player_stat_nulls.COUNT_FIELDS) behind each settled stat of a rec
_REC_COUNT_COLS = {"goals": ("goals",), "assists": ("assists",), "shots_on_target": ("shots_on_target",),
                   "yellow_card": ("yellow_cards",), "red_card": ("red_cards",)}


def _fixture_players(fixture_id: int) -> dict:
    """{'teams': {team_id: [player rec]}, 'by_id': {player_id: rec}, 'answered': bool} from
    /fixtures/players, every count read with THE shared null rule (player_model.player_stat_nulls,
    owner decision 2026-11-04 - one helper, no copy in the settler):
      * a value the provider SENT is kept (a sent 0 is a 0);
      * a null COUNT is 0 when the player PLAYED (games.minutes > 0) and the provider returned
        player statistics for the fixture (fixture_answered); otherwise it stays None (pending).
    `rule_zero` on a rec lists the stats the rule read as 0 from a provider null: a result decided
    by one of them is recorded beside the ledger (NULL_RULE_SETTLED_FILE). Minutes are the gate,
    never zero-filled (None when the provider sent none)."""
    from player_model import player_stat_nulls as PSN
    resp = _af_response("/fixtures/players", {"fixture": fixture_id}, 168)
    answered = PSN.fixture_answered(resp)
    teams, by_id = {}, {}
    for block in resp:
        tid = (block.get("team") or {}).get("id")
        recs = []
        for pe in block.get("players") or []:
            p = pe.get("player") or {}
            s = (pe.get("statistics") or [{}])[0] or {}
            games = s.get("games") or {}
            c = PSN.player_counts(s, answered)
            yel, red = c["yellow_cards"], c["red_cards"]
            rec = {"player_id": p.get("id"), "player_name": p.get("name"), "team_id": tid,
                   "minutes_played": PSN.minutes_of(s), "substitute": games.get("substitute"),
                   "goals": c["goals"], "assists": c["assists"],
                   "shots_on_target": c["shots_on_target"],
                   "yellow_card": None if yel is None else int(yel > 0),
                   "red_card": None if red is None else int(red > 0)}
            rec["rule_zero"] = sorted(k for k, cols in _REC_COUNT_COLS.items()
                                      if rec[k] == 0 and all(PSN.raw_value(s, col) is None for col in cols))
            recs.append(rec)
            if rec["player_id"] is not None:
                by_id[int(rec["player_id"])] = rec
        if tid is not None:
            teams[tid] = recs
    return {"teams": teams, "by_id": by_id, "answered": answered}


def _goal_events_complete(events: list, fixture: dict) -> bool:
    """The event feed holds every goal: scored goal events (own goals included, missed penalties
    not) add up to the final score."""
    g = fixture.get("goals") or {}
    if g.get("home") is None or g.get("away") is None:
        return False
    n = sum(1 for e in events if e.get("type") == "Goal" and e.get("detail") != "Missed Penalty")
    return n == int(g["home"]) + int(g["away"])


def _fill_null_stats(rec: dict, fixture_id: int, fixture: dict, players: dict) -> dict:
    """Replace stats the null rule left None with values PROVEN from the fixture's own record;
    leave them None otherwise. Lazy: the extra endpoints are called only when a needed stat is None.

    Since fix round 10 (owner 2026-11-04) _fixture_players reads every count with the shared null
    rule, so for a player who PLAYED a stat is None only when the provider sent something that is
    not a count. This proof path is kept for that case, unchanged for goals / assists / cards (the
    event feed when its goals add up to the final score). The team 'Shots on Goal' total is read
    with THE team rule, src.af_team_stats (a null in an answered team block is 0; an unanswered
    block is missing), not as a raw value."""
    from src import af_team_stats as AFTS
    out = dict(rec)
    pid = int(rec["player_id"])
    if any(out.get(k) is None for k in ("goals", "assists", "yellow_card", "red_card")):
        ev = _af_response("/fixtures/events", {"fixture": fixture_id}, 168)
        if _goal_events_complete(ev, fixture):
            if out.get("red_card") is None:
                out["red_card"] = int(any(e.get("type") == "Card"
                                          and str(e.get("detail", "")) in ("Red Card", "Second Yellow card")
                                          and (e.get("player") or {}).get("id") == pid for e in ev))
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
                total = AFTS.team_count(blk.get("statistics"), "Shots on Goal")
        known = [r["shots_on_target"] for r in players["teams"].get(tid, [])
                 if r.get("shots_on_target") is not None]
        # The team total is fully accounted for by the players who have a value -> the rest are 0.
        if total is not None and total == sum(int(v) for v in known):
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


# ── RESULTS DECIDED BY THE NULL RULE ARE RECORDED BESIDE THE LEDGER (owner 2026-11-04, fix10) ──────
# Before fix round 10 a prop whose deciding stat the provider left null stayed pending (shots on
# target unless the team total reconciled; goals / assists / cards unless the event feed was
# complete). The shared null rule now settles it. Every such settlement is written - besides the
# normal ledger write that fills the pending row - to output/corrections/
# player_ledger_null_rule_settled.csv with the value before (pending: '') and after, the stats the
# rule read as 0 and the reason, so the rows the rule decided can always be told apart and restated.
# The event feed keeps its role for goals / assists / cards (_event_check): where the sheet's null
# would decide the result, the feed is read as before fix round 10. A feed that does not name the
# player leaves the rule's 0 standing. A COMPLETE feed (its goals add up to the final score) that
# names him decides the stat, exactly as _fill_null_stats did (that settlement is not a rule
# decision). An incomplete feed that names him contradicts the null and proves nothing, so the row
# stays pending ('rule_zero_contradicted_by_events') rather than be settled on either side.
NULL_RULE_SETTLED_FILE = config.OUTPUT_DIR / "corrections" / "player_ledger_null_rule_settled.csv"
NULL_RULE_SETTLED_COLS = ["ledger", "signal_date", "match_date", "league", "home_team", "away_team",
                          "player_name", "player_id", "fixture_id", "market", "tier",
                          "result_before", "result_after", "pnl_before", "pnl_after", "minutes",
                          "rule_zero_stats", "reason", "resolved_date"]
NULL_RULE_REASON = ("provider null read as 0 under the owner's null rule 2026-11-04 (player_stat_nulls: "
                    "player played, provider returned player statistics); was pending before fix round 10")
#: the rec stats each prop market base reads (see _resolve_player_market / card_market_result)
_MARKET_RECS = {"goals": ("goals",), "sot": ("shots_on_target",), "assists": ("assists",),
                "cards": ("yellow_card", "red_card")}
_EVENT_CHECKED = ("goals", "assists", "yellow_card", "red_card")


def _rule_decided(market: str, rec: dict, result: str) -> list[str]:
    """The stats the null rule read as 0 that DECIDED `result` (a LOSS only: a 0 can never win)."""
    import re
    if result != "LOSS":
        return []
    m = re.fullmatch(r"([a-z]+?)(\d*)", str(market or "").strip().lower())
    base = m.group(1) if m else ""
    return [k for k in _MARKET_RECS.get(base, ()) if k in (rec.get("rule_zero") or ())]


def _event_check(rec: dict, stats: list[str], fixture_id: int, fixture: dict) -> tuple[dict | None, list[str], str]:
    """The event-feed check of the goals / assists / cards nulls the rule read as 0 (`stats`).
    Returns (rec to settle on, the stats still decided by the rule, '') or (None, [], why pending).
      * the feed does not name the player for any of them -> the rule's 0 stands;
      * it names him and is COMPLETE (its goals add up to the final score) -> the stat is taken
        from the feed, exactly as before fix round 10 (_fill_null_stats), not from the rule;
      * it names him but is not complete -> the sheet and the feed disagree and neither is proven:
        the row stays pending.
    One cached /fixtures/events call, made only when such a null decides the result."""
    want = [k for k in stats if k in _EVENT_CHECKED]
    if not want:
        return rec, list(stats), ""
    pid = int(rec["player_id"])
    ev = _af_response("/fixtures/events", {"fixture": fixture_id}, 168)
    who = lambda e, k="player": (e.get(k) or {}).get("id") == pid        # noqa: E731
    scored = [e for e in ev if e.get("type") == "Goal" and e.get("detail") not in ("Own Goal", "Missed Penalty")]
    cards = [str(e.get("detail", "")) for e in ev if e.get("type") == "Card" and who(e)]
    feed = {"goals": sum(1 for e in scored if who(e)),
            "assists": sum(1 for e in scored if who(e, "assist")),
            "yellow_card": int(any(d.startswith("Yellow") for d in cards)),
            "red_card": int(any(d in ("Red Card", "Second Yellow card") for d in cards))}
    named = [k for k in want if feed[k]]
    if not named:
        return rec, list(stats), ""
    if not _goal_events_complete(ev, fixture):
        return None, [], f"sheet null read as 0 but the (incomplete) event feed names the player for {named}"
    out = dict(rec)
    for k in named:
        out[k] = feed[k]
    return out, [k for k in stats if k not in named], ""


def record_null_rule_settlements(ledger_name: str, frame: pd.DataFrame, updates: list[dict],
                                 dst: Path | None = None) -> int:
    """Append the updates the null rule decided (u['rule_zero_stats']) to NULL_RULE_SETTLED_FILE,
    one row per ledger row and result (a re-run never duplicates). Returns rows appended. The
    default file sits in the corrections folder BESIDE the ledger being settled (PLAYER_LEDGER_FILE's
    folder: output/corrections/ in production)."""
    dst = dst or (Path(PLAYER_LEDGER_FILE).parent / "corrections" / NULL_RULE_SETTLED_FILE.name)
    rows = []
    for u in updates:
        if not u.get("rule_zero_stats"):
            continue
        r = frame.loc[u["idx"]]
        g = lambda c: "" if c not in r.index or pd.isna(r[c]) else str(r[c])   # noqa: E731
        rows.append({"ledger": ledger_name, **{c: g(c) for c in (
            "signal_date", "match_date", "league", "home_team", "away_team", "player_name",
            "player_id", "fixture_id", "market", "tier")},
            "fixture_id": str(u.get("fixture_id") or g("fixture_id")),
            "result_before": g("result"), "result_after": u["result"],
            "pnl_before": g("pnl"), "pnl_after": str(u["pnl"]), "minutes": str(u.get("minutes", "")),
            "rule_zero_stats": "|".join(u["rule_zero_stats"]), "reason": NULL_RULE_REASON,
            "resolved_date": u["resolved_date"]})
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=NULL_RULE_SETTLED_COLS)
    key = ["ledger", "signal_date", "match_date", "player_id", "player_name", "market", "result_after"]
    if dst.exists():
        old = pd.read_csv(dst, dtype=str, keep_default_na=False)
        seen = set(map(tuple, old[key].astype(str).to_numpy()))
        new = new[[tuple(x) not in seen for x in new[key].astype(str).to_numpy()]]
    if new.empty:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(dst, mode="a", header=not dst.exists(), index=False, lineterminator="\n")
    log.info(f"{ledger_name}: {len(new)} result(s) decided by the null rule recorded -> {dst}")
    return len(new)


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

    updates, fixture_ids, skipped = _grade_player_rows(pending, today_str, dry_run)

    if skipped:
        log.info(f"player_ledger: left pending this run - {dict(sorted(skipped.items()))}")

    if not updates and not fixture_ids:
        log.info("  No player prop results resolved.")
        return

    n_rule = sum(1 for u in updates if u.get("rule_zero_stats"))
    if dry_run:
        voids   = sum(1 for u in updates if u["result"] == "VOID")
        wins    = sum(1 for u in updates if u["result"] == "WIN")
        losses  = sum(1 for u in updates if u["result"] == "LOSS")
        print(f"\n  [DRY RUN] Would update {len(updates)} player prop row(s): "
              f"{wins}W / {losses}L / {voids} VOID ({n_rule} decided by the null rule), and write "
              f"{len(fixture_ids)} fixture_id(s). No files written.")
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
    # `pending` is a copy taken before the writes above: it holds the values before.
    record_null_rule_settlements("output/player_ledger.csv", pending, updates)

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


# ── THE 111 RESTORED PLAYER ROWS ARE GRADED IN THEIR SIDECAR (round 5, owner decision) ─────────────
# output/corrections/player_ledger_restored_2026-11.csv holds tips lost from player_ledger.csv by bot
# commits (scripts/check_ledger_history.py --restore), with their first-sight values. They are never
# appended to the ledger. They are graded HERE, by the very same per-row rules as the ledger
# (_grade_player_rows), into new `graded_*` columns beside the first-sight ones - the restored
# `result` / `pnl` / `notes` columns keep the values first seen and are never rewritten.
RESTORED_PLAYER_FILE = config.OUTPUT_DIR / "corrections" / "player_ledger_restored_2026-11.csv"
_RESTORED_GRADE_COLS = ("graded_is_played", "graded_result", "graded_pnl", "graded_notes",
                        "graded_resolved_date", "graded_fixture_id")


def grade_restored_player_sidecar(path: Path | None = None, dry_run: bool = False) -> dict:
    """Grade the restored rows that have no graded_result yet; write only the graded_* columns."""
    import os
    p = path or RESTORED_PLAYER_FILE
    if not p.exists():
        return {"graded": 0}
    raw = p.read_bytes()
    eol = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    side = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in _RESTORED_GRADE_COLS:
        if c not in side.columns:
            side[c] = ""
    if not os.getenv("APIFOOTBALL_KEY", ""):
        log.info("restored player rows: APIFOOTBALL_KEY not set — not graded this run")
        return {"graded": 0}
    today_str = str(datetime.utcnow().date())
    pending = side[(side["graded_result"].str.strip() == "") & (side["match_date"] < today_str)]
    if pending.empty:
        return {"graded": 0}
    # The grader reads the row's ledger columns; the restored file carries every one of them.
    updates, fixture_ids, skipped = _grade_player_rows(pending, today_str, dry_run)
    if skipped:
        log.info(f"restored player rows: left pending - {dict(sorted(skipped.items()))}")
    if dry_run or (not updates and not fixture_ids):
        return {"graded": 0, "would_grade": len(updates), "pending": skipped}
    for i, fid in fixture_ids.items():
        side.at[i, "graded_fixture_id"] = str(fid)
    for u in updates:
        side.at[u["idx"], "graded_is_played"] = u["is_played"]
        side.at[u["idx"], "graded_result"] = u["result"]
        side.at[u["idx"], "graded_pnl"] = str(u["pnl"])
        # This file is itself the beside-record (graded_* next to the first-sight values), and its
        # own CI step writes it alone: a result the null rule decided is marked in graded_notes.
        side.at[u["idx"], "graded_notes"] = "; ".join(x for x in (
            u.get("notes", ""),
            f"null rule 2026-11-04: {'|'.join(u['rule_zero_stats'])} null read as 0"
            if u.get("rule_zero_stats") else "") if x)
        side.at[u["idx"], "graded_resolved_date"] = u["resolved_date"]
    side.to_csv(p, index=False, lineterminator=eol)
    log.info(f"restored player rows: {len(updates)} graded beside the first-sight values -> {p}")
    return {"graded": len(updates), "pending": skipped}


def _grade_player_rows(pending: pd.DataFrame, today_str: str,
                       dry_run: bool = False) -> tuple[list[dict], dict, dict]:
    """THE per-row player-prop grading rules (fixture, status, 48h rule, match sheet, appearance,
    90 minutes, price). Returns (updates, fixture_ids found for rows without one, skip reasons).
    Shared by player_ledger.csv and the restored-rows sidecar."""
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
                fixture_cache[ck] = _find_prop_fixture_near(league, league_id, date_str, home, away)
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
            # 48-hour rule (owner 2026-11-02): PST / ABD void only once the window has closed.
            if status != "CANC" and not _window_closed(date_str, today_str):
                _skip("postponed_within_48h_window", f"{what}: fixture {fid} status {status}")
                continue
            label = _VOID_STATUSES[status]
            log.info(f"  {home} vs {away} ({date_str}) — {label}, marking VOID")
            updates.append({"idx": idx, "is_played": "False", "result": "VOID",
                            "pnl": "0.0", "notes": label, "resolved_date": today_str})
            continue
        if status not in _FT_STATUSES:
            _skip("not_finished", f"{what}: fixture {fid} status {status!r}")
            continue
        # The stored fixture id is the match's identity: when API-Football played it more than 48
        # hours after the date the tip was logged for, it was rescheduled -> VOID (owner rule).
        played_on = str((fixture.get("fixture") or {}).get("date") or "")[:10]
        if played_on and _window_closed(date_str, played_on):
            label = f"RESCHEDULED (played {played_on}, >48h after {date_str})"
            log.info(f"  {what}: {label} (VOID)")
            updates.append({"idx": idx, "is_played": "False", "result": "VOID",
                            "pnl": "0.0", "notes": label, "resolved_date": today_str})
            continue

        # ── Step 2: the fixture's player sheet ───────────────────────────────
        if fid not in players_cache:
            players_cache[fid] = _fixture_players(fid)
        players = players_cache[fid]

        # NO STATS FOR THE WHOLE FIXTURE IS NOT "DID NOT PLAY". An API error, an empty body or stats
        # not yet published all look like this, and until 2026-11-01 every prop on such a fixture
        # was voided as DNP for good (Data Truth audit finding 16). Leave it pending and retry.
        # NEVER VOIDED ON AGE (estate scan P2, 2026-11-02): this used to void the row after 7 days
        # ('NO_STATS_7D') although the fixture had FINISHED - the match was played, only the
        # provider's sheet is missing. Age is not evidence; the row stays pending, counted below.
        if not players["by_id"]:
            _age = (pd.Timestamp(today_str) - pd.Timestamp(date_str)).days
            _skip("no_player_stats" + ("_7d_plus" if _age >= 7 else "_yet"), f"{what} ({_age} days)")
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

        # ── Step 5: resolve WIN/LOSS - on 90 minutes (extra time ignored) ─────
        if status in _EXTRA_TIME_STATUSES:
            reg, why_reg = _regulation_stats(rec, fid, fixture)
            if reg is None:
                _skip("extra_time_regulation_unknown", f"{what}: {why_reg}")
                continue
            result = _resolve_player_market(market, reg)
        else:
            result = _resolve_player_market(market, rec)
            if not result:
                result = _resolve_player_market(market, _fill_null_stats(rec, fid, fixture, players))
        if not result:
            _skip("stat_unknown", f"{what}: market not settleable or stat not published")
            continue
        # The extra-time path rebuilds goals / assists / cards from the event feed (no rule zero
        # there); only its shots-on-target bound comes from the sheet under the rule.
        if status in _EXTRA_TIME_STATUSES:
            rule_zero = ["shots_on_target"] if (result == "LOSS" and str(market).startswith("sot")
                                                and "shots_on_target" in (rec.get("rule_zero") or ())) else []
        else:
            rule_zero = _rule_decided(market, rec, result)
            if rule_zero:
                checked, rule_zero, clash = _event_check(rec, rule_zero, fid, fixture)
                if checked is None:
                    _skip("rule_zero_contradicted_by_events", f"{what}: {clash}")
                    continue
                if checked is not rec:                  # the complete feed named him: it decides
                    result = _resolve_player_market(market, checked)
                    rule_zero = _rule_decided(market, {**checked, "rule_zero": rule_zero}, result)
                    if not result:
                        _skip("stat_unknown", f"{what}: market not settleable on the event feed")
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
                        "pnl": pnl, "notes": note, "resolved_date": today_str,
                        "rule_zero_stats": rule_zero, "minutes": minutes, "fixture_id": fid})
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

    return updates, fixture_ids, skipped


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


# ── SHARP P/L IS BOOKED AT THE SIGNAL-TIME PRICE (round 2 V1, 2026-11-02) ─────────────────────────
# Every sharp signal is a price DROP: opening_odds > current_odds on 100% of the 2,010 settled rows.
# The grader booked P/L at opening_odds, a price that was already gone when the signal fired, so the
# record showed +22.22u (+1.1%) where the price a follower could actually take gives -159.14u (-7.9%).
# current_odds is the price at signal time and is frozen: player_model.ledger.append_sharp_signals
# writes it once and never updates an existing key (0 of 2,302 keys changed it across all 678
# commits of sharp_ledger.csv). Rows settled from this change on carry pnl_basis='signal_price';
# rows settled before it (pnl_basis blank, booked at opening_odds) are NOT rewritten - their
# restated P/L is the sidecar output/corrections/sharp_ledger_restated.csv (restate_sharp_ledger).
SHARP_PNL_BASIS = "signal_price"
SHARP_RESTATED_FILE = config.OUTPUT_DIR / "corrections" / "sharp_ledger_restated.csv"


def _sharp_entry_odds(row) -> float:
    """The signal-time price of a sharp row (current_odds), NaN when absent. Never opening_odds."""
    try:
        v = float(row.get("current_odds"))
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def restate_sharp_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Sidecar rows for the sharp rows booked at opening_odds: every settled WIN/LOSS row without
    pnl_basis. The ledger is not touched; original pnl and odds are kept beside the restatement."""
    d = ledger.copy()
    if "pnl_basis" not in d.columns:
        d["pnl_basis"] = ""
    d = d[d["result"].isin(["WIN", "LOSS"]) & (d["pnl_basis"].fillna("").astype(str).str.strip() == "")]
    rows = []
    for _, r in d.iterrows():
        entry = _sharp_entry_odds(r)
        if entry > 1.0:
            restated = round(entry - 1.0, 4) if r["result"] == "WIN" else -1.0
            note = "booked at opening_odds (gone at signal time); restated at current_odds (signal-time price)"
        else:
            restated = np.nan
            note = "no signal-time price recorded: restated P/L left empty"
        rows.append({"signal_date": r["signal_date"], "match_date": r["match_date"],
                     "league": r["league"], "home_team": r["home_team"], "away_team": r["away_team"],
                     "side": r["side"], "signal": r.get("signal", ""), "result": r["result"],
                     "original_odds": r["opening_odds"], "original_pnl": r["pnl"],
                     "restated_odds": r.get("current_odds", ""),
                     "restated_pnl": "" if not np.isfinite(restated) else restated,
                     "note": note})
    return pd.DataFrame(rows, columns=["signal_date", "match_date", "league", "home_team",
                                       "away_team", "side", "signal", "result", "original_odds",
                                       "original_pnl", "restated_odds", "restated_pnl", "note"])


def write_sharp_restatement(src: Path | None = None, dst: Path | None = None) -> pd.DataFrame:
    """Write output/corrections/sharp_ledger_restated.csv from the committed sharp ledger."""
    src, dst = src or SHARP_LEDGER_FILE, dst or SHARP_RESTATED_FILE
    out = restate_sharp_ledger(pd.read_csv(src, dtype=str, keep_default_na=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, lineterminator="\n")
    o = pd.to_numeric(out["original_pnl"], errors="coerce")
    n = pd.to_numeric(out["restated_pnl"], errors="coerce")
    log.info(f"sharp restatement: {len(out)} row(s); P/L {o.sum():+.2f}u as booked -> "
             f"{n.sum():+.2f}u at the signal-time price -> {dst}")
    return out


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
    reasons: dict[str, int] = {}
    left: list[str] = []

    for idx, row in pending.iterrows():
        league   = row["league"]
        home     = row["home_team"]
        away     = row["away_team"]
        date_str = row["match_date"]
        side     = row["side"]

        if not side:
            log.debug(f"  sharp: no side parsed for {row.get('market_label', '')} — skipping")
            reasons["no_side"] = reasons.get("no_side", 0) + 1
            left.append(date_str)
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

        # LAST RESORT: API-Football, as the O/U and side-market paths already do. Without it a sharp
        # tip in a league with no football-data feed (MLS, Argentina, Brazil ...) could only settle
        # inside OddsAPI's 3-day scores window: 96 of the 164 sharp rows pending > 7 days on
        # 2026-11-02 were MLS. Lazy, so a healthy run costs no calls.
        # The primary score goes through the knockout guard (90 minutes, round 5).
        ev, held = _team_event(league, home, away, date_str, [scores_cache.get(league, [])])
        match_result = _resolve_sharp_side(side, ev["home_score"], ev["away_score"]) if ev else None

        if not match_result:
            if held:
                why = held
            elif ev is None:
                label, why = _postponement_evidence(league, date_str, home, away)
                if label:
                    log.info(f"  sharp: {home} vs {away} {date_str} — {label} (API-Football), marking VOID")
                    updates.append({"idx": idx, "result": "VOID", "pnl": 0.0, "basis": ""})
                    continue
            else:
                why = f"side {side!r} not gradable"
            reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
            left.append(date_str)
            log.debug(f"  sharp: {home} vs {away} {date_str} pending: {why}")
            continue

        # No usable entry price -> the RESULT is graded, the P/L is not. This defaulted to odds 1.0,
        # so a WIN was booked at 0.0 with no flag (Data Truth audit finding 27).
        entry = _sharp_entry_odds(row)
        if not entry > 1.0:
            pnl = np.nan
            log.warning(f"  sharp: {home} vs {away} {date_str} has no signal-time price — result kept, P/L left empty")
        else:
            pnl = round(entry - 1.0, 4) if match_result == "WIN" else -1.0

        updates.append({"idx": idx, "result": match_result, "pnl": pnl,
                        "basis": SHARP_PNL_BASIS if entry > 1.0 else ""})
        log.info(
            f"  {'[DRY]' if dry_run else '[ OK]'} "
            f"{home} vs {away} ({side}) {date_str} → {match_result}  PnL={pnl:+.3f}u"
        )

    _log_pending("sharp_ledger", reasons, today_str, left)
    if not updates:
        log.info("  No sharp money results found in any scores source.")
        return

    if dry_run:
        print(f"\n  [DRY RUN] Would update {len(updates)} sharp row(s) "
              f"({sum(1 for u in updates if u['result'] == 'VOID')} VOID). No files written.")
        return

    resolved_today = str(datetime.utcnow().date())
    if "pnl_basis" not in ledger.columns:
        ledger["pnl_basis"] = ""
    ledger["pnl_basis"] = ledger["pnl_basis"].astype(object)
    for u in updates:
        ledger.at[u["idx"], "result"]        = u["result"]
        # An unpriced row's P/L stays EMPTY, not the text 'nan'.
        ledger.at[u["idx"], "pnl"]           = "" if np.isnan(u["pnl"]) else str(u["pnl"])
        ledger.at[u["idx"], "resolved_date"] = resolved_today
        ledger.at[u["idx"], "pnl_basis"]     = u.get("basis", "")

    ledger.to_csv(SHARP_LEDGER_FILE, index=False)

    wins      = sum(1 for u in updates if u["result"] == "WIN")
    losses    = sum(1 for u in updates if u["result"] == "LOSS")
    voids     = sum(1 for u in updates if u["result"] == "VOID")
    priced    = [u["pnl"] for u in updates if u["result"] != "VOID" and not np.isnan(u["pnl"])]
    log.info(f"sharp_ledger: {len(updates)} resolved — {wins}W/{losses}L/{voids} VOID — "
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
    reasons: dict[str, int] = {}
    left: list[str] = []

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
        # The primary score goes through the knockout guard (90 minutes, round 5).
        ev_found, held = _team_event(league, home, away, date_str, [scores_cache.get(league, [])])
        if ev_found is None:
            label, why = (None, held) if held else _postponement_evidence(league, date_str, home, away)
            if label:
                log.info(f"  side {market}: {home} vs {away} {date_str} — {label} (API-Football), marking VOID")
                updates.append({"idx": idx, "result": "VOID", "pnl": 0.0, "closing_odds": np.nan,
                                "clv_pct": np.nan, "notes": label})
                continue
            reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
            left.append(date_str)
            log.debug(f"  side {market}: {home} vs {away} {date_str} pending: {why}")
            continue

        res = _resolve_side_market(market, ev_found["home_score"], ev_found["away_score"])
        if not res:
            reasons["market_not_gradable"] = reasons.get("market_not_gradable", 0) + 1
            left.append(date_str)
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

    _log_pending("side_bets_ledger", reasons, today_str, left)
    if not updates:
        log.info("  No side-market results found in any scores source.")
        return
    if dry_run:
        print(f"\n  [DRY RUN] Would update {len(updates)} side-market row(s) "
              f"({sum(1 for u in updates if u['result'] == 'VOID')} VOID). No files written.")
        return

    if "notes" not in ledger.columns:
        ledger["notes"] = ""
    for u in updates:
        ledger.at[u["idx"], "result"] = u["result"]
        # An unpriced row's P/L stays EMPTY, not the text 'nan'.
        ledger.at[u["idx"], "pnl"]    = "" if np.isnan(u["pnl"]) else str(u["pnl"])
        if not np.isnan(u["closing_odds"]):
            ledger.at[u["idx"], "closing_odds"] = str(u["closing_odds"])
        if not np.isnan(u["clv_pct"]):
            ledger.at[u["idx"], "clv_pct"] = str(u["clv_pct"])
        if u.get("notes"):
            ledger.at[u["idx"], "notes"] = _join_note(ledger.at[u["idx"], "notes"], u["notes"])
    ledger.to_csv(SIDE_LEDGER_FILE, index=False)

    wins   = sum(1 for u in updates if u["result"] == "WIN")
    losses = sum(1 for u in updates if u["result"] == "LOSS")
    voids  = sum(1 for u in updates if u["result"] == "VOID")
    priced = [u["pnl"] for u in updates if u["result"] != "VOID" and not np.isnan(u["pnl"])]
    log.info(f"side_bets_ledger: {len(updates)} resolved ({voids} VOID) — {wins}W/{losses}L — "
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


def _read_fd_ht(path: Path, tag: str) -> pd.DataFrame:
    """fd_history's half-time scores as the loader serves the history (fix16/readers): the owner's
    corrections through src.history_corrections.serve_history - proven cup / awarded rows and
    duplicate copies out (a removed copy's HT fills the kept copy's EMPTY cell, the same match),
    one-day-off rows on the provider's date. The full-time score is read only so a listed row is
    recognised. Columns: league, date, home_team, away_team, hh, ha."""
    from src.history_corrections import SERVE_COLS, serve_history
    d = pd.read_parquet(path, columns=SERVE_COLS + ["ht_home_goals", "ht_away_goals"])
    d = serve_history(d, tag=tag)
    return d[["league", "date", "home_team", "away_team", "ht_home_goals", "ht_away_goals"]].rename(
        columns={"ht_home_goals": "hh", "ht_away_goals": "ha"})


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
            if path.name == "fd_history.parquet":
                frames.append(_read_fd_ht(path, "ht_observations.fd_history"))
                continue
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
            _frames.append(_read_fd_ht(_fdp, "ht_ledger.fd_history"))
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
        # The committed stores through the same matcher as every other grader: exact date, then
        # the 48-hour reschedule window; exactly one fixture and one score, or nothing.
        lk = _canon_league(league)
        pool = []
        for k in range(_RESCHEDULE_DAYS[0], _RESCHEDULE_DAYS[1] + 1):
            dk = str((pd.Timestamp(day) + pd.Timedelta(days=k)).date())
            g = _by_day.get((dk, lk))
            if g is not None and not g.empty:
                pool += [{"home_team": str(h), "away_team": str(a), "home_score": 0, "away_score": 0,
                          "ht_home": hh, "ht_away": ha, "date_str": dk}
                         for h, a, hh, ha in zip(g["home_team"], g["away_team"], g["hh"], g["ha"])]
        ev = _find_event(home, away, pool, day)
        if ev is not None:
            return (ev["ht_home"], ev["ht_away"])
        # LAST RESORT: API-Football's completed fixtures for THIS match's season carry HT scores.
        # Without it a fixture missing from both committed stores stayed pending for ever
        # (Data Truth audit finding 24). Same rules: names resolved in that day's whole pool (a
        # one-name pool let any token-subset pair through), exactly one match.
        ev = _find_event(home, away, fetch_scores_apifootball(str(league), day), day)
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
    reasons: dict[str, int] = {}
    left: list[str] = []
    for i in pending.index:
        home = led.at[i, "home_team"]; away = led.at[i, "away_team"]
        d = str(led.at[i, "match_date"])[:10]; mkt = str(led.at[i, "market"])
        sc = _ht_score(led.at[i, "league"], home, away, d)
        if sc is None:
            label, why = _postponement_evidence(led.at[i, "league"], d, home, away)
            if label:
                log.info(f"  ht {mkt}: {home} vs {away} {d} — {label} (API-Football), marking VOID")
                led.at[i, "result"] = "VOID"
                led.at[i, "pnl"] = 0.0
                led.at[i, "notes"] = _join_note(led.at[i, "notes"], label)
                graded += 1
                continue
            reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
            left.append(d)
            continue
        res = _resolve_ht(mkt, sc[0], sc[1])
        if not res:
            reasons["ht_score_incomplete"] = reasons.get("ht_score_incomplete", 0) + 1
            left.append(d)
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

    _log_pending("ht_ledger", reasons, today, left)
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
            # Knockout guard (round 5): final_total and full-time sides are on 90 minutes.
            ev, held = _knockout_guard(league, home, away, ev["date_str"], ev)
            if held:
                log.debug(f"  live: {home} vs {away} ({d}) pending: {held}")
                continue
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

    guard = None
    if providers is None:
        guard = _knockout_guard                    # 90 minutes: AET / PEN from API-Football

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
                                    identity=_shadow_identity(), now=now,
                                    extra_time_guard=guard)
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
    parser.add_argument("--restate-sharp", action="store_true",
                        help="ONLY write the settlement sidecars under output/corrections/ "
                             "(sharp restatement, card regrade); offline, ledgers untouched")
    parser.add_argument("--restate-extra-time", action="store_true",
                        help="ONLY write output/corrections/team_ledgers_extra_time_restated.csv "
                             "(settled rows on AET/PEN fixtures re-graded on 90 minutes; needs "
                             "API-Football - run it in the collection job; ledgers untouched)")
    parser.add_argument("--grade-restored", action="store_true",
                        help="ONLY grade output/corrections/player_ledger_restored_2026-11.csv into its "
                             "graded_* columns (needs API-Football; player_ledger.csv untouched)")
    args = parser.parse_args()

    # Corrections sidecars: explicit, never part of the 2-hourly run (they need API-Football calls
    # and the settlement workflow does not commit output/corrections/). Combinable.
    if args.restate_extra_time or args.grade_restored:
        if args.grade_restored:
            grade_restored_player_sidecar(dry_run=args.dry_run)
        if args.restate_extra_time:
            write_extra_time_restatement()
        return

    if args.restate_sharp:
        write_sharp_restatement()
        write_card_regrade()
        return
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

    # Plan this run's API-Football status lookups (postponement evidence) across every ledger that
    # can void on it, so the per-run budget is shared and rotates (see _plan_status_checks).
    global _STATUS_ALLOWED
    _STATUS_ALLOWED = _plan_status_checks(_pending_frames_for_status(today_str), today_str)

    # Match tips to results
    filled = 0
    updates: list[dict] = []
    voids: list[dict] = []          # postponed / cancelled / abandoned on API-Football evidence
    reasons: dict[str, int] = {}
    left: list[str] = []

    for idx, row in pending.iterrows():
        league = row["league"]
        # Primary pools (football-data, then OddsAPI) through the knockout guard (90 minutes), then
        # API-Football as the LAST RESORT - only reached when no primary source resolved the fixture,
        # so a healthy run makes no fallback call. That is what settles leagues with no football-data
        # feed once they age past OddsAPI's 3-day scores window.
        # A pool whose event cannot grade the side (an HT side with no HT score) hands on to the next.
        found, held = None, ""
        for pool in [*scores_cache.get(league, []), None]:
            if pool is None:
                ev, held = _team_event(league, row["home_team"], row["away_team"],
                                       row["match_date"], [])
            else:
                ev, held = _team_event(league, row["home_team"], row["away_team"],
                                       row["match_date"], [pool], fallback=False)
                if ev is None and not held:
                    continue
            if held:
                break
            found = _grade_ou_event(ev, row["side"])
            if found is not None:
                break
        if held:
            reasons[held.split(" (")[0]] = reasons.get(held.split(" (")[0], 0) + 1
            left.append(row["match_date"])
            log.debug(f"  pending: {row['home_team']} vs {row['away_team']} ({row['match_date']}): {held}")
            continue

        if found is None:
            label, why = _postponement_evidence(league, str(row["match_date"])[:10],
                                                row["home_team"], row["away_team"])
            if label:
                log.info(f"  {row['home_team']} vs {row['away_team']} ({row['match_date']}) — "
                         f"{label} (API-Football), marking VOID")
                voids.append({"idx": idx, "label": label})
            else:
                reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
                left.append(row["match_date"])
                log.debug(f"  pending: {row['home_team']} vs {row['away_team']} ({row['match_date']}): {why}")
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

    _log_pending("bets_ledger", reasons, today_str, left)
    if not updates and not voids:
        log.info("No pending matches found in any scores source. If matches were recently played, try --days with a larger value or re-run after the local CSVs are refreshed.")
    elif args.dry_run:
        print(f"\n  [DRY RUN] Would update {filled} row(s) and void {len(voids)} "
              f"(postponed/cancelled/abandoned on API-Football evidence). No files written.")

    # Write back to ledger
    if voids and not args.dry_run:
        if "notes" not in ledger.columns:
            ledger["notes"] = ""
        ledger["notes"] = ledger["notes"].astype(object)
        for v in voids:
            ledger.at[v["idx"], "result"] = "VOID"
            ledger.at[v["idx"], "pnl"] = "0.0"
            ledger.at[v["idx"], "notes"] = _join_note(ledger.at[v["idx"], "notes"], v["label"])
        if not updates:
            ledger.to_csv(LEDGER_FILE, index=False)
        log.info(f"bets_ledger: {len(voids)} row(s) VOID on API-Football evidence "
                 f"({', '.join(sorted({v['label'] for v in voids}))})")
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
    if _KNOCKOUT_UNCHECKED:
        log.info(f"knockout guard: primary-source scores settled WITHOUT an AET/PEN check "
                 f"(no source to check against) - {dict(sorted(_KNOCKOUT_UNCHECKED.items()))}")
    # LAST and guarded, like ht_observations: an evidence log must never take a ledger down.
    try:
        settle_shadow_predictions(days=args.days, dry_run=args.dry_run,
                                  journal=args.shadow_journal)
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"shadow settlement skipped: {e}")


if __name__ == "__main__":
    main()
