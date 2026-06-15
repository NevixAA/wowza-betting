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


# ── Team name normalisation ───────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase + strip accents for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()


def _names_match(a: str, b: str) -> bool:
    """True if two team names refer to the same club (normalised exact or substring)."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    # substring both ways (handles "Man City" vs "Manchester City" style)
    if na in nb or nb in na:
        return True
    # first-word match for short names
    if na.split()[0] == nb.split()[0] and len(na.split()[0]) >= 4:
        return True
    return False


# ── football-data.co.uk scores fetch ─────────────────────────────────────────
# "new" format: football-data.co.uk/new/{code}.csv  (cols: Home/Away/HG/AG/Date, no HT)
# "std" format: football-data.co.uk/mmz4281/2526/{code}.csv (cols: HomeTeam/AwayTeam/FTHG/FTAG/HTHG/HTAG/Date)
# Tuple: (fmt, code, home_col, away_col, ft_hg, ft_ag, date_col, ht_hg, ht_ag)

_FD_SOURCES = {
    # new format — no HT columns available
    "Austrian Bundesliga":      ("new", "AUT", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "Sweden Allsvenskan":       ("new", "SWE", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "Denmark Superliga":        ("new", "DNK", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "Japan J-League":           ("new", "JPN", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "USA MLS":                  ("new", "USA", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "China Super League":       ("new", "CHN", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    "Ireland Premier Division": ("new", "IRL", "Home", "Away", "HG",   "AG",   "Date", None,   None),
    # standard format — includes HTHG/HTAG (half-time scores)
    "Bundesliga 2":             ("std", "D2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "La Liga 2":                ("std", "SP2", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "Ligue 2":                  ("std", "F2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "League One":               ("std", "E2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "League Two":               ("std", "E3",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "Greek Super League":       ("std", "G1",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
    "Belgian First Division A": ("std", "B1",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "HTHG", "HTAG"),
}

_FD_NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
_FD_STD_URL = "https://www.football-data.co.uk/mmz4281/2526/{code}.csv"

def fetch_scores_fd(league: str) -> list[dict]:
    """Fetch completed results from football-data.co.uk (free, no quota). Includes HT scores where available."""
    if league not in _FD_SOURCES:
        return []

    fmt, code, home_col, away_col, hg_col, ag_col, date_col, ht_hg_col, ht_ag_col = _FD_SOURCES[league]

    try:
        from io import StringIO

        url = _FD_NEW_URL.format(code=code) if fmt == "new" else _FD_STD_URL.format(code=code)
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log.debug(f"  FD {league}: HTTP {r.status_code}")
            return []

        raw = pd.read_csv(StringIO(r.text), encoding="utf-8-sig",
                          on_bad_lines="skip", low_memory=False)

        if fmt == "new" and "Season" in raw.columns:
            raw = raw[raw["Season"].astype(str).str.contains("2025|2026")].copy()

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

            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if home not in score_map or away not in score_map:
                continue

            dt_raw = event.get("commence_time", "")
            try:
                dt = pd.to_datetime(dt_raw, utc=True).tz_localize(None)
            except Exception:
                continue

            results.append({
                "home_team":   home,
                "away_team":   away,
                "home_score":  score_map[home],
                "away_score":  score_map[away],
                "date_str":    str(dt.date()),
            })
        return results

    except Exception as e:
        log.debug(f"  OddsAPI scores {sport_key} exception: {e}")
        return []


# ── CLV lookup ───────────────────────────────────────────────────────────────

def _closing_odds(home: str, away: str, match_date: str, side: str) -> float:
    """
    Return the LAST recorded odds snapshot before kick-off from odds_history_v9.json.
    This approximates the closing line.
    """
    if not config.ODDS_HISTORY_JSON.exists():
        return np.nan
    try:
        history = json.loads(config.ODDS_HISTORY_JSON.read_text(encoding="utf-8"))
        key = f"{home} vs {away} | {match_date}"
        snapshots = history.get(key, [])
        if not snapshots:
            # Try normalised team names
            norm_key = next(
                (k for k in history
                 if _norm(k.split(" vs ")[0]) == _norm(home)
                 and len(k.split(" vs ")) > 1
                 and _norm(k.split(" vs ")[1].split(" | ")[0]) == _norm(away)),
                None,
            )
            snapshots = history.get(norm_key, []) if norm_key else []
        if not snapshots:
            return np.nan
        last = snapshots[-1]
        field = "under" if side == "UNDER" else "over"
        val = last.get(field)
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan


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
    for ev in completed:
        if ev["date_str"] != date_str:
            continue
        if not (_names_match(home, ev["home_team"]) and _names_match(away, ev["away_team"])):
            continue

        total = ev["home_score"] + ev["away_score"]
        ht_total = ev.get("ht_home", -1) + ev.get("ht_away", -1) if "ht_home" in ev else None

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
            extras["ht_score"] = f"{ev.get('ht_home',0)}-{ev.get('ht_away',0)}"
            extras["ht_total"] = ht_total
        return result, total, extras

    return None


# ── Player prop result resolver ───────────────────────────────────────────────

def _resolve_player_market(market: str, stats: dict) -> str:
    """Determine WIN/LOSS for a player prop market given their fixture stats."""
    if market == "goals":
        return "WIN" if int(stats.get("goals", 0)) >= 1 else "LOSS"
    if market == "sot":
        return "WIN" if int(stats.get("shots_on_target", 0)) >= 1 else "LOSS"
    if market == "cards":
        return "WIN" if int(stats.get("yellow_card", 0)) >= 1 else "LOSS"
    if market == "assists":
        return "WIN" if int(stats.get("assists", 0)) >= 1 else "LOSS"
    return ""


def _fetch_fixture_player_stats(league_id: int, season: str, date_str: str, home: str, away: str) -> dict[str, dict]:
    """
    Returns {norm(player_name): {goals, shots_on_target, yellow_card, assists}}
    for one match. Uses API-Football (APIFOOTBALL_KEY env var required).
    """
    import os
    api_key = os.getenv("APIFOOTBALL_KEY", "")
    if not api_key:
        return {}

    try:
        from player_model.api_football import find_fixture_id, get_fixture_player_stats
    except ImportError:
        log.debug("player_model not importable — cannot resolve player results")
        return {}

    fixture_id = find_fixture_id(league_id, season, date_str, home, away)
    if not fixture_id:
        log.debug(f"  fixture not found: {home} vs {away} {date_str}")
        return {}

    raw = get_fixture_player_stats(fixture_id)
    return {_norm(p["player_name"]): p for p in raw if p.get("player_name")}


def update_player_results(days: int = 3, dry_run: bool = False) -> None:
    """Resolve player prop outcomes in player_ledger.csv using API-Football."""
    if not PLAYER_LEDGER_FILE.exists():
        log.info("player_ledger.csv not found — skipping player prop resolution")
        return

    ledger = pd.read_csv(PLAYER_LEDGER_FILE, dtype=str)

    # Add new columns if missing (backward compat with ledgers written before this version)
    for col in ("is_played", "notes"):
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

    try:
        from player_model.config import PROP_LEAGUES
        from player_model.api_football import find_fixture_id, get_fixture_player_stats, get_fixture_status
    except ImportError:
        log.warning("player_model not importable — cannot resolve player results")
        return

    import os
    if not os.getenv("APIFOOTBALL_KEY", ""):
        log.info("APIFOOTBALL_KEY not set — skipping player prop resolution")
        return

    # Cache: fixture_id and full stats per (home, away, date)
    fixture_cache: dict[str, int | None]   = {}  # cache_key → fixture_id (None = not FT)
    stats_cache:   dict[str, dict]         = {}  # cache_key → {norm_name: stats}
    updates: list[dict] = []

    for idx, row in pending.iterrows():
        home      = row["home_team"]
        away      = row["away_team"]
        date_str  = row["match_date"]
        league    = row["league"]
        market    = row["market"]
        player    = row["player_name"]
        cache_key = f"{home}|{away}|{date_str}"

        league_id = PROP_LEAGUES.get(league)
        if not league_id:
            log.debug(f"  {league}: no PROP_LEAGUES entry — skipping")
            continue

        yr     = int(date_str[:4])
        # Single-year tournaments (WC=1, Euros=4, Copa America=9) use the match year directly.
        # Club leagues that span two calendar years use yr-1 for Jan-Jun matches.
        _SINGLE_YEAR_LEAGUES = {1, 4, 9}
        season = str(yr) if league_id in _SINGLE_YEAR_LEAGUES else (
            str(yr - 1) if int(date_str[5:7]) < 7 else str(yr)
        )

        # ── Step 1: look up fixture as FT (completed) ─────────────────────────
        if cache_key not in fixture_cache:
            fixture_cache[cache_key] = find_fixture_id(league_id, season, date_str, home, away)

        fixture_id = fixture_cache[cache_key]

        if fixture_id is None:
            # Fixture not found as FT — check actual status if match is ≥1 day old
            days_since = (datetime.utcnow().date() - pd.to_datetime(date_str).date()).days
            if days_since >= 1:
                status, _ = get_fixture_status(league_id, season, date_str, home, away)
                if status in ("PST", "CANC", "ABD"):
                    label = {"PST": "POSTPONED", "CANC": "CANCELLED", "ABD": "ABANDONED"}.get(status, status)
                    log.info(f"  {home} vs {away} ({date_str}) — {label}, marking VOID")
                    updates.append({"idx": idx, "is_played": "False", "result": "VOID",
                                    "pnl": "0.0", "notes": label, "resolved_date": today_str})
                else:
                    log.debug(f"  {home} vs {away} ({date_str}) — status={status}, leaving pending")
            continue

        # ── Step 2: fetch all player stats for this fixture ───────────────────
        if cache_key not in stats_cache:
            raw = get_fixture_player_stats(fixture_id)
            stats_cache[cache_key] = {_norm(p["player_name"]): p for p in raw if p.get("player_name")}

        stats_map = stats_cache[cache_key]

        # ── Step 3: find this player in the stats ─────────────────────────────
        player_key = _norm(player)
        stats = stats_map.get(player_key)
        if stats is None:
            # Fuzzy fallback: first-5-chars match
            for k, v in stats_map.items():
                if len(player_key) >= 5 and len(k) >= 5 and (player_key[:5] in k or k[:5] in player_key):
                    stats = v
                    break

        if stats is None:
            # Player not in fixture stats at all — did not play (DNP)
            log.info(f"  {player} not found in stats for {home} vs {away} — DNP (VOID)")
            updates.append({"idx": idx, "is_played": "True", "result": "VOID",
                            "pnl": "0.0", "notes": "DNP", "resolved_date": today_str})
            continue

        # ── Step 4: check minutes played ──────────────────────────────────────
        minutes = int(stats.get("minutes_played") or 0)
        if minutes == 0:
            log.info(f"  {player} had 0 minutes in {home} vs {away} — DNP (VOID)")
            updates.append({"idx": idx, "is_played": "True", "result": "VOID",
                            "pnl": "0.0", "notes": "DNP (0 min)", "resolved_date": today_str})
            continue

        # ── Step 5: resolve WIN/LOSS ──────────────────────────────────────────
        result = _resolve_player_market(market, stats)
        if not result:
            continue

        try:
            mkt_odds = float(row["market_odds"])
        except (TypeError, ValueError):
            mkt_odds = 1.0
        pnl = round(mkt_odds - 1.0, 4) if result == "WIN" else -1.0

        updates.append({"idx": idx, "is_played": "True", "result": result,
                        "pnl": pnl, "notes": "", "resolved_date": today_str})
        log.info(
            f"  {'[DRY]' if dry_run else '[ OK]'} "
            f"{player} ({market}, {minutes}min) {home} vs {away} ({date_str}) → {result}  PnL={pnl:+.3f}u"
        )

    if not updates:
        log.info("  No player prop results resolved.")
        return

    if dry_run:
        voids   = sum(1 for u in updates if u["result"] == "VOID")
        wins    = sum(1 for u in updates if u["result"] == "WIN")
        losses  = sum(1 for u in updates if u["result"] == "LOSS")
        print(f"\n  [DRY RUN] Would update {len(updates)} player prop row(s): "
              f"{wins}W / {losses}L / {voids} VOID. No files written.")
        return

    resolved_today = today_str
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
    total_pnl = sum(float(u["pnl"]) for u in updates if u["result"] != "VOID")
    log.info(f"player_ledger: {len(updates)} resolved — {wins}W / {losses}L / {voids} VOID — PnL={total_pnl:+.3f}u")
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

        completed = scores_cache.get(league, [])
        match_result = None
        for ev in completed:
            if ev["date_str"] != date_str:
                continue
            if not (_names_match(home, ev["home_team"]) and _names_match(away, ev["away_team"])):
                continue
            match_result = _resolve_sharp_side(side, ev["home_score"], ev["away_score"])
            break

        if match_result is None:
            log.debug(f"  sharp: {home} vs {away} {date_str} not found in scores")
            continue

        try:
            opening_odds = float(row["opening_odds"])
        except (TypeError, ValueError):
            opening_odds = 1.0
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
        ledger.at[u["idx"], "pnl"]           = str(u["pnl"])
        ledger.at[u["idx"], "resolved_date"] = resolved_today

    ledger.to_csv(SHARP_LEDGER_FILE, index=False)

    wins      = sum(1 for u in updates if u["result"] == "WIN")
    losses    = sum(1 for u in updates if u["result"] == "LOSS")
    total_pnl = sum(u["pnl"] for u in updates)
    log.info(f"sharp_ledger: {len(updates)} resolved — {wins}W/{losses}L — PnL={total_pnl:+.3f}u")
    log.info(f"  Saved → {SHARP_LEDGER_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=3,
                        help="How many past days to fetch scores for (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing the ledger")
    args = parser.parse_args()

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

    # Fetch scores — FD first (free), OddsAPI as fallback when FD is stale
    scores_cache: dict[str, list[dict]] = {}
    for league in pending["league"].unique():
        if league in _FD_SOURCES:
            log.info(f"  Fetching scores for {league} from football-data.co.uk ...")
            fd_results = fetch_scores_fd(league)
            scores_cache[league] = fd_results

            # Fall back to OddsAPI only for RECENT missing dates (within args.days)
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d")
            pending_dates = set(pending[
                (pending["league"] == league) & (pending["match_date"] >= cutoff)
            ]["match_date"].tolist())
            fd_dates = {ev["date_str"] for ev in fd_results}
            missing_dates = {d for d in pending_dates if d not in fd_dates}
            if missing_dates and config.ODDS_API_SPORT_KEYS.get(league):
                sport_key = config.ODDS_API_SPORT_KEYS[league]
                log.info(f"  FD missing dates {missing_dates} — falling back to OddsAPI...")
                api_results = fetch_scores(sport_key, args.days)
                log.info(f"    → OddsAPI returned {len(api_results)} completed events")
                scores_cache[league] = fd_results + api_results
            continue

        sport_key = config.ODDS_API_SPORT_KEYS.get(league)
        if not sport_key:
            log.info(f"  {league}: no scores source configured — skipping")
            continue
        if sport_key not in scores_cache:
            log.info(f"  Fetching scores for {league} ({sport_key}) last {args.days} days ...")
            scores_cache[sport_key] = fetch_scores(sport_key, args.days)
            log.info(f"    → {len(scores_cache[sport_key])} completed events returned")

    # Match tips to results
    filled = 0
    updates: list[dict] = []

    for idx, row in pending.iterrows():
        league = row["league"]
        if league in _FD_SOURCES:
            completed = scores_cache.get(league, [])
        else:
            sport_key = config.ODDS_API_SPORT_KEYS.get(league)
            if not sport_key:
                continue
            completed = scores_cache.get(sport_key, [])
        found = _find_result(
            home=row["home_team"],
            away=row["away_team"],
            date_str=row["match_date"],
            side=row["side"],
            completed=completed,
        )

        if found is None:
            log.debug(f"  Not found: {row['home_team']} vs {row['away_team']} ({row['match_date']})")
            continue

        result_str, total_goals, extras = found
        odds = float(row["odds"])

        if result_str == "WIN":
            pnl = round(odds - 1.0, 4)
        else:
            pnl = -1.0

        # Closing Line Value
        cl = _closing_odds(row["home_team"], row["away_team"], row["match_date"], row["side"])
        clv = round((odds - cl) / cl * 100, 2) if (not np.isnan(cl) and cl > 0) else np.nan

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
            ledger.at[u["idx"], "pnl"]    = str(u["pnl"])
            if not np.isnan(u["closing_odds"]):
                ledger.at[u["idx"], "closing_odds"] = str(u["closing_odds"])
            if not np.isnan(u["clv_pct"]):
                ledger.at[u["idx"], "clv_pct"] = str(u["clv_pct"])
            if u.get("ht_score"):
                ledger.at[u["idx"], "ht_score"] = u["ht_score"]

        ledger.to_csv(LEDGER_FILE, index=False)

    if updates and not args.dry_run:
        total_pnl  = sum(u["pnl"] for u in updates)
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

    # Also resolve player prop and sharp money ledgers
    update_player_results(days=args.days, dry_run=args.dry_run)
    update_sharp_results(days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
