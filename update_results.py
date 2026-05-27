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

LEDGER_FILE = config.OUTPUT_DIR / "bets_ledger.csv"


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
# "new" format: football-data.co.uk/new/{code}.csv  (cols: Home/Away/HG/AG/Date)
# "std" format: football-data.co.uk/mmz4281/2526/{code}.csv (cols: HomeTeam/AwayTeam/FTHG/FTAG/Date)

_FD_SOURCES = {
    # new format
    "Austrian Bundesliga":      ("new", "AUT", "Home", "Away", "HG",   "AG",   "Date"),
    "Sweden Allsvenskan":       ("new", "SWE", "Home", "Away", "HG",   "AG",   "Date"),
    "Denmark Superliga":        ("new", "DNK", "Home", "Away", "HG",   "AG",   "Date"),
    "Japan J-League":           ("new", "JPN", "Home", "Away", "HG",   "AG",   "Date"),
    "USA MLS":                  ("new", "USA", "Home", "Away", "HG",   "AG",   "Date"),
    "China Super League":       ("new", "CHN", "Home", "Away", "HG",   "AG",   "Date"),
    "Ireland Premier Division": ("new", "IRL", "Home", "Away", "HG",   "AG",   "Date"),
    # standard format (2025/26 season)
    "Bundesliga 2":             ("std", "D2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "La Liga 2":                ("std", "SP2", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "Ligue 2":                  ("std", "F2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "League One":               ("std", "E2",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "League Two":               ("std", "E3",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "Greek Super League":       ("std", "G1",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
    "Belgian First Division A": ("std", "B1",  "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"),
}

_FD_NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
_FD_STD_URL = "https://www.football-data.co.uk/mmz4281/2526/{code}.csv"

def fetch_scores_fd(league: str) -> list[dict]:
    """Fetch completed results from football-data.co.uk (free, no quota)."""
    if league not in _FD_SOURCES:
        return []

    fmt, code, home_col, away_col, hg_col, ag_col, date_col = _FD_SOURCES[league]

    try:
        from io import StringIO

        if fmt == "new":
            url = _FD_NEW_URL.format(code=code)
        else:  # std
            url = _FD_STD_URL.format(code=code)

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
                results.append({
                    "home_team":  str(row[home_col]).strip(),
                    "away_team":  str(row[away_col]).strip(),
                    "home_score": hg,
                    "away_score": ag,
                    "date_str":   str(dt.date()),
                })
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
) -> tuple[str, float] | None:
    """
    Search `completed` for this fixture.
    Returns (result_str, pnl) or None if not found.
    """
    for ev in completed:
        if ev["date_str"] != date_str:
            continue
        if not (_names_match(home, ev["home_team"]) and _names_match(away, ev["away_team"])):
            continue

        total = ev["home_score"] + ev["away_score"]
        if side == "OVER":
            won = total > 2.5
        elif side == "UNDER":
            won = total <= 2.5
        else:
            return None

        result = "WIN" if won else "LOSS"
        return result, total

    return None


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
        log.info("No pending results to fill in.")
        return

    log.info(f"Found {len(pending)} tip(s) with missing results (match_date < {today_str})")

    # Fetch scores — OddsAPI for major leagues, football-data.co.uk for others
    scores_cache: dict[str, list[dict]] = {}
    for league in pending["league"].unique():
        if league in _FD_SOURCES:
            log.info(f"  Fetching scores for {league} from football-data.co.uk ...")
            scores_cache[league] = fetch_scores_fd(league)
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

        result_str, total_goals = found
        odds = float(row["odds"])

        if result_str == "WIN":
            pnl = round(odds - 1.0, 4)
        else:
            pnl = -1.0

        # Closing Line Value
        cl = _closing_odds(row["home_team"], row["away_team"], row["match_date"], row["side"])
        clv = round((odds - cl) / cl * 100, 2) if (not np.isnan(cl) and cl > 0) else np.nan

        updates.append({
            "idx":          idx,
            "result":       result_str,
            "pnl":          pnl,
            "total_goals":  total_goals,
            "closing_odds": cl,
            "clv_pct":      clv,
        })

        clv_str = f"  CLV={clv:+.1f}%" if not np.isnan(clv) else ""
        log.info(
            f"  {'[DRY]' if args.dry_run else '[ OK]'} "
            f"{row['home_team']} vs {row['away_team']} "
            f"({row['match_date']})  "
            f"{row['side']} @ {odds}  "
            f"Goals={total_goals}  → {result_str}  PnL={pnl:+.3f}u{clv_str}"
        )
        filled += 1

    if not updates:
        log.info("No pending matches found in any scores source. If matches were recently played, try --days with a larger value or re-run after the local CSVs are refreshed.")
        return

    if args.dry_run:
        print(f"\n  [DRY RUN] Would update {filled} row(s). No files written.")
        return

    # Write back to ledger
    for u in updates:
        ledger.at[u["idx"], "result"] = u["result"]
        ledger.at[u["idx"], "pnl"]    = str(u["pnl"])
        if not np.isnan(u["closing_odds"]):
            ledger.at[u["idx"], "closing_odds"] = str(u["closing_odds"])
        if not np.isnan(u["clv_pct"]):
            ledger.at[u["idx"], "clv_pct"] = str(u["clv_pct"])

    ledger.to_csv(LEDGER_FILE, index=False)

    # Print summary
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


if __name__ == "__main__":
    main()
