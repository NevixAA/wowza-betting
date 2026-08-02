"""
FORWARD standard side-market odds capture (open -> moving -> closing).
=====================================================================
Sibling of capture_nf_odds_forward.py, for the STANDARD 2nd-division leagues.
`output/standard_sidemarket_odds_history.csv` was BACKFILL-ONLY (static since 2026-05-18) —
no scheduled job appended it, so standard BTTS / O/U 1.5 / 3.5 had NO forward odds at all.
This fetches TODAY's odds for UPCOMING standard fixtures and APPENDS every DISTINCT price, so
each (fixture, market) accumulates a real open->moving->closing curve as the season plays.

FREE: uses API-Football /odds (Bet365, id 8) on the APIFOOTBALL_KEY quota — NO OddsAPI credits.
ISOLATION: writes ONLY output/standard_sidemarket_odds_history.csv (standard side/main odds).
Does not train/predict any model; does not touch new-format, props, or shared model code.

Markets captured (match the existing schema): btts_yes/no, over25/under25, over15/under15,
over35/under35. Dedup keeps every DISTINCT price per (match_date, match, market) -> movement.
"""
import os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config
from player_model.api_football import get_upcoming_fixtures

_KEY = os.getenv("APIFOOTBALL_KEY", "")
_BASE = "https://v3.football.api-sports.io"
_HEADERS = {"x-apisports-key": _KEY}
BET365 = 8
OUT = PROJ / "output" / "standard_sidemarket_odds_history.csv"
COLS = ["snapshot_date", "snapshot_ts", "match_date", "league", "match", "market", "odds"]
NEXT_N = 12          # look-ahead fixtures per league
_MIN_QUOTA = 300     # stop if API-Football quota gets low (Pro tier is 7,500/day)


def _parse_all_odds(data: dict) -> dict:
    """Bet365 BTTS yes/no + O/U 1.5/2.5/3.5 (over+under) -> {market: odds}."""
    out = {}
    for entry in data.get("response", []):
        for bk in entry.get("bookmakers", []):
            if bk.get("id") != BET365:
                continue
            for bet in bk.get("bets", []):
                name = bet.get("name", "")
                vals = bet.get("values", [])
                if "Both Teams Score" in name or "BTTS" in name:
                    for v in vals:
                        try:
                            if v.get("value") == "Yes":
                                out["btts_yes"] = float(v["odd"])
                            elif v.get("value") == "No":
                                out["btts_no"] = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            pass
                elif "Over/Under" in name:
                    for v in vals:
                        label = v.get("value", "")
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        for ln, key in (("1.5", "15"), ("2.5", "25"), ("3.5", "35")):
                            if f"Over {ln}" in label:
                                out[f"over{key}"] = odd
                            elif f"Under {ln}" in label:
                                out[f"under{key}"] = odd
    return out


def _fetch_odds(fixture_id: int) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(f"{_BASE}/odds", headers=_HEADERS,
                             params={"fixture": fixture_id, "bookmaker": BET365}, timeout=20)
            if r.status_code == 429:
                time.sleep(15); continue
            remaining = int(r.headers.get("x-ratelimit-requests-remaining", 9999))
            if remaining < _MIN_QUOTA:
                print(f"  [!] quota low ({remaining}) — stopping"); return {"_stop": True}
            if r.status_code != 200:
                return {}
            data = r.json()
            if data.get("errors"):
                if "rateLimit" in str(data["errors"]) or "requests" in str(data["errors"]).lower():
                    time.sleep(15); continue
                return {}
            return _parse_all_odds(data)
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(2)
    return {}


def run() -> int:
    if not _KEY:
        print("[std_odds] APIFOOTBALL_KEY not set — skipping"); return 0
    now = datetime.now(timezone.utc)
    snap_date = now.strftime("%Y-%m-%d")
    snap_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    leagues = sorted(set(config.STANDARD_FORMAT_LEAGUES) & set(config.API_FOOTBALL_IDS))
    rows, stop = [], False
    for league in leagues:
        if stop:
            break
        lid = config.API_FOOTBALL_IDS[league]
        season = config.API_FOOTBALL_SEASONS.get(league, str(now.year))
        try:
            fixtures = get_upcoming_fixtures(lid, season, next_n=NEXT_N)
        except Exception as e:
            print(f"  {league}: fixtures fetch failed ({e})"); continue
        n_lg = 0
        for fx in fixtures:
            fid = fx.get("fixture", {}).get("id")
            mdate = (fx.get("fixture", {}).get("date", "") or "")[:10]
            teams = fx.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            if not (fid and home and away):
                continue
            odds = _fetch_odds(fid)
            if odds.get("_stop"):
                stop = True; break
            for market, odd in odds.items():
                rows.append({"snapshot_date": snap_date, "snapshot_ts": snap_ts,
                             "match_date": mdate, "league": league,
                             "match": f"{home} vs {away}", "market": market, "odds": odd})
            n_lg += 1
        print(f"  {league}: {n_lg} upcoming fixtures priced")
    if not rows:
        print("[std_odds] no rows captured"); return 0
    new = pd.DataFrame(rows, columns=COLS)
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
    # keep every DISTINCT price per (fixture, market) -> open..moving..close curve
    new = new.sort_values("snapshot_ts").drop_duplicates(
        subset=["match_date", "match", "market", "odds"], keep="first")
    new.to_csv(OUT, index=False)
    print(f"[std_odds] appended -> {OUT.name} (total {len(new):,})")
    return len(rows)


if __name__ == "__main__":
    run()
