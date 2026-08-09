"""
FORWARD new-format odds capture (v10 NF mission 1).
====================================================
Fixes the gap found 2026-07-09: `output/newformat_odds_history.csv` was BACKFILL-ONLY
(no scheduled job appended it → it stopped at 2026-06-23 while summer NF leagues play on).
This script fetches TODAY's odds for UPCOMING new-format fixtures and APPENDS them, so the
NF O/U+BTTS odds accumulate forward — ending the re-backfill treadmill (like props already do).

ISOLATION: touches ONLY `output/newformat_odds_history.csv` (new-format team odds).
Does NOT touch the standard model, its data, props, or any shared model code.

Source: API-Football /odds, Bet365 (id 8). Markets captured (match the existing schema):
over25/under25, btts_yes/btts_no, over15/under15, over35/under35.
Run in CI daily (has APIFOOTBALL_KEY). Idempotent: dedups on (snapshot_date, match, market).
"""
import os, sys, json, time
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
OUT = PROJ / "output" / "newformat_odds_history.csv"
COLS = ["snapshot_date", "snapshot_ts", "match_date", "league", "match", "market", "odds"]
NEXT_N = 20          # look-ahead fixtures per league
_MIN_QUOTA = 200     # stop if API quota gets low


def _parse_all_odds(data: dict) -> dict:
    """Bet365 O/U 2.5/1.5/3.5 (over+under) + BTTS yes/no -> {market: odds}."""
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
                elif "Over/Under" in name and "Half" not in name:   # FT only — NOT half-time O/U
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
                elif name == "Match Winner":            # 1X2 (h2h) — free, same /odds call
                    for v in vals:
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        val = v.get("value")
                        if val == "Home":
                            out["h2h_home"] = odd
                        elif val == "Draw":
                            out["h2h_draw"] = odd
                        elif val == "Away":
                            out["h2h_away"] = odd
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
        print("[nf_odds] APIFOOTBALL_KEY not set — skipping"); return 0
    now = datetime.now(timezone.utc)
    snap_date = now.strftime("%Y-%m-%d")
    snap_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    leagues = sorted(config.NEW_FORMAT_LEAGUES & set(config.API_FOOTBALL_IDS))
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
        print("[nf_odds] no rows captured"); return 0
    new = pd.DataFrame(rows, columns=COLS)
    if OUT.exists():
        old = pd.read_csv(OUT)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    # keep price CHANGES per (fixture, market) -> full open..moving..close curve. Was one row
    # per DAY (dropped intraday moves); consecutive-distinct keeps every change AND the true
    # last snapshot even when the line reverts to an earlier value.
    combined = combined.sort_values(["match_date", "match", "market", "snapshot_ts"])
    _prev = combined.groupby(["match_date", "match", "market"])["odds"].shift()
    combined = combined[combined["odds"].ne(_prev)].sort_values("snapshot_ts")
    combined.to_csv(OUT, index=False)
    print(f"[nf_odds] appended {len(new)} rows -> {OUT.name} (total {len(combined):,})")
    return len(new)


if __name__ == "__main__":
    run()
