"""Exchange back + LAY prices for our leagues (collection only - owner approved 2026-10-20).

    python scripts/capture_exchange_odds.py              # appends output/exchange_quotes/<YYYY-MM>.csv
    python scripts/capture_exchange_odds.py --dry-run    # list what would be asked, spend nothing

WHY. The hedging research (back-to-lay) and the prop research have only ever seen ONE side of each
price. Exchange quotes give both: a back AND a lay price, near-zero margin, from the sharpest venues
there are. Nothing here feeds a model, a tip, a stake or Telegram.

WHAT EXISTS (probed 2026-10-20, 22 credits): in our 2nd divisions the exchanges carry `h2h`/`h2h_lay`;
Matchbook also BACKS totals and BTTS (1.90/2.08 - a ~1% margin reference); O/U and BTTS LAY
(`alternate_totals_lay`, `btts_lay`) appear only on top-flight fixtures so far; the anytime-scorer
lay key is accepted but was empty 3 days out. `totals_lay` does NOT exist (HTTP 422). Prop
Under/No quotes are already logged since 2026-10-14 in output/prop_quotes/ (only Pinnacle posts any).
So this captures what our leagues actually have and records the rest as soon as it appears.

COST. Event lists are free. A per-event call is billed per market RETURNED (0 when empty), so this is
~2-4 credits per priced event. Hard guards, both fail CLOSED (stop asking, keep what was captured):
  * DAILY_BUDGET credits per run (default 150, env EXCHANGE_DAILY_BUDGET)
  * stops when the account has fewer than FLOOR credits left (default 15,000, env EXCHANGE_CREDIT_FLOOR)
    - production predict and the prop snapshots share this quota and always come first.
Batch job only, once a day - never inside the 5-minute predict loop (CLAUDE.md wall-clock rule).

The book KEY is logged as well as the title: OddsAPI titles Betfair Exchange UK just "Betfair", which
reads like the sportsbook.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://api.the-odds-api.com/v4"
BOOKS = "betfair_ex_uk,betfair_ex_eu,matchbook,smarkets"
MARKETS = "h2h,h2h_lay,alternate_totals,alternate_totals_lay,btts,btts_lay,player_goal_scorer_anytime_lay"
HORIZON_H = 36
OUT_DIR = ROOT / "output" / "exchange_quotes"
FIELDS = ["captured_at", "league", "sport_key", "event_id", "commence_time", "home_team", "away_team",
          "bookmaker_key", "bookmaker_title", "market", "outcome", "description", "point", "price",
          "last_update"]


def our_leagues() -> dict:
    """ENABLED leagues with an OddsAPI key - our leagues only (owner rule), paper leagues included."""
    import config
    return {lg: k for lg, k in sorted(config.ODDS_API_SPORT_KEYS.items()) if lg in config.ENABLED_LEAGUES}


def rows_from_event(ev: dict, league: str, sport_key: str, now: str) -> list[dict]:
    out = []
    for bm in ev.get("bookmakers") or []:
        for mk in bm.get("markets") or []:
            for oc in mk.get("outcomes") or []:
                out.append({"captured_at": now, "league": league, "sport_key": sport_key,
                            "event_id": ev.get("id"), "commence_time": ev.get("commence_time"),
                            "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                            "bookmaker_key": bm.get("key"), "bookmaker_title": bm.get("title"),
                            "market": mk.get("key"), "outcome": oc.get("name"),
                            "description": oc.get("description"), "point": oc.get("point"),
                            "price": oc.get("price"), "last_update": mk.get("last_update") or bm.get("last_update")})
    return out


def _append(rows: list[dict], now: datetime) -> Path | None:
    if not rows:
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{now:%Y-%m}.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        print("[exchange] ODDS_API_KEY not set - nothing captured")
        return 0
    budget = int(os.getenv("EXCHANGE_DAILY_BUDGET", "150"))
    floor = int(os.getenv("EXCHANGE_CREDIT_FLOOR", "15000"))
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = now_dt + timedelta(hours=HORIZON_H)
    spent, rows, asked, priced, stop = 0, [], 0, 0, None
    queue = []                                                         # event lists are free: collect all first
    for league, sk in our_leagues().items():
        try:
            r = requests.get(f"{BASE}/sports/{sk}/events", params={"apiKey": key}, timeout=30)
            events = r.json() if r.status_code == 200 else []
        except Exception as e:                                         # noqa: BLE001 - one league never stops the rest
            print(f"[exchange] {league}: event list failed ({type(e).__name__})")
            continue
        for ev in events if isinstance(events, list) else []:
            try:
                ko = datetime.fromisoformat(str(ev.get("commence_time")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if now_dt < ko <= until:                                   # pre-match only, next 36h
                queue.append((ko, league, sk, ev))
    # Soonest kickoff first: when the budget binds, it is spent on the prices closest to the close.
    queue.sort(key=lambda q: q[0])
    for ko, league, sk, ev in queue:
        if a.dry_run:
            print(f"  would ask {league}: {ev.get('home_team')} v {ev.get('away_team')} {ko:%m-%d %H:%M}")
            asked += 1
            continue
        if spent >= budget:
            stop = f"daily budget {budget} reached ({len(queue) - asked} events not asked)"
            break
        try:
            r = requests.get(f"{BASE}/sports/{sk}/events/{ev['id']}/odds", timeout=30,
                             params={"apiKey": key, "bookmakers": BOOKS, "markets": MARKETS,
                                     "oddsFormat": "decimal", "includeLinks": "false"})
        except Exception as e:                                     # noqa: BLE001
            print(f"[exchange] {league} {ev.get('id')}: {type(e).__name__}")
            continue
        asked += 1
        spent += int(r.headers.get("x-requests-last") or 0)
        remaining = int(float(r.headers.get("x-requests-remaining") or 0))
        if r.status_code == 200:
            got = rows_from_event(r.json(), league, sk, now)
            priced += bool(got)
            rows += got
        elif r.status_code == 422:
            print(f"[exchange] 422 from OddsAPI ({r.text[:120]}) - market list rejected, stopping")
            stop = "422"
            break
        if remaining and remaining < floor:
            stop = f"account below the {floor}-credit floor ({remaining} left) - production comes first"
            break
    path = _append(rows, now_dt)
    by_mkt = {}
    for x in rows:
        by_mkt[x["market"]] = by_mkt.get(x["market"], 0) + 1
    print(f"[exchange] events asked {asked}, priced {priced}, rows {len(rows)}, credits {spent}"
          f"{' - stopped: ' + stop if stop else ''}")
    print(f"[exchange] rows by market: {by_mkt or '{}'} -> {path or 'nothing written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
