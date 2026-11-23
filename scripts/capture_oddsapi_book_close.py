"""Per-book OddsAPI near-kickoff close capture -> PRIVATE store (owner decision 2026-11-23).

    python scripts/capture_oddsapi_book_close.py --store DIR --predictions output/predictions.csv \
        --mode dry-run                                   # free call only: remaining credits + the plan
    python scripts/capture_oddsapi_book_close.py --store DIR ... --mode test --run-cap 20 --max-calls 3
    python scripts/capture_oddsapi_book_close.py --store DIR ... --mode capture --run-cap 30 \
        --day-cap 100 --month-cap 1000 --loop-minutes 110

capture  buys each (league, kickoff) slot in the bands CLOSE_30 (10, 30] and CLOSE_10 (1, 10] minutes
         before kickoff (src/oddsapi_book_close.py), looping until no band opens inside --loop-minutes.
test     a capped real run: the next --max-calls slots (any lead time), stored apart under <store>/test/
         with band TEST, so the close data is never mixed with a test.
dry-run  one free call (GET /v4/sports) for x-requests-remaining, the guards and the slots; no paid call.

Every paid call passes src.oddsapi_book_close.Budget.can_spend first (floor 260,000, run / day / month caps).
One `[oddsapi_usage]` JSON line per run feeds src/integrity/oddsapi_quota.py. Never prints a price or the key.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oddsapi_book_close as BC  # noqa: E402

API = "https://api.the-odds-api.com/v4"
log = logging.getLogger("book_close")


def _key() -> str:
    k = os.getenv("ODDS_API_KEY", "").strip()
    if not k:
        raise SystemExit("ODDS_API_KEY is not set")
    return k


def free_remaining(budget: BC.Budget, get=requests.get, sport_key: str | None = None) -> None:
    """GET /v4/sports (and, when that answer carries no usage header, one sport's /events) is not billed;
    it refreshes x-requests-remaining. A billed answer stops the run."""
    urls = [f"{API}/sports"] + ([f"{API}/sports/{sport_key}/events"] if sport_key else [])
    for url in urls:
        r = get(url, params={"apiKey": _key()}, timeout=20)
        spent = budget.read_headers(getattr(r, "headers", {}))
        budget.calls_free += 1
        if spent:
            raise SystemExit(f"a free listing was billed ({spent}): stopping")
        if budget.remaining is not None:
            return


def buy(slot: dict, budget: BC.Budget, store: Path, *, mode: str, run_id: str, get=requests.get,
        now=BC.utcnow) -> dict:
    """One paid bulk /odds call for one slot. Returns its journal record."""
    ko = slot["ko"]
    rec = {"sport_key": slot["sport_key"], "kickoff_utc": BC.iso(ko), "band": slot["band"],
           "attempted_at": BC.iso(now()), "status": "", "http_status": "", "events": 0, "rows": 0,
           "credits": 0, "run_id": run_id, "mode": mode}
    params = {"apiKey": _key(), "regions": BC.REGIONS, "markets": BC.MARKETS, "oddsFormat": "decimal",
              "dateFormat": "iso",
              "commenceTimeFrom": BC.iso(ko - BC.SLOT_WINDOW), "commenceTimeTo": BC.iso(ko + BC.SLOT_WINDOW)}
    try:
        r = get(f"{API}/sports/{slot['sport_key']}/odds", params=params, timeout=20)
    except requests.RequestException as e:
        rec["status"] = f"ERROR_{type(e).__name__}"
        return rec
    rec["http_status"] = str(getattr(r, "status_code", ""))
    budget.calls_paid += 1
    spent = budget.read_headers(getattr(r, "headers", {}))
    if spent is None:
        budget.cost_unknown = True          # stops every further paid call (can_spend)
    else:
        budget.book(spent, now())
        rec["credits"] = spent
    if getattr(r, "status_code", 0) != 200:
        rec["status"] = f"HTTP_{rec['http_status']}"
        return rec
    try:
        events = r.json() or []
    except ValueError:
        rec["status"] = "BAD_JSON"
        return rec
    rows = BC.quote_rows(events, league=slot["league"], sport_key=slot["sport_key"], band=slot["band"],
                         captured=now())
    rec["events"] = len(events)
    rec["rows"] = BC.append_rows(store, rows, sub="test" if mode == "test" else "data")
    rec["status"] = "OK" if rows else "EMPTY"
    return rec


def run(args, *, get=requests.get, now=BC.utcnow, sleep=time.sleep) -> dict:
    store = Path(args.store).resolve()
    try:
        store.relative_to(ROOT)
        raise SystemExit("--store is inside the public repo: refused (prices go to the private store only)")
    except ValueError:
        pass
    usage_path = store / "ledger" / "usage.json"
    budget = BC.Budget.load(usage_path, floor=args.floor, run_cap=args.run_cap, day_cap=args.day_cap,
                            month_cap=args.month_cap)
    pred = pd.read_csv(args.predictions, low_memory=False) if Path(args.predictions).exists() else pd.DataFrame()
    slots = BC.slots_from_predictions(pred)
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    summary = Counter()
    by_league = Counter()

    free_remaining(budget, get, slots["sport_key"].iloc[0] if len(slots) else None)
    start = now()
    end = start + timedelta(minutes=args.loop_minutes)

    if args.mode == "dry-run":
        summary["slots_upcoming"] = int((slots["ko"] > pd.Timestamp(start)).sum()) if len(slots) else 0
        summary["due_now"] = len(BC.due(slots, BC.read_journal(store), start))
    elif args.mode == "test":
        upcoming = slots[slots["ko"] > pd.Timestamp(start + timedelta(minutes=2))].head(args.max_calls)
        for s in upcoming.itertuples(index=False):
            if not budget.can_spend(BC.CALL_COST, now()):
                break
            slot = {"league": s.league, "sport_key": s.sport_key, "ko": s.ko.to_pydatetime(), "band": BC.TEST_BAND}
            rec = buy(slot, budget, store, mode="test", run_id=run_id, get=get, now=now)
            BC.append_journal(store, rec)
            budget.save(usage_path)
            summary[rec["status"]] += 1
            summary["rows"] += rec["rows"]
            by_league[s.league] += rec["credits"]
    else:
        while True:
            t = now()
            for slot in BC.due(slots, BC.read_journal(store), t):
                if not budget.can_spend(BC.CALL_COST, now()):
                    break
                rec = buy(slot, budget, store, mode="capture", run_id=run_id, get=get, now=now)
                BC.append_journal(store, rec)
                budget.save(usage_path)
                summary[rec["status"]] += 1
                summary[f"band_{slot['band']}"] += 1
                summary["rows"] += rec["rows"]
                by_league[slot["league"]] += rec["credits"]
            if budget.stopped:
                break
            nxt = BC.next_band_open(slots, now())
            pending = BC.due(slots, BC.read_journal(store), now())
            if pending:                          # a failed attempt with one try left: retry after a pause
                wake = now() + timedelta(seconds=args.poll_seconds)
            elif nxt is None:
                break
            else:
                wake = nxt
            if wake >= end:
                break
            sleep(max(1.0, (wake - now()).total_seconds()))

    budget.save(usage_path)
    nxt = BC.next_band_open(slots, now())
    out = {"source": "oddsapi_book_close", "workflow": os.getenv("GITHUB_WORKFLOW", ""), "run_id": run_id,
           "mode": args.mode, "market": "totals_close", "regions": BC.REGIONS,
           "calls": budget.calls_paid, "free_calls": budget.calls_free, "credits": budget.run_spent,
           "credits_by_market": {"totals_close": budget.run_spent}, "credits_by_league": dict(by_league),
           "requests_remaining": budget.remaining, "stopped": budget.stopped, "floor": args.floor,
           "caps": {"run": args.run_cap, "day": args.day_cap, "month": args.month_cap},
           "day_spent": budget.usage["by_day"][now().strftime("%Y-%m-%d")],
           "month_spent": budget.usage["by_month"][now().strftime("%Y-%m")],
           "slots_on_board": int(len(slots)), "summary": dict(summary),
           "next_band_open": BC.iso(nxt) if nxt else None}
    print("[oddsapi_usage] " + json.dumps(out, sort_keys=True, default=str), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--predictions", default=str(ROOT / "output" / "predictions.csv"))
    ap.add_argument("--mode", choices=("dry-run", "test", "capture"), required=True)
    ap.add_argument("--floor", type=int, default=BC.FLOOR_MIN)
    ap.add_argument("--run-cap", type=int, required=True)
    ap.add_argument("--day-cap", type=int, default=100)
    ap.add_argument("--month-cap", type=int, default=1000)
    ap.add_argument("--max-calls", type=int, default=3, help="test mode: most slots bought")
    ap.add_argument("--loop-minutes", type=float, default=0.0)
    ap.add_argument("--poll-seconds", type=float, default=90.0)
    a = ap.parse_args(argv)
    try:
        run(a)
    except ValueError as e:                   # Budget refuses a floor below the reserve / a negative cap
        raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
