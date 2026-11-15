"""SF-PAPER-1 live capture + paper ledger (owner 2026-11-15, "Paper track + capture").

    python scripts/sharp_paper_capture.py --store DIR --ledgers DIR --mode dry-run      # free calls only
    python scripts/sharp_paper_capture.py --store DIR --ledgers DIR --mode test --run-cap 12 --max-fixtures 4
    python scripts/sharp_paper_capture.py --store DIR --ledgers DIR --mode capture --loop-minutes 170
    python scripts/sharp_paper_capture.py --store DIR --ledgers DIR --mode evaluate      # no API call

WHAT A PASS DOES. Reads the live tips (v9 output/bets_ledger.csv, output/side_bets_ledger.csv) of the
protocol's cells (src.sharp_paper.tip_universe), lists each league's upcoming provider events (GET
/v4/sports/{key}/events: free), matches them (src.sharp_paper.match_events) and, for every matched fixture
whose slot is due (T72 = [60, 72] h, T6 = [4, 6] h, CLOSE = 2-15 min before kickoff) and not yet in the
journal, buys ONE event-odds snapshot: markets totals,alternate_totals, region eu (2 credits when both
markets come back). Every book's exact 1.5 / 2.5 / 3.5 pair is stored.

BUDGET (never below the floor; counted against the month):
  * before every paid call: x-requests-remaining (from the previous answer; the free events call refreshes
    it) minus the call's cost must stay >= --floor (260,000 by default, the 5M plan's floor);
  * --run-cap, --day-cap (UTC day) and --month-cap credits, journaled in <store>/ledger/usage.json;
  * one [oddsapi_usage] JSON line per pass (counts and credits; never the key, never a price).

PRIVATE. --store must be outside this public repo (the licensed prices go there: wowza-exec branch
sharp-paper). The log prints counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sharp_paper as SP  # noqa: E402

API = "https://api.the-odds-api.com/v4"
CALL_COST = 2                       # markets totals + alternate_totals, one region
CAP_COLS = ["pick_key", "league", "slot", "captured_at", "ko", "lead_h", "event_id", "bookmaker", "line", "over",
            "under", "book_last_update"]
JOURNAL_COLS = ["pick_key", "slot", "event_id", "captured_at", "lead_h", "status", "rows", "credits"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Budget:
    def __init__(self, store: Path, floor: int, run_cap: int, day_cap: int, month_cap: int):
        self.path = store / "ledger" / "usage.json"
        u = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"by_day": {}, "by_month": {}}
        # tallies of credits spent: a day / month with no call has spent nothing (a counter, not a value)
        self.u = {"by_day": Counter(u["by_day"]), "by_month": Counter(u["by_month"])}
        self.cost_unknown = False
        self.floor, self.run_cap, self.day_cap, self.month_cap = floor, run_cap, day_cap, month_cap
        self.run_spent, self.remaining, self.stopped = 0, None, None
        self.calls_paid, self.calls_free = 0, 0

    def can_spend(self, cost: int) -> bool:
        now = utcnow()
        day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        why = None
        if self.cost_unknown:
            why = "a paid answer carried no x-requests-last: its cost is unknown, no further paid call"
        elif self.remaining is None:
            why = "remaining credits unknown (no free call answered): no paid call"
        elif self.remaining - cost < self.floor:
            why = f"floor: {self.remaining} remaining - {cost} < {self.floor}"
        elif self.run_spent + cost > self.run_cap:
            why = f"run cap {self.run_cap}"
        elif self.u["by_day"][day] + cost > self.day_cap:
            why = f"day cap {self.day_cap}"
        elif self.u["by_month"][month] + cost > self.month_cap:
            why = f"month cap {self.month_cap}"
        if why:
            self.stopped = why
            return False
        return True

    def headers(self, r) -> int | None:
        h = getattr(r, "headers", None) or {}
        try:
            if h.get("x-requests-remaining") not in (None, ""):
                self.remaining = int(float(h["x-requests-remaining"]))
        except (TypeError, ValueError):
            pass
        v = h.get("x-requests-last")
        if v in (None, ""):
            return None                     # unknown, never assumed 0
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    def book(self, spent: int) -> None:
        now = utcnow()
        day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        self.run_spent += spent
        self.u["by_day"][day] += spent
        self.u["by_month"][month] += spent

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: dict(v) for k, v in self.u.items()}, indent=1, sort_keys=True),
                             encoding="utf-8")


def _get(url: str, params: dict, budget: Budget, paid: bool):
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("ODDS_API_KEY is not set")
    r = requests.get(url, params={**params, "apiKey": key}, timeout=20)
    spent = budget.headers(r)
    if paid:
        budget.calls_paid += 1
        if spent is None:
            budget.cost_unknown = True      # stops every further paid call of this run (can_spend)
            return r, None
        budget.book(spent)
    else:
        budget.calls_free += 1
        if spent:
            raise SystemExit(f"a listing call was billed ({spent}): stopping (the events listing must be free)")
    return r, spent


def read_ledgers(ledgers: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    out = []
    for f in ("bets_ledger.csv", "side_bets_ledger.csv"):
        p = Path(ledgers) / f
        out.append(pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else None)
    return out[0], out[1]


def read_journal(store: Path) -> pd.DataFrame:
    p = store / "ledger" / "journal.csv"
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=JOURNAL_COLS)


def read_capture(store: Path, sub: str = "capture") -> pd.DataFrame:
    fs = sorted((store / "data" / sub).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=CAP_COLS)


def one_pass(args, P, budget: Budget, aliases: dict, seq: int) -> dict:
    from src import league_registry as LR
    test = args.mode == "test"
    bets, side = read_ledgers(args.ledgers)
    today = utcnow().strftime("%Y-%m-%d")
    picks, n = SP.tip_universe(bets, side, P, test=test, today=today)
    horizon = (utcnow() + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    picks = picks[(picks["match_date"] >= (utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                  & (picks["match_date"] <= horizon)]
    fixtures = picks.drop_duplicates(["match_date", "league", "home", "away"])
    if test and args.max_fixtures:
        fixtures = fixtures.head(args.max_fixtures)
    j = read_journal(args.store)
    done = {(r.pick_key, r.slot) for r in j.itertuples() if r.status in ("ok", "empty")}
    st = {"picks": int(len(picks)), "fixtures": int(len(fixtures)), "matched": 0, "unmatched": 0, "ambiguous": 0,
          "slots_due": 0, "snapshots": 0, "rows": 0, "skipped_budget": 0, "post_kickoff": 0, **{f"universe_{k}": v for k, v in n.items()}}
    caps, jrows = [], []
    for lg, fx in fixtures.groupby("league"):
        events = []
        for sk in LR.odds_api_sport_keys_for(lg):
            r, _ = _get(f"{API}/sports/{sk}/events", {"dateFormat": "iso"}, budget, paid=False)
            if r.status_code == 200:
                events += [e | {"sport_key": sk} for e in (r.json() or [])]
        m, ms = SP.match_events(fx, events, (aliases or {}).get(lg))
        for k in ("matched", "unmatched", "ambiguous"):
            st[k] += ms[k]
        for (md, _lg, h, a), ev in m.items():
            key = f"{md}|{lg}|{h}|{a}"
            ko = pd.Timestamp(ev["commence_time"]).tz_convert("UTC")
            slots = SP.due_slots(ko, pd.Timestamp(utcnow()), {s for (k2, s) in done if k2 == key}, P, test=test)
            for slot in slots:
                st["slots_due"] += 1
                if args.mode == "dry-run":
                    continue
                if not budget.can_spend(CALL_COST):
                    st["skipped_budget"] += 1
                    continue
                req_at = utcnow()
                r, spent = _get(f"{API}/sports/{ev['sport_key']}/events/{ev['id']}/odds",
                                {"regions": "eu", "markets": "totals,alternate_totals", "oddsFormat": "decimal",
                                 "dateFormat": "iso"}, budget, paid=True)
                at = utcnow()                                     # stamped AFTER the answer: conservative
                lead = SP.lead_hours(ko, pd.Timestamp(at))
                if r.status_code != 200:
                    jrows.append({"pick_key": key, "slot": slot, "event_id": ev["id"], "captured_at": at.isoformat(),
                                  "lead_h": round(lead, 3), "status": f"http_{r.status_code}", "rows": 0, "credits": spent})
                    continue
                rows, ps = SP.parse_event_odds(r.json(), pd.Timestamp(at))
                st["post_kickoff"] += ps["post_kickoff"]
                for x in rows:
                    caps.append({"pick_key": key, "league": lg, "slot": slot, "captured_at": at.isoformat(),
                                 "ko": ko.isoformat(), "lead_h": round(lead, 3), "event_id": ev["id"], **x})
                st["snapshots"] += 1
                st["rows"] += len(rows)
                jrows.append({"pick_key": key, "slot": slot, "event_id": ev["id"], "captured_at": at.isoformat(),
                              "lead_h": round(lead, 3), "status": "ok" if rows else "empty", "rows": len(rows),
                              "credits": spent})
                done.add((key, slot))
    sub = "test" if test else "capture"
    if caps:
        d = args.store / "data" / sub
        d.mkdir(parents=True, exist_ok=True)
        tag = os.getenv("GITHUB_RUN_ID", utcnow().strftime("%Y%m%dT%H%M%S"))
        pd.DataFrame(caps, columns=CAP_COLS).to_parquet(d / f"{tag}-{seq:03d}.parquet", index=False)
    if jrows and not test:
        p = args.store / "ledger" / "journal.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(jrows, columns=JOURNAL_COLS).to_csv(p, mode="a", header=not p.exists(), index=False)
    if jrows and test:
        p = args.store / "ledger" / "test_journal.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(jrows, columns=JOURNAL_COLS).to_csv(p, mode="a", header=not p.exists(), index=False)
    budget.save()
    return st


def evaluate(args, P) -> dict:
    bets, side = read_ledgers(args.ledgers)
    picks, n = SP.tip_universe(bets, side, P)
    cap = read_capture(args.store)
    paper = SP.build_paper(picks, cap, P)
    d = args.store / "paper"
    d.mkdir(parents=True, exist_ok=True)
    paper.to_parquet(d / "paper_ledger.parquet", index=False)
    s = SP.summarize(paper, P) | {"universe": n, "protocol_sha256": SP.protocol_sha(),
                                  "evaluated_at": utcnow().isoformat()}
    (d / "summary.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    return {"picks": int(len(picks)), "status_counts": s["status_counts"], "review_due": s["review_due"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--ledgers", required=True, type=Path)
    ap.add_argument("--aliases", type=Path, help="the private store's ledger/provider_aliases.json (proven names)")
    ap.add_argument("--mode", choices=["dry-run", "test", "capture", "evaluate"], required=True)
    ap.add_argument("--floor", type=int, default=260_000)
    ap.add_argument("--run-cap", type=int, default=400)
    ap.add_argument("--day-cap", type=int, default=1_500)
    ap.add_argument("--month-cap", type=int, default=30_000)
    ap.add_argument("--max-fixtures", type=int, default=0)
    ap.add_argument("--loop-minutes", type=float, default=0)
    ap.add_argument("--interval-minutes", type=float, default=10)
    a = ap.parse_args(argv)
    if a.floor < 260_000:
        raise SystemExit("--floor below 260,000 is refused (the 5M plan's floor)")
    a.store = SP.assert_private(a.store)
    a.store.mkdir(parents=True, exist_ok=True)
    P = SP.load_protocol()
    print(f"[sharp_paper] protocol {P['protocol_id']} sha256 {SP.protocol_sha()[:12]} mode {a.mode}")
    if a.mode == "evaluate":
        print("[sharp_paper] " + json.dumps(evaluate(a, P), default=str))
        return 0
    aliases = {}
    if a.aliases and a.aliases.exists():
        aliases = json.loads(a.aliases.read_text(encoding="utf-8")).get("map", {})
    budget = Budget(a.store, a.floor, a.run_cap, a.day_cap, a.month_cap)
    start, seq = time.time(), 0
    while True:
        seq += 1
        st = one_pass(a, P, budget, aliases, seq)
        print(f"[sharp_paper] pass {seq}: " + json.dumps(st, sort_keys=True))
        if a.loop_minutes <= 0 or (time.time() - start) / 60 + a.interval_minutes > a.loop_minutes:
            break
        time.sleep(a.interval_minutes * 60)
    print("[oddsapi_usage] " + json.dumps({
        "source": "sharp_paper_capture", "workflow": os.getenv("GITHUB_WORKFLOW", ""), "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "mode": a.mode, "calls_paid": budget.calls_paid, "calls_free": budget.calls_free, "credits": budget.run_spent,
        "requests_remaining": budget.remaining, "stopped": budget.stopped, "floor": a.floor,
        "credits_today": budget.u["by_day"][utcnow().strftime("%Y-%m-%d")],
        "credits_month": budget.u["by_month"][utcnow().strftime("%Y-%m")]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
