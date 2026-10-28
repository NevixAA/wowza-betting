"""OddsAPI HISTORICAL backfill of full-match Over/Under 1.5 and 3.5 prices — RESEARCH ONLY.

    python scripts/backfill_std_ou15_history.py --check     # 0 credits: balance, ledger, RUN/SKIP decision
    python scripts/backfill_std_ou15_history.py             # spend, inside every guard below
    python scripts/backfill_std_ou15_history.py --status    # print progress, no network

Owner approved 2026-10-28 ("wait for the credit reset, then run it").

WHAT. The 7 Standard bet leagues (Championship, League One, League Two, Bundesliga 2, La Liga 2,
Ligue 2, Serie B), seasons 2025-26 FIRST, then 2024-25. One snapshot per match at 00:00Z on match
day (the same convention as the 2026-07-01 BTTS backfill in standard_sidemarket_odds_history.csv);
if a match kicks off within 30 minutes of that, the snapshot moves to kickoff - 2h so it is always
pre-match.

WHERE IT WRITES — and why NOT the training file.
    output/research/std_ou15_history_backfill.csv      the prices
    output/research/std_ou15_backfill_state.json       resumable state + credit ledger
`src/data_loader._enrich_with_standard_sidemarket_odds` already reads
output/standard_sidemarket_odds_history.csv into TRAINING (commit f280ac42). Writing there would
change production training data without approval. NOTHING in production reads output/research/.
Merging these rows into the training file is a LATER OWNER DECISION.

HOW (probed 2026-10-28, 34 credits, 3 of 3 events priced two-sided at 1.5 and 3.5):
    GET /v4/historical/sports/{sport}/events?date=D             1 credit (0 when empty)
    GET /v4/historical/sports/{sport}/events/{id}/odds
        ?date=T&regions=eu&markets=alternate_totals             10 credits per event
O/U 1.5 and 3.5 are NOT in the featured `totals` market (that carries the main 2.5 line) — they are
lines of the additional market `alternate_totals`, and ONE request returns every line, so 3.5 costs
nothing extra. Event lists reach only ~1-6 days ahead (Championship 2024-09-14 listed that weekend
only), so discovery steps ONE day at a time; a day with nothing listed costs 0.

PARSING — exact, never substring (the 2026-08-23 first-half-BTTS lesson, commit 8d9962c6):
market key == "alternate_totals" exactly, outcome name == "Over"/"Under" exactly, point == 1.5 /
3.5 exactly, no `description` (team/player totals carry one), and a book counts ONLY when it posts
BOTH sides of the same point exactly once. A first-half key (alternate_totals_h1, totals_h1, ...)
can therefore never contaminate a row.

CREDIT GUARDS (production comes first — predict, the std/nf captures, props and the exchange
capture draw on the same OddsAPI pool):
  (a) START: no spending unless x-requests-remaining >= OU15_START_THRESHOLD (60,000) — i.e. the
      monthly reset has happened.
  (b) FLOOR: stop when remaining would drop below
          floor = max(OU15_FLOOR_MIN 25,000,
                      live_burn/day x days_to_next_reset x OU15_RESERVE_SAFETY 1.15)
      live_burn is MEASURED from the ledger this script keeps (account usage minus our own spend,
      last 8 days), and never taken below OU15_LIVE_BURN_PER_DAY (3,400 — measured 2026-10-28:
      92,630 used by day ~28 of the cycle, reset assumed on the 1st). Days to the next reset come
      from the reset the ledger observed (+OU15_CYCLE_DAYS 30); unknown = a full cycle.
      Set OU15_DYNAMIC_RESERVE=0 to use the fixed 25,000 floor alone (owner decision only).
  (c) per-run cap OU15_RUN_CAP (8,000 credits).
  (d) HTTP 401 / 422 / 429 stop the run at once with a clear message (exit 3); what was already
      captured is written first.
Every response's x-requests-remaining is checked; the run stops before a call that could cross the
floor or the cap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.the-odds-api.com/v4"
OUT_DIR = ROOT / "output" / "research"
OUT_CSV = OUT_DIR / "std_ou15_history_backfill.csv"
STATE = OUT_DIR / "std_ou15_backfill_state.json"
TRAINING_FILE = ROOT / "output" / "standard_sidemarket_odds_history.csv"   # READ ONLY (names)

LEAGUES = {                       # league -> OddsAPI sport key (matches config.ODDS_API_SPORT_KEYS)
    "Championship": "soccer_efl_champ",
    "League One": "soccer_england_league1",
    "League Two": "soccer_england_league2",
    "Bundesliga 2": "soccer_germany_bundesliga2",
    "La Liga 2": "soccer_spain_segunda_division",
    "Ligue 2": "soccer_france_ligue_two",
    "Serie B": "soccer_italy_serie_b",
}
# Most recent season first: if one cycle cannot pay for both, the newer one is finished first.
SEASONS = [("2025-2026", date(2025, 7, 25), date(2026, 6, 7)),
           ("2024-2025", date(2024, 7, 26), date(2025, 6, 7))]
SEASON_START = {s: b for s, b, _e in SEASONS}

MARKET_KEY = "alternate_totals"
REGIONS = "eu"
POINTS = {1.5: "15", 3.5: "35"}
PRICE_CLASS = "ODDSAPI_HISTORICAL_SNAPSHOT"
EST_EVENT_COST = 10               # 10 x 1 market x 1 region (probed)
EST_LIST_COST = 1
COLS = ["snapshot_date", "snapshot_ts", "match_date", "kickoff_utc", "league", "match", "market",
        "odds", "price_class", "requested_ts", "event_id", "home_team_oddsapi", "away_team_oddsapi",
        "name_resolved", "point", "side", "n_books", "bookmakers", "odds_min", "odds_max",
        "odds_pinnacle", "book_prices"]
README_NOTE = ("RESEARCH ONLY - not read by any production code. OddsAPI historical snapshot of "
               "full-match O/U 1.5/3.5 (alternate_totals, regions=eu); odds = median over books "
               "posting BOTH sides. Merging into output/standard_sidemarket_odds_history.csv "
               "(training) is a later OWNER decision. Writer: scripts/backfill_std_ou15_history.py")


class HardStop(Exception):
    """401 / 422 / 429: stop everything, keep what was captured."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "") or default))
    except ValueError:
        return default


def settings() -> dict:
    return {
        "start_threshold": _env_int("OU15_START_THRESHOLD", 60000),
        "floor_min": _env_int("OU15_FLOOR_MIN", 25000),
        "run_cap": _env_int("OU15_RUN_CAP", 8000),
        "live_burn_default": _env_int("OU15_LIVE_BURN_PER_DAY", 3400),
        "cycle_days": _env_int("OU15_CYCLE_DAYS", 30),
        "safety": float(os.getenv("OU15_RESERVE_SAFETY", "") or 1.15),
        "dynamic": (os.getenv("OU15_DYNAMIC_RESERVE", "1").strip() or "1") not in {"0", "false", "no"},
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── parsing ──────────────────────────────────────────────────────────────────────────────────────

def book_pairs(event: dict, points=tuple(POINTS)) -> dict:
    """{point: {book_key: (over, under)}} from one event-odds payload's `data`.

    Exact matching only. A book contributes a point when, inside the market whose key is EXACTLY
    "alternate_totals", it has exactly one "Over" and exactly one "Under" at exactly that point,
    neither carrying a `description`, both prices > 1.0. Anything else is dropped for that book.
    """
    out = {p: {} for p in points}
    for bm in event.get("bookmakers") or []:
        bkey = bm.get("key")
        if not bkey:
            continue
        for mk in bm.get("markets") or []:
            if mk.get("key") != MARKET_KEY:
                continue
            for p in points:
                overs, unders = [], []
                for oc in mk.get("outcomes") or []:
                    if oc.get("description"):
                        continue
                    try:
                        pt = float(oc.get("point"))
                        price = float(oc.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if pt != p or not price > 1.0:
                        continue
                    if oc.get("name") == "Over":
                        overs.append(price)
                    elif oc.get("name") == "Under":
                        unders.append(price)
                if len(overs) == 1 and len(unders) == 1:
                    o, u = overs[0], unders[0]
                    if 1.0 <= 1 / o + 1 / u <= 1.25:          # a real two-way book, not garbage
                        out[p][bkey] = (o, u)
    return out


def rows_for_event(payload: dict, meta: dict, requested_ts: str, name_map=None) -> list[dict]:
    """CSV rows (over15/under15/over35/under35) for one historical event-odds response."""
    data = payload.get("data") or {}
    snap = payload.get("timestamp") or requested_ts
    ko = data.get("commence_time") or meta["commence"]
    home_o = data.get("home_team") or meta["home"]
    away_o = data.get("away_team") or meta["away"]
    h, a, resolved = home_o, away_o, False
    if name_map:
        rh, ra = name_map(meta["league"], home_o), name_map(meta["league"], away_o)
        if rh and ra and rh != ra:
            h, a, resolved = rh, ra, True
    rows = []
    for p, books in book_pairs(data).items():
        if not books:
            continue
        names = sorted(books)
        for side, idx in (("over", 0), ("under", 1)):
            prices = [books[b][idx] for b in names]
            rows.append({
                "snapshot_date": snap[:10], "snapshot_ts": snap, "match_date": ko[:10],
                "kickoff_utc": _iso(_parse_ts(ko)), "league": meta["league"],
                "match": f"{h} vs {a}", "market": f"{side}{POINTS[p]}",
                "odds": round(statistics.median(prices), 3), "price_class": PRICE_CLASS,
                "requested_ts": requested_ts, "event_id": meta["id"],
                "home_team_oddsapi": home_o, "away_team_oddsapi": away_o,
                "name_resolved": int(resolved), "point": p, "side": side,
                "n_books": len(names), "bookmakers": ";".join(names),
                "odds_min": min(prices), "odds_max": max(prices),
                "odds_pinnacle": books["pinnacle"][idx] if "pinnacle" in books else "",
                "book_prices": "|".join(f"{b}:{books[b][idx]}" for b in names),
            })
    return rows


def snapshot_ts_for(commence: str) -> str:
    """00:00Z on match day; kickoff - 2h if the match starts within 30 minutes of midnight."""
    ko = _parse_ts(commence)
    snap = ko.replace(hour=0, minute=0, second=0, microsecond=0)
    if ko - snap < timedelta(minutes=30):
        snap = ko - timedelta(hours=2)
    return _iso(snap)


# ── team names (read-only use of the training file's spellings) ──────────────────────────────────

def make_name_map():
    """league, OddsAPI name -> the spelling used in standard_sidemarket_odds_history.csv, or None."""
    try:
        sys.path.insert(0, str(ROOT))
        from src.team_names import resolve
    except Exception:                                                  # noqa: BLE001
        return None
    cands: dict[str, set] = {}
    if TRAINING_FILE.exists():
        with TRAINING_FILE.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lg = r.get("league")
                if lg in LEAGUES and " vs " in (r.get("match") or ""):
                    cands.setdefault(lg, set()).update(r["match"].split(" vs ", 1))
    lists = {lg: sorted(v) for lg, v in cands.items()}
    cache: dict = {}

    def name_map(league, name):
        k = (league, name)
        if k not in cache:
            cache[k] = resolve(name, lists.get(league, []))
        return cache[k]
    return name_map


# ── state + ledger ───────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
    st.setdefault("version", 1)
    st.setdefault("cursor", {})        # "sport|season" -> next date to list (YYYY-MM-DD)
    st.setdefault("events", {})        # id -> {league, sport, season, home, away, commence, status}
    st.setdefault("ledger", [])        # [{ts, used, remaining, spent_total}]
    st.setdefault("spent_total", 0)
    st.setdefault("reset_ts", None)
    st.setdefault("complete", False)
    st["_readme"] = README_NOTE        # the CSV stays a plain CSV; its provenance note lives here
    return st


def save_state(st: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    tmp.replace(STATE)


def record_balance(st: dict, used: int, remaining: int, now: datetime) -> None:
    """Append a ledger point; a DROP in `used` means the monthly reset happened."""
    led = st["ledger"]
    if led and used < led[-1]["used"]:
        st["reset_ts"] = _iso(now)
    led.append({"ts": _iso(now), "used": used, "remaining": remaining,
                "spent_total": st["spent_total"]})
    del led[:-90]


def measured_live_burn(st: dict, now: datetime, window_days: float = 8.0):
    """Credits/day the REST of the estate used (account usage minus this backfill), or None."""
    pts = [p for p in st["ledger"]
           if (now - _parse_ts(p["ts"])).total_seconds() <= window_days * 86400
           and (not st.get("reset_ts") or p["ts"] >= st["reset_ts"])]
    if len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    days = (_parse_ts(b["ts"]) - _parse_ts(a["ts"])).total_seconds() / 86400
    if days < 1.5:
        return None
    live = (b["used"] - a["used"]) - (b["spent_total"] - a["spent_total"])
    return max(0.0, live / days)


def compute_floor(st: dict, now: datetime, cfg: dict) -> tuple[int, str]:
    if not cfg["dynamic"]:
        return cfg["floor_min"], f"fixed floor {cfg['floor_min']}"
    meas = measured_live_burn(st, now)
    burn = max(meas or 0.0, cfg["live_burn_default"])
    if st.get("reset_ts"):
        nxt = _parse_ts(st["reset_ts"]) + timedelta(days=cfg["cycle_days"])
        days_left = max(1.0, (nxt - now).total_seconds() / 86400)
        basis = f"reset seen {st['reset_ts'][:10]}"
    else:
        days_left, basis = float(cfg["cycle_days"]), "reset not yet seen -> full cycle"
    reserve = int(math.ceil(burn * days_left * cfg["safety"]))
    floor = max(cfg["floor_min"], reserve)
    why = (f"live burn {burn:,.0f}/day (measured {'n/a' if meas is None else f'{meas:,.0f}'}, "
           f"min {cfg['live_burn_default']:,}) x {days_left:.1f} days to reset ({basis}) x "
           f"{cfg['safety']} = {reserve:,}; floor = max({cfg['floor_min']:,}, that) = {floor:,}")
    return floor, why


def decide(remaining: int, st: dict, now: datetime, cfg: dict) -> tuple[bool, int, str]:
    """(run?, credit budget for this run, reason). Pure: no network."""
    if st.get("complete"):
        return False, 0, "backfill COMPLETE - nothing to do"
    if remaining < cfg["start_threshold"]:
        return False, 0, (f"remaining {remaining:,} < START_THRESHOLD {cfg['start_threshold']:,} "
                          "(waiting for the monthly reset)")
    floor, why = compute_floor(st, now, cfg)
    budget = min(cfg["run_cap"], remaining - floor)
    if budget < EST_EVENT_COST + EST_LIST_COST:
        return False, 0, f"remaining {remaining:,} leaves no room above the floor: {why}"
    return True, budget, f"budget {budget:,} (cap {cfg['run_cap']:,}); {why}"


# ── network ──────────────────────────────────────────────────────────────────────────────────────

class Client:
    def __init__(self, key: str, budget: int, floor: int, session=None):
        import requests
        self.key, self.budget, self.floor = key, budget, floor
        self.http = session or requests.Session()
        self.spent, self.remaining, self.used = 0, None, None

    def can_afford(self, est: int) -> bool:
        if self.spent + est > self.budget:
            return False
        return self.remaining is None or self.remaining - est >= self.floor

    def get(self, path: str, **params):
        params["apiKey"] = self.key
        for attempt in range(3):
            try:
                r = self.http.get(f"{BASE}{path}", params=params, timeout=60)
                break
            except Exception as e:                                       # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"network error on {path.split('?')[0]}: {type(e).__name__}")
                time.sleep(3)
        self.spent += int(float(r.headers.get("x-requests-last") or 0))
        if r.headers.get("x-requests-remaining") is not None:
            self.remaining = int(float(r.headers["x-requests-remaining"]))
        if r.headers.get("x-requests-used") is not None:
            self.used = int(float(r.headers["x-requests-used"]))
        if r.status_code in (401, 422, 429):
            msg = {401: "401 Unauthorized - ODDS_API_KEY rejected or plan lacks historical access",
                   422: "422 Unprocessable - request rejected (market/date/params)",
                   429: "429 Too Many Requests - rate limited / quota exhausted"}[r.status_code]
            raise HardStop(f"{msg}: {r.text[:160]}")
        return r


def balance(key: str, session=None) -> tuple[int, int]:
    """(used, remaining) via GET /v4/sports — costs 0 credits."""
    c = Client(key, 0, 0, session)
    r = c.get("/sports")
    if r.status_code != 200 or c.remaining is None:
        raise RuntimeError(f"balance check failed: HTTP {r.status_code}")
    return int(c.used or 0), int(c.remaining)


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        w =csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def _discover(c: Client, st: dict, sport: str, league: str, season: str, end: date,
              today: date) -> str | None:
    """List events day by day from the cursor; returns a stop reason or None when finished."""
    ck = f"{sport}|{season}"
    d = date.fromisoformat(st["cursor"].get(ck) or SEASON_START[season].isoformat())
    last = min(end, today - timedelta(days=1))
    n_new = 0
    while d <= last:
        if not c.can_afford(EST_LIST_COST):
            st["cursor"][ck] = d.isoformat()
            return "budget/floor reached during discovery"
        r = c.get(f"/historical/sports/{sport}/events", date=f"{d.isoformat()}T00:00:00Z")
        if r.status_code == 200:
            for ev in (r.json() or {}).get("data") or []:
                eid, ko = ev.get("id"), ev.get("commence_time") or ""
                if not eid or not ko:
                    continue
                kd = date.fromisoformat(ko[:10])
                if not (SEASON_START[season] <= kd <= end):
                    continue
                old = st["events"].get(eid)
                if old and old["status"] != "pending":
                    continue
                n_new += old is None
                # a later list carries the fresher kickoff (postponements, TV moves)
                st["events"][eid] = {"league": league, "sport": sport, "season": season,
                                     "home": ev.get("home_team"), "away": ev.get("away_team"),
                                     "commence": ko, "status": "pending"}
        elif r.status_code != 404:
            print(f"  [discover] {league} {d}: HTTP {r.status_code} - will retry next run")
            st["cursor"][ck] = d.isoformat()
            return None
        d += timedelta(days=1)
        st["cursor"][ck] = d.isoformat()
    print(f"  [discover] {league} {season}: listed to {st['cursor'][ck]}, +{n_new} events")
    return None


def run(key: str, budget: int, floor: int, batch: int = 50, session=None) -> int:
    st = load_state()
    c = Client(key, budget, floor, session)
    name_map = make_name_map()
    today = _now().date()
    stop, buf, n_done = None, [], 0

    def flush():
        append_rows(buf)
        buf.clear()
        st["spent_total"] = st.get("spent_total", 0) + c.spent - flush.seen
        flush.seen = c.spent
        save_state(st)
    flush.seen = 0

    try:
        for season, _start, end in SEASONS:
            for league, sport in LEAGUES.items():
                stop = _discover(c, st, sport, league, season, end, today)
                flush()
                if stop:
                    break
                pend = sorted((e for e, m in st["events"].items()
                               if m["status"] == "pending" and m["season"] == season and m["sport"] == sport),
                              key=lambda e: st["events"][e]["commence"])
                for eid in pend:
                    m = st["events"][eid]
                    if _parse_ts(m["commence"]) > _now():
                        continue
                    if not c.can_afford(EST_EVENT_COST):
                        stop = "budget/floor reached"
                        break
                    ts = snapshot_ts_for(m["commence"])
                    r = c.get(f"/historical/sports/{sport}/events/{eid}/odds", date=ts,
                              regions=REGIONS, markets=MARKET_KEY, oddsFormat="decimal")
                    if r.status_code == 200:
                        rows = rows_for_event(r.json() or {}, dict(m, id=eid), ts, name_map)
                        m["status"] = "done" if rows else "no_two_sided_price"
                        buf.extend(rows)
                    elif r.status_code == 404:
                        m["status"] = "not_found_at_snapshot"
                    else:
                        print(f"  [odds] {eid}: HTTP {r.status_code} - left pending")
                        continue
                    n_done += 1
                    if n_done % batch == 0:
                        flush()
                        print(f"  [odds] {n_done} events, {c.spent:,} credits, remaining {c.remaining}")
                flush()
                if stop:
                    break
            if stop:
                break
    except HardStop as e:
        flush()
        print(f"[ou15] HARD STOP: {e}")
        return 3
    finally:
        if buf:
            flush()
    pending = sum(m["status"] == "pending" for m in st["events"].values())
    listed = all(st["cursor"].get(f"{sp}|{s}", "") > e.isoformat()
                 for s, _b, e in SEASONS for sp in LEAGUES.values())
    st["complete"] = bool(listed and pending == 0)
    if c.used is not None and c.remaining is not None:
        record_balance(st, c.used, c.remaining, _now())
    save_state(st)
    print(f"[ou15] events processed {n_done}, credits spent {c.spent:,}, remaining {c.remaining}, "
          f"pending {pending}, complete={st['complete']}{' - stopped: ' + stop if stop else ''}")
    return 0


def status() -> None:
    st = load_state()
    by = {}
    for m in st["events"].values():
        k = (m["season"], m["league"])
        by.setdefault(k, {}).setdefault(m["status"], 0)
        by[k][m["status"]] += 1
    for k in sorted(by):
        print(k, by[k])
    print("spent_total", st["spent_total"], "complete", st["complete"], "reset_ts", st["reset_ts"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="0-credit balance check + decision only")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        return 0
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        print("[ou15] ODDS_API_KEY not set - nothing done")
        return 0
    cfg = settings()
    st = load_state()
    now = _now()
    used, remaining = balance(key)
    record_balance(st, used, remaining, now)
    save_state(st)
    go, budget, why = decide(remaining, st, now, cfg)
    print(f"[ou15] account remaining {remaining:,}; DECISION={'RUN' if go else 'SKIP'}: {why}")
    gh = os.getenv("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"run={'true' if go else 'false'}\n")
    if a.check or not go:
        return 0
    floor, _ = compute_floor(st, now, cfg)
    return run(key, budget, floor)


if __name__ == "__main__":
    raise SystemExit(main())
