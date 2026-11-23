"""Per-book OddsAPI near-kickoff close capture (owner decision 2026-11-23, "approve all").

WHY. Certified CLV compares the ENTRY price with a CLOSE observed <= 30 min before kickoff. The entry is the
first-listed OddsAPI eu book quoting the side (src/predict._fetch_odds_api), but 427 of 510 certified closes
since 2026-10-23 were Bet365 captures via API-Football: a cross-book CLV (reports/P1_PRICE_CLV_COVERAGE.md,
sections 2, 7, 10). This capture buys the SAME OddsAPI feed (region eu, market totals: every book's lines)
twice per (league, kickoff slot) near kickoff, so a same-book close exists for the entry book that predict now
records (odds_over25_book / odds_under25_book, ledger `entry_bookmaker`, shadow `entry_bookmaker`).

BANDS. One bulk /odds call per (sport key, kickoff instant) and band:
    CLOSE_30  (10, 30] minutes before kickoff
    CLOSE_10  ( 1, 10] minutes before kickoff
`commenceTimeFrom/To` = kickoff -/+ 1 min, so every event of that league kicking off at that instant is
priced by the one call. Cost = markets x regions = 1 credit (OddsAPI bills x-requests-last; we book the
header, never an assumption, and an answer without it stops every further paid call of the run).

BUDGET (all enforced BEFORE a paid call, see Budget.can_spend):
  * floor: x-requests-remaining - cost >= FLOOR_MIN (260,000, the live reserve of the OddsAPI plan);
    a lower floor is refused at start;
  * per-run, per-UTC-day and per-month credit caps, journaled in <store>/ledger/usage.json;
  * remaining unknown (no free call answered) -> no paid call;
  * at most MAX_ATTEMPTS paid attempts per (slot, band): a provider error is retried once, never looped.

PRIVATE. Prices are written ONLY under --store (a clone of the private repo NevixAA/wowza-exec, branch
`oddsapi-book-close`); the script refuses a store inside this public repo. Logs carry counts and credits,
never a price and never the key.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

FLOOR_MIN = 260_000                 # memory oddsapi-5m-plan: live reserve 260k; never spend below it
CALL_COST = 1                       # totals x eu
REGIONS = "eu"                      # the entry feed's region (src/predict._fetch_odds_api)
MARKETS = "totals"
BANDS = (("CLOSE_30", 10.0, 30.0), ("CLOSE_10", 1.0, 10.0))   # (name, lo, hi]: minutes before kickoff
TEST_BAND = "TEST"
MAX_ATTEMPTS = 2
SLOT_WINDOW = timedelta(minutes=1)

ROW_COLS = ["captured_at", "band", "minutes_to_kickoff", "league", "sport_key", "kickoff_utc", "match_date",
            "event_id", "home_team", "away_team", "bookmaker", "book_last_update", "market", "side", "point",
            "odds", "source"]
JOURNAL_COLS = ["sport_key", "kickoff_utc", "band", "attempted_at", "status", "http_status", "events", "rows",
                "credits", "run_id", "mode"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── slots ──────────────────────────────────────────────────────────────────────────────────────────
def slots_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Distinct (league, sport_key, kickoff_utc) of the board. Rows without a sport key or a parseable
    kickoff cannot be priced by this feed and are dropped (counted by the caller), never guessed."""
    cols = {"league", "_oa_sport_key", "kickoff_utc"}
    if pred is None or pred.empty or not cols <= set(pred.columns):
        return pd.DataFrame(columns=["league", "sport_key", "ko"])
    d = pd.DataFrame({"league": pred["league"].astype(str),
                      "sport_key": pred["_oa_sport_key"].astype("object"),
                      "ko": pd.to_datetime(pred["kickoff_utc"], errors="coerce", utc=True)})
    d = d[d["sport_key"].notna() & (d["sport_key"].astype(str).str.strip() != "") & d["ko"].notna()]
    d = d.assign(sport_key=d["sport_key"].astype(str).str.strip())
    return d.drop_duplicates(["sport_key", "ko"]).sort_values("ko").reset_index(drop=True)


def band_of(minutes_to_ko: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo < minutes_to_ko <= hi:
            return name
    return None


def due(slots: pd.DataFrame, journal: pd.DataFrame, now: datetime) -> list[dict]:
    """(slot, band) pairs whose band is open NOW and that are neither captured (status OK / EMPTY) nor out of
    attempts. Earliest kickoff first."""
    done, tries = set(), Counter()
    if journal is not None and not journal.empty:
        for r in journal.itertuples(index=False):
            k = (str(r.sport_key), str(r.kickoff_utc), str(r.band))
            tries[k] += 1
            if str(r.status) in ("OK", "EMPTY"):
                done.add(k)
    out = []
    for s in slots.itertuples(index=False):
        mins = (s.ko.to_pydatetime() - now).total_seconds() / 60.0
        b = band_of(mins)
        if b is None:
            continue
        k = (s.sport_key, iso(s.ko.to_pydatetime()), b)
        if k in done or tries[k] >= MAX_ATTEMPTS:
            continue
        out.append({"league": s.league, "sport_key": s.sport_key, "ko": s.ko.to_pydatetime(), "band": b,
                    "minutes_to_kickoff": round(mins, 2)})
    return out


def next_band_open(slots: pd.DataFrame, now: datetime) -> datetime | None:
    """The earliest future instant a band opens (kickoff - hi). None when no band opens after now."""
    best = None
    for s in slots.itertuples(index=False):
        ko = s.ko.to_pydatetime()
        for _, lo, hi in BANDS:
            t = ko - timedelta(minutes=hi) + timedelta(seconds=1)
            if t > now and (best is None or t < best):
                best = t
    return best


# ── budget ─────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Budget:
    """Credit guard. `remaining` comes only from a provider header; caps are credits."""
    floor: int
    run_cap: int
    day_cap: int
    month_cap: int
    usage: dict = field(default_factory=lambda: {"by_day": Counter(), "by_month": Counter()})
    remaining: int | None = None
    run_spent: int = 0
    cost_unknown: bool = False
    stopped: str | None = None
    calls_paid: int = 0
    calls_free: int = 0

    def __post_init__(self):
        if self.floor < FLOOR_MIN:
            raise ValueError(f"floor {self.floor} is below the OddsAPI reserve {FLOOR_MIN}: refused")
        for k in ("run_cap", "day_cap", "month_cap"):
            if int(getattr(self, k)) < 0:
                raise ValueError(f"{k} must be >= 0")

    @classmethod
    def load(cls, path: Path, **kw) -> "Budget":
        b = cls(**kw)
        if path.exists():
            u = json.loads(path.read_text(encoding="utf-8"))
            b.usage = {"by_day": Counter(u.get("by_day", {})), "by_month": Counter(u.get("by_month", {}))}
        return b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: dict(v) for k, v in self.usage.items()}, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)

    def can_spend(self, cost: int, now: datetime | None = None) -> bool:
        now = now or utcnow()
        day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        why = None
        if self.cost_unknown:
            why = "a paid answer carried no x-requests-last: cost unknown, no further paid call"
        elif self.remaining is None:
            why = "remaining credits unknown: no paid call"
        elif self.remaining - cost < self.floor:
            why = f"floor: {self.remaining} remaining - {cost} < {self.floor}"
        elif self.run_spent + cost > self.run_cap:
            why = f"run cap {self.run_cap}"
        elif self.usage["by_day"][day] + cost > self.day_cap:
            why = f"day cap {self.day_cap}"
        elif self.usage["by_month"][month] + cost > self.month_cap:
            why = f"month cap {self.month_cap}"
        if why:
            self.stopped = why
            return False
        return True

    def read_headers(self, headers) -> int | None:
        """Update `remaining`; return x-requests-last (this call's cost) or None when absent."""
        h = headers or {}
        try:
            if h.get("x-requests-remaining") not in (None, ""):
                self.remaining = int(float(h["x-requests-remaining"]))
        except (TypeError, ValueError):
            pass
        v = h.get("x-requests-last")
        if v in (None, ""):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    def book(self, spent: int, now: datetime | None = None) -> None:
        now = now or utcnow()
        self.run_spent += spent
        self.usage["by_day"][now.strftime("%Y-%m-%d")] += spent
        self.usage["by_month"][now.strftime("%Y-%m")] += spent


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def quote_rows(events: list, *, league: str, sport_key: str, band: str, captured: datetime) -> list[dict]:
    """Every book's totals outcome of every pre-kickoff event. A price <= 1.0 is no price (suspended)
    and an event already started is in-play: both are dropped."""
    rows = []
    for ev in events or []:
        ko = pd.to_datetime(ev.get("commence_time"), errors="coerce", utc=True)
        if pd.isna(ko):
            continue
        ko = ko.to_pydatetime()
        if ko <= captured:
            continue
        mins = round((ko - captured).total_seconds() / 60.0, 2)
        for bm in ev.get("bookmakers") or []:
            for mk in bm.get("markets") or []:
                if mk.get("key") != "totals":
                    continue
                for o in mk.get("outcomes") or []:
                    nm, pt, pr = o.get("name"), o.get("point"), o.get("price")
                    try:
                        pr = float(pr)
                        pt = float(pt)
                    except (TypeError, ValueError):
                        continue
                    if nm not in ("Over", "Under") or pr <= 1.0:
                        continue
                    rows.append({"captured_at": iso(captured), "band": band, "minutes_to_kickoff": mins,
                                 "league": league, "sport_key": sport_key, "kickoff_utc": iso(ko),
                                 "match_date": ko.strftime("%Y-%m-%d"), "event_id": ev.get("id", ""),
                                 "home_team": ev.get("home_team", ""), "away_team": ev.get("away_team", ""),
                                 "bookmaker": bm.get("key", ""),
                                 "book_last_update": mk.get("last_update") or bm.get("last_update") or "",
                                 "market": f"OU{str(pt).replace('.', '')}", "side": nm.upper(), "point": pt,
                                 "odds": pr, "source": f"oddsapi_{REGIONS}_{MARKETS}"})
    return rows


def append_rows(store: Path, rows: list[dict], sub: str = "data") -> int:
    """Append to <store>/<sub>/<YYYY-MM>/<match_date>.csv (by kickoff date). Returns rows written."""
    if not rows:
        return 0
    d = pd.DataFrame(rows, columns=ROW_COLS)
    for day, g in d.groupby("match_date"):
        p = store / sub / day[:7] / f"{day}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(p, mode="a", header=not p.exists(), index=False)
    return len(d)


def read_journal(store: Path) -> pd.DataFrame:
    p = store / "ledger" / "journal.csv"
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=JOURNAL_COLS)


def append_journal(store: Path, rec: dict) -> None:
    p = store / "ledger" / "journal.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec], columns=JOURNAL_COLS).to_csv(p, mode="a", header=not p.exists(), index=False)
