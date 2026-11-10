"""
OddsAPI HISTORICAL collector (owner 2026-11-10: "use all the 5M credits this month to achieve more
data and gain more edge in all the models"). CODE ONLY: the bought history is stored in the PRIVATE
repo NevixAA/wowza-exec, orphan branch `odds-history`, never in a public repo and never in a log.
=================================================================================================
Terms of use (The Odds API): storing and training models are allowed, REDISTRIBUTION IS PROHIBITED.
So nothing here prints a price, a club name, an event id or a URL: logs show counts and credits only.

Pricing (provider docs, v4 guide, read 2026-11-10):
  * /historical/sports/{sport}/odds (featured h2h / totals, one snapshot of a whole sport):
    10 x markets x regions, free when empty, history from 2020-06-06 (5-minute snapshots from 2022-09);
  * /historical/sports/{sport}/events/{id}/odds (additional markets, props, correct_score):
    10 x markets RETURNED x regions, free when empty, history only after 2023-05-03T05:30:00Z;
  * /historical/sports/{sport}/events: 1 credit, free when empty;
  * /sports: free (used to read x-requests-remaining before the first paid call).

The universe is OUR fixtures (the committed banks af_history / fd_history with the owner's history
corrections, plus player_history for the props-only leagues), restricted to leagues with an OddsAPI
sport key. A provider event is bought ONLY when it is the provider's own listing of one of our
fixtures, proven by src.team_names.resolve within ONE league and ONE match day (never a cross-league
join); so the dry-run count is an exact upper bound of the calls. Unmatched fixtures and unmatched
provider events are recorded in the private ledger, never guessed.

Tiers run strictly in the owner's priority order; a later tier never re-buys a market an earlier
tier already requested for the same match / league-day, snapshot and region (coverage rule).
HARD FLOOR: no call is made when x-requests-remaining minus the call's maximum cost would fall below
MIN_FLOOR (260,000: about 102k live burn to Dec 1 + 150k safety, owner 2026-11-10); a per-run cap
and a wall-clock budget bound each run. A call whose cost the provider does not report stops the run.

Nothing in this module has a default price, a filled value or a guessed kickoff: a missing price is
not a row, a fixture without a provider event is not bought, a league-day without a matched event
has no snapshot.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.the-odds-api.com/v4"

# ── provider facts (docs) and owner rules ─────────────────────────────────────────────────────────
FEATURED_HISTORY_START = datetime(2020, 6, 6, tzinfo=timezone.utc)
EVENT_HISTORY_START = datetime(2023, 5, 3, 5, 30, tzinfo=timezone.utc)
CREDITS_PER_MARKET_REGION = 10
LOOKUP_MAX_COST = 1
PLAN_CREDITS = 5_000_000          # the owner's plan this month (owner 2026-11-10)
MIN_FLOOR = 260_000               # owner 2026-11-10: live needs ~102k to Dec 1 + 150k safety margin
HISTORY_END = "2026-11-09"        # the last COMPLETE match day when this plan was written (2026-11-10)

# A local match DATE of ours covers kickoffs from 12:00 UTC the day before to 12:00 UTC the day after
# (J-League evenings are the same UTC day, Liga MX / MLS nights are the next UTC day). Matching is by
# club names inside this window, so the window only bounds the candidates; it never pairs anything.
DAY_WINDOW_BEFORE = timedelta(hours=12)
DAY_WINDOW_AFTER = timedelta(hours=36)
LOOKUP_LEAD = timedelta(hours=6)   # the events listing is read 6h before the window: 06:00 UTC the day before

CLOSE = ("close", timedelta(minutes=-5))
OPEN48 = ("T-48h", timedelta(hours=-48))
EIGHT = (("T-72h", timedelta(hours=-72)), ("T-48h", timedelta(hours=-48)), ("T-24h", timedelta(hours=-24)),
         ("T-12h", timedelta(hours=-12)), ("T-6h", timedelta(hours=-6)), ("T-2h", timedelta(hours=-2)),
         ("T-1h", timedelta(hours=-1)), CLOSE)

PROBE_FILE = ROOT / "output" / "collect" / "oddsapi_history_probe.json"
S_PROBE_NAME = "probe_s.json"      # written into the PRIVATE store by --probe-s

# exit codes of a run
EXIT_DONE, EXIT_TIME, EXIT_RUN_CAP, EXIT_FLOOR, EXIT_ERROR = 0, 10, 20, 30, 40


@dataclass(frozen=True)
class Tier:
    name: str
    kind: str                        # "featured" (sport snapshot per league-day) | "event" (per match)
    markets: tuple                   # requested markets; "props" tiers take them from the probe per league
    regions: tuple
    offsets: tuple                   # ((label, timedelta), ...) relative to kickoff (event) / first kickoff
    leagues: str                     # "team" | "new_format" | "props"
    start: str
    end: str
    gate: str | None = None          # None | "props_probe" | "s_probe"
    purpose: str = ""


TIERS: tuple[Tier, ...] = (
    Tier("B1", "event", ("btts", "alternate_totals"), ("eu",), (CLOSE,), "team", "2023-05-03", HISTORY_END,
         purpose="BTTS + every alternate total line (O0.5..O6.5+) at close: bars evidence + specials"),
    Tier("B2", "featured", ("totals",), ("eu",), (CLOSE,), "new_format", "2020-06-06", "2023-05-02",
         purpose="New-Format O/U main line before the event-market history starts"),
    Tier("B4", "featured", ("h2h", "totals"), ("eu",), EIGHT, "team", "2023-06-14", "2026-06-13",
         purpose="sharp-money line movement, 8 snapshots per league-day"),
    Tier("B3", "event", ("props",), ("uk", "eu", "us", "us2"), (CLOSE,), "props", "2023-05-03", HISTORY_END,
         gate="props_probe", purpose="player props at close (probe-confirmed markets only)"),
    Tier("S", "event", ("correct_score",), ("uk", "eu"), (CLOSE,), "team", "2023-05-03", HISTORY_END,
         gate="s_probe", purpose="correct score grid at close (specials); only leagues the S probe confirms"),
    Tier("C1", "event", ("btts", "alternate_totals"), ("eu",), (OPEN48,), "team", "2023-05-03", HISTORY_END,
         purpose="B1 at T-48h: line movement / CLV evidence"),
    Tier("C3", "featured", ("h2h", "totals"), ("eu",), (OPEN48, CLOSE), "team", "2020-06-06", "2023-06-13",
         purpose="1X2 + O/U open and close before B4"),
    Tier("C2", "event", ("props",), ("uk", "eu", "us", "us2"), (OPEN48,), "props", "2023-05-03", HISTORY_END,
         gate="props_probe", purpose="props at T-48h: props CLV"),
    Tier("C4", "event", ("btts", "alternate_totals"), ("uk",), (CLOSE,), "team", "2023-05-03", HISTORY_END,
         purpose="B1 with the uk books added"),
    Tier("C5", "featured", ("h2h", "totals"), ("eu",), EIGHT, "team", "2020-06-06", "2023-06-13",
         purpose="B4's 8 snapshots back to 2020-06-06 (only if credits remain)"),
)
TIER_ORDER = tuple(t.name for t in TIERS)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)


def day_window(league_date: str) -> tuple[datetime, datetime]:
    d = datetime.strptime(league_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d - DAY_WINDOW_BEFORE, d + DAY_WINDOW_AFTER


# ── league facts (registry; no table typed here) ──────────────────────────────────────────────────
def sport_key(league: str) -> str | None:
    from src import league_registry as LR
    return LR.odds_api_sport_key(league)


def new_format_leagues() -> set:
    from src import league_registry as LR
    return set(LR.derive().new_format)


def probe_props_markets(probe_file: Path = PROBE_FILE) -> dict:
    """{league: (markets...)} the P7 probe saw RETURNED on its probed event (EPL 4, Ligue 1 2, MLS 3
    on 2026-11-09). A league the probe found nothing for is absent: nothing is bought there."""
    if not Path(probe_file).exists():
        return {}
    p = json.loads(Path(probe_file).read_text(encoding="utf-8"))
    out = {}
    for lg, r in (p.get("leagues") or {}).items():
        mk = sorted(((r.get("props") or {}).get("markets") or {}).keys())
        if mk:
            out[lg] = tuple(mk)
    return out


def s_probe_leagues(store_root: Path | None) -> set | None:
    """Leagues where the S probe saw `correct_score` returned. None = the probe has not run (the
    S tier is then NOT run; the dry run counts it for every team league as the maximum)."""
    if store_root is None:
        return None
    f = Path(store_root) / "ledger" / S_PROBE_NAME
    if not f.exists():
        return None
    p = json.loads(f.read_text(encoding="utf-8"))
    return {lg for lg, r in (p.get("leagues") or {}).items() if r.get("correct_score_returned")}


# ── our fixture universe ─────────────────────────────────────────────────────────────────────────
def load_fixtures(root: Path = ROOT) -> pd.DataFrame:
    """[league, date (YYYY-MM-DD), home, away, source] from the committed banks, with the owner's
    history corrections (fd_history through serve_history, af_history through the af rule), keyed
    leagues only. player_history adds the props-only leagues (Premier League, Ligue 1, ...)."""
    from src.af_history_corrections import apply_af_history_corrections
    from src.history_corrections import serve_history
    parts = []
    af = root / "output" / "af_history.parquet"
    if af.exists():
        a = apply_af_history_corrections(pd.read_parquet(af), tag="oddsapi_history.af_history")
        parts.append(a.assign(source="af")[["league", "date", "home_team", "away_team", "source"]])
    fd = root / "output" / "fd_history.parquet"
    if fd.exists():
        f = serve_history(pd.read_parquet(fd), tag="oddsapi_history.fd_history")
        parts.append(f.assign(source="fd")[["league", "date", "home_team", "away_team", "source"]])
    ph = root / "player_history.parquet"
    if ph.exists():
        import pyarrow.dataset as ds
        p = ds.dataset(str(ph), format="parquet").to_table(
            columns=["league", "date", "home_team", "away_team"]).to_pandas().drop_duplicates()
        parts.append(p.assign(source="ph"))
    if not parts:
        return pd.DataFrame(columns=["league", "date", "home", "away", "source"])
    h = pd.concat(parts, ignore_index=True)
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h = h.dropna(subset=["league", "date", "home_team", "away_team"])
    keyed = {lg for lg in h["league"].unique() if sport_key(lg)}
    h = h[h["league"].isin(keyed)]
    return (h.rename(columns={"home_team": "home", "away_team": "away"})
             .drop_duplicates(["league", "date", "home", "away", "source"])
             .reset_index(drop=True)[["league", "date", "home", "away", "source"]])


_SOURCE_RANK = {"af": 0, "fd": 1, "ph": 2}


def dedupe_day(rows: list[tuple]) -> list[tuple]:
    """One league-day's fixtures across the sources -> one list of (home, away). The source with the
    most fixtures is the base; another source's fixture is the SAME match when either club resolves
    (src.team_names.resolve) onto a club already playing that day (a club plays once a day); a
    fixture neither of whose clubs resolves is a different match and is added."""
    from src.team_names import resolve
    by_src: dict = {}
    for h, a, s in rows:
        by_src.setdefault(s, []).append((h, a))
    order = sorted(by_src, key=lambda s: (-len(by_src[s]), _SOURCE_RANK[s]))
    out = list(dict.fromkeys(by_src[order[0]]))
    for s in order[1:]:
        for h, a in by_src[s]:
            playing = [x for f in out for x in f]
            if (h, a) in out or resolve(h, playing) or resolve(a, playing):
                continue
            out.append((h, a))
    return out


def fixture_days(fixtures: pd.DataFrame) -> dict:
    """{(league, date): [(home, away), ...]} deduplicated across the sources."""
    out = {}
    for (lg, d), g in fixtures.groupby(["league", "date"], sort=True):
        out[(lg, d)] = dedupe_day(list(zip(g["home"], g["away"], g["source"])))
    return out


# ── the plan (abstract units; identical in the dry run and the run) ───────────────────────────────
@dataclass(frozen=True)
class Unit:
    tier: str
    kind: str
    league: str
    date: str
    home: str | None          # event tiers: our fixture; featured: None
    away: str | None
    label: str
    offset_s: int
    markets: tuple
    regions: tuple

    @property
    def max_cost(self) -> int:
        return CREDITS_PER_MARKET_REGION * len(self.markets) * len(self.regions)


def _tier_leagues(t: Tier, days: dict, props: dict, s_ok: set | None, nf: set) -> set:
    present = {lg for lg, _ in days}
    if t.leagues == "props":
        return present & set(props)
    team = {lg for lg in present if not _props_only(lg)}
    if t.leagues == "new_format":
        return team & nf
    if t.gate == "s_probe" and s_ok is not None:
        return team & s_ok
    return team


def _props_only(league: str) -> bool:
    """A props-only league (top divisions / cups) has no team model: no bars, no team tiers."""
    from src import league_registry as LR
    e = LR.entry(league) or {}
    return e.get("model_type") not in ("standard", "new_format")


def build_plan(days: dict, tiers: Iterable[Tier] = TIERS, props: dict | None = None,
               s_ok: set | None = None, nf: set | None = None) -> dict:
    """{tier: [Unit...]} plus the league-days each tier needs a lookup for (first tier that needs it).
    Coverage rule: (kind, league, date, home, away, offset, region) -> markets already requested;
    a later tier asks only for what is missing and drops a unit with nothing missing."""
    props = probe_props_markets() if props is None else props
    nf = new_format_leagues() if nf is None else nf
    cov: dict = {}
    looked: set = set()
    plan = {"units": {}, "lookups": {}}
    for t in tiers:
        lgs = _tier_leagues(t, days, props, s_ok, nf)
        units, lookups = [], []
        for (lg, d), fx in days.items():
            if lg not in lgs or not (t.start <= d <= t.end):
                continue
            mk = props[lg] if t.markets == ("props",) else t.markets
            anchors = [(None, None)] if t.kind == "featured" else list(fx)
            new = False
            for h, a in anchors:
                for label, off in t.offsets:
                    ident = (t.kind, lg, d, h, a, int(off.total_seconds()))
                    miss = {}
                    for r in t.regions:
                        m = tuple(x for x in mk if x not in cov.get(ident + (r,), set()))
                        if m:
                            miss.setdefault(m, []).append(r)
                    for m, rs in miss.items():
                        units.append(Unit(t.name, t.kind, lg, d, h, a, label, int(off.total_seconds()),
                                          m, tuple(rs)))
                        for r in rs:
                            cov.setdefault(ident + (r,), set()).update(m)
                        new = True
            if new and (lg, d) not in looked:
                looked.add((lg, d))
                lookups.append((lg, d))
        plan["units"][t.name] = units
        plan["lookups"][t.name] = lookups
    return plan


def summarise_plan(plan: dict, days: dict, sec_per_call: tuple = (0.5, 1.0),
                   conditional: tuple = ("Saudi Pro League",)) -> dict:
    """Exact calls and MAXIMUM credits per tier (a call is billed only for markets returned; empty
    answers are free, so the real spend is at most this)."""
    usable = PLAN_CREDITS - MIN_FLOOR
    out, cum = {"tiers": {}, "order": list(plan["units"]), "usable_credits": usable,
                "plan_credits": PLAN_CREDITS, "floor": MIN_FLOOR, "history_end": HISTORY_END}, 0
    tiers = {t.name: t for t in TIERS}
    for name, units in plan["units"].items():
        lk = plan["lookups"][name]
        cr = sum(u.max_cost for u in units) + LOOKUP_MAX_COST * len(lk)
        calls = len(units) + len(lk)
        cond = [u for u in units if u.league in conditional]
        cond_lk = [x for x in lk if x[0] in conditional]
        per_league, per_year = Counter(), Counter()
        for u in units:
            per_league[u.league] += u.max_cost
            per_year[u.date[:4]] += u.max_cost
        for lg, d in lk:
            per_league[lg] += LOOKUP_MAX_COST
            per_year[d[:4]] += LOOKUP_MAX_COST
        cum += cr
        t = tiers[name]
        out["tiers"][name] = {
            "kind": t.kind, "markets": list(t.markets), "regions": list(t.regions),
            "snapshots": [lab for lab, _ in t.offsets], "window": [t.start, t.end], "gate": t.gate,
            "purpose": t.purpose,
            "league_days": len({(u.league, u.date) for u in units}),
            "matches": len({(u.league, u.date, u.home, u.away) for u in units if u.kind == "event"}),
            "odds_calls": len(units), "lookup_calls": len(lk), "calls": calls,
            "max_credits": cr, "max_credits_odds": cr - len(lk), "max_credits_lookups": len(lk),
            "conditional_saudi_credits": sum(u.max_cost for u in cond) + len(cond_lk),
            "cumulative_max_credits": cum, "fits_in_usable": cum <= usable,
            "hours_at_sec_per_call": {str(s): round(calls * s / 3600, 1) for s in sec_per_call},
            "max_credits_by_league": dict(sorted(per_league.items())),
            "max_credits_by_year": dict(sorted(per_year.items())),
        }
    out["total_max_credits"] = cum
    out["total_calls"] = sum(v["calls"] for v in out["tiers"].values())
    return out


# ── provider-event matching (one league, one match day) ───────────────────────────────────────────
def match_events(fixtures: list[tuple], events: list[dict], league_date: str) -> tuple[dict, dict]:
    """({(home, away): event}, report). An event is ours only when BOTH clubs resolve
    (src.team_names.resolve, OddsAPI spelling -> our spelling) onto one fixture of this league-day,
    its kickoff is inside the day window, and the pairing is one-to-one; anything else is reported."""
    from src.team_names import resolve
    lo, hi = day_window(league_date)
    homes, aways = [h for h, _ in fixtures], [a for _, a in fixtures]
    cand: dict = {}
    rep = {"events_in_window": 0, "unmatched_events": [], "ambiguous": [], "unmatched_fixtures": []}
    for e in events:
        try:
            ko = parse_iso(e["commence_time"])
        except Exception:                                           # noqa: BLE001
            continue
        if not (lo <= ko < hi):
            continue
        rep["events_in_window"] += 1
        rh, ra = resolve(str(e.get("home_team") or ""), homes), resolve(str(e.get("away_team") or ""), aways)
        if rh and ra and (rh, ra) in fixtures:
            cand.setdefault((rh, ra), []).append(e)
        else:
            rep["unmatched_events"].append((e.get("home_team"), e.get("away_team")))
    out = {}
    for fx, es in cand.items():
        if len({x["id"] for x in es}) == 1:
            out[fx] = es[0]
        else:
            rep["ambiguous"].append(fx)
    rep["unmatched_fixtures"] = [fx for fx in fixtures if fx not in out]
    return out, rep


# ── response -> rows (real prices only) ──────────────────────────────────────────────────────────
ODDS_COLS = ["tier", "unit_key", "endpoint", "sport_key", "league", "league_date", "snapshot_label",
             "requested_at", "snapshot_ts", "event_id", "commence_time", "home_team", "away_team",
             "bookmaker", "book_last_update", "market", "market_last_update", "outcome_name",
             "outcome_description", "point", "price"]
EVENT_COLS = ["unit_key", "sport_key", "league", "league_date", "lookup_at", "snapshot_ts", "event_id",
              "commence_time", "home_team", "away_team"]


def _real_price(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v > 1.0 else None


def _real_point(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def odds_rows(events: list[dict], meta: dict) -> tuple[list[dict], int]:
    """One row per (event, book, market, outcome) whose price is a real decimal price; an outcome
    without one is NOT a row (counted in the second return value), never filled."""
    rows, dropped = [], 0
    for e in events:
        for b in e.get("bookmakers") or []:
            for m in b.get("markets") or []:
                for o in m.get("outcomes") or []:
                    p = _real_price(o.get("price"))
                    if p is None:
                        dropped += 1
                        continue
                    rows.append({**meta, "event_id": e.get("id"), "commence_time": e.get("commence_time"),
                                 "home_team": e.get("home_team"), "away_team": e.get("away_team"),
                                 "bookmaker": b.get("key"), "book_last_update": b.get("last_update"),
                                 "market": m.get("key"), "market_last_update": m.get("last_update"),
                                 "outcome_name": o.get("name"), "outcome_description": o.get("description"),
                                 "point": _real_point(o.get("point")), "price": p})
    return rows, dropped


# ── the store (a checkout of the PRIVATE branch) ──────────────────────────────────────────────────
class Store:
    """data/odds/*.parquet, data/events/*.parquet (shards, zstd, far under 100 MB each), ledger/units/
    <run>.csv (one line per finished request), ledger/runs/<run>.json (credits), ledger/match/<run>.csv
    (unmatched fixtures / events). A unit is DONE when a ledger line says ok / empty / http_4xx."""
    TERMINAL = {"ok", "empty", "http_404", "http_422", "before_history"}
    SHARD_ROWS = 250_000

    def __init__(self, root: Path, run_id: str):
        self.root, self.run_id = Path(root), run_id
        self.buf = {"odds": [], "events": []}
        self.ledger_buf, self.match_buf, self.seq = [], [], 0
        self.done = self._load_done()
        self.lookups = self._load_lookups()

    def _load_done(self) -> set:
        done = set()
        for f in sorted((self.root / "ledger" / "units").glob("*.csv")):
            with f.open(encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("status") in self.TERMINAL:
                        done.add(r["unit_key"])
        return done

    def _load_lookups(self) -> dict:
        out: dict = {}
        files = sorted((self.root / "data" / "events").glob("*.parquet"))
        if files:
            ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            for k, g in ev.groupby("unit_key"):
                out[k] = g.drop_duplicates("event_id").to_dict("records")
        return out

    def add(self, kind: str, rows: list[dict]):
        self.buf[kind].extend(rows)
        if len(self.buf[kind]) >= self.SHARD_ROWS:
            self.flush()

    def finish_unit(self, unit_key: str, tier: str, status: str, credits: int | None, rows: int,
                    snapshot_ts: str | None):
        self.ledger_buf.append({"unit_key": unit_key, "tier": tier, "status": status,
                                "credits": "" if credits is None else credits, "rows": rows,
                                "snapshot_ts": snapshot_ts or "",
                                "finished_at": iso(datetime.now(timezone.utc))})
        if status in self.TERMINAL:
            self.done.add(unit_key)

    def flush(self):
        """Shards FIRST, then the ledger lines that declare them done (a crash in between re-buys a
        unit, it never marks an unstored unit done)."""
        for kind, cols in (("odds", ODDS_COLS), ("events", EVENT_COLS)):
            if self.buf[kind]:
                d = self.root / "data" / kind
                d.mkdir(parents=True, exist_ok=True)
                self.seq += 1
                pd.DataFrame(self.buf[kind], columns=cols).to_parquet(
                    d / f"{self.run_id}-{self.seq:05d}.parquet", compression="zstd", index=False)
                self.buf[kind] = []
        if self.ledger_buf:
            f = self.root / "ledger" / "units" / f"{self.run_id}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            new = not f.exists()
            with f.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.ledger_buf[0]))
                if new:
                    w.writeheader()
                w.writerows(self.ledger_buf)
            self.ledger_buf = []
        if self.match_buf:
            f = self.root / "ledger" / "match" / f"{self.run_id}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            new = not f.exists()
            with f.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["league", "league_date", "kind", "home", "away"])
                if new:
                    w.writeheader()
                w.writerows(self.match_buf)
            self.match_buf = []

    def write_run(self, info: dict):
        f = self.root / "ledger" / "runs" / f"{self.run_id}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ── metered HTTP ─────────────────────────────────────────────────────────────────────────────────
class Stop(Exception):
    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


class Meter:
    """Every paid call passes here. Before: floor, run cap, time. After: x-requests-last and
    x-requests-remaining must both be present, and the cost may not exceed the call's maximum."""

    def __init__(self, key: str, floor: int, run_cap: int, max_seconds: float, session=None,
                 pause: float = 0.25, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if floor < MIN_FLOOR:
            raise ValueError(f"floor {floor} is below the owner's minimum {MIN_FLOOR}")
        if run_cap <= 0:
            raise ValueError("the per-run credit cap must be positive")
        import requests
        self.key, self.floor, self.run_cap = key, floor, run_cap
        self.s = session or requests.Session()
        self.pause, self.clock, self.sleep = pause, clock, sleep
        self.t0, self.max_seconds = clock(), max_seconds
        self.spent, self.calls, self.remaining = 0, 0, None
        self.remaining_start = None
        self.by_tier: Counter = Counter()

    def _get(self, path: str, params: dict):
        try:
            return self.s.get(f"{API}{path}", params={"apiKey": self.key, **params}, timeout=60)
        except Exception as e:                                      # noqa: BLE001
            # the exception text can hold the URL (and so the key): only its type is ever reported
            raise Stop(EXIT_ERROR, f"network error {type(e).__name__}") from None

    def read_remaining(self):
        """GET /sports is free: it reads the credits left before the first paid call."""
        r = self._get("/sports", {})
        rem = r.headers.get("x-requests-remaining")
        if r.status_code != 200 or rem is None:
            raise Stop(EXIT_ERROR, f"credits remaining unknown (status {r.status_code})")
        self.remaining = self.remaining_start = int(float(rem))
        return self.remaining

    def check(self, max_cost: int):
        if self.remaining is None:
            raise Stop(EXIT_ERROR, "credits remaining unknown")
        if self.clock() - self.t0 >= self.max_seconds:
            raise Stop(EXIT_TIME, "time budget reached")
        if self.remaining - max_cost < self.floor:
            raise Stop(EXIT_FLOOR, "credit floor reached")
        if self.spent + max_cost > self.run_cap:
            raise Stop(EXIT_RUN_CAP, "per-run credit cap reached")

    def call(self, path: str, params: dict, max_cost: int, tier: str):
        """(status, body | None, credits). Retries 429 / 5xx with back-off (each attempt metered)."""
        waits = (2, 5, 15, 30, 60)
        for attempt in range(len(waits) + 1):
            self.check(max_cost)
            r = self._get(path, params)
            self.calls += 1
            last, rem = r.headers.get("x-requests-last"), r.headers.get("x-requests-remaining")
            if last is None or rem is None:
                raise Stop(EXIT_ERROR, f"cost not reported (status {r.status_code})")
            last, self.remaining = int(float(last)), int(float(rem))
            self.spent += last
            self.by_tier[tier] += last
            if last > max_cost:
                raise Stop(EXIT_ERROR, f"a call cost {last} > its planned maximum {max_cost}")
            if self.pause:
                self.sleep(self.pause)
            if r.status_code == 200:
                return 200, r.json(), last
            if r.status_code in (401, 403):
                raise Stop(EXIT_ERROR, f"auth refused (status {r.status_code})")
            if r.status_code in (404, 422):
                return r.status_code, None, last
            if attempt < len(waits) and (r.status_code == 429 or r.status_code >= 500):
                self.sleep(waits[attempt])
                continue
            return r.status_code, None, last
        return None, None, 0


# ── the run ──────────────────────────────────────────────────────────────────────────────────────
def unit_key(endpoint: str, sport: str, ref: str, snapshot: str, markets: tuple, regions: tuple) -> str:
    return "|".join([endpoint, sport, ref, snapshot, ",".join(sorted(markets)), ",".join(sorted(regions))])


def lookup_at(league_date: str) -> datetime:
    return day_window(league_date)[0] - LOOKUP_LEAD


def lookup_key(sport: str, league_date: str) -> str:
    lo, hi = day_window(league_date)
    return unit_key("events", sport, f"{iso(lo)}/{iso(hi)}", iso(lookup_at(league_date)), (), ())


def say(msg: str):
    """The ONLY output of a run: counts and credits (Actions logs must never carry data)."""
    print(f"[oddsapi-history] {msg}", flush=True)


class Collector:
    def __init__(self, store: Store, meter: Meter, days: dict, plan: dict, tiers: Iterable[str] = TIER_ORDER):
        self.store, self.m, self.days, self.plan = store, meter, days, plan
        self.tiers = [t for t in TIER_ORDER if t in set(tiers)]
        self.stats: dict = {}
        self._matched: dict = {}

    def _st(self, tier, k, n=1):
        self.stats.setdefault(tier, Counter())[k] += n

    def events_for(self, league: str, d: str, tier: str) -> dict | None:
        """{(home, away): provider event} for one league-day (the lookup bought once, then reused)."""
        if (league, d) in self._matched:
            return self._matched[(league, d)]
        sk = sport_key(league)
        k = lookup_key(sk, d)
        if k in self.store.done:
            evs = self.store.lookups.get(k, [])
        else:
            lo, hi = day_window(d)
            at = lookup_at(d)
            status, body, c = self.m.call(f"/historical/sports/{sk}/events",
                                          {"date": iso(at), "commenceTimeFrom": iso(lo), "commenceTimeTo": iso(hi)},
                                          LOOKUP_MAX_COST, tier)
            self._st(tier, "lookup_calls")
            self._st(tier, "credits", c)
            data = (body or {}).get("data") or [] if status == 200 else []
            rows = [{"unit_key": k, "sport_key": sk, "league": league, "league_date": d, "lookup_at": iso(at),
                     "snapshot_ts": (body or {}).get("timestamp"), "event_id": e.get("id"),
                     "commence_time": e.get("commence_time"), "home_team": e.get("home_team"),
                     "away_team": e.get("away_team")} for e in data if e.get("id")]
            self.store.add("events", rows)
            st = "ok" if rows else ("empty" if status == 200 else f"http_{status}")
            self.store.finish_unit(k, tier, st, c, len(rows), (body or {}).get("timestamp"))
            self.store.lookups[k] = rows
            evs = rows
        fx = self.days.get((league, d), [])
        matched, rep = match_events(fx, [{"id": e["event_id"], **e} for e in evs], d)
        for h, a in rep["unmatched_fixtures"]:
            self.store.match_buf.append({"league": league, "league_date": d, "kind": "unmatched_fixture",
                                         "home": h, "away": a})
        for h, a in rep["unmatched_events"]:
            self.store.match_buf.append({"league": league, "league_date": d, "kind": "unmatched_event",
                                         "home": h, "away": a})
        for h, a in rep["ambiguous"]:
            self.store.match_buf.append({"league": league, "league_date": d, "kind": "ambiguous",
                                         "home": h, "away": a})
        self._st(tier, "fixtures_unmatched", len(rep["unmatched_fixtures"]))
        self._matched[(league, d)] = matched
        return matched

    def do_unit(self, u: Unit):
        matched = self.events_for(u.league, u.date, u.tier)
        sk = sport_key(u.league)
        if u.kind == "event":
            e = matched.get((u.home, u.away))
            if e is None:
                self._st(u.tier, "units_no_provider_event")
                return
            snap = parse_iso(e["commence_time"]) + timedelta(seconds=u.offset_s)
            if snap < EVENT_HISTORY_START:
                self._st(u.tier, "units_before_history")
                return
            k = unit_key("event_odds", sk, e["event_id"], iso(snap), u.markets, u.regions)
            path = f"/historical/sports/{sk}/events/{e['event_id']}/odds"
        else:
            if not matched:
                self._st(u.tier, "units_no_provider_event")
                return
            first = min(parse_iso(e["commence_time"]) for e in matched.values())
            snap = first + timedelta(seconds=u.offset_s)
            if snap < FEATURED_HISTORY_START:
                self._st(u.tier, "units_before_history")
                return
            k = unit_key("odds", sk, "-", iso(snap), u.markets, u.regions)
            path = f"/historical/sports/{sk}/odds"
        if k in self.store.done:
            self._st(u.tier, "units_already_stored")
            return
        status, body, c = self.m.call(path, {"date": iso(snap), "regions": ",".join(u.regions),
                                             "markets": ",".join(u.markets), "oddsFormat": "decimal",
                                             "dateFormat": "iso"}, u.max_cost, u.tier)
        self._st(u.tier, "odds_calls")
        self._st(u.tier, "credits", c)
        if status != 200:
            self.store.finish_unit(k, u.tier, f"http_{status}", c, 0, None)
            self._st(u.tier, f"http_{status}")
            return
        data = (body or {}).get("data")
        evs = [data] if isinstance(data, dict) else list(data or [])
        if u.kind == "featured":
            lo, hi = day_window(u.date)
            evs = [e for e in evs if e.get("commence_time") and lo <= parse_iso(e["commence_time"]) < hi]
        meta = {"tier": u.tier, "unit_key": k, "endpoint": "event_odds" if u.kind == "event" else "odds",
                "sport_key": sk, "league": u.league, "league_date": u.date, "snapshot_label": u.label,
                "requested_at": iso(snap), "snapshot_ts": (body or {}).get("timestamp")}
        rows, dropped = odds_rows(evs, meta)
        self.store.add("odds", rows)
        self.store.finish_unit(k, u.tier, "ok" if rows else "empty", c, len(rows), meta["snapshot_ts"])
        self._st(u.tier, "units_ok" if rows else "units_empty")
        self._st(u.tier, "rows", len(rows))
        self._st(u.tier, "outcomes_without_price", dropped)

    def run(self, flush_every: int = 200) -> tuple[int, str]:
        code, reason, n = EXIT_DONE, "all planned tiers done", 0
        try:
            self.m.read_remaining()
            say(f"credits remaining at start {self.m.remaining_start:,}; floor {self.m.floor:,}; "
                f"run cap {self.m.run_cap:,}")
            for t in self.tiers:
                units = self.plan["units"].get(t, [])
                say(f"tier {t}: {len(units):,} planned units")
                for u in units:
                    self.do_unit(u)
                    n += 1
                    if n % flush_every == 0:
                        self.store.flush()
                        self._progress()
                self.store.flush()
                self._progress()
        except Stop as s:
            code, reason = s.code, s.reason
        finally:
            self.store.flush()
            info = {"run_id": self.store.run_id, "exit_code": code, "stop_reason": reason,
                    "calls": self.m.calls, "credits_spent": self.m.spent,
                    "credits_by_tier": self.m.by_tier, "remaining_start": self.m.remaining_start,
                    "remaining_end": self.m.remaining, "floor": self.m.floor, "run_cap": self.m.run_cap,
                    "stats": self.stats, "finished_at": iso(datetime.now(timezone.utc))}
            self.store.write_run(info)
            say(f"stop: {reason}; calls {self.m.calls:,}; credits spent {self.m.spent:,}; "
                f"remaining {self.m.remaining if self.m.remaining is not None else 'unknown'}")
        return code, reason

    def _progress(self):
        for t, s in self.stats.items():
            say(f"tier {t}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(s.items())))


# ── the S probe (correct_score; one event per league, the P7 probe's own events) ──────────────────
def probe_s(meter: Meter, store_root: Path, probe_file: Path = PROBE_FILE,
            regions: tuple = ("uk", "eu")) -> dict:
    """For every league the P7 probe found an event for: correct_score at kickoff-1h on THAT event.
    At most 10 x 1 x len(regions) credits per league, free when empty. Writes ledger/probe_s.json
    into the PRIVATE store (market presence, books, outcome counts; no price)."""
    p = json.loads(Path(probe_file).read_text(encoding="utf-8"))
    meter.read_remaining()
    res = {"probed_at": iso(datetime.now(timezone.utc)), "regions": list(regions), "leagues": {}}
    try:
        for lg, r in sorted((p.get("leagues") or {}).items()):
            ev = r.get("event")
            if not isinstance(ev, dict) or not ev.get("id"):
                res["leagues"][lg] = {"skipped": "no P7 probe event"}
                continue
            at = parse_iso(ev["kickoff"]) - timedelta(hours=1)
            status, body, c = meter.call(f"/historical/sports/{r['sport_key']}/events/{ev['id']}/odds",
                                         {"date": iso(at), "regions": ",".join(regions),
                                          "markets": "correct_score", "oddsFormat": "decimal"},
                                         CREDITS_PER_MARKET_REGION * len(regions), "S-probe")
            d = (body or {}).get("data") or {} if status == 200 else {}
            books = [b for b in d.get("bookmakers") or []
                     if any(m.get("key") == "correct_score" for m in b.get("markets") or [])]
            res["leagues"][lg] = {"status": status, "credits": c, "correct_score_returned": bool(books),
                                  "books": len(books),
                                  "outcomes": sum(len(m.get("outcomes") or []) for b in books
                                                  for m in b.get("markets") or [] if m.get("key") == "correct_score")}
    except Stop as s:
        res["stopped"] = s.reason
    res["credits_spent"], res["calls"] = meter.spent, meter.calls
    f = Path(store_root) / "ledger" / S_PROBE_NAME
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    say(f"S probe: {sum(1 for v in res['leagues'].values() if v.get('correct_score_returned'))} of "
        f"{len(res['leagues'])} leagues returned correct_score; credits {meter.spent:,}")
    return res


def plan_digest(plan: dict) -> str:
    h = hashlib.sha256()
    for t, units in plan["units"].items():
        h.update(f"{t}:{len(units)}:{sum(u.max_cost for u in units)}:{len(plan['lookups'][t])};".encode())
    return h.hexdigest()[:16]
