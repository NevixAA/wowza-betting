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
    # per model track ((track, ((label, timedelta), ...)), ...): replaces `offsets` for a league of that
    # track (the registry's model_type); a league of no listed track gets no unit
    offsets_by_track: tuple | None = None

    def offsets_for(self, league: str) -> tuple:
        if self.offsets_by_track is None:
            return self.offsets
        return dict(self.offsets_by_track).get(league_model_type(league), ())


# BT, the BET-TIME snapshot (owner 2026-11-11: "a closing price may never be a bet price"). The lead
# time live tips are really made at, MEASURED on v9's live ledgers (data-truth-100 91cc1e805,
# output/bets_ledger.csv + output/side_bets_ledger.csv, source == live, 2026-04-26..2026-11-09): first
# sight = the earliest generated_at (UTC, src/ledger.py datetime.utcnow) per fixture x market, kickoff =
# the provider's commence_time of that fixture (exact OddsAPI names, +-1 day; 96-98% joined).
#   new_format: n 1,663, median 152.8 h (IQR 98.4-166.3)    standard: n 677, median 124.9 h (IQR 66.1-164.5)
# The tracks differ by 28 h, so each gets its own median, rounded to the hour.
BT_LEAD_HOURS = {"new_format": 153, "standard": 125}
BT_LABEL = "bet"


TIERS: tuple[Tier, ...] = (
    Tier("B1", "event", ("btts", "alternate_totals"), ("eu",), (CLOSE,), "team", "2023-05-03", HISTORY_END,
         purpose="BTTS + every alternate total line (O0.5..O6.5+) at close: bars evidence + specials"),
    Tier("B2", "featured", ("totals",), ("eu",), (CLOSE,), "new_format", "2020-06-06", "2023-05-02",
         purpose="New-Format O/U main line before the event-market history starts"),
    Tier("B4", "featured", ("h2h", "totals"), ("eu",), EIGHT, "team", "2023-06-14", "2026-06-13",
         purpose="sharp-money line movement, 8 snapshots per league-day"),
    Tier("BT", "event", ("btts", "alternate_totals"), ("eu",), (), "team", "2023-05-03", HISTORY_END,
         purpose="BET-TIME snapshot: B1's events and markets at the measured live-tip lead time per track",
         offsets_by_track=tuple((tr, ((BT_LABEL, timedelta(hours=-h)),)) for tr, h in BT_LEAD_HOURS.items())),
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
def load_served_history() -> tuple[pd.DataFrame, dict]:
    """v9's SERVED history, exactly as its clean retrain reads it: src.data_loader.load_all_matches
    (force=True) with the football-data download switched off and every socket refused while it runs
    (committed rows only, every loader correction applied: history corrections, provider dates, ONE
    club ONE spelling ONE fixture through src.identity_repair - Altach/SCR Altach, SK Rapid/Rapid
    Vienna, the China renames). Returns ([league, date, home, away], {league: {spelling: canonical}}
    = data_loader.IDENTITY_ALIASES as applied). The loader's own log lines are silenced: they name
    clubs, and this runs in a workflow whose log carries counts only."""
    import contextlib
    import io
    import logging
    import socket
    from src import data_loader as DL
    real = (socket.socket.connect, socket.create_connection)

    def _no_net(*a, **k):
        raise OSError("network blocked (oddsapi_history reads committed history only)")
    socket.socket.connect, socket.create_connection = _no_net, _no_net
    dl_fetch, dl_cov = DL._ci_download_all, DL._report_training_coverage
    DL._ci_download_all = lambda have=None: []
    DL._report_training_coverage = lambda out: None
    prev = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = DL.load_all_matches(force=True)
        identity = {lg: dict(m) for lg, m in DL.IDENTITY_ALIASES.items()}
    finally:
        socket.socket.connect, socket.create_connection = real
        DL._ci_download_all, DL._report_training_coverage = dl_fetch, dl_cov
        logging.disable(prev)
    h = raw[["league", "date", "home_team", "away_team"]].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h = h.dropna().rename(columns={"home_team": "home", "away_team": "away"})
    return h.drop_duplicates().reset_index(drop=True), identity


def load_fixtures(root: Path = ROOT, served: tuple | None = None, include: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """([league, date, home, away, source], identity aliases). Team leagues come from v9's served
    history (canonical clubs, one fixture each); keyed leagues the served history does not hold (the
    props-only top divisions: Premier League, Ligue 1, ...) come from player_history. Keyed leagues
    only, plus `include` (the keyless leagues, for --probe-keys only: never planned)."""
    h, identity = served if served is not None else load_served_history()
    parts = [h.assign(source="v9")]
    ph = root / "player_history.parquet"
    if ph.exists():
        import pyarrow.dataset as ds
        p = ds.dataset(str(ph), format="parquet").to_table(
            columns=["league", "date", "home_team", "away_team"]).to_pandas().drop_duplicates()
        p = p[~p["league"].isin(set(h["league"]))]
        p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        parts.append(p.rename(columns={"home_team": "home", "away_team": "away"}).assign(source="ph"))
    f = pd.concat(parts, ignore_index=True).dropna(subset=["league", "date", "home", "away"])
    keyed = {lg for lg in f["league"].unique() if sport_key(lg)} | set(include)
    f = f[f["league"].isin(keyed)]
    return (f.drop_duplicates(["league", "date", "home", "away", "source"])
             .reset_index(drop=True)[["league", "date", "home", "away", "source"]], identity)


def club_spellings(identity: dict, days: dict) -> dict:
    """{league: {canonical club: {every spelling v9 maps onto it}}} over the clubs of `days`."""
    out: dict = {}
    for (lg, _), fx in days.items():
        m = out.setdefault(lg, {})
        for h, a in fx:
            m.setdefault(h, {h})
            m.setdefault(a, {a})
    for lg, amap in identity.items():
        for alias, canon in amap.items():
            if canon in out.get(lg, {}):
                out[lg][canon].add(alias)
    return out


_SOURCE_RANK = {"v9": 0, "af": 1, "fd": 2, "ph": 3}


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


def league_model_type(league: str) -> str | None:
    """The registry's model_type of the league ('standard' / 'new_format'), None for props-only."""
    from src import league_registry as LR
    return (LR.entry(league) or {}).get("model_type")


def snapshot_role(tier: str, label: str) -> str:
    """The role of a stored snapshot. BET: the bet-time snapshot (tier BT) - the ONLY price a bet may
    be evaluated at. CLOSE: a closing snapshot (label 'close': B1, B2, B3, B4, C3, C4) - CLV only, never a
    bet price. OPEN: the T-48h opening snapshots of C1, C2, C3. MOVEMENT: B4 / C5's intermediate line
    snapshots (T-72h..T-1h)."""
    if tier == "BT":
        return "BET"
    if label == "close":
        return "CLOSE"
    if tier in ("C1", "C2", "C3") and label == OPEN48[0]:
        return "OPEN"
    return "MOVEMENT"


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
                for label, off in t.offsets_for(lg):
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
            "snapshots": ([lab for lab, _ in t.offsets] if t.offsets_by_track is None else
                          {tr: [f"{lab} {int(off.total_seconds() // 3600)}h" for lab, off in offs]
                           for tr, offs in t.offsets_by_track}), "window": [t.start, t.end], "gate": t.gate,
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


# ── provider-event matching (one league; the match day, or +-1 day with both clubs proven) ────────
MIN_ALIAS_DATES = 3   # the identity_repair / prove_af_history_aliases evidence rule, not a new number


class Resolver:
    """OddsAPI spelling -> ONE of our canonical clubs among `clubs` (one league, one side, one day).
    A club is hit when the league's PROVEN provider alias names it, or src.team_names.resolve proves
    the name is one of v9's spellings of that club (canonical + every identity alias v9 applied).
    Exactly one club hit -> that club; none or several -> None. Nothing fuzzy."""

    def __init__(self, spell: dict, proven: dict | None = None):
        self.spell, self.proven = spell, proven or {}
        self._memo: dict = {}

    def club(self, league: str, name, clubs) -> str | None:
        from src.team_names import resolve
        name = "" if name is None else str(name)
        clubs = tuple(sorted(set(clubs)))
        k = (league, name, clubs)
        if k in self._memo:
            return self._memo[k]
        p = self.proven.get(league, {}).get(name)
        if p is not None:
            hit = p if p in clubs else None
        else:
            sp = self.spell.get(league, {})
            hits = [c for c in clubs if any(resolve(name, [x]) == x for x in sorted(sp.get(c, {c})))]
            hit = hits[0] if len(hits) == 1 else None
        self._memo[k] = hit
        return hit


def _in_window(e: dict, league_date: str) -> bool:
    lo, hi = day_window(league_date)
    try:
        return lo <= parse_iso(e["commence_time"]) < hi
    except Exception:                                               # noqa: BLE001
        return False


def _adjacent(d: str) -> tuple[str, str]:
    x = datetime.strptime(d, "%Y-%m-%d")
    return ((x - timedelta(days=1)).strftime("%Y-%m-%d"), (x + timedelta(days=1)).strftime("%Y-%m-%d"))


def match_day(league: str, d: str, fixtures: list[tuple], events_by_day: dict, R: Resolver) -> tuple[dict, dict]:
    """({(home, away): event}, report) for one league-day. First the provider events of THIS day's
    window; a fixture still unmatched may then take an event from the stored listings of the day
    before / after (a UTC vs local date), only with BOTH clubs proven. One-to-one: a fixture with two
    event ids, or an event claimed by two fixtures, is ambiguous and bought for neither."""
    rep = {"events_in_window": 0, "unmatched_events": [], "ambiguous": [], "unmatched_fixtures": [],
           "matched_adjacent_day": 0}
    if not fixtures:
        return {}, rep
    homes, aways = [h for h, _ in fixtures], [a for _, a in fixtures]
    prim = [e for e in events_by_day.get((league, d), []) if _in_window(e, d)]
    rep["events_in_window"] = len({e["id"] for e in prim})
    seen = {e["id"] for e in prim}
    adj = []
    for dd in _adjacent(d):
        for e in events_by_day.get((league, dd), []):
            if e["id"] not in seen:
                seen.add(e["id"])
                adj.append(e)
    cand: dict = {}

    def scan(evs, only=None, tag="day"):
        for e in evs:
            rh, ra = R.club(league, e.get("home_team"), homes), R.club(league, e.get("away_team"), aways)
            if rh and ra and (rh, ra) in fixtures and (only is None or (rh, ra) in only):
                cand.setdefault((rh, ra), {})[e["id"]] = (e, tag)
    scan(prim)
    left = {f for f in fixtures if f not in cand}
    if left and adj:
        scan(adj, only=left, tag="adjacent")
    # A RESCHEDULED match: the listing still holds the stale event id beside the new one, both clubs
    # proven on both. The match was played on OUR date (the results provider's), so the one event
    # whose UTC kickoff date is that date is it; none or several such -> ambiguous, bought for neither.
    rep["rescheduled_picked"] = 0
    for f, es in list(cand.items()):
        if len(es) > 1:
            on_day = {k: v for k, v in es.items() if str(v[0].get("commence_time", ""))[:10] == d}
            if len(on_day) == 1:
                cand[f] = on_day
                rep["rescheduled_picked"] += 1
    claims: dict = {}
    for f, es in cand.items():
        for eid in es:
            claims.setdefault(eid, []).append(f)
    out = {}
    for f, es in cand.items():
        if len(es) != 1 or len(claims[next(iter(es))]) != 1:
            rep["ambiguous"].append(f)
            continue
        e, tag = next(iter(es.values()))
        out[f] = e
        rep["matched_adjacent_day"] += tag == "adjacent"
    used = {e["id"] for e in out.values()}
    rep["unmatched_events"] = [(e.get("home_team"), e.get("away_team")) for e in prim if e["id"] not in used]
    rep["unmatched_fixtures"] = [f for f in fixtures if f not in out]
    return out, rep


def match_events(fixtures: list[tuple], events: list[dict], league_date: str, league: str = "_",
                 R: Resolver | None = None) -> tuple[dict, dict]:
    """One league-day against one plain event list."""
    return match_day(league, league_date, fixtures, {(league, league_date): events}, R or Resolver({}))


def prove_provider_aliases(days: dict, events_by_day: dict, spell: dict, min_dates: int = MIN_ALIAS_DATES,
                           max_rounds: int = 6) -> dict:
    """League-scoped OddsAPI spelling -> our canonical club, PROVEN by forced-opponent evidence on the
    provider's own listings (the prove_af_history_aliases rule; listings carry no score):
      * on league-day D an event's one side is proven to be our club X, X plays exactly one fixture
        that day on that side, X v Y, Y is not matched by any other event that day, and the event's
        other name N is not proven to be anyone -> evidence (N, Y, D);
      * (N -> Y) is PROVEN when it has evidence on >= min_dates distinct days and N never has
        evidence for another club; two names claiming Y that appear in the SAME listing are refused;
      * rounds iterate to a fixpoint: a pair proven in round k anchors the forced opponent in k+1.
    Returns {"map": {league: {N: Y}}, "evidence": {...}, "refused": {...}, "rounds": n}."""
    known: dict = {}
    res: dict = {"map": {}, "evidence": {}, "refused": {}, "rounds": 0}
    for rnd in range(1, max_rounds + 1):
        R = Resolver(spell, known)
        ev: dict = {}
        cooccur: dict = {}
        for (lg, d), fx in days.items():
            evs = [e for e in events_by_day.get((lg, d), []) if _in_window(e, d)]
            if not evs or not fx:
                continue
            homes, aways = [h for h, _ in fx], [a for _, a in fx]
            cooccur.setdefault(lg, []).append({str(e.get("home_team")) for e in evs}
                                              | {str(e.get("away_team")) for e in evs})
            res_e = [(e, R.club(lg, e.get("home_team"), homes), R.club(lg, e.get("away_team"), aways))
                     for e in evs]
            matched_clubs = {c for _, rh, ra in res_e if rh and ra for c in (rh, ra)}
            for e, rh, ra in res_e:
                if rh and ra:
                    continue
                for anc, other_name, side in ((rh, e.get("away_team"), 0), (ra, e.get("home_team"), 1)):
                    if not anc:
                        continue
                    f = [x for x in fx if x[side] == anc]
                    if len(f) != 1:
                        continue
                    y = f[0][1 - side]
                    if y in matched_clubs:
                        continue
                    ev.setdefault(lg, {}).setdefault(str(other_name), {}).setdefault(y, set()).add(d)
        # a pair proven in an earlier round stays proven: its names now resolve, so they produce no
        # new evidence (re-deriving them from scratch would make the rounds oscillate)
        new_map: dict = {lg: dict(m) for lg, m in known.items()}
        evidence: dict = {lg: dict(m) for lg, m in res.get("evidence", {}).items()}
        refused = Counter()
        for lg, by_name in ev.items():
            cands = {}
            for n, ys in by_name.items():
                if n in new_map.get(lg, {}):
                    continue
                if len(ys) > 1:
                    refused["name_points_at_two_clubs"] += 1
                    continue
                (y, ds), = ys.items()
                if len(ds) < min_dates:
                    refused["fewer_than_min_dates"] += 1
                    continue
                cands[n] = (y, ds)
            by_club: dict = {}
            for n, (y, _) in cands.items():
                by_club.setdefault(y, []).append(n)
            for y, ns in by_club.items():
                ns = ns + [n for n, c in new_map.get(lg, {}).items() if c == y and n not in ns]
                if len(ns) > 1 and any(sum(n in names for n in ns) > 1 for names in cooccur.get(lg, [])):
                    refused["club_claimed_by_two_names_together"] += len(ns)
                    continue
                for n in ns:
                    if n not in cands:
                        continue
                    ds = cands[n][1]
                    new_map.setdefault(lg, {})[n] = y
                    evidence.setdefault(lg, {})[n] = {"club": y, "dates": len(ds), "first": min(ds),
                                                      "last": max(ds)}
        res = {"map": new_map, "evidence": evidence, "refused": dict(refused), "rounds": rnd}
        if new_map == known:
            break
        known = new_map
    return res


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
        self.events_by_day: dict = {}
        files = sorted((self.root / "data" / "events").glob("*.parquet"))
        if files:
            ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            for k, g in ev.groupby("unit_key"):
                rows = g.drop_duplicates("event_id").to_dict("records")
                out[k] = rows
                self._index(rows[0]["league"], rows[0]["league_date"], rows)
        return out

    def _index(self, league: str, d: str, rows: list[dict]):
        """events_by_day[(league, date)]: the listing's events with `id` (what the matcher reads)."""
        have = self.events_by_day.setdefault((league, d), [])
        ids = {e["id"] for e in have}
        have += [{**r, "id": r["event_id"]} for r in rows if r["event_id"] not in ids]

    def put_lookup(self, key: str, league: str, d: str, rows: list[dict]):
        self.lookups[key] = rows
        self._index(league, d, rows)

    def write_identity(self, identity: dict):
        """ledger/identity_aliases.json: v9's applied identity map this run matched with (the
        adapter reads the same spellings)."""
        f = self.root / "ledger" / "identity_aliases.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(identity, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_aliases(self, proof: dict):
        """ledger/provider_aliases.json: the proven OddsAPI spellings (private: derived from listings)."""
        f = self.root / "ledger" / "provider_aliases.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(proof, indent=1, sort_keys=True, default=sorted) + "\n", encoding="utf-8")

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


RETRY_WAITS = (5, 20, 60)        # owner 2026-11-10: at most 3 retries per request, exponential back-off
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_FAILED_PER_RUN = 20          # more requests than this failing after their retries stops the run


class Meter:
    """Every paid call passes here. Before: floor, run cap, time. After: x-requests-last and
    x-requests-remaining must both be present on a successful answer, and the cost may not exceed
    the call's maximum.

    Provider faults (HTTP 500/502/503/504, a timeout or a dropped connection) and 429 are retried at
    most len(RETRY_WAITS) times with back-off. Right after each failed attempt the free GET /sports is
    read (no other collector call in between): a cost the attempt did not report is the drop from the
    last known remaining value to that reading, booked to the request ('unreported_cost_reconciled');
    more than the request's maximum stops the run. After the back-off sleep /sports is read again:
    the shared key's live spend meanwhile is excluded (live_spend_during_backoff). A request still
    failing after its retries returns 'failed' (the caller leaves it NOT done); more than
    MAX_FAILED_PER_RUN of them stop the run. Other 4xx are never retried."""

    def __init__(self, key: str, floor: int, run_cap: int, max_seconds: float, session=None,
                 pause: float = 0.25, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, on_reconciled: Callable | None = None):
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
        self.failed_requests = 0
        self.retries = 0
        self.reconciled: list = []       # [{unit_key, tier, credits}] booked to failed requests
        self.live_spend_during_backoff = 0   # the shared key's spend while we slept: never ours
        self.on_reconciled = on_reconciled

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

    def sports_list(self) -> list:
        """GET /sports?all=true (in and out of season). Free per the v4 guide ("This endpoint does not
        count against the usage quota"); a response that reports any cost, or none, stops the run."""
        r = self._get("/sports", {"all": "true"})
        last, rem = r.headers.get("x-requests-last"), r.headers.get("x-requests-remaining")
        if r.status_code != 200 or rem is None or last is None:
            raise Stop(EXIT_ERROR, f"sports list unreadable (status {r.status_code})")
        if int(float(last)) != 0:
            raise Stop(EXIT_ERROR, f"the sports list was not free (x-requests-last {last})")
        self.remaining = int(float(rem))
        return r.json()

    def sports_remaining(self) -> int:
        """x-requests-remaining from GET /sports, which the provider docs say "does not count against
        the usage quota" (v4 guide). A missing header stops the run: nothing is ever estimated."""
        r = self._get("/sports", {})
        rem = r.headers.get("x-requests-remaining")
        if r.status_code != 200 or rem is None:
            raise Stop(EXIT_ERROR, f"credits remaining unknown from /sports (status {r.status_code})")
        return int(float(rem))

    def _settle_fault(self, key, tier: str, max_cost: int, reported: bool):
        """Right after a failed attempt, with no other collector call in between: read /sports. If
        the attempt reported no cost, the drop from the last known remaining value to that reading is
        its unexplained cost, booked to the request ('unreported_cost_reconciled'); more than the
        request's maximum stops the run. Returns the reading."""
        now = self.sports_remaining()
        if not reported:
            drop = self.remaining - now
            if drop > max_cost:
                raise Stop(EXIT_ERROR, f"unexplained credit drop {drop} > one request's maximum {max_cost}")
            if drop > 0:
                self.spent += drop
                self.by_tier[tier] += drop
                self.reconciled.append({"unit_key": key, "tier": tier, "credits": drop})
                if self.on_reconciled:
                    self.on_reconciled(key, tier, drop)
        self.remaining = now
        return now

    def _backoff(self, wait: float):
        """Sleep, then read /sports again: what the shared key (live predictions) spent meanwhile is
        NOT ours; it is recorded as live_spend_during_backoff and the baseline moves to the reading."""
        before = self.remaining
        self.sleep(wait)
        after = self.sports_remaining()
        if before - after > 0:
            self.live_spend_during_backoff += before - after
        self.remaining = after

    def call(self, path: str, params: dict, max_cost: int, tier: str, key: str | None = None):
        """(status, body | None, credits). status 'failed' = still a provider fault after the retries."""
        for attempt in range(len(RETRY_WAITS) + 1):
            self.check(max_cost)
            try:
                r = self._get(path, params)
            except Stop:
                r = None                                 # timeout / dropped connection: a provider fault
            self.calls += 1
            status = None if r is None else r.status_code
            last = None if r is None else r.headers.get("x-requests-last")
            rem = None if r is None else r.headers.get("x-requests-remaining")
            if r is None or status in RETRYABLE_STATUS:
                reported = last is not None and rem is not None
                if reported:
                    last = int(float(last))
                    self.spent += last
                    self.by_tier[tier] += last
                    self.remaining = int(float(rem))
                self._settle_fault(key or path, tier, max_cost, reported)
                if attempt < len(RETRY_WAITS):
                    self.retries += 1
                    self._backoff(RETRY_WAITS[attempt])
                    continue
                self.failed_requests += 1
                if self.failed_requests > MAX_FAILED_PER_RUN:
                    raise Stop(EXIT_ERROR, f"more than {MAX_FAILED_PER_RUN} requests failed after retries")
                return "failed", None, None
            if last is None or rem is None:
                raise Stop(EXIT_ERROR, f"cost not reported (status {status})")
            last = int(float(last))
            self.remaining = int(float(rem))
            self.spent += last
            self.by_tier[tier] += last
            if last > max_cost:
                raise Stop(EXIT_ERROR, f"a call cost {last} > its planned maximum {max_cost}")
            if self.pause:
                self.sleep(self.pause)
            if status == 200:
                return 200, r.json(), last
            if status in (401, 403):
                raise Stop(EXIT_ERROR, f"auth refused (status {status})")
            return status, None, last                    # other 4xx: never retried
        return "failed", None, None


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
    """Per tier, in the owner's priority order: (1) the provider's event listing of every league-day
    the tier needs (1 credit each, bought once and reused by every tier and every later run); (2) the
    league-scoped OddsAPI aliases PROVEN from every stored listing (prove_provider_aliases), written
    to ledger/provider_aliases.json; (3) the odds units, each bought only for a proven provider event.
    A unit whose fixture has no proven event is NOT written to the ledger, so a later run retries it
    (with the aliases the new listings prove) without re-buying the listing."""

    def __init__(self, store: Store, meter: Meter, days: dict, plan: dict, tiers: Iterable[str] = TIER_ORDER,
                 spell: dict | None = None):
        self.store, self.m, self.days, self.plan = store, meter, days, plan
        self.tiers = [t for t in TIER_ORDER if t in set(tiers)]
        self.spell = spell if spell is not None else club_spellings({}, days)
        self.stats: dict = {}
        self._matched: dict = {}
        self.R = Resolver(self.spell)
        self.proof: dict = {}
        if self.m.on_reconciled is None:
            # an unreported cost found later is booked to the failed request (never done: retryable)
            self.m.on_reconciled = lambda k, t, c: self.store.finish_unit(k, t, "unreported_cost_reconciled",
                                                                          c, 0, None)

    def _st(self, tier, k, n=1):
        self.stats.setdefault(tier, Counter())[k] += n

    def ensure_lookup(self, league: str, d: str, tier: str):
        sk = sport_key(league)
        k = lookup_key(sk, d)
        if k in self.store.done:
            return
        lo, hi = day_window(d)
        at = lookup_at(d)
        status, body, c = self.m.call(f"/historical/sports/{sk}/events",
                                      {"date": iso(at), "commenceTimeFrom": iso(lo), "commenceTimeTo": iso(hi)},
                                      LOOKUP_MAX_COST, tier, key=k)
        self._st(tier, "lookup_calls")
        if status == "failed":
            self._st(tier, "lookups_failed_after_retries")          # not done: retried on resume
            self.store.finish_unit(k, tier, "failed_after_retries", None, 0, None)
            return
        self._st(tier, "credits", c)
        data = ((body or {}).get("data") or []) if status == 200 else []
        rows = [{"unit_key": k, "sport_key": sk, "league": league, "league_date": d, "lookup_at": iso(at),
                 "snapshot_ts": (body or {}).get("timestamp"), "event_id": e.get("id"),
                 "commence_time": e.get("commence_time"), "home_team": e.get("home_team"),
                 "away_team": e.get("away_team")} for e in data if e.get("id")]
        self.store.add("events", rows)
        st = "ok" if rows else ("empty" if status == 200 else f"http_{status}")
        self.store.finish_unit(k, tier, st, c, len(rows), (body or {}).get("timestamp"))
        self.store.put_lookup(k, league, d, rows)

    def reprove(self):
        """Re-derive the proven aliases from EVERY stored listing; drop the cached day matches."""
        self.proof = prove_provider_aliases(self.days, self.store.events_by_day, self.spell)
        self.R = Resolver(self.spell, self.proof["map"])
        self._matched = {}
        self.store.write_aliases(self.proof)
        say(f"provider aliases proven {sum(len(v) for v in self.proof['map'].values()):,} "
            f"in {len(self.proof['map'])} league(s); refused {sum(self.proof['refused'].values()):,}")

    def events_for(self, league: str, d: str, tier: str) -> dict:
        """{(home, away): provider event} for one league-day."""
        if (league, d) in self._matched:
            return self._matched[(league, d)]
        self.ensure_lookup(league, d, tier)
        fx = self.days.get((league, d), [])
        matched, rep = match_day(league, d, fx, self.store.events_by_day, self.R)
        for kind, pairs in (("unmatched_fixture", rep["unmatched_fixtures"]),
                            ("unmatched_event", rep["unmatched_events"]), ("ambiguous", rep["ambiguous"])):
            for h, a in pairs:
                self.store.match_buf.append({"league": league, "league_date": d, "kind": kind,
                                             "home": h, "away": a})
        self._st(tier, "fixtures_matched", len(matched))
        self._st(tier, "fixtures_unmatched", len(rep["unmatched_fixtures"]))
        self._st(tier, "fixtures_matched_adjacent_day", rep["matched_adjacent_day"])
        self._matched[(league, d)] = matched
        return matched

    def do_unit(self, u: Unit):
        matched = self.events_for(u.league, u.date, u.tier)
        sk = sport_key(u.league)
        if u.kind == "event":
            e = matched.get((u.home, u.away))
            if e is None:
                self._st(u.tier, "units_no_provider_event")      # NOT in the ledger: retried next run
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
                                             "dateFormat": "iso"}, u.max_cost, u.tier, key=k)
        self._st(u.tier, "odds_calls")
        if status == "failed":
            self._st(u.tier, "units_failed_after_retries")          # not done: retried on resume
            self.store.finish_unit(k, u.tier, "failed_after_retries", None, 0, None)
            return
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
                days = list(dict.fromkeys((u.league, u.date) for u in units))
                say(f"tier {t}: {len(units):,} planned units over {len(days):,} league-days")
                for i, (lg, d) in enumerate(days, 1):         # (1) listings first
                    self.ensure_lookup(lg, d, t)
                    if i % flush_every == 0:
                        self.store.flush()
                self.store.flush()
                self.reprove()                                 # (2) aliases from every listing
                for u in units:                                # (3) odds
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
                    "retries": self.m.retries, "failed_requests": self.m.failed_requests,
                    "unreported_cost_reconciled": self.m.reconciled,
                    "live_spend_during_backoff": self.m.live_spend_during_backoff,
                    "stats": self.stats, "finished_at": iso(datetime.now(timezone.utc))}
            self.store.write_run(info)
            say(f"stop: {reason}; calls {self.m.calls:,}; credits spent {self.m.spent:,}; "
                f"remaining {self.m.remaining if self.m.remaining is not None else 'unknown'}; "
                f"retries {self.m.retries:,}; failed after retries {self.m.failed_requests:,}; "
                f"reconciled {sum(x['credits'] for x in self.m.reconciled):,}; "
                f"live spend during back-off {self.m.live_spend_during_backoff:,}")
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


# ── probe-keys: keyless leagues against the provider's own sports list (owner 2026-11-11) ──────────
# How each league WITHOUT a registry key can be recognised in the provider's soccer list: a country
# token AND a competition token in the key / title / description. These only select CANDIDATES from
# the provider's own list; a league with no candidate or with two stays keyless (never guessed), and a
# selected key must still return a real historical event before it may enter the registry.
KEYLESS_SEARCH = {
    "Portuguese Primeira Liga": (("portugal", "portuguese"), ("primeira",)),
    "National League": (("england", "english"), ("national league",)),
    "Scottish Championship": (("scotland", "scottish"), ("championship",)),
    "Scottish League One": (("scotland", "scottish"), ("league one", "league 1")),
    "Scottish League Two": (("scotland", "scottish"), ("league two", "league 2")),
    "Romanian Superliga": (("romania", "romanian"), ("liga 1", "liga i", "superliga")),
}
PROBE_KEYS_NAME = "probe_keys.json"
PROBE_KEYS_MARKETS = ("btts", "alternate_totals")
PROBE_KEYS_WINDOW = ("2024-03-01", "2025-03-31")      # the P7 probe's window


def find_candidate_keys(sports: list, league: str) -> list:
    """The provider's soccer (non-outright) sports whose key / title / description carry the league's
    country token and competition token (KEYLESS_SEARCH)."""
    country, comp = KEYLESS_SEARCH[league]
    out = []
    for s in sports or []:
        if str(s.get("group", "")).lower() != "soccer" or s.get("has_outrights"):
            continue
        text = " ".join([str(s.get("key", "")).replace("_", " "), str(s.get("title", "")),
                         str(s.get("description", ""))]).lower()
        if any(c in text for c in country) and any(c in text for c in comp):
            out.append(s)
    return out


def _event_summary(data: dict) -> dict:
    """Presence only (books, outcome counts, lines): no price leaves the store."""
    out = {"books": len(data.get("bookmakers") or []), "markets": {}}
    for b in data.get("bookmakers") or []:
        for m in b.get("markets") or []:
            s = out["markets"].setdefault(m.get("key"), {"books": 0, "outcomes": 0, "points": set()})
            s["books"] += 1
            s["outcomes"] += len(m.get("outcomes") or [])
            s["points"] |= {o["point"] for o in m.get("outcomes") or [] if _real_point(o.get("point")) is not None}
    for s in out["markets"].values():
        s["points"] = sorted(s["points"])
    return out


def probe_keys(meter: Meter, store: Store, days: dict, spell: dict, leagues=tuple(KEYLESS_SEARCH),
               window: tuple = PROBE_KEYS_WINDOW) -> dict:
    """(1) GET /sports?all=true (free; stops if the provider bills it); (2) per keyless league with
    exactly ONE candidate key: on our busiest match day in `window`, the provider's event listing
    (<= 1 credit) and one btts + alternate_totals event-odds call at kickoff - 1 h (<= 20 credits,
    free when empty) on the earliest event PROVEN to be one of our fixtures (else the earliest listed
    event, flagged unproven). The listing and the odds are kept in the store (tier 'PK'); the summary
    (presence, books, lines, match counts, credits; no price) goes to ledger/probe_keys.json."""
    meter.read_remaining()
    sports = meter.sports_list()
    res = {"probed_at": iso(datetime.now(timezone.utc)), "window": list(window),
           "soccer_keys_listed": sum(str(s.get("group", "")).lower() == "soccer" for s in sports),
           "leagues": {}}
    R = Resolver(spell)
    try:
        for lg in leagues:
            cands = find_candidate_keys(sports, lg)
            r = {"candidates": [{"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
                                for s in cands], "selected_key": None}
            res["leagues"][lg] = r
            if len(cands) != 1:
                r["reason"] = "no candidate key in the provider's list" if not cands else "more than one candidate"
                continue
            sk = cands[0]["key"]
            r["selected_key"] = sk
            ds = [(len(fx), d) for (l, d), fx in days.items() if l == lg and window[0] <= d <= window[1]]
            if not ds:
                r["reason"] = "no match day of ours in the window"
                continue
            d = max(ds)[1]
            r["probe_date"] = d
            lo, hi = day_window(d)
            k = lookup_key(sk, d)
            status, body, c = meter.call(f"/historical/sports/{sk}/events",
                                         {"date": iso(lookup_at(d)), "commenceTimeFrom": iso(lo),
                                          "commenceTimeTo": iso(hi)}, LOOKUP_MAX_COST, "PK", key=k)
            data = ((body or {}).get("data") or []) if status == 200 else []
            rows = [{"unit_key": k, "sport_key": sk, "league": lg, "league_date": d, "lookup_at": iso(lookup_at(d)),
                     "snapshot_ts": (body or {}).get("timestamp"), "event_id": e.get("id"),
                     "commence_time": e.get("commence_time"), "home_team": e.get("home_team"),
                     "away_team": e.get("away_team")} for e in data if e.get("id")]
            if status != "failed":
                store.add("events", rows)
                store.finish_unit(k, "PK", "ok" if rows else ("empty" if status == 200 else f"http_{status}"),
                                  c, len(rows), (body or {}).get("timestamp"))
                store.put_lookup(k, lg, d, rows)
            matched, rep = match_day(lg, d, days[(lg, d)], store.events_by_day, R)
            r["lookup"] = {"status": status, "credits": c, "events_in_window": rep["events_in_window"],
                           "fixtures": len(days[(lg, d)]), "fixtures_matched": len(matched)}
            listed = [e for e in store.events_by_day.get((lg, d), []) if _in_window(e, d)]
            pick = sorted(matched.values(), key=lambda e: e["commence_time"]) or \
                sorted(listed, key=lambda e: e["commence_time"])
            if not pick:
                r["event_odds"] = {"skipped": "the provider lists no event that day"}
                continue
            e = pick[0]
            at = parse_iso(e["commence_time"]) - timedelta(hours=1)
            ok = unit_key("event_odds", sk, e["id"], iso(at), PROBE_KEYS_MARKETS, ("eu",))
            status, body, c = meter.call(f"/historical/sports/{sk}/events/{e['id']}/odds",
                                         {"date": iso(at), "regions": "eu", "markets": ",".join(PROBE_KEYS_MARKETS),
                                          "oddsFormat": "decimal", "dateFormat": "iso"},
                                         CREDITS_PER_MARKET_REGION * len(PROBE_KEYS_MARKETS), "PK", key=ok)
            proven = e["id"] in {x["id"] for x in matched.values()}
            if status != 200:
                r["event_odds"] = {"status": status, "credits": c, "proven_fixture": proven}
                if status != "failed":
                    store.finish_unit(ok, "PK", f"http_{status}", c, 0, None)
                continue
            ev = (body or {}).get("data") or {}
            meta = {"tier": "PK", "unit_key": ok, "endpoint": "event_odds", "sport_key": sk, "league": lg,
                    "league_date": d, "snapshot_label": "T-1h", "requested_at": iso(at),
                    "snapshot_ts": (body or {}).get("timestamp")}
            orows, _ = odds_rows([ev] if ev else [], meta)
            store.add("odds", orows)
            store.finish_unit(ok, "PK", "ok" if orows else "empty", c, len(orows), meta["snapshot_ts"])
            r["event_odds"] = {"status": status, "credits": c, "proven_fixture": proven,
                               "snapshot_ts": meta["snapshot_ts"], **_event_summary(ev)}
    except Stop as s:
        res["stopped"] = s.reason
    store.flush()
    res["credits_spent"], res["calls"] = meter.spent, meter.calls
    f = store.root / "ledger" / PROBE_KEYS_NAME
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1, sort_keys=True, default=sorted) + "\n", encoding="utf-8")
    found = sum(1 for v in res["leagues"].values() if v.get("selected_key"))
    hist = sum(1 for v in res["leagues"].values() if (v.get("event_odds") or {}).get("books"))
    say(f"probe-keys: {len(res['leagues'])} leagues, {found} with one key in the provider's list, "
        f"{hist} with historical event odds; credits {meter.spent:,}")
    return res


def plan_digest(plan: dict) -> str:
    h = hashlib.sha256()
    for t, units in plan["units"].items():
        h.update(f"{t}:{len(units)}:{sum(u.max_cost for u in units)}:{len(plan['lookups'][t])};".encode())
    return h.hexdigest()[:16]
