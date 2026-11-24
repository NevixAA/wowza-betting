"""TIP TRUTH GATE - no duplicates, no unreal data, no unreal markets (owner 2026-11-24, verbatim: "no duplicates, no
unreal data, not unreal markets"). Fail closed: a tip is FROZEN or SENT only when every check below holds; otherwise
it is blocked with a logged reason and never sent.

WHY. Argentina Primera Division 2026-W48: T1 and T3 were the same match (Rosario Central v Barracas Central, BTTS
Yes, both staked) under two OddsAPI event ids (kickoffs 2026-11-28 15:00Z and 2026-11-29 21:00Z). The candidate
builder de-duplicated by (market, match DATE, home, away), so two event ids one day apart were two "matches".

THE FOUR CHECKS (one reason per tip, the first that fails, in this order):
  1. REAL FIXTURE   the tip resolves to EXACTLY ONE API-Football fixture: same league (registry api_football_id), both
                    clubs proven (API-Football team ids from output/af_history.parquet, club-name rule of
                    src/fixture_identity), the AF kickoff date within the +-1 day window of src/fixture_identity, AF
                    status not postponed / cancelled / abandoned. The AF kickoff is AUTHORITATIVE. Sources:
                    output/collect/upcoming_fixtures.parquet (NS / TBD) and output/strength/competition_fixtures.parquet
                    (played). An OddsAPI-only event with no AF fixture is blocked (NO_AF_FIXTURE).
  2. NO DUPLICATES  one canonical match (its AF fixture id) carries at most ONE tip per market x side across all
                    slots, tiers, weeks and senders; and only ONE provider event per canonical match - the event whose
                    kickoff is closest to the AF kickoff - may feed tips (a stale / rescheduled event id is dropped,
                    STALE_EVENT_ID). Same-direction different markets stay allowed (src/tip_coherence).
  3. REAL PRICE     a captured bet-time price (odds > 1) from ONE NAMED bookmaker of the approved order
                    (src/bars_opt/oddsapi_prices.BOOK_ORDER, then the owner's O1 fallback books), two-sided (the other
                    side's price was captured with it), with a capture time strictly before the send / freeze time.
                    A cross-book MEDIAN of real captured quotes is accepted when it names at least MEDIAN_MIN_BOOKS (2)
                    books (today's engine basis; PR #104's O/U 2.5 median of 3+ books passes too); a median naming fewer
                    is blocked (MEDIAN_ONLY_PRICE). No price without a book and a capture time is ever accepted -
                    nothing is defaulted or filled.
  4. REAL MARKET    the market / line / side is a market with its league's own owner-applied bar
                    (MARKET_NOT_IN_APPROVED_BARS), and the named book actually quoted that market / side for that
                    match in a captured snapshot at or before the price's capture time
                    (output/book_odds_snapshots.csv + archive; props: output/player_prop_odds_history.csv)
                    (MARKET_NOT_OFFERED_BY_BOOK).

Pure apart from TipTruth.load (reads the committed files once). Never prints a price.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TRUTH_VERSION = "tip-truth/1.0"

# ── reasons ──────────────────────────────────────────────────────────────────────────────────────
NO_AF_FIXTURE = "NO_AF_FIXTURE"
AF_TEAM_UNRESOLVED = "AF_TEAM_UNRESOLVED"
AF_AMBIGUOUS = "AF_FIXTURE_AMBIGUOUS"
AF_KICKOFF_MISMATCH = "AF_KICKOFF_MISMATCH"
AF_NOT_PLAYABLE = "AF_POSTPONED_OR_CANCELLED"
NO_LEAGUE_AF_ID = "NO_LEAGUE_AF_ID"
DUPLICATE = "DUPLICATE_MATCH_PICK"
STALE_EVENT = "STALE_EVENT_ID"
NO_PRICE = "NO_PRICE"
NO_BOOK = "NO_NAMED_BOOK"
MEDIAN_ONLY = "MEDIAN_ONLY_PRICE"
BOOK_NOT_APPROVED = "BOOK_NOT_APPROVED"
NO_CAPTURE = "NO_CAPTURE_TIME"
CAPTURED_LATE = "CAPTURED_AFTER_SEND"
ONE_SIDED = "ONE_SIDED_PRICE"
NOT_IN_BARS = "MARKET_NOT_IN_APPROVED_BARS"
NOT_OFFERED = "MARKET_NOT_OFFERED_BY_BOOK"

FIXTURE_REASONS = (NO_LEAGUE_AF_ID, AF_TEAM_UNRESOLVED, NO_AF_FIXTURE, AF_AMBIGUOUS, AF_KICKOFF_MISMATCH,
                   AF_NOT_PLAYABLE)
DUP_REASONS = (DUPLICATE, STALE_EVENT)
PRICE_REASONS = (NO_PRICE, NO_BOOK, MEDIAN_ONLY, BOOK_NOT_APPROVED, NO_CAPTURE, CAPTURED_LATE, ONE_SIDED)
MARKET_REASONS = (NOT_IN_BARS, NOT_OFFERED)
CHECK_OF = {**{r: "1_real_fixture" for r in FIXTURE_REASONS}, **{r: "2_no_duplicates" for r in DUP_REASONS},
            **{r: "3_real_price" for r in PRICE_REASONS}, **{r: "4_real_market" for r in MARKET_REASONS}}

DAY_WINDOW = 1                                   # src.fixture_identity.DAY_WINDOW (the settlement finder's +-1 day)
#: API-Football statuses of a match that will not be played as scheduled
BAD_STATUS = {"PST", "CANC", "ABD", "SUSP", "AWD", "WO", "INT"}


def _approved_books() -> tuple[str, ...]:
    from src.bars_opt import oddsapi_prices as OP
    return tuple(OP.BOOK_ORDER) + tuple(OP.O1_FALLBACK_BOOKS)


#: provider spellings of the approved books -> the OddsAPI key (API-Football names props books "Bet365", "1xBet")
_BOOK_ALIAS = {"bet365": "bet365", "pinnacle": "pinnacle", "williamhill": "williamhill", "1xbet": "onexbet",
               "onexbet": "onexbet", "ladbrokes": "ladbrokes_uk", "ladbrokesuk": "ladbrokes_uk"}


def book_key(name) -> str:
    """The OddsAPI key of a bookmaker name ('' when blank). Unknown books keep their normalised spelling."""
    s = str(name or "").strip()
    if not s or s.lower() == "nan":
        return ""
    k = re.sub(r"[^a-z0-9]", "", s.lower())
    return _BOOK_ALIAS.get(k, s.lower())


#: a cross-book median is a real price only over at least this many NAMED books (src.btts_odds.book_label writes
#: "median:" + the sorted distinct book keys). Today's engine median uses 2+; PR #104 (O/U 2.5) uses 3+.
MEDIAN_MIN_BOOKS = 2


def median_books(label) -> list[str] | None:
    """The named books of a "median:a;b;c" label (None when the label is not a median)."""
    s = str(label or "").strip()
    if not s.lower().startswith("median"):
        return None
    _, _, rest = s.partition(":")
    return sorted({book_key(b) for b in rest.split(";") if book_key(b)})


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _ts(v) -> pd.Timestamp:
    t = pd.to_datetime(v, utc=True, errors="coerce")
    return t if isinstance(t, pd.Timestamp) else pd.NaT


# ── 1. API-Football fixtures ─────────────────────────────────────────────────────────────────────
class AFIndex:
    """API-Football fixtures (upcoming + played) and the league's club -> AF team id map."""

    def __init__(self, fixtures: pd.DataFrame, names: pd.DataFrame, league_ids: dict):
        f = fixtures.copy()
        f["fixture_id"] = f["fixture_id"].astype("int64")
        f["kickoff"] = pd.to_datetime(f["kickoff"], utc=True, errors="coerce")
        f["day"] = f["kickoff"].dt.tz_localize(None).dt.normalize()
        self.fx = f
        self.by_id = {int(r.fixture_id): r for r in f.itertuples(index=False)}
        self.by_league = {int(k): g for k, g in f.groupby("league_id")}
        self.league_ids = league_ids
        self.names: dict = {}                       # league -> {spelling: team id}
        for lg, g in names.groupby("league"):
            m: dict = {}
            for n, i in zip(g["name"], g["team_id"]):
                if pd.notna(i):
                    m.setdefault(str(n), set()).add(int(i))
            self.names[str(lg)] = {n: next(iter(ids)) for n, ids in m.items() if len(ids) == 1}
        self._memo: dict = {}

    @classmethod
    def load(cls, root: Path | None = None) -> "AFIndex":
        root = Path(root) if root else ROOT
        parts = []
        up = root / "output" / "collect" / "upcoming_fixtures.parquet"
        if up.exists():
            u = pd.read_parquet(up)
            u = pd.DataFrame({"fixture_id": u["fixture_id"], "league_id": u["league_id"],
                              "kickoff": pd.to_datetime(u["kickoff_ts"], unit="s", utc=True),
                              "status": u["status"].astype(str), "home_id": u["home_id"], "away_id": u["away_id"]})
            parts.append(u)
        cf = root / "output" / "strength" / "competition_fixtures.parquet"
        names = []
        if cf.exists():
            c = pd.read_parquet(cf, columns=["fixture_id", "league_id", "league", "date", "home_id", "home_name",
                                             "away_id", "away_name", "status"])
            parts.append(pd.DataFrame({"fixture_id": c["fixture_id"], "league_id": c["league_id"],
                                       "kickoff": pd.to_datetime(c["date"], utc=True, errors="coerce"),
                                       "status": c["status"].astype(str), "home_id": c["home_id"],
                                       "away_id": c["away_id"]}))
            for side in ("home", "away"):
                names.append(pd.DataFrame({"league": c["league"].astype(str), "name": c[f"{side}_name"].astype(str),
                                           "team_id": c[f"{side}_id"]}))
        ah = root / "output" / "af_history.parquet"
        if ah.exists():
            h = pd.read_parquet(ah, columns=["league", "home_team", "home_team_id", "away_team", "away_team_id"])
            for side in ("home", "away"):
                names.append(pd.DataFrame({"league": h["league"].astype(str), "name": h[f"{side}_team"].astype(str),
                                           "team_id": h[f"{side}_team_id"]}))
        fx = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["fixture_id", "league_id", "kickoff", "status", "home_id", "away_id"]))
        # an upcoming row (exact kickoff) wins over a played row's date-only kickoff for the same id
        fx = fx.drop_duplicates("fixture_id", keep="first")
        nm = pd.concat(names, ignore_index=True) if names else pd.DataFrame(columns=["league", "name", "team_id"])
        nm = nm.dropna(subset=["team_id"]).drop_duplicates()
        from src import league_registry as LR
        ids = {}
        for e in LR.leagues() if callable(getattr(LR, "leagues", None)) else []:
            e = e if isinstance(e, dict) else getattr(e, "__dict__", {})
            if e.get("api_football_id") is not None:
                ids[str(e.get("name"))] = int(e["api_football_id"])
        return cls(fx, nm, ids)

    def league_id(self, league: str) -> int | None:
        lid = self.league_ids.get(str(league))
        if lid is None:
            try:
                from src import league_registry as LR
                e = LR.entry(str(league)) or {}
                lid = e.get("api_football_id")
            except Exception:                                          # noqa: BLE001
                lid = None
        return int(lid) if lid is not None else None

    def team_id(self, league: str, name: str) -> int | None:
        key = (str(league), str(name))
        if key in self._memo:
            return self._memo[key]
        pool = self.names.get(str(league), {})
        tid = pool.get(str(name))
        if tid is None and pool:
            from src.fixture_identity import _Clubs
            clubs = _Clubs(list(pool) + [str(name)])
            hits = {pool[n] for n in pool if clubs.same(str(name), n)}
            tid = next(iter(hits)) if len(hits) == 1 else None
        self._memo[key] = tid
        return tid

    def resolve(self, league: str, kickoff, home: str, away: str, af_fixture_id=None) -> dict:
        """{reason, af_fixture_id, af_kickoff_utc, af_status}; reason '' = exactly one real AF fixture."""
        ko = _ts(kickoff)
        out = {"reason": "", "af_fixture_id": None, "af_kickoff_utc": pd.NaT, "af_status": ""}
        if af_fixture_id not in (None, "") and str(af_fixture_id).strip().isdigit():
            r = self.by_id.get(int(str(af_fixture_id).strip()))
            if r is None:
                return out | {"reason": NO_AF_FIXTURE}
            cands = [r]
        else:
            lid = self.league_id(league)
            if lid is None:
                return out | {"reason": NO_LEAGUE_AF_ID}
            hid, aid = self.team_id(league, home), self.team_id(league, away)
            if hid is None or aid is None:
                return out | {"reason": AF_TEAM_UNRESOLVED}
            g = self.by_league.get(lid)
            if g is None or pd.isna(ko):
                return out | {"reason": NO_AF_FIXTURE}
            day = ko.tz_localize(None).normalize()
            m = g[(g["home_id"] == hid) & (g["away_id"] == aid) & ((g["day"] - day).abs() <= pd.Timedelta(days=DAY_WINDOW))]
            cands = list(m.itertuples(index=False))
            if not cands:
                return out | {"reason": NO_AF_FIXTURE}
            if len(cands) > 1:
                return out | {"reason": AF_AMBIGUOUS}
        r = cands[0]
        out.update(af_fixture_id=int(r.fixture_id), af_kickoff_utc=r.kickoff, af_status=str(r.status))
        if str(r.status) in BAD_STATUS:
            return out | {"reason": AF_NOT_PLAYABLE}
        if pd.notna(ko) and pd.notna(r.kickoff) and abs((r.day - ko.tz_localize(None).normalize()).days) > DAY_WINDOW:
            return out | {"reason": AF_KICKOFF_MISMATCH}
        return out


# ── 4. captured book snapshots ───────────────────────────────────────────────────────────────────
TEAM_SNAP_MARKET = {"OU25": "OU25", "BTTS": "BTTS", "OVER15": "OU15", "OVER35": "OU35"}


class Snapshots:
    """First capture time of (book, league, match, market, side) in the team snapshots, and of (book, player_id,
    market, match_date) in the props quote history. Prices are never read."""

    def __init__(self, team: pd.DataFrame | None, props: pd.DataFrame | None):
        self.team: dict = {}
        if team is not None and len(team):
            t = team.copy()
            t["k"] = list(zip(t["bookmaker"].map(book_key), t["league"].astype(str), t["match"].astype(str),
                              t["market"].astype(str), t["side"].astype(str)))
            t["ts"] = pd.to_datetime(t["snapshot_ts"], utc=True, errors="coerce")
            t["d"] = pd.to_datetime(t["match_date"], errors="coerce")
            for k, g in t.groupby("k"):
                self.team[k] = list(zip(g["d"], g["ts"]))
        self.props: dict = {}
        if props is not None and len(props):
            p = props.copy()
            p["ts"] = pd.to_datetime(p["snapshot_ts"], utc=True, errors="coerce")
            for k, g in p.groupby([p["bookmaker"].map(book_key), p["player_id"].astype(str), p["market"].astype(str)]):
                self.props[k] = list(zip(pd.to_datetime(g["match_date"], errors="coerce"), g["ts"]))

    @classmethod
    def load(cls, out_dir: Path | None = None) -> "Snapshots":
        od = Path(out_dir) if out_dir else ROOT / "output"
        cols = ["snapshot_ts", "league", "match_date", "match", "bookmaker", "market", "side"]
        frames = []
        for p in [od / "book_odds_snapshots.csv"] + sorted((od / "book_odds_snapshots_archive").glob("*.csv")):
            if p.exists():
                try:
                    frames.append(pd.read_csv(p, usecols=lambda c: c in cols, dtype=str))
                except (ValueError, OSError) as e:
                    log.warning(f"[tip_truth] snapshot file {p.name} unreadable ({type(e).__name__})")
        team = pd.concat(frames, ignore_index=True) if frames else None
        props = None
        pp = od / "player_prop_odds_history.csv"
        if pp.exists():
            props = pd.read_csv(pp, usecols=lambda c: c in ("snapshot_ts", "match_date", "market", "bookmaker",
                                                            "player_id"), dtype=str).dropna(subset=["player_id"])
        return cls(team, props)

    def team_offered(self, book: str, league: str, home: str, away: str, market: str, side: str,
                     match_date, by: pd.Timestamp) -> bool:
        k = (book, str(league), f"{home} vs {away}", TEAM_SNAP_MARKET.get(str(market), str(market)), str(side))
        d = pd.to_datetime(str(match_date)[:10], errors="coerce")
        for sd, ts in self.team.get(k, []):
            if pd.notna(ts) and ts <= by and (pd.isna(d) or pd.isna(sd) or abs((sd - d).days) <= DAY_WINDOW):
                return True
        return False

    def prop_offered(self, book: str, player_id: str, market: str, match_date, by: pd.Timestamp) -> bool:
        d = pd.to_datetime(str(match_date)[:10], errors="coerce")
        pid = str(player_id).split(".")[0]
        for sd, ts in self.props.get((book, pid, str(market)), []):
            if pd.notna(ts) and ts <= by and (pd.isna(d) or pd.isna(sd) or abs((sd - d).days) <= DAY_WINDOW):
                return True
        return False


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────
class TipTruth:
    """check(row, now, kind) -> (reason, info). kind 'team' (TOP-3 / side / HT tips keyed by league + clubs) or
    'props' (fixture_id = API-Football id). `bars_ok(league, market, side) -> bool` is the caller's own approved-bar
    test (rule 4); None = no market has an approved bar (fail closed)."""

    def __init__(self, af: AFIndex, snaps: Snapshots, bars_ok=None, books: tuple | None = None):
        self.af, self.snaps, self.bars_ok = af, snaps, bars_ok
        self.books = tuple(books) if books is not None else _approved_books()

    @classmethod
    def load(cls, out_dir: Path | None = None, root: Path | None = None, bars_ok=None) -> "TipTruth":
        return cls(AFIndex.load(root), Snapshots.load(out_dir), bars_ok)

    def fixture(self, r: dict, kind: str = "team") -> dict:
        return self.af.resolve(r.get("league"), r.get("kickoff_utc") or r.get("match_date"),
                               r.get("home_team"), r.get("away_team"),
                               af_fixture_id=r.get("fixture_id") if kind == "props" else r.get("af_fixture_id"))

    def price(self, r: dict, now, kind: str = "team") -> str:
        if not _f(r.get("odds")) > 1.0:
            return NO_PRICE
        raw = str(r.get("entry_bookmaker") or "").strip()
        if not raw or raw.lower() == "nan":
            return NO_BOOK
        books = median_books(raw)
        if books is not None:
            # the cross-book MEDIAN basis (coordinator 2026-11-24): today's engine median (2+ named books) and
            # PR #104's O/U 2.5 median of 3+ real books are both a real price; a "median" naming < 2 books is not
            if len(books) < MEDIAN_MIN_BOOKS:
                return MEDIAN_ONLY
        elif book_key(raw) not in self.books:
            return BOOK_NOT_APPROVED
        cap = _ts(r.get("entry_captured_at"))
        if pd.isna(cap):
            return NO_CAPTURE
        if not cap < _ts(now):
            return CAPTURED_LATE
        if kind == "props":
            if str(r.get("fair_basis") or "") != "two_sided_devig":
                return ONE_SIDED
        elif not _f(r.get("opposite_side_odds")) > 1.0:
            return ONE_SIDED
        return ""

    def market(self, r: dict, kind: str = "team") -> str:
        ok = self.bars_ok(r.get("league"), r.get("market"), r.get("side")) if self.bars_ok else False
        if not ok:
            return NOT_IN_BARS
        book = book_key(r.get("entry_bookmaker"))
        by = _ts(r.get("entry_captured_at"))
        med = median_books(r.get("entry_bookmaker"))
        if med is not None and kind != "props":
            # a median price: at least MEDIAN_MIN_BOOKS of its named books quoted that market / side in a snapshot
            n = sum(self.snaps.team_offered(b, r.get("league"), r.get("home_team"), r.get("away_team"),
                                            r.get("market"), r.get("side"), r.get("match_date"), by) for b in med)
            return "" if n >= MEDIAN_MIN_BOOKS else NOT_OFFERED
        if kind == "props":
            hit = self.snaps.prop_offered(book, r.get("player_id"), r.get("market"), r.get("match_date"), by)
        else:
            hit = self.snaps.team_offered(book, r.get("league"), r.get("home_team"), r.get("away_team"),
                                          r.get("market"), r.get("side"), r.get("match_date"), by)
        return "" if hit else NOT_OFFERED

    def check(self, r: dict, now, kind: str = "team") -> tuple[str, dict]:
        """(first failing reason or '', fixture info). Order: 1 real fixture, 3 real price, 4 real market (2 is
        a property of a SET of tips: dedupe())."""
        info = self.fixture(r, kind)
        if info["reason"]:
            return info["reason"], info
        why = self.price(r, now, kind) or self.market(r, kind)
        return why, info

    def gate(self, df: pd.DataFrame, now, kind: str = "team", prior: set | None = None,
             key_cols=("market", "side")) -> pd.DataFrame:
        """`df` + columns truth_reason, canonical_match (AF fixture id), af_kickoff_utc. Rows failing 1/3/4 carry
        their reason; among the rest, dedupe() flags stale event ids and duplicate (match, market, side) picks
        (including against `prior`: keys already frozen / sent). Pure apart from logging."""
        d = df.copy()
        reasons, canon, afko = [], [], []
        for r in d.to_dict("records"):
            why, info = self.check(r, r.get("_now") or now, kind)
            reasons.append(why)
            canon.append("" if info["af_fixture_id"] is None else f"af:{info['af_fixture_id']}")
            afko.append(info["af_kickoff_utc"])
        d["truth_reason"], d["canonical_match"], d["af_kickoff_utc"] = reasons, canon, afko
        return dedupe(d, prior, key_cols)


def dedupe(d: pd.DataFrame, prior: set | None = None, key_cols=("market", "side"), event_col="fixture_id",
           order_col: str | None = None) -> pd.DataFrame:
    """Rule 2 on rows already carrying canonical_match / af_kickoff_utc / truth_reason. Per canonical match only
    the provider event whose kickoff is closest to the AF kickoff feeds tips (others: STALE_EVENT_ID); then one
    row per (canonical match, *key_cols) - the first in `order_col` order (else row order) - and none whose key
    is in `prior` (DUPLICATE_MATCH_PICK)."""
    d = d.copy()
    prior = set(prior or ())
    live = d[(d["truth_reason"] == "") & (d["canonical_match"] != "")]
    if event_col in d.columns and len(live):
        ko = pd.to_datetime(live["kickoff_utc"], utc=True, errors="coerce")
        afk = pd.to_datetime(live["af_kickoff_utc"], utc=True, errors="coerce")
        gap = (ko - afk).abs()
        for cm, g in live.assign(_gap=gap).groupby("canonical_match"):
            ev = g.groupby(g[event_col].astype(str))["_gap"].min().sort_values(kind="mergesort")
            if len(ev) > 1:
                keep = ev.index[0]
                stale = g.index[g[event_col].astype(str) != keep]
                d.loc[stale, "truth_reason"] = STALE_EVENT
    seen = set(prior)
    idx = d.index if order_col is None else d.sort_values(order_col, kind="mergesort").index
    for i in idx:
        if d.at[i, "truth_reason"] or not d.at[i, "canonical_match"]:
            continue
        k = (d.at[i, "canonical_match"],) + tuple(str(d.at[i, c]) for c in key_cols)
        if k in seen:
            d.at[i, "truth_reason"] = DUPLICATE
        else:
            seen.add(k)
    return d


def pick_keys(d: pd.DataFrame, key_cols=("market", "side")) -> set:
    """(canonical match, *key_cols) of the rows that passed the gate."""
    ok = d[(d["truth_reason"] == "") & (d["canonical_match"] != "")]
    return {(cm,) + tuple(str(x) for x in rest) for cm, *rest in
            zip(ok["canonical_match"], *[ok[c] for c in key_cols])}


def log_blocks(d: pd.DataFrame, who: str) -> None:
    """One log line per blocked tip (reason only; never a price)."""
    for r in d[d["truth_reason"] != ""].to_dict("records"):
        log.warning(f"[tip_truth] BLOCK {who}: {r.get('league')} {r.get('home_team')} v {r.get('away_team')} "
                    f"{r.get('market')} {r.get('side', '')} - {r['truth_reason']}")


# ── loading (production) and the test seam ──────────────────────────────────────────────────────
#: tests/conftest.py installs a pass-all factory so the many synthetic-fixture tests of the TOP-3 / props / notifier
#: paths keep testing what they test; tests/test_tip_truth.py exercises the real gate. Production never sets it.
_factory = None


def default(out_dir: Path | None = None, root: Path | None = None, bars_ok=None):
    """The production gate (TipTruth.load). Raises when it cannot load: every caller then freezes / sends NOTHING."""
    if _factory is not None:
        return _factory(out_dir=out_dir, root=root, bars_ok=bars_ok)
    return TipTruth.load(out_dir, root, bars_ok)


class PassAll:
    """Test double: every tip is a real fixture of its own provider event; rule 2 still applies."""

    def fixture(self, r: dict, kind: str = "team") -> dict:
        return {"reason": "", "af_fixture_id": None, "af_kickoff_utc": pd.NaT, "af_status": ""}

    def check(self, r: dict, now, kind: str = "team") -> tuple[str, dict]:
        return "", self.fixture(r, kind)

    def gate(self, df: pd.DataFrame, now, kind: str = "team", prior: set | None = None,
             key_cols=("market", "side")) -> pd.DataFrame:
        d = df.copy()
        d["truth_reason"] = ""
        d["canonical_match"] = ["ev:" + str(x) for x in d["fixture_id"]] if "fixture_id" in d.columns else ""
        d["af_kickoff_utc"] = d["kickoff_utc"] if "kickoff_utc" in d.columns else pd.NaT
        return dedupe(d, prior, key_cols)


def prior_keys(truth, d: pd.DataFrame, kind: str = "team", key_cols=("market", "side")) -> set:
    """(canonical match, *key_cols) of tips ALREADY frozen / sent (their price status does not matter: a second
    tip on the same match x market x side is a duplicate either way)."""
    if d is None or d.empty:
        return set()
    g = truth.gate(d, None, kind, key_cols=key_cols)
    g = g[g["canonical_match"] != ""]
    return {(cm,) + tuple(str(x) for x in rest) for cm, *rest in zip(g["canonical_match"], *[g[c] for c in key_cols])}


# ── scan of the frozen ledgers (data truth rule 18 + the owner's per-failure counts) ───────────────────────────────
LEDGERS = {"top3": ("top3_league_ledger.csv", "team", ("market", "side")),
           "props_top3": ("props_top3_ledger.csv", "props", ("player_id", "market"))}
CORRECTIONS = Path("output") / "corrections" / "tip_truth_corrections.csv"
CORR_COLS = ["pick_id", "ledger", "league_week_id", "league", "home_team", "away_team", "market", "selection",
             "truth_check", "truth_reason", "action", "owner_action", "decided_at", "note"]


def _bars_for(kind: str, root: Path):
    if kind == "props":
        from src import props_top3 as PT
        return PT.truth_bars_ok
    from src import top3_league as TL
    return TL.truth_bars_ok(TL.bars_context(root))


def scan(root: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Every frozen TOP-3 / props TOP-3 pick with its truth verdict AT ITS ISSUE TIME (price captured before issue),
    rule 2 applied in issue order across the whole ledger. Columns: ledger, pick_id, ..., truth_reason, truth_check."""
    root = Path(root) if root else ROOT
    od = Path(out_dir) if out_dir else root / "output"
    af = AFIndex.load(root)
    snaps = Snapshots.load(od)
    out = []
    for name, (fname, kind, keys) in LEDGERS.items():
        p = od / fname
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str).fillna("")
        if d.empty:
            continue
        t = TipTruth(af, snaps, _bars_for(kind, root))
        d["_now"] = d["issued_at"]
        g = t.gate(d.sort_values(["issued_at", "pick_id"], kind="mergesort"), None, kind, key_cols=keys)
        # every check on its own (the owner's per-failure counts), not only the first failing one
        recs = g.to_dict("records")
        g["fixture_reason"] = [t.fixture(r, kind)["reason"] for r in recs]
        g["price_reason"] = [t.price(r, r["issued_at"], kind) for r in recs]
        g["market_reason"] = [t.market(r, kind) for r in recs]
        g["ledger"] = name
        g["selection"] = g["side"] if kind == "team" else g.get("player_name", g["player_id"])
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["ledger", "pick_id", "truth_reason", "truth_check"])
    res = pd.concat(out, ignore_index=True)
    res["truth_check"] = res["truth_reason"].map(lambda r: CHECK_OF.get(r, "") if r else "")
    return res.drop(columns=["_now"], errors="ignore")


def read_corrections(root: Path | None = None) -> pd.DataFrame:
    p = (Path(root) if root else ROOT) / CORRECTIONS
    if not p.exists():
        return pd.DataFrame(columns=CORR_COLS)
    return pd.read_csv(p, dtype=str).fillna("")


def counts(res: pd.DataFrame) -> dict:
    """{ledger: {picks, pass, fail, by_check, by_reason}}."""
    out = {}
    for name, g in res.groupby("ledger"):
        bad = g[g["truth_reason"] != ""]
        each = {}
        for col in ("fixture_reason", "price_reason", "market_reason"):
            if col in g.columns:
                each[col] = {k: int(v) for k, v in g.loc[g[col] != "", col].value_counts().sort_index().items()}
        out[name] = {"picks": int(len(g)), "pass": int(len(g) - len(bad)), "fail": int(len(bad)), "each_check": each,
                     "by_check": {k: int(v) for k, v in bad["truth_check"].value_counts().sort_index().items()},
                     "by_reason": {k: int(v) for k, v in bad["truth_reason"].value_counts().sort_index().items()}}
    return out


def duplicate_picks(res: pd.DataFrame) -> pd.DataFrame:
    """Rule 2 on its own, whatever checks 1/3/4 say: picks that repeat an EARLIER pick's (canonical match, market,
    side) [props: (match, player, market)] in the same ledger. The canonical match is the AF fixture when resolved,
    else the src/fixture_identity group of the provider event ids (same league, same clubs, +-1 day)."""
    from src import fixture_identity as FI
    out = []
    for name, g in res.groupby("ledger"):
        g = g.sort_values(["issued_at", "pick_id"], kind="mergesort").copy()
        if name == "props_top3":
            g["_c"] = "af:" + g["fixture_id"].astype(str)
            keys = ["_c", "player_id", "market"]
        else:
            ref = [c if c else "oa:" + f for c, f in zip(g["canonical_match"], g["fixture_id"])]
            fx = pd.DataFrame({"fixture_ref": ref, "league_slug": g["league"].values, "league": g["league"].values,
                               "kickoff_utc": g["kickoff_utc"].values, "home_team": g["home_team"].values,
                               "away_team": g["away_team"].values})
            canon = FI.resolve_identities(fx).set_index("fixture_ref")["canonical_fixture_ref"]
            g["_c"] = [canon.get(r, r) for r in ref]
            keys = ["_c", "market", "side"]
        dup = g[g.duplicated(keys, keep="first")]
        out.append(dup.drop(columns="_c"))
    return pd.concat(out, ignore_index=True) if out else res.iloc[0:0]
