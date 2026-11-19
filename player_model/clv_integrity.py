"""
Props CLV integrity - one status per CLV record (Phase 2 part 7, 2026-11-06).
=============================================================================
A prop CLV value compares the price a tip was logged at (ENTRY) with a later price (CLOSE). It is
only a measurement of the same quote when both prices are of the same player, market, line, side
and bookmaker basis, the entry price was really on offer for that fixture when the tip was logged,
and the close was quoted BEFORE kickoff. This module decides that, per record, from the recorded
prices only. It never computes or judges an edge: it labels records.

STATUSES (one per record)
  VALID_SAME_BOOK       identity proven, entry and close both before kickoff, and the entry and the
                        close are the same named bookmaker (an API-Football per-book row)
  VALID_CROSS_BOOK      identity and timing proven, but the close is best-of-books or another book
                        (OddsAPI rows carry no bookmaker: each is already the best across its books).
                        This is NOT same-book CLV; it is labelled, never relabelled
  IDENTITY_UNPROVEN     the entry price never appears for this player / market / fixture in the price
                        history at or before the tip time, the close price never appears for it, or
                        the player name maps to two provider ids in that fixture (namesake)
  CLOSE_AFTER_KICKOFF   the close price was first quoted at or after kickoff (an in-play price)
  ENTRY_AFTER_KICKOFF   the tip itself was logged at or after kickoff
  TIMING_UNPROVEN       no single recorded kickoff for the fixture (unknown, or two recorded values),
                        or the close price was seen both before and after kickoff
  NO_CLOSE              no close was recorded (or none exists before kickoff)
Only the two VALID_* statuses may carry a CLV value. The checks run in the order above, so a record
has exactly one status; `reason` names the check that decided it.

IDENTITY. Market strings carry the line (sot = 1+ shots on target, sot2 = 2+), and the history holds
the Over / Yes side only (odds_fetcher._best_price), the side every prop tip takes. The player is
matched on the normalised name (odds_fetcher._norm) inside the SAME fixture (odds_fetcher.same_fixture).
Provider ids are used where both sides carry one: a price row with a player_id must carry the record's
id, and a name that player_ledger maps to two ids on that date is a namesake (IDENTITY_UNPROVEN).

KICKOFF. The record's own `kickoff_utc` (written at entry since 2026-11-06), else a supplied map of
(match date, match) -> kickoff. Nothing is inferred from the match date.
"""
from __future__ import annotations

import pandas as pd

from player_model.odds_fetcher import _norm, same_fixture

VALID_SAME_BOOK = "VALID_SAME_BOOK"
VALID_CROSS_BOOK = "VALID_CROSS_BOOK"
IDENTITY_UNPROVEN = "IDENTITY_UNPROVEN"
CLOSE_AFTER_KICKOFF = "CLOSE_AFTER_KICKOFF"
ENTRY_AFTER_KICKOFF = "ENTRY_AFTER_KICKOFF"
TIMING_UNPROVEN = "TIMING_UNPROVEN"
NO_CLOSE = "NO_CLOSE"
STATUSES = (VALID_SAME_BOOK, VALID_CROSS_BOOK, IDENTITY_UNPROVEN, CLOSE_AFTER_KICKOFF,
            ENTRY_AFTER_KICKOFF, TIMING_UNPROVEN, NO_CLOSE)
VALID = (VALID_SAME_BOOK, VALID_CROSS_BOOK)
#: statuses close_out never revisits (TIMING_UNPROVEN is retried: a kickoff may still be recorded)
FINAL = (VALID_SAME_BOOK, VALID_CROSS_BOOK, IDENTITY_UNPROVEN, CLOSE_AFTER_KICKOFF,
         ENTRY_AFTER_KICKOFF, NO_CLOSE)

#: the source whose history rows name one bookmaker; OddsAPI rows are best-of-books (no bookmaker)
PER_BOOK_SOURCE = "api_football"
_ODDS_EPS = 1e-9


# ── parsing ────────────────────────────────────────────────────────────────────────────────────
def parse_bet_id(bet_id) -> tuple[str, str, str] | None:
    """PLAYER|date|player|market -> (date, player, market), or None."""
    parts = str(bet_id or "").split("|", 3)
    if len(parts) != 4 or parts[0] != "PLAYER":
        return None
    return parts[1][:10], parts[2], parts[3]


def utc(v) -> pd.Timestamp:
    """A recorded timestamp as UTC, or NaT (never a guessed time)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return pd.NaT
    t = pd.to_datetime(v, errors="coerce", utc=True)
    return t if isinstance(t, pd.Timestamp) else pd.NaT


def _odds(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 1.0 else None


def _id(v) -> str | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f == f and f > 0 else None


# ── price history ──────────────────────────────────────────────────────────────────────────────
def prepare_history(oh: pd.DataFrame, per_book: bool = False) -> pd.DataFrame:
    """Typed copy of player_prop_odds_history rows: _pk (normalised player), _d (match date), _ts
    (snapshot time UTC), _odds, _book ('source|bookmaker', lower) and _named_book (the bookmaker of a
    per-book row, else None). `per_book=True` for output/prop_quotes, where every source names its book;
    there only the Over / Yes side is kept (the side every prop tip takes)."""
    d = oh.copy() if oh is not None else pd.DataFrame()
    if per_book and "side" in d.columns:
        d = d[d["side"].astype(str).str.strip().str.lower().isin(["yes", "over"])]
    if d.empty or "player" not in d.columns:
        return pd.DataFrame(columns=list(d.columns) + ["_pk", "_d", "_ts", "_odds", "_book", "_named_book"])
    d["_pk"] = d["player"].astype(str).map(_norm)
    d["_d"] = d["match_date"].astype(str).str[:10]
    d["_ts"] = pd.to_datetime(d.get("snapshot_ts"), errors="coerce", utc=True)
    d["_odds"] = pd.to_numeric(d["odds"], errors="coerce")
    src = (d["source"].astype(object).where(d["source"].notna(), "oddsapi") if "source" in d.columns
           else pd.Series("oddsapi", index=d.index, dtype=object)).astype(str).str.strip().str.lower()
    book = (d["bookmaker"].astype(object).where(d["bookmaker"].notna(), "") if "bookmaker" in d.columns
            else pd.Series("", index=d.index, dtype=object)).astype(str).str.strip()
    book = book.where(~book.str.lower().isin(["nan", "none"]), "")
    d["_book"] = src + "|" + book.str.lower()
    d["_named_book"] = book.where(((src == PER_BOOK_SOURCE) | per_book) & (book != ""), None)
    return d


def index_history(h: pd.DataFrame) -> dict:
    """{(match date, normalised player, market): rows} for fast per-record lookups."""
    if h is None or h.empty:
        return {}
    return {k: g for k, g in h.groupby(["_d", "_pk", "market"], sort=False)}


def identity_rows(idx: dict, date: str, player: str, market: str, match: str,
                  player_id: str | None = None) -> pd.DataFrame:
    """History rows of THIS player, market (= line) and fixture on that match date. A row that carries
    a provider player_id must carry `player_id` (when the record has one)."""
    m = idx.get((str(date)[:10], _norm(player), str(market)))
    if m is None or m.empty:
        return pd.DataFrame()
    if match and "match" in m.columns:
        ok = {mm for mm in m["match"].astype(str).unique() if same_fixture(str(match), mm)}
        m = m[m["match"].astype(str).isin(ok)]
    if player_id and "player_id" in m.columns and len(m):
        rid = m["player_id"].map(_id)
        m = m[rid.isna() | (rid == player_id)]
    return m


def entry_rows(m: pd.DataFrame, odds_bet, ts_bet) -> pd.DataFrame:
    """Rows proving the entry: the same price, quoted at or before the tip time."""
    o, t = _odds(odds_bet), utc(ts_bet)
    if m is None or m.empty or o is None or pd.isna(t):
        return pd.DataFrame()
    return m[((m["_odds"] - o).abs() < _ODDS_EPS) & m["_ts"].notna() & (m["_ts"] <= t)]


def books(rows: pd.DataFrame) -> set:
    """Named bookmakers of `rows` (lower case); a best-of-books row contributes '*'."""
    if rows is None or rows.empty:
        return set()
    return {(str(b).lower() if b is not None and b == b else "*") for b in rows["_named_book"]}


def best_of_books(pre: pd.DataFrame) -> tuple[float | None, set]:
    """(close, books of that close): each book's LAST quote, then the best of them - the same basis as
    the best-of-books entry (clv_tracker.close_best_of_books)."""
    d = pre[pre["_odds"] > 1.0] if pre is not None and not pre.empty else pd.DataFrame()
    if d.empty:
        return None, set()
    last = d.sort_values("_ts", kind="mergesort").groupby("_book", sort=False).tail(1)
    best = float(last["_odds"].max())
    return best, books(last[(last["_odds"] - best).abs() < _ODDS_EPS])


def book_status(entry_books: set, close_books: set) -> str:
    """VALID_SAME_BOOK only when one named book supplied the entry and that same book the close."""
    if len(entry_books) == 1 and entry_books == close_books and "*" not in entry_books:
        return VALID_SAME_BOOK
    return VALID_CROSS_BOOK


def same_book_close(q: pd.DataFrame, odds_bet, ts_bet, odds_close, kickoff) -> str | None:
    """The named book whose quote at entry was `odds_bet` (quoted at or before the tip time) AND whose
    last quote before kickoff is `odds_close`, from per-book quotes (output/prop_quotes); else None.
    Quotes are stored first time seen per (book, price), so a book that returned to an earlier price
    keeps its earlier row: "last quote" is the latest recorded change, as in the history."""
    o_b, o_c, t, ko = _odds(odds_bet), _odds(odds_close), utc(ts_bet), utc(kickoff)
    if q is None or q.empty or o_b is None or o_c is None or pd.isna(t) or pd.isna(ko):
        return None
    q = q[q["_named_book"].notna() & q["_ts"].notna()]
    for b, g in q.groupby("_named_book", sort=True):
        if not (((g["_odds"] - o_b).abs() < _ODDS_EPS) & (g["_ts"] <= t)).any():
            continue
        pre = g[g["_ts"] < ko].sort_values("_ts", kind="mergesort")
        if len(pre) and abs(float(pre["_odds"].iloc[-1]) - o_c) < _ODDS_EPS:
            return str(b).lower()
    return None


# ── namesakes (provider ids) ───────────────────────────────────────────────────────────────────
def ledger_ids(ledger: pd.DataFrame | None) -> dict:
    """{(normalised player, market, match date): set of provider player_ids} from player_ledger."""
    out: dict = {}
    if ledger is None or ledger.empty or "player_id" not in ledger.columns:
        return out
    for name, mk, d, pid in zip(ledger["player_name"].astype(str), ledger["market"].astype(str),
                                ledger["match_date"].astype(str).str[:10], ledger["player_id"]):
        p = _id(pid)
        if p is not None:
            out.setdefault((_norm(name), mk, d), set()).add(p)
    return out


def entry_reason(m: pd.DataFrame, odds_bet) -> str:
    """Why the entry is unproven: no rows at all, the price only quoted after the tip, or never."""
    if m is None or m.empty:
        return "no history rows for this player/market/fixture"
    o = _odds(odds_bet)
    if o is not None and ((m["_odds"] - o).abs() < _ODDS_EPS).any():
        return "entry price quoted for this player/market/fixture only after the tip time"
    return "entry price never quoted for this player/market/fixture"


# ── the writer's decision (a record that has not closed yet) ───────────────────────────────────
def decide_close(rec: dict, m: pd.DataFrame, kickoff, namesake: bool = False) -> dict:
    """{status, reason, close, close_books, entry_books} for an OPEN record once its match has kicked
    off. The close is the best-of-books close over quotes strictly BEFORE kickoff; a record that fails
    identity or timing gets its status and no close."""
    out = {"status": None, "reason": "", "close": None, "close_books": set(), "entry_books": set()}
    ko = utc(kickoff)
    ts_bet = utc(rec.get("ts_bet"))
    if namesake:
        return {**out, "status": IDENTITY_UNPROVEN, "reason": "namesake: two provider ids for this name"}
    ent = entry_rows(m, rec.get("odds_bet"), rec.get("ts_bet"))
    if ent.empty:
        return {**out, "status": IDENTITY_UNPROVEN, "reason": entry_reason(m, rec.get("odds_bet"))}
    out["entry_books"] = books(ent)
    if pd.isna(ko):
        return {**out, "status": TIMING_UNPROVEN, "reason": "no recorded kickoff"}
    if ts_bet >= ko:
        return {**out, "status": ENTRY_AFTER_KICKOFF, "reason": "tip logged at or after kickoff"}
    close, cb = best_of_books(m[m["_ts"].notna() & (m["_ts"] < ko)])
    if close is None:
        return {**out, "status": NO_CLOSE, "reason": "no quote before kickoff"}
    st = book_status(out["entry_books"], cb)
    return {**out, "status": st, "close": close, "close_books": cb,
            "reason": "" if st == VALID_SAME_BOOK else "close is best-of-books or another book"}


# ── the audit's decision (a record as stored, closed or not) ───────────────────────────────────
def classify_stored(rec: dict, m: pd.DataFrame, kickoff, kickoff_ambiguous: bool = False,
                    namesake: bool = False, quotes: pd.DataFrame | None = None) -> dict:
    """{status, reason, entry_books, close_books} for a STORED record: is the stored close a
    same-identity, pre-kickoff quote? The close is traced to the history rows quoting the stored price
    for this player / market / fixture. `quotes` = this record's per-book quotes (output/prop_quotes,
    identity rows): when given, SAME BOOK is decided on them (same_book_close), because a history that
    kept one row per PRICE (production before 2026-11-02) records only the first book seen at a price."""
    out = {"status": None, "reason": "", "entry_books": set(), "close_books": set()}
    close = _odds(rec.get("odds_close"))
    if close is None:
        st = str(rec.get("clv_status") or "").strip()
        return {**out, "status": NO_CLOSE,
                "reason": (f"writer status {st}: {rec.get('clv_status_reason', '')}" if st
                           else "no close recorded")}
    if namesake:
        return {**out, "status": IDENTITY_UNPROVEN, "reason": "namesake: two provider ids for this name"}
    ent = entry_rows(m, rec.get("odds_bet"), rec.get("ts_bet"))
    if ent.empty:
        return {**out, "status": IDENTITY_UNPROVEN, "reason": entry_reason(m, rec.get("odds_bet"))}
    out["entry_books"] = books(ent)
    cand = m[((m["_odds"] - close).abs() < _ODDS_EPS) & m["_ts"].notna()]
    if cand.empty:
        return {**out, "status": IDENTITY_UNPROVEN,
                "reason": "close price never quoted for this player/market/fixture"}
    ko = utc(kickoff)
    if pd.isna(ko):
        return {**out, "status": TIMING_UNPROVEN,
                "reason": "kickoff recorded with two values" if kickoff_ambiguous else "no recorded kickoff"}
    if utc(rec.get("ts_bet")) >= ko:
        return {**out, "status": ENTRY_AFTER_KICKOFF, "reason": "tip logged at or after kickoff"}
    pre, post = cand[cand["_ts"] < ko], cand[cand["_ts"] >= ko]
    if pre.empty:
        return {**out, "status": CLOSE_AFTER_KICKOFF, "reason": "close price first quoted at or after kickoff"}
    if not post.empty:
        return {**out, "status": TIMING_UNPROVEN, "reason": "close price quoted both before and after kickoff"}
    if quotes is not None:
        b = same_book_close(quotes, rec.get("odds_bet"), rec.get("ts_bet"), close, ko)
        if b is not None:
            return {**out, "status": VALID_SAME_BOOK, "entry_books": {b}, "close_books": {b}}
        return {**out, "status": VALID_CROSS_BOOK,
                "reason": ("no per-book quotes for this player/market/fixture" if quotes.empty
                           else "no book quoted the entry price and then closed at the stored close")}
    out["close_books"] = books(pre)
    st = book_status(out["entry_books"], out["close_books"])
    return {**out, "status": st, "reason": "" if st == VALID_SAME_BOOK else "close is best-of-books or another book"}
