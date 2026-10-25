"""API-Football per-book quotes as a SECOND price source for the market_edge_v2 shadow. SHADOW ONLY.

    from src.edge_af_quotes import af_book_pairs, merge_pairs

WHY (owner-approved 2026-10-25). OddsAPI per-book quotes (src/predict._LAST_BOOK_ROWS,
src/btts_odds._LAST_BOOK_ROWS, output/book_odds_snapshots.csv) almost never carry Under 1.5, so 176 of
183 OU15 rows and 130 of 183 OU35 rows in output/edge_engine_shadow.csv had no two-sided price
(docs/EDGE_ENGINE_AUDIT_2026_10.md). The batch capture jobs (std_odds_capture.yml / nf_odds_capture.yml
via src/book_quotes.py) already store 7-8 API-Football books with BOTH sides of O/U 1.5 / 2.5 / 3.5 and
BTTS in output/book_quotes/<date>.csv. This module reads those COMMITTED files. It makes no API call,
so it adds nothing to the 5-minute predict loop's quota and only a file read to its wall-clock.

RULES (each one has a test in tests/test_edge_af_quotes.py)
  * SAME BOOK, SAME SNAPSHOT. Over is paired with Under (Yes with No) only when both rows carry the
    same bookmaker AND the same snapshot_ts. book_quotes is a CHANGE-LOG, so a later row on ONE side
    means the latest state of that book is not a same-snapshot pair: the book is passed on as
    ONE-SIDED (the engine counts it and never uses it as a market). No carry-forward synthesis, and
    never book A's over with book B's under.
  * STRICTLY PRE-KICKOFF and not in the future: snapshot_ts <= now and < kickoff (both the board's
    kickoff and API-Football's own), minutes_to_kickoff > 0.
  * ONE LINE PER FAMILY. market "ou" line 1.5 -> OU15, 2.5 -> OU25, 3.5 -> OU35; "btts" -> BTTS.
    Any other line (and every half-time "ht_ou" row) is dropped, so 1.5 is never mixed with 2.5.
  * NAMES. API-Football spells clubs differently from the board ("Sheffield Utd", "Argentinos JRS").
    Each board club is resolved with src.team_names.resolve against the API-Football clubs of the SAME
    league only; an ambiguous resolution returns None and is refused. A fixture is matched when
      BOTH_TEAMS   both clubs resolve and exactly one API-Football fixture has that home/away pair
                   with kickoff within KICKOFF_TOL_BOTH of the board's; or
      ONE_TEAM_KO  one club resolves, the other resolves to nothing (NOT to a different club), and
                   exactly one API-Football fixture in the league kicks off within KICKOFF_TOL_ONE of
                   the board's with the resolved club on the SAME side.
    Anything else is refused and counted (unmatched / refused_ambiguous / kickoff_mismatch).
  * TIMESTAMP + SOURCE. Every pair carries its own snapshot_ts (BookQuote.ts), so the engine's
    price_timestamp / price_age_minutes / stale gate apply unchanged. Books are named "af:<slug>" so
    `books_used` in the shadow log shows which source each book came from.
  * ONE VOTE PER PHYSICAL BOOK. Pinnacle, Marathonbet, 1xBet, William Hill and BetVictor are quoted by
    both sources; merge_pairs keeps the FRESHER two-sided quote of the two (OddsAPI on a tie) so the
    consensus median never counts one bookmaker twice.
  * Validation (odds > 1, overround inside [overround_min, overround_max] which rejects arbitrage and
    garbage, stale limits) is the engine's own per-book validation (src/market_edge._evaluate) and is
    applied to these quotes exactly as to OddsAPI's.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import market_edge as ME
from src.team_names import resolve

log = logging.getLogger(__name__)

PREFIX = "af:"
#: book_quotes (market, line) -> engine family. Everything else is not a family we price.
LINE_FAMILY = {("ou", 1.5): "OU15", ("ou", 2.5): "OU25", ("ou", 3.5): "OU35"}
#: API-Football bookmaker -> the OddsAPI key of the SAME physical book (dedup in merge_pairs).
#: "Betfair" is deliberately absent: OddsAPI's betfair_ex_* is the exchange, API-Football's is not.
SAME_BOOK = {"pinnacle": "pinnacle", "marathonbet": "marathonbet", "1xbet": "onexbet",
             "william hill": "williamhill", "betvictor": "betvictor"}
KICKOFF_TOL_BOTH = pd.Timedelta(hours=3)
KICKOFF_TOL_ONE = pd.Timedelta(minutes=15)
#: Parts read: today and the previous LOOKBACK_DAYS. A FAR opening price is re-written once it falls
#: out of the capture's two-part dedup window, so three days hold every live entity; anything older
#: is far past the engine's stale limit anyway.
LOOKBACK_DAYS = 2
USECOLS = ["snapshot_ts", "fixture_id", "kickoff_utc", "minutes_to_kickoff", "ladder_band", "league",
           "home_team", "away_team", "bookmaker", "market", "side", "line", "odds"]


def default_dir() -> Path:
    from src.book_quotes import QUOTES_DIR
    return QUOTES_DIR


def book_slug(name) -> str:
    return PREFIX + re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def physical_book(name: str) -> str:
    """The identity used to stop one bookmaker voting twice: OddsAPI key, or the af: slug."""
    n = str(name)
    if n.startswith(PREFIX):
        raw = n[len(PREFIX):].replace("_", " ")
        return SAME_BOOK.get(raw, n)
    return n


def source_of(name: str) -> str:
    return "apifootball" if str(name).startswith(PREFIX) else "oddsapi"


def load(now: pd.Timestamp, quotes_dir: Path | None = None) -> pd.DataFrame:
    """The committed parts that can hold a quote usable at `now` (day parts named YYYY-MM-DD.csv)."""
    d = Path(quotes_dir) if quotes_dir else default_dir()
    if not d.exists():
        return pd.DataFrame(columns=USECOLS)
    lo = (now - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    hi = now.strftime("%Y-%m-%d")
    frames = []
    for f in sorted(d.glob("*.csv")):
        if lo <= f.stem <= hi:
            try:
                frames.append(pd.read_csv(f, usecols=lambda c: c in USECOLS))
            except Exception as e:                                    # noqa: BLE001
                log.warning(f"[edge_af] unreadable {f.name} ({type(e).__name__})")
    frames = [x for x in frames if len(x)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=USECOLS)


def _family(q: pd.DataFrame) -> pd.Series:
    mk = q["market"].astype(str).str.lower()
    line = pd.to_numeric(q["line"], errors="coerce")
    fam = pd.Series(None, index=q.index, dtype=object)
    for (m, ln), f in LINE_FAMILY.items():
        fam[(mk == m) & np.isclose(line.fillna(-9.0), ln)] = f
    fam[(mk == "btts") & line.isna()] = "BTTS"
    return fam


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def match_fixtures(board: pd.DataFrame, fx: pd.DataFrame, stats: dict) -> dict:
    """{board (match_date, 'home vs away'): (af fixture_id, method)} for board rows that match
    exactly one API-Football fixture under the rules in the module docstring."""
    out: dict = {}
    if board is None or board.empty or fx is None or fx.empty:
        return out
    by_league: dict = {}
    for fid, lg, ko, h, a in zip(fx["fixture_id"], fx["league"].astype(str), _utc(fx["kickoff_utc"]),
                                 fx["home_team"].astype(str), fx["away_team"].astype(str)):
        by_league.setdefault(lg, []).append((fid, ko, h, a))
    names = {lg: sorted({x[2] for x in v} | {x[3] for x in v}) for lg, v in by_league.items()}
    for r in board.to_dict("records"):
        lg, home, away = str(r.get("league")), str(r.get("home_team")), str(r.get("away_team"))
        key = (str(r.get("date"))[:10], f"{home} vs {away}")
        ko = ME._utc(r.get("kickoff_utc"))
        g = by_league.get(lg)
        if g is None or ko is None:
            stats["unmatched_no_league_quotes"] += 1
            continue
        h, a = resolve(home, names[lg]), resolve(away, names[lg])
        if h is not None and h == a:
            stats["refused_ambiguous"] += 1
            continue
        if h is not None and a is not None:
            same = [x for x in g if x[2] == h and x[3] == a]
            if not same:
                stats["unmatched_not_captured"] += 1
                continue
            ids = {x[0] for x in same if pd.notna(x[1]) and abs(x[1] - ko) <= KICKOFF_TOL_BOTH}
            if len(ids) != 1:
                stats["refused_kickoff_mismatch" if not ids else "refused_ambiguous"] += 1
                continue
            out[key] = (ids.pop(), "BOTH_TEAMS")
            continue
        if h is None and a is None:
            stats["unmatched_names"] += 1
            continue
        ids = {x[0] for x in g if pd.notna(x[1]) and abs(x[1] - ko) <= KICKOFF_TOL_ONE
               and (x[2] == h if h is not None else x[3] == a)}
        if len(ids) != 1:
            stats["unmatched_names" if not ids else "refused_ambiguous"] += 1
            continue
        out[key] = (ids.pop(), "ONE_TEAM_KO")
    # one API-Football fixture can price ONE board fixture: a fixture claimed twice is refused for both
    claims = pd.Series([v[0] for v in out.values()], dtype=object).value_counts()
    dup = set(claims[claims > 1].index)
    if dup:
        stats["refused_ambiguous"] += sum(1 for v in out.values() if v[0] in dup)
        out = {k: v for k, v in out.items() if v[0] not in dup}
    return out


def _pairs_for(q: pd.DataFrame) -> list:
    """Per (fixture, book, family): the rows at that book's LATEST snapshot_ts. Both sides there ->
    one same-snapshot pair. One side only -> the newer row supersedes any older pair, so the book is
    one-sided (never carried forward). Returns [(fixture_id, family, BookQuote, kind)]."""
    out = []
    if q.empty:
        return out
    k = ["fixture_id", "bookmaker", "_fam"]
    q = q.sort_values("_ts", kind="mergesort").drop_duplicates(k + ["_ts", "_side"], keep="last")
    last = q[q["_ts"] == q.groupby(k)["_ts"].transform("max")]
    last = last.assign(_ab=np.where(last["_side"].isin(["OVER", "YES"]), "A", "B"))
    pv = last.pivot_table(index=k + ["_ts"], columns="_ab", values="odds", aggfunc="last")
    pv = pv.reindex(columns=["A", "B"])
    for (fid, bk, fam, t), oa, ob in zip(pv.index, pv["A"], pv["B"]):
        oa = None if pd.isna(oa) else float(oa)
        ob = None if pd.isna(ob) else float(ob)
        kind = "pair" if (oa is not None and ob is not None) else "one_sided"
        out.append((fid, fam, ME.BookQuote(book_slug(bk), oa, ob, t), kind))
    return out


def af_book_pairs(board: pd.DataFrame, now: pd.Timestamp, quotes_dir: Path | None = None,
                  raw: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """({(match_date, 'home vs away', family): [BookQuote]}, stats). Never raises."""
    t0 = time.perf_counter()
    stats = {k: 0 for k in ("files_rows", "board_fixtures", "matched_both_teams", "matched_one_team_ko",
                            "unmatched_no_league_quotes", "unmatched_not_captured", "unmatched_names",
                            "refused_ambiguous", "refused_kickoff_mismatch", "rows_post_kickoff_or_future",
                            "rows_other_line_or_market", "pairs_same_snapshot", "one_sided_superseded")}
    out: dict = {}
    try:
        q = load(now, quotes_dir) if raw is None else raw
        stats["files_rows"] = int(len(q))
        if board is None or board.empty or q.empty:
            return out, stats
        bk = pd.to_datetime(board["kickoff_utc"], errors="coerce", utc=True) \
            if "kickoff_utc" in board.columns else pd.Series(pd.NaT, index=board.index)
        board = board[bk > now]                                      # pre-kickoff fixtures only
        stats["board_fixtures"] = int(len(board))
        if board.empty:
            return out, stats
        fx = q.drop_duplicates("fixture_id", keep="last")[["fixture_id", "league", "kickoff_utc",
                                                           "home_team", "away_team"]]
        m = match_fixtures(board, fx, stats)
        stats["matched_both_teams"] = sum(1 for v in m.values() if v[1] == "BOTH_TEAMS")
        stats["matched_one_team_ko"] = sum(1 for v in m.values() if v[1] == "ONE_TEAM_KO")
        if not m:
            return out, stats
        board_ko = {}
        for r in board.to_dict("records"):
            k = (str(r.get("date"))[:10], f"{r.get('home_team')} vs {r.get('away_team')}")
            if k in m:
                board_ko[m[k][0]] = ME._utc(r.get("kickoff_utc"))
        q = q[q["fixture_id"].isin(list(board_ko))].copy()
        q["_fam"] = _family(q)
        n0 = len(q)
        q = q[q["_fam"].notna()]
        stats["rows_other_line_or_market"] = int(n0 - len(q))
        q["_ts"] = _utc(q["snapshot_ts"])
        q["_side"] = q["side"].astype(str).str.upper()
        q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
        bko = q["fixture_id"].map(board_ko)
        ako = _utc(q["kickoff_utc"])
        mtk = pd.to_numeric(q["minutes_to_kickoff"], errors="coerce")
        ok = (q["_ts"].notna() & (q["_ts"] <= now) & (q["_ts"] < bko) & (q["_ts"] < ako) & (mtk > 0))
        stats["rows_post_kickoff_or_future"] = int((~ok).sum())
        q = q[ok]
        ou = q["_fam"].str.startswith("OU")
        q = q[(ou & q["_side"].isin(["OVER", "UNDER"])) | (~ou & q["_side"].isin(["YES", "NO"]))]
        inv = {fid: k for k, (fid, _) in m.items()}
        for fid, fam, bq, kind in _pairs_for(q):
            md, match = inv[fid]
            out.setdefault((md, match, fam), []).append(bq)
            stats["pairs_same_snapshot" if kind == "pair" else "one_sided_superseded"] += 1
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"[edge_af] API-Football quotes skipped ({type(e).__name__}: {e})")
        stats["error"] = f"{type(e).__name__}: {e}"
    finally:
        stats["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return out, stats


def _two_sided(bq) -> bool:
    return bq.odds_a is not None and bq.odds_b is not None


def merge_pairs(oddsapi: dict, af: dict) -> dict:
    """OddsAPI pairs + API-Football pairs, one vote per physical book (the fresher two-sided quote
    wins; OddsAPI on a tie or when neither side has a two-sided quote)."""
    out = {k: list(v) for k, v in (oddsapi or {}).items()}
    for k, quotes in (af or {}).items():
        cur = out.setdefault(k, [])
        idx = {physical_book(x.book): i for i, x in enumerate(cur)}
        for bq in quotes:
            pb = physical_book(bq.book)
            if pb not in idx:
                idx[pb] = len(cur)
                cur.append(bq)
                continue
            old = cur[idx[pb]]
            if _two_sided(bq) and (not _two_sided(old) or
                                   (ME._utc(bq.ts) is not None and ME._utc(old.ts) is not None
                                    and ME._utc(bq.ts) > ME._utc(old.ts))):
                cur[idx[pb]] = bq
    return out
