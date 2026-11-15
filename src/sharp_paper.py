"""SF-PAPER-1: the sharp-agreement PAPER track (owner 2026-11-15, "Paper track + capture").

    P = load_protocol()                                   # config/sharp_paper/protocol.json (hash-registered)
    picks, n = tip_universe(bets_ledger, side_ledger, P)  # live tips of the approved O/U MODEL cells
    rows = parse_event_odds(body, captured_at, ko)        # one book's exact-line O/U pairs of one snapshot
    paper = build_paper(picks, capture, P)                # sharp class at T-6 h, bet price, CLV, 1u P/L
    stats = summarize(paper, P)                           # counts and statistics only

WHAT IT IS. For every live model tip of an approved O/U MODEL cell (TP-2026-11-15-bade2c0b) from 2026-11-21,
the sharp-money class read about 6 h before kickoff (research sharp-filter/1, W72_6: the fair-probability move
of Pinnacle and of the other books since about 72 h), a paper bet at Pinnacle's exact-line price in that same
T-6 h snapshot, flat 1 unit, settled with the ledger's own result, and the CLV against the captured close.

PAPER ONLY. Nothing here stakes, places, publishes or writes into v9's ledgers: every writer refuses a path
inside this public repository (assert_private), and no production module imports this one
(tests/test_sharp_paper.py). The OddsAPI prices it reads and writes are licensed (no redistribution), so
they live only in the PRIVATE store (wowza-exec branch sharp-paper); summarize() returns counts and
statistics only.

NO DEFAULTS. A missing pair, too few books, a league without a learned threshold, an unmatched event or a
missed slot is recorded as missing (NO_DATA / NO_PRICE / counts), never filled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = ROOT / "config" / "sharp_paper" / "protocol.json"
TRACK = "PAPER"
STAKE_UNITS = 1
SHARP_BOOK = "pinnacle"
MIN_PRICE = 1.01
MSIDE_LINE = {"over25_over": 2.5, "over25_under": 2.5, "over15_over": 1.5, "over35_over": 3.5}
MSIDE_SIGN = {"over25_over": 1, "over25_under": -1, "over15_over": 1, "over35_over": 1}
CLASS = {1.0: "AGREE", -1.0: "DISAGREE", 0.0: "NONE"}


class NotPrivateError(RuntimeError):
    """A write of licensed prices or paper rows was aimed inside the public repository."""


def load_protocol(path: Path = PROTOCOL_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_sha(path: Path = PROTOCOL_FILE) -> str:
    """sha256 of the committed bytes (LF line ends; a Windows CRLF checkout hashes the same)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def assert_private(path) -> Path:
    p = Path(path).resolve()
    if p == ROOT or ROOT in p.parents:
        raise NotPrivateError(f"{p} is inside the public repository: licensed prices and paper rows stay private")
    return p


# ── universe ─────────────────────────────────────────────────────────────────────────────────

def _cells(P: dict) -> dict:
    return {(c["mside"], c["league"]): c for c in P["universe"]["cells"]}


def tip_universe(bets: pd.DataFrame | None, side: pd.DataFrame | None, P: dict, *, test: bool = False,
                 today: str | None = None) -> tuple[pd.DataFrame, dict]:
    """The paper universe: live tips of the protocol's cells. test=True (the capped capture test only) drops
    the tier and start-date rules and keeps upcoming fixtures (match_date >= today) of the cells' leagues."""
    U = P["universe"]
    cells = _cells(P)
    parts, n = [], {}
    if bets is not None and len(bets):
        b = bets.copy()
        b["mside"] = b["side"].astype(str).str.upper().map({"OVER": "over25_over", "UNDER": "over25_under"})
        parts.append(b.assign(ledger="bets_ledger"))
    if side is not None and len(side):
        s = side.copy()
        ok = s["side"].astype(str).str.upper().isin(["OVER", "YES"])
        s["mside"] = np.where(ok, s["market"].astype(str).str.lower().map({"over15": "over15_over",
                                                                            "over35": "over35_over"}), None)
        parts.append(s.assign(ledger="side_bets_ledger"))
    if not parts:
        return pd.DataFrame(columns=PICK_COLS), n
    d = pd.concat(parts, ignore_index=True)
    n["ledger_rows"] = int(len(d))
    d = d[d["source"].astype(str) == "live"]
    d = d[d["mside"].notna()]
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d = d[d["match_date"].notna()]
    if test:
        d = d[d["match_date"] >= (today or pd.Timestamp.utcnow().strftime("%Y-%m-%d"))]
    else:
        d = d[d["match_date"] >= P["start_match_date"]]
        d = d[d["signal_tier"].astype(str).isin(U["tip_tiers"])]
    n["live_tips_in_scope"] = int(len(d))
    incell = [(m, lg) in cells for m, lg in zip(d["mside"], d["league"])]
    d = d[np.array(incell, dtype=bool)]
    n["tips_in_cells"] = int(len(d))
    d = d.sort_values("generated_at").drop_duplicates(["match_date", "league", "home_team", "away_team", "mside"])
    out = pd.DataFrame({
        "pick_id": d["match_date"] + "|" + d["league"] + "|" + d["home_team"] + "|" + d["away_team"] + "|" + d["mside"],
        "match_date": d["match_date"], "league": d["league"], "home": d["home_team"], "away": d["away_team"],
        "mside": d["mside"], "track": [cells[(m, lg)]["track"] for m, lg in zip(d["mside"], d["league"])],
        "line": d["mside"].map(MSIDE_LINE), "side_sign": d["mside"].map(MSIDE_SIGN),
        "generated_at": d["generated_at"], "result": d["result"].astype(str), "ledger": d["ledger"]})
    n["picks"] = int(len(out))
    return out.reset_index(drop=True), n


PICK_COLS = ["pick_id", "match_date", "league", "home", "away", "mside", "track", "line", "side_sign",
             "generated_at", "result", "ledger"]


# ── provider events ──────────────────────────────────────────────────────────────────────────

def match_events(fixtures: pd.DataFrame, events: list[dict], aliases: dict | None = None) -> tuple[dict, dict]:
    """{(match_date, league, home, away): event} for ONE league's fixtures and that league's provider events
    (its own sport keys only). An event's clubs must resolve onto the fixture's clubs (src.team_names.resolve,
    or the store's proven provider alias), its kickoff inside [match_date - 12 h, match_date + 36 h), one to
    one; an ambiguous pair is refused."""
    from src.team_names import resolve
    aliases = aliases or {}
    out, used, st = {}, set(), {"fixtures": 0, "matched": 0, "unmatched": 0, "ambiguous": 0}
    for f in fixtures.drop_duplicates(["match_date", "league", "home", "away"]).itertuples():
        st["fixtures"] += 1
        d0 = pd.Timestamp(f.match_date, tz="UTC")
        hits = []
        for e in events:
            ko = pd.to_datetime(e.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(ko) or not (d0 - pd.Timedelta(hours=12) <= ko < d0 + pd.Timedelta(hours=36)):
                continue
            h = aliases.get(e.get("home_team")) or resolve(e.get("home_team"), [f.home])
            a = aliases.get(e.get("away_team")) or resolve(e.get("away_team"), [f.away])
            if h == f.home and a == f.away:
                hits.append(e)
        if len(hits) == 1 and hits[0]["id"] not in used:
            out[(f.match_date, f.league, f.home, f.away)] = hits[0]
            used.add(hits[0]["id"])
            st["matched"] += 1
        elif len(hits) > 1:
            st["ambiguous"] += 1
        else:
            st["unmatched"] += 1
    return out, st


def lead_hours(ko, at) -> float:
    return (pd.Timestamp(ko).tz_convert("UTC") - pd.Timestamp(at).tz_convert("UTC")).total_seconds() / 3600.0


def due_slots(ko, now, done: set, P: dict, test: bool = False) -> list[str]:
    """Slots whose lead window contains `now` and that the journal has not answered yet. test=True: one TEST
    slot at any pre-kickoff lead (the capped capture test only)."""
    ld = lead_hours(ko, now)
    if ld <= 0:
        return []
    if test:
        return [] if "TEST" in done else ["TEST"]
    return [s for s, w in P["capture"]["slots"].items() if w["lead_h"][0] <= ld <= w["lead_h"][1] and s not in done]


def parse_event_odds(body: dict, captured_at, lines=(1.5, 2.5, 3.5)) -> tuple[list[dict], dict]:
    """One provider answer -> one row per (book, exact line) with BOTH Over and Under real prices
    (> MIN_PRICE), captured before the kickoff the answer states. A book quoting two prices for one outcome
    and line is ambiguous: dropped and counted."""
    st = {"books": 0, "rows": 0, "one_sided": 0, "ambiguous": 0, "post_kickoff": 0, "low_price": 0}
    ko = pd.to_datetime((body or {}).get("commence_time"), utc=True, errors="coerce")
    at = pd.Timestamp(captured_at).tz_convert("UTC")
    if pd.isna(ko) or at >= ko:
        st["post_kickoff"] = 1
        return [], st
    rows = []
    for bm in (body or {}).get("bookmakers", []) or []:
        st["books"] += 1
        q: dict = {}
        for mk in bm.get("markets", []) or []:
            if mk.get("key") not in ("totals", "alternate_totals"):
                continue
            for o in mk.get("outcomes", []) or []:
                pt, nm, pr = o.get("point"), o.get("name"), o.get("price")
                if pt is None or nm not in ("Over", "Under") or pr is None:
                    continue
                pt = float(pt)
                if not any(abs(pt - l) < 1e-9 for l in lines):
                    continue
                pr = float(pr)
                if pr <= MIN_PRICE:
                    st["low_price"] += 1
                    continue
                q.setdefault((pt, nm), set()).add(pr)
        for pt in sorted({k[0] for k in q}):
            o, u = q.get((pt, "Over")), q.get((pt, "Under"))
            if not o or not u:
                st["one_sided"] += 1
                continue
            if len(o) > 1 or len(u) > 1:
                st["ambiguous"] += 1
                continue
            rows.append({"bookmaker": bm.get("key"), "line": pt, "over": next(iter(o)), "under": next(iter(u)),
                         "book_last_update": bm.get("last_update")})
            st["rows"] += 1
    return rows, st


# ── signal ───────────────────────────────────────────────────────────────────────────────────

def fair_over(over, under) -> float | None:
    from src.market_edge import power_devig
    p = power_devig(float(over), float(under))
    return None if p is None else float(p)


def moves(open_rows: pd.DataFrame, sig_rows: pd.DataFrame, P: dict) -> dict:
    """PIN and CONS move of the fair Over probability on the exact 2.5 line, T72 -> T6."""
    S = P["signal"]
    out = {"pin": None, "cons": None, "cons_books": 0}
    a = open_rows[open_rows["line"] == 2.5]
    b = sig_rows[sig_rows["line"] == 2.5]
    if a.empty or b.empty:
        return out
    m = a.merge(b, on="bookmaker", suffixes=("_o", "_s"))
    if m.empty:
        return out
    m["d"] = [None if (fo := fair_over(o1, u1)) is None or (fs := fair_over(o2, u2)) is None else fs - fo
              for o1, u1, o2, u2 in zip(m["over_o"], m["under_o"], m["over_s"], m["under_s"])]
    m = m[m["d"].notna()]
    pin = m[m["bookmaker"] == SHARP_BOOK]
    if len(pin):
        out["pin"] = float(pin["d"].iloc[0])
    others = m[m["bookmaker"] != SHARP_BOOK]
    out["cons_books"] = int(len(others))
    if len(others) >= 3:
        out["cons"] = float(np.median(others["d"].astype(float)))
    return out


def _dir(v, thr) -> float | None:
    if v is None or thr is None:
        return None
    return 1.0 if v > thr else (-1.0 if v < -thr else 0.0)


def classify(mv: dict, league: str, side_sign: int, P: dict) -> dict:
    """{PIN, CONS, BOTH: AGREE / DISAGREE / NONE / MIXED / NO_DATA} for a pick on the side with sign
    side_sign (+1 Over, -1 Under), against the LEAGUE's own frozen thresholds."""
    th = P["signal"]["league_thresholds"].get(league) or {}
    pd_ = _dir(mv.get("pin"), (th.get("pin") or {}).get("q75"))
    cd_ = _dir(mv.get("cons"), (th.get("cons") or {}).get("q75"))
    pd_ = None if pd_ is None else pd_ * side_sign + 0.0
    cd_ = None if cd_ is None else cd_ * side_sign + 0.0
    pin = CLASS[pd_] if pd_ is not None else "NO_DATA"
    cons = CLASS[cd_] if cd_ is not None else "NO_DATA"
    if pd_ is None or cd_ is None:
        both = "NO_DATA"
    else:
        both = pin if pd_ == cd_ else "MIXED"
    return {"PIN": pin, "CONS": cons, "BOTH": both}


# ── the paper ledger ─────────────────────────────────────────────────────────────────────────

PAPER_COLS = ["pick_id", "match_date", "league", "track", "mside", "line", "side_sign", "cls_CONS", "cls_PIN",
              "cls_BOTH", "cons_books", "status", "bet_odds", "close_odds", "clv", "won", "pnl", "stake_units",
              "track_kind", "signal_lead_h", "open_lead_h", "result"]


def build_paper(picks: pd.DataFrame, cap: pd.DataFrame, P: dict) -> pd.DataFrame:
    """cap: captured rows (pick_key = match_date|league|home|away, slot, captured_at, ko, bookmaker, line,
    over, under). One paper row per pick: classes, the Pinnacle exact-line bet price at T6 (NO_PRICE when
    absent), CLV vs the CLOSE slot, 1-unit P/L from the ledger's own WIN / LOSS (VOID / unsettled: no P/L)."""
    rows = []
    cap = cap if cap is not None else pd.DataFrame(columns=["pick_key", "slot"])
    for p in picks.itertuples():
        key = f"{p.match_date}|{p.league}|{p.home}|{p.away}"
        c = cap[cap["pick_key"] == key]
        o, s, cl = (c[c["slot"] == x] for x in ("T72", "T6", "CLOSE"))
        mv = moves(o, s, P) if len(o) and len(s) else {"pin": None, "cons": None, "cons_books": 0}
        cls = classify(mv, p.league, int(p.side_sign), P)
        r = {"pick_id": p.pick_id, "match_date": p.match_date, "league": p.league, "track": p.track,
             "mside": p.mside, "line": p.line, "side_sign": p.side_sign, "cls_CONS": cls["CONS"],
             "cls_PIN": cls["PIN"], "cls_BOTH": cls["BOTH"], "cons_books": mv["cons_books"],
             "stake_units": STAKE_UNITS, "track_kind": TRACK, "result": p.result,
             "signal_lead_h": float(s["lead_h"].iloc[0]) if len(s) and "lead_h" in s else None,
             "open_lead_h": float(o["lead_h"].iloc[0]) if len(o) and "lead_h" in o else None}
        pin_s = s[(s["bookmaker"] == SHARP_BOOK) & (s["line"] == p.line)] if len(s) else s
        if len(s) == 0:
            r["status"] = "NO_SIGNAL_SNAPSHOT"
        elif pin_s.empty:
            r["status"] = "NO_PRICE"
        else:
            r["bet_odds"] = float(pin_s["over"].iloc[0] if p.side_sign == 1 else pin_s["under"].iloc[0])
            pin_c = cl[(cl["bookmaker"] == SHARP_BOOK) & (cl["line"] == p.line)] if len(cl) else cl
            if len(pin_c):
                r["close_odds"] = float(pin_c["over"].iloc[0] if p.side_sign == 1 else pin_c["under"].iloc[0])
                r["clv"] = r["bet_odds"] / r["close_odds"] - 1.0
            if p.result in ("WIN", "LOSS"):
                r["won"] = p.result == "WIN"
                r["pnl"] = (r["bet_odds"] - 1.0) * STAKE_UNITS if r["won"] else -1.0 * STAKE_UNITS
                r["status"] = "SETTLED"
            else:
                r["status"] = "OPEN_OR_VOID"
        rows.append(r)
    return pd.DataFrame(rows, columns=PAPER_COLS)


def summarize(paper: pd.DataFrame, P: dict) -> dict:
    """Counts and statistics only (no price, no per-row value): per cell x class of the primary and
    secondary signals: settled bets, ROI, win rate minus implied, mean CLV; status counts."""
    out = {"protocol_id": P["protocol_id"], "track": TRACK, "status_counts": paper["status"].value_counts().to_dict()
           if len(paper) else {}, "cells": []}
    s = paper[paper["status"] == "SETTLED"]
    for sig in [P["signal"]["primary"]] + list(P["signal"]["secondary"]):
        for (lg, ms, cl), g in s.groupby(["league", "mside", f"cls_{sig}"]):
            out["cells"].append({"signal": sig, "league": lg, "mside": ms, "cls": cl, "bets": int(len(g)),
                                 "roi": round(float(g["pnl"].mean()), 4),
                                 "win_minus_implied": round(float(g["won"].mean() - (1 / g["bet_odds"]).mean()), 4),
                                 "clv": None if g["clv"].isna().all() else round(float(g["clv"].mean()), 4),
                                 "clv_n": int(g["clv"].notna().sum())})
    out["review_due"] = sorted({f"{c['league']}|{c['mside']}" for c in out["cells"]
                                if c["signal"] == P["signal"]["primary"] and c["cls"] == "AGREE" and c["bets"] >= 150})
    return out
