"""
BTTS odds — the whole board, via OddsAPI's per-event endpoint.
=============================================================
BTTS produced ZERO tips in every league for months. Not a threshold problem and not a model
problem: `p_btts` was populated on all 147 board fixtures while `odds_btts` was populated on
**none**, so there was never a price to compute an edge against.

WHY IT WAS NEVER FETCHED. BTTS is not available on OddsAPI's bulk `/odds` endpoint — asking for
`markets=totals,btts` returns HTTP 422 and kills the ENTIRE call, which is what silently wiped
every tip in v9.2 (2026-06-22). The existing comment in predict.py records that and says BTTS
"needs the per-event endpoint"; nobody built it.

WHY THE API-FOOTBALL ARCHIVES ARE NOT THE ANSWER. They do hold BTTS, and most of it looks sound
(10,049 rows, common values 1.62-1.97 = implied 51-62%). But joining them to a live board returned
53 of 53 fixtures above the SNIPER threshold at a median +28% edge with **zero** negative edges —
the low tail of a mixed-quality archive, selected precisely because bad rows produce the biggest
apparent edges. A market that never disagrees with you is not a market.

OddsAPI per-event BTTS is clean by comparison. Birmingham City v Bristol City returned Yes
1.73-1.85 / No 1.97-2.14 across four books — implied ~54-58%, exactly where real BTTS sits.

COST, stated plainly because the naive version is genuinely expensive:

    1 credit per event. 147 board fixtures x 26 predict runs = ~3,822/day = 115k/month,
    against a 100,000 allowance. That does not fit.

    This module instead refetches a fixture every REFRESH_HOURS (default 2), and on EVERY run once
    it is inside NEAR_HOURS (default 6) of kickoff:

        147 fixtures / 2h  ->  ~1,760/day
        plus ~12 imminent x 26 runs -> ~310/day
        ~2,100/day, ~63k/month, inside the allowance with headroom

    That buys the whole board AND a genuine open->moving->closing curve where the line actually
    moves. Set BTTS_REFRESH_HOURS=0 to fetch everything every run if the allowance ever permits.

PERSISTENCE WITHOUT A NEW FILE. Quotes are written to, and read back from,
`output/book_odds_snapshots.csv` — which predict already writes, already commits, and which already
carries kickoff_utc. So the refresh clock survives across runs for free and BTTS lands in the same
per-book archive as the O/U prices, rather than in a parallel store that could drift out of step.

CONSENSUS, NOT BEST PRICE, feeds `odds_btts`. The tier is decided on what the market BELIEVES, so
the median across books is the right input; the best executable price belongs in the EV. Mixing
them is how one book's outlier becomes a fake edge.
"""
from __future__ import annotations

import logging
import os
import statistics
from datetime import datetime, timezone

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

_URL = "https://api.the-odds-api.com/v4/sports/{sport}/events/{eid}/odds"


def _cfg(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _last_seen(path) -> dict[str, pd.Timestamp]:
    """Newest BTTS snapshot per fixture, from the committed archive. The cross-run clock."""
    if not path.exists():
        return {}
    try:
        prev = pd.read_csv(path, usecols=lambda c: c in ("snapshot_ts", "match", "market"))
        prev = prev[prev["market"].astype(str).str.upper() == "BTTS"]
        if prev.empty:
            return {}
        ts = pd.to_datetime(prev["snapshot_ts"], errors="coerce", utc=True)
        return (prev.assign(_t=ts).dropna(subset=["_t"])
                    .groupby("match")["_t"].max().to_dict())
    except Exception:
        return {}


def append_book_rows(new_rows: list[dict]) -> None:
    """Append per-book quotes to book_odds_snapshots.csv, keeping consecutive-distinct only.

    Price-changes only, so a refetch that finds an unmoved market writes nothing and the file
    grows with real movement rather than with polling.
    """
    if not new_rows:
        return
    path = config.OUTPUT_DIR / "book_odds_snapshots.csv"
    df = pd.DataFrame(new_rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    key = ["match_date", "match", "market", "side", "bookmaker"]
    for c in key + ["odds", "snapshot_ts"]:
        if c not in df.columns:
            return
    df = df.sort_values(key + ["snapshot_ts"])
    prev = df.groupby(key)["odds"].shift()
    df = df[df["odds"].ne(prev)].sort_values("snapshot_ts")
    df.to_csv(path, index=False)


def fetch(rows: list[dict]) -> list[dict]:
    """Fill `odds_btts` across the board. Returns rows unchanged on any failure.

    A missing research price must never cost a tip, so every failure path is a pass-through.
    """
    if not rows or not getattr(config, "ODDS_API_KEY", ""):
        return rows
    refresh_h = _cfg("BTTS_REFRESH_HOURS", 2.0)
    near_h = _cfg("BTTS_NEAR_HOURS", 6.0)
    max_calls = int(_cfg("BTTS_MAX_CALLS", 200))
    regions = os.getenv("BTTS_REGIONS", "uk,eu")

    now = pd.Timestamp(datetime.now(timezone.utc))
    snap_path = config.OUTPUT_DIR / "book_odds_snapshots.csv"
    seen = _last_seen(snap_path)
    snap_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    out, calls, filled, book_rows = [], 0, 0, []
    n_fresh = n_nokey = n_err = 0
    for r in rows:
        match = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
        # hours to kickoff — unparseable means "treat as far out", never skip silently
        try:
            ko = pd.Timestamp(r.get("kickoff_utc"))
            ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
            lead_h = (ko - now).total_seconds() / 3600.0
        except Exception:
            lead_h = None

        # Inside NEAR_HOURS: refetch every run, because that is where the line moves and the
        # closing price cannot be recovered later. Otherwise honour the refresh clock.
        near = lead_h is not None and 0 <= lead_h <= near_h
        prior = seen.get(match)
        if not near and prior is not None and refresh_h > 0:
            if (now - prior).total_seconds() / 3600.0 < refresh_h:
                n_fresh += 1
                out.append(r)
                continue
        eid, skey = r.get("_oa_event_id"), r.get("_oa_sport_key")
        if not eid or not skey:
            n_nokey += 1
            out.append(r)
            continue
        if calls >= max_calls:
            out.append(r)
            continue
        try:
            resp = requests.get(_URL.format(sport=skey, eid=eid),
                                params={"apiKey": config.ODDS_API_KEY, "regions": regions,
                                        "markets": "btts", "oddsFormat": "decimal"},
                                timeout=15)
            calls += 1
            if resp.status_code != 200:
                n_err += 1
                out.append(r)
                continue
            data = resp.json()
        except Exception:
            n_err += 1
            out.append(r)
            continue

        yes, no = [], []
        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "btts":
                    continue
                for oc in mkt.get("outcomes", []):
                    nm = str(oc.get("name", "")).upper()
                    pr = oc.get("price")
                    if not pr or float(pr) <= 1.0:
                        continue
                    if nm == "YES":
                        yes.append(float(pr))
                    elif nm == "NO":
                        no.append(float(pr))
                    book_rows.append({
                        "snapshot_ts": snap_ts, "league": r.get("league", ""),
                        "match_date": str(r.get("date", ""))[:10],
                        "kickoff_utc": str(r.get("kickoff_utc", "")), "match": match,
                        "bookmaker": bm.get("key", ""), "market": "BTTS",
                        "side": nm, "odds": float(pr)})

        # BOTH sides required. A one-sided quote carries an unknown margin, so using it would bias
        # the price by whatever that book charges — and a coherence check needs the pair.
        if yes and no:
            r = dict(r)
            r["odds_btts"] = float(statistics.median(yes))
            r["odds_btts_no"] = float(statistics.median(no))
            r["btts_books"] = len(yes)
            # de-vigged P(BTTS yes), recorded so a later reader never has to re-derive it
            iy, ino = 1.0 / r["odds_btts"], 1.0 / r["odds_btts_no"]
            r["btts_overround"] = round(iy + ino - 1.0, 4)
            r["btts_fair_yes"] = round(iy / (iy + ino), 4)
            filled += 1
        out.append(r)

    try:
        append_book_rows(book_rows)
    except Exception as e:
        log.warning(f"[btts] archive append failed: {e}")
    log.info(f"[btts] {calls} credit(s) spent, filled {filled}/{len(rows)} fixtures "
             f"(skipped {n_fresh} within {refresh_h:g}h refresh, {n_nokey} no event id, "
             f"{n_err} errors; near-kickoff window {near_h:g}h)")
    return out
