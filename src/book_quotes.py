"""§10 — per-bookmaker quotes on a kickoff-relative ladder.

    from src.book_quotes import parse_books, ladder_band, append_quotes

WHAT IS WRONG TODAY. The side-market capture asks API-Football for ONE bookmaker
(`params={"fixture": fid, "bookmaker": 8}`) and stores one price per market. Everything needed
for market microstructure — who moved first, how wide the books are, what the best executable
price was — is discarded at the request, before it is ever seen.

Dropping that filter costs **the same number of API calls**. The response simply carries every
book instead of one. The only extra cost is parsing and disk.

AND THE CLOSE IS NOT THE LAST SNAPSHOT. CLV is currently computed against whatever row happened
to be written last, which may be six hours before kickoff. A price six hours out is not a
closing line, and a "CLV" measured against it is measuring something else. `ladder_band` tags
every quote by how long before kickoff it was taken, so a close can be defined as the last quote
inside T-30m and a row that never got near kickoff can be excluded rather than silently treated
as one.

THE LADDER. T-6h, T-3h, T-1h, T-30m, T-10m, as §10 specifies, plus FAR for anything earlier and
POST for anything after kickoff (which must never be used as a close — a price after kick-off is
an in-play price, and treating one as a closing line manufactures enormous fake CLV).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

QUOTES_FILE = config.OUTPUT_DIR / "book_quotes.csv"

#: §10's required fields. `line` is separate from `market` so Over 2.5 and Over 3.5 are the same
#: market at different lines rather than two unrelated strings — which is what lets a future
#: study ask about the line itself.
COLUMNS = [
    "snapshot_ts", "fixture_id", "kickoff_utc", "minutes_to_kickoff", "ladder_band",
    "league", "model_type", "home_team", "away_team",
    "bookmaker_id", "bookmaker", "market", "side", "line", "odds", "source",
]

#: Upper edge of each band, in minutes before kickoff. Ordered tightest-first.
LADDER = [("T-10m", 10), ("T-30m", 30), ("T-1h", 60), ("T-3h", 180), ("T-6h", 360)]

#: A quote inside this window is eligible to be the closing line.
CLOSING_WINDOW_MIN = 30


def ladder_band(minutes_to_kickoff: float | None) -> str:
    """Which rung of the ladder a quote sits on.

    POST is deliberately its own band and never folded into T-10m: a price taken after kickoff
    is an in-play price. Treating one as a close would produce large positive CLV out of nothing,
    which is the most flattering possible error and therefore the one most likely to be believed.
    """
    if minutes_to_kickoff is None or (isinstance(minutes_to_kickoff, float)
                                      and np.isnan(minutes_to_kickoff)):
        return "UNKNOWN"
    if minutes_to_kickoff < 0:
        return "POST"
    for name, edge in LADDER:
        if minutes_to_kickoff <= edge:
            return name
    return "FAR"


def minutes_to_kickoff(kickoff_utc, snapshot_ts) -> float | None:
    try:
        ko = pd.Timestamp(kickoff_utc)
        ts = pd.Timestamp(snapshot_ts)
        if ko.tzinfo is None:
            ko = ko.tz_localize("UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return float((ko - ts).total_seconds() / 60.0)
    except Exception:                                                 # noqa: BLE001
        return None


# ── parsing ───────────────────────────────────────────────────────────────────────────────────
#: BET IDS, NOT NAMES. API-Football returns several bets whose NAME contains "over/under" and
#: which are not full-match goals:
#:
#:     id 5    Goals Over/Under                  <- the one we want
#:     id 6    Goals Over/Under First Half       <- first-half goals
#:     id 26   Goals Over/Under - Second Half    <- NOT full match
#:     id 57   Home Corners Over/Under           <- not goals at all
#:     id 58   Away Corners Over/Under
#:     id 197  Over/Under 15m-30m                <- a time window
#:     id 198  Over/Under 30m-45m
#:
#: A first version matched on the name containing "over/under" and swallowed all seven. On a
#: live fixture that produced "O/U 2.5" quotes ranging 1.25-3.50 and a cross-book anchor of
#: p=0.254 where the real market was ~0.63. The capture script already learned this the hard
#: way -- 21.9% of its archive once had over25 >= over35, which is impossible -- and its own
#: comment says the numeric id is the primary test "because it cannot be broken by a rename".
BET_IDS = {5: "ou", 6: "ht_ou", 8: "btts", 1: "h2h"}


def _classify(bet_id, bet_name: str, value: str) -> tuple[str, str, float | None] | None:
    """(market, side, line) for a quote, or None when the bet is not one we model.

    Keyed on the numeric bet id. An unrecognised id is dropped silently and deliberately: a
    market we cannot name confidently is worse than a market we do not store.
    """
    market = BET_IDS.get(bet_id)
    if market is None:
        return None
    v = (value or "").strip()
    vl = v.lower()

    if market == "btts":
        return ("btts", vl, None) if vl in ("yes", "no") else None

    if market == "h2h":
        m = {"home": "home", "draw": "draw", "away": "away"}.get(vl)
        return ("h2h", m, None) if m else None

    # ou / ht_ou -> "Over 2.5" / "Under 2.5"
    parts = v.split()
    if len(parts) != 2 or parts[0].lower() not in ("over", "under"):
        return None
    try:
        line = float(parts[1])
    except ValueError:
        return None
    return (market, parts[0].lower(), line)


def parse_books(payload: dict, fixture_id: int, kickoff_utc: str, league: str,
                home: str, away: str, snapshot_ts: str, model_type: str = "",
                source: str = "api_football") -> list[dict]:
    """Every bookmaker's quote for one fixture, flattened to one row per (book, market, side).

    Consensus is NOT computed here and nothing is discarded. The whole point is to keep what the
    current capture throws away; a consensus can always be derived later, while a book that was
    never stored is gone.
    """
    rows: list[dict] = []
    mins = minutes_to_kickoff(kickoff_utc, snapshot_ts)
    band = ladder_band(mins)
    for entry in (payload or {}).get("response", []) or []:
        for bk in entry.get("bookmakers", []) or []:
            bid, bname = bk.get("id"), (bk.get("name") or "").strip()
            for bet in bk.get("bets", []) or []:
                for val in bet.get("values", []) or []:
                    cls = _classify(bet.get("id"), bet.get("name", ""),
                                    str(val.get("value", "")))
                    if cls is None:
                        continue
                    try:
                        odds = float(val.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if odds <= 1.0:
                        continue
                    market, side, line = cls
                    rows.append({
                        "snapshot_ts": snapshot_ts, "fixture_id": fixture_id,
                        "kickoff_utc": kickoff_utc,
                        "minutes_to_kickoff": round(mins, 1) if mins is not None else None,
                        "ladder_band": band, "league": league, "model_type": model_type,
                        "home_team": home, "away_team": away,
                        "bookmaker_id": bid, "bookmaker": bname,
                        "market": market, "side": side, "line": line,
                        "odds": odds, "source": source,
                    })
    return rows


def append_quotes(rows: list[dict], path: Path | None = None) -> int:
    """Append, keeping only DISTINCT consecutive prices per (fixture, book, market, side, line).

    Same discipline as the existing archives: a price that has not changed is not new
    information, and storing it every run would make the file enormous for nothing. What it
    costs is that row spacing is not evidence of sampling frequency — a fact already recorded
    against this estate's NEAR-loop analysis, and the reason `ladder_band` is stored explicitly
    rather than inferred from gaps.
    """
    if not rows:
        return 0
    p = path or QUOTES_FILE
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]

    key = ["fixture_id", "bookmaker_id", "market", "side", "line"]

    def _k(frame):
        """Dedup key as a single string.

        `line` is NaN for BTTS and 1X2, and NaN NEVER EQUALS ITSELF — so a tuple key containing
        one can never match its own earlier row, and every unchanged BTTS price would be written
        again on every run. Found by the test that asserts a re-append writes nothing: it let 2
        of 8 rows through, and those 2 were exactly the line-less markets.
        """
        return (frame["fixture_id"].astype(str) + "|"
                + frame["bookmaker_id"].astype(str) + "|"
                + frame["market"].astype(str) + "|"
                + frame["side"].astype(str) + "|"
                + frame["line"].fillna(-1).astype(str))

    if p.exists():
        old = pd.read_csv(p)
        if len(old):
            o = old.sort_values("snapshot_ts").copy()
            o["_k"] = _k(o)
            last = o.groupby("_k")["odds"].last()
        else:
            last = pd.Series(dtype=float)
        df = df.assign(_k=_k(df))
        prev = df["_k"].map(last)
        df = df[prev.isna() | ~np.isclose(prev.fillna(-1).astype(float),
                                          df["odds"].astype(float))].drop(columns="_k")
        if df.empty:
            return 0
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return int(len(df))


def closing_quote(df: pd.DataFrame, fixture_id, market: str, side: str,
                  line: float | None = None) -> dict | None:
    """The LAST pre-kickoff quote inside the closing window, or None.

    Returns None rather than falling back to an earlier price. A six-hour-old quote is not a
    close, and the honest answer when no close was captured is that there isn't one — which is
    what lets coverage be measured instead of assumed.
    """
    m = df[(df["fixture_id"] == fixture_id) & (df["market"] == market)
           & (df["side"] == side)]
    if line is not None:
        m = m[m["line"] == line]
    m = m[(m["minutes_to_kickoff"] >= 0) & (m["minutes_to_kickoff"] <= CLOSING_WINDOW_MIN)]
    if m.empty:
        return None
    r = m.sort_values("minutes_to_kickoff").iloc[0]      # smallest minutes = closest to kickoff
    return {"odds": float(r["odds"]), "bookmaker": r["bookmaker"],
            "minutes_to_kickoff": float(r["minutes_to_kickoff"]),
            "band": r["ladder_band"]}


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many fixture-markets reached each rung. The measurement §10 exists to enable."""
    if df.empty:
        return pd.DataFrame()
    g = (df.groupby("ladder_band")
           .agg(rows=("odds", "size"),
                fixtures=("fixture_id", "nunique"),
                books=("bookmaker_id", "nunique")).reset_index())
    order = {b: i for i, (b, _) in enumerate(LADDER)}
    order.update({"FAR": 98, "POST": 99, "UNKNOWN": 100})
    return g.sort_values("ladder_band", key=lambda s: s.map(order))
