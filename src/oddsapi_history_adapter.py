"""
Read side of the OddsAPI history (src/oddsapi_history.py): attach the REAL stored prices to our
prediction / bars universe rows by fixture id. Output stays LOCAL or PRIVATE (redistribution of the
provider's data is prohibited): the CLI refuses an output path inside this public repo.
=================================================================================================
Matching is the collector's own rule (src.oddsapi_history.match_events): within ONE league (only that
league's OddsAPI sport keys, never a cross-league join) and ONE match day, both clubs resolved by
src.team_names.resolve onto one fixture, one-to-one. Every fixture without a proven event and every
stored event without a fixture is returned in the unmatched report.

Nothing is filled: a market the provider did not price has no row; a summary is computed only from
real book prices (n_books, best, median) and is NaN where there is none.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import oddsapi_history as OH

FIXTURE_KEY = ["league", "date", "home", "away"]
ODDS_DEDUPE = ["unit_key", "event_id", "bookmaker", "market", "outcome_name", "outcome_description", "point"]
UNMATCHED_COLS = ["kind", "league", "date", "home", "away", "event_id", "commence_time"]
TOTAL_MARKETS = {"totals", "alternate_totals"}
TEAM_TOTAL_MARKETS = {"team_totals", "alternate_team_totals"}


def load_store(store_root: Path) -> pd.DataFrame:
    """Every stored odds row (deduplicated: a unit re-bought after a crash counts once)."""
    files = sorted((Path(store_root) / "data" / "odds").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=OH.ODDS_COLS)
    o = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return o.drop_duplicates(ODDS_DEDUPE).reset_index(drop=True)


def fixture_id(df: pd.DataFrame) -> pd.Series:
    """The universe's own fixture id: row_id when present, else 'date|league|home|away' (the same
    format the bars universe uses)."""
    if "row_id" in df.columns:
        return df["row_id"].astype(str)
    d = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return d + "|" + df["league"].astype(str) + "|" + df["home"].astype(str) + "|" + df["away"].astype(str)


def _line_name(p: float) -> str | None:
    """1.5 -> 'over15', 6.5 -> 'over65'; quarter / whole lines have no universe market name."""
    if p is None or not np.isfinite(p) or abs((p % 1) - 0.5) > 1e-9:
        return None
    return "over" + f"{p:.1f}".replace(".", "")


def classify(market: str, name, desc, point, home: str, away: str) -> dict | None:
    """One provider outcome -> our market vocabulary, or None when it cannot be placed exactly."""
    name = None if name is None else str(name)
    pt = None if point is None or (isinstance(point, float) and np.isnan(point)) else float(point)
    if market == "btts" and name in ("Yes", "No"):
        return {"market": "btts", "side": name.lower(), "line": None, "team": None, "player": None}
    if market in TOTAL_MARKETS and name in ("Over", "Under") and pt is not None:
        return {"market": _line_name(pt) or "total_line", "side": "yes" if name == "Over" else "no",
                "line": pt, "team": None, "player": None}
    if market in TEAM_TOTAL_MARKETS and name in ("Over", "Under") and pt is not None:
        team = "home" if desc == home else ("away" if desc == away else None)
        if team is None:
            return None
        return {"market": f"team_total_{team}", "side": "yes" if name == "Over" else "no", "line": pt,
                "team": team, "player": None}
    if market == "h2h":
        side = "home" if name == home else ("away" if name == away else ("draw" if name == "Draw" else None))
        return None if side is None else {"market": "1x2", "side": side, "line": None, "team": None, "player": None}
    if market.startswith("correct_score"):
        return None if not name else {"market": market, "side": name, "line": pt, "team": None, "player": None}
    if market.startswith("player_"):
        return None if not desc else {"market": market, "side": name, "line": pt, "team": None,
                                      "player": str(desc)}
    return None


def load_context(store_root: Path, universe: pd.DataFrame) -> dict:
    """The collector's own matching context, read from the PRIVATE store: every stored provider
    listing (events_by_day, keyed by league and match day: one league's sport key only), the proven
    OddsAPI aliases (ledger/provider_aliases.json) and v9's identity spellings the run used
    (ledger/identity_aliases.json)."""
    root = Path(store_root)
    st = OH.Store(root, "read-only")
    proven = {}
    f = root / "ledger" / "provider_aliases.json"
    if f.exists():
        proven = json.loads(f.read_text(encoding="utf-8")).get("map", {})
    identity = {}
    f = root / "ledger" / "identity_aliases.json"
    if f.exists():
        identity = json.loads(f.read_text(encoding="utf-8"))
    return {"events_by_day": st.events_by_day, "identity": identity, "proven": proven}


def _context_from_odds(odds: pd.DataFrame) -> dict:
    """Without a store: the events the odds rows themselves carry, keyed by their league-day."""
    ebd: dict = {}
    ev = odds.drop_duplicates("event_id")
    for r in ev.itertuples():
        ebd.setdefault((r.league, r.league_date), []).append(
            {"id": r.event_id, "commence_time": r.commence_time, "home_team": r.home_team,
             "away_team": r.away_team})
    return {"events_by_day": ebd, "identity": {}, "proven": {}}


def match_universe(universe: pd.DataFrame, odds: pd.DataFrame, ctx: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """({fixture_id: event_id}, unmatched report rows), by the collector's own rule
    (src.oddsapi_history.match_day: one league, the match day or +-1 day with both clubs proven,
    one-to-one, proven aliases and v9 identity spellings only)."""
    ctx = ctx or _context_from_odds(odds)
    u = universe.copy()
    u["date"] = pd.to_datetime(u["date"]).dt.strftime("%Y-%m-%d")
    u["fixture_id"] = fixture_id(universe).values
    fx = u.drop_duplicates(FIXTURE_KEY)[FIXTURE_KEY + ["fixture_id"]]
    days = {(lg, d): list(zip(g["home"], g["away"])) for (lg, d), g in fx.groupby(["league", "date"])}
    R = OH.Resolver(OH.club_spellings(ctx["identity"], days), ctx["proven"])
    ebd = ctx["events_by_day"]
    out, rep, used = {}, [], set()
    from src import league_registry as LR
    for (lg, d), g in fx.groupby(["league", "date"]):
        if not LR.odds_api_sport_keys_for(lg):
            rep += [{"kind": "league_without_sport_key", "league": lg, "date": d, "home": h, "away": a}
                    for h, a in zip(g["home"], g["away"])]
            continue
        matched, r = OH.match_day(lg, d, days[(lg, d)], ebd, R)
        ids = dict(zip(zip(g["home"], g["away"]), g["fixture_id"]))
        for f, e in matched.items():
            out[ids[f]] = e["id"]
            used.add(e["id"])
        for h, a in r["unmatched_fixtures"]:
            rep.append({"kind": "fixture_without_event", "league": lg, "date": d, "home": h, "away": a})
    leagues = set(fx["league"])
    seen = set()
    for (lg, _), evs in ebd.items():
        if lg not in leagues:
            continue
        for e in evs:
            if e["id"] in used or e["id"] in seen:
                continue
            seen.add(e["id"])
            rep.append({"kind": "event_without_fixture", "league": lg, "date": None, "home": e.get("home_team"),
                        "away": e.get("away_team"), "event_id": e["id"], "commence_time": e.get("commence_time")})
    return out, pd.DataFrame(rep, columns=UNMATCHED_COLS)


def OH_league(sport_key: str) -> str | None:
    from src import league_registry as LR
    return LR.sport_key_to_league().get(sport_key)


MIN_PRICE = 1.01            # owner 2026-11-11: a quote at or below 1.01 is not a usable price
PURPOSE_ROLE = {"bet": "BET", "close": "CLOSE", "open": "OPEN"}


class CloseAsBetError(ValueError):
    """A bet price was requested from a snapshot whose role is not BET (owner rule: a closing price
    may never be a bet price)."""


def actual_kickoffs(odds: pd.DataFrame, ctx: dict | None = None) -> pd.Series:
    """{event_id: kickoff} as the provider LAST stated it: the commence_time carried by the latest
    snapshot (odds rows and stored listings) of each event. A rescheduled match moves with it."""
    parts = [odds[["event_id", "snapshot_ts", "commence_time"]]]
    if ctx:
        rows = [(e.get("id"), e.get("snapshot_ts"), e.get("commence_time"))
                for evs in ctx.get("events_by_day", {}).values() for e in evs]
        if rows:
            parts.append(pd.DataFrame(rows, columns=["event_id", "snapshot_ts", "commence_time"]))
    k = pd.concat(parts, ignore_index=True)
    k["ts"] = pd.to_datetime(k["snapshot_ts"], utc=True, errors="coerce")
    k["ko"] = pd.to_datetime(k["commence_time"], utc=True, errors="coerce")
    k = k.dropna(subset=["ko"]).sort_values("ts", na_position="first")
    return k.drop_duplicates("event_id", keep="last").set_index("event_id")["ko"]


def attach(universe: pd.DataFrame, odds: pd.DataFrame, ctx: dict | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(long prices, summary, unmatched). long: one row per fixture x book x outcome x snapshot with
    our market vocabulary and the snapshot's ROLE (BET / CLOSE / OPEN / MOVEMENT,
    src.oddsapi_history.snapshot_role); summary: per fixture x market x line x side x role x snapshot
    label, n_books / best / median of the real prices.

    THE SHARED READ RULES (owner 2026-11-11), applied before anything is returned:
      * a quote whose snapshot is at or after the event's ACTUAL kickoff (the provider's latest stated
        commence_time; a rescheduled match moves) OR at or after the kickoff that snapshot itself
        carried is dropped, and so is a quote with no snapshot time (pre-kickoff cannot be proven);
      * a price at or below MIN_PRICE (1.01) is dropped.
    Counts in long.attrs['dropped']."""
    fmap, unmatched = match_universe(universe, odds, ctx)
    if not fmap:
        return pd.DataFrame(), pd.DataFrame(), unmatched
    inv = pd.DataFrame({"fixture_id": list(fmap), "event_id": list(fmap.values())})
    o = odds.merge(inv, on="event_id", how="inner")
    ko = o["event_id"].map(actual_kickoffs(odds, ctx))
    ts = pd.to_datetime(o["snapshot_ts"], utc=True, errors="coerce")
    own = pd.to_datetime(o["commence_time"], utc=True, errors="coerce")
    no_ts = ts.isna() | ko.isna()
    # post-kickoff by EITHER statement: the provider's latest kickoff, or the kickoff the snapshot itself
    # carried (a quote taken when the provider said the match had started is never a pre-match price)
    post = ~no_ts & ((ts >= ko) | (own.notna() & (ts >= own)))
    low = ~no_ts & ~post & (pd.to_numeric(o["price"], errors="coerce") <= MIN_PRICE)
    dropped = {"no_snapshot_ts": int(no_ts.sum()), "post_kickoff": int(post.sum()),
               "price_le_1_01": int(low.sum())}
    keep_rows = ~(no_ts | post | low)
    o = o[keep_rows].copy()
    o["actual_kickoff"] = ko[keep_rows].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    o = o.reset_index(drop=True)
    o["role"] = [OH.snapshot_role(t, lab) for t, lab in zip(o["tier"], o["snapshot_label"])]
    cls = [classify(m, n, ds, p, h, a) for m, n, ds, p, h, a in
           zip(o["market"], o["outcome_name"], o["outcome_description"], o["point"], o["home_team"], o["away_team"])]
    keep = [c is not None for c in cls]
    o = o[keep].reset_index(drop=True)
    c = pd.DataFrame([x for x in cls if x is not None], columns=["market", "side", "line", "team", "player"])
    o = o.rename(columns={"market": "provider_market"})
    # a FEATURED snapshot is timed off the day's FIRST kickoff, not this match's: its labels say so
    # ('day-close', 'day-T-48h'), so 'close' always means this match's own kickoff - 5 minutes
    o["snapshot_label"] = np.where(o["endpoint"] == "event_odds", o["snapshot_label"], "day-" + o["snapshot_label"])
    long = pd.concat([o[["fixture_id", "league", "event_id", "tier", "endpoint", "role", "snapshot_label",
                         "requested_at", "snapshot_ts", "commence_time", "actual_kickoff", "bookmaker",
                         "provider_market", "price"]], c], axis=1)
    grp = ["fixture_id", "market", "line", "team", "player", "side", "role", "snapshot_label"]
    k = long.assign(team=long["team"].fillna(""), player=long["player"].fillna(""))
    # ONE quote per book: a book seen twice for the same outcome (main `totals` and `alternate_totals`,
    # or two tiers) counts once - its latest snapshot, then the provider market name, as stored.
    # A market without a line keeps line NaN (grouped with dropna=False, never given a stand-in value).
    k = (k.sort_values(grp + ["bookmaker", "snapshot_ts", "provider_market"])
          .drop_duplicates(grp + ["bookmaker"], keep="last"))
    s = k.groupby(grp, sort=True, dropna=False)["price"].agg(n_books="count", best="max",
                                                             median="median").reset_index()
    long.attrs["dropped"] = dropped
    return long, s, unmatched


def prices_for(df: pd.DataFrame, purpose: str, role: str | None = None) -> pd.DataFrame:
    """The rows of `df` (attach's long frame or summary) that may serve `purpose`: 'bet' -> role BET
    only, 'close' -> CLOSE, 'open' -> OPEN. Asking for a bet price from any other role raises
    CloseAsBetError; nothing ever falls back to another role."""
    if purpose not in PURPOSE_ROLE:
        raise ValueError(f"purpose must be one of {sorted(PURPOSE_ROLE)}")
    want = PURPOSE_ROLE[purpose]
    if role is not None and role != want:
        if purpose == "bet":
            raise CloseAsBetError(f"a bet price may only come from role BET, not {role}")
        raise ValueError(f"purpose {purpose} reads role {want}, not {role}")
    if "role" not in df.columns:
        raise ValueError("the frame carries no snapshot role: build it with attach()")
    return df[df["role"] == want]


def attach_to_universe(universe: pd.DataFrame, summary: pd.DataFrame, *, purpose: str,
                       role: str | None = None) -> pd.DataFrame:
    """The universe rows with oh_<purpose>_{yes,no}_{best,median,n_books} for their OWN market
    (over15 / over25 / over35 / btts ...), from the role that purpose may read (prices_for: a bet
    price only from BET). One snapshot label per role and fixture is expected; should a role hold two
    labels for one fixture, the row is left unpriced rather than picking one."""
    u = universe.copy()
    u["fixture_id"] = fixture_id(universe).values
    cols = [f"oh_{purpose}_{side}_{k}" for side in ("yes", "no") for k in ("best", "median", "n_books")]
    if summary is None or summary.empty:
        prices_for(pd.DataFrame(columns=["role"]), purpose, role)            # the same refusal
        for col in cols:
            u[col] = np.nan
        return u
    s = prices_for(summary, purpose, role)
    s = s[(s["team"] == "") & (s["player"] == "")]
    for side in ("yes", "no"):
        part = s[s["side"] == side][["fixture_id", "market", "best", "median", "n_books"]]
        part = part[~part.duplicated(["fixture_id", "market"], keep=False)]
        part = part.rename(columns={k: f"oh_{purpose}_{side}_{k}" for k in ("best", "median", "n_books")})
        u = u.merge(part, on=["fixture_id", "market"], how="left")
    return u
