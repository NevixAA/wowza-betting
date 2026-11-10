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


def match_universe(universe: pd.DataFrame, odds: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """({fixture_id: event_id}, unmatched report rows)."""
    from src import league_registry as LR
    u = universe.copy()
    u["date"] = pd.to_datetime(u["date"]).dt.strftime("%Y-%m-%d")
    u["fixture_id"] = fixture_id(universe).values
    fx = u.drop_duplicates(FIXTURE_KEY)[FIXTURE_KEY + ["fixture_id"]]
    ev = odds.drop_duplicates("event_id")[["event_id", "sport_key", "commence_time", "home_team", "away_team"]]
    ev = ev.assign(ko=pd.to_datetime(ev["commence_time"], utc=True, errors="coerce")).dropna(subset=["ko"])
    by_sport = {k: g for k, g in ev.groupby("sport_key")}
    out, rep, used = {}, [], set()
    for (lg, d), g in fx.groupby(["league", "date"]):
        keys = LR.odds_api_sport_keys_for(lg)
        if not keys:
            rep += [{"kind": "league_without_sport_key", "league": lg, "date": d, "home": h, "away": a}
                    for h, a in zip(g["home"], g["away"])]
            continue
        lo, hi = OH.day_window(d)
        cands = pd.concat([by_sport[k] for k in keys if k in by_sport]) if any(k in by_sport for k in keys) \
            else ev.iloc[0:0]
        cands = cands[(cands["ko"] >= pd.Timestamp(lo)) & (cands["ko"] < pd.Timestamp(hi))]
        fixtures = list(zip(g["home"], g["away"]))
        matched, r = OH.match_events(fixtures, [{"id": e.event_id, "commence_time": e.commence_time,
                                                 "home_team": e.home_team, "away_team": e.away_team}
                                                for e in cands.itertuples()], d)
        ids = dict(zip(zip(g["home"], g["away"]), g["fixture_id"]))
        for f, e in matched.items():
            out[ids[f]] = e["id"]
            used.add(e["id"])
        for h, a in r["unmatched_fixtures"]:
            rep.append({"kind": "fixture_without_event", "league": lg, "date": d, "home": h, "away": a})
    for e in ev.itertuples():
        if e.event_id not in used:
            rep.append({"kind": "event_without_fixture", "league": OH_league(e.sport_key), "date": None,
                        "home": e.home_team, "away": e.away_team, "event_id": e.event_id,
                        "commence_time": e.commence_time})
    return out, pd.DataFrame(rep, columns=UNMATCHED_COLS)


def OH_league(sport_key: str) -> str | None:
    from src import league_registry as LR
    return LR.sport_key_to_league().get(sport_key)


def attach(universe: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(long prices, summary, unmatched). long: one row per fixture x book x outcome x snapshot with
    our market vocabulary; summary: per fixture x market x line x side x snapshot label, n_books /
    best / median of the real prices."""
    fmap, unmatched = match_universe(universe, odds)
    if not fmap:
        return pd.DataFrame(), pd.DataFrame(), unmatched
    inv = pd.DataFrame({"fixture_id": list(fmap), "event_id": list(fmap.values())})
    o = odds.merge(inv, on="event_id", how="inner")
    cls = [classify(m, n, ds, p, h, a) for m, n, ds, p, h, a in
           zip(o["market"], o["outcome_name"], o["outcome_description"], o["point"], o["home_team"], o["away_team"])]
    keep = [c is not None for c in cls]
    o = o[keep].reset_index(drop=True)
    c = pd.DataFrame([x for x in cls if x is not None])
    o = o.rename(columns={"market": "provider_market"})
    # a FEATURED snapshot is timed off the day's FIRST kickoff, not this match's: its labels say so
    # ('day-close', 'day-T-48h'), so 'close' always means this match's own kickoff - 5 minutes
    o["snapshot_label"] = np.where(o["endpoint"] == "event_odds", o["snapshot_label"], "day-" + o["snapshot_label"])
    long = pd.concat([o[["fixture_id", "league", "event_id", "tier", "endpoint", "snapshot_label", "requested_at",
                         "snapshot_ts", "commence_time", "bookmaker", "provider_market", "price"]], c], axis=1)
    grp = ["fixture_id", "market", "line", "team", "player", "side", "snapshot_label"]
    k = long.assign(team=long["team"].fillna(""), player=long["player"].fillna(""))
    # ONE quote per book: a book seen twice for the same outcome (main `totals` and `alternate_totals`,
    # or two tiers) counts once - its latest snapshot, then the provider market name, as stored.
    # A market without a line keeps line NaN (grouped with dropna=False, never given a stand-in value).
    k = (k.sort_values(grp + ["bookmaker", "snapshot_ts", "provider_market"])
          .drop_duplicates(grp + ["bookmaker"], keep="last"))
    s = k.groupby(grp, sort=True, dropna=False)["price"].agg(n_books="count", best="max",
                                                             median="median").reset_index()
    return long, s, unmatched


def attach_to_universe(universe: pd.DataFrame, summary: pd.DataFrame, label: str = "close") -> pd.DataFrame:
    """The universe rows with oh_<label>_{yes,no}_{best,median,n_books} for their OWN market
    (over15 / over25 / over35 / btts ...). A row the history does not price keeps NaN."""
    u = universe.copy()
    u["fixture_id"] = fixture_id(universe).values
    if summary is None or summary.empty:
        for side in ("yes", "no"):
            for k in ("best", "median", "n_books"):
                u[f"oh_{label}_{side}_{k}"] = np.nan
        return u
    s = summary[(summary["snapshot_label"] == label) & (summary["team"] == "") & (summary["player"] == "")]
    for side in ("yes", "no"):
        part = s[s["side"] == side][["fixture_id", "market", "best", "median", "n_books"]].rename(
            columns={k: f"oh_{label}_{side}_{k}" for k in ("best", "median", "n_books")})
        u = u.merge(part, on=["fixture_id", "market"], how="left")
    return u
