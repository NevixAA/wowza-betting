"""
props_harness_clv.py - closing-line value for the Phase 2 props retrain harness (player_model.props_harness).

BUILD ONLY: on real data it runs inside the owner-gated `props_harness --run`; the dry-run only COUNTS
the two-sided pairs (pair_counts). Tests use synthetic quotes.

SOURCE. The props quote log (output/prop_quotes/<day>.csv: snapshot_ts, match_date, league, match,
player, market, side, point, bookmaker, odds, source). One (snapshot_ts, source, match_date, match) group
is one provider answer.

THE FAIR PROBABILITY AT THE CLOSE (owner decision 2026-11-05, player_model.fair_basis):
  two_sided_devig    a REAL pair - the Yes/Over and the No/Under price from the same book in the same
                     answer (odds_fetcher._add_fixture_pairs, power de-vig) - median over books, taken
                     from the LAST answer on or before the match day that holds a pair.
  assumed_overround  no pair: the last best Yes/Over price divided by the ASSUMED overround
                     (predict._devig_fair_prob, config.OVERROUND_BY_ODDS). Every such row is flagged
                     `assumed_margin` and summarised APART; it is never pooled with two-sided rows.

CLV. For a row the model would have backed at the first price seen (model probability x that price
> 1: the model's own probability, no threshold, no hand adjustment), clv = first price x the fair
probability at the close - 1. Positive = the price taken beat the closing fair price.

KNOWN LIMITS (reported, not hidden). The quote log stores a price only the first time it is seen, so
an answer whose one side did not move holds no pair there; kickoff times are not in the log, so "the
close" is the last answer dated on or before the match day.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .fair_basis import ASSUMED_OVERROUND, TWO_SIDED_DEVIG

QUOTES_DIR = config.OUTPUT_DIR / "prop_quotes"
_YES_SIDES = {"yes", "over"}
CLV_COLS = ["market", "date", "player_name", "match", "p_model", "price_open", "p_fair_close",
            "fair_basis", "assumed_margin", "n_books_close", "selected", "clv", "y"]


def load_quotes(qdir: Path | None = None) -> pd.DataFrame:
    d = QUOTES_DIR if qdir is None else Path(qdir)
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return pd.DataFrame(columns=["snapshot_ts", "match_date", "match", "player", "market", "side",
                                     "point", "bookmaker", "odds", "source"])
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _prep(quotes: pd.DataFrame) -> pd.DataFrame:
    from .odds_fetcher import _norm
    q = quotes.copy()
    q["side"] = q["side"].astype(str).str.strip().str.lower()
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q["match_date"] = q["match_date"].astype(str).str[:10]
    q["player_norm"] = q["player"].astype(str).map(_norm)
    q["snapshot_ts"] = q["snapshot_ts"].astype(str)
    # the close is on or before the match day (no kickoff time in the log)
    return q[q["snapshot_ts"].str[:10] <= q["match_date"]]


def closing_pairs(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per (match_date, match, player_norm, market): the median two-sided fair probability over books
    in the LAST answer that holds a pair. Columns: match_date, match, player_norm, market, p_fair,
    n_books, snapshot_ts."""
    from .odds_fetcher import _add_fixture_pairs
    q = _prep(quotes)
    rows = []
    for (ts, _src, md, match), g in q.groupby(["snapshot_ts", "source", "match_date", "match"], sort=False):
        fp: dict = {}
        if not _add_fixture_pairs(fp, md, match, g.to_dict("records")):
            continue
        for key, pairs in next(iter(fp.values())).items():
            player, _, market = key.rpartition("|")
            ps = [float(p["p_fair"]) for p in pairs if p.get("p_fair") is not None]
            if ps:
                rows.append({"match_date": md, "match": match, "player_norm": player, "market": market,
                             "p_fair": float(statistics.median(ps)), "n_books": len(ps), "snapshot_ts": ts})
    cols = ["match_date", "match", "player_norm", "market", "p_fair", "n_books", "snapshot_ts"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("snapshot_ts", kind="stable")
    return out.groupby(["match_date", "match", "player_norm", "market"], as_index=False).last()[cols]


def yes_prices(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per (match_date, match, player_norm, market): the best Yes/Over price at the FIRST and at the
    LAST answer that quotes it."""
    q = _prep(quotes)
    q = q[q["side"].isin(_YES_SIDES) & (q["odds"] > 1.0)]
    cols = ["match_date", "match", "player_norm", "market", "price_open", "price_close"]
    if q.empty:
        return pd.DataFrame(columns=cols)
    k = ["match_date", "match", "player_norm", "market"]
    best = q.groupby(k + ["snapshot_ts"], as_index=False)["odds"].max().sort_values("snapshot_ts", kind="stable")
    g = best.groupby(k, as_index=False)
    out = g.first().rename(columns={"odds": "price_open"})[k + ["price_open"]]
    out["price_close"] = g.last()["odds"].to_numpy()
    return out[cols]


def pair_counts(quotes: pd.DataFrame) -> dict:
    """Dry-run view: how many player-markets have a REAL two-sided closing price (counts only)."""
    cp = closing_pairs(quotes)
    yp = yes_prices(quotes)
    out = {"quotes": int(len(quotes)), "player_markets_priced": int(len(yp)),
           "player_markets_two_sided": int(len(cp)), "by_market": {}}
    for m in sorted(set(yp["market"]) | set(cp["market"])):
        out["by_market"][m] = {"priced": int((yp["market"] == m).sum()),
                               "two_sided": int((cp["market"] == m).sum())}
    if len(cp):
        out["two_sided_fixtures"] = int(cp[["match_date", "match"]].drop_duplicates().shape[0])
        out["two_sided_dates"] = [str(cp["match_date"].min()), str(cp["match_date"].max())]
    return out


def _resolve(names: list[str], player: str) -> str | None:
    """The quoted name for `player`: exact normalised name, else a UNIQUE initial + surname match."""
    from .odds_fetcher import _initial_surname, _norm
    n = _norm(player)
    if n in names:
        return n
    key = _initial_surname(player)
    if not key:
        return None
    hits = [x for x in names if _initial_surname(x) == key]
    return hits[0] if len(hits) == 1 else None


def clv_rows(scored: pd.DataFrame, quotes: pd.DataFrame, p_col: str) -> pd.DataFrame:
    """One row per scored row that has a Yes/Over price in its own fixture. `scored` needs market, date,
    player_name, home_team, away_team and the probability column `p_col` (used as is)."""
    from .odds_fetcher import _dates_close, same_fixture
    from .predict import _devig_fair_prob
    yp, cp = yes_prices(quotes), closing_pairs(quotes)
    if yp.empty or scored.empty:
        return pd.DataFrame(columns=CLV_COLS)
    cp_idx = cp.set_index(["match_date", "match", "player_norm", "market"]) if len(cp) else None
    qfx = yp[["match_date", "match"]].drop_duplicates().to_records(index=False).tolist()
    s = scored.assign(_date=pd.to_datetime(scored["date"]).dt.strftime("%Y-%m-%d"),
                      _match=scored["home_team"].astype(str) + " vs " + scored["away_team"].astype(str))
    rows = []
    for (day, match), g in s.groupby(["_date", "_match"], sort=False):
        fx = [(d, m) for d, m in qfx if _dates_close(d, day) and same_fixture(match, m)]
        if len(fx) != 1:                                # no quote, or two quoted fixtures: ambiguous
            continue
        qd, qm = fx[0]
        here = yp[(yp["match_date"] == qd) & (yp["match"] == qm)]
        for market, gm in g.groupby("market", sort=False):
            hm = here[here["market"] == market].set_index("player_norm")
            names = list(hm.index)
            for _, r in gm.iterrows():
                who = _resolve(names, r["player_name"])
                if who is None:
                    continue
                price_open = float(hm.loc[who, "price_open"])
                pair = None
                if cp_idx is not None and (qd, qm, who, market) in cp_idx.index:
                    pair = cp_idx.loc[(qd, qm, who, market)]
                if pair is not None:
                    p_fair, basis, nb = float(pair["p_fair"]), TWO_SIDED_DEVIG, int(pair["n_books"])
                else:
                    p_fair, basis, nb = _devig_fair_prob(float(hm.loc[who, "price_close"])), ASSUMED_OVERROUND, 0
                p = float(r[p_col])
                sel = bool(np.isfinite(p) and p * price_open > 1.0)
                rows.append({"market": market, "date": day, "player_name": r["player_name"], "match": qm,
                             "p_model": p, "price_open": price_open, "p_fair_close": p_fair,
                             "fair_basis": basis, "assumed_margin": basis == ASSUMED_OVERROUND,
                             "n_books_close": nb, "selected": sel,
                             "clv": price_open * p_fair - 1.0 if np.isfinite(p_fair) else np.nan,
                             "y": float(r["y"]) if "y" in r.index else np.nan})
    return pd.DataFrame(rows, columns=CLV_COLS)


def summarise(rows: pd.DataFrame) -> dict:
    """Per market and fair_basis, never pooled across the two bases: priced rows, rows the model would
    have backed and their mean CLV. On two-sided rows with an outcome, the log loss of the model and of
    the closing fair price on the SAME rows (the market as the benchmark)."""
    out = {}
    if rows.empty:
        return out
    for (m, basis), g in rows.groupby(["market", "fair_basis"]):
        sel = g[g["selected"] & g["clv"].notna()]
        rec = {"priced_rows": int(len(g)), "selected_rows": int(len(sel)),
               "mean_clv_selected": float(sel["clv"].mean()) if len(sel) else None,
               "assumed_margin": basis == ASSUMED_OVERROUND}
        lab = g[g["y"].notna() & g["p_model"].notna() & g["p_fair_close"].notna()]
        if basis == TWO_SIDED_DEVIG and len(lab):
            y = lab["y"].to_numpy(dtype=float)
            rec["benchmark_rows"] = int(len(lab))
            rec["log_loss_model"] = float(_ll(y, lab["p_model"]).mean())
            rec["log_loss_closing_fair"] = float(_ll(y, lab["p_fair_close"]).mean())
        out.setdefault(m, {})[basis] = rec
    return out


def _ll(y, p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
