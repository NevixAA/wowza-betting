"""Side-market edge on two-sided de-vigged prices.

SINCE 2026-11-01 THIS IS PRODUCTION (owner cutover): pipeline._generate_side_bets prices side edges
with `production_fair_probs` below and skips fixtures with no two-sided pair. The shadow comparison
against the old rule keeps running for the record. Rollback: config.SIDE_EDGE_FAIR_PRICE.

THE OLD RULE (legacy_1_08):

    fair_prob = (1 / odds) / 1.08          edge = p_model - fair_prob

That assumes every side-market price carries an 8% margin on the quoted side. A real two-sided
BTTS pair from one book is usually 4-7% overround in total, split across both sides, so dividing a
single side's implied probability by 1.08 shaves 3-4% off the market's opinion and hands the model
a 2-3pp "edge" before any fixture-specific information exists.

SHADOW (this module): per bookmaker, BOTH sides of the same market from the SAME book, both quoted
strictly before kickoff and at or before the decision time, de-vigged with the power method
(src.market_edge.power_devig — the tested port of Pro's src/market/devig). The consensus is the
median over books. A fixture with no two-sided pair is ONE_SIDED: it has NO de-vigged price and is
never described as having one.

The shadow tier uses the SAME thresholds production would use for that league and market (approved
per-league bars from models/best_params_side_markets.json, else 0.10/0.08/0.04), so the only thing
that differs is the fair-price definition. Classification of each production tip:

    CONFIRMED_BY_DEVIG     shadow tier is not AVOID either
    ONLY_BECAUSE_OF_1_08   a de-vigged price exists and the shadow edge is below the VALUABLE bar
    UNVERIFIABLE_ONE_SIDED no two-sided pre-kickoff pair exists for this fixture/market
Nothing here changes a tip, a tier, a stake or a message.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src import side_thresholds as _ST
from src.market_edge import overround, power_devig

SHADOW_VERSION = "side_edge_devig_shadow-1.0"
PROD_OVERROUND = 1.08
FAMILY = {  # production market -> (snapshot market, side A, side B, p column, odds column)
    "btts": ("BTTS", "YES", "NO", "p_btts", "odds_btts"),
    "over15": ("OU15", "OVER", "UNDER", "p_over15", "odds_over15"),
    "over35": ("OU35", "OVER", "UNDER", "p_over35", "odds_over35"),
}
#: the declared global policy bar (config.SIDE_GLOBAL_BARS; was a typed (0.10, 0.08, 0.04) here)
DEFAULT_BARS = _ST.global_bars()


def _utc(x):
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return t


def thresholds(league_params: dict, market: str, league: str) -> tuple[float, float, float]:
    """Exactly pipeline._generate_side_bets' threshold choice: the SAME function
    (src/side_thresholds.side_bars, fix9/defaults 2026-11-04 - it was duplicated here with the
    global bar typed as a literal default). threshold_source: thresholds_with_source."""
    return thresholds_with_source(league_params, market, league)[:3]


def thresholds_with_source(league_params: dict, market: str, league: str) -> tuple[float, float, float, str]:
    """(sniper, marksman, valuable, 'LEAGUE' | 'GLOBAL')."""
    lp = (league_params.get(market, {}) or {}).get(league, {}) or {}
    return _ST.side_bars(lp)


def tier(edge: float, bars: tuple) -> str:
    if edge is None or not np.isfinite(edge):
        return "AVOID"
    s, m, v = bars
    if edge >= s:
        return "SNIPER"
    if edge >= m:
        return "MARKSMAN"
    if edge >= v:
        return "VALUABLE"
    return "AVOID"


def book_pairs(quotes: pd.DataFrame, match: str, snap_market: str, side_a: str, side_b: str,
               asof: pd.Timestamp, kickoff: pd.Timestamp | None) -> list[dict]:
    """Per-book two-sided pairs, last value per side carried forward up to `asof`, pre-kickoff.

    `quotes` columns: snapshot_ts (UTC), match, bookmaker, market, side, odds. A book contributes
    only when it quoted BOTH sides; quotes at/after kickoff are never an entry price."""
    q = quotes[(quotes["match"] == match) & (quotes["market"] == snap_market)]
    if q.empty:
        return []
    q = q[q["_ts"] <= asof]
    if kickoff is not None and not pd.isna(kickoff):
        q = q[q["_ts"] < kickoff]
    if q.empty:
        return []
    last = q.sort_values("_ts").groupby(["bookmaker", "side"]).tail(1)
    out = []
    for book, g in last.groupby("bookmaker"):
        sides = dict(zip(g["side"].str.upper(), zip(g["odds"], g["_ts"])))
        if side_a in sides and side_b in sides:
            (oa, ta), (ob, tb) = sides[side_a], sides[side_b]
            p = power_devig(float(oa), float(ob))
            if p is not None:
                out.append({"book": book, "odds_a": float(oa), "odds_b": float(ob),
                            "p_fair": p, "overround": overround(float(oa), float(ob)),
                            "ts": max(ta, tb)})
    return out


def evaluate_rows(rows: pd.DataFrame, quotes: pd.DataFrame, league_params: dict,
                  asof_col: str = "asof") -> pd.DataFrame:
    """rows: one per (fixture, market) with league, home_team, away_team, kickoff_utc, market,
    p_model, odds (production execution price), asof (decision time, UTC)."""
    q = quotes.copy()
    q["_ts"] = _utc(q["snapshot_ts"])
    q = q.dropna(subset=["_ts"])
    q["side"] = q["side"].astype(str).str.upper()
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q = q[q["odds"] > 1.0]
    out = []
    for _, r in rows.iterrows():
        mk = r["market"]
        snap_mk, sa, sb, _, _ = FAMILY[mk]
        match = f"{r['home_team']} vs {r['away_team']}"
        ko = _utc(pd.Series([r.get("kickoff_utc")])).iloc[0]
        asof = _utc(pd.Series([r[asof_col]])).iloc[0]
        pairs = book_pairs(q, match, snap_mk, sa, sb, asof, ko)
        p_model = float(r["p_model"])
        odds = float(r["odds"]) if pd.notna(r.get("odds")) else np.nan
        bars = thresholds(league_params, mk, r["league"])
        fair_prod = (1.0 / odds) / PROD_OVERROUND if odds and odds > 1 else np.nan
        edge_prod = p_model - fair_prod if np.isfinite(fair_prod) else np.nan
        rec = {"shadow_version": SHADOW_VERSION, "asof": str(asof), "league": r["league"],
               "home_team": r["home_team"], "away_team": r["away_team"],
               "kickoff_utc": str(r.get("kickoff_utc", "")), "market": mk,
               "model_type": r.get("model_type", ""), "p_model": p_model, "odds": odds,
               "fair_prod_1_08": fair_prod, "edge_prod": edge_prod,
               "tier_prod": r.get("tier_prod") or tier(edge_prod, bars),
               "bars": "/".join(f"{b:g}" for b in bars), "n_two_sided_books": len(pairs)}
        if pairs:
            pf = float(statistics.median(p["p_fair"] for p in pairs))
            rec.update(devig_state="DEVIGGED_TWO_SIDED", devig_method="power",
                       p_fair_devig=pf, books=";".join(sorted(p["book"] for p in pairs)),
                       median_pair_overround=float(statistics.median(p["overround"] for p in pairs)),
                       quote_ts_latest=str(max(p["ts"] for p in pairs)),
                       implied_side_margin=(1.0 / odds) / pf - 1.0 if np.isfinite(odds) else np.nan)
            rec["edge_devig"] = p_model - pf
            rec["tier_shadow"] = tier(rec["edge_devig"], bars)
        else:
            rec.update(devig_state="ONE_SIDED_NOT_DEVIGGED", devig_method="", p_fair_devig=np.nan,
                       books="", median_pair_overround=np.nan, quote_ts_latest="",
                       implied_side_margin=np.nan, edge_devig=np.nan, tier_shadow="UNAVAILABLE")
        if rec["tier_prod"] == "AVOID":
            rec["classification"] = "NOT_A_PRODUCTION_TIP"
        elif rec["devig_state"] != "DEVIGGED_TWO_SIDED":
            rec["classification"] = "UNVERIFIABLE_ONE_SIDED"
        elif rec["tier_shadow"] == "AVOID":
            rec["classification"] = "ONLY_BECAUSE_OF_1_08"
        else:
            rec["classification"] = "CONFIRMED_BY_DEVIG"
        out.append(rec)
    return pd.DataFrame(out)


def board_rows(board: pd.DataFrame) -> pd.DataFrame:
    """One row per (board fixture, side market) that production can price."""
    parts = []
    asof = board["generated_at"].iloc[0] if "generated_at" in board.columns and len(board) else None
    for mk, (_, _, _, pcol, ocol) in FAMILY.items():
        if pcol not in board.columns or ocol not in board.columns:
            continue
        d = board.dropna(subset=[pcol, ocol])
        if d.empty:
            continue
        parts.append(pd.DataFrame({
            "league": d["league"], "home_team": d["home_team"], "away_team": d["away_team"],
            "kickoff_utc": d.get("kickoff_utc", pd.Series("", index=d.index)), "market": mk,
            "model_type": d.get("model_type", pd.Series("", index=d.index)),
            "p_model": d[pcol], "odds": d[ocol], "asof": asof}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ledger_rows(ledger: pd.DataFrame, kickoffs: dict | None = None) -> pd.DataFrame:
    """Production side tips as logged (first sight), decision time = generated_at (UTC)."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    d = ledger[ledger["market"].isin(FAMILY)].copy()
    d = d.dropna(subset=["model_prob", "odds"])
    ko = None
    if kickoffs:
        ko = [kickoffs.get(f"{h} vs {a}", "") for h, a in zip(d["home_team"], d["away_team"])]
    return pd.DataFrame({
        "league": d["league"], "home_team": d["home_team"], "away_team": d["away_team"],
        "kickoff_utc": ko if ko is not None else "", "market": d["market"],
        "model_type": d.get("model_type", ""), "p_model": d["model_prob"], "odds": d["odds"],
        "asof": pd.to_datetime(d["generated_at"], errors="coerce").dt.tz_localize("UTC"),
        "tier_prod": d["signal_tier"], "match_date": d["match_date"]})


def summarise(ev: pd.DataFrame) -> dict:
    if ev is None or ev.empty:
        return {"rows": 0}
    tips = ev[ev["classification"] != "NOT_A_PRODUCTION_TIP"]
    out = {"rows": int(len(ev)), "production_tips": int(len(tips)),
           "classification": {k: int(v) for k, v in tips["classification"].value_counts().items()},
           "by_market": {}}
    for mk, g in tips.groupby("market"):
        out["by_market"][mk] = {k: int(v) for k, v in g["classification"].value_counts().items()}
    dv = ev[ev["devig_state"] == "DEVIGGED_TWO_SIDED"]
    if len(dv):
        out["devigged_rows"] = int(len(dv))
        out["median_pair_overround"] = round(float(dv["median_pair_overround"].median()), 4)
        out["median_edge_prod_minus_devig_pp"] = round(float((dv["edge_prod"] - dv["edge_devig"]).median() * 100), 2)
        diff = (dv["edge_prod"] - dv["edge_devig"]) * 100
        out["mean_edge_prod_minus_devig_pp"] = round(float(diff.mean()), 2)
        out["p90_edge_prod_minus_devig_pp"] = round(float(diff.quantile(0.9)), 2)
        out["rows_prod_edge_inflated_ge_2pp"] = int((diff >= 2).sum())
        # The 1.08 rule IS proportional de-vig when a book's pair overround is exactly 8%; the
        # inflation appears where the real overround is lower (sharp books, exchanges).
        low = dv["median_pair_overround"] < 0.06
        out["rows_pair_overround_below_6pct"] = int(low.sum())
        if low.any():
            out["mean_inflation_pp_when_overround_below_6pct"] = round(float(diff[low].mean()), 2)
    return out


def append_first_sight(ev: pd.DataFrame, path: Path) -> int:
    """Append-only, first sight per (shadow_version, league, home, away, market, asof-source)."""
    if ev is None or ev.empty:
        return 0
    key = ["shadow_version", "league", "home_team", "away_team", "market"]
    if path.exists():
        prev = pd.read_csv(path, low_memory=False)
        seen = set(map(tuple, prev[key].astype(str).to_numpy()))
        new = ev[[tuple(x) not in seen for x in ev[key].astype(str).to_numpy()]]
        if new.empty:
            return 0
        new.to_csv(path, mode="a", header=False, index=False, columns=list(prev.columns))
        return int(len(new))
    ev.to_csv(path, index=False)
    return int(len(ev))


def load_quotes(output_dir: Path, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-book side-market quotes from the committed change-log (current file + archive).
    `since` (UTC) drops older snapshot rows — production only needs the live slate."""
    cols = ["snapshot_ts", "match_date", "kickoff_utc", "match", "bookmaker", "market", "side", "odds"]
    out = Path(output_dir)
    parts = []
    for p in [out / "book_odds_snapshots.csv", *sorted((out / "book_odds_snapshots_archive").glob("*.csv"))]:
        try:
            parts.append(pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False))
        except Exception:                                           # noqa: BLE001
            continue
    q = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    q = q[q["market"].isin(["BTTS", "OU15", "OU35"])]
    if since is not None and len(q):
        q = q[_utc(q["snapshot_ts"]) >= since]
    return q


def production_fair_probs(df: pd.DataFrame, market: str, quotes: pd.DataFrame,
                          asof: pd.Timestamp) -> pd.Series:
    """PRODUCTION side-market fair probability (owner cutover 2026-11-01): the median over books of
    the power-de-vigged same-book two-sided pair, pre-kickoff, carried forward to `asof`.
    NaN where no two-sided pair exists — the caller must not tip those (fail closed).

    df: league, home_team, away_team, [kickoff_utc]; index preserved."""
    if df.empty or quotes is None or quotes.empty:
        return pd.Series(np.nan, index=df.index)
    q = quotes.copy()
    q["_ts"] = _utc(q["snapshot_ts"])
    q = q.dropna(subset=["_ts"])
    q["side"] = q["side"].astype(str).str.upper()
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q = q[q["odds"] > 1.0]
    ko_map = (q.dropna(subset=["kickoff_utc"]).drop_duplicates("match", keep="last")
               .set_index("match")["kickoff_utc"].astype(str).to_dict()) if "kickoff_utc" in q else {}
    snap_mk, sa, sb, _, _ = FAMILY[market]
    out = {}
    for idx, r in df.iterrows():
        match = f"{r['home_team']} vs {r['away_team']}"
        ko = _utc(pd.Series([r.get("kickoff_utc") or ko_map.get(match)])).iloc[0]
        pairs = book_pairs(q, match, snap_mk, sa, sb, asof, ko)
        out[idx] = float(statistics.median(p["p_fair"] for p in pairs)) if pairs else np.nan
    return pd.Series(out, dtype=float).reindex(df.index)


def load_league_params(models_dir: Path) -> dict:
    p = Path(models_dir) / "best_params_side_markets.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return {}
