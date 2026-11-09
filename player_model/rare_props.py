"""Rare props research picks: goals2, goals3, sot3, sot4 (owner 2026-11-09).

    "for those markets its suppose to be 2-3 max tips in a GW, only if the formulas find awesome
    opportunities"

RESEARCH ONLY. A pick here is paper, labelled RESEARCH, never staked, never on the normal tip board.
The four markets stay hidden from normal tips (config.HIDDEN_PENDING_CLEAN_RETRAIN): their clean
retrain (run 42524325567) found no model better than the player base rate.

WHAT A PICK NEEDS, in order (the first failing step is the row's reason, every row is counted):
  1. NO_BAR_YET           the market has no 'exceptional' bar in config/rare_props_bars.json. The bar
                          comes from the bars optimization study, per market: the minimum EV lower
                          bound AND the probability margin that defines the lower bound, each with its
                          source. No typed bar, no global bar: a market without its own is silent.
  2. NO_PRICE             no real bookmaker price for the player-market.
  3. NO_TWO_SIDED_PRICE   no real two-sided price (same book, both sides, de-vigged). An assumed
                          margin is never presented as a fair price, so such a row is not ranked.
  4. BELOW_BAR            ev_lower_bound under the market's bar.
  5. ALREADY_LOGGED       the same player-market of the fixture was picked by an earlier run.
  6. GW_CAP_REACHED       the gameweek already holds MAX_PICKS_PER_GAMEWEEK picks (all four markets
                          together, counted from the append-only log), or a stronger candidate took
                          the last slot this run.
  7. SELECTED             ranked by ev_lower_bound (descending; ties by kickoff, then player id).

  p_lower_bound   = model probability - the market's probability margin (from the study)
  ev_lower_bound  = p_lower_bound x price - 1, at the real price taken
  edge_lb_vs_fair = p_lower_bound - the de-vigged two-sided fair probability

GAMEWEEK. The ISO week of the tip day (src.gameweek rule GW-1 without the league): 'YYYY-Www'.
OUTPUT. output/rare_props_research.csv, append-only, first sight per (fixture_id, player_id, market).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .fair_basis import TWO_SIDED_DEVIG

RARE_MARKETS = tuple(sorted(config.HIDDEN_PENDING_CLEAN_RETRAIN))
#: owner 2026-11-09: "2-3 max tips in a GW" - the upper end, across the four markets together
MAX_PICKS_PER_GAMEWEEK = 3
BARS_PATH = config.BASE_DIR / "config" / "rare_props_bars.json"
LOG_PATH = config.OUTPUT_DIR / "rare_props_research.csv"
CANDIDATES_PATH = config.OUTPUT_DIR / "rare_props_candidates.csv"
STATE = "RESEARCH"
REASONS = ("NO_BAR_YET", "NO_PRICE", "NO_TWO_SIDED_PRICE", "BELOW_BAR", "ALREADY_LOGGED",
           "GW_CAP_REACHED", "SELECTED")
KEY = ["fixture_id", "player_id", "market"]
LOG_COLS = ["logged_at", "gameweek", "date", "kickoff_utc", "league", "match", "fixture_id", "player_id",
            "player_name", "team", "position", "market", "model_prob", "p_lower_margin", "p_lower_bound",
            "market_odds", "fair_prob_two_sided", "fair_basis", "ev_lower_bound", "edge_lb_vs_fair",
            "bar_ev_lower_bound", "bar_source", "rank", "state", "paper_only"]
CANDIDATE_COLS = ["date", "kickoff_utc", "league", "match", "fixture_id", "player_id", "player_name",
                  "team", "position", "market", "model_prob"]


def load_bars(path: Path | None = None) -> dict:
    """{market: {'ev_lower_bound_min', 'p_lower_margin', 'source'}} for the rare markets that HAVE a
    complete bar from the study. A missing file, a market without an entry, or an entry missing a
    number or its source gives that market no bar (NO_BAR_YET). Nothing is filled in."""
    p = BARS_PATH if path is None else Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for m in RARE_MARKETS:
        e = (raw.get("markets") or {}).get(m)
        if not isinstance(e, dict):
            continue
        try:
            bar, margin = float(e["ev_lower_bound_min"]), float(e["p_lower_margin"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(bar) and np.isfinite(margin) and margin >= 0 and str(e.get("source") or "").strip()):
            continue
        out[m] = {"ev_lower_bound_min": bar, "p_lower_margin": margin, "source": str(e["source"])}
    return out


def gameweek_of(kickoff_utc, match_date) -> str:
    """'YYYY-Www' of the tip day (src.gameweek GW-1 without the league); '' when unassignable."""
    from src.gameweek import league_week_id
    wid = league_week_id("ALL", kickoff_utc, None if match_date is None else str(match_date)[:10])
    return wid.split("|", 1)[1] if "|" in wid else ""


def evaluate(cands: pd.DataFrame, bars: dict, logged: pd.DataFrame | None = None,
             now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """(picks to log, {reason: count}). `cands` holds the rare player-markets with model_prob and,
    where priced, market_odds / fair_implied / fair_basis (predict.enrich_with_odds columns)."""
    counts = {r: 0 for r in REASONS}
    if cands is None or cands.empty:
        return pd.DataFrame(columns=LOG_COLS), counts
    c = cands[cands["market"].isin(RARE_MARKETS)].copy()
    c["gameweek"] = [gameweek_of(k, d) for k, d in zip(c.get("kickoff_utc", pd.Series(None, index=c.index)),
                                                        c.get("date", pd.Series(None, index=c.index)))]
    c["reason"] = None
    c["rank"] = np.nan
    has_bar = c["market"].isin(list(bars))
    c.loc[~has_bar, "reason"] = "NO_BAR_YET"
    odds = pd.to_numeric(c.get("market_odds"), errors="coerce") if "market_odds" in c else pd.Series(np.nan, index=c.index)
    c.loc[c["reason"].isna() & ~(odds > 1.0), "reason"] = "NO_PRICE"
    fair = pd.to_numeric(c.get("fair_implied"), errors="coerce") if "fair_implied" in c else pd.Series(np.nan, index=c.index)
    basis = c["fair_basis"].astype(str) if "fair_basis" in c else pd.Series("", index=c.index)
    two = (basis == TWO_SIDED_DEVIG) & (fair > 0) & (fair < 1)
    c.loc[c["reason"].isna() & ~two, "reason"] = "NO_TWO_SIDED_PRICE"
    margin = c["market"].map(lambda m: bars.get(m, {}).get("p_lower_margin", np.nan))
    barv = c["market"].map(lambda m: bars.get(m, {}).get("ev_lower_bound_min", np.nan))
    p = pd.to_numeric(c["model_prob"], errors="coerce")
    c["p_lower_margin"] = margin
    c["p_lower_bound"] = (p - margin).clip(lower=0.0)
    c["ev_lower_bound"] = c["p_lower_bound"] * odds - 1.0
    c["edge_lb_vs_fair"] = c["p_lower_bound"] - fair
    c["fair_prob_two_sided"] = fair.where(two)
    c["bar_ev_lower_bound"] = barv
    c["bar_source"] = c["market"].map(lambda m: bars.get(m, {}).get("source"))
    c.loc[c["reason"].isna() & ~(c["ev_lower_bound"] >= barv), "reason"] = "BELOW_BAR"
    ok = c[c["reason"].isna()].copy()
    if len(ok):
        done = set()
        if logged is not None and len(logged):
            done = set(zip(logged["fixture_id"].astype(str), logged["player_id"].astype(str), logged["market"]))
        seen = pd.Series([(str(f), str(pl), m) in done for f, pl, m in
                          zip(ok["fixture_id"], ok["player_id"], ok["market"])], index=ok.index)
        c.loc[seen[seen].index, "reason"] = "ALREADY_LOGGED"      # a pick is logged once (first sight)
        ok = ok[~seen]
        ok = ok.sort_values(["ev_lower_bound", "kickoff_utc", "player_id"], ascending=[False, True, True],
                            kind="mergesort")
        from collections import Counter
        used = Counter(logged["gameweek"].astype(str).tolist() if logged is not None and len(logged) else [])
        keep = []
        for i, r in ok.iterrows():
            gw = r["gameweek"]
            if not gw or used[gw] >= MAX_PICKS_PER_GAMEWEEK:
                c.at[i, "reason"] = "GW_CAP_REACHED"
                continue
            used[gw] += 1
            c.at[i, "reason"] = "SELECTED"
            c.at[i, "rank"] = used[gw]
            keep.append(i)
    for r in REASONS:
        counts[r] = int((c["reason"] == r).sum())
    picks = c[c["reason"] == "SELECTED"].sort_values(["gameweek", "rank"], kind="mergesort").copy()
    picks["logged_at"] = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    picks["state"] = STATE
    picks["paper_only"] = True
    if "fair_basis" not in picks:
        picks["fair_basis"] = None
    return picks.reindex(columns=LOG_COLS), counts


def read_log(path: Path | None = None) -> pd.DataFrame:
    p = LOG_PATH if path is None else Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=LOG_COLS)


def run(priced_cands: pd.DataFrame, bars_path: Path | None = None, log_path: Path | None = None,
        now: datetime | None = None) -> dict:
    """Evaluate this run's rare candidates and APPEND the picks (first sight per KEY). Returns the
    reason counts plus the number appended; prints them. Never raises into the props run."""
    lp = LOG_PATH if log_path is None else Path(log_path)
    logged = read_log(lp)
    picks, counts = evaluate(priced_cands, load_bars(bars_path), logged, now=now)
    if len(picks):
        out = pd.concat([logged.reindex(columns=LOG_COLS), picks], ignore_index=True)
        out = out.drop_duplicates(KEY, keep="first")
        lp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(lp, index=False)
    counts["appended"] = int(len(picks))
    print(f"[rare_props] RESEARCH picks: {counts}")
    return counts


def attach_prices(cands: pd.DataFrame, odds_mapped: dict, fair_mapped: dict) -> pd.DataFrame:
    """market_odds / fair_implied / fair_basis from the props odds answer of THIS run
    (odds_fetcher.match_odds_to_tips / match_fair_to_tips, keyed by tip_price_key). A fair
    probability exists only from a real two-sided price (fair_basis two_sided_devig); without one
    fair_implied and fair_basis stay empty - never an assumed margin."""
    from .odds_fetcher import tip_price_key
    if cands is None or cands.empty:
        return cands
    c = cands.copy()
    keys = [tip_price_key(p, m, d, mt) for p, m, d, mt in zip(c["player_name"], c["market"], c["date"], c["match"])]
    c["market_odds"] = [odds_mapped.get(k) for k in keys]
    c["fair_implied"] = [fair_mapped.get(k) for k in keys]
    c["fair_basis"] = [TWO_SIDED_DEVIG if pd.notna(f) else None for f in c["fair_implied"]]
    return c


def write_candidates(rows: list[dict], path: Path | None = None) -> int:
    """This run's rare player-markets (predict.run_player_predictions), replacing the previous
    run's file. Returns the row count."""
    p = CANDIDATES_PATH if path is None else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CANDIDATE_COLS).to_csv(p, index=False)
    return len(rows)


def read_candidates(path: Path | None = None) -> pd.DataFrame:
    p = CANDIDATES_PATH if path is None else Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=CANDIDATE_COLS)
