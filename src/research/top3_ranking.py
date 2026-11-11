"""TOP-3 ranking rules - the ONE implementation of R1, R2 and R3 (protocol config/top3_ranking_paper_protocol_v3.json).

Owner 2026-11-11 ("Clean rebuild"): the live TOP-3 is ranked by R3 (served from prep/bars-oddsapi-hist, which
imports it from HERE); R1 and R2 run as PAPER tracks (src/research/top3_paper.py). The old EV ranking is
dropped as a track; it stays below only as a DIAGNOSTIC rule for the research scripts.

PUBLIC API (pure functions, pandas in / pandas out):
    cell_stats(prior_bets, prior_rows) -> stats      per-cell statistics (the freeze computes them once)
    stats_from_freeze(record) -> stats               the frozen statistics of a freeze record (dict)
    rank_order(frame, stats, rule, tiebreak=())      the frame sorted best first, with key / has_stat / no_stat
    select(bets, stats, rule)                         research replay: TOP-3 per (track, league, ISO week)
    MSIDE_OF[(panel market, side)]                    -> study market-side ("over25_over", ...)
    in_pool(cell) -> bool                            the cell may feed the TOP-3 pool (MODEL cells only)

A frame to rank needs: track, league, mside (the CELL), fair, p_used, odds and ev_est (= p_used * odds - 1 from
the cell's own served source; used only as the tie-break / no-statistic order).

Why (output/research/top3_ranking/diagnosis.json): ev_est compares numbers on different scales across cells
- a RAW-source cell carries a far larger EV than a calibrated BLEND / SHRINK cell for the same realised value -
and inside a cell EV carries no ranking signal, so an EV ranking takes the most over-confident tips.

Every statistic is learned per CELL = (track, league, market-side) from that cell's OWN history dated before
the freeze, shrunk at most toward the SAME league's other markets. No number is pooled across leagues and no
constant is invented: MIN_OWN_BETS (40) and Z95 are the study's declared ones (src.bars_opt.study).

  R1_CELL_LCB    the cell's own walk-forward BET ROI, lower 95% bound (mean - Z95 * max-of-three SE)
  R2_CAL_EV      calibrated EV (fair + lam * (p_used - fair)) * odds - 1; lam = least squares of (won - fair)
                 on (p_used - fair) over the cell's own prior PRICED ROWS, shrunk toward the same league's
                 other markets (DerSimonian-Laird inside the league), clipped to [0, 1]
  R3_CAL_EV_LCB  as R2 with lam_lo = clip(lam - Z95 * se_own(lam), 0, 1)

POOL (owner 2026-11-11 "Own list"): a LEAGUE_WIDE cell (source NONE, mode LEAGUE_WIDE - no model probability)
is shown on its own line outside the TOP-3 slots, so it is never in the TOP-3 / TIER2 pool of any track
(R1, R2 or the live R3): in_pool(cell) is False for it.

A tip whose cell has no statistic for the rule gets none (nothing filled in): it ranks AFTER every tip that
has one, among those by ev_est, and is flagged no_stat = 1.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.bars_opt import study as S

RULES = ("R1_CELL_LCB", "R2_CAL_EV", "R3_CAL_EV_LCB")
LIVE_RULE = "R3_CAL_EV_LCB"
PAPER_RULES = ("R1_CELL_LCB", "R2_CAL_EV")
DIAGNOSTIC_RULES = ("EV_PRODUCTION", "DEV_CELL_EB")       # research scripts only; never a track
MIN_OWN_BETS = S.MIN_OWN_BETS          # 40, declared in the study (own bets for a bar / switch / TIER2 band)
Z95 = 1.96                             # the study's 95% interval (study.score: mean +- 1.96 * max-of-three SE)
CELL = ["track", "league", "mside"]
STAT_COLS = ["n", "roi", "se", "lcb", "eb", "lam", "lam_se", "lam_rows"]
#: panel market x side (src/top3_league.py candidates) -> study market-side
MSIDE_OF = {("OU25", "OVER"): "over25_over", ("OU25", "UNDER"): "over25_under", ("BTTS", "YES"): "btts_yes",
            ("OVER15", "OVER"): "over15_over", ("OVER35", "OVER"): "over35_over"}


def in_pool(cell: dict | None) -> bool:
    """True only for a MODEL cell (its own probability source). A LEAGUE_WIDE cell (source NONE) is the owner's
    own list, outside the TOP-3 slots (owner 2026-11-11 "Own list"); a missing cell is never in the pool."""
    return (isinstance(cell, dict) and cell.get("mode") == S.MODE_MODEL
            and cell.get("source") not in (None, S.LEAGUE_WIDE_SOURCE))


def _dl_shrink(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse-variance mean + DerSimonian-Laird tau^2 over the cells of ONE league; posterior means."""
    v = np.maximum(v, 1e-12)
    if len(m) < 2:
        return m.copy()
    iw = 1 / v
    mu = float((iw * m).sum() / iw.sum())
    q = float((iw * (m - mu) ** 2).sum())
    c = float(iw.sum() - (iw ** 2).sum() / iw.sum())
    tau2 = max(0.0, (q - (len(m) - 1)) / c) if c > 0 else 0.0
    return mu + (tau2 / (tau2 + v)) * (m - mu)


def _own_slope_se(g: pd.DataFrame) -> float:
    """SE of the cell's OWN least-squares slope (study.shrink_weights' per-group variance s2 / sxx)."""
    dd = (g["p_used"] - g["fair"]).to_numpy(float)
    r = g["won"].to_numpy(float) - g["fair"].to_numpy(float)
    sxx = float((dd ** 2).sum())
    if len(g) < 3 or sxx <= 0:
        return float("nan")
    w = float((dd * r).sum() / sxx)
    return math.sqrt(float(((r - w * dd) ** 2).sum() / (len(g) - 1)) / sxx)


def _lams(rows: pd.DataFrame | None) -> pd.DataFrame:
    cols = CELL + ["lam", "lam_se", "lam_rows"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=cols)
    out = []
    cnt = rows.groupby(CELL).size()
    for (tr, lg), g in rows.groupby(["track", "league"]):
        big = [m for m in g["mside"].unique() if cnt.get((tr, lg, m), 0) >= MIN_OWN_BETS]
        own = g[g["mside"].isin(big)]
        if own.empty:
            continue
        w = S.shrink_weights(own.assign(league=own["mside"], p_side=own["p_used"]))
        out += [{"track": tr, "league": lg, "mside": m, "lam": float(v),
                 "lam_se": _own_slope_se(own[own["mside"] == m]), "lam_rows": int(cnt[(tr, lg, m)])}
                for m, v in w.items()]
    return pd.DataFrame(out, columns=cols)


def cell_stats(prior_bets: pd.DataFrame | None, prior_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per cell. From prior BETS (track, league, mside, date, pnl): n, roi, se, lcb (R1), eb.
    From prior priced ROWS (track, league, mside, fair, p_used, won): lam, lam_se, lam_rows (R2, R3).
    NaN = no statistic for that rule; never filled."""
    lam = _lams(prior_rows)
    if prior_bets is None or prior_bets.empty:
        return lam.assign(n=0, roi=np.nan, se=np.nan, lcb=np.nan, eb=np.nan)[CELL + STAT_COLS]
    recs = []
    for key, g in prior_bets.groupby(CELL):
        pnl = g["pnl"].to_numpy(float)
        days = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").to_numpy()
        se = S.max_se(pnl, days)
        ok = len(g) >= MIN_OWN_BETS and np.isfinite(se)
        recs.append(dict(zip(CELL, key)) | {"n": len(g), "roi": float(pnl.mean()), "se": se if ok else np.nan,
                                             "lcb": float(pnl.mean() - Z95 * se) if ok else np.nan})
    st = pd.DataFrame(recs)
    st["eb"] = np.nan
    for _, idx in st.groupby(["track", "league"]).groups.items():
        sub = st.loc[idx]
        okk = sub["se"].notna()
        if okk.any():
            st.loc[sub.index[okk], "eb"] = _dl_shrink(sub.loc[okk, "roi"].to_numpy(float),
                                                      sub.loc[okk, "se"].to_numpy(float) ** 2)
    st = st.merge(lam, on=CELL, how="outer")
    st["n"] = st["n"].fillna(0).astype(int)
    return st[CELL + STAT_COLS]


def stats_from_freeze(record: dict) -> pd.DataFrame:
    """The frozen per-cell statistics of a freeze record (scripts/top3_paper_freeze.py), as cell_stats returns
    them. JSON null -> NaN (no statistic)."""
    st = pd.DataFrame(record.get("cells") or [], columns=CELL + STAT_COLS)
    for c in STAT_COLS:
        st[c] = pd.to_numeric(st[c], errors="coerce")
    st["n"] = st["n"].fillna(0).astype(int)
    return st


def rank_key(frame: pd.DataFrame, stats: pd.DataFrame, rule: str) -> pd.DataFrame:
    """frame + the cell's statistics + key (higher = taken first), has_stat and no_stat for `rule`."""
    if rule not in RULES + DIAGNOSTIC_RULES:
        raise ValueError(f"unknown ranking rule {rule!r}")
    b = frame.drop(columns=[c for c in STAT_COLS if c in frame.columns])
    if len(stats):
        b = b.merge(stats[CELL + STAT_COLS], on=CELL, how="left")
    else:
        b = b.assign(**{c: np.nan for c in STAT_COLS})
    fair, p, odds = (pd.to_numeric(b[c], errors="coerce") for c in ("fair", "p_used", "odds"))
    if rule == "EV_PRODUCTION":
        b["key"] = pd.to_numeric(b["ev_est"], errors="coerce")
    elif rule == "R1_CELL_LCB":
        b["key"] = b["lcb"]
    elif rule == "DEV_CELL_EB":
        b["key"] = b["eb"]
    else:
        lam = b["lam"] if rule == "R2_CAL_EV" else (b["lam"] - Z95 * b["lam_se"]).clip(0, 1)
        b["key"] = (fair + lam * (p - fair)) * odds - 1.0
    b["has_stat"] = b["key"].notna()
    b["no_stat"] = (~b["has_stat"]).astype(int)
    return b


def rank_order(frame: pd.DataFrame, stats: pd.DataFrame, rule: str, tiebreak: tuple = ()) -> pd.DataFrame:
    """`frame` best first under `rule`: tips with a statistic by key (desc), then those without; ties by
    ev_est (desc), then the `tiebreak` columns (asc, e.g. kickoff, fixture id). Stable."""
    if frame is None or frame.empty:
        cols = list(frame.columns) if frame is not None else []
        return pd.DataFrame(columns=cols + ["key", "has_stat", "no_stat"])
    b = rank_key(frame, stats, rule)
    b["_ev"] = pd.to_numeric(b["ev_est"], errors="coerce")
    cols, asc = ["no_stat", "key", "_ev", *tiebreak], [True, False, False, *([True] * len(tiebreak))]
    return b.sort_values(cols, ascending=asc, na_position="last", kind="mergesort").drop(columns="_ev")


def select(bets: pd.DataFrame, stats: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Research replay on study bet frames: TOP-3 per (track, league, ISO week); direction rule and never-pad
    as src.bars_opt.study.top3. EV_PRODUCTION reproduces study.top3(bets, "TOP_ONLY_V") pick for pick."""
    if bets is None or bets.empty:
        return pd.DataFrame()
    b = bets.copy()
    b["gw"] = b["date"].dt.strftime("%G-W%V")
    order = rank_order(b, stats, rule, tiebreak=("date",))
    picks = []
    for _, g in order.groupby(["track", "league", "gw"], sort=False):
        taken, dirs = [], {}
        for r in g.itertuples(index=False):
            d = "OVER" if r.mside in S.OVER_DIRECTION else "UNDER"
            if dirs.get(r.row_id, d) != d:
                continue
            dirs[r.row_id] = d
            taken.append(r)
            if len(taken) == 3:
                break
        picks += taken
    return pd.DataFrame(picks, columns=order.columns)
