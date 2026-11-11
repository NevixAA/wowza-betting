"""RESEARCH ONLY - pre-registered alternative TOP-3 ranking rules (protocol config/top3_ranking_paper_protocol.json).

Nothing here is imported by production. The rules change ONLY the order in which a league-week's qualifying
tips are taken; the eligible pool (every tip at or above its league's own VALUABLE bar for that market), the
cap of 3 per league per gameweek, never padding, the direction rule and the TIER2 label are those of
src.bars_opt.study.top3 (TOP_ONLY_V) / src.top3_league.

Why (reports/TOP3_RANKING_DIAGNOSIS.md): the production key ev_est = p_used * odds - 1 compares numbers on
different scales across cells - a RAW-source cell (uncalibrated model probability) carries a far larger EV
than a BLEND / SHRINK cell (calibrated) for the same realised value - and inside a cell EV carries no
ranking signal. So the key keeps taking the most over-confident tips.

Every statistic is learned per CELL = (track, league, market-side) from that cell's OWN history dated before
the ranking date, shrunk at most toward the SAME league's other markets. No number is pooled across leagues
and no constant is invented: the minimum own bets / rows (MIN_OWN_BETS = 40) and the 95% level (Z95) are the
study's declared ones (src.bars_opt.study, src.league_bars).

  R1_CELL_LCB    cell's own walk-forward BET ROI, lower 95% bound (mean - Z95 * max-of-three SE)
  R2_CAL_EV      per-tip EV with the cell's calibrated probability p_cal = fair + lam * (p_used - fair);
                 lam = least squares of (won - fair) on (p_used - fair) over the cell's own prior PRICED ROWS
                 (every priced match of that league x market-side, tipped or not, scored by the cell's own
                 probability source), shrunk toward the same league's other markets (DerSimonian-Laird
                 tau^2 learned inside the league) and clipped to [0, 1] (study.shrink_weights with the
                 market-side as the group, inside ONE league)
  R3_CAL_EV_LCB  as R2 with lam_lo = clip(lam - Z95 * se_own(lam), 0, 1): the EV the tip still has at the
                 low end of its cell's calibration - a large model-vs-market gap counts only as far as the
                 cell's own history supports it
  DEV_CELL_EB    (development only, NOT a paper track: it ranked like R1 on the selection window) the cell's
                 bet ROI shrunk toward the league's own all-market ROI

A cell without the rule's statistic (too few own bets / rows) has none - nothing is filled in: its tips rank
AFTER every tip that has one, among themselves by the production key (ev_est); they are counted per pick
(no_stat). Ties are broken by the production key, then kickoff date.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.bars_opt import study as S

RULES = ("EV_PRODUCTION", "R1_CELL_LCB", "R2_CAL_EV", "R3_CAL_EV_LCB")
DEV_ONLY = ("DEV_CELL_EB",)
MIN_OWN_BETS = S.MIN_OWN_BETS          # 40, declared in the study (own bets for a bar / switch / TIER2 band)
Z95 = 1.96                             # the study's 95% interval (study.score: mean +- 1.96 * max-of-three SE)
CELL = ["track", "league", "mside"]
STAT_COLS = ["n", "roi", "se", "lcb", "eb", "lam", "lam_se", "lam_rows"]


def _dl_shrink(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse-variance mean + DerSimonian-Laird tau^2 over the cells of ONE league; posterior means.
    One cell: its own value."""
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
    """SE of the cell's OWN least-squares slope (study.shrink_weights' per-league variance s2 / sxx)."""
    dd = (g["p_used"] - g["fair"]).to_numpy(float)
    r = g["won"].to_numpy(float) - g["fair"].to_numpy(float)
    sxx = float((dd ** 2).sum())
    if len(g) < 3 or sxx <= 0:
        return float("nan")
    w = float((dd * r).sum() / sxx)
    return math.sqrt(float(((r - w * dd) ** 2).sum() / (len(g) - 1)) / sxx)


def _lams(rows: pd.DataFrame | None) -> pd.DataFrame:
    """R2 / R3 calibration per cell from prior priced rows (cells with >= MIN_OWN_BETS rows)."""
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


def cell_stats(prior: pd.DataFrame | None, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per cell: n, roi, se, lcb, eb from its prior BETS (R1, DEV_CELL_EB) and lam, lam_se from its
    prior priced ROWS (R2, R3). NaN = no statistic for that rule; never filled."""
    lam = _lams(rows)
    if prior is None or prior.empty:
        return lam.assign(n=0, roi=np.nan, se=np.nan, lcb=np.nan, eb=np.nan)[CELL + STAT_COLS]
    recs = []
    for key, g in prior.groupby(CELL):
        pnl = g["pnl"].to_numpy(float)
        days = g["date"].dt.strftime("%Y-%m-%d").to_numpy()
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


def rank_key(bets: pd.DataFrame, stats: pd.DataFrame, rule: str) -> pd.DataFrame:
    """bets + has_stat (bool) and key (higher = taken first) for `rule`."""
    if len(stats):
        b = bets.merge(stats, on=CELL, how="left")
    else:
        b = bets.assign(**{c: np.nan for c in STAT_COLS})
    if rule == "EV_PRODUCTION":
        b["key"] = b["ev_est"]
    elif rule == "R1_CELL_LCB":
        b["key"] = b["lcb"]
    elif rule == "DEV_CELL_EB":
        b["key"] = b["eb"]
    elif rule in ("R2_CAL_EV", "R3_CAL_EV_LCB"):
        lam = b["lam"] if rule == "R2_CAL_EV" else (b["lam"] - Z95 * b["lam_se"]).clip(0, 1)
        b["key"] = (b["fair"] + lam * (b["p_used"] - b["fair"])) * b["odds"] - 1.0
    else:
        raise ValueError(rule)
    b["has_stat"] = b["key"].notna()
    return b


def select(bets: pd.DataFrame, stats: pd.DataFrame, rule: str) -> pd.DataFrame:
    """TOP-3 per (track, league, ISO week) under `rule`; direction rule and never-pad as study.top3.
    EV_PRODUCTION reproduces study.top3(bets, "TOP_ONLY_V") pick for pick."""
    if bets is None or bets.empty:
        return pd.DataFrame()
    b = rank_key(bets, stats, rule)
    b["gw"] = b["date"].dt.strftime("%G-W%V")
    b["no_stat"] = (~b["has_stat"]).astype(int)
    b["slot"] = np.where((b["band"] == "BAND_VM") & b["tier2_ok"].astype(bool), "TIER2", "TOP")
    order = b.sort_values(["no_stat", "key", "ev_est", "date"], ascending=[True, False, False, True],
                          na_position="last", kind="mergesort")
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
