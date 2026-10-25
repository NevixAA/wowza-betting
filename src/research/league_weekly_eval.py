"""Analytics for the weekly per-league research panel (output/league_weekly_research.csv). PURE.

    from src.research import league_weekly_eval as EV
    EV.league_table(panel)            # league x market x model_type x side x subset -> metrics
    EV.rank_monotonicity(panel)       # rank 1..5 per league x market x model_type x side
    EV.absolute_vs_relative(panel)    # EV_LB > 0 (absolute score) vs rank 1 (relative rank)
    EV.weekly_grid(panel, league, market)   # gameweek x #1..#5 grid (the dashboard's history table)
    EV.evaluate(panel)                # all of the above as JSON-ready dicts

The dashboard page (pages/18_League_Research.py) renders THESE functions' outputs; it never aggregates
on its own (tests/test_league_weekly.py checks the page's data path reproduces them exactly).

UNIT OF ANALYSIS. league x market x model_type x side. OU25 OVER and UNDER are separate rows (a top-5 can
mix sides; the rank is the rank within the league-week, both sides together). Nothing is pooled across
model types or leagues except `rank_monotonicity(pooled=True)`, which is labelled POOLED_ACROSS_LEAGUES.

SUBSETS  RANK1 (rank 1), TOP2, TOP3, TOP5 (every member), QUALIFIED (decision_state QUALIFIED/STRONG),
         PRODUCTION_SELECTED (production tipped the same fixture x market x side: the settled
         `production_selected_final` when known, else the first-sight `production_selected`).

METRICS  N settled, independent gameweeks, win rate (every settled row); flat ROI on rows with a two-sided
         price only (INSUFFICIENT rows never certify performance) with a 90% GAMEWEEK BLOCK-BOOTSTRAP CI
         (2000 resamples, seed crc32(key)); certified CLV mean / median / share positive and coverage
         (rows with a certified close / priced settled rows); Brier and log loss of the model, the
         de-vigged first-sight market and the decision probability on the same rows; max drawdown and
         longest losing streak in kickoff order (flat 1u).
STATUS   NOT_ENOUGH_DATA below 30 priced settled rows or 8 independent gameweeks (the regime registry's
         adequacy bar) - the numbers are still shown, the status says they cannot be read yet. OK above it.
         Evidence type is TRUE_FORWARD: the panel exists only from its first run; nothing is replayed.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

GROUP = ["league", "market", "model_type", "side"]
SUBSETS = ("RANK1", "TOP2", "TOP3", "TOP5", "QUALIFIED", "PRODUCTION_SELECTED")
PRICED_STATES = ("TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK")
MIN_N, MIN_WEEKS = 30, 8
N_BOOT, CI = 2000, 0.90
EPS = 1e-6
METRIC_COLS = ["status", "n_rows", "n_settled", "n_priced_settled", "n_insufficient", "gameweeks", "win_rate",
               "flat_pnl", "roi", "roi_ci_lo", "roi_ci_hi", "clv_n", "clv_coverage", "clv_mean", "clv_median",
               "clv_positive_share", "n_prob", "brier_model", "brier_market", "brier_decision", "logloss_model",
               "logloss_market", "logloss_decision", "dlogloss_model_minus_market", "max_drawdown",
               "longest_losing_streak"]


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Typed copy of the stored panel (dtype=str in, numbers and flags out). Never mutates the input."""
    d = panel.copy() if panel is not None else pd.DataFrame()
    if d.empty:
        return d.assign(**{c: pd.Series(dtype=float) for c in ("rank", "won", "pnl")})
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)  # noqa: E731
    d["rank"] = num("league_market_rank")
    d["ev_lb"] = num("ev_lower_bound")
    d["odds_f"] = num("odds")
    d["pnl"] = num("flat_pnl")
    d["clv"] = num("clv_pct")
    d["clv_p"] = num("clv_prob")
    d["p_model"] = num("p_model_raw")
    d["p_market"] = num("p_market_fair")
    d["p_dec"] = num("p_decision")
    res = d["result"].astype(str).str.upper() if "result" in d.columns else pd.Series("", index=d.index)
    d["settled"] = res.isin(["WIN", "LOSS"])
    d["won"] = np.where(res == "WIN", 1.0, np.where(res == "LOSS", 0.0, np.nan))
    pq = d["price_quality_state"] if "price_quality_state" in d.columns else pd.Series("", index=d.index)
    d["priced"] = pq.isin(PRICED_STATES)
    fin = d["production_selected_final"].astype(str).str.strip() if "production_selected_final" in d.columns \
        else pd.Series("", index=d.index)
    first = _bool(d["production_selected"]) if "production_selected" in d.columns else pd.Series(False, index=d.index)
    d["prod_sel"] = np.where(fin != "", _bool(fin), first)
    d["qualified"] = d["decision_state"].isin(["QUALIFIED", "STRONG"]) if "decision_state" in d.columns else False
    d["_ko"] = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    return d


def subset(d: pd.DataFrame, name: str) -> pd.DataFrame:
    if d.empty:
        return d
    if name == "RANK1":
        return d[d["rank"] == 1]
    if name in ("TOP2", "TOP3", "TOP5"):
        return d[d["rank"] <= int(name[-1])]
    if name == "QUALIFIED":
        return d[d["qualified"].astype(bool)]
    if name == "PRODUCTION_SELECTED":
        return d[d["prod_sel"].astype(bool)]
    raise ValueError(name)


def _seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def block_bootstrap_roi(p: pd.DataFrame, key: str) -> tuple[float | None, float | None]:
    """90% CI of flat ROI, resampling whole gameweeks. None with fewer than 2 gameweeks."""
    if p.empty or p["league_week_id"].nunique() < 2:
        return None, None
    g = p.groupby("league_week_id")["pnl"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(_seed(key))
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    roi = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.quantile(roi, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return float(lo), float(hi)


def _ll(p: pd.Series, y: pd.Series) -> float:
    q = p.clip(EPS, 1 - EPS)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def _streak(won: list) -> int:
    best = cur = 0
    for w in won:
        cur = cur + 1 if w == 0 else 0
        best = max(best, cur)
    return best


def _r(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, nd)


def metrics(d: pd.DataFrame, key: str = "") -> dict:
    """Every metric for one already-subsetted, prepared frame."""
    s = d[d["settled"]] if len(d) else d
    p = s[s["priced"] & s["pnl"].notna()] if len(s) else s
    weeks = int(s["league_week_id"].nunique()) if len(s) else 0
    out = dict.fromkeys(METRIC_COLS)
    out.update(n_rows=int(len(d)), n_settled=int(len(s)), n_priced_settled=int(len(p)),
               n_insufficient=int((~d["priced"].astype(bool)).sum()) if len(d) else 0, gameweeks=weeks)
    out["status"] = "OK" if (len(p) >= MIN_N and weeks >= MIN_WEEKS) else "NOT_ENOUGH_DATA"
    if len(s):
        out["win_rate"] = _r(s["won"].mean())
    if len(p):
        out["flat_pnl"] = _r(p["pnl"].sum())
        out["roi"] = _r(p["pnl"].sum() / len(p))
        lo, hi = block_bootstrap_roi(p, key)
        out["roi_ci_lo"], out["roi_ci_hi"] = _r(lo), _r(hi)
        c = p["clv"].dropna()
        out["clv_n"] = int(len(c))
        out["clv_coverage"] = _r(len(c) / len(p))
        if len(c):
            out["clv_mean"], out["clv_median"] = _r(c.mean(), 3), _r(c.median(), 3)
            out["clv_positive_share"] = _r((c > 0).mean())
        o = p.sort_values("_ko", kind="mergesort")
        eq = o["pnl"].cumsum()
        out["max_drawdown"] = _r(float((eq.cummax().clip(lower=0) - eq).max()) if len(eq) else 0.0)
        out["longest_losing_streak"] = _streak(o["won"].tolist())
    q = s[s["p_model"].notna() & s["p_market"].notna() & s["p_dec"].notna()] if len(s) else s
    out["n_prob"] = int(len(q))
    if len(q):
        y = q["won"]
        for nm, col in (("model", "p_model"), ("market", "p_market"), ("decision", "p_dec")):
            out[f"brier_{nm}"] = _r(float(((q[col] - y) ** 2).mean()), 5)
            out[f"logloss_{nm}"] = _r(_ll(q[col], y), 5)
        out["dlogloss_model_minus_market"] = _r(out["logloss_model"] - out["logloss_market"], 5)
    return out


def league_table(panel: pd.DataFrame, prepared: bool = False) -> pd.DataFrame:
    """One row per league x market x model_type x side x subset. Sorted, deterministic."""
    d = panel if prepared else prepare(panel)
    rows = []
    if not d.empty:
        for k, g in d.groupby(GROUP, sort=True):
            for sub in SUBSETS:
                key = "|".join(map(str, k)) + "|" + sub
                rows.append({**dict(zip(GROUP, k)), "subset": sub, **metrics(subset(g, sub), key)})
    return pd.DataFrame(rows, columns=GROUP + ["subset"] + METRIC_COLS)


def rank_monotonicity(panel: pd.DataFrame, pooled: bool = False, prepared: bool = False) -> pd.DataFrame:
    """Per rank 1..5: N, win rate, ROI (priced), mean EV_LB, mean CLV. Per league x market x model_type x side,
    or POOLED_ACROSS_LEAGUES (market x model_type x side) when pooled=True."""
    d = panel if prepared else prepare(panel)
    grp = GROUP[1:] if pooled else GROUP
    cols = grp + ["rank", "n_rows", "n_settled", "win_rate", "roi", "mean_ev_lower_bound", "clv_mean", "scope"]
    rows = []
    if not d.empty:
        for k, g in d.groupby(grp, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            for rk in range(1, 6):
                x = g[g["rank"] == rk]
                s = x[x["settled"]]
                p = s[s["priced"] & s["pnl"].notna()]
                rows.append({**dict(zip(grp, k)), "rank": rk, "n_rows": int(len(x)), "n_settled": int(len(s)),
                             "win_rate": _r(s["won"].mean()) if len(s) else None,
                             "roi": _r(p["pnl"].sum() / len(p)) if len(p) else None,
                             "mean_ev_lower_bound": _r(x["ev_lb"].mean(), 5) if x["ev_lb"].notna().any() else None,
                             "clv_mean": _r(p["clv"].mean(), 3) if p["clv"].notna().any() else None,
                             "scope": "POOLED_ACROSS_LEAGUES" if pooled else "LEAGUE"})
    return pd.DataFrame(rows, columns=cols)


def absolute_vs_relative(panel: pd.DataFrame, prepared: bool = False) -> pd.DataFrame:
    """ABSOLUTE = members with ev_lower_bound > 0 (any rank); RELATIVE = rank 1 (any score)."""
    d = panel if prepared else prepare(panel)
    rows = []
    if not d.empty:
        for k, g in d.groupby(GROUP, sort=True):
            key = "|".join(map(str, k))
            ab, rel = g[g["ev_lb"] > 0], g[g["rank"] == 1]
            for nm, x in (("ABSOLUTE_EV_LB_POSITIVE", ab), ("RELATIVE_RANK1", rel)):
                m = metrics(x, f"{key}|{nm}")
                rows.append({**dict(zip(GROUP, k)), "selection": nm,
                             "overlap_rows": int(len(set(ab.index) & set(rel.index))),
                             **{c: m[c] for c in ("status", "n_rows", "n_settled", "n_priced_settled", "roi",
                                                  "roi_ci_lo", "roi_ci_hi", "clv_mean", "win_rate")}})
    return pd.DataFrame(rows, columns=GROUP + ["selection", "overlap_rows", "status", "n_rows", "n_settled",
                                               "n_priced_settled", "roi", "roi_ci_lo", "roi_ci_hi", "clv_mean",
                                               "win_rate"])


def _cell(r: pd.Series) -> str:
    res = str(r.get("result", "") or "").upper() or "pending"
    clv = r.get("clv")
    clv_s = f"CLV {clv:+.1f}%" if clv is not None and not pd.isna(clv) else "CLV -"
    p = " | PROD" if bool(r.get("prod_sel")) else ""
    side = str(r.get("side", ""))
    return f"{r.get('home_team', '')} v {r.get('away_team', '')} {side} | {res} | {clv_s} | {r.get('decision_state', '')}{p}"


def weekly_grid(panel: pd.DataFrame, league: str, market: str, prepared: bool = False) -> pd.DataFrame:
    """Gameweek (newest first) x #1..#5, each cell 'Home v Away SIDE | RESULT | CLV | state [| PROD]'."""
    d = panel if prepared else prepare(panel)
    cols = ["league_week_id"] + [f"#{i}" for i in range(1, 6)]
    if d.empty:
        return pd.DataFrame(columns=cols)
    x = d[(d["league"] == league) & (d["market"] == market)]
    rows = []
    for wk, g in x.groupby("league_week_id", sort=True):
        row = {"league_week_id": wk}
        for _, r in g.iterrows():
            row[f"#{int(r['rank'])}"] = _cell(r)
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols).fillna("")
    return out.sort_values("league_week_id", ascending=False, kind="mergesort").reset_index(drop=True)


def previous_weeks_summary(panel_league: pd.DataFrame, latest: str | None) -> dict:
    """One league's weeks BEFORE `latest`: per market Top1/Top3/Top5 ROI and N, mean CLV and market-relative
    log loss (model minus de-vigged market, Top5)."""
    d = prepare(panel_league)
    if d.empty:
        return {}
    d = d[d["league_week_id"] < latest] if latest else d
    out = {}
    for m, g in d.groupby("market", sort=True):
        t5 = metrics(subset(g, "TOP5"), f"{m}|TOP5")
        out[m] = {"gameweeks": t5["gameweeks"],
                  **{f"{s.lower()}_{k}": metrics(subset(g, s), f"{m}|{s}")[k]
                     for s in ("RANK1", "TOP3", "TOP5") for k in ("n_priced_settled", "roi")},
                  "top5_clv_mean": t5["clv_mean"], "top5_clv_coverage": t5["clv_coverage"],
                  "top5_dlogloss_model_minus_market": t5["dlogloss_model_minus_market"],
                  "status": t5["status"]}
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def evaluate(panel: pd.DataFrame) -> dict:
    d = prepare(panel)
    return {
        "evidence_type": "TRUE_FORWARD",
        "status_rule": f"NOT_ENOUGH_DATA below {MIN_N} priced settled rows or {MIN_WEEKS} gameweeks",
        "rows": int(len(d)), "settled_rows": int(d["settled"].sum()) if len(d) else 0,
        "league_table": _records(league_table(d, prepared=True)),
        "rank_monotonicity": _records(rank_monotonicity(d, prepared=True)),
        "rank_monotonicity_pooled": _records(rank_monotonicity(d, pooled=True, prepared=True)),
        "absolute_vs_relative": _records(absolute_vs_relative(d, prepared=True)),
    }
