"""The pre-registered bars optimization study (reports/P4_BARS_OPTIMIZATION.md, sections 1-12).

Cell family = (track, market-side); market-sides: over25_over, over25_under, btts_yes, over15_over, over35_over.
Configuration = probability source x edge measure x odds range (4 x 2 x 6 = 48) plus the league-wide source
NONE x odds range (6) = 54 per family (amendment 3). Inside a configuration every league gets its OWN bars from
src.league_bars.choose_bars (learned shrinkage weight, reported); a league whose bars would be a pooled stand-in
(basis SHRUNK_TO_POOL), thin, or has no bar with a positive shrunk ROI is OFF - never served a global or pooled
number (owner: "no global vars").

After a family's configuration is chosen, each league competes MODEL vs LEAGUE_WIDE on its own inner bets
(amendment 3, 12.1). Each MODEL league may carry a TIER2 band VALUABLE..MARKSMAN, only with its own positive
SELECT evidence and EV > 0 per bet (12.2).

Windows: SELECT = priced rows before 2026-09-01 (every choice); FINAL = 2026-09-01..2026-11-08, scored once with
the frozen, committed choice (choices.json; run_final refuses when a refit differs from it).

Reused from Pro (wowzaV9-Pro src/studies, ported as pure functions, credited):
  learned_tiers._design / fit   -> the LEARNED probability source (L2 logistic, C = 0.5 frozen)
  evidence                       -> max-of-three SE, White reality check (centred block bootstrap)
Every tip-deciding constant below is declared in the pre-registration (section 11) and reported.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import league_bars as LB
from src.market_edge import power_devig

# ── declared constants (pre-registration sections 11-12) ─────────────────────────────────────────
SELECT_END = pd.Timestamp("2026-09-01")          # SELECT = date < SELECT_END
FINAL_START = pd.Timestamp("2026-09-01")
FINAL_END = pd.Timestamp("2026-11-08")
SOURCES = ("RAW", "BLEND", "SHRINK", "LEARNED")
MEASURES = ("EDGE", "EV")
LEAGUE_WIDE_SOURCE = "NONE"                       # no model filter: every eligible match (amendment 3)
LEAGUE_WIDE_MEASURE = "ALL"
MAX_ODDS = (None, 2.50, 3.50)
MIN_ODDS = (None, 1.40)
INNER_BLOCKS = 5                                  # 5 consecutive blocks, blocks 2..5 scored
MIN_INNER_BETS = 100
MIN_OWN_BETS = LB.MIN_BETS_FIT                    # 40: own bets for a bar, a LEAGUE_WIDE switch, a TIER2 band
LEARNED_C = 0.5                                   # Pro learned_tiers.C, frozen
N_BOOT = 2000
MONTHLY_MIN_HISTORY_MONTHS = 6
MIN_FINAL_BETS = LB.MIN_HOLDOUT_BETS              # 150, the owner's strict rule
HOLM_ALPHA = LB.HOLM_ALPHA_ONE_SIDED              # 0.025 one-sided
MARKET_SIDES = {"over25_over": ("over25", "yes"), "over25_under": ("over25", "no"),
                "btts_yes": ("btts", "yes"), "over15_over": ("over15", "yes"), "over35_over": ("over35", "yes")}
OVER_DIRECTION = {"over25_over", "btts_yes", "over15_over", "over35_over"}   # BTTS Yes counts as Over

OFF_THIN = "INSUFFICIENT_BETS"
OFF_POOL = "OWN_DATA_TOO_THIN"
OFF_NO_VALUABLE = "NO_POSITIVE_BAR"
OFF_FAMILY = "INSUFFICIENT_SELECTION_EVIDENCE"
MODE_MODEL, MODE_LW = "MODEL", "LEAGUE_WIDE"


def configs() -> list[tuple]:
    model = [(s, m, hi, lo) for s in SOURCES for m in MEASURES for hi in MAX_ODDS for lo in MIN_ODDS]
    lw = [(LEAGUE_WIDE_SOURCE, LEAGUE_WIDE_MEASURE, hi, lo) for hi in MAX_ODDS for lo in MIN_ODDS]
    return model + lw


def cfg_name(c) -> str:
    s, m, hi, lo = c
    return f"{s}|{m}|max{hi if hi is not None else '-'}|min{lo if lo is not None else '-'}"


def parse_cfg(name: str) -> tuple:
    s, m, hi, lo = name.split("|")

    def num(x):
        return None if x.endswith("-") else float(x[3:])
    return (s, m, num(hi), num(lo))


def lw_of(cfg) -> tuple:
    return (LEAGUE_WIDE_SOURCE, LEAGUE_WIDE_MEASURE, cfg[2], cfg[3])


# ── family frames ────────────────────────────────────────────────────────────────────────────

def family_frame(u: pd.DataFrame, mside: str) -> pd.DataFrame:
    """Priced rows of one market-side with the side's model probability, fair probability, odds, closing
    odds and outcome. Rows without a real pair, an outcome or a probability are excluded, never filled."""
    market, side = MARKET_SIDES[mside]
    d = u[(u["market"] == market) & u["yes"].notna() & u["no"].notna() & u["y"].notna() & u["p"].notna()].copy()
    fair = np.array([power_devig(float(a), float(b)) for a, b in zip(d["yes"], d["no"])], dtype=object)
    ok = np.array([f is not None for f in fair], dtype=bool)
    d = d[ok].copy()
    fy = np.array([float(f) for f in fair[ok]])
    y = d["y"].to_numpy(float)
    if side == "yes":
        d["p_side"], d["fair"], d["odds"], d["odds_c"], d["won"] = d["p"].to_numpy(float), fy, d["yes"], d["yes_c"], y == 1
    else:
        d["p_side"], d["fair"], d["odds"], d["odds_c"], d["won"] = 1 - d["p"].to_numpy(float), 1 - fy, d["no"], d["no_c"], y == 0
    d["odds"] = d["odds"].astype(float)
    d["odds_c"] = d["odds_c"].astype(float)
    d["pnl"] = np.where(d["won"], d["odds"] - 1.0, -1.0)
    d["mside"] = mside
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d[["track", "mside", "league", "row_id", "date", "p_side", "fair", "odds", "odds_c", "won", "pnl"]] \
        .reset_index(drop=True)


def select_rows(f: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    return f[f["date"] < before]


# ── probability sources (fitted on training rows only) ───────────────────────────────────────

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _learned_design(d: pd.DataFrame, leagues: list[str]) -> np.ndarray:
    """Pro learned_tiers._design, one side per family (so no side column)."""
    raw = (d["p_side"] - d["fair"]).to_numpy(float)
    base = np.c_[_logit(d["fair"]), _logit(d["p_side"]), raw, np.log(d["odds"].to_numpy(float))]
    lg = np.stack([(d["league"] == x).to_numpy(float) for x in leagues], axis=1)
    return np.c_[base, lg, lg * raw[:, None]]


def fit_source(src: str, t: pd.DataFrame):
    """Returns a function rows -> p_used (NaN where the source has no fitted value for the row)."""
    if src in ("RAW",):
        return lambda e: e["p_side"].to_numpy(float)
    if src == LEAGUE_WIDE_SOURCE:
        return lambda e: np.full(len(e), np.nan)
    if src == "BLEND":
        from sklearn.linear_model import LogisticRegression
        X = np.c_[_logit(t["p_side"]), _logit(t["fair"])]
        m = LogisticRegression(penalty=None, max_iter=2000).fit(X, t["won"].astype(int))
        return lambda e: m.predict_proba(np.c_[_logit(e["p_side"]), _logit(e["fair"])])[:, 1] if len(e) else np.array([])
    if src == "SHRINK":
        w = shrink_weights(t)
        return lambda e: np.where(e["league"].isin(list(w)).to_numpy(),
                                  e["fair"].to_numpy(float) + e["league"].map(w).to_numpy(float)
                                  * (e["p_side"] - e["fair"]).to_numpy(float), np.nan)
    if src == "LEARNED":
        from sklearn.linear_model import LogisticRegression
        leagues = sorted(t["league"].unique())
        m = LogisticRegression(C=LEARNED_C, max_iter=3000).fit(_learned_design(t, leagues), t["won"].astype(int))

        def pred(e):
            out = np.full(len(e), np.nan)
            known = e["league"].isin(leagues).to_numpy()      # a league unseen in training: no value
            if known.any():
                out[known] = m.predict_proba(_learned_design(e[known], leagues))[:, 1]
            return out
        return pred
    raise ValueError(src)


def shrink_weights(t: pd.DataFrame) -> dict:
    """Per league w in p = fair + w (p_model - fair): least squares of (won - fair) on (p_model - fair),
    shrunk toward the inverse-variance mean by a DerSimonian-Laird tau^2 learned from the leagues; clipped
    to [0, 1]. A league with < 3 rows or no spread has no estimate (and no SHRINK value)."""
    est = {}
    for lg, g in t.groupby("league"):
        dd = (g["p_side"] - g["fair"]).to_numpy(float)
        r = g["won"].to_numpy(float) - g["fair"].to_numpy(float)
        sxx = float((dd ** 2).sum())
        if len(g) < 3 or sxx <= 0:
            continue
        w = float((dd * r).sum() / sxx)
        s2 = float(((r - w * dd) ** 2).sum() / (len(g) - 1))
        est[lg] = (w, s2 / sxx)
    if not est:
        return {}
    ws = np.array([v[0] for v in est.values()])
    vs = np.array([max(v[1], 1e-12) for v in est.values()])
    iw = 1 / vs
    mu = float((iw * ws).sum() / iw.sum())
    k = len(ws)
    q = float((iw * (ws - mu) ** 2).sum())
    c = float(iw.sum() - (iw ** 2).sum() / iw.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if (k >= 2 and c > 0) else 0.0
    return {lg: float(np.clip(mu + (tau2 / (tau2 + v[1]) if tau2 + v[1] > 0 else 0.0) * (v[0] - mu), 0, 1))
            for lg, v in est.items()}


def measure(e: pd.DataFrame, p_used: np.ndarray, how: str) -> np.ndarray:
    if how == "EDGE":
        return p_used - e["fair"].to_numpy(float)
    if how == "EV":
        return p_used * e["odds"].to_numpy(float) - 1.0
    if how == LEAGUE_WIDE_MEASURE:              # league-wide: every row clears every bar of the grid
        return np.full(len(e), np.inf)
    raise ValueError(how)


def in_odds_range(e: pd.DataFrame, hi, lo) -> np.ndarray:
    o = e["odds"].to_numpy(float)
    ok = np.ones(len(e), bool)
    if hi is not None:
        ok &= o <= hi
    if lo is not None:
        ok &= o >= lo
    return ok


# ── bars per league (src.league_bars, own data only) + TIER2 ────────────────────────────────

def league_bars(t: pd.DataFrame, league_wide: bool) -> dict:
    """t: training rows with league, edge, pnl, p_used, odds. {league: bars | {'off': reason}}."""
    leagues = sorted(t["league"].unique())
    curves, pooled, tau2 = LB._bars_for(t, leagues)
    out = {}
    for lg in leagues:
        b = LB.choose_bars(curves[lg], pooled, tau2)
        if b["status"] == LB.STATUS_THIN:
            out[lg] = {"off": OFF_THIN}
        elif b["bar_basis"] != "OWN_BETS":
            out[lg] = {"off": OFF_POOL, "own_weight": b["shrink_weight_at_valuable"]}
        elif b["note"]:
            out[lg] = {"off": OFF_NO_VALUABLE}
        else:
            b = {k: b[k] for k in ("valuable_th", "marksman_th", "sniper_th", "bar_basis",
                                    "shrink_weight_at_valuable", "wf_bets", "wf_roi", "wf_roi_shrunk")}
            if league_wide:
                b |= {"tier2": False, "tier2_bets": 0, "tier2_roi": None}
            else:
                g = t[t["league"] == lg]
                ev = g["p_used"].to_numpy(float) * g["odds"].to_numpy(float) - 1.0
                band = ((g["edge"] >= b["valuable_th"]) & (g["edge"] < b["marksman_th"])).to_numpy() & (ev > 0)
                n = int(band.sum())
                roi = float(g.loc[band, "pnl"].mean()) if n else None
                b |= {"tier2": bool(n >= MIN_OWN_BETS and roi is not None and roi > 0), "tier2_bets": n,
                      "tier2_roi": None if roi is None else round(roi, 4)}
            out[lg] = b
    return out


def fit_cfg(t: pd.DataFrame, cfg, src_fns: dict | None = None):
    """(bars, scorer) for cfg fitted on t. scorer(rows) -> rows with p_used / edge (odds range applied)."""
    src, how, hi, lo = cfg
    fn = (src_fns or {}).get(src) or fit_source(src, t)

    def scorer(e):
        ee = e[in_odds_range(e, hi, lo)].copy()
        ee["p_used"] = fn(ee) if len(ee) else np.array([])
        ee["edge"] = measure(ee, ee["p_used"].to_numpy(float), how) if len(ee) else np.array([])
        return ee[~np.isnan(ee["edge"].to_numpy(float))] if len(ee) else ee
    tt = scorer(t)
    bars = league_bars(tt, src == LEAGUE_WIDE_SOURCE) if len(tt) else {}
    return bars, scorer


def apply_bars(ee: pd.DataFrame, bars: dict, mode: str) -> pd.DataFrame:
    """Rows of ee at or above their league's VALUABLE bar (OFF / unseen leagues: no bet), with the tier,
    the band (TOP_M = at or above MARKSMAN; BAND_VM = VALUABLE..MARKSMAN), TIER2 eligibility and the
    expected value used for ranking (model: p_used * odds - 1; LEAGUE_WIDE: the league's shrunk SELECT ROI)."""
    if ee.empty:
        return ee.assign(tier=pd.Series(dtype=object), band=pd.Series(dtype=object), tier2_ok=pd.Series(dtype=bool),
                         ev_est=pd.Series(dtype=float), mode=pd.Series(dtype=object))
    keep = np.zeros(len(ee), bool)
    tier = np.full(len(ee), "", dtype=object)
    band = np.full(len(ee), "", dtype=object)
    t2 = np.zeros(len(ee), bool)
    ev = np.full(len(ee), np.nan)
    lg, ed = ee["league"].to_numpy(), ee["edge"].to_numpy(float)
    evm = ee["p_used"].to_numpy(float) * ee["odds"].to_numpy(float) - 1.0
    for name, b in bars.items():
        if "off" in b:
            continue
        m = (lg == name) & (ed >= b["valuable_th"])
        keep |= m
        if mode == MODE_LW:
            tier[m], band[m] = MODE_LW, "TOP_M"
            ev[m] = b["wf_roi_shrunk"]
            continue
        tier[m & (ed >= b["sniper_th"])] = "SNIPER"
        tier[m & (ed >= b["marksman_th"]) & (ed < b["sniper_th"])] = "MARKSMAN"
        tier[m & (ed < b["marksman_th"])] = "VALUABLE"
        band[m & (ed >= b["marksman_th"])] = "TOP_M"
        band[m & (ed < b["marksman_th"])] = "BAND_VM"
        t2[m & (ed < b["marksman_th"]) & (evm > 0)] = bool(b["tier2"])
        ev[m] = evm[m]
    out = ee[keep].copy()
    out["tier"], out["band"], out["tier2_ok"], out["ev_est"] = tier[keep], band[keep], t2[keep], ev[keep]
    out["mode"] = mode
    return out


# ── nested inner validation and the choice ───────────────────────────────────────────────────

def inner_blocks(dates: pd.Series) -> list[pd.Timestamp]:
    """Start dates of blocks 2..INNER_BLOCKS (equal numbers of matchdays, time order) plus the end."""
    days = np.sort(pd.to_datetime(dates).unique())
    if len(days) < INNER_BLOCKS * 2:
        return []
    cuts = np.array_split(days, INNER_BLOCKS)
    return [pd.Timestamp(c[0]) for c in cuts[1:]] + [pd.Timestamp(days[-1]) + pd.Timedelta(days=1)]


def inner_validate(sel: pd.DataFrame) -> dict:
    """Every configuration's inner out-of-sample bets over blocks 2..5. {cfg_name: bets frame}."""
    starts = inner_blocks(sel["date"])
    res = {cfg_name(c): [] for c in configs()}
    for k in range(len(starts) - 1):
        t = sel[sel["date"] < starts[k]]
        e = sel[(sel["date"] >= starts[k]) & (sel["date"] < starts[k + 1])]
        if t.empty or e.empty:
            continue
        fns = {s: fit_source(s, t) for s in SOURCES + (LEAGUE_WIDE_SOURCE,)}
        for c in configs():
            bars, scorer = fit_cfg(t, c, fns)
            mode = MODE_LW if c[0] == LEAGUE_WIDE_SOURCE else MODE_MODEL
            res[cfg_name(c)].append(apply_bars(scorer(e), bars, mode).assign(fold=k))
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in res.items()}


def reality_check(bets: dict, seed: int = 0) -> dict:
    """White's reality check (Pro src/studies/evidence: max over configurations, centred block bootstrap
    over match dates) on the inner ROI of configurations with >= MIN_INNER_BETS bets."""
    eligible = {k: v for k, v in bets.items() if len(v) >= MIN_INNER_BETS}
    if not eligible:
        return {"configs": 0, "p": None}
    days = sorted(set().union(*[set(v["date"]) for v in eligible.values()]))
    di = {d: i for i, d in enumerate(days)}
    S = np.zeros((len(eligible), len(days)))
    N = np.zeros((len(eligible), len(days)))
    names = list(eligible)
    for j, k in enumerate(names):
        g = eligible[k].groupby("date")["pnl"].agg(["sum", "count"])
        for d, r in g.iterrows():
            S[j, di[d]] += r["sum"]
            N[j, di[d]] += r["count"]
    roi = S.sum(1) / N.sum(1)
    rng = np.random.default_rng(seed)
    best = roi.max()
    hits = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(days), len(days))
        n = N[:, idx].sum(1)
        r = np.where(n > 0, S[:, idx].sum(1) / np.where(n > 0, n, 1), roi)
        hits += int((r - roi).max() >= best)
    return {"configs": len(names), "best": names[int(roi.argmax())], "best_roi": round(float(best), 4),
            "p": round((hits + 1) / (N_BOOT + 1), 4)}


def choose(bets: dict) -> tuple[tuple | None, pd.DataFrame]:
    tab = pd.DataFrame([{"config": k, "bets": len(v), "roi": float(v["pnl"].mean()) if len(v) else np.nan}
                        for k, v in bets.items()])
    elig = tab[tab["bets"] >= MIN_INNER_BETS]
    if elig.empty:
        return None, tab
    name = elig.sort_values(["roi", "bets"], ascending=[False, False]).iloc[0]["config"]
    return parse_cfg(name), tab


def select_core(sel: pd.DataFrame, seed: int = 0) -> dict:
    """The whole pre-registered choice on `sel`: the family configuration, then per league MODEL vs
    LEAGUE_WIDE on its own inner bets, then the frozen bars (with mode and TIER2)."""
    bets = inner_validate(sel) if len(sel) else {}
    rc = reality_check(bets, seed) if bets else {"configs": 0, "p": None}
    chosen, tab = choose(bets) if bets else (None, pd.DataFrame())
    out = {"reality_check": rc, "inner_table": tab.to_dict("records") if len(tab) else [],
           "chosen": cfg_name(chosen) if chosen else None, "off": None if chosen else OFF_FAMILY,
           "bars": {}, "league_switch": []}
    if chosen is None:
        return out
    lw = lw_of(chosen)
    model_bars, _ = fit_cfg(sel, chosen)
    lw_bars, _ = fit_cfg(sel, lw)
    mb, lb = bets[cfg_name(chosen)], bets[cfg_name(lw)]
    for lg in sorted(set(model_bars) | set(lw_bars)):
        use_lw = chosen[0] == LEAGUE_WIDE_SOURCE
        if not use_lw:
            m_l, l_l = mb[mb["league"] == lg] if len(mb) else mb, lb[lb["league"] == lg] if len(lb) else lb
            if (len(m_l) >= MIN_OWN_BETS and len(l_l) >= MIN_OWN_BETS and l_l["pnl"].mean() > m_l["pnl"].mean()
                    and lg in lw_bars and "off" not in lw_bars[lg]):
                use_lw = True
                out["league_switch"].append({"league": lg, "model_inner": [len(m_l), round(float(m_l["pnl"].mean()), 4)],
                                             "league_wide_inner": [len(l_l), round(float(l_l["pnl"].mean()), 4)]})
        src = lw_bars if use_lw else model_bars
        if lg in src:
            out["bars"][lg] = src[lg] | {"mode": MODE_LW if use_lw else MODE_MODEL}
    return out


def bet_with(sel: pd.DataFrame, e: pd.DataFrame, choice: dict) -> tuple[pd.DataFrame, dict]:
    """Bets on rows e under a frozen choice (refitted on sel, which must be the same data the choice was
    made on). Returns (bets, recomputed bars) so callers can verify the refit equals the frozen choice."""
    if not choice.get("chosen"):
        return pd.DataFrame(), {}
    chosen = parse_cfg(choice["chosen"])
    model_bars, m_scorer = fit_cfg(sel, chosen)
    lw_bars, l_scorer = fit_cfg(sel, lw_of(chosen))
    rebuilt, parts = {}, []
    for lg, b in choice["bars"].items():
        src = lw_bars if b["mode"] == MODE_LW else model_bars
        if lg in src:
            rebuilt[lg] = src[lg] | {"mode": b["mode"]}
    for mode, bars_all, scorer in ((MODE_MODEL, model_bars, m_scorer), (MODE_LW, lw_bars, l_scorer)):
        mine = {lg: b for lg, b in choice["bars"].items() if b["mode"] == mode}
        if not mine or e.empty:
            continue
        ee = scorer(e[e["league"].isin(list(mine))])
        parts.append(apply_bars(ee, {lg: bars_all[lg] for lg in mine if lg in bars_all}, mode))
    keep = [p for p in parts if len(p)]
    return (pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()), rebuilt


# ── statistics ───────────────────────────────────────────────────────────────────────────────

def max_se(pnl: np.ndarray, days: np.ndarray) -> float:
    """Pro evidence: the LARGEST of the independent, matchday-block and ISO-week-block SEs of the mean."""
    n = len(pnl)
    if n < 2:
        return float("nan")
    ses = [float(np.std(pnl, ddof=1) / math.sqrt(n))]
    mu = pnl.mean()
    for key in (days, pd.to_datetime(days).strftime("%G-%V")):
        s = pd.Series(pnl).groupby(np.asarray(key))
        tot, cnt = s.sum(), s.size()
        k = len(tot)
        if k >= 2:
            ses.append(float(math.sqrt(((tot - mu * cnt) ** 2).sum() * k / (k - 1)) / n))
    return max(ses)


def score(b: pd.DataFrame, seed: int = 0) -> dict:
    if b is None or b.empty:
        return {"bets": 0, "roi": None, "ci95": [None, None], "p_one_sided": None, "clv": None, "clv_n": 0}
    pnl = b["pnl"].to_numpy(float)
    days = b["date"].dt.strftime("%Y-%m-%d").to_numpy()
    lo, hi = LB.block_ci(pnl, days)
    se = max_se(pnl, days)
    m = float(pnl.mean())
    cands_lo = [x for x in (lo, m - 1.96 * se) if np.isfinite(x)]
    cands_hi = [x for x in (hi, m + 1.96 * se) if np.isfinite(x)]
    c = b[b["odds_c"].notna() & (b["odds_c"] > 1)]
    clv = (c["odds"] / c["odds_c"] - 1.0) if len(c) else pd.Series(dtype=float)
    p = LB.one_sided_p(pnl, days, seed=seed)
    return {"bets": int(len(b)), "roi": round(m, 4), "matchdays": int(len(np.unique(days))),
            "ci95": [round(float(min(cands_lo)), 4) if cands_lo else None,
                     round(float(max(cands_hi)), 4) if cands_hi else None],
            "p_one_sided": None if not np.isfinite(p) else round(float(p), 4),
            "clv": round(float(clv.mean()), 4) if len(clv) else None, "clv_n": int(len(clv))}


def verdicts(cells: list[dict]) -> None:
    """Holm across every tested cell, then PROVEN / PROMISING / UNPROVEN (OFF / NO_PRICE_DATA stay)."""
    tested = [i for i, r in enumerate(cells) if r.get("verdict") is None and r["final"]["bets"] > 0]
    ph = LB.holm(np.array([np.nan if cells[i]["final"]["p_one_sided"] is None else cells[i]["final"]["p_one_sided"]
                           for i in tested], float))
    for i, adj in zip(tested, ph):
        r, f = cells[i], cells[i]["final"]
        r["p_holm"] = None if not np.isfinite(adj) else round(float(adj), 4)
        lo = f["ci95"][0]
        if (f["bets"] >= MIN_FINAL_BETS and r["p_holm"] is not None and r["p_holm"] < HOLM_ALPHA
                and lo is not None and lo > 0):
            r["verdict"] = "PROVEN"
        elif f["roi"] is not None and f["roi"] > 0 and (f["clv"] is None or f["clv"] > 0):
            r["verdict"] = "PROMISING"
        else:
            r["verdict"] = "UNPROVEN"
    for r in cells:
        if r.get("verdict") is None:
            r["verdict"] = "UNPROVEN"
    for r in cells:
        r["family_size"] = len(tested)


# ── TOP-3 per league per gameweek ────────────────────────────────────────────────────────────

TOP3_VARIANTS = ("TOP_ONLY_V", "TOP_ONLY_M", "TOP_M_PLUS_TIER2")


def top3(bets: pd.DataFrame, variant: str = "TOP_ONLY_V") -> pd.DataFrame:
    """At most 3 per (track, league, ISO week) across market-sides; direction rule on one fixture (Under 2.5
    never beside an Over-direction tip); never padded.
      TOP_ONLY_V       every bet at or above VALUABLE ranks by expected value (the 2026-11-08 rule)
      TOP_ONLY_M       only bets at or above MARKSMAN (LEAGUE_WIDE bets count as TOP)
      TOP_M_PLUS_TIER2 TOP_M bets first, then TIER2 band bets (labelled TIER2) fill empty slots"""
    if bets is None or bets.empty:
        return pd.DataFrame()
    b = bets.copy()
    b["gw"] = b["date"].dt.strftime("%G-W%V")
    if variant == "TOP_ONLY_V":
        b["slot"] = "TOP"
    elif variant == "TOP_ONLY_M":
        b = b[b["band"] == "TOP_M"].assign(slot="TOP")
    elif variant == "TOP_M_PLUS_TIER2":
        b = b[(b["band"] == "TOP_M") | b["tier2_ok"].astype(bool)].copy()
        b["slot"] = np.where(b["band"] == "TOP_M", "TOP", "TIER2")
    else:
        raise ValueError(variant)
    b["slot_rank"] = (b["slot"] == "TIER2").astype(int)
    picks = []
    order = b.sort_values(["slot_rank", "ev_est", "date"], ascending=[True, False, True])
    for _, g in order.groupby(["track", "league", "gw"], sort=False):
        taken, dirs = [], {}
        for r in g.itertuples(index=False):
            d = "OVER" if r.mside in OVER_DIRECTION else "UNDER"
            if dirs.get(r.row_id, d) != d:
                continue
            dirs[r.row_id] = d
            taken.append(r)
            if len(taken) == 3:
                break
        picks += taken
    return pd.DataFrame(picks, columns=order.columns)


def top3_summary(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"picks": 0}
    s = score(t)
    s.pop("p_one_sided", None)
    gw = t["gw"] if "gw" in t else t["date"].dt.strftime("%G-W%V")
    by_m = {m: {"picks": int(len(g)), "roi": round(float(g["pnl"].mean()), 4)} for m, g in t.groupby("mside")}
    by_l = {f"{tr}|{lg}": {"picks": int(len(g)), "roi": round(float(g["pnl"].mean()), 4)}
            for (tr, lg), g in t.groupby(["track", "league"])}
    out = s | {"league_weeks": int(pd.DataFrame({"a": t["track"], "b": t["league"], "c": gw}).drop_duplicates().shape[0]),
               "by_market": by_m, "by_league": by_l}
    if "slot" in t:
        out["by_slot"] = {k: score(g) | {"p_one_sided": None} for k, g in t.groupby("slot")}
    return out


# ── phase 1: SELECT (choices frozen and committed before FINAL is read) ───────────────────

def load_families(universe_dir: Path) -> list[tuple]:
    fams = []
    for track in ("standard", "new_format"):
        p = universe_dir / f"universe_{track}.parquet"
        if not p.exists():
            continue
        u = pd.read_parquet(p)
        for mside in MARKET_SIDES:
            fams.append((family_frame(u, mside), track, mside))
    return fams


def baseline_bars(sel: pd.DataFrame) -> dict:
    """The -13.3% method: RAW / EDGE / no caps, bars on SELECT, every league with bars ON (no OFF rules)."""
    bb = {}
    if len(sel):
        tt = sel.assign(edge=sel["p_side"] - sel["fair"])
        leagues = sorted(tt["league"].unique())
        curves, pooled, tau2 = LB._bars_for(tt, leagues)
        for lg in leagues:
            x = LB.choose_bars(curves[lg], pooled, tau2)
            if x["status"] != LB.STATUS_THIN:
                bb[lg] = {k: x[k] for k in ("valuable_th", "marksman_th", "sniper_th", "wf_roi_shrunk")} | {"tier2": False}
    return bb


def select_family(args) -> dict:
    f, track, mside, seed = args
    sel = select_rows(f, SELECT_END)
    core = select_core(sel, seed)
    return {"track": track, "mside": mside, "select_rows": int(len(sel)),
            "select_leagues": sorted(sel["league"].unique()), "configs_searched": len(configs())} | core | \
           {"baseline_bars": baseline_bars(sel)}


def run_select(universe_dir: Path, out_dir: Path, seed: int = 0, workers: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = load_families(universe_dir)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(select_family, [(f, t, m, seed) for f, t, m in fams]))
    for r in res:
        pd.DataFrame(r.pop("inner_table")).to_csv(out_dir / f"inner_{r['track']}_{r['mside']}.csv", index=False)
    (out_dir / "choices.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return {"families": [{k: r[k] for k in ("track", "mside", "select_rows", "chosen", "off", "reality_check",
                                            "league_switch")} for r in res],
            "configs_total": sum(r["configs_searched"] for r in res)}


# ── phase 2: FINAL, scored once with the frozen choices ──────────────────────────────────────

def _norm(x):
    return json.loads(json.dumps(x, default=str))


def final_family(args) -> dict:
    f, choice = args
    sel = select_rows(f, SELECT_END)
    fin = f[(f["date"] >= FINAL_START) & (f["date"] <= FINAL_END)]
    fb, rebuilt = bet_with(sel, fin, choice)
    if _norm(rebuilt) != _norm(choice["bars"]):
        raise RuntimeError(f"{choice['track']}/{choice['mside']}: refit bars differ from the committed choice")
    ee = fin.assign(p_used=fin["p_side"], edge=fin["p_side"] - fin["fair"])
    bb = apply_bars(ee, choice["baseline_bars"], MODE_MODEL) if len(ee) else pd.DataFrame()
    return {"track": choice["track"], "mside": choice["mside"], "chosen": choice["chosen"], "off": choice["off"],
            "bars": choice["bars"], "select_leagues": choice["select_leagues"],
            "final_leagues": sorted(fin["league"].unique()), "final_rows": int(len(fin)),
            "final_bets": fb, "baseline_bets": bb}


def _cells(res: dict) -> list[dict]:
    cells = []
    fb = res["final_bets"]
    for lg in sorted(set(res["select_leagues"]) | set(res["final_leagues"])):
        b = res["bars"].get(lg)
        cell = {"track": res["track"], "mside": res["mside"], "league": lg, "config": res["chosen"]}
        empty = score(pd.DataFrame())
        if res["chosen"] is None:
            cell |= {"verdict": "OFF", "reason": res["off"], "final": empty}
        elif b is None:
            cell |= {"verdict": "OFF", "reason": "NO_SELECT_ROWS", "final": empty}
        elif "off" in b:
            cell |= {"verdict": "OFF", "reason": b["off"], "mode": b.get("mode"), "final": empty}
            if "own_weight" in b:
                cell["own_weight"] = b["own_weight"]
        else:
            cell |= {k: b[k] for k in ("mode", "valuable_th", "marksman_th", "sniper_th", "shrink_weight_at_valuable",
                                       "wf_bets", "wf_roi", "wf_roi_shrunk", "tier2", "tier2_bets", "tier2_roi")}
            mine = fb[fb["league"] == lg] if len(fb) else pd.DataFrame()
            cell["final"] = score(mine)
            cell["final_tier2"] = score(mine[mine["tier2_ok"].astype(bool)]) if len(mine) else empty
            cell["verdict"] = None
        cells.append(cell)
    return cells


def run_final(universe_dir: Path, out_dir: Path, workers: int = 6) -> dict:
    raw = json.loads((out_dir / "choices.json").read_text(encoding="utf-8"))
    choices = {(c["track"], c["mside"]): c for c in raw}
    fams = load_families(universe_dir)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(final_family, [(f, choices[(t, m)]) for f, t, m in fams]))
    cells = [c for r in results for c in _cells(r)]
    verdicts(cells)
    keep = [r["final_bets"] for r in results if len(r["final_bets"])]
    fb = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    keepb = [r["baseline_bets"] for r in results if len(r["baseline_bets"])]
    bb = pd.concat(keepb, ignore_index=True) if keepb else pd.DataFrame()
    tops = {v: top3(fb, v) for v in TOP3_VARIANTS}
    t_base = top3(bb, "TOP_ONLY_V")
    t2 = fb[fb["tier2_ok"].astype(bool)] if len(fb) else pd.DataFrame()
    summary = {
        "families": len(results), "configs_total": len(configs()) * len(results),
        "verdict_counts": pd.DataFrame(cells).groupby(["track", "mside", "verdict"]).size()
                          .rename("n").reset_index().to_dict("records"),
        "all_final_bets": {"optimized": top3_summary(fb), "baseline": top3_summary(bb),
                           "tier2_band": top3_summary(t2)},
        "top3_final": {v: top3_summary(t) for v, t in tops.items()} | {"baseline": top3_summary(t_base)},
    }
    pd.DataFrame(cells).to_json(out_dir / "cells.json", orient="records", indent=1, default_handler=str)
    for v, t in tops.items():
        t.to_csv(out_dir / f"top3_final_{v}.csv", index=False)
    t_base.to_csv(out_dir / "top3_final_baseline.csv", index=False)
    fb.to_parquet(out_dir / "final_bets.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return {"summary": summary, "cells": cells}


# ── monthly re-selection (pre-registration 10.3) ─────────────────────────────────────────────

def _month_bets(f: pd.DataFrame, choice: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sel = f[f["date"] < start]
    e = f[(f["date"] >= start) & (f["date"] < end)]
    if e.empty or not choice.get("chosen"):
        return pd.DataFrame()
    b, _ = bet_with(sel, e, choice)
    return b


def _static_bets(f: pd.DataFrame, choice: dict, chosen_on: pd.Timestamp, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    sel = f[f["date"] < chosen_on]
    e = f[(f["date"] >= start) & (f["date"] < end)]
    if e.empty or not choice.get("chosen"):
        return pd.DataFrame()
    b, _ = bet_with(sel, e, choice)
    return b


def monthly_sim_family(args) -> dict:
    """(a) over SELECT: months from MONTHLY_MIN_HISTORY_MONTHS after the first priced month up to August
    2026; MONTHLY re-selection vs STATIC (chosen once before the first simulated month)."""
    f, track, mside = args
    sel = f[f["date"] < SELECT_END]
    base = {"track": track, "mside": mside, "months": [], "monthly_bets": pd.DataFrame(), "static_bets": pd.DataFrame()}
    if sel.empty:
        return base
    first = (sel["date"].min() + pd.DateOffset(months=MONTHLY_MIN_HISTORY_MONTHS)).to_period("M").to_timestamp()
    months = list(pd.date_range(first, pd.Timestamp("2026-08-01"), freq="MS"))
    if not months:
        return base
    static = select_core(sel[sel["date"] < months[0]])
    rows, mb, sb = [], [], []
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        ch = select_core(sel[sel["date"] < m])
        bm = _month_bets(sel, ch, m, nxt)
        bs = _static_bets(sel, static, months[0], m, nxt)
        if len(bm):
            mb.append(bm.assign(month=str(m.date())[:7]))
        if len(bs):
            sb.append(bs.assign(month=str(m.date())[:7]))
        rows.append({"month": str(m.date())[:7], "monthly_config": ch["chosen"],
                     "monthly_bets": int(len(bm)), "monthly_pnl": round(float(bm["pnl"].sum()), 3) if len(bm) else 0.0,
                     "static_bets": int(len(bs)), "static_pnl": round(float(bs["pnl"].sum()), 3) if len(bs) else 0.0})
    return {"track": track, "mside": mside, "static_config": static["chosen"], "months": rows,
            "monthly_bets": pd.concat(mb, ignore_index=True) if mb else pd.DataFrame(),
            "static_bets": pd.concat(sb, ignore_index=True) if sb else pd.DataFrame()}


def monthly_final_family(args) -> dict:
    """(b) the monthly cycle on FINAL: Sep with the SELECT choice, Oct re-selected on data < Oct, Nov 1-8 on
    data < Nov. Labelled MONTHLY_ROLLING; the main verdict stays STATIC."""
    f, track, mside = args
    out = []
    for m in (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")):
        end = min(m + pd.offsets.MonthBegin(1), FINAL_END + pd.Timedelta(days=1))
        ch = select_core(f[f["date"] < m])
        b = _month_bets(f, ch, m, end)
        if len(b):
            out.append(b.assign(month=str(m.date())[:7], config=ch["chosen"]))
    return {"track": track, "mside": mside, "bets": pd.concat(out, ignore_index=True) if out else pd.DataFrame()}


def run_monthly(universe_dir: Path, out_dir: Path, workers: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = load_families(universe_dir)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sims = list(ex.map(monthly_sim_family, [(f, t, m) for f, t, m in fams]))
        fins = list(ex.map(monthly_final_family, [(f, t, m) for f, t, m in fams]))
    cat = (lambda xs: pd.concat([x for x in xs if len(x)], ignore_index=True) if any(len(x) for x in xs) else pd.DataFrame())
    mb, sb = cat([s["monthly_bets"] for s in sims]), cat([s["static_bets"] for s in sims])
    fb = cat([x["bets"] for x in fins])
    res = {"sim_select": {"monthly": score(mb), "static": score(sb),
                          "monthly_top3": {v: top3_summary(top3(mb, v)) for v in TOP3_VARIANTS},
                          "static_top3": {v: top3_summary(top3(sb, v)) for v in TOP3_VARIANTS},
                          "by_month": {m: {"monthly": score(mb[mb["month"] == m]), "static": score(sb[sb["month"] == m])}
                                       for m in sorted(set(mb["month"]) | set(sb["month"]))} if len(mb) or len(sb) else {},
                          "families": [{k: s[k] for k in ("track", "mside", "static_config", "months") if k in s}
                                       for s in sims]},
           "final_monthly_rolling": {"bets": score(fb),
                                     "top3": {v: top3_summary(top3(fb, v)) for v in TOP3_VARIANTS},
                                     "by_month": {m: score(g) for m, g in fb.groupby("month")} if len(fb) else {}}}
    (out_dir / "monthly.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def monthly_proposal(universe_dir: Path, month: str, out_dir: Path) -> dict:
    """One monthly re-selection (the scheduled job): every family chosen on all priced rows before the month;
    per-league bars written as a PROPOSAL candidate (never a production file)."""
    m = pd.Timestamp(f"{month}-01")
    props = []
    for f, track, mside in load_families(universe_dir):
        ch = select_core(f[f["date"] < m])
        if not ch["chosen"]:
            props.append({"month": month, "track": track, "mside": mside, "league": None, "off": OFF_FAMILY})
        for lg, b in sorted(ch["bars"].items()):
            props.append({"month": month, "track": track, "mside": mside, "league": lg, "config": ch["chosen"]} | b)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"bars_opt_{month}.json"
    p.write_text(json.dumps(props, indent=1, default=str), encoding="utf-8")
    return {"month": month, "entries": len(props), "path": str(p)}
