"""Quality layers - Best / Second / Third ranked by LEARNED expected profit, not by gap size (owner 2026-10-26).

    python -m src.research.quality_tiers --study --snapshots <prematch_snapshots.parquet>
        -> docs/quality_tiers_2026_10/*.csv|json   (discovery / holdout tables, seed file)
    python -m src.research.quality_tiers --shadow
        -> output/quality_tier_shadow.csv           (first-sight log of every NEW tip; research only)

RESEARCH + SHADOW ONLY. Nothing here changes a tip, a tier, a stake, a threshold, a ledger row or a
Telegram message. The live tiers (SNIPER / MARKSMAN / VALUABLE) are assigned by the size of the gap
between the model and the bookmaker; measured this season the biggest gap is not the best tip on
O/U 2.5. This module asks a narrower question: if tips were layered by a learned estimate of how
good they are, would the top layer actually beat the bottom one, market by market?

THE SCORE (frozen here, before any holdout result; docs/QUALITY_TIERS_STUDY_2026_10.md)
  p_win   L2 logistic regression (C=0.5, standardised inputs) of WIN on, per tip:
            lf    logit(fair market probability): de-vigged two-sided where the opposite price exists
                  (O/U 2.5 always; BTTS when the No price exists), else 1/odds / 1.08 (production's own
                  side-market overround)
            lgap  logit(model p) - logit(fair p)  - the gap, on the log-odds scale
            long  odds >= 3.00
            lrate league rating AS KNOWN AT THE TIME: the league's shrunk past flat ROI on that track x
                  market, sum(pnl) / (n + 50), from tips settled strictly before the day
            market dummies (BTTS / Over 1.5 / Over 3.5 vs O/U 2.5), an Under dummy, and market x lf,
            market x lgap interactions (so the gap may rank BTTS while not ranking O/U 2.5)
          ONE MODEL PER TRACK (standard / new_format) - the two are never pooled (invariant 1).
  fit     walk-forward: tips first seen on day D are scored by a model fitted ONLY on tips whose match
          was played strictly before D (settled WIN/LOSS). Needs >= MIN_TRAIN rows, else no score.
  score   expected profit = p_win x odds - 1 (flat 1u).

THE LAYERS (both definitions frozen here; the choice is made on the DISCOVERY window only)
  (a) TERCILE  per tip day, within track x market: rank by score, top third BEST, middle SECOND,
               bottom THIRD (rank percentile (r - 0.5) / n; a one-tip day is SECOND).
  (b) FIXED    two score cut-offs per track x market = the 1/3 and 2/3 quantiles of the walk-forward
               scores in the discovery window; applied unchanged afterwards.
  Choice rule: per cell, the definition with the larger discovery Best - Third ROI difference.

VERDICT per track x market on the untouched HOLDOUT (frozen before it was looked at)
  REAL_3_LAYERS    holdout ROI Best > Second > Third (point estimates) AND the matchday block-bootstrap
                   one-sided p of (Best - Third <= 0), BH-adjusted across evaluated cells, < 0.10
  ONE_LAYER        everything else - the cell is shown as a single layer ("Tip"); sub-reason
                   ORDER_SUGGESTIVE (ordered but not significant), ORDER_FAILS, or INSUFFICIENT
                   (holdout n < 45 or the model never had enough history to score the cell).

DATA: the tips actually sent (both v9 ledgers, live, paper leagues dropped, one per fixture x market -
the same population as src/top_picks.load_tips) with every feature taken at FIRST SIGHT from the
committed predictions.csv snapshot of the run that created the tip. The ledgers upgrade tier/edge (and
side-market odds) in place, so ledger values are never used where a first-sight value exists.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

VERSION = "qt_v1"
C = 0.5                      # L2 strength - same frozen value as Pro learned_tiers; never tuned
MIN_TRAIN = 150              # settled tips per track before a day can be scored
LEAGUE_K = 50.0              # league-rating shrinkage (pseudo-tips at ROI 0)
LONG_ODDS = 3.00
SIDE_OVERROUND = 1.08        # pipeline._generate_side_bets
SEASON_START = "2026-08-15"
DISCOVERY_SHARE = 0.60
MIN_HOLDOUT_N = 45
FDR_Q = 0.10
N_BOOT = 2000
SEED = 20261026
LAYERS = ("BEST", "SECOND", "THIRD")
LAYER_RANK = {"BEST": 3, "SECOND": 2, "THIRD": 1}
GAP_TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
TRACKS = ("new_format", "standard")
CELL_MARKETS = ("OU25", "BTTS", "OVER15", "OVER35")
SIDE_MARKETS = ("BTTS", "OVER15", "OVER35")
EPS = 1e-4

STUDY_DIR = config.BASE_DIR / "docs" / "quality_tiers_2026_10"
SEED_FILE = STUDY_DIR / "seed_first_sight_tips.csv"
CUTS_FILE = STUDY_DIR / "frozen_cuts_v1.json"
SHADOW_FILE = config.OUTPUT_DIR / "quality_tier_shadow.csv"

#: Training columns: what a settled first-sight tip must carry to be learned from.
TRAIN_COLS = ["key", "day", "gen_day", "match_day", "league", "model_type", "market", "side",
              "odds", "opp_odds", "p_model", "result"]
SHADOW_KEY = ["match_day", "home_team", "away_team", "market"]
SHADOW_COLS = ["observed_at", "ledger_generated_at", "lag_min", "day", "match_day", "kickoff_utc", "league",
               "home_team", "away_team", "model_type", "market", "side", "odds", "opp_odds", "p_model",
               "p_fair", "raw_edge", "gap_tier", "model_version", "quality_version", "fitted_through",
               "train_n", "p_win", "score", "layer_fixed", "cut_lo", "cut_hi"]


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ───────────────────────────── features ─────────────────────────────

def fair_prob(odds, opp_odds, market) -> np.ndarray:
    """De-vigged two-sided where the opposite price exists, else 1/odds / SIDE_OVERROUND."""
    o = np.asarray(odds, float)
    op = np.asarray(opp_odds, float)
    imp = 1.0 / o
    two = np.isfinite(op) & (op > 1.0)
    fair = np.where(two, imp / (imp + 1.0 / np.where(two, op, 2.0)), imp / SIDE_OVERROUND)
    return np.clip(fair, 0.02, 0.98)


def add_basic(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["p_fair"] = fair_prob(d["odds"], d["opp_odds"], d["market"])
    d["raw_edge"] = d["p_model"] - np.where(d["market"].eq("OU25"), 1.0 / d["odds"], 1.0 / d["odds"] / SIDE_OVERROUND)
    d["pnl"] = np.where(d["result"].eq("WIN"), d["odds"] - 1.0, np.where(d["result"].eq("LOSS"), -1.0, np.nan))
    return d


def league_rating(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Shrunk past flat ROI of each row's track x market x league, from `train` only."""
    if train.empty:
        return np.zeros(len(rows))
    g = train.groupby(["model_type", "market", "league"])["pnl"].agg(["sum", "size"])
    r = (g["sum"] / (g["size"] + LEAGUE_K)).to_dict()
    return np.array([r.get((t, m, l), 0.0) for t, m, l in zip(rows["model_type"], rows["market"], rows["league"])])


def _train_rating(train: pd.DataFrame) -> np.ndarray:
    """League rating for TRAINING rows: each row sees only rows played strictly before its own day."""
    out = np.zeros(len(train))
    t = train.reset_index(drop=True)
    for day, idx in t.groupby("gen_day").groups.items():
        out[np.asarray(idx)] = league_rating(t[t["match_day"] < day], t.loc[idx])
    return out


def design(d: pd.DataFrame, rating: np.ndarray) -> np.ndarray:
    lf = _logit(d["p_fair"])
    lgap = _logit(d["p_model"]) - lf
    cols = [lf, lgap, (d["odds"].to_numpy(float) >= LONG_ODDS).astype(float), np.asarray(rating, float),
            (d["side"].astype(str).str.upper() == "UNDER").to_numpy(float)]
    for m in SIDE_MARKETS:
        dm = (d["market"] == m).to_numpy(float)
        cols += [dm, dm * lf, dm * lgap]
    return np.column_stack(cols)


@dataclass
class Fit:
    model: object
    track: str
    fitted_through: str
    n: int


def fit_track(train: pd.DataFrame, track: str) -> Fit | None:
    """L2 logistic on one track's settled tips. None when there is too little history."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t = train[(train["model_type"] == track) & train["result"].isin(["WIN", "LOSS"])]
    if len(t) < MIN_TRAIN or t["result"].nunique() < 2:
        return None
    t = add_basic(t)
    X = design(t, _train_rating(t))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    m.fit(X, (t["result"] == "WIN").astype(int))
    m._train = t                                                       # for the league rating of new rows
    return Fit(m, track, str(t["match_day"].max()), int(len(t)))


def score(fit: Fit, rows: pd.DataFrame) -> pd.DataFrame:
    """p_win and expected profit for `rows` (same track as the fit)."""
    r = add_basic(rows)
    X = design(r, league_rating(fit.model._train, r))
    p = fit.model.predict_proba(X)[:, 1]
    return pd.DataFrame({"p_win": p, "score": p * r["odds"].to_numpy(float) - 1.0}, index=rows.index)


def walk_forward(rec: pd.DataFrame) -> pd.DataFrame:
    """Score every tip with a model fitted only on tips played strictly before its first-sight day."""
    out = rec.copy()
    for c in ("p_win", "score", "train_n"):
        out[c] = np.nan
    out["fitted_through"] = ""
    for track in TRACKS:
        tr_all = rec[rec["model_type"] == track]
        for day in sorted(tr_all["gen_day"].unique()):
            train = tr_all[(tr_all["match_day"] < day) & tr_all["result"].isin(["WIN", "LOSS"])]
            f = fit_track(train, track)
            if f is None:
                continue
            idx = tr_all.index[tr_all["gen_day"] == day]
            s = score(f, rec.loc[idx])
            out.loc[idx, "p_win"] = s["p_win"].values
            out.loc[idx, "score"] = s["score"].values
            out.loc[idx, "train_n"] = f.n
            out.loc[idx, "fitted_through"] = f.fitted_through
    return out


# ───────────────────────────── layers ─────────────────────────────

def layer_tercile(d: pd.DataFrame, col: str = "score", by=("model_type", "market", "day")) -> pd.Series:
    """Definition (a). Deterministic: ties broken by the stable row key, never by row order."""
    out = pd.Series(pd.NA, index=d.index, dtype=object)
    ok = d[col].notna()
    x = d[ok].sort_values([*by, col, "key"], ascending=[True] * len(by) + [False, True], kind="mergesort")
    r = x.groupby(list(by)).cumcount() + 1
    n = x.groupby(list(by))[col].transform("size")
    pct = (r - 0.5) / n
    out.loc[x.index] = np.select([pct < 1 / 3, pct < 2 / 3], ["BEST", "SECOND"], "THIRD")
    return out


def learn_cuts(disc: pd.DataFrame, col: str = "score") -> dict:
    """Definition (b): per track x market, the 1/3 and 2/3 quantiles of discovery scores."""
    cuts = {}
    for (t, m), g in disc[disc[col].notna()].groupby(["model_type", "market"]):
        lo, hi = np.quantile(g[col].to_numpy(float), [1 / 3, 2 / 3])
        cuts[f"{t}|{m}"] = [round(float(lo), 6), round(float(hi), 6)]
    return cuts


def layer_fixed(d: pd.DataFrame, cuts: dict, col: str = "score") -> pd.Series:
    out = pd.Series(pd.NA, index=d.index, dtype=object)
    for i, t, m, s in zip(d.index, d["model_type"], d["market"], d[col]):
        c = cuts.get(f"{t}|{m}")
        if c is None or pd.isna(s):
            continue
        out.at[i] = "BEST" if s >= c[1] else ("SECOND" if s >= c[0] else "THIRD")
    return out


# ───────────────────────────── evidence ─────────────────────────────

def _roi(x: pd.DataFrame) -> float:
    return float(x["pnl"].mean()) if len(x) else np.nan


def boot_roi(x: pd.DataFrame, rng) -> tuple[float, float]:
    """95% matchday block-bootstrap CI of flat ROI."""
    if len(x) < 5:
        return (np.nan, np.nan)
    days = x["match_day"].to_numpy()
    u = np.unique(days)
    pn = x["pnl"].to_numpy(float)
    idx = {k: np.where(days == k)[0] for k in u}
    b = [pn[np.concatenate([idx[k] for k in rng.choice(u, len(u))])].mean() for _ in range(N_BOOT)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def boot_diff(x: pd.DataFrame, col: str, hi: str, lo: str, rng) -> dict:
    """Best - Third ROI difference, matchday blocks resampled jointly (paired by day)."""
    days = x["match_day"].to_numpy()
    u = np.unique(days)
    if len(u) < 3:
        return {"diff": np.nan, "ci": [np.nan, np.nan], "p_one": np.nan}
    lay = x[col].to_numpy()
    pn = x["pnl"].to_numpy(float)
    idx = {k: np.where(days == k)[0] for k in u}
    obs = pn[lay == hi].mean() - pn[lay == lo].mean() if (lay == hi).any() and (lay == lo).any() else np.nan
    diffs = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        a, b = s[lay[s] == hi], s[lay[s] == lo]
        if len(a) and len(b):
            diffs.append(pn[a].mean() - pn[b].mean())
    diffs = np.asarray(diffs)
    if not len(diffs) or not np.isfinite(obs):
        return {"diff": obs, "ci": [np.nan, np.nan], "p_one": np.nan}
    return {"diff": float(obs), "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_one": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))}


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaN kept out of the family)."""
    q = pd.Series(np.nan, index=p.index)
    v = p.dropna().sort_values()
    if v.empty:
        return q
    m = len(v)
    adj = (v * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    q.loc[adj.index] = adj
    return q


def layer_table(x: pd.DataFrame, col: str, order=LAYERS, clv: bool = True) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for lay in order:
        g = x[x[col] == lay]
        lo, hi = boot_roi(g, rng)
        r = {"layer": lay, "n": int(len(g)), "matchdays": int(g["match_day"].nunique()),
             "hit": round(float((g["result"] == "WIN").mean()), 3) if len(g) else np.nan,
             "roi": round(_roi(g), 4) if len(g) else np.nan,
             "roi_ci95": [round(lo, 4), round(hi, 4)] if np.isfinite(lo) else [np.nan, np.nan],
             "mean_odds": round(float(g["odds"].mean()), 3) if len(g) else np.nan}
        if clv and "clv_cert" in g.columns:
            c = g["clv_cert"].dropna()
            c6 = g["clv_6h"].dropna()
            r.update(clv_cert_n=int(len(c)), clv_cert_mean=round(float(c.mean()), 2) if len(c) else np.nan,
                     clv_6h_n=int(len(c6)), clv_6h_mean=round(float(c6.mean()), 2) if len(c6) else np.nan)
        rows.append(r)
    return rows


def monotonic(x: pd.DataFrame, col: str, order=LAYERS) -> dict:
    from scipy.stats import spearmanr
    y = x[x[col].isin(order)]
    rank = {lay: len(order) - i for i, lay in enumerate(order)}
    rois = [_roi(y[y[col] == lay]) for lay in order]
    ordered = all(np.isfinite(rois)) and all(rois[i] > rois[i + 1] for i in range(len(rois) - 1))
    if len(y) >= 10 and y[col].nunique() > 1:
        rho, p = spearmanr(y[col].map(rank), y["pnl"])
    else:
        rho, p = np.nan, np.nan
    d = boot_diff(y, col, order[0], order[-1], np.random.default_rng(SEED + 1))
    return {"ordered": bool(ordered), "spearman": round(float(rho), 4) if np.isfinite(rho) else np.nan,
            "spearman_p": round(float(p), 4) if np.isfinite(p) else np.nan,
            "top_minus_bottom": round(d["diff"], 4) if np.isfinite(d["diff"]) else np.nan,
            "top_minus_bottom_ci95": [round(v, 4) for v in d["ci"]], "p_one": d["p_one"]}


def brier(x: pd.DataFrame) -> dict:
    """Does the learned P(win) beat the de-vigged price (and the raw model) on these rows? Lower = better."""
    if len(x) < 10:
        return {}
    y = (x["result"] == "WIN").to_numpy(float)
    return {"n": int(len(x)), "p_win": round(float(np.mean((x["p_win"] - y) ** 2)), 4),
            "market_fair": round(float(np.mean((x["p_fair"] - y) ** 2)), 4),       # p_fair is the tip's own side
            "raw_model": round(float(np.mean((x["p_model"] - y) ** 2)), 4),
            "mean_p_win": round(float(x["p_win"].mean()), 4), "hit": round(float(y.mean()), 4)}


def ranking_compare(x: pd.DataFrame) -> dict:
    """Spearman of each ranking signal with pnl, inside one cell (score vs p_win-only vs gap-only)."""
    from scipy.stats import spearmanr
    out = {}
    for name, col in (("score", "score"), ("p_win_only", "p_win"), ("gap_only", "raw_edge")):
        y = x.dropna(subset=[col, "pnl"])
        if len(y) >= 10:
            rho, p = spearmanr(y[col], y["pnl"])
            out[name] = {"spearman": round(float(rho), 4), "p": round(float(p), 4), "n": int(len(y))}
    return out


# ───────────────────────────── study ─────────────────────────────

def _cell_market(m: str) -> str:
    m = str(m)
    return "OU25" if m.startswith("OU25") else m.upper()


def build_record(snapshots: pd.DataFrame, tips: pd.DataFrame | None = None) -> pd.DataFrame:
    """Season tips with FIRST-SIGHT features from the committed snapshot of the run that created them.

    tips: src.top_picks.load_tips() frame (both ledgers, live, paper leagues dropped, first per fixture x market).
    snapshots: every committed predictions.csv row (Pro experiments/valuable_2026_10/prematch_snapshots.parquet,
    refreshed). Match = same league/date/home/away, the snapshot generated closest to the ledger's
    generated_at, no later than +15 min and no earlier than -3 h. Unmatched tips are dropped and counted.
    """
    from src.top_picks import load_tips
    t = (tips if tips is not None else load_tips()).copy()
    t = t[t["match_day"] >= SEASON_START].copy()
    t["cell_market"] = t["market"].map(_cell_market)
    t["side"] = np.where(t["cell_market"].eq("OU25"), t["market"].str.replace("OU25_", "", regex=False),
                         np.where(t["cell_market"].eq("BTTS"), "YES", "OVER"))
    t["lgen"] = pd.to_datetime(t["generated_at"], utc=True, errors="coerce", format="mixed")
    s = snapshots.copy()
    s["gen"] = pd.to_datetime(s["generated_at"], utc=True)
    s["fxk"] = s["league"].astype(str) + "|" + s["date"].astype(str).str[:10] + "|" + s["home_team"] + "|" + s["away_team"]
    t["fxk"] = t["league"].astype(str) + "|" + t["match_day"] + "|" + t["home_team"] + "|" + t["away_team"]
    s = s[s["fxk"].isin(set(t["fxk"]))].sort_values("gen")
    by = dict(tuple(s.groupby("fxk")))
    rows, miss = [], 0
    for i, r in t.iterrows():
        g = by.get(r["fxk"])
        if g is None or pd.isna(r["lgen"]):
            miss += 1
            continue
        dt = (g["gen"] - r["lgen"]).dt.total_seconds() / 60
        ok = g[(dt <= 15) & (dt >= -180)]
        if ok.empty:
            miss += 1
            continue
        sn = ok.loc[(ok["gen"] - r["lgen"]).abs().idxmin()]
        m, side = r["cell_market"], r["side"]
        if m == "OU25":
            p = sn["p_over25"] if side == "OVER" else 1 - sn["p_over25"]
            odds = float(r["odds"])                                   # OU25 ledger odds are never upgraded
            opp = sn["odds_under25"] if side == "OVER" else sn["odds_over25"]
            tier = str(sn["tier_over"] if side == "OVER" else sn["tier_under"]).upper()
            snap_odds = sn["odds_over25"] if side == "OVER" else sn["odds_under25"]
        else:
            col = {"BTTS": "btts", "OVER15": "over15", "OVER35": "over35"}[m]
            p, odds = sn[f"p_{col}"], float(sn[f"odds_{col}"]) if pd.notna(sn[f"odds_{col}"]) else float(r["odds"])
            opp = sn["odds_btts_no"] if m == "BTTS" else np.nan
            tier = ""                                                  # recomputed below with production's rule
            snap_odds = sn[f"odds_{col}"]
        rows.append({"key": r["fxk"] + "|" + m + "|" + side, "fxk": r["fxk"], "day": r["day"],
                     "gen_day": r["lgen"].strftime("%Y-%m-%d"), "match_day": r["match_day"], "league": r["league"],
                     "home_team": r["home_team"], "away_team": r["away_team"], "model_type": r["model_type"],
                     "market": m, "side": side, "odds": odds, "opp_odds": float(opp) if pd.notna(opp) else np.nan,
                     "p_model": float(p) if pd.notna(p) else np.nan, "gap_tier_snap": tier,
                     "ledger_tier": r["signal_tier"], "snap_odds": snap_odds, "snap_lag_min": float(dt.loc[sn.name]),
                     "model_version": sn.get("model_version"), "kickoff": sn.get("kickoff"), "result": r["result"]})
    d = pd.DataFrame(rows)
    d = d.dropna(subset=["p_model", "odds"])
    d = d[d["odds"] > 1.01].reset_index(drop=True)
    d = add_basic(d)
    d["gap_tier"] = np.where(d["market"].eq("OU25"), d["gap_tier_snap"], _side_tier(d))
    d["gap_tier"] = d["gap_tier"].replace({"VALUE": "VALUABLE"})
    # where the first-sight snapshot itself was below the floor (rare: the tip was created by a
    # run whose own snapshot was not committed), fall back to the ledger's tier and flag it
    bad = ~d["gap_tier"].isin(GAP_TIERS)
    d["gap_tier_fallback"] = bad
    d.loc[bad, "gap_tier"] = d.loc[bad, "ledger_tier"]
    d.attrs["unmatched"] = miss
    return d


def _side_tier(d: pd.DataFrame) -> np.ndarray:
    """Production's side-market rule (pipeline._generate_side_bets) with today's approved per-league gates."""
    try:
        from src.gate_resolver import resolve_effective_tier_gates
    except Exception:                                                  # noqa: BLE001
        resolve_effective_tier_gates = None
    out = []
    for lg, m, e, t in zip(d["league"], d["market"], d["raw_edge"], d["model_type"]):
        if m == "OU25":
            out.append("")
            continue
        sn, mk = 0.10, 0.08
        if resolve_effective_tier_gates is not None:
            try:
                g = resolve_effective_tier_gates(lg, m.lower(), t)
                sn, mk = g.sniper, g.marksman
            except Exception:                                          # noqa: BLE001
                pass
        out.append("SNIPER" if e >= sn else "MARKSMAN" if e >= mk else "VALUABLE" if e >= 0.04 else "AVOID")
    return np.array(out, dtype=object)


def add_clv(d: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Close = last committed snapshot price strictly before kickoff. clv_cert: close within 30 min of
    kickoff (v9's certified-close rule); clv_6h: within 6 h (Pro all-fixture rule). CLV% = odds/close - 1."""
    s = snapshots.copy()
    s["gen"] = pd.to_datetime(s["generated_at"], utc=True)
    s["kick"] = pd.to_datetime(s["kickoff_utc"], utc=True, errors="coerce")
    s = s[s["kick"].notna() & (s["gen"] < s["kick"])]
    s["fxk"] = s["league"].astype(str) + "|" + s["date"].astype(str).str[:10] + "|" + s["home_team"] + "|" + s["away_team"]
    s = s[s["fxk"].isin(set(d["fxk"]))].sort_values("gen").drop_duplicates("fxk", keep="last")
    s["gap_min"] = (s["kick"] - s["gen"]).dt.total_seconds() / 60
    last = s.set_index("fxk")
    colmap = {("OU25", "OVER"): "odds_over25", ("OU25", "UNDER"): "odds_under25", ("BTTS", "YES"): "odds_btts",
              ("OVER15", "OVER"): "odds_over15", ("OVER35", "OVER"): "odds_over35"}
    close, gap = [], []
    for fx, m, side in zip(d["fxk"], d["market"], d["side"]):
        if fx in last.index and (m, side) in colmap:
            close.append(last.at[fx, colmap[(m, side)]])
            gap.append(last.at[fx, "gap_min"])
        else:
            close.append(np.nan)
            gap.append(np.nan)
    close, gap = np.asarray(close, float), np.asarray(gap, float)
    clv = (d["odds"].to_numpy(float) / close - 1) * 100
    d = d.copy()
    d["clv_cert"] = np.where(gap <= 30, clv, np.nan)
    d["clv_6h"] = np.where(gap <= 360, clv, np.nan)
    return d


def split_day(d: pd.DataFrame) -> str:
    """First day of the HOLDOUT: 60% of the distinct settled scored tip days fall before it (all cells)."""
    days = sorted(d.loc[d["score"].notna() & d["pnl"].notna(), "day"].unique())
    return days[int(np.floor(len(days) * DISCOVERY_SHARE))]


def run_study(snapshots: pd.DataFrame, tips: pd.DataFrame | None = None, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else STUDY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = build_record(snapshots, tips)
    rec = add_clv(rec, snapshots)
    wf = walk_forward(rec)
    wf = wf.assign(settled=wf["result"].isin(["WIN", "LOSS"]))
    cut = split_day(wf)
    wf["window"] = np.where(wf["day"] < cut, "DISCOVERY", "HOLDOUT")
    ev = wf[wf["settled"] & wf["score"].notna()].copy()
    disc, hold = ev[ev["window"] == "DISCOVERY"], ev[ev["window"] == "HOLDOUT"]
    cuts = learn_cuts(disc)
    for name, col in (("score", "score"), ("p_win", "p_win"), ("gap", "raw_edge")):
        ev[f"L_tercile_{name}"] = layer_tercile(ev, col)
    ev["L_fixed_score"] = layer_fixed(ev, cuts)

    cells, verdict_rows = {}, []
    for t in TRACKS:
        for m in CELL_MARKETS:
            c = ev[(ev["model_type"] == t) & (ev["market"] == m)]
            all_tips = wf[(wf["model_type"] == t) & (wf["market"] == m) & wf["settled"]]
            if all_tips.empty:
                continue
            cd, ch = c[c["window"] == "DISCOVERY"], c[c["window"] == "HOLDOUT"]
            r = {"track": t, "market": m, "season_tips_settled": int(len(all_tips)), "scored": int(len(c)),
                 "discovery_n": int(len(cd)), "holdout_n": int(len(ch))}
            dsc = {}
            for defn, col in (("TERCILE", "L_tercile_score"), ("FIXED", "L_fixed_score")):
                mo = monotonic(cd, col) if len(cd) >= 10 else {"top_minus_bottom": np.nan, "ordered": False}
                dsc[defn] = mo
            a, b = dsc["TERCILE"]["top_minus_bottom"], dsc["FIXED"]["top_minus_bottom"]
            chosen = "FIXED" if (np.isfinite(b) and (not np.isfinite(a) or b > a)) else "TERCILE"
            col = {"TERCILE": "L_tercile_score", "FIXED": "L_fixed_score"}[chosen]
            r.update(chosen_definition=chosen, discovery=dsc,
                     discovery_layers=layer_table(cd, col), holdout_layers=layer_table(ch, col),
                     holdout_monotonic=monotonic(ch, col) if len(ch) >= 10 else {},
                     holdout_gap_tiers=layer_table(ch, "gap_tier", GAP_TIERS),
                     holdout_gap_monotonic=monotonic(ch, "gap_tier", GAP_TIERS) if len(ch) >= 10 else {},
                     discovery_gap_tiers=layer_table(cd, "gap_tier", GAP_TIERS),
                     holdout_ranking=ranking_compare(ch), discovery_ranking=ranking_compare(cd),
                     holdout_alt_layers={"p_win_only": monotonic(ch, "L_tercile_p_win") if len(ch) >= 10 else {},
                                         "gap_only": monotonic(ch, "L_tercile_gap") if len(ch) >= 10 else {}},
                     holdout_all={"n": int(len(ch)), "roi": round(_roi(ch), 4) if len(ch) else np.nan},
                     holdout_brier=brier(ch),
                     season_gap_tiers_first_sight=layer_table(all_tips, "gap_tier", GAP_TIERS, clv=False),
                     season_gap_tiers_ledger=layer_table(all_tips, "ledger_tier", GAP_TIERS, clv=False))
            cells[f"{t}|{m}"] = r
            verdict_rows.append({"cell": f"{t}|{m}", "n": len(ch),
                                 "ordered": bool(r["holdout_monotonic"].get("ordered", False)),
                                 "p_one": r["holdout_monotonic"].get("p_one", np.nan)})
    v = pd.DataFrame(verdict_rows).set_index("cell")
    fam = v[v["n"] >= MIN_HOLDOUT_N]["p_one"]
    v["q_bh"] = bh(fam).reindex(v.index)
    for cell, r in cells.items():
        n, ordered, q = v.at[cell, "n"], v.at[cell, "ordered"], v.at[cell, "q_bh"]
        if n < MIN_HOLDOUT_N:
            verdict, why = "ONE_LAYER", "INSUFFICIENT"
        elif ordered and np.isfinite(q) and q < FDR_Q:
            verdict, why = "REAL_3_LAYERS", "ORDERED_AND_SIGNIFICANT"
        elif ordered:
            verdict, why = "ONE_LAYER", "ORDER_SUGGESTIVE"
        else:
            verdict, why = "ONE_LAYER", "ORDER_FAILS"
        r.update(q_bh=None if not np.isfinite(q) else round(float(q), 4), verdict=verdict, verdict_reason=why)
    # the same test applied to the CURRENT gap tiers on the same holdout rows, for comparison
    gp = pd.Series({k: c["holdout_gap_monotonic"].get("p_one", np.nan) for k, c in cells.items()
                    if c["holdout_n"] >= MIN_HOLDOUT_N})
    gq = bh(gp)
    for k, c in cells.items():
        q = gq.get(k, np.nan)
        c["gap_q_bh"] = None if not np.isfinite(q) else round(float(q), 4)

    meta = {"version": VERSION, "built_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season_start": SEASON_START, "holdout_starts": cut,
            "first_scored_day": str(ev["day"].min()), "last_day": str(ev["day"].max()),
            "tips_in_record": int(len(rec)), "unmatched_tips": int(rec.attrs.get("unmatched", 0)),
            "gap_tier_fallback": int(rec["gap_tier_fallback"].sum()),
            "gap_tier_vs_ledger_agree": round(float((rec["gap_tier"] == rec["ledger_tier"]).mean()), 3),
            "settled_scored": int(len(ev)), "frozen": {"C": C, "MIN_TRAIN": MIN_TRAIN, "LEAGUE_K": LEAGUE_K,
                                                       "LONG_ODDS": LONG_ODDS, "DISCOVERY_SHARE": DISCOVERY_SHARE,
                                                       "MIN_HOLDOUT_N": MIN_HOLDOUT_N, "FDR_Q": FDR_Q,
                                                       "N_BOOT": N_BOOT, "SEED": SEED},
            "cuts_fixed_from_discovery": cuts}
    res = {"meta": meta, "cells": cells}
    (out_dir / "study_results.json").write_text(json.dumps(_clean(res), indent=1, default=_js, allow_nan=False) + "\n",
                                                encoding="utf-8")
    CUTS = {"version": VERSION, "learned_on": f"DISCOVERY {meta['first_scored_day']} .. day < {cut}",
            "cuts": cuts, "chosen_definition": {k: c["chosen_definition"] for k, c in cells.items()},
            "verdict": {k: c["verdict"] for k, c in cells.items()}}
    (out_dir / "frozen_cuts_v1.json").write_text(json.dumps(CUTS, indent=1) + "\n", encoding="utf-8")
    keep = ["key", "day", "gen_day", "match_day", "league", "model_type", "market", "side", "odds", "opp_odds",
            "p_model", "gap_tier", "model_version", "result", "window", "p_win", "score", "fitted_through",
            "L_tercile_score", "L_fixed_score", "clv_cert", "clv_6h"]
    rows = wf.merge(ev[["key", "L_tercile_score", "L_fixed_score"]], on="key", how="left")
    rows[keep].round(5).sort_values(["gen_day", "key"]).to_csv(out_dir / "seed_first_sight_tips.csv", index=False,
                                                               lineterminator="\n")
    return res


def _clean(o):
    """Strict JSON: NaN/inf -> null, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ───────────────────────────── shadow (first sight, research only) ─────────────────────────────

def frozen_cuts() -> dict:
    try:
        return json.loads(CUTS_FILE.read_text(encoding="utf-8")).get("cuts", {})
    except Exception:                                                  # noqa: BLE001
        return {}


def shadow_store(path: Path | None = None):
    from src.research.first_sight import FirstSightStore
    return FirstSightStore(Path(path) if path else SHADOW_FILE, key=SHADOW_KEY, columns=SHADOW_COLS,
                           observed_col="observed_at", kickoff_col="kickoff_utc", digest=True)


def training_history(seed_path: Path | None = None, shadow_path: Path | None = None,
                     tips: pd.DataFrame | None = None) -> pd.DataFrame:
    """Settled first-sight tips to learn from: the frozen seed (season to 2026-10-26) + shadow rows whose
    result the ledgers now hold. Never ledger features (they are upgraded in place)."""
    parts = []
    sp = Path(seed_path) if seed_path else SEED_FILE
    if sp.exists():
        s = pd.read_csv(sp, dtype=str, keep_default_na=False)
        s["home_team"] = s["key"].str.split("|").str[2]
        s["away_team"] = s["key"].str.split("|").str[3]
        parts.append(s)
    shp = Path(shadow_path) if shadow_path else SHADOW_FILE
    if shp.exists() and shp.stat().st_size > 0:
        sh = pd.read_csv(shp, dtype=str, keep_default_na=False)
        if len(sh):
            sh["key"] = sh["league"] + "|" + sh["match_day"] + "|" + sh["home_team"] + "|" + sh["away_team"] + "|" + \
                sh["market"] + "|" + sh["side"]
            sh["gen_day"] = sh["ledger_generated_at"].str[:10]
            sh["result"] = ""
            parts.append(sh)
    if not parts:
        return pd.DataFrame(columns=TRAIN_COLS)
    h = pd.concat([p[TRAIN_COLS + ["home_team", "away_team"]] for p in parts], ignore_index=True)
    h = h.drop_duplicates("key", keep="first").reset_index(drop=True)
    h["result"] = h["result"].fillna("").astype(str).str.upper()
    # results come from the ledgers (WIN/LOSS are never upgraded); features never do
    pend = ~h["result"].isin(["WIN", "LOSS"])
    if pend.any():
        from src.top_picks import load_tips
        t = tips if tips is not None else load_tips()
        if t is not None and len(t):
            t = t.assign(cell=t["market"].map(_cell_market))
            res = t.drop_duplicates(["match_day", "home_team", "away_team", "cell"]).set_index(
                ["match_day", "home_team", "away_team", "cell"])["result"]
            k = pd.MultiIndex.from_arrays([h.loc[pend, c] for c in ("match_day", "home_team", "away_team", "market")])
            h.loc[pend, "result"] = pd.Series(res.reindex(k).to_numpy(), index=h.index[pend]).fillna("").astype(str)
    for c in ("odds", "opp_odds", "p_model"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    return h[TRAIN_COLS]


def shadow_rows(now: pd.Timestamp | None = None, tips: pd.DataFrame | None = None,
                preds: pd.DataFrame | None = None, history: pd.DataFrame | None = None,
                cuts: dict | None = None) -> pd.DataFrame:
    """One row per tip not yet kicked off: first-sight values + p_win / score / fixed layer."""
    from src.top_picks import load_tips
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    all_t = tips if tips is not None else load_tips()
    t = all_t
    if t is None or t.empty:
        return pd.DataFrame(columns=SHADOW_COLS)
    t = t[t["match_day"] >= now.strftime("%Y-%m-%d")].copy() if "match_day" in t.columns else t
    if t.empty:
        return pd.DataFrame(columns=SHADOW_COLS)
    if preds is None:
        p = config.OUTPUT_DIR / "predictions.csv"
        preds = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()
    t["market_c"] = t["market"].map(_cell_market)
    t["side"] = np.where(t["market_c"].eq("OU25"), t["market"].str.replace("OU25_", "", regex=False),
                         np.where(t["market_c"].eq("BTTS"), "YES", "OVER"))
    pk = {}
    if len(preds):
        for r in preds.to_dict("records"):
            pk[(str(r.get("date", ""))[:10], str(r.get("home_team")), str(r.get("away_team")))] = r
    rows = []
    for r in t.to_dict("records"):
        pr = pk.get((r["match_day"], str(r["home_team"]), str(r["away_team"])))
        if not pr:
            # not on the current board: predictions.csv is pre-match only (invariant 5), so a tip missing
            # from it may already have kicked off - a first-sight log never records what it cannot prove
            # was seen before kickoff
            continue
        m, side = r["market_c"], r["side"]
        odds = float(r["odds"])
        if m == "OU25":
            pov = pr.get("p_over25")
            p_model = (pov if side == "OVER" else 1 - pov) if pd.notna(pov) else r["model_p"]
            opp = pr.get("odds_under25") if side == "OVER" else pr.get("odds_over25")
        else:
            p_model = r.get("model_prob") if pd.notna(r.get("model_prob")) else r["model_p"]
            opp = pr.get("odds_btts_no") if m == "BTTS" else np.nan
        ko = pr.get("kickoff_utc") if pd.notna(pr.get("kickoff_utc")) else r.get("kickoff_utc")
        lg = pd.to_datetime(r.get("generated_at"), utc=True, errors="coerce")
        rows.append({"observed_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "ledger_generated_at": lg.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(lg) else "",
                     "lag_min": round((now - lg).total_seconds() / 60, 1) if pd.notna(lg) else np.nan,
                     "day": r["day"], "match_day": r["match_day"], "kickoff_utc": ko, "league": r["league"],
                     "home_team": r["home_team"], "away_team": r["away_team"], "model_type": r["model_type"],
                     "market": m, "side": side, "odds": odds,
                     "opp_odds": float(opp) if opp is not None and pd.notna(opp) else np.nan,
                     "p_model": float(p_model), "gap_tier": r["signal_tier"],
                     "model_version": r.get("model_version", ""), "result": ""})
    if not rows:
        return pd.DataFrame(columns=SHADOW_COLS)
    d = pd.DataFrame(rows)
    d = add_basic(d)
    d["quality_version"] = VERSION
    cuts = cuts if cuts is not None else frozen_cuts()
    h = history if history is not None else training_history(tips=all_t)
    day = now.strftime("%Y-%m-%d")
    for c in ("p_win", "score", "train_n", "cut_lo", "cut_hi"):
        d[c] = np.nan
    d["fitted_through"] = ""
    d["layer_fixed"] = ""
    for track in TRACKS:
        idx = d.index[d["model_type"] == track]
        if not len(idx):
            continue
        f = fit_track(h[h["match_day"] < day], track)
        if f is None:
            continue
        s = score(f, d.loc[idx])
        d.loc[idx, "p_win"] = s["p_win"].round(5).values
        d.loc[idx, "score"] = s["score"].round(5).values
        d.loc[idx, "train_n"] = f.n
        d.loc[idx, "fitted_through"] = f.fitted_through
    lay = layer_fixed(d, cuts)
    d["layer_fixed"] = lay.fillna("").values
    for i in d.index:
        c = cuts.get(f"{d.at[i, 'model_type']}|{d.at[i, 'market']}")
        if c:
            d.at[i, "cut_lo"], d.at[i, "cut_hi"] = c
    for c in ("p_fair", "raw_edge"):
        d[c] = d[c].round(5)
    return d.reindex(columns=SHADOW_COLS)


def log_shadow(path: Path | None = None, **kw) -> int:
    """Append first-sight rows for new tips. Never raises; returns rows appended."""
    try:
        rows = shadow_rows(**kw)
        n = shadow_store(path).append(rows)
        log.info(f"[quality_tiers] shadow: {n} new row(s) -> {Path(path) if path else SHADOW_FILE}")
        return n
    except Exception as e:                                             # noqa: BLE001 - research recorder
        log.warning(f"[quality_tiers] shadow skipped: {type(e).__name__}: {e}")
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--snapshots", default=None, help="prematch_snapshots.parquet (Pro experiments/valuable_2026_10)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shadow", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.study:
        res = run_study(pd.read_parquet(a.snapshots), out_dir=a.out)
        print(json.dumps({k: {"verdict": v["verdict"], "why": v["verdict_reason"], "holdout_n": v["holdout_n"]}
                          for k, v in res["cells"].items()}, indent=1))
    if a.shadow:
        print(f"quality-layer shadow: {log_shadow()} new row(s)")


if __name__ == "__main__":
    main()
