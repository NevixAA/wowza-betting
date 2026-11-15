"""Scores and the pre-registered comparison. Forecasting quality only - never a profit number.

    metrics(y, p)            log loss, Brier, ECE, calibration slope/intercept, n
    calibration_table(y, p)  reliability bins (mean p, observed rate, n)
    clv(df, p, market)       entry/closing price ratio of the sides the model would take, ONLY where a
                             real entry AND a real closing price exist; otherwise n = 0
    compare(rows, cfg)       per (stage, track, market, candidate) vs the reference on COMMON rows:
                             match-date paired bootstrap, one-sided p, Holm across the whole family,
                             verdict from the thresholds fixed in config/phase2_retrain.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15

BETTER = "BETTER_THAN_REFERENCE"
NOT_SHOWN = "NOT_SHOWN_BETTER"
INSUFFICIENT = "INSUFFICIENT_N"
NOT_COMPARABLE = "NOT_COMPARABLE"


def _arr(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], np.clip(p[ok], EPS, 1 - EPS)


def row_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins: int) -> float:
    y, p = _arr(y, p)
    if not len(y):
        return float("nan")
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def calibration_table(y, p, bins: int) -> list[dict]:
    y, p = _arr(y, p)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int((idx == b).sum()),
             "mean_p": float(p[idx == b].mean()), "observed": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def calibration_line(y, p) -> tuple[float, float]:
    """(intercept, slope) of logit P(y) on logit p by IRLS; (nan, nan) when not estimable."""
    y, p = _arr(y, p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if len(y) < 30 or np.std(x) < 1e-9 or len(set(y)) < 2:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    try:
        for _ in range(50):
            mu = 1 / (1 + np.exp(-(X @ beta)))
            H = X.T @ (X * np.clip(mu * (1 - mu), 1e-9, None)[:, None])
            step = np.linalg.solve(H, X.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        return float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


def metrics(y, p, bins: int) -> dict:
    yy, pp = _arr(y, p)
    if not len(yy):
        return {"n": 0}
    a, b = calibration_line(yy, pp)
    return {"n": int(len(yy)), "log_loss": float(row_log_loss(yy, pp).mean()),
            "brier": float(np.mean((pp - yy) ** 2)), "ece": ece(yy, pp, bins),
            "calibration_intercept": a, "calibration_slope": b,
            "mean_p": float(pp.mean()), "observed_rate": float(yy.mean())}


# ── CLV ─────────────────────────────────────────────────────────────────────────────────────

def clv(df: pd.DataFrame, p_yes, market: dict, market_name: str) -> dict:
    """Per side: n counted, mean and median of entry/close - 1. A side is counted only when its
    entry AND closing price are real (> 1) and p_side x entry > 1. Columns
    odds_<market>_<side>_entry / _close come from --fd-raw; without them every n is 0 (nothing is
    approximated)."""
    p_yes = np.asarray(p_yes, float)
    out = {}
    for side in market.get("sides") or ():
        e_col, c_col = f"odds_{market_name}_{side}_entry", f"odds_{market_name}_{side}_close"
        if market.get("price") is None or e_col not in df.columns or c_col not in df.columns:
            out[side] = {"n": 0, "reason": f"no real entry and closing price columns ({e_col}, {c_col})"}
            continue
        e = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(df[c_col], errors="coerce").to_numpy(dtype=float)
        ps = p_yes if side in ("over", "yes") else 1 - p_yes
        take = (e > 1) & (c > 1) & np.isfinite(ps) & (ps * e > 1)
        v = e[take] / c[take] - 1
        out[side] = {"n": int(take.sum()), "eligible_rows": int(((e > 1) & (c > 1)).sum()),
                     "mean": float(v.mean()) if len(v) else None,
                     "median": float(np.median(v)) if len(v) else None}
    return out


# ── the pre-registered comparison ───────────────────────────────────────────────────────────

def bootstrap_p(delta, days, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    """One-sided p that the mean of `delta` (candidate - reference log loss) is < 0, resampling
    whole match dates. Returns (p, ci_lo, ci_hi, n_dates); p = (k + 1) / (B + 1), k = number of
    bootstrap means >= 0."""
    by = (pd.DataFrame({"d": pd.Series(days).astype(str).str[:10].to_numpy(), "x": np.asarray(delta, float)})
          .groupby("d")["x"].agg(["sum", "size"]))
    nd = int(len(by))
    if nd == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    means = by["sum"].to_numpy()[idx].sum(1) / by["size"].to_numpy()[idx].sum(1)
    k = int((means >= 0).sum())
    return (k + 1) / (n_boot + 1), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), nd


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1). NaN stays NaN and is not counted."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if not m:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def compare(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """`pred`: one row per (stage, track, market, candidate, row_id) with y, p, date, league.
    Returns one row per comparison with the verdict. Thresholds come only from cfg['comparison']."""
    c = cfg["comparison"]
    ref = c["reference"]
    rows = []
    for (stage, track, market), g in pred.groupby(["stage", "track", "market"], sort=True):
        wide = g.pivot_table(index=["row_id", "date", "y"], columns="candidate", values="p",
                             aggfunc="first", dropna=False).reset_index()
        for cand in sorted(set(g["candidate"]) - {ref}):
            if ref not in wide.columns or cand not in wide.columns:
                rows.append({"stage": stage, "track": track, "market": market, "candidate": cand,
                             "reference": ref, "n_common": 0, "matchdays": 0,
                             "verdict": NOT_COMPARABLE, "p_one_sided": float("nan")})
                continue
            w = wide[np.isfinite(wide[ref].astype(float)) & np.isfinite(wide[cand].astype(float))]
            r = {"stage": stage, "track": track, "market": market, "candidate": cand, "reference": ref,
                 "n_common": int(len(w)), "matchdays": int(w["date"].nunique())}
            if not len(w):
                rows.append(r | {"verdict": NOT_COMPARABLE, "p_one_sided": float("nan")})
                continue
            y = w["y"].to_numpy(float)
            dl = row_log_loss(y, w[cand]) - row_log_loss(y, w[ref])
            db = (np.clip(w[cand].to_numpy(float), 0, 1) - y) ** 2 - (np.clip(w[ref].to_numpy(float), 0, 1) - y) ** 2
            p, lo, hi, nd = bootstrap_p(dl, w["date"], int(c["n_boot"]), int(c["seed"]))
            r.update(delta_log_loss=float(dl.mean()), delta_brier=float(db.mean()),
                     ci95_delta_log_loss=[lo, hi], p_one_sided=p)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_one_sided"].to_numpy(float))
    out["verdict"] = [_verdict(r, c) for r in out.to_dict("records")]
    out["family_size"] = int(np.isfinite(out["p_one_sided"].to_numpy(float)).sum())
    return out


def _verdict(r: dict, c: dict) -> str:
    if r.get("verdict") == NOT_COMPARABLE or not np.isfinite(r.get("p_holm", np.nan)):
        return NOT_COMPARABLE
    if r["n_common"] < int(c["min_n_rows"]) or r["matchdays"] < int(c["min_matchdays"]):
        return INSUFFICIENT
    if (r["p_holm"] < float(c["family_alpha"])
            and -r["delta_log_loss"] >= float(c["min_effect_logloss"])
            and r["delta_brier"] <= float(c["max_brier_worsening"])):
        return BETTER
    return NOT_SHOWN


def per_league(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (stage, track, market, candidate, league) metrics; leagues under min_league_n are kept
    and marked, never dropped."""
    bins = int(cfg["comparison"]["ece_bins"])
    min_n = int(cfg["comparison"]["min_league_n"])
    rows = []
    for key, g in pred.groupby(["stage", "track", "market", "candidate", "league"], sort=True):
        m = metrics(g["y"], g["p"], bins)
        rows.append(dict(zip(["stage", "track", "market", "candidate", "league"], key))
                    | m | {"below_min_league_n": m["n"] < min_n})
    return pd.DataFrame(rows)
