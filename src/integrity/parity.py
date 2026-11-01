"""Train/serve feature parity (R2-C finding 18).

For every serving feature of a model: training mean/std/missing rate vs live (the committed board,
output/predictions.csv) mean/std/missing rate, a constant-live flag, PSI on training-decile bins
and a PASS/WARN/FAIL verdict under an explicit rule.

THE FAILURE THIS EXISTS FOR. Until 2026-11-01 new-format serving built `home_advantage` as a
hard-coded 1.0 while training computed a per-team rolling rate (mean ~0.45). Nothing compared the
two, so the model scored every NF fixture with a value it had never seen in training for months.
`classify` makes that impossible to miss: a feature that varies in training and is constant on a
live board of several fixtures is FAIL, whatever its value.

RULES (PARITY_RULE_VERSION)
  FAIL  live all-missing while training is mostly present (train missing < 50%)
  FAIL  constant live (>= MIN_LIVE_N non-null, one distinct value) while training varies
  FAIL  PSI >= PSI_FAIL and |mean shift| >= 1 training std
  FAIL  live missing exceeds training missing by >= 50pp (the live value is mostly imputed)
  WARN  PSI >= PSI_WARN, or |mean shift| >= 1 training std, or live missing exceeds training
        missing by >= 25pp
  PASS  otherwise
EXPECTED exceptions are declared per feature (calendar-driven features legitimately sit on one
value on a single matchday); they cap the verdict at WARN and the reason is written out.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PARITY_RULE_VERSION = "feature-parity-1.0"
MIN_LIVE_N = 5
PSI_WARN = 0.25
PSI_FAIL = 1.0
MISSING_JUMP = 0.25
MISSING_JUMP_FAIL = 0.50

#: feature -> (rule, reason). A declared rule caps the verdict at WARN.
EXPECTED = {
    "season_stage_ratio": ("calendar", "one matchday: every live fixture sits at the same stage of its season"),
    "is_late_season": ("calendar", "one matchday: binary calendar flag is legitimately constant live"),
    "home_pitch_artificial": ("sparse_binary", "few artificial pitches: a small board can be all zeros"),
    "home_coach_is_caretaker": ("sparse_binary", "rare event: a small board can be all zeros"),
    "away_coach_is_caretaker": ("sparse_binary", "rare event: a small board can be all zeros"),
    "key_attacker_absent_home": ("sparse_binary", "rare event: a small board can be all zeros"),
    "key_attacker_absent_away": ("sparse_binary", "rare event: a small board can be all zeros"),
    "bookmaker_overround": ("market_derived", "price-derived: historical source (best-of-books / "
                            "closing) differs from the live one-book or best-of-books price"),
    "implied_prob_over": ("market_derived", "price-derived"),
    "implied_prob_under": ("market_derived", "price-derived"),
}


def psi(train: pd.Series, live: pd.Series, bins: int = 10, eps: float = 1e-4) -> float | None:
    """Population stability index on training-quantile bins. None when either side is empty."""
    t = pd.to_numeric(train, errors="coerce").dropna().to_numpy(dtype=float)
    l = pd.to_numeric(live, errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) == 0 or len(l) == 0:
        return None
    edges = np.unique(np.quantile(t, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                       # constant in training
        same = np.isclose(l, edges[0]).mean()
        return 0.0 if same == 1.0 else float((1 - same) * math.log((1 - same + eps) / eps))
    edges[0], edges[-1] = -np.inf, np.inf
    pt = np.histogram(t, edges)[0] / len(t)
    pl = np.histogram(l, edges)[0] / len(l)
    pt = np.clip(pt, eps, None)
    pl = np.clip(pl, eps, None)
    return float(np.sum((pl - pt) * np.log(pl / pt)))


def feature_stats(train: pd.Series, live: pd.Series) -> dict:
    t = pd.to_numeric(train, errors="coerce")
    l = pd.to_numeric(live, errors="coerce")
    tn, ln = t.dropna(), l.dropna()

    def _f(x):
        return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)
    return {
        "train_n": int(len(t)), "live_n": int(len(l)),
        "train_mean": _f(tn.mean()) if len(tn) else None,
        "train_std": _f(tn.std(ddof=0)) if len(tn) else None,
        "live_mean": _f(ln.mean()) if len(ln) else None,
        "live_std": _f(ln.std(ddof=0)) if len(ln) else None,
        "train_missing_rate": _f(t.isna().mean()) if len(t) else None,
        "live_missing_rate": _f(l.isna().mean()) if len(l) else None,
        "train_distinct": int(tn.nunique()),
        "live_distinct": int(ln.nunique()),
        "live_non_null": int(len(ln)),
        "psi": _f(psi(t, l)),
    }


def classify(feature: str, s: dict) -> tuple[str, list]:
    """Verdict + reasons for one feature's stats (see module docstring)."""
    reasons, verdict = [], "PASS"
    tm, lm = s.get("train_missing_rate"), s.get("live_missing_rate")
    t_std = s.get("train_std") or 0.0
    constant_live = (s.get("live_non_null", 0) >= MIN_LIVE_N and s.get("live_distinct", 0) == 1
                     and s.get("train_distinct", 0) > 1 and t_std > 0)
    s["constant_live"] = bool(constant_live)
    shift = None
    if s.get("train_mean") is not None and s.get("live_mean") is not None and t_std > 0:
        shift = abs(s["live_mean"] - s["train_mean"]) / t_std
    s["mean_shift_train_sd"] = None if shift is None else round(float(shift), 4)
    p = s.get("psi")

    if s.get("live_n", 0) and s.get("live_non_null", 0) == 0 and tm is not None and tm < 0.5:
        verdict = "FAIL"
        reasons.append("live all-missing while training is mostly present")
    if constant_live:
        verdict = "FAIL"
        reasons.append(f"constant live value across {s['live_non_null']} fixtures while training "
                       f"varies (train sd {t_std:.4g})")
    if p is not None and p >= PSI_FAIL and shift is not None and shift >= 1.0:
        verdict = "FAIL"
        reasons.append(f"PSI {p:.3f} >= {PSI_FAIL} with mean shift {shift:.2f} train sd")
    if tm is not None and lm is not None and lm - tm >= MISSING_JUMP_FAIL and s.get("live_non_null", 0):
        verdict = "FAIL"
        reasons.append(f"live missing {lm:.0%} vs train {tm:.0%}: most live values are imputed")
    if verdict != "FAIL":
        if p is not None and p >= PSI_WARN:
            verdict = "WARN"
            reasons.append(f"PSI {p:.3f} >= {PSI_WARN}")
        if shift is not None and shift >= 1.0:
            verdict = "WARN"
            reasons.append(f"mean shift {shift:.2f} train sd")
        if tm is not None and lm is not None and lm - tm >= MISSING_JUMP:
            verdict = "WARN"
            reasons.append(f"live missing {lm:.0%} vs train {tm:.0%}")
    if feature in EXPECTED and verdict == "FAIL":
        rule, why = EXPECTED[feature]
        verdict = "WARN"
        reasons.append(f"capped at WARN by expected rule '{rule}': {why}")
    s["expected_rule"] = EXPECTED.get(feature, ("train_serve_identical", "same construction "
                                               "train and serve; distributions should match"))[0]
    return verdict, reasons


def parity_report(train: pd.DataFrame, live: pd.DataFrame, features: list, track: str,
                  model: str) -> dict:
    rows, counts = {}, {"PASS": 0, "WARN": 0, "FAIL": 0}
    for f in features:
        tcol = train[f] if f in train.columns else pd.Series(dtype=float)
        lcol = live[f] if f in live.columns else pd.Series([np.nan] * len(live), dtype=float)
        s = feature_stats(tcol, lcol)
        if f not in live.columns:
            s["note"] = "feature column absent from the live board"
        v, r = classify(f, s)
        s["status"], s["reasons"] = v, r
        rows[f] = s
        counts[v] += 1
    status = "FAIL" if counts["FAIL"] else ("WARN" if counts["WARN"] else "PASS")
    return {"track": track, "model": model, "rule_version": PARITY_RULE_VERSION,
            "train_rows": int(len(train)), "live_rows": int(len(live)), "status": status,
            "counts": counts, "features": rows}
