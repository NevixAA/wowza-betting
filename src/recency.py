"""
Derived recency weighting for the R2 repaired challengers (finding 5), FROZEN 2026-11-01.
========================================================================================

`config.TRAINING_DECAY_WEIGHTS` is a hand-typed {season label: weight} map. It froze: it has no
2025/26 or 2026/27 key, so the two newest seasons fell back to 1.0 while 2024/25 got 4.0, and
calendar-year labels ("2025") never matched at all. Repaired recipes never read it.

Instead a weight is a FUNCTION of the match date and an `as_of` date (the last training date,
never a holdout date). The candidate functions and their parameters below were written down
BEFORE any R2 challenger was evaluated and are not tuned afterwards; the holdout only REPORTS
them side by side. The primary candidate is fixed a priori (PRIMARY), so no holdout number
decides which weighting a challenger uses.

    season_age      weight by the age of the canonical, date-derived season (0 = as_of's season):
                    {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.5, older: 1.0}; COVID seasons 2019/20 and 2020/21
                    weight 0 (as the legacy map). Identical in shape to
                    src.standard_repair.derived_weights, which the serving Standard repaired model
                    was trained with - the continuity choice, hence PRIMARY.
    exp_half_life   0.5 ** (age_days / 365): one-year half-life.
    bounded_linear  max(0.25, 1 - age_days / 1460): linear over four years, floor 0.25.
    uniform         1.0 everywhere (the FULL / unweighted control).

Every payload persists the function name, its params, the effective weight and row count per
season, and the effective sample size  ESS = (sum w)^2 / sum w^2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RECENCY_VERSION = "r2-recency-1.0"
FROZEN_ON = "2026-11-01"
CANDIDATES = {
    "season_age": {"weights_by_age": {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.5}, "older": 1.0,
                   "zero_seasons": ["2019/20", "2020/21"]},
    "exp_half_life": {"half_life_days": 365.0},
    "bounded_linear": {"span_days": 1460.0, "floor": 0.25},
    "uniform": {},
}
PRIMARY = "season_age"


def _season_key(d: pd.Series, leagues: pd.Series) -> pd.Series:
    from src.features_repaired import season_key
    return season_key(d, leagues)


def weights(df: pd.DataFrame, function: str, as_of) -> pd.Series:
    """Index-aligned weights for `df` (needs date, league). `as_of` = last TRAINING date."""
    if function not in CANDIDATES:
        raise ValueError(f"unknown recency function {function!r}; frozen set: {sorted(CANDIDATES)}")
    p = CANDIDATES[function]
    d = pd.to_datetime(df["date"]).dt.normalize()
    ref = pd.Timestamp(as_of).normalize()
    if (d > ref).any():
        raise ValueError("recency weights asked for rows AFTER as_of (as_of must be the last "
                         "training date, never a holdout date)")
    age_days = (ref - d).dt.days.astype(float)
    if function == "uniform":
        w = pd.Series(1.0, index=df.index)
    elif function == "exp_half_life":
        w = 0.5 ** (age_days / p["half_life_days"])
    elif function == "bounded_linear":
        w = (1.0 - age_days / p["span_days"]).clip(lower=p["floor"])
    else:
        lg = df["league"].astype(str)
        sk = _season_key(d, lg)
        ref_sk = _season_key(pd.Series([ref] * len(df), index=df.index), lg)
        age = ref_sk.str.slice(0, 4).astype(int) - sk.str.slice(0, 4).astype(int)
        w = age.map(lambda a: p["weights_by_age"].get(int(a), p["older"])).astype(float)
        w[sk.isin(set(p["zero_seasons"]))] = 0.0
    return pd.Series(w.to_numpy(dtype=float), index=df.index)


def ess(w) -> float:
    w = np.asarray(w, dtype=float)
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def summary(df: pd.DataFrame, w: pd.Series, function: str, as_of) -> dict:
    """What gets persisted: function, params, effective weight per season, ESS."""
    sk = _season_key(pd.to_datetime(df["date"]), df["league"].astype(str))
    t = pd.DataFrame({"season": sk.to_numpy(), "w": np.asarray(w, float)})
    g = t.groupby("season")["w"].agg(["size", "mean", "sum"])
    tot = float(t["w"].sum()) or 1.0
    return {"version": RECENCY_VERSION, "frozen_on": FROZEN_ON, "function": function,
            "params": CANDIDATES[function], "as_of": str(pd.Timestamp(as_of).date()),
            "rows": int(len(t)), "ess": round(ess(t["w"]), 1),
            "ess_share": round(ess(t["w"]) / max(1, len(t)), 4),
            "per_season": {s: {"rows": int(r["size"]), "mean_weight": round(float(r["mean"]), 4),
                               "weight_share": round(float(r["sum"]) / tot, 4)}
                           for s, r in g.iterrows()}}
