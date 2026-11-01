"""New-format O/U 2.5 recalibration to OUTCOMES (owner decision 2026-11-01).

FINDING. On 457 settled new-format O/U 2.5 fixtures (output/v91_shadow.csv, first-sight p_v9) the
over rate was 58.4%, the market said 60.2% and the model 47.8%: overs under-rated by ~10pp, model
log loss worse than the market's (90% CI excludes zero). Staked NF Unders since 2026-08-10: 396
bets, 37.6% hit, -44.6u - more than the whole staked book's loss.

WHY NOT FIT ON THOSE 457 ROWS. They were predicted BEFORE the 2026-11-01 serving fixes (feature
parity, rest-day/HT alignment, safe joins, identity). On the fixed pipeline the live model-market
gap is -4.5pp, not ~-12pp; with the market's own historical +1.8pp tilt that puts today's bias near
-2.7pp. A shift fitted on the old rows would push the corrected model ~10pp up into an OVER bias.

WHAT THIS DOES. One parameter - a logit intercept shift b, by maximum likelihood on settled NF
fixtures predicted AFTER `SINCE` (the fixed pipeline):  p_cal = sigmoid(logit(p) + b). It is
APPLIED only when n >= MIN_N settled fixtures AND b's 95% CI excludes zero; before that it is
reported, never applied. Refitted from the committed log on every predict run (cheap), so it
tracks the model as it changes. Rollback: NF_CALIBRATION=0.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output" / "v91_shadow.csv"
SINCE = "2026-11-01T17:00:00Z"      # first board built with every 2026-11-01 serving fix
MIN_N = 150


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(log_path: Path = LOG, since: str = SINCE) -> dict:
    """{'n', 'shift', 'se', 'ci95', 'active', 'reason'} from settled NF O/U 2.5 first-sight rows."""
    out = {"since": since, "n": 0, "shift": 0.0, "se": None, "ci95": None, "active": False}
    try:
        s = pd.read_csv(log_path, low_memory=False)
    except Exception as e:                                             # noqa: BLE001
        return {**out, "reason": f"no log ({type(e).__name__})"}
    s = s[(s["model_type"].astype(str) == "new_format") & (s["market"].astype(str) == "ou25")]
    s = s[pd.to_datetime(s["logged_at"], errors="coerce", utc=True) >= pd.Timestamp(since)]
    hg, ag = pd.to_numeric(s["home_goals"], errors="coerce"), pd.to_numeric(s["away_goals"], errors="coerce")
    p = pd.to_numeric(s["p_v9"], errors="coerce")
    ok = hg.notna() & ag.notna() & p.notna()
    y, x = ((hg + ag)[ok] > 2.5).astype(float).to_numpy(), _logit(p[ok])
    n = int(ok.sum())
    out["n"] = n
    if n < 20:
        return {**out, "reason": f"only {n} settled fixtures since {since} (needs {MIN_N})"}
    b = 0.0
    for _ in range(50):                                                # Newton on the intercept
        q = 1 / (1 + np.exp(-(x + b)))
        g, h = float((y - q).sum()), float((q * (1 - q)).sum())
        if h <= 0:
            break
        step = g / h
        b += step
        if abs(step) < 1e-8:
            break
    q = 1 / (1 + np.exp(-(x + b)))
    se = 1 / math.sqrt(max(float((q * (1 - q)).sum()), 1e-9))
    lo, hi = b - 1.96 * se, b + 1.96 * se
    active = n >= MIN_N and (lo > 0 or hi < 0)
    reason = ("applied" if active else
              f"not applied: n={n} < {MIN_N}" if n < MIN_N else "not applied: shift CI includes zero")
    return {**out, "shift": round(b, 4), "se": round(se, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "active": bool(active), "reason": reason}


def apply(p: pd.Series, cal: dict) -> pd.Series:
    """Calibrated probabilities when `cal` is active (and NF_CALIBRATION != 0), else `p` unchanged."""
    if not cal.get("active") or os.getenv("NF_CALIBRATION", "1").strip() == "0":
        return p
    z = _logit(p.to_numpy(float)) + float(cal["shift"])
    return pd.Series(1 / (1 + np.exp(-z)), index=p.index).where(p.notna())
