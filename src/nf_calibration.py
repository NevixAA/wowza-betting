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


#: The serving-champion registry. The shift is fitted against ONE model's errors, so it may only be
#: fitted on rows that model predicted: rows logged before the serving NF champion was installed
#: belong to the previous model (owner 2026-11-09, clean retrain promotion).
CHAMPIONS = ROOT / "models" / "registry" / "current_champions.json"


def champion_since(path: Path = CHAMPIONS) -> str | None:
    """`champion_since` of the serving new-format O/U 2.5 champion, or None when it cannot be read."""
    import json
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8"))["newformat"]["champion_since"]
    except Exception:                                                  # noqa: BLE001
        return None
    return str(v) if v else None


def fit_window(since: str = SINCE, champions: Path = CHAMPIONS) -> tuple[str | None, str]:
    """The first logged_at a calibration row may carry: the later of the fixed-pipeline start and the
    serving champion's install time. (None, reason) when the champion's install time is unknown:
    a shift cannot be tied to the model it would correct, so nothing is fitted (fail closed)."""
    cs = champion_since(champions)
    if cs is None:
        return None, "serving new-format champion's champion_since unreadable - not fitted"
    a, b = pd.Timestamp(since), pd.Timestamp(cs)
    a = a.tz_localize("UTC") if a.tzinfo is None else a
    b = b.tz_localize("UTC") if b.tzinfo is None else b
    return (since, "fixed-pipeline start") if a >= b else (cs, "serving champion installed")


def fit(log_path: Path = LOG, since: str = SINCE, champions: Path = CHAMPIONS) -> dict:
    """{'n', 'shift', 'se', 'ci95', 'active', 'reason'} from settled NF O/U 2.5 first-sight rows
    predicted by the serving champion (rows logged since max(`since`, its champion_since))."""
    window, basis = fit_window(since, champions)
    out = {"since": window, "since_basis": basis, "n": 0, "shift": 0.0, "se": None, "ci95": None,
           "active": False}
    if window is None:
        return {**out, "reason": basis}
    since = window
    try:
        s = pd.read_csv(log_path, low_memory=False)
    except Exception as e:                                             # noqa: BLE001
        return {**out, "reason": f"no log ({type(e).__name__})"}
    from src.ledger_corrections import drop_v91_duplicates     # one fixture logged on two dates: once
    s = drop_v91_duplicates(s)
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
