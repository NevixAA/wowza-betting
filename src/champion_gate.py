"""CHAMPION PROMOTION GATE — high-water mark (owner 2026-10-18).

    from src.champion_gate import decide
    v = decide(y, p_champion, p_challenger, leagues=..., dates=..., incumbent_exists=True)
    v["verdict"]   # PROMOTE | KEEP_CHAMPION | SHADOW_MORE

RETRAINING IS AN EXPERIMENT. PROMOTION IS A PRIVILEGE. Every retrain produces a CHALLENGER. It
replaces the serving champion only on convincing evidence that it is better, measured on the SAME
fixtures, the same labels and the same scoring function. "Not worse" is not enough any more.

────────────────────────────────────────────────────────────────────────────────────────────
WHAT CHANGED AND WHY
────────────────────────────────────────────────────────────────────────────────────────────
The previous live rule (pipeline._train_one, 2026-10-04) blocked a challenger that scored worse
and promoted ANY improvement, however small. On 2026-10-18 five of seven models promoted, every
one on a gain whose 95% block-bootstrap CI included zero (e.g. over15 -0.00006). That is churn,
not learning.

It also scored the wrong thing: the MEAN OF THE BASE MODELS' log losses, not the probabilities
production serves (the meta-stacked ensemble). The two disagree. On 2026-10-11 new-format was
BLOCKED because the base-model mean rose (0.68270 -> 0.68314) while the served ensemble was
clearly better (CI [-0.0052, -0.0011]). This gate scores the served probabilities.

────────────────────────────────────────────────────────────────────────────────────────────
EVALUATION LAYERS (brief §12)
────────────────────────────────────────────────────────────────────────────────────────────
  B. ROLLING FRESH OOS — the chronological holdout train_model already builds: the newest block,
     unseen by the challenger, after the champion's own training cutoff. It moves forward with
     every retrain, so it is never one holdout optimised against forever.
  C. SECOND CONFIRMATION — the LATER HALF of that holdout, scored on its own. A gain that lives
     only in the older half is not promoted.
  D. PROSPECTIVE SHADOW — a SHADOW_MORE challenger is parked (model_registry.set_challenger) and
     scores every new fixture next to the champion; Pro grades the head-to-head.
  A. A FROZEN HISTORICAL BENCHMARK is deliberately NOT a promotion input: any fixed past slice is
     inside both models' training data, so it measures memorisation. It would be a sanity check
     at best, and the prediction canary already plays that role.

────────────────────────────────────────────────────────────────────────────────────────────
THRESHOLDS — from this estate's evidence, not by feel
────────────────────────────────────────────────────────────────────────────────────────────
MATERIAL_LL = 0.001. The measured bias of the meta-stack leak is 0.0006-0.0014 log loss
(promotion_gate.MIN_EFFECT_LOG_LOSS); an improvement smaller than our own measurement error is
not one we can see. It is also roughly the half-width of the block-bootstrap CI on a 4,000-12,000
row holdout (2026-10-05..18 retrain_log: 0.0005-0.0021), so a gain below it is inside the noise.

LOCK-IN GUARD (brief §32). The bar is a promotion bar, not a freeze: a challenger whose CI still
admits a material gain goes to SHADOW_MORE and is measured prospectively; a materially better one
promotes the same day. KNOWN TENSION, recorded rather than hidden: a 2026-10-04 replay of 1,568
simulated comparisons found no promotion rule beat having no gate, and retraining beats freezing
(project_shadow_learning). The owner chose stability over that small average edge on 2026-10-18;
Pro's `model_lineage` report measures whether the choice costs anything, every week.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.promotion_gate import block_bootstrap_ci, prediction_canary

PROMOTE, KEEP, SHADOW = "PROMOTE", "KEEP_CHAMPION", "SHADOW_MORE"
VERDICTS = (PROMOTE, KEEP, SHADOW)

MIN_N = 1000                   # holdout rows before anything but KEEP is possible
MATERIAL_LL = 0.001            # see module docstring
MAX_BRIER_WORSE = 1e-4         # Brier may not deteriorate
MAX_ECE_WORSE = 0.01           # calibration may not deteriorate by more than this (ECE, absolute)
MIN_LEAGUE_N = 150             # below this a league's number is noise and may not veto
MAX_LEAGUE_REGRESSION = 0.02   # a league losing more log loss than this blocks promotion


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _metrics(y, p) -> dict:
    return {"n": int(len(y)), "log_loss": round(float(_ll(y, p).mean()), 6),
            "brier": round(float(np.mean((p - y) ** 2)), 6), "ece": round(_ece(y, p), 6)}


def _force_promote(model_type: str) -> bool:
    """Owner-approved ARCHITECTURE changes (new features, new target) cannot be scored against
    an incompatible champion; this is their explicit, logged door. Comma-separated model types."""
    raw = os.getenv("CHAMPION_FORCE_PROMOTE", "")
    return model_type in {s.strip() for s in raw.split(",") if s.strip()}


def decide(y, p_champion, p_challenger, *, model_type: str = "",
           leagues=None, dates=None, incumbent_exists: bool = True,
           same_rows: bool = True) -> dict:
    """Return the verdict and the full audit record (brief §52). NEVER RAISES — fail closed.

    `p_champion` and `p_challenger` must be the two models' SERVED probabilities for the same
    rows of the same holdout (`same_rows` is the caller's assertion that they are).
    """
    rec = {"verdict": KEEP, "reason": "", "holdout_n": 0, "ll_champion": None,
           "ll_challenger": None, "ll_delta": None, "ll_ci": None, "brier_delta": None,
           "ece_delta": None, "later_half_ll_delta": None, "league_regressions": [],
           "safe_to_run": False, "better_than_champion": False, "checks": {},
           "champion_metrics": None, "challenger_metrics": None}
    try:
        if not incumbent_exists:
            rec.update(verdict=PROMOTE, reason="BOOTSTRAP: no serving champion exists for this "
                       "model type", safe_to_run=True)
            return rec
        if not same_rows:
            rec["reason"] = "EVALUATION_FAILED: champion and challenger not scored on the same rows"
            return rec
        y = np.asarray(y, float)
        pc = np.asarray(p_champion, float)
        pn = np.asarray(p_challenger, float)
        if not (len(y) == len(pc) == len(pn)):
            rec["reason"] = "EVALUATION_FAILED: length mismatch between labels and predictions"
            return rec
        m = np.isfinite(y) & np.isfinite(pc) & np.isfinite(pn)
        if m.sum() < len(y):
            # Dropping rows from ONE side would break same-observations; dropping from both
            # keeps it. Report how many.
            rec["checks"]["dropped_non_finite"] = int((~m).sum())
        y, pc, pn = y[m], pc[m], pn[m]
        lg = None if leagues is None else pd.Series(list(leagues))[m].reset_index(drop=True)
        dt = None if dates is None else pd.to_datetime(pd.Series(list(dates))[m],
                                                       errors="coerce").reset_index(drop=True)
        n = int(len(y))
        rec["holdout_n"] = n
        if n < MIN_N:
            rec["reason"] = f"EVALUATION_FAILED: holdout n={n} < {MIN_N}"
            return rec

        cm, nm = _metrics(y, pc), _metrics(y, pn)
        rec["champion_metrics"], rec["challenger_metrics"] = cm, nm
        rec["ll_champion"], rec["ll_challenger"] = cm["log_loss"], nm["log_loss"]
        d = nm["log_loss"] - cm["log_loss"]
        rec["ll_delta"] = round(d, 6)
        rec["brier_delta"] = round(nm["brier"] - cm["brier"], 6)
        rec["ece_delta"] = round(nm["ece"] - cm["ece"], 6)

        per_row = _ll(y, pn) - _ll(y, pc)
        # Block bootstrap assumes chronological order — football is serially correlated.
        order = np.argsort(dt.to_numpy()) if dt is not None and dt.notna().all() else np.arange(n)
        lo, hi = block_bootstrap_ci(per_row[order])
        rec["ll_ci"] = [round(lo, 6), round(hi, 6)]

        # Layer C — the later half of the holdout on its own.
        later = order[n // 2:]
        rec["later_half_ll_delta"] = round(float(per_row[later].mean()), 6)

        ok_canary, why_canary = prediction_canary(pc, pn)
        rec["safe_to_run"] = bool(ok_canary)

        regs = []
        if lg is not None:
            for name, idx in pd.Series(range(n), index=lg.astype(str)).groupby(level=0):
                ii = idx.to_numpy()
                if len(ii) >= MIN_LEAGUE_N:
                    dl = float(per_row[ii].mean())
                    if dl > MAX_LEAGUE_REGRESSION:
                        regs.append({"league": name, "n": int(len(ii)), "ll_delta": round(dl, 5)})
        rec["league_regressions"] = regs

        checks = {
            "material_improvement": d <= -MATERIAL_LL,
            "ci_excludes_zero": hi < 0,
            "second_period_confirms": rec["later_half_ll_delta"] < 0,
            "brier_not_worse": rec["brier_delta"] <= MAX_BRIER_WORSE,
            "calibration_held": rec["ece_delta"] <= MAX_ECE_WORSE,
            "no_protected_league_regression": not regs,
            "prediction_canary": ok_canary,
        }
        rec["checks"].update({k: bool(v) for k, v in checks.items()})
        rec["better_than_champion"] = bool(d < 0 and hi < 0)

        if _force_promote(model_type) and ok_canary:
            rec.update(verdict=PROMOTE, reason="OWNER_OVERRIDE via CHAMPION_FORCE_PROMOTE "
                       f"(delta {d:+.5f}, CI [{lo:+.5f}, {hi:+.5f}])")
            return rec
        if not ok_canary:
            rec.update(verdict=KEEP, reason=f"challenger failed the prediction canary: {why_canary}")
        elif d >= 0:
            rec.update(verdict=KEEP, reason=f"challenger not better: delta {d:+.6f} "
                       f"({cm['log_loss']:.5f} -> {nm['log_loss']:.5f}); worse or tied keeps "
                       f"the champion")
        elif all(checks.values()):
            rec.update(verdict=PROMOTE, reason=f"materially better: delta {d:+.5f}, CI "
                       f"[{lo:+.5f}, {hi:+.5f}], later half {rec['later_half_ll_delta']:+.5f}, "
                       f"Brier {rec['brier_delta']:+.5f}, ECE {rec['ece_delta']:+.4f}")
        elif lo <= -MATERIAL_LL:
            failed = [k for k, v in checks.items() if not v]
            rec.update(verdict=SHADOW, reason=f"possibly better (delta {d:+.5f}, CI [{lo:+.5f}, "
                       f"{hi:+.5f}]) but not proven: {', '.join(failed)}")
        else:
            rec.update(verdict=KEEP, reason=f"statistically tied: delta {d:+.6f}, CI [{lo:+.5f}, "
                       f"{hi:+.5f}] does not even admit a material ({MATERIAL_LL}) gain")
        return rec
    except Exception as e:                                             # noqa: BLE001
        rec.update(verdict=KEEP, reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")
        return rec
