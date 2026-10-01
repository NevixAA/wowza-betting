"""Champion / challenger promotion gate. Evidence required, not absence of catastrophe.

    from src.promotion_gate import evaluate_promotion
    verdict = evaluate_promotion(incumbent_scores, candidate_scores, y, p_inc, p_cand, leagues)

────────────────────────────────────────────────────────────────────────────────────────────
WHY THE OLD GATE HAD TO GO
────────────────────────────────────────────────────────────────────────────────────────────
`pipeline.py` promoted whenever the candidate's log loss had not risen by more than a tolerance
(0.030, briefly 0.005). "Not catastrophically worse" is not evidence of improvement, and the
retrain log shows what that permits: **17 of 56 logged decisions promoted a model that got
WORSE**, including

    2026-09-29 standard   0.68778 -> 0.68782   (+0.00004)  PROMOTED
    2026-09-26 ht_over05  0.64789 -> 0.66716   (+0.01927)  PROMOTED

Daily retraining under that rule is not learning; it is a random walk on the live model.

And the deltas being decided on are smaller than the measurement error in the thing measuring
them. The corrected-stack study found the meta-leak alone inflates log loss by 0.0006-0.0014,
while the gate was ruling on differences of 0.00004-0.003. A gate cannot resolve a signal
finer than its own bias.

────────────────────────────────────────────────────────────────────────────────────────────
WHAT REPLACES IT
────────────────────────────────────────────────────────────────────────────────────────────
Four explicit statuses, and a small regression stays a CHALLENGER rather than quietly shipping:

    CHAMPION    serving production
    CHALLENGER  measured, not yet better enough to replace the champion
    RESEARCH    promising but not yet measurable to the standard required
    REJECTED    materially worse, or failed an integrity check

A candidate must EARN promotion by clearing every check in `GATES`. There is no tolerance band
inside which a worse model ships. Improvement must also be distinguishable from noise: the
bootstrap CI on the delta has to exclude zero, which is the check that would have stopped all
17 of those promotions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHAMPION, CHALLENGER, RESEARCH, REJECTED = "CHAMPION", "CHALLENGER", "RESEARCH", "REJECTED"

#: Minimum holdout rows before any verdict other than RESEARCH is possible.
MIN_N = 1000
#: Minimum rows in a league before its regression can veto a promotion. Below this a bad
#: subgroup number is noise, and letting it block would make the gate superstitious.
MIN_LEAGUE_N = 150
#: Calibration may not deteriorate by more than this (ECE, absolute).
MAX_ECE_DETERIORATION = 0.01
#: A league may not lose more than this much log loss relative to the incumbent.
MAX_LEAGUE_REGRESSION = 0.02
#: Bootstrap resamples for the delta CI.
N_BOOT = 2000
#: MINIMUM MATERIAL IMPROVEMENT in log loss, on top of statistical significance.
#:
#: Significance alone is not enough. With a large holdout a 0.0000001 improvement can have a CI
#: that excludes zero, and promoting on it is the old random walk wearing a confidence interval.
#: The floor is set at 0.001 because that is ABOVE the measured bias of the meta-stack leak
#: (0.0006-0.0014 log loss): an improvement smaller than our own measurement error is not an
#: improvement we can see. A candidate below this stays a CHALLENGER.
MIN_EFFECT_LOG_LOSS = 0.001


@dataclass
class Verdict:
    status: str
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    delta_log_loss: float | None = None
    delta_brier: float | None = None
    ci_log_loss: tuple[float, float] | None = None
    n: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_log_loss"] = list(self.ci_log_loss) if self.ci_log_loss else None
        return d


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_bootstrap_ci(per_row_delta: np.ndarray, block: int = 50,
                       n_boot: int = N_BOOT, seed: int = 20261001) -> tuple[float, float]:
    """95% CI on the mean delta, resampling CONTIGUOUS BLOCKS rather than single rows.

    Football is serially correlated — a cold scoring week moves many fixtures together — so an
    iid bootstrap understates the interval and manufactures significance. Blocks preserve that
    dependence. Negative delta = the candidate is better (lower loss).
    """
    rng = np.random.default_rng(seed)
    n = len(per_row_delta)
    if n < block * 3:
        block = max(1, n // 10)
    starts = np.arange(0, max(n - block, 1))
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng.choice(starts, k, replace=True)])[:n]
        means[i] = per_row_delta[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prediction_canary(p_inc: np.ndarray, p_cand: np.ndarray) -> tuple[bool, str]:
    """Has the candidate's probability DISTRIBUTION collapsed or exploded?

    A model can improve log loss while becoming useless for selection — by predicting the base
    rate for everything. Spread is what a betting layer consumes, so a model that has lost it
    fails regardless of its loss. This is the check that would have caught the half-time models
    predicting a flat 0.70 for every fixture.
    """
    s_inc, s_cand = float(np.std(p_inc)), float(np.std(p_cand))
    if s_cand < 0.25 * s_inc:
        return False, f"probability spread collapsed: sd {s_inc:.4f} -> {s_cand:.4f}"
    # NO RATIO TEST ON EXPANSION. An earlier version failed a candidate whose spread grew from
    # sd 0.029 to 0.146 — but that was a near-flat incumbent being replaced by a model that
    # actually discriminates, which is the outcome we want. A wider spread is only a fault when
    # it is NOT supported by the outcomes, and that is calibration's job (the ECE and
    # cal_slope checks), not a crude ratio here. What remains is the structural pathology:
    # probabilities pinned against 0 or 1, which manufactures enormous fake edges.
    pinned = float(np.mean((p_cand < 0.02) | (p_cand > 0.98)))
    if pinned > 0.10:
        return False, f"{pinned:.1%} of predictions pinned at 0 or 1"
    return True, f"spread sd {s_inc:.4f} -> {s_cand:.4f}, {pinned:.1%} pinned"


def evaluate_promotion(inc: dict, cand: dict, y: np.ndarray,
                       p_inc: np.ndarray, p_cand: np.ndarray,
                       leagues: pd.Series | None = None,
                       dataset_id_inc: str | None = None,
                       dataset_id_cand: str | None = None,
                       second_period_delta: float | None = None) -> Verdict:
    """Decide CHAMPION / CHALLENGER / RESEARCH / REJECTED from evidence.

    `inc` and `cand` are the metric dicts from model_validation.score() on the SAME holdout.
    `second_period_delta` is the candidate's log-loss delta on an INDEPENDENT later period;
    None means the replication has not been run, which caps the verdict at CHALLENGER.
    """
    v = Verdict(status=RESEARCH, n=int(len(y)))
    y = np.asarray(y, float)
    d_ll = cand["log_loss"] - inc["log_loss"]
    d_br = cand["brier"] - inc["brier"]
    v.delta_log_loss, v.delta_brier = round(d_ll, 6), round(d_br, 6)

    def chk(name, ok, detail):
        v.checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            v.reasons.append(f"{name}: {detail}")
        return ok

    # 1. same data, or the comparison is meaningless
    same_data = (dataset_id_inc is None or dataset_id_cand is None
                 or dataset_id_inc == dataset_id_cand)
    chk("same_dataset", same_data,
        "identical dataset id" if same_data else
        f"scored on DIFFERENT data ({dataset_id_inc} vs {dataset_id_cand}) — not comparable")

    # 2. enough rows to resolve anything
    chk("min_sample", len(y) >= MIN_N, f"n={len(y)} (need >= {MIN_N})")

    # 3. log loss must not be worse. No tolerance band.
    chk("log_loss_not_worse", d_ll <= 0,
        f"delta {d_ll:+.6f} ({inc['log_loss']:.5f} -> {cand['log_loss']:.5f})")

    # 4. Brier must not materially deteriorate
    chk("brier_not_worse", d_br <= 1e-4, f"delta {d_br:+.6f}")

    # 5. improvement distinguishable from noise — the check the old gate lacked entirely
    per_row = _ll(y, np.asarray(p_cand, float)) - _ll(y, np.asarray(p_inc, float))
    lo, hi = block_bootstrap_ci(per_row)
    v.ci_log_loss = (round(lo, 6), round(hi, 6))
    chk("ci_excludes_zero", hi < 0,
        f"95% block-bootstrap CI [{lo:+.6f}, {hi:+.6f}] "
        f"{'excludes' if hi < 0 else 'includes'} zero")

    # 5b. and the improvement must be big enough to be worth acting on, not merely significant
    chk("material_improvement", d_ll <= -MIN_EFFECT_LOG_LOSS,
        f"delta {d_ll:+.6f} vs required <= {-MIN_EFFECT_LOG_LOSS:+.6f}")

    # 6. calibration must not materially deteriorate
    d_ece = cand["ece"] - inc["ece"]
    chk("calibration_held", d_ece <= MAX_ECE_DETERIORATION,
        f"ECE {inc['ece']:.4f} -> {cand['ece']:.4f} ({d_ece:+.4f})")

    # 7. distribution canary
    ok_canary, why = prediction_canary(np.asarray(p_inc, float), np.asarray(p_cand, float))
    chk("prediction_canary", ok_canary, why)

    # 8. no catastrophic league regression
    worst = None
    if leagues is not None and len(leagues) == len(y):
        rows = []
        for lg, idx in pd.Series(range(len(y)), index=list(leagues)).groupby(level=0):
            ii = idx.to_numpy()
            if len(ii) < MIN_LEAGUE_N:
                continue
            rows.append((lg, float(per_row[ii].mean()), len(ii)))
        if rows:
            worst = max(rows, key=lambda r: r[1])
        chk("no_league_regression", worst is None or worst[1] <= MAX_LEAGUE_REGRESSION,
            "none" if worst is None else
            f"worst league {worst[0]} delta {worst[1]:+.5f} on n={worst[2]}")
    else:
        v.checks["no_league_regression"] = {"pass": True, "detail": "no league labels supplied"}

    # 9. reproduced on a second, independent forward period
    chk("second_period", second_period_delta is not None and second_period_delta <= 0,
        "not run" if second_period_delta is None else f"delta {second_period_delta:+.6f}")

    # ── verdict ───────────────────────────────────────────────────────────────────────────
    hard = ("same_dataset", "min_sample", "prediction_canary")
    if not all(v.checks[k]["pass"] for k in hard):
        v.status = REJECTED if not v.checks["prediction_canary"]["pass"] else RESEARCH
    elif d_ll > 0.005 or not v.checks["brier_not_worse"]["pass"]:
        # materially worse — this is where 17 historical promotions actually belonged
        v.status = REJECTED
    elif all(c["pass"] for c in v.checks.values()):
        v.status = CHAMPION
    else:
        v.status = CHALLENGER
    return v
