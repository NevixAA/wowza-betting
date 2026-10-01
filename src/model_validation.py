"""Leak-free chronological evaluation of candidate probability models. RESEARCH HARNESS.

    from src.model_validation import evaluate_candidates
    res = evaluate_candidates(feat_df, target="over25")

THIS MODULE CHANGES NO PRODUCTION BEHAVIOUR. It does not train, save or serve the live models.
`src/model.py` is untouched; this exists to measure candidates honestly so a promotion decision
can rest on evidence instead of on a number that was never out-of-sample.

────────────────────────────────────────────────────────────────────────────────────────────
THE DEFECT THIS EXISTS TO MEASURE AROUND
────────────────────────────────────────────────────────────────────────────────────────────
`src/model.py` builds its meta-stack like this (lines 240-245):

    meta_X = column_stack([base.predict_proba(X_test) for base in results])
    meta_clf.fit(meta_X, y_test)                     # <- fitted ON THE TEST LABELS
    meta_proba = meta_clf.predict_proba(meta_X)      # <- scored on the SAME block
    roc_auc_score(y_test, meta_proba)                # <- therefore in-sample

The code's comment argues this is safe because the base models never saw X_test. That is true of
the BASE models and false of the META model, which is the one being scored. Two consequences:

  1. every meta metric ever reported is optimistic by an unknown amount;
  2. worse, the meta has now SEEN the holdout labels, so that block is burned for any
     downstream comparison — including the promotion gate, which scores incumbent against
     candidate on exactly that split.

The meta is not merely mis-measured, it is live: `predict_proba` uses it when present
(src/model.py:349). So this is a production model whose only evidence is in-sample.

────────────────────────────────────────────────────────────────────────────────────────────
THE CORRECTED ARCHITECTURE
────────────────────────────────────────────────────────────────────────────────────────────
Four chronological blocks, each used for exactly one job, no block reused:

    FIT        base learners trained here
    CAL        Platt/isotonic calibration of each base learner
    META       the stacker is trained here, on calibrated base predictions
    HOLDOUT    touched ONCE, by every candidate, for the reported numbers

No random split anywhere: football drifts, and a shuffled split lets a model learn from matches
that had not happened yet. The holdout is the last block in time, so every candidate is judged
on the most recent football, which is also the regime it would actually bet into.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Chronological block boundaries as cumulative fractions. The holdout is deliberately the
#: largest of the three evaluation blocks: it carries every reported number, so its sample size
#: is what every confidence interval is built from.
BLOCKS = (0.60, 0.72, 0.86, 1.00)      # fit | cal | meta | holdout

#: Below this a block cannot support a conclusion. The gate reports INSUFFICIENT rather than a
#: number, because a tiny holdout produces confident-looking metrics that mean nothing.
MIN_BLOCK = 300


@dataclass(frozen=True)
class Split:
    """Index ranges for the four blocks, plus the dataset identity they came from."""
    fit: tuple[int, int]
    cal: tuple[int, int]
    meta: tuple[int, int]
    holdout: tuple[int, int]
    n: int
    date_min: str
    date_max: str
    dataset_id: str

    def sizes(self) -> dict:
        return {k: v[1] - v[0] for k, v in
                (("fit", self.fit), ("cal", self.cal),
                 ("meta", self.meta), ("holdout", self.holdout))}

    def ok(self) -> bool:
        return min(self.sizes().values()) >= MIN_BLOCK


def dataset_id(df: pd.DataFrame, target: str, feature_cols: list[str]) -> str:
    """Stable identity for the exact data a result was produced on.

    Two results are only comparable if this matches. Without it, a "better" log loss can simply
    mean a different, easier test set — which is the failure mode the production promotion gate
    already has, where the test slice grows between runs and consecutive models are scored on
    different rows.
    """
    d = df.dropna(subset=[target])
    payload = {
        "n": int(len(d)),
        "target": target,
        "date_min": str(pd.to_datetime(d["date"], errors="coerce").min()),
        "date_max": str(pd.to_datetime(d["date"], errors="coerce").max()),
        "features": sorted(feature_cols),
        "base_rate": round(float(pd.to_numeric(d[target], errors="coerce").mean()), 6),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def model_sha(model) -> str:
    """Identity of a fitted estimator — its class and hyperparameters, not its weights.

    Enough to tell "the same architecture refitted" from "a different architecture", which is
    what a promotion record needs to be auditable.
    """
    try:
        params = {k: str(v) for k, v in sorted(model.get_params(deep=False).items())}
    except Exception:                                                 # noqa: BLE001
        params = {}
    payload = {"cls": type(model).__name__, "params": params}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def chronological_split(df: pd.DataFrame, target: str,
                        feature_cols: list[str], blocks=BLOCKS) -> Split:
    """Four non-overlapping blocks in date order. Sorted stably so ties cannot permute."""
    d = (df.dropna(subset=[target])
           .sort_values(["date"], kind="mergesort")      # stable: same input -> same split
           .reset_index(drop=True))
    n = len(d)
    b = [0] + [int(n * f) for f in blocks]
    return Split(fit=(b[0], b[1]), cal=(b[1], b[2]), meta=(b[2], b[3]), holdout=(b[3], b[4]),
                 n=n,
                 date_min=str(pd.to_datetime(d["date"], errors="coerce").min())[:10],
                 date_max=str(pd.to_datetime(d["date"], errors="coerce").max())[:10],
                 dataset_id=dataset_id(d, target, feature_cols))


# ── metrics ───────────────────────────────────────────────────────────────────────────────────
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |claimed - realised| weighted by bin population."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of a logistic fit of outcome on logit(p).

    Slope 1.0 / intercept 0.0 is perfect. Slope BELOW 1 is the signature of overconfidence —
    the model's probabilities are spread wider than reality supports, which is exactly the
    estate's measured defect (+12.6pp on settled bets).
    """
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    lr = LogisticRegression(max_iter=1000).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Every metric the promotion gate reads, from one pass over one block."""
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    slope, intercept = calibration_line(y, p)
    base = float(((y.mean() - y) ** 2).mean())
    brier = float(brier_score_loss(y, p))
    return {
        "n": int(len(y)),
        "log_loss": round(float(log_loss(y, p)), 6),
        "brier": round(brier, 6),
        "auc": round(float(roc_auc_score(y, p)), 6) if len(np.unique(y)) > 1 else float("nan"),
        "ece": round(ece(y, p), 6),
        "cal_slope": round(slope, 4),
        "cal_intercept": round(intercept, 4),
        "brier_skill": round(1 - brier / base, 6) if base else float("nan"),
        "claimed": round(float(p.mean()), 6),
        "realised": round(float(y.mean()), 6),
    }


# ── candidates ────────────────────────────────────────────────────────────────────────────────
def _base_models(seed: int = 42):
    """The same three learners v9 trains, so the comparison is like-for-like."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    import lightgbm as lgb
    return [
        ("logistic", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ("gradient_boost", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=seed)),
        ("lightgbm", lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)),
    ]


def _hgb():
    """Pro's strongest single learner. Best log-loss gain in Pro's own 4-fold chronological
    study on both BTTS (+0.0101 vs logreg's +0.0040) and Over 2.5 (+0.0142 vs +0.0115)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42)


def evaluate_candidates(df: pd.DataFrame, target: str = "over25",
                        feature_cols: list[str] | None = None,
                        blocks=BLOCKS) -> tuple[pd.DataFrame, Split]:
    """Score every candidate architecture on ONE untouched chronological holdout.

    Candidates, as §3 of the upgrade brief requires:
      v9_meta_leaky      reproduces production exactly, INCLUDING fitting the meta on the block
                         it is scored on — carried so the size of the illusion is measurable
      mean_blend         unweighted mean of the calibrated base learners
      meta_corrected     stacker trained on the META block, scored on HOLDOUT (the fix)
      best_single        the strongest individual base learner
      pro_hgb            Pro's HistGradientBoosting challenger

    Every candidate sees the same FIT and CAL blocks and is judged on the same HOLDOUT, so the
    differences are architecture and nothing else.
    """
    from sklearn.calibration import CalibratedClassifierCV
    # sklearn >= 1.6 removed cv='prefit'; a pre-fitted estimator is wrapped instead. Using
    # the old argument raises InvalidParameterError, and falling back to cv=N would REFIT on
    # the calibration block — which would quietly turn CAL into a second training block and
    # reintroduce exactly the kind of block reuse this module exists to prevent.
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from src.model import _prep, FEATURE_COLS

    cols = feature_cols or FEATURE_COLS
    d = (df.dropna(subset=[target]).sort_values(["date"], kind="mergesort")
           .reset_index(drop=True))
    sp = chronological_split(d, target, cols, blocks)
    if not sp.ok():
        log.warning(f"[validation] blocks too small for {target}: {sp.sizes()}")
        return pd.DataFrame(), sp

    X, used = _prep(d, cols)
    y = pd.to_numeric(d[target], errors="coerce").to_numpy(float)

    def blk(rng):
        a, b = rng
        return X.iloc[a:b], y[a:b]

    X_fit, y_fit = blk(sp.fit)
    X_cal, y_cal = blk(sp.cal)
    X_meta, y_meta = blk(sp.meta)
    X_hold, y_hold = blk(sp.holdout)

    # 1. base learners: fit on FIT, calibrate on CAL. Neither block is ever scored.
    calibrated, shas = {}, {}
    for name, est in _base_models():
        est.fit(X_fit, y_fit)
        cc = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid")
        cc.fit(X_cal, y_cal)
        calibrated[name] = cc
        shas[name] = model_sha(est)

    P_meta = np.column_stack([m.predict_proba(X_meta)[:, 1] for m in calibrated.values()])
    P_hold = np.column_stack([m.predict_proba(X_hold)[:, 1] for m in calibrated.values()])

    rows = []

    # 2. mean blend — no stacker, nothing to leak
    rows.append({"candidate": "mean_blend", **score(y_hold, P_hold.mean(axis=1)),
                 "model_sha": "-", "note": "unweighted mean of calibrated base learners"})

    # 3. best single learner, chosen ON THE META BLOCK so the holdout stays untouched.
    #    Choosing it on the holdout would be selection leakage — a quieter version of the same
    #    bug this module exists to fix.
    ll = {n: score(y_meta, calibrated[n].predict_proba(X_meta)[:, 1])["log_loss"]
          for n in calibrated}
    best = min(ll, key=ll.get)
    rows.append({"candidate": f"best_single[{best}]",
                 **score(y_hold, calibrated[best].predict_proba(X_hold)[:, 1]),
                 "model_sha": shas[best], "note": "selected on META block, scored on HOLDOUT"})

    # 4. corrected meta stack — trained on META, scored on HOLDOUT. This is the fix.
    meta = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(P_meta, y_meta)
    rows.append({"candidate": "meta_corrected",
                 **score(y_hold, meta.predict_proba(P_hold)[:, 1]),
                 "model_sha": model_sha(meta),
                 "note": "stacker fitted on META block only"})

    # 5. production's actual behaviour, leak and all — fitted on HOLDOUT and scored there.
    #    Reported so the gap between this and meta_corrected IS the size of the illusion.
    leaky = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(P_hold, y_hold)
    rows.append({"candidate": "v9_meta_leaky",
                 **score(y_hold, leaky.predict_proba(P_hold)[:, 1]),
                 "model_sha": model_sha(leaky),
                 "note": "REPRODUCES THE BUG — fitted on the block it is scored on"})

    # 6. Pro's HGB challenger, same fit/cal discipline
    h = _hgb().fit(X_fit, y_fit)
    hc = CalibratedClassifierCV(FrozenEstimator(h), method="sigmoid").fit(X_cal, y_cal)
    rows.append({"candidate": "pro_hgb", **score(y_hold, hc.predict_proba(X_hold)[:, 1]),
                 "model_sha": model_sha(h), "note": "Pro's strongest single learner"})

    out = pd.DataFrame(rows)
    out.insert(0, "target", target)
    out["dataset_id"] = sp.dataset_id
    out["n_fit"], out["n_cal"] = sp.sizes()["fit"], sp.sizes()["cal"]
    out["n_meta"], out["n_holdout"] = sp.sizes()["meta"], sp.sizes()["holdout"]
    out["holdout_from"] = sp.date_min
    out["holdout_to"] = sp.date_max
    return out.sort_values("log_loss").reset_index(drop=True), sp
