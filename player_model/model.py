"""
Player Prop ML Models v2
========================
One model per market. Uses Platt scaling (sigmoid calibration) instead of
IsotonicRegression — better calibration on small samples per the Senior ML audit.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

TARGETS = config.MARKET_TARGETS


def _prep(df: pd.DataFrame, feat_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = feat_cols or [c for c in config.PLAYER_FEATURE_COLS if c in df.columns]
    # Fill missing columns with 0 rather than crashing — graceful degradation at predict time
    missing = [c for c in cols if c not in df.columns]
    if missing:
        import logging
        logging.getLogger(__name__).warning(f"[predict] {len(missing)} feature cols missing from feat_df, filling 0: {missing[:8]}")
        df = df.copy()
        for c in missing:
            df[c] = 0.0
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median().fillna(0.0))
    return X, cols


class _UsableColumns(BaseEstimator, TransformerMixin):
    """Keep only columns that vary in the TRAINING data. sklearn 1.9's HistGradientBoosting
    binner raises on an all-NaN or single-valued column; a constant column carries no
    information for a tree anyway. Decided at fit time, applied identically at predict time."""

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.keep_ = [j for j in range(A.shape[1])
                      if np.isfinite(A[:, j]).sum() > 1
                      and np.nanmax(A[:, j]) > np.nanmin(A[:, j])]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class _PlattCalibratedModel:
    """Pipeline + Platt scaling (LogisticRegression on raw probabilities)."""
    def __init__(self, pipe, platt):
        self._pipe  = pipe
        self._platt = platt

    def predict_proba(self, X):
        raw = self._pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self._platt.predict_proba(raw)[:, 1]
        cal = np.clip(cal, 0.001, 0.999)
        return np.column_stack([1 - cal, cal])


def split_bounds(n: int) -> tuple[int, int]:
    """(split, cal_split) row indices of the chronological fit | calibration | holdout split.
    ONE definition, used by train() and by the promotion gate's holdout, so they cannot diverge."""
    split = int(n * 0.80)
    return split, int(split * 0.85)


def train(df: pd.DataFrame, market: str, return_holdout: bool = False):
    """Train LogReg + GradientBoosting with Platt calibration for one market.

    return_holdout=True returns (results, holdout_frame, training_cutoff): the untouched
    chronological holdout rows exactly as train() scored them (every column, so the promotion gate
    can score the serving champion on the SAME rows), and the date of the last row the candidate
    saw (fit + calibration)."""
    if "n_prev_games" in df.columns:
        df = df[df["n_prev_games"] >= 1].copy()
    elif "appearances" in df.columns:
        df = df[df["appearances"] >= config.MIN_APPEARANCES].copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    df = df.reset_index(drop=True)

    target_col = TARGETS[market]
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")

    # Rows with NO label are not trainable. target_sot4 is empty on 5,403 rows collected before
    # that target existed (friendlies, Ireland, Finland, internationals; kept by the union merge),
    # and `.astype(int)` raised on them -> mode_train caught the ValueError as "[SKIP] sot4", so
    # the sot4 model was never retrained. Found 2026-10-12. Dropping, not imputing: never invent
    # a label.
    n_unlabelled = int(df[target_col].isna().sum())
    if n_unlabelled:
        print(f"  [{market}] dropping {n_unlabelled:,} rows with no {target_col}")
        df = df[df[target_col].notna()].reset_index(drop=True)

    y = df[target_col].astype(int)
    X, feat_cols = _prep(df)

    n = len(df)
    if n < 200:
        raise ValueError(f"Not enough samples for {market}: {n}")

    split, cal_split = split_bounds(n)

    X_fit, y_fit   = X.iloc[:cal_split], y.iloc[:cal_split]
    X_cal, y_cal   = X.iloc[cal_split:split], y.iloc[cal_split:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    estimators = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ]),
        # HistGradientBoosting since 2026-10-13 (was classic GradientBoostingClassifier: single-
        # threaded, linear in rows; at 855k rows the weekly retrain took 3h50m and had silently
        # timed out every Sunday from 09-27). Same trees (300 x depth 3, lr 0.05, min leaf 20);
        # the key stays "gradient_boost" so predict and stored payloads are unchanged.
        # Validated in Pro first (src/studies/props_learner.py, output/studies/PROPS_LEARNER.md):
        # all 8 markets non-inferior under a rule fixed before the results, 6 of 8 significantly
        # better, ~70x faster. Owner-approved 2026-10-12.
        "gradient_boost": Pipeline([
            ("usable", _UsableColumns()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, early_stopping=False, random_state=42,
            )),
        ]),
    }

    results = {}
    for name, est in estimators.items():
        est.fit(X_fit, y_fit)
        # Platt scaling — LogisticRegression on raw probabilities
        raw_cal = est.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        platt   = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(raw_cal, y_cal)

        calibrated = _PlattCalibratedModel(est, platt)
        p_test = calibrated.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else 0.5
        # labels=[0,1] so a single-class test slice (common for rare longshot markets like
        # sot3/goals2) does NOT raise -> otherwise train() aborts to [SKIP] and silently keeps
        # the STALE previous model for that market.
        ll  = log_loss(y_test, p_test, labels=[0, 1])

        results[name] = {
            "model":       calibrated,
            "feature_cols": feat_cols,
            "metrics":     {"auc": auc, "log_loss": ll, "n_train": split, "n_test": len(y_test)},
        }
        print(f"  [{market}] {name}: AUC={auc:.3f}  LogLoss={ll:.3f}")

    if return_holdout:
        cutoff = None
        if "date" in df.columns and split > 0:
            cutoff = str(pd.to_datetime(df["date"].iloc[split - 1], errors="coerce"))[:10]
        return results, df.iloc[split:], cutoff
    return results


def build_payload(results: dict, market: str) -> dict:
    """The serving-file payload, exactly as the pre-gate save_model wrote it."""
    return {
        "market":      market,
        "models":      {k: v["model"] for k, v in results.items()},
        "feature_cols": results[next(iter(results))]["feature_cols"],
        "metrics":     {k: v["metrics"] for k, v in results.items()},
    }


def payload_bytes(results: dict, market: str) -> bytes:
    """Byte format of a serving file (pickle of build_payload), so readers are unchanged."""
    return pickle.dumps(build_payload(results, market))


def save_model(results: dict, market: str) -> None:
    """REMOVED as a write path (owner decision 2026-10-25). It used to overwrite the serving file
    unconditionally; the ONLY code that may now change a serving props model is
    player_model/props_gate.py (promotion, explicit bootstrap, logged rollback)."""
    raise PermissionError(
        f"save_model({market!r}) refused: props serving files are written only by "
        f"player_model.props_gate (canonical promotion gate, PG-PROPS)")


def load_model(market: str) -> Optional[dict]:
    path = config.MODEL_FILES[market]
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(df: pd.DataFrame, payload: dict) -> pd.Series:
    """Ensemble average probability from both models."""
    feat_cols = payload["feature_cols"]
    X, _ = _prep(df, feat_cols)
    probas = [m.predict_proba(X)[:, 1] for m in payload["models"].values()]
    return pd.Series(np.mean(probas, axis=0), index=df.index)
