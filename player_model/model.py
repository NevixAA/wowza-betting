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
from sklearn.ensemble import GradientBoostingClassifier
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


def train(df: pd.DataFrame, market: str) -> dict:
    """Train LogReg + GradientBoosting with Platt calibration for one market."""
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

    y = df[target_col].astype(int)
    X, feat_cols = _prep(df)

    n = len(df)
    if n < 200:
        raise ValueError(f"Not enough samples for {market}: {n}")

    split     = int(n * 0.80)
    cal_split = int(split * 0.85)

    X_fit, y_fit   = X.iloc[:cal_split], y.iloc[:cal_split]
    X_cal, y_cal   = X.iloc[cal_split:split], y.iloc[cal_split:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    estimators = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ]),
        "gradient_boost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
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
        ll  = log_loss(y_test, p_test)

        results[name] = {
            "model":       calibrated,
            "feature_cols": feat_cols,
            "metrics":     {"auc": auc, "log_loss": ll, "n_train": split, "n_test": len(y_test)},
        }
        print(f"  [{market}] {name}: AUC={auc:.3f}  LogLoss={ll:.3f}")

    return results


def save_model(results: dict, market: str) -> None:
    path = config.MODEL_FILES[market]
    payload = {
        "market":      market,
        "models":      {k: v["model"] for k, v in results.items()},
        "feature_cols": results[next(iter(results))]["feature_cols"],
        "metrics":     {k: v["metrics"] for k, v in results.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved {path.name}")


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
