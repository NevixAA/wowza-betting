"""
Player prop ML models — one per market.
Same pattern as src/model.py: LogReg + GradientBoosting + isotonic calibration.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

from . import config

# Target column per market
TARGETS = {
    "goals":   "target_goals",
    "assists": "target_assists",
    "sot":     "target_sot",
    "cards":   "target_cards",
}


def _binarise(df: pd.DataFrame, market: str) -> pd.Series:
    col = TARGETS[market]
    return df[col].astype(int)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in config.PLAYER_FEATURE_COLS if c in df.columns]
    X = df[feat_cols].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median().fillna(0.0))
    return X


def train(df: pd.DataFrame, market: str) -> dict:
    """
    Train LogReg + GradientBoosting with isotonic calibration for one market.
    Returns results dict with model, metrics, feature_cols.
    """
    df = df[df["appearances"] >= config.MIN_APPEARANCES].copy()
    df = df.sort_values("date").reset_index(drop=True)

    y = _binarise(df, market)
    X = _prep(df)
    feat_cols = list(X.columns)

    n = len(df)
    if n < 200:
        raise ValueError(f"Not enough samples for {market}: {n}")

    split     = int(n * 0.80)
    cal_split = int(split * 0.85)

    X_fit, y_fit = X.iloc[:cal_split], y.iloc[:cal_split]
    X_cal, y_cal = X.iloc[cal_split:split], y.iloc[cal_split:split]
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
        raw_cal = est.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal, y_cal)

        raw_test = est.predict_proba(X_test)[:, 1]
        p_test   = iso.transform(raw_test)
        auc  = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else 0.5
        ll   = log_loss(y_test, p_test)

        results[name] = {
            "model":       (est, iso),
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
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median().fillna(0.0))

    probas = []
    for est, iso in payload["models"].values():
        raw = est.predict_proba(X)[:, 1]
        probas.append(iso.transform(raw))

    return pd.Series(np.mean(probas, axis=0), index=df.index)
