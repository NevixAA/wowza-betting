"""
Model training, calibration, and evaluation.

Two models trained and ensembled:
  1. LogisticRegression         (fast, interpretable baseline)
  2. GradientBoostingClassifier (primary — best single-model performance)

Both are calibrated with isotonic regression (cv='prefit') on the test split.
Train/test split is strictly chronological — no shuffling, no leakage.

Ensemble: simple average of both calibrated models' probabilities.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)


# ── Calibrated model wrapper (module-level so pickle works) ───────────────────

class _CalibratedModel:
    """Wraps a sklearn Pipeline + IsotonicRegression calibrator."""
    def __init__(self, pipe, iso):
        self._pipe = pipe
        self._iso  = iso

    def predict_proba(self, X):
        raw = self._pipe.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        cal = np.clip(cal, 0.001, 0.999)
        return np.column_stack([1 - cal, cal])


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Rolling form
    "home_scored_last5",    "away_scored_last5",
    "home_conceded_last5",  "away_conceded_last5",
    "home_over25_last5",    "away_over25_last5",
    # Strength vs league
    "home_attack_str",      "away_attack_str",
    "home_defense_str",     "away_defense_str",
    # Match context
    "home_advantage",
    "home_rest_days",       "away_rest_days",
    # Odds-derived (market consensus)
    "implied_prob_over",    "implied_prob_under",
    # Set piece proxies (CSV)
    "combined_corners_pg",  "combined_fouls_pg",  "combined_sot_ratio",
    "referee_foul_avg",
    # Actual SP goals (Sofascore)
    "home_sp_goals_pg",     "away_sp_goals_pg",
    "home_pen_goals_pg",    "away_pen_goals_pg",
    "home_fk_goals_pg",     "away_fk_goals_pg",
    # Half-time rolling form (only populated for standard-format leagues)
    "home_ht_scored_last5",   "away_ht_scored_last5",
    "home_ht_conceded_last5", "away_ht_conceded_last5",
    "home_ht_attack_str",     "away_ht_attack_str",
    "combined_ht_goals_avg",
    # HT tendency rates (% of recent games with HT goal)
    "home_ht_over05_rate",    "away_ht_over05_rate",
    "home_ht_over15_rate",    "away_ht_over15_rate",
]


def _prep(df: pd.DataFrame, feat_cols: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """Select and impute feature columns."""
    cols = [c for c in (feat_cols or FEATURE_COLS) if c in df.columns]
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    # Fill with column median; for columns that are entirely NaN use a hard default
    col_medians = X.median()
    col_medians = col_medians.fillna(0.0)   # e.g. referee_foul_avg all-NaN → 0
    X = X.fillna(col_medians)
    # Referee foul avg: use 22.0 if still NaN (global football average)
    if "referee_foul_avg" in X.columns:
        X["referee_foul_avg"] = X["referee_foul_avg"].fillna(22.0)
    return X, cols


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    target: str = "over25",
    train_ratio: float = 0.8,
) -> dict:
    """
    Train on the chronological training split, calibrate on the test split.
    Returns dict: {model_name: {"model", "metrics", "feature_cols"}}
    """
    df = df.dropna(subset=[target]).copy().sort_values("date").reset_index(drop=True)
    split   = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train, feat_cols = _prep(train_df)
    X_test,  _         = _prep(test_df, feat_cols)
    y_train = train_df[target].values
    y_test  = test_df[target].values

    # Hold out last 15% of train for calibration (still chronological, no leakage)
    cal_split  = int(len(X_train) * 0.85)
    X_fit, y_fit = X_train.iloc[:cal_split], y_train[:cal_split]
    X_cal, y_cal = X_train.iloc[cal_split:], y_train[cal_split:]

    models_cfg = [
        ("logistic", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ("gradient_boost", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )),
    ]

    results = {}
    for name, base_clf in models_cfg:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    base_clf),
        ])
        pipe.fit(X_fit, y_fit)

        # Isotonic calibration on cal split (chronologically after fit split)
        raw_proba_cal = pipe.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_proba_cal, y_cal)

        calibrated = _CalibratedModel(pipe, iso)

        proba = calibrated.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy":  round(accuracy_score(y_test, proba >= 0.5), 4),
            "auc":       round(roc_auc_score(y_test, proba), 4),
            "log_loss":  round(log_loss(y_test, proba), 5),
            "n_train":   int(len(X_fit)),
            "n_cal":     int(len(X_cal)),
            "n_test":    int(len(X_test)),
        }
        log.info(
            f"  {name:20s}  acc={metrics['accuracy']:.3f}"
            f"  auc={metrics['auc']:.3f}  logloss={metrics['log_loss']:.4f}"
        )
        results[name] = {
            "model":        calibrated,
            "metrics":      metrics,
            "feature_cols": feat_cols,
        }

    return results


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(results: dict, target: str = "over25",
                model_file: Optional[Path] = None) -> None:
    path = Path(model_file) if model_file else config.MODEL_FILE
    payload = {
        "target":       target,
        "models":       {k: v["model"] for k, v in results.items()},
        "feature_cols": results[next(iter(results))]["feature_cols"],
        "metrics":      {k: v["metrics"] for k, v in results.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    metrics_stem = path.stem  # e.g. "model_v9_standard"
    metrics_file = config.MODELS_DIR / f"metrics_{metrics_stem}.json"
    metrics_file.write_text(
        json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2),
        encoding="utf-8",
    )
    log.info(f"Models saved → {path}")


def load_models(model_file: Optional[Path] = None) -> dict:
    path = Path(model_file) if model_file else config.MODEL_FILE
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba(df: pd.DataFrame, payload: Optional[dict] = None,
                  model_file: Optional[Path] = None) -> pd.Series:
    """
    Return ensemble P(over 2.5) for each row in df.
    Ensemble = simple average of all trained model probabilities.
    """
    if payload is None:
        payload = load_models(model_file)

    feat_cols = payload["feature_cols"]
    X, _ = _prep(df, feat_cols)

    probas = [
        model.predict_proba(X)[:, 1]
        for model in payload["models"].values()
    ]
    ensemble = np.mean(probas, axis=0)
    return pd.Series(ensemble, index=df.index, name="p_over25")


# ── Feature importances ───────────────────────────────────────────────────────

def get_feature_importances(payload: dict) -> pd.DataFrame:
    """
    Extract feature importances from the GradientBoosting model.
    Returns a sorted DataFrame.
    """
    gb_model = payload["models"].get("gradient_boost")
    feat_cols = payload["feature_cols"]

    if gb_model is None:
        return pd.DataFrame()

    try:
        # Our _CalibratedModel stores the pipeline as _pipe
        pipe = gb_model._pipe
        gb = pipe.named_steps["clf"]
        importances = gb.feature_importances_
        df = pd.DataFrame({
            "feature":    feat_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_%"] = (df["importance"] * 100).round(2)
        return df
    except Exception:
        return pd.DataFrame()
