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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)


# ── Calibrated model wrapper (module-level so pickle works) ───────────────────

class _CalibratedModel:
    """Wraps a sklearn Pipeline + Platt scaling (sigmoid) calibrator.
    Platt scaling uses LogisticRegression on raw probabilities — regularized,
    no overfit risk on small calibration sets (unlike IsotonicRegression).
    """
    def __init__(self, pipe, calibrator):
        self._pipe       = pipe
        self._calibrator = calibrator

    def predict_proba(self, X):
        raw = self._pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self._calibrator.predict_proba(raw)[:, 1]
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
    "home_advantage",       "away_home_adv_factor",
    "home_rest_days",       "away_rest_days",
    # Set piece proxies — rolling historical averages (match-day actuals removed)
    "home_corners_pg_roll", "away_corners_pg_roll",
    "home_fouls_pg_roll",   "away_fouls_pg_roll",
    "combined_sot_ratio",
    # Dropped: referee_foul_avg, *_sp_goals_pg, *_pen_goals_pg, *_fk_goals_pg
    # All had 0% importance in both LR and GBM — pure noise (ML audit 2026-06-10)
    # Half-time rolling form (only populated for standard-format leagues)
    "home_ht_scored_last5",   "away_ht_scored_last5",
    "home_ht_conceded_last5", "away_ht_conceded_last5",
    "home_ht_attack_str",     "away_ht_attack_str",
    "combined_ht_goals_avg",
    # HT tendency rates (% of recent games with HT goal)
    "home_ht_over05_rate",    "away_ht_over05_rate",
    "home_ht_over15_rate",    "away_ht_over15_rate",
    # Market microstructure
    "bookmaker_overround",    # over-round tightness signals market confidence
    "p_over25_poisson_dc",    # Dixon-Coles corrected Poisson P(over 2.5)
]


def _prep(df: pd.DataFrame, feat_cols: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """Select and impute feature columns."""
    cols = [c for c in (feat_cols or FEATURE_COLS) if c in df.columns]
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    col_medians = X.median().fillna(0.0)
    X = X.fillna(col_medians)
    return X, cols


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    target: str = "over25",
    train_ratio: float = 0.8,
    sample_weight=None,
) -> dict:
    """
    Train on the chronological training split, calibrate on the test split.
    sample_weight: array-like of per-row weights (time-decay etc.), same length as df.
    Returns dict: {model_name: {"model", "metrics", "feature_cols"}}
    """
    import numpy as np
    df = df.dropna(subset=[target]).copy().sort_values("date").reset_index(drop=True)
    split   = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train, feat_cols = _prep(train_df)
    X_test,  _         = _prep(test_df, feat_cols)
    y_train = train_df[target].values
    y_test  = test_df[target].values

    # Align sample weights to the sorted/reset df
    if sample_weight is not None:
        w_all = np.array(sample_weight)[df.index] if len(sample_weight) == len(df) else \
                np.ones(len(df))
        w_train = w_all[:split]
    else:
        w_train = None

    # Hold out last 15% of train for calibration (still chronological, no leakage)
    cal_split  = int(len(X_train) * 0.85)
    X_fit, y_fit = X_train.iloc[:cal_split], y_train[:cal_split]
    X_cal, y_cal = X_train.iloc[cal_split:], y_train[cal_split:]
    w_fit = w_train[:cal_split] if w_train is not None else None
    w_cal = w_train[cal_split:] if w_train is not None else None

    models_cfg = [
        ("logistic", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ("gradient_boost", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )),
    ]

    # LightGBM as 3rd ensemble member — leaf-wise growth, ~10× faster than GBM
    try:
        from lightgbm import LGBMClassifier
        models_cfg.append(("lightgbm", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
        )))
    except ImportError:
        pass

    results = {}
    for name, base_clf in models_cfg:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    base_clf),
        ])
        fit_params = {}
        if w_fit is not None:
            fit_params["clf__sample_weight"] = w_fit
        pipe.fit(X_fit, y_fit, **fit_params)

        # Platt scaling calibration (sigmoid) on cal split — regularized, no overfit risk
        # Replaces IsotonicRegression which overfits on small cal sets (<300 samples)
        raw_proba_cal = pipe.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(raw_proba_cal, y_cal,
                  sample_weight=w_cal if w_cal is not None else None)

        calibrated = _CalibratedModel(pipe, platt)

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

    # ── Meta-logistic learned ensemble blend ─────────────────────────────────
    # Fit on the test split: base model predictions → logistic → final probability.
    # The test split was never seen by any base model, so there is no leakage.
    if len(results) >= 2:
        try:
            meta_X = np.column_stack([
                results[n]["model"].predict_proba(X_test)[:, 1] for n in results
            ])
            meta_clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            meta_clf.fit(meta_X, y_test)
            meta_proba = meta_clf.predict_proba(meta_X)[:, 1]
            log.info(
                f"  {'meta_logistic':20s}  "
                f"auc={round(roc_auc_score(y_test, meta_proba), 4):.3f}"
            )
            results["__meta__"] = {
                "model":        meta_clf,
                "metrics":      {},
                "feature_cols": list(results.keys()),
            }
        except Exception as e:
            log.warning(f"Meta-logistic training failed, falling back to mean: {e}")

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
    Uses meta-logistic learned blend if available, otherwise simple mean.
    """
    if payload is None:
        payload = load_models(model_file)

    feat_cols  = payload["feature_cols"]
    X, _       = _prep(df, feat_cols)
    base_models = {k: v for k, v in payload["models"].items() if k != "__meta__"}
    meta_model  = payload["models"].get("__meta__")

    probas = [m.predict_proba(X)[:, 1] for m in base_models.values()]

    if meta_model is not None and len(probas) >= 2:
        try:
            meta_X   = np.column_stack(probas)
            ensemble = meta_model.predict_proba(meta_X)[:, 1]
        except Exception:
            ensemble = np.mean(probas, axis=0)
    else:
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
