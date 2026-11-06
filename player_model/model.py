"""
Player Prop ML Models v2
========================
One model per market. Uses Platt scaling (sigmoid calibration) instead of
IsotonicRegression — better calibration on small samples per the Senior ML audit.
"""
from __future__ import annotations

import pickle
import warnings
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
from . import props_contract as PC

TARGETS = config.MARKET_TARGETS


def _prep(df: pd.DataFrame, feat_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Legacy (batch-median) preprocessing. Data Truth (2026-11-01): a feature column that is
    absent, or has no value anywhere in the batch, stays NaN (it used to become 0.0 for every
    row). Only the NaN-native estimator can score such rows; see _ensemble_proba."""
    cols = feat_cols or [c for c in config.PLAYER_FEATURE_COLS if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        import logging
        logging.getLogger(__name__).warning(
            f"[predict] {len(missing)} feature cols absent from feat_df, left NaN: {missing[:8]}")
        df = df.copy()
        for c in missing:
            df[c] = np.nan
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return X, cols


def _nan_native(model) -> bool:
    """True when the calibrated model's estimator accepts NaN (HistGradientBoosting)."""
    pipe = getattr(model, "_pipe", model)
    final = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    return isinstance(final, HistGradientBoostingClassifier)


_ENSEMBLE_LOG: dict = {"rows_nan_native_only": 0, "rows_unscorable": 0}


# ── MEDIAN-FILL TELEMETRY (owner decision 2026-11-04, fix round 10) ───────────────────────────────
# The serving props models fill a missing feature with a MEDIAN: the legacy payloads (every serving
# model until the Phase 2 retrain) with the median of the scoring batch (_prep), a repaired payload
# with the FIT-block median of its persisted PropsImputer (its SEMANTIC_ZERO features are filled with
# 0, not a median, and are not counted). How many cells each run fills was never measured. This
# counts them per (market, feature) while predict_proba runs, READ-ONLY - it never touches the
# matrix the model scores, so serving is unchanged (owner rule: no serving change until the Phase 2
# retrain). pipeline --mode predict resets it before the run and appends one block per run to
# output/props_median_fill.csv (write_median_fill).
_MEDIAN_FILL: dict = {}          # (market, imputer) -> {"rows": int, "cells": {feature: int}}
MEDIAN_FILL_FILE = config.OUTPUT_DIR / "props_median_fill.csv"
MEDIAN_FILL_COLS = ["run_at", "run_id", "market", "imputer", "rows_scored", "feature", "cells_filled"]


def median_fill_reset() -> None:
    _MEDIAN_FILL.clear()


def _record_median_fill(market, imputer: str, df: pd.DataFrame, X: pd.DataFrame,
                        skip: frozenset = frozenset()) -> None:
    """Count the cells of X that were missing in the raw input and hold a value after the median
    fill (a column with no value anywhere stays NaN in the legacy path and is not a fill)."""
    try:
        from collections import Counter
        slot = _MEDIAN_FILL.setdefault((str(market), imputer), {"rows": 0, "cells": Counter()})
        slot["rows"] += int(len(X))
        for c in X.columns:
            if c in skip:
                continue
            raw = (pd.to_numeric(df[c], errors="coerce") if c in df.columns
                   else pd.Series(np.nan, index=df.index))
            n = int((raw.isna().to_numpy() & X[c].notna().to_numpy()).sum())
            if n:
                slot["cells"].update({c: n})          # a tally (Counter), not a default value
    except Exception as e:                      # telemetry must never break scoring
        import logging
        logging.getLogger(__name__).warning(f"[props_model] median-fill telemetry skipped: {e}")


def median_fill_rows(run_at: str, run_id: str = "") -> list[dict]:
    """This run's counts as rows: one per (market, imputer, feature) with a fill; a market scored
    with no fill at all gets one row with an empty feature and 0, so every scored market is seen."""
    rows = []
    for (market, imputer), slot in sorted(_MEDIAN_FILL.items()):
        cells = slot["cells"]
        if not cells:
            rows.append({"run_at": run_at, "run_id": run_id, "market": market, "imputer": imputer,
                         "rows_scored": slot["rows"], "feature": "", "cells_filled": 0})
        for f in sorted(cells):
            rows.append({"run_at": run_at, "run_id": run_id, "market": market, "imputer": imputer,
                         "rows_scored": slot["rows"], "feature": f, "cells_filled": cells[f]})
    return rows


def write_median_fill(dst=None, run_at: str | None = None, run_id: str | None = None) -> int:
    """Append this run's median-fill counts to output/props_median_fill.csv. Returns rows written."""
    import os
    from datetime import datetime, timezone
    from pathlib import Path
    dst = Path(dst or MEDIAN_FILL_FILE)
    run_at = run_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = os.getenv("GITHUB_RUN_ID", "") if run_id is None else run_id
    rows = median_fill_rows(run_at, run_id)
    if not rows:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MEDIAN_FILL_COLS).to_csv(
        dst, mode="a", header=not dst.exists(), index=False, lineterminator="\n")
    return len(rows)


def _ensemble_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Mean probability over the payload's models. A row that still has a missing input after the
    legacy batch-median step is scored ONLY by the NaN-native estimator(s) (no invented value is
    fed to the logistic model); a row no estimator can score gets NaN (no tip). Counts are logged
    by predict_proba."""
    complete = X.notna().all(axis=1).to_numpy()
    out = np.full(len(X), np.nan)
    probs_all, probs_nan = [], []
    for m in models.values():
        if _nan_native(m):
            probs_nan.append(m.predict_proba(X)[:, 1])
        elif complete.any():
            p = np.full(len(X), np.nan)
            p[complete] = m.predict_proba(X[complete])[:, 1]
            probs_all.append(p)
    if probs_all or probs_nan:
        stack = np.vstack(probs_all + probs_nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN column -> NaN, intended
            out = np.nanmean(stack, axis=0)
    if not probs_nan:
        out[~complete] = np.nan
    _ENSEMBLE_LOG["rows_nan_native_only"] += int((~complete).sum()) if probs_nan else 0
    _ENSEMBLE_LOG["rows_unscorable"] += int(np.isnan(out).sum())
    return out


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


def recipe_estimators() -> dict:
    """The props recipe's two unfitted estimators (one definition: train() and the Phase 2 props
    harness, player_model.props_harness, build them here, so a harness candidate is the recipe)."""
    return {
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


def split_bounds(n: int) -> tuple[int, int]:
    """(split, cal_split) row indices of the chronological fit | calibration | holdout split.
    ONE definition, used by train() and by the promotion gate's holdout, so they cannot diverge."""
    split = int(n * 0.80)
    return split, int(split * 0.85)


def train(df: pd.DataFrame, market: str, return_holdout: bool = False, recipe: str = "legacy"):
    """Train LogReg + GradientBoosting with Platt calibration for one market.

    recipe="legacy" (DEFAULT, what the weekly retrain and the promotion gate use): unchanged
    preprocessing, `_prep` batch medians over the whole frame (holdout included).
    recipe="props_r2_repaired" (R2-D challenger, never served without owner approval): the
    persisted PropsImputer fitted on the FIT block only, applied identically to calibration,
    holdout and live; REQUIRED_FAIL_ROW rows are excluded from fit/calibration and get no
    holdout probability. Same split bounds, same estimators, same hyper-parameters.

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
    n = len(df)
    if recipe == "legacy":
        X, feat_cols = _prep(df)
        # A feature with no value in the whole training frame carries nothing to learn and has no
        # median: dropped from this candidate (logged), never filled with a typed 0.0.
        _dead = [c for c in feat_cols if X[c].isna().all()]
        if _dead:
            print(f"  [{market}] {len(_dead)} feature(s) with no value in training dropped: {_dead[:12]}")
            feat_cols = [c for c in feat_cols if c not in _dead]
            X = X[feat_cols]
    elif recipe == PC.RECIPE_REPAIRED:
        if n < 200:
            raise ValueError(f"Not enough samples for {market}: {n}")
        _split, _cal = split_bounds(n)
        # FIT block only: calibration and holdout statistics never reach the imputer.
        imputer = PC.PropsImputer().fit(df.iloc[:_cal], list(config.PLAYER_FEATURE_COLS))
        X, diag = imputer.transform(df)
        feat_cols = list(imputer.feature_cols_)
    else:
        raise ValueError(f"unknown props recipe {recipe!r}")

    if n < 200:
        raise ValueError(f"Not enough samples for {market}: {n}")

    split, cal_split = split_bounds(n)

    X_fit, y_fit   = X.iloc[:cal_split], y.iloc[:cal_split]
    X_cal, y_cal   = X.iloc[cal_split:split], y.iloc[cal_split:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    r2_meta = None
    if recipe == PC.RECIPE_REPAIRED:
        ok = ~diag["required_fail"].to_numpy()
        fit_ok, cal_ok, test_ok = ok[:cal_split], ok[cal_split:split], ok[split:]
        X_fit, y_fit = X_fit[fit_ok], y_fit[fit_ok]
        X_cal, y_cal = X_cal[cal_ok], y_cal[cal_ok]
        X_test, y_test = X_test[test_ok], y_test[test_ok]
        _d = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else None

        def _rng(a, b):
            if _d is None or b <= a:
                return None
            return [str(_d.iloc[a])[:10], str(_d.iloc[b - 1])[:10]]
        r2_meta = {
            "recipe": PC.RECIPE_REPAIRED,
            "imputer": imputer.state(),
            "imputer_version": PC.IMPUTER_VERSION,
            "contract_version": imputer.contract["contract_version"],
            "blocks": {"fit": [0, cal_split], "calibration": [cal_split, split], "holdout": [split, n],
                       "fit_dates": _rng(0, cal_split), "calibration_dates": _rng(cal_split, split),
                       "holdout_dates": _rng(split, n)},
            "required_fail_excluded": {"fit": int((~fit_ok).sum()),
                                       "calibration": int((~cal_ok).sum()),
                                       "holdout": int((~test_ok).sum())},
            "training_rows": int(n),
        }

    estimators = recipe_estimators()

    results = {}
    for name, est in estimators.items():
        est.fit(X_fit, y_fit)
        # Platt scaling — LogisticRegression on raw probabilities
        raw_cal = est.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        platt   = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(raw_cal, y_cal)

        calibrated = _PlattCalibratedModel(est, platt)
        p_test = calibrated.predict_proba(X_test)[:, 1]
        # A single-class holdout has no AUC (was reported as 0.5, i.e. "no skill").
        auc = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else float("nan")
        # labels=[0,1] so a single-class test slice (common for rare longshot markets like
        # sot3/goals2) does NOT raise -> otherwise train() aborts to [SKIP] and silently keeps
        # the STALE previous model for that market.
        ll  = log_loss(y_test, p_test, labels=[0, 1])

        results[name] = {
            "model":       calibrated,
            "feature_cols": feat_cols,
            "metrics":     {"auc": auc, "log_loss": ll, "n_train": split, "n_test": len(y_test)},
        }
        if r2_meta is not None:
            results[name]["r2"] = r2_meta
        if return_holdout:
            # The holdout probabilities exactly as THIS train() scored them, aligned to the
            # returned holdout frame (NaN where the repaired contract refused a row). Not part
            # of build_payload, so serving bytes are unaffected.
            full = np.full(n - split, np.nan)
            full[np.asarray(X_test.index) - split] = p_test
            results[name]["holdout_p"] = full
        print(f"  [{market}] {name}: AUC={auc:.3f}  LogLoss={ll:.3f}")

    if return_holdout:
        cutoff = None
        if "date" in df.columns and split > 0:
            cutoff = str(pd.to_datetime(df["date"].iloc[split - 1], errors="coerce"))[:10]
        return results, df.iloc[split:], cutoff
    return results


def build_payload(results: dict, market: str) -> dict:
    """The serving-file payload, exactly as the pre-gate save_model wrote it."""
    payload = {
        "market":      market,
        "models":      {k: v["model"] for k, v in results.items()},
        "feature_cols": results[next(iter(results))]["feature_cols"],
        "metrics":     {k: v["metrics"] for k, v in results.items()},
    }
    # R2 challenger metadata. Added ONLY for a repaired recipe, so a legacy payload's keys (and
    # therefore its pickle bytes / artifact sha) are exactly what they were before R2.
    r2 = results[next(iter(results))].get("r2")
    if r2:
        payload.update(r2)
    return payload


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


def payload_recipe(payload: dict) -> str:
    """A payload without a `recipe` key is a pre-R2 (legacy) payload - every serving props model
    as of 2026-11-01."""
    return payload.get("recipe") or PC.RECIPE_LEGACY


def predict_with_diagnostics(df: pd.DataFrame, payload: dict, _fill_telemetry: bool = False) -> pd.DataFrame:
    """Per-row probability plus the R2 integrity record. Columns: p, n_missing, n_degraded,
    n_required_missing, required_fail, imputer_version, contract_version, recipe.

    Repaired payload: persisted train-fitted imputer, batch-invariant, REQUIRED_FAIL_ROW -> p NaN.
    Legacy payload: EXACTLY the pre-R2 computation (batch medians, absent column -> 0.0); the
    diagnostics then describe the raw input and imputer_version says BATCH_MEDIAN_LEGACY."""
    if payload_recipe(payload) == PC.RECIPE_REPAIRED:
        imp = PC.PropsImputer.from_state(payload["imputer"])
        X, diag = imp.transform(df)
        if list(X.columns) != list(payload["feature_cols"]):
            raise ValueError("payload feature_cols differ from its persisted imputer")
        if _fill_telemetry:                      # serving calls only (predict_proba), not shadows
            zero = frozenset(f for f, c in (payload["imputer"].get("contract_classes") or {}).items()
                             if c == "SEMANTIC_ZERO")
            _record_median_fill(payload.get("market"), "fit_median", df, X, skip=zero)
        probas = [m.predict_proba(X)[:, 1] for m in payload["models"].values()]
        p = pd.Series(np.mean(probas, axis=0), index=df.index)
        p[diag["required_fail"].to_numpy()] = np.nan
        out = diag.copy()
        out.insert(0, "p", p)
        out["recipe"] = PC.RECIPE_REPAIRED
        return out
    feat_cols = payload["feature_cols"]
    X, _ = _prep(df, feat_cols)
    out = pd.DataFrame({"p": _ensemble_proba(payload["models"], X)}, index=df.index)
    raw = pd.DataFrame({c: (pd.to_numeric(df[c], errors="coerce") if c in df.columns
                            else pd.Series(np.nan, index=df.index)) for c in feat_cols},
                       index=df.index)
    out["n_missing"] = raw.isna().sum(axis=1).astype(int)
    out["n_degraded"] = -1          # legacy payloads carry no contract
    out["n_required_missing"] = -1
    out["required_fail"] = False    # legacy never fails a row (that is the defect)
    out["imputer_version"] = "BATCH_MEDIAN_LEGACY"
    out["contract_version"] = "NONE"
    out["recipe"] = PC.RECIPE_LEGACY
    return out


def predict_proba(df: pd.DataFrame, payload: dict) -> pd.Series:
    """Ensemble average probability from both models.

    Legacy payloads (no `recipe` key; all serving props models as of 2026-11-01) take the
    original code path unchanged - batch-median `_prep` - so serving probabilities are
    bit-identical to pre-R2 until an owner-approved retrain. Repaired payloads use the persisted
    train-fitted imputer and return NaN on rows that fail the feature contract."""
    if payload_recipe(payload) == PC.RECIPE_REPAIRED:
        return predict_with_diagnostics(df, payload, _fill_telemetry=True)["p"]
    feat_cols = payload["feature_cols"]
    X, _ = _prep(df, feat_cols)
    _record_median_fill(payload.get("market"), "batch_median_legacy", df, X)
    before = dict(_ENSEMBLE_LOG)
    p = pd.Series(_ensemble_proba(payload["models"], X), index=df.index)
    n_nn = _ENSEMBLE_LOG["rows_nan_native_only"] - before["rows_nan_native_only"]
    n_un = _ENSEMBLE_LOG["rows_unscorable"] - before["rows_unscorable"]
    if n_nn or n_un:
        print(f"[props_model] {payload.get('market')}: {n_nn}/{len(df)} row(s) with a missing input "
              f"scored by the NaN-native estimator only; {n_un} unscorable (no probability)")
    return p
