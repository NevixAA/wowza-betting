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
    # xG and inside-box shots (API-Football cache; NaN for standard leagues → median-imputed)
    "home_xg_last5",        "away_xg_last5",
    "home_insidebox_last5", "away_insidebox_last5",
    # Season-to-date venue splits — complements last-5 with full-season tendencies
    "home_season_goals_h",    "away_season_goals_a",
    "home_season_conceded_h", "away_season_conceded_a",
    "home_cs_rate_h",         "away_cs_rate_a",
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
    # Phase 5: Formation / lineup features (from /fixtures/lineups, TTL 2h)
    "home_attack_formation",    # starting formation attacking intent 0-1
    "away_attack_formation",
    "combined_attack_intent",   # sum of both — overall match attacking expectation
    "home_forward_count",       # confirmed forwards in starting XI
    "away_forward_count",
    # Phase 6: Head-to-head history (from /fixtures/headtohead, TTL 168h)
    "h2h_over25_rate",          # % of last 10 H2H matches that ended over 2.5
    "h2h_avg_goals",            # avg goals per game in last 10 H2H
    "h2h_home_win_rate",        # % of last 10 H2H won by the home team
    # Phase 7: API-Football cross-validated odds (Bet365 via /odds, TTL 6h)
    "api_implied_over25",       # Bet365 implied P(over 2.5) — second bookmaker signal
    "api_overround",            # Bet365 overround — market tightness / confidence proxy
    # Phase 7 extensions — additional Bet365 market signals (same /odds call)
    "api_implied_btts",          # Bet365 implied P(BTTS) — independent O/U signal
    "api_implied_over35",        # Bet365 implied P(over 3.5) — high-scoring confirmation
    "api_implied_over15",        # Bet365 implied P(over 1.5) — defensive match signal
    "api_implied_draw",          # Bet365 implied P(draw) — draw prob correlates with under
    # Phase 9 — Season stage (from /fixtures/rounds, TTL 24h)
    "season_stage_ratio",        # current_round / total_rounds: late-season pressure
    "is_late_season",            # binary: stage_ratio >= 0.80
    # Phase 10 — Coach stability (from /coachs, TTL 7d)
    "home_coach_is_caretaker",   # binary: coach tenure < 14 days
    "away_coach_is_caretaker",
    # Rolling possession and blocked shots (from fixture stats backfill)
    "home_possession_last5",     # avg ball possession % home team last 5 matches
    "away_possession_last5",
    "home_blocked_last5",        # avg blocked shots last 5
    "away_blocked_last5",
    # Pitch type — artificial turf suppresses goals (Finland/Sweden/Norway)
    "home_pitch_artificial",
]


def _prep(df: pd.DataFrame, feat_cols: list[str] = None,
          imputer=None) -> tuple[pd.DataFrame, list[str]]:
    """Select and impute feature columns.

    imputer=None (LEGACY, every serving payload): fill with the median OF THIS FRAME — the
    batch-median defect (P0 bug 1): a row's value depends on the other rows passed with it.
    imputer=<manifest> (recipe v9_repaired): fill from the persisted TRAINING distribution, so a
    row's value depends only on that row. All declared features are kept, in payload order.
    """
    if imputer is not None:
        from src.imputer import apply_imputer
        cols = list(feat_cols or FEATURE_COLS)
        X, _deg, _rep = apply_imputer(df, imputer, cols)
        return X, cols
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
    feature_cols: "list[str] | None" = None,
    recipe: str = "legacy_v9",
) -> dict:
    """
    Train on the chronological training split, calibrate on the test split.
    sample_weight: array-like of per-row weights (time-decay etc.), same length as df.
    Returns dict: {model_name: {"model", "metrics", "feature_cols"}}

    recipe="legacy_v9" (DEFAULT — pipeline.mode_train never passes anything else) is the code
    below, unchanged, defects included: batch-median imputation (bug 1), the meta-logistic fitted
    AND reported on the same test block (bug 6), and weights indexed by a reset index (bug 8).
    recipe="v9_repaired" -> train_repaired() (research only).
    """
    if recipe == "v9_repaired":
        return train_repaired(df, target=target, sample_weight=sample_weight,
                              feature_cols=feature_cols)
    if recipe != "legacy_v9":
        raise ValueError(f"unknown model recipe {recipe!r}")
    import numpy as np
    df = df.dropna(subset=[target]).copy().sort_values("date").reset_index(drop=True)
    split   = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    # feature_cols=None -> full FEATURE_COLS (standard model, unchanged). A caller can pass a
    # reduced list (e.g. the new-format model drops features that are populated at train but
    # absent at predict, which _prep would otherwise impute to 0.0 and the scaler would blow up).
    X_train, feat_cols = _prep(train_df, feature_cols)
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
            n_jobs=-1,
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


# ── Recipe v9_repaired (Standard forensic repair 2026-10-28) — RESEARCH ONLY ──────

MODEL_RECIPE_VERSION = "v9_repaired-1.0"
#: chronological block boundaries (fractions of the time-ordered labelled rows). The holdout is
#: the newest 20% — the same rows `chronological_split` holds out — so legacy and repaired are
#: scored on IDENTICAL untouched fixtures.
FOUR_BLOCKS = {"base": (0.00, 0.64), "calibration": (0.64, 0.72),
               "meta": (0.72, 0.80), "holdout": (0.80, 1.00)}


class TrainResults(dict):
    """A plain {model_name: {...}} dict (every existing reader keeps working) that also carries
    the recipe metadata `build_payload` persists: imputer, blocks, versions."""
    def __init__(self, *a, meta: "dict | None" = None, **k):
        super().__init__(*a, **k)
        self.meta = dict(meta or {})


def _rows_with_weights(df: pd.DataFrame, target: str, sample_weight) -> pd.DataFrame:
    """Attach the weight TO THE ROW before any filter or sort (bug 8).

    A pd.Series is aligned by index label; an array must have one entry per row of `df` as
    passed. A length mismatch is an error — never a silent fallback to ones.
    """
    d = df.copy()
    if sample_weight is None:
        d["_w"] = 1.0
    elif isinstance(sample_weight, pd.Series):
        w = sample_weight.reindex(d.index)
        if w.isna().any():
            raise ValueError("sample_weight Series does not cover every row of df")
        d["_w"] = w.to_numpy(dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if len(w) != len(d):
            raise ValueError(f"sample_weight has {len(w)} entries for {len(d)} rows")
        d["_w"] = w
    d = d.dropna(subset=[target])
    tie = [c for c in ("league", "home_team", "away_team") if c in d.columns]
    return d.sort_values(["date"] + tie, kind="mergesort").reset_index(drop=True)


def four_block_split(df: pd.DataFrame, target: str, sample_weight=None) -> dict:
    """BASE TRAIN -> CALIBRATION -> META TRAIN -> UNTOUCHED HOLDOUT, strictly chronological.

    Boundaries are cut on whole dates where possible: a date that straddles a cut goes to the
    LATER block, so no block shares a matchday with the block before it.
    """
    d = _rows_with_weights(df, target, sample_weight)
    n = len(d)
    dates = pd.to_datetime(d["date"]).to_numpy()
    cuts = {}
    for name, (lo, hi) in FOUR_BLOCKS.items():
        cuts[name] = int(round(n * hi)) if hi < 1 else n
    bounds, start = {}, 0
    for name in FOUR_BLOCKS:
        end = cuts[name]
        if end < n and end > start:
            # move the cut back to the first row of the straddling date
            first_of_day = int(np.searchsorted(dates, dates[end], side="left"))
            end = max(start, first_of_day)
        bounds[name] = (start, end)
        start = end
    return {name: d.iloc[a:b].copy() for name, (a, b) in bounds.items()}


def train_repaired(df: pd.DataFrame, target: str = "over25", sample_weight=None,
                   feature_cols: "list[str] | None" = None) -> TrainResults:
    """Same model family as legacy (LR + GB + LightGBM, Platt, meta-logistic), with every P0
    training defect removed:

      bug 1  imputer fitted on BASE rows only, persisted, applied to every block and to serving;
      bug 6  calibrators fit on CALIBRATION, meta-logistic on META, all metrics on the untouched
             HOLDOUT — no reported number comes from rows any stage was fitted on;
      bug 8  weights travel with their rows (`_rows_with_weights`).
    """
    from src.imputer import fit_imputer, IMPUTER_VERSION
    feat_cols = list(feature_cols or FEATURE_COLS)
    blocks = four_block_split(df, target, sample_weight)
    base, cal, meta_b, hold = (blocks["base"], blocks["calibration"], blocks["meta"],
                               blocks["holdout"])
    if min(len(base), len(cal), len(meta_b), len(hold)) < 20:
        raise ValueError(f"too few rows for four-block training: "
                         f"{[len(b) for b in blocks.values()]}")
    imputer = fit_imputer(base, feat_cols)
    X_base, _ = _prep(base, feat_cols, imputer=imputer)
    X_cal, _ = _prep(cal, feat_cols, imputer=imputer)
    X_meta, _ = _prep(meta_b, feat_cols, imputer=imputer)
    X_hold, _ = _prep(hold, feat_cols, imputer=imputer)
    y_base, y_cal = base[target].to_numpy(), cal[target].to_numpy()
    y_meta, y_hold = meta_b[target].to_numpy(), hold[target].to_numpy()
    w_base, w_cal = base["_w"].to_numpy(), cal["_w"].to_numpy()

    models_cfg = [
        ("logistic", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ("gradient_boost", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42)),
    ]
    try:
        from lightgbm import LGBMClassifier
        models_cfg.append(("lightgbm", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)))
    except ImportError:
        pass

    def _m(y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return {"accuracy": round(accuracy_score(y, p >= 0.5), 4),
                "auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
                "log_loss": round(log_loss(y, p, labels=[0, 1]), 5),
                "brier": round(float(np.mean((p - y) ** 2)), 5)}

    results = TrainResults()
    for name, base_clf in models_cfg:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
        pipe.fit(X_base, y_base, clf__sample_weight=w_base)
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(pipe.predict_proba(X_cal)[:, 1].reshape(-1, 1), y_cal, sample_weight=w_cal)
        calibrated = _CalibratedModel(pipe, platt)
        metrics = _m(y_hold, calibrated.predict_proba(X_hold)[:, 1])
        metrics.update(n_train=int(len(base)), n_cal=int(len(cal)), n_meta=int(len(meta_b)),
                       n_test=int(len(hold)), evaluated_on="holdout")
        results[name] = {"model": calibrated, "metrics": metrics, "feature_cols": feat_cols}

    served_hold = None
    if len(results) >= 2:
        meta_X = np.column_stack([results[k]["model"].predict_proba(X_meta)[:, 1] for k in results])
        meta_clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        meta_clf.fit(meta_X, y_meta)
        hold_X = np.column_stack([results[k]["model"].predict_proba(X_hold)[:, 1] for k in results])
        served_hold = meta_clf.predict_proba(hold_X)[:, 1]
        mm = _m(y_hold, served_hold)
        mm.update(n_meta=int(len(meta_b)), n_test=int(len(hold)), evaluated_on="holdout")
        results["__meta__"] = {"model": meta_clf, "metrics": mm, "feature_cols": list(results)}
    else:
        served_hold = results[next(iter(results))]["model"].predict_proba(X_hold)[:, 1]

    def _rng(b):
        return [str(pd.Timestamp(b["date"].min()).date()), str(pd.Timestamp(b["date"].max()).date())]
    results.meta = {
        "recipe": "v9_repaired", "model_recipe_version": MODEL_RECIPE_VERSION,
        "imputer": imputer.to_dict(), "imputer_version": IMPUTER_VERSION,
        "blocks": {k: {"rows": int(len(b)), "dates": _rng(b)} for k, b in blocks.items()},
        "served_holdout_metrics": _m(y_hold, served_hold),
        "target": target,
    }
    log.info(f"  [v9_repaired] served holdout {results.meta['served_holdout_metrics']}  "
             f"blocks={ {k: v['rows'] for k, v in results.meta['blocks'].items()} }")
    return results


def build_payload(results: dict, target: str = "over25") -> dict:
    """The payload `save_models` writes. Legacy results -> exactly the legacy dict; repaired
    results additionally carry `imputer`, `recipe` and the block/version metadata."""
    payload = {
        "target":       target,
        "models":       {k: v["model"] for k, v in results.items()},
        "feature_cols": results[next(iter(results))]["feature_cols"],
        "metrics":      {k: v["metrics"] for k, v in results.items()},
    }
    payload.update(getattr(results, "meta", {}) or {})
    return payload


def score_served(payload: dict, df: pd.DataFrame, target: str) -> dict:
    """Log loss / Brier of the probability predict_proba actually SERVES (bug 7).

    `score_payload` averages the sub-models' log losses and skips `__meta__`, but production
    serves the meta-logistic blend, so that number describes a model nobody serves. This scores
    the served probability. Diagnostics per sub-model are included, never used for a decision.
    """
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    p = np.clip(np.asarray(predict_proba(df, payload=payload), dtype=float), 1e-15, 1 - 1e-15)
    ok &= np.isfinite(p)
    out = {"n": int(ok.sum()),
           "log_loss": float(log_loss(y[ok], p[ok], labels=[0, 1])) if ok.any() else float("nan"),
           "brier": float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan"),
           "submodel_mean_log_loss_diagnostic": score_payload(payload, df[ok], target)}
    return out


def compare_served(champion: dict, challenger: dict, df: pd.DataFrame, target: str) -> dict:
    """Canonical comparison: served champion probability vs served challenger probability on the
    IDENTICAL rows. Negative `delta_log_loss` = challenger better. Read-only; decides nothing."""
    a, b = score_served(champion, df, target), score_served(challenger, df, target)
    return {"n": a["n"], "champion": a, "challenger": b,
            "delta_log_loss": b["log_loss"] - a["log_loss"],
            "delta_brier": b["brier"] - a["brier"]}


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(results: dict, target: str = "over25",
                model_file: Optional[Path] = None) -> None:
    """Write a trained payload to `model_file` — NEVER to a serving model file.

    Since 2026-10-25 the serving files (config.MODEL_FILE_*, HT_MODEL_FILE_*) are written only by
    src/promotion_gate.py, so a research or one-off script cannot silently change what serves.
    """
    path = Path(model_file) if model_file else config.MODEL_FILE
    serving = {Path(p).resolve() for p in (
        config.MODEL_FILE_STANDARD, config.MODEL_FILE_NEWFORMAT, config.MODEL_FILE_BTTS,
        config.MODEL_FILE_OVER15, config.MODEL_FILE_OVER35, config.HT_MODEL_FILE_05,
        config.HT_MODEL_FILE_15)}
    if path.resolve() in serving:
        raise PermissionError(f"{path.name} is a SERVING model file; only src/promotion_gate.py "
                              f"may write it (submit the candidate to the gate instead)")
    payload = build_payload(results, target)
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


def chronological_split(df: pd.DataFrame, target: str, train_ratio: float = 0.8):
    """The exact split `train_model` will use. Exposed so the promotion gate can score the
    INCUMBENT on the same rows the candidate is scored on."""
    d = df.dropna(subset=[target]).copy().sort_values("date").reset_index(drop=True)
    return d.iloc[:int(len(d) * train_ratio)], d.iloc[int(len(d) * train_ratio):]


def score_payload(payload: dict, df: pd.DataFrame, target: str) -> float:
    """Mean per-sub-model log loss of an already-trained payload on `df`.

    WHY THIS EXISTS. The promotion gate used to compare the incumbent's log loss AS RECORDED
    WHEN IT WAS TRAINED against the candidate's log loss on today's holdout. Those are two
    different test sets, and on 2026-09-23 the difference stopped being academic: a data fix grew
    the training frame 33-56% overnight and every model reported a large "improvement"
    (standard 0.68885 -> 0.64066) that was really a different, larger holdout. A controlled
    experiment on IDENTICAL test fixtures the same day found no improvement at all from the extra
    data, so the two cannot both be measuring model quality.

    Scoring the incumbent here, on the candidate's own holdout, makes the comparison mean what it
    appears to mean. It costs one prediction pass and no retraining.

    Deliberately the MEAN OF SUB-MODEL log losses rather than the ensemble's, because that is
    what the candidate side reports; comparing a mean against an ensemble would swap one
    apples-to-oranges problem for another.
    """
    import numpy as np
    from sklearn.metrics import log_loss as _ll
    X, _ = _prep(df, payload.get("feature_cols"), imputer=payload.get("imputer"))
    y = df[target].values
    out = []
    for name, m in (payload.get("models") or {}).items():
        if name == "__meta__":
            continue
        try:
            p = np.clip(m.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
            out.append(_ll(y, p))
        except Exception:                                            # noqa: BLE001
            continue
    return float(np.mean(out)) if out else float("nan")


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
    # payload["imputer"] exists only on v9_repaired payloads; serving payloads have none, so
    # this is the legacy call exactly.
    X, _       = _prep(df, feat_cols, imputer=payload.get("imputer"))
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
