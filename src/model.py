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

#: Calibrator / meta-logistic input scales (train_repaired calibration_input).
CALIBRATION_INPUTS = ("probability", "logit")
_LOGIT_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p / (1 - p))


class _CalibratedModel:
    """Wraps a sklearn Pipeline + Platt scaling (sigmoid) calibrator.
    Platt scaling uses LogisticRegression on raw probabilities — regularized,
    no overfit risk on small calibration sets (unlike IsotonicRegression).
    """
    def __init__(self, pipe, calibrator, calibration_input: str = "probability"):
        self._pipe       = pipe
        self._calibrator = calibrator
        # "probability" (every payload before 2026-11-11): Platt reads the raw probability.
        # "logit": Platt reads logit(raw probability), so the identity map is a coefficient of 1
        # and the L2 penalty no longer pulls every row toward the block's base rate.
        self._input      = calibration_input

    def predict_proba(self, X):
        raw = self._pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        if getattr(self, "_input", "probability") == "logit":   # pickles from before: attribute absent
            raw = _logit(raw)
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

def _legacy_sorted_with_weights(df: pd.DataFrame, target: str, sample_weight):
    """Labelled rows of `df` in date order, plus each row's OWN weight (or None).

    `sample_weight` is positional to `df` as passed (one entry per row) or, as the legacy contract
    allowed, to its labelled rows only. A pd.Series is aligned by index label. The weight is
    attached as a column BEFORE the sort, so it cannot be separated from its row. The sort call is
    the legacy one, so the row order (and the unweighted numerics) is unchanged.
    """
    lab = df[target].notna().to_numpy()
    d = df[lab].copy()
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            ws = sample_weight.reindex(d.index)
            if ws.isna().any():
                raise ValueError("sample_weight Series does not cover every labelled row of df")
            w = ws.to_numpy(dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if len(w) == len(df):
                w = w[lab]
            elif len(w) != len(d):
                raise ValueError(f"sample_weight has {len(w)} entries for {len(df)} rows "
                                 f"({len(d)} labelled) - compute the weights on the frame passed "
                                 f"to train()")
        d[WEIGHT_COL] = w
    d = d.sort_values("date").reset_index(drop=True)
    w_sorted = d.pop(WEIGHT_COL).to_numpy() if sample_weight is not None else None
    return d, w_sorted


def train(
    df: pd.DataFrame,
    target: str = "over25",
    train_ratio: float = 0.8,
    sample_weight=None,
    feature_cols: "list[str] | None" = None,
    recipe: str = "legacy_v9",
    **repaired_options,
) -> dict:
    """
    Train on the chronological training split, calibrate on the test split.
    sample_weight: array-like of per-row weights (time-decay etc.), same length as df.
    Returns dict: {model_name: {"model", "metrics", "feature_cols"}}

    recipe="legacy_v9" (DEFAULT for direct callers: src/backtest, research scripts) keeps two
    known defects: batch-median imputation (bug 1) and the meta-logistic fitted on the test block
    (bug 6). Bug 8 (weights indexed by a reset index) is fixed here too since 2026-11-02.
    pipeline.mode_train NO LONGER uses it: every scheduled retrain goes through
    `retrain_recipe(label)` -> a repaired recipe (persisted train-fitted imputer, meta fitted on its
    own block, holdout untouched). See RETRAIN_RECIPE.
    recipe="v9_repaired" -> train_repaired() (the serving Standard model's trainer).
    recipe in R2_RECIPES ("std_/nf_/side_/ht_v9_repaired") -> train_repaired() with the R2
    options (holdout_start, imputer_options, provenance).
    """
    if recipe in REPAIRED_RECIPES:
        return train_repaired(df, target=target, sample_weight=sample_weight,
                              feature_cols=feature_cols, recipe=recipe, **repaired_options)
    if recipe != "legacy_v9":
        raise ValueError(f"unknown model recipe {recipe!r}")
    if repaired_options:
        raise TypeError(f"legacy_v9 takes no repaired options: {sorted(repaired_options)}")
    import numpy as np
    # Weights TRAVEL WITH THEIR ROWS (bug 8 fixed for legacy too, 2026-11-02). They used to be
    # read positionally (`w[df.index]`) AFTER the sort + reset_index, so a frame not already in
    # date order gave every row another row's recency weight.
    df, w_all = _legacy_sorted_with_weights(df, target, sample_weight)
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

    w_train = w_all[:split] if w_all is not None else None

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


#: R2 (2026-11-01): ONE parameterised repaired architecture for every non-Standard team model.
#: The recipe NAME selects only the default metadata label; the architecture is identical.
#: "v9_repaired" keeps its exact 2026-10-28 numerics (the serving Standard model's trainer).
R2_RECIPES = ("nf_v9_repaired", "side_v9_repaired", "ht_v9_repaired", "std_v9_repaired")
REPAIRED_RECIPES = ("v9_repaired",) + R2_RECIPES
R2_RECIPE_VERSION = "r2-repaired-1.0"
#: R2 within-pre-holdout fractions: the same 0.64 / 0.08 / 0.08 shape as FOUR_BLOCKS once the
#: holdout (newest 20%) is set aside, i.e. 80% / 10% / 10% of the pre-holdout rows.
R2_PRE_HOLDOUT_BLOCKS = {"base": (0.00, 0.80), "calibration": (0.80, 0.90), "meta": (0.90, 1.00)}
WEIGHT_COL = "_sample_weight"


def _rows_with_weights(df: pd.DataFrame, target: str, sample_weight) -> pd.DataFrame:
    """Attach the weight TO THE ROW before any filter or sort (bug 8 / finding 4).

    A pd.Series is aligned by index label; an array must have one entry per row of `df` as
    passed. A length mismatch is an error — never a silent fallback to ones. After the sort the
    weight of every row is re-derived from its ORIGINAL position (`_row_pos`) and compared with
    the travelling column: any disagreement raises instead of training on misaligned weights.
    """
    d = df.copy()
    if sample_weight is None:
        w = np.ones(len(d))
    elif isinstance(sample_weight, pd.Series):
        ws = sample_weight.reindex(d.index)
        if ws.isna().any():
            raise ValueError("sample_weight Series does not cover every row of df")
        w = ws.to_numpy(dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if len(w) != len(d):
            raise ValueError(f"sample_weight has {len(w)} entries for {len(d)} rows")
    d[WEIGHT_COL] = w
    d["_row_pos"] = np.arange(len(d))
    d = d.dropna(subset=[target])
    tie = [c for c in ("league", "home_team", "away_team") if c in d.columns]
    d = d.sort_values(["date"] + tie, kind="mergesort").reset_index(drop=True)
    if not np.array_equal(w[d["_row_pos"].to_numpy()], d[WEIGHT_COL].to_numpy()):
        raise AssertionError("sample weights no longer belong to their rows after the sort")
    return d


def _whole_date_cuts(d: pd.DataFrame, fracs: dict) -> dict:
    """Cut `d` (date-sorted) at the given fractions; a date straddling a cut goes to the LATER
    block, so no block shares a matchday with the block before it."""
    n = len(d)
    dates = pd.to_datetime(d["date"]).to_numpy()
    bounds, start = {}, 0
    for name, (_lo, hi) in fracs.items():
        end = int(round(n * hi)) if hi < 1 else n
        if end < n and end > start:
            first_of_day = int(np.searchsorted(dates, dates[end], side="left"))
            end = max(start, first_of_day)
        bounds[name] = (start, end)
        start = end
    return {name: d.iloc[a:b].copy() for name, (a, b) in bounds.items()}


def four_block_split(df: pd.DataFrame, target: str, sample_weight=None,
                     holdout_start=None) -> dict:
    """BASE TRAIN -> CALIBRATION -> META TRAIN -> UNTOUCHED HOLDOUT, strictly chronological.

    holdout_start=None (v9_repaired, unchanged): cut at FOUR_BLOCKS fractions of the rows.
    holdout_start=<date> (R2): the holdout is every row dated ON/AFTER it — a boundary fixed
    outside the trainer, so several challengers trained on DIFFERENT row sets (source regimes,
    tracks) are scored on the IDENTICAL untouched rows; the earlier rows split 80/10/10.
    Both cut on whole dates.
    """
    d = _rows_with_weights(df, target, sample_weight)
    if holdout_start is None:
        return _whole_date_cuts(d, FOUR_BLOCKS)
    hs = pd.Timestamp(holdout_start)
    dd = pd.to_datetime(d["date"])
    pre, hold = d[dd < hs].reset_index(drop=True), d[dd >= hs].reset_index(drop=True)
    out = _whole_date_cuts(pre, R2_PRE_HOLDOUT_BLOCKS)
    out["holdout"] = hold.copy()
    return out


def assert_blocks_disjoint(blocks: dict) -> None:
    """No two consecutive blocks share a calendar date (same-day league aggregates cannot leak
    from a later block into an earlier one), and each block is strictly after the previous."""
    order = [k for k in ("base", "calibration", "meta", "holdout") if k in blocks]
    for a, b in zip(order, order[1:]):
        if blocks[a].empty or blocks[b].empty:
            continue
        if pd.to_datetime(blocks[a]["date"]).max() >= pd.to_datetime(blocks[b]["date"]).min():
            raise AssertionError(f"block {a} overlaps block {b} in time")


def _block_meta(blocks: dict) -> dict:
    def _rng(b):
        return [str(pd.Timestamp(b["date"].min()).date()), str(pd.Timestamp(b["date"].max()).date())]
    return {k: {"rows": int(len(b)), "dates": _rng(b)} for k, b in blocks.items()}


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return {"accuracy": round(accuracy_score(y, p >= 0.5), 4),
            "auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
            "log_loss": round(log_loss(y, p, labels=[0, 1]), 5),
            "brier": round(float(np.mean((p - y) ** 2)), 5),
            "ece": round(expected_calibration_error(y, p), 5)}


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Equal-width 10-bin ECE (|mean p - mean y| weighted by bin share)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return float("nan")
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _base_models():
    cfg = [
        ("logistic", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ("gradient_boost", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42)),
    ]
    try:
        from lightgbm import LGBMClassifier
        cfg.append(("lightgbm", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)))
    except ImportError:
        pass
    return cfg


def train_repaired(df: pd.DataFrame, target: str = "over25", sample_weight=None,
                   feature_cols: "list[str] | None" = None, recipe: str = "v9_repaired",
                   holdout_start=None, imputer_options: "dict | None" = None,
                   provenance: "dict | None" = None,
                   calibration_input: str = "probability") -> TrainResults:
    """Same model family as legacy (LR + GB + LightGBM, Platt, meta-logistic), with every P0
    training defect removed:

      bug 1  imputer fitted on BASE rows only, persisted, applied to every block and to serving;
      bug 6  calibrators fit on CALIBRATION, meta-logistic on META, all metrics on the untouched
             HOLDOUT — no reported number comes from rows any stage was fitted on;
      bug 8  weights travel with their rows (`_rows_with_weights`, verified after the sort).

    recipe "v9_repaired" with no R2 keyword = the 2026-10-28 trainer, numerically unchanged.
    R2 recipes (nf_/side_/ht_v9_repaired) additionally: a 1.1 imputer (no-training-values
    features DROPPED and recorded, never silently constant), an externally fixed holdout
    boundary, asserted block disjointness, holdout-label independence evidence, and the full
    payload metadata (versions, cutoff, frame hash, feature list, blocks, weights, provenance).

    calibration_input "probability" (default) is the recipe as served since 2026-10-28: the Platt
    calibrators and the meta-logistic read PROBABILITIES under an L2 penalty (C=1). On that scale the
    identity map needs a coefficient near 4, so the penalty shrinks every probability toward the
    block's base rate - the Phase 2 champion's BTTS spread was 0.02 and its league means all fell
    in 0.52-0.53 (2026-11-11). "logit" fits both on log-odds instead (identity = coefficient 1);
    the payload records it (`calibration_input`) and predict_proba applies the same transform.
    """
    from src.imputer import fit_imputer, IMPUTER_VERSION
    if recipe not in REPAIRED_RECIPES:
        raise ValueError(f"unknown repaired recipe {recipe!r}")
    if calibration_input not in CALIBRATION_INPUTS:
        raise ValueError(f"calibration_input must be one of {CALIBRATION_INPUTS}, not {calibration_input!r}")
    _cin = _logit if calibration_input == "logit" else (lambda p: np.asarray(p, dtype=float))
    r2 = recipe in R2_RECIPES or holdout_start is not None or imputer_options is not None
    feat_cols = list(feature_cols or FEATURE_COLS)
    blocks = four_block_split(df, target, sample_weight, holdout_start=holdout_start)
    base, cal, meta_b, hold = (blocks["base"], blocks["calibration"], blocks["meta"],
                               blocks["holdout"])
    if min(len(base), len(cal), len(meta_b), len(hold)) < 20:
        raise ValueError(f"too few rows for four-block training: "
                         f"{[len(b) for b in blocks.values()]}")
    dropped = []
    if r2:
        assert_blocks_disjoint(blocks)
        opts = {"all_missing": "drop", **(imputer_options or {})}
        imputer = fit_imputer(base, feat_cols, **opts)
        dropped = list(imputer.all_missing_features)
        if dropped:
            feat_cols = [c for c in feat_cols if c not in dropped]
            log.info(f"  [{recipe}] dropped {len(dropped)} feature(s) with no training values: "
                     f"{dropped}")
            imputer = fit_imputer(base, feat_cols, **opts)
    else:
        imputer = fit_imputer(base, feat_cols)
    X_base, _ = _prep(base, feat_cols, imputer=imputer)
    X_cal, _ = _prep(cal, feat_cols, imputer=imputer)
    X_meta, _ = _prep(meta_b, feat_cols, imputer=imputer)
    X_hold, _ = _prep(hold, feat_cols, imputer=imputer)
    y_base, y_cal = base[target].to_numpy(), cal[target].to_numpy()
    y_meta, y_hold = meta_b[target].to_numpy(), hold[target].to_numpy()
    w_base, w_cal = base[WEIGHT_COL].to_numpy(), cal[WEIGHT_COL].to_numpy()

    if r2:
        _m = _metrics
    else:
        def _m(y, p):                                   # exact 2026-10-28 metric dict
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return {"accuracy": round(accuracy_score(y, p >= 0.5), 4),
                    "auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
                    "log_loss": round(log_loss(y, p, labels=[0, 1]), 5),
                    "brier": round(float(np.mean((p - y) ** 2)), 5)}

    results = TrainResults()
    for name, base_clf in _base_models():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
        pipe.fit(X_base, y_base, clf__sample_weight=w_base)
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(_cin(pipe.predict_proba(X_cal)[:, 1]).reshape(-1, 1), y_cal, sample_weight=w_cal)
        calibrated = _CalibratedModel(pipe, platt, calibration_input=calibration_input)
        metrics = _m(y_hold, calibrated.predict_proba(X_hold)[:, 1])
        metrics.update(n_train=int(len(base)), n_cal=int(len(cal)), n_meta=int(len(meta_b)),
                       n_test=int(len(hold)), evaluated_on="holdout")
        results[name] = {"model": calibrated, "metrics": metrics, "feature_cols": feat_cols}

    served_hold = None
    if len(results) >= 2:
        meta_X = _cin(np.column_stack([results[k]["model"].predict_proba(X_meta)[:, 1] for k in results]))
        meta_clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        meta_clf.fit(meta_X, y_meta)
        hold_X = _cin(np.column_stack([results[k]["model"].predict_proba(X_hold)[:, 1] for k in results]))
        served_hold = meta_clf.predict_proba(hold_X)[:, 1]
        mm = _m(y_hold, served_hold)
        mm.update(n_meta=int(len(meta_b)), n_test=int(len(hold)), evaluated_on="holdout")
        results["__meta__"] = {"model": meta_clf, "metrics": mm, "feature_cols": list(results)}
    else:
        served_hold = results[next(iter(results))]["model"].predict_proba(X_hold)[:, 1]

    results.meta = {
        "recipe": "v9_repaired", "model_recipe_version": MODEL_RECIPE_VERSION,
        "imputer": imputer.to_dict(), "imputer_version": IMPUTER_VERSION,
        "blocks": _block_meta(blocks),
        "served_holdout_metrics": _m(y_hold, served_hold),
        "target": target,
    }
    if calibration_input != "probability":
        # only a non-default scale is written, so every payload trained before stays byte-identical
        results.meta["calibration_input"] = calibration_input
    if r2:
        results.meta.update(_r2_meta(recipe, blocks, feat_cols, dropped, imputer, target,
                                     holdout_start, provenance))
        results.holdout_frame = hold
        results.holdout_p = served_hold
    log.info(f"  [{recipe}] served holdout {results.meta['served_holdout_metrics']}  "
             f"blocks={ {k: v['rows'] for k, v in results.meta['blocks'].items()} }")
    return results


def _r2_meta(recipe, blocks, feat_cols, dropped, imputer, target, holdout_start, provenance):
    """Full payload metadata (finding 1). Nothing here is used to score; it is the audit trail."""
    import hashlib
    import os
    import subprocess
    pre = pd.concat([blocks["base"], blocks["calibration"], blocks["meta"]])
    keys = [c for c in ("date", "league", "home_team", "away_team", target) if c in pre.columns]
    fh = hashlib.sha256(pd.util.hash_pandas_object(
        pre[keys].astype(str), index=False).to_numpy().tobytes()).hexdigest()[:16]
    sha = os.getenv("GITHUB_SHA", "").strip()
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parents[1], timeout=20).stdout.strip()
        except Exception:                                              # noqa: BLE001
            sha = "unknown"
    prov = dict(provenance or {})
    d_pre = pd.to_datetime(pre["date"])
    return {
        "recipe": recipe, "model_recipe_version": R2_RECIPE_VERSION,
        "imputer_version": imputer.version,
        "feature_cols": list(feat_cols), "n_features": len(feat_cols),
        "dropped_all_missing_features": list(dropped),
        "block_definition": {"holdout_start": (str(pd.Timestamp(holdout_start).date())
                                               if holdout_start is not None else None),
                             "pre_holdout_fractions": R2_PRE_HOLDOUT_BLOCKS
                             if holdout_start is not None else FOUR_BLOCKS,
                             "whole_date_cuts": True, "disjoint_asserted": True},
        "holdout_independence": {"imputer_fit": "base", "base_models_fit": "base",
                                 "calibrators_fit": "calibration", "meta_fit": "meta",
                                 "holdout_labels_used_for_fitting": False},
        "weights": {"column": WEIGHT_COL, "travel_with_rows": True, "alignment_verified": True,
                    "base_ess": round(float(blocks["base"][WEIGHT_COL].sum() ** 2
                                            / max(1e-12, (blocks["base"][WEIGHT_COL] ** 2).sum())), 1),
                    "recency": prov.pop("recency", None)},
        "training_start": str(d_pre.min().date()), "training_end": str(d_pre.max().date()),
        "training_cutoff": prov.pop("training_cutoff", str(d_pre.max().date())),
        "training_rows": int(len(pre)), "frame_hash": fh, "code_sha": sha or "unknown",
        "feature_recipe_version": prov.pop("feature_recipe_version", None),
        "identity_version": prov.pop("identity_version", None),
        "provenance": prov,
    }


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


# ── Scheduled retrain (pipeline.mode_train) — ONE trainer, ONE router ─────────────────────────
#
# 2026-11-02 (estate scan, training findings a-d). pipeline._train_one used the legacy recipe for
# every model: batch-median imputation at serving (a fixture's probability depended on the other
# fixtures scored with it), the meta-logistic fitted on the very rows the promotion gate then
# scored, and side models (BTTS / O1.5 / O3.5) trained on Standard + New Format rows POOLED and
# served to both. A scheduled retrain now trains every label on a repaired recipe, and the side
# labels per league track. Payloads already on disk keep serving exactly as before (legacy
# payloads have no `imputer` and no `tracks`), until a retrain replaces them through the gate.

#: pipeline._train_one label -> repaired recipe. A label missing here is an error, not legacy.
RETRAIN_RECIPE = {
    "standard": "std_v9_repaired", "newformat": "nf_v9_repaired",
    "btts": "side_v9_repaired", "over15": "side_v9_repaired", "over35": "side_v9_repaired",
    "ht_over05": "ht_v9_repaired", "ht_over15": "ht_v9_repaired",
}
#: Labels whose ONE serving file is read for both league tracks: trained per track, routed per row.
TRACK_SPLIT_LABELS = frozenset({"btts", "over15", "over35"})
#: The track the payload's top-level models belong to (every flat reader keeps working on it).
PRIMARY_TRACK = "standard"
#: Tracks carried as nested payloads under payload["tracks"].
ROUTED_TRACKS = ("new_format",)


def retrain_recipe(label: str) -> str:
    if label not in RETRAIN_RECIPE:
        raise ValueError(f"no retrain recipe for model label {label!r}")
    return RETRAIN_RECIPE[label]


def track_of(leagues) -> pd.Series:
    """'standard' | 'new_format' | None per row, from the league registry
    (config.model_type_for_league). The ONE routing rule for track-split training and serving."""
    s = pd.Series(leagues).reset_index(drop=True)
    lut = {lg: config.model_type_for_league(lg) for lg in pd.unique(s.dropna())}
    return s.map(lut).where(s.notna(), None)


def train_by_track(df: pd.DataFrame, target: str, sample_weight=None,
                   feature_cols: "list[str] | None" = None, recipe: str = "side_v9_repaired",
                   holdout_start=None, provenance: "dict | None" = None) -> TrainResults:
    """One repaired model per league track, on that track's rows only.

    The Standard model is the payload's top level (so feature importances, the gate manifest and
    every flat reader see a normal payload); each routed track is a complete nested payload under
    meta["tracks"][track]. A routed track with too few rows is recorded as None — its fixtures then
    get NO probability at serving (missing stays missing), never the Standard model's. Rows whose
    league is in no track are excluded and counted. The Standard track failing raises.
    """
    if recipe not in REPAIRED_RECIPES:
        raise ValueError(f"track-split training needs a repaired recipe, got {recipe!r}")
    d = df.reset_index(drop=True)
    w = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            ws = sample_weight.reindex(df.index)
            if ws.isna().any():
                raise ValueError("sample_weight Series does not cover every row of df")
            w = ws.to_numpy(dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
        if len(w) != len(d):
            raise ValueError(f"sample_weight has {len(w)} entries for {len(d)} rows")
    tr = track_of(d["league"]).to_numpy()

    def _one(name):
        m = tr == name
        sub = d[m]
        sw = pd.Series(w[m], index=sub.index) if w is not None else None
        return train(sub, target=target, sample_weight=sw, feature_cols=feature_cols,
                     recipe=recipe, holdout_start=holdout_start,
                     provenance={**(provenance or {}), "track": name})

    counts = {name: int((tr == name).sum()) for name in (PRIMARY_TRACK,) + ROUTED_TRACKS}
    n_none = int(sum(t is None for t in tr))
    if n_none:
        log.warning(f"  [{target}] {n_none} row(s) in no league track excluded from training")
    primary = _one(PRIMARY_TRACK)
    tracks, status = {}, {PRIMARY_TRACK: "trained"}
    for name in ROUTED_TRACKS:
        try:
            tracks[name] = build_payload(_one(name), target)
            status[name] = "trained"
        except ValueError as e:
            tracks[name] = None
            status[name] = f"not trained: {e}"
            log.warning(f"  [{target}] {name} track not trained ({e}) — its fixtures will get no "
                        f"{target} probability from this payload")
    primary.meta.update({
        "tracks": tracks,
        "track_split": {"primary": PRIMARY_TRACK, "routed": list(ROUTED_TRACKS),
                        "router": "src.model.track_of (config.model_type_for_league)",
                        "rows": counts, "rows_in_no_track": n_none, "status": status},
    })
    return primary


def train_for_retrain(df: pd.DataFrame, label: str, target: str = "over25", sample_weight=None,
                      feature_cols: "list[str] | None" = None, holdout_start=None,
                      provenance: "dict | None" = None) -> TrainResults:
    """THE scheduled-retrain trainer (pipeline._train_one). Repaired recipe for every label;
    side labels trained per track. `holdout_start` should be the first date of the promotion
    gate's holdout, so no row the gate scores was used to fit any stage of the candidate."""
    recipe = retrain_recipe(label)
    if label in TRACK_SPLIT_LABELS:
        return train_by_track(df, target, sample_weight=sample_weight, feature_cols=feature_cols,
                              recipe=recipe, holdout_start=holdout_start, provenance=provenance)
    return train(df, target=target, sample_weight=sample_weight, feature_cols=feature_cols,
                 recipe=recipe, holdout_start=holdout_start, provenance=provenance)


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
        config.HT_MODEL_FILE_15, config.MODEL_FILE_STANDARD_REPAIRED)}
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
    if payload.get("tracks") is not None:
        return _predict_by_track(df, payload)
    return _predict_flat(df, payload)


def _predict_by_track(df: pd.DataFrame, payload: dict) -> pd.Series:
    """A track-split payload (`train_by_track`): each row is scored ONLY by the model trained on
    its own track (routing = `track_of`, the function the trainer split on). A row whose track has
    no trained model, or whose league is in no track, gets NO probability (NaN) — never another
    track's model."""
    out = pd.Series(np.nan, index=df.index, name="p_over25")
    if df.empty:
        return out
    tr = track_of(df["league"]).to_numpy() if "league" in df.columns else np.full(len(df), None)
    for name in (PRIMARY_TRACK,) + ROUTED_TRACKS:
        m = tr == name
        if not m.any():
            continue
        sub = payload if name == PRIMARY_TRACK else (payload.get("tracks") or {}).get(name)
        if sub is None:
            log.warning(f"  {int(m.sum())} {name} row(s): no {name} model in this payload — "
                        f"left without a probability")
            continue
        out[m] = _predict_flat(df[m], sub).to_numpy()
    return out


def _predict_flat(df: pd.DataFrame, payload: dict) -> pd.Series:
    feat_cols  = payload["feature_cols"]
    # A repaired payload WITHOUT its persisted imputer would silently fall back to the
    # batch-median path (finding 2): refuse it. Legacy payloads carry no `recipe` key.
    if payload.get("recipe") in REPAIRED_RECIPES and not payload.get("imputer"):
        raise ValueError(f"{payload.get('recipe')} payload has no persisted imputer — refusing "
                         f"to impute from the scoring batch")
    # payload["imputer"] exists only on v9_repaired payloads; serving payloads have none, so
    # this is the legacy call exactly.
    X, _       = _prep(df, feat_cols, imputer=payload.get("imputer"))
    base_models = {k: v for k, v in payload["models"].items() if k != "__meta__"}
    meta_model  = payload["models"].get("__meta__")

    probas = [m.predict_proba(X)[:, 1] for m in base_models.values()]

    if meta_model is not None and len(probas) >= 2:
        try:
            meta_X   = np.column_stack(probas)
            if payload.get("calibration_input", "probability") == "logit":
                meta_X = _logit(meta_X)              # the scale the meta-logistic was fitted on
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
