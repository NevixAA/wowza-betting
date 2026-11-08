"""
props_harness.py - the Phase 2 PROPS CLEAN RETRAIN HARNESS (key propsharness, 2026-11-06).

BUILD ONLY. Nothing here trains on real data until the owner approves: `--run` refuses unless every
owner decision in config/experiments/props_clean_retrain/plan.json is set AND the environment says
PHASE2_RETRAIN_ENABLED=true (set only by .github/workflows/props_clean_retrain.yml, whose job is
gated on the repository variable of the same name). `--dry-run` reads the committed data and COUNTS
rows; it fits nothing and scores nothing. The tests exercise every component on synthetic data.

DATA (owner 2026-11-04 / 2026-11-06).
  * Rows: player_history.parquet plus the collected UEFA cup rows (cup_history; a cup fixture decided
    after extra time is context only, never a training row), with the re-fetched counts applied
    (history_refetch.with_refetched). A row is a TRAINING row only when its counts are verified:
    count_source 'refetched' or 'provider_rule'. 'zeros_unverified' rows are context only.
  * Feature rule (owner decision, plan.feature_rule): a verified row's rolling features are built
    from the player's EARLIER rows, which may still be unverified. 'label_only' accepts that;
    'window_verified' also needs the player's previous config.ROLLING_N rows verified;
    'all_prior_verified' needs every earlier row of the player verified.
  * Features and labels are rebuilt from the raw counts (feature_engineering.build_features), never
    read from the stored derived columns. The 7 profile features the frozen payloads learned from
    placeholders (age 25 / 180 cm / 75 kg on every row) are RE-ADMITTED: build_features attaches the
    collected profiles (output/collect/player_profiles.parquet) and leaves NaN where none is known,
    so training and serving read the same real values (closes the P9 train/serve gap).
  * The owner's null rule is applied at parse (player_stat_nulls); a missing count is a missing label
    and the row is dropped for that market, never read as 0.

TARGETS. The props markets as the tips use them (config.MARKET_TARGETS, labels from
feature_engineering._label): sot (SOT 1+), sot2 (SOT 2+), goals (anytime scorer), assists, cards.

CANDIDATES (same rows, same windows).
  recipe_retrained  the current recipe (model.recipe_estimators: logistic + HistGradientBoosting,
                    Platt-calibrated on the window's last matchdays, mean of the two) with an
                    imputer fitted on the FIT block of the training window only and saved with the
                    model. A feature with no value in the fit block is dropped, never filled.
  hgb_nan_native    the recipe's HistGradientBoosting alone on the raw features: nothing is filled
                    at train or serve; Platt-calibrated on the same calibration block.
  player_base_rate  the player's own rate in the training window, shrunk to the window's market rate
                    by a beta prior whose strength is FITTED (method of moments) on that window; a
                    player with no training row gets the window's market rate.

PROTOCOL. Walk-forward by matchday (one calendar match date, every league) before the holdout:
plan.walk_forward.n_folds folds of step_matchdays each, every fold trained on all earlier
matchdays. Then ONE fit on everything before holdout_start, scored once on [holdout_start,
holdout_end]. Per market and candidate: log loss, Brier, ECE (10 bins), calibration intercept and
slope, a reliability table, and on the holdout the paired difference to player_base_rate with a
week-block bootstrap interval. CLV (props_harness_clv) uses REAL two-sided prices
(fair_basis 'two_sided_devig'); a row priced on one side only is reported apart, flagged with the
assumed margin, and never pooled with the two-sided rows.

HAND ADJUSTMENTS: none. Owner 2026-11-05: tiers use the model's own probability, so the harness never
applies predict.hand_adjusted_prob (or any multiplier) to a candidate's probability.

Nothing here writes a serving file (models/model_player_*.pkl) or touches the props promotion gate.

Usage:
    python -m player_model.props_harness --dry-run [--holdout-start D --holdout-end D] [--out F]
    python -m player_model.props_harness --run          (GitHub Actions, owner-gated)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .player_stat_nulls import SOURCE_COL, UNVERIFIED, VERIFIED_SOURCES

HARNESS_VERSION = "PROPS-HARNESS-1.0"
PLAN_VERSION = "PROPS-CLEAN-RETRAIN-1.0"
PLAN_PATH = config.BASE_DIR / "config" / "experiments" / "props_clean_retrain" / "plan.json"
OUT_DIR = config.OUTPUT_DIR / "props_harness"
MODEL_DIR = config.MODELS_DIR / "challengers" / "props_clean"
HISTORY_PATH = config.BASE_DIR / "player_history.parquet"
ENABLE_ENV = "PHASE2_RETRAIN_ENABLED"

#: owner 2026-11-05: the model's own probability decides; recorded in every output
HAND_ADJUSTMENT = "none"
MARKETS_IN_SCOPE = ("sot", "sot2", "goals", "assists", "cards")
#: market -> (raw count column, threshold): the same reading build_features uses for target_<market>
LABEL_RULE = {"goals": ("goals", 1), "goals2": ("goals", 2), "assists": ("assists", 1),
              "sot": ("shots_on_target", 1), "sot2": ("shots_on_target", 2),
              "sot3": ("shots_on_target", 3), "sot4": ("shots_on_target", 4),
              "cards": ("yellow_cards", 1)}
FEATURE_RULES = ("label_only", "window_verified", "all_prior_verified")
#: feature rule -> the population column that implements it (population())
RULE_COLUMN = {"label_only": "label_verified", "window_verified": "window_verified",
               "all_prior_verified": "all_prior_verified"}
CANDIDATES = ("recipe_retrained", "hgb_nan_native", "player_base_rate")
#: min_verified_share = the TRAINING (pre-holdout) floor, min_verified_share_holdout = the HOLDOUT floor
#: (owner 2026-11-08: 95% / 85%); both must pass under the owner's feature rule or nothing is fitted
OWNER_KEYS = ("holdout_start", "holdout_end", "min_verified_share", "min_verified_share_holdout",
              "markets", "feature_rule")
#: build_features drops a row under this many minutes (bench / DNP); the population mirrors it
MIN_ROW_MINUTES = 10
#: the recipe's fit : calibration ratio (model.split_bounds: calibration = the last 15% of training)
FIT_SHARE = 0.85
#: model.train refuses a market with fewer rows; the harness uses the same floor per window
MIN_TRAIN_ROWS = 200
EPS = 1e-6
SPLITS = ("pre_holdout", "holdout", "after_holdout")


# ── plan ─────────────────────────────────────────────────────────────────────────────────────────

def load_plan(path: Path | None = None) -> dict:
    p = PLAN_PATH if path is None else Path(path)
    with open(p, encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError(f"props retrain plan version {plan.get('plan_version')!r} != {PLAN_VERSION!r}")
    if plan.get("hand_adjustment") != HAND_ADJUSTMENT:
        raise ValueError("the props retrain plan must say hand_adjustment 'none' (owner 2026-11-05)")
    return plan


def missing_owner_decisions(plan: dict) -> list[str]:
    d = plan.get("owner_decisions") or {}
    return [k for k in OWNER_KEYS if d.get(k) is None or d.get(k) == []]


def validate_decisions(d: dict) -> None:
    """Raise on a decision that cannot be used (wrong market, rule, share or date order)."""
    hs, he = pd.Timestamp(d["holdout_start"]), pd.Timestamp(d["holdout_end"])
    if he < hs:
        raise ValueError(f"holdout_end {he.date()} is before holdout_start {hs.date()}")
    bad = [m for m in d["markets"] if m not in MARKETS_IN_SCOPE]
    if bad:
        raise ValueError(f"markets outside the harness scope {MARKETS_IN_SCOPE}: {bad}")
    if d["feature_rule"] not in FEATURE_RULES:
        raise ValueError(f"feature_rule {d['feature_rule']!r} not one of {FEATURE_RULES}")
    for k in ("min_verified_share", "min_verified_share_holdout"):
        share = float(d[k])
        if not 0.0 < share <= 1.0:
            raise ValueError(f"{k} {share} must be in (0, 1]")


def owner_decisions(plan: dict) -> dict:
    """The owner's decisions, validated. Raises while any is unset: --run never trains on a proposal."""
    miss = missing_owner_decisions(plan)
    if miss:
        raise PermissionError(f"props retrain refused: owner decision(s) not set in {PLAN_PATH.name}: {miss}")
    d = dict(plan["owner_decisions"])
    validate_decisions(d)
    return d


def run_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() == "true"


# ── data ─────────────────────────────────────────────────────────────────────────────────────────

def source_frame(history: pd.DataFrame | None = None, cups: pd.DataFrame | None = None,
                 refetch: pd.DataFrame | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """player_history + the cup rows it does not hold, `harness_source` 'history' | 'cup_rows', the
    re-fetched counts applied and `count_source` on every row (history_refetch.with_refetched)."""
    from .cup_history import cup_rows, new_cup_rows
    from .history_refetch import with_refetched
    if history is None:
        history = _read_history(columns)
    h = history.assign(harness_source="history")
    add = new_cup_rows(h, cup_rows() if cups is None else cups)
    if columns is not None and not add.empty:
        add = add[[c for c in add.columns if c in set(columns) | {"fixture_status", SOURCE_COL}]]
    parts = [h] if add.empty else [h, add.assign(harness_source="cup_rows")]
    raw = pd.concat(parts, ignore_index=True, sort=False)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = with_refetched(raw, refetch)
    return raw.sort_values("date", kind="stable").reset_index(drop=True)


def _read_history(columns: list[str] | None) -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=columns or [])
    if columns is None:
        return pd.read_parquet(HISTORY_PATH)
    import pyarrow.dataset as ds
    have = set(ds.dataset(str(HISTORY_PATH), format="parquet").schema.names)
    return pd.read_parquet(HISTORY_PATH, columns=[c for c in columns if c in have])


def population(raw: pd.DataFrame, verified_extra: pd.Series | None = None) -> pd.DataFrame:
    """The rows build_features keeps as training rows (a real player id, >= MIN_ROW_MINUTES minutes,
    at least one earlier such row of the player), cup rows decided after extra time removed, with the
    verification columns:
        label_verified       this row's counts are verified (count_source refetched / provider_rule)
        window_verified      ... and the player's previous config.ROLLING_N rows are verified
        all_prior_verified   ... and every earlier row of the player is verified
    `verified_extra` (bool, aligned to raw) marks rows to treat as verified as well - the dry-run's
    'after the re-fetch' view. Earlier rows are counted in the same minutes-filtered frame
    build_features rolls over."""
    from .cup_history import EXTRA_TIME
    pid = pd.to_numeric(raw["player_id"], errors="coerce")
    mins = pd.to_numeric(raw["minutes"], errors="coerce")
    keep = (pid > 0) & (mins >= MIN_ROW_MINUTES)
    f = raw[keep.to_numpy()].copy()
    f["player_id"] = pid[keep.to_numpy()].astype("int64").to_numpy()
    ver = f[SOURCE_COL].isin(VERIFIED_SOURCES) if SOURCE_COL in f.columns \
        else pd.Series(False, index=f.index)
    if verified_extra is not None:
        ver = ver | verified_extra.reindex(f.index).fillna(False).astype(bool)
    f["label_verified"] = ver.astype(bool).to_numpy()
    f = f.sort_values(["player_id", "date"], kind="stable")
    g = f.groupby("player_id", sort=False)
    f["n_prev"] = g.cumcount()
    v = f["label_verified"].astype(int)
    prev_v = v.groupby(f["player_id"]).shift(1)
    n = config.ROLLING_N
    roll_sum = prev_v.groupby(f["player_id"]).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
    roll_cnt = prev_v.groupby(f["player_id"]).rolling(n, min_periods=1).count().reset_index(level=0, drop=True)
    win_ok = (roll_sum.reindex(f.index) == roll_cnt.reindex(f.index)) & roll_cnt.reindex(f.index).gt(0)
    unver_before = (1 - v).groupby(f["player_id"]).cumsum() - (1 - v)
    f["window_verified"] = (f["label_verified"] & win_ok).to_numpy()
    f["all_prior_verified"] = (f["label_verified"] & unver_before.eq(0) & f["n_prev"].ge(1)).to_numpy()
    f = f[f["n_prev"] >= 1]
    if "fixture_status" in f.columns:
        f = f[~f["fixture_status"].isin(EXTRA_TIME)]
    return f.sort_values("date", kind="stable")


def eligible(pop: pd.DataFrame, rule: str) -> pd.Series:
    if rule not in FEATURE_RULES:
        raise ValueError(f"feature rule {rule!r} not one of {FEATURE_RULES}")
    return pop[RULE_COLUMN[rule]].astype(bool)


def label(df: pd.DataFrame, market: str) -> pd.Series:
    """1 / 0 / NaN (no recorded count) for one market, from the raw count column."""
    from .feature_engineering import _label
    col, k = LABEL_RULE[market]
    return _label(df, col, k)


def split_of(dates: pd.Series, holdout_start, holdout_end) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    hs, he = pd.Timestamp(holdout_start), pd.Timestamp(holdout_end)
    out = pd.Series("pre_holdout", index=d.index, dtype=object)
    out[d >= hs] = "holdout"
    out[d > he] = "after_holdout"
    return out


def matchdays(dates: pd.Series) -> list[pd.Timestamp]:
    """Distinct calendar match dates, ascending."""
    d = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    return sorted(d.unique())


def walk_forward_folds(days: list, holdout_start, step: int, n_folds: int) -> list[dict]:
    """The last n_folds blocks of `step` matchdays before holdout_start: fold k trains on every
    matchday before its first test day. Fixed on the calendar of ALL population rows, so a fold's
    dates do not move as rows become verified."""
    if step < 1 or n_folds < 1:
        raise ValueError("walk-forward step and n_folds must be >= 1")
    hs = pd.Timestamp(holdout_start)
    pre = [pd.Timestamp(x) for x in days if pd.Timestamp(x) < hs]
    folds = []
    end = len(pre)
    for k in range(n_folds):
        start = end - step
        if start < 1:                                  # no training day before this block
            break
        folds.append({"test_start": pre[start], "test_end": pre[end - 1], "n_matchdays": end - start})
        end = start
    folds.reverse()
    for i, f in enumerate(folds):
        f["fold"] = i
    return folds


def calibration_cut(dates: pd.Series) -> pd.Timestamp:
    """The first matchday of the calibration block: the first date whose rows take the window past
    FIT_SHARE (the fit block holds at most FIT_SHARE of the rows, and at least one matchday). Cut on
    a date, so one matchday never sits in both blocks."""
    d = pd.to_datetime(dates, errors="coerce").dt.normalize()
    counts = d.value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("training window has fewer than two matchdays: no calibration block")
    cum = counts.cumsum() / counts.sum()
    pos = max(1, counts.index.get_loc(cum[cum > FIT_SHARE].index[0]))
    return counts.index[pos]


def harness_features(frame: pd.DataFrame) -> list[str]:
    """config.PLAYER_FEATURE_COLS + the 7 re-admitted profile features (real profiles or NaN)."""
    cols = list(config.PLAYER_FEATURE_COLS) + sorted(config.PROFILE_FEATURES_WITHHELD)
    return [c for c in dict.fromkeys(cols) if c in frame.columns]


# ── candidates ───────────────────────────────────────────────────────────────────────────────────

class RecipeRetrained:
    """The current props recipe on a train-fitted, persisted imputer (fit block medians)."""
    name = "recipe_retrained"

    def fit(self, X_fit, y_fit, X_cal, y_cal, **_):
        from .model import _PlattCalibratedModel, recipe_estimators
        from sklearn.linear_model import LogisticRegression
        med = X_fit.apply(pd.to_numeric, errors="coerce").median()
        self.dropped_unfittable_ = [c for c in X_fit.columns if pd.isna(med.get(c))]
        self.feature_cols_ = [c for c in X_fit.columns if c not in self.dropped_unfittable_]
        self.fill_ = {c: float(med[c]) for c in self.feature_cols_}
        self.models_ = {}
        Xf, Xc = self._fill(X_fit), self._fill(X_cal)
        for name, est in recipe_estimators().items():
            est.fit(Xf, y_fit)
            platt = LogisticRegression(C=1.0, max_iter=1000)
            platt.fit(est.predict_proba(Xc)[:, 1].reshape(-1, 1), y_cal)
            self.models_[name] = _PlattCalibratedModel(est, platt)
        return self

    def _fill(self, X):
        cols = {c: pd.to_numeric(X[c], errors="coerce").fillna(self.fill_[c]) if c in X.columns
                else pd.Series(self.fill_[c], index=X.index) for c in self.feature_cols_}
        return pd.DataFrame(cols, index=X.index)[self.feature_cols_].astype(float)

    def predict(self, X, **_):
        Xf = self._fill(X)
        return np.mean([m.predict_proba(Xf)[:, 1] for m in self.models_.values()], axis=0)

    def state(self) -> dict:
        return {"imputer": "fit_block_median", "fill_values": dict(self.fill_),
                "feature_cols": list(self.feature_cols_),
                "dropped_unfittable": list(self.dropped_unfittable_)}


class HGBNaNNative:
    """The recipe's HistGradientBoosting alone, NaN-native: no value is filled anywhere."""
    name = "hgb_nan_native"

    def fit(self, X_fit, y_fit, X_cal, y_cal, **_):
        from .model import _PlattCalibratedModel, recipe_estimators
        from sklearn.linear_model import LogisticRegression
        self.feature_cols_ = list(X_fit.columns)
        est = recipe_estimators()["gradient_boost"]
        est.fit(self._x(X_fit), y_fit)
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(est.predict_proba(self._x(X_cal))[:, 1].reshape(-1, 1), y_cal)
        self.model_ = _PlattCalibratedModel(est, platt)
        return self

    def _x(self, X):
        return pd.DataFrame({c: (pd.to_numeric(X[c], errors="coerce") if c in X.columns
                                 else pd.Series(np.nan, index=X.index)) for c in self.feature_cols_},
                            index=X.index).astype(float)

    def predict(self, X, **_):
        return self.model_.predict_proba(self._x(X))[:, 1]

    def state(self) -> dict:
        return {"imputer": "none (NaN-native)", "feature_cols": list(self.feature_cols_)}


class PlayerBaseRate:
    """The player's own training-window rate, shrunk to the window's market rate by a beta prior
    whose strength is fitted by the method of moments on the same window."""
    name = "player_base_rate"

    def fit(self, X_fit, y_fit, X_cal, y_cal, players_fit=None, players_cal=None, **_):
        y = pd.concat([pd.Series(np.asarray(y_fit, dtype=float)), pd.Series(np.asarray(y_cal, dtype=float))],
                      ignore_index=True)
        p = pd.concat([pd.Series(np.asarray(players_fit)), pd.Series(np.asarray(players_cal))],
                      ignore_index=True)
        g = pd.DataFrame({"p": p, "y": y}).groupby("p")["y"].agg(["sum", "count"])
        mu = float(y.mean())
        self.market_rate_ = mu
        rate = g["sum"] / g["count"]
        w = g["count"] / g["count"].sum()
        var_obs = float((w * (rate - mu) ** 2).sum())
        noise = float((w * mu * (1 - mu) / g["count"]).sum())
        tau2 = var_obs - noise
        # no between-player spread beyond binomial noise -> every player gets the market rate
        self.prior_strength_ = (mu * (1 - mu) / tau2 - 1.0) if tau2 > 0 else float("inf")
        m = self.prior_strength_
        if np.isfinite(m) and m > 0:
            self.rates_ = ((g["sum"] + m * mu) / (g["count"] + m)).to_dict()
        else:
            self.rates_ = {k: mu for k in g.index}
        return self

    def predict(self, X, players=None, **_):
        return pd.Series(np.asarray(players)).map(self.rates_).astype(float) \
            .where(lambda s: s.notna(), self.market_rate_).to_numpy()

    def state(self) -> dict:
        return {"imputer": "none", "market_rate": self.market_rate_,
                "prior_strength": self.prior_strength_, "players": len(self.rates_)}


CANDIDATE_CLASSES = {c.name: c for c in (RecipeRetrained, HGBNaNNative, PlayerBaseRate)}


def fit_window(train: pd.DataFrame, features: list[str], y_col: str, names=CANDIDATES) -> dict:
    """Fit every candidate on one training window (fit block | calibration block cut by date).
    Returns {name: fitted candidate} plus '_window' metadata. Raises ValueError on a window that
    cannot be fitted (too few rows, a single class in a block)."""
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"training window has {len(train)} rows (< {MIN_TRAIN_ROWS})")
    cut = calibration_cut(train["date"])
    d = pd.to_datetime(train["date"]).dt.normalize()
    fit, cal = train[d < cut], train[d >= cut]
    y_fit, y_cal = fit[y_col].astype(int), cal[y_col].astype(int)
    if y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError("a single class in the fit or calibration block")
    out = {"_window": {"fit_rows": int(len(fit)), "calibration_rows": int(len(cal)),
                       "fit_dates": [str(d.min().date()), str((cut - pd.Timedelta(days=1)).date())],
                       "calibration_dates": [str(cut.date()), str(d.max().date())]}}
    for n in names:
        out[n] = CANDIDATE_CLASSES[n]().fit(fit[features], y_fit, cal[features], y_cal,
                                            players_fit=fit["player_id"].to_numpy(),
                                            players_cal=cal["player_id"].to_numpy())
    return out


def score(fitted: dict, test: pd.DataFrame, features: list[str]) -> dict:
    """{candidate: probabilities} - each candidate's OWN probability, no adjustment of any kind."""
    return {n: np.asarray(m.predict(test[features], players=test["player_id"].to_numpy()), dtype=float)
            for n, m in fitted.items() if not n.startswith("_")}


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────

def _ll(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def reliability(y, p, bins: int = 10) -> list[dict]:
    q = pd.qcut(pd.Series(np.asarray(p, dtype=float)), bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float), "b": q.to_numpy()})
    g = df.groupby("b").agg(n=("y", "size"), mean_p=("p", "mean"), rate=("y", "mean"))
    return [{"bin": int(b), "n": int(r.n), "mean_p": float(r.mean_p), "observed": float(r.rate)}
            for b, r in g.iterrows()]


def metrics(y, p) -> dict:
    """Log loss, Brier, ECE (10 quantile bins), calibration intercept / slope, reliability table."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), "unscored": int((~ok).sum())}
    if not len(y):
        return out
    rel = reliability(y, p)
    a = b = None
    if len(np.unique(y)) > 1:
        from sklearn.linear_model import LogisticRegression
        pc = np.clip(p, EPS, 1 - EPS)
        lp = np.log(pc / (1 - pc)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y.astype(int))
        a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
    out.update({"events": int(y.sum()), "base_rate": float(y.mean()), "mean_p": float(p.mean()),
                "log_loss": float(_ll(y, p).mean()), "brier": float(((p - y) ** 2).mean()),
                "ece10": float(sum(r["n"] * abs(r["observed"] - r["mean_p"]) for r in rel) / len(y)),
                "calib_intercept": a, "calib_slope": b, "reliability": rel})
    return out


def block_ci(delta, blocks, reps: int = 1000, seed: int = 7) -> dict:
    """Week-block bootstrap of mean(delta): players of one week share matches, so rows of a week move
    together. Returns mean, 95% interval and the number of blocks."""
    delta = np.asarray(delta, dtype=float)
    ub, inv = np.unique(np.asarray(blocks), return_inverse=True)
    sums, cnts = np.bincount(inv, weights=delta), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(ub)
    bs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, k, k)
        bs[i] = sums[idx].sum() / cnts[idx].sum()
    return {"mean": float(delta.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "week_blocks": int(k)}


# ── evaluation (real data only through --run, owner-gated) ───────────────────────────────────────

def evaluate_market(frame: pd.DataFrame, market: str, features: list[str], decisions: dict,
                    folds: list[dict], names=CANDIDATES) -> tuple[dict, pd.DataFrame, dict]:
    """Walk-forward folds + the fixed-date holdout for one market on an ELIGIBLE frame (verified rows
    under the decided feature rule, label present). Returns (report, holdout scored rows, fitted
    holdout candidates)."""
    y_col = config.MARKET_TARGETS[market]
    f = frame[frame[y_col].notna()].copy()
    d = pd.to_datetime(f["date"]).dt.normalize()
    hs, he = pd.Timestamp(decisions["holdout_start"]), pd.Timestamp(decisions["holdout_end"])
    rep = {"market": market, "target": y_col, "rows": int(len(f)), "walk_forward": [], "holdout": {}}
    wf_parts = []
    for fold in folds:
        tr = f[d < fold["test_start"]]
        te = f[(d >= fold["test_start"]) & (d <= fold["test_end"])]
        rec = {"fold": fold["fold"], "test_dates": [str(fold["test_start"].date()), str(fold["test_end"].date())],
               "train_rows": int(len(tr)), "test_rows": int(len(te))}
        try:
            fitted = fit_window(tr, features, y_col, names)
        except ValueError as e:
            rec["status"] = f"SKIPPED: {e}"
            rep["walk_forward"].append(rec)
            continue
        ps = score(fitted, te, features)
        rec.update(status="OK", window=fitted["_window"],
                   metrics={n: metrics(te[y_col], p) for n, p in ps.items()})
        rep["walk_forward"].append(rec)
        wf_parts.append(pd.DataFrame({"y": te[y_col].to_numpy(), **{f"p_{n}": p for n, p in ps.items()}}))
    if wf_parts:
        wf = pd.concat(wf_parts, ignore_index=True)
        rep["walk_forward_pooled"] = {n: metrics(wf["y"], wf[f"p_{n}"]) for n in names}
    tr, te = f[d < hs], f[(d >= hs) & (d <= he)]
    rep["holdout"] = {"dates": [str(hs.date()), str(he.date())], "train_rows": int(len(tr)), "test_rows": int(len(te))}
    try:
        fitted = fit_window(tr, features, y_col, names)
    except ValueError as e:
        rep["holdout"]["status"] = f"SKIPPED: {e}"
        return rep, pd.DataFrame(), {}
    ps = score(fitted, te, features)
    rep["holdout"].update(status="OK", window=fitted["_window"],
                          metrics={n: metrics(te[y_col], p) for n, p in ps.items()})
    wk = pd.to_datetime(te["date"]).dt.strftime("%G-%V").to_numpy()
    y = te[y_col].to_numpy(dtype=float)
    if "player_base_rate" in ps:
        base = _ll(y, ps["player_base_rate"])
        rep["holdout"]["log_loss_minus_base_rate"] = {
            n: block_ci(_ll(y, p) - base, wk) for n, p in ps.items() if n != "player_base_rate" and len(y)}
    keep = [c for c in ("date", "player_id", "player_name", "fixture_id", "league", "home_team", "away_team",
                        "is_home", "harness_source", SOURCE_COL) if c in te.columns]
    scored = te[keep].copy()
    scored.insert(0, "market", market)
    scored["y"] = y
    for n, p in ps.items():
        scored[f"p_{n}"] = p
    return rep, scored, fitted


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Features and labels rebuilt from the raw counts over the full context (every row, verified or
    not), with the population's verification columns joined back on (fixture_id, player_id)."""
    from .feature_engineering import build_features
    pop = population(raw)
    built = build_features(raw)
    key = ["fixture_id", "player_id"]
    ver = pop[key + ["label_verified", "window_verified", "all_prior_verified"]].drop_duplicates(key)
    for c in key:
        built[c] = pd.to_numeric(built[c], errors="coerce")
        ver[c] = pd.to_numeric(ver[c], errors="coerce")
    return built.merge(ver, on=key, how="inner")


def run(plan: dict | None = None, raw: pd.DataFrame | None = None, out_dir: Path | None = None,
        model_dir: Path | None = None, quotes: pd.DataFrame | None = None) -> dict:
    """The gated real run. Refuses unless PHASE2_RETRAIN_ENABLED=true and every owner decision is set,
    and stops (no fit) when the verified share is under the owner's minimum."""
    if not run_enabled():
        raise PermissionError(f"props retrain refused: {ENABLE_ENV} is not 'true' (owner approval gate)")
    plan = load_plan() if plan is None else plan
    dec = owner_decisions(plan)
    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    model_dir = MODEL_DIR if model_dir is None else Path(model_dir)
    raw = source_frame() if raw is None else raw
    frame = build_frame(raw)
    split = split_of(frame["date"], dec["holdout_start"], dec["holdout_end"])
    in_scope = split.isin(["pre_holdout", "holdout"]).to_numpy()
    ok = eligible(frame, dec["feature_rule"]).to_numpy()
    share = float(ok[in_scope].mean()) if in_scope.any() else 0.0
    pre = (split == "pre_holdout").to_numpy()
    hol = (split == "holdout").to_numpy()
    share_train = float(ok[pre].mean()) if pre.any() else 0.0
    share_holdout = float(ok[hol].mean()) if hol.any() else 0.0
    report = {"harness_version": HARNESS_VERSION, "plan_version": plan["plan_version"],
              "generated_at": _now(), "code_sha": _git_sha(), "evidence_class": "RETROSPECTIVE",
              "hand_adjustment": HAND_ADJUSTMENT, "owner_decisions": dec,
              "promotion": "NONE - no serving file is written; any cutover is an owner decision",
              "verified_share": share, "verified_share_train": share_train,
              "verified_share_holdout": share_holdout,
              "rows_in_scope": int(in_scope.sum()), "markets": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    short = []
    if share_train < float(dec["min_verified_share"]):
        short.append(f"training share {share_train:.4f} < {float(dec['min_verified_share']):.4f}")
    if share_holdout < float(dec["min_verified_share_holdout"]):
        short.append(f"holdout share {share_holdout:.4f} < {float(dec['min_verified_share_holdout']):.4f}")
    if short:
        report["status"] = ("STOPPED: verified share under the owner's minimum (" + "; ".join(short)
                            + "); nothing was fitted")
        _write_json(out_dir / "props_harness_report.json", report)
        return report
    elig = frame[ok]
    features = harness_features(elig)
    report["features"] = {"n": len(features), "sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
                          "profile_readmitted": sorted(config.PROFILE_FEATURES_WITHHELD & set(features)),
                          "profile_coverage": {c: float(elig[c].notna().mean()) for c in
                                               sorted(config.PROFILE_FEATURES_WITHHELD & set(features))}}
    wf = plan["walk_forward"]
    folds = walk_forward_folds(matchdays(frame["date"]), dec["holdout_start"],
                               int(wf["step_matchdays"]), int(wf["n_folds"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    scored_parts = []
    for market in dec["markets"]:
        rep, scored, fitted = evaluate_market(elig, market, features, dec, folds)
        for n, m in fitted.items():
            if n.startswith("_"):
                continue
            raw_bytes = pickle.dumps({"market": market, "candidate": n, "harness_version": HARNESS_VERSION,
                                      "hand_adjustment": HAND_ADJUSTMENT, "model": m, "state": m.state(),
                                      "window": fitted["_window"], "owner_decisions": dec})
            (model_dir / f"props_clean_{market}_{n}.pkl").write_bytes(raw_bytes)
            rep.setdefault("artifacts", {})[n] = {"path": f"models/challengers/props_clean/props_clean_{market}_{n}.pkl",
                                                  "sha256": hashlib.sha256(raw_bytes).hexdigest()}
        report["markets"][market] = rep
        if len(scored):
            scored_parts.append(scored)
    if scored_parts:
        from . import props_harness_clv as CLV
        allsc = pd.concat(scored_parts, ignore_index=True)
        allsc.to_csv(out_dir / "props_harness_holdout_scored.csv.gz", index=False)
        q = CLV.load_quotes() if quotes is None else quotes
        report["clv"] = {n: CLV.summarise(CLV.clv_rows(allsc, q, f"p_{n}")) for n in CANDIDATES}
    report["status"] = "OK"
    _write_json(out_dir / "props_harness_report.json", report)
    return report


# ── dry run (committed data: counts only, nothing fitted or scored) ──────────────────────────────

DRY_COLUMNS = ["date", "fixture_id", "player_id", "player_name", "league", "home_team", "away_team",
               "minutes", "goals", "assists", "shots_on_target", "yellow_cards", SOURCE_COL]


def proven_refetch_fixtures() -> tuple[set, dict]:
    """Fixture ids the re-fetch plan (scripts/collect_player_refetch.plan) can fetch: proven against
    the fixture lists, already journaled or still to do. Offline: the plan reads committed files."""
    sys.path.insert(0, str(config.BASE_DIR / "scripts"))
    import collect_player_refetch as CPR
    p = CPR.plan()
    ids = set(pd.to_numeric(p["todo"]["fixture_id"], errors="coerce").dropna().astype(int))
    j = CPR.read_parquet(CPR.JOURNAL)
    if len(j):
        rows_ids = set(pd.to_numeric(p["rows_frame"]["fixture_id"], errors="coerce").dropna().astype(int))
        ids |= set(pd.to_numeric(j["fixture_id"], errors="coerce").dropna().astype(int)) & rows_ids
    info = {k: p[k] for k in ("rows", "rows_by_source", "rows_verified_at_parse", "rows_to_verify",
                              "fixtures", "proven_fixtures", "journaled_fixtures", "planned_calls",
                              "refused_fixtures", "refused_rows") if k in p}
    return ids, info


def _counts(pop: pd.DataFrame, split: pd.Series, markets, rule: str | None) -> dict:
    out = {}
    sel = pd.Series(True, index=pop.index) if rule is None else eligible(pop, rule)
    for m in markets:
        lab = label(pop, m).notna()
        out[m] = {s: int((sel & lab & split.eq(s)).sum()) for s in SPLITS}
    return out


def dry_run(raw: pd.DataFrame, holdout_start, holdout_end, markets, walk_forward: dict,
            proven_fixtures: set | None = None, quotes: pd.DataFrame | None = None,
            profiles: pd.DataFrame | None = None, holdout_basis: str = "proposed") -> dict:
    """Row counts per split, market and feature rule - verified TODAY and verified AFTER the re-fetch
    (upper bound: every not-yet-verified row of a fixture the re-fetch plan has proven). Nothing is
    fitted or scored."""
    pop_today = population(raw)
    after = None
    if proven_fixtures is not None:
        fid = pd.to_numeric(raw["fixture_id"], errors="coerce")
        after = fid.isin(list(proven_fixtures))
    pop_after = population(raw, verified_extra=after) if after is not None else None
    split = split_of(pop_today["date"], holdout_start, holdout_end)
    in_scope = split.isin(["pre_holdout", "holdout"])
    days = matchdays(pop_today["date"])
    folds = walk_forward_folds(days, holdout_start, int(walk_forward["step_matchdays"]),
                               int(walk_forward["n_folds"]))
    rep = {
        "harness_version": HARNESS_VERSION, "generated_at": _now(), "code_sha": _git_sha(),
        "mode": "DRY_RUN (counts only: nothing fitted, scored or priced)",
        "hand_adjustment": HAND_ADJUSTMENT,
        "data": {"source_rows": int(len(raw)),
                 "source_rows_by_source": raw["harness_source"].value_counts().to_dict(),
                 "count_source": raw[SOURCE_COL].value_counts().to_dict(),
                 "oldest": str(pd.to_datetime(raw["date"]).min().date()),
                 "newest": str(pd.to_datetime(raw["date"]).max().date()),
                 "population_rows": int(len(pop_today)),
                 "population_rows_in_scope": int(in_scope.sum())},
        "holdout": {"start": str(pd.Timestamp(holdout_start).date()), "end": str(pd.Timestamp(holdout_end).date()),
                    "basis": holdout_basis},
        "walk_forward": {"step_matchdays": int(walk_forward["step_matchdays"]),
                         "n_folds_requested": int(walk_forward["n_folds"]), "n_folds": len(folds),
                         "folds": [{"fold": f["fold"], "test_start": str(f["test_start"].date()),
                                    "test_end": str(f["test_end"].date())} for f in folds],
                         "matchdays_before_holdout": int(sum(pd.Timestamp(x) < pd.Timestamp(holdout_start)
                                                             for x in days))},
        "counts_all_rows": _counts(pop_today, split, markets, None),
        "verified_today": {}, "verified_after_refetch_upper_bound": {}, "verified_share": {},
    }
    split_after = split_of(pop_after["date"], holdout_start, holdout_end) if pop_after is not None else None
    for rule in FEATURE_RULES:
        rep["verified_today"][rule] = _counts(pop_today, split, markets, rule)
        share = {"today": float(eligible(pop_today, rule)[in_scope].mean()) if in_scope.any() else 0.0}
        if pop_after is not None:
            rep["verified_after_refetch_upper_bound"][rule] = _counts(pop_after, split_after, markets, rule)
            sc = split_after.isin(["pre_holdout", "holdout"])
            share["after_refetch_upper_bound"] = float(eligible(pop_after, rule)[sc].mean()) if sc.any() else 0.0
        rep["verified_share"][rule] = share
    rep["walk_forward"]["test_rows_verified"] = {}
    for rule in FEATURE_RULES:
        per = {}
        for view, pop in (("today", pop_today), ("after_refetch_upper_bound", pop_after)):
            if pop is None:
                continue
            d = pd.to_datetime(pop["date"]).dt.normalize()
            sel = eligible(pop, rule)
            per[view] = [int((sel & (d >= f["test_start"]) & (d <= f["test_end"])).sum()) for f in folds]
        rep["walk_forward"]["test_rows_verified"][rule] = per
    rep["notes"] = [
        "verified_after_refetch_upper_bound treats every unverified row of a fixture the re-fetch plan has "
        "proven as verified. The real number is lower: a player the answer cannot prove, or an empty answer, "
        "stays unverified, and a re-fetched count can become missing, which removes that row's label.",
        "Label counts use the stored counts (the pre-fix parser wrote no blank), so on unverified rows "
        "they are an upper bound for the same reason.",
    ]
    rep["profiles"] = profile_coverage(pop_today, profiles)
    if quotes is not None:
        from . import props_harness_clv as CLV
        rep["two_sided_prices"] = CLV.pair_counts(quotes)
    return rep


def profile_coverage(pop: pd.DataFrame, profiles: pd.DataFrame | None = None) -> dict:
    """Share of population rows whose player has a collected real height / birth date."""
    from .feature_engineering import collected_profiles
    prof = collected_profiles() if profiles is None else profiles
    if prof is None or prof.empty:
        return {"players_with_profile": 0, "rows_with_height": 0, "rows_with_birth_date": 0, "rows": int(len(pop))}
    p = prof.set_index("player_id")
    h = pop["player_id"].map(p["height_cm"]).notna()
    b = pop["player_id"].map(p["birth_date"]).notna()
    return {"players_with_profile": int(len(p)), "rows": int(len(pop)),
            "rows_with_height": int(h.sum()), "rows_with_birth_date": int(b.sum()),
            "placeholders_used": "none (age 25 / 180 cm / 75 kg are read as missing)"}


# ── helpers / CLI ────────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR, text=True).strip()
    except Exception:                                                   # noqa: BLE001
        return os.environ.get("GITHUB_SHA", "UNIDENTIFIABLE")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 props clean retrain harness")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="count rows per split / market (no fit)")
    g.add_argument("--run", action="store_true", help="the owner-gated real run (GitHub Actions only)")
    ap.add_argument("--holdout-start")
    ap.add_argument("--holdout-end")
    ap.add_argument("--out", help="write the dry-run JSON here")
    a = ap.parse_args(argv)
    plan = load_plan()
    if a.run:
        rep = run(plan)
        print(json.dumps({k: rep[k] for k in ("status", "verified_share", "rows_in_scope")}, indent=2))
        return 0
    dec, prop = plan.get("owner_decisions") or {}, plan.get("proposed") or {}
    if a.holdout_start or a.holdout_end:
        hs, he, basis = a.holdout_start, a.holdout_end, "command line"
    elif dec.get("holdout_start") and dec.get("holdout_end"):
        hs, he, basis = dec["holdout_start"], dec["holdout_end"], "owner"
    else:
        hs, he, basis = prop["holdout_start"], prop["holdout_end"], "PROPOSED (not an owner decision)"
    markets = dec.get("markets") or list(MARKETS_IN_SCOPE)
    raw = source_frame(columns=DRY_COLUMNS)
    proven, plan_info = proven_refetch_fixtures()
    from . import props_harness_clv as CLV
    rep = dry_run(raw, hs, he, markets, plan["walk_forward"], proven_fixtures=proven, quotes=CLV.load_quotes(),
                  holdout_basis=basis)
    rep["refetch_plan"] = plan_info
    rep["owner_decisions_missing"] = missing_owner_decisions(plan)
    text = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
