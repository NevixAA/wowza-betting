"""
Training-only feature imputer, persisted inside the model payload (Standard forensic repair 2026-10-28).
======================================================================================================

WHY. `src/model._prep` (legacy) fills every missing feature with the median OF THE FRAME BEING
SCORED (`X.median()`). That has two consequences, both measured in the 2026-10-28 forensic audit:

  * HOLDOUT LEAK — during training the test/holdout block is imputed with its OWN medians, i.e.
    with information from the rows being evaluated.
  * BATCH DEPENDENCE — live, a fixture's probability depends on which OTHER fixtures happen to be
    on the board (and, for side markets, on the New Format fixtures pooled into the same call).

The repaired recipe fits the imputer on the BASE-TRAIN block only, stores it in the payload
(`payload["imputer"]`, versioned) and applies it unchanged to calibration, meta-training,
holdout, prospective and live rows. A payload without an `imputer` key keeps the legacy
behaviour, which is why every serving model is byte-for-byte unaffected.

Logic copied (not imported) from wowzaV9-Pro `src/features/imputers.py` per the owner rule that
v9 never imports Pro. Differences from Pro, all deliberate:
  * `to_dict` / `from_dict` so the manifest can live inside a pickle payload;
  * a feature with NO training values (e.g. xG for Standard) is filled with `fill_all_missing`
    (0.0, the same constant legacy `_prep` used) — on train AND serve, so it is at least a
    consistent constant instead of a value the scaler maps to z=-37;
  * required-feature degradation is reported, never used to drop a row (the caller decides).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

IMPUTER_VERSION = "v9-train-only-imputer-1.0"

#: invariant 8 — a fixture with no rolling-form history must never look like an ordinary one.
DEFAULT_REQUIRED = ("home_scored_last5", "away_scored_last5",
                    "home_conceded_last5", "away_conceded_last5")


@dataclass
class FeatureStats:
    name: str
    dtype: str
    count: int
    missing_rate: float
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    q01: float | None = None
    q05: float | None = None
    q25: float | None = None
    q75: float | None = None
    q95: float | None = None
    q99: float | None = None
    required: bool = False
    degenerate: bool = False


@dataclass
class ImputerManifest:
    version: str = IMPUTER_VERSION
    fitted_on_rows: int = 0
    fitted_on_dates: tuple = ("", "")
    fill_all_missing: float = 0.0
    clip_to_training_range: bool = True
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"version": self.version, "fitted_on_rows": self.fitted_on_rows,
                "fitted_on_dates": list(self.fitted_on_dates),
                "fill_all_missing": self.fill_all_missing,
                "clip_to_training_range": self.clip_to_training_range,
                "features": {k: asdict(v) for k, v in self.features.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ImputerManifest":
        m = cls(version=d.get("version", IMPUTER_VERSION),
                fitted_on_rows=int(d.get("fitted_on_rows", 0)),
                fitted_on_dates=tuple(d.get("fitted_on_dates", ["", ""])),
                fill_all_missing=float(d.get("fill_all_missing", 0.0)),
                clip_to_training_range=bool(d.get("clip_to_training_range", True)))
        for k, v in (d.get("features") or {}).items():
            m.features[k] = FeatureStats(**v)
        return m

    @property
    def feature_names(self) -> list:
        return list(self.features)


def as_manifest(obj) -> "ImputerManifest | None":
    if obj is None:
        return None
    if isinstance(obj, ImputerManifest):
        return obj
    if isinstance(obj, dict):
        return ImputerManifest.from_dict(obj)
    raise TypeError(f"not an imputer manifest: {type(obj).__name__}")


def fit_imputer(train_df: pd.DataFrame, feature_cols: list, *, required=DEFAULT_REQUIRED,
                date_col: str = "date", clip_to_training_range: bool = True) -> ImputerManifest:
    """Fit on the BASE-TRAIN block and nothing else (calibration / meta / holdout rows = leak)."""
    req = set(required or ())
    m = ImputerManifest(fitted_on_rows=int(len(train_df)),
                        clip_to_training_range=clip_to_training_range)
    if date_col in train_df.columns and len(train_df):
        d = pd.to_datetime(train_df[date_col], errors="coerce")
        m.fitted_on_dates = (str(d.min())[:10], str(d.max())[:10])
    for c in feature_cols:
        if c not in train_df.columns:
            m.features[c] = FeatureStats(name=c, dtype="absent", count=0, missing_rate=1.0,
                                         required=c in req)
            continue
        s = pd.to_numeric(train_df[c], errors="coerce")
        n_valid = int(s.notna().sum())
        st = FeatureStats(name=c, dtype=str(train_df[c].dtype), count=n_valid,
                          missing_rate=round(1.0 - n_valid / max(1, len(s)), 6),
                          required=c in req)
        if n_valid:
            st.mean, st.std = float(s.mean()), float(s.std(ddof=0))
            st.median, st.min, st.max = float(s.median()), float(s.min()), float(s.max())
            for q, attr in ((0.01, "q01"), (0.05, "q05"), (0.25, "q25"),
                            (0.75, "q75"), (0.95, "q95"), (0.99, "q99")):
                setattr(st, attr, float(s.quantile(q)))
            st.degenerate = bool(st.std == 0.0)
        m.features[c] = st
    return m


@dataclass
class ImputeReport:
    rows: int
    imputed_counts: dict = field(default_factory=dict)
    degraded_rows: int = 0
    missing_required: dict = field(default_factory=dict)
    unknown_features: list = field(default_factory=list)


def apply_imputer(df: pd.DataFrame, manifest, feature_cols: list | None = None):
    """Fill from the TRAINING distribution. Returns (X, degraded_mask, report).

    X holds exactly `feature_cols` (default: the manifest's features) in that order, numeric.
    The result for a row depends ONLY on that row and the manifest — never on the batch.
    """
    manifest = as_manifest(manifest)
    cols = list(feature_cols) if feature_cols is not None else manifest.feature_names
    rep = ImputeReport(rows=int(len(df)))
    degraded = pd.Series(False, index=df.index)
    out = {}
    for name in cols:
        st = manifest.features.get(name)
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce").astype(float)
        else:
            s = pd.Series(np.nan, index=df.index, dtype=float)
            rep.unknown_features.append(name)
        miss = s.isna()
        n_miss = int(miss.sum())
        if st is not None and st.required and n_miss:
            degraded |= miss
            rep.missing_required[name] = n_miss
        fill = (st.median if (st is not None and st.median is not None)
                else manifest.fill_all_missing)
        if n_miss:
            s = s.fillna(fill)
            rep.imputed_counts[name] = n_miss
        if (manifest.clip_to_training_range and st is not None
                and st.min is not None and st.max is not None):
            s = s.clip(lower=st.min, upper=st.max)
        out[name] = s
    X = pd.DataFrame(out, index=df.index, columns=cols)
    rep.degraded_rows = int(degraded.sum())
    return X, degraded, rep


def batch_median_would_differ(serve_df: pd.DataFrame, manifest, *, tol: float = 1e-9) -> pd.DataFrame:
    """Diagnostic: how far the SERVING batch median sits from the TRAINING median."""
    manifest = as_manifest(manifest)
    rows = []
    for name, st in manifest.features.items():
        if name not in serve_df.columns or st.median is None:
            continue
        s = pd.to_numeric(serve_df[name], errors="coerce")
        if s.notna().sum() == 0:
            continue
        bm = float(s.median())
        if abs(bm - st.median) > tol:
            span = (st.max - st.min) if (st.max is not None and st.min is not None) else None
            rows.append({"feature": name, "training_median": st.median, "batch_median": bm,
                         "abs_diff": abs(bm - st.median),
                         "diff_vs_training_range": (abs(bm - st.median) / span) if span else None})
    return (pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame())
