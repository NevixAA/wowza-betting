"""One-off rebuild of the prev_season_* and shot_accuracy_l10 columns of the stored player_history.

    python -m player_model.pipeline --mode rebuild-prev-season            # DRY RUN: counts only
    python -m player_model.pipeline --mode rebuild-prev-season --write    # owner-approved write

WHY (found 2026-11-24, Pro props learner run 45358726547). props6 2/5 (43d4eebdf, 2026-11-02) added
13 previous-season features and shot_accuracy_l10 to PLAYER_FEATURE_COLS, and props6 5/5
(33fcf85d3) said such columns "arrive for every row together, through a rebuild of
player_history". That rebuild never ran. On 2026-11-24 the nine PLAYER prev_season_* columns were
empty on all 893,422 stored rows, the four team ones and shot_accuracy_l10 held values only on
rows collected since 2026-05 / 2026-06. Serving (build_upcoming_features) and the props clean
harness (props_harness.build_frame) compute them from the raw counts over the full history, so
the served models see real values; any reader of the STORED columns saw nothing.

WHAT IT DOES. One build_features pass over the stored history (exactly what serving does, minus
the upcoming rows), then, per (fixture_id, player_id), the rebuilt value of the 14 columns goes
into a cell that is stored EMPTY. Nothing else changes:
  * keep-first: a stored non-empty cell is never rewritten; a rebuilt value that differs from it
    is counted (disagreements) and left alone;
  * no other column changes, and the row set and order are identical (asserted before writing);
  * no-shrink: fewer rows than stored -> refused;
  * a row build_features does not emit (minutes filter, first appearance, no real player_id)
    gets no value: missing stays missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_model.feature_engineering import PREV_SEASON_COLUMNS, build_features

KEY = ["fixture_id", "player_id"]
COLUMNS = (*PREV_SEASON_COLUMNS, "shot_accuracy_l10")
REL_TOL = 1e-5        # stored floats are float32 (pipeline._save_history)


def _keys(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[KEY].apply(pd.to_numeric, errors="coerce"))


def rebuild(existing: pd.DataFrame, built: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """(history with the empty COLUMNS cells filled from the rebuild, report). `built` is the
    build_features frame of `existing` (computed here when None; injectable for tests)."""
    if built is None:
        built = build_features(existing)
    out = existing.copy()
    ek = _keys(existing)
    bk = _keys(built)
    dup = int(bk.duplicated().sum())
    if dup:
        raise ValueError(f"build_features emitted {dup} duplicate (fixture_id, player_id) keys")
    pos = bk.get_indexer(ek)                        # -1: the build did not emit this stored row
    hit = pos >= 0
    rep = {"rows": int(len(existing)), "rows_rebuilt": int(hit.sum()),
           "rows_not_emitted_by_build": int((~hit).sum()), "columns": {}}
    for c in COLUMNS:
        new = np.full(len(existing), np.nan)
        if c in built.columns:
            new[hit] = pd.to_numeric(built[c], errors="coerce").to_numpy(dtype=float)[pos[hit]]
        if c in out.columns:
            old = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
        else:
            old = np.full(len(existing), np.nan)
        stored, fresh = ~np.isnan(old), ~np.isnan(new)
        fill = ~stored & fresh
        both = stored & fresh
        differ = both & ~np.isclose(old, new, rtol=REL_TOL, atol=1e-7, equal_nan=False)
        merged = np.where(fill, new, old)
        if c in out.columns:
            out[c] = pd.Series(merged, index=out.index).astype(out[c].dtype if out[c].dtype.kind == "f"
                                                                else "float64")
        else:
            out[c] = merged
        rep["columns"][c] = {"stored_non_null": int(stored.sum()), "rebuilt_non_null": int(fresh.sum()),
                             "cells_filled": int(fill.sum()), "after_non_null": int((stored | fill).sum()),
                             "stored_kept_disagreeing": int(differ.sum())}
    verify(existing, out)
    return out, rep


def verify(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Refuse unless only COLUMNS changed and only in cells that were empty."""
    if len(after) < len(before):
        raise ValueError(f"no-shrink: {len(after)} rows < {len(before)} stored")
    if len(after) != len(before) or not _keys(after).equals(_keys(before)):
        raise ValueError("row set or order changed")
    extra = [c for c in after.columns if c not in before.columns and c not in COLUMNS]
    if extra:
        raise ValueError(f"columns added outside the rebuild set: {extra}")
    for c in before.columns:
        if c in COLUMNS:
            o = pd.to_numeric(before[c], errors="coerce").to_numpy(dtype=float)
            n = pd.to_numeric(after[c], errors="coerce").to_numpy(dtype=float)
            kept = ~np.isnan(o)
            if not np.array_equal(o[kept], n[kept]):
                raise ValueError(f"keep-first: a stored {c} value changed")
        elif not before[c].equals(after[c]):
            raise ValueError(f"column {c} changed; only {COLUMNS} may")
