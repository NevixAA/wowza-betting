"""
preserving_upsert.py - merge new rows into a stored table WITHOUT losing anything it does not own.

THE DEFECT THIS EXISTS FOR (2026-11-16). src/data_loader's af_history write-back did
`concat([stored, new]).drop_duplicates(key, keep="last")` where `new` held only the shot / xG columns.
For every key it touched, the stored provider row was REPLACED by a shots-only row: corners, fouls,
cards, the full-time score (and, from PR #34 on, fixture_id) of that row went from present to missing.
Git history of output/af_history.parquet shows 28,412 provider rows lost that way. Several collectors
had the same shape in smaller form (`old[COLS]` / `old.reindex(columns=COLS)` before the concat drops
every stored column the writer does not list).

THE RULE. A writer OWNS some columns. `upsert` therefore
  * keeps every stored row, in its stored order, and every stored column, with its stored dtype;
  * for a stored row whose key the writer supplies, changes ONLY the owned columns, and only as the
    mode says:
        "fill"       a missing stored cell takes the new value; a present one is never changed
        "update"     a non-missing new value replaces the stored one; a missing new value never
                     erases a stored one (missing stays missing, present stays present)
        "replace"    the owned cells take the new row's cells as they are (a fresh provider answer
                     that legitimately says "not given") - still never touches non-owned columns
        "keep"       the stored row wins entirely (first answer kept)
  * appends a new row whose key is not stored, with every non-owned column missing (never a default);
  * never removes a row - not even a stored duplicate. Removing rows is a correction, decided
    elsewhere and written beside the original.

Keys are compared normalised (dates per `date_keys` to YYYY-MM-DD, whole floats as integers, text as
text), so 123 and 123.0, or a Timestamp and its ISO string, are the same key. Stored cells are written
back exactly as read: the round trip of an untouched row is identical, value and dtype.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODES = ("fill", "update", "replace", "keep")


@dataclass
class UpsertReport:
    stored_rows: int = 0
    new_rows: int = 0
    matched_keys: int = 0
    appended_rows: int = 0
    cells_changed: Counter = field(default_factory=Counter)

    @property
    def changed(self) -> bool:
        return bool(self.appended_rows or any(self.cells_changed.values()))


def _norm_key_col(s: pd.Series, as_date: bool) -> pd.Series:
    if as_date:
        d = pd.to_datetime(s, errors="coerce", utc=False)
        try:
            out = d.dt.strftime("%Y-%m-%d")
        except (AttributeError, TypeError):   # mixed tz -> per-value
            out = pd.Series([x.strftime("%Y-%m-%d") if pd.notna(x) else None for x in d], index=s.index)
        return out.fillna("<NA>").astype(str)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.map(lambda x: x.isoformat() if pd.notna(x) else "<NA>").astype(str)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        def _n(x):
            if pd.isna(x):
                return "<NA>"
            f = float(x)
            return str(int(f)) if f.is_integer() else repr(f)
        return s.map(_n).astype(str)

    def _o(x):
        if x is None or (not isinstance(x, str) and pd.isna(x)):
            return "<NA>"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return str(int(x)) if float(x).is_integer() else repr(float(x))
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        return str(x)
    return s.map(_o).astype(str)


def key_tuples(df: pd.DataFrame, key: list[str], date_keys=()) -> pd.Series:
    """One hashable normalised key per row."""
    if df.empty:
        return pd.Series([], index=df.index, dtype=object)
    parts = [_norm_key_col(df[k], k in date_keys) for k in key]
    return pd.Series(list(zip(*parts)), index=df.index, dtype=object)


def _same(a, b) -> bool:
    na_a, na_b = _isna(a), _isna(b)
    if na_a or na_b:
        return na_a and na_b
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _isna(x) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _cast_like(col: pd.Series, dtype) -> pd.Series:
    try:
        return col.astype(dtype)
    except (TypeError, ValueError):
        return col


def upsert(stored: pd.DataFrame | None, new: pd.DataFrame, key: list[str], owned: list[str] | None = None,
           mode: str = "update", date_keys=(), new_keep: str = "last") -> tuple[pd.DataFrame, UpsertReport]:
    """Return (merged, report). `owned` defaults to every column of `new` that is not a key column."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    missing_key = [k for k in key if k not in new.columns]
    if missing_key:
        raise KeyError(f"new rows lack key column(s) {missing_key}")
    owned = [c for c in (owned if owned is not None else new.columns) if c not in key and c in new.columns]
    rep = UpsertReport(new_rows=int(len(new)))
    new = new[list(key) + owned].copy()
    new["_k"] = key_tuples(new, key, date_keys)
    new = new.drop_duplicates("_k", keep=new_keep)

    if stored is None or (len(stored) == 0 and len(stored.columns) == 0):
        out = new.drop(columns="_k").reset_index(drop=True)
        rep.appended_rows = int(len(out))
        return out, rep

    out = stored.copy()
    rep.stored_rows = int(len(out))
    for c in key + owned:                        # a column the store has never had: added, all missing
        if c not in out.columns:
            out[c] = pd.Series(pd.NA if pd.api.types.is_extension_array_dtype(new[c]) else np.nan,
                               index=out.index, dtype=new[c].dtype if c in owned else object)
    sk = key_tuples(out, key, date_keys)
    nk = new.set_index("_k")
    hit = sk.isin(nk.index)
    rep.matched_keys = int(sk[hit].nunique())

    if mode != "keep" and hit.any():
        rows = out.index[hit.to_numpy()]
        src = nk.loc[sk[hit].to_numpy()]
        for c in owned:
            cur = out.loc[rows, c]
            val = pd.Series(src[c].to_numpy(), index=rows)
            differs = pd.Series([not _same(a, b) for a, b in zip(cur.to_numpy(), val.to_numpy())],
                                index=rows, dtype=bool)
            if mode == "fill":
                take = cur.isna() & val.notna()
            elif mode == "update":
                take = val.notna() & differs
            else:                                # replace
                take = differs
            n = int(take.sum())
            if n:
                idx = rows[take.to_numpy()]
                if out[c].dtype != object:
                    try:
                        out.loc[idx, c] = val[take].astype(out[c].dtype).to_numpy()
                    except (TypeError, ValueError):
                        out[c] = out[c].astype(object)
                        out.loc[idx, c] = val[take].to_numpy()
                else:
                    out.loc[idx, c] = val[take].to_numpy()
                rep.cells_changed[c] += n

    add = new[~new["_k"].isin(set(sk))].drop(columns="_k")
    rep.appended_rows = int(len(add))
    if len(add):
        dtypes = out.dtypes
        add = add.reindex(columns=out.columns)
        out = pd.concat([out, add], ignore_index=True)
        for c in out.columns:                    # the stored dtype stays the column's dtype
            if out[c].dtype != dtypes[c]:
                out[c] = _cast_like(out[c], dtypes[c])
    return out, rep


def union_columns(old: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`old` with every column in `columns` present (added as missing) and NO column of `old` dropped -
    the column-preserving replacement for `old[columns]` / `old.reindex(columns=columns)`."""
    out = old.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = None
    return out[list(columns) + [c for c in out.columns if c not in columns]]
