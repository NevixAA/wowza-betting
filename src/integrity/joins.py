"""Fixture-level left enrichment that can never expand, shift or guess (R2-C finding 12).

THE DEFECT THIS EXISTS FOR. `src/data_loader._enrich_with_api_shots` merged af_history.parquet on
(home, away, date) WITHOUT league and then copied the merged column back with a positional
`zip(df[col].values, merged[col].values)`. af_history holds several rows per key (two spellings of
one club that normalise alike, and whole leagues filed twice under another league's name), so the
left merge returned MORE rows than the left frame. `zip` stops at the shorter input, so from the
first duplicated key onward every value is copied onto the wrong fixture — a silent shift, not an
error. The same shape exists in the half-time merge and in the live per-league shots merge.

THE CONTRACT `safe_left_enrich` ENFORCES
  * the key carries fixture identity: `fixture_id`, or `league` + date + both clubs. A key without
    league or fixture id is refused (ValueError) unless the caller opts out explicitly;
  * the right side is collapsed to ONE row per key before the merge. Duplicate rows that agree are
    one row; where two rows disagree on a value, that CELL is ambiguous and is left unfilled — never
    first-wins, never last-wins;
  * merged rows == left rows, asserted, and every value is written back to the row it came from
    (the merge carries the left row number and the assignment is checked against it);
  * NaN-only fill by default: an existing value is never overwritten.
Every call leaves a JoinReport (row counts, duplicate right keys, ambiguous cells, unmapped left
rows, filled cells) in a process-wide registry; `write_join_health` writes them out.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

JOIN_HELPER_VERSION = "safe_left_enrich-1.0"

_REPORTS: dict[str, dict] = {}


class JoinIntegrityError(AssertionError):
    """Raised when an enrichment would change the row count or misalign rows."""


@dataclass
class JoinReport:
    name: str
    keys: list
    helper_version: str = JOIN_HELPER_VERSION
    mode: str = "safe"                  # safe | legacy_audit
    left_rows: int = 0
    right_rows: int = 0
    right_null_key_rows: int = 0
    right_duplicate_key_groups: int = 0
    right_duplicate_rows: int = 0
    ambiguous_key_groups: int = 0
    ambiguous_cells: dict = field(default_factory=dict)
    left_null_key_rows: int = 0
    matched_left_rows: int = 0
    unmapped_left_rows: int = 0
    output_rows: int = 0
    duplicate_output_rows: int = 0
    filled_cells: dict = field(default_factory=dict)
    overwritten_cells: int = 0
    legacy_expansion_rows: int | None = None
    legacy_misaligned_rows: int | None = None
    status: str = "PASS"
    notes: list = field(default_factory=list)

    def finalise(self) -> "JoinReport":
        if self.output_rows != self.left_rows or self.duplicate_output_rows:
            self.status = "FAIL"
        elif self.legacy_expansion_rows or self.legacy_misaligned_rows:
            self.status = "FAIL"
        elif self.ambiguous_key_groups:
            self.status = "WARN"
        else:
            self.status = "PASS"
        return self


def _has_identity(keys: Iterable[str]) -> bool:
    k = set(keys)
    return "fixture_id" in k or "league" in k


def collapse_right(right: pd.DataFrame, keys: list, value_cols: list,
                   report: JoinReport | None = None) -> pd.DataFrame:
    """One row per key. A cell whose non-null values disagree becomes NaN (ambiguous)."""
    r = right[keys + value_cols].copy()
    null_key = r[keys].isna().any(axis=1)
    if report is not None:
        report.right_rows = int(len(right))
        report.right_null_key_rows = int(null_key.sum())
    r = r[~null_key]
    dup = r.duplicated(keys, keep=False)
    if report is not None:
        report.right_duplicate_rows = int(dup.sum())
        report.right_duplicate_key_groups = int(r.loc[dup, keys].drop_duplicates().shape[0])
    if not dup.any():
        return r.reset_index(drop=True)
    single = r[~dup]
    multi = r[dup]
    g = multi.groupby(keys, sort=False, dropna=False)
    first = g[value_cols].first()               # first NON-NULL per column
    nun = g[value_cols].nunique(dropna=True)
    conflict = nun > 1
    first = first.mask(conflict)
    if report is not None:
        report.ambiguous_key_groups = int(conflict.any(axis=1).sum())
        report.ambiguous_cells = {c: int(conflict[c].sum()) for c in value_cols if conflict[c].any()}
    return pd.concat([single, first.reset_index()], ignore_index=True)


def safe_left_enrich(left: pd.DataFrame, right: pd.DataFrame, *, keys: list, value_cols: list,
                     name: str, target_cols: dict | None = None, fill: str = "nan_only",
                     require_identity: bool = True, register: bool = True
                     ) -> tuple[pd.DataFrame, JoinReport]:
    """Left-enrich `left` with `value_cols` from `right` on `keys`. Returns (new frame, report).

    `target_cols` maps a right column to the left column it fills (default: same name).
    `fill`: "nan_only" (default; never overwrites) or "overwrite".
    The left frame's index and row order are preserved exactly.
    """
    if fill not in ("nan_only", "overwrite"):
        raise ValueError(f"fill must be nan_only|overwrite, got {fill!r}")
    if require_identity and not _has_identity(keys):
        raise ValueError(f"[{name}] join key {keys} carries no league or fixture_id — a fixture "
                         f"join without competition identity can attach another league's match")
    missing = [k for k in keys if k not in left.columns or k not in right.columns]
    if missing:
        raise KeyError(f"[{name}] key column(s) missing: {missing}")
    target_cols = dict(target_cols or {})
    rep = JoinReport(name=name, keys=list(keys))
    rep.left_rows = int(len(left))
    value_cols = [c for c in value_cols if c in right.columns and c not in keys]

    r = collapse_right(right, list(keys), value_cols, rep)
    tag = "__r2c_rhs__"
    r = r.rename(columns={c: f"{c}{tag}" for c in value_cols})
    r["__r2c_hit__"] = True

    lk = left[list(keys)].copy()
    lk["__r2c_row__"] = np.arange(len(left))
    rep.left_null_key_rows = int(lk[list(keys)].isna().any(axis=1).sum())
    try:
        merged = lk.merge(r, on=list(keys), how="left", validate="many_to_one", sort=False)
    except pd.errors.MergeError as e:                     # collapse_right makes this impossible
        raise JoinIntegrityError(f"[{name}] right side not unique after collapse: {e}") from e
    if len(merged) != len(left):
        raise JoinIntegrityError(f"[{name}] merge changed row count {len(left)} -> {len(merged)}")
    merged = merged.sort_values("__r2c_row__", kind="mergesort")
    if not np.array_equal(merged["__r2c_row__"].to_numpy(), np.arange(len(left))):
        raise JoinIntegrityError(f"[{name}] merged rows do not map 1:1 onto left rows")

    out = left.copy()
    hit = merged["__r2c_hit__"].fillna(False).astype(bool).to_numpy()
    rep.matched_left_rows = int(hit.sum())
    rep.unmapped_left_rows = int(len(left) - hit.sum())
    for c in value_cols:
        tgt = target_cols.get(c, c)
        new = merged[f"{c}{tag}"].to_numpy()
        if tgt not in out.columns:
            out[tgt] = np.nan
        cur = out[tgt]
        if fill == "nan_only":
            take = cur.isna().to_numpy() & pd.notna(new)
        else:
            take = pd.notna(new)
            rep.overwritten_cells += int((take & cur.notna().to_numpy()).sum())
        if take.any():
            vals = cur.to_numpy(dtype=object, copy=True)
            vals[take] = new[take]
            out[tgt] = pd.Series(vals, index=out.index).infer_objects()
            if pd.api.types.is_numeric_dtype(pd.Series(new[pd.notna(new)])):
                out[tgt] = pd.to_numeric(out[tgt], errors="coerce")
        rep.filled_cells[tgt] = int(take.sum())
    rep.output_rows = int(len(out))
    rep.duplicate_output_rows = int(out.index.duplicated().sum())
    rep.finalise()
    if register:
        _REPORTS[name] = asdict(rep)
    return out, rep


def audit_legacy_merge(left: pd.DataFrame, right: pd.DataFrame, on: list, name: str,
                       register: bool = True) -> JoinReport:
    """Measure what a legacy `left.merge(right, on=on, how='left')` + positional copy-back does:
    how many rows the merge adds, and how many left rows a positional zip would misalign."""
    rep = JoinReport(name=name, keys=list(on), mode="legacy_audit")
    rep.left_rows = int(len(left))
    rep.right_rows = int(len(right))
    rk = right[list(on)].dropna()
    dup = rk.duplicated(list(on), keep=False)
    rep.right_duplicate_rows = int(dup.sum())
    rep.right_duplicate_key_groups = int(rk[dup].drop_duplicates().shape[0])
    lk = left[list(on)].reset_index(drop=True)
    m = lk.merge(rk.assign(__h=1), on=list(on), how="left")
    rep.output_rows = int(len(m))
    rep.legacy_expansion_rows = int(len(m) - len(left))
    n = len(lk)
    head = m[list(on)].iloc[:n].reset_index(drop=True)
    same = (head.astype(str).to_numpy() == lk.astype(str).to_numpy()).all(axis=1)
    rep.legacy_misaligned_rows = int((~same).sum())
    rep.matched_left_rows = int(lk.merge(rk.drop_duplicates(), on=list(on), how="inner").shape[0]) \
        if len(rk) else 0
    if not _has_identity(on):
        rep.notes.append("key omits league/fixture_id")
    rep.finalise()
    if rep.output_rows != rep.left_rows:
        rep.status = "FAIL"
    if register:
        _REPORTS[name] = asdict(rep)
    return rep


def audit_league_free_lookup(left: pd.DataFrame, right: pd.DataFrame, keys: list,
                             name: str, league_col: str = "league",
                             register: bool = True) -> JoinReport:
    """Audit a legacy pivot/reindex lookup keyed WITHOUT league (row count cannot change, so the
    risk is a value from ANOTHER competition). Counts right keys that span several leagues and
    left rows whose key is found in the right side only under a different league name. A league
    spelled differently in two sources shows up here too — the report says which pairs."""
    rep = JoinReport(name=name, keys=list(keys), mode="legacy_audit")
    rep.left_rows = int(len(left))
    rep.right_rows = int(len(right))
    rep.legacy_expansion_rows = 0
    rep.legacy_misaligned_rows = 0
    rep.notes.append("key omits league/fixture_id; pivot aggfunc picks one value per key")
    if league_col not in right.columns or league_col not in left.columns:
        rep.notes.append("league column unavailable on one side; cross-league risk not measurable")
        rep.status = "WARN"
        if register:
            _REPORTS[name] = asdict(rep)
        return rep
    r = right[list(keys) + [league_col]].dropna(subset=list(keys)).drop_duplicates()
    span = r.groupby(list(keys))[league_col].nunique()
    rep.right_duplicate_key_groups = int((span > 1).sum())
    rl = r.groupby(list(keys))[league_col].agg(lambda s: tuple(sorted(set(map(str, s)))))
    lk = left[list(keys) + [league_col]].copy()
    m = lk.merge(rl.rename("_rlg").reset_index(), on=list(keys), how="inner")
    rep.matched_left_rows = int(len(m))
    other = m[[str(a) not in b for a, b in zip(m[league_col], m["_rlg"])]]
    rep.ambiguous_key_groups = int(len(other))         # left rows filled from another league label
    if len(other):
        pairs = (other[league_col].astype(str) + " <- " + other["_rlg"].map("|".join))
        rep.ambiguous_cells = {k: int(v) for k, v in pairs.value_counts().head(15).items()}
    rep.unmapped_left_rows = int(rep.left_rows - rep.matched_left_rows)
    rep.output_rows = rep.left_rows
    rep.status = "WARN" if len(other) or rep.right_duplicate_key_groups else "PASS"
    if register:
        _REPORTS[name] = asdict(rep)
    return rep


def reports() -> dict:
    return dict(_REPORTS)


def reset() -> None:
    _REPORTS.clear()


def write_join_health(path: str | Path, extra: dict | None = None) -> dict:
    statuses = [r["status"] for r in _REPORTS.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else
                                                  ("PASS" if statuses else "UNKNOWN"))
    doc = {"artifact": "join_integrity_health", "helper_version": JOIN_HELPER_VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "status": overall, "joins": _REPORTS}
    if extra:
        doc.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return doc
