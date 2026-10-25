"""A light, dependency-free table validator. Contracts are DATA; `validate` never raises.

    from src.contracts import validate, SCHEMAS
    violations = validate(df, SCHEMAS["STAKE_PLAN_ROW"])      # [] when the frame honours the contract

WHY NOT PYDANTIC. pydantic happens to be in the local venv but is not in requirements.txt, so CI
would not have it; and a row-by-row model over a 36k-row change-log is slow. A contract here is a
frozen list of `Col`s plus cross-column `Rule`s, evaluated column-wise with pandas.

A VIOLATION is a dict, never an exception:
    {"schema", "version", "kind", "column", "n_rows", "examples", "detail"}
kinds: not_a_frame | missing_column | null | dtype | allowed | range | pattern | duplicate_key |
       rule | internal_error
`examples` holds at most `max_examples` offending VALUES (row positions for a rule), so a report
can show what was wrong without copying the frame.

NULLS. Frames reach the validator both typed (from a writer) and as dtype=str with
keep_default_na=False (from a committed CSV), so "", "nan", "NaN", "None", "<NA>", "NaT" all count
as null on every path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

NULL_TOKENS = ("", "nan", "NaN", "None", "<NA>", "NaT", "none", "null")
DTYPES = ("str", "float", "int", "datetime", "date", "bool")


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str = "str"
    required: bool = True              # the column must exist
    nullable: bool = True              # a present column may hold blanks
    allowed: tuple | None = None       # categorical vocabulary (compared as stripped strings)
    min: float | None = None           # inclusive numeric bounds
    max: float | None = None
    pattern: str | None = None         # full-match regex on the string value
    doc: str = ""


@dataclass(frozen=True)
class Rule:
    """A cross-column invariant. `check(df)` returns a boolean Series, True where a row VIOLATES.
    The rule is skipped (and says so) when any column in `requires` is absent."""
    name: str
    check: Callable[[pd.DataFrame], pd.Series]
    doc: str = ""
    requires: tuple = ()


@dataclass(frozen=True)
class Schema:
    name: str
    version: str
    description: str
    columns: tuple
    key: tuple = ()                    # unique key; duplicates are reported when `unique_key`
    unique_key: bool = False
    rules: tuple = ()
    kind: str = ""                     # contract family (e.g. research_observation)
    writers: tuple = field(default=())  # file(s) this contract describes

    def col(self, name: str) -> Col | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "kind": self.kind,
                "description": self.description, "writers": list(self.writers),
                "key": list(self.key), "unique_key": self.unique_key,
                "columns": [{"name": c.name, "dtype": c.dtype, "required": c.required,
                             "nullable": c.nullable, "allowed": list(c.allowed) if c.allowed else None,
                             "min": c.min, "max": c.max, "pattern": c.pattern, "doc": c.doc}
                            for c in self.columns],
                "rules": [{"name": r.name, "doc": r.doc, "requires": list(r.requires)} for r in self.rules]}


# ── helpers usable by rules ─────────────────────────────────────────────────────────────────────
def null_mask(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(NULL_TOKENS)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ts(s: pd.Series) -> pd.Series:
    """UTC timestamps; naive values are read as UTC, unparseable as NaT."""
    s2 = s.where(~null_mask(s), None)
    try:
        return pd.to_datetime(s2, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s2, errors="coerce", utc=True)


def text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


# ── validation ──────────────────────────────────────────────────────────────────────────────────
def _v(schema: Schema, kind: str, column, n: int, examples=(), detail: str = "") -> dict:
    ex = []
    for e in list(examples):
        if isinstance(e, (np.integer,)):
            e = int(e)
        elif isinstance(e, (np.floating,)):
            e = float(e)
        elif not isinstance(e, (int, float, str, bool)) and e is not None:
            e = str(e)
        ex.append(e)
    return {"schema": schema.name, "version": schema.version, "kind": kind, "column": column,
            "n_rows": int(n), "examples": ex, "detail": detail}


def _examples(values: pd.Series, k: int) -> list:
    out = []
    for v in values.astype(str).unique()[:k]:
        out.append(v[:80])
    return out


def _check_col(df: pd.DataFrame, c: Col, schema: Schema, k: int) -> list[dict]:
    out = []
    s = df[c.name]
    if isinstance(s, pd.DataFrame):                                   # duplicated header
        return [_v(schema, "duplicate_column", c.name, len(df), detail="column name appears more than once")]
    nm = null_mask(s)
    if not c.nullable and nm.any():
        out.append(_v(schema, "null", c.name, nm.sum(), detail="blank in a non-nullable column"))
    val = s[~nm]
    if val.empty:
        return out
    if c.dtype in ("float", "int"):
        x = num(val)
        bad = x.isna()
        if c.dtype == "int":
            bad = bad | ((x - x.round()).abs() > 1e-9)
        if bad.any():
            out.append(_v(schema, "dtype", c.name, bad.sum(), _examples(val[bad], k), f"expected {c.dtype}"))
        x = x[~bad]
        if c.min is not None and (x < c.min).any():
            out.append(_v(schema, "range", c.name, (x < c.min).sum(), _examples(x[x < c.min], k), f"< min {c.min}"))
        if c.max is not None and (x > c.max).any():
            out.append(_v(schema, "range", c.name, (x > c.max).sum(), _examples(x[x > c.max], k), f"> max {c.max}"))
    elif c.dtype in ("datetime", "date"):
        t = ts(val)
        bad = t.isna()
        if bad.any():
            out.append(_v(schema, "dtype", c.name, bad.sum(), _examples(val[bad], k), f"expected {c.dtype}"))
    elif c.dtype == "bool":
        bad = ~text(val).str.lower().isin(("true", "false", "1", "0", "1.0", "0.0"))
        if bad.any():
            out.append(_v(schema, "dtype", c.name, bad.sum(), _examples(val[bad], k), "expected bool"))
    if c.allowed is not None:
        bad = ~text(val).isin([str(a) for a in c.allowed])
        if bad.any():
            out.append(_v(schema, "allowed", c.name, bad.sum(), _examples(val[bad], k),
                          f"allowed: {list(c.allowed)}"))
    if c.pattern is not None:
        rx = re.compile(c.pattern)
        bad = ~text(val).map(lambda v: bool(rx.fullmatch(v)))
        if bad.any():
            out.append(_v(schema, "pattern", c.name, bad.sum(), _examples(val[bad], k), f"pattern {c.pattern}"))
    return out


def validate(df, schema: Schema, max_examples: int = 3) -> list[dict]:
    """Every way `df` breaks `schema`, as a list of violation dicts. NEVER RAISES: an internal
    failure is itself reported as an `internal_error` violation."""
    try:
        if not isinstance(df, pd.DataFrame):
            return [_v(schema, "not_a_frame", None, 0, detail=type(df).__name__)]
        out = []
        cols = set(map(str, df.columns))
        for c in schema.columns:
            if c.name not in cols:
                if c.required:
                    out.append(_v(schema, "missing_column", c.name, len(df), detail="required column absent"))
                continue
            try:
                out += _check_col(df, c, schema, max_examples)
            except Exception as e:                                    # noqa: BLE001
                out.append(_v(schema, "internal_error", c.name, 0, detail=f"{type(e).__name__}: {e}"[:200]))
        if schema.unique_key and schema.key and set(schema.key) <= cols and len(df):
            k = df[list(schema.key)].astype(str)
            dup = k.duplicated(keep="first")
            if dup.any():
                out.append(_v(schema, "duplicate_key", list(schema.key), dup.sum(),
                              ["|".join(r) for r in k[dup].head(max_examples).values.tolist()],
                              "key must be unique"))
        for r in schema.rules:
            missing = [c for c in r.requires if c not in cols]
            if missing or not len(df):
                continue
            try:
                bad = pd.Series(r.check(df), index=df.index).fillna(False).astype(bool)
            except Exception as e:                                    # noqa: BLE001
                out.append(_v(schema, "internal_error", r.name, 0, detail=f"{type(e).__name__}: {e}"[:200]))
                continue
            if bad.any():
                pos = np.flatnonzero(bad.to_numpy())[:max_examples].tolist()
                out.append(_v(schema, "rule", r.name, bad.sum(), [f"row {p}" for p in pos], r.doc))
        return out
    except Exception as e:                                            # noqa: BLE001 - never raise
        return [_v(schema, "internal_error", None, 0, detail=f"{type(e).__name__}: {e}"[:200])]


def summarize(violations: list[dict]) -> dict:
    """{'n_violations': int (distinct findings), 'n_rows_affected': int (sum), 'by_kind': {...}}"""
    by = {}
    for v in violations:
        by[v["kind"]] = by.get(v["kind"], 0) + 1
    return {"n_violations": len(violations), "n_rows_affected": int(sum(v["n_rows"] for v in violations)),
            "by_kind": by}
