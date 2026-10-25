"""WOWZA data contracts (CONTRACTS-1, 2026-10-25). See src/contracts/schemas.py.

    from src.contracts import validate, SCHEMAS, committed_violations
    validate(df, SCHEMAS["STAKE_PLAN_ROW"])        # list of violations, never raises
    python -m src.contracts                         # validate every committed file, print a summary
    python -m src.contracts --json                  # same, as JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.contracts.schema import Col, Rule, Schema, summarize, validate  # noqa: F401
from src.contracts.schemas import COMMITTED_FILES, CONTRACTS_VERSION, SCHEMAS  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def load_frame(rel: str, root: Path | None = None) -> pd.DataFrame | None:
    """A committed file as strings exactly as stored (None when absent)."""
    p = (root or ROOT) / rel
    if not p.exists():
        return None
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        return pd.DataFrame(d.get("experiments", []) if isinstance(d, dict) else d)
    return pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)


def committed_violations(root: Path | None = None, files: dict | None = None) -> dict:
    """{file: {schema, rows, n_violations, violations}} for every committed contract file. Never raises."""
    out = {}
    for rel, name in (files or COMMITTED_FILES).items():
        try:
            df = load_frame(rel, root)
            if df is None:
                out[rel] = {"schema": name, "present": False, "rows": 0, "n_violations": 0, "violations": []}
                continue
            v = validate(df, SCHEMAS[name])
            out[rel] = {"schema": name, "schema_version": SCHEMAS[name].version, "present": True,
                        "rows": int(len(df)), **summarize(v), "violations": v}
        except Exception as e:                                         # noqa: BLE001
            out[rel] = {"schema": name, "present": None, "rows": 0, "n_violations": 1,
                        "violations": [{"kind": "internal_error", "detail": f"{type(e).__name__}: {e}"[:200]}]}
    return out
