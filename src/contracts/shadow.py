"""SHADOW-MODE contract checks at the points where frames are written. OBSERVE ONLY.

    from src.contracts.shadow import observe
    observe("stake_plan.csv", written_frame, "STAKE_PLAN_ROW")     # never raises, returns violations

`observe` validates the frame a writer has ALREADY written and records the result in
output/contract_violations.json (one entry per frame, each with its own validated_at, plus a file
level generated_at). It never changes, filters, re-orders or blocks what is written: a writer calls
it after its own write, with the exact frame it wrote, inside its own try/except.

The file is RUNNER-LOCAL (gitignored): predict, daily_summary and evidence_refresh would otherwise all
race to commit it, and a tracked file modified by a job that does not commit it is the shape of the
2026-09 rebase incident. The committed signal is output/system_health.json, whose contract check
validates the committed files themselves (`committed_violations`). Path override for tests:
WOWZA_CONTRACT_VIOLATIONS=<path>; WOWZA_CONTRACTS_SHADOW=0 switches observation off entirely.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "output" / "contract_violations.json"
MAX_PER_FRAME = 50


def violations_path() -> Path:
    p = os.getenv("WOWZA_CONTRACT_VIOLATIONS")
    return Path(p) if p else DEFAULT_PATH


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(frame: str, schema_name: str, rows: int, violations: list[dict], path: Path | None = None,
           source: str = "writer") -> None:
    """Merge one frame's result into the violations file. Never raises."""
    try:
        from src.contracts.schema import summarize
        from src.contracts.schemas import CONTRACTS_VERSION, SCHEMAS
        p = Path(path) if path else violations_path()
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                doc = {}
        except Exception:                                              # noqa: BLE001
            doc = {}
        frames = doc.get("frames") if isinstance(doc.get("frames"), dict) else {}
        s = SCHEMAS.get(schema_name)
        frames[frame] = {"validated_at": _now(), "source": source, "schema": schema_name,
                         "schema_version": s.version if s else None, "rows": int(rows),
                         **summarize(violations), "violations": violations[:MAX_PER_FRAME]}
        out = {"about": "Contract violations observed in SHADOW mode (src/contracts). Nothing written was "
                        "changed because of them. Runner-local; the committed summary is in "
                        "output/system_health.json.",
               "generated_at": _now(), "contracts_version": CONTRACTS_VERSION,
               "n_violations": int(sum(f.get("n_violations", 0) for f in frames.values())),
               "frames": dict(sorted(frames.items()))}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:                                             # noqa: BLE001
        log.debug(f"[contracts] could not record {frame}: {type(e).__name__}: {e}")


def observe(frame: str, df, schema_name: str, path: Path | None = None) -> list[dict]:
    """Validate a frame a writer just wrote and record the result. NEVER RAISES, never mutates `df`."""
    if os.getenv("WOWZA_CONTRACTS_SHADOW", "1") == "0":
        return []
    try:
        from src.contracts.schema import validate
        from src.contracts.schemas import SCHEMAS
        v = validate(df, SCHEMAS[schema_name])
        record(frame, schema_name, len(df) if hasattr(df, "__len__") else 0, v, path)
        if v:
            log.info(f"[contracts] {frame}: {len(v)} {schema_name} violation(s) (shadow only, nothing changed)")
        return v
    except Exception as e:                                             # noqa: BLE001
        log.debug(f"[contracts] observe {frame} skipped: {type(e).__name__}: {e}")
        return []
