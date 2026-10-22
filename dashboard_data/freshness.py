"""Freshness banners for decision-facing pages (P1-3, 2026-10-22).

Reads the small output/artifact_manifest.json written by src/artifact_freshness.py (never file mtime).
A decision-facing artifact that is STALE / UNKNOWN / MISSING gets a red banner instead of being shown
like current truth; a dated study shows its date; a historical record says it is historical.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "output" / "artifact_manifest.json"


def manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {"artifacts": []}


def status(rel: str, m: dict | None = None) -> dict:
    for r in (m or manifest()).get("artifacts", []):
        if r["artifact"] == rel:
            return r
    return {"artifact": rel, "freshness": "UNKNOWN", "valid_for_decisions": False,
            "reason": "not in the artifact manifest - treated as stale"}


def banner(st, rel: str, label: str | None = None, m: dict | None = None) -> dict:
    """Render the freshness verdict for one artifact; returns the manifest row."""
    r = status(rel, m)
    name = label or rel
    f = r.get("freshness")
    if f in ("STALE", "UNKNOWN", "MISSING"):
        st.error(f"**STALE — NOT VALID FOR CURRENT PERFORMANCE DECISIONS** · {name}: {r.get('reason', '')}")
    elif f == "HISTORICAL":
        st.warning(f"**HISTORICAL RECORD** · {name}: {r.get('reason', '')}")
    elif f == "DATED_STUDY":
        st.info(f"**Study as of {str(r.get('generated_at') or '?')[:10]}** · {name} — evidence of that date, not live.")
    else:
        st.caption(f"✅ {name}: fresh (generated {r.get('generated_at')}, limit {r.get('max_age_hours')} h)")
    return r
