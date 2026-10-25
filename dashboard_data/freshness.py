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


def age_hours(ts) -> float | None:
    """Hours since a RECORDED timestamp (never file mtime). None when absent or unparseable."""
    from datetime import datetime, timezone
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 3600.0)
    except Exception:                                                 # noqa: BLE001
        return None


def recorded(st, rel: str, generated_at, max_age_hours: float, label: str | None = None,
             m: dict | None = None) -> str:
    """Freshness for an artifact: the manifest's verdict when the manifest tracks it, otherwise the
    age of the timestamp RECORDED inside the artifact against `max_age_hours`. An unknown age is
    treated as stale. Returns FRESH | STALE | HISTORICAL | DATED_STUDY | UNKNOWN."""
    m = m or manifest()
    if any(r.get("artifact") == rel for r in m.get("artifacts", [])):
        r = banner(st, rel, label, m)
        f = r.get("freshness") or "UNKNOWN"
        return "STALE" if f in ("STALE", "UNKNOWN", "MISSING") else f
    name = label or rel
    a = age_hours(generated_at)
    if a is None or a > max_age_hours:
        why = "no recorded timestamp" if a is None else f"{a:.1f} h old, limit {max_age_hours:g} h"
        st.error(f"**STALE — NOT VALID FOR CURRENT DECISIONS** · {name}: {why}")
        return "STALE" if a is not None else "UNKNOWN"
    st.caption(f"✅ {name}: fresh (generated {generated_at}, {a:.1f} h old, limit {max_age_hours:g} h; "
               f"age from the artifact's own timestamp)")
    return "FRESH"


# ── header chips (visual redesign stage 3, 2026-10-25) ──────────────────────────────────────────────
# The SAME verdicts as banner() / recorded(), returned as one chip for ui.page_header(freshness=...) so a
# page states its data age once, beside its title. A STALE / UNKNOWN / MISSING artifact still says
# NOT VALID FOR DECISIONS in the chip itself; pages that need the long explanation keep banner().
def header_badge(rel: str, label: str | None = None, m: dict | None = None) -> tuple[str, str]:
    """(state, markdown chip) for a manifest-tracked artifact. state: FRESH | STALE | HISTORICAL | DATED_STUDY."""
    import dashboard_ui as ui
    r = status(rel, m)
    f = r.get("freshness") or "UNKNOWN"
    name = label or rel
    gen = str(r.get("generated_at") or "?")[:16].replace("T", " ")
    if f in ("STALE", "UNKNOWN", "MISSING"):
        return "STALE", ui.freshness_badge("STALE", f"{name} · NOT VALID FOR DECISIONS")
    if f == "HISTORICAL":
        return f, ui.freshness_badge("HISTORICAL", name)
    if f == "DATED_STUDY":
        return f, ui.freshness_badge("DATED_STUDY", f"{name} · as of {gen[:10]}")
    lim = r.get("max_age_hours")
    return "FRESH", ui.freshness_badge("FRESH", f"{name} · generated {gen} UTC" + (f" · limit {lim} h" if lim else ""))


def ts_badge(ts, cadence: str = "daily", label: str = "") -> tuple[str, str]:
    """(state, chip) from a timestamp recorded INSIDE the data (never file mtime); unknown age = UNKNOWN."""
    import dashboard_ui as ui
    from dashboard_data.core import freshness_from_timestamp
    f = freshness_from_timestamp(ts, cadence)
    return f.state, ui.freshness_badge(f.state, " · ".join(x for x in (label, f.label) if x))


def recorded_badge(rel: str, generated_at, max_age_hours: float, label: str | None = None,
                   m: dict | None = None) -> tuple[str, str]:
    """(state, chip) with EXACTLY recorded()'s verdict, without rendering anything. A page shows the chip
    beside its title and calls recorded() itself only when the state is not FRESH (the long banner)."""
    import dashboard_ui as ui
    m = m or manifest()
    if any(r.get("artifact") == rel for r in m.get("artifacts", [])):
        st_, chip = header_badge(rel, label, m)
        return st_, chip
    name = label or rel
    a = age_hours(generated_at)
    if a is None or a > max_age_hours:
        why = "no recorded timestamp" if a is None else f"{a:.1f} h old, limit {max_age_hours:g} h"
        return ("STALE" if a is not None else "UNKNOWN"), ui.freshness_badge(
            "STALE", f"{name} · NOT VALID FOR DECISIONS · {why}")
    return "FRESH", ui.freshness_badge("FRESH", f"{name} · generated {generated_at} · {a:.1f} h old, "
                                                f"limit {max_age_hours:g} h")
