"""System Health — is the estate fit to make decisions today? (owner architecture brief 2026-10-25, s.30)

A VIEWER of two small CI-written files:
  * output/system_health.json   (src/system_health.build_artifact, evidence_refresh.yml) — overall
    PASS / WARN / FAIL with its reasons, every check WITH the explicit rule that produced its status,
    the protected-track research lock as recorded by CI, the weekly research panel check, first-sight
    store integrity and data-contract violations;
  * output/artifact_manifest.json (src/artifact_freshness.py) — every decision-facing artifact and its
    freshness; anything STALE / UNKNOWN / MISSING is labelled NOT VALID FOR CURRENT DECISIONS.

Nothing here runs a check, calls git or re-derives a status: the research lock is shown exactly as
system_health.json recorded it. Freshness comes from recorded timestamps, never file mtime.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - admin, like Overview

from dashboard_data import freshness, system_view  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers
# so the page and its data modules are always the same version.
import importlib  # noqa: E402
freshness = importlib.reload(freshness)
system_view = importlib.reload(system_view)

st.set_page_config(page_title="Wowza | System Health", page_icon="🩺", layout="wide")
import dashboard_ui as ui  # noqa: E402

_SUB = ("Display only. Every status below was computed by CI and is shown with the rule that produced it; "
        "this page re-derives nothing.")


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    return system_view.load()


sh, man = _load()
if not sh:
    ui.page_header("System Health", _SUB, eyebrow_text="System · health", state="FAIL")
    st.error("**output/system_health.json is missing or unreadable** — health is UNKNOWN, which is treated "
             "as FAIL. evidence_refresh.yml writes it daily.")
    st.stop()

# ── overall verdict (visual redesign 2026-10-25, stage 3: header chips; values as CI recorded them) ─
overall = str(sh.get("overall", "UNKNOWN"))
_fstate, _fchip = freshness.recorded_badge("output/system_health.json", sh.get("generated_at"), 36, "system health", man)
ui.page_header("System Health", _SUB, eyebrow_text="System · health", state=overall, freshness=_fchip)
if _fstate != "FRESH":
    freshness.recorded(st, "output/system_health.json", sh.get("generated_at"), 36, "system health", man)

reasons = sh.get("reasons") or []
msg = f"**Overall: {overall}**" + ("" if not reasons else " — " + " · ".join(map(str, reasons)))
{"PASS": st.success, "WARN": st.warning}.get(overall, st.error)(msg)
st.caption(sh.get("about", ""))

cnt = sh.get("counts") or {}
ui.metric_grid([{"label": "GREEN", "value": cnt.get("GREEN", 0)}, {"label": "AMBER", "value": cnt.get("AMBER", 0)},
                {"label": "RED", "value": cnt.get("RED", 0)}, {"label": "NO_DATA", "value": cnt.get("NO_DATA", 0)},
                {"label": "Contract violations", "value": sh.get("contract_violations", 0)}], per_row=3)

v = sh.get("versions") or {}
sel = v.get("selection_engine_version") or {}
st.caption(f"Policy `{v.get('policy_version')}` · selection engine: production `{sel.get('production')}`, "
           f"shadow `{sel.get('shadow')}` · contracts `{v.get('contracts_version')}` · code `{v.get('code_commit')}`")

# ── protected tracks (research lock) ───────────────────────────────────────────────────────────
ui.section_header("Protected tracks — research lock")
lock = sh.get("research_lock") or {}
modified = sh.get("protected_tracks_modified")
if lock.get("error"):
    st.error(f"**Lock could not be verified by CI** (treated as FAIL): {lock['error']}")
elif modified:
    st.error("**PROTECTED TRACKS MODIFIED** — a production state, threshold, stake or eligibility of a "
             "protected track moved without owner approval. System health is FAIL until resolved.")
    if lock.get("changes"):
        st.dataframe(pd.DataFrame(lock["changes"]), hide_index=True, width="stretch")
else:
    st.success("🔒 protected_tracks_modified = **False** — protected tracks are unchanged since the owner-approved "
               "snapshot. Research may observe them (PROTECTED — SHADOW ONLY), never change them.")
if lock.get("unapproved_new_protected_tracks"):
    st.warning(f"Unapproved new protected tracks: {lock['unapproved_new_protected_tracks']}")
st.caption(f"Snapshot approved by {lock.get('approved_by', '?')} at {lock.get('approved_at', '?')} · "
           f"lock digest `{str(v.get('research_lock_digest') or '')[:16]}` · as recorded by CI in "
           f"system_health.json (this page does not run the lock check).")

# ── every check, with its rule ─────────────────────────────────────────────────────────────────
ui.section_header("Checks — status, value and the rule that set it")
st.dataframe(system_view.checks(sh), hide_index=True, width="stretch",
             column_config={"rule": st.column_config.TextColumn(width="large")})

# ── weekly research panel ──────────────────────────────────────────────────────────────────────
ui.section_header("Weekly research panel")
wk = system_view.check(sh, "Weekly research panel")
if wk:
    st.markdown(f"{system_view.STATUS_ICON.get(wk.get('status'), wk.get('status'))} — {wk.get('value', '')}")
    st.caption(f"Rule: {wk.get('rule', '')}")
    if wk.get("detail"):
        st.caption(str(wk["detail"]))
else:
    st.info("No weekly research panel check recorded in system_health.json.")

# ── artifact freshness ─────────────────────────────────────────────────────────────────────────
ui.section_header("Artifact freshness (manifest)")
mt = system_view.manifest_table(man)
if mt.empty:
    st.error("**output/artifact_manifest.json is missing or empty** — every artifact is UNKNOWN and treated as "
             "STALE — NOT VALID FOR CURRENT DECISIONS.")
else:
    n_stale = int(mt["verdict"].eq(system_view.STALE_LABEL).sum())
    (st.error if n_stale else st.success)(
        f"{n_stale} of {len(mt)} tracked artifacts are **{system_view.STALE_LABEL}**" if n_stale
        else f"All {len(mt)} tracked artifacts are fresh or labelled (HISTORICAL / DATED_STUDY).")
    st.caption(f"Manifest generated {man.get('generated_at')} · calculation {man.get('calculation_version')}. "
               "Ages come from timestamps recorded inside each artifact, never from file mtime.")
    st.dataframe(mt, hide_index=True, width="stretch")

# ── first-sight integrity ──────────────────────────────────────────────────────────────────────
ui.section_header("First-sight store integrity")
fs = system_view.first_sight(sh)
fsc = system_view.check(sh, "First-sight store integrity")
if fsc:
    st.caption(f"{system_view.STATUS_ICON.get(fsc.get('status'), fsc.get('status'))} {fsc.get('value', '')} · "
               f"rule: {fsc.get('rule', '')}")
if fs.empty:
    st.info("No first-sight stores recorded.")
else:
    st.dataframe(fs, hide_index=True, width="stretch")

# ── data contracts ─────────────────────────────────────────────────────────────────────────────
cv = system_view.contract_violations(sh)
with st.expander(f"Data-contract violations ({len(cv)})", expanded=False):
    if cv.empty:
        st.success("No contract violations recorded.")
    else:
        st.dataframe(cv, hide_index=True, width="stretch")
