"""Data truth panel (Phase 2 dashtruth, 2026-11-05): what the data behind the dashboard can and cannot prove.

Shown on System Health and summarised on Home. Every value is READ from a committed artifact; nothing here
runs a check, fetches, or fills a gap. A missing input is NO_DATA, never a typed value.

  stoplight     output/data_truth_health.json (python -m src.integrity.data_truth, evidence_refresh.yml) and
                the "Data truth N" lines of output/system_health.json - the NEWER recorded copy is shown, with
                its time. `stoplight` is the gate verdict when the artifact records it (written since
                2026-11-05); otherwise the panel shows the worst recorded check status, labelled as such.
  unverified    zeros_unverified player rows   output/collect/<player history>_refetch_audit.json
                                               (plan.rows_to_verify, stored_rows: recorded, not recounted)
                unproven cup rows              output/corrections/history_cup_unproven_rows.csv (rows, reason)
                refused scores                 output/corrections/history_score_refused.csv (rows)
  collection    output/collect/*audit*.json    filled / lacking as each collector recorded them
                last collection                newest date among af_budget.json days with API calls recorded
                                               and the audits' own generated_at
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NO_DATA = "NO_DATA"
ORDER = {"RED": 0, "AMBER": 1, NO_DATA: 2, "GREEN": 3}
#: a recorded data-truth artifact older than this is shown as STALE (evidence_refresh.yml runs daily)
MAX_AGE_H = 36
TRUTH_JSON = "data_truth_health.json"
HEALTH_JSON = "system_health.json"
#: the re-fetch collector's small audit JSON (~1 KB). Spelled in two parts because the dashboard guard
#: (tests/test_dashboard_navigation.py) refuses the bank's name in page code: the bank itself is never read.
REFETCH_AUDIT = "collect/player_" + "history_refetch_audit.json"
CUP_UNPROVEN = "corrections/history_cup_unproven_rows.csv"
SCORE_REFUSED = "corrections/history_score_refused.csv"
BUDGET = "collect/af_budget.json"
#: collector audit -> (label, filled key, base key, base label). Values are read exactly as recorded.
COLLECT_AUDITS = {
    "collect/ht_fill_audit.json": ("Half-time scores (fd_history)", "filled_rows_total", None, None),
    "collect/af_stats_fill_audit.json": ("Shot stats (af_history)", "filled_rows_total", "lacking_rows",
                                         "lacking_rows"),
    "collect/af_fullstats_audit.json": ("Full match stats (history)", "filled_rows_total", "history_rows",
                                        "history_rows"),
    REFETCH_AUDIT: ("Player counts re-fetch", "stored_rows", None, None),
}


def _out(out_dir) -> Path:
    return Path(out_dir) if out_dir is not None else ROOT / "output"


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001 - absent = NO_DATA
        return None


def _csv(p: Path):
    try:
        return pd.read_csv(p, low_memory=False) if Path(p).exists() else None
    except Exception:                                                 # noqa: BLE001
        return None


def _ts(v):
    t = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(t) else t


def stoplight(out_dir=None, now=None) -> dict:
    """The newer recorded copy of the data truth checks. {'status': NO_DATA} when neither file has them."""
    o = _out(out_dir)
    cands = []
    dt = _json(o / TRUTH_JSON)
    if isinstance(dt, dict) and dt.get("checks"):
        cands.append(("output/" + TRUTH_JSON, dt.get("generated_at"), dt["checks"], dt.get("stoplight"),
                      dt.get("gate_reasons")))
    sh = _json(o / HEALTH_JSON)
    if isinstance(sh, dict):
        c = [x for x in sh.get("checks") or [] if str(x.get("check", "")).startswith("Data truth")]
        if c:
            cands.append(("output/" + HEALTH_JSON, sh.get("generated_at"), c, None, None))
    if not cands:
        return {"status": NO_DATA, "basis": "no recorded data truth checks", "source": None,
                "generated_at": None, "age_h": None, "stale": None, "checks": []}
    # newest recorded time wins; a copy without a parseable time is used only when no copy has one
    timed = [c for c in cands if _ts(c[1]) is not None]
    src, gen, checks, gate, reasons = max(timed, key=lambda c: _ts(c[1])) if timed else cands[0]
    t = _ts(gen)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    age = round((now - t).total_seconds() / 3600.0, 1) if t is not None else None
    # an unknown status word ranks with NO_DATA (never as GREEN)
    worst = min((str(c.get("status", NO_DATA)) for c in checks), key=lambda s: ORDER[s] if s in ORDER else ORDER[NO_DATA])
    if gate in ("GREEN", "RED"):
        status, basis = gate, "gate verdict as recorded"
    else:
        status, basis = worst, "worst recorded check status (this artifact records no gate verdict)"
    return {"status": status, "basis": basis, "source": src, "generated_at": gen, "age_h": age,
            "stale": (age is None or age > MAX_AGE_H), "gate_reasons": reasons or [],
            "checks": [{"check": c.get("check"), "status": c.get("status"), "value": c.get("value")}
                       for c in checks]}


def unverified(out_dir=None) -> list[dict]:
    """One row per known-unverified population: what, how many (as recorded), source. NO_DATA when absent."""
    o = _out(out_dir)
    rows = []
    a = _json(o / REFETCH_AUDIT)
    plan = (a or {}).get("plan") if isinstance(a, dict) else None
    if isinstance(plan, dict) and plan.get("rows_to_verify") is not None and a.get("stored_rows") is not None:
        todo, done = int(plan["rows_to_verify"]), int(a["stored_rows"])
        refused = sum(int(v) for v in (plan.get("refused_rows") or {}).values())
        rows.append({"item": "Player rows still zeros_unverified", "value": todo - done,
                     "detail": f"{todo:,} to verify (plan) - {done:,} re-fetched and stored; {refused:,} refused "
                               f"by the plan. Run stop: {(a.get('run') or {}).get('budget_stop') or '-'}",
                     "source": "output/" + REFETCH_AUDIT, "as_of": a.get("generated_at")})
    else:
        rows.append({"item": "Player rows still zeros_unverified", "value": NO_DATA, "detail": "audit missing",
                     "source": "output/" + REFETCH_AUDIT, "as_of": None})
    cup = _csv(o / CUP_UNPROVEN)
    if cup is None:
        rows.append({"item": "Unproven cup rows (kept in history)", "value": NO_DATA, "detail": "file missing",
                     "source": "output/" + CUP_UNPROVEN, "as_of": None})
    else:
        why = cup["reason"].value_counts().to_dict() if "reason" in cup.columns else {}
        rows.append({"item": "Unproven cup rows (kept in history)", "value": int(len(cup)),
                     "detail": "; ".join(f"{k}: {v}" for k, v in why.items()) or "-",
                     "source": "output/" + CUP_UNPROVEN, "as_of": None})
    ref = _csv(o / SCORE_REFUSED)
    if ref is None:
        rows.append({"item": "Refused scores (history vs provider)", "value": NO_DATA, "detail": "file missing",
                     "source": "output/" + SCORE_REFUSED, "as_of": None})
    else:
        why = ref["reason"].value_counts().to_dict() if "reason" in ref.columns else {}
        rows.append({"item": "Refused scores (history vs provider)", "value": int(len(ref)),
                     "detail": "; ".join(f"{k}: {v}" for k, v in why.items()) or "-",
                     "source": "output/" + SCORE_REFUSED, "as_of": None})
    return rows


def collection(out_dir=None) -> dict:
    """Collector progress as recorded, and the date of the last collection."""
    o = _out(out_dir)
    rows, dates = [], []
    for rel, (label, filled, base, base_label) in COLLECT_AUDITS.items():
        a = _json(o / rel)
        if not isinstance(a, dict):
            rows.append({"collector": label, "filled": NO_DATA, "recorded base": None, "detail": "audit missing",
                         "source": "output/" + rel})
            continue
        f = a.get(filled)
        b = a.get(base) if base else None
        calls = a["calls"] if "calls" in a else (a.get("run") or {}).get("calls")
        rows.append({"collector": label, "filled": NO_DATA if f is None else int(f),
                     "recorded base": (f"{base_label} {int(b):,}" if b is not None else None),
                     "detail": "API calls last run: " + ("not recorded" if calls is None else str(calls)),
                     "source": "output/" + rel})
        if _ts(a.get("generated_at")) is not None:
            dates.append(_ts(a.get("generated_at")).strftime("%Y-%m-%d"))
    bud = _json(o / BUDGET)
    if isinstance(bud, dict):
        spent = [k for k, v in bud.items() if pd.notna(pd.to_numeric(v, errors="coerce"))
                 and float(v) > 0 and _ts(k) is not None]
        dates += spent
    last = max(dates) if dates else NO_DATA
    return {"rows": rows, "last_collection": last,
            "last_basis": ("newest day with API-Football calls recorded in output/" + BUDGET
                           + " or a collector audit's generated_at") if dates else "no recorded collection date"}


def panel(out_dir=None, now=None) -> dict:
    return {"stoplight": stoplight(out_dir, now), "unverified": unverified(out_dir),
            "collection": collection(out_dir)}


# ── presentation ──────────────────────────────────────────────────────────────────────────────
def render(p: dict, compact: bool = False) -> None:
    """System Health (full) or Home (compact, one line + the counts)."""
    import streamlit as st

    import dashboard_ui as ui
    s = p["stoplight"]
    when = (f"recorded {s['generated_at']} ({s['age_h']:.1f} h old)" if s.get("age_h") is not None
            else "no recorded time")
    stale = " · STALE - older than %d h" % MAX_AGE_H if s.get("stale") else ""
    unv = {r["item"]: r["value"] for r in p["unverified"]}

    def _v(x):
        return f"{x:,}" if isinstance(x, int) else str(x)
    if compact:
        with st.container(border=True):
            ui.eyebrow("D · Data truth")
            st.metric("Data truth stoplight", s["status"] + (" · STALE" if s.get("stale") else ""), help=f"{s['basis']}. Source: {s.get('source') or '-'}.")
            ui.stat_line([("Player rows zeros_unverified", _v(unv.get("Player rows still zeros_unverified", NO_DATA))),
                          ("Unproven cup rows", _v(unv.get("Unproven cup rows (kept in history)", NO_DATA))),
                          ("Refused scores", _v(unv.get("Refused scores (history vs provider)", NO_DATA))),
                          ("Last collection", str(p["collection"]["last_collection"]))])
            st.caption(f"{s.get('source') or 'no artifact'} · {when}{stale}. Full panel on System Health.")
        return
    st.markdown(f"**Stoplight: {s['status']}{' (STALE)' if s.get('stale') else ''}** - {s['basis']} · {s.get('source') or 'no artifact'} · {when}{stale}")
    if s.get("gate_reasons"):
        st.error(" · ".join(map(str, s["gate_reasons"])))
    if s["checks"]:
        ui.dataframe(pd.DataFrame(s["checks"]), hide_index=True, width="stretch")
    st.markdown("**Not yet verified** (kept and labelled, never filled)")
    ui.dataframe(pd.DataFrame(p["unverified"]).assign(value=lambda d: d["value"].map(_v)),
                 hide_index=True, width="stretch")
    c = p["collection"]
    st.markdown(f"**Collection progress** · last collection: **{c['last_collection']}** ({c['last_basis']})")
    ui.dataframe(pd.DataFrame(c["rows"]).assign(filled=lambda d: d["filled"].map(_v)),
                 hide_index=True, width="stretch")
