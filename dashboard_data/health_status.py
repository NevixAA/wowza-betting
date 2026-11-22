"""Four independent health indicators (owner P0 deep-repair item 2, 2026-11-22). Never one GREEN for all four.

Until 2026-11-22 the dashboard showed ONE "Data truth stoplight" that preferred the CI gate verdict over the
worst check: output/data_truth_health.json said GREEN while its own "Data truth 3" check was RED and
output/system_health.json was FAIL. A gate answers "may this commit merge?"; it does not say the data is
clean. Each question now gets its own indicator, read from committed artifacts only (nothing is recomputed
or filled; a missing artifact is NO_DATA, never a pass):

  ci_gate                     CI / change control: the data truth --gate verdict (gated rules + the
                              known-defaults ratchet) AND the research lock (protected_tracks_modified).
                              GREEN | RED | NO_DATA.
  operational_data_quality    the WORST current data truth check, gated or not (newest recorded copy of
                              output/data_truth_health.json or the "Data truth" lines of system_health.json).
                              GREEN | AMBER | RED | NO_DATA.
  production_model_integrity  output/model_integrity_health.json: every active serving track (train/serve
                              parity, chronology, holdout, batch invariance, active serving joins, training
                              lineage). The legacy join replay is shown beside it and does not count.
                              PASS | WARN | FAIL | NO_DATA.
  full_system_certification   CERTIFIED only when every REQUIRED component has CURRENT evidence (artifact
                              present, generated_at parseable and <= MAX_AGE_H old) and passes. Otherwise
                              NOT CERTIFIED with each missing / stale / failing component listed.

Every failing item carries {component, status, detail, source, generated_at}.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NO_DATA = "NO_DATA"
MAX_AGE_H = 36                                        # the daily evidence / integrity jobs; GitHub cron runs late
TRUTH_JSON, HEALTH_JSON = "data_truth_health.json", "system_health.json"
MODEL_JSON, JOIN_JSON, PARITY_JSON = "model_integrity_health.json", "join_integrity_health.json", "feature_parity_health.json"
_TRUTH_ORDER = {"RED": 0, "AMBER": 1, NO_DATA: 2, "GREEN": 3}
INDICATORS = ("ci_gate", "operational_data_quality", "production_model_integrity", "full_system_certification")
LABELS = {"ci_gate": "CI / change-control gate", "operational_data_quality": "Operational data quality",
          "production_model_integrity": "Production model integrity",
          "full_system_certification": "Full-system certification"}
CERTIFIED, NOT_CERTIFIED = "CERTIFIED", "NOT CERTIFIED"


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001 - absent = NO_DATA
        return None


def _ts(v):
    t = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(t) else t


def _now(now):
    n = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return n.tz_localize("UTC") if n.tzinfo is None else n


def _age(gen, now):
    t = _ts(gen)
    return None if t is None else round((now - t).total_seconds() / 3600.0, 1)


def _item(component, status, detail, source, generated_at, **extra):
    return {"component": component, "status": status, "detail": detail, "source": source,
            "generated_at": generated_at, **extra}


def load_docs(out_dir=None) -> dict:
    o = Path(out_dir) if out_dir is not None else ROOT / "output"
    return {n: _json(o / n) for n in (TRUTH_JSON, HEALTH_JSON, MODEL_JSON, JOIN_JSON, PARITY_JSON)}


# ── 1 CI / change-control gate ────────────────────────────────────────────────────────────────
def ci_gate(docs: dict, now=None) -> dict:
    now = _now(now)
    dt, sh = docs.get(TRUTH_JSON), docs.get(HEALTH_JSON)
    cands = []
    if isinstance(dt, dict) and dt.get("stoplight") in ("GREEN", "RED"):
        cands.append(("output/" + TRUTH_JSON, dt.get("generated_at"), dt["stoplight"], dt.get("gate_reasons") or []))
    g = (sh or {}).get("data_truth_gate") if isinstance(sh, dict) else None
    if isinstance(g, dict) and g.get("stoplight") in ("GREEN", "RED"):
        cands.append(("output/" + HEALTH_JSON, sh.get("generated_at"), g["stoplight"], g.get("gate_reasons") or []))
    failing = []
    if not cands:
        gate, src, gen = NO_DATA, None, None
        failing.append(_item("data_truth --gate", NO_DATA, "no recorded gate verdict", "output/" + TRUTH_JSON, None))
    else:
        timed = [c for c in cands if _ts(c[1]) is not None]
        src, gen, gate, reasons = max(timed, key=lambda c: _ts(c[1])) if timed else cands[0]
        failing += [_item("data_truth --gate", "RED", str(r), src, gen) for r in reasons]
        if gate == "RED" and not reasons:
            failing.append(_item("data_truth --gate", "RED", "gate RED (no reason recorded)", src, gen))
    lock = (sh or {}).get("research_lock") if isinstance(sh, dict) else None
    if not isinstance(sh, dict) or "protected_tracks_modified" not in sh:
        lock_status = NO_DATA
        failing.append(_item("research lock", NO_DATA, "no recorded research lock", "output/" + HEALTH_JSON, None))
    elif sh.get("protected_tracks_modified") or (lock or {}).get("error"):
        lock_status = "RED"
        failing.append(_item("research lock", "RED", (lock or {}).get("error") or
                             "protected_tracks_modified=true: production moved without re-approval",
                             "output/" + HEALTH_JSON, sh.get("generated_at")))
    else:
        lock_status = "GREEN"
    parts = [gate, lock_status]
    status = "RED" if "RED" in parts else NO_DATA if NO_DATA in parts else "GREEN"
    return {"indicator": "ci_gate", "label": LABELS["ci_gate"], "status": status,
            "meaning": "may a commit merge: the data truth --gate (gated rules + ratchet) and the research lock. "
                       "It does NOT say the current data is clean.",
            "parts": {"data_truth_gate": gate, "research_lock": lock_status},
            "source": src, "generated_at": gen, "age_h": _age(gen, now), "failing": failing}


# ── 2 operational data quality ────────────────────────────────────────────────────────────────
def truth_checks(docs: dict):
    """(source, generated_at, checks) of the NEWER recorded copy of the data truth checks, or None."""
    cands = []
    dt, sh = docs.get(TRUTH_JSON), docs.get(HEALTH_JSON)
    if isinstance(dt, dict) and dt.get("checks"):
        cands.append(("output/" + TRUTH_JSON, dt.get("generated_at"), dt["checks"]))
    if isinstance(sh, dict):
        c = [x for x in sh.get("checks") or [] if str(x.get("check", "")).startswith("Data truth")]
        if c:
            cands.append(("output/" + HEALTH_JSON, sh.get("generated_at"), c))
    if not cands:
        return None
    timed = [c for c in cands if _ts(c[1]) is not None]
    return max(timed, key=lambda c: _ts(c[1])) if timed else cands[0]


def worst_truth(statuses) -> str:
    s = [str(x) for x in statuses]
    # an unknown status word ranks with NO_DATA, never as GREEN
    return min(s, key=lambda x: _TRUTH_ORDER.get(x, _TRUTH_ORDER[NO_DATA])) if s else NO_DATA


def _gated(check: str) -> bool | None:
    try:
        from src.integrity.data_truth import is_gated
        return bool(is_gated(check))
    except Exception:                                                 # noqa: BLE001 - slim tree
        return None


def operational(docs: dict, now=None) -> dict:
    now = _now(now)
    got = truth_checks(docs)
    if got is None:
        return {"indicator": "operational_data_quality", "label": LABELS["operational_data_quality"],
                "status": NO_DATA, "meaning": "worst current data truth check, gated or not",
                "source": None, "generated_at": None, "age_h": None,
                "failing": [_item("data truth checks", NO_DATA, "no recorded data truth checks",
                                  "output/" + TRUTH_JSON, None)]}
    src, gen, checks = got
    status = worst_truth(c.get("status", NO_DATA) for c in checks)
    failing = [_item(c.get("check"), str(c.get("status", NO_DATA)), str(c.get("value", "")), src, gen,
                     gated=_gated(str(c.get("check", ""))))
               for c in sorted(checks, key=lambda c: _TRUTH_ORDER.get(str(c.get("status")), 2))
               if str(c.get("status")) != "GREEN"]
    return {"indicator": "operational_data_quality", "label": LABELS["operational_data_quality"], "status": status,
            "meaning": "the WORST current data truth check, gated or not (an ungated RED is RED here)",
            "source": src, "generated_at": gen, "age_h": _age(gen, now), "failing": failing}


# ── 3 production model integrity ──────────────────────────────────────────────────────────────
def model_integrity(docs: dict, now=None) -> dict:
    now = _now(now)
    mi = docs.get(MODEL_JSON)
    src = "output/" + MODEL_JSON
    if not isinstance(mi, dict) or not mi.get("status"):
        return {"indicator": "production_model_integrity", "label": LABELS["production_model_integrity"],
                "status": NO_DATA, "meaning": "active serving tracks: parity, chronology, joins, lineage",
                "source": src, "generated_at": None, "age_h": None, "scopes": {},
                "failing": [_item("model integrity", NO_DATA, "artifact missing", src, None)]}
    gen = mi.get("generated_at")
    status = str(mi.get("overall_model_integrity_status") or mi.get("status"))
    status = status if status in ("PASS", "WARN", "FAIL") else NO_DATA
    scopes = {k: mi.get(k, NO_DATA) for k in ("active_serving_join_status", "active_training_lineage_status",
                                               "legacy_replay_status")}
    failing = []
    for k, t in (mi.get("tracks") or {}).items():
        if t.get("status") == "PASS":
            continue
        bad = {n: (t.get(n) or {}).get("status") for n in
               ("chronology", "parity", "batch_invariance", "weight_alignment", "holdout_independence",
                "price_provenance", "join_integrity", "training_lineage")
               if isinstance(t.get(n), dict) and (t.get(n) or {}).get("status") not in (None, "PASS")}
        fails = (t.get("parity") or {}).get("fail_features") or []
        detail = ", ".join(f"{n} {s}" for n, s in bad.items()) or t.get("note", "")
        if fails:
            detail += f" (parity FAIL features: {', '.join(map(str, fails))})"
        failing.append(_item(f"model {k}", str(t.get("status")), detail, src, gen))
    for k in ("active_serving_join_status", "active_training_lineage_status"):
        if scopes[k] not in ("PASS",):
            failing.append(_item(k, str(scopes[k]), "scope not PASS" if scopes[k] != NO_DATA else
                                 "not recorded by this artifact version", src, gen))
    return {"indicator": "production_model_integrity", "label": LABELS["production_model_integrity"],
            "status": status,
            "meaning": "active serving tracks (train/serve parity, chronology, holdout, active serving joins, "
                       "training lineage). legacy_replay_status is historical evidence and does not count.",
            "scopes": scopes, "source": src, "generated_at": gen, "age_h": _age(gen, now),
            "failing": sorted(failing, key=lambda x: 0 if x["status"] == "FAIL" else 1 if x["status"] in (NO_DATA, "UNVERIFIED")
                              else 2)}


# ── 4 full-system certification ───────────────────────────────────────────────────────────────
#: component -> (artifact, how its pass is read)
REQUIRED = (
    ("system health", HEALTH_JSON, lambda d: str(d.get("overall")), ("PASS",)),
    ("data truth (operational)", TRUTH_JSON,
     lambda d: worst_truth(c.get("status", NO_DATA) for c in d.get("checks") or []) if d.get("checks") else NO_DATA,
     ("GREEN",)),
    ("data truth gate", TRUTH_JSON, lambda d: str(d.get("stoplight", NO_DATA)), ("GREEN",)),
    ("model integrity", MODEL_JSON, lambda d: str(d.get("overall_model_integrity_status") or d.get("status")), ("PASS",)),
    ("active serving joins", JOIN_JSON,
     lambda d: str(d.get("active_serving_join_status") or _join_active(d)), ("PASS",)),
    ("train/serve feature parity", PARITY_JSON, lambda d: str(d.get("status")), ("PASS",)),
    ("training lineage", MODEL_JSON, lambda d: str(d.get("active_training_lineage_status", NO_DATA)), ("PASS",)),
)


def _join_active(d: dict) -> str:
    try:
        from src.integrity.training_lineage import join_scopes
        return join_scopes(d)["active_serving_join_status"]
    except Exception:                                                 # noqa: BLE001 - slim tree
        return NO_DATA


def certification(docs: dict, now=None, max_age_h: float = MAX_AGE_H) -> dict:
    now = _now(now)
    failing, components = [], []
    for name, art, read, ok in REQUIRED:
        d = docs.get(art)
        src = "output/" + art
        if not isinstance(d, dict):
            st_, why, gen, age = NO_DATA, "evidence missing", None, None
        else:
            gen = d.get("generated_at")
            age = _age(gen, now)
            st_ = read(d)
            if age is None:
                why = "evidence has no parseable generated_at"
            elif age > max_age_h:
                why = f"evidence stale: {age:.1f} h old (limit {max_age_h} h)"
            elif st_ not in ok:
                why = f"status {st_} (needs {'/'.join(ok)})"
            else:
                why = ""
        c = {"component": name, "status": st_, "source": src, "generated_at": gen, "age_h": age, "ok": not why}
        components.append(c)
        if why:
            failing.append(_item(name, st_, why, src, gen))
    status = CERTIFIED if not failing else NOT_CERTIFIED
    return {"indicator": "full_system_certification", "label": LABELS["full_system_certification"],
            "status": status,
            "meaning": f"every required component has current evidence (<= {max_age_h} h) and passes",
            "components": components, "source": None, "generated_at": None, "age_h": None, "failing": failing}


def indicators(out_dir=None, now=None, docs: dict | None = None) -> dict:
    """The four indicators. `docs` overrides artifacts by file name (system_health.build_artifact passes the
    artifact it is writing)."""
    d = load_docs(out_dir)
    d.update(docs or {})
    return {"ci_gate": ci_gate(d, now), "operational_data_quality": operational(d, now),
            "production_model_integrity": model_integrity(d, now),
            "full_system_certification": certification(d, now)}


def all_failing(ind: dict) -> list[dict]:
    return [{"indicator": LABELS[k], **f} for k in INDICATORS for f in (ind.get(k) or {}).get("failing", [])]


# ── presentation ──────────────────────────────────────────────────────────────────────────────
_TONE = {"GREEN": "PASS", "PASS": "PASS", CERTIFIED: "PASS", "AMBER": "WARN", "WARN": "WARN",
         "RED": "FAIL", "FAIL": "FAIL", NOT_CERTIFIED: "FAIL", NO_DATA: "NO_DATA", "UNVERIFIED": "WARN"}


def render(ind: dict, compact: bool = False) -> None:
    """Four separate indicators (Home: compact; System Health: with every failing check)."""
    import streamlit as st

    import dashboard_ui as ui
    cols = st.columns(4)
    for col, k in zip(cols, INDICATORS):
        x = ind.get(k) or {}
        s = str(x.get("status", NO_DATA))
        when = f"{x['generated_at']} ({x['age_h']:.1f} h old)" if x.get("age_h") is not None else "-"
        col.metric(LABELS[k], s, help=f"{x.get('meaning', '')}. Source: {x.get('source') or 'several artifacts'} · {when}.")
        col.markdown(ui.status_chip(_TONE.get(s, "NO_DATA"), label=s))
    mi = ind.get("production_model_integrity") or {}
    sc = mi.get("scopes") or {}
    if sc:
        st.caption("Model integrity scopes - active serving joins: **%s** · active training lineage: **%s** · "
                   "legacy join replay (historical, not counted): **%s**"
                   % (sc.get("active_serving_join_status"), sc.get("active_training_lineage_status"),
                      sc.get("legacy_replay_status")))
    rows = all_failing(ind)
    if compact:
        st.caption(f"{len(rows)} failing / missing item(s) across the four indicators - listed on System Health.")
        return
    if rows:
        st.markdown("**Every failing or missing item** (component · status · source artifact · generated_at)")
        ui.dataframe(pd.DataFrame(rows)[["indicator", "component", "status", "detail", "source", "generated_at"]],
                     hide_index=True, width="stretch")
