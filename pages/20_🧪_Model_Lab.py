"""Model Lab — who serves, who is waiting, and what the canonical promotion gate decided.

Owner architecture brief 2026-10-25, s.30. A VIEWER of small registry/lineage files:
  * models/registry/current_champions.json         the serving champion per model (the pointer)
  * models/registry/<type>/challenger.json         the challenger slot (metadata only, no model file)
  * output/model_lineage/promotion_decisions.csv   PG-1.0 decisions (team models)
  * output/model_lineage/props_promotion_decisions.csv  PG-PROPS-1.0 decisions (player props)
  * output/model_lineage/shadow_experiments_eval.json   H-MODEL-01 paired prospective comparison
  * output/retrain_log.json                        what the latest retrain logged

RETRAINING IS FREQUENT, PROMOTION IS RARE. Only src/promotion_gate.py (PG-1.0) and
player_model/props_gate.py (PG-PROPS-1.0) may change a serving model, and only after TWO independent
periods against the serving champion, per track: the untouched holdout, then >= 1,000 settled
prospective rows logged after the challenger was frozen. This page shows their records; it decides
nothing and loads no model.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - admin, like Model Performance

from dashboard_data import freshness, model_lab  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
freshness = importlib.reload(freshness)
model_lab = importlib.reload(model_lab)

st.set_page_config(page_title="Wowza | Model Lab", page_icon="🧪", layout="wide")
import dashboard_ui as ui  # noqa: E402

ui.page_header("Model Lab", "Who serves, who is waiting, and what the canonical promotion gate decided. Display "
               "only — this page decides nothing and loads no model.", eyebrow_text="Research Lab · model lineage",
               freshness=ui.status_chip("SHADOW", label="CHALLENGERS ARE SHADOW · NEVER SERVED"))


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    pe, pe_at = model_lab.prospective()
    rk, rt = model_lab.latest_retrain()
    return {"champ": model_lab.champions(), "slots": model_lab.challenger_slots(),
            "dec": model_lab.decisions(), "props_dec": model_lab.decisions(model_lab.PROPS_DECISIONS),
            "gate": model_lab.gate_config(), "pe": pe, "pe_at": pe_at, "rk": rk, "rt": rt}


d = _load()
g = d["gate"]

with st.container(border=True):
    st.markdown("**Retraining is frequent. Promotion is rare.**")
    st.markdown(
        f"A retrain produces a *challenger*, never a champion. Only the canonical gate — **{g['team'] or 'PG-1.0'}** "
        f"for team models, **{g['props'] or 'PG-PROPS-1.0'}** for player props — may change what serves, and only "
        f"after **two independent periods**, both against the champion actually serving, per track:\n"
        f"1. **Holdout** — untouched chronological holdout, n ≥ {g['holdout_min_n'] or 1000} per track, log-loss "
        f"improvement ≥ {g['material_ll'] or 0.001} with a CI excluding zero, Brier/calibration/canary/league checks. "
        f"Passing makes it QUALIFIED and parks it in the challenger slot (SHADOW_MORE) — not promoted.\n"
        f"2. **Prospective** — n ≥ {g['prospective_min_n'] or 1000} settled first-sight rows per track logged after "
        f"the freeze, log-loss delta < 0, Brier non-inferior. Only then PROMOTE "
        f"(slot expires after {g['max_age_days'] or 180} days).")
    st.caption("Prospective volume is ~100–200 settled rows per track per week, so a promotion takes months. "
               "That is the intended rate.")

# ── champions ──────────────────────────────────────────────────────────────────────────────────
ui.section_header("Serving champions")
if d["champ"].empty:
    st.error("models/registry/current_champions.json is missing or unreadable.")
else:
    st.dataframe(d["champ"], hide_index=True, width="stretch",
                 column_config={"promotion_reason": st.column_config.TextColumn(width="large")})
    st.caption("`current_champions.json` is the serving pointer. LEGACY_GIT = reconstructed from git history; "
               "ROLLBACK / BOOTSTRAP are explicit, logged operations.")

# ── challenger slots ───────────────────────────────────────────────────────────────────────────
ui.section_header("Challenger slots")
if d["slots"].empty:
    st.info("No challenger is parked in any slot.")
else:
    st.dataframe(d["slots"], hide_index=True, width="stretch")
    st.caption("A parked challenger is SHADOW: never served, tiered or staked. Its holdout numbers are period 1 only.")

# ── gate decisions ─────────────────────────────────────────────────────────────────────────────
ui.section_header("Promotion decisions")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown(f"**Team models — {g['team'] or 'PG-1.0'}**")
        if d["dec"].empty:
            st.warning("output/model_lineage/promotion_decisions.csv missing or empty — no gate decision recorded.")
        else:
            st.metric("Decisions", len(d["dec"]),
                      help="Every PG-1.0 decision, with gate version and metrics, appended by src/promotion_gate.py")
            st.caption(f"Last: {d['dec']['decided_at'].iloc[0]} · verdicts: "
                       + ", ".join(f"{k} {v}" for k, v in d["dec"]["verdict"].value_counts().items()))
with c2:
    with st.container(border=True):
        st.markdown(f"**Player props — {g['props'] or 'PG-PROPS-1.0'}**")
        if d["props_dec"].empty:
            st.info("No props gate decision recorded yet (output/model_lineage/props_promotion_decisions.csv "
                    "is written by the first props retrain under PG-PROPS-1.0). Props stay paper-only regardless.")
        else:
            st.metric("Decisions", len(d["props_dec"]))
            st.caption(f"Last: {d['props_dec']['decided_at'].iloc[0]}")
if not d["dec"].empty:
    st.dataframe(d["dec"], hide_index=True, width="stretch",
                 column_config={"reason": st.column_config.TextColumn(width="large")})
if not d["props_dec"].empty:
    with st.expander("Props gate decisions"):
        st.dataframe(d["props_dec"], hide_index=True, width="stretch")

# ── prospective comparison ─────────────────────────────────────────────────────────────────────
ui.section_header("Prospective samples — H-MODEL-01 (challenger vs serving champion)")
freshness.recorded(st, "output/model_lineage/shadow_experiments_eval.json", d["pe_at"], 48, "shadow experiment eval")
pe = d["pe"]
if pe.empty:
    st.info("No prospective evaluation recorded yet.")
else:
    cols = st.columns(len(pe))
    for col, (_, r) in zip(cols, pe.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{r['track']}**")
                st.metric("Settled paired rows", f"{r['settled_paired_n']:,} / {r['required_n']:,}")
                st.progress(float(r["progress"]))
                st.caption(f"Δ log loss {r['delta_log_loss']:+.4f} · Δ Brier {r['delta_brier']:+.4f} "
                           f"(negative = challenger better)")
                if not r["evaluable"]:
                    st.caption(f"⏳ Below the {r['required_n']:,}-row per-track minimum: **not evidence yet**, "
                               "shown for monitoring only.")
    st.dataframe(pe.drop(columns=["progress"]), hide_index=True, width="stretch")
    st.caption("Shadow only: the challenger is never served. The minimum applies to EACH track separately "
               "(owner 2026-10-25); the window starts at registration.")

# ── latest retrain ─────────────────────────────────────────────────────────────────────────────
ui.section_header(f"Latest retrain — {d['rk'] or 'none recorded'}")
if d["rt"].empty:
    st.info("output/retrain_log.json has no runs.")
else:
    st.dataframe(d["rt"], hide_index=True, width="stretch",
                 column_config={"why": st.column_config.TextColumn(width="large")})
    st.caption("These are the verdicts the retrain LOGGED. A logged PROMOTE is not the serving state: the serving "
               "model is whatever current_champions.json points to, and any later gate action (e.g. an owner "
               "ROLLBACK) is in the decisions table above.")
