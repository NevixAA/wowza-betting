"""Presentation for pages/7_🔬_Pro.py — nav label "Research" (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page reads exactly the Pro adapter calls it read before (dashboard_data.pro); this
module arranges them: the evidence strip at the top, one card per frozen forward test with its TRUE
FORWARD evidence badge, the Price Trader state as a chip, and the VALUABLE league x market heatmap in the
shared altair grammar (it was the one chart on the page drawn outside it).

SPEED, NOT VALUES. Over HTTP every Pro call is one or more sequential network round trips (raw files +
GitHub contents-API listings), which is why the page measured ~25-37 s cold. `load_all` makes the SAME
calls concurrently (threads overlap the waiting; every call still goes through dashboard_data.core's
disk cache and listing memo) and the page caches the result for five minutes. Same requests, same
answers, same objects.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

import dashboard_ui as ui
from src.tier_labels import TIER_LABEL

#: every Pro adapter the page reads (name -> attribute on dashboard_data.pro)
CALLS = ("system_contract", "candidate_forward", "valuable", "price_trader", "model_lineage", "research_index",
         "shadow_learning", "canonical_stores", "health_artifacts", "tip_scoreboard", "studies")


def load_all(pro, max_workers: int = 8) -> dict:
    """{name: pro.<name>()} for every call above, run concurrently. A failing call returns {} for that name
    only (exactly what the page did with a missing artifact), never an exception for the page."""
    from dashboard_data import overview_view
    try:
        overview_view.prefetch_pro_listings()          # canonical-store listings, concurrently (same memo)
    except Exception:                                  # noqa: BLE001 - a speed-up must never break the page
        pass

    def _one(name):
        fn = getattr(pro, name, None)
        if fn is None:
            return name, {}
        try:
            return name, fn()
        except Exception:                              # noqa: BLE001
            return name, {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return dict(ex.map(_one, CALLS))


# ── forward tests ────────────────────────────────────────────────────────────────────────────────
def forward_card(h: dict, container=None) -> None:
    """One frozen candidate gate: TRUE FORWARD n against its requirement is the main number; ROI only with its
    CI, matchdays and CLV coverage beside it; the historical discovery is shown as NOT COUNTED."""
    f = h.get("TRUE_FORWARD") or {}
    hist = h.get("HISTORICAL_DISCOVERY") or {}
    c = container or st
    with c.container(border=True):
        ui.eyebrow(f"{h['id']} · {h['league']} · {h['market'].upper()} · {h['model_type']}")
        st.metric("TRUE FORWARD n", f"{f.get('n', 0)} / {h.get('required_n')}")
        ui.stat_line([("Matchdays", f"{f.get('matchdays', 0)} / {h.get('required_matchdays')}"),
                      ("ROI", f"{f.get('roi_pct')}%"), ("ROI CI", f"{f.get('roi_ci95')}"),
                      ("CLV coverage", f"{f.get('clv_coverage_pct')}%"), ("CLV mean", f"{f.get('clv_mean')}"),
                      ("BH q", f"{f.get('q_bh')}")])
        st.markdown(ui.evidence_badge("TRUE FORWARD", f.get("n", 0), span=f"{f.get('matchdays', 0)} matchdays",
                                      state=str(h.get("status") or "")) + " "
                    + ui.sample_marker(f.get("n", 0) or 0))
        st.progress(min(1.0, float(h.get("progress_pct") or 0) / 100), text=f"{h.get('progress_pct')}% of required n")
        st.caption(f"gate **{h.get('candidate_gate')}** vs production {TIER_LABEL['SNIPER']} "
                   f"{(h.get('production_gates') or {}).get('sniper')} · status **{h.get('status')}** · "
                   f"HISTORICAL DISCOVERY fit {hist.get('fit_roi')} · validation {hist.get('validation_roi')} · "
                   f"holdout {hist.get('holdout_roi')} — post-discovery rows: shown in the study, NOT COUNTED")


# ── Price Trader ─────────────────────────────────────────────────────────────────────────────────
_TRADER_TONE = {"COLLECTING": ("ACCRUING", "WARNING"), "PRICE_SIGNAL": ("PASS", "POSITIVE"),
                "EXECUTABLE_EDGE": ("PASS", "POSITIVE"), "NO_EDGE": ("FAIL", "NEGATIVE")}


def trader_chip(state: str) -> str:
    key, tone = _TRADER_TONE.get(str(state), ("UNKNOWN", "NEUTRAL"))
    return ui.status_chip(key, label=str(state).replace("_", " "), tone=tone)


# ── VALUABLE heatmap ─────────────────────────────────────────────────────────────────────────────
def valuable_heat_frame(lm: pd.DataFrame) -> pd.DataFrame:
    """The cells the page drew before: league x market, N >= 10, shrunk ROI % (pivot mean, same as before)."""
    h = lm[lm.N >= 10].pivot_table(index="league", columns="market", values="ROI_shrunk_pct")
    if h.empty:
        return pd.DataFrame(columns=["league", "market", "ROI_shrunk_pct"])
    long = h.stack().rename("ROI_shrunk_pct").reset_index().dropna(subset=["ROI_shrunk_pct"])   # empty cell = no cell
    n = lm[lm.N >= 10].groupby(["league", "market"])["N"].sum().rename("N").reset_index()
    return long.merge(n, on=["league", "market"], how="left")


def valuable_heatmap(d: pd.DataFrame):
    """Diverging around zero on the polarity tokens, the value printed in every cell, n in the tooltip."""
    import altair as alt
    t = ui.tokens()
    pos, neg = ui.polarity()
    lim = float(max(abs(d["ROI_shrunk_pct"].min()), abs(d["ROI_shrunk_pct"].max()), 1.0))
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"], tickColor=t["BORDER"])
    base = alt.Chart(d).encode(x=alt.X("market:N", title=None, axis=alt.Axis(labelAngle=0, **axis)),
                               y=alt.Y("league:N", title=None, axis=alt.Axis(**axis)))
    rect = base.mark_rect().encode(
        color=alt.Color("ROI_shrunk_pct:Q", title="shrunk ROI %",
                        scale=alt.Scale(domain=[-lim, 0, lim], range=[neg, t["SURFACE_MUTED"], pos])),
        tooltip=[alt.Tooltip("league:N"), alt.Tooltip("market:N"),
                 alt.Tooltip("ROI_shrunk_pct:Q", title="shrunk ROI %", format="+.1f"),
                 alt.Tooltip("N:Q", title="n", format=",")])
    text = base.mark_text(color=t["TEXT_PRIMARY"]).encode(text=alt.Text("ROI_shrunk_pct:Q", format=".0f"))
    return (rect + text).properties(height=max(320, 26 * d["league"].nunique()))
