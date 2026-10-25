"""Production — what production is doing right now: current bets and stakes, track states, protected tracks.

Owner architecture brief 2026-10-25, s.30. DISPLAY ONLY. A VIEWER of:
  * output/stake_plan.csv      today's and upcoming tips with the approved policy applied
                               (deployment_state, final stake, the reason for every zero)
  * config/track_registry.json the canonical read-model of every track's production state and research
                               regime; PROTECTED tracks are shown as PROTECTED — SHADOW ONLY
Policies change only through a reviewed PR (config/bankroll_policy.json); nothing here edits anything.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - admin: shows the full production state

from dashboard_data import freshness, production_view  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
freshness = importlib.reload(freshness)
production_view = importlib.reload(production_view)

st.set_page_config(page_title="Wowza | Production", page_icon="🏭", layout="wide")
st.title("🏭 Production")
st.caption("Display only — current bets, stakes and track states as production recorded them. Policies change "
           "through a reviewed PR, never here. Stakes are % of a notional bankroll.")


@st.cache_data(ttl=300, show_spinner=False)
def _load(today: str):
    plan = production_view.stake_plan(today=today)
    meta, reg = production_view.registry()
    return plan, production_view.plan_summary(plan), meta, reg, production_view.protected(reg)


today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
plan, summ, meta, reg, prot = _load(today)

# ── stake plan ─────────────────────────────────────────────────────────────────────────────────
st.subheader("Current bets and stakes")
freshness.recorded(st, "output/stake_plan.csv", None, 12, "stake plan")   # tracked by the manifest
if plan.empty:
    st.info(f"No tips dated {today} or later in output/stake_plan.csv — predict writes it on its next run.")
else:
    staked = plan[plan["final_fraction"] > 0]
    c = st.columns(4)
    c[0].metric("Tips sized", f"{len(plan):,}")
    c[1].metric("With a stake", f"{len(staked):,}")
    c[2].metric("Days covered", plan["date"].nunique())
    c[3].metric("Policy", str(plan["policy_version"].dropna().iloc[0]) if plan["policy_version"].notna().any() else "?")
    st.dataframe(summ, hide_index=True, width="stretch",
                 column_config={"stake_pct_total": st.column_config.NumberColumn("stake % total", format="%.3f")})
    show = st.toggle("Show tips without a stake", value=False)
    rows = plan if show else staked
    st.dataframe(rows[["date", "fixture", "league", "track", "market", "side", "signal_tier", "odds",
                       "deployment_state", "stake_pct", "reason", "decision_state"]],
                 hide_index=True, width="stretch",
                 column_config={"stake_pct": st.column_config.NumberColumn("final stake %", format="%.3f"),
                                "decision_state": st.column_config.TextColumn(
                                    "market-first state (shadow)",
                                    help="market_edge_v2 shadow decision for the same tip - display only")})
    st.caption(f"Generated {plan['generated_at'].dropna().max() if plan['generated_at'].notna().any() else '?'} UTC. "
               "Every zero stake carries its reason (paper league, PAPER/BLOCKED track, odds outside caps, ...).")

# ── track registry ─────────────────────────────────────────────────────────────────────────────
st.subheader("Track states (canonical track registry)")
if reg.empty:
    st.warning("config/track_registry.json missing or unreadable.")
else:
    g = meta.get("global") or {}
    sel = meta.get("selection_engine_version") or {}
    c = st.columns(3)
    with c[0]:
        with st.container(border=True):
            st.markdown("**Real-money tracks**")
            st.markdown("\n".join(f"- `{t}`" for t in g.get("real_money_tracks") or []) or "none")
    with c[1]:
        with st.container(border=True):
            st.markdown("**Blocked tracks**")
            st.markdown("\n".join(f"- `{t}`" for t in g.get("blocked_tracks") or []) or "none")
    with c[2]:
        with st.container(border=True):
            st.markdown("**Paper leagues / engines**")
            st.markdown(f"Paper: {', '.join(g.get('paper_leagues') or []) or 'none'}  \n"
                        f"Selection: production `{sel.get('production')}`, shadow `{sel.get('shadow')}`  \n"
                        f"Player props: {g.get('player_props')} · Cloudbet: {g.get('cloudbet')}")
    st.caption(f"{meta.get('registry_version')} for {meta.get('generated_for')} · policy {meta.get('policy_version')} "
               f"· {meta.get('n_entries')} entries · production states: {meta.get('production_state_counts')}")
    states = sorted(reg["production_state"].dropna().unique().tolist())
    pick = st.multiselect("Production state", states, default=[s for s in states if s != "OFF"] or states)
    st.dataframe(reg[reg["production_state"].isin(pick)], hide_index=True, width="stretch",
                 column_config={"evidence_reason": st.column_config.TextColumn(width="large")})
    st.caption("Evidence is shown with its n and independent match-days; an ROI inside evidence_reason is never "
               "proof on its own.")

# ── protected tracks ───────────────────────────────────────────────────────────────────────────
st.subheader("Protected tracks")
if prot.empty:
    st.info("No protected tracks in the registry.")
else:
    st.warning(f"**PROTECTED — SHADOW ONLY** · {len(prot)} track(s). Research may observe them, never change their "
               "production state, thresholds, stakes or eligibility (owner lock; the research-lock check in System "
               "Health turns FAIL if they move).")
    st.dataframe(prot[["key", "production_state", "research_regime", "owner_locked", "real_money", "evidence_status",
                       "evidence_n", "independent_matchdays"]], hide_index=True, width="stretch")
