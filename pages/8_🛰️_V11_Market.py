"""V11 — the market-first shadow. It has never placed a bet and it never will from this page.

WHY THIS PAGE EXISTS. Until now nothing in the dashboard read v11 at all. It has been logging
decisions for months against a genuinely different philosophy from v9's, and the only way to see
any of it was to open the repo.

THE DIFFERENCE IN ONE LINE. v9 starts from the model and treats disagreement with the bookmaker
as edge. v11 starts from the bookmaker's own de-vigged price and asks whether the model adds
anything on top. Out-of-sample research said the first approach is a longshot machine, so v11
defaults to NO_BET and only tiers when several independent conditions hold at once.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_data import ui, v11

st.set_page_config(page_title="Wowza | V11", page_icon="🛰️", layout="wide")
st.title("🛰️ V11 — market-first shadow")

if not v11.available():
    ui.absent(v11.absent_reason())
    st.stop()

ui.state_banner(v11.STATE,
                "V11 observes and logs. It has no Telegram, no stakes and no write path into v9.")

health = v11.health()
c1, c2, c3 = st.columns(3)
with c1:
    ui.kpi("Shadow decisions logged", f"{health['shadow_rows']:,}", "reading v9's public data")
with c2:
    raw = (health.get("research_health") or {}).get("raw") or {}
    ui.kpi("Market snapshots", f"{raw.get('snapshots_rows', 0):,}",
           "price changes only — see the note below")
with c3:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ui.freshness_chip(health["freshness"], "shadow log")

st.divider()

tab_res, tab_move, tab_mom = st.tabs(
    ["Does the model add anything?", "Price movement", "Momentum"])

# ── the residual test: the only question that matters for v11 ─────────────────────────────────
with tab_res:
    st.subheader("The residual test")
    st.markdown(
        "**The question, stated so it can be wrong.** Take the bookmakers' prices, strip the "
        "margin out of them, and you have the market's own probability. Now add Wowza's model on "
        "top. Does the combined number predict results *better than the market alone*?"
    )
    st.markdown(
        "**Why it is not the same as accuracy.** A model can be accurate and still worthless — "
        "if everything it knows is already in the price, adding it changes nothing. Standalone "
        "AUC cannot tell those apart. Brier and log loss measured *after the price is known* can."
    )
    r = v11.residual()
    if not r.get("available"):
        st.info(r.get("why", "no residual results"))
    else:
        d = pd.DataFrame(r["table"])
        # Name the two columns explicitly. The baseline is the market alone; the comparison is
        # the market WITH Wowza folded in. Picking them positionally is how you end up comparing
        # a column to itself and reporting a delta of exactly zero as a finding.
        base_col = next((c for c in d.columns
                         if c.startswith("brier") and "wowza" not in c), None)
        with_col = next((c for c in d.columns
                         if c.startswith("brier") and "wowza" in c), None)
        if base_col and with_col and "n" in d.columns:
            n = pd.to_numeric(d["n"], errors="coerce").fillna(0)
            tot = int(n.sum())
            w = n / n.sum() if n.sum() else n
            # Weighted by n — an unweighted mean lets a 12-fixture segment count as much as a
            # 1,048-fixture one, which is how a thin segment gets to set the headline.
            bm = float((pd.to_numeric(d[base_col], errors="coerce") * w).sum())
            bmw = float((pd.to_numeric(d[with_col], errors="coerce") * w).sum())
            k1, k2, k3 = st.columns(3)
            with k1:
                ui.kpi("Market alone", f"{bm:.4f}", "Brier — lower is better")
            with k2:
                ui.kpi("Market + Wowza", f"{bmw:.4f}", "Brier — lower is better")
            with k3:
                delta = bm - bmw
                ui.kpi("What Wowza adds", f"{delta:+.5f}", f"n = {tot:,} fixtures",
                       tone=ui.polarity()[0] if delta > 0.001 else ui.NEUTRAL)
            st.markdown(
                f"**Read it plainly.** The market alone scores {bm:.4f}. Adding Wowza moves it to "
                f"{bmw:.4f} — a change of {delta:+.5f} on a scale where 0 is perfect and 0.25 is "
                f"a coin flip. On {tot:,} fixtures that is **indistinguishable from nothing**. "
                f"The honest reading is that on this evidence the model tells you nothing the "
                f"price has not already told you.")
            st.caption("That is a result, not a bug. It is the reason v11 defaults to NO_BET and "
                       "the reason the residual test, not AUC, is the measure that governs here.")
            # ONE scale, two series. Brier and log loss are separate charts by design.
            long_b = d.melt(id_vars=[c for c in ("n",) if c in d.columns],
                            value_vars=[base_col, with_col],
                            var_name="series", value_name="value")
            long_b["row"] = long_b.groupby("series").cumcount().astype(str)
            st.altair_chart(
                ui.compare_bars(long_b, "row", "value", "series",
                                title="Brier per segment — market vs market + Wowza",
                                y_title="Brier (lower is better)"),
                use_container_width=False)
        ui.table(d)

# ── movement ──────────────────────────────────────────────────────────────────────────────────
with tab_move:
    m = v11.movement()
    if not m.get("available"):
        st.info(m.get("why", "no movement detail"))
    else:
        st.subheader("How prices move between our entry and the close")
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.kpi("Observations", f"{m['rows']:,}", "entry → close pairs")
        with c2:
            ui.kpi("Fixtures", f"{m['fixtures']:,}", "with a usable close")
        with c3:
            ui.freshness_chip(m["freshness"], "movement detail")
        st.warning(
            "**Read every snapshot table as a change-log, not a panel.** The writers store "
            "consecutive *distinct* values only, so at any single instant the file contains only "
            "the entities that just moved. Grouping by (key, timestamp) therefore undercounts "
            "depth badly — measured 3 books per bet where the real figure is 8, a 2.7x "
            "understatement. Carry the last observation forward per entity before aggregating. "
            "This one storage decision has already produced three separate wrong conclusions.")
        rh = m.get("research_health") or {}
        if rh:
            st.markdown("**Instrument health**")
            st.caption(f"calculation version `{rh.get('calculation_version', '?')}` · "
                       f"git `{str(rh.get('git_sha', ''))[:8]}` · "
                       f"generated {rh.get('generated_at', '?')}")
            raw = rh.get("raw") or {}
            if raw:
                ui.table(pd.DataFrame([{"field": k, "value": v} for k, v in raw.items()]))

# ── momentum ──────────────────────────────────────────────────────────────────────────────────
with tab_mom:
    mm = v11.momentum()
    if not mm.get("available"):
        st.info(mm.get("why", "no momentum results"))
    else:
        st.subheader("Momentum controls")
        st.error(
            "**Every momentum number published before 2026-09-10 is void.** The script assigned a "
            "`pd.merge_asof` result back onto another frame by index. `merge_asof` returns a fresh "
            "RangeIndex in sorted-key order, so `sort_index()` did nothing and every value landed "
            "on the wrong row. The much-quoted figures — mean reversion 0.995, fixed anchor 0.753, "
            "shuffled residual 0.711 all beating v9's 0.703 — are **unmeasured, not disproven**. "
            "Do not quote them.")
        st.success(f"**What is on this page is post-fix.** {mm['provenance']} · "
                   f"calculation version `{mm.get('calc_version')}` · "
                   f"sample status `{mm.get('sample_status')}`.")
        st.markdown(
            f"Of **{mm['rows']}** fitted terms, **{mm['terms_excluding_zero']}** have a confidence "
            "interval that excludes zero. A term whose interval straddles zero has not been shown "
            "to do anything, however large its point estimate looks.")
        d = pd.DataFrame(mm["table"])
        if {"term", "coef", "ci_lo", "ci_hi"} <= set(d.columns):
            d = d.copy()
            d["excludes_zero"] = (d["ci_lo"] > 0) | (d["ci_hi"] < 0)
            d["label"] = d["term"].astype(str) + d["excludes_zero"].map({True: "  ✔", False: ""})
            st.altair_chart(ui.pnl_bars(d.head(20), "label", "coef",
                                        h=24 * min(len(d), 20) + 60,
                                        title="Fitted coefficients (✔ = interval excludes zero)"),
                            use_container_width=False)
            st.caption("The ✔ is the secondary cue — the bars are not distinguishable by colour "
                       "alone, and significance is never encoded in colour on this page.")
        ui.table(d)
