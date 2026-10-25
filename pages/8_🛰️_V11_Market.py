"""V11 — the market-first shadow. It has never placed a bet and it never will from this page.

WHY THIS PAGE EXISTS. Until now nothing in the dashboard read v11 at all. It has been logging
decisions for months against a genuinely different philosophy from v9's, and the only way to see
any of it was to open the repo.

THE DIFFERENCE IN ONE LINE. v9 starts from the model and treats disagreement with the bookmaker
as edge. v11 starts from the bookmaker's own de-vigged price and asks whether the model adds
anything on top. Out-of-sample research said the first approach is a longshot machine, so v11
defaults to NO_BET and only tiers when several independent conditions hold at once.

Visual redesign 2026-10-25 (stage 3, nav label "Market Lab"): page header with the shadow state and the
shadow log's freshness, finding cards with their n and evidence kind, no dividers, and every remote read
cached for five minutes (they were re-fetched on every widget interaction). PRESENTATION ONLY.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

import dashboard_ui as ui  # noqa: E402
from dashboard_data import pro, v11  # noqa: E402

# Streamlit Cloud keeps imported modules across a redeploy (see the Pro page); reload so this page and
# the helper are always the same version.
import importlib  # noqa: E402
pro = importlib.reload(pro)

st.set_page_config(page_title="Wowza | V11", page_icon="🛰️", layout="wide")
_TITLE, _SUB = "Market Lab", "V11 — the market-first shadow. It observes and logs: no Telegram, no stakes, no write path into v9."
_EYEBROW = "Research Lab · V11 market-first shadow"


@st.cache_data(ttl=300, show_spinner="Reading v11…")
def _load() -> dict:
    """Every remote read the page makes, once per five minutes."""
    return {"health": v11.health(), "residual": v11.residual(), "movement": v11.movement(),
            "momentum": v11.momentum(),
            "player_v3": pro.verdict("output/player_v3/player_v3_verdict.json"),
            "price_trader": pro.verdict("output/price_trader/price_trader_verdict.json")}


if not v11.available():
    ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW)
    ui.empty_state("NOT_CONNECTED", v11.absent_reason())
    st.stop()

D = _load()
health = D["health"]
ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW,
               freshness=ui.status_chip("SHADOW", label=f"{v11.STATE} · LOGS, NEVER STAKES") + " "
               + ui.freshness_badge(health["freshness"].state, f"shadow log · {health['freshness'].label}"))

raw = (health.get("research_health") or {}).get("raw") or {}
c = st.columns(2)
ui.metric_card("Shadow decisions logged", f"{health['shadow_rows']:,}", title="Reading v9's public data",
               status=ui.status_chip("SHADOW"), container=c[0])
ui.metric_card("Market snapshots", f"{raw.get('snapshots_rows', 0):,}", title="Price changes only",
               supporting=[("Stored as", "a change-log, not a panel — see Price movement")], container=c[1])

tab_res, tab_move, tab_mom = st.tabs(
    ["Does the model add anything?", "Price movement", "Momentum"])

# ── the residual test: the only question that matters for v11 ─────────────────────────────────
with tab_res:
    ui.section_header("The residual test", container=st)
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
    r = D["residual"]
    if not r.get("available"):
        ui.empty_state("NO_DATA", r.get("why", "no residual results"))
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
            delta = bm - bmw
            k = st.columns(3)
            ui.metric_card("What Wowza adds", f"{delta:+.5f}", title="Brier improvement over the market",
                           supporting=[("n", f"{tot:,} fixtures")],
                           status="" if delta > 0.001 else ui.status_chip(
                               "NO_DATA", label="INDISTINGUISHABLE FROM NOTHING", tone="NEUTRAL"),
                           evidence=ui.evidence_badge("SHADOW LOG", tot, state="TEAM MARKETS"), container=k[0])
            ui.metric_card("Market alone", f"{bm:.4f}", title="Brier — lower is better", container=k[1])
            ui.metric_card("Market + Wowza", f"{bmw:.4f}", title="Brier — lower is better", container=k[2])
            st.markdown(
                f"**Read it plainly.** The market alone scores {bm:.4f}. Adding Wowza moves it to "
                f"{bmw:.4f} — a change of {delta:+.5f} on a scale where 0 is perfect and 0.25 is "
                f"a coin flip. On {tot:,} fixtures that is **indistinguishable from nothing**. "
                f"The honest reading is that on this evidence the model tells you nothing the "
                f"price has not already told you.")
            st.caption("That is a result, not a bug. It is the reason v11 defaults to NO_BET and "
                       "the reason the residual test, not AUC, is the measure that governs here.")
            # SCOPE (owner, 2026-10-14): this test is TEAM markets only. Pro's Player V3 ran the same test on
            # player props and found the opposite, so the page states which market each result covers.
            pv = D["player_v3"]
            if pv:
                by = pv.get("PLAYER_WOWZA_ADDS_TO_MARKET_BY_MARKET", {})
                st.info(f"**Scope: team fixtures only.** The same test on **player props** (Pro, Player V3, "
                        f"{pv.get('generated_at', '?')}) says the model *does* add to the price: "
                        f"{', '.join(f'{k} {v}' for k, v in by.items())}. That is accuracy only — no prop bet has "
                        f"shown a profit yet, so props stay paper (pre-registered forward test running).")
            # ONE scale, two series. Brier and log loss are separate charts by design.
            long_b = d.melt(id_vars=[c for c in ("n",) if c in d.columns],
                            value_vars=[base_col, with_col],
                            var_name="series", value_name="value")
            long_b["row"] = long_b.groupby("series").cumcount().astype(str)
            st.altair_chart(
                ui.compare_bars(long_b, "row", "value", "series",
                                title="Brier per segment — market vs market + Wowza",
                                y_title="Brier (lower is better)"),
                width="stretch")
        with st.expander("Residual table — every segment"):
            ui.table(d)

# ── movement ──────────────────────────────────────────────────────────────────────────────────
with tab_move:
    m = D["movement"]
    if not m.get("available"):
        ui.empty_state("NO_DATA", m.get("why", "no movement detail"))
    else:
        ui.section_header("How prices move between our entry and the close", container=st)
        st.markdown(ui.freshness_badge(m["freshness"].state, f"movement detail · {m['freshness'].label}"))
        k = st.columns(2)
        ui.metric_card("Observations", f"{m['rows']:,}", title="Entry → close pairs", container=k[0])
        ui.metric_card("Fixtures", f"{m['fixtures']:,}", title="With a usable close", container=k[1])
        # Pro's Price Trader lab re-tested these movement figures as REAL trades (2026-10-14). Shown first so a
        # 70%+ toward rate is never read as money on the table.
        pt = D["price_trader"]
        if pt:
            st.error(
                f"**Movement is not a tradeable edge (Pro Price Trader, {pt.get('generated_at', '?')}).** The toward "
                "rates here measure fair-probability drift across many snapshots. Entered at Wowza's real signal, at "
                "prices that really existed: the price moved toward Wowza on only ~41% of trades, a model-free "
                "placebo did as well, and locking in a profit lost ~10% per trade (the bookmaker margin is bigger "
                f"than the moves). Verdict: beats placebo **{pt.get('MOVEMENT_BEATS_PLACEBO')}**, profitable after "
                f"costs **{pt.get('FULL_GREEN_PROFITABLE_AFTER_COSTS')}**, prospective "
                f"**{pt.get('PROSPECTIVE_CONFIRMATION')}** (paper engine running).")
        st.warning(
            "**Read every snapshot table as a change-log, not a panel.** The writers store "
            "consecutive *distinct* values only, so at any single instant the file contains only "
            "the entities that just moved. Grouping by (key, timestamp) therefore undercounts "
            "depth badly — measured 3 books per bet where the real figure is 8, a 2.7x "
            "understatement. Carry the last observation forward per entity before aggregating. "
            "This one storage decision has already produced three separate wrong conclusions.")
        rh = m.get("research_health") or {}
        if rh:
            ui.eyebrow("Instrument health")
            st.caption(f"calculation version `{rh.get('calculation_version', '?')}` · "
                       f"git `{str(rh.get('git_sha', ''))[:8]}` · "
                       f"generated {rh.get('generated_at', '?')}")
            raw = rh.get("raw") or {}
            if raw:
                ui.table(pd.DataFrame([{"field": k, "value": v} for k, v in raw.items()]))

# ── momentum ──────────────────────────────────────────────────────────────────────────────────
with tab_mom:
    mm = D["momentum"]
    if not mm.get("available"):
        ui.empty_state("NO_DATA", mm.get("why", "no momentum results"))
    else:
        ui.section_header("Momentum controls", container=st)
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
        ui.metric_card("Terms whose interval excludes zero", f"{mm['terms_excluding_zero']} of {mm['rows']}",
                       title="Fitted terms", evidence=ui.evidence_badge("POST-FIX", mm["rows"],
                                                                         state=str(mm.get("sample_status") or "")))
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
                                        title="Fitted coefficients (✔ = interval excludes zero)",
                                        axis_title="coefficient", value_title="coef", fmt="+.4f"),
                            width="stretch")
            st.caption("The ✔ is the secondary cue — the bars are not distinguishable by colour "
                       "alone, and significance is never encoded in colour on this page.")
        with st.expander("Coefficient table"):
            ui.table(d)
