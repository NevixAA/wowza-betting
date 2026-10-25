"""Market Intelligence — price coverage, close (CLV) coverage, direction conflicts, legacy vs market-first.

Owner architecture brief 2026-10-25, s.30. A VIEWER of small CI-written artifacts (see
dashboard_data/market_intel.py for the list). Two selection philosophies are shown side by side and
are always labelled:
  * LEGACY RAW EDGE — what production uses today: p_model − 1/odds, tiered SNIPER/MARKSMAN/VALUABLE.
  * MARKET-FIRST (market_edge_v2, SHADOW ONLY) — de-vigged market fair price, decision probability,
    its lower bound, EV lower bound and a state (BET / PAPER / NO_BET). Never staked from here.
Nothing on this page prices, de-vigs or decides; it shows what the writers recorded.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - admin, like Market Lab (V11)

from dashboard_data import freshness, market_intel  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
freshness = importlib.reload(freshness)
market_intel = importlib.reload(market_intel)

st.set_page_config(page_title="Wowza | Market Intelligence", page_icon="📡", layout="wide")
import dashboard_ui as ui  # noqa: E402

ui.page_header("Market Intelligence", "Research and monitoring only — nothing here is a bet. Market-first rows are "
               "SHADOW ONLY.", eyebrow_text="Research Lab · prices and closes",
               freshness=ui.status_chip("SHADOW", label="MARKET-FIRST · SHADOW ONLY") + " "
               + ui.status_chip("LIVE", label="LEGACY RAW EDGE · PRODUCTION"))


@st.cache_data(ttl=300, show_spinner=False)
def _load(today: str):
    meta, cov = market_intel.edge_health()
    rows, conf_sum = market_intel.conflicts()
    up, lsum = market_intel.legacy_vs_market(today=today)
    return {"meta": meta, "cov": cov, "clv": market_intel.clv_health(), "conf": rows, "conf_sum": conf_sum,
            "up": up, "lsum": lsum}


d = _load(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
PCT = st.column_config.NumberColumn(format="%.1f%%")

# ── de-vig coverage ────────────────────────────────────────────────────────────────────────────
ui.section_header("De-vig coverage — two-sided prices per market × track")
m = d["meta"]
freshness.recorded(st, "output/edge_engine_health.json", m.get("generated_at"), 72, "edge-engine health")
if d["cov"].empty:
    st.warning("output/edge_engine_health.json missing or has no coverage block.")
else:
    ui.metric_grid([{"label": "Engine", "value": str(m.get("selection_engine_version"))},
                    {"label": "Config", "value": str(m.get("config_status"))},
                    {"label": "Rows scored (last run)", "value": f"{m.get('rows_scored') or 0:,}"},
                    {"label": "Book quotes (last run)", "value": f"{m.get('current_run_book_quotes') or 0:,}"}],
                   per_row=2)
    if m.get("shadow_only") is not True:
        st.error("shadow_only is not true in edge_engine_health.json — the market-first engine must stay shadow.")
    st.dataframe(d["cov"].assign(two_sided_share=d["cov"]["two_sided_share"] * 100), hide_index=True,
                 width="stretch", column_config={"two_sided_share": PCT})
    st.caption(f"De-vig method `{m.get('devig_method')}`. A fixture without a two-sided price cannot be de-vigged, so "
               "the market-first engine cannot judge it. `legacy_tipped` = rows the legacy raw-edge rule tiered.")

# ── CLV / close coverage ───────────────────────────────────────────────────────────────────────
ui.section_header("Certified close coverage (CLV)")
cl = d["clv"]
freshness.recorded(st, "output/clv_collection_health.json", cl.get("generated_at"), 30, "CLV collection health")
if not cl.get("generated_at"):
    st.warning("output/clv_collection_health.json missing or unreadable.")
else:
    s, a, col = cl["since"], cl["all_time"], cl["collection"]
    ui.metric_grid([
        {"label": f"Certified T-30 since {str(cl.get('measure_from'))[:10]}", "value": f"{s.get('certified_t30_pct')}%",
         "help": f"n = {s.get('eligible')} settled shadow rows"},
        {"label": "Certified T-10 (same window)", "value": f"{s.get('certified_t10_pct')}%",
         "help": f"n = {s.get('eligible')}"},
        {"label": "Certified T-30, all time", "value": f"{a.get('certified_t30_pct')}%",
         "help": f"n = {a.get('eligible')}"},
        {"label": f"Collection T-30, last {col.get('window_days')} d", "value": f"{col.get('t30_pct')}%",
         "help": f"{col.get('fixture_markets')} fixture-markets over {col.get('fixtures')} fixtures"}], per_row=2)
    st.caption(f"CLV coverage n: {s.get('eligible')} settled shadow rows since {str(cl.get('measure_from'))[:10]} · "
               f"{a.get('eligible')} all time.")
    st.caption(f"Close rule: {cl.get('close_rule')}. Check: {cl['check'].get('status', '?')} "
               f"{'; '.join(cl['check'].get('reasons') or [])}. A CLV figure on a stale close is not certified "
               "and is not evidence.")
    t1, t2, t3 = st.tabs(["Certified, by league", "Collection, by track / market", "Collection, by league"])
    with t1:
        st.dataframe(cl["since_by_league"].sort_values("eligible", ascending=False), hide_index=True,
                     width="stretch", column_config={"certified_t30_pct": PCT, "certified_t10_pct": PCT})
        st.dataframe(cl["since_by_market"], hide_index=True, width="stretch",
                     column_config={"certified_t30_pct": PCT, "certified_t10_pct": PCT})
    with t2:
        cc = {"t60_pct": PCT, "t30_pct": PCT, "t10_pct": PCT}
        st.dataframe(cl["collection_by_track"], hide_index=True, width="stretch", column_config=cc)
        st.dataframe(cl["collection_by_market"], hide_index=True, width="stretch", column_config=cc)
    with t3:
        st.dataframe(cl["collection_by_league"].sort_values("t30_pct"), hide_index=True, width="stretch",
                     column_config={"t60_pct": PCT, "t30_pct": PCT, "t10_pct": PCT})

# ── direction conflicts ────────────────────────────────────────────────────────────────────────
ui.section_header("Direction conflicts — suppressed opposite-direction tips")
if d["conf"].empty:
    st.info("No direction conflict recorded (output/direction_conflicts.csv is written only when the direction "
            "guard suppresses a tip).")
else:
    st.metric("Suppressed tips (all time)", f"{len(d['conf']):,}")
    st.dataframe(d["conf_sum"], hide_index=True, width="stretch")
    with st.expander("Latest suppressed rows"):
        st.dataframe(d["conf"].head(200), hide_index=True, width="stretch")
    st.caption("A suppressed tip is not sent, not ledgered and not staked: the earlier tip on that fixture stands.")

# ── legacy vs market-first ─────────────────────────────────────────────────────────────────────
ui.section_header("Legacy raw edge vs market-first decision")
with st.container(border=True):
    st.markdown("**LEGACY RAW EDGE** (production): `p_model − 1/odds` → SNIPER / MARKSMAN / VALUABLE.  \n"
                "**MARKET-FIRST** (market_edge_v2, *shadow only*): de-vigged market fair → decision probability → "
                "lower bound → EV lower bound → state BET / PAPER / NO_BET.")
if d["lsum"].empty:
    st.info("output/edge_engine_shadow.csv has no rows yet.")
else:
    st.dataframe(d["lsum"], hide_index=True, width="stretch",
                 column_config={"median_ev_lb": st.column_config.NumberColumn(format="%.3f")})
    st.caption("All first-sight rows. A legacy tier where the market-first state is NO_BET is a disagreement, not "
               "proof either way: neither column is a P/L, and none of these counts is an ROI.")
    up = d["up"]
    st.markdown(f"**Upcoming fixtures — {len(up):,} rows**")
    if not up.empty:
        num = st.column_config.NumberColumn
        st.dataframe(up, hide_index=True, width="stretch", column_config={
            "legacy_raw_edge": num("LEGACY raw edge", format="%.3f"),
            "p_market_fair": num("market fair", format="%.3f"), "p_decision": num("decision p", format="%.3f"),
            "p_lower_bound": num("lower bound", format="%.3f"), "ev_lower_bound": num("EV_LB", format="%.3f"),
            "p_model_raw": num("p_model", format="%.3f")})

try:                                                   # the link only resolves inside the app's navigation
    st.page_link("pages/8_🛰️_V11_Market.py", label="Market Lab (V11 market-first shadow)", icon="🛰️")
except Exception:                                      # noqa: BLE001
    st.caption("See also: Market Lab (V11 market-first shadow) in Research Lab.")
