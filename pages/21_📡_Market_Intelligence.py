"""Market Intelligence — price coverage, close (CLV) coverage, direction conflicts, legacy vs market-first.

Owner architecture brief 2026-10-25, s.30. A VIEWER of small CI-written artifacts (see
dashboard_data/market_intel.py for the list). Two selection philosophies are shown side by side and
are always labelled:
  * LEGACY RAW EDGE — what production uses today: p_model − 1/odds, tiered SNIPER/MARKSMAN/VALUABLE.
  * MARKET-FIRST (market_edge_v2, SHADOW ONLY) — de-vigged market fair price, decision probability,
    its lower bound, EV lower bound and a state (BET / PAPER / NO_BET). Never staked from here.
Nothing on this page prices, de-vigs or decides; it shows what the writers recorded.

MIGRATION COMPARISON (market-first brief, 2026-10-26): a RETROSPECTIVE section (first-sight replay,
output/edge_engine_comparison.json + output/edge_engine_backtest.csv) and a PROSPECTIVE section
(output/edge_engine_shadow.csv + Pro H-EDGE2 forward progress), per track x market, PAPER and DEPLOYABLE
never pooled. Values: dashboard_data/market_intel.migration_comparison / shadow_sample / edge2_status
(pinned by tests/test_market_intel_comparison.py).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - admin, like Market Lab (V11)

from dashboard_data import freshness, market_intel, pro  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
freshness = importlib.reload(freshness)
market_intel = importlib.reload(market_intel)

st.set_page_config(page_title="Wowza | Market Intelligence", page_icon="📡", layout="wide")
import dashboard_ui as ui  # noqa: E402
from src.tier_labels import TIER_LABEL  # noqa: E402

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
    ui.dataframe(d["cov"].assign(two_sided_share=d["cov"]["two_sided_share"] * 100), hide_index=True,
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
        ui.dataframe(cl["since_by_league"].sort_values("eligible", ascending=False), hide_index=True,
                     width="stretch", column_config={"certified_t30_pct": PCT, "certified_t10_pct": PCT})
        ui.dataframe(cl["since_by_market"], hide_index=True, width="stretch",
                     column_config={"certified_t30_pct": PCT, "certified_t10_pct": PCT})
    with t2:
        cc = {"t60_pct": PCT, "t30_pct": PCT, "t10_pct": PCT}
        ui.dataframe(cl["collection_by_track"], hide_index=True, width="stretch", column_config=cc)
        ui.dataframe(cl["collection_by_market"], hide_index=True, width="stretch", column_config=cc)
    with t3:
        ui.dataframe(cl["collection_by_league"].sort_values("t30_pct"), hide_index=True, width="stretch",
                     column_config={"t60_pct": PCT, "t30_pct": PCT, "t10_pct": PCT})

# ── direction conflicts ────────────────────────────────────────────────────────────────────────
ui.section_header("Direction conflicts — suppressed opposite-direction tips")
if d["conf"].empty:
    st.info("No direction conflict recorded (output/direction_conflicts.csv is written only when the direction "
            "guard suppresses a tip).")
else:
    st.metric("Suppressed tips (all time)", f"{len(d['conf']):,}")
    ui.dataframe(d["conf_sum"], hide_index=True, width="stretch")
    with st.expander("Latest suppressed rows"):
        ui.dataframe(d["conf"].head(200), hide_index=True, width="stretch")
    st.caption("A suppressed tip is not sent, not ledgered and not staked: the earlier tip on that fixture stands.")

# ── legacy vs market-first ─────────────────────────────────────────────────────────────────────
ui.section_header("Legacy raw edge vs market-first decision")
with st.container(border=True):
    st.markdown("**LEGACY RAW EDGE** (production): `p_model − 1/odds` → "
                f"{TIER_LABEL['SNIPER']} / {TIER_LABEL['MARKSMAN']} / {TIER_LABEL['VALUABLE']} (gap size).  \n"
                "**MARKET-FIRST** (market_edge_v2, *shadow only*): de-vigged market fair → decision probability → "
                "lower bound → EV lower bound → state BET / PAPER / NO_BET.")
if d["lsum"].empty:
    st.info("output/edge_engine_shadow.csv has no rows yet.")
else:
    ui.dataframe(d["lsum"], hide_index=True, width="stretch",
                 column_config={"median_ev_lb": st.column_config.NumberColumn(format="%.3f")})
    st.caption("All first-sight rows. A legacy tier where the market-first state is NO_BET is a disagreement, not "
               "proof either way: neither column is a P/L, and none of these counts is an ROI.")
    up = d["up"]
    st.markdown(f"**Upcoming fixtures — {len(up):,} rows**")
    if not up.empty:
        num = st.column_config.NumberColumn
        ui.dataframe(up, hide_index=True, width="stretch", column_config={
            "legacy_raw_edge": num("LEGACY raw edge", format="%.3f"),
            "p_market_fair": num("market fair", format="%.3f"), "p_decision": num("decision p", format="%.3f"),
            "p_lower_bound": num("lower bound", format="%.3f"), "ev_lower_bound": num("EV_LB", format="%.3f"),
            "p_model_raw": num("p_model", format="%.3f")})

# ── LEGACY vs MARKET-FIRST migration comparison (market-first brief) ───────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _load_migration():
    return market_intel.migration_comparison()


@st.cache_data(ttl=600, show_spinner=False)
def _load_edge2():
    if not pro.available():
        return None
    return pro.edge2_forward()


ui.section_header("Migration comparison — legacy raw edge vs market-first",
                  "Per track × market, never pooled. PAPER and DEPLOYABLE are separate rows, never summed.")
mig = _load_migration()
mm = mig["meta"]
ui.state_banner("RETROSPECTIVE", f"first-sight replay of committed boards · {mm.get('note') or ''}")
st.caption(f"Dated replay: output/edge_engine_comparison.json generated {mm.get('generated_at')} "
           f"({mm.get('engine_version')}). A replay does not age into forward evidence and is not refreshed daily.")
if mig["engines"].empty:
    ui.empty_state("NO_DATA", "output/edge_engine_comparison.json or output/edge_engine_backtest.csv is missing — "
                   "run `python -m src.research.edge_replay`.")
else:
    inp = mm.get("inputs") or {}
    ui.metric_grid([{"label": "Replay rows", "value": f"{inp.get('replay_rows') or 0:,}",
                     "help": "first-sight fixture × market rows"},
                    {"label": "Settled replay rows", "value": f"{inp.get('settled_rows') or 0:,}"},
                    {"label": "Engine config", "value": str(mm.get("config_status"))},
                    {"label": "Bankroll policy (DEPLOYABLE split)", "value": str(mig["policy_version"])}],
                   per_row=2)
    st.caption("**LEGACY** = production's raw-edge tier at first sight (not the ledger: its rows logged before "
               "2026-10-26 can carry a tier upgraded in place; since then the first logged tier stands). **DEPLOYABLE** = a legacy selection today's bankroll policy would stake (track state, paper "
               "league, odds caps, tier multiplier); **PAPER** = one it would not. That split applies today's "
               "policy to the whole window, and today's policy was set partly on this window, so it describes, it "
               "does not test. Every market-first (NEW) row is PAPER: market_edge_v2 is shadow only. ROI CI = "
               "95% matchday block bootstrap; CLV = certified closes only (≤ 30 min before kickoff).")
    rec = mig["reconcile"]
    if not rec.empty and not bool(rec["reconciled"].all()):
        st.error("The DEPLOYABLE + PAPER split does not add up to the comparison JSON's LEGACY figure for: "
                 + ", ".join(rec.loc[~rec["reconciled"], "cell"]) + ". Do not read the split.")
    elif not rec.empty:
        st.caption("Reconciled: DEPLOYABLE + PAPER = the JSON's LEGACY n and P/L in every cell.")

    gain = mig["brier_gain"].assign(label=lambda d: d["cell"] + " (n=" + d["n"].astype(str) + ")")
    agree = mig["agreement"]
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(ui.pnl_bars(gain, "label", "brier_gain", n_col="n", h=260,
                                    title="Does the model beat the de-vigged market?",
                                    axis_title="Brier gain, model − market (+ = model better)",
                                    value_title="Brier gain", fmt="+.4f", n_title="two-sided rows"),
                        width="stretch")
    with c2:
        ev = agree[agree["new_engine"] == "NEW_EVLB"]
        sel = ev.melt(id_vars=["cell", "two_sided_rows"], value_vars=["legacy_only", "new_only", "both_same_side"],
                      var_name="selected by", value_name="fixtures")
        sel["cell"] = sel["cell"] + " (n=" + sel["two_sided_rows"].astype(str) + ")"
        sel["selected by"] = sel["selected by"].map({"legacy_only": "legacy only", "new_only": "new only",
                                                     "both_same_side": "both, same side"})
        st.altair_chart(ui.compare_bars(sel, "cell", "fixtures", "selected by",
                                        order=["legacy only", "new only", "both, same side"], h=260, fmt=",.0f",
                                        title="Do the two engines pick the same fixtures?",
                                        y_title="fixtures selected (two-sided rows)"), width="stretch")
    st.caption("Brier gain CI and every count are in the tables below. NEW here = EV lower bound > 0 (the engine's "
               "value path, evidence gate ignored).")

    num = st.column_config.NumberColumn
    eng_cfg = {"n": num("bets", format="%d"), "matchdays": num("matchdays", format="%d"),
               "hit_pct": num("hit %", format="%.1f%%"), "roi_pct": num("ROI %", format="%+.1f%%"),
               "roi_ci_lo": num("ROI CI lo", format="%+.1f%%"), "roi_ci_hi": num("ROI CI hi", format="%+.1f%%"),
               "pnl_u": num("P/L u", format="%+.2f"), "max_drawdown_u": num("max DD u", format="%.2f"),
               "clv_n": num("CLV n", format="%d"), "clv_coverage_pct": num("CLV cov %", format="%.1f%%"),
               "clv_mean_pct": num("CLV mean %", format="%+.2f%%")}
    prob_cfg = {"brier": num("Brier", format="%.5f"), "logloss": num("log loss", format="%.5f"),
                "calibration_slope": num("calibration slope", format="%.3f"), "ece": num("ECE", format="%.4f"),
                "n": num("n", format="%d")}
    cells = list(dict.fromkeys(mig["engines"]["cell"]))
    for tab, cell in zip(st.tabs(cells), cells):
        with tab:
            e = mig["engines"][mig["engines"]["cell"] == cell]
            status = str(e["cell_status"].iloc[0])
            g = gain[gain["cell"] == cell].iloc[0]
            st.markdown(ui.status_chip("RETROSPECTIVE") + " " + (
                ui.status_chip("PASS", label=f"CELL OK · n={int(g['n'])} two-sided") if status == "OK"
                else ui.status_chip("SMALL_SAMPLE", label=f"INSUFFICIENT · n={int(g['n'])} two-sided")))
            ui.table(e.drop(columns=["cell", "track", "market", "cell_status"]), auto_format=False,
                     column_config=eng_cfg)
            pr = mig["probability"][mig["probability"]["cell"] == cell]
            ui.table(pr.drop(columns=["cell", "track", "market"]), auto_format=False, column_config=prob_cfg)
            a = agree[agree["cell"] == cell]
            ui.table(a.drop(columns=["cell", "track", "market"]), auto_format=False, column_config={
                "agreement_rate": num("agreement rate", format="percent"),
                "selection_overlap": num("selection overlap", format="percent")})
            st.caption(f"Model − market Brier gain {g['brier_gain']:+.5f}, 95% CI [{g['ci_lo']:+.5f}, "
                       f"{g['ci_hi']:+.5f}] on n = {int(g['n'])} two-sided rows (side A = OVER/YES). Agreement rate = "
                       "(both same side + neither) / two-sided rows; selection overlap = both same side / rows "
                       "either engine selected." + ("" if status == "OK" else " INSUFFICIENT: fewer than 30 "
                                                     "two-sided rows — not readable as a rate."))

    # PROSPECTIVE: the live first-sight shadow log + Pro's H-EDGE2 forward family
    ui.section_header("Migration comparison — prospective (forward)",
                      "What is accruing now. Nothing forward is CONFIRMED; the evaluator cannot assign it.")
    ui.state_banner("PROSPECTIVE", "output/edge_engine_shadow.csv (v9 predict) + Pro H-EDGE2 forward progress")
    e2 = _load_edge2()
    e2_meta, e2_tab = market_intel.edge2_status(e2) if e2 else ({}, pd.DataFrame())
    smeta, stab = market_intel.shadow_sample(forward_start=e2_meta.get("forward_start"))
    if stab.empty:
        ui.empty_state("NO_DATA", "output/edge_engine_shadow.csv has no rows yet.")
    else:
        ui.metric_grid([{"label": "Shadow rows (first sight)", "value": f"{smeta['rows']:,}"},
                        {"label": "Predict runs logged", "value": f"{smeta['runs']:,}"},
                        {"label": "First row", "value": str(smeta["first_prediction_ts"])[:16]},
                        {"label": "Last row", "value": str(smeta["last_prediction_ts"])[:16]}], per_row=2)
        cols = ["cell", "rows", "two_sided", "since_forward_start", "kicked_off", "states"]
        ui.table(stab[[c for c in cols if c in stab.columns]], auto_format=False, column_config={
            "rows": num("rows", format="%d"), "two_sided": num("two-sided", format="%d"),
            "since_forward_start": num("since H-EDGE2 start", format="%d"),
            "kicked_off": num("kicked off", format="%d"), "states": st.column_config.TextColumn("decision states")})
        st.caption("Rows, not bets: every state is shadow. OVER35 is blocked and logged for evidence only. The shadow "
                   "log has no result columns; forward settlement is done by Pro's H-EDGE2 evaluator.")
    if e2 is None:
        ui.empty_state("NOT_CONNECTED", pro.absent_reason())
    elif e2_tab.empty:
        ui.empty_state("NO_DATA", "Pro output/edge_reassessment/edge2_forward_progress.json has no hypotheses.")
    else:
        freshness.recorded(st, "pro:output/edge_reassessment/edge2_forward_progress.json", e2_meta.get("generated_at"),
                           72, "H-EDGE2 forward progress")
        if e2_meta["invalid_status"]:
            st.error("H-EDGE2 reported a status this page does not accept (CONFIRMED is never assignable): "
                     + ", ".join(e2_meta["invalid_status"]))
        ui.show_status(*sorted(set(e2_tab["status"])))
        ui.table(e2_tab, auto_format=False, column_config={
            "progress_pct": num("progress %", format="%.1f%%"),
            "pre_registration_not_counted": num("pre-registration rows (not counted)", format="%d")})
        st.caption(f"Forward start {e2_meta.get('forward_start')}. Registration intact: "
                   f"{e2_meta.get('registration_intact')}; engine hashes match registration: "
                   f"{e2_meta.get('engine_hashes_match_registration')}; production changed: "
                   f"{e2_meta.get('production_changed')}. ELIGIBLE_FOR_REVIEW needs the minimum settled two-sided "
                   "rows AND 8 gameweeks, then owner review.")

try:                                                   # the link only resolves inside the app's navigation
    st.page_link("pages/8_🛰️_V11_Market.py", label="Market Lab (V11 market-first shadow)", icon="🛰️")
except Exception:                                      # noqa: BLE001
    st.caption("See also: Market Lab (V11 market-first shadow) in Research Lab.")
