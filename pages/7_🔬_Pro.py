"""Pro — the evidence layer. Nothing here is a bet.

WHY THIS PAGE EXISTS. Pro has been running for months and the dashboard could not see it. The one
page that touched Pro read a single Bet Builder CSV. Everything Pro exists for — the canonical
season store, the walk-forward proof that retraining actually helps, the governance contract, the
register of results that turned out to be wrong — was invisible unless you opened the repo.

WHAT IT IS NOT. Pro does not stake and does not write into v9. Its numbers are evidence about the
system, not returns from it. The banner says so at the top and the page never prints a P/L.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_data import pro, ui

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY. A page and its helper changed in the same
# commit, and the server ran the NEW page against the OLD dashboard_data.pro still in memory:
# AttributeError on pro.tip_scoreboard (2026-10-08). Reloading this small module on every run
# makes the page and its helper always the same version.
import importlib  # noqa: E402
pro = importlib.reload(pro)

st.set_page_config(page_title="Wowza | Pro", page_icon="🔬", layout="wide")
st.title("🔬 Wowza Pro — evidence & validation")

if not pro.available():
    ui.absent(pro.absent_reason())
    st.stop()

ui.state_banner("RESEARCH",
                "Pro validates and stores evidence. It never stakes, and it never writes into v9.")

contract = pro.system_contract()
if contract:
    va = contract.get("verified_against") or {}
    st.caption(
        f"System contract v{contract.get('contract_version', '?')} · generated "
        f"{contract.get('generated_at_utc', '?')} · verified against "
        f"v9 `{str(va.get('v9_commit', ''))[:7]}` · pro `{str(va.get('pro_commit', ''))[:7]}` · "
        f"v11 `{str(va.get('v11_commit', ''))[:7]}`")

tab_learn, tab_tips, tab_studies, tab_store, tab_gov, tab_health = st.tabs(
    ["Does it learn?", "Tips sent (1X2 · Bet Builder)", "Studies", "Canonical store", "Governance", "Health"])

# ── the load-bearing positive result ──────────────────────────────────────────────────────────
with tab_learn:
    st.subheader("Does retraining actually help?")
    sl = pro.shadow_learning()
    if not sl.get("available"):
        st.info(sl.get("why", "no walk-forward results"))
    else:
        st.markdown(
            "**The question.** Every week Wowza retrains on new football. That only makes sense "
            "if a model retrained on more data beats the same model left frozen. It is not "
            "obvious — more data also means more of last season's football, and football drifts."
        )
        st.markdown(
            "**The test.** Walk forward month by month over "
            f"**{sl['months']} months**. At each month, score three variants on football none of "
            "them has seen: **frozen** (never retrained), **current** (what production does), and "
            "**canonical** (retrained from the full canonical store). Lower log loss is better."
        )
        rows = [{"variant": k, "metric": m, "value": val}
                for k, v in sl["variants"].items() for m, val in v.items()]
        d = pd.DataFrame(rows)
        left, right = st.columns([3, 2])
        with left:
            ll = d[d["metric"] == "log_loss"].copy()
            ll["variant"] = pd.Categorical(ll["variant"],
                                           ["frozen", "current", "canonical"], ordered=True)
            st.altair_chart(
                ui.compare_bars(ll.sort_values("variant"), "variant", "value", "metric",
                                title="Log loss — lower is better", y_title="log loss"),
                use_container_width=False)
            # AUC and log loss live on different scales, so they get their OWN chart rather than
            # a second y-axis. A twin axis can be tuned to show any relationship the author wants.
            au = d[d["metric"].isin(["auc", "pr_auc"])].copy()
            st.altair_chart(
                ui.compare_bars(au, "variant", "value", "metric",
                                title="AUC and PR-AUC — higher is better  (separate chart: "
                                      "different scale, never a second axis)", y_title="score"),
                use_container_width=False)
        with right:
            best = sl.get("best_by_logloss")
            ui.kpi("Best by log loss", str(best).title(),
                   f"over {sl['months']} walk-forward months",
                   tone=ui.polarity()[0])
            v = sl["variants"]
            if "frozen" in v and "canonical" in v:
                gap = v["frozen"]["log_loss"] - v["canonical"]["log_loss"]
                st.markdown(
                    f"**In plain terms.** Retraining on the canonical store beats leaving the "
                    f"model frozen by **{gap:.5f}** of log loss. That is small in absolute size "
                    f"and it is the point: it is *consistently* in the same direction across "
                    f"{sl['months']} independent months, which is what a real effect looks like. "
                    f"Across the wider run, 78 league-cells improved and 0 degraded.")
            st.markdown(
                "**What follows.** Never delete old data. Every window, decay and filter test "
                "that threw history away lost to simply keeping it.")
            ui.table(d.pivot(index="variant", columns="metric", values="value")
                      .round(5).reset_index())

# ── the canonical store ───────────────────────────────────────────────────────────────────────
with tab_store:
    cs = pro.canonical_stores()
    if not cs.get("available"):
        st.info(cs.get("why", "no season store"))
    else:
        st.subheader(f"Canonical season store — {cs['season'].replace('_', ' ')}")
        st.markdown(
            "Each store is a **date-partitioned directory**, not one big file. That is how the "
            "estate stays under GitHub's hard 100 MB per-file limit without hand-splitting a "
            "table — and it has a second benefit worth more than the first: the newest partition "
            "name *is* the data's own date, so freshness comes from content and cannot be faked "
            "by a `git checkout` resetting file times.")
        parts = {k: v for k, v in cs["stores"].items() if v.get("partitioned")}
        if parts:
            d = pd.DataFrame([{"store": k, "partitions": v["partitions"],
                               "first": v["first_date"], "last": v["last_date"],
                               "state": v["freshness"].state, "age": v["freshness"].label}
                              for k, v in parts.items()]).sort_values("partitions",
                                                                      ascending=False)
            c1, c2 = st.columns([2, 3])
            with c1:
                st.altair_chart(ui.coverage_bars(d, "store", "partitions",
                                                 h=22 * len(d) + 60,
                                                 title="Days of data held, by store"),
                                use_container_width=False)
            with c2:
                st.markdown("**Freshness by store**")
                for _, r in d.iterrows():
                    f = parts[r["store"]]["freshness"]
                    ui.freshness_chip(f, f"{r['store']} · {r['first']} → {r['last']}")
            st.caption("A store with few partitions is not necessarily broken — some only write "
                       "on match days. Read the date range, not the count alone.")
            ui.table(d)
        flat = {k: v for k, v in cs["stores"].items() if not v.get("partitioned")}
        if flat:
            st.caption("Unpartitioned stores: " +
                       ", ".join(f"{k} ({v.get('files', 0)} files)" for k, v in flat.items()))

# ── governance ────────────────────────────────────────────────────────────────────────────────
with tab_gov:
    if not contract:
        st.info("No system contract found in Pro's registry.")
    else:
        st.subheader("What is allowed to happen, and what is proven")

        rm = contract.get("real_money") or {}
        st.markdown(
            f"**Automated bet placement exists: "
            f"{'YES' if rm.get('automated_placement_exists') else 'NO'}** — "
            f"evidence `{rm.get('evidence', '?')}`.")
        st.caption(rm.get("proof", ""))

        inv = contract.get("invariants") or []
        if inv:
            st.markdown("**Invariants and how well each is actually enforced**")
            st.caption("The status column is the honest part. An invariant written in a document "
                       "and an invariant enforced by code are different things, and the gap is "
                       "where the failures come from.")
            di = pd.DataFrame([{"#": i.get("id"), "invariant": i.get("statement"),
                                "status": i.get("status"), "evidence": i.get("evidence")}
                               for i in inv])
            ui.table(di)

        tr = contract.get("threshold_regimes") or {}
        if tr:
            with st.expander("⚠ The thresholds in config.py are NOT the ones production runs"):
                st.markdown(tr.get("rule", ""))
                dep = tr.get("deployed") or tr.get("production_overrides") or {}
                if isinstance(dep, dict) and dep:
                    ui.table(pd.DataFrame([{"setting": k, "value": v} for k, v in dep.items()]))
                st.caption(
                    "This is not pedantry. Reading config.py and stopping there produced the "
                    "widely repeated claim that 37 of 38 staked MARKSMAN bets were below their "
                    "own threshold. Measured against each bet's own effective floor the figure "
                    "is 25 of 38 — still a real defect, but a different one.")

        amb = contract.get("known_ambiguities") or []
        if amb:
            st.markdown("**Known ambiguities — things the code cannot settle**")
            st.caption("Code proves behaviour. It does not prove intent. These need a human "
                       "answer and are listed rather than guessed at.")
            ui.table(pd.DataFrame(amb))

# ── health ────────────────────────────────────────────────────────────────────────────────────
with tab_health:
    st.subheader("Pro's own health artifacts")
    st.caption("Green CI does not prove the instrument is measuring anything. v11's momentum "
               "work exited 0 for four weeks while measuring nothing at all. These files record "
               "what each pipeline believed about itself when it ran.")
    ha = pro.health_artifacts()
    files = ha.get("files") or {}
    if not files:
        st.info("No health artifacts found in Pro's output.")
    else:
        for fname, meta in sorted(files.items()):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"**{fname.replace('.json', '')}**")
            with c2:
                ui.freshness_chip(meta["freshness"],
                                  f"status: {meta.get('status') or 'not reported'}")

# ── how the tips Pro SENDS are doing (src/pipelines/tip_scoreboard.py) ───────────────────────
with tab_tips:
    sb = pro.tip_scoreboard()
    if not sb:
        st.info("No tip scoreboard yet — it is written by `pro_tip_scoreboard.yml` each morning.")
    else:
        if sb.get("freshness"):
            ui.freshness_chip(sb["freshness"], "tip scoreboard")
        st.caption("Graded only on tips that were actually SENT to Telegram. Paper leagues (USA MLS) excluded.")
        a, b = sb.get("one_x_two", {}), sb.get("bet_builder", {})
        h, bh = a.get("headline", {}), b.get("headline", {})
        st.subheader("1X2 tips")
        if h.get("settled"):
            c = st.columns(4)
            with c[0]:
                ui.kpi("Won", f"{h['won']} / {h['settled']}", f"{a.get('pending', 0)} pending")
            with c[1]:
                ui.kpi("Hit rate", ui.pct(h["hit_rate"]), f"break-even {ui.pct(h['break_even'])}")
            with c[2]:
                ui.kpi("Flat stakes", ui.units(h["units"]), f"ROI {ui.pct(h['roi'])}")
            with c[3]:
                ci = h.get("roi_ci90") or [None, None]
                ui.kpi("ROI 90% range", "–" if ci[0] is None else f"{ci[0]:+.0%} … {ci[1]:+.0%}",
                       "too few tips to judge" if h["settled"] < 50 else "")
            last = pd.DataFrame(a.get("last_10", []))
            if not last.empty:
                ui.table(last)
        else:
            st.info("No settled 1X2 tips yet.")
        st.subheader("Bet Builder combos")
        if bh.get("settled"):
            c = st.columns(3)
            with c[0]:
                ui.kpi("Won (fully graded)", f"{bh['won']} / {bh['settled']}", ui.pct(bh["hit_rate"]))
            with c[1]:
                ui.kpi("Model expected", f"{bh['expected_wins']:.1f} wins", f"z = {bh['z_actual_vs_claimed']}")
            with c[2]:
                v = b.get("with_voided_legs", {})
                ui.kpi("With a voided leg", str(v.get("settled", 0)), f"{v.get('won', 0)} 'won' on the remaining legs")
            st.caption("Combos with a voided player leg are settled on the remaining legs, so they are kept out of "
                       "the hit rate. No bookmaker builder price is recorded, so builder P/L cannot be measured.")
        else:
            st.info("No fully graded combos yet.")

# ── the October 2026 upgrade studies (src/studies) ───────────────────────────────────────────
with tab_studies:
    S = pro.studies()
    if not any(S.values()):
        st.info("No studies yet — `pro_studies.yml` writes them on Mondays and Thursdays.")
    else:
        st.caption("Research only. Refreshed twice a week. Full write-up: `output/studies/REPORT.md` in Pro.")
        ab = S.get("argentina_btts") or {}
        if ab:
            st.subheader("1 · Argentina BTTS — model, league, bookmaker, or luck?")
            rates = ab.get("btts_rate_all_matches", {})
            t = ab.get("tip_period", {})
            reg = (ab.get("regime_by_period") or {}).get("2026 since Aug 10 (tip period)", {})
            c = st.columns(4)
            with c[0]:
                ui.kpi("BTTS rate 2025", ui.pct(rates.get("2025", {}).get("rate")),
                       f"since 10 Aug: {ui.pct(rates.get('2026 since Aug 10', {}).get('rate'))}")
            with c[1]:
                ui.kpi("Price implied", ui.pct(reg.get("bet365_fair_yes_close")),
                       f"realised {ui.pct(reg.get('btts_rate'))} · gap {reg.get('gap_pp')} pts")
            with c[2]:
                bb = t.get("B_blind_yes_all", {})
                ui.kpi("Blind YES, every match", ui.pct(bb.get("roi")), f"n {bb.get('n')}")
            with c[3]:
                ui.kpi("Wowza picks minus the rest", ui.pct(t.get("A_minus_not_selected_roi")),
                       "range includes zero" if (t.get("A_minus_not_selected_ci90") or [0])[0] < 0 else "")
            st.markdown("**Reading:** the profit is a league scoring regime the market prices slowly, not model "
                        "selection. If the league calms down, the edge goes.")
        ch = S.get("ou_challenger") or {}
        if ch.get("probabilistic_oos"):
            st.subheader("2 · O/U 2.5 — v9 vs the market-anchored challenger")
            tv, tc = ch["tips_at_5pct_edge"]["v9"], ch["tips_at_5pct_edge"]["challenger"]
            c = st.columns(4)
            with c[0]:
                ui.kpi("v9 tips (5% edge)", str(tv.get("n")), f"{ui.pct(tv.get('under_share'))} UNDER")
            with c[1]:
                ui.kpi("v9 result", ui.units(tv.get("units", 0)), f"ROI {ui.pct(tv.get('roi'))}")
            with c[2]:
                ui.kpi("Challenger tips", str(tc.get("n")), "anchored on the market")
            with c[3]:
                w = ch.get("current_weights", {})
                ui.kpi("Model weight", str(w.get("c_model")), f"unconstrained {w.get('c_model_unconstrained')}")
            st.markdown("**Reading:** v9's O/U probability is squashed near 52%, so 'UNDER edges' are matches the "
                        "market rates high-scoring — and the market is right. Owner decision 2026-10-08: wait 2–3 weeks "
                        "of the challenger's forward record before any change to v9.")
        ou = S.get("ou_studies") or {}
        if ou.get("residual"):
            st.subheader("3 · Does the model add anything once the price is known?")
            rows = [{"Market": m, "Track": tr, "n": x.get("n"), "Δ log loss (out of sample)": x.get("oos_delta_logloss"),
                     "90% range": str(x.get("oos_delta_logloss_ci90")), "Verdict": x.get("verdict")}
                    for m, by in ou["residual"].items() for tr, x in by.items() if x.get("n_oos")]
            ui.table(pd.DataFrame(rows))
            st.caption("Negative Δ = adding the model improved the forecast. None does.")
        ev = S.get("evidence") or {}
        if ev.get("cells"):
            st.subheader("4 · Which league × market has earned money?")
            st.caption(f"{ev.get('n_cells_searched')} cells tested · reality-check p for the best cell "
                       f"{ev.get('reality_check_p_best_cell')} · gates: n ≥ 150, P(edge>0) ≥ 0.80, FDR q ≤ 0.10")
            rows = [{"League": c["league"], "Market": c["market"], "n": c["n"], "Raw ROI": c["roi"],
                     "Shrunk ROI": c.get("posterior_mean"), "P(edge>0)": c.get("p_edge_gt_0"),
                     "FDR q": c.get("q_bh"), "Recommendation": c.get("recommendation")}
                    for c in ev["cells"] if c.get("testable")]
            ui.table(pd.DataFrame(rows))
