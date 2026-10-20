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

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

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

(tab_latest, tab_champ, tab_trader, tab_valuable, tab_learn, tab_tips, tab_studies, tab_store, tab_gov,
 tab_health) = st.tabs(
    ["Latest findings", "Model champions", "Price Trader", "VALUABLE research", "Does it learn?",
     "Tips sent (1X2 · Bet Builder)", "Team studies", "Canonical store", "Governance", "Health"])

# ── VALUABLE deep dive (owner brief 2026-10-20) — READ-ONLY research, selection logic untouched ─────
with tab_valuable:
    vd = pro.valuable()
    sm = vd.get("valuable_summary", pd.DataFrame())
    if sm.empty:
        st.info("No VALUABLE deep dive yet — Pro writes `output/valuable_deep_dive/` (src/studies/valuable).")
    else:
        st.markdown(
            "**Research only — nothing here changes which tips are sent.** VALUABLE is not one region: on O/U 2.5 "
            "the size of the edge does not predict results; on BTTS it does, and 4–8% is the one band that held up "
            "out of sample. Seven hypotheses are frozen and tested from 21 Oct on a shadow of every candidate.")
        val = sm[sm.tier == "VALUABLE"]
        pooled = val[val.market.str.startswith("ALL")].set_index("window")
        c = st.columns(3)
        for col, (w, lab) in zip(c, (("last14", "Last 14 days"), ("last30", "Last 30 days"), ("since_aug26", "Since 26 Aug"))):
            if w in pooled.index:
                r = pooled.loc[w]
                col.metric(lab, f"{r.ROI_pct:+.1f}%", f"n {int(r.N)} · 90% CI {r.ROI_ci90_lo:+.0f}..{r.ROI_ci90_hi:+.0f}",
                           delta_color="off")
        st.caption("Pooled across markets, as the owner audit counts it. By market below.")
        ui.table(val[~val.market.str.startswith("ALL")][["window", "market", "N", "ROI_pct", "ROI_ci90_lo", "ROI_ci90_hi", "CLV_pct"]]
                 .rename(columns={"ROI_pct": "ROI %", "ROI_ci90_lo": "CI low", "ROI_ci90_hi": "CI high", "CLV_pct": "CLV %"}))
        st.subheader("Edge bands (O/U 2.5 VALUABLE since 26 Aug)")
        st.markdown("3–4%: **−5.8%** (n 196) · 4–6%: **+3.1%** (n 120) · 6–8%: **+16.3%** (n 31) — CLV *falls* as the "
                    "edge rises (+0.37 → −0.17 → −0.74), so the price does not confirm the high band. "
                    "BTTS fair edge 4–8%: new-format **+19.6%** (n 87), standard **+31.1%** (n 34).")
        fb = vd.get("valuable_fair_edge_bands", pd.DataFrame())
        if not fb.empty:
            with st.expander("Fair-edge bands (two-sided de-vig)"):
                ui.table(fb[(fb.kind == "aggregate") & (fb.N > 0)][["band", "market", "model_type", "N", "ROI_pct", "ROI_ci90_lo",
                                                                     "ROI_ci90_hi", "CLV_pct"]])
        lm = vd.get("valuable_league_market", pd.DataFrame())
        if not lm.empty:
            st.subheader("League × market heatmap (shrunk ROI, VALUABLE, all live history)")
            import plotly.express as px                                    # heavy import kept out of module scope
            h = lm[lm.N >= 10].pivot_table(index="league", columns="market", values="ROI_shrunk_pct")
            if not h.empty:
                fig = px.imshow(h, color_continuous_scale="RdBu", color_continuous_midpoint=0, text_auto=".0f",
                                aspect="auto", labels={"color": "shrunk ROI %"})
                fig.update_layout(height=max(320, 26 * len(h)), margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
            st.caption("Cells with ≥10 settled tips, shrunk toward their tier × market × model average. "
                       "No cell survives multiple-testing control (best q 0.61).")
        sh = vd.get("shadow_live") or vd.get("shadow") or {}
        st.caption(f"Low-edge shadow (every candidate incl. below the floor, never bet): "
                   f"{sh.get('rows', 0)} rows · {sh.get('settled', 0)} settled · status "
                   f"{(vd.get('shadow') or {}).get('status', '?')}")

# ── Price Trader research lab (owner brief 2026-10-19) — PAPER / SHADOW only ────────────────────
with tab_trader:
    pt = pro.price_trader()
    v = pt.get("verdict") or {}
    if not v:
        st.info("No Price Trader v2 verdict yet — Pro writes it daily (pro_price_trader.yml).")
    else:
        state = v.get("EVIDENCE_STATE", "?")
        colour = {"COLLECTING": "🟡", "PRICE_SIGNAL": "🟢", "EXECUTABLE_EDGE": "🟢", "NO_EDGE": "🔴"}.get(state, "⚪")
        st.markdown(f"## {colour} {state}")
        st.caption("Paper/shadow research only: no orders, no stakes, no messages. Raw price movement is never "
                   "called an edge without the economic test (costs, breakeven, controls, forward window).")
        ds = v.get("data_status") or {}
        h = pt.get("health") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signals with a price path", h.get("tradable", "?"))
        c2.metric("Exchange back prices", ds.get("EXCHANGE_BACK_PRICES_RECORDED", "?"),
                  help=f"share of path rows: {ds.get('exchange_opposite_coverage_of_path_rows')}")
        c3.metric("Executable exchange data", ds.get("EXECUTABLE_EXCHANGE_DATA_AVAILABLE", "?"),
                  help="no lay ladder / matched size is captured anywhere")
        c4.metric("Capacity", "UNKNOWN" if ds.get("CAPACITY_UNKNOWN") == "YES" else "known")
        hp = v.get("historical_primary") or {}
        fg = v.get("forward_gates") or {}
        st.markdown(f"**Primary setup (historical, exploratory):** {hp.get('n')} hedged trades, mean "
                    f"{(hp.get('mean_pnl_per_unit') or 0):+.1%} per unit, full-green rate "
                    f"{(hp.get('full_green_rate') or 0):.1%}.  **Forward window** (from 2026-10-20): "
                    f"{fg.get('N_completed_forward', 0)} of {fg.get('N_required', 150)} trades, "
                    f"{fg.get('match_days', 0)} of {fg.get('days_required', 30)} match days.")
        hz = pt.get("horizon_rule_results")
        if hz is not None and len(hz):
            sub = hz[(hz["venue"] == "book") & (hz["slippage"] == 0) & (hz["rule"] == "GREEN_2PCT")]
            st.markdown("**By entry horizon** — observed favourable rate vs the rate needed to break even after costs")
            ui.table(sub[["horizon", "fixtures", "completed", "favourable_move_rate", "median_clv",
                          "full_green_rate", "mean_pnl_per_unit", "observed_favourable",
                          "required_favourable", "tradeable"]])
        ct = pt.get("controls")
        if ct is not None and len(ct):
            with st.expander("Wowza vs controls at the same entries (opposite, random, shuffled, market-only)"):
                ui.table(ct[ct["rule"] == "GREEN_2PCT"][["horizon", "control", "n", "wowza_toward_rate",
                                                          "control_toward_rate", "wowza_mean_pnl",
                                                          "control_mean_pnl", "pnl_diff_ci90"]])
        st.markdown(f"**States (primary):** {v.get('states_primary')} · **labels:** {v.get('labels_primary')}")
        integ = pt.get("integrity") or {}
        if integ:
            st.caption(f"Research integrity: {integ.get('summary')} · blocked: "
                       f"{integ.get('EDGE_VERDICTS_BLOCKED_FOR') or 'nothing'}")
        st.caption("PRODUCTION_SELECTION_CHANGED = NO · THRESHOLDS = NO · STAKING = NO · REAL_BETTING_ENABLED = NO")

# ── champion high-water mark (owner 2026-10-18) ───────────────────────────────────────────────
with tab_champ:
    import json as _json
    st.subheader("Model champions — the high-water mark")
    st.markdown(
        "**Retraining is an experiment; promotion is a privilege.** Every retrain trains a *challenger*. "
        "It replaces the serving model only when it is clearly better on the same fixtures. Worse or "
        "tied keeps the champion; *possibly* better is shadowed — it predicts the same live fixtures "
        "next to the champion, without ever producing a tip, until the evidence decides. Weekly "
        "results still swing with football; what may not slip is the quality of the model serving.")
    _root = Path(__file__).resolve().parents[1]
    try:
        _ptr = _json.loads((_root / "models" / "registry" / "current_champions.json").read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        _ptr = {}
    try:
        _dec = pd.read_csv(_root / "output" / "model_lineage" / "retrain_batches.csv")
    except Exception:                                                  # noqa: BLE001
        _dec = pd.DataFrame()
    ml = pro.model_lineage()
    view = ml.get("current_champions_view", pd.DataFrame())
    if len(view):
        ui.table(view.rename(columns=str))
    elif _ptr:
        ui.table(pd.DataFrame([{"model": k, "champion": v.get("model_version"),
                                "since": v.get("champion_since")} for k, v in sorted(_ptr.items())]))
    else:
        st.info("No champion registry yet.")

    # lineage timeline: one line per model, last 12 decisions, oldest -> newest
    if len(_dec):
        st.markdown("**Lineage — last 12 retrains per model** (oldest → newest)")
        _sym = {"PROMOTE": "⬆️ promoted", "KEEP_CHAMPION": "⏸ kept", "SHADOW_MORE": "🔬 shadow",
                "LEGACY_PROMOTED": "↑ old-rule promote", "LEGACY_BLOCKED": "⛔ old-rule block"}
        for mt, g in _dec.groupby("model_type"):
            chain = " → ".join(_sym.get(str(v), str(v)) for v in g["verdict"].tail(12))
            st.markdown(f"- **{mt}**: {chain}")
        st.caption("Before 2026-10-18 the old rule promoted almost every retrain (each champion lived about a "
                   "day). From 2026-10-18 only a proven improvement promotes.")

    a = ml.get("answers") or {}
    if a:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrains analysed", a.get("RETRAIN_ATTEMPTS_ANALYZED"))
        c2.metric("Promoted", a.get("PROMOTIONS"))
        c3.metric("Churn rate", a.get("MODEL_CHURN_RATE"))
        c4.metric("Head-to-head graded", a.get("H2H_GRADED"))
        st.caption(f"Best proven improvement: {a.get('BEST_PROVEN_CHAMPION_IMPROVEMENT')}")
    wt = ml.get("retrain_week_table", pd.DataFrame())
    if len(wt):
        with st.expander("Retrain → what happened next (the week table)"):
            st.caption("Future results = settled bets the model generated until the next retrain decision. "
                       "WEEKLY_RESULT_VARIANCE_HIGH: one week never demotes a champion.")
            ui.table(wt.tail(40))
    for key, title in (("challenger_shadow_results", "Shadow challengers vs champion (same live fixtures)"),
                       ("champion_health", "Champion health"), ("model_churn", "Churn by model")):
        df = ml.get(key, pd.DataFrame())
        if len(df):
            with st.expander(title):
                ui.table(df)

# ── every study's latest verdict, newest first (Pro output/research_index.json) ──────────────
with tab_latest:
    ri = pro.research_index()
    if not ri.get("items"):
        st.info("No research index yet — Pro writes `output/research_index.json` after each study run.")
    else:
        st.markdown(
            "**What Pro has learned, newest first.** Each line is read from the study's own verdict file, "
            "rebuilt after every run — when a study learns something new, it appears here by itself.")
        st.markdown(
            "**Two results that look like they contradict each other, and don't:** for **team** markets "
            "(O/U, BTTS) the model adds nothing once the bookmaker's price is known (Team studies tab, V11 page). "
            "For **player props** Player V3 *does* add a little to the price — but only in accuracy; no bet has "
            "yet shown a profit, so props stay paper. Both are true; they are different markets.")
        ui.freshness_chip(ri["freshness"], "research index")
        for it in ri["items"]:
            with st.expander(f"**{it['title']}** — {it.get('status')}", expanded=False):
                st.caption(f"area: {it.get('area')} · study run: {it.get('date') or '?'}")
                key = it.get("key") or {}
                if key:
                    ui.table(pd.DataFrame([{"item": k, "value": str(v)} for k, v in key.items()]))
                if it.get("report_url"):
                    st.markdown(f"[Full report]({it['report_url']})")

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
        st.info("No studies yet — `pro_studies.yml` writes them daily.")
    else:
        st.caption("Research only, TEAM markets. Refreshed daily. Full write-up: `output/studies/REPORT.md` in Pro. "
                   "Player, calibration and price-trading studies are on the Latest findings tab.")
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
            st.caption("Negative Δ = adding the model improved the forecast. None does — on these TEAM markets. "
                       "Player props are different: see Player Model V3 on the Latest findings tab.")
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
