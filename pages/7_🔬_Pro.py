"""Pro — the evidence layer. Nothing here is a bet.

WHY THIS PAGE EXISTS. Pro has been running for months and the dashboard could not see it. The one
page that touched Pro read a single Bet Builder CSV. Everything Pro exists for — the canonical
season store, the walk-forward proof that retraining actually helps, the governance contract, the
register of results that turned out to be wrong — was invisible unless you opened the repo.

WHAT IT IS NOT. Pro does not stake and does not write into v9. Its numbers are evidence about the
system, not returns from it. The banner says so at the top and the page never prints a P/L.

Visual redesign 2026-10-25 (stage 3, nav label "Research"): page header with the RESEARCH state and the
research index's freshness; an evidence strip (forward tests, Price Trader, champions, tips sent) before the
tabs; every finding as a card with its n / evidence kind; the VALUABLE heatmap in the shared altair grammar.
SPEED: every Pro read is made concurrently (dashboard_data/research_view.load_all) and cached for five
minutes — the page measured ~25-37 s cold over HTTP with the reads made one after another. Same calls,
same values. PRESENTATION ONLY.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

import dashboard_ui as ui  # noqa: E402
from dashboard_data import pro  # noqa: E402
from dashboard_data import research_view as V  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY. A page and its helper changed in the same
# commit, and the server ran the NEW page against the OLD dashboard_data.pro still in memory:
# AttributeError on pro.tip_scoreboard (2026-10-08). Reloading this small module on every run
# makes the page and its helper always the same version.
import importlib  # noqa: E402
pro = importlib.reload(pro)
V = importlib.reload(V)

st.set_page_config(page_title="Wowza | Pro", page_icon="🔬", layout="wide")
_TITLE, _SUB = "Research", "Pro — the evidence layer. It validates and stores evidence; it never stakes and never writes into v9."
_EYEBROW = "Research Lab · Pro evidence & validation"
_STATE = ui.status_chip("RESEARCH", label="RESEARCH · NOT A BET")


@st.cache_data(ttl=300, show_spinner="Reading Pro…")
def _load() -> dict:
    return V.load_all(pro)


if not pro.available():
    ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW, freshness=_STATE)
    ui.empty_state("NOT_CONNECTED", pro.absent_reason())
    st.stop()

P = _load()
ri = P["research_index"] or {}
_rif = ri.get("freshness")
ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW,
               freshness=_STATE + (" " + ui.freshness_badge(_rif.state, f"research index · {_rif.label}") if _rif else ""))

contract = P["system_contract"]
if contract:
    va = contract.get("verified_against") or {}
    st.caption(
        f"System contract v{contract.get('contract_version', '?')} · generated "
        f"{contract.get('generated_at_utc', '?')} · verified against "
        f"v9 `{str(va.get('v9_commit', ''))[:7]}` · pro `{str(va.get('pro_commit', ''))[:7]}` · "
        f"v11 `{str(va.get('v11_commit', ''))[:7]}`")

# ═══ STATUS — the evidence strip ═════════════════════════════════════════════════════════════════
cf = P["candidate_forward"] or {}
pt = P["price_trader"] or {}
ml = P["model_lineage"] or {}
sb = P["tip_scoreboard"] or {}
_v = pt.get("verdict") or {}
_fg = _v.get("forward_gates") or {}
_a = ml.get("answers") or {}
_h1 = (sb.get("one_x_two") or {}).get("headline", {})
s1, s2 = st.columns(2), st.columns(2)
ui.research_card(title="Frozen forward tests", label="Candidate gates", value=f"{len(cf.get('hypotheses', []))}",
                 supporting=[("Status", str(cf.get("status_counts") or "—"))],
                 evidence=ui.evidence_badge("TRUE FORWARD", None, span=f"from {str(cf.get('forward_start'))[:10]}")
                 if cf else "", state="RESEARCH", container=s1[0])
ui.research_card(title="Price Trader", label="Evidence state", value=str(_v.get("EVIDENCE_STATE", "—")).replace("_", " "),
                 supporting=[("Forward trades", f"{_fg.get('N_completed_forward', 0)} / {_fg.get('N_required', 150)}"),
                             ("Match days", f"{_fg.get('match_days', 0)} / {_fg.get('days_required', 30)}")],
                 state="SHADOW", n=_fg.get("N_completed_forward", 0) if _v else None, container=s1[1])
ui.research_card(title="Model champions", label="Promoted", value=str(_a.get("PROMOTIONS", "—")),
                 supporting=[("Retrains analysed", str(_a.get("RETRAIN_ATTEMPTS_ANALYZED", "—"))),
                             ("Churn rate", str(_a.get("MODEL_CHURN_RATE", "—")))], state="RESEARCH", container=s2[0])
ui.research_card(title="1X2 tips sent", label="Hit rate", value=ui.pct(_h1.get("hit_rate")) if _h1.get("settled") else "—",
                 supporting=[("Won", f"{_h1.get('won', 0)} / {_h1.get('settled', 0)}"),
                             ("Flat stakes", ui.units(_h1["units"]) if _h1.get("settled") else "—"),
                             ("ROI", ui.pct(_h1.get("roi")) if _h1.get("settled") else "—")],
                 state="RESEARCH", n=_h1.get("settled") if _h1.get("settled") else None, container=s2[1])

(tab_latest, tab_champ, tab_trader, tab_valuable, tab_forward, tab_learn, tab_tips, tab_studies, tab_store, tab_gov,
 tab_health) = st.tabs(
    ["Latest findings", "Model champions", "Price Trader", "VALUABLE research", "Forward tests", "Does it learn?",
     "Tips sent (1X2 · Bet Builder)", "Team studies", "Canonical store", "Governance", "Health"])

# ── Frozen candidate forward tests (P1-4, 2026-10-22) - three windows NEVER mixed ─────────────────
with tab_forward:
    if not cf:
        ui.empty_state("NO_DATA", "No forward-test progress yet — Pro writes it daily (pro_valuable_shadow.yml).")
    else:
        st.markdown(f"**{len(cf.get('hypotheses', []))} frozen candidate gates** (registry frozen 2026-10-20, forward "
                    f"start {str(cf.get('forward_start'))[:10]}). Only TRUE FORWARD rows can pass or fail a test. "
                    "A retrospective winner is a candidate, never 'confirmed' — CONFIRMED cannot be assigned here.")
        st.caption(f"status: {cf.get('status_counts')} · updated {cf.get('generated_at')}")
        hyps = cf.get("hypotheses", [])
        for i in range(0, len(hyps), 2):
            for col, h in zip(st.columns(2), hyps[i:i + 2]):
                V.forward_card(h, container=col)

# ── VALUABLE deep dive (owner brief 2026-10-20) — READ-ONLY research, selection logic untouched ─────
with tab_valuable:
    vd = P["valuable"] or {}
    sm = vd.get("valuable_summary", pd.DataFrame())
    if sm.empty:
        ui.empty_state("NO_DATA", "No VALUABLE deep dive yet — Pro writes `output/valuable_deep_dive/` (src/studies/valuable).")
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
                ui.metric_card(lab, f"{r.ROI_pct:+.1f}%", title="VALUABLE · pooled ROI", delta_color="off",
                               delta=f"n {int(r.N)} · 90% CI {r.ROI_ci90_lo:+.0f}..{r.ROI_ci90_hi:+.0f}",
                               supporting=[("CLV %", f"{r.CLV_pct}" if "CLV_pct" in r.index else "—")],
                               status=ui.status_chip("PAPER", label="PAPER · NEVER STAKED"), n=int(r.N), container=col)
        st.caption("Pooled across markets, as the owner audit counts it. By market below.")
        ui.table(val[~val.market.str.startswith("ALL")][["window", "market", "N", "ROI_pct", "ROI_ci90_lo", "ROI_ci90_hi", "CLV_pct"]]
                 .rename(columns={"ROI_pct": "ROI %", "ROI_ci90_lo": "CI low", "ROI_ci90_hi": "CI high", "CLV_pct": "CLV %"}))
        ui.section_header("Edge bands (O/U 2.5 VALUABLE since 26 Aug)", container=st)
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
            ui.section_header("League × market heatmap (shrunk ROI, VALUABLE, all live history)", container=st)
            hd = V.valuable_heat_frame(lm)
            if not hd.empty:
                st.altair_chart(V.valuable_heatmap(hd), width="stretch")
            st.caption("Cells with ≥10 settled tips, shrunk toward their tier × market × model average. "
                       "No cell survives multiple-testing control (best q 0.61).")
        sh = vd.get("shadow_live") or vd.get("shadow") or {}
        st.caption(f"Low-edge shadow (every candidate incl. below the floor, never bet): "
                   f"{sh.get('rows', 0)} rows · {sh.get('settled', 0)} settled · status "
                   f"{(vd.get('shadow') or {}).get('status', '?')}")

# ── Price Trader research lab (owner brief 2026-10-19) — PAPER / SHADOW only ────────────────────
with tab_trader:
    v = _v
    if not v:
        ui.empty_state("NO_DATA", "No Price Trader v2 verdict yet — Pro writes it daily (pro_price_trader.yml).")
    else:
        state = v.get("EVIDENCE_STATE", "?")
        st.markdown(V.trader_chip(state) + " " + ui.status_chip("SHADOW", label="PAPER / SHADOW ONLY"))
        st.caption("Paper/shadow research only: no orders, no stakes, no messages. Raw price movement is never "
                   "called an edge without the economic test (costs, breakeven, controls, forward window).")
        ds = v.get("data_status") or {}
        h = pt.get("health") or {}
        ui.metric_grid([
            {"label": "Signals with a price path", "value": h.get("tradable", "?")},
            {"label": "Exchange back prices", "value": ds.get("EXCHANGE_BACK_PRICES_RECORDED", "?"),
             "help": f"share of path rows: {ds.get('exchange_opposite_coverage_of_path_rows')}"},
            {"label": "Executable exchange data", "value": ds.get("EXECUTABLE_EXCHANGE_DATA_AVAILABLE", "?"),
             "help": "no lay ladder / matched size is captured anywhere"},
            {"label": "Capacity", "value": "UNKNOWN" if ds.get("CAPACITY_UNKNOWN") == "YES" else "known"},
        ], per_row=2)
        hp = v.get("historical_primary") or {}
        fg = v.get("forward_gates") or {}
        st.markdown(f"**Primary setup (historical, exploratory):** {hp.get('n')} hedged trades, mean "
                    f"{(hp.get('mean_pnl_per_unit') or 0):+.1%} per unit, full-green rate "
                    f"{(hp.get('full_green_rate') or 0):.1%}.  **Forward window** (from 2026-10-20): "
                    f"{fg.get('N_completed_forward', 0)} of {fg.get('N_required', 150)} trades, "
                    f"{fg.get('match_days', 0)} of {fg.get('days_required', 30)} match days.")
        st.markdown(ui.evidence_badge("TRUE FORWARD", fg.get("N_completed_forward", 0),
                                      span=f"{fg.get('match_days', 0)} match days", state=state)
                    + " " + ui.sample_marker(fg.get("N_completed_forward", 0) or 0))
        hz = pt.get("horizon_rule_results")
        if hz is not None and len(hz):
            sub = hz[(hz["venue"] == "book") & (hz["slippage"] == 0) & (hz["rule"] == "GREEN_2PCT")]
            ui.eyebrow("By entry horizon — observed favourable rate vs the rate needed to break even after costs")
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
    ui.section_header("Model champions — the high-water mark", container=st)
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
    a = _a
    if a:
        ui.metric_grid([{"label": "Retrains analysed", "value": a.get("RETRAIN_ATTEMPTS_ANALYZED")},
                        {"label": "Promoted", "value": a.get("PROMOTIONS")},
                        {"label": "Churn rate", "value": a.get("MODEL_CHURN_RATE")},
                        {"label": "Head-to-head graded", "value": a.get("H2H_GRADED")}], per_row=2)
        st.caption(f"Best proven improvement: {a.get('BEST_PROVEN_CHAMPION_IMPROVEMENT')}")
    view = ml.get("current_champions_view", pd.DataFrame())
    if len(view):
        ui.table(view.rename(columns=str))
    elif _ptr:
        ui.table(pd.DataFrame([{"model": k, "champion": v.get("model_version"),
                                "since": v.get("champion_since")} for k, v in sorted(_ptr.items())]))
    else:
        ui.empty_state("NO_DATA", "No champion registry yet.")

    # lineage timeline: one line per model, last 12 decisions, oldest -> newest
    if len(_dec):
        ui.eyebrow("Lineage — last 12 retrains per model (oldest → newest)")
        _sym = {"PROMOTE": "⬆️ promoted", "KEEP_CHAMPION": "⏸ kept", "SHADOW_MORE": "🔬 shadow",
                "LEGACY_PROMOTED": "↑ old-rule promote", "LEGACY_BLOCKED": "⛔ old-rule block"}
        for mt, g in _dec.groupby("model_type"):
            chain = " → ".join(_sym.get(str(v), str(v)) for v in g["verdict"].tail(12))
            st.markdown(f"- **{mt}**: {chain}")
        st.caption("Before 2026-10-18 the old rule promoted almost every retrain (each champion lived about a "
                   "day). From 2026-10-18 only a proven improvement promotes.")
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
    if not ri.get("items"):
        ui.empty_state("NO_DATA", "No research index yet — Pro writes `output/research_index.json` after each study run.")
    else:
        st.markdown(
            "**What Pro has learned, newest first.** Each line is read from the study's own verdict file, "
            "rebuilt after every run — when a study learns something new, it appears here by itself.")
        st.markdown(
            "**Two results that look like they contradict each other, and don't:** for **team** markets "
            "(O/U, BTTS) the model adds nothing once the bookmaker's price is known (Team studies tab, Market Lab "
            "page). For **player props** Player V3 *does* add a little to the price — but only in accuracy; no bet has "
            "yet shown a profit, so props stay paper. Both are true; they are different markets.")
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
    ui.section_header("Does retraining actually help?", container=st)
    sl = P["shadow_learning"] or {}
    if not sl.get("available"):
        ui.empty_state("NO_DATA", sl.get("why", "no walk-forward results"))
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
                width="stretch")
            # AUC and log loss live on different scales, so they get their OWN chart rather than
            # a second y-axis. A twin axis can be tuned to show any relationship the author wants.
            au = d[d["metric"].isin(["auc", "pr_auc"])].copy()
            st.altair_chart(
                ui.compare_bars(au, "variant", "value", "metric",
                                title="AUC and PR-AUC — higher is better  (separate chart: "
                                      "different scale, never a second axis)", y_title="score"),
                width="stretch")
        with right:
            best = sl.get("best_by_logloss")
            ui.metric_card("Best by log loss", str(best).title(), title="Walk-forward winner",
                           supporting=[("Months", f"{sl['months']}")],
                           evidence=ui.evidence_badge("WALK-FORWARD", None, span=f"over {sl['months']} walk-forward months"))
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
    cs = P["canonical_stores"] or {}
    if not cs.get("available"):
        ui.empty_state("NO_DATA", cs.get("why", "no season store"))
    else:
        ui.section_header(f"Canonical season store — {cs['season'].replace('_', ' ')}", container=st)
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
                                width="stretch")
            with c2:
                ui.eyebrow("Freshness by store")
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
        ui.empty_state("NO_DATA", "No system contract found in Pro's registry.")
    else:
        ui.section_header("What is allowed to happen, and what is proven", container=st)

        rm = contract.get("real_money") or {}
        st.markdown(
            f"**Automated bet placement exists: "
            f"{'YES' if rm.get('automated_placement_exists') else 'NO'}** — "
            f"evidence `{rm.get('evidence', '?')}`.")
        st.caption(rm.get("proof", ""))

        inv = contract.get("invariants") or []
        if inv:
            ui.eyebrow("Invariants and how well each is actually enforced")
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
            ui.eyebrow("Known ambiguities — things the code cannot settle")
            st.caption("Code proves behaviour. It does not prove intent. These need a human "
                       "answer and are listed rather than guessed at.")
            ui.table(pd.DataFrame(amb))

# ── health ────────────────────────────────────────────────────────────────────────────────────
with tab_health:
    ui.section_header("Pro's own health artifacts", container=st)
    st.caption("Green CI does not prove the instrument is measuring anything. v11's momentum "
               "work exited 0 for four weeks while measuring nothing at all. These files record "
               "what each pipeline believed about itself when it ran.")
    ha = P["health_artifacts"] or {}
    files = ha.get("files") or {}
    if not files:
        ui.empty_state("NO_DATA", "No health artifacts found in Pro's output.")
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
    if not sb:
        ui.empty_state("NO_DATA", "No tip scoreboard yet — it is written by `pro_tip_scoreboard.yml` each morning.")
    else:
        if sb.get("freshness"):
            ui.freshness_chip(sb["freshness"], "tip scoreboard")
        st.caption("Graded only on tips that were actually SENT to Telegram. Paper leagues (USA MLS) excluded.")
        a, b = sb.get("one_x_two", {}), sb.get("bet_builder", {})
        h, bh = a.get("headline", {}), b.get("headline", {})
        ui.section_header("1X2 tips", container=st)
        if h.get("settled"):
            ci = h.get("roi_ci90") or [None, None]
            c = st.columns(2)
            ui.metric_card("Hit rate", ui.pct(h["hit_rate"]), title="1X2 · tips sent",
                           supporting=[("Won", f"{h['won']} / {h['settled']}"), ("Pending", f"{a.get('pending', 0)} pending"),
                                       ("Break-even", ui.pct(h["break_even"]))],
                           n=h["settled"], container=c[0])
            ui.metric_card("Flat stakes", ui.units(h["units"]), title="Result at flat 1u",
                           supporting=[("ROI", ui.pct(h["roi"])),
                                       ("ROI 90% range", "–" if ci[0] is None else f"{ci[0]:+.0%} … {ci[1]:+.0%}")],
                           status=ui.status_chip("SMALL_SAMPLE", label="TOO FEW TIPS TO JUDGE")
                           if h["settled"] < 50 else "", n=h["settled"], container=c[1])
            last = pd.DataFrame(a.get("last_10", []))
            if not last.empty:
                ui.table(last)
        else:
            ui.empty_state("NO_DATA", "No settled 1X2 tips yet.")
        ui.section_header("Bet Builder combos", container=st)
        if bh.get("settled"):
            v = b.get("with_voided_legs", {})
            ui.metric_card("Won (fully graded)", f"{bh['won']} / {bh['settled']}", title="Bet Builder · tips sent",
                           supporting=[("Hit rate", ui.pct(bh["hit_rate"])),
                                       ("Model expected", f"{bh['expected_wins']:.1f} wins"),
                                       ("z", f"z = {bh['z_actual_vs_claimed']}"),
                                       ("With a voided leg", str(v.get("settled", 0))),
                                       ("'Won' on the remaining legs", f"{v.get('won', 0)} 'won' on the remaining legs")],
                           n=bh["settled"])
            st.caption("Combos with a voided player leg are settled on the remaining legs, so they are kept out of "
                       "the hit rate. No bookmaker builder price is recorded, so builder P/L cannot be measured.")
        else:
            ui.empty_state("NO_DATA", "No fully graded combos yet.")

# ── the October 2026 upgrade studies (src/studies) ───────────────────────────────────────────
with tab_studies:
    S = P["studies"] or {}
    if not any(S.values()):
        ui.empty_state("NO_DATA", "No studies yet — `pro_studies.yml` writes them daily.")
    else:
        st.caption("Research only, TEAM markets. Refreshed daily. Full write-up: `output/studies/REPORT.md` in Pro. "
                   "Player, calibration and price-trading studies are on the Latest findings tab.")
        ab = S.get("argentina_btts") or {}
        if ab:
            ui.section_header("1 · Argentina BTTS — model, league, bookmaker, or luck?", container=st)
            rates = ab.get("btts_rate_all_matches", {})
            t = ab.get("tip_period", {})
            reg = (ab.get("regime_by_period") or {}).get("2026 since Aug 10 (tip period)", {})
            bb = t.get("B_blind_yes_all", {})
            c = st.columns(2)
            ui.metric_card("Wowza picks minus the rest", ui.pct(t.get("A_minus_not_selected_roi")), title="Selection skill",
                           supporting=[("Blind YES, every match", ui.pct(bb.get("roi"))), ("n", f"n {bb.get('n')}")],
                           status=ui.status_chip("NO_DATA", label="RANGE INCLUDES ZERO", tone="NEUTRAL")
                           if (t.get("A_minus_not_selected_ci90") or [0])[0] < 0 else "", container=c[0])
            ui.metric_card("BTTS rate 2025", ui.pct(rates.get("2025", {}).get("rate")), title="League regime",
                           supporting=[("Since 10 Aug", f"since 10 Aug: {ui.pct(rates.get('2026 since Aug 10', {}).get('rate'))}"),
                                       ("Price implied", ui.pct(reg.get("bet365_fair_yes_close"))),
                                       ("Realised", f"realised {ui.pct(reg.get('btts_rate'))} · gap {reg.get('gap_pp')} pts")],
                           container=c[1])
            st.markdown("**Reading:** the profit is a league scoring regime the market prices slowly, not model "
                        "selection. If the league calms down, the edge goes.")
        ch = S.get("ou_challenger") or {}
        if ch.get("probabilistic_oos"):
            ui.section_header("2 · O/U 2.5 — v9 vs the market-anchored challenger", container=st)
            tv, tc = ch["tips_at_5pct_edge"]["v9"], ch["tips_at_5pct_edge"]["challenger"]
            w = ch.get("current_weights", {})
            c = st.columns(2)
            ui.metric_card("v9 result", ui.units(tv.get("units", 0)), title="v9 tips at a 5% edge",
                           supporting=[("Tips", str(tv.get("n"))), ("UNDER share", f"{ui.pct(tv.get('under_share'))} UNDER"),
                                       ("ROI", f"ROI {ui.pct(tv.get('roi'))}")],
                           n=tv.get("n"), container=c[0])
            ui.metric_card("Challenger tips", str(tc.get("n")), title="Anchored on the market",
                           supporting=[("Model weight", str(w.get("c_model"))),
                                       ("Unconstrained", f"unconstrained {w.get('c_model_unconstrained')}")],
                           container=c[1])
            st.markdown("**Reading:** v9's O/U probability is squashed near 52%, so 'UNDER edges' are matches the "
                        "market rates high-scoring — and the market is right. Owner decision 2026-10-08: wait 2–3 weeks "
                        "of the challenger's forward record before any change to v9.")
        ou = S.get("ou_studies") or {}
        if ou.get("residual"):
            ui.section_header("3 · Does the model add anything once the price is known?", container=st)
            rows = [{"Market": m, "Track": tr, "n": x.get("n"), "Δ log loss (out of sample)": x.get("oos_delta_logloss"),
                     "90% range": str(x.get("oos_delta_logloss_ci90")), "Verdict": x.get("verdict")}
                    for m, by in ou["residual"].items() for tr, x in by.items() if x.get("n_oos")]
            ui.table(pd.DataFrame(rows))
            st.caption("Negative Δ = adding the model improved the forecast. None does — on these TEAM markets. "
                       "Player props are different: see Player Model V3 on the Latest findings tab.")
        ev = S.get("evidence") or {}
        if ev.get("cells"):
            ui.section_header("4 · Which league × market has earned money?", container=st)
            st.caption(f"{ev.get('n_cells_searched')} cells tested · reality-check p for the best cell "
                       f"{ev.get('reality_check_p_best_cell')} · gates: n ≥ 150, P(edge>0) ≥ 0.80, FDR q ≤ 0.10")
            rows = [{"League": c["league"], "Market": c["market"], "n": c["n"], "Raw ROI": c["roi"],
                     "Shrunk ROI": c.get("posterior_mean"), "P(edge>0)": c.get("p_edge_gt_0"),
                     "FDR q": c.get("q_bh"), "Recommendation": c.get("recommendation")}
                    for c in ev["cells"] if c.get("testable")]
            ui.table(pd.DataFrame(rows))
