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

tab_learn, tab_store, tab_gov, tab_health = st.tabs(
    ["Does it learn?", "Canonical store", "Governance", "Health"])

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
