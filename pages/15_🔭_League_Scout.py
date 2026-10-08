"""League Scout — the ~200 leagues Wowza watches but does not bet, and how far each has got.

WHAT IT IS. Pro's scout (src/scout) records odds (Pinnacle, Bet365, cross-book median) for O/U
1.5/2.5/3.5 and BTTS every two hours, stores results and five seasons of history, and freezes a
baseline goals model's probability for each fixture before kickoff. Historical odds cannot be
bought back, so a league's market can only be judged on prices recorded from the day the scout
started (2026-10-08).

HOW A LEAGUE IS JUDGED, once it has settled fixtures with a captured close:
    market skill   how much better the closing price is than the league's base rate
    residual z     does the model add information to the close? (> ~2 = yes)
    paper ROI      model edge > 5 points at Bet365, flat stakes — and how one-sided it is

Nothing here is a bet. A league moves toward v9 only as a paper league, and only at READABLE.
Reads one small JSON written by the scout report; never the scout's parquet store.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_data import pro, ui

st.set_page_config(page_title="Wowza | League Scout", page_icon="🔭", layout="wide")
st.title("🔭 League Scout")

if not pro.available():
    ui.absent(pro.absent_reason())
    st.stop()

ui.state_banner("RESEARCH", "Collecting evidence on leagues Wowza does not bet. Nothing here is a tip.")

s = pro.scout_status()
if not s.get("available"):
    ui.absent(s.get("why", "no scout data"))
    st.stop()
if s.get("freshness"):
    ui.freshness_chip(s["freshness"], "scout report")
run = pro.scout_last_run()
if run:
    st.caption(f"Last scout run: {run.get('finished', '?')[:16]} UTC · {run.get('api_calls', '?')} API calls · "
               f"status {run.get('status', '?')} · quota left {run.get('quota_remaining', '?')}")

L = pd.DataFrame(s["leagues"])
STAGES = ["READABLE", "EARLY", "COLLECTING", "PRICING", "HISTORY_ONLY", "NOT_STARTED"]
ICON = {"READABLE": "🟢", "EARLY": "🟡", "COLLECTING": "🔵", "PRICING": "🔵",
        "HISTORY_ONLY": "⚪", "NOT_STARTED": "⚪"}

# ── summary ───────────────────────────────────────────────────────────────────────────────────
c = st.columns(6)
with c[0]:
    ui.kpi("Leagues watched", str(len(L)), f"{int((L['priority'] == 1).sum())} priority 1")
with c[1]:
    ui.kpi("Pricing now", str(int((L["upcoming_priced"] > 0).sum())), "leagues with upcoming odds")
with c[2]:
    ui.kpi("Fixtures priced", f"{int(L['fixtures_priced'].sum()):,}", "since 2026-10-08")
with c[3]:
    ui.kpi("Results stored", f"{int(L['results'].sum()):,}", "history + this season")
with c[4]:
    ui.kpi("Model frozen", f"{int(L['model_frozen'].sum()):,}", "probabilities before kickoff")
with c[5]:
    ui.kpi("Readable", str(int((L["status"] == "READABLE").sum())), "300+ settled with a close")

with st.expander("What the stages mean"):
    for k in STAGES:
        if k in (s.get("status_meaning") or {}):
            st.markdown(f"{ICON[k]} **{k}** — {s['status_meaning'][k]}")
    st.markdown("A verdict needs about **8–12 weeks** of collection for a busy league, longer for a small one.")

# ── filters ───────────────────────────────────────────────────────────────────────────────────
f1, f2, f3, f4 = st.columns([1, 2, 2, 2])
with f1:
    prio = st.selectbox("Priority", ["All", "1 — owner's list", "2"], key="scout_prio")
with f2:
    countries = st.multiselect("Country", sorted(L["country"].unique()), key="scout_country")
with f3:
    stages = st.multiselect("Stage", [x for x in STAGES if x in set(L["status"])], key="scout_stage")
with f4:
    q = st.text_input("Search league", key="scout_q")
V = L.copy()
if prio.startswith("1"):
    V = V[V["priority"] == 1]
elif prio == "2":
    V = V[V["priority"] == 2]
if countries:
    V = V[V["country"].isin(countries)]
if stages:
    V = V[V["status"].isin(stages)]
if q:
    V = V[V["league"].str.contains(q, case=False) | V["country"].str.contains(q, case=False)]

order = {k: i for i, k in enumerate(STAGES)}
V = V.assign(_o=V["status"].map(order)).sort_values(["_o", "priority", "settled_with_close", "fixtures_priced"],
                                                     ascending=[True, True, False, False])
show = pd.DataFrame({
    "Stage": V["status"].map(lambda x: f"{ICON.get(x, '')} {x}"),
    "Priority": V["priority"], "Country": V["country"], "League": V["league"],
    "Settled with close": V["settled_with_close"], "Upcoming priced": V["upcoming_priced"],
    "Fixtures priced": V["fixtures_priced"], "Results": V["results"],
    "History seasons": V["history_seasons"], "Model frozen": V["model_frozen"],
    "Last price": V["last_price_seen"].fillna("–").astype(str).str[:16],
})
st.subheader(f"{len(V)} league(s)")
ui.table(show)

# ── per-league verdicts ──────────────────────────────────────────────────────────────────────
withv = V[V["markets"].map(lambda m: bool(m))]
st.subheader("Verdicts by market")
if withv.empty:
    st.info("No league has settled fixtures with a captured closing price yet. Verdicts appear here as "
            "leagues reach COLLECTING (any settled), EARLY (100+) and READABLE (300+).")
else:
    rows = []
    for _, r in withv.iterrows():
        for mk, m in r["markets"].items():
            res = m.get("residual") or {}
            rows.append({"League": f"{r['country']} — {r['league']}", "Market": mk, "n": m.get("n"),
                         "Stage": m.get("status"), "Rate": m.get("rate"),
                         "Market skill vs base rate": m.get("market_skill_vs_base"),
                         "Model residual z": res.get("z"), "Paper bets": m.get("paper_bets"),
                         "Paper ROI": m.get("paper_roi"), "Share on one side": m.get("paper_first_side_share")})
    ui.table(pd.DataFrame(rows))
    st.caption("Market skill near 0 = the closing price barely beats the league average (a soft market). "
               "Residual z above ~2 = the model knows something the close did not. A paper record that is "
               "almost all one side is the failure signature to distrust.")
