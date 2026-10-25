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

Visual redesign 2026-10-25 (stage 3): page header with the scout report's freshness, a stage funnel and
three summary cards first, stage chips (glyph + word, no colour-only circles), secondary filters under
"Advanced filters", and the remote reads cached for five minutes. PRESENTATION ONLY.
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

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY. A page and its helper changed in the same
# commit, and the server ran the NEW page against the OLD dashboard_data.pro still in memory:
# AttributeError on pro.tip_scoreboard (2026-10-08). Reloading this small module on every run
# makes the page and its helper always the same version.
import importlib  # noqa: E402
pro = importlib.reload(pro)

st.set_page_config(page_title="Wowza | League Scout", page_icon="🔭", layout="wide")
_TITLE, _SUB = "League Scout", "Collecting evidence on leagues Wowza does not bet. Nothing here is a tip."
_EYEBROW = "Research Lab · league expansion"
_STATE = ui.status_chip("RESEARCH", label="RESEARCH · NOT A TIP")


@st.cache_data(ttl=300, show_spinner="Reading the scout report…")
def _load() -> dict:
    s = pro.scout_status()
    return {"status": s, "run": pro.scout_last_run() if s.get("available") else {},
            "promotion": (pro.promotion_review() if hasattr(pro, "promotion_review") else {}) if s.get("available") else {}}


if not pro.available():
    ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW, freshness=_STATE)
    ui.empty_state("NOT_CONNECTED", pro.absent_reason())
    st.stop()

D = _load()
s = D["status"]
if not s.get("available"):
    ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW, freshness=_STATE)
    ui.empty_state("NOT_CONNECTED", s.get("why", "no scout data"))
    st.stop()
_f = s.get("freshness")
ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW,
               freshness=_STATE + " " + (ui.freshness_badge(_f.state, f"scout report · {_f.label}") if _f
                                         else ui.freshness_badge("UNKNOWN", "scout report · no completed run")))
run = D["run"]
if run:
    st.caption(f"Last scout run: {run.get('finished', '?')[:16]} UTC · {run.get('api_calls', '?')} API calls · "
               f"status {run.get('status', '?')} · quota left {run.get('quota_remaining', '?')}")

L = pd.DataFrame(s["leagues"])
STAGES = ["READABLE", "EARLY", "COLLECTING", "PRICING", "HISTORY_ONLY", "NOT_STARTED"]
#: glyph + word (the word carries the stage; the glyph is a secondary cue, never colour alone)
ICON = {"READABLE": "✓", "EARLY": "◔", "COLLECTING": "◌", "PRICING": "◌",
        "HISTORY_ONLY": "○", "NOT_STARTED": "○"}

# ═══ STATUS — how far the watch list has got ═════════════════════════════════════════════════════
c = st.columns(3)
ui.metric_card("Readable", str(int((L["status"] == "READABLE").sum())), title="300+ settled with a close",
               supporting=[("Leagues watched", str(len(L))), ("Priority 1", f"{int((L['priority'] == 1).sum())}")],
               container=c[0])
ui.metric_card("Pricing now", str(int((L["upcoming_priced"] > 0).sum())), title="Leagues with upcoming odds",
               supporting=[("Fixtures priced", f"{int(L['fixtures_priced'].sum()):,}"), ("Since", "2026-10-08")],
               container=c[1])
ui.metric_card("Results stored", f"{int(L['results'].sum()):,}", title="History + this season",
               supporting=[("Model frozen (probabilities before kickoff)", f"{int(L['model_frozen'].sum()):,}")],
               container=c[2])

_funnel = (L["status"].value_counts().reindex(STAGES).dropna().astype(int).rename("leagues")
           .rename_axis("stage").reset_index())
if len(_funnel):
    left, right = st.columns([3, 2])
    with left:
        # A bar column, not a chart: the same picture without loading a chart library on a page that
        # never needed one (keeps the page as light as it was).
        ui.eyebrow("Leagues by stage")
        ui.compact_table(_funnel.assign(stage=_funnel["stage"].map(lambda x: f"{ICON.get(x, '')} {x}")),
                         {"leagues": ui.bar_col("Leagues", max_value=float(len(L)), fmt="%d")})
    with right:
        ui.eyebrow("What the stages mean")
        for k in STAGES:
            if k in (s.get("status_meaning") or {}):
                st.markdown(f"**{ICON[k]} {k}** — {s['status_meaning'][k]}")
        st.caption("A verdict needs about **8–12 weeks** of collection for a busy league, longer for a small one.")

# ═══ DETAIL — every league ═══════════════════════════════════════════════════════════════════════
f3, f4 = st.columns(2)
with f3:
    stages = st.multiselect("Stage", [x for x in STAGES if x in set(L["status"])], key="scout_stage")
with f4:
    q = st.text_input("Search league", key="scout_q")
with st.expander("Advanced filters"):
    f1, f2 = st.columns([1, 2])
    with f1:
        prio = st.selectbox("Priority", ["All", "1 — owner's list", "2"], key="scout_prio")
    with f2:
        countries = st.multiselect("Country", sorted(L["country"].unique()), key="scout_country")
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
ui.section_header(f"{len(V)} league(s)")
ui.table(show, column_config={"Settled with close": ui.bar_col("Settled with close", max_value=300, fmt="%d",
                                                               help="READABLE at 300 settled fixtures with a close")})

# ═══ RESEARCH — per-league verdicts ══════════════════════════════════════════════════════════════
withv = V[V["markets"].map(lambda m: bool(m))]
ui.section_header("Verdicts by market", "Market skill near 0 = the closing price barely beats the league average (a "
                  "soft market). Residual z above ~2 = the model knows something the close did not.")
if withv.empty:
    ui.empty_state("NO_DATA", "No league has settled fixtures with a captured closing price yet. Verdicts appear here "
                              "as leagues reach COLLECTING (any settled), EARLY (100+) and READABLE (300+).")
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
    st.caption("A paper record that is almost all one side is the failure signature to distrust. A paper ROI is "
               "never read without its n beside it.")

# ═══ RESEARCH — weekly promotion review (Pro src/scout/promotion) ════════════════════════════════
ui.section_header("Weekly promotion review")
pr = D["promotion"]
if not pr:
    ui.empty_state("NO_DATA", "No weekly promotion review yet — it runs every Sunday in Pro.")
else:
    if pr.get("freshness"):
        ui.freshness_chip(pr["freshness"], "promotion review")
    k = pr.get("counts", {})
    ui.metric_grid([{"label": lab, "value": k.get(key, 0)}
                    for lab, key in (("New candidates", "new_candidates"), ("Approved (shadow)", "promoted"),
                                     ("Validation failed", "failed"), ("Data-quality blocked", "blocked"),
                                     ("Still collecting", "still_collecting"))], per_row=3)
    st.caption(f"Week {pr.get('week')} · {pr.get('leagues_monitored')} leagues monitored · a league that passes stops "
               f"at SHADOW_APPROVED (paper only) until the existing-league regression test exists — no v9 config "
               f"change and no money is ever decided here.")
    det = pr.get("details", {})
    rows = []
    for name, d in det.items():
        dist = d.get("distance") or {}
        rows.append({"League": name, "State": d.get("new_state"), "Scout": d.get("scout_status", ""),
                     "Score": d.get("promotion_score"),
                     "Settled w/ close needed": dist.get("settled_with_close"),
                     "Results needed": dist.get("results"), "Seasons needed": dist.get("history_seasons"),
                     "Why": "; ".join(map(str, d.get("reason", [])))[:120]})
    if rows:
        T = pd.DataFrame(rows)
        T["_d"] = T[["Settled w/ close needed", "Results needed"]].fillna(0).sum(axis=1)
        ui.eyebrow("Distance to promotion — closest leagues first")
        ui.table(T.sort_values(["Score", "_d"], ascending=[False, True], na_position="last")
                 .drop(columns="_d").head(25))
    reg = pr.get("registry") or {}
    if reg.get("divergences"):
        with st.expander(f"League registry: {len(reg['divergences'])} list divergence(s) in v9 config"):
            for d in reg["divergences"]:
                st.write("- " + d)
