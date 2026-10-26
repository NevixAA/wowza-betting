"""Leagues — every league's results, tips, rating and averages, then one league in detail.

Owner request 2026-10-25: "a page for it in the app? and telegram? 'leagues' with the ability to see
each league results and tip and rating and avg's". The numbers come from dashboard_data/leagues.py,
the same functions the Monday Telegram "Leagues" picture draws (telegram_bot/daily_pack.leagues_image),
so the app and Telegram cannot disagree.

Basis (the same as the rest of the app): live tips from the two committed ledgers, settled, match
date from PERFORMANCE_CUTOFF_DATE, flat 1u. Results = staked tiers (SNIPER + MARKSMAN); VALUABLE is
shown on its own line as paper. A paper league (config.PAPER_LEAGUES) is shown, labelled, and kept
out of every all-league total.

Visual redesign 2026-10-25 (stage 2b): league cards with chips and a trend line, the same overview
table with numeric columns, and a league profile of three cards instead of ten equal metrics.
PRESENTATION ONLY - components live in dashboard_data/leagues_view.py.

A VIEWER: two small ledgers, two small JSON configs, Pro's ratings JSON and one small Pro CSV.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES) - same tier as the Tips dashboard

import dashboard_ui as ui  # noqa: E402
from src.tier_labels import GAP_EXPLAINER, PAPER_TIER_LABEL, STAKED_LABEL  # noqa: E402
from dashboard_data import core, leagues  # noqa: E402
from dashboard_data import leagues_view as V  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helper
# so the page and its data module are always the same version.
import importlib  # noqa: E402
leagues = importlib.reload(leagues)
V = importlib.reload(V)

st.set_page_config(page_title="Wowza | Leagues", page_icon="🏟️", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def _load(today: str):
    d = leagues.settled()
    up = leagues.upcoming(today, 7)
    rat = leagues.ratings()
    pol, reg = leagues.policy(), leagues.regimes()
    ov = leagues.overview(d, today, up=up, rat=rat, pol=pol, reg=reg)
    tr = V.trends(d, ov["league"].tolist()) if len(ov) else {}
    return d, up, rat, pol, reg, ov, tr


@st.cache_data(ttl=1800, show_spinner=False)
def _pro_weekly():
    return leagues.pro_weekly()


from src.tip_day import today as _tip_today  # noqa: E402
TODAY = _tip_today()
_SUB = (f"Every league's record, tips, rating and averages. Results are staked tiers ({STAKED_LABEL}), settled, "
        f"flat 1u, from the performance cutoff; {PAPER_TIER_LABEL} is paper and shown on its own line. Paper leagues "
        f"are shown and labelled, never counted in totals. {GAP_EXPLAINER} Not betting advice.")
try:
    d, up, rat, pol, reg, ov, tr = _load(TODAY)
except Exception as e:                                                  # noqa: BLE001 - a viewer never dies
    ui.page_header("Leagues", _SUB, eyebrow_text="Results by league")
    ui.empty_state("ERROR", f"League data unavailable ({type(e).__name__}: {e})")
    st.stop()

weekly = _pro_weekly()
_stamps = leagues.freshness_stamps(weekly)
ui.page_header("Leagues", _SUB, eyebrow_text="Results by league", freshness=V.freshness_header(_stamps))

if ov.empty:
    ui.empty_state("NO_DATA", "No league data yet.")
    st.stop()

# ── headline ──────────────────────────────────────────────────────────────────────────────────
tot = leagues.totals(d)
ui.metric_card("P/L", ui.units(tot["pl"]), title="All leagues · staked tiers", help="Flat 1u stakes",
               supporting=[("Staked tips settled", f"{tot['tips']:,}"), ("Hit", V.fmt_pct(tot["hit"], 1)),
                           ("ROI", V.fmt_roi(tot["roi"])), ("W-L", V.wl_help(tot["won"], tot["lost"], tot["void"])),
                           ("Leagues", f"{int((ov['tips'] > 0).sum())} with staked tips · {len(ov)} listed")],
               status=ui.status_chip("LIVE", label=f"STAKED · {STAKED_LABEL.upper()}") + " "
               + ui.status_chip("PAPER", label="PAPER LEAGUES EXCLUDED"))

# ── 1. all-leagues overview ───────────────────────────────────────────────────────────────────
ui.section_header("All leagues", "Best P/L first, paper leagues last. Cards carry the decision; the table is "
                  "the same one the Monday Telegram picture draws, sortable by any column.")
tab = leagues.overview_table(ov)          # the SAME table the Monday Telegram picture draws
tab = tab.assign(Trend=[tr.get(lg) for lg in ov["league"]])
_cards, _table = st.tabs(["Cards", "Table"])
with _cards:
    V.league_cards(ov, tr)
with _table:
    ui.compact_table(tab, V.table_config(tab, "Trend"), height=min(38 * (len(tab) + 1) + 4, 860))
st.caption("**Rating** = Pro's league rating 1-10 (the stake plan's sizing; '—' when missing or older than 8 days). "
           "**State** = production state from the bankroll policy (the strongest market of the league's track; detail "
           "below). **Regime** = research regime (PROTECTED / REPAIR / LEARNING, count of markets) - labels where "
           "research effort goes, never a stake. **Model % vs Implied %** = average model probability against 1/odds "
           "(O/U 2.5 model probability derived as 1/odds + edge). Avg edge mixes O/U 2.5 (vs the vigged price) and side "
           "markets (vs the de-vigged price) - compare per market below. **Upcoming** = sent tips, tip days "
           f"{TODAY} to +7.")

# ── 2. one league ────────────────────────────────────────────────────────────────────────────
ui.section_header("League profile")
names = ov["league"].tolist()
pick = st.selectbox("League", names, format_func=lambda n: n + (" (paper league)" if leagues.config.is_paper_league(n)
                                                                 else ""), key="lg_pick")
row = ov[ov["league"] == pick].iloc[0]
g = d[d["league"] == pick]

ui.eyebrow(pick)
st.markdown(" ".join(x for x in (V.state_chip(row["state"]), V.regime_chip(row["regime"]), V.track_chip(row["track"]),
                                 f":gray-badge[rating {V.rating(row['rating'])}]") if x))
if row["paper"]:
    ui.state_banner("PAPER", "Paper league: tips are collected and settled but never sent, counted or bet.")
V.profile(row.to_dict())
st.caption(f"Production state by market: {row['state_detail']}.")


def _stats_table(x: pd.DataFrame, first: str) -> pd.DataFrame:
    out = pd.DataFrame({
        first.title(): x[first],
        "Tips": x["tips"],
        "W-L": [leagues.wl(w, l, v) for w, l, v in zip(x["won"], x["lost"], x["void"])],
        "Hit %": (x["hit"] * 100).round(1), "ROI %": (x["roi"] * 100).round(1), "P/L (u)": x["pl"].round(2),
        "Avg odds": x["avg_odds"].round(2), "Avg edge %": x["avg_edge"].round(1),
        "Model %": (x["avg_model_prob"] * 100).round(1), "Implied %": (x["avg_implied"] * 100).round(1)})
    if "regime" in x.columns:
        out["Regime"] = x["regime"]
    return out


ui.section_header("Market breakdown", "Sides never pooled: O/U 2.5 Over and Under are separate rows.")
left, right = st.columns(2)
with left:
    ui.eyebrow("By market")
    basis = st.radio("Tiers", [f"Staked ({STAKED_LABEL})", f"{PAPER_TIER_LABEL} (paper)"], horizontal=True,
                     key="lg_mkt_basis", label_visibility="collapsed")
    bm = leagues.by_market(g, leagues.STAKED if basis.startswith("Staked") else ("VALUABLE",), reg=reg, league=pick)
    if bm.empty:
        ui.empty_state("NO_DATA", "No settled tips on this basis.")
    else:
        _bm = _stats_table(bm, "market")
        ui.compact_table(_bm, V.stats_config(_bm))
with right:
    ui.eyebrow("By tier · all markets")
    bt = leagues.by_tier(g)
    if bt.empty:
        ui.empty_state("NO_DATA", "No settled tips.")
    else:
        _bt = _stats_table(bt, "tier")
        ui.compact_table(_bt, V.stats_config(_bt))

# recent results + cumulative chart
ui.section_header("Recent trend", "Cumulative staked P/L by match date, and the last 30 settled tips (all tiers).")
left, right = st.columns([2, 3])
with left:
    cu = leagues.cumulative(g)
    if len(cu) > 1:
        st.altair_chart(ui.cumulative_line(cu, "date", "cum_pl", h=380), width="stretch")
    else:
        ui.empty_state("NO_DATA", "Not enough settled staked tips to draw a line.")
with right:
    rc = leagues.recent(g, 30)
    if rc.empty:
        ui.empty_state("NO_DATA", "No settled tips yet.")
    else:
        ui.compact_table(rc.rename(columns={"date": "Date", "match": "Match", "market": "Market", "tier": "Tier",
                                            "odds": "Odds", "result": "Result", "pnl": "P/L (u)"}),
                         {"Odds": ui.num_col("Odds", fmt="%.2f"), "P/L (u)": ui.num_col("P/L (u)", fmt="%+.2f")},
                         height=420)

# upcoming tips (tip days, the same rows the Telegram tips image draws)
ui.section_header("Upcoming tips", "Sent tips, tip days today to +7.")
u = up[up["league"] == pick] if len(up) and "league" in up.columns else pd.DataFrame()
if u.empty:
    ui.empty_state("NO_DATA", "No sent tips for this league in the next 7 tip days"
                   + (" (a paper league's tips are never sent)." if row["paper"] else "."))
else:
    u = u.sort_values(["tip_day", "kickoff_utc"], kind="mergesort")
    V.upcoming_cards(u)
    ui.compact_table(pd.DataFrame({
        "Tip day": u["tip_day"], "Kickoff (UTC)": u["time"],
        "Match": u["home_team"].astype(str) + " v " + u["away_team"].astype(str),
        "Market": u["market"], "Tier": u["tier"], "Odds": pd.to_numeric(u["odds"], errors="coerce").round(2),
        "Edge %": pd.to_numeric(u["edge"], errors="coerce").round(1)}),
        {"Odds": ui.num_col("Odds", fmt="%.2f"), "Edge %": ui.num_col("Edge %", fmt="%.1f")})

# research status: regimes per market + Pro season ranking
ui.section_header("Research status", "Research regime per market (where research effort goes, never a stake) and "
                  "Pro's season ranking as context.")
st.markdown(V.regime_line(pick, reg))
pc = leagues.pro_context(weekly, pick)
with st.expander("Pro season ranking (research context)", expanded=not pc.empty):
    if pc.empty:
        ui.empty_state("NOT_CONNECTED", "Pro's weekly league table is not reachable or does not list this league.")
    else:
        ui.compact_table(pd.DataFrame({
            "Week": pc["week"], "Window": pc["window"].map({"season": "Season to date",
                                                           "last_30_days": "Last 30 days"}).fillna(pc["window"]),
            "Rank": pc["rank"], "Tips": pc["tips"], "Hit %": (pc["hit_rate"] * 100).round(0),
            "Avg odds": pc["avg_odds"], "ROI %": (pc["roi"] * 100).round(0),
            "Best market": pc["best_market"], "Label": pc["label"]}),
            {"Rank": ui.num_col("Rank", fmt="%d"), "Tips": ui.num_col("Tips", fmt="%d"),
             "Hit %": ui.num_col("Hit %", fmt="%.0f"), "Avg odds": ui.num_col("Avg odds", fmt="%.2f"),
             "ROI %": ui.num_col("ROI %", fmt="%+.0f")})
        st.caption("Pro's table counts ALL settled tips per league (every tier) and ranks by ROI shrunk toward "
                   "the average - a different basis from the staked numbers above, shown as context.")

# freshness of every input - recorded timestamps, never file mtime
ui.section_header("Data sources", "Every age comes from a timestamp recorded inside the input, never from file mtime.")
_pol_ts = _stamps.pop("bankroll policy (generated)", None)
for label, ts in _stamps.items():
    cad = "weekly" if "regime" in label else "daily"
    ui.freshness_chip(core.freshness_from_timestamp(ts, cad, label) if ts else None, label)
st.caption(f"League ratings: {'loaded for ' + str(len(rat)) + ' leagues' if rat else 'unavailable or stale - shown as —'} "
           f"(Pro, hidden when older than 8 days). Bankroll policy {pol.get('policy_version', '?')} generated "
           f"{_pol_ts or '?'} - changes only through a reviewed PR; shown, not edited, here.")
