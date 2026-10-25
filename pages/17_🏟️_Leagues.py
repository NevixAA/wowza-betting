"""Leagues — every league's results, tips, rating and averages, then one league in detail.

Owner request 2026-10-25: "a page for it in the app? and telegram? 'leagues' with the ability to see
each league results and tip and rating and avg's". The numbers come from dashboard_data/leagues.py,
the same functions the Monday Telegram "Leagues" picture draws (telegram_bot/daily_pack.leagues_image),
so the app and Telegram cannot disagree.

Basis (the same as the rest of the app): live tips from the two committed ledgers, settled, match
date from PERFORMANCE_CUTOFF_DATE, flat 1u. Results = staked tiers (SNIPER + MARKSMAN); VALUABLE is
shown on its own line as paper. A paper league (config.PAPER_LEAGUES) is shown, labelled, and kept
out of every all-league total.

A VIEWER: two small ledgers, two small JSON configs, Pro's ratings JSON and one small Pro CSV.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES) - same tier as the Tips dashboard

from dashboard_data import core, leagues, ui  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helper
# so the page and its data module are always the same version.
import importlib  # noqa: E402
leagues = importlib.reload(leagues)

st.set_page_config(page_title="Wowza | Leagues", page_icon="🏟️", layout="wide")
st.title("🏟️ Leagues")
st.caption("Every league's record, tips, rating and averages. Results are **staked tiers (SNIPER + MARKSMAN)**, "
           "settled, flat 1u, from the performance cutoff; VALUABLE is paper and shown on its own line. "
           "Paper leagues are shown and labelled, never counted in totals. Not betting advice.")


@st.cache_data(ttl=300, show_spinner=False)
def _load(today: str):
    d = leagues.settled()
    up = leagues.upcoming(today, 7)
    rat = leagues.ratings()
    pol, reg = leagues.policy(), leagues.regimes()
    ov = leagues.overview(d, today, up=up, rat=rat, pol=pol, reg=reg)
    return d, up, rat, pol, reg, ov


@st.cache_data(ttl=1800, show_spinner=False)
def _pro_weekly():
    return leagues.pro_weekly()


from src.tip_day import today as _tip_today  # noqa: E402
TODAY = _tip_today()
try:
    d, up, rat, pol, reg, ov = _load(TODAY)
except Exception as e:                                                  # noqa: BLE001 - a viewer never dies
    st.error(f"League data unavailable ({type(e).__name__}: {e})")
    st.stop()

if ov.empty:
    st.info("No league data yet.")
    st.stop()


def _fmt_pct(x, dp=0):
    return "–" if x is None or pd.isna(x) else f"{100 * float(x):.{dp}f}%"


def _fmt_roi(x):
    return "–" if x is None or pd.isna(x) else f"{100 * float(x):+.1f}%"


def _fmt_num(x, dp=2):
    return "–" if x is None or pd.isna(x) else f"{float(x):.{dp}f}"


def _rating(x):
    return "–" if x is None or pd.isna(x) else f"{int(x)}/10"


# ── headline ──────────────────────────────────────────────────────────────────────────────────
tot = leagues.totals(d)
c = st.columns(5)
c[0].metric("Leagues", f"{int((ov['tips'] > 0).sum())} with staked tips", help=f"{len(ov)} leagues listed")
c[1].metric("Staked tips settled", f"{tot['tips']:,}", help="SNIPER + MARKSMAN, paper leagues excluded")
c[2].metric("Hit", _fmt_pct(tot["hit"], 1), help=f"{tot['won']}W-{tot['lost']}L"
            + (f", {tot['void']} void" if tot["void"] else ""))
c[3].metric("ROI", _fmt_roi(tot["roi"]))
c[4].metric("P/L", ui.units(tot["pl"]), help="Flat 1u stakes")

# ── 1. all-leagues overview ───────────────────────────────────────────────────────────────────
st.subheader("All leagues")
tab = leagues.overview_table(ov)          # the SAME table the Monday Telegram picture draws


def _sign_colour(v):
    if v is None or pd.isna(v) or v == 0:
        return ""
    pos, neg = ui.polarity()
    return f"color: {pos if v > 0 else neg}; font-weight: 600"


_sty = tab.style.map(_sign_colour, subset=["ROI %", "P/L (u)", "P/L 7d (u)", "VALUABLE (paper) P/L"]).format(
    {"Hit %": "{:.1f}", "ROI %": "{:+.1f}", "P/L (u)": "{:+.2f}", "Avg odds": "{:.2f}", "Avg edge %": "{:.1f}",
     "Model %": "{:.1f}", "Implied %": "{:.1f}", "P/L 7d (u)": "{:+.2f}", "VALUABLE (paper) P/L": "{:+.2f}"},
    na_rep="–")
st.dataframe(_sty, hide_index=True, width="stretch", height=min(38 * (len(tab) + 1) + 4, 860))
st.caption("Click a column header to sort. **Rating** = Pro's league rating 1-10 (the stake plan's sizing; "
           "'–' when missing or older than 8 days). **State** = production state from the bankroll policy "
           "(the strongest market of the league's track; detail below). **Regime** = research regime "
           "(PROTECTED / REPAIR / LEARNING, count of markets) - labels where research effort goes, never a "
           "stake. **Model % vs Implied %** = average model probability against 1/odds (O/U 2.5 model "
           "probability derived as 1/odds + edge). Avg edge mixes O/U 2.5 (vs the vigged price) and side "
           "markets (vs the de-vigged price) - compare per market below. **Upcoming** = sent tips, tip days "
           f"{TODAY} to +7.")

# ── 2. one league ────────────────────────────────────────────────────────────────────────────
st.divider()
names = ov["league"].tolist()
pick = st.selectbox("League", names, format_func=lambda n: n + (" (paper league)" if leagues.config.is_paper_league(n)
                                                                 else ""), key="lg_pick")
row = ov[ov["league"] == pick].iloc[0]
g = d[d["league"] == pick]

if row["paper"]:
    ui.state_banner("PAPER", "Paper league: tips are collected and settled but never sent, counted or bet.")
st.subheader(pick)
m = st.columns(5)
m[0].metric("Rating", _rating(row["rating"]))
m[1].metric("Track", row["track"])
m[2].metric("State", row["state"], help=row["state_detail"])
m[3].metric("Regime", row["regime"])
m[4].metric("Staked tips", f"{row['tips']}", help=f"{row['won']}W-{row['lost']}L"
            + (f", {row['void']} void" if row["void"] else ""))
m = st.columns(5)
m[0].metric("Hit", _fmt_pct(row["hit"], 1))
m[1].metric("ROI", _fmt_roi(row["roi"]))
m[2].metric("P/L", ui.units(row["pl"]), delta=f"{row['pl_7d']:+.2f}u last 7 days", delta_color="normal")
m[3].metric("Avg odds", _fmt_num(row["avg_odds"]))
m[4].metric("Avg edge", "–" if pd.isna(row["avg_edge"]) else f"{row['avg_edge']:.1f}%")
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


def _show(x: pd.DataFrame):
    st.dataframe(x.style.map(_sign_colour, subset=["ROI %", "P/L (u)"]).format(
        {"Hit %": "{:.1f}", "ROI %": "{:+.1f}", "P/L (u)": "{:+.2f}", "Avg odds": "{:.2f}",
         "Avg edge %": "{:.1f}", "Model %": "{:.1f}", "Implied %": "{:.1f}"}, na_rep="–"),
        hide_index=True, width="stretch")


left, right = st.columns(2)
with left:
    st.markdown("**By market** — sides never pooled")
    basis = st.radio("Tiers", ["Staked (SNIPER + MARKSMAN)", "VALUABLE (paper)"], horizontal=True,
                     key="lg_mkt_basis", label_visibility="collapsed")
    bm = leagues.by_market(g, leagues.STAKED if basis.startswith("Staked") else ("VALUABLE",), reg=reg, league=pick)
    if bm.empty:
        st.caption("No settled tips on this basis.")
    else:
        _show(_stats_table(bm, "market"))
with right:
    st.markdown("**By tier** — all markets")
    bt = leagues.by_tier(g)
    if bt.empty:
        st.caption("No settled tips.")
    else:
        _show(_stats_table(bt, "tier"))

# upcoming tips (tip days, the same rows the Telegram tips image draws)
st.markdown("**Upcoming tips** — tip days today to +7")
u = up[up["league"] == pick] if len(up) and "league" in up.columns else pd.DataFrame()
if u.empty:
    st.caption("No sent tips for this league in the next 7 tip days"
               + (" (a paper league's tips are never sent)." if row["paper"] else "."))
else:
    u = u.sort_values(["tip_day", "kickoff_utc"], kind="mergesort")
    st.dataframe(pd.DataFrame({
        "Tip day": u["tip_day"], "Kickoff (UTC)": u["time"],
        "Match": u["home_team"].astype(str) + " v " + u["away_team"].astype(str),
        "Market": u["market"], "Tier": u["tier"], "Odds": pd.to_numeric(u["odds"], errors="coerce").round(2),
        "Edge %": pd.to_numeric(u["edge"], errors="coerce").round(1)}), hide_index=True, width="stretch")

# recent results + cumulative chart
left, right = st.columns([3, 2])
with left:
    st.markdown("**Recent results** — last 30 settled tips, all tiers")
    rc = leagues.recent(g, 30)
    if rc.empty:
        st.caption("No settled tips yet.")
    else:
        st.dataframe(rc.rename(columns={"date": "Date", "match": "Match", "market": "Market", "tier": "Tier",
                                        "odds": "Odds", "result": "Result", "pnl": "P/L (u)"})
                     .style.map(_sign_colour, subset=["P/L (u)"]).format({"Odds": "{:.2f}", "P/L (u)": "{:+.2f}"},
                                                                          na_rep="–"),
                     hide_index=True, width="stretch", height=420)
with right:
    st.markdown("**Cumulative P/L** — staked tiers, by match date")
    cu = leagues.cumulative(g)
    if len(cu) > 1:
        st.altair_chart(ui.cumulative_line(cu, "date", "cum_pl", h=380), width="stretch")
    else:
        st.caption("Not enough settled staked tips to draw a line.")

# Pro season ranking context
weekly = _pro_weekly()
pc = leagues.pro_context(weekly, pick)
with st.expander("Pro season ranking (research context)", expanded=not pc.empty):
    if pc.empty:
        st.caption("Pro's weekly league table is not reachable or does not list this league.")
    else:
        st.dataframe(pd.DataFrame({
            "Week": pc["week"], "Window": pc["window"].map({"season": "Season to date",
                                                           "last_30_days": "Last 30 days"}).fillna(pc["window"]),
            "Rank": pc["rank"], "Tips": pc["tips"], "Hit %": (pc["hit_rate"] * 100).round(0),
            "Avg odds": pc["avg_odds"], "ROI %": (pc["roi"] * 100).round(0),
            "Best market": pc["best_market"], "Label": pc["label"]}), hide_index=True, width="stretch")
        st.caption("Pro's table counts ALL settled tips per league (every tier) and ranks by ROI shrunk toward "
                   "the average - a different basis from the staked numbers above, shown as context.")

# freshness - recorded timestamps, never file mtime
st.divider()
_stamps = leagues.freshness_stamps(weekly)
_pol_ts = _stamps.pop("bankroll policy (generated)", None)
for label, ts in _stamps.items():
    cad = "weekly" if "regime" in label else "daily"
    ui.freshness_chip(core.freshness_from_timestamp(ts, cad, label) if ts else None, label)
st.caption(f"League ratings: {'loaded for ' + str(len(rat)) + ' leagues' if rat else 'unavailable or stale - shown as –'} "
           f"(Pro, hidden when older than 8 days). Bankroll policy {pol.get('policy_version', '?')} generated "
           f"{_pol_ts or '?'} - changes only through a reviewed PR; shown, not edited, here.")
