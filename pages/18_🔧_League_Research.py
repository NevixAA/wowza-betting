"""League Repair Lab — the weekly per-league research panel (owner principle 2026-10-25).

"Every league generates evidence every week. Not every league generates bets every week." Each tracked
league x market gets a top-5 of its gameweek's fixtures, ranked by market_edge_v2's EV lower bound and
frozen before the week's first kickoff, then graded with certified (T-30m) CLV. RESEARCH ONLY: nothing on
this page is a bet, and nothing here changes a production state, threshold, stake or tip.

Leagues are listed REPAIR first, then LEARNING, then PROTECTED (a PROTECTED track is shown with a
"PROTECTED — SHADOW ONLY" banner: research may observe it, never change it).

A VIEWER: one column-projected CSV (~400 rows a week) + small JSONs. Every aggregate comes from
src/research/league_weekly_eval.py through dashboard_data/league_research.py.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - same tier as League Scout

from dashboard_data import league_research as LR  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
from src.research import league_weekly_eval as _EV  # noqa: E402
_EV = importlib.reload(_EV)
LR = importlib.reload(LR)

st.set_page_config(page_title="Wowza | League Repair Lab", page_icon="🔧", layout="wide")
st.title("🔧 League Repair Lab")
st.caption("Weekly per-league research panel: the top-5 fixtures per league × market × gameweek by the market-first "
           "engine's EV lower bound, frozen before the week's first kickoff and graded afterwards. **Research only — "
           "these rows are not bets** and change no production state, threshold or stake.")


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    return (LR.load_panel(), LR.registry_entries(), LR.load_json("league_market_health.json"),
            LR.load_json("weekly_league_report.json"))


try:
    panel, entries, health, report = _load()
except Exception as e:                                                  # noqa: BLE001 - a viewer never dies
    st.error(f"Research panel unavailable ({type(e).__name__}: {e})")
    st.stop()


def _pct(x, dp=0):
    return "–" if x is None or pd.isna(x) else f"{100 * float(x):.{dp}f}%"


# ── headline (league_market_health.json) ─────────────────────────────────────────────────────
c = st.columns(6)
c[0].metric("Tracked leagues", health.get("tracked_leagues", "–"),
            help=f"{health.get('leagues_with_current_week_data', '–')} with rows this gameweek")
c[1].metric("Missing leagues", len(health.get("missing_leagues") or []),
            help="Fixtures this gameweek but no panel rows. 'No fixtures' is not missing.")
c[2].metric("Candidates (this week)", f"{health.get('actual_candidate_rows', '–')}/{health.get('expected_candidate_rows', '–')}")
c[3].metric("Panel rows", health.get("panel_rows", "–"),
            help=f"settled {health.get('settled_rows', '–')}, pending {health.get('pending_rows', '–')}")
c[4].metric("CLV coverage", _pct(health.get("clv_coverage")), help="certified close (≤30 min, pre-kickoff)")
ptm = health.get("protected_tracks_modified")
c[5].metric("Protected tracks modified", "NO" if ptm is False else "YES" if ptm else "–")
if ptm:
    st.error("protected_tracks_modified = true: production moved without re-approval (src/research_lock).")
if health.get("missing_leagues"):
    st.warning(f"Fixtures but no panel rows this gameweek: {', '.join(health['missing_leagues'])}")
st.caption(f"Panel health generated {health.get('generated_at', '–')} · panel started {health.get('panel_start', '–')} · "
           "evidence type TRUE_FORWARD (nothing replayed).")

# ── all leagues ──────────────────────────────────────────────────────────────────────────────
st.subheader("Leagues (REPAIR → LEARNING → PROTECTED)")
ov = LR.overview(panel, entries, health)
st.dataframe(ov, hide_index=True, width="stretch", height=min(38 * (len(ov) + 1) + 4, 820))

order = LR.league_order(entries)
if not order:
    st.info("No tracked leagues in config/track_registry.json.")
    st.stop()
label = {lg: f"{lg}  ·  {rg}" for lg, rg in order}
league = st.selectbox("League", [lg for lg, _ in order], format_func=lambda x: label[x])
regime = LR.league_regime(league, entries)
tracks = LR.track_rows(league, entries)
if (tracks["research_regime"] == "PROTECTED").any():
    prot = tracks[tracks["research_regime"] == "PROTECTED"]
    st.warning("PROTECTED — SHADOW ONLY: " + ", ".join(f"{m} {s}" for m, s in zip(prot["market"], prot["side"]))
               + ". Research observes these tracks; it never changes them.")

hdr = LR.league_header(panel, league)
h = st.columns(4)
h[0].metric("Research regime", regime)
h[1].metric("Gameweeks", hdr["gameweeks"])
h[2].metric("Observations", hdr["observations"])
cur = ((health.get("leagues") or {}).get(league) or {}).get("current") or {}
h[3].metric("This gameweek", cur.get("state", "–"), help=cur.get("league_week_id", ""))
st.caption(f"Prediction engine(s): {', '.join(hdr['prediction_engines']) or '–'} · selection engine(s): "
           f"{', '.join(hdr['selection_engines']) or '–'}")
st.dataframe(tracks, hide_index=True, width="stretch")

table = LR.metric_table(panel, league)
latest = ((report.get("leagues") or {}).get(league) or {})
for tab, market in zip(st.tabs(list(LR.MARKETS)), LR.MARKETS):
    with tab:
        st.markdown(f"**Latest gameweek {latest.get('latest_league_week_id') or '–'} — top 5**")
        top = pd.DataFrame((latest.get("top5") or {}).get(market) or [])
        if top.empty:
            st.caption("No rows for this market yet.")
        else:
            st.dataframe(top, hide_index=True, width="stretch")
        st.markdown("**Rank 1 / Top 2 / Top 3 / Top 5 / Qualified / Production-selected** (sides separate)")
        t = table[table["market"] == market]
        if t.empty:
            st.caption("NOT_ENOUGH_DATA — no frozen rows yet.")
        else:
            st.dataframe(t.drop(columns=["league", "market"]), hide_index=True, width="stretch")
        st.markdown("**Weekly history** (gameweek × #1..#5: fixture side | result | CLV | state | PROD)")
        grid = LR.history_grid(panel, league, market)
        if grid.empty:
            st.caption("No frozen gameweeks yet.")
        else:
            st.dataframe(grid, hide_index=True, width="stretch")
st.caption("ROI is flat 1u on rows with a two-sided price; INSUFFICIENT-price rows are graded but never certify "
           "performance. CI = 90% gameweek block bootstrap. NOT_ENOUGH_DATA below 30 priced settled rows or 8 "
           "gameweeks. Brier / log loss: model vs de-vigged first-sight market vs decision probability.")
