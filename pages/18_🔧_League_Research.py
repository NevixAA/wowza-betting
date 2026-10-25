"""League Repair Lab — the weekly per-league research panel (owner principle 2026-10-25).

"Every league generates evidence every week. Not every league generates bets every week." Each tracked
league x market gets a top-5 of its gameweek's fixtures, ranked by market_edge_v2's EV lower bound and
frozen before the week's first kickoff, then graded with certified (T-30m) CLV. RESEARCH ONLY: nothing on
this page is a bet, and nothing here changes a production state, threshold, stake or tip.

Leagues are listed REPAIR first, then LEARNING, then PROTECTED (a PROTECTED track carries the governance
chip "🛡 PROTECTED · SHADOW ONLY": research may observe it, never change it).

Visual redesign 2026-10-25 (stage 2b): overview strip, league cards by regime, a league profile card, the
TOP 1/2/3/5 strip, the weekly rank heatmap, production vs market-first shadow, and this gameweek's top-5 as
cards. PRESENTATION ONLY - components live in dashboard_data/league_research_view.py.

A VIEWER: one column-projected CSV (~400 rows a week) + small JSONs. Every aggregate comes from
src/research/league_weekly_eval.py through dashboard_data/league_research.py.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - same tier as League Scout

import dashboard_ui as ui  # noqa: E402
from dashboard_data import league_research as LR  # noqa: E402
from dashboard_data import league_research_view as V  # noqa: E402

# STREAMLIT CLOUD KEEPS IMPORTED MODULES ACROSS A REDEPLOY (see 15_League_Scout): reload the helpers.
import importlib  # noqa: E402
from src.research import league_weekly_eval as _EV  # noqa: E402
_EV = importlib.reload(_EV)
LR = importlib.reload(LR)
V = importlib.reload(V)

st.set_page_config(page_title="Wowza | League Repair Lab", page_icon="🔧", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    """Every page-wide read, once per 5 minutes: the panel, registry, two JSONs, the league overview and the
    typed panel the heatmaps share (league_weekly_eval.prepare once, not once per market)."""
    p, e, h, r = (LR.load_panel(), LR.registry_entries(), LR.load_json("league_market_health.json"),
                  LR.load_json("weekly_league_report.json"))
    return p, e, h, r, LR.overview(p, e, h), _EV.prepare(p)


@st.cache_data(ttl=300, show_spinner=False)
def _metric_table(league: str):
    """league_weekly_eval.league_table for one league (seeded bootstrap, deterministic): cached so a widget
    click does not recompute ~0.35 s of bootstraps. Same function, same panel, same numbers."""
    return LR.metric_table(_load()[0], league)


try:
    panel, entries, health, report, ov, prepared = _load()
except Exception as e:                                                  # noqa: BLE001 - a viewer never dies
    ui.page_header("League Repair Lab", "Weekly per-league research panel.", eyebrow_text="Research Lab")
    ui.empty_state("ERROR", f"Research panel unavailable ({type(e).__name__}: {e})")
    st.stop()

ui.page_header("League Repair Lab",
               "The top-5 fixtures per league × market × gameweek by the market-first engine's EV lower bound, "
               "frozen before the week's first kickoff and graded afterwards. Research only — these rows are not "
               "bets and change no production state, threshold or stake.",
               eyebrow_text="Research Lab · weekly panel", state="RESEARCH", freshness=V.freshness(health))

# ── overview strip (league_market_health.json + the registry's regimes) ─────────────────────
order = LR.league_order(entries)
V.overview_strip(health, order)
ui.needs_attention(V.attention(health), healthy_text="Protected tracks unchanged; no league is missing panel rows.")
st.caption(f"Panel health generated {health.get('generated_at', '—')} · panel started {health.get('panel_start', '—')} · "
           "evidence type TRUE_FORWARD (nothing replayed).")

# ── all leagues, REPAIR -> LEARNING -> PROTECTED ───────────────────────────────────────────────
ui.section_header("Leagues", "REPAIR first, then LEARNING, then PROTECTED. Observations are panel rows; CLV "
                  "coverage is certified closes over priced settled rows.")
V.league_cards(ov)
with st.expander("All leagues as a table"):
    ui.compact_table(ov, V.overview_table_config(), height=min(38 * (len(ov) + 1) + 4, 820))

if not order:
    ui.empty_state("NO_DATA", "No tracked leagues in config/track_registry.json.")
    st.stop()

# ── one league ─────────────────────────────────────────────────────────────────────────────────
ui.section_header("League profile")
label = {lg: f"{lg}  ·  {rg}" for lg, rg in order}
league = st.selectbox("League", [lg for lg, _ in order], format_func=lambda x: label[x], key="lr_league")
regime = LR.league_regime(league, entries)
tracks = LR.track_rows(league, entries)
hdr = LR.league_header(panel, league)
cur = ((health.get("leagues") or {}).get(league) or {}).get("current") or {}
_ov_row = ov[ov["league"] == league].iloc[0].to_dict() if len(ov) and (ov["league"] == league).any() else {}
V.profile_card(league, regime, hdr, _ov_row, cur, tracks, V.engines(league, entries))
if (tracks["research_regime"] == "PROTECTED").any():
    prot = tracks[tracks["research_regime"] == "PROTECTED"]
    ui.state_banner("PROTECTED", "· SHADOW ONLY · " + ", ".join(f"{m} {s}" for m, s in zip(prot["market"], prot["side"]))
                    + " — research observes these tracks; it never changes them.")
with st.expander("Tracks (registry, verbatim)"):
    ui.compact_table(tracks, V.tracks_config())

table = _metric_table(league)
latest = ((report.get("leagues") or {}).get(league) or {})
for tab, market in zip(st.tabs(list(LR.MARKETS)), LR.MARKETS):
    with tab:
        t = table[table["market"] == market]

        ui.section_header(f"This gameweek — top 5 · {latest.get('latest_league_week_id') or '—'}",
                          "Ranked by EV lower bound. PROVISIONAL rows may still change until the week lock.")
        top = V.top_frame((latest.get("top5") or {}).get(market) or [])
        if top.empty:
            ui.empty_state("NO_DATA", "No rows for this market yet.")
        else:
            V.top5_cards(top)
            with st.expander("Full top-5 table"):
                ui.compact_table(top, V.top_config())

        ui.section_header("TOP 1 / TOP 2 / TOP 3 / TOP 5", "Flat 1u ROI on priced settled rows, mean certified "
                          "CLV, n · gameweeks. Sides are never pooled.")
        V.rank_strip(t)

        ui.section_header("Weekly rank heatmap", "Gameweek × rank: result, CLV and decision state in every cell.")
        V.heatmap(prepared, league, market, prepared=True)
        grid = LR.history_grid(panel, league, market)
        if not grid.empty:
            with st.expander("Weekly history as a table"):
                ui.compact_table(grid)

        ui.section_header("Current production vs market-first shadow",
                          "Production-selected rows against the shadow engine's top-5 and qualified rows, and the "
                          "three probabilities scored on the same top-5 rows. One measure per chart — no dual axes.")
        V.engine_compare(t)

        with st.expander("All metrics — Rank 1 / Top 2 / Top 3 / Top 5 / Qualified / Production-selected"):
            if t.empty:
                ui.empty_state("NO_DATA", "NOT_ENOUGH_DATA — no frozen rows yet.")
            else:
                ui.compact_table(t.drop(columns=["league", "market"]), V.metric_config())

st.caption("ROI is flat 1u on rows with a two-sided price; INSUFFICIENT-price rows are graded but never certify "
           "performance. CI = 90% gameweek block bootstrap. NOT_ENOUGH_DATA below 30 priced settled rows or 8 "
           "gameweeks. Brier / log loss: model vs de-vigged first-sight market vs decision probability.")
