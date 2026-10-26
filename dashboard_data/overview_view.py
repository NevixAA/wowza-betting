"""Presentation for pages/0_🏠_Overview.py (admin), visual redesign 2026-10-25 (stage 2b).

PRESENTATION ONLY. The page reads exactly what it read before - src.system_health.evaluate(), v9 / Pro /
v11 / Fantasy adapters in dashboard_data - and this module only arranges it: the overall RED / AMBER /
GREEN line, one health card per check, the staked-P/L cards and one subsystem card per repo.

`prefetch_pro_listings` is the one speed change. Pro's canonical store walk makes one GitHub contents
call per store, one after another (~11 s cold when Pro is read over HTTP). It issues the SAME calls
concurrently so dashboard_data.core's in-process listing memo is warm when pro.canonical_stores() runs.
Same requests, same answers, same memo - only the waiting overlaps.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import streamlit as st

import dashboard_ui as ui
from src.tier_labels import STAKED_LABEL
from dashboard_data import core

ORDER = {"RED": 0, "AMBER": 1, "NO_DATA": 2, "GREEN": 3}
ROLE = {
    "v9": ("LIVE", "V9 live", "Production. Sends tips, stakes paper, learns every week."),
    "pro": ("RESEARCH", "Pro research", "Canonical evidence, validation, Bet Builder. Never writes into v9."),
    "v11": ("SHADOW", "V11 shadow", "Market-first shadow. Logs decisions, places nothing."),
    "fantasy": ("RESEARCH", "Fantasy", "FPL projections inside v9. Scored in points, never in units."),
}


# ── speed: overlap Pro's sequential directory listings (same calls, same memo) ─────────────────────
def prefetch_pro_listings(max_workers: int = 8) -> None:
    if core.repo_path("pro") is not None or "pro" not in core.API_BASE:
        return                                   # local checkout: listings are instant already
    seasons = sorted((n for n in core.remote_listing("pro", "data") if n.startswith("season_")), reverse=True)
    if not seasons:
        return
    stores = core.remote_listing("pro", f"data/{seasons[0]}")
    if not stores:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(lambda s: core.remote_listing("pro", f"data/{seasons[0]}/{s}"), sorted(stores)))


# ── system health ─────────────────────────────────────────────────────────────────────────────
def counts(checks: list[dict]) -> dict:
    return {k: sum(c.get("status") == k for c in checks) for k in ("RED", "AMBER", "GREEN", "NO_DATA")}


def overall(checks: list[dict]) -> str:
    """RED if any check is red, AMBER if any is amber, GREEN if every check is green, else NO_DATA."""
    n = counts(checks)
    if n["RED"]:
        return "RED"
    if n["AMBER"]:
        return "AMBER"
    return "GREEN" if checks and n["GREEN"] == len(checks) else ("NO_DATA" if not checks else "GREEN")


def count_line(checks: list[dict]) -> str:
    n = counts(checks)
    return f"{n['RED']} red · {n['AMBER']} amber · {n['GREEN']} green"


def health_cards(checks: list[dict], per_row: int = 3) -> None:
    """One card per check, worst first (stable within a colour). Rule and detail travel with the value."""
    if not checks:
        ui.empty_state("NO_DATA", "No system-health checks returned.")
        return
    srt = sorted(checks, key=lambda c: ORDER.get(c.get("status"), 2))
    for i in range(0, len(srt), per_row):
        for col, c in zip(st.columns(per_row), srt[i:i + per_row]):
            ui.health_card(title=c["check"], status=c.get("status", "NO_DATA"), value=str(c.get("value", "")),
                           rule=f"rule: {c['rule']}" + (f" · {c['detail']}" if c.get("detail") else ""),
                           container=col)


# ── production ────────────────────────────────────────────────────────────────────────────────
def staked_cards(perf: dict) -> None:
    """The staked headline + one card per track, SNIPER + MARKSMAN only (VALUABLE is paper)."""
    tracks = {k: val for k, val in perf["by_track"].items() if k != "unknown"}
    staked_n = sum(t["staked_only"]["n"] for t in tracks.values())
    staked_pnl = sum(t["staked_only"]["pnl"] for t in tracks.values())
    items = sorted(tracks.items())
    cols = st.columns(min(3, len(items) + 1))
    ui.metric_card("Staked P/L", ui.units(staked_pnl), title="V9 · all tracks", container=cols[0],
                   supporting=[("Staked bets", f"{staked_n:,}"), ("Tiers", f"{STAKED_LABEL} only")],
                   status=ui.status_chip("LIVE", label="STAKED"), n=staked_n)
    for i, (name, t) in enumerate(items):
        s_ = t["staked_only"]
        be = s_.get("break_even")
        ui.metric_card(name.replace("_", "-"), ui.units(s_["pnl"]), title="Track · staked",
                       container=cols[(i + 1) % len(cols)],
                       supporting=[("Staked", f"{s_['n']:,}"), ("Hit", ui.pct(s_.get("hit"), None)),
                                   ("Break-even", ui.pct(be, None) if be is not None else "—"),
                                   ("ROI", ui.pct(s_.get("roi"), None))],
                       n=s_["n"])


def tier_table_config() -> dict:
    return {"pnl": ui.num_col("P/L (u)", fmt="%+.2f"), "n": ui.num_col("n", fmt="%d"),
            "roi": ui.pct_col("ROI")}


# ── data feeds: one card per subsystem ─────────────────────────────────────────────────────────
def subsystem_header(key: str, label: str, found) -> None:
    state, title, blurb = ROLE[key]
    ui.eyebrow(title)
    st.markdown(f"<div class='wz-title'>{ui._html.escape(label)}</div>", unsafe_allow_html=True)
    st.markdown(ui.status_chip(state) + f" :gray[{ui._md_safe(blurb)}]")
    if key != "fantasy":
        st.caption(f"📁 `{found.name}`" if found is not None else "○ not checked out beside v9 — read over HTTP "
                   "from the public repo when reachable")
