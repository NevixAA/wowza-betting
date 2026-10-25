"""Presentation for pages/17_🏟️_Leagues.py, visual redesign 2026-10-25 (stage 2b).

PRESENTATION ONLY. Every value comes from dashboard_data/leagues.py - the module the Monday Telegram
"Leagues" picture also draws - and is only arranged here: league cards, the profile's three cards,
chips and column configs. The overview TABLE the page shows is still leagues.overview_table(ov), the
exact frame Telegram draws (tests/test_leagues_page.py renders the page and compares).

`trends` calls leagues.cumulative() per league (the same running staked P/L the profile chart draws) so
each card can carry its line; nothing new is computed.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui
from dashboard_data import core, leagues

#: overview_table columns -> display format (numbers stay numbers; sorting works)
TABLE_FORMATS = {"Tips": "%d", "Hit %": "%.1f", "ROI %": "%+.1f", "P/L (u)": "%+.2f", "Avg odds": "%.2f",
                 "Avg edge %": "%.1f", "Model %": "%.1f", "Implied %": "%.1f", "P/L 7d (u)": "%+.2f",
                 "VALUABLE (paper) P/L": "%+.2f", "Upcoming": "%d"}
STATS_FORMATS = {"Tips": "%d", "Hit %": "%.1f", "ROI %": "%+.1f", "P/L (u)": "%+.2f", "Avg odds": "%.2f",
                 "Avg edge %": "%.1f", "Model %": "%.1f", "Implied %": "%.1f"}


# ── the page's original text helpers, moved verbatim (missing renders as an em dash) ─────────────
def fmt_pct(x, dp=0):
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.{dp}f}%"


def fmt_roi(x):
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):+.1f}%"


def fmt_num(x, dp=2):
    return "—" if x is None or pd.isna(x) else f"{float(x):.{dp}f}"


def rating(x):
    return "—" if x is None or pd.isna(x) else f"{int(x)}/10"


def wl_help(won, lost, void) -> str:
    return f"{won}W-{lost}L" + (f", {void} void" if void else "")


# ── chips ───────────────────────────────────────────────────────────────────────────────────────
_STATE_TONE = {"ACTIVE_STRONG": "ACTIVE", "PROBATION": "PAPER", "DEGRADED": "PAPER"}


def state_chip(state: str) -> str:
    """Production state (bankroll policy). PAPER LEAGUE and '–' stay neutral."""
    s = str(state or "")
    if s in ("", "–", "—"):
        return ui.status_chip("NO_DATA", label="NO POLICY TRACK")
    if s == leagues.PAPER_STATE:
        return ui.status_chip("PAPER", label="PAPER LEAGUE")
    key = _STATE_TONE.get(s, s)
    icon, word, tone = ui.status_meta(key)
    return ui.status_chip(key, label=s.replace("_", " ") if key != s else word)


def regime_chip(regime: str) -> str:
    """'REPAIR (2)' -> the REPAIR chip carrying the count; PROTECTED is governance, not an error."""
    s = str(regime or "")
    if s in ("", "–", "—"):
        return ""
    return ui.status_chip(s.split(" ")[0], label=s)


def track_chip(track: str) -> str:
    return f":gray-badge[{ui._md_safe(track)}]"


# ── overview: cards + table ───────────────────────────────────────────────────────────────────
def trends(d: pd.DataFrame, names) -> dict:
    """{league: running staked P/L list} from leagues.cumulative (one point per match day)."""
    out = {}
    for lg in names:
        cu = leagues.cumulative(d[d["league"] == lg])
        out[lg] = cu["cum_pl"].round(2).tolist() if len(cu) > 1 else None
    return out


def league_cards(ov: pd.DataFrame, tr: dict, per_row: int = 3) -> None:
    rows = [r for r in ov.to_dict("records") if r["tips"] > 0 and not r["paper"]]
    if not rows:
        ui.empty_state("NO_DATA", "No league has a settled staked tip yet.")
        return
    for i in range(0, len(rows), per_row):
        for col, r in zip(st.columns(per_row), rows[i:i + per_row]):
            ui.metric_card(
                "P/L (staked)", ui.units(r["pl"]), title=r["league"], container=col,
                supporting=[("ROI", fmt_roi(r["roi"])), ("Hit", fmt_pct(r["hit"], 1)),
                            ("Tips", f"{r['tips']} · {leagues.wl(r['won'], r['lost'], r['void'])}"),
                            ("Rating", rating(r["rating"])), ("Last 7d", f"{r['pl_7d']:+.2f}u")],
                status=" ".join(x for x in (state_chip(r["state"]), regime_chip(r["regime"]),
                                            track_chip(r["track"])) if x),
                n=r["tips"], chart_data=tr.get(r["league"]))
    paper = [r["league"] for r in ov.to_dict("records") if r["paper"]]
    quiet = [r["league"] for r in ov.to_dict("records") if not r["paper"] and r["tips"] == 0]
    if paper or quiet:
        st.caption(" · ".join(x for x in (
            ("Paper leagues (shown in the table, never totalled): " + ", ".join(paper)) if paper else "",
            ("No settled staked tips yet: " + ", ".join(quiet)) if quiet else "") if x))


def table_config(tab: pd.DataFrame, trend_col: str | None = None) -> dict:
    cfg = {c: ui.num_col(c, fmt=f) for c, f in TABLE_FORMATS.items() if c in tab.columns}
    if trend_col:
        cfg[trend_col] = ui.spark_col(trend_col, help="Running staked P/L by match date; each row on its own "
                                                      "scale - compare the shape, read P/L for the level.")
    return cfg


def stats_config(x: pd.DataFrame) -> dict:
    return {c: ui.num_col(c, fmt=f) for c, f in STATS_FORMATS.items() if c in x.columns}


# ── profile ─────────────────────────────────────────────────────────────────────────────────────
def profile(row: dict) -> None:
    """Three cards, one main number each: record, context, prices."""
    a, b, c = st.columns(3)
    ui.metric_card("P/L", ui.units(row["pl"]), title="Record · staked tiers", container=a,
                   delta=f"{row['pl_7d']:+.2f}u last 7 days", delta_color="normal",
                   supporting=[("Hit", fmt_pct(row["hit"], 1)), ("ROI", fmt_roi(row["roi"])),
                               ("Staked tips", f"{row['tips']}"),
                               ("W-L", wl_help(row["won"], row["lost"], row["void"]))],
                   n=row["tips"])
    ui.metric_card("Rating", rating(row["rating"]), title="Context", container=b,
                   supporting=[("Track", row["track"]), ("State", row["state"]), ("Regime", row["regime"]),
                               ("Upcoming tips", f"{row['upcoming']}")],
                   status=" ".join(x for x in (state_chip(row["state"]), regime_chip(row["regime"])) if x))
    ui.metric_card("Avg odds", fmt_num(row["avg_odds"]), title="Prices", container=c,
                   supporting=[("Avg edge", "—" if pd.isna(row["avg_edge"]) else f"{row['avg_edge']:.1f}%"),
                               ("Model %", fmt_pct(row["avg_model_prob"], 1)),
                               ("Implied %", fmt_pct(row["avg_implied"], 1)),
                               ("Best market", row["best_market"]), ("Worst market", row["worst_market"])])


def regime_line(league: str, reg: dict) -> str:
    r = leagues.regime_entries(league, reg)
    if not r:
        return ui.status_chip("NO_DATA", label="NO RESEARCH REGIME")
    return " ".join(ui.status_chip(v, label=f"{m} · {v}") for m, v in r.items())


def upcoming_cards(u: pd.DataFrame, n: int = 3) -> None:
    cols = st.columns(3)
    for col, r in zip(cols, u.head(n).to_dict("records")):
        ui.selection_card(league=str(r.get("tip_day", "")), kickoff=f"{r.get('time', '')} UTC",
                          home=r.get("home_team", ""), away=r.get("away_team", ""), pick=str(r.get("market", "")),
                          odds=pd.to_numeric(r.get("odds"), errors="coerce"), tier=str(r.get("tier", "")),
                          show_plan=False, container=col,
                          supporting=[("Edge", _edge(r.get("edge")))])


def _edge(x) -> str:
    v = pd.to_numeric(x, errors="coerce")
    return "—" if pd.isna(v) else f"{float(v):.1f}%"


def freshness_header(stamps: dict) -> str:
    """The page's ONE freshness statement: the newest tip recorded in the ledgers."""
    ts = stamps.get("ledgers (newest tip recorded)")
    f = core.freshness_from_timestamp(ts, "daily", "ledgers") if ts else None
    if f is None:
        return ui.freshness_badge("UNKNOWN", "ledgers: no timestamp")
    return ui.freshness_badge(f.state, f"ledgers · newest tip {f.label}")
