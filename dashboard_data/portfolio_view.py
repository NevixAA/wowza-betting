"""Presentation for pages/12_💼_Portfolio.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page's simulate() still walks the same ledgers with the same sizing rules; this
module only arranges its output: one summary card per market (one main number, the rest supporting),
the equity curve against the starting bankroll, and the weekly / per-bet tables with numeric columns.
Every string here is formatted exactly as the page formatted it before the redesign.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui

#: What each simulated market IS, so a curve is never read as money it was not. Props are paper by
#: invariant 2; the side-market ledger has never been staked. The two O/U tracks are simulations of the
#: tips production sent - the bankroll itself is notional, so they carry SIMULATED, not LIVE.
MARKET_STATE = {"Standard O/U": ("SIMULATED", None), "New-Format": ("SIMULATED", None),
                "Player Props": ("PAPER", "PAPER · NEVER STAKED"), "Side/BTTS": ("PAPER", "PAPER · NEVER STAKED")}


def state_chip(market: str) -> str:
    state, label = MARKET_STATE.get(market, ("SIMULATED", None))
    if state == "SIMULATED":
        return ui.status_chip("SIMULATED", label="SIMULATED · NOTIONAL BANK", tone="INFO")
    return ui.status_chip(state, label=label)


def headline(s: dict) -> dict:
    """The page's own formatted figures for one market (identical strings to the pre-redesign metrics)."""
    return {"final": f"{s['final']:,.0f}U", "roi": f"{s['roi'] * 100:+.1f}%", "bets": f"{s['n_bets']}",
            "win": f"{s['win_pct'] * 100:.0f}% win", "best": f"+{s['best_week']:,.0f}U",
            "worst": f"{s['worst_week']:,.0f}U", "max_dd": f"{s['max_dd'] * 100:.1f}%", "weeks": f"{s['weeks']} wks"}


def summary_card(market: str, s: dict, container=None) -> None:
    """One card per market: final bankroll is the main number; ROI never stands without n and weeks."""
    h = headline(s)
    ui.metric_card("Final bankroll", h["final"], title=market, delta=h["roi"],
                   supporting=[("Bets", h["bets"]), ("Win rate", h["win"]), ("Weeks", h["weeks"]),
                               ("Max drawdown", h["max_dd"]), ("Best / worst week", f"{h['best']} / {h['worst']}")],
                   status=state_chip(market), n=s["n_bets"], container=container)


def empty_card(market: str, container=None) -> None:
    with (container or st).container(border=True):
        ui.eyebrow(market)
        st.markdown(state_chip(market) + " " + ui.status_chip("NO_DATA"))
        st.caption(f"No settled bets for {market} under the current tier/source filters.")


def equity_frame(eq: pd.DataFrame, start_bank: float) -> pd.DataFrame:
    """The curve the page drew before (st.line_chart of bank by week, seeded one week earlier at the
    starting bankroll), as a long frame with the bets of each week for the tooltip."""
    curve = eq.set_index("week")["bank"]
    curve.loc[curve.index.min() - pd.Timedelta(days=7)] = start_bank
    out = curve.sort_index().rename("bank").reset_index()
    out["bets"] = out["week"].map(eq.set_index("week")["bets"]).fillna(0).astype(int)
    return out


def equity_chart(eq: pd.DataFrame, start_bank: float, h: int = 260):
    """Bankroll by week against a rule at the starting bankroll (above the rule = up on the start)."""
    import altair as alt
    d = equity_frame(eq, start_bank)
    t = ui.tokens()
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"], tickColor=t["BORDER"])
    rule = alt.Chart(pd.DataFrame({"y": [float(start_bank)]})).mark_rule(
        color=t["BORDER_STRONG"], strokeDash=[4, 4], size=1).encode(y="y:Q")
    line = alt.Chart(d).mark_line(size=2, color=ui.repo_colour("v9"), point=True).encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(grid=False, **axis)),
        y=alt.Y("bank:Q", title="bankroll (U)", scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True, gridColor=t["BORDER"], **axis)),
        tooltip=[alt.Tooltip("week:T", title="week"), alt.Tooltip("bank:Q", format=",.0f", title="bankroll (U)"),
                 alt.Tooltip("bets:Q", title="bets that week", format=",")])
    return (rule + line).properties(height=h, title="Bankroll by week — dashed line is the starting bankroll")


def weekly_frame(eq: pd.DataFrame) -> pd.DataFrame:
    """Same values and rounding as the pre-redesign weekly table."""
    return eq.assign(week=eq["week"].dt.strftime("%Y-%m-%d"), staked=eq["staked"].round(1), pnl=eq["pnl"].round(1),
                     bank=eq["bank"].round(0), drawdown=(eq["drawdown"] * 100).round(1),
                     )[["week", "bets", "staked", "pnl", "bank", "drawdown"]]


WEEKLY_CONFIG = {"week": ui.text_col("Week"), "bets": st.column_config.NumberColumn("Bets", format="%d"),
                 "staked": ui.num_col("Staked (U)", fmt="%.1f"), "pnl": ui.num_col("P/L (U)", fmt="%+.1f"),
                 "bank": ui.num_col("Bankroll (U)", fmt="%.0f"), "drawdown": ui.num_col("Drawdown %", fmt="%.1f")}


def games_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Same values and rounding as the pre-redesign per-bet table."""
    gv = games.sort_values("date", ascending=False).copy()
    gv["date"] = pd.to_datetime(gv["date"]).dt.strftime("%Y-%m-%d")
    gv["odds"] = gv["odds"].round(2)
    gv["stake"] = gv["stake"].round(2)
    gv["pnl"] = gv["pnl"].round(2)
    return gv[["date", "league", "bet", "tier", "odds", "stake", "result", "pnl"]]


GAMES_CONFIG = {"odds": ui.num_col("Odds"), "stake": ui.num_col("Stake (U)"), "pnl": ui.num_col("P/L (U)", fmt="%+.2f")}
