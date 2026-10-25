"""Presentation for pages/11_👤_Player_Props.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page still loads output/player_tips.csv and output/player_ledger.csv exactly as
before (same de-duplication, same priced / unpriced split, same filters, same per-day limits). The old
page drew each day as a hand-built HTML table with its own stylesheet and hardcoded tier colours; this
module returns the SAME rows as numeric frames, rendered with st.dataframe + column_config, plus the
top-of-page cards. Props are PAPER-ONLY, permanently (invariant 2): nothing here is a bet.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import dashboard_ui as ui

TZ_NAME = "Asia/Jerusalem"
MARKET_LABEL = {
    "goals": "⚽ Anytime scorer", "goals2": "⚽ 2+ goals", "goals3": "⚽ Hat-trick",
    "assists": "🅰️ Assist",
    "sot": "🎯 1+ shot on target", "sot2": "🎯 2+ shots on target",
    "sot3": "🎯 3+ shots on target", "sot4": "🎯 4+ shots on target",
    "cards": "🟨 Booked",
}


def _tz():
    try:
        return ZoneInfo(TZ_NAME)
    except Exception:                                       # noqa: BLE001
        return ZoneInfo("UTC")


def board(d: pd.DataFrame, priced_view: bool, limit: int) -> tuple[pd.DataFrame, list[dict]]:
    """(rows shown, per-day headers) — the rows the old HTML board drew, in the same order.

    Same sort (edge desc for priced, model probability desc for unpriced, within each local day), same
    per-day `limit`, same day label. Numbers stay numbers: model / implied / EV in percent points, edge in
    percentage points, odds as odds.
    """
    tz = _tz()
    d = d.copy()
    ko = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    d["_local"] = ko.dt.tz_convert(tz)
    d["_day"] = d["_local"].dt.strftime("%Y-%m-%d").fillna(d["date"].dt.strftime("%Y-%m-%d"))
    if priced_view:
        d["_edge"] = (d["model_prob"] - 1 / d["market_odds"]) * 100
        d = d.sort_values(["_day", "_edge"], ascending=[True, False])
    else:
        d = d.sort_values(["_day", "model_prob"], ascending=[True, False])
    shown = d.groupby("_day", group_keys=False).head(limit)
    today = pd.Timestamp.now(tz=tz).date()
    days = []
    for day, g in shown.groupby("_day", sort=True):
        dd = pd.Timestamp(day)
        rel = {0: "Today", 1: "Tomorrow"}.get((dd.date() - today).days, "")
        total = int((d["_day"] == day).sum())
        days.append({"day": day, "title": f"{rel + ' · ' if rel else ''}{dd:%a %d %b}", "total": total,
                     "shown": len(g)})
    out = pd.DataFrame({
        "_day": shown["_day"],
        "Kickoff": shown["_local"].map(lambda t: t.strftime("%H:%M") if pd.notna(t) else "–"),
        "Player": shown["player_name"].astype(str),
        "Team": shown.get("team", pd.Series("", index=shown.index)).fillna("").astype(str),
        "Pos": shown.get("position", pd.Series(None, index=shown.index)),
        "Match": shown["match"].astype(str),
        "League": shown["league"].astype(str),
        "Market": shown["market"].map(lambda m: MARKET_LABEL.get(m, str(m))),
        "Model %": shown["model_prob"] * 100,
    })
    if priced_view:
        out["Book odds"] = shown["market_odds"]
        out["Implied %"] = 100 / shown["market_odds"]
        out["Edge (pts)"] = shown["_edge"]
        out["EV %"] = shown["ev"] * 100 if "ev" in shown.columns else None
        out["Tier"] = shown.get("tier", pd.Series("", index=shown.index)).astype(str)
    else:
        out["Fair odds"] = pd.to_numeric(shown.get("fair_odds"), errors="coerce")
    return out, days


def board_config(priced_view: bool) -> dict:
    cfg = {"Model %": ui.num_col("Model", fmt="%.0f%%", help="The model's probability."),
           "Kickoff": ui.text_col("Kickoff", help=f"Local time ({TZ_NAME})"), "Pos": ui.text_col("Pos")}
    if priced_view:
        cfg.update({"Book odds": ui.num_col("Book odds"),
                    "Implied %": ui.num_col("Implied", fmt="%.0f%%", help="1 / book odds (includes the margin)."),
                    "Edge (pts)": ui.num_col("Edge", fmt="%+.1f pts",
                                             help="Model probability minus the bookmaker's implied probability."),
                    "EV %": ui.num_col("EV", fmt="%+.0f%%"), "Tier": ui.text_col("Tier")})
    else:
        cfg["Fair odds"] = ui.num_col("Fair odds", help="1 / model probability.")
    return cfg


def render_board(d: pd.DataFrame, priced_view: bool, limit: int) -> pd.DataFrame:
    """One day heading + one sortable table per local day (kickoff in Israel time)."""
    rows, days = board(d, priced_view, limit)
    cfg = board_config(priced_view)
    for day in days:
        more = f" · top {day['shown']} shown" if day["total"] > day["shown"] else ""
        ui.eyebrow(f"{day['title']} — {day['total']} prop{'s' if day['total'] != 1 else ''}{more}")
        ui.compact_table(rows[rows["_day"] == day["day"]].drop(columns="_day"), cfg)
    return rows


def top_priced(priced: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The k priced props where the model sits furthest above the price (paper only)."""
    if priced.empty:
        return priced
    p = priced.assign(_edge=(priced["model_prob"] - 1 / priced["market_odds"]) * 100)
    return p.sort_values("_edge", ascending=False).head(k)


def prop_card(r, container=None) -> None:
    """A priced prop: the model probability is the main number; price, implied, edge and EV beside it."""
    ko = pd.to_datetime(r.get("kickoff_utc"), errors="coerce", utc=True)
    when = ko.tz_convert(_tz()).strftime("%a %d %b %H:%M") if pd.notna(ko) else str(r.get("date"))[:10]
    ui.metric_card(MARKET_LABEL.get(r["market"], str(r["market"])), f"{r['model_prob'] * 100:.0f}%",
                   title=f"{r['player_name']} · {r.get('team', '')}",
                   supporting=[("Book odds", f"{r['market_odds']:.2f}"), ("Implied", f"{100 / r['market_odds']:.0f}%"),
                               ("Edge", f"{r['_edge']:+.1f} pts"),
                               ("EV", f"{r['ev']:+.0%}" if pd.notna(r.get("ev")) else "—"),
                               ("Kickoff", when)],
                   status=ui.status_chip("PAPER", label="PAPER · NOT A BET")
                   + (" " + ui.status_chip(str(r["tier"])) if pd.notna(r.get("tier")) and str(r.get("tier")) else ""),
                   container=container)
    (container or st).caption(f"{r['match']} · {r['league']}")


RESULTS_CONFIG = {"market_odds": ui.num_col("Odds"), "pnl": ui.num_col("P/L (u)", fmt="%+.2f")}
RAW_CONFIG = {"model_prob": ui.pct_col("Model"), "fair_odds": ui.num_col("Fair odds"),
              "market_odds": ui.num_col("Book odds"), "ev": ui.num_col("EV", fmt="%+.3f"),
              "confidence": ui.num_col("Confidence", fmt="%.3f")}
