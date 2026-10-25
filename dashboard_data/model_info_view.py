"""Presentation for pages/4_ℹ️_Model_Info.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page reads the same feature-importance CSVs and the same backtest-by-league CSV;
the two figures drawn outside the shared grammar (hardcoded dark backgrounds, a red/grey/green ramp) are
replaced by charts in the
shared altair grammar. Values unchanged: the importance bars are labelled with the file's own
importance_% and the backtest bars are the file's roi_% with its bets in the tooltip.
"""
from __future__ import annotations

import pandas as pd

import dashboard_ui as ui


def importance_bars(fi: pd.DataFrame, top: int = 12, h: int = 380):
    """Top-`top` features by importance — one hue (magnitude, not polarity), the % printed on each bar."""
    import altair as alt
    t = ui.tokens()
    d = fi.head(top).sort_values("importance")
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"], tickColor=t["BORDER"])
    base = alt.Chart(d).encode(y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(grid=False, **axis)))
    bars = base.mark_bar(cornerRadiusEnd=3, size=16, color=ui.token("PRIMARY")).encode(
        x=alt.X("importance:Q", title="importance", axis=alt.Axis(grid=True, gridColor=t["BORDER"], **axis)),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f"),
                 alt.Tooltip("importance_%:Q", title="share %", format=".1f")])
    text = base.mark_text(align="left", dx=4, color=t["TEXT_SECONDARY"]).encode(
        x="importance:Q", text=alt.Text("importance_%:Q", format=".1f"))
    return (bars + text).properties(height=h)


def backtest_bars(df: pd.DataFrame):
    """Walk-forward backtest ROI % by league around zero, bets in the tooltip (never P/L units)."""
    n_col = "bets" if "bets" in df.columns else None
    return ui.pnl_bars(df, "league", "roi_%", n_col, h=26 * len(df) + 60,
                       title="Walk-forward backtest ROI % by league (post-COVID, standard model)",
                       axis_title="ROI %", value_title="ROI %", fmt="+.2f")


THRESH_CONFIG = {"SNIPER Threshold": ui.num_col("SNIPER threshold", fmt="%.0f%%")}
BACKTEST_CONFIG = {"bets": ui.num_col("Bets", fmt="%.0f"), "wins": ui.num_col("Wins", fmt="%.0f"),
                   "profit": ui.num_col("Profit (u)", fmt="%+.2f"), "roi_%": ui.num_col("ROI %", fmt="%+.2f")}
