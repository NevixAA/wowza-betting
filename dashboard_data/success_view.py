"""Presentation for pages/5_🎯_Success_Rates.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page still unions the same ledgers, applies the same filters and computes the
same model x market x tier table and roll-ups; this module only draws them. The two charts were drawn outside
the shared grammar with hardcoded colours — they are now altair, in the shared grammar: one measure per chart, the
zero / break-even reference drawn, n in every tooltip, and colour never the only cue (shape and the
legend word carry 'clears zero'; the bar label carries above / below break-even).
"""
from __future__ import annotations

import pandas as pd

import dashboard_ui as ui


def _axis(t, **kw):
    import altair as alt
    return alt.Axis(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"],
                    tickColor=t["BORDER"], **kw)


def ci_frame(shown: pd.DataFrame) -> pd.DataFrame:
    """Cells with a numeric interval, labelled — the rows the chart draws (same rule as before)."""
    ch = shown.dropna(subset=["roi_lo", "roi_hi"]).copy()
    if ch.empty:
        return ch
    ch["cell"] = ch["model"].astype(str) + " · " + ch["market"].astype(str) + " · " + ch["tier"].astype(str)
    ch["decisive"] = (ch["roi_lo"] > 0) | (ch["roi_hi"] < 0)
    ch["reading"] = ch["decisive"].map({True: "interval clears zero", False: "interval spans zero"})
    return ch.sort_values("roi_pct")


def ci_chart(ch: pd.DataFrame):
    """ROI with its 95% interval per cell, against a zero rule. Decisive cells: filled diamond + legend word."""
    import altair as alt
    t = ui.tokens()
    h = max(240, 34 * len(ch))
    base = alt.Chart(ch).encode(y=alt.Y("cell:N", sort=list(ch["cell"]), title=None, axis=_axis(t, grid=False)))
    dom = ["interval clears zero", "interval spans zero"]
    rng = [t["PRIMARY"], t["NO_DATA"]]
    whisk = base.mark_rule(size=1.6).encode(
        x=alt.X("roi_lo:Q", title="ROI % (95% CI)", axis=_axis(t, grid=True, gridColor=t["BORDER"])),
        x2="roi_hi:Q", color=alt.Color("reading:N", scale=alt.Scale(domain=dom, range=rng), legend=None))
    pts = base.mark_point(size=110, filled=True).encode(
        x="roi_pct:Q",
        color=alt.Color("reading:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("reading:N", scale=alt.Scale(domain=dom, range=["diamond", "circle"]), legend=None),
        tooltip=[alt.Tooltip("cell:N"), alt.Tooltip("roi_pct:Q", title="ROI %", format=".1f"),
                 alt.Tooltip("roi_lo:Q", title="CI low", format=".1f"),
                 alt.Tooltip("roi_hi:Q", title="CI high", format=".1f"),
                 alt.Tooltip("n:Q", title="n", format=","), alt.Tooltip("reading:N")])
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(
        color=t["BORDER_STRONG"], strokeDash=[4, 4], size=1).encode(x="z:Q")
    return (whisk + pts + zero).properties(height=h)


def rollup_chart(t_: pd.DataFrame, by: str):
    """Achieved win % per group against its own break-even (1 / average odds), n in the tooltip.

    Bar colour supports the reading (above break-even / below / no trustworthy break-even) and the same
    word is in the tooltip; the break-even is a tick, drawn only where >= 80% of the bets were priced."""
    import altair as alt
    t = ui.tokens()
    d = t_.copy()
    d["group"] = d[by].astype(str)
    d["reading"] = [("no break-even (priced < 80%)" if b != b else ("above break-even" if a >= b else
                                                                    "below break-even"))
                    for a, b in zip(d["win_pct"], d["need_pct"])]
    dom = ["above break-even", "below break-even", "no break-even (priced < 80%)"]
    rng = [t["POSITIVE"], t["NEGATIVE"], t["NO_DATA"]]
    bars = alt.Chart(d).mark_bar(size=22, cornerRadiusEnd=3).encode(
        x=alt.X("group:N", title=None, sort=list(d["group"]), axis=_axis(t, grid=False, labelAngle=0)),
        y=alt.Y("win_pct:Q", title="win %", axis=_axis(t, grid=True, gridColor=t["BORDER"])),
        color=alt.Color("reading:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("group:N"), alt.Tooltip("win_pct:Q", title="achieved %", format=".1f"),
                 alt.Tooltip("need_pct:Q", title="break-even %", format=".1f"),
                 alt.Tooltip("n:Q", title="n", format=","), alt.Tooltip("reading:N")])
    be = alt.Chart(d.dropna(subset=["need_pct"])).mark_tick(thickness=3, size=30, color=t["TEXT_PRIMARY"]).encode(
        x=alt.X("group:N", sort=list(d["group"])), y="need_pct:Q",
        tooltip=[alt.Tooltip("group:N"), alt.Tooltip("need_pct:Q", title="needs %", format=".1f")])
    return (bars + be).properties(height=260)
