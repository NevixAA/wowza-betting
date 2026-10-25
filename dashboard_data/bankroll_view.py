"""Presentation for pages/16_💰_Bankroll.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page reads exactly what it read before — config/bankroll_policy.json (the production
policy), Pro's monthly review (dashboard_data.pro.bankroll) and output/stake_plan.csv — and nothing here
sizes, recommends or edits a policy. This module turns those into: the decision cards, one card per track
(state chip, base stake, Pro's recommendation, evidence n beside every ROI), the per-day stake bars against
the daily cap, and the drawdown path. Every percentage keeps the page's pct() formatting.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui


def pct(x) -> str:
    """The page's own formatter (unchanged): a fraction as a percent with two decimals, '—' when missing."""
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.2f}%"


def track_rows(prod: dict, rec: dict, dec: pd.DataFrame) -> pd.DataFrame:
    """One row per production track, NUMERIC (fractions x100 as percent points; formatted at render time).
    The same fields the pre-redesign table showed as pct() strings."""
    rows = []
    for k, t in (prod.get("tracks") or {}).items():
        r = (rec.get("tracks") or {}).get(k, {})
        d = dec[dec["track"] == k].iloc[0] if len(dec) and (dec["track"] == k).any() else None

        def _p(x):
            return None if x is None or pd.isna(x) else 100 * float(x)
        rows.append({"Track": k.replace("|", " · "), "State": t.get("state"),
                     "Base (production)": _p(t.get("base_fraction")), "Recommended": _p(r.get("base_fraction")),
                     "Decision": None if d is None else d["decision"], "Max odds": str(t.get("max_odds", "—")),
                     "n": None if d is None else d.get("n"), "Matchdays": None if d is None else d.get("matchdays"),
                     "ROI": None if d is None else _p(d.get("roi")),
                     "Shrunk ROI": None if d is None else _p(d.get("shrunk_roi")),
                     "Why": None if d is None else str(d.get("reasons", ""))[:160]})
    return pd.DataFrame(rows)


TRACK_CONFIG = {"Base (production)": ui.num_col("Base (production) %", fmt="%.2f%%"),
                "Recommended": ui.num_col("Recommended %", fmt="%.2f%%"),
                "ROI": ui.num_col("ROI %", fmt="%.2f%%"), "Shrunk ROI": ui.num_col("Shrunk ROI %", fmt="%.2f%%"),
                "n": st.column_config.NumberColumn("n", format="%d"),
                "Matchdays": st.column_config.NumberColumn("Matchdays", format="%d"),
                "Why": ui.text_col("Why", width="large")}

#: production state -> chip (ACTIVE / ACTIVE_LOW / PAPER / BLOCKED are in the shared vocabulary)
_ORDER = {"ACTIVE": 0, "ACTIVE_LOW": 1, "PAPER": 2, "BLOCKED": 3}


def _fmt_pp(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x):.2f}%"


def track_cards(rows: pd.DataFrame, per_row: int = 3) -> None:
    """One card per track, ACTIVE first: the production base stake is the main number; the ROI is shown
    only beside its n and matchdays (an ROI alone is never evidence)."""
    if rows.empty:
        ui.empty_state("NO_DATA", "config/bankroll_policy.json has no tracks.")
        return
    srt = rows.assign(_o=rows["State"].map(_ORDER).fillna(9)).sort_values(["_o", "Track"], kind="mergesort")
    recs = srt.to_dict("records")
    for i in range(0, len(recs), per_row):
        for col, r in zip(st.columns(per_row), recs[i:i + per_row]):
            n = r.get("n")
            n_txt = "—" if n is None or pd.isna(n) else f"{int(n):,}"
            md = r.get("Matchdays")
            ui.metric_card("Base stake (production)", _fmt_pp(r["Base (production)"]), title=r["Track"],
                           supporting=[("Pro recommends", _fmt_pp(r["Recommended"])),
                                       ("Decision", r.get("Decision") or "—"),
                                       ("ROI · n · days", f"{_fmt_pp(r['ROI'])} · n={n_txt} · "
                                                          f"{'—' if md is None or pd.isna(md) else int(md)}"),
                                       ("Shrunk ROI", _fmt_pp(r["Shrunk ROI"])), ("Max odds", r["Max odds"])],
                           status=ui.status_chip(r.get("State") or "UNKNOWN"),
                           n=None if n is None or pd.isna(n) else int(n), container=col)


def plan_rows(sp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(staked rows table, total stake % per day) — the same arithmetic as before the redesign."""
    staked = sp[sp["final_fraction"] > 0]
    by_day = (staked.groupby("date")["final_fraction"].sum() * 100).round(2).rename("% of bankroll").reset_index()
    by_day["tips"] = by_day["date"].map(staked.groupby("date").size()).fillna(0).astype(int)   # tooltip n only
    show = staked.assign(stake_pct=(staked["final_fraction"] * 100).round(3))[
        ["date", "fixture", "league", "track", "signal_tier", "odds", "stake_pct"]].sort_values(
        ["date", "stake_pct"], ascending=[True, False])
    return show.rename(columns={"stake_pct": "stake % of bankroll"}), by_day


PLAN_CONFIG = {"odds": ui.num_col("Odds"), "stake % of bankroll": ui.num_col("Stake % of bankroll", fmt="%.3f")}


def day_bars(by_day: pd.DataFrame, cap_fraction, h: int = 220):
    """Total stake per day (% of bankroll) against the daily cap — one measure, one scale, n in the tooltip."""
    import altair as alt
    t = ui.tokens()
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"], tickColor=t["BORDER"])
    bars = alt.Chart(by_day).mark_bar(cornerRadiusEnd=3, size=22, color=ui.token("PRIMARY")).encode(
        x=alt.X("date:N", title=None, axis=alt.Axis(grid=False, labelAngle=0, **axis)),
        y=alt.Y("% of bankroll:Q", title="% of bankroll", axis=alt.Axis(grid=True, gridColor=t["BORDER"], **axis)),
        tooltip=[alt.Tooltip("date:N"), alt.Tooltip("% of bankroll:Q", format=".2f"),
                 alt.Tooltip("tips:Q", title="tips staked", format=",")])
    if cap_fraction is None or pd.isna(cap_fraction):
        return bars.properties(height=h)
    cap = alt.Chart(pd.DataFrame({"cap": [100 * float(cap_fraction)]})).mark_rule(
        color=t["BORDER_STRONG"], strokeDash=[4, 4], size=1).encode(y="cap:Q")
    return (bars + cap).properties(height=h, title="Total stake per day — dashed line is the daily cap")


def drawdown_chart(path: pd.DataFrame, h: int = 240):
    """Replayed bankroll and its running peak on ONE scale (the gap between them is the drawdown)."""
    import altair as alt
    t = ui.tokens()
    axis = dict(labelColor=t["TEXT_MUTED"], titleColor=t["TEXT_MUTED"], domainColor=t["BORDER"], tickColor=t["BORDER"])
    d = path[["date", "bankroll", "peak"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    long = d.melt("date", var_name="series", value_name="value")
    return alt.Chart(long).mark_line(size=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(grid=False, **axis)),
        y=alt.Y("value:Q", title="bankroll (start 100)", scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True, gridColor=t["BORDER"], **axis)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["bankroll", "peak"],
                                                    range=[ui.repo_colour("v9"), t["TEXT_MUTED"]]),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["bankroll", "peak"], range=[[1, 0], [4, 4]]),
                                  legend=None),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=".1f")],
    ).properties(height=h)


CANDIDATE_CONFIG = {"Ending": ui.num_col("Ending"), "Max DD": ui.num_col("Max DD", fmt="%.3f"),
                    "Median (resampled)": ui.num_col("Median (resampled)"), "5th pct": ui.num_col("5th pct"),
                    "P(DD>10%)": ui.num_col("P(DD>10%)", fmt="%.3f"), "P(DD>20%)": ui.num_col("P(DD>20%)", fmt="%.3f"),
                    "bets": st.column_config.NumberColumn("Bets", format="%d"), "score": ui.num_col("Score", fmt="%.3f")}
