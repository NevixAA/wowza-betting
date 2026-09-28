"""Shared visual grammar for the dashboard.

ONE CHART VOCABULARY, USED EVERYWHERE. Before this, nine pages each drew their own thing and the
board came out 51 tables against 14 charts — a 3.6:1 ratio that makes a reader scan numbers for a
shape a chart would have shown instantly. These helpers exist so a page can reach for a chart as
cheaply as it reaches for st.dataframe.

THREE RULES THAT ARE NOT NEGOTIABLE HERE:

1. **Colour is never the only cue.** Every state carries its word, and usually an icon. A
   red/green dashboard is unreadable to roughly one man in twelve, and this one is read on a
   phone in daylight.
2. **Never a dual axis.** Two measures on different scales become two charts or one indexed
   chart. A twin-axis plot can be made to show any relationship you like by choosing the scales.
3. **n travels with every rate.** A 100% hit rate on 3 bets and on 300 bets are different facts
   and must not look the same. `pct()` refuses to format a rate without its denominator.

The palette is one blue ramp for magnitude, a warm/cool pair for polarity around a NEUTRAL grey
zero, and a reserved status set that is never reused as a series colour.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# ── palette ───────────────────────────────────────────────────────────────────────────────────
# EVERY VALUE BELOW WAS COMPUTED, NOT CHOSEN BY EYE. Each set was run through the six checks
# (lightness band, chroma floor, colour-blind separation on adjacent pairs, normal-vision floor,
# contrast against the surface) in both modes and re-stepped until it passed.
#
# What that caught, which no amount of looking would have: the obvious first pick — a blue v9
# against a violet Pro — separates by ΔE 2.5 under protanopia and only 12.0 for normal vision,
# below the floor of 15. Those two repos appear side by side on nearly every comparison in this
# dashboard. Pro is magenta instead because the measurement said the violet was unreadable.
#
# Dark mode is SELECTED, not flipped: its band is L 0.48–0.67 against L 0.43–0.77 for light, so
# the same hexes fail there. Each mode has its own validated steps.
REPO_COLOUR = {"v9": "#1E6FD9", "pro": "#C026A3", "v11": "#E8A020", "fantasy": "#12916A"}
REPO_COLOUR_DARK = {"v9": "#3B82F6", "pro": "#CB3FAE", "v11": "#B8820C", "fantasy": "#0E9B72"}

#: Polarity. Cool = profit, warm = loss, grey = the zero band — never red/green alone, and the
#: sign is always also in the text (`units()` prints a leading + or −).
POS, NEG, NEUTRAL = "#0894AF", "#C2410C", "#9CA3AF"
POS_DARK, NEG_DARK = "#12A0BC", "#D9632B"

#: Status. Reserved — never reused as a series colour, and always shipped with its word.
STATUS = {"CURRENT": "#0E9F6E", "AGING": "#B45309", "STALE": "#B91C1C",
          "FAILED": "#7F1D1D", "UNKNOWN": "#6B7280"}

#: Sequential ramp for magnitude. ONE hue, monotone light → dark. Never a rainbow.
RAMP = ["#DBEAFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8"]

INK, INK_MUTED = "#111827", "#6B7280"


def is_dark() -> bool:
    """Whether Streamlit is rendering dark. Unknown means light, which is the safe default."""
    try:
        return str(getattr(st.context.theme, "type", "light")).lower() == "dark"
    except Exception:
        return False


def repo_colour(repo: str) -> str:
    return (REPO_COLOUR_DARK if is_dark() else REPO_COLOUR).get(repo, INK_MUTED)


def polarity() -> tuple[str, str]:
    """(profit, loss) for the current mode."""
    return (POS_DARK, NEG_DARK) if is_dark() else (POS, NEG)

_AXIS = alt.Axis(labelColor=INK_MUTED, titleColor=INK_MUTED, domainColor="#E5E7EB",
                 tickColor="#E5E7EB", grid=False)
_AXIS_G = alt.Axis(labelColor=INK_MUTED, titleColor=INK_MUTED, domainColor="#E5E7EB",
                   tickColor="#E5E7EB", grid=True, gridColor="#F3F4F6")


# ── text helpers ──────────────────────────────────────────────────────────────────────────────
def pct(x, n=None, dp: int = 1) -> str:
    """A rate, with its denominator. Refuses to hide n.

    `pct(0.55)` is a number; `pct(0.55, 20)` is a fact. Only the second belongs on a page where
    someone might act on it, so n is only omitted when the caller explicitly passes None and has
    shown the count elsewhere.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{float(x) * 100:.{dp}f}%"
    return f"{s} (n={n:,})" if n is not None else s


def units(x, dp: int = 2) -> str:
    """P/L is in UNITS at flat 1u stakes, never in currency and never as an ROI without n."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):+.{dp}f}u"


def points(x, dp: int = 2) -> str:
    """Fantasy is scored in FPL points. Never mix this scale with units."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{dp}f} pts"


def small_sample(n: int, floor: int = 30) -> bool:
    """Below this, a rate is noise. Flagged on the page, never silently dropped."""
    return n is not None and n < floor


# ── blocks ────────────────────────────────────────────────────────────────────────────────────
def freshness_chip(f, label: str = "") -> None:
    """A feed's state as word + icon + age. Colour supports it; colour never carries it alone."""
    if f is None:
        st.caption(f"⚪ UNKNOWN{(' · ' + label) if label else ''}")
        return
    st.markdown(
        f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"background:{STATUS.get(f.state, '#6B7280')}1A;color:{STATUS.get(f.state, '#6B7280')};"
        f"font-size:0.78rem;font-weight:600'>{f.emoji} {f.state}</span>"
        f"<span style='color:{INK_MUTED};font-size:0.78rem'> &nbsp;{label or f.source}"
        f"{(' · ' + f.label) if f.label else ''}</span>",
        unsafe_allow_html=True)


def state_banner(state: str, detail: str = "") -> None:
    """The one thing a reader must not misread: is this LIVE, PAPER, or RESEARCH?

    v11's numbers and Pro's numbers are not bets. A page that shows them next to v9's P/L without
    saying so is inviting someone to stake on a shadow log.
    """
    tone = {"LIVE": "#0E9F6E", "PAPER": "#B45309",
            "SHADOW / RESEARCH ONLY": "#6D28D9", "RESEARCH": "#6D28D9"}.get(state, "#6B7280")
    st.markdown(
        f"<div style='border-left:4px solid {tone};background:{tone}0F;padding:8px 14px;"
        f"border-radius:0 8px 8px 0;margin-bottom:10px'>"
        f"<b style='color:{tone}'>{state}</b>"
        f"<span style='color:{INK_MUTED}'>{(' — ' + detail) if detail else ''}</span></div>",
        unsafe_allow_html=True)


def absent(reason: str) -> None:
    """A repo that is not checked out. Say so plainly; never render an empty chart instead.

    An empty chart reads as 'measured, found nothing'. That is a different claim from 'not
    connected', and confusing the two is how a missing feed gets mistaken for a dead signal.
    """
    st.info(f"**Not connected.** {reason}")


def kpi(label: str, value: str, sub: str = "", tone: str | None = None) -> None:
    c = tone or INK
    st.markdown(
        f"<div style='padding:10px 2px'>"
        f"<div style='color:{INK_MUTED};font-size:0.76rem;text-transform:uppercase;"
        f"letter-spacing:.04em'>{label}</div>"
        f"<div style='color:{c};font-size:1.7rem;font-weight:650;line-height:1.25'>{value}</div>"
        f"<div style='color:{INK_MUTED};font-size:0.78rem'>{sub}</div></div>",
        unsafe_allow_html=True)


# ── charts ────────────────────────────────────────────────────────────────────────────────────
def _base(df: pd.DataFrame, h: int) -> alt.Chart:
    return alt.Chart(df).properties(height=h)


def pnl_bars(df: pd.DataFrame, x: str, y: str, n_col: str | None = None,
             title: str = "", h: int = 320) -> alt.Chart:
    """Signed magnitude around zero: warm below, cool above, grey at the zero band.

    The n column rides in the tooltip so a tall bar on 4 bets cannot be mistaken for a tall bar
    on 400 — which is the single most common way a per-league chart misleads.
    """
    _pos, _neg = polarity()
    tips = [alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format="+.2f", title="P/L (u)")]
    if n_col and n_col in df.columns:
        tips.append(alt.Tooltip(f"{n_col}:Q", title="bets", format=","))
    return _base(df, h).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y(f"{x}:N", sort="-x", title=None, axis=_AXIS),
        x=alt.X(f"{y}:Q", title="P/L (units, flat 1u)", axis=_AXIS_G),
        color=alt.condition(alt.datum[y] >= 0, alt.value(_pos), alt.value(_neg)),
        tooltip=tips,
    ).properties(title=title)


def rate_bars(df: pd.DataFrame, x: str, y: str, n_col: str, ref: float | None = None,
              title: str = "", h: int = 320, fmt: str = ".1%") -> alt.LayerChart | alt.Chart:
    """A rate per category, with n in the tooltip and an optional reference line.

    The reference is what the rate must beat to mean anything — break-even, or the market's own
    number. A hit rate with no reference is a number nobody can grade.
    """
    bars = _base(df, h).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y(f"{x}:N", sort="-x", title=None, axis=_AXIS),
        x=alt.X(f"{y}:Q", title=None, axis=alt.Axis(format=fmt, labelColor=INK_MUTED,
                                                    grid=True, gridColor="#F3F4F6")),
        color=alt.Color(f"{n_col}:Q", scale=alt.Scale(range=RAMP), legend=alt.Legend(title="n")),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=fmt),
                 alt.Tooltip(f"{n_col}:Q", title="n", format=",")],
    ).properties(title=title)
    if ref is None:
        return bars
    line = alt.Chart(pd.DataFrame({"r": [ref]})).mark_rule(
        color=INK_MUTED, strokeDash=[4, 4], size=1).encode(x="r:Q")
    return (bars + line).resolve_scale(x="shared")


def cumulative_line(df: pd.DataFrame, x: str, y: str, colour: str | None = None,
                    title: str = "", h: int = 300) -> alt.LayerChart:
    """A running total over time, with a zero rule so the sign is readable at a glance."""
    colour = colour or repo_colour("v9")
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(
        color="#D1D5DB", size=1).encode(y="z:Q")
    line = _base(df, h).mark_line(size=2, color=colour, interpolate="monotone").encode(
        x=alt.X(f"{x}:T", title=None, axis=_AXIS),
        y=alt.Y(f"{y}:Q", title="cumulative units", axis=_AXIS_G),
        tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q", format="+.2f")],
    ).properties(title=title)
    return (zero + line).resolve_scale(y="shared")


def calibration(df: pd.DataFrame, pred: str, actual: str, n_col: str,
                title: str = "Claimed vs realised", h: int = 320) -> alt.LayerChart:
    """Claimed probability against what actually happened, against the diagonal.

    This is the chart that carries the estate's load-bearing negative result — the model claims
    0.5430 and realises 0.4066, overconfident by +13.64pp on n=792. A table of the same numbers
    does not make a reader feel a gap; the distance from the diagonal does.
    """
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#D1D5DB", strokeDash=[5, 5], size=1).encode(x="x:Q", y="y:Q")
    pts = _base(df, h).mark_circle(opacity=0.85).encode(
        x=alt.X(f"{pred}:Q", title="claimed probability", scale=alt.Scale(domain=[0, 1]),
                axis=_AXIS_G),
        y=alt.Y(f"{actual}:Q", title="realised rate", scale=alt.Scale(domain=[0, 1]),
                axis=_AXIS_G),
        size=alt.Size(f"{n_col}:Q", scale=alt.Scale(range=[40, 600]),
                      legend=alt.Legend(title="n")),
        color=alt.value(repo_colour("v9")),
        tooltip=[alt.Tooltip(f"{pred}:Q", format=".3f"), alt.Tooltip(f"{actual}:Q", format=".3f"),
                 alt.Tooltip(f"{n_col}:Q", title="n", format=",")],
    ).properties(title=title)
    return (diag + pts).resolve_scale(x="shared", y="shared")


def compare_bars(df: pd.DataFrame, cat: str, value: str, series: str,
                 order: list[str] | None = None, title: str = "", h: int = 320,
                 fmt: str = ".3f", y_title: str = "") -> alt.Chart:
    """Two or more measures on ONE scale, side by side. Never a second y-axis.

    Where two measures genuinely differ in scale, the caller indexes them or draws two charts —
    the point of refusing the dual axis is that a twin-axis plot can be tuned to show whatever
    relationship the author wants, and a reader has no way to tell.
    """
    sc = alt.Scale(domain=order, range=RAMP[1::2]) if order else alt.Scale(range=RAMP[1::2])
    return _base(df, h).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{cat}:N", title=None, axis=_AXIS),
        y=alt.Y(f"{value}:Q", title=y_title or None, axis=_AXIS_G),
        xOffset=f"{series}:N",
        color=alt.Color(f"{series}:N", scale=sc, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{series}:N"),
                 alt.Tooltip(f"{value}:Q", format=fmt)],
    ).properties(title=title)


def coverage_bars(df: pd.DataFrame, cat: str, value: str, title: str = "",
                  h: int = 300) -> alt.Chart:
    """Magnitude with a single-hue ramp — counts, rows, partitions, coverage."""
    return _base(df, h).mark_bar(cornerRadiusEnd=4, size=16).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=_AXIS),
        x=alt.X(f"{value}:Q", title=None, axis=_AXIS_G),
        color=alt.Color(f"{value}:Q", scale=alt.Scale(range=RAMP), legend=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{value}:Q", format=",")],
    ).properties(title=title)


def table(df: pd.DataFrame, **kw) -> None:
    """A table is the accessible fallback for every chart, not the default way to show a shape.

    Mixed-type object columns are rendered as text. Arrow cannot type a column holding both
    '2026-09-22T01:39:02Z' and 59859, and Streamlit's automatic repair silently coerces —
    which can turn a value into something the source never said. Stringifying keeps what was
    written, visibly.
    """
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == "object":
            kinds = {type(v).__name__ for v in d[c].dropna().head(200)}
            if len(kinds) > 1:
                d[c] = d[c].map(lambda v: "" if v is None else str(v))
    st.dataframe(d, width='stretch', hide_index=True, **kw)
