"""
Shared dashboard presentation helpers.
=====================================
There was no shared UI module, which is the root of the dashboard looking inconsistent: seven
pages each rolled their own headers, their own auto-refresh, and raw `st.dataframe` calls with no
column formatting anywhere in the entire app. Every table rendered as an undifferentiated grid and
every section carried the same visual weight, so nothing read as important.

This is presentation only. It imports no model code, touches no output file, and is used by no
workflow — the dashboard is not referenced in any of v9's ~20 workflows, so changes here cannot
affect notify or collect.

THREE THINGS IT FIXES

1. **Numbers stay numbers.** The Fantasy page formatted probabilities as STRINGS (`"73%"`) before
   display, which silently broke sorting: clicking P(goal) descending gave 9%, 8%, 73%, 45%, 100%
   in that order, because they sort lexicographically. `pct_col` and `num_col` format at RENDER
   time via column_config, so the underlying value stays numeric and sorts correctly.

2. **Deprecated APIs, both already past their removal dates.** `components.v1.html` (removal
   announced for 2026-06-01) was used on four pages for a JS auto-refresh, and
   `use_container_width` (2025-12-31) appears 29 times across eight pages. `autorefresh` here uses
   `st_autorefresh`, which is already in requirements.txt and already used correctly by
   `1_Dashboard.py` — the other pages simply never got updated.

3. **Bars instead of bare digits.** `ProgressColumn` renders a value as a filled bar, which turns
   a column of numbers into something scannable at a glance. This is the single biggest visual
   change available and it costs one config object per column.

Colour choices are deliberately left to Streamlit's theme. Hardcoding hex values breaks the
viewer's light/dark setting, and a dashboard that only reads well in one theme is a regression for
whoever uses the other.
"""
from __future__ import annotations

import streamlit as st


def autorefresh(minutes: int = 2, key: str = "refresh") -> None:
    """Re-run the page every `minutes`. Degrades to no refresh rather than to a broken page.

    Replaces `components.v1.html("<script>setTimeout(...reload...)</script>")`, whose announced
    removal date has passed. The JS hack also reloaded the whole browser tab rather than re-running
    the script, so it discarded widget state — a filter the user had set was silently reset every
    two minutes.
    """
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=minutes * 60 * 1000, key=key)
    except Exception:
        # A missing optional dependency must not take the page down; the data is still current
        # on any manual interaction.
        pass


def page_header(title: str, subtitle: str = "", *, badge: str = "") -> None:
    """One consistent page header.

    `st.title` + `st.caption` was already the de-facto pattern; this makes it uniform and adds an
    optional badge so a page can state what family it belongs to (FANTASY vs betting) without
    spending a whole line of body text on it.
    """
    st.title(title)
    line = subtitle
    if badge:
        line = f"{badge} &nbsp;·&nbsp; {subtitle}" if subtitle else badge
    if line:
        st.caption(line, unsafe_allow_html=True)


def pct_col(label: str, *, help: str = "", width: str | None = None):
    """A 0-1 probability shown as a percentage, still sorting numerically."""
    return st.column_config.NumberColumn(label, help=help or None, format="percent",
                                         width=width)


def bar_col(label: str, *, max_value: float, help: str = "", fmt: str = "%.2f"):
    """A value rendered as a filled bar. Needs an explicit max — a bar without a
    scale is decoration, and an auto-scaled one changes meaning as the data changes."""
    return st.column_config.ProgressColumn(label, help=help or None, format=fmt,
                                           min_value=0, max_value=max_value)


def num_col(label: str, *, fmt: str = "%.2f", help: str = ""):
    return st.column_config.NumberColumn(label, help=help or None, format=fmt)


def money_col(label: str = "£m", *, help: str = ""):
    return st.column_config.NumberColumn(label, help=help or None, format="£%.1f")


# Column-name -> how to render it. Keyed on the DISPLAY label, because every page renames
# columns for display before calling st.dataframe.
#
# One shared map rather than a hand-written config per table. There are 15 tables on the Fantasy
# page alone and 11 more across the other pages; configuring each by hand guarantees they drift
# apart, which is how the dashboard came to look like seven different products. Anything not
# recognised here simply renders as Streamlit's default, so an unknown column is never hidden or
# mangled — it just misses out on formatting.
# Percentage columns, split by the CONVENTION OF THEIR SOURCE rather than detected from the
# data. Both conventions genuinely exist here: the prop models emit 0-1 probabilities, while
# FPL's ownership figures arrive as 0-100.
#
# I tried detecting the scale from the values (max > 1.5 => already a percentage) and it is
# WRONG, because a filter can move the max. The Differentials tab has a max-ownership slider
# starting at 1.0, so a user narrowing it to 1% leaves a column whose maximum is below the
# threshold — and 0.8% would then render as 80%. A format that changes when you move a slider is
# worse than one that is merely inconvenient to maintain.
_PCT_FRACTION = ("Start%", "P(goal)", "P(assist)", "P(SOT2+)", "P(start)", "Hit %", "Win %")
_PCT_OF_100 = ("Owned %", "Own %", "League own %", "Selected %")
_PCT = _PCT_FRACTION + _PCT_OF_100
_MONEY = ("£m", "Price")
_POINTS = ("Exp pts", "Pts", "xPts", "Projected", "Total pts", "xpts")
_PLAIN2 = ("Pts/£", "Def", "CS", "Bon", "xPts·rot", "Avg FDR", "FDR", "MAE", "Bias",
           "err_ours", "err_fpl", "mae_ours", "mae_fpl")
_INT = ("#", "GW", "gw", "Pens (career)", "Set-piece goals", "n_dgw", "n_bgw", "players")


def auto_config(df, *, bars: tuple[str, ...] = ()) -> dict:
    """A column_config for whatever recognised columns `df` has.

    `bars` names columns to render as ProgressColumn; their max is taken from the data actually
    being shown, so the bar is a comparison within this table rather than against an arbitrary
    constant.
    """
    import pandas as pd
    cfg: dict = {}
    for c in df.columns:
        if c in bars:
            try:
                mx = float(pd.to_numeric(df[c], errors="coerce").max())
            except Exception:
                mx = 1.0
            cfg[c] = bar_col(c, max_value=max(mx, 0.1))
        elif c in _PCT_OF_100:
            # Already a percentage: show the number with a % suffix. Using format="percent" here
            # multiplies by 100 again and renders FPL's 74.9% ownership as 7,490%.
            cfg[c] = num_col(c, fmt="%.1f%%")
        elif c in _PCT_FRACTION:
            cfg[c] = pct_col(c)
        elif c in _MONEY:
            cfg[c] = money_col(c)
        elif c in _POINTS or c in _PLAIN2:
            cfg[c] = num_col(c, fmt="%.2f")
        elif c in _INT:
            cfg[c] = st.column_config.NumberColumn(c, format="%d", width="small")
    return cfg


def table(df, *, bars: tuple[str, ...] = (), height: int | None = None,
          container=None, **kw) -> None:
    """`st.dataframe` with shared formatting applied. Drop-in for the raw calls.

    `height` is OMITTED when None rather than passed through. Streamlit rejects
    `height=None` with "Height must be either a positive integer, 'stretch', or 'content'",
    and because most of these tables sit inside a `try/except` that shows a warning, the first
    version failed SILENTLY on two of them — the page still rendered, with two sections replaced
    by "Differentials unavailable: Invalid height value: None". What caught it was the dataframe
    count dropping from 15 to 13, not any error surfacing.
    """
    target = container if container is not None else st
    if height is not None:
        kw["height"] = height
    target.dataframe(df, hide_index=True, width="stretch",
                     column_config=auto_config(df, bars=bars), **kw)


def metric_row(items: list[dict], *, per_row: int = 4) -> None:
    """Metrics laid out in rows of `per_row`.

    `st.columns(len(items))` with a variable-length list produces one very wide row that squeezes
    each metric to unreadable width as the list grows. Chunking keeps them legible.
    """
    if not items:
        return
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        for col, it in zip(st.columns(len(chunk)), chunk):
            col.metric(it.get("label", ""), it.get("value", ""),
                       delta=it.get("delta"), help=it.get("help"))


def player_card(container, *, name: str, position: str, points: float,
                team: str = "", price: float | None = None, fixture: str = "",
                p_goal: float | None = None, p_assist: float | None = None,
                flag: str = "") -> None:
    """A captaincy/pick card with the useful numbers VISIBLE.

    The previous version was a bare `st.metric` with price, next fixture and the goal/assist
    probabilities all buried in a `help=` tooltip — invisible on touch devices and invisible to
    anyone who does not know to hover. The whole point of a captaincy pick is comparing those
    numbers across three candidates, so they belong on the face of the card.
    """
    with container.container(border=True):
        st.markdown(f"**{flag}{name}**")
        meta = " · ".join(x for x in (position, team,
                                     f"£{price:.1f}m" if price is not None else "") if x)
        if meta:
            st.caption(meta)
        st.metric("Expected points", f"{points:.2f}")
        bits = []
        if p_goal is not None and p_goal == p_goal:
            bits.append(f"⚽ {p_goal:.0%}")
        if p_assist is not None and p_assist == p_assist:
            bits.append(f"🅰️ {p_assist:.0%}")
        if bits:
            st.caption(" · ".join(bits))
        if fixture:
            st.caption(f"Next: {fixture}")
