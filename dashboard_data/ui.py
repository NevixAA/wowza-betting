"""Back-compat shim: the dashboard's visual grammar now lives in ONE module, `dashboard_ui`.

Until the 2026-10-25 visual redesign this file held a second, parallel set of helpers (palette,
chips, banners, charts, number formats) that disagreed with `dashboard_ui.py`. Everything moved
there; this module only re-exports it so the pages that import `dashboard_data.ui` keep working
unchanged. Do not add helpers here — add them to dashboard_ui.py.

The one behavioural difference kept on purpose: `table()` here does NOT apply dashboard_ui's
automatic column formats. Pages calling it pass frames whose columns ("Hit %" on a 0-100 scale,
for instance) were never written for that map, and auto-formatting them would change what they
show. Mixed-type columns are still stringified, exactly as before.
"""
from __future__ import annotations

import dashboard_ui as _ds
from dashboard_ui import (  # noqa: F401  (re-exported)
    CATEGORICAL, DARK, INK, INK_MUTED, LIGHT, NEG, NEG_DARK, NEUTRAL, POS, POS_DARK, RAMP,
    REPO_COLOUR, REPO_COLOUR_DARK, STATUS, STATUSES, TONE_COLOUR, absent, calibration, compare_bars,
    coverage_bars, cumulative_line, empty_state, evidence_badge, fmt_clv, fmt_money, fmt_odds,
    fmt_prob, fmt_roi, fmt_units, fmt_xpts, freshness_badge, freshness_chip, is_dark, kpi,
    metric_card, metric_grid, page_header, pct, pnl_bars, points, polarity, rate_bars, repo_colour,
    sample_marker, section_header, small_sample, state_banner, status_chip, token, tokens, units,
)


def table(df, **kw) -> None:
    """A table is the accessible fallback for every chart, not the default way to show a shape."""
    _ds.table(df, auto_format=False, **kw)
