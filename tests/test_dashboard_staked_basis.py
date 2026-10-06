"""The dashboard must never present a paper tier as money.

VALUABLE is half-stake monitor. Pooling it into a P/L headline inflated the standard track from
-9.17u to -65.11u and the league chart from -35.66u to -123.16u — numbers the owner was reading
as results.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard_data import v9

PAGE = Path(__file__).resolve().parents[1] / "pages" / "0_🏠_Overview.py"


def test_by_league_is_staked_only_by_default():
    a = v9.by_league(days=None)
    b = v9.by_league(days=None, staked_only=False)
    if a.empty:
        return
    assert a["n"].sum() < b["n"].sum(), "the default is already pooling the paper tier"


def test_every_track_exposes_a_staked_only_figure():
    p = v9.performance()
    if not p.get("available"):
        return
    for name, t in p["by_track"].items():
        s = t["staked_only"]
        assert s["n"] <= t["n"], f"{name}: more staked bets than tips"
        for k in ("n", "pnl", "roi", "hit", "break_even"):
            assert k in s, f"{name}: staked_only is missing {k}"


def test_a_hit_rate_never_travels_without_its_break_even():
    """45% is excellent at 2.40 and a disaster at 1.70. A hit rate shown without the bar it
    must clear invites exactly the wrong read."""
    bl = v9.by_league(days=None)
    if bl.empty:
        return
    assert {"break_even", "excess_hit"} <= set(bl.columns)
    assert bl["break_even"].notna().any()


def test_break_even_is_mean_of_inverse_odds_not_inverse_of_mean():
    import numpy as np
    f = pd.DataFrame({"odds": [1.5, 4.5]})
    got = v9._break_even(f)
    assert got == round(float(np.mean([1 / 1.5, 1 / 4.5])), 4)
    assert got != round(float(1 / np.mean([1.5, 4.5])), 4)


def test_the_overview_page_asks_for_staked_only():
    src = PAGE.read_text(encoding="utf-8")
    assert "staked_only=True" in src


def test_the_overview_cards_use_the_staked_figures():
    """The headline said 'SNIPER + MARKSMAN only' while the cards beside it showed all tips —
    three numbers that looked comparable and were not."""
    src = PAGE.read_text(encoding="utf-8")
    block = src[src.index("for c, (name, t) in zip"):src.index("st.caption", src.index("for c, (name, t) in zip"))]
    assert 'staked_only' in block
    assert 't["pnl"]' not in block and "t['pnl']" not in block


def test_the_league_chart_no_longer_compares_against_an_estate_mean():
    """One shared reference line makes a short-priced league look strong purely from its prices."""
    src = PAGE.read_text(encoding="utf-8")
    assert "estate mean" not in src
    assert "excess_hit" in src
