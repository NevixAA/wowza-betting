"""Player Props results section (owner 2026-11-09: "the % isn't correct, the last 2 months were crazy results
and we don't show it"): period filter, every tip tier incl. VALUABLE, month and market tables, AVOID kept out,
P/L only from priced rows, model-era labels."""
import math

import pandas as pd
import pytest

from dashboard_data import props_view as V


def _led():
    rows = [
        # date, tier, result, odds, pnl, market
        ("2026-06-20", "MARKSMAN", "LOSS", 5.0, -1.0, "goals"),
        ("2026-06-21", "SNIPER", "LOSS", 6.0, -1.0, "goals"),
        ("2026-10-05", "VALUABLE", "WIN", 4.0, 3.0, "goals"),
        ("2026-10-06", "VALUABLE", "LOSS", 4.0, -1.0, "sot"),
        ("2026-10-07", "VALUABLE", "WIN", None, None, "sot"),        # settled but unpriced: counted, no P/L
        ("2026-10-08", "VALUABLE", "VOID", 3.0, None, "sot"),
        ("2026-11-02", "VALUABLE", None, 3.0, None, "cards"),        # pending
        ("2026-11-03", "MARKSMAN", "WIN", 5.0, 4.0, "assists"),
        ("2026-11-03", "AVOID", "WIN", 2.0, 1.0, "sot"),             # logged, not a tip
        ("2026-11-04", "AVOID", "LOSS", None, None, "sot"),
    ]
    return pd.DataFrame(rows, columns=["match_date", "tier", "result", "market_odds", "pnl", "market"])


TODAY = pd.Timestamp("2026-11-09")


def test_period_starts():
    assert V.period_start("Last 30 days", TODAY) == pd.Timestamp("2026-10-10")
    assert V.period_start("Last 60 days", TODAY) == pd.Timestamp("2026-09-10")
    assert V.period_start("This month", TODAY) == pd.Timestamp("2026-11-01")
    assert V.period_start("All time", TODAY) is None
    assert V.DEFAULT_PERIOD == "Last 60 days" and V.DEFAULT_PERIOD in V.PERIODS
    with pytest.raises(ValueError):
        V.period_start("Last week", TODAY)


def test_period_filter_drops_old_rows():
    d = V.prepare_ledger(_led())
    x = V.in_period(d, "Last 60 days", TODAY)
    assert x["md"].min() >= pd.Timestamp("2026-09-10")
    assert len(V.in_period(d, "All time", TODAY)) == len(d)


def test_tier_table_shows_valuable_and_keeps_avoid_out():
    d = V.in_period(V.prepare_ledger(_led()), "Last 60 days", TODAY)
    t = V.tier_table(d).set_index("Tier")
    assert {"VALUABLE", "MARKSMAN", "All tips"} <= set(t.index) and "AVOID" not in t.index
    v = t.loc["VALUABLE"]
    assert (v["Settled"], v["Wins"], v["Priced"]) == (3, 2, 2)      # the unpriced WIN is counted, no P/L
    assert v["Profit (u)"] == 2.0 and math.isclose(v["ROI %"], 100.0)  # (3 - 1) / 2 priced
    assert (v["Void"], v["Pending"]) == (1, 1)
    allt = t.loc["All tips"]
    assert allt["Settled"] == 4 and allt["Profit (u)"] == 6.0       # AVOID's +1 is not in the tip record
    assert V.avoid_count(d) == 2


def test_all_time_includes_the_june_rows():
    t = V.tier_table(V.in_period(V.prepare_ledger(_led()), "All time", TODAY)).set_index("Tier")
    assert t.loc["SNIPER", "Settled"] == 1 and t.loc["MARKSMAN", "Settled"] == 2


def test_month_and_market_tables():
    d = V.in_period(V.prepare_ledger(_led()), "All time", TODAY)
    m = V.month_table(d)
    assert list(m["Month"])[0] == "2026-11"                         # newest first
    octv = m[(m["Month"] == "2026-10") & (m["Tier"] == "VALUABLE")].iloc[0]
    assert (octv["Settled"], octv["Wins"], octv["Profit (u)"]) == (3, 2, 2.0)
    mk = V.market_table(d).set_index("Market")
    assert V.MARKET_LABEL["goals"] in mk.index and mk.loc[V.MARKET_LABEL["goals"], "Settled"] == 3


def test_small_sample_marked():
    t = V.tier_table(V.prepare_ledger(_led())).set_index("Tier")
    assert t.loc["VALUABLE", "Sample"] == "small sample"


def test_empty_period_and_empty_ledger():
    d = V.prepare_ledger(_led())
    assert V.tier_table(V.in_period(d, "This month", pd.Timestamp("2030-01-15"))).empty
    assert V.month_table(d.iloc[0:0]).empty and V.market_table(d.iloc[0:0]).empty


def test_model_era_labels():
    d = V.prepare_ledger(_led())
    assert set(V.model_era(d, None)) == {"pre-repair models"}
    e = V.model_era(d, "2026-11-01")
    assert e[d["md"] >= pd.Timestamp("2026-11-01")].eq("clean models").all()
    assert e[d["md"] < pd.Timestamp("2026-11-01")].eq("pre-repair models").all()
