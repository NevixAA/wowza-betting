"""Player Props earlier-signals record (owner 2026-11-09: "the % isn't correct ..."; 2026-11-22: no grouping by the
retired labels): period filter, one summary, month and market tables, AVOID kept out, P/L only from priced rows,
model-era labels."""
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


def test_record_table_is_one_summary_and_keeps_avoid_out():
    # owner 2026-11-22: the record is no longer grouped by the retired SNIPER / MARKSMAN / VALUABLE / PAPER labels
    d = V.in_period(V.prepare_ledger(_led()), "Last 60 days", TODAY)
    t = V.record_table(d).set_index("Record")
    assert list(t.index) == ["All prop signals"] and "Tier" not in V.record_table(d).columns
    allt = t.loc["All prop signals"]
    assert (allt["Settled"], allt["Wins"], allt["Priced"]) == (4, 3, 3)    # the unpriced WIN is counted, no P/L
    assert allt["Profit (u)"] == 6.0                                       # AVOID's +1 is not in the record
    assert (allt["Void"], allt["Pending"]) == (1, 1)
    assert V.avoid_count(d) == 2


def test_all_time_includes_the_june_rows():
    t = V.record_table(V.in_period(V.prepare_ledger(_led()), "All time", TODAY)).set_index("Record")
    assert t.loc["All prop signals", "Settled"] == 6


def test_month_and_market_tables():
    d = V.in_period(V.prepare_ledger(_led()), "All time", TODAY)
    m = V.month_table(d)
    assert list(m["Month"])[0] == "2026-11" and "Tier" not in m.columns   # newest first, no label grouping
    octv = m[m["Month"] == "2026-10"].iloc[0]
    assert (octv["Settled"], octv["Wins"], octv["Profit (u)"]) == (3, 2, 2.0)
    mk = V.market_table(d).set_index("Market")
    assert V.MARKET_LABEL["goals"] in mk.index and mk.loc[V.MARKET_LABEL["goals"], "Settled"] == 3


def test_small_sample_marked():
    t = V.record_table(V.prepare_ledger(_led())).set_index("Record")
    assert t.loc["All prop signals", "Sample"] == "small sample"


def test_empty_period_and_empty_ledger():
    d = V.prepare_ledger(_led())
    assert V.record_table(V.in_period(d, "This month", pd.Timestamp("2030-01-15"))).empty
    assert V.month_table(d.iloc[0:0]).empty and V.market_table(d.iloc[0:0]).empty


def test_model_era_labels():
    d = V.prepare_ledger(_led())
    assert set(V.model_era(d, None)) == {"pre-repair models"}
    e = V.model_era(d, "2026-11-01")
    assert e[d["md"] >= pd.Timestamp("2026-11-01")].eq("clean models").all()
    assert e[d["md"] < pd.Timestamp("2026-11-01")].eq("pre-repair models").all()
