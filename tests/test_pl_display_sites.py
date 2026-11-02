"""P/L sums and displays go through src.units at every site (issue #19): an exact half must round
half-up wherever it is shown - allocation cells, Telegram digests, dashboard views."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stored values whose exact sum is -0.015 but whose float sum is -0.01499999...
HALF = [0.1, 0.1, 0.1, -0.315]


def test_allocation_cell_pnl_is_exact_half_up():
    from src import allocation as A
    d = pd.DataFrame({"league": "Championship", "market": "ou25", "model_type": "standard",
                      "match_date": pd.date_range("2026-10-01", periods=4, freq="D"),
                      "odds": 2.0, "pnl": HALF, "result": ["WIN", "WIN", "WIN", "LOSS"],
                      "signal_tier": "SNIPER"})
    cells, _ = A.allocate(d)
    assert cells[0].pnl == -0.02


def test_builder_summary_total_is_exact():
    from dashboard_data import builder_view as B
    s = pd.DataFrame({"combo_result": ["WON", "WON", "WON", "LOST"],
                      "fair_combo_odds": [1.1, 1.1, 1.1, 2.0]})
    # returns 0.1 + 0.1 + 0.1 - 1.0, exact -0.7 (the float sum is -0.7000000000000001)
    assert B.settled_summary(s)["pnl"] == -0.7


def test_leagues_summarise_is_exact():
    from dashboard_data import leagues as L
    g = pd.DataFrame({"pnl": HALF, "odds": 2.0, "edge_pct": 5.0, "model_prob": 0.5, "implied": 0.5})
    s = L.summarise(g)
    assert s["pl"] == -0.015
    t = L.overview_table(pd.DataFrame([dict(league="X", paper=False, track="standard", rating=None, state="",
                                            regime="", won=3, lost=1, void=0, hit=0.75, roi=-0.00375,
                                            avg_odds=2.0, avg_edge=5.0, avg_model_prob=0.5, avg_implied=0.5,
                                            tips=4, pl=s["pl"], pl_7d=-0.015, valuable_pl=0.0, upcoming=0,
                                            best_market="", worst_market="")]))
    assert t["P/L (u)"].iloc[0] == -0.02 and t["P/L 7d (u)"].iloc[0] == -0.02


def test_notifier_and_daily_pack_have_no_float_pl_formatting():
    for rel in ("telegram_bot/notifier.py", "telegram_bot/daily_pack.py", "src/allocation.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert ":+.2f}u" not in src, rel
        assert "['pnl'].sum()" not in src and '["pnl"].sum()' not in src, rel


def test_notifier_cuts_on_match_date_like_every_other_page():
    from telegram_bot import notifier as N
    src = (ROOT / "telegram_bot" / "notifier.py").read_text(encoding="utf-8")
    assert '["generated_at"].astype(str).str[:10] >= app_config.PERFORMANCE_CUTOFF_DATE' not in src
    import config
    cut = pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE)
    d = pd.DataFrame({"match_date": [str((cut - pd.Timedelta(days=1)).date()), str(cut.date()), "", None],
                      "generated_at": [str(cut.date())] * 4})
    assert N._since_cutoff(d)["match_date"].tolist() == [str(cut.date())]
