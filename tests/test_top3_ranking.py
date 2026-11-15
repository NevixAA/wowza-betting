"""Research TOP-3 ranking rules (src/research/top3_ranking.py): parity with the study rule, never padded,
direction rule, per-cell statistics only, missing statistics never filled."""
import numpy as np
import pandas as pd

from src.bars_opt import study as S
from src.research import top3_ranking as R

ALL_RULES = R.DIAGNOSTIC_RULES[:1] + R.RULES + R.DIAGNOSTIC_RULES[1:]


def _bets(rows):
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    for k, v in {"track": "t", "pnl": 0.5, "odds": 2.0, "odds_c": 2.0, "tier2_ok": False, "band": "TOP_M",
                 "fair": 0.45, "p_used": 0.55, "won": True}.items():
        if k not in d:
            d[k] = v
    return d


def _rows():
    rows = [{"league": "A", "date": "2026-09-07", "row_id": f"f{i}", "mside": "btts_yes", "ev_est": 0.1 + i / 100}
            for i in range(5)]
    rows += [{"league": "B", "date": "2026-09-07", "row_id": "g1", "mside": "btts_yes", "ev_est": 0.2}]
    rows += [{"league": "A", "date": "2026-09-08", "row_id": "f4", "mside": "over25_under", "ev_est": 0.9}]
    return _bets(rows)


def test_ev_production_reproduces_the_study_rule():
    b = _rows()
    a = S.top3(b, "TOP_ONLY_V")
    t = R.select(b, pd.DataFrame(), "EV_PRODUCTION")
    assert list(a["row_id"] + a["mside"]) == list(t["row_id"] + t["mside"])


def test_every_rule_caps_at_three_never_pads_and_keeps_direction():
    b = _rows()
    for rule in ALL_RULES:
        t = R.select(b, pd.DataFrame(), rule)
        assert t.groupby(["league", "gw"]).size().max() == 3
        assert (t["league"] == "B").sum() == 1
        f4 = t[t["row_id"] == "f4"]
        assert f4["mside"].nunique() <= 1


def test_missing_statistic_ranks_last_and_is_never_filled():
    rng = np.random.default_rng(0)
    prior = _bets([{"league": "A", "date": f"2026-0{1 + i % 8}-1{i % 9}", "row_id": f"p{i}", "mside": "over25_over",
                    "ev_est": 0.1, "pnl": float(rng.choice([1.0, -1.0]))} for i in range(R.MIN_OWN_BETS)])
    st = R.cell_stats(prior)
    assert set(st["mside"]) == {"over25_over"} and st["lcb"].notna().all()
    b = _bets([{"league": "A", "date": "2026-09-07", "row_id": "x", "mside": "btts_yes", "ev_est": 0.9},
               {"league": "A", "date": "2026-09-07", "row_id": "y", "mside": "over25_over", "ev_est": 0.0}])
    t = R.select(b, st, "R1_CELL_LCB")
    assert list(t["row_id"]) == ["y", "x"]                 # the cell with a statistic first, the other not dropped
    assert t.set_index("row_id").loc["x", "no_stat"] == 1


def test_statistics_never_cross_leagues():
    rows = pd.DataFrame({"track": "t", "league": ["A"] * 60 + ["B"] * 60, "mside": "btts_yes",
                         "date": pd.Timestamp("2026-08-01"), "fair": 0.5, "odds": 2.0,
                         "p_used": np.r_[np.full(60, 0.6), np.full(60, 0.6)],
                         "won": np.r_[np.ones(60, bool), np.zeros(60, bool)]})
    lam = R.cell_stats(None, rows).set_index("league")["lam"]
    assert lam["A"] == 1.0 and lam["B"] == 0.0            # each league its own slope, clipped to [0, 1]
