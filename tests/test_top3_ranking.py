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


# ── protocol v4: the missing-statistic fallback never uses ev_est (owner "don't use the old ranking") ──
def _frame():
    return pd.DataFrame({"track": "t", "league": "A", "mside": ["m_lcb", "m_lam", "m_eb", "m_none", "m_none2"],
                         "fair": 0.5, "p_used": 0.6, "odds": 2.0, "ev_est": [0.01, 0.02, 0.03, 0.9, 0.8],
                         "kickoff_utc": ["2026-11-21T15:00Z", "2026-11-21T15:00Z", "2026-11-21T15:00Z",
                                         "2026-11-21T17:00Z", "2026-11-21T14:00Z"],
                         "fixture_id": ["1", "2", "3", "4", "5"]})


def _st():
    nan = np.nan
    return pd.DataFrame([
        {"track": "t", "league": "A", "mside": "m_lcb", "n": 50, "roi": .1, "se": .05, "lcb": .002, "eb": .1,
         "lam": nan, "lam_se": nan, "lam_rows": nan},
        {"track": "t", "league": "A", "mside": "m_lam", "n": 0, "roi": nan, "se": nan, "lcb": nan, "eb": nan,
         "lam": .5, "lam_se": .1, "lam_rows": 200},
        {"track": "t", "league": "A", "mside": "m_eb", "n": 50, "roi": .1, "se": .05, "lcb": nan, "eb": .07,
         "lam": nan, "lam_se": nan, "lam_rows": nan}])


def test_fallback_order_is_rule_then_r3_then_cell_eb_then_kickoff():
    o = R.rank_order(_frame(), _st(), "R1_CELL_LCB", tiebreak=("kickoff_utc", "fixture_id"))
    assert list(o["mside"]) == ["m_lcb", "m_lam", "m_eb", "m_none2", "m_none"]
    assert list(o["rank_basis"]) == ["RULE", "R3_KEY", "CELL_EB", "KICKOFF", "KICKOFF"]
    assert list(o["no_stat"]) == [0, 1, 1, 1, 1]


def test_ev_est_never_changes_a_rule_ranking():
    f = _frame()
    for rule in R.RULES:
        a = R.rank_order(f, _st(), rule, tiebreak=("kickoff_utc", "fixture_id"))["fixture_id"].tolist()
        for perm in ([5, -1, 3, 0, 2], [0, 0, 0, 0, 0]):
            g = f.assign(ev_est=perm)
            assert R.rank_order(g, _st(), rule, tiebreak=("kickoff_utc", "fixture_id"))["fixture_id"].tolist() == a
        g = f.drop(columns="ev_est")
        assert R.rank_order(g, _st(), rule, tiebreak=("kickoff_utc", "fixture_id"))["fixture_id"].tolist() == a
