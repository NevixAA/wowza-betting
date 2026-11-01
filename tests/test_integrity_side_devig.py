"""SHADOW side-market de-vig vs production's 1.08 rule (decision-engine finding, 2026-11-01).

    python -m pytest tests/test_integrity_side_devig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integrity import side_edge_devig as SE  # noqa: E402
from src.market_edge import power_devig  # noqa: E402


def _q(rows):
    return pd.DataFrame(rows, columns=["snapshot_ts", "match", "bookmaker", "market", "side", "odds"])


def _row(**kw):
    r = {"league": "League Two", "home_team": "H", "away_team": "A",
         "kickoff_utc": "2026-11-01T15:00:00Z", "market": "btts", "p_model": 0.54, "odds": 1.90,
         "asof": "2026-11-01T12:00:00Z", "model_type": "standard"}
    r.update(kw)
    return pd.DataFrame([r])


def test_power_devig_symmetric_and_bounded():
    assert power_devig(1.9, 1.9) == pytest.approx(0.5)
    p = power_devig(1.5, 2.6)
    assert 0.6 < p < 0.67
    assert power_devig(1.0, 2.0) is None and power_devig(None, 2.0) is None


def test_tip_that_exists_only_because_of_the_1_08_assumption():
    # Low-margin pair (overround ~3.1%): production divides 1/1.90 by 1.08 -> fair 0.487, edge 5.3pp
    # (VALUABLE). De-vigged fair ~0.510 -> edge 3.0pp, below the 4pp bar.
    q = _q([["2026-11-01T10:00:00Z", "H vs A", "sharp", "BTTS", "YES", 1.90],
            ["2026-11-01T10:00:00Z", "H vs A", "sharp", "BTTS", "NO", 1.98]])
    ev = SE.evaluate_rows(_row(), q, {})
    r = ev.iloc[0]
    assert r.tier_prod == "VALUABLE" and r.tier_shadow == "AVOID"
    assert r.classification == "ONLY_BECAUSE_OF_1_08" and r.devig_state == "DEVIGGED_TWO_SIDED"
    assert r.edge_prod == pytest.approx(0.54 - (1 / 1.90) / 1.08)


def test_eight_percent_book_reproduces_production_and_is_confirmed():
    q = _q([["2026-11-01T10:00:00Z", "H vs A", "b1", "BTTS", "YES", 1.80],
            ["2026-11-01T10:00:00Z", "H vs A", "b1", "BTTS", "NO", 1.92]])
    ev = SE.evaluate_rows(_row(p_model=0.62, odds=1.80), q, {})
    assert ev.iloc[0].classification == "CONFIRMED_BY_DEVIG"
    assert abs(ev.iloc[0].edge_prod - ev.iloc[0].edge_devig) < 0.01


def test_pairs_only_within_one_book_and_carry_forward():
    q = _q([["2026-11-01T09:00:00Z", "H vs A", "b1", "BTTS", "YES", 1.80],
            ["2026-11-01T09:00:00Z", "H vs A", "b2", "BTTS", "NO", 1.95],     # never paired across books
            ["2026-11-01T08:00:00Z", "H vs A", "b3", "BTTS", "YES", 1.85],
            ["2026-11-01T11:00:00Z", "H vs A", "b3", "BTTS", "NO", 1.95],     # carried-forward pair
            ["2026-11-01T13:00:00Z", "H vs A", "b4", "BTTS", "YES", 1.70],    # after decision time
            ["2026-11-01T13:00:00Z", "H vs A", "b4", "BTTS", "NO", 2.10]])
    ev = SE.evaluate_rows(_row(), q, {})
    assert ev.iloc[0].n_two_sided_books == 1 and ev.iloc[0].books == "b3"


def test_threshold_choice_matches_pipeline(monkeypatch):
    """Shadow tiers must use production's bars; prove it against pipeline._generate_side_bets."""
    import pipeline
    params = {"btts": {"League Two": {"approved": True, "sniper_th": 0.06, "marksman_th": 0.05},
                       "Serie B": {"approved": False, "sniper_th": 0.01, "marksman_th": 0.01}}}
    monkeypatch.setattr(pipeline, "_load_side_market_thresholds", lambda: params)
    preds = pd.DataFrame({"date": ["2026-11-01"] * 3, "league": ["League Two", "Serie B", "Championship"],
                          "home_team": ["a", "b", "c"], "away_team": ["x", "y", "z"],
                          "p_btts": [0.56, 0.56, 0.62], "odds_btts": [1.95, 1.95, 1.95],
                          "model_type": ["standard"] * 3})
    prod = pipeline._generate_side_bets(preds, {"btts": None}).set_index("league")["signal_tier"]
    for _, r in preds.iterrows():
        edge = r.p_btts - (1 / r.odds_btts) / 1.08
        t = SE.tier(edge, SE.thresholds(params, "btts", r.league))
        assert t == prod.get(r.league, "AVOID"), r.league


def test_append_first_sight_never_rewrites(tmp_path):
    p = tmp_path / "s.csv"
    q = _q([["2026-11-01T10:00:00Z", "H vs A", "b", "BTTS", "YES", 1.9],
            ["2026-11-01T10:00:00Z", "H vs A", "b", "BTTS", "NO", 1.98]])
    ev = SE.evaluate_rows(_row(), q, {})
    assert SE.append_first_sight(ev, p) == 1
    before = p.read_bytes()
    ev2 = SE.evaluate_rows(_row(p_model=0.70), q, {})
    assert SE.append_first_sight(ev2, p) == 0 and p.read_bytes() == before


@pytest.fixture(autouse=True)
def _legacy_side_fair_price(monkeypatch):
    """Synthetic rows here carry no book quotes, and these tests pin other behaviour. The two-sided
    de-vig fair price (production since 2026-11-01) has its own tests: test_side_devig_production.py."""
    import config as _cfg
    monkeypatch.setattr(_cfg, "SIDE_EDGE_FAIR_PRICE", "legacy_1_08")