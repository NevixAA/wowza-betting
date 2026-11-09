"""Side markets price edges on the two-sided de-vig in PRODUCTION (owner cutover 2026-11-01).

    python -m pytest tests/test_side_devig_production.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
import pipeline
from src.integrity import side_edge_devig as SE
from src.market_edge import power_devig

@pytest.fixture(autouse=True)
def _own_bars(monkeypatch):
    """Owner 2026-11-09: no global bars. These tests predate the cutover and assert the tiers the
    removed global path served, so every league gets exactly those values as its OWN owner-applied
    bars (tests/own_bars.install_historical)."""
    from tests.own_bars import install_historical
    install_historical(monkeypatch)


NOW = pd.Timestamp.now(tz="UTC")
KO = (NOW + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pred(home, p_btts, odds_btts):
    return {"date": NOW.strftime("%Y-%m-%d"), "league": "Brazil Serie A", "home_team": home,
            "away_team": f"{home} B", "p_btts": p_btts, "odds_btts": odds_btts,
            "model_type": "new_format", "kickoff_utc": KO}


def _quotes(rows):
    ts = (NOW - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([{"snapshot_ts": ts, "match_date": NOW.strftime("%Y-%m-%d"), "kickoff_utc": KO,
                          "match": m, "bookmaker": b, "market": "BTTS", "side": s, "odds": o}
                         for m, b, s, o in rows])


@pytest.fixture(autouse=True)
def _iso(monkeypatch):
    monkeypatch.setattr(pipeline, "_load_side_market_thresholds", lambda: {})
    monkeypatch.setattr(config, "SIDE_EDGE_FAIR_PRICE", "devig_two_sided")


def _run(monkeypatch, preds, quotes):
    monkeypatch.setattr(SE, "load_quotes", lambda *a, **k: quotes)
    return pipeline._generate_side_bets(pd.DataFrame(preds), {"btts": None})


def test_fair_prob_is_the_devigged_pair(monkeypatch):
    q = _quotes([("A vs A B", "bk1", "YES", 1.80), ("A vs A B", "bk1", "NO", 2.00)])
    out = _run(monkeypatch, [_pred("A", 0.66, 1.80)], q)
    assert len(out) == 1
    fair = power_devig(1.80, 2.00)
    assert out["fair_prob"].iloc[0] == pytest.approx(fair)
    assert out["edge"].iloc[0] == pytest.approx(0.66 - fair)


def test_one_sided_fixture_gets_no_tip(monkeypatch):
    q = _quotes([("A vs A B", "bk1", "YES", 1.80)])                 # NO side never quoted
    assert _run(monkeypatch, [_pred("A", 0.90, 1.80)], q).empty


def test_two_sides_from_different_books_is_not_a_pair(monkeypatch):
    q = _quotes([("A vs A B", "bk1", "YES", 1.80), ("A vs A B", "bk2", "NO", 2.00)])
    assert _run(monkeypatch, [_pred("A", 0.90, 1.80)], q).empty


def test_no_quotes_at_all_fails_closed(monkeypatch):
    assert _run(monkeypatch, [_pred("A", 0.90, 1.80)], _quotes([]).iloc[0:0]).empty


def test_edge_that_only_existed_under_1_08_is_dropped(monkeypatch):
    # Sharp pair (2% overround): 1.08 hands the model ~5pp; the real de-vig leaves < 4pp.
    q = _quotes([("A vs A B", "bk1", "YES", 1.92), ("A vs A B", "bk1", "NO", 2.00)])
    p = 0.53
    assert p - (1 / 1.92) / 1.08 >= 0.04                             # legacy would tip it
    assert _run(monkeypatch, [_pred("A", p, 1.92)], q).empty


def test_blind_fixture_gets_no_side_tip(monkeypatch):
    q = _quotes([("A vs A B", "bk1", "YES", 1.80), ("A vs A B", "bk1", "NO", 2.00),
                 ("C vs C B", "bk1", "YES", 1.80), ("C vs C B", "bk1", "NO", 2.00)])
    seen = dict(_pred("A", 0.66, 1.80), home_scored_last5=1.2, away_scored_last5=1.0,
                home_conceded_last5=1.1, away_conceded_last5=1.3)
    blind = dict(_pred("C", 0.66, 1.80), home_scored_last5=float("nan"), away_scored_last5=1.0,
                 home_conceded_last5=1.1, away_conceded_last5=1.3)
    out = _run(monkeypatch, [seen, blind], q)
    assert list(out["home_team"]) == ["A"]


def test_rollback_switch_restores_legacy(monkeypatch):
    monkeypatch.setattr(config, "SIDE_EDGE_FAIR_PRICE", "legacy_1_08")
    out = _run(monkeypatch, [_pred("A", 0.66, 1.80)], _quotes([]).iloc[0:0])
    assert len(out) == 1
    assert out["fair_prob"].iloc[0] == pytest.approx((1 / 1.80) / 1.08)
