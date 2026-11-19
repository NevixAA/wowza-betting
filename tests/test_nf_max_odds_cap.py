"""The GLOBAL new-format O/U 2.5 price cap (config.NF_MAX_OU_ODDS, owner 2026-10-22) no longer decides a served
tip (owner 2026-11-09: "no global vars"). An odds limit now belongs to the league itself: a bars_opt cell's
odds_min / odds_max (tests/test_bars_serving.py), or a legacy entry's own min_over_odds / min_under_odds."""
import pandas as pd

import config
from src.betting import evaluate_value
import pytest


@pytest.fixture(autouse=True)
def _own_bars(monkeypatch):
    """Owner 2026-11-09: no global bars. Every league gets the historical values as its OWN owner-applied
    bars (tests/own_bars.install_historical)."""
    from tests.own_bars import install_historical
    install_historical(monkeypatch)


def _row(league, p_over, odds_over, odds_under):
    return {"league": league, "date": "2026-10-25", "home_team": "A", "away_team": "B", "p_over25": p_over,
            "odds_over25": odds_over, "odds_under25": odds_under,
            "home_scored_last5": 7, "away_scored_last5": 6, "home_conceded_last5": 5, "away_conceded_last5": 6}


def _nf_and_std():
    nf = next(l for l in sorted(config.NEW_FORMAT_LEAGUES) if l in config.ENABLED_LEAGUES)
    std = next(l for l in sorted(config.STANDARD_FORMAT_LEAGUES) if l in config.ENABLED_LEAGUES)
    return nf, std


def test_global_nf_cap_no_longer_blocks_a_tip():
    nf, std = _nf_and_std()
    df = evaluate_value(pd.DataFrame([_row(nf, 0.45, 3.40, 1.33), _row(std, 0.45, 3.40, 1.33)]))
    for _, r in df.iterrows():
        assert r["bet"] == "OVER" and not bool(r["max_odds_blocked"])


def test_global_nf_cap_value_is_not_read_by_serving(monkeypatch):
    nf, _ = _nf_and_std()
    monkeypatch.setattr(config, "NF_MAX_OU_ODDS", 1.01)            # would block every price if it were read
    df = evaluate_value(pd.DataFrame([_row(nf, 0.60, 2.40, 1.62)]))
    assert df.iloc[0]["bet"] == "OVER" and not bool(df.iloc[0]["max_odds_blocked"])


def test_global_min_odds_value_is_not_read_by_serving(monkeypatch):
    nf, _ = _nf_and_std()
    monkeypatch.setattr(config, "MIN_OVER_ODDS", 99.0)             # would block every price if it were read
    df = evaluate_value(pd.DataFrame([_row(nf, 0.60, 2.40, 1.62)]))
    assert df.iloc[0]["bet"] == "OVER"
