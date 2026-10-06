"""MLS is paper-only from 2026-10-06: no tips, but still trained on and still collected.

The distinction matters and is easy to break. Removing a league from ENABLED_LEAGUES stops
PREDICTION and the backtest; it must NOT stop the league contributing to training or to odds
and results collection, or the decision silently becomes "delete the data" instead of
"stop betting it".
"""
from __future__ import annotations

import config

MLS = "USA MLS"


def test_mls_no_longer_generates_tips():
    assert MLS not in config.ENABLED_LEAGUES


def test_mls_still_trains_the_model():
    """ENABLED_LEAGUES gates prediction and the backtest. train_model() is fed the full
    frame, so a league outside it still improves the model — which is the whole point of
    STANDARD_FORMAT_LEAGUES being a superset (invariant 7)."""
    assert MLS in config.NEW_FORMAT_LEAGUES


def test_mls_still_has_an_odds_source_so_collection_continues():
    assert MLS in config.ODDS_API_SPORT_KEYS
    assert config.model_type_for_league(MLS) == "new_format"


def test_the_decision_records_its_evidence_in_config():
    """Invariant 7's rule: a league excluded from prediction carries its measured ROI beside
    it, so nobody re-litigates it from memory."""
    src = (config.__file__ and open(config.__file__, encoding="utf-8").read()) or ""
    i = src.index('# "USA MLS"')
    block = src[i:i + 1200]
    assert "-24.72u" in block and "EDGE_NOT_RANKING_OUTCOMES" in block
    assert "PREDICTION ONLY" in block or "prediction only" in block.lower()


def test_re_enabling_is_a_one_line_change():
    src = open(config.__file__, encoding="utf-8").read()
    assert 'Re-enable by uncommenting this one line' in src


def test_no_other_league_was_disabled_by_accident():
    """A config edit that silently dropped a second league would be invisible."""
    expected = {
        "League One", "League Two", "Bundesliga 2", "La Liga 2", "Ligue 2",
        "Championship", "Serie B",
        "Austrian Bundesliga", "Sweden Allsvenskan", "Norway Eliteserien",
        "Finland Veikkausliiga", "Denmark Superliga", "Ireland Premier Division",
        "Argentina Primera Division", "Brazil Serie A", "Japan J-League",
        "Mexico Liga MX", "China Super League", "Romanian Superliga",
    }
    assert config.ENABLED_LEAGUES == expected, (
        f"unexpected: +{config.ENABLED_LEAGUES - expected} -{expected - config.ENABLED_LEAGUES}")
