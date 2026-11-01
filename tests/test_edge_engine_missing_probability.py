"""R2 finding 30: a missing model probability fails closed; it is never a 0.5 stand-in.

    python -m pytest tests/test_edge_engine_missing_probability.py -q
"""
import math

import pytest

from src import edge_engine as EE
from src import market_edge as ME


def test_candidate_has_no_default_probability():
    assert EE.Candidate(market="over25", over_odds=1.9).p_model is None


@pytest.mark.parametrize("bad", [None, float("nan"), -0.01, 1.01, "x"])
def test_classify_missing_or_invalid_probability_is_no_bet(bad):
    r = EE.classify(EE.Candidate(market="over25", over_odds=1.9, under_odds=1.95, p_model=bad,
                                 lineup_confirmed=True))
    assert r["tier"] == "NO_BET" and r["reason"] == EE.MISSING_MODEL_PROBABILITY
    assert r["p_blend"] is None


def test_real_probability_still_reaches_the_blend():
    r = EE.classify(EE.Candidate(market="over25", over_odds=1.9, under_odds=1.95, p_model=0.55,
                                 lineup_confirmed=True, n_eff=1000))
    assert r["reason"] != EE.MISSING_MODEL_PROBABILITY and r["p_blend"] is not None


def test_shadow_engine_market_edge_already_fails_closed_on_missing_model():
    """The serving shadow engine (market_edge.evaluate via edge_engine_shadow) refuses a missing
    p_model with MISSING_MODEL -- pinned here so it cannot regress to a default."""
    import pandas as pd
    now = pd.Timestamp("2026-11-01T10:00:00Z")
    c = ME.Candidate(family="OU25", track="standard", p_model=None,
                     kickoff=now + pd.Timedelta(hours=5), now=now)
    e = ME.evaluate(c, ME.load_params() if hasattr(ME, "load_params") else {})
    assert e["decision_state"] == "NO_BET" and "MISSING_MODEL" in e["decision_reason"]
    c2 = ME.Candidate(family="OU25", track="standard", p_model=math.nan,
                      kickoff=now + pd.Timedelta(hours=5), now=now)
    assert "MISSING_MODEL" in ME.evaluate(c2, {})["decision_reason"]
