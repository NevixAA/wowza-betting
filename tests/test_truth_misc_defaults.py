"""Small Data Truth fixes: league tiers, injury flags, training weights (findings 33, 40, 47).

    python -m pytest tests/test_truth_misc_defaults.py -q
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from player_model import league_quality as LQ
from player_model.league_quality import league_tier
from src.api_football_ou import injury_features_from_context


# dt/props (2026-11-01): the hand LEAGUE_TIERS table is gone (owner rule: measured strength or
# missing). No measured cross-league strength exists in the repos, so every league - including the
# second divisions finding 40 was about, and an unknown league that used to get tier 3 - is NaN.
@pytest.mark.parametrize("league", ["La Liga 2", "Bundesliga 2", "Brazil Serie A", "Austrian Bundesliga",
                                    "Premier League", "World Cup - Qualification Europe",
                                    "Totally Unknown League", None])
def test_no_hand_tier_every_league_is_missing(league):
    assert math.isnan(league_tier(league))
    assert not hasattr(LQ, "LEAGUE_TIERS") and not hasattr(LQ, "NATIONAL_TEAM_TIERS")


def test_enrich_writes_nan_quality_columns():
    rows = [{"player_id": 1, "date": "2026-01-01", "league": "Premier League", "team": "A"}]
    out = LQ.enrich_league_quality(rows)[0]
    assert all(math.isnan(out[c]) for c in LQ.QUALITY_COLUMNS)


def test_injury_flags_are_nan_when_not_fetched():
    out = injury_features_from_context("A", "B", "2026-11-05", {})
    assert all(math.isnan(v) for v in out.values())
    ctx = {"fixture_ids": {}, "injuries": {}, "top_scorers": {}}
    out = injury_features_from_context("A", "B", "2026-11-05", ctx)
    assert all(math.isnan(v) for v in out.values())


def test_training_weights_follow_the_match_date():
    from src.standard_repair import derived_weights
    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2025-10-01", "2024-10-01", "2020-10-01"]),
                       "league": "Championship", "season": ["2026/27", "2025/26", "2024/25", "2020/21"]})
    w = derived_weights(df).tolist()
    assert w[0] == 4.0 and w[1] == 3.0 and w[2] == 2.0 and w[3] == 0.0       # newest highest, COVID 0
