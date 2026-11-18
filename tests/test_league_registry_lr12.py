"""LR-1.2 (2026-11-02): football-data codes + file family, historical league spellings, the
national-team competitions of player_history, and the evidence-based odds_api_props flag.

    python -m pytest tests/test_league_registry_lr12.py -q
"""
import copy
import json
from pathlib import Path

import pytest

from src import league_registry as LR
from src.season_resolver import resolve_api_season, is_known

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads(LR.REGISTRY_FILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("alias,canon", [
    ("Portugal Primeira Liga", "Portuguese Primeira Liga"), ("Primeira Liga", "Portuguese Primeira Liga"),
    ("Turkey Super Lig", "Turkish Super Lig"),
    ("Belgium Pro League", "Belgian First Division A"), ("Belgian Pro League", "Belgian First Division A"),
    ("Netherlands Eredivisie", "Dutch Eredivisie"), ("Eredivisie", "Dutch Eredivisie"),
    ("Switzerland Super League", "Swiss Super League"),
])
def test_historical_spellings_resolve_to_the_same_competition(alias, canon):
    assert LR.canonical_name(alias) == canon


@pytest.mark.parametrize("name", ["Denmark 1st Div", "Switzerland Challenge League",
                                  "Switzerland 1. Liga Promotion"])
def test_different_competitions_are_not_aliased(name):
    # 'Denmark 1st Div' is a different (second-tier) competition AND the label of 2,041 misfiled
    # Bundesliga 2 rows; the Swiss lower divisions are not the Super League.
    assert LR.canonical_name(name) is None


def test_football_data_codes_and_family():
    want = {"Premier League": "E0", "Bundesliga": "D1", "La Liga": "SP1", "Serie A": "I1",
            "Ligue 1": "F1", "Swiss Super League": "SWZ", "Poland Ekstraklasa": "POL"}
    for lg, code in want.items():
        assert LR.entry(lg)["football_data_code"] == code
    for e in REG["leagues"]:
        if e.get("model_type") and e.get("football_data_code"):
            assert e["football_data_format"] == {"standard": "mmz4281", "new_format": "new"}[e["model_type"]]


def test_odds_api_props_only_where_oddsapi_priced_props():
    cov = json.loads((ROOT / "output" / "prop_odds_coverage.json").read_text(encoding="utf-8"))
    priced = {k for k, v in cov.items()
              if not k.startswith("apifootball:") and (v.get("priced_events") or 0) > 0}
    flagged = dict(LR.odds_api_prop_sport_keys())
    assert flagged and all(k in priced or set(LR.odds_api_sport_keys_for(n)) & priced
                           for n, k in flagged.items())
    # probed and never priced -> false
    for n in ("Championship", "League One", "Bundesliga 2", "Europa League", "Conference League"):
        assert LR.entry(n)["odds_api_props"] is False, n
    assert all(isinstance(e["odds_api_props"], bool) for e in REG["leagues"])


def test_national_team_rows_are_classified_and_season_is_unknown_not_guessed():
    for n in ("Friendlies", "World Cup - Qualification Europe", "Africa Cup of Nations"):
        assert LR.competition_type(n) == "national_teams"
        assert not is_known(resolve_api_season(n, "2026-03-31"))


def _mut(fn):
    r = copy.deepcopy(REG)
    fn(r)
    return LR.validate(r)


def _lg(r, name):
    return next(e for e in r["leagues"] if e["name"] == name)


def test_validator_guards_the_new_fields():
    assert _mut(lambda r: None) == []
    assert any("odds_api_props" in m for m in _mut(lambda r: _lg(r, "Serie B").pop("odds_api_props")))
    assert any("odds_api_props" in m for m in _mut(lambda r: _lg(r, "Ligue 2").update(
        odds_api_props=True, odds_api_sport_key=None)))
    assert any("contradicts" in m for m in _mut(lambda r: _lg(r, "Serie B").update(football_data_format="new")))
    assert any("together" in m for m in _mut(lambda r: _lg(r, "Premier League").update(football_data_format=None)))
    # a null season rule is only allowed for a no-model, no-API-id entry
    assert any("season_rule" in m for m in _mut(lambda r: _lg(r, "Serie B").update(season_rule=None)))
    assert any("season_rule" in m for m in _mut(lambda r: _lg(r, "Friendlies").update(api_football_id=10)))
