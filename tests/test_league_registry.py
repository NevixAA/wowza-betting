"""R2-A finding 7: config/leagues.json is the canonical league registry; config.py derives from it.

Equality tests: every config league table equals what it was before the registry existed
(tests/fixtures/config_league_tables_pre_registry_2026_11_01.json, origin/main 4e9c93357), except the
three alias keys that are now ALIASES rather than duplicate keys. Plus structural rules: no duplicate
names, no alias that is a canonical name (cycle), no alias with two targets, no provider id shared by
two canonical leagues (a collapse), exact-match resolution only.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import league_registry as LR  # noqa: E402

SNAP = json.loads((ROOT / "tests" / "fixtures" / "config_league_tables_pre_registry_2026_11_01.json")
                  .read_text(encoding="utf-8"))["tables"]
#: the only intended differences: alias spellings that were separate keys are now aliases
ALIAS_KEYS_REMOVED = {"FOOTBALL_DATA_LEAGUES": {"Austria Bundesliga"}, "LEAGUE_TIMEZONES": {"Austria Bundesliga"},
                      "API_FOOTBALL_SEASONS": {"Danish Superliga"}, "ODDS_API_SPORT_KEYS": {"World Cup"}}
REG = json.loads(LR.REGISTRY_FILE.read_text(encoding="utf-8"))
#: leagues that GAINED a value in LR-1.1 (prop-only OddsAPI keys; single-zone-country timezones)
LR_1_1_ADDS = {
    "ODDS_API_SPORT_KEYS": {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
                            "Champions League", "Europa League", "Conference League"},
    "LEAGUE_TIMEZONES": {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Swiss Super League",
                         "Czech Liga", "Croatia HNL", "Serbia Super Liga", "Ukraine Premier League"},
    # LR-1.2 (2026-11-02): football-data codes for no-model leagues (big-5 from data_loader._HT_EXTRA,
    # SWZ/POL from the deleted Pro backfill dict)
    "FOOTBALL_DATA_LEAGUES": {"Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1",
                              "Swiss Super League", "Poland Ekstraklasa"},
}


def _cfg_table(name):
    v = getattr(config, name)
    if isinstance(v, set):
        return sorted(v)
    auto = set(config.AUTO_LEAGUES)
    return {k: x for k, x in dict(v).items() if k not in auto}


@pytest.mark.parametrize("name", ["STANDARD_FORMAT_LEAGUES", "NEW_FORMAT_LEAGUES", "API_FOOTBALL_ONLY_LEAGUES",
                                  "ENABLED_LEAGUES", "PAPER_LEAGUES"])
def test_league_sets_equal_pre_registry(name):
    auto = set(config.AUTO_LEAGUES)
    assert [x for x in _cfg_table(name) if x not in auto] == SNAP[name]


@pytest.mark.parametrize("name", ["API_FOOTBALL_EXTRA_SEASONS", "FOOTBALL_DATA_LEAGUES", "LEAGUE_TIMEZONES",
                                  "API_FOOTBALL_IDS", "ODDS_API_SPORT_KEYS", "SOFASCORE_TOURNAMENT_IDS",
                                  "PROVIDER_UNSUPPORTED"])
def test_league_maps_equal_pre_registry_minus_aliases(name):
    old = {k: v for k, v in SNAP[name] if k not in ALIAS_KEYS_REMOVED.get(name, set())}
    have = _cfg_table(name)
    # LR-1.1 only ADDS values copied from where the code already held them (see
    # test_league_registry_facts.py for their provenance); every pre-registry value is unchanged.
    added = {k: have[k] for k in set(have) - set(old)}
    assert set(added) == LR_1_1_ADDS.get(name, set())
    assert {k: v for k, v in have.items() if k not in added} == old


def test_api_football_seasons_equal_pre_registry_on_2026_11_01():
    """The resolver reproduces the hand-rolled table on the day it was last verified; it adds the
    leagues that used to fall back to API_SEASON ("2025") or str(now.year)."""
    from src.season_resolver import resolve_api_season
    old = {k: v for k, v in SNAP["API_FOOTBALL_SEASONS"]}
    for k, v in old.items():
        assert resolve_api_season(k, "2026-11-01") == v, k


def test_enabled_iteration_order_preserved_for_predict():
    """predict iterates ODDS_API_SPORT_KEYS in order: the enabled subsequence must be unchanged."""
    old = [k for k, _ in SNAP["ODDS_API_SPORT_KEYS"] if k in config.ENABLED_LEAGUES]
    new = [k for k in config.ODDS_API_SPORT_KEYS if k in config.ENABLED_LEAGUES]
    assert new == old
    old_ids = [k for k, _ in SNAP["API_FOOTBALL_IDS"]]
    assert [k for k in config.API_FOOTBALL_IDS if k in set(old_ids)] == old_ids


def test_aliases_resolve_but_are_not_keys():
    assert config.FOOTBALL_DATA_LEAGUES["Austria Bundesliga"] == "AUT"
    assert config.LEAGUE_TIMEZONES.get("Austria Bundesliga") == "Europe/Vienna"
    assert config.API_FOOTBALL_SEASONS.get("Danish Superliga") == config.API_FOOTBALL_SEASONS["Denmark Superliga"]
    assert config.ODDS_API_SPORT_KEYS.get("World Cup") == "soccer_fifa_world_cup"
    assert "Danish Superliga" in config.API_FOOTBALL_IDS
    for t in (config.FOOTBALL_DATA_LEAGUES, config.LEAGUE_TIMEZONES, config.API_FOOTBALL_SEASONS,
              config.ODDS_API_SPORT_KEYS, config.API_FOOTBALL_IDS):
        assert not set(t) & set(LR.alias_map()), "an alias is iterated as a key"
    assert LR.canonical_name("Austria Bundesliga") == "Austrian Bundesliga"
    assert LR.canonical_name("Danish Superliga") == "Denmark Superliga"


@pytest.mark.parametrize("near", ["austria bundesliga", "Austrian  Bundesliga", "Denmark SuperLiga", "Championshp",
                                  "championship", "La Liga2"])
def test_resolution_is_exact_no_fuzzy(near):
    assert LR.canonical_name(near) is None


def test_league_map_pickles_with_aliases():
    m = pickle.loads(pickle.dumps(config.FOOTBALL_DATA_LEAGUES))
    assert m["Austria Bundesliga"] == "AUT" and dict(m) == dict(config.FOOTBALL_DATA_LEAGUES)
    assert copy.deepcopy(config.LEAGUE_TIMEZONES).get("Austria Bundesliga") == "Europe/Vienna"
    with pytest.raises(KeyError):
        config.FOOTBALL_DATA_LEAGUES["Austria Bundesliga II"]


def test_shipped_registry_is_valid_and_consistent_with_config():
    assert LR.validate(REG) == []
    cons = LR.consistency_with_config(config)
    assert all(v["status"] == "PASS" for v in cons.values()), cons
    assert config.LEAGUE_REGISTRY_VERSION == REG["registry_version"]


def test_every_model_league_has_api_id_and_season_rule():
    for e in REG["leagues"]:
        # LR-1.2: null only for a no-model, no-API-id collection entry (national-team rows)
        assert e["season_rule"] or (e["model_type"] is None and e["api_football_id"] is None
                                    and not e["enabled"]), e["name"]
        if e["model_type"]:
            assert e["api_football_id"], e["name"]
        if e["enabled"] and not e.get("provider_unsupported"):
            assert e["odds_api_sport_key"], e["name"]


def test_player_model_prop_league_ids_match_registry():
    """Ligue 2 sat on id 65 in one copy and 62 in the other for months: ids must agree everywhere."""
    from player_model import config as pcfg
    for name, lid in pcfg.PROP_LEAGUES.items():
        c = LR.canonical_name(name)
        if c is None:
            continue
        assert LR.entry(c)["api_football_id"] == lid, name


def _mut(fn):
    r = copy.deepcopy(REG)
    fn(r)
    return LR.validate(r)


def test_validation_catches_duplicates_cycles_and_collapses():
    lg = lambda r, n: next(e for e in r["leagues"] if e["name"] == n)          # noqa: E731
    assert any("duplicate league name" in m for m in _mut(lambda r: r["leagues"].append(copy.deepcopy(lg(r, "Serie B")))))
    assert any("alias cycle" in m for m in _mut(lambda r: lg(r, "Serie B").setdefault("aliases", []).append("Championship")))
    assert any("points to two leagues" in m for m in _mut(lambda r: lg(r, "Serie B")["aliases"].append("Austria Bundesliga")))
    assert any("api_football_id 40" in m for m in _mut(lambda r: lg(r, "Serie B").update(api_football_id=40)))
    assert any("odds_api_sport_key" in m for m in _mut(lambda r: lg(r, "Serie B").update(odds_api_sport_key="soccer_efl_champ")))
    assert any("enabled league must have a model_type" in m for m in _mut(lambda r: lg(r, "Premier League").update(enabled=True)))
    assert any("season_rule" in m for m in _mut(lambda r: lg(r, "Serie B").update(season_rule={"type": "guess"})))


def test_invalid_registry_refuses_to_load(tmp_path):
    bad = copy.deepcopy(REG)
    bad["leagues"].append(copy.deepcopy(bad["leagues"][0]))
    p = tmp_path / "leagues.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(LR.RegistryError):
        LR.load_registry(p)


def test_health_file(tmp_path):
    h = LR.write_health(routing={"unknown_league_fixtures": [], "status": "PASS"}, path=tmp_path / "h.json")
    d = json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))
    assert d["status"] in ("PASS", "WARN") and d["errors"] == [] and d["counts"]["leagues"] == len(REG["leagues"])
    assert d["aliases"]["Austria Bundesliga"] == "Austrian Bundesliga"
    assert h["config_consistency"]["ENABLED_LEAGUES"]["status"] == "PASS"
