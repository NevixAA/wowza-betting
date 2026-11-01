"""LR-1.1: per-league facts the code used to type next to league names now live in config/leagues.json.

Provenance tests prove each registry value equals the code table it was copied from (so nothing was
guessed); accessor tests pin the API the other fixers derive from; validation tests prove a malformed
fact refuses to load.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import league_registry as LR  # noqa: E402

REG = json.loads(LR.REGISTRY_FILE.read_text(encoding="utf-8"))


def _literal(rel, name):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


# ── provenance: every value is a copy of what the code already held ─────────────────────────────
# The player_model tables these facts were copied from are now DERIVED from the registry (dt/props,
# 2026-11-01): the provenance check becomes "the code reads the registry, and types no copy".
def test_props_flag_equals_player_model_prop_leagues():
    from player_model import config as PMC
    assert dict(PMC.PROP_LEAGUES) == dict(LR.prop_leagues())
    with pytest.raises((KeyError, ValueError)):         # no typed (literal) copy left in the source
        _literal("player_model/config.py", "PROP_LEAGUES")


def test_prop_sport_keys_equal_odds_fetcher_table():
    from player_model import odds_fetcher as OF
    assert dict(OF.PROP_SPORT_KEYS) == dict(LR.prop_sport_keys())
    assert OF.SPORT_KEY_TO_LEAGUE == LR.sport_key_to_league()
    assert OF.SPORT_KEY_TO_LEAGUE["soccer_uefa_champs_league_qualification"] == "Champions League"
    assert "soccer_uefa_champs_league_qualification" in LR.odds_api_sport_keys_for("Champions League")


def test_added_timezones_come_from_the_country_table():
    country_tz = _literal("scripts/auto_onboard.py", "COUNTRY_TZ")
    for e in REG["leagues"]:
        if e["competition_type"] == "club_league" and e.get("timezone") and e["country"] in country_tz:
            assert e["timezone"] == country_tz[e["country"]], e["name"]


def test_fpl_league_and_fbref_and_same_club_pairs_match_code():
    # dt/fantasy (2026-11-01) derives FANTASY_LEAGUE from the registry; the value copied into the
    # registry was the literal "Premier League" that fantasy.py held before.
    from player_model import fantasy as FAN
    assert LR.fpl_league() == FAN.FANTASY_LEAGUE == "Premier League"
    from player_model import config as PMC
    assert PMC.FBREF_LEAGUES                            # derived from provider_names.fbref
    for name, (fid, slug) in PMC.FBREF_LEAGUES.items():
        assert LR.provider_name(name, "fbref") == {"id": fid, "slug": slug}, name
    # The pairs data_loader._SAME_CLUB_NO_OVERLAP held until dt/features (2026-11-01) moved its
    # reader onto the registry; the hand dict is gone from code.
    moved = {"Mexico Liga MX": [("U.N.A.M. - Pumas", "UNAM Pumas")],
             "La Liga 2": [("Celta B", "Celta de Vigo II")],
             "Championship": [("Sheffield United", "Sheffield Utd")],
             "League One": [("Sheffield United", "Sheffield Utd")]}
    for name, pairs in moved.items():
        assert LR.same_club_spellings(name) == [tuple(p) for p in pairs], name
    with pytest.raises(KeyError):
        _literal("src/data_loader.py", "_SAME_CLUB_NO_OVERLAP")


# ── accessors ───────────────────────────────────────────────────────────────────────────────────
def test_accessors_accept_aliases_and_return_none_when_unknown():
    assert LR.prop_leagues()["World Cup"] == 1                         # alias lookup
    assert LR.competition_type("World Cup") == "national_teams"
    assert LR.competition_type("Champions League") == "club_cup"
    assert LR.competition_type("Austria Bundesliga") == "club_league"
    assert LR.timezone("Austria Bundesliga") == "Europe/Vienna"
    assert LR.timezone("Premier League") == "Europe/London"
    for fn in (LR.timezone, LR.competition_type, LR.odds_api_sport_key):
        assert fn("Not A League") is None and fn(None) is None
    assert LR.odds_api_sport_keys_for("Not A League") == []
    assert LR.same_club_spellings("Serie B") == []
    assert LR.provider_name("USA MLS", "fbref") is None


def test_multi_country_competitions_have_no_timezone_never_utc():
    for n in LR.leagues_of_type("club_cup") + LR.leagues_of_type("national_teams"):
        assert LR.timezone(n) is None, n
    assert LR.timezone("Cyprus First Division") is None              # no repo source holds it


def test_leagues_of_type_partitions_the_registry():
    allnames = [e["name"] for e in REG["leagues"]]
    parts = [LR.leagues_of_type(t) for t in LR.COMPETITION_TYPES]
    assert sorted(sum(parts, [])) == sorted(allnames)
    assert LR.leagues_of_type("national_teams") == ["World Cup 2026"]
    with pytest.raises(ValueError):
        LR.leagues_of_type("MID")


def test_sport_key_to_league_is_one_to_one():
    m = LR.sport_key_to_league()
    assert m["soccer_epl"] == "Premier League" and m["soccer_efl_champ"] == "Championship"
    assert len(m) == sum(len(LR.odds_api_sport_keys_for(e["name"])) for e in REG["leagues"])


def test_every_enabled_league_has_a_real_timezone():
    for e in REG["leagues"]:
        if e["enabled"]:
            assert LR.timezone(e["name"]), e["name"]


# ── validation ──────────────────────────────────────────────────────────────────────────────────
def _mut(fn):
    r = copy.deepcopy(REG)
    fn(r)
    return LR.validate(r)


def _lg(r, n):
    return next(e for e in r["leagues"] if e["name"] == n)


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: _lg(r, "Serie B").update(competition_type="cup"), "competition_type"),
    (lambda r: _lg(r, "Serie B").pop("competition_type"), "competition_type"),
    (lambda r: _lg(r, "Serie B").update(props="yes"), "props must be true or false"),
    (lambda r: _lg(r, "Serie B").pop("fpl"), "fpl must be true or false"),
    (lambda r: _lg(r, "Serie B").update(fpl=True), "FPL scores exactly one league"),
    (lambda r: _lg(r, "Champions League").update(fpl=True), "fpl league must be a club_league"),
    (lambda r: _lg(r, "Serie B").update(timezone="Europe/Atlantis"), "not an IANA zone"),
    (lambda r: _lg(r, "Serie B").update(timezone=""), "non-empty IANA name"),
    (lambda r: _lg(r, "Serie B").update(timezone=None), "enabled club league must have a timezone"),
    (lambda r: _lg(r, "Serie B").update(odds_api_extra_sport_keys=["soccer_efl_champ"]), "used by two leagues"),
    (lambda r: _lg(r, "Serie B").update(odds_api_extra_sport_keys=["x", "x"]), "distinct non-empty"),
    (lambda r: _lg(r, "Serie B").update(same_club_spellings=[["A"]]), "same_club_spellings"),
    (lambda r: _lg(r, "Cyprus First Division").update(props=True, api_football_id=None),
     "props league must have an api_football_id"),
])
def test_validation_rejects_malformed_facts(mutate, needle):
    if "IANA zone" in needle and not LR._iana_zones():
        pytest.skip("no tz database on this interpreter")
    assert any(needle in m for m in _mut(mutate)), needle


def test_shipped_registry_valid():
    assert LR.validate(REG) == []
