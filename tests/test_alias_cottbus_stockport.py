"""2026-11-19: two provider spellings the strict resolver (2026-11-02, no stem / token-subset tiers)
stopped bridging, proven at fixture level (see the alias store comments in src/team_names.py).

Pro's fixture_key v2 keys one fixture from several sources; with 'Energie Cottbus' (API-Football)
and 'Stockport County FC' (OddsAPI) unresolved, one fixture got two v2 keys and 19 v1 keys became
ambiguous. These pin the two aliases and the refusals around them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_names import resolve  # noqa: E402

BUNDES_2 = ["Cottbus", "Hertha", "Karlsruhe", "Darmstadt", "Paderborn"]
LEAGUE_ONE = ["Bolton", "Stockport", "Wigan", "Wycombe", "Reading"]


def test_energie_cottbus_resolves_to_cottbus():
    assert resolve("Energie Cottbus", BUNDES_2) == "Cottbus"
    assert resolve("FC Energie Cottbus", BUNDES_2) == "Cottbus"


def test_stockport_county_fc_resolves_to_stockport():
    assert resolve("Stockport County FC", LEAGUE_ONE) == "Stockport"
    assert resolve("Stockport County", LEAGUE_ONE) == "Stockport"


def test_absent_club_is_never_paired_with_another():
    assert resolve("Energie Cottbus", ["Hertha", "Karlsruhe"]) is None
    assert resolve("Stockport County FC", ["Bolton", "Wigan", "Notts County"]) is None


def test_a_pool_holding_two_spellings_of_one_club_is_refused_by_the_alias_tier():
    # a duplicate identity in the pool: the alias tier refuses to pick half; exact equality and the
    # identical-token-set tier (unchanged by this entry) still decide on their own proof
    assert resolve("Energie Cottbus", ["Cottbus", "Energie Cottbus"]) == "Energie Cottbus"
    assert resolve("Stockport County FC", ["Stockport", "Stockport County"]) == "Stockport County"
    assert resolve("Stockport County FC", ["Stockport", "Stockport Town"]) == "Stockport"
