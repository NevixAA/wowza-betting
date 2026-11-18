"""One canonical referee name per referee: evidence-only alias derivation (player_model/referee_names.py)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from player_model import referee_names as RN

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("a,b", [
    ("R. Abatti", "Ramon Abatti Abel, Brazil"),
    ("Ramon Abatti", "Ramon Abatti Abel, Brazil"),
    ("A. Da Silva", "Alexis Da Silva"),
    ("G. Gonzales", "Guido Gonzales Jr"),
    ("J. Sánchez", "Jose Sanchez, Spain"),
])
def test_compatible_spellings(a, b):
    assert RN.compatible(a, b) and RN.compatible(b, a)


@pytest.mark.parametrize("a,b", [
    ("S. Boiko", "Ricardo Montero Araya"),        # a provider RE-ASSIGNMENT on fixture 1490598
    ("R. Abatti", "Rodrigo Jose Pereira de Lima, Brazil"),
    ("Ramon", "Ramon Abatti Abel, Brazil"),       # a first name alone proves nothing
    ("", "Ramon Abatti"),
])
def test_incompatible_spellings(a, b):
    assert not RN.compatible(a, b)


def test_derive_maps_only_evidenced_compatible_pairs_per_league():
    obs = {
        (1, "Brazil Serie A"): {"R. Abatti", "Ramon Abatti Abel, Brazil"},
        (2, "Brazil Serie A"): {"Ramon Abatti", "Ramon Abatti Abel, Brazil"},
        (3, "USA MLS"): {"S. Boiko", "Ricardo Montero Araya"},   # not the same person
        (4, "USA MLS"): {"T. Ford"},                             # one spelling: no evidence
    }
    amap, rep = RN.derive_aliases(obs)
    assert amap == {"Brazil Serie A": {"R. Abatti": "Ramon Abatti Abel, Brazil",
                                       "Ramon Abatti": "Ramon Abatti Abel, Brazil"}}
    assert rep["incompatible_pairs_not_aliased"] == 1 and rep["aliases"] == 2


def test_an_abbreviation_bridging_two_people_is_ambiguous_not_guessed():
    obs = {
        (1, "Brazil Serie A"): {"R. Pereira", "Rodrigo Jose Pereira de Lima, Brazil"},
        (2, "Brazil Serie A"): {"R. Pereira", "Rafael Pereira Costa, Brazil"},
    }
    amap, rep = RN.derive_aliases(obs)
    assert amap == {}
    assert rep["ambiguous_groups"] == [{"league": "Brazil Serie A", "names": [
        "R. Pereira", "Rafael Pereira Costa, Brazil", "Rodrigo Jose Pereira de Lima, Brazil"]}]


def test_canonicalise_is_league_scoped_and_leaves_unknown_names():
    aliases = {"Brazil Serie A": {"R. Abatti": "Ramon Abatti Abel, Brazil"}}
    ref = pd.Series(["R. Abatti", "R. Abatti", "T. Ford", None])
    lg = pd.Series(["Brazil Serie A", "Serie A", "Brazil Serie A", "Brazil Serie A"])
    out = RN.canonicalise(ref, lg, aliases)
    assert out.tolist()[:3] == ["Ramon Abatti Abel, Brazil", "R. Abatti", "T. Ford"]
    assert pd.isna(out.iloc[3])
    rows = [{"referee": "R. Abatti", "league": "Brazil Serie A"}, {"referee": "", "league": "x"}]
    assert RN.canonicalise_rows(rows, aliases) == 1 and rows[0]["referee"] == "Ramon Abatti Abel, Brazil"


def test_shipped_alias_file_is_consistent():
    d = json.loads(RN.ALIASES_PATH.read_text(encoding="utf-8"))
    assert d["version"] == RN.ALIASES_VERSION
    amap = RN.load_aliases()
    assert amap, "referee_aliases.json is empty"
    for lg, m in amap.items():
        canon = set(m.values())
        assert not canon & set(m), f"{lg}: a canonical name is itself an alias (chain)"
        for alias, c in m.items():
            assert RN.compatible(alias, c), (lg, alias, c)
    # the three Brazil fixtures rewritten on main on 2026-11-17/18
    br = amap["Brazil Serie A"]
    assert br["R. Abatti"] == "Ramon Abatti Abel, Brazil"
    assert br["W. Sampaio"] == "Wilton Pereira Sampaio, Brazil"
    assert br["R. Pereira"] == "Rodrigo Jose Pereira de Lima, Brazil"


def test_builder_reads_git_objects_only():
    src = (ROOT / "scripts" / "build_referee_aliases.py").read_text(encoding="utf-8")
    assert "requests" not in src and "_api_get" not in src and "api_football" not in src
