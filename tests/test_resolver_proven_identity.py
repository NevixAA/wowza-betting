"""resolve() returns a club only when it is PROVABLY the same club (2026-11-02, owner rule).

The stem and token-subset tiers paired different clubs whenever the right one was absent from the
pool. Proof is now: normalised equality, the alias store, or the same identity-token set.

    python -m pytest tests/test_resolver_proven_identity.py -q
"""
import pytest

from src import team_names as TN
from src.team_names import resolve, resolve_result


@pytest.mark.parametrize("name,pool", [
    ("Platense", ["Boca Juniors", "River Plate"]),               # 'plate' is a stem of 'platense'
    ("Estudiantes", ["Estudiantes de Rio Cuarto", "Talleres Cordoba"]),
    ("Estudiantes", ["Estudiantes Rio Cuarto"]),
    ("Beijing FC", ["Beijing Renhe", "Shanghai Port"]),
    ("Dundee United", ["Dundee", "Hearts"]),                     # two clubs in one league
    ("Independiente Rivadavia", ["Independiente", "Racing Club"]),
    ("Shenzhen Peng City FC", ["Shenzhen", "Beijing Guoan"]),
    ("Manchester City", ["Manchester United"]),
    ("GFC Ajaccio", ["AC Ajaccio"]),
    ("Bristol", ["Bristol City"]),
])
def test_a_different_club_is_never_returned_when_the_right_one_is_absent(name, pool):
    assert resolve(name, pool) is None
    assert resolve_result(name, pool) is None


@pytest.mark.parametrize("name,pool,want", [
    ("Estudiantes", ["Estudiantes L.P.", "Estudiantes de Rio Cuarto"], "Estudiantes L.P."),
    ("Beijing FC", ["Beijing Guoan", "Beijing Renhe"], "Beijing Guoan"),
    ("Dundee Utd", ["Dundee", "Dundee United"], "Dundee United"),   # abbreviation spelled out
    ("Independiente Rivadavia", ["Independiente", "Ind. Rivadavia"], "Ind. Rivadavia"),
    ("Platense", ["River Plate", "Platense"], "Platense"),
    ("Cadiz CF", ["Cádiz", "Almeria"], "Cádiz"),                       # club-form word dropped
    ("Real Valladolid CF", ["Valladolid", "Real Madrid"], "Valladolid"),
    ("Kalmar FF", ["Kalmar", "Malmo FF"], "Kalmar"),
    ("Bodø/Glimt", ["Bodo/Glimt", "Molde"], "Bodo/Glimt"),            # transliteration, not deletion
    ("Kasımpaşa", ["Kasimpasa", "Rizespor"], "Kasimpasa"),
    ("Arminia Bielefeld", ["Bielefeld", "Karlsruhe"], "Bielefeld"),   # alias store, with evidence
    ("Karlsruher SC", ["Karlsruhe", "Bielefeld"], "Karlsruhe"),
    ("Man City", ["Manchester City", "Manchester United"], "Manchester City"),
])
def test_proven_spellings_still_resolve(name, pool, want):
    assert resolve(name, pool) == want


def test_alias_store_holds_the_old_api_football_ou_pairs():
    for a, b in [("Vasco da Gama", "Vasco"), ("RB Bragantino", "Bragantino"),
                 ("Red Bull Bragantino", "Bragantino"), ("Atlanta United", "Atlanta Utd"),
                 ("Internazionale", "Inter"), ("Sporting CP", "Sporting")]:
        assert resolve(a, [b, "Some Other Club"]) == b, (a, b)
        assert resolve(b, [a, "Some Other Club"]) == a, (a, b)


def test_two_spellings_of_one_club_in_the_pool_are_refused():
    assert resolve("Rapid Wien", ["SK Rapid", "Rapid Vienna"]) is None
    assert resolve("Rapid Wien", ["SK Rapid"]) == "SK Rapid"
    assert resolve("Rapid Wien", ["Rapid Vienna"]) == "Rapid Vienna"


def test_alias_groups_are_closed_transitively():
    g = TN._alias_groups(id(TN._ALIASES), TN._ALIASES)
    assert {"peterboro", "peterborough", "peterborough united"} <= g["peterborough united"]


def test_the_stem_tiers_are_gone():
    assert not hasattr(TN, "_prefix_eq") and not hasattr(TN, "_pairs_up")
