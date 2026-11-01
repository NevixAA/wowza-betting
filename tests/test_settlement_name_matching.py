"""Settlement matches the SAME club and never a different one (Data Truth audit 17, 18).

    python -m pytest tests/test_settlement_name_matching.py -q
"""
from __future__ import annotations

import pytest

import update_results as U


@pytest.mark.parametrize("a,b", [
    ("LA Galaxy", "Los Angeles Galaxy"), ("D.C. United", "DC United"),
    ("Bragantino-SP", "RB Bragantino"), ("St Patricks Athletic", "St. Patricks"),
    ("St Patricks Athletic", "St Patrick's Athl."), ("Hiroshima Sanfrecce FC", "Sanfrecce Hiroshima"),
    ("Bodø/Glimt", "Bodo/Glimt"), ("HamKam", "Ham-Kam"), ("TPS Turku", "Turku PS"),
    ("Fredrikstad FK", "Fredrikstad"), ("Man City", "Manchester City"),
    ("Arminia Bielefeld", "Bielefeld"), ("Cadiz CF", "Cádiz"),
])
def test_same_club_matches(a, b):
    assert U._names_match(a, b) and U._names_match(b, a)


@pytest.mark.parametrize("a,b", [
    ("Real Madrid", "Real Sociedad"), ("Manchester City", "Manchester United"),
    ("Celta Vigo", "Celta de Vigo II"), ("Sporting", "Sporting Gijon"),
    ("Bristol City", "Bristol Rovers"), ("Sheffield United", "Sheffield Wednesday"),
])
def test_different_clubs_never_match(a, b):
    assert not U._names_match(a, b) and not U._names_match(b, a)
