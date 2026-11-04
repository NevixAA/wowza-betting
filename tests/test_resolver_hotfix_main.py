"""Resolver hotfix on main (2026-11-04): known different clubs are refused, proven aliases added.

Measured on main's served pools before the change: the guessing tiers paired 'Platense' with
'River Plate'; 5 live clubs resolved to nothing. data-truth-100 replaces the guessing tiers
entirely (with its identity merge); this keeps main's tiers and only closes the proven gaps.
"""
from __future__ import annotations

import pytest

from src.team_names import resolve, resolve_result

ARG = ["River Plate", "Boca Juniors", "Racing Club", "Lanus", "Velez Sarsfield", "Independiente",
       "Estudiantes L.P.", "Gimnasia L.P.", "Gimnasia Mendoza", "Estudiantes de Rio Cuarto"]


@pytest.mark.parametrize("name,pool", [
    ("Platense", ARG),
    ("Independiente Rivadavia", ARG),
    ("Shenzhen Peng City FC", ["Shenzhen", "Beijing Guoan", "Shanghai Port"]),
    ("Dundee Utd", ["Dundee", "Celtic", "Rangers"]),
    ("Dundee United", ["Dundee", "Celtic", "Rangers"]),
])
def test_known_different_clubs_are_refused(name, pool):
    assert resolve(name, pool) is None
    assert resolve_result(name, pool) is None


def test_the_right_club_still_resolves_when_present():
    assert resolve("Platense", ARG + ["Platense"]) == "Platense"
    assert resolve("Independiente", ARG) == "Independiente"
    assert resolve("River Plate", ARG) == "River Plate"


@pytest.mark.parametrize("name,pool,want", [
    ("Estudiantes", ARG, "Estudiantes L.P."),
    ("Gimnasia La Plata", ARG, "Gimnasia L.P."),
    ("Cercle Brugge KSV", ["Cercle Brugge", "Club Brugge", "Anderlecht"], "Cercle Brugge"),
    ("Beijing FC", ["Beijing Guoan", "Shanghai Port"], "Beijing Guoan"),
    ("Tokyo Verdy", ["Verdy", "Kashima", "Urawa Reds"], "Verdy"),
])
def test_proven_aliases_resolve(name, pool, want):
    assert resolve(name, pool) == want


@pytest.mark.parametrize("name,pool,want", [
    # same clubs main's tiers already resolved correctly: must not regress
    ("TPS Turku", ["TPS", "HJK Helsinki", "KuPS"], "TPS"),
    ("WSG Tirol", ["Tirol", "Sturm Graz", "LASK Linz"], "Tirol"),
    ("Stockport County FC", ["Stockport", "Wigan", "Bolton"], "Stockport"),
    ("FC Energie Cottbus", ["Cottbus", "Hertha", "Hamburg"], "Cottbus"),
    ("Lincoln City", ["Lincoln", "Bolton"], "Lincoln"),
    ("Manchester City", ["Man City", "Man United"], "Man City"),
])
def test_main_tiers_unchanged_for_real_matches(name, pool, want):
    assert resolve(name, pool) == want
