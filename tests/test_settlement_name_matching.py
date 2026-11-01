"""Settlement matches the SAME club and never a different one (Data Truth audit 17, 18).

    python -m pytest tests/test_settlement_name_matching.py -q
"""
from __future__ import annotations

import numpy as np
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


# ── pool-scoped settlement matching (Data Truth 2026-11-01) ───────────────────────────────────
def _ev(h, a, d="2026-10-24", hs=1, as_=0):
    return {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_, "date_str": d}


def test_match_event_refuses_an_ambiguous_pool():
    # Pairwise, 'Bristol' matches 'Bristol City' (one-name pool); in the date's pool it is ambiguous.
    assert U._names_match("Bristol", "Bristol City")
    evs = [_ev("Bristol City", "Roma"), _ev("Bristol Rovers", "Roma")]
    assert U._match_event("Bristol", "Roma", evs, {"2026-10-24"}) is None


def test_match_event_requires_the_date():
    assert U._match_event("Inter", "Roma", [_ev("Inter", "Roma", d="2026-10-23")], {"2026-10-24"}) is None


def test_match_event_same_fixture_from_two_reports_is_one():
    evs = [_ev("Inter", "Roma"), _ev("Inter", "Roma")]
    assert U._match_event("Inter", "Roma", evs, {"2026-10-24"})["home_score"] == 1
    evs[1] = _ev("Inter", "Roma", hs=3)                      # same fixture, two different scores
    assert U._match_event("Inter", "Roma", evs, {"2026-10-24"}) is None


def test_fd_sources_come_from_the_registry():
    from src import league_registry as R
    want = {e["name"]: e["football_data_code"] for e in R.leagues()
            if e.get("football_data_code") and e.get("model_type")}
    assert {k: v[1] for k, v in U._FD_SOURCES.items()} == want
    assert U._FD_SOURCES["Championship"][0] == "std" and U._FD_SOURCES["USA MLS"][0] == "new"


def test_closing_price_is_league_scoped(monkeypatch):
    import pandas as pd
    arch = pd.DataFrame([
        {"snapshot_ts": "2026-10-24T10:00:00", "match_date": "2026-10-24", "league": "League Two",
         "match": "Barrow vs Gillingham", "market": "over25", "odds": 2.1},
        {"snapshot_ts": "2026-10-24T11:00:00", "match_date": "2026-10-24", "league": "League One",
         "match": "Barrow vs Gillingham", "market": "over25", "odds": 9.9}])
    arch["_h"] = arch["match"].str.split(" vs ").str[0]
    arch["_a"] = arch["match"].str.split(" vs ").str[1]
    arch["_dk"] = arch["match_date"]
    arch["_lk"] = arch["league"]
    monkeypatch.setattr(U, "_CSV_ARCHIVES", [arch])
    assert U._closing_for_market("Barrow AFC", "Gillingham", "2026-10-24", "over25",
                                 league="League Two") == 2.1
    # No league -> no close (it used to search every league on the date).
    assert np.isnan(U._closing_for_market("Barrow AFC", "Gillingham", "2026-10-24", "over25"))


def test_resolve_ht_unknown_half_is_not_zero():
    assert U._resolve_ht("ht_under05", None, 0) == ""
    assert U._resolve_ht("ht_under05", 0, 0) == "WIN"
