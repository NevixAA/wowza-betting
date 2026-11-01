"""The current season's finished-fixture list is refreshed; finished seasons stay cached (finding 35).

    python -m pytest tests/test_af_results_cache_ttl.py -q
"""
from __future__ import annotations

from src import api_football_ou as AF
from src.season_resolver import resolve_api_season


def _ttl(monkeypatch, season, league):
    seen = {}
    monkeypatch.setattr(AF, "_get", lambda path, params, ttl_h=0: seen.update(ttl=ttl_h) or None)
    AF.fetch_league_fixture_results(40, season, league)
    return seen["ttl"]


def test_current_season_is_short_lived(monkeypatch):
    assert _ttl(monkeypatch, resolve_api_season("Championship"), "Championship") == 6


def test_finished_season_stays_cached(monkeypatch):
    prev = str(int(resolve_api_season("Championship")) - 1)
    assert _ttl(monkeypatch, prev, "Championship") == 8760


def test_unknown_league_is_short_lived(monkeypatch):
    assert _ttl(monkeypatch, "2024", "") == 6
