"""Every enabled league needs its OWN API-Football season (2026-10-22; resolver since R2-A 2026-11-01).

A missing entry used to fall back silently to config.API_SEASON ("2025", a finished season). That is
how Denmark Superliga ran on 2025 for months: its entry was keyed "Danish Superliga". Seasons are now
resolved per league from config/leagues.json (src/season_resolver); there is no generic fallback.
"""
import config
from src.season_resolver import resolve_api_season


def test_every_enabled_league_has_an_explicit_season():
    missing = [l for l in sorted(config.ENABLED_LEAGUES) if l not in config.API_FOOTBALL_SEASONS]
    assert not missing, f"no resolved API_FOOTBALL_SEASONS entry: {missing}"


def test_known_calendar_shifts():
    # pinned to the date the hand table was verified, so the test does not age out at a rollover
    assert resolve_api_season("Denmark Superliga", "2026-11-01") == "2026"
    assert resolve_api_season("Danish Superliga", "2026-11-01") == "2026"     # alias, same league
    assert resolve_api_season("Japan J-League", "2026-11-01") == "2027"       # autumn-spring from Aug 2026
