"""Every enabled league needs its OWN API-Football season entry (2026-10-22).

A missing entry does not fail - it silently falls back to API_SEASON (a finished season). That is how
Denmark Superliga ran on 2025 for months: its entry was keyed "Danish Superliga".
"""
import config


def test_every_enabled_league_has_an_explicit_season():
    missing = [l for l in sorted(config.ENABLED_LEAGUES) if l not in config.API_FOOTBALL_SEASONS]
    assert not missing, f"no API_FOOTBALL_SEASONS entry (would fall back to {config.API_SEASON}): {missing}"


def test_known_calendar_shifts():
    assert config.API_FOOTBALL_SEASONS["Denmark Superliga"] == "2026"
    assert config.API_FOOTBALL_SEASONS["Japan J-League"] == "2027"     # autumn-spring calendar from Aug 2026
