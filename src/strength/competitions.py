"""
Cross-league competitions for the measured club strength (owner decision A, 2026-11-02).

Two leagues share one rating scale only when their clubs meet. The UEFA cups are registry leagues
(competition_type 'club_cup': Champions League, Europa League, Conference League) and are collected
from the registry. The continental cups below are NOT registry leagues yet, so their API-Football
league ids are listed here - provider identifiers (which URL to fetch), not data values. The
collector (scripts/collect_competition_fixtures.py) VERIFIES each id against API-Football /leagues
before using it: the provider must report it as a 'Cup' in country 'World', or it is skipped and
logged. Registry owner: when these become registry entries (competition_type club_cup), delete them
here; registry cups always take precedence.
"""
from __future__ import annotations

CROSS_LEAGUE_CUPS: dict[int, str] = {
    13: "CONMEBOL Libertadores",
    11: "CONMEBOL Sudamericana",
    17: "AFC Champions League",
    16: "CONCACAF Champions League",
    15: "FIFA Club World Cup",          # the only real link between the continents' components
}

# Competition key of a non-registry cup in the strength tables (registry leagues use their
# canonical registry name).
CUP_KEYS: dict[str, int] = {f"af_cup:{i}": i for i in CROSS_LEAGUE_CUPS}


def cup_key(league_id) -> str | None:
    try:
        i = int(league_id)
    except (TypeError, ValueError):
        return None
    return f"af_cup:{i}" if i in CROSS_LEAGUE_CUPS else None
