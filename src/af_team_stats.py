"""
af_team_stats.py - THE reading of API-Football's per-TEAM match statistics (owner, 2026-11-04).

WHY THIS EXISTS. /fixtures/statistics sends one block per team: a list of {type, value}. For a count
the team did not record it sends `null`, not 0 - a match with no red card reads 'Red Cards': null, a
side that never won a corner 'Corner Kicks': null. Measured on the committed --fullstats sidecar
(output/collect/af_history_fullstats_fill.parquet, 63,805 answered matches, 2026-11-04): home 'Red
Cards' null in 49,384 matches, 'Offsides' 3,393, 'Yellow Cards' 1,896 - while a sent 0 is rare for
the same types (red 8,964). Until this module every parser read such a null as missing, so a quiet
match looked exactly like a match the provider never covered (fix round 7 had turned the old
`int(x or 0)` - which also zero-filled a fixture with NO answer - into "null is always missing").

OWNER DECISION 2026-11-04 (fix round 9, KEY=bank): for a fixture the provider ANSWERED with
statistics, a null team COUNT means 0. No answer means missing. Percentages and xG /
goals_prevented stay missing when null. Read here as:
  * ANSWERED = the team's statistics block carries at least one non-null value. A block of only
    nulls (the 4 'empty_answer' fixtures of the fullstats audit), an empty list or no block for the
    team is NO answer: every value missing.
  * a COUNT type (`COUNT_TYPES`) listed with value null in an answered block -> 0;
  * a type the answered block does not list at all -> missing (the provider gave nothing to read a
    0 from; it lists every type it tracks, with null when there was none);
  * `NOT_COUNTS` (Ball Possession, Passes %, expected_goals, goals_prevented) -> a null stays missing;
  * a value the provider SENT is kept as sent (a sent 0 is a 0); unparseable -> missing, never 0.

'Total passes' / 'Passes accurate' are counts but are NOT zero-filled (`PASS_COUNTS`, owner question
2026-11-04): on the same sidecar they are null in 2,767 answered matches and a sent 0 occurs 3 times
in 61,038 - no side completes a match without a pass, so that null is "not tracked", and reading it
as 0 would invent a value. They stay missing until the owner decides.

'Free Kicks' is a count the provider sends in some answers (the fullstats audit reported it as an
unknown type); it is stored and read under the same rule.
"""
from __future__ import annotations

from typing import Optional

#: provider count types: a null in an ANSWERED team block is 0 (owner 2026-11-04)
COUNT_TYPES = frozenset({
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots", "Shots insidebox",
    "Shots outsidebox", "Fouls", "Corner Kicks", "Offsides", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Free Kicks",
})
#: counts whose null stays MISSING (see the module docstring; open owner question)
PASS_COUNTS = frozenset({"Total passes", "Passes accurate"})
#: not counts: a null stays missing, never 0
NOT_COUNTS = frozenset({"Ball Possession", "Passes %", "expected_goals", "goals_prevented"})


def _is_null(v) -> bool:
    if v is None or (isinstance(v, float) and v != v):
        return True
    return isinstance(v, str) and v.strip().lower() in ("", "none", "null")


def _number(v) -> Optional[float]:
    """A SENT provider value as a float ('55%' -> 55.0); null / unparseable -> None."""
    if _is_null(v) or isinstance(v, bool):
        return None
    t = str(v).strip().rstrip("%").strip()
    try:
        return float(t)
    except ValueError:
        return None


def answered(stats_list) -> bool:
    """True when this team's statistics block carries at least one non-null value."""
    return any(not _is_null((s or {}).get("value")) for s in (stats_list or []))


def team_values(stats_list) -> dict:
    """{provider type: value} for one team's block under the owner's null rule. Not answered -> {}
    (every type missing). Values are floats (counts are whole numbers)."""
    if not answered(stats_list):
        return {}
    out = {}
    for s in stats_list:
        typ = (s or {}).get("type")
        if typ is None:
            continue
        v = s.get("value")
        if _is_null(v):
            out[typ] = 0.0 if typ in COUNT_TYPES else None
        else:
            out[typ] = _number(v)
    return out


def team_value(stats_list, stat_type: str) -> Optional[float]:
    """One statistic of one team's block under the owner's null rule (None = missing)."""
    return team_values(stats_list).get(stat_type)


def team_count(stats_list, stat_type: str) -> Optional[int]:
    """team_value as an int for a count type (None = missing)."""
    v = team_value(stats_list, stat_type)
    return None if v is None else int(v)
