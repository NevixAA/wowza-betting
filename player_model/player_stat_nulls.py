"""
player_stat_nulls.py - THE reading of API-Football's per-player match counts (owner, 2026-11-04).

WHY THIS EXISTS. API-Football /fixtures/players sends `null`, not 0, for a count a player did not
record in a match he PLAYED (no goal -> goals.total null, no shot -> shots.on null). Two readings were
wrong in turn:
  * until 2026-11-04 the parsers wrote `int(x or 0)`: every null became 0, including the null of a
    fixture the provider never covered, so "the provider has no player statistics for this match"
    looked exactly like "he had no shot" (player_history: 876,130 rows, 0 blanks in any count);
  * fix round 7 (2026-11-04) made every null missing: right for an uncovered fixture, wrong for the
    ordinary "he played and did not score" - a real 0 became a hole in every target.
Owner decision 2026-11-04 (fix round 8, props): read a null count as 0 ONLY when BOTH hold
  1. the player PLAYED that match: games.minutes > 0, and
  2. the provider ANSWERED that statistic for the fixture: at least one player of the same
     /fixtures/players answer (either team) carries a non-null value for it.
Otherwise it stays missing: a player with no / 0 minutes, a fixture with no player statistics at
all, or a statistic nobody in the fixture has (e.g. no penalty was taken, so every penalty field is
null for everyone - the provider gave nothing to read a 0 from). A value the provider SENT is kept
as sent (a sent 0 is a 0); an unparseable value is missing, never 0.

THE FIELDS (`COUNT_FIELDS`, player_history column -> provider path). All are per-player counts the
rule above applies to. Provider behaviour, as documented by API-Football and stated by the owner
2026-11-04; the re-fetch audit (scripts/collect_player_refetch.py, `fields`) MEASURES it per field on
the real answers (null-and-played / sent 0 / sent value), so this table is checked, not assumed:
  goals.total, goals.assists, shots.total, shots.on, passes.key, tackles.total, tackles.interceptions,
  tackles.blocks, dribbles.attempts, dribbles.success, dribbles.past, fouls.drawn, fouls.committed,
  duels.total, duels.won, offsides, penalty.won, penalty.commited (sic, the provider's spelling),
  penalty.scored, penalty.missed, penalty.saved, goals.saves .......... null when the player had none
  goals.conceded, cards.yellow, cards.red, passes.total .............. usually SENT as 0; a null is
                                                                         read by the same rule
NOT COUNTS (`NOT_COUNTS`) - never zero-filled, a null stays missing:
  games.rating      the provider's match rating; null = not rated (0 is not a rating)
  passes.accuracy   accurate passes (a % in the season endpoint); a ratio-like value, not "none"
  games.minutes     the gate itself (null / 0 = did not play -> nothing is read as 0)
  games.number, games.position, games.captain, games.substitute - identity / flags
"""
from __future__ import annotations

from typing import Iterable, Optional

#: player_history column -> (provider group, key) in /fixtures/players statistics[0]
COUNT_FIELDS: dict[str, tuple[str, str]] = {
    "goals": ("goals", "total"),
    "assists": ("goals", "assists"),
    "shots_total": ("shots", "total"),
    "shots_on_target": ("shots", "on"),
    "yellow_cards": ("cards", "yellow"),
    "red_cards": ("cards", "red"),
    "key_passes": ("passes", "key"),
    "passes_total": ("passes", "total"),
    "fouls_committed": ("fouls", "committed"),
    "fouls_drawn": ("fouls", "drawn"),
    "duels_total": ("duels", "total"),
    "duels_won": ("duels", "won"),
    "offsides": ("offsides", ""),                 # a top-level number, not a group
    "tackles_total": ("tackles", "total"),
    "tackles_blocks": ("tackles", "blocks"),
    "interceptions": ("tackles", "interceptions"),
    "dribbles_attempted": ("dribbles", "attempts"),
    "dribbles_success": ("dribbles", "success"),
    "dribbles_past": ("dribbles", "past"),
    "goals_conceded": ("goals", "conceded"),
    "saves": ("goals", "saves"),
    "penalty_scored": ("penalty", "scored"),
    "penalty_missed": ("penalty", "missed"),
    "penalty_won": ("penalty", "won"),
    "penalty_committed": ("penalty", "commited"),  # the provider's spelling
    "penalty_saved": ("penalty", "saved"),
}

#: provider values that are NOT counts: a null stays missing, never 0
NOT_COUNTS: dict[str, tuple[str, str]] = {
    "rating": ("games", "rating"),
    "passes_accuracy": ("passes", "accuracy"),
}

#: the columns player_history stores (tackles_blocks / penalty_committed are re-fetch extras)
HISTORY_COUNT_COLS = tuple(c for c in COUNT_FIELDS if c not in ("tackles_blocks", "penalty_committed"))


def as_count(v) -> Optional[int]:
    """A count the provider SENT, or None (null / "" / unparseable). Never 0 for a missing value."""
    if v is None or isinstance(v, bool) or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _stats0(player_entry: dict) -> dict:
    sl = (player_entry or {}).get("statistics") or [{}]
    return (sl[0] if sl else {}) or {}


def raw_value(stats: dict, col: str):
    """The provider's raw value of `col` in one player's statistics dict (None when absent)."""
    group, key = COUNT_FIELDS.get(col) or NOT_COUNTS[col]
    g = (stats or {}).get(group)
    if not key:
        return g if not isinstance(g, dict) else None
    return g.get(key) if isinstance(g, dict) else None


def minutes_of(stats: dict) -> Optional[int]:
    return as_count(((stats or {}).get("games") or {}).get("minutes"))


def fixture_answered(response: Iterable[dict] | None) -> frozenset:
    """The COUNT_FIELDS columns the provider answered for this fixture: at least one player of the
    /fixtures/players answer (either team) carries a parseable value for it. An empty answer (no
    player statistics at all) answers nothing."""
    got = set()
    for block in response or []:
        for pe in (block or {}).get("players") or []:
            st = _stats0(pe)
            for col in COUNT_FIELDS:
                if col not in got and as_count(raw_value(st, col)) is not None:
                    got.add(col)
        if len(got) == len(COUNT_FIELDS):
            break
    return frozenset(got)


def read_count(raw, minutes: Optional[int], answered: bool) -> Optional[int]:
    """THE rule for one value: sent -> as sent; null AND played (minutes > 0) AND answered for the
    fixture -> 0; anything else -> None. A non-null value that does not parse is missing (it is not
    the provider's 'none')."""
    v = as_count(raw)
    if v is not None:
        return v
    if raw is not None and not (isinstance(raw, float) and raw != raw):
        return None                                   # something was sent but it is not a count
    if minutes is not None and minutes > 0 and answered:
        return 0
    return None


def player_counts(stats: dict, answered: frozenset | set | None) -> dict:
    """Every COUNT_FIELDS column of one player's statistics dict under the rule. `answered` is
    fixture_answered() of the SAME fixture's answer; None means the fixture-level proof is not
    available to the caller, so every null stays missing (fix round 7 behaviour)."""
    mins = minutes_of(stats)
    ans = answered or frozenset()
    return {col: read_count(raw_value(stats, col), mins, col in ans) for col in COUNT_FIELDS}


def rating_of(stats: dict) -> Optional[float]:
    """games.rating as a float; null / "" / unparseable -> None (NOT a count: never 0)."""
    r = ((stats or {}).get("games") or {}).get("rating")
    if r is None or (isinstance(r, str) and not r.strip()):
        return None
    try:
        return float(r)
    except (TypeError, ValueError):
        return None
