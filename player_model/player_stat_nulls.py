"""
player_stat_nulls.py - THE reading of API-Football's per-player match counts (owner, 2026-11-04).

WHY THIS EXISTS. API-Football /fixtures/players sends `null`, not 0, for a count a player did not
record in a match he PLAYED (no goal -> goals.total null, no shot -> shots.on null). Three readings
were wrong in turn:
  * until 2026-11-04 the parsers wrote `int(x or 0)`: every null became 0, including the null of a
    fixture the provider never covered, so "the provider has no player statistics for this match"
    looked exactly like "he had no shot" (player_history: 876,130 rows, 0 blanks in any count);
  * fix round 7 (2026-11-04) made every null missing: right for an uncovered fixture, wrong for the
    ordinary "he played and did not score" - a real 0 became a hole in every target.
  * fix round 8 (2026-11-04) read a null as 0 only when the player played AND some other player of
    the same answer carried a value for that statistic. That left "no penalty was taken" (every
    penalty field null for everyone) missing for players who plainly took no penalty.
THE RULE (owner decision 2026-11-04, refined in fix round 9 'players'). A null per-player COUNT is 0
when BOTH hold
  1. the player PLAYED that match: games.minutes > 0, and
  2. the provider RETURNED PLAYER STATISTICS for the fixture: the /fixtures/players answer holds at
     least one player entry with a statistics block (`fixture_answered`),
EVEN IF no player in the match has a value for that statistic (no penalty taken -> penalty_* = 0 for
everyone who played). Otherwise it stays missing: a fixture the provider did not answer (empty
answer / no statistics), a player with no or 0 minutes. A value the provider SENT is kept as sent (a
sent 0 is a 0); an unparseable value is missing, never 0. Ratings and percentages (NOT_COUNTS) stay
missing when null.

PROVENANCE (owner 2026-11-04, fix round 9). Every row a parser builds from a raw provider answer with
this rule carries `count_source` = PARSED_WITH_RULE. Such a row is verified at parse time: it is
never 'zeros_unverified' and never needs a /fixtures/players re-fetch. Rows written before the fix
(the `or 0` parser) carry no count_source; player_model.history_refetch marks them
'zeros_unverified' until scripts/collect_player_refetch.py reaches them ('refetched').

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
NULL IS MISSING (`NULL_IS_MISSING`, fix/props-feature-bugs 2026-11-11) - counts whose null is NOT the
provider's "none", so a null stays missing even when the player played:
  goals.conceded    MEASURED on the re-fetched answers (scripts/collect_player_refetch.py) against the
                    fixture score: a goalkeeper's null goals.conceded had a scoring opponent in 75% of
                    1,489 full-match keeper rows, and the provider's null share rose from 0% (to
                    2026-02) to 47% (2026-10) and 90% (2026-11). Read as 0 by the rule above, it put
                    0 beside a real score on 1,048 of the 1,086 contradicting keeper rows since
                    2026-09, inflating gk_save_rate -> the served opp_gk_save_rate toward 1.0. The
                    readers (history_refetch.with_refetched) also blank the stored re-fetched rows
                    whose filled_cols show this null-read-as-0; the stored value stays beside.
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

#: counts whose null is not "none" (measured, see the module docstring): a null stays missing even
#: for a player who played
NULL_IS_MISSING = frozenset({"goals_conceded"})

#: provider values that are NOT counts: a null stays missing, never 0
NOT_COUNTS: dict[str, tuple[str, str]] = {
    "rating": ("games", "rating"),
    "passes_accuracy": ("passes", "accuracy"),
}

#: the columns player_history stores (tackles_blocks / penalty_committed are re-fetch extras)
HISTORY_COUNT_COLS = tuple(c for c in COUNT_FIELDS if c not in ("tackles_blocks", "penalty_committed"))

#: `count_source` - where a row's per-player counts came from (owner 2026-11-04, fix round 9)
SOURCE_COL = "count_source"
PARSED_WITH_RULE = "provider_rule"        # parsed from the raw answer with this rule: verified
PARSED_NO_PROOF = "provider_no_fixture_proof"   # parser had no fixture-level proof: nulls missing
REFETCHED = "refetched"                   # re-read by scripts/collect_player_refetch.py
UNVERIFIED = "zeros_unverified"           # written by the `or 0` parser, not re-read yet
#: count_source values whose counts need no re-fetch
VERIFIED_SOURCES = frozenset({PARSED_WITH_RULE, REFETCHED})


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


def fixture_answered(response: Iterable[dict] | None) -> bool:
    """Did the provider RETURN PLAYER STATISTICS for this fixture? True when the /fixtures/players
    answer holds at least one player entry with a non-empty statistics block. An empty answer, or
    one whose entries carry no statistics, is 'not answered': every null of it stays missing.
    (Fix round 8 returned the set of statistics some player had a value for; the owner's refined
    rule of 2026-11-04 asks only whether the fixture was answered.)"""
    for block in response or []:
        for pe in (block or {}).get("players") or []:
            if _stats0(pe):
                return True
    return False


def parsed_source(answered: Optional[bool]) -> str:
    """count_source of a row parsed from a raw answer: PARSED_WITH_RULE when the parser held the
    fixture-level proof, PARSED_NO_PROOF when it did not (its nulls stayed missing)."""
    return PARSED_WITH_RULE if answered is not None else PARSED_NO_PROOF


def read_count(raw, minutes: Optional[int], answered: bool) -> Optional[int]:
    """THE rule for one value: sent -> as sent; null AND played (minutes > 0) AND the provider
    returned player statistics for the fixture -> 0; anything else -> None. A non-null value that
    does not parse is missing (it is not the provider's 'none')."""
    v = as_count(raw)
    if v is not None:
        return v
    if raw is not None and not (isinstance(raw, float) and raw != raw):
        return None                                   # something was sent but it is not a count
    if minutes is not None and minutes > 0 and answered:
        return 0
    return None


def player_counts(stats: dict, answered: Optional[bool]) -> dict:
    """Every COUNT_FIELDS column of one player's statistics dict under the rule. `answered` is
    fixture_answered() of the SAME fixture's answer; None means the fixture-level proof is not
    available to the caller, so every null stays missing (fix round 7 behaviour)."""
    if answered is not None and not isinstance(answered, bool):
        # fix round 8 passed a per-statistic set; a caller still doing so must be fixed, not guessed
        raise TypeError("player_counts: `answered` is fixture_answered()'s bool (owner rule 2026-11-04)")
    mins = minutes_of(stats)
    return {col: (as_count(raw_value(stats, col)) if col in NULL_IS_MISSING
                  else read_count(raw_value(stats, col), mins, bool(answered)))
            for col in COUNT_FIELDS}


def rating_of(stats: dict) -> Optional[float]:
    """games.rating as a float; null / "" / unparseable -> None (NOT a count: never 0)."""
    r = ((stats or {}).get("games") or {}).get("rating")
    if r is None or (isinstance(r, str) and not r.strip()):
        return None
    try:
        return float(r)
    except (TypeError, ValueError):
        return None
