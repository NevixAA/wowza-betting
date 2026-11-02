"""
Current-club rule (invariant 12) - ONE definition for tip time (predict.py) and grading time
(clv_tracker.settle_results, scripts/rebuild_player_history.py).

A prop belongs to the player's CLUB in THAT fixture. Ghost tips measured 2026-11-02 on the
committed player_ledger: 337 graded rows whose player_id played a DIFFERENT fixture that day
(De Paul graded for Atletico on 2026-08-23 while at Inter Miami; Anthony Gordon for Newcastle while
at Barcelona; Casemiro for Manchester United while at Inter Miami). They were graded off the other
club's match because a lookup joined on (player_id, date) without checking the fixture.

Club identity is src.team_names.resolve (normalised equality, alias store, identity tokens;
ambiguity refused).
"""
from __future__ import annotations

import pandas as pd

from . import config


def _tn():
    return config._v9_src("team_names")


def _blank(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or str(x).strip() in ("", "nan", "None")


def same_club(a, b) -> bool | None:
    """True / False when both names are present; None when either is missing (unknown)."""
    if _blank(a) or _blank(b):
        return None
    a, b = str(a), str(b)
    if a == b:
        return True
    TN = _tn()
    # The resolver's tiers (normalised equality, alias store, identity tokens): "Verona" /
    # "Hellas Verona" and "Montreal Impact" / "CF Montreal" are alias-store pairs.
    return TN.resolve(a, [b]) == b or TN.resolve(b, [a]) == a


def club_conflict(tip_club, latest_club, latest_date, fixture_day,
                  days: int | None = None) -> bool:
    """True when the player's latest CLUB appearance (latest_club on latest_date) is for another
    club than `tip_club` and lies within `days` (config.CURRENT_CLUB_CONFLICT_DAYS) of the fixture:
    the evidence contradicts the tip's club, so the tip is refused. Unknown evidence -> False."""
    days = config.CURRENT_CLUB_CONFLICT_DAYS if days is None else days
    if latest_date is None or pd.isna(latest_date):
        return False
    gap = (pd.Timestamp(fixture_day).normalize() - pd.Timestamp(latest_date).normalize()).days
    if gap > days:
        return False
    return same_club(tip_club, latest_club) is False


def same_fixture_sides(tip_home, tip_away, rec_home, rec_away) -> bool:
    """A stats row belongs to the tipped fixture only when BOTH sides are the same clubs (either
    order is refused: home/away are part of the fixture's identity)."""
    return same_club(tip_home, rec_home) is True and same_club(tip_away, rec_away) is True


def side_of(team, home, away) -> bool | None:
    """True if `team` is the HOME club of (home, away), False if the AWAY club, None when the name
    proves neither or both (never guessed). Exact equality first, then club identity."""
    if _blank(team):
        return None
    if str(team) == str(home) and str(team) != str(away):
        return True
    if str(team) == str(away) and str(team) != str(home):
        return False
    if _blank(home) or _blank(away) or str(home) == str(away):
        return None
    # resolve() refuses an ambiguous name (None), so a name matching both sides proves nothing.
    hit = _tn().resolve(str(team), [str(home), str(away)])
    return True if hit == str(home) else False if hit == str(away) else None
