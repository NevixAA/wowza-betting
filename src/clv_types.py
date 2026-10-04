"""§7 — three different things called "CLV", kept apart.

    from src.clv_types import same_book_clv, sharp_reference_clv, consensus_clv

ONE NUMBER CANNOT ANSWER THREE QUESTIONS. The estate currently reports a single `clv`, and
depending on what it was computed against it has meant any of:

    "did I get a good price at the book I actually used?"        execution quality
    "did the sharp market move my way?"                          informational quality
    "was my price good against the market as a whole?"           consensus value

Those have different failure modes and different fixes. Mixing them into one aggregate means a
genuine execution gain can be cancelled by an informational loss and the total reads as nothing —
which is not hypothetical here: best-price execution is worth +1.99pp CI [+1.37, +2.67] on this
estate, a real and separable gain that no blended CLV figure would show.

WHAT "NO CLOSE" MEANS, AND WHY IT IS NEVER FILLED IN. A missing close is not a zero and is not
an earlier price. Substituting a T-6h quote for a missing T-30m close is the single most
flattering error available — it reports the price as if it had been validated near kickoff when
it was not — so every path here returns a status instead of a number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.book_quotes import CLOSING_WINDOW_MIN
from src.market_anchor import MIN_BOOKS_FOR_MEDIAN, power_devig

log = logging.getLogger(__name__)

SAME_BOOK = "same_book_clv"
SHARP_REFERENCE = "sharp_reference_clv"
CONSENSUS = "consensus_clv"

# ── statuses. Each names a DIFFERENT missing thing, because each needs a different fix. ────────
OK = "OK"
NO_CLOSE = "NO_CLOSE"                       # nothing at all inside the closing window
NO_SAME_BOOK_CLOSE = "NO_SAME_BOOK_CLOSE"   # the market closed, but not at the book we used
NO_SHARP_CLOSE = "NO_SHARP_CLOSE"           # no designated sharp source near kickoff
INSUFFICIENT_BOOKS = "INSUFFICIENT_BOOKS"   # too few two-sided books to form a consensus
POST_KICKOFF_ONLY = "POST_KICKOFF_ONLY"     # only in-play quotes — never a close
ONE_SIDED = "ONE_SIDED"                     # cannot de-vig, so no fair probability exists

#: Books designated sharp, best first.
#:
#: This is NOT a quality ranking and NOT a statement about which book pays best. Pinnacle is the
#: reference because it runs a low margin and accepts sharp action instead of limiting it, so its
#: close carries information; Betfair is an exchange, so its price is the closest thing here to a
#: traded consensus. Bet365 — the book this estate actually captures most — is deliberately NOT
#: on this list: it is where we bet, not a source of truth about what the price should be.
SHARP_NAMES = ("Pinnacle", "Betfair")

#: Opposite side, per market. A market without a defined opposite cannot be de-vigged at all.
_OPPOSITE = {"over": "under", "under": "over", "yes": "no", "no": "yes"}


@dataclass(frozen=True)
class CLV:
    value: float | None          # probability points; positive = we beat the close
    status: str
    kind: str
    detail: str = ""
    n_books: int = 0
    close_minutes_to_kickoff: float | None = None

    @property
    def usable(self) -> bool:
        """Check this before using `value`. A status is not a number."""
        return self.value is not None and self.status == OK


def _window(df: pd.DataFrame, fixture_id, market: str, side: str,
            line: float | None) -> pd.DataFrame:
    m = df[(df["fixture_id"] == fixture_id) & (df["market"] == market) & (df["side"] == side)]
    if line is not None and "line" in m.columns:
        m = m[m["line"].fillna(-1) == line]
    return m


def _closing_rows(m: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rows inside the closing window, or a status saying precisely why there are none."""
    if m.empty:
        return m, NO_CLOSE
    mins = pd.to_numeric(m["minutes_to_kickoff"], errors="coerce")
    pre = m[mins >= 0]
    if pre.empty:
        # Quotes exist but all are in-play. Distinct from "nothing was captured": it means the
        # capture ran too late, which is a scheduling fix, not a coverage fix.
        return pre, POST_KICKOFF_ONLY
    inside = pre[pd.to_numeric(pre["minutes_to_kickoff"], errors="coerce") <= CLOSING_WINDOW_MIN]
    return (inside, OK) if len(inside) else (inside, NO_CLOSE)


def _nearest(frame: pd.DataFrame) -> pd.Series:
    """The row closest to kickoff. Smallest minutes_to_kickoff wins."""
    return frame.sort_values("minutes_to_kickoff").iloc[0]


def same_book_clv(df: pd.DataFrame, fixture_id, market: str, side: str,
                  entry_odds: float, bookmaker_id, line: float | None = None) -> CLV:
    """Entry at book A against book A's OWN close. EXECUTION QUALITY.

    Deliberately NOT de-vigged. This asks whether we took a good price at the book we used, and
    that book's margin is common to both ends of the comparison. De-vigging would silently turn
    it into a different question.

    A missing close here is `NO_SAME_BOOK_CLOSE`, not `NO_CLOSE`: the market may well have closed
    elsewhere, and the fix is capturing that book, not capturing more books.
    """
    m = _window(df, fixture_id, market, side, line)
    m = m[m["bookmaker_id"] == bookmaker_id]
    inside, status = _closing_rows(m)
    if status != OK:
        return CLV(None, NO_SAME_BOOK_CLOSE if status == NO_CLOSE else status, SAME_BOOK,
                   f"book {bookmaker_id} has no close inside T-{CLOSING_WINDOW_MIN}m")
    r = _nearest(inside)
    close = float(r["odds"])
    if not (entry_odds > 1 and close > 1):
        return CLV(None, NO_CLOSE, SAME_BOOK, "unusable odds")
    # Expressed in probability points so all three kinds are on one scale.
    return CLV(round(1.0 / close - 1.0 / entry_odds, 6), OK, SAME_BOOK,
               f"book {bookmaker_id}: {entry_odds} -> {close}", 1,
               float(r["minutes_to_kickoff"]))


def sharp_reference_clv(df: pd.DataFrame, fixture_id, market: str, side: str,
                        entry_prob: float, line: float | None = None,
                        sharp_names: tuple[str, ...] = SHARP_NAMES) -> CLV:
    """Entry probability against a DESIGNATED sharp close, two-sided and de-vigged.

    INFORMATIONAL QUALITY: did the market hardest to beat move our way? Tries each sharp source
    in order and uses the first that closed on BOTH sides — one side of a Pinnacle market cannot
    be separated from its margin, so a one-sided sharp close is refused rather than halved.
    """
    other = _OPPOSITE.get(side)
    if other is None:
        return CLV(None, ONE_SIDED, SHARP_REFERENCE, f"no opposite side defined for '{side}'")
    for name in sharp_names:
        def at(s):
            f = _window(df, fixture_id, market, s, line)
            return f[f["bookmaker"].astype(str).str.contains(name, case=False, na=False)]
        inside_a, st_a = _closing_rows(at(side))
        if st_a != OK:
            continue                      # this source has no close; try the next
        inside_b, st_b = _closing_rows(at(other))
        if st_b != OK:
            return CLV(None, ONE_SIDED, SHARP_REFERENCE,
                       f"{name} closed one-sided — margin cannot be separated from opinion")
        ra, rb = _nearest(inside_a), _nearest(inside_b)
        fair = power_devig([float(ra["odds"]), float(rb["odds"])])
        if not fair:
            return CLV(None, ONE_SIDED, SHARP_REFERENCE, f"{name}: de-vig failed")
        return CLV(round(fair[0] - entry_prob, 6), OK, SHARP_REFERENCE,
                   f"{name} two-sided close, de-vigged", 1, float(ra["minutes_to_kickoff"]))
    return CLV(None, NO_SHARP_CLOSE, SHARP_REFERENCE,
               f"none of {list(sharp_names)} closed inside T-{CLOSING_WINDOW_MIN}m")


def consensus_clv(df: pd.DataFrame, fixture_id, market: str, side: str,
                  entry_prob: float, line: float | None = None) -> CLV:
    """Entry probability against the cross-book MEDIAN de-vigged close. CONSENSUS VALUE.

    A book contributes only when it closed on BOTH sides inside the window. Ten books carrying
    only the Over are still impossible to de-vig, so they return INSUFFICIENT_BOOKS rather than
    a confident-looking number built on the margin.
    """
    other = _OPPOSITE.get(side)
    if other is None:
        return CLV(None, ONE_SIDED, CONSENSUS, f"no opposite side defined for '{side}'")
    ia, sa = _closing_rows(_window(df, fixture_id, market, side, line))
    ib, _sb = _closing_rows(_window(df, fixture_id, market, other, line))
    if sa != OK:
        return CLV(None, sa, CONSENSUS, "nothing closed inside the window")

    def last_per_book(frame: pd.DataFrame) -> dict:
        """Each book's quote CLOSEST TO KICKOFF, not its first or its latest row."""
        if frame.empty:
            return {}
        f = frame.sort_values("minutes_to_kickoff")
        return {k: float(v) for k, v in f.groupby("bookmaker_id")["odds"].first().items()}

    oa, ob = last_per_book(ia), last_per_book(ib)
    both = sorted(set(oa) & set(ob))
    if len(both) < MIN_BOOKS_FOR_MEDIAN:
        return CLV(None, INSUFFICIENT_BOOKS, CONSENSUS,
                   f"{len(both)} two-sided book(s), need >= {MIN_BOOKS_FOR_MEDIAN}", len(both))
    ps = [f[0] for f in (power_devig([oa[k], ob[k]]) for k in both) if f]
    if len(ps) < MIN_BOOKS_FOR_MEDIAN:
        return CLV(None, INSUFFICIENT_BOOKS, CONSENSUS,
                   f"de-vig succeeded on only {len(ps)} book(s)", len(ps))
    return CLV(round(float(np.median(ps)) - entry_prob, 6), OK, CONSENSUS,
               f"median of {len(ps)} two-sided books", len(ps),
               float(pd.to_numeric(ia["minutes_to_kickoff"], errors="coerce").min()))
