"""§7 — the three CLV kinds must stay distinguishable, and a missing close must stay missing."""
from __future__ import annotations

import pandas as pd
import pytest

from src import clv_types as ct


def q(book_id, book, side, odds, mins, fid=1, market="ou", line=2.5):
    return {"fixture_id": fid, "bookmaker_id": book_id, "bookmaker": book, "market": market,
            "side": side, "line": line, "odds": odds, "minutes_to_kickoff": mins,
            "ladder_band": "T-30m", "snapshot_ts": "2026-10-04T17:30:00Z"}


def frame(rows):
    return pd.DataFrame(rows)


# ── same_book: execution quality ──────────────────────────────────────────────────────────────
def test_same_book_uses_the_same_book():
    """A close at a DIFFERENT book is not this book's close, however many books closed."""
    df = frame([q(2, "Pinnacle", "over", 1.80, 10.0)])
    r = ct.same_book_clv(df, 1, "ou", "over", entry_odds=1.90, bookmaker_id=8)
    assert r.status == ct.NO_SAME_BOOK_CLOSE and r.value is None


def test_same_book_clv_is_positive_when_the_price_shortened():
    df = frame([q(8, "Bet365", "over", 1.80, 10.0)])
    r = ct.same_book_clv(df, 1, "ou", "over", entry_odds=1.90, bookmaker_id=8)
    assert r.usable and r.kind == ct.SAME_BOOK and r.value > 0


def test_same_book_is_not_devigged():
    """Execution quality compares raw prices at one book; its margin is common to both ends."""
    df = frame([q(8, "Bet365", "over", 1.80, 10.0)])
    r = ct.same_book_clv(df, 1, "ou", "over", entry_odds=1.90, bookmaker_id=8)
    assert r.value == pytest.approx(1 / 1.80 - 1 / 1.90, abs=1e-6)  # value is 6dp


# ── sharp_reference: informational quality ────────────────────────────────────────────────────
def test_sharp_requires_a_designated_sharp_source():
    df = frame([q(8, "Bet365", "over", 1.80, 10.0), q(8, "Bet365", "under", 2.05, 10.0)])
    assert ct.sharp_reference_clv(df, 1, "ou", "over", 0.52).status == ct.NO_SHARP_CLOSE


def test_a_one_sided_sharp_close_is_refused_not_halved():
    """One side of a Pinnacle market cannot be separated from its margin."""
    df = frame([q(2, "Pinnacle", "over", 1.90, 10.0)])
    assert ct.sharp_reference_clv(df, 1, "ou", "over", 0.52).status == ct.ONE_SIDED


def test_sharp_two_sided_close_devigs():
    df = frame([q(2, "Pinnacle", "over", 1.95, 10.0), q(2, "Pinnacle", "under", 1.95, 10.0)])
    r = ct.sharp_reference_clv(df, 1, "ou", "over", entry_prob=0.45)
    assert r.usable and r.kind == ct.SHARP_REFERENCE
    assert r.value == pytest.approx(0.05, abs=0.01)   # fair ~0.50 vs entry 0.45


def test_bet365_is_not_treated_as_sharp():
    """Where we bet is not a source of truth about what the price should be."""
    assert "Bet365" not in ct.SHARP_NAMES


# ── consensus ─────────────────────────────────────────────────────────────────────────────────
def _two_sided(n_books):
    rows = []
    for i in range(n_books):
        rows += [q(100 + i, f"Book{i}", "over", 1.95, 10.0),
                 q(100 + i, f"Book{i}", "under", 1.95, 10.0)]
    return frame(rows)


def test_consensus_needs_enough_two_sided_books():
    r = ct.consensus_clv(_two_sided(2), 1, "ou", "over", 0.50)
    assert r.status == ct.INSUFFICIENT_BOOKS and r.n_books == 2


def test_ten_one_sided_books_are_still_insufficient():
    """The brief's case: quantity of one-sided quotes never substitutes for two-sidedness."""
    rows = [q(100 + i, f"Book{i}", "over", 1.95, 10.0) for i in range(10)]
    assert ct.consensus_clv(frame(rows), 1, "ou", "over", 0.50).status == ct.INSUFFICIENT_BOOKS


def test_consensus_works_with_three_two_sided_books():
    r = ct.consensus_clv(_two_sided(3), 1, "ou", "over", entry_prob=0.45)
    assert r.usable and r.kind == ct.CONSENSUS and r.n_books == 3


# ── the shared rule: a missing close stays missing ────────────────────────────────────────────
def test_a_t6h_quote_never_substitutes_for_a_missing_close():
    """The most flattering error available, so it is tested for all three kinds."""
    df = frame([q(8, "Bet365", "over", 1.90, 360.0), q(8, "Bet365", "under", 2.00, 360.0),
                q(2, "Pinnacle", "over", 1.92, 360.0), q(2, "Pinnacle", "under", 1.98, 360.0)])
    assert ct.same_book_clv(df, 1, "ou", "over", 1.90, 8).value is None
    assert ct.sharp_reference_clv(df, 1, "ou", "over", 0.5).value is None
    assert ct.consensus_clv(df, 1, "ou", "over", 0.5).value is None


def test_post_kickoff_only_is_its_own_status():
    """In-play quotes are not a late close — they are a scheduling failure, and the status
    has to say which, because the fixes are different."""
    df = frame([q(8, "Bet365", "over", 1.90, -5.0)])
    assert ct.same_book_clv(df, 1, "ou", "over", 1.90, 8).status == ct.POST_KICKOFF_ONLY


def test_each_kind_tags_itself_so_aggregates_cannot_silently_mix():
    df = frame([q(8, "Bet365", "over", 1.80, 10.0), q(8, "Bet365", "under", 2.10, 10.0),
                q(2, "Pinnacle", "over", 1.85, 10.0), q(2, "Pinnacle", "under", 2.05, 10.0),
                q(3, "BookC", "over", 1.83, 10.0), q(3, "BookC", "under", 2.07, 10.0)])
    kinds = {ct.same_book_clv(df, 1, "ou", "over", 1.90, 8).kind,
             ct.sharp_reference_clv(df, 1, "ou", "over", 0.52).kind,
             ct.consensus_clv(df, 1, "ou", "over", 0.52).kind}
    assert kinds == {ct.SAME_BOOK, ct.SHARP_REFERENCE, ct.CONSENSUS}


def test_the_book_closest_to_kickoff_is_the_one_used():
    """Not the first row, not the last written — the nearest to kickoff."""
    df = frame([q(8, "Bet365", "over", 1.70, 25.0), q(8, "Bet365", "over", 1.95, 5.0)])
    r = ct.same_book_clv(df, 1, "ou", "over", entry_odds=1.90, bookmaker_id=8)
    assert r.close_minutes_to_kickoff == 5.0
