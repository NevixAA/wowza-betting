"""Two edges, kept apart. The circularity test is the one that matters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cloudbet.consensus import both_edges, consensus_probability, soft_price_edge


def q(book, side, odds, fid=1, market="ou", line=2.5, mins=20.0):
    return {"fixture_id": fid, "bookmaker": book, "market": market, "side": side,
            "odds": odds, "line": line, "minutes_to_kickoff": mins}


def market(n_books=6, over=1.95, under=1.95, extra=()):
    rows = []
    for i in range(n_books):
        rows += [q(f"Book{i}", "over", over), q(f"Book{i}", "under", under)]
    rows += list(extra)
    return pd.DataFrame(rows)


# ── the consensus itself ──────────────────────────────────────────────────────────────────────
def test_a_fair_market_devigs_to_a_half():
    c = consensus_probability(market(), 1, "ou", "over", 2.5)
    assert c.usable and c.probability == pytest.approx(0.5, abs=1e-6) and c.n_books == 6


def test_a_book_quoting_one_side_does_not_count():
    """With one side you cannot separate the margin from the opinion."""
    rows = [q("A", "over", 1.95), q("A", "under", 1.95),
            q("B", "over", 1.95), q("B", "under", 1.95),
            q("C", "over", 2.10)]                       # one-sided
    c = consensus_probability(pd.DataFrame(rows), 1, "ou", "over", 2.5)
    assert c.n_books == 2 and c.status == "INSUFFICIENT_BOOKS" and c.probability is None


def test_ten_one_sided_books_are_still_insufficient():
    rows = [q(f"B{i}", "over", 2.0) for i in range(10)]
    assert consensus_probability(pd.DataFrame(rows), 1, "ou", "over", 2.5).probability is None


def test_the_quote_closest_to_kickoff_wins():
    rows = [q("A", "over", 1.50, mins=300), q("A", "over", 2.00, mins=5),
            q("A", "under", 2.00, mins=5), q("B", "over", 2.00, mins=5),
            q("B", "under", 2.00, mins=5), q("C", "over", 2.00, mins=5),
            q("C", "under", 2.00, mins=5)]
    c = consensus_probability(pd.DataFrame(rows), 1, "ou", "over", 2.5)
    assert c.probability == pytest.approx(0.5, abs=1e-6)


def test_post_kickoff_quotes_are_excluded():
    rows = []
    for i in range(3):
        rows += [q(f"B{i}", "over", 2.0, mins=-5), q(f"B{i}", "under", 2.0, mins=-5)]
    assert consensus_probability(pd.DataFrame(rows), 1, "ou", "over", 2.5).probability is None


# ── LEAVE-ONE-OUT: the test this module exists for ────────────────────────────────────────────
def test_the_book_being_priced_is_excluded_from_its_own_consensus():
    """Leave-one-out is the discipline even when the median happens to absorb the outlier."""
    df = market(6, extra=[q("Cloudbet", "over", 2.30), q("Cloudbet", "under", 1.70)])
    c_all = consensus_probability(df, 1, "ou", "over", 2.5)
    c_exc = consensus_probability(df, 1, "ou", "over", 2.5, exclude=("Cloudbet",))
    assert c_all.n_books == 7 and c_exc.n_books == 6
    assert "Cloudbet" not in c_exc.books
    # With six books agreeing, the MEDIAN absorbs one outlier entirely — which is precisely why
    # a median is used rather than a mean. The circularity still shows up in the dispersion.
    assert c_all.probability == c_exc.probability
    assert c_all.dispersion > (c_exc.dispersion or 0)


def test_including_the_priced_book_measurably_contaminates_the_consensus():
    """MEASURED, not asserted from intuition. On identical books a median absorbs one outlier
    entirely; on a realistically scattered market it never does.

    Simulated over 400 markets per book count, books scattered N(0.50, 0.03), Cloudbet soft by
    6pp. Shift in the consensus from including Cloudbet in its own comparison:

        other books   median   p95
                  3   1.11pp   3.49pp
                  6   0.52pp   1.80pp
                  8   0.39pp   1.34pp

    A soft-price edge worth betting is roughly 3-5pp, so self-comparison eats 10-17% of it at
    six books and can eat all of it at three — and it bites hardest on thin markets, which is
    exactly where a soft price is most likely to exist.
    """
    rng = np.random.default_rng(0)
    for n_books, floor in ((3, 0.004), (6, 0.002)):
        shifts = []
        for _ in range(120):
            ps = np.clip(rng.normal(0.50, 0.03, n_books), 0.05, 0.95)
            rows = []
            for i, p in enumerate(ps):
                rows += [q(f"B{i}", "over", 1 / (p * 1.05)),
                         q(f"B{i}", "under", 1 / ((1 - p) * 1.05))]
            rows += [q("Cloudbet", "over", 1 / (0.44 * 1.05)),
                     q("Cloudbet", "under", 1 / (0.56 * 1.05))]
            d = pd.DataFrame(rows)
            a = consensus_probability(d, 1, "ou", "over", 2.5)
            b = consensus_probability(d, 1, "ou", "over", 2.5, exclude=("Cloudbet",))
            if a.probability is not None and b.probability is not None:
                shifts.append(abs(a.probability - b.probability))
        assert shifts, f"no usable markets at {n_books} books"
        assert float(np.median(shifts)) > floor, (
            f"{n_books} books: median shift {np.median(shifts):.4f} — the contamination this "
            f"module exists to avoid did not appear, so the test is not testing it")


def test_soft_price_edge_always_excludes_the_book_it_prices():
    df = market(6, extra=[q("Cloudbet", "over", 2.30), q("Cloudbet", "under", 1.70)])
    edge, c = soft_price_edge(df, 1, "ou", "over", 2.30)
    assert "Cloudbet" not in c.books
    # consensus 0.50 vs implied 1/2.30 = 0.4348 -> they are cheap by ~6.5pp
    assert edge == pytest.approx(0.5 - 1 / 2.30, abs=1e-6)


def test_a_book_in_line_with_the_market_shows_no_softness():
    df = market(6, extra=[q("Cloudbet", "over", 1.95), q("Cloudbet", "under", 1.95)])
    edge, _ = soft_price_edge(df, 1, "ou", "over", 1.95)
    # equal to the market's RAW price, so the gap is just the vig — and it is negative.
    assert edge < 0


# ── the two edges must stay separate ──────────────────────────────────────────────────────────
def test_the_two_edges_are_reported_separately_and_never_summed():
    df = market(6, extra=[q("Cloudbet", "over", 2.30), q("Cloudbet", "under", 1.70)])
    r = both_edges(0.56, 2.30, df, 1, "ou", "over", 2.5)
    assert r["edge_vs_model"] == pytest.approx(0.56 - 1 / 2.30, abs=1e-6)
    assert r["edge_vs_consensus"] == pytest.approx(0.50 - 1 / 2.30, abs=1e-6)
    assert r["edge_vs_model"] != r["edge_vs_consensus"]
    assert "edge" not in r, "a combined edge would invite adding two different claims together"


def test_agree_is_recorded_but_nothing_filters_on_it_yet():
    """Which of model-only / market-only / both actually performs is an empirical question.
    The data to answer it is recorded; the answer is not assumed."""
    df = market(6, extra=[q("Cloudbet", "over", 2.30), q("Cloudbet", "under", 1.70)])
    assert both_edges(0.56, 2.30, df, 1, "ou", "over", 2.5)["agree"] is True
    assert both_edges(0.40, 2.30, df, 1, "ou", "over", 2.5)["agree"] is False
    import inspect
    from src.cloudbet import selection
    assert "agree" not in inspect.getsource(selection), "the bot is filtering on an untested rule"


def test_dispersion_is_reported_so_a_thin_disagreeing_market_is_visible():
    tight = market(6)
    wide = pd.DataFrame([q("A", "over", 1.60), q("A", "under", 2.50),
                         q("B", "over", 2.30), q("B", "under", 1.70),
                         q("C", "over", 1.95), q("C", "under", 1.95)])
    assert consensus_probability(wide, 1, "ou", "over", 2.5).dispersion > \
        (consensus_probability(tight, 1, "ou", "over", 2.5).dispersion or 0)
