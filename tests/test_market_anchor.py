"""Tests for the market anchor hierarchy. The critical one is the refusal on one-sided prices."""
from __future__ import annotations
import numpy as np, pytest
from src.market_anchor import (CROSS_BOOK_MEDIAN, EXCHANGE_FAIR, INSUFFICIENT, SINGLE_BOOK,
                               anchor, power_devig, proportional_devig)


def test_a_one_sided_price_returns_no_probability():
    """THE test. A one-sided quote cannot separate margin from opinion."""
    a = anchor(over_quotes={"bet365": 1.91})
    assert a.probability is None
    assert a.status == INSUFFICIENT
    assert not a.usable


def test_many_one_sided_books_are_still_insufficient():
    """Ten overs and no unders is ten times no market, not a consensus."""
    a = anchor(over_quotes={f"b{i}": 1.9 + i / 100 for i in range(10)})
    assert a.status == INSUFFICIENT and a.probability is None


def test_a_book_quoting_only_one_side_does_not_count():
    a = anchor(over_quotes={"a": 1.91, "b": 1.92, "c": 1.93},
               under_quotes={"a": 1.95})
    assert a.status == SINGLE_BOOK and a.n_books == 1


def test_hierarchy_prefers_exchange_then_median_then_single():
    q_o = {"a": 1.91, "b": 1.92, "c": 1.93}
    q_u = {"a": 1.95, "b": 1.94, "c": 1.93}
    assert anchor(q_o, q_u).status == CROSS_BOOK_MEDIAN
    assert anchor(q_o, q_u, exchange_over=2.00, exchange_under=2.00).status == EXCHANGE_FAIR
    assert anchor({"a": 1.91}, {"a": 1.95}).status == SINGLE_BOOK


def test_devig_removes_the_margin():
    p = power_devig([1.91, 1.95])
    assert abs(sum(p) - 1.0) < 1e-9
    assert sum(1 / o for o in (1.91, 1.95)) > 1.0      # the raw market had a margin
    assert p[0] < 1 / 1.91, "the de-vigged probability must be below the raw implied one"


def test_power_and_proportional_differ_and_power_is_not_symmetric():
    """On a lopsided market the two methods must disagree — that is why the choice matters."""
    pw, pr = power_devig([1.25, 4.20]), proportional_devig([1.25, 4.20])
    assert abs(pw[0] - pr[0]) > 1e-4


def test_absurd_overround_is_refused():
    """1.40/1.40 implies a 43% margin — that is a bad parse or a stale pair, not a market."""
    assert anchor({"a": 1.40}, {"a": 1.40}).status == INSUFFICIENT


def test_arbitrage_pair_is_refused():
    """Sub-1.0 overround means the two sides came from different moments or markets."""
    assert anchor({"a": 2.20}, {"a": 2.20}).status == INSUFFICIENT


def test_usable_flag_guards_callers():
    assert anchor({"a": 1.91}, {"a": 1.95}).usable is True
    assert anchor({"a": 1.91}).usable is False


def test_exchange_commission_is_applied():
    """A traded exchange price is not a fair price until commission is removed."""
    a = anchor(exchange_over=2.10, exchange_under=1.90)
    b = anchor({"x": 2.10}, {"x": 1.90})
    assert a.status == EXCHANGE_FAIR and b.status == SINGLE_BOOK
    assert a.probability != b.probability
