"""Pricing and fixture matching. Both failures here cost real money, in opposite ways:
a missed price costs a bet; a wrong match costs the stake on a game we never meant to bet."""
from __future__ import annotations

import pytest

from src.cloudbet.feed import (MARKET_MAP, Quote, discover_markets, find_event, match_team,
                               price_for)


def ev(home="Velez Sarsfield", away="Platense", markets=None):
    return {"home": {"name": home}, "away": {"name": away}, "markets": markets or {}}


def totals(**lines):
    """Build a total_goals market: totals(**{'2.5': (1.95, 1.90)}) -> over/under prices."""
    return {"soccer.total_goals": {"submarkets": {
        f"total={k}": {"selections": [
            {"outcome": "over", "price": v[0], "status": "SELECTION_ENABLED"},
            {"outcome": "under", "price": v[1], "status": "SELECTION_ENABLED"}]}
        for k, v in lines.items()}}}


# ── pricing ───────────────────────────────────────────────────────────────────────────────────
def test_the_right_line_is_picked_from_many():
    e = ev(markets=totals(**{"1.5": (1.30, 3.40), "2.5": (1.95, 1.90), "3.5": (3.60, 1.28)}))
    q, why = price_for(e, "ou25", "OVER")
    assert q and q.odds == 1.95 and q.line == 2.5, why
    assert price_for(e, "over15", "OVER")[0].odds == 1.30
    assert price_for(e, "over35", "OVER")[0].odds == 3.60


def test_the_market_url_matches_cloudbets_documented_shape():
    q, _ = price_for(ev(markets=totals(**{"2.5": (1.95, 1.90)})), "ou25", "UNDER")
    assert q.market_url == "soccer.total_goals/under?total=2.5"


def test_a_missing_market_names_what_was_actually_there():
    """An unmapped market must be diagnosable from one log line, not a debugging session."""
    q, why = price_for(ev(markets={"soccer.match_odds": {"submarkets": {}}}), "btts", "YES")
    assert q is None and "soccer.match_odds" in why


def test_a_missing_line_names_the_lines_that_existed():
    q, why = price_for(ev(markets=totals(**{"1.5": (1.30, 3.40)})), "ou25", "OVER")
    assert q is None and "total=1.5" in why


def test_a_suspended_selection_is_not_a_price():
    e = {"home": {"name": "A"}, "away": {"name": "B"}, "markets": {"soccer.total_goals": {
        "submarkets": {"total=2.5": {"selections": [
            {"outcome": "over", "price": 1.95, "status": "SELECTION_DISABLED"}]}}}}}
    assert price_for(e, "ou25", "OVER")[0] is None


def test_an_impossible_price_is_rejected():
    e = ev(markets=totals(**{"2.5": (1.0, 0.5)}))
    assert price_for(e, "ou25", "OVER")[0] is None


def test_discover_markets_reports_the_real_keys():
    """The map is a hypothesis until a live payload confirms it. This is how it gets fixed."""
    d = discover_markets(ev(markets=totals(**{"2.5": (1.95, 1.90)})))
    assert d == {"soccer.total_goals": ["over", "under"]}


# ── fixture matching (invariant 11) ───────────────────────────────────────────────────────────
def test_noise_tokens_do_not_prevent_a_match():
    assert match_team("Velez Sarsfield", ["CA Velez Sarsfield", "Platense"])[0] \
        == "CA Velez Sarsfield"
    assert match_team("Kaiserslautern", ["1. FC Kaiserslautern"])[0] == "1. FC Kaiserslautern"


def test_manchester_city_never_matches_manchester_united():
    got, why = match_team("Manchester City", ["Manchester United"])
    assert got is None, f"matched the wrong club: {got}"


def test_an_ambiguous_name_is_refused_not_guessed():
    """`Real Valladolid CF` once matched any club starting 'Real', leaving 46% of standard
    fixtures with no form data. Refusing is the correct behaviour."""
    got, why = match_team("Real Sociedad", ["Real Sociedad B", "Real Sociedad"])
    assert got == "Real Sociedad" or (got is None and "ambiguous" in why)


def test_a_name_sharing_nothing_is_refused():
    assert match_team("Platense", ["Boca Juniors", "River Plate"])[0] is None


def test_both_clubs_must_resolve_to_the_SAME_event():
    """The dangerous case: each club matches, but in two different fixtures. Betting that is
    betting a game we never meant to bet."""
    events = [ev("Velez Sarsfield", "Boca Juniors"), ev("River Plate", "Platense")]
    e, why = find_event(events, "Velez Sarsfield", "Platense")
    assert e is None and "different events" in why


def test_a_clean_fixture_resolves():
    events = [ev("CA Velez Sarsfield", "CA Platense"), ev("Boca Juniors", "River Plate")]
    e, why = find_event(events, "Velez Sarsfield", "Platense")
    assert e is not None, why


def test_every_market_we_bet_is_mapped():
    from src.cloudbet.selection import DEFAULT_MARKETS
    assert set(DEFAULT_MARKETS) <= set(MARKET_MAP)
