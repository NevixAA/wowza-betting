"""The selection rules are what stand between the model and real money. Test them as such."""
from __future__ import annotations

import pandas as pd
import pytest

from src.cloudbet.client import Selection
from src.cloudbet.selection import SelectionRules, select, remaining_today


def cand(fid, edge, tier="SNIPER", market="ou25", side="OVER", px=1.95, **kw):
    d = {"fixture_id": fid, "market": market, "side": side, "edge": edge,
         "signal_tier": tier, "cloudbet_price": px,
         "reference_id": f"ref-{fid}-{market}-{side}"}
    d.update(kw)
    return d


def frame(rows):
    return pd.DataFrame(rows)


# ── the owner's rules ─────────────────────────────────────────────────────────────────────────
def test_only_sniper_and_marksman():
    d, rej = select(frame([cand(1, 0.20, "SNIPER"), cand(2, 0.20, "MARKSMAN"),
                           cand(3, 0.20, "VALUABLE"), cand(4, 0.20, "AVOID")]))
    assert set(d.fixture_id) == {1, 2} and rej.reasons["tier"] == 2


def test_edge_must_be_above_five_percent():
    d, _ = select(frame([cand(1, 0.051), cand(2, 0.050), cand(3, 0.049)]))
    assert list(d.fixture_id) == [1], "5.0% exactly is not ABOVE 5%"


def test_best_edge_is_placed_first():
    d, _ = select(frame([cand(1, 0.07), cand(2, 0.21), cand(3, 0.12)]))
    assert list(d.fixture_id) == [2, 3, 1]


def test_daily_cap_is_hard_and_trims_the_weakest():
    rows = [cand(i, 0.05 + i / 1000) for i in range(1, 31)]
    d, rej = select(frame(rows))
    assert len(d) == 20
    assert d.edge.min() > frame(rows).edge.nsmallest(10).max(), "the cap cut the best, not the worst"
    assert rej.reasons["over_daily_cap"] == 10


def test_only_enabled_markets():
    d, _ = select(frame([cand(1, 0.2, market="ou25"), cand(2, 0.2, market="btts"),
                         cand(3, 0.2, market="over15"), cand(4, 0.2, market="over35"),
                         cand(5, 0.2, market="ht_over05"), cand(6, 0.2, market="h2h")]))
    assert set(d.fixture_id) == {1, 2, 3, 4}


# ── the guards that protect real money ────────────────────────────────────────────────────────
def test_never_both_sides_of_one_fixture():
    """Measured on the real ledger: 14 fixtures carry tips on both sides, two at staked tiers.
    A human ignores the second; a bot would place both and guarantee paying the spread twice."""
    d, rej = select(frame([cand(1, 0.20, side="OVER"), cand(1, 0.14, side="UNDER")]))
    assert len(d) == 1 and d.iloc[0]["side"] == "OVER", "kept the higher-edge side"
    assert rej.reasons["second_side_same_fixture"] == 1


def test_a_fixture_bet_on_an_earlier_run_is_not_bet_again():
    d, rej = select(frame([cand(1, 0.30, side="UNDER"), cand(2, 0.10)]),
                    placed_fixtures={"1"})
    assert list(d.fixture_id) == [2]
    assert rej.reasons["fixture_already_bet"] == 1


def test_an_already_placed_reference_is_never_replaced():
    rows = [cand(1, 0.20), cand(2, 0.19)]
    d, _ = select(frame(rows), already_placed={"ref-1-ou25-OVER"})
    assert list(d.fixture_id) == [2]


def test_no_cloudbet_price_means_no_bet():
    """An edge against a price nobody will fill is not an edge."""
    d, rej = select(frame([cand(1, 0.30, px=float("nan")), cand(2, 0.30, px=1.0),
                           cand(3, 0.10, px=2.10)]))
    assert list(d.fixture_id) == [3]
    assert rej.reasons["no_cloudbet_price"] == 2


def test_rules_are_configurable_without_editing_code():
    d, _ = select(frame([cand(i, 0.06) for i in range(1, 11)]),
                  SelectionRules(min_edge=0.09, max_bets_per_day=3))
    assert len(d) == 0, "min_edge was not applied"
    d, _ = select(frame([cand(i, 0.10 + i / 100) for i in range(1, 11)]),
                  SelectionRules(max_bets_per_day=3))
    assert len(d) == 3


def test_remaining_today_never_goes_negative():
    assert remaining_today(25) == 0 and remaining_today(0) == 20 and remaining_today(18) == 2


def test_an_empty_board_is_not_an_error():
    d, _ = select(pd.DataFrame())
    assert d.empty


# ── idempotency ───────────────────────────────────────────────────────────────────────────────
def _sel(price=1.95, edge=0.1):
    return Selection(fixture_id="fx1", event_id="ev1",
                     market_url="soccer.total_goals/over?total=2.5", league="La Liga 2",
                     home_team="A", away_team="B", kickoff_utc="2026-10-06T18:00:00Z",
                     market="ou25", side="OVER", price=price, edge=edge, tier="SNIPER",
                     model_type="standard")


def test_reference_id_is_stable_across_processes():
    """A fresh uuid on retry turns every network blip into a double stake."""
    assert _sel().reference_id == _sel().reference_id


def test_reference_id_ignores_price_and_edge():
    """A retry after a price refresh is the SAME bet. If price were in the key, a one-tick move
    would defeat the dedup and stake twice."""
    assert _sel(price=1.95).reference_id == _sel(price=2.05, edge=0.2).reference_id


def test_reference_id_differs_across_selections():
    a = _sel()
    b = Selection(**{**a.__dict__, "side": "UNDER",
                     "market_url": "soccer.total_goals/under?total=2.5"})
    assert a.reference_id != b.reference_id
