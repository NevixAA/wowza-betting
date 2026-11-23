"""P1 item 7 (2026-11-22) - a close is only ever compared with an entry on the SAME fixture, market,
line and side, and the CLV record says honestly whether entry and close came from the same feed.

* market / line / side identity: an over25 close never answers an under25, over15 or btts row;
  a fixture's close is never taken from another league or another date.
* feed identity: the shadow entry is the predict run's OddsAPI price (first listed book, book name
  unrecorded); a close from a Bet365 capture archive is CROSS-FEED and is labelled so; nothing
  is ever labelled same-BOOK because no archive proves which OddsAPI book was used.
* clv_types.same_book_clv refuses another book's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clv_collection_health as cch
from src import shadow_settlement as ss
from src.clv_types import NO_SAME_BOOK_CLOSE, OK, same_book_clv

KO = pd.Timestamp("2026-11-21T15:00:00Z")


def _frame(rows, league="Championship", date="2026-11-21"):
    return pd.DataFrame([{"snapshot_ts": (KO - pd.Timedelta(minutes=m)).strftime(
                              "%Y-%m-%dT%H:%M:%SZ"),
                          "match_date": date, "kickoff_utc": "2026-11-21T15:00:00Z",
                          "league": league, "match": "Leeds United vs Norwich City",
                          "market": mk, "odds": o} for mk, m, o in rows])


def _close(archives, key, league="Championship"):
    return ss.closing_price(ss.index_archives(archives), "Leeds United", "Norwich City",
                            "2026-11-21", KO, key, league=league, max_min=ss.CLOSE_MAX_MIN)


# ── market, line and side identity ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("market,side,key", [
    ("ou25", "OVER", "over25"), ("ou25", "UNDER", "under25"),
    ("over15", "OVER", "over15"), ("ou15", "UNDER", "under15"),
    ("over35", "OVER", "over35"), ("ou35", "UNDER", "under35"),
    ("btts", "YES", "btts_yes"), ("btts", "NO", "btts_no"),
])
def test_each_market_side_maps_to_exactly_its_own_archive_key(market, side, key):
    assert ss.MARKETS[ss.norm_market(market)][2][ss.norm_side(side)] == key


def test_an_over25_close_never_answers_another_line_or_side():
    arch = [("a", _frame([("over25", 5, 1.9)]))]
    assert _close(arch, "over25")["status"] == ss.CLOSE_OK
    for other in ("under25", "over15", "over35", "btts_yes"):
        c = _close(arch, other)
        assert c["status"] == ss.NO_CLOSE and np.isnan(c["odds"]), other


def test_a_same_named_fixture_in_another_league_never_supplies_the_close():
    arch = [("a", _frame([("over25", 5, 1.9)], league="Serie B"))]
    c = _close(arch, "over25", league="Championship")
    assert c["status"] == ss.NO_CLOSE and np.isnan(c["odds"])


def test_a_close_from_another_date_is_never_used():
    arch = [("a", _frame([("over25", 5, 1.9)], date="2026-11-22"))]
    assert _close(arch, "over25")["status"] == ss.NO_CLOSE


# ── feed identity (entry OddsAPI vs close source) ────────────────────────────────────────────
def test_entry_feed_archives_are_the_predict_dense_archives():
    assert ss.ENTRY_FEED_ARCHIVE == {"standard": "standard_odds_history.csv",
                                     "new_format": "newformat_odds_dense.csv"}


@pytest.mark.parametrize("mt,src,status,want", [
    ("standard", "standard_odds_history.csv", "OK", ss.SAME_FEED),
    ("new_format", "newformat_odds_dense.csv", "OK", ss.SAME_FEED),
    ("standard", "standard_sidemarket_odds_history.csv", "OK", ss.CROSS_FEED),
    ("new_format", "newformat_odds_history.csv", "OK", ss.CROSS_FEED),
    ("new_format", "odds_heartbeat/nf_capture", "OK", ss.CROSS_FEED),
    # a NEW-FORMAT entry against the STANDARD dense archive is not its own feed
    ("new_format", "standard_odds_history.csv", "OK", ss.CROSS_FEED),
    ("standard", "standard_odds_history.csv", ss.STALE_CLOSE, ss.NO_CLOSE_FEED),
    ("standard", "", "OK", ss.NO_CLOSE_FEED),
    ("", "standard_odds_history.csv", "OK", ss.UNKNOWN_FEED),
])
def test_close_feed_identity(mt, src, status, want):
    assert ss.close_feed_identity(mt, src, status) == want


def test_no_label_ever_claims_same_book():
    labels = {ss.SAME_FEED, ss.CROSS_FEED, ss.NO_CLOSE_FEED, ss.UNKNOWN_FEED}
    assert not any("SAME_BOOK" in x for x in labels)
    assert ss.SAME_FEED.endswith("BOOK_UNRECORDED")


def test_health_counts_feed_identity_only_for_certified_closes():
    df = pd.DataFrame([
        {"result": "WIN", "close_status": "OK", "close_minutes_to_kickoff": 3.0,
         "close_source": "standard_odds_history.csv", "model_type": "standard"},
        {"result": "LOSS", "close_status": "OK", "close_minutes_to_kickoff": 2.0,
         "close_source": "newformat_odds_history.csv", "model_type": "new_format"},
        # recorded as OK but 45 minutes out: not certified, so no feed claim
        {"result": "WIN", "close_status": "OK", "close_minutes_to_kickoff": 45.0,
         "close_source": "standard_odds_history.csv", "model_type": "standard"},
    ]).assign(kickoff_utc="2026-11-21 15:00:00", league="Championship", market="ou25")
    s = cch.shadow_section(df)
    assert s["close_feed_identity"] == {ss.CROSS_FEED: 1, ss.NO_CLOSE_FEED: 1, ss.SAME_FEED: 1}


# ── same-book CLV (clv_types) ────────────────────────────────────────────────────────────────
def _quotes():
    return pd.DataFrame([
        {"fixture_id": 1, "market": "OU", "side": "over", "line": 2.5, "bookmaker_id": 8,
         "bookmaker": "Bet365", "odds": 1.90, "minutes_to_kickoff": 5.0},
        {"fixture_id": 1, "market": "OU", "side": "over", "line": 2.5, "bookmaker_id": 11,
         "bookmaker": "1xBet", "odds": 1.95, "minutes_to_kickoff": 4.0},
        {"fixture_id": 1, "market": "OU", "side": "over", "line": 3.5, "bookmaker_id": 8,
         "bookmaker": "Bet365", "odds": 2.80, "minutes_to_kickoff": 3.0},
    ])


def test_same_book_clv_uses_only_the_entry_books_close_on_the_same_line():
    r = same_book_clv(_quotes(), 1, "OU", "over", 2.0, bookmaker_id=8, line=2.5)
    assert r.status == OK and r.usable
    assert r.close_minutes_to_kickoff == 5.0          # Bet365's own 2.5 close, not 1xBet / 3.5


def test_same_book_clv_refuses_when_only_another_book_closed():
    r = same_book_clv(_quotes(), 1, "OU", "over", 2.0, bookmaker_id=6, line=2.5)
    assert r.status == NO_SAME_BOOK_CLOSE and r.value is None
