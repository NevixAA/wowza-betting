"""Estate scan 2026-11-02 finding (f): prop CLV must compare like with like.

The ENTRY is the best price across books (odds_fetcher._best_price; API-Football only ever raises
it). The CLOSE used to be the last history row, which for API-Football is ONE bookmaker's quote,
so a book that merely logged last could manufacture CLV. Now both are best-of-books, and the
basis is written on every new entry. No file is written by these tests."""
from __future__ import annotations

import pandas as pd

from player_model import clv_tracker

DAY = "2026-10-20"
MATCH = "X vs Y"


def _close(monkeypatch, hist, odds_bet):
    import src.rolling_store as rs
    # the entry is quoted at 09:00, logged at 09:30; kickoff 18:00 (Phase 2 part 7 integrity checks)
    rec = {"bet_id": f"PLAYER|{DAY}|Ana Lima|sot", "match": MATCH, "odds_close": "",
           "odds_bet": str(odds_bet), "ts_bet": f"{DAY}T09:30:00+00:00", "kickoff_utc": f"{DAY}T18:00:00Z"}
    closes = {}
    monkeypatch.setattr(clv_tracker.clv_capture, "_load", lambda: [rec])
    monkeypatch.setattr(clv_tracker.clv_capture, "capture_close",
                        lambda bid, o, u, *a: closes.__setitem__(bid, o))
    monkeypatch.setattr(clv_tracker.clv_capture, "mark_status", lambda *a: None)
    monkeypatch.setattr(clv_tracker, "_player_ids", lambda: {})
    monkeypatch.setattr(rs, "read_all", lambda path, dates=None: hist.copy())
    assert clv_tracker.close_out() == 1
    return closes[rec["bet_id"]]


def _row(ts, odds, book, source="api_football"):
    return {"snapshot_ts": f"{DAY}T{ts}:00Z", "match_date": DAY, "match": MATCH,
            "player": "ana lima", "market": "sot", "odds": odds, "source": source,
            "bookmaker": book}


def test_close_is_best_of_books_not_last_logged_book(monkeypatch):
    # entry = best of {Bet365 2.10, Betano 2.30} = 2.30. Bet365 then logs 2.00 last; Betano never
    # moved. Best-vs-best close = 2.30 (CLV 0); the old rule took Bet365's 2.00 (+15% phantom CLV).
    hist = pd.DataFrame([_row("09:00", 2.10, "Bet365"), _row("09:00", 2.30, "Betano"),
                         _row("13:00", 2.00, "Bet365")])
    assert _close(monkeypatch, hist, 2.30) == 2.30


def test_oddsapi_rows_are_one_best_of_books_series(monkeypatch):
    # OddsAPI rows carry no bookmaker: each is already best-of-books at its snapshot, so the latest
    # one is the close (an earlier, higher best price is not).
    hist = pd.DataFrame([_row("09:00", 3.0, None, "oddsapi"), _row("13:00", 2.6, None, "oddsapi")])
    assert _close(monkeypatch, hist, 3.0) == 2.6


def test_new_entries_are_labelled_with_the_basis(monkeypatch):
    logged = []
    monkeypatch.setattr(clv_tracker.clv_capture, "_load", lambda: [])
    monkeypatch.setattr(clv_tracker.clv_capture, "log_bet",
                        lambda *a, **k: logged.append(k.get("notes")))
    tips = pd.DataFrame([{"tier": "PAPER", "market_odds": 2.3, "date": DAY,
                          "player_name": "Ana Lima", "market": "sot", "match": MATCH}])
    assert clv_tracker.log_new_tips(tips) == 1
    assert logged == [f"PAPER;{clv_tracker.CLV_BASIS}"]
