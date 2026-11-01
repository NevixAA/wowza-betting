"""BTTS refresh clock (2026-11-01): an unmoved price must not be refetched inside the refresh window.

Before the fix the clock was read from book_odds_snapshots.csv, which stores price CHANGES only, so
every unmoved market looked stale and the whole board was bought on every predict run.

    python -m pytest tests/test_btts_fetch_clock.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import btts_odds  # noqa: E402


class _Resp:
    status_code = 200
    headers = {"x-requests-last": "2"}

    def json(self):
        return {"bookmakers": [{"key": "b1", "markets": [{"key": "btts", "outcomes": [
            {"name": "Yes", "price": 1.85}, {"name": "No", "price": 1.95}]}]}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test", raising=False)
    monkeypatch.setattr(btts_odds, "append_book_rows", lambda rows: None)   # archive not under test
    calls = []
    monkeypatch.setattr(btts_odds.requests, "get", lambda *a, **k: calls.append(a) or _Resp())
    return tmp_path, calls


def _rows(hours_to_ko=48):
    ko = (datetime.now(timezone.utc) + timedelta(hours=hours_to_ko)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [{"home_team": "Bari", "away_team": "Pisa", "kickoff_utc": ko, "date": ko[:10],
             "_oa_event_id": "e1", "_oa_sport_key": "soccer_italy_serie_b"}]


def test_unmoved_price_is_not_rebought_inside_refresh_window(env):
    _, calls = env
    out = btts_odds.fetch(_rows())
    assert len(calls) == 1 and out[0]["odds_btts"] == 1.85
    btts_odds.fetch(_rows())                                   # same price, minutes later
    assert len(calls) == 1, "an unmoved market inside 2h must be skipped, not bought again"


def test_near_kickoff_is_still_refetched_every_run(env):
    _, calls = env
    btts_odds.fetch(_rows(hours_to_ko=3))
    btts_odds.fetch(_rows(hours_to_ko=3))
    assert len(calls) == 2, "inside the near-kickoff window every run fetches (closing price)"


def test_stale_clock_entry_triggers_refetch(env):
    tmp, calls = env
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp / "btts_fetch_clock.json").write_text('{"Bari vs Pisa": "%s"}' % old, encoding="utf-8")
    btts_odds.fetch(_rows())
    assert len(calls) == 1


def test_unreadable_clock_means_never_fetched(env):
    tmp, calls = env
    (tmp / "btts_fetch_clock.json").write_text("not json", encoding="utf-8")
    btts_odds.fetch(_rows())
    assert len(calls) == 1, "an unknown age is infinitely old, never fresh"


def test_failed_call_is_not_recorded_as_fetched(env, monkeypatch):
    tmp, calls = env

    class _Fail(_Resp):
        status_code = 401

    monkeypatch.setattr(btts_odds.requests, "get", lambda *a, **k: calls.append(a) or _Fail())
    btts_odds.fetch(_rows())
    btts_odds.fetch(_rows())
    assert len(calls) == 2, "an out-of-quota 401 must not freeze the fixture for 2h"
