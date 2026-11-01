"""Over 1.5 / 3.5 prices from alternate_totals, targeted and capped (Data Truth finding 9, option B).

    python -m pytest tests/test_alt_totals_targeted.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import config
from src import btts_odds as B


class _Resp:
    status_code = 200
    headers = {"x-requests-last": "2"}

    def __init__(self, markets):
        self._m = markets

    def json(self):
        mk = [{"key": "btts", "outcomes": [{"name": "Yes", "price": 1.9}, {"name": "No", "price": 1.9}]}]
        if "alternate_totals" in self._m:
            mk.append({"key": "alternate_totals", "outcomes": [
                {"name": "Over", "point": 1.5, "price": 1.30}, {"name": "Under", "point": 1.5, "price": 3.40},
                {"name": "Over", "point": 3.5, "price": 3.10}, {"name": "Under", "point": 3.5, "price": 1.36}]})
        return {"bookmakers": [{"key": "bk", "markets": mk}]}


@pytest.fixture()
def calls(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(config, "ODDS_API_KEY", "k", raising=False)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(B, "append_book_rows", lambda rows: None)
    monkeypatch.setattr(B, "_save_clock", lambda *a, **k: None)
    monkeypatch.setattr(B, "_log_usage", lambda **k: None)
    monkeypatch.setattr(B.requests, "get", lambda url, params=None, timeout=0:
                        seen.append(params["markets"]) or _Resp(params["markets"]))
    return seen


def _row(i, hours):
    ko = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"home_team": f"H{i}", "away_team": f"A{i}", "league": "Championship", "date": ko[:10],
            "kickoff_utc": ko, "_oa_event_id": f"e{i}", "_oa_sport_key": "soccer_efl_champ"}


def test_near_kickoff_fixture_gets_both_lines(calls):
    out = B.fetch([_row(1, 10)])
    assert calls == ["btts,alternate_totals"]
    assert out[0]["odds_over15"] == 1.30 and out[0]["odds_over35"] == 3.10


def test_far_fixture_pays_only_for_btts(calls):
    out = B.fetch([_row(1, 72)])
    assert calls == ["btts"]
    assert "odds_over15" not in out[0]


def test_alt_calls_are_capped_per_run(calls, monkeypatch):
    monkeypatch.setenv("ALT_TOTALS_MAX_CALLS", "2")
    B.fetch([_row(i, 5) for i in range(4)])
    assert calls.count("btts,alternate_totals") == 2 and calls.count("btts") == 2
