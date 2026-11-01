"""Props slate = every fixture in the window, not the next five kickoffs (2026-11-01).

    python -m pytest tests/test_props_fixture_window.py -q
"""
from __future__ import annotations

import inspect

from player_model import api_football as AF
from player_model import predict as P


def test_fixtures_between_asks_for_a_date_window_not_next_n(monkeypatch):
    seen = {}

    def fake_get(path, params, cache_hours=0):
        seen.update(path=path, params=params)
        return {"response": [{"fixture": {"id": i}} for i in range(18)]}   # a full UCL matchday

    monkeypatch.setattr(AF, "_get", fake_get)
    out = AF.get_fixtures_between(2, "2026", "2026-11-01", "2026-11-05")
    assert len(out) == 18
    assert seen["path"] == "/fixtures"
    assert seen["params"] == {"league": 2, "season": "2026", "from": "2026-11-01", "to": "2026-11-05"}
    assert "next" not in seen["params"]


def test_predict_slate_uses_the_window_and_skips_started_games():
    src = inspect.getsource(P)
    assert "get_fixtures_between" in src
    assert "next_n=n_fix" not in src
    assert '("NS", "TBD")' in src
