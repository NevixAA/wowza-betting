"""Historical season eras match Pro's evidence-backed calendar (Data Truth audit finding 56).

    python -m pytest tests/test_season_eras_history.py -q
"""
from __future__ import annotations

import pytest

from src.season_resolver import resolve_api_season


@pytest.mark.parametrize("league,day,want", [
    ("Argentina Primera Division", "2019-11-01", "2019"),   # split season 2019/20
    ("Argentina Primera Division", "2020-02-01", "2019"),
    ("Argentina Primera Division", "2020-11-01", "2020"),   # Copa Diego Maradona
    ("Argentina Primera Division", "2021-01-15", "2020"),
    ("Argentina Primera Division", "2022-07-15", "2022"),   # calendar year
    ("Brazil Serie A", "2021-02-10", "2020"),               # COVID-extended 2020
    ("Brazil Serie A", "2021-06-10", "2021"),
    ("China Super League", "2022-01-10", "2021"),           # 2021 finished in Jan 2022
    ("China Super League", "2022-06-10", "2022"),
    ("China Super League", "2026-11-01", "2026"),            # today unchanged
    ("Brazil Serie A", "2026-11-01", "2026"),
    ("Argentina Primera Division", "2026-11-01", "2026"),
])
def test_historical_eras(league, day, want):
    assert resolve_api_season(league, day) == want
