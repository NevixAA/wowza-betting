"""Props seasons come from the canonical registry, never a European-only rule (Data Truth 38, 41).

    python -m pytest tests/test_props_seasons_registry.py -q
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from player_model import config as PC
from player_model import data_fetcher as DF
from src.season_resolver import resolve_api_season

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("league,day,want", [
    ("Japan J-League", date(2026, 11, 1), "2027"),       # autumn-spring since 2026-07
    ("USA MLS", date(2027, 3, 1), "2027"),               # calendar year: not the July rule's 2026
    ("Brazil Serie A", date(2027, 3, 1), "2027"),
    ("Championship", date(2027, 3, 1), "2026"),           # European: start year
    ("Championship", date(2026, 11, 1), "2026"),
])
def test_prop_season_is_league_aware(league, day, want):
    assert PC.prop_season(league, day) == want
    assert DF._current_season(league, day) == want


def test_prop_seasons_table_matches_the_resolver_for_every_prop_league():
    assert PC.PROP_SEASONS == {lg: resolve_api_season(lg) for lg in PC.PROP_LEAGUES}


def test_no_literal_season_fallbacks_left():
    pat = re.compile(r'PROP_SEASONS\.get\([^)]*"20\d\d"|"season":\s*"20\d\d"')
    hits = [f"{p.relative_to(ROOT)}:{i}" for d in ("player_model", "telegram_bot", "scripts")
            for p in (ROOT / d).glob("*.py")
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if pat.search(ln.split("#", 1)[0])]
    assert not hits, hits
