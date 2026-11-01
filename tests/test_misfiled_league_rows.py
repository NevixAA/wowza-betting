"""No fixture is filed under the wrong league (Data Truth audit 2026-11-01, findings 4/5/31).

    python -m pytest tests/test_misfiled_league_rows.py -q
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader as DL

ROOT = Path(__file__).resolve().parents[1]
DUTCH = {"Ajax", "PSV", "Feyenoord", "AZ Alkmaar", "Twente", "Heerenveen", "Utrecht"}


def test_unknown_league_and_split_label_in_calendar_league_are_dropped():
    d = pd.DataFrame({
        "league": ["Norway Eliteserien", "Norway Eliteserien", "Denmark 1st Div", "Championship",
                   "Danish Superliga"],
        "season": ["2024", "2023/24", "2023/24", "2023/24", "2023/24"],
        "home_team": ["Brann", "Ajax", "Bochum", "Hull", "Brondby"]})
    kept = DL._drop_misfiled_league_rows(d, "t")
    assert list(kept["home_team"]) == ["Brann", "Hull", "Brondby"]       # alias resolves, kept


@pytest.mark.parametrize("name", ["af_history.parquet", "backtest_all_leagues.csv"])
def test_committed_banks_hold_no_misfiled_rows(name):
    p = ROOT / "output" / name
    if not p.exists():
        pytest.skip(f"no {name}")
    d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    assert len(DL._drop_misfiled_league_rows(d, "t")) == len(d)
    norway = d[d["league"] == "Norway Eliteserien"]
    assert not (set(norway["home_team"]) & DUTCH)


def test_safe_joins_and_price_filter_are_on_by_default(monkeypatch):
    monkeypatch.delenv("WOWZA_SAFE_ENRICH_JOINS", raising=False)
    monkeypatch.delenv("WOWZA_PRICE_MAPPING_FILTER", raising=False)
    assert DL._safe_joins() and DL._price_mapping_filter()
