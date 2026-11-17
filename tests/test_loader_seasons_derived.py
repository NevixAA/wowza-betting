"""History downloads roll with the calendar and cover every registry league (Data Truth 32-46).

    python -m pytest tests/test_loader_seasons_derived.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

from src import data_loader as DL
from src import league_registry as LR

_RAW = (Path(__file__).resolve().parents[1] / "src" / "data_loader.py").read_text(encoding="utf-8")
# Code only: comments and docstring prose quote the old defects on purpose.
SRC = "\n".join(ln.split("#", 1)[0] for ln in _RAW.splitlines()
                if not ln.strip().startswith(("#", '"', "'")) and "This was" not in ln)


def test_every_model_league_with_a_football_data_code_is_downloaded():
    std, new = DL._fd_download_tables()
    want = {e["name"]: e["football_data_code"] for e in LR.leagues()
            if e.get("football_data_code") and e.get("model_type")}
    assert {**std, **new} == want
    assert "Romanian Superliga" in new                      # enabled, was missing from the hand table
    assert all(e.get("model_type") == "standard" for e in LR.leagues() if e["name"] in std)
    assert all(e.get("model_type") == "new_format" for e in LR.leagues() if e["name"] in new)


def test_a_league_with_no_model_is_never_filed_in_the_new_bucket():
    # LR-1.2: the big-5 hold mmz4281 codes but no model_type; before the skip they would have
    # been downloaded from /new/{code}.csv by elimination.
    std, new = DL._fd_download_tables()
    for lg in ("Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1",
               "Swiss Super League", "Poland Ekstraklasa"):
        assert LR.entry(lg)["football_data_code"] and lg not in std and lg not in new
        assert LR.football_data_format(lg) == ("new" if lg in ("Swiss Super League", "Poland Ekstraklasa")
                                               else "mmz4281")


def test_no_literal_season_year_lists_or_pinned_folders():
    assert not re.search(r"20\d\d\|20\d\d\|20\d\d", SRC), "literal season alternation"
    assert "mmz4281/2526" not in SRC, "pinned football-data season folder"


def test_new_format_season_assigned_by_index_not_position():
    assert ".values[:len(df)]" not in SRC
