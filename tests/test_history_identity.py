"""One club, one fixture, one season label in the merged history (Data Truth audit finding 37).

    python -m pytest tests/test_history_identity.py -q
"""
from __future__ import annotations

import pandas as pd

from src import data_loader as DL


def _row(d, lg, h, a, hg, ag, season):
    return {"date": pd.Timestamp(d), "league": lg, "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag, "season": season}


def test_season_labels_follow_each_leagues_rule():
    d = pd.DataFrame([_row("2025-03-01", "Championship", "A", "B", 1, 0, "2025"),
                      _row("2025-09-01", "Championship", "A", "B", 1, 0, "2025"),
                      _row("2026-03-01", "USA MLS", "C", "D", 1, 0, "2025/26"),
                      _row("2026-03-01", "Austrian Bundesliga", "E", "F", 1, 0, "2026"),
                      _row("2026-11-01", "Japan J-League", "G", "H", 1, 0, "2026"),
                      _row("2026-03-01", "Mystery League", "I", "J", 1, 0, "keep")])
    assert DL._canonical_season_labels(d).tolist() == ["2024/25", "2025/26", "2026", "2025/26",
                                                       "2026/27", "keep"]


def test_duplicate_spellings_collapse_to_one_club_and_one_fixture():
    rows = []
    for i, day in enumerate(["2025-09-06", "2025-09-13", "2025-09-20", "2025-09-27"]):
        rows.append(_row(day, "Championship", "Stoke City", f"T{i}", 2, 1, "2025"))
        rows.append(_row(day, "Championship", "Stoke", f"T{i}", 2, 1, "2025/26"))     # same match
    out = DL._canonical_identity_and_seasons(pd.DataFrame(rows))
    assert len(out) == 4
    assert out["home_team"].nunique() == 1
    assert set(out["season"]) == {"2025/26"}


def test_rollback_switch(monkeypatch):
    monkeypatch.setenv("WOWZA_IDENTITY_DEDUPE", "0")
    d = pd.DataFrame([_row("2025-09-06", "Championship", "Stoke", "X", 2, 1, "2025")])
    assert DL._canonical_identity_and_seasons(d)["season"].tolist() == ["2025"]
