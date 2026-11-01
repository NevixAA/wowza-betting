"""Form features find the club's history whatever the bookmaker calls it (Data Truth 6, 8).

    python -m pytest tests/test_form_name_resolution.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_upcoming_features
from src.team_names import resolve


@pytest.mark.parametrize("name,history,want", [
    ("Saint Etienne", ["St Etienne", "Lens"], "St Etienne"),
    ("Austria Wien", ["Austria Vienna", "Austria Lustenau", "Austria Klagenfurt"], "Austria Vienna"),
    ("Atletico Mineiro", ["Atletico-MG", "Atletico GO", "Atletico Paranaense"], "Atletico-MG"),
    ("Argentinos Juniors", ["Argentinos JRS", "Boca Juniors"], "Argentinos JRS"),
    ("Blackburn Rovers", ["Blackburn", "Bristol Rvs"], "Blackburn"),
    ("Queens Park Rangers", ["QPR", "Preston"], "QPR"),
])
def test_bookmaker_names_resolve_to_their_history_club(name, history, want):
    assert resolve(name, history) == want


def _hist():
    rng = np.random.default_rng(1)
    rows = []
    d = pd.Timestamp("2026-08-01")
    clubs = ["Blackburn", "QPR", "Preston", "Hull"]
    for k in range(12):
        h, a = clubs[k % 4], clubs[(k + 1 + k // 4) % 4]
        if h == a:
            continue
        hg, ag = int(rng.poisson(1.4)), int(rng.poisson(1.1))
        rows.append({"date": d + pd.Timedelta(days=7 * k), "league": "Championship", "home_team": h,
                     "away_team": a, "home_goals": hg, "away_goals": ag,
                     "ht_home_goals": min(hg, 1), "ht_away_goals": min(ag, 1),
                     "home_shots": 10, "away_shots": 8, "home_sot": 4, "away_sot": 3,
                     "home_corners": 5, "away_corners": 4, "home_fouls": 11, "away_fouls": 12,
                     "odds_over25": 2.0, "odds_under25": 1.8})
    df = pd.DataFrame(rows)
    df["total_goals"] = df.home_goals + df.away_goals
    df["over25"] = (df.total_goals > 2.5).astype(float)
    df["btts"] = ((df.home_goals > 0) & (df.away_goals > 0)).astype(float)
    return df


def test_rest_days_resolve_bookmaker_names():
    hist = _hist()
    up = pd.DataFrame([{"date": hist.date.max() + pd.Timedelta(days=5), "league": "Championship",
                        "home_team": "Blackburn Rovers", "away_team": "Queens Park Rangers",
                        "odds_over25": 2.0, "odds_under25": 1.8}])
    f = build_upcoming_features(up, hist).iloc[0]
    assert pd.notna(f["home_rest_days"]) and pd.notna(f["away_rest_days"])


def test_ht_rates_resolve_bookmaker_names():
    hist = _hist()
    up = pd.DataFrame([{"date": hist.date.max() + pd.Timedelta(days=5), "league": "Championship",
                        "home_team": "Blackburn Rovers", "away_team": "Queens Park Rangers",
                        "odds_over25": 2.0, "odds_under25": 1.8}])
    f = build_upcoming_features(up, hist).iloc[0]
    for c in ("home_ht_over05_rate", "away_ht_over05_rate", "away_ht_scored_last5"):
        assert pd.notna(f[c]), c
