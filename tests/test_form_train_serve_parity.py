"""Train/serve parity of the rolling-form features (estate scan P0-1, 2026-11-02).

Serving built "last 5" as list(home.tail(5)) + list(away.tail(5)) and kept the last 5 of that list,
i.e. the team's last 5 AWAY games once it had played 5 away, while training uses the last 5 matches
in date order. Here a team scores 3 in every home game and 0 in every away game, alternating:
training gives its next match 1.2, the old serving code gave 0.0.

The test compares serving with training for the SAME next match: training is run with that match
appended as a played row, serving with it as an upcoming fixture.

    python -m pytest tests/test_form_train_serve_parity.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as FE

LEAGUE = "Sweden Allsvenskan"
FORM = ["home_scored_last5", "home_conceded_last5", "home_over25_last5", "home_shots_last5", "home_sot_last5",
        "home_ht_scored_last5", "home_ht_conceded_last5", "home_ht_over05_rate", "home_ht_over15_rate",
        "home_corners_pg_roll", "home_fouls_pg_roll",
        "away_scored_last5", "away_conceded_last5", "away_over25_last5", "away_shots_last5", "away_sot_last5",
        "away_ht_scored_last5", "away_ht_conceded_last5", "away_ht_over05_rate", "away_ht_over15_rate",
        "away_corners_pg_roll", "away_fouls_pg_roll"]


def _match(date, home, away, hg, ag, *, hs=10, as_=8, hht=None, aht=None):
    return {"date": pd.Timestamp(date), "league": LEAGUE, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "total_goals": hg + ag, "over25": int(hg + ag > 2.5),
            "home_shots": hs, "away_shots": as_, "home_sot": hs // 2, "away_sot": as_ // 2,
            "home_corners": hs // 2, "away_corners": as_ // 2, "home_fouls": 11, "away_fouls": 13,
            "ht_home_goals": hht if hht is not None else min(hg, 1), "ht_away_goals": aht if aht is not None else 0,
            "odds_over25": 1.9, "odds_under25": 1.95}


def _history() -> pd.DataFrame:
    rows, d0 = [], pd.Timestamp("2026-05-02")
    for i in range(12):                      # A: home 3-1 (12 shots), away 1-0 loss (4 shots), alternating
        d = d0 + pd.Timedelta(days=7 * i)
        if i % 2 == 0:
            rows.append(_match(d, "A", f"X{i}", 3, 1, hs=12, as_=6, hht=2, aht=0))
        else:
            rows.append(_match(d, f"Y{i}", "A", 1, 0, hs=9, as_=4, hht=0, aht=0))
        rows.append(_match(d, "B", f"Z{i}", 1, 1))      # B: steady 1-1 at home
    return pd.DataFrame(rows)


def _training_row(hist: pd.DataFrame, nxt: dict) -> pd.Series:
    full = pd.concat([hist, pd.DataFrame([nxt])], ignore_index=True)
    out = FE.build_features(full)
    hit = out[(out["home_team"] == nxt["home_team"]) & (out["date"] == nxt["date"])]
    assert len(hit) == 1
    return hit.iloc[0]


def _serving_row(hist: pd.DataFrame, nxt: dict) -> pd.Series:
    up = pd.DataFrame([{k: nxt[k] for k in ("date", "league", "home_team", "away_team", "odds_over25", "odds_under25")}])
    out = FE.build_upcoming_features(up, hist)
    assert len(out) == 1
    return out.iloc[0]


@pytest.mark.parametrize("home,away", [("A", "B"), ("B", "A")])
def test_serving_form_equals_training_form_for_the_same_next_match(home, away):
    hist = _history()
    nxt = _match("2026-07-25", home, away, 0, 0)
    tr, sv = _training_row(hist, nxt), _serving_row(hist, nxt)
    for col in FORM:
        assert col in sv.index, col
        a, b = tr[col], sv[col]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b, abs=1e-12), f"{col}: training {a} vs serving {b}"


def test_last5_is_chronological_not_last_five_away_games():
    hist = _history()
    sv = _serving_row(hist, _match("2026-07-25", "A", "B", 0, 0))
    # A's last 5 matches in date order: away 0, home 3, away 0, home 3, away 0 -> 6 / 5
    assert sv["home_scored_last5"] == pytest.approx(1.2)
    assert not np.isclose(sv["home_scored_last5"], 0.0)


def test_no_history_is_missing_not_zero():
    hist = _history()
    sv = _serving_row(hist, _match("2026-07-25", "NEWCLUB", "B", 0, 0))
    assert pd.isna(sv["home_scored_last5"]) and pd.isna(sv["home_ht_over05_rate"])
