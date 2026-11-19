"""Serving league aggregates read the rows training read (2026-11-15).

Every trainer (pipeline.mode_train, the Phase 2 harness) drops the COVID seasons before
build_features, so the trained league_avg_goals (-> attack/defence strengths) and the league
away-goal average (-> home_advantage / away_home_adv_factor) never saw them. Serving
(build_upcoming_features) averaged the FULL history: up to 0.107 goals per game apart on the
committed data. Here a league plays 10 goals a game in its "COVID" weeks: on the old serving path
the served strengths differ from the trained ones; on the fix they are equal.

    python -m pytest tests/test_serve_covid_league_aggregates.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import feature_engineering as FE

LEAGUE = "Serie B"
CUT = pd.Timestamp("2025-09-15")          # rows before this date are the synthetic COVID season
COLS = ["league_avg_goals", "home_attack_str", "away_attack_str", "home_defense_str",
        "away_defense_str", "home_advantage", "away_home_adv_factor"]


def _m(date, home, away, hg, ag):
    return {"date": pd.Timestamp(date), "league": LEAGUE, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "total_goals": hg + ag, "over25": float(hg + ag > 2.5),
            "home_shots": 10, "away_shots": 8, "home_sot": 5, "away_sot": 4, "home_corners": 5,
            "away_corners": 4, "home_fouls": 11, "away_fouls": 13, "ht_home_goals": 0, "ht_away_goals": 0,
            "odds_over25": 1.9, "odds_under25": 1.95}


def _history() -> pd.DataFrame:
    rows, d0 = [], pd.Timestamp("2025-08-02")
    for i in range(16):
        d = d0 + pd.Timedelta(days=7 * i)
        covid = d < CUT
        rows.append(_m(d, "A", f"X{i}", 6 if covid else 2, 4 if covid else 1))
        rows.append(_m(d, f"Y{i}", "B", 5 if covid else 1, 5 if covid else 1))
        rows.append(_m(d, "B", f"Z{i}", 6 if covid else 1, 4 if covid else 0))
    return pd.DataFrame(rows)


@pytest.fixture()
def covid(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_COVID_SEASONS", True)
    monkeypatch.setattr(config, "covid_season_mask",
                        lambda df, *a, **k: pd.Series(pd.to_datetime(df["date"]).to_numpy() < np.datetime64(CUT),
                                                      index=df.index))


def _trained_and_served(hist):
    nxt = _m(hist["date"].max() + pd.Timedelta(days=7), "A", "B", 1, 1)
    trained_rows = hist[~config.covid_season_mask(hist).to_numpy()]      # what every trainer keeps
    tr = FE.build_features(pd.concat([trained_rows, pd.DataFrame([nxt])], ignore_index=True))
    tr = tr[tr["date"] == nxt["date"]].iloc[0]
    up = pd.DataFrame([{k: nxt[k] for k in ("date", "league", "home_team", "away_team",
                                            "odds_over25", "odds_under25")}])
    sv = FE.build_upcoming_features(up, hist).iloc[0]                   # serving reads the FULL history
    return tr, sv


def test_served_league_aggregates_equal_the_trained_ones(covid):
    tr, sv = _trained_and_served(_history())
    for c in COLS:
        assert np.isfinite(tr[c]), c
        assert sv[c] == pytest.approx(tr[c], abs=1e-12), c
    assert sv["league_avg_goals"] == pytest.approx(2.0)                  # (3 + 2 + 1) / 3, no COVID 10s


def test_the_old_full_history_average_is_what_differed(covid):
    hist = _history()
    full = hist.groupby("league")["total_goals"].mean()[LEAGUE]
    kept = FE.league_aggregate_history(hist).groupby("league")["total_goals"].mean()[LEAGUE]
    assert full != pytest.approx(kept)                                    # the skew the fix removes
    tr, _sv = _trained_and_served(hist)
    assert tr["league_avg_goals"] == pytest.approx(kept)


def test_without_the_covid_rule_serving_reads_every_row(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_COVID_SEASONS", False)
    hist = _history()
    assert len(FE.league_aggregate_history(hist)) == len(hist)


def test_parity_check_builds_its_training_side_from_the_trainers_rows(covid, monkeypatch):
    import socket
    # the parity script blocks the network on import; keep that inside this test
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    import scripts.team_train_serve_parity as parity
    hist = _history()
    assert len(parity.training_rows(hist)) == int((hist["date"] >= CUT).sum())
