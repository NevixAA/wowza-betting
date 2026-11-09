"""Train/serve parity fixes for the props feature builder (owner 2026-11-09, "fix first").
Synthetic data only; the real-data parity is scripts/props_train_serve_parity.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402


def _table():
    d = pd.to_datetime(["2026-10-01", "2026-10-08", "2026-10-15", "2026-10-22", "2026-10-25", "2026-10-29"])
    t = pd.DataFrame({"_tkey": "T1", "date": d, "league": "Championship", "fixture_id": range(6),
                      "goals_scored": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    t["_day"] = t["date"].dt.normalize()
    for src in list(FE._TEAM_ROLL) + list(FE._LEAGUE_NORMS.values()):
        if src not in t:
            t[src] = np.nan
    return t


def test_team_state_uses_only_days_before_the_fixture(monkeypatch):
    monkeypatch.setattr(config, "MIN_OPP_MATCHES", 1)
    t = _table()
    fdate = pd.Timestamp("2026-10-25")                     # a same-day result and a later one exist
    got = FE._team_now(t, 3, fdate)["team_goals_pg_roll"]
    assert got == pytest.approx(np.mean([2.0, 3.0, 4.0]))  # 10-08, 10-15, 10-22; never 10-25 / 10-29
    assert FE._team_now(t, 3)["team_goals_pg_roll"] == pytest.approx(5.0)   # no date: every row


def test_league_rows_follow_the_training_string_and_find_continental_cups():
    t = pd.DataFrame({"league": ["Championship", "CONMEBOL Libertadores", "League One"], "x": [1, 2, 3]})
    assert FE._league_rows(t, "Championship")["x"].tolist() == [1]
    assert FE._league_rows(t, "CONMEBOL Libertadores")["x"].tolist() == [2]   # no registry name
    assert FE._same_league(t["league"], "CONMEBOL Libertadores").sum() == 0      # the old lookup: nothing
    assert FE._league_rows(t, "No Such League").empty


def test_league_now_never_counts_the_match_day(monkeypatch):
    monkeypatch.setattr(config, "MIN_LEAGUE_MATCHES", 1)
    t = _table()
    out = FE._league_now(t, "Championship", pd.Timestamp("2026-10-25"))
    assert out["lg_goals_per_team"] == pytest.approx(np.mean([1.0, 2.0, 3.0, 4.0]))


def _ref_history():
    rows = []
    for i, (day, ref, y) in enumerate([("2026-10-01", "M. Oliver", 4), ("2026-10-08", "M. Oliver", 6),
                                      ("2026-10-15", "A. Taylor", 2), ("2026-10-25", "M. Oliver", 9)]):
        for k in range(2):
            rows.append({"fixture_id": i, "date": day, "league": "Premier League", "referee": ref,
                         "player_id": 100 * i + k + 1, "minutes": 90, "yellow_cards": y / 2})
    return pd.DataFrame(rows)


def test_referee_profile_uses_only_fixtures_before_the_match(monkeypatch):
    monkeypatch.setattr(config, "MIN_REF_MATCHES", 1)
    monkeypatch.setattr(config, "MIN_LEAGUE_MATCHES", 1)
    h = _ref_history()
    before = FE.referee_profile_from_history(h, "M. Oliver", "Premier League", before="2026-10-25")
    assert before["n_games"] == 2 and before["yellows_per_game"] == pytest.approx(5.0)
    allheld = FE.referee_profile_from_history(h, "M. Oliver", "Premier League")
    assert allheld["n_games"] == 3                          # without a date: every held fixture


def test_referee_tables_use_the_training_rows():
    h = _ref_history()
    h.loc[0, "player_id"] = 0                               # no real id: training drops the row
    h.loc[1, "minutes"] = 5                                 # bench minutes: training drops the row
    f = FE._fixture_yellows(FE._training_rowset(h))
    assert 0 not in set(f["fixture_id"])                    # both rows of fixture 0 dropped


@pytest.mark.parametrize("p, hist_pos, want", [
    ({"lineup_position": "D", "position": "M"}, ["M"], ("D", FE.POSITION_FROM_LINEUP)),
    ({"position": "M"}, ["F"], ("M", FE.POSITION_FROM_LATEST)),
    ({}, ["F"], ("F", FE.POSITION_FROM_LATEST)),
    ({"lineup_position": "Sub"}, [None], ("", FE.POSITION_NONE)),
    ({}, [], ("", FE.POSITION_NONE)),
])
def test_upcoming_position_has_a_real_source(p, hist_pos, want):
    phist = pd.DataFrame({"position": hist_pos}) if hist_pos else pd.DataFrame({"position": []})
    assert FE._upcoming_position(p, phist) == want


def test_serving_code_carries_the_fixture_date_and_the_position_source():
    src = (ROOT / "player_model" / "feature_engineering.py").read_text(encoding="utf-8")
    assert "_opponent_state(ctx, okey[0], league, n, fdate)" in src
    assert "_league_now(ctx[\"table\"], league, fdate)" in src
    assert "team_match_table(_training_rowset(h))" in src
    assert 'r["position_source"] = _src' in src
    assert 'r["pos_forward"] = r["pos_midfielder"] = r["pos_defender"] = NAN' in src
    pipe = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    assert "referee_profile_from_history(history, referee, league,\n" in pipe.replace("\r\n", "\n")
    pred = (ROOT / "player_model" / "predict.py").read_text(encoding="utf-8")
    assert '_rec["lineup_position"] = _lp' in pred
