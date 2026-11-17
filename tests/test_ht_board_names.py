"""fix13: HT features meet board club names through the proven-identity resolver.

src.ht_features.attach_ht_features keys its team accumulators by (league, name). Until fix13 the
board's spelling was used as-is, so 'Wigan Athletic' (board) never met 'Wigan' (history) and the
fixture got only the league prior (ht_home_n = 0): on the committed board 38 of 174 fixtures had
team HT history on both sides. Board names now go through src.team_names.resolve_identity with the
applied identity map; an unproven name is left alone (no team HT history), counted and logged.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src import ht_features as HT


def _hist():
    """Two seasons of a small League One: Wigan and Stockport have real HT history."""
    clubs = ["Wigan", "Stockport", "Bolton", "Reading"]
    rows, d = [], pd.Timestamp("2025-08-01")
    for k in range(6):
        for i, h in enumerate(clubs):
            for j, a in enumerate(clubs):
                if h == a:
                    continue
                rows.append({"date": d, "league": "League One", "home_team": h, "away_team": a,
                             "home_goals": (i + k) % 3, "away_goals": (j + k) % 2,
                             "ht_home_goals": (i + k) % 2, "ht_away_goals": (j * k) % 2})
                d += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


AMAP = {"League One": {"Stockport County": "Stockport"}}


def _board(home, away, league="League One"):
    return pd.DataFrame([{"date": pd.Timestamp("2026-11-08"), "league": league,
                          "home_team": home, "away_team": away}])


def test_board_spelling_reaches_the_history_club():
    h = _hist()
    exact = HT.attach_ht_features(h, _board("Wigan", "Stockport"))
    board = HT.attach_ht_features(h, _board("Wigan Athletic", "Stockport County FC"), identity_aliases=AMAP)
    # same club, same history -> the very same HT features, only the caller's names differ
    for c in HT.HT_FEATURE_COLS:
        assert np.allclose(exact[c].to_numpy(float), board[c].to_numpy(float), equal_nan=True), c
    assert board.loc[0, "ht_home_n"] > 0 and board.loc[0, "ht_away_n"] > 0
    assert list(board[["home_team", "away_team"]].iloc[0]) == ["Wigan Athletic", "Stockport County FC"]
    rep = HT.LAST_NAME_REPORT
    assert rep["resolved"] == 2 and rep["unresolved"] == 0
    assert rep["resolved_names"] == {"League One|Wigan Athletic": "Wigan",
                                     "League One|Stockport County FC": "Stockport"}


def test_the_identity_map_is_what_proves_a_renamed_club():
    h = _hist()
    no_map = HT.attach_ht_features(h, _board("Bolton", "Stockport County FC"))
    assert no_map.loc[0, "ht_away_n"] == 0                       # resolve() alone cannot prove it
    assert HT.LAST_NAME_REPORT["unresolved_names"] == ["League One|Stockport County FC"]
    with_map = HT.attach_ht_features(h, _board("Bolton", "Stockport County FC"), identity_aliases=AMAP)
    assert with_map.loc[0, "ht_away_n"] > 0


def test_an_unproven_name_stays_without_team_ht_history_counted_and_logged(caplog):
    h = _hist()
    with caplog.at_level(logging.WARNING, logger="src.ht_features"):
        out = HT.attach_ht_features(h, _board("Wigan Athletic", "Platense"), identity_aliases=AMAP)
    assert out.loc[0, "ht_home_n"] > 0
    assert out.loc[0, "ht_away_n"] == 0                          # never guessed onto another club
    assert out.loc[0, "away_team"] == "Platense"
    rep = HT.LAST_NAME_REPORT
    assert rep["unresolved"] == 1 and rep["unresolved_names"] == ["League One|Platense"]
    assert "Platense" in caplog.text and "not proven" in caplog.text


def test_a_name_is_resolved_only_within_its_own_league():
    h = _hist()
    out = HT.attach_ht_features(h, _board("Wigan Athletic", "Stockport County FC", league="League Two"),
                                identity_aliases=AMAP)
    assert out.loc[0, "ht_home_n"] == 0 and out.loc[0, "ht_away_n"] == 0
    assert HT.LAST_NAME_REPORT["resolved"] == 0 and HT.LAST_NAME_REPORT["unresolved"] == 2


def test_rows_keep_the_callers_order_and_names():
    h = _hist()
    up = pd.concat([_board("Wigan Athletic", "Bolton"), _board("Reading", "Stockport County FC"),
                    _board("Bolton", "Wigan")], ignore_index=True)
    up.loc[0, "date"] = pd.Timestamp("2026-11-09")              # not in date order on purpose
    out = HT.attach_ht_features(h, up, identity_aliases=AMAP)
    assert out[["home_team", "away_team"]].astype(str).values.tolist() == \
        up[["home_team", "away_team"]].astype(str).values.tolist()
    assert (out["ht_home_n"] > 0).all() and (out["ht_away_n"] > 0).all()


def test_training_path_is_unchanged():
    h = _hist()
    pd.testing.assert_frame_equal(HT.attach_ht_features(h), HT.build_ht_features(h))


def test_predict_passes_the_applied_identity_map():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "predict.py").read_text(encoding="utf-8")
    assert 'attach_ht_features(historical, feat,\n' in src
    assert 'identity_aliases=getattr(_dl, "IDENTITY_ALIASES", None))' in src.split("attach_ht_features(historical, feat,")[1][:200]
