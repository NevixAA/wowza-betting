"""One-off prev_season_* / shot_accuracy_l10 rebuild of the stored player_history (2026-11-24).

    python -m pytest tests/test_history_rebuild.py
"""
import numpy as np
import pandas as pd
import pytest

from player_model import history_rebuild as HR


def _stored():
    return pd.DataFrame({
        "fixture_id": [1, 2, 3, 4], "player_id": [7, 7, 8, 8],
        "goals": [1.0, 0.0, np.nan, 2.0],
        "prev_season_goals_pg": pd.Series([np.nan, np.nan, np.nan, np.nan], dtype="float32"),
        "shot_accuracy_l10": pd.Series([0.25, np.nan, np.nan, np.nan], dtype="float32"),
    })


def _built():
    # row (3, 8) is not emitted by the build (e.g. first appearance); order differs from stored
    return pd.DataFrame({"fixture_id": [4, 2, 1], "player_id": [8, 7, 7],
                         "prev_season_goals_pg": [0.4, 0.5, np.nan],
                         "shot_accuracy_l10": [0.3, 0.2, 0.9],
                         "goals": [99.0, 99.0, 99.0]})


def test_fills_only_empty_cells_and_keeps_stored_values():
    out, rep = HR.rebuild(_stored(), _built())
    assert out["prev_season_goals_pg"].tolist()[1] == pytest.approx(0.5)
    assert out["prev_season_goals_pg"].tolist()[3] == pytest.approx(0.4)
    assert np.isnan(out["prev_season_goals_pg"].iloc[0])           # rebuilt NaN stays missing
    assert np.isnan(out["prev_season_goals_pg"].iloc[2])           # not emitted: stays missing
    assert out["shot_accuracy_l10"].iloc[0] == pytest.approx(0.25)  # keep-first, 0.9 not written
    c = rep["columns"]["shot_accuracy_l10"]
    assert c["stored_kept_disagreeing"] == 1 and c["cells_filled"] == 2 and c["stored_non_null"] == 1
    assert rep["rows_not_emitted_by_build"] == 1


def test_no_other_column_changes():
    s = _stored()
    out, _ = HR.rebuild(s, _built())
    pd.testing.assert_series_equal(out["goals"], s["goals"])       # the build's goals=99 never land
    assert list(out.columns)[:len(s.columns)] == list(s.columns)          # only rebuild columns added
    assert set(out.columns) - set(s.columns) <= set(HR.COLUMNS)
    assert out["prev_season_goals_pg"].dtype == np.float32


def test_verify_refuses_shrink_changed_columns_and_rewrites():
    s = _stored()
    with pytest.raises(ValueError, match="no-shrink"):
        HR.verify(s, s.iloc[:3])
    bad = s.copy(); bad["goals"] = 0.0
    with pytest.raises(ValueError, match="goals"):
        HR.verify(s, bad)
    bad = s.copy(); bad.loc[0, "shot_accuracy_l10"] = 0.5
    with pytest.raises(ValueError, match="keep-first"):
        HR.verify(s, bad)


def test_duplicate_build_keys_refused():
    b = pd.concat([_built(), _built().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        HR.rebuild(_stored(), b)
