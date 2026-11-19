"""The provider-proven home/away side (owner 2026-11-09, P7 item 6) is what props training and serving
read; the stored side and opponent stay beside. Synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import history_refetch as HR  # noqa: E402
from player_model.player_stat_nulls import PARSED_WITH_RULE, SOURCE_COL  # noqa: E402


def _history():
    return pd.DataFrame([
        # 'Pachuca' vs 'CF Pachuca': stored as away, the provider proves home
        {"fixture_id": 1, "player_id": 10, "team": "Pachuca", "home_team": "CF Pachuca", "away_team": "Puebla",
         "is_home": 0, "opponent": "CF Pachuca", "goals": 0},
        # a correct row, also re-fetched
        {"fixture_id": 1, "player_id": 20, "team": "Puebla", "home_team": "CF Pachuca", "away_team": "Puebla",
         "is_home": 0, "opponent": "CF Pachuca", "goals": 1},
        # a row parsed with the rule: its counts stay, its side is still corrected (identity, not a count)
        {"fixture_id": 2, "player_id": 10, "team": "Pachuca", "home_team": "CF Pachuca", "away_team": "Leon",
         "is_home": 0, "opponent": "CF Pachuca", "goals": 2, SOURCE_COL: PARSED_WITH_RULE},
        # not re-fetched at all
        {"fixture_id": 3, "player_id": 30, "team": "Leon", "home_team": "Leon", "away_team": "Toluca",
         "is_home": 1, "opponent": "Toluca", "goals": 0},
    ])


def _refetch():
    return pd.DataFrame([
        {"fixture_id": 1, "player_id": 10, "is_home": 1, "is_home_orig": 0, "is_home_corrected": True, "goals": 0},
        {"fixture_id": 1, "player_id": 20, "is_home": 0, "is_home_orig": 0, "is_home_corrected": False, "goals": 1},
        {"fixture_id": 2, "player_id": 10, "is_home": 1, "is_home_orig": 0, "is_home_corrected": True, "goals": 0},
    ])


def test_proven_side_replaces_the_stored_one_and_the_original_stays_beside():
    out = HR.with_refetched(_history(), _refetch()).set_index(["fixture_id", "player_id"])
    assert out.loc[(1, 10), "is_home"] == 1 and out.loc[(1, 10), "is_home_orig"] == 0
    assert out.loc[(1, 10), "opponent"] == "Puebla" and out.loc[(1, 10), "opponent_orig"] == "CF Pachuca"
    assert out.loc[(1, 10), "is_home_source"] == HR.SIDE_PROVIDER_PROVEN
    assert out.loc[(2, 10), "is_home"] == 1 and out.loc[(2, 10), "goals"] == 2      # rule row: counts kept
    for k in ((1, 20), (3, 30)):
        assert out.loc[k, "is_home"] == out.loc[k, "is_home_orig"]
        assert out.loc[k, "is_home_source"] == HR.SIDE_FROM_HISTORY
    assert out.loc[(3, 30), "opponent"] == "Toluca"


def test_the_input_is_not_rewritten_and_no_file_means_no_change():
    h = _history()
    HR.with_refetched(h, _refetch())
    assert h["is_home"].tolist() == [0, 0, 0, 1]
    none = HR.with_refetched(_history(), pd.DataFrame())
    assert none["is_home"].tolist() == [0, 0, 0, 1]


def test_features_read_the_corrected_side():
    """build_features derives the opponent (and so the team keys) from is_home."""
    from player_model import feature_engineering as FE
    src = (ROOT / "player_model" / "feature_engineering.py").read_text(encoding="utf-8")
    assert 'df["opponent"] = df["away_team"].where(df["is_home"].astype(bool), df["home_team"])' in src
    harness = (ROOT / "player_model" / "props_harness.py").read_text(encoding="utf-8")
    assert "raw = with_refetched(raw, refetch)" in harness            # training and serving: source_frame
    assert FE.build_upcoming_features is not None
