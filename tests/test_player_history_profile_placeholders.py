"""D4 (owner 2026-11-23): stored profile placeholders -> missing, old values kept in the sidecar; a real
value equal to a placeholder number is never touched."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import null_player_history_profile_placeholders as N  # noqa: E402


def _df():
    return pd.DataFrame({
        "fixture_id": [1, 2, 3], "player_id": [10, 11, 12], "date": ["2026-01-01"] * 3, "team": ["A"] * 3,
        "age": [25.0, 24.37, np.nan], "height_cm": [180.0, 180.0, 175.0], "weight_kg": [75.0, 75.0, 70.0],
        "age_peak_delta": [0.0, 2.63, np.nan], "opp_def_mean_height": [180.0, 183.2, 181.0],
        "height_diff_vs_opp_def": [0.0, -3.2, -6.0], "height_advantage_score": [0.0, -0.32, -0.6],
        "height_aerial_interaction": [0.25, 0.3, 0.1], "minutes": [90, 80, 70]})


def test_only_the_four_value_signature_is_rewritten():
    new, side, counts = N.null_placeholders(_df())
    assert counts["placeholder_rows"] == 1
    for c in ("age", "height_cm", "weight_kg", "age_peak_delta", *N.COMPOSITES):
        assert pd.isna(new.loc[0, c])
    assert new.loc[1, "height_cm"] == 180.0 and new.loc[1, "weight_kg"] == 75.0     # real 180 cm / 75 kg kept
    assert counts["height_cm_equal_to_placeholder_kept_real"] == 1
    pd.testing.assert_frame_equal(new.loc[1:], _df().loc[1:], check_dtype=False)
    assert (new["minutes"] == _df()["minutes"]).all()                                # raw stats untouched


def test_sidecar_keeps_keys_and_old_values():
    _, side, _ = N.null_placeholders(_df())
    assert side[["fixture_id", "player_id"]].values.tolist() == [[1, 10]]
    assert side.loc[0, "old_age"] == 25.0 and side.loc[0, "old_height_cm"] == 180.0
    assert side.loc[0, "old_height_aerial_interaction"] == 0.25
