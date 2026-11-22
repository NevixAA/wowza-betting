"""Owner P0 item 6 (2026-11-22): a real ROI needs real entry odds, bookmaker, capture time, market,
line and settlement; a synthetic price (base-rate implied odds, a fill constant) never enters it."""
from __future__ import annotations

import pandas as pd
import pytest

from src.integrity import synthetic_roi as S
from src.integrity.price_provenance import label_side_prices

ROW = {"entry_odds": 2.10, "bookmaker": "BookA", "capture_time": "2026-11-20T12:00:00Z",
       "kickoff_utc": "2026-11-20T15:00:00Z", "market": "player_sot", "line": 0.5, "settlement": "WIN"}


def test_complete_real_row_passes():
    S.assert_real_roi_rows(pd.DataFrame([ROW]))


@pytest.mark.parametrize("field", S.REAL_ROI_FIELDS)
def test_each_missing_field_is_refused(field):
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{k: v for k, v in ROW.items() if k != field}]))
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, field: None}]))


def test_unsettled_and_post_kickoff_capture_refused():
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, "settlement": "PENDING"}]))
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, "capture_time": "2026-11-20T15:00:00Z"}]))
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, "entry_odds": 1.0}]))


@pytest.mark.parametrize("col,val", [("price_provenance", "SYNTHETIC_PRICE_DIAGNOSTIC_ONLY"),
                                     ("profitability_label", "INVALID_FOR_PROFITABILITY"),
                                     ("roi_status", "INVALID_FOR_REAL_BETTING_ROI"),
                                     ("price_synthetic", True)])
def test_synthetic_labels_refused(col, val):
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, col: val}]))


def test_side_fill_constant_labelled_synthetic_is_refused():
    d = label_side_prices(pd.DataFrame([{"odds_btts": 1.85}]), "btts")
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([{**ROW, "price_provenance": d["price_provenance"].iloc[0]}]))


def test_the_quarantined_calibration_rows_can_never_be_a_real_roi():
    d = S.load_for_audit("output/player_props_calibration.json")
    frame = S.calibration_rows_as_roi_frame(d)
    assert len(frame) >= 7
    probs = S.real_roi_problems(frame)
    assert (probs != "").all()
    assert probs.str.contains("no bookmaker").all() and probs.str.contains("no capture_time").all()
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(frame)


def test_mixed_frame_is_refused_as_a_whole():
    with pytest.raises(S.SyntheticROIError):
        S.assert_real_roi_rows(pd.DataFrame([ROW, {**ROW, "bookmaker": ""}]))
