"""Hotfix on main (2026-11-04): an assists tip without a real bookmaker quote is never priced.

enrich_no_odds_markets used to invent market_odds = 1 / (base_rate x 1.10) and tier the tip on it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from player_model import predict as P


def _tips():
    return pd.DataFrame([
        {"market": "assists", "model_prob": 0.45, "market_odds": np.nan, "tier": "AVOID",
         "confidence": 1.0, "lazy_factors": "", "ges": 0.5, "position": "M", "data_source": ""},
        {"market": "assists", "model_prob": 0.45, "market_odds": 2.60, "tier": "VALUABLE",
         "confidence": 1.0, "lazy_factors": "", "ges": 0.5, "position": "M", "data_source": "oddsapi"},
        {"market": "sot", "model_prob": 0.55, "market_odds": np.nan, "tier": "AVOID",
         "confidence": 1.0, "lazy_factors": "", "ges": 0.5, "position": "F", "data_source": ""},
    ])


def test_unpriced_assists_tip_stays_unpriced_and_avoid(tmp_path, monkeypatch):
    # a calibration file that the OLD code turned into a synthetic price
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    (tmp_path / "player_props_calibration.json").write_text(json.dumps({"assists": {"base_rate": 0.20}}))
    out = P.enrich_no_odds_markets(_tips())
    assert pd.isna(out.loc[0, "market_odds"]) and out.loc[0, "tier"] == "AVOID"
    assert "ev" not in out.columns or pd.isna(out.loc[0, "ev"])


def test_real_quote_and_other_markets_are_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    before = _tips()
    out = P.enrich_no_odds_markets(before.copy())
    pd.testing.assert_frame_equal(out, before)


def test_empty_frame():
    assert P.enrich_no_odds_markets(pd.DataFrame()).empty
