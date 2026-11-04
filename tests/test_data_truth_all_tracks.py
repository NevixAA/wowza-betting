"""Data truth rule 3 on EVERY model track (owner 2026-11-04).

Until 2026-11-04 rule 3 checked two columns on Standard fixtures only, so the new-format half-time
gap of 2026-11-02 (NF fixtures served with HT form ~99% missing) stayed GREEN. Now each served
(model_type, market) is checked against the feature list of the payload that scores it, plus the HT
form and rest-day inputs, and a gap turns the light AMBER (> 20%) or RED (> 50%).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integrity import data_truth as DT  # noqa: E402

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)
FEATS = ["home_scored_last5", "home_corners_pg_roll"]
ALL_INPUTS = FEATS + list(DT.HT_FORM_INPUTS) + list(DT.REST_INPUTS)


def _board(n_std=10, n_nf=10, nf_ht_missing=0, nf_corners_missing=0):
    rows = []
    for mt, n in (("standard", n_std), ("new_format", n_nf)):
        for i in range(n):
            r = {c: 1.0 for c in ALL_INPUTS}
            r.update(model_type=mt, odds_over25=1.9)
            if mt == "new_format" and i < nf_ht_missing:
                r.update({c: np.nan for c in DT.HT_FORM_INPUTS})
            if mt == "new_format" and i < nf_corners_missing:
                r["home_corners_pg_roll"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _features(nf=FEATS):
    out = {("standard", "O/U 2.5"): FEATS, ("new_format", "O/U 2.5"): nf}
    for m in ("BTTS", "Over 1.5", "Over 3.5"):
        out[("standard", m)] = FEATS
        out[("new_format", m)] = FEATS
    return out


def test_the_2026_11_02_nf_ht_gap_is_red_on_every_nf_track():
    t = DT.track_gaps(_board(nf_ht_missing=10), _features())
    for m in ("O/U 2.5", "BTTS", "Over 1.5", "Over 3.5"):
        assert t[f"new_format|{m}"]["status"] == DT.RED
        assert t[f"new_format|{m}"]["missing"]["home_ht_over05_rate"] == 100.0
        assert t[f"standard|{m}"]["status"] == DT.GREEN
    # the old rule looked at Standard rows only: the same board read GREEN there
    std = _board(nf_ht_missing=10).query("model_type == 'standard'")
    assert std["home_ht_over05_rate"].isna().mean() == 0


def test_amber_between_20_and_50_percent_and_green_at_or_below_20():
    assert DT.track_gaps(_board(nf_corners_missing=3), _features())["new_format|BTTS"]["status"] == DT.AMBER
    assert DT.track_gaps(_board(nf_corners_missing=2), _features())["new_format|BTTS"]["status"] == DT.GREEN
    assert DT.track_gaps(_board(nf_corners_missing=6), _features())["new_format|BTTS"]["status"] == DT.RED


def test_a_payload_input_the_board_does_not_carry_is_100_percent_missing():
    t = DT.track_gaps(_board(), _features(nf=FEATS + ["home_xg_last5"]))
    assert t["new_format|O/U 2.5"]["status"] == DT.RED
    assert t["new_format|O/U 2.5"]["missing"] == {"home_xg_last5": 100.0}


def test_unreadable_payload_is_never_green_and_unserved_track_is_noted():
    f = _features()
    f[("new_format", "BTTS")] = "UNREADABLE: FileNotFoundError"
    f[("new_format", "Over 3.5")] = None
    t = DT.track_gaps(_board(), f)
    assert t["new_format|BTTS"]["status"] == DT.AMBER and "not checked" in t["new_format|BTTS"]["note"]
    assert t["new_format|Over 3.5"]["status"] == DT.GREEN and "no model" in t["new_format|Over 3.5"]["note"]


def test_track_split_payload_reads_each_tracks_own_feature_list():
    pl = {"feature_cols": ["a", "b"], "tracks": {"new_format": {"feature_cols": ["c"]}}}
    assert DT._payload_cols(pl, "standard") == ["a", "b"]
    assert DT._payload_cols(pl, "new_format") == ["c"]
    assert DT._payload_cols({"feature_cols": ["a"], "tracks": {"new_format": None}}, "new_format") is None
    assert DT._payload_cols({"feature_cols": ["a"]}, "new_format") == ["a"]       # pooled legacy payload


def test_missing_rolls_the_worst_track_up_and_names_it(tmp_path, monkeypatch):
    _board(nf_ht_missing=10).to_csv(tmp_path / "predictions.csv", index=False)
    for f in DT.BANK_FILES:
        pd.DataFrame({"date": [pd.Timestamp("2026-11-02")]}).to_parquet(tmp_path / f)
    monkeypatch.setattr(DT, "OUT", tmp_path)
    c = DT.missing(NOW, features=_features())
    assert c["status"] == DT.RED and "new_format|O/U 2.5" in c["value"]
    assert set(c["tracks"]) == {f"{mt}|{m}" for mt in DT.MODEL_TYPES for m in ("O/U 2.5", "BTTS", "Over 1.5", "Over 3.5")}
    _board().to_csv(tmp_path / "predictions.csv", index=False)
    assert DT.missing(NOW, features=_features())["status"] == DT.GREEN


def test_committed_payloads_cover_every_served_track():
    f = DT.serving_features()
    want = {(mt, m) for mt in DT.MODEL_TYPES for m in ("O/U 2.5", "BTTS", "Over 1.5", "Over 3.5")}
    want |= {("standard", "HT Over 0.5"), ("standard", "HT Over 1.5")}
    assert set(f) == want
    for k, cols in f.items():
        assert cols is None or (isinstance(cols, list) and cols), k


def test_rule_3_stays_operational_and_the_ratchet_is_untouched():
    """Rule 3 reports (system health) but does not block a push; the ratchet keeps its meaning."""
    assert not any(g.startswith("Data truth 3") for g in DT.GATED)
    red3 = [{"check": "Data truth 3: no missing data", "status": DT.RED, "value": "x", "detail": ""}]
    assert not [f for f in DT.gate(None, red3) if "Data truth 3" in f]
    # the ceiling may still only go down
    ceiling = int(DT._ratchet()["max_known_defaults"])
    assert any("may only go down" in f for f in DT.gate(ceiling - 1, []))
