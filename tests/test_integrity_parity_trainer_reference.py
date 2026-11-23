"""D3 (owner 2026-11-23 "Yes, do all"): the train/serve parity monitor builds its TRAINING reference
exactly like the trainers (protocol.load_committed_frame + protocol.eligible_rows per payload part,
training_start..training_end), never the legacy base_history replay."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import integrity_feature_parity as FP  # noqa: E402


def test_reference_is_the_trainers_frame_not_the_legacy_replay():
    src = (ROOT / "scripts" / "integrity_feature_parity.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]                       # past the module docstring
    assert "load_committed_frame" in code and "eligible_rows" in code
    assert "base_history" not in code and "_enrich_with_api_shots" not in code
    assert "MODEL_FILE_STANDARD_REPAIRED" not in code and 'recipe="v9_repaired"' not in code


def test_payload_part_picks_the_serving_track():
    std = {"target": "btts", "feature_cols": ["a"], "tracks": {"new_format": {"target": "btts", "feature_cols": ["b"]}}}
    assert FP.payload_part(std, "standard")["feature_cols"] == ["a"]
    assert FP.payload_part(std, "new_format")["feature_cols"] == ["b"]
    assert FP.payload_part({"feature_cols": ["c"]}, "new_format")["feature_cols"] == ["c"]


def test_every_serving_payload_part_is_monitored():
    keys = set(FP._payload_specs())
    for t in ("standard", "new_format"):
        for m in ("OU25", "btts", "over15", "over35", "ht_over05", "ht_over15"):
            assert f"{t}|{m}" in keys
    # the model-health aggregator reads every key it needs from this file
    from src.integrity.model_health import PARITY_KEY
    assert set(PARITY_KEY.values()) <= keys


def test_training_rows_are_the_eligible_rows_inside_the_fit_window(monkeypatch):
    from src.retrain_harness import protocol as P
    feat = pd.DataFrame({"date": pd.to_datetime(["2016-06-30", "2016-08-05", "2020-01-01", "2026-11-01",
                                                 "2026-11-02", "2026-11-08"]), "x": range(6)})
    seen = {}

    def fake(f, cfg, track, market):
        seen["args"] = (track, market)
        return f
    monkeypatch.setattr(P, "eligible_rows", fake)
    cfg = {"markets": {"over25": {"target": "over25"}, "ht_over05": {"target": "ht_over05"}}}
    part = {"target": "ht_over05", "training_start": "2016-08-05", "training_end": "2026-11-01"}
    rows, market = FP.training_rows(feat, cfg, "standard", part)
    assert market == "ht_over05" and seen["args"] == ("standard", "ht_over05")
    assert rows["x"].tolist() == [1, 2, 3]              # before the start and the recipe's own tail excluded
