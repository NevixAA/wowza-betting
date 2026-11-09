"""The one-time clean-data switch (owner 2026-11-09, "Override for this switch.").

    python -m pytest tests/test_owner_clean_switch.py -q
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import model as M
from src import model_registry as reg
from src import promotion_gate as PG

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def tmpreg(tmp_path, monkeypatch):
    files = {mt: tmp_path / "models" / f"model_{mt}.pkl" for mt in reg.MODEL_FILES}
    monkeypatch.setattr(reg, "MODEL_FILES", files)
    monkeypatch.setattr(reg, "REGISTRY_DIR", tmp_path / "models" / "registry")
    monkeypatch.setattr(reg, "POINTER_FILE", tmp_path / "models" / "registry" / "cc.json")
    lin = tmp_path / "out" / "model_lineage"
    monkeypatch.setattr(reg, "LINEAGE_DIR", lin)
    monkeypatch.setattr(reg, "CHAMPION_HISTORY", lin / "champion_history.csv")
    monkeypatch.setattr(reg, "RETRAIN_BATCHES", lin / "retrain_batches.csv")
    monkeypatch.setattr(reg, "HIGH_WATER", lin / "high_water_mark.json")
    monkeypatch.setattr(PG.config, "MODELS_DIR", tmp_path / "models")
    reg._version_cache.clear()
    yield tmp_path
    reg._version_cache.clear()


def _payload(target="over25", recipe="std_v9_repaired", tracks=None):
    p = {"target": target, "recipe": recipe, "imputer": {"version": "1.1"}, "feature_cols": ["a", "b"],
         "models": {"logistic": object.__class__}, "served_holdout_metrics": {"n": 50, "log_loss": 0.68}}
    if tracks is not None:
        p["tracks"] = tracks
    return p


def _raw(p):
    return pickle.dumps(p)


def _record(tmp_path, arts, oid="clean-data-switch-2026-11-09"):
    f = tmp_path / "overrides.json"
    f.write_text(json.dumps({"overrides": [{
        "id": oid, "owner_quote": "Override for this switch.", "approved_by": "Nevo (owner)",
        "approved_on": "2026-11-09", "scope": "the clean-data switch only",
        "rule_overridden": "PG-1.0 prospective period",
        "artifacts": {mt: {"sha256": reg.sha256_bytes(r)} for mt, r in arts.items()}}]}), encoding="utf-8")
    return f


def test_installs_the_listed_artifact_and_records_the_owner(tmpreg):
    raw = _raw(_payload())
    rec = _record(tmpreg, {"standard": raw})
    m = PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)
    assert reg.MODEL_FILES["standard"].read_bytes() == raw
    ptr = reg.read_pointer()["standard"]
    assert ptr["model_version"] == m["model_version"] and ptr["promotion_source"] == PG.OWNER_CLEAN_SWITCH
    assert m["owner_approval"]["owner_quote"] == "Override for this switch."
    d = pd.read_csv(PG.decisions_file(), dtype=str, keep_default_na=False)
    row = d[d["stage"] == PG.OWNER_CLEAN_SWITCH].iloc[-1]
    assert row["verdict"] == PG.PROMOTE and row["promoted"].lower() == "true"
    assert "[clean-data-switch-2026-11-09]" in row["reason"] and "Override for this switch." in row["reason"]


def test_refuses_an_artifact_the_record_does_not_name(tmpreg):
    rec = _record(tmpreg, {"standard": _raw(_payload())})
    other = _raw({**_payload(), "feature_cols": ["a", "b", "c"]})
    with pytest.raises(PermissionError, match="does not cover"):
        PG.owner_clean_switch("standard", other, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)
    assert not reg.MODEL_FILES["standard"].exists()


def test_is_one_time(tmpreg):
    raw = _raw(_payload())
    rec = _record(tmpreg, {"standard": raw})
    m = PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)
    with pytest.raises(PermissionError, match="one-time"):
        PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0, supersede=m["model_version"])


def test_unknown_override_and_missing_fields_refuse(tmpreg):
    raw = _raw(_payload())
    rec = _record(tmpreg, {"standard": raw})
    with pytest.raises(PermissionError, match="no single owner override"):
        PG.owner_clean_switch("standard", raw, override_id="nope", overrides_path=rec, now=T0)
    body = json.loads(rec.read_text(encoding="utf-8"))
    body["overrides"][0]["owner_quote"] = ""
    rec.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(PermissionError, match="owner_quote"):
        PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)


def test_existing_champion_needs_an_exact_supersede(tmpreg):
    raw = _raw(_payload())
    rec = _record(tmpreg, {"standard": raw})
    ptr = {"standard": {"model_version": "standard_20261018T080423Z_64313878"}}
    reg._atomic_write_json(reg.POINTER_FILE, ptr)
    with pytest.raises(PermissionError, match="supersede"):
        PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)
    PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                          overrides_path=rec, now=T0, supersede="standard_20261018T080423Z_64313878")


@pytest.mark.parametrize("bad", [
    {"recipe": "v9"}, {"imputer": None}, {"target": "btts"}, {"feature_cols": []},
])
def test_refuses_a_payload_serving_cannot_read_cleanly(tmpreg, bad):
    raw = _raw({**_payload(), **bad})
    rec = _record(tmpreg, {"standard": raw})
    with pytest.raises(ValueError):
        PG.owner_clean_switch("standard", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)


def test_side_labels_need_both_tracks(tmpreg):
    nf = _payload(target="btts", recipe="side_v9_repaired")
    good = _raw(_payload(target="btts", recipe="side_v9_repaired", tracks={"new_format": nf}))
    flat = _raw(_payload(target="btts", recipe="side_v9_repaired"))
    rec = _record(tmpreg, {"btts": good})
    PG.owner_clean_switch("btts", good, override_id="clean-data-switch-2026-11-09", overrides_path=rec, now=T0)
    rec2 = _record(tmpreg, {"btts": flat}, oid="other")
    with pytest.raises(ValueError, match="track-split"):
        PG.owner_clean_switch("btts", flat, override_id="other", overrides_path=rec2, now=T0)


def test_frozen_types_are_not_installable_here(tmpreg):
    raw = _raw(_payload())
    rec = _record(tmpreg, {"standard_repaired": raw})
    with pytest.raises(PermissionError, match="not a model type"):
        PG.owner_clean_switch("standard_repaired", raw, override_id="clean-data-switch-2026-11-09",
                              overrides_path=rec, now=T0)


def test_archive_record_keeps_the_pointer(tmpreg):
    ptr = {"standard_repaired": {"model_version": "standard_repaired_x"}}
    reg._atomic_write_json(reg.POINTER_FILE, ptr)
    PG.record_archive("standard_repaired", "owner 2026-11-09: stops serving")
    assert reg.read_pointer()["standard_repaired"]["model_version"] == "standard_repaired_x"
    d = pd.read_csv(PG.decisions_file(), dtype=str, keep_default_na=False)
    assert d.iloc[-1]["stage"] == "ARCHIVE" and d.iloc[-1]["promoted"].lower() == "false"


def test_track_split_payload_routes_each_row_to_its_own_track():
    """The assembled side payload is the train_by_track serving format: predict_proba sends every
    Standard row to the top-level model and every New-Format row to tracks['new_format']."""
    import config
    calls = []

    class _Const:
        def __init__(self, v):
            self.v = v

        def predict_proba(self, X):
            calls.append(self.v)
            return np.column_stack([1 - np.full(len(X), self.v), np.full(len(X), self.v)])

    def _p(v):
        return {"target": "btts", "feature_cols": ["a"], "models": {"m": _Const(v)}}   # routing only
    pay = {**_p(0.2), "tracks": {"new_format": _p(0.8)}}
    std, nf = sorted(config.STANDARD_FORMAT_LEAGUES)[0], sorted(config.NEW_FORMAT_LEAGUES)[0]
    df = pd.DataFrame({"league": [std, nf, std], "a": [1.0, 2.0, 3.0]})
    p = M.predict_proba(df, payload=pay).to_numpy()
    assert np.allclose(p, [0.2, 0.8, 0.2])
