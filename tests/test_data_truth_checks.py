"""The four Data Truth rules turn RED on a violation (src/integrity/data_truth, 2026-11-01).

    python -m pytest tests/test_data_truth_checks.py -q
"""
from __future__ import annotations

import json

import pandas as pd

from src.integrity import data_truth as DT


def test_every_baseline_entry_has_a_reason():
    b = json.loads(DT.BASELINE.read_text(encoding="utf-8"))
    assert b["entries"]
    assert all(e.get("reason") and not e["reason"].startswith("TODO") for e in b["entries"])


def test_committed_code_has_no_default_outside_the_baseline():
    assert DT.defaults()["status"] == DT.GREEN, DT.defaults()["detail"]


def test_a_new_invented_default_turns_red(monkeypatch, tmp_path):
    f = tmp_path / "src" / "predict.py"
    f.parent.mkdir(parents=True)
    f.write_text('x = df["home_rest_days"].fillna(7.0)\n', encoding="utf-8")
    monkeypatch.setattr(DT, "ROOT", tmp_path)
    monkeypatch.setattr(DT, "SCAN_FILES", ["src/predict.py"])
    assert DT.defaults()["status"] == DT.RED


def test_an_unknown_league_name_turns_red(monkeypatch, tmp_path):
    pd.DataFrame({"league": ["Championship", "Denmark 1st Div"]}).to_csv(tmp_path / "predictions.csv", index=False)
    monkeypatch.setattr(DT, "OUT", tmp_path)
    monkeypatch.setattr(DT, "BANK_FILES", [])
    assert DT.naming()["status"] == DT.RED


def test_missing_serving_features_turn_red(monkeypatch, tmp_path):
    pd.DataFrame({"model_type": ["standard"] * 4, "odds_over25": [2.0] * 4,
                  "home_rest_days": [None, None, None, 5.0], "home_ht_over05_rate": [0.5] * 4}
                 ).to_csv(tmp_path / "predictions.csv", index=False)
    monkeypatch.setattr(DT, "OUT", tmp_path)
    monkeypatch.setattr(DT, "BANK_FILES", [])
    assert DT.missing()["status"] == DT.RED


def test_league_relations_check_runs_green_on_the_committed_banks():
    assert DT.leagues()["status"] == DT.GREEN, DT.leagues()["value"]


def test_system_health_carries_the_four_rules():
    import inspect
    from src import system_health as SH
    assert "data_truth" in inspect.getsource(SH.build_artifact)
