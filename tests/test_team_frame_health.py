"""team_frame findings, health side (CLAUDE_PROMPT_2 P2, 2026-11-02). Each test FAILS on
data-truth-100 HEAD 84e06800:
  * the research lock does not watch REQUIRE_FORM_DATA / REQUIRE_FORM_DATA_UNTIL;
  * health checks read GREEN when their input is missing;
  * no data-truth check stops duplicate matches coming back.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import research_lock as L
from src import system_health as SH
from src.integrity import data_truth as DT

SNAP = json.loads(L.SNAPSHOT_PATH.read_text(encoding="utf-8"))
NOW = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)

YML = """
jobs:
  predict:
    steps:
      - env:
          REQUIRE_FORM_DATA: "0"
          REQUIRE_FORM_DATA_UNTIL: "2026-09-15"
          ODDS_API_KEY: ${{ secrets.X }}
"""
BETTING = ('_guard_off_until = os.getenv("REQUIRE_FORM_DATA_UNTIL", "2026-09-15")\n'
           'x = os.getenv("REQUIRE_FORM_DATA", "1").strip()\n')


# ── research lock watches the blind-fixture guard ───────────────────────────────────────────────
def test_form_guard_block_reads_deployed_values_and_code_defaults():
    b = L.form_guard_block(YML, BETTING)
    assert b == {"deployed": {"REQUIRE_FORM_DATA": "0", "REQUIRE_FORM_DATA_UNTIL": "2026-09-15"},
                 "code_default": {"REQUIRE_FORM_DATA": "1", "REQUIRE_FORM_DATA_UNTIL": "2026-09-15"}}
    assert L.form_guard_block("jobs: {}\n", "")["deployed"] == {"REQUIRE_FORM_DATA": None,
                                                                "REQUIRE_FORM_DATA_UNTIL": None}


def test_live_repo_form_guard_is_fingerprinted():
    fg = L.form_guard_block()
    assert fg["deployed"]["REQUIRE_FORM_DATA"] is not None          # predict.yml sets it
    assert fg["code_default"]["REQUIRE_FORM_DATA"] == "1"          # betting.py fails closed


def test_moving_the_form_guard_after_approval_is_a_production_change():
    approved = L.form_guard_block(YML, BETTING)
    snap = copy.deepcopy(SNAP)
    snap["fingerprint"]["global"]["form_guard"] = approved
    same = L.check(snapshot=snap, form_guard=approved)
    assert not any("form_guard" in c for c in same["changes"])
    assert same["unapproved_new_watched"] == []
    moved = L.form_guard_block(YML.replace('"2026-09-15"', '"2027-06-01"'), BETTING)
    r = L.check(snapshot=snap, form_guard=moved)
    assert r["protected_tracks_modified"] is True
    assert any("form_guard.deployed.REQUIRE_FORM_DATA_UNTIL" in c for c in r["changes"])


def test_snapshot_older_than_the_form_guard_reports_it_awaiting_approval_not_as_a_change():
    snap = copy.deepcopy(SNAP)
    snap["fingerprint"]["global"].pop("form_guard", None)
    r = L.check(snapshot=snap, form_guard=L.form_guard_block(YML, BETTING))
    assert r["unapproved_new_watched"] == ["global.form_guard"]
    assert not any("form_guard" in c for c in r["changes"])
    assert r["protected_tracks_modified"] is False, r["changes"]
    assert SH.protected_tracks_check(r)["status"] == SH.AMBER


# ── a missing input is never GREEN ──────────────────────────────────────────────────────────────
def test_standard_serving_without_a_board_is_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(SH, "OUT", tmp_path)
    assert SH.standard_serving_health()["status"] == SH.GREY


def test_stale_artifacts_without_a_count_is_red():
    assert SH.stale_artifacts({"generated_at": "2026-11-02T00:00:00Z"})["status"] == SH.RED


def test_workflow_health_without_a_count_is_no_data():
    assert SH.workflow_health({"generated_at": "2026-11-02T00:00:00Z"})["status"] == SH.GREY


def test_research_panel_without_panel_health_is_amber():
    lock = {"protected_tracks_modified": False, "unapproved_new_protected_tracks": [],
            "unapproved_new_watched": []}
    reg = {"regime_counts": {"PROTECTED": 1}, "tracked_leagues": ["Championship"]}
    assert SH.weekly_research_panel(reg, lock, None, NOW)["status"] == SH.AMBER


def test_promotion_gate_without_champions_is_red(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "model_lineage").mkdir(parents=True)
    pd.DataFrame({"decided_at": ["2026-11-01T00:00:00Z"], "model_type": ["standard"],
                  "promoted": ["true"], "candidate_version": ["v1"], "gate_version": ["PG-1.0"]}
                 ).to_csv(out / "model_lineage" / "promotion_decisions.csv", index=False)
    monkeypatch.setattr(SH, "OUT", out)
    monkeypatch.setattr(SH, "ROOT", tmp_path)
    assert SH.promotion_gate_health(NOW)["status"] == SH.RED


def test_missing_first_sight_store_is_not_green(tmp_path, monkeypatch):
    from src.research.first_sight import FirstSightStore
    monkeypatch.setattr(SH, "first_sight_stores", lambda: {
        "gone.csv": FirstSightStore(tmp_path / "gone.csv", key=["a"], columns=["a"])})
    assert SH.first_sight_integrity()[0]["status"] == SH.GREY


def test_absent_contract_file_is_not_green(monkeypatch):
    import src.contracts as C
    monkeypatch.setattr(C, "committed_violations", lambda root=None: {
        "output/x.csv": {"schema": "x", "present": False, "rows": 0, "n_violations": 0, "violations": []}})
    assert SH.contracts_check()[0]["status"] == SH.GREY


def test_feature_health_records_an_empty_model_as_no_data(tmp_path, monkeypatch):
    from src import health_check as H
    monkeypatch.setattr(H, "FEATURE_HEALTH_FILE", tmp_path / "feature_health.json")
    feat = pd.DataFrame({"model_type": ["standard"], "f1": [1.0]})
    st = H.record_feature_health(feat, {"standard": ["f1"], "new_format": ["f1"]})
    assert st["models"]["new_format"]["status"] == "NO_DATA"
    assert st["models"]["standard"]["status"] == "DEGRADED"          # one row: constant
    assert H.record_feature_health(feat, {})["status"] == "NO_DATA"


def test_data_truth_naming_and_leagues_without_inputs_are_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(DT, "OUT", tmp_path)
    assert DT.naming()["status"] == DT.GREY
    assert DT.leagues()["status"] == DT.GREY


def test_data_truth_missing_bank_is_red(tmp_path, monkeypatch):
    pd.DataFrame({"odds_over25": [1.9], "model_type": ["new_format"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    monkeypatch.setattr(DT, "OUT", tmp_path)
    c = DT.missing(NOW)
    assert c["status"] == DT.RED and "fd_history.parquet missing" in c["value"]


# ── data truth 5: duplicate matches cannot return ───────────────────────────────────────────────
def _bank(rows):
    return pd.DataFrame([{"date": pd.Timestamp(d), "league": lg, "season": "2023", "home_team": h,
                          "away_team": a, "home_goals": float(hg), "away_goals": float(ag)}
                         for d, lg, h, a, hg, ag in rows])


BANK = [("2023-06-03", "USA MLS", "Philadelphia Union", "CF Montreal", 3, 0),
        ("2023-06-04", "USA MLS", "Philadelphia Union", "CF Montreal", 3, 0),
        ("2023-06-10", "USA MLS", "Philadelphia Union", "Orlando City", 1, 1),
        ("2024-08-04", "Bundesliga 2", "Greuther Furth", "PreuÃ\x9fen MÃ¼nster", 3, 1),
        ("2024-08-04", "Bundesliga 2", "Greuther Furth", "Preußen Münster", 3, 1)]


def test_data_truth_one_match_green_through_the_loader_pipeline():
    c = DT.one_match({"fd_cache": _bank(BANK)})
    assert c["status"] == DT.GREEN, c["value"]
    assert "5 bank rows -> 3" in c["value"]


def test_data_truth_one_match_red_when_the_pipeline_lets_a_copy_through(monkeypatch):
    from src import data_loader as DL
    monkeypatch.setattr(DL, "one_club_one_match", lambda d: d)
    c = DT.one_match({"fd_cache": _bank(BANK)})
    assert c["status"] == DT.RED
    assert c["value"].startswith("2 duplicate pair(s)")


def test_data_truth_one_match_without_the_bank_is_no_data():
    assert DT.one_match({"fd_cache": pd.DataFrame()})["status"] == DT.GREY


def test_one_match_is_part_of_the_data_truth_evaluation(monkeypatch):
    for n in ("naming", "defaults", "leagues"):
        monkeypatch.setattr(DT, n, lambda: DT._c(n, DT.GREEN, "", ""))
    monkeypatch.setattr(DT, "missing", lambda now=None: DT._c("m", DT.GREEN, "", ""))
    monkeypatch.setattr(DT, "one_match", lambda: DT._c("Data truth 5: one match, one row", DT.RED, "x", ""))
    assert any(c["check"].startswith("Data truth 5") for c in DT.evaluate(NOW))
