"""Owner P0 item 5 (2026-11-22): props production certification from the real serialized artifacts.

output/props_real_data_provenance.json is computed by src/integrity/props_certification.py from the
installed props artifacts; Data truth 13a reads it through real_data_provenance.props_status, which
re-hashes the artifacts (STALE on any difference) and never reports VERIFIED for a failing file.
"""
from __future__ import annotations

import json

import pytest
import sklearn

from player_model import config as pc
from src.integrity import props_certification as PCERT
from src.integrity import real_data_provenance as RDP

ARTS = PCERT.installed_artifacts()
COMMITTED = PCERT.OUT


@pytest.fixture(scope="module")
def evidence():
    run_ids = {str((PCERT.load_raw(PCERT.read_bytes(a["path"])).get("provenance") or {}).get("run_id")) for a in ARTS}
    assert len(run_ids) == 1, run_ids
    ev = PCERT.run_evidence(next(iter(run_ids)))
    assert ev is not None, "committed run evidence missing (reports/p0_props_cert)"
    return ev


@pytest.mark.parametrize("art", ARTS, ids=[a["key"] for a in ARTS])
def test_artifact_provenance_checks(art, evidence):
    r = PCERT.certify_artifact(art, None, evidence, sklearn.__version__)
    failing = {k: v["detail"] for k, v in r["checks"].items() if not v["ok"] and k != "no_batch_statistics"}
    assert failing == {}, failing                     # batch invariance: tests/test_props_batch_invariance.py
    assert r["withheld_or_proxy_features"] == []
    assert r["sha256"] == (art["manifest"] or {}).get("artifact_sha256")


def test_production_feature_schema_has_no_withheld_or_proxy_feature():
    bad = PCERT._forbidden_features()
    assert bad and not (bad & set(pc.PLAYER_FEATURE_COLS))


def test_legacy_batch_median_prep_unreachable_in_production():
    serving = {a["market"]: PCERT.load_raw(PCERT.read_bytes(a["path"])) for a in ARTS if a["role"] != PCERT.RESEARCH}
    r = PCERT.prep_reachability(serving)
    assert r["ok"] and not r["reachable_in_production"], r
    assert r["legacy_payloads_scored"] == [] and r["runtime_probe"]["prep_calls"] == 0
    assert {v for k, v in r["payload_recipes"].items() if not k.startswith("r2_shadow:")} == {"props_clean_harness"}
    assert r["prep_callers_outside_model_py"] == [] and r["train_callers_production"] == []


def test_committed_certification_is_current_and_verified():
    d = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert d["verdict"] == PCERT.VERIFIED and d["failing"] == []
    assert d["counts"]["artifacts"] == len(ARTS) == d["counts"]["verified"]
    st = RDP.props_status()
    assert st["status"] == "VERIFIED", st


def _write(tmp_path, monkeypatch, d):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "props_real_data_provenance.json").write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setattr(RDP, "ROOT", tmp_path)


def test_status_pending_without_file(tmp_path, monkeypatch):
    monkeypatch.setattr(RDP, "ROOT", tmp_path)
    assert RDP.props_status()["status"] == "PENDING"


def test_status_failed_never_verified(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"verdict": "FAILED", "failing": ["x"], "installed_hashes": PCERT.installed_hashes()})
    assert RDP.props_status()["status"] == "FAILED"


def test_status_stale_when_an_artifact_changes(tmp_path, monkeypatch):
    h = PCERT.installed_hashes()
    h[next(iter(h))] = "0" * 64
    _write(tmp_path, monkeypatch, {"verdict": "VERIFIED", "failing": [], "installed_hashes": h})
    assert RDP.props_status()["status"] == "STALE"


def test_13a_green_only_when_props_verified(monkeypatch):
    h = {"A_registered_historical_exceptions": {"count": 0, "documented": [], "inactive_artifact_violations": []},
         "B_active_production_violations": {"count": 0, "rows": [], "refused_active_payloads": []},
         "C_new_violations": {"count": 0, "rows": []}, "serving_coverage_gaps": []}
    for st, want in (("VERIFIED", "GREEN"), ("FAILED", "AMBER"), ("STALE", "AMBER"), ("PENDING", "AMBER"),
                     ("REPORTED", "AMBER")):
        assert RDP.checks({**h, "props": {"status": st}})[0]["status"] == want
