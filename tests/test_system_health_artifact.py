"""output/system_health.json (src/system_health.build_artifact): explicit overall verdict, FAIL on a
moved protected track, and the new checks' red/amber/green rules."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import system_health as S

NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
LOCK_OK = {"protected_tracks_modified": False, "changes": [], "unapproved_new_protected_tracks": [],
           "approved_by": "t", "approved_at": "2026-10-25T08:00:00Z", "current_digest": "d"}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "model_lineage").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "models" / "registry").mkdir(parents=True)
    monkeypatch.setattr(S, "OUT", out)
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "HEALTH_FILE", out / "system_health.json")
    monkeypatch.setattr(S, "evaluate", lambda now=None: [S._c("x", S.GREEN, "ok", "r")])
    monkeypatch.setattr(S, "_research_lock_status", lambda: dict(LOCK_OK))
    monkeypatch.setattr(S, "first_sight_stores", lambda: {})
    return tmp_path


def _decisions(root, rows):
    pd.DataFrame(rows).to_csv(root / "output" / "model_lineage" / "promotion_decisions.csv", index=False)


def test_protected_track_change_forces_fail(sandbox, monkeypatch):
    monkeypatch.setattr(S, "_research_lock_status",
                        lambda: {**LOCK_OK, "protected_tracks_modified": True, "changes": ["global.x: 1 -> 2"]})
    a = S.build_artifact(NOW, prior={})
    assert a["overall"] == "FAIL" and a["protected_tracks_modified"] is True
    assert any("protected_tracks_modified" in r for r in a["reasons"])


def test_unverifiable_lock_is_fail(sandbox, monkeypatch):
    monkeypatch.setattr(S, "_research_lock_status", lambda: {"error": "boom"})
    assert S.build_artifact(NOW, prior={})["overall"] == "FAIL"


def test_overall_reasons_are_explicit(sandbox):
    a = S.build_artifact(NOW, prior={})
    assert a["overall"] in ("PASS", "WARN", "FAIL")
    flagged = [c for c in a["checks"] if c["status"] != S.GREEN]
    assert len(a["reasons"]) == len(flagged)
    assert all(c["rule"] for c in a["checks"])
    assert {"generated_at", "artifact_version", "versions", "contract_violations", "first_sight",
            "version_field_coverage"} <= set(a)


def test_promotion_gate_rules(sandbox):
    assert S.promotion_gate_health(NOW)["status"] == S.RED                         # missing
    (sandbox / "models" / "registry" / "current_champions.json").write_text(
        json.dumps({"newformat": {"model_version": "nf_A"}}), encoding="utf-8")
    _decisions(sandbox, [{"decided_at": "2026-10-25T08:46:05Z", "gate_version": "PG-1.0", "model_type": "newformat",
                          "candidate_version": "nf_A", "promoted": "True"}])
    assert S.promotion_gate_health(NOW)["status"] == S.GREEN
    _decisions(sandbox, [{"decided_at": "2026-10-25T08:46:05Z", "gate_version": "PG-1.0", "model_type": "newformat",
                          "candidate_version": "nf_B", "promoted": "True"}])
    assert S.promotion_gate_health(NOW)["status"] == S.RED                         # champion moved without the gate
    _decisions(sandbox, [{"decided_at": "2026-10-01T08:46:05Z", "gate_version": "PG-1.0", "model_type": "newformat",
                          "candidate_version": "nf_A", "promoted": "True"}])
    assert S.promotion_gate_health(NOW)["status"] == S.AMBER                       # stale


def test_edge_shadow_must_stay_shadow(sandbox):
    p = sandbox / "output" / "edge_engine_health.json"
    assert S.edge_shadow_health(NOW)["status"] == S.RED
    h = {"generated_at": "2026-10-25T11:00:00Z", "shadow_only": True, "rows_scored": 10}
    p.write_text(json.dumps(h), encoding="utf-8")
    assert S.edge_shadow_health(NOW)["status"] == S.GREEN
    p.write_text(json.dumps({**h, "shadow_only": False}), encoding="utf-8")
    assert S.edge_shadow_health(NOW)["status"] == S.RED
    p.write_text(json.dumps({**h, "generated_at": "2026-10-24T06:00:00Z"}), encoding="utf-8")
    assert S.edge_shadow_health(NOW)["status"] == S.AMBER


def test_first_sight_tamper_turns_red(sandbox, monkeypatch, tmp_path):
    from src.research.first_sight import FirstSightStore
    p = tmp_path / "log.csv"
    st = FirstSightStore(p, key=["k"], columns=["k", "v"])
    st.append(pd.DataFrame({"k": ["a", "b"], "v": ["1", "2"]}))
    monkeypatch.setattr(S, "first_sight_stores", lambda: {"log.csv": st})
    chk, detail = S.first_sight_integrity(None)
    assert chk["status"] == S.GREEN
    prior = {"log.csv": detail["log.csv"]}
    pd.DataFrame({"k": ["a", "b"], "v": ["9", "2"]}).to_csv(p, index=False)
    chk, _ = S.first_sight_integrity(prior)
    assert chk["status"] == S.RED


def test_write_artifact_uses_the_previous_file_as_prior(sandbox, monkeypatch, tmp_path):
    from src.research.first_sight import FirstSightStore
    p = tmp_path / "log.csv"
    st = FirstSightStore(p, key=["k"], columns=["k", "v"])
    st.append(pd.DataFrame({"k": ["a"], "v": ["1"]}))
    monkeypatch.setattr(S, "first_sight_stores", lambda: {"log.csv": st})
    S.write_artifact(now=NOW)
    a = S.write_artifact(now=NOW)
    assert a["first_sight"]["log.csv"]["prior_match"] is True
    assert json.loads(S.HEALTH_FILE.read_text(encoding="utf-8"))["generated_at"] == "2026-10-25T12:00:00Z"


def test_dashboard_evaluate_is_unchanged_in_shape():
    """evaluate() (rendered on the Overview page) keeps its 10 checks; the heavy checks live only in
    the artifact."""
    import inspect
    src = inspect.getsource(S.evaluate)
    assert "contracts_check" not in src and "first_sight_integrity" not in src
