"""P0 item 2 (2026-11-22): four independent health indicators, never one GREEN for all four.

The case the owner found: the CI gate was GREEN while a data truth check was RED and system health FAIL,
and the dashboard showed the gate verdict as the data truth stoplight.
"""
import json
from pathlib import Path

from dashboard_data import health_status as HS
from dashboard_data import truth_view as TV
from src.integrity import data_truth as DT

NOW = "2026-11-22T18:00:00Z"
T_DT = "2026-11-22T15:00:00Z"
T_SH = "2026-11-22T14:00:00Z"
T_MI = "2026-11-22T16:00:00Z"


def _w(d: Path, name: str, doc: dict) -> None:
    (d / name).write_text(json.dumps(doc), encoding="utf-8")


def _checks(rule3="RED"):
    return [{"check": "Data truth 1: one canonical name", "status": "GREEN", "value": "0", "rule": "r", "detail": ""},
            {"check": "Data truth 3: no missing data", "status": rule3,
             "value": "team:model_ht_over05.pkl: 1 served feature(s) hold one value", "rule": "r", "detail": ""},
            {"check": "Data truth 5: one match, one row", "status": "GREEN", "value": "0", "rule": "r", "detail": ""}]


def _scenario(tmp_path: Path, *, rule3="RED", model="FAIL", lineage="PASS", parity="PASS", overall="FAIL",
              ptm=False) -> Path:
    """CI gate GREEN (only ungated rule 3 is RED), model integrity as given, joins active PASS / legacy FAIL."""
    _w(tmp_path, "data_truth_health.json", {"generated_at": T_DT, "checks": _checks(rule3), "stoplight": "GREEN",
                                            "gate_reasons": []})
    _w(tmp_path, "system_health.json", {
        "generated_at": T_SH, "overall": overall, "protected_tracks_modified": ptm,
        "research_lock": {"protected_tracks_modified": ptm},
        "checks": _checks(rule3) + [{"check": "Side-market routing", "status": "GREEN", "value": "", "rule": "",
                                     "detail": ""}]})
    _w(tmp_path, "model_integrity_health.json", {
        "generated_at": T_MI, "status": model, "overall_model_integrity_status": model,
        "active_serving_join_status": "PASS", "active_training_lineage_status": lineage,
        "legacy_replay_status": "FAIL",
        "tracks": {"standard|HT_O05": {"status": model, "parity": {"status": "FAIL",
                                                                    "fail_features": ["home_has_ht_history"]},
                                       "join_integrity": {"status": "PASS"},
                                       "training_lineage": {"status": lineage}}}})
    _w(tmp_path, "join_integrity_health.json", {
        "generated_at": T_MI, "status": "FAIL", "active_serving_join_status": "PASS", "legacy_replay_status": "FAIL",
        "joins": {"fd_history_ht_fill": {"mode": "safe", "status": "PASS", "left_rows": 10, "output_rows": 10},
                  "fd_history_ht_fill.legacy": {"mode": "legacy_audit", "status": "FAIL", "left_rows": 10,
                                                "output_rows": 12}}})
    _w(tmp_path, "feature_parity_health.json", {"generated_at": T_MI, "status": parity})
    return tmp_path


def test_data_truth_gate_vs_operational_health(tmp_path):
    out = _scenario(tmp_path)
    # the real gate: an ungated RED (rule 3, operational) does not block a merge ...
    checks = json.loads((out / "data_truth_health.json").read_text())["checks"]
    assert not DT.is_gated("Data truth 3: no missing data")
    assert DT.gate(checks=checks) == []
    # ... and the four indicators keep the questions apart
    ind = HS.indicators(out, now=NOW)
    assert ind["ci_gate"]["status"] == "GREEN"
    assert ind["operational_data_quality"]["status"] == "RED"
    assert ind["production_model_integrity"]["status"] == "FAIL"
    assert ind["full_system_certification"]["status"] == "NOT CERTIFIED"
    # every failing check carries component, source artifact and generated_at
    red = [f for f in ind["operational_data_quality"]["failing"] if f["status"] == "RED"]
    assert [f["component"] for f in red] == ["Data truth 3: no missing data"]
    assert red[0]["source"] == "output/data_truth_health.json" and red[0]["generated_at"] == T_DT
    assert red[0]["gated"] is False
    for k in HS.INDICATORS:
        for f in ind[k]["failing"]:
            assert {"component", "status", "detail", "source", "generated_at"} <= set(f)
    comps = {f["component"] for f in ind["full_system_certification"]["failing"]}
    assert {"system health", "data truth (operational)", "model integrity"} <= comps
    # the legacy replay is reported, never counted into model integrity
    assert ind["production_model_integrity"]["scopes"]["legacy_replay_status"] == "FAIL"
    assert ind["production_model_integrity"]["scopes"]["active_serving_join_status"] == "PASS"


def test_dashboard_status_consistency(tmp_path):
    out = _scenario(tmp_path)
    s = TV.stoplight(out, now=NOW)
    # the dashboard's data truth status is the WORST check, never the gate verdict
    assert s["status"] == "RED" and s["gate"] == "GREEN"
    assert "gated or not" in s["basis"]
    p = TV.panel(out, now=NOW)
    ind = HS.indicators(out, now=NOW)
    assert {k: v["status"] for k, v in p["indicators"].items()} == {k: v["status"] for k, v in ind.items()}
    assert p["indicators"]["operational_data_quality"]["status"] == s["status"]
    # one GREEN never stands for all four
    assert sorted(v["status"] for v in ind.values()) != ["GREEN"] * 4
    assert len({v["status"] for v in ind.values()}) == 4
    # system_health.build_artifact passes the artifact it writes: the newer copy wins, same verdicts
    newer = json.loads((out / "system_health.json").read_text())
    newer.update(generated_at="2026-11-22T17:00:00Z",
                 data_truth_gate={"stoplight": "GREEN", "gate_reasons": []})
    ind2 = HS.indicators(out, now=NOW, docs={"system_health.json": newer})
    assert ind2["ci_gate"]["source"] == "output/system_health.json"
    assert ind2["operational_data_quality"]["source"] == "output/system_health.json"
    assert {k: v["status"] for k, v in ind2.items()} == {k: v["status"] for k, v in ind.items()}


def test_gate_red_and_research_lock_turn_ci_red(tmp_path):
    out = _scenario(tmp_path, ptm=True)
    ind = HS.indicators(out, now=NOW)
    assert ind["ci_gate"]["status"] == "RED"
    assert any(f["component"] == "research lock" for f in ind["ci_gate"]["failing"])


def test_missing_or_stale_evidence_is_never_certified(tmp_path):
    out = _scenario(tmp_path, rule3="GREEN", model="PASS", overall="PASS")
    ind = HS.indicators(out, now=NOW)
    assert ind["full_system_certification"]["status"] == "CERTIFIED"   # reachable from real evidence
    assert ind["operational_data_quality"]["status"] == "GREEN"
    (out / "feature_parity_health.json").unlink()
    ind = HS.indicators(out, now=NOW)
    assert ind["full_system_certification"]["status"] == "NOT CERTIFIED"
    assert [f["detail"] for f in ind["full_system_certification"]["failing"]] == ["evidence missing"]
    _scenario(tmp_path, rule3="GREEN", model="PASS", overall="PASS")
    ind = HS.indicators(out, now="2026-11-25T18:00:00Z")                 # every artifact > 36 h old
    assert ind["full_system_certification"]["status"] == "NOT CERTIFIED"
    assert all("stale" in f["detail"] for f in ind["full_system_certification"]["failing"])


def test_no_artifacts_is_no_data(tmp_path):
    ind = HS.indicators(tmp_path, now=NOW)
    assert ind["ci_gate"]["status"] == "NO_DATA"
    assert ind["operational_data_quality"]["status"] == "NO_DATA"
    assert ind["production_model_integrity"]["status"] == "NO_DATA"
    assert ind["full_system_certification"]["status"] == "NOT CERTIFIED"
