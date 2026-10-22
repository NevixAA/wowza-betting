"""P1-3 guards: stale evidence is never shown as current; metadata cannot disagree with wiring."""
import json
from datetime import datetime, timezone
from pathlib import Path

from src import artifact_freshness as F

ROOT = Path(__file__).resolve().parents[1]


def _with(tmp_path, monkeypatch, files: dict, artifacts):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "ARTIFACTS", artifacts)


def test_recorded_timestamp_decides_not_mtime(tmp_path, monkeypatch):
    art = [("output/a.json", "v9", "x", ("json", ["generated_at"]), "OPERATIONAL", 24, True, None)]
    _with(tmp_path, monkeypatch, {"output/a.json": json.dumps({"generated_at": "2026-10-01T00:00:00Z"})}, art)
    m = F.evaluate(now=datetime(2026, 10, 22, tzinfo=timezone.utc), pro_root=tmp_path)
    r = m["artifacts"][0]                                    # file was just written (fresh mtime) - still STALE
    assert r["freshness"] == "STALE" and not r["valid_for_decisions"]
    assert m["n_stale_decision_facing"] == 1


def test_missing_timestamp_is_treated_as_stale(tmp_path, monkeypatch):
    art = [("output/b.json", "v9", "x", ("json", ["generated_at"]), "OPERATIONAL", 24, True, None)]
    _with(tmp_path, monkeypatch, {"output/b.json": json.dumps({"tracks": {}})}, art)
    r = F.evaluate(pro_root=tmp_path)["artifacts"][0]
    assert r["freshness"] == "UNKNOWN" and not r["valid_for_decisions"]


def test_fresh_csv_column_and_study_and_historical(tmp_path, monkeypatch):
    art = [("output/c.csv", "v9", "x", ("csv", "generated_at"), "OPERATIONAL", 24, True, None),
           ("output/s.json", "v9", "x", ("json", ["generated_at"]), "STUDY", None, True, None),
           ("output/h.json", "v9", "x", ("json", ["generated_at"]), "HISTORICAL", None, False, "y")]
    _with(tmp_path, monkeypatch, {"output/c.csv": "a,generated_at\n1,2026-10-22 08:00:00\n",
                                  "output/s.json": json.dumps({"generated_at": "2026-10-20"}),
                                  "output/h.json": json.dumps({"generated_at": "2026-10-06"})}, art)
    rows = {r["artifact"]: r for r in F.evaluate(now=datetime(2026, 10, 22, 10, tzinfo=timezone.utc),
                                                pro_root=tmp_path)["artifacts"]}
    assert rows["output/c.csv"]["freshness"] == "FRESH"
    assert rows["output/s.json"]["freshness"] == "DATED_STUDY"
    assert rows["output/h.json"]["freshness"] == "HISTORICAL" and not rows["output/h.json"]["valid_for_decisions"]


def test_dashboard_banner_flags_stale_and_unknown():
    from dashboard_data import freshness as dfresh

    class _St:
        def __init__(self):
            self.calls = []

        def error(self, m): self.calls.append(("error", m))
        def warning(self, m): self.calls.append(("warning", m))
        def info(self, m): self.calls.append(("info", m))
        def caption(self, m): self.calls.append(("caption", m))

    st = _St()
    dfresh.banner(st, "output/x.json", m={"artifacts": [{"artifact": "output/x.json", "freshness": "STALE",
                                                          "reason": "old"}]})
    dfresh.banner(st, "output/never_registered.json", m={"artifacts": []})
    assert [c[0] for c in st.calls] == ["error", "error"]
    assert "NOT VALID FOR CURRENT PERFORMANCE DECISIONS" in st.calls[0][1]


def test_wiring_metadata_agrees_with_production():
    """pipeline.py sizes tips through src/stake_plan.py, so nothing may still claim the policy is not wired."""
    pipeline = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    wired = "from src.stake_plan import write" in pipeline
    policy = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    reader = (ROOT / "src" / "bankroll_policy.py").read_text(encoding="utf-8")
    if wired:
        assert "NOT_WIRED" not in policy.get("status", "")
        assert "NOT WIRED into predict" not in reader


def test_evidence_scripts_do_not_hardcode_today():
    import re
    for s in ("track_evidence.py", "allocation_registry.py", "v91_readiness.py"):
        src = (ROOT / "scripts" / s).read_text(encoding="utf-8")
        assert not re.search(r'^TODAY\s*=\s*pd\.Timestamp\("20\d\d-', src, re.M), s
