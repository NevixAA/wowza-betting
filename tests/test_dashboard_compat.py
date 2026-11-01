"""R2 finding 32: dashboard code/data SHA skew must be visible, and an incompatible or unknown schema
must never be shown as healthy.

    python -m pytest tests/test_dashboard_compat.py -q
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CODE = [{"path": "app.py", "bytes": 10}, {"path": "pages/a.py", "bytes": 20}]
DATA = [{"path": "output/a.csv"}]


@pytest.fixture
def slim(tmp_path, monkeypatch):
    m = _load("dashboard_data/remote_mirror.py", "remote_mirror_compat_under_test")
    dep = tmp_path / "deploy"
    dep.mkdir()
    for k, v in {"ROOT": tmp_path, "MANIFEST": dep / "dashboard_manifest.json",
                 "SLIM_MARKER": dep / "SLIM", "SYNC_FILE": dep / ".mirror_sync.json",
                 "BUILD_INFO": dep / "BUILD_INFO.json", "V9_MANIFEST_COPY": dep / ".v9_manifest_at_data.json",
                 "FULL_CHECKOUT_SENTINEL": tmp_path / ".github" / "workflows" / "predict.yml"}.items():
        monkeypatch.setattr(m, k, v)
    monkeypatch.setenv("WOWZA_SLIM", "1")
    (dep / "dashboard_manifest.json").write_text(json.dumps(
        {"schema_version": m.DASHBOARD_SCHEMA_VERSION, "code": CODE, "data": DATA}), encoding="utf-8")
    (dep / "BUILD_INFO.json").write_text(json.dumps(
        {"dashboard_build_id": "abc123", "v9_head_at_build": "1" * 40, "schema_version": m.DASHBOARD_SCHEMA_VERSION,
         "built_at_utc": "2026-11-01T00:00:00Z"}), encoding="utf-8")
    (dep / ".mirror_sync.json").write_text(json.dumps(
        {"synced_at_utc": "2026-11-01T01:00:00Z", "attempted_at_utc": "2026-11-01T01:10:00Z", "sha": "2" * 40}),
        encoding="utf-8")
    return m, dep


def _data_manifest(dep, **kw):
    base = {"schema_version": 1, "v9_commit": "3" * 40, "code": CODE, "data": DATA}
    base.update(kw)
    (dep / ".v9_manifest_at_data.json").write_text(json.dumps(base), encoding="utf-8")


def test_compatible_build_exposes_every_sha_and_sync_time(slim):
    m, dep = slim
    _data_manifest(dep, schema_version=m.DASHBOARD_SCHEMA_VERSION)
    c = m.compatibility()
    assert c["status"] == "COMPATIBLE" and c["healthy"] is True and c["warning"] is None
    assert c["dashboard_build_id"] == "abc123" and c["source_v9_sha"] == "1" * 40
    assert c["mirrored_data_sha"] == "2" * 40 and c["data_manifest_v9_commit"] == "3" * 40
    assert c["last_successful_sync_utc"] and c["last_attempted_sync_utc"]
    assert m.status()["compat"]["status"] == "COMPATIBLE"


def test_schema_mismatch_is_incompatible_and_never_healthy(slim):
    m, dep = slim
    _data_manifest(dep, schema_version=m.DASHBOARD_SCHEMA_VERSION + 1)
    c = m.compatibility()
    assert c["status"] == "INCOMPATIBLE" and c["healthy"] is False and "schema" in c["warning"]


def test_data_manifest_without_schema_is_incompatible(slim):
    m, dep = slim
    d = {"v9_commit": "3" * 40, "code": CODE, "data": DATA}
    (dep / ".v9_manifest_at_data.json").write_text(json.dumps(d), encoding="utf-8")
    assert m.compatibility()["status"] == "INCOMPATIBLE"


def test_newer_v9_dashboard_code_or_data_is_a_stale_build(slim):
    m, dep = slim
    _data_manifest(dep, schema_version=m.DASHBOARD_SCHEMA_VERSION,
                   code=CODE + [{"path": "pages/new.py", "bytes": 5}], data=DATA + [{"path": "output/new.csv"}])
    c = m.compatibility()
    assert c["status"] == "STALE_BUILD" and c["healthy"] is False
    assert "pages/new.py" in c["code_files_differ"] and c["data_files_new"] == ["output/new.csv"]


def test_unknown_data_manifest_is_not_assumed_healthy(slim):
    m, _ = slim
    c = m.compatibility()
    assert c["status"] == "UNKNOWN" and c["healthy"] is False


def test_missing_build_info_and_never_synced(slim):
    m, dep = slim
    (dep / ".mirror_sync.json").unlink()
    assert m.compatibility()["status"] == "NOT_SYNCED"
    (dep / "BUILD_INFO.json").unlink()
    assert m.compatibility()["status"] == "BUILD_INFO_MISSING"


def test_full_checkout_is_local_and_healthy(monkeypatch):
    m = _load("dashboard_data/remote_mirror.py", "remote_mirror_compat_local")
    assert m.FULL_CHECKOUT_SENTINEL.exists()
    c = m.compatibility()
    assert c["status"] == "LOCAL_CHECKOUT" and c["healthy"] is True


def test_sync_fetches_the_data_side_manifest_at_the_pinned_sha(slim, monkeypatch):
    m, dep = slim
    (dep / ".mirror_sync.json").unlink()
    sha = "a" * 40
    v9_manifest = {"schema_version": m.DASHBOARD_SCHEMA_VERSION, "v9_commit": sha, "code": CODE, "data": DATA}
    calls = []

    def fake_get(url, headers=None, timeout=30):
        calls.append(url)
        if url.endswith("/commits/main"):
            return 200, sha.encode(), {}
        if url.endswith(f"/{sha}/deploy/dashboard_manifest.json"):
            return 200, json.dumps(v9_manifest).encode(), {}
        if url.endswith(f"/{sha}/output/a.csv"):
            return 200, b"x\n1\n", {"ETag": '"e"'}
        return 404, b"", {}

    monkeypatch.setattr(m, "_get", fake_get)
    r = m.sync()
    assert r["mode"] == "full"
    assert json.loads((dep / ".v9_manifest_at_data.json").read_text(encoding="utf-8"))["v9_commit"] == sha
    # the BUILD's own manifest is never overwritten by the data side's
    assert "schema_version" in json.loads((dep / "dashboard_manifest.json").read_text(encoding="utf-8"))
    assert json.loads((dep / "dashboard_manifest.json").read_text(encoding="utf-8")).get("v9_commit") is None
    assert m.compatibility()["status"] == "COMPATIBLE"
    calls.clear()
    m.sync()                                     # unchanged sha: no downloads at all
    assert all("raw.githubusercontent" not in c for c in calls)


def test_builder_writes_build_info_and_manifest_carries_schema(tmp_path):
    b = _load("scripts/build_dashboard_repo.py", "build_dashboard_repo_compat")
    man = json.loads((ROOT / "deploy" / "dashboard_manifest.json").read_text(encoding="utf-8"))
    assert man.get("schema_version") == b.schema_version()
    r = b.build(tmp_path / "slim")
    info = json.loads((Path(r["out"]) / "deploy" / "BUILD_INFO.json").read_text(encoding="utf-8"))
    assert info["schema_version"] == b.schema_version() and len(info["dashboard_build_id"]) == 16
    assert info["v9_head_at_build"] and info["v9_commit_of_manifest"]
    assert "deploy/.v9_manifest_at_data.json" in (Path(r["out"]) / ".gitignore").read_text(encoding="utf-8")
    # same code -> same build id
    assert b.build(tmp_path / "slim2")["build_id"] == r["build_id"]
