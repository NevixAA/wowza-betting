"""The SLIM dashboard deployment (2026-10-26) - manifest, runtime mirror, slim-tree builder.

The dashboard deploys from a generated code-only repo; its data is mirrored at runtime from v9's raw
URLs. A page that imports a module, or reads a v9 file, that deploy/dashboard_manifest.json does not
list works on a full checkout and silently breaks the deployment. These tests are the fast, static
half of that guard (the full AppTest trace is `python scripts/dashboard_manifest.py --check`):

  1. every module reachable by import statements from app.py and every routed page (imports inside
     functions and button branches included) is in the manifest's code list;
  2. every data-file name written literally in the viewer layer that resolves to a v9 file is in the
     manifest (as data, or as a deliberate exclusion the page degrades without);
  3. the mirror is a no-op outside slim mode, and in slim mode writes atomically, keeps the old copy on
     failure, and skips downloads when v9's head commit has not moved;
  4. the builder copies code only (no data, no models) and the tree stays small.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "deploy" / "dashboard_manifest.json").read_text(encoding="utf-8"))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DM = _load("scripts/dashboard_manifest.py", "dashboard_manifest_under_test")
CODE = {e["path"] for e in MANIFEST["code"]}
DATA = {e["path"] for e in MANIFEST["data"]}


def test_every_page_and_import_is_in_the_manifest():
    entries = DM.entry_files(ROOT)
    assert set(entries) <= CODE, sorted(set(entries) - CODE)
    missing = sorted(DM.static_closure(ROOT, entries) - CODE)
    assert not missing, ("modules the dashboard imports are not in deploy/dashboard_manifest.json - run "
                         f"`python scripts/dashboard_manifest.py --write`: {missing}")
    for rel in CODE:
        assert (ROOT / rel).is_file(), f"manifest code file missing: {rel}"


def test_every_literal_data_file_is_in_the_manifest():
    bad = DM.unmatched_literals(MANIFEST, ROOT)
    assert not bad, ("viewer code names v9 data files the slim deployment would not mirror - run "
                     f"`python scripts/dashboard_manifest.py --write`: {bad}")


def test_manifest_is_sane():
    assert not (CODE & DATA)
    assert ".streamlit/config.toml" in CODE and "dashboard_data/remote_mirror.py" in CODE
    for e in MANIFEST["data"]:
        name = e["path"].rsplit("/", 1)[-1]
        assert name not in DM.SECRET_NAMES and not e["path"].startswith(("..", "/"))
        assert e["bytes"] is None or e["bytes"] <= DM.MAX_DATA_BYTES, e
        assert not e["path"].endswith((".pkl", ".joblib")), f"a model is not dashboard data: {e['path']}"
    excluded = {e["path"] for e in MANIFEST.get("excluded", [])}
    assert "player_history.parquet" in excluded or not (ROOT / "player_history.parquet").exists()


# ── runtime mirror ────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mirror(tmp_path, monkeypatch):
    m = _load("dashboard_data/remote_mirror.py", "remote_mirror_under_test")
    (tmp_path / "deploy").mkdir()
    man = {"data": [{"path": "output/a.csv"}, {"path": "config/b.json"}, {"path": "output/gone.json"}]}
    (tmp_path / "deploy" / "dashboard_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "MANIFEST", tmp_path / "deploy" / "dashboard_manifest.json")
    monkeypatch.setattr(m, "SLIM_MARKER", tmp_path / "deploy" / "SLIM")
    monkeypatch.setattr(m, "SYNC_FILE", tmp_path / "deploy" / ".mirror_sync.json")
    monkeypatch.setattr(m, "FULL_CHECKOUT_SENTINEL", tmp_path / ".github" / "workflows" / "predict.yml")
    server = {"sha": "a" * 40, "files": {"output/a.csv": b"x,y\n1,2\n", "config/b.json": b"{}"},
              "changed": None, "calls": [], "fail": set()}

    def fake_get(url, headers=None, timeout=30):
        server["calls"].append(url)
        if url.endswith("/commits/main"):
            return (200, server["sha"].encode(), {}) if server["sha"] else (403, b"", {})
        if "/compare/" in url:
            if server["changed"] is None:
                return 403, b"", {}
            return 200, json.dumps({"files": [{"filename": f} for f in server["changed"]]}).encode(), {}
        rel = url.split("raw.githubusercontent.com/NevixAA/wowza-betting/", 1)[1].split("/", 1)[1]
        if rel in server["fail"]:
            return 500, b"", {}
        body = server["files"].get(rel)
        if body is None:
            return 404, b"", {}
        etag = f'"{hash(body)}"'
        if (headers or {}).get("If-None-Match") == etag:
            return 304, b"", {}
        return 200, body, {"ETag": etag}

    monkeypatch.setattr(m, "_get", fake_get)
    return m, server, tmp_path


def test_mirror_is_a_noop_outside_slim_mode(mirror, monkeypatch):
    m, server, root = mirror
    monkeypatch.setenv("WOWZA_SLIM", "0")
    (root / "deploy" / "SLIM").write_text("x")
    assert m.ensure_fresh() == {"slim": False}
    assert server["calls"] == [] and not (root / "output").exists()


def test_a_full_checkout_is_never_mirrored_into(monkeypatch):
    m = _load("dashboard_data/remote_mirror.py", "remote_mirror_full_checkout")
    monkeypatch.setenv("WOWZA_SLIM", "1")
    assert m.FULL_CHECKOUT_SENTINEL.exists() and m.slim_mode() is False


def test_mirror_cold_sync_then_cheap_resyncs(mirror, monkeypatch):
    m, server, root = mirror
    monkeypatch.setenv("WOWZA_SLIM", "1")
    st = m.ensure_fresh()                                   # cold: waits for the first sync
    assert st["synced_at_utc"] and st["sha"] == "a" * 40
    assert (root / "output" / "a.csv").read_bytes() == b"x,y\n1,2\n"
    assert (root / "config" / "b.json").read_bytes() == b"{}"
    assert not (root / "output" / "gone.json").exists()   # 404 upstream: recorded missing, not fatal
    assert "?" not in "".join(server["calls"]) and f"/{'a' * 40}/output/a.csv" in "".join(server["calls"])

    # head commit unchanged -> no file downloads at all
    server["calls"].clear()
    r = m.sync()
    assert r["mode"] == "unchanged-sha" and all("raw.githubusercontent" not in c for c in server["calls"])

    # head moved, compare API answers -> only the changed manifest file is fetched
    server["sha"], server["changed"] = "b" * 40, ["output/a.csv", "src/unrelated.py"]
    server["files"]["output/a.csv"] = b"x,y\n3,4\n"
    server["calls"].clear()
    r = m.sync()
    raw = [c for c in server["calls"] if "raw.githubusercontent" in c]
    assert r["mode"] == "compare" and len(raw) == 1 and raw[0].endswith("/output/a.csv")
    assert (root / "output" / "a.csv").read_bytes() == b"x,y\n3,4\n"


def test_mirror_keeps_old_copy_on_failure_and_uses_etags(mirror, monkeypatch):
    m, server, root = mirror
    monkeypatch.setenv("WOWZA_SLIM", "1")
    m.sync()
    server["sha"] = None                                     # API rate-limited: full conditional sweep
    server["fail"] = {"output/a.csv"}
    r = m.sync()
    assert r["mode"] == "full" and r["counts"]["failed"] == 1 and r["counts"]["unchanged"] == 1
    assert (root / "output" / "a.csv").read_bytes() == b"x,y\n1,2\n"     # old copy kept
    assert not list(root.rglob("*.tmp"))                                  # no torn temp files


def test_mirror_never_blocks_a_warm_render(mirror, monkeypatch):
    m, server, root = mirror
    monkeypatch.setenv("WOWZA_SLIM", "1")
    m.ensure_fresh()
    import time
    t0 = time.time()
    m.ensure_fresh(ttl_s=0)                                  # due: refresh runs in the background
    assert time.time() - t0 < 1.0
    th = m._STATE["thread"]
    th.join(5)
    assert not th.is_alive()


# ── builder ───────────────────────────────────────────────────────────────────────────────────────
def test_builder_copies_code_only(tmp_path):
    b = _load("scripts/build_dashboard_repo.py", "build_dashboard_repo_under_test")
    r = b.build(tmp_path / "slim")
    out = Path(r["out"])
    assert (out / "deploy" / "SLIM").exists() and (out / "app.py").exists()
    assert (out / "requirements.txt").exists() and (out / ".streamlit" / "config.toml").exists()
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert not (files & DATA), sorted(files & DATA)[:5]
    assert not [f for f in files if f.endswith((".pkl", ".parquet", ".joblib"))]
    assert r["bytes"] < 10_000_000, r
    ignore = (out / ".gitignore").read_text(encoding="utf-8")
    assert "/output/" in ignore and ".streamlit/secrets.toml" in ignore
    with pytest.raises(SystemExit):
        b.build(ROOT / "deploy" / "nope")                    # never inside the v9 checkout
