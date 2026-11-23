"""Stale-module guard (dashboard_build_guard, owner-approved 2026-11-23).

Reproduces the live error class - a NEW page calling an OLD in-memory helper
(TypeError: top3_summaries() got an unexpected keyword argument 'staked') - in a throwaway app tree,
and proves the guard makes it impossible: a build change purges and re-imports, the same build never
purges again (no loop), and a missing BUILD_INFO is a named problem, never a crash.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

import dashboard_build_guard as G

PKG = "wg_probe_dashboard_data"           # unique name: never collides with the real dashboard_data
TOP = (PKG, "wg_probe_ui")


def _write_build(root: Path, bid: str, staked: bool) -> None:
    (root / "deploy").mkdir(exist_ok=True)
    (root / "deploy" / "BUILD_INFO.json").write_text(json.dumps({"dashboard_build_id": bid}), encoding="utf-8")
    pkg = root / PKG
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    sig = "def top3_summaries(rows, staked=None):\n    return ('new', staked)\n" if staked else \
          "def top3_summaries(rows):\n    return ('old',)\n"
    (pkg / "clean_record.py").write_text(sig, encoding="utf-8")
    (root / "wg_probe_ui.py").write_text(f"BUILD = {bid!r}\n", encoding="utf-8")
    importlib.invalidate_caches()


@pytest.fixture()
def app_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for n in [n for n in sys.modules if n.split(".")[0] in TOP]:
        sys.modules.pop(n, None)


def _page_calls_staked():
    """What the new Clean Record page does: import the helper, call it with staked=."""
    from wg_probe_dashboard_data import clean_record as CR       # noqa: PLC0415
    return CR.top3_summaries([], staked=True)


def _guard(root, modules, clears):
    return G.ensure_current_build(root=root, modules=modules, toplevel=TOP,
                                  clear_caches=lambda: clears.append(1), self_name="wg_probe_guard")


def test_build_change_reimports_and_signature_mismatch_impossible(app_tree):
    state = {}
    modules = sys.modules
    modules.pop(G.STATE_KEY, None)
    clears: list = []
    try:
        _write_build(app_tree, "build-A", staked=False)
        r = _guard(app_tree, modules, clears)
        # sys.modules holds other project modules (the real repo's) but none under this throwaway root
        assert r["status"] in ("RECORDED", "PURGED")
        import wg_probe_ui  # noqa: F401
        from wg_probe_dashboard_data import clean_record as old  # noqa: F401
        assert old.top3_summaries([]) == ("old",)

        _write_build(app_tree, "build-B", staked=True)       # the sync pushes a new build into the process
        # Without the guard: the stale module is still in memory -> exactly the live TypeError.
        with pytest.raises(TypeError, match="staked"):
            _page_calls_staked()

        r = _guard(app_tree, modules, clears)
        assert r["status"] == "PURGED" and r["deployed_build"] == "build-B" and r["loaded_build"] == "build-A"
        assert f"{PKG}.clean_record" in r["purged"] and "wg_probe_ui" in r["purged"]
        assert clears, "Streamlit caches must be cleared on a new build"
        assert _page_calls_staked() == ("new", True)          # same run: fresh import, new signature
        import wg_probe_ui as ui2
        assert ui2.BUILD == "build-B"
        state["ok"] = True
    finally:
        modules.pop(G.STATE_KEY, None)
    assert state["ok"]


def test_same_build_never_purges_again_no_loop(app_tree):
    modules = {}                                              # isolated module table
    clears: list = []
    _write_build(app_tree, "build-A", staked=True)
    assert _guard(app_tree, modules, clears)["status"] == "RECORDED"
    modules[f"{PKG}.clean_record"] = type(sys)("x")
    for _ in range(5):                                        # every rerun of every page
        r = _guard(app_tree, modules, clears)
        assert r["status"] == "OK" and r["purged"] == []
    assert not clears and f"{PKG}.clean_record" in modules
    _write_build(app_tree, "build-B", staked=True)
    assert _guard(app_tree, modules, clears)["status"] == "PURGED"
    assert [_guard(app_tree, modules, clears)["status"] for _ in range(3)] == ["OK"] * 3
    assert len(clears) == 1 and modules[G.STATE_KEY].purges == 1


def test_modules_from_a_build_before_the_guard_are_purged_once(app_tree):
    modules = {}
    _write_build(app_tree, "build-A", staked=True)
    m = type(sys)(PKG)
    m.__file__ = str(app_tree / PKG / "__init__.py")
    modules[PKG] = m                                          # imported by an older, unrecorded build
    r = _guard(app_tree, modules, [])
    assert r["status"] == "PURGED" and PKG not in modules
    assert _guard(app_tree, modules, [])["status"] == "OK"


@pytest.mark.parametrize("content,problem", [(None, "BUILD_INFO_MISSING"), ("{not json", "BUILD_INFO_UNREADABLE"),
                                             ("{}", "BUILD_INFO_UNREADABLE")])
def test_build_info_missing_is_named_problem_no_crash(app_tree, content, problem):
    modules = {}
    _write_build(app_tree, "build-A", staked=True)
    bi = app_tree / "deploy" / "BUILD_INFO.json"
    if content is None:
        bi.unlink()
    else:
        bi.write_text(content, encoding="utf-8")
    modules[f"{PKG}.clean_record"] = type(sys)("x")
    r = _guard(app_tree, modules, [])
    assert r["status"] == problem and r["purged"] == [] and f"{PKG}.clean_record" in modules


def test_installed_library_with_project_name_is_never_purged(tmp_path):
    lib = type(sys)("utils")
    lib.__file__ = str(tmp_path / "venv" / "lib" / "site-packages" / "utils" / "__init__.py")
    assert G.project_modules({"utils": lib}, root=tmp_path / "app") == []


def test_real_repo_project_modules_cover_the_owner_list():
    import dashboard_access, dashboard_ui  # noqa: F401,E401
    from dashboard_data import core  # noqa: F401
    names = set(G.project_modules())
    for n in ("dashboard_access", "dashboard_ui", "dashboard_words", "dashboard_data", "dashboard_data.core"):
        assert n in names
    assert "streamlit" not in names and "pandas" not in names and "__main__" not in names


def test_app_router_runs_guard_before_any_other_project_import():
    src = (Path(G.__file__).parent / "app.py").read_text(encoding="utf-8")
    guard_at = src.index("dashboard_build_guard.ensure_current_build()")
    for other in ("from dashboard_access", "from dashboard_data", "import dashboard_ui", "import config"):
        if other in src:
            assert src.index(other) > guard_at, other
    assert "st.navigation" in src and "nav.run()" in src     # every page goes through this router
