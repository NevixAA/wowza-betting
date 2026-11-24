"""Stale-module guard for the deployed dashboard (owner-approved 2026-11-23).

THE ERROR CLASS. Streamlit Cloud pulls a new commit of the slim dashboard repo into the RUNNING
process. Page scripts are re-read from disk (Streamlit's script cache is cleared when sessions
disconnect), but every project module already in sys.modules stays the OLD build. A new page then
calls an old helper:
    TypeError: top3_summaries() got an unexpected keyword argument 'staked'      (Clean Record, 2026-11-23)
    AttributeError: module 'dashboard_data.pro' has no attribute 'tip_scoreboard' (League Scout, 2026-10-08)
Per-page `importlib.reload(V)` patched single helpers on single pages; it never covered the helpers'
own imports or the pages without it.

THE GUARD. app.py is the router (st.navigation): it runs on EVERY rerun of EVERY page, before the page.
Its first project import is this module, and its first call is ensure_current_build(), BEFORE any
other project import. It compares
  * the deployed build id - deploy/BUILD_INFO.json["dashboard_build_id"], written by
    scripts/build_dashboard_repo.py in the sync (a digest of the code files, so it changes exactly
    when code changes), with
  * the build id recorded when the project modules now in memory were imported.
On a mismatch it removes every project module from sys.modules (any module loaded from a file
under the app root - dashboard_data.*, dashboard_ui, dashboard_access, dashboard_words, src.*,
player_model.*, utils.*, config, this guard - plus those top-level names), clears st.cache_data and
st.cache_resource (cached values can hold objects of the old classes), and records the new id.
Because this runs before app.py or the page imports anything else, the SAME run then imports the new
code - no st.rerun() is needed, so a rerun loop is impossible: the next run sees recorded == deployed.

The record lives in sys.modules[STATE_KEY] (a plain namespace, not a project file), so it survives the
purge. First run in a process: if project modules are already in memory but nothing was recorded
(they were imported by a build that predates this guard), that is treated as an unknown build and
purged once.

BUILD_INFO missing or unreadable (a full v9 checkout, a broken sync): returns the named problem
BUILD_INFO_MISSING / BUILD_INFO_UNREADABLE, purges nothing and never raises. A local `streamlit run`
reloads changed modules through its own file watcher.
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_INFO_REL = Path("deploy") / "BUILD_INFO.json"
STATE_KEY = "_wowza_build_guard_state"

#: Project top-level names, also purged when a module carries no __file__ (namespace packages). A module
#: with a file is purged only when that file is under the app root, so a same-named installed library is
#: never touched.
PROJECT_TOPLEVEL = ("dashboard_data", "dashboard_ui", "dashboard_access", "dashboard_words",
                    "dashboard_fantasy_tools", "dashboard_build_guard", "src", "player_model", "utils",
                    "config")

_LOCK = threading.Lock()


def deployed_build(root: Path = ROOT) -> tuple[str | None, str | None]:
    """(build_id, problem). problem is None, BUILD_INFO_MISSING or BUILD_INFO_UNREADABLE."""
    p = Path(root) / BUILD_INFO_REL
    if not p.exists():
        return None, "BUILD_INFO_MISSING"
    try:
        bid = json.loads(p.read_text(encoding="utf-8")).get("dashboard_build_id")
    except Exception:                                                # noqa: BLE001
        return None, "BUILD_INFO_UNREADABLE"
    if not bid:
        return None, "BUILD_INFO_UNREADABLE"
    return str(bid), None


def _state(modules: dict) -> types.ModuleType:
    st_ = modules.get(STATE_KEY)
    if st_ is None:
        st_ = types.ModuleType(STATE_KEY)          # a module object so sys.modules stays well-typed
        st_.loaded_build = None
        st_.purges = 0
        modules[STATE_KEY] = st_
    return st_


def _under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        rp = Path(path).resolve()
    except (OSError, ValueError):
        return False
    if any(part in ("site-packages", "dist-packages") for part in rp.parts):
        return False
    try:
        rp.relative_to(root)
        return True
    except ValueError:
        return False


def project_modules(modules: dict | None = None, root: Path = ROOT,
                    toplevel: tuple[str, ...] = PROJECT_TOPLEVEL) -> list[str]:
    """Names in `modules` that belong to the app: loaded from a file under `root`, or under a project
    top-level name. __main__ (Streamlit's running script) and the guard state are never included."""
    modules = sys.modules if modules is None else modules
    root = Path(root).resolve()
    out = []
    for name, mod in list(modules.items()):
        if name in ("__main__", STATE_KEY) or mod is None:
            continue
        f = getattr(mod, "__file__", None)
        paths = [f] if f else list(getattr(mod, "__path__", None) or [])
        if any(_under(p, root) for p in paths):
            out.append(name)
        elif name.split(".")[0] in toplevel and not f and not any(
                "site-packages" in str(p) or "dist-packages" in str(p) for p in paths):
            out.append(name)                         # project namespace package / fileless entry
    return out


def _clear_streamlit_caches() -> None:
    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:                                                # noqa: BLE001
        pass                                                         # no Streamlit runtime (tests, CLI)


def ensure_current_build(root: Path = ROOT, modules: dict | None = None,
                         toplevel: tuple[str, ...] = PROJECT_TOPLEVEL, clear_caches=_clear_streamlit_caches,
                         self_name: str = __name__) -> dict:
    """Make the project modules in memory match the deployed build. Never raises.

    Returns {"status": OK | RECORDED | PURGED | BUILD_INFO_MISSING | BUILD_INFO_UNREADABLE,
             "deployed_build", "loaded_build", "purged": [module names]}.
    """
    modules = sys.modules if modules is None else modules
    try:
        bid, problem = deployed_build(root)
        with _LOCK:
            state = _state(modules)
            before = state.loaded_build
            if problem:
                return {"status": problem, "deployed_build": None, "loaded_build": before, "purged": []}
            if before == bid:
                return {"status": "OK", "deployed_build": bid, "loaded_build": before, "purged": []}
            names = [n for n in project_modules(modules, root, toplevel) if n != self_name]
            if before is None and not names:
                state.loaded_build = bid                             # fresh process: nothing to purge
                return {"status": "RECORDED", "deployed_build": bid, "loaded_build": None, "purged": []}
            # New build (or modules of an unrecorded build already in memory): drop them all, this guard
            # included so an updated guard takes effect next run, then let the imports below load fresh.
            for n in names:
                modules.pop(n, None)
            if self_name in modules and modules is sys.modules:
                modules.pop(self_name, None)
            importlib.invalidate_caches()
            state.loaded_build = bid
            state.purges += 1
        clear_caches()
        return {"status": "PURGED", "deployed_build": bid, "loaded_build": before, "purged": sorted(names)}
    except Exception as e:                                           # noqa: BLE001
        return {"status": f"GUARD_ERROR: {type(e).__name__}: {e}", "deployed_build": None,
                "loaded_build": None, "purged": []}
