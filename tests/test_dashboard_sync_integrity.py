"""Dashboard sync integrity (deep-repair brief 2026-11-22, P1 item 12).

NevixAA/wowza-dashboard is GENERATED from this repo: its sync_from_v9.yml checks out v9 main, runs
scripts/build_dashboard_repo.py and mirrors the result with rsync --delete. This pins what that sync
relies on, so a drift shows up here rather than on the deployed page:

  * every manifest CODE file is tracked here, and every DATA file is tracked or explicitly `optional`
    (the runtime mirror records a 404 as missing; an unflagged absence would be a silent hole);
  * the build is EXACTLY the manifest's code files plus the generated set, byte-equal (LF) to v9's
    sources, with BUILD_INFO stating this HEAD, the manifest's v9_commit and the right counts;
  * the files the dashboard shares with the other repos (league registry, team names, prediction ledger
    core) are shipped, so the deployed copies are v9's bytes by construction;
  * with DASHBOARD_LOCAL pointing at a wowza-dashboard checkout, its tracked tree equals this build
    (BUILD_INFO / README differ only in their build stamps) - skipped otherwise, never passed.

Which Streamlit build is actually SERVING is not provable from here (the app is login-protected); that
stays UNVERIFIED in reports unless someone checks the running app.

    python -m pytest tests/test_dashboard_sync_integrity.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "deploy" / "dashboard_manifest.json").read_text(encoding="utf-8"))
GENERATED = {"README.md", ".gitignore", "requirements.txt", "deploy/SLIM", "deploy/BUILD_INFO.json",
             "deploy/dashboard_manifest.json", ".streamlit/config.toml"}
STAMPED = {"README.md", "deploy/BUILD_INFO.json"}
SHARED_WITH_OTHER_REPOS = ("config/leagues.json", "src/team_names.py", "src/prediction_ledger.py")


def _paths(key):
    return [e if isinstance(e, str) else e["path"] for e in MANIFEST[key]]


def _tracked(root: Path) -> set[str]:
    out = subprocess.run(["git", "-C", str(root), "-c", "core.quotepath=off", "ls-files"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        pytest.skip(f"not a git checkout: {root}")
    return {x for x in out.stdout.split("\n") if x}


def _lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).parts[0] not in (".git", ".github")}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("build_dashboard_repo_sync", ROOT / "scripts" /
                                                  "build_dashboard_repo.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    out = tmp_path_factory.mktemp("slim") / "slim"
    r = b.build(out)
    return Path(r["out"])


def test_manifest_code_files_are_all_tracked():
    tracked = _tracked(ROOT)
    missing = [p for p in _paths("code") if p not in tracked]
    assert not missing, f"manifest code files not in this checkout: {missing}"


def test_manifest_data_files_tracked_or_flagged_optional():
    tracked = _tracked(ROOT)
    silent = [e["path"] for e in MANIFEST["data"] if e["path"] not in tracked and not e.get("optional")]
    assert not silent, f"data files the dashboard downloads but v9 does not carry: {silent}"


def test_shared_files_are_shipped():
    shipped = set(_paths("code")) | set(_paths("data"))
    assert not [f for f in SHARED_WITH_OTHER_REPOS if f not in shipped]


def test_build_is_exactly_the_manifest(built):
    got = _files(built)
    code = set(_paths("code"))
    assert got - code - GENERATED == set(), f"unlisted files in the build: {sorted(got - code - GENERATED)}"
    assert code - got == set(), f"manifest code files missing from the build: {sorted(code - got)}"
    diff = [p for p in code if _lf(built / p) != _lf(ROOT / p)]
    assert not diff, f"built copies differ from v9's sources: {diff}"


def test_build_info_states_this_head_and_counts(built):
    info = json.loads((built / "deploy" / "BUILD_INFO.json").read_text(encoding="utf-8"))
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0:
        assert info["v9_head_at_build"] == head.stdout.strip()
    assert info["v9_commit_of_manifest"] == MANIFEST["v9_commit"]
    assert info["code_files"] == len(MANIFEST["code"]) and info["data_files_mirrored"] == len(MANIFEST["data"])
    assert info["schema_version"] == MANIFEST["schema_version"]


def test_live_dashboard_checkout_equals_this_build(built):
    dash = os.getenv("DASHBOARD_LOCAL", "")
    if not dash or not (Path(dash) / "deploy" / "BUILD_INFO.json").is_file():
        pytest.skip("set DASHBOARD_LOCAL to a NevixAA/wowza-dashboard checkout to compare the deployed tree")
    dash = Path(dash)
    tracked = {p for p in _tracked(dash) if not p.startswith(".github/")}
    got = _files(built)
    assert tracked == got, (f"only in dashboard: {sorted(tracked - got)}; "
                            f"only in this build: {sorted(got - tracked)}")
    diff = [p for p in sorted(got - STAMPED) if _lf(built / p) != _lf(dash / p)]
    assert not diff, f"dashboard files differ from a fresh build of this checkout: {diff}"
    info = json.loads((dash / "deploy" / "BUILD_INFO.json").read_text(encoding="utf-8"))
    assert info["v9_commit_of_manifest"] == MANIFEST["v9_commit"]
    assert info["schema_version"] == MANIFEST["schema_version"]
