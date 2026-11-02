"""ops6 item 2: update_results.yml runs the corrections sidecars in CI (they need API-Football) and
commits output/corrections/*, budget-aware: once a day on the 07 UTC run, or on a manual dispatch
with the `corrections` input. Text checks on the YAML, no runner."""
from __future__ import annotations

from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "update_results.yml"


def _doc():
    return yaml.safe_load(WF.read_text(encoding="utf-8"))


def _step():
    for st in _doc()["jobs"]["update"]["steps"]:
        if "--grade-restored" in str(st.get("run") or ""):
            return st
    return None


def _cmds(run: str) -> str:
    return "\n".join(l for l in run.splitlines() if not l.lstrip().startswith("#"))


def test_corrections_step_exists_and_runs_both_flags():
    st = _step()
    assert st is not None, "no step runs update_results.py --grade-restored"
    run = _cmds(st["run"])
    assert "python update_results.py --grade-restored --restate-extra-time" in run
    assert st["env"]["APIFOOTBALL_KEY"] == "${{ secrets.APIFOOTBALL_KEY }}"


def test_corrections_step_commits_only_the_two_sidecars():
    run = _cmds(_step()["run"])
    assert "output/corrections/player_ledger_restored_2026-11.csv" in run
    assert "output/corrections/team_ledgers_extra_time_restated.csv" in run
    assert "git add -A" not in run and "-X theirs" not in run
    # never pushes to a branch other than the checked-out one's origin main
    assert "git push" in run and "--force" not in run


def test_corrections_step_is_budget_gated():
    st = _step()
    run = _cmds(st["run"])
    # once a day on the scheduled 07:xx UTC fire, or on an explicit dispatch input
    assert 'date -u +%H' in run and '"07"' in run
    assert "inputs.corrections" in str(st.get("env")) or "inputs.corrections" in run
    inputs = (_doc().get(True) or _doc().get("on"))["workflow_dispatch"]["inputs"]
    assert inputs["corrections"]["default"] in (False, "false")


def test_corrections_step_runs_after_results_commit():
    names = [s.get("name") for s in _doc()["jobs"]["update"]["steps"]]
    assert names.index(_step()["name"]) > names.index("Commit and push")
