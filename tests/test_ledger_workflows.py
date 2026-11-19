"""Workflow wiring for the ledger race fix (estate scan 2026-11-02, P1 + oneoff label).

Every workflow step that stages a ledger the row-level sync covers (src.ledger.SYNC_LEDGERS) must
run `src.ledger sync-snapshot` and then `sync-push` BEFORE its own `git add` / `pull --rebase -X
theirs`, so the blind rebase never carries a ledger again. Text checks on the YAML, no runner.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.ledger import SYNC_LEDGERS

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
# Steps that publish a ledger WITHOUT a rebase (reset to origin + re-apply a journal) are row-safe
# already and are exempt: oneoff_dnp_recheck.
EXEMPT = {"oneoff_dnp_recheck.yml"}


def _steps(path: Path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in (doc.get("jobs") or {}).values():
        for st in job.get("steps") or []:
            run = str(st.get("run") or "")
            # commands only: comments mention `git add` and `-X theirs` too
            yield st.get("name", ""), "\n".join(l for l in run.splitlines()
                                                 if not l.lstrip().startswith("#"))


def _ledger_staging_steps():
    names = [Path(p).name for p in SYNC_LEDGERS]
    for wf in sorted(WF.glob("*.yml")):
        if wf.name in EXEMPT:
            continue
        for name, run in _steps(wf):
            if "git add" in run and any(n in run for n in names):
                yield wf.name, name, run


def test_there_are_ledger_writers_to_check():
    found = {w for w, _, _ in _ledger_staging_steps()}
    assert {"predict.yml", "update_results.yml", "daily_summary.yml", "player_props.yml",
            "sharp_tracker.yml", "worldcup.yml"} <= found


def test_every_ledger_commit_goes_through_the_row_level_sync_first():
    bad = []
    for wf, name, run in _ledger_staging_steps():
        snap = run.find("src.ledger sync-snapshot")
        push = run.find("src.ledger sync-push")
        add = run.find("git add")
        rebase = run.find("-X theirs")
        if not (0 <= snap < push < add) or (rebase >= 0 and rebase < push):
            bad.append(f"{wf} / {name}")
    assert not bad, f"ledger staged without the row-level sync first: {bad}"


def test_a_failed_sync_turns_the_step_red():
    for wf, name, run in _ledger_staging_steps():
        assert "ledger_sync_failed=1" in run and re.search(r'\$ledger_sync_failed" = "0" \]', run), \
            f"{wf} / {name}: a failed ledger sync must fail the step"


def test_update_results_runs_never_overlap():
    doc = yaml.safe_load((WF / "update_results.yml").read_text(encoding="utf-8"))
    assert doc["concurrency"]["group"] == "update-results"
    assert doc["concurrency"]["cancel-in-progress"] is False


# ── oneoff_dnp_recheck: nightly by owner intent, stops itself ─────────────────

def test_oneoff_dnp_recheck_label_matches_its_schedule_and_it_disables_itself():
    text = (WF / "oneoff_dnp_recheck.yml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    on = doc.get("on", doc.get(True))
    assert on["schedule"], "it runs nightly by owner intent"
    assert "Manual only" not in text, "the label said manual-only while a cron ran it nightly"
    job = doc["jobs"]["recheck"]
    assert job["permissions"].get("actions") == "write"
    steps = {s.get("name"): s for s in job["steps"]}
    pool = steps["Anything left to re-check?"]
    assert "notes\") == \"DNP\"" in pool["run"] and "VOID" in pool["run"]
    for n in ("Install dependencies", "Recheck DNP voids"):
        assert "steps.pool.outputs.left != '0'" in steps[n]["if"]
    assert "^checked 0 fixture" in steps["Recheck DNP voids"]["run"]
    dis = steps["Done - disable this workflow"]
    assert "gh workflow disable oneoff_dnp_recheck.yml" in dis["run"]
    assert "steps.pool.outputs.left == '0'" in dis["if"] and "steps.run.outputs.done == '1'" in dis["if"]
