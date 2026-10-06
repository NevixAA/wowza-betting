"""Any workflow that runs the full API-Football enrichment MUST cache it.

This failure has now happened three times in this estate, each time the same way: a workflow
gets APIFOOTBALL_KEY and WOWZA_FULL_ENRICH wired in, nobody adds actions/cache, every run
starts cold and refetches the whole history, and the job dies on its timeout.

  2026-08-15  predict.yml        2-3 min -> 10+ against a 15-minute cap. Reverted.
  2026-09-06  retrain.yml        enrichment switch added the same day as predict's cache,
                                 without its own. No retrain committed until fixed.
  2026-10-01  backtest_matrix    all five jobs exceeded 2h and were cancelled, freezing
                                 best_params_standard.json and the per-league gates at 09-01.

The third one was the expensive one: it was not a slow job, it was thresholds fitted to a model
that no longer existed, and a timed-out matrix reports as a grey partial rather than a red
failure so nothing raised it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _steps(doc: dict):
    for job in (doc.get("jobs") or {}).values():
        for st in (job.get("steps") or []):
            yield job, st


def _enriching(doc: dict) -> bool:
    """True when a job actually turns the full enrichment on."""
    for _job, st in _steps(doc):
        env = st.get("env") or {}
        if str(env.get("WOWZA_FULL_ENRICH", "")) == "1" and "APIFOOTBALL_KEY" in env:
            return True
    return False


def _caches_enrichment(doc: dict) -> bool:
    for _job, st in _steps(doc):
        if "actions/cache" in str(st.get("uses", "")):
            if "api_football_cache" in str((st.get("with") or {}).get("path", "")):
                return True
    return False


ALL = sorted(WF.glob("*.yml"))


@pytest.mark.parametrize("path", ALL, ids=[p.name for p in ALL])
def test_full_enrichment_is_always_cached(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not _enriching(doc):
        pytest.skip("this workflow does not run the full enrichment")
    assert _caches_enrichment(doc), (
        f"{path.name} runs WOWZA_FULL_ENRICH=1 with APIFOOTBALL_KEY but has no "
        f"actions/cache for api_football_cache. Every run will start cold and refetch the "
        f"whole history — this is the defect that froze the per-league gates for five weeks.")


def test_the_backtest_matrix_specifically_is_fixed():
    doc = yaml.safe_load((WF / "backtest_matrix.yml").read_text(encoding="utf-8"))
    assert _enriching(doc) and _caches_enrichment(doc)
    job = doc["jobs"]["backtest"]
    assert job["timeout-minutes"] >= 240, (
        "the 2026-10-01 run took 2h 1m 56s against a 120-minute cap; the new cap must leave "
        "real headroom, not clear the measured time by minutes")


def test_the_backtest_commits_the_standard_optimizer():
    """best_params_standard.json sets Championship's live 0.07 gate. If the backtest does not
    stage it, the only artifact that can move a per-league threshold never lands."""
    txt = (WF / "backtest_matrix.yml").read_text(encoding="utf-8")
    assert "models/best_params_*.json" in txt or "best_params_standard.json" in txt


def test_retrain_now_refits_the_gates_too():
    """Daily learning with monthly gate-fitting is what let June thresholds survive every data
    fix that followed."""
    src = (Path(__file__).resolve().parents[1] / "retrain.py").read_text(encoding="utf-8")
    assert "optimize_standard_thresholds" in src
    assert "best_params_standard.json" in (WF / "retrain.yml").read_text(encoding="utf-8")
