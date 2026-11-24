"""Monthly side-model candidate (owner 2026-11-24): gate skip, never auto-installs, held-out-only comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.side_models import candidate as C

ROOT = Path(__file__).resolve().parents[1]


# ── gate ───────────────────────────────────────────────────────────────────────────────────────────────
def test_gate_runs_only_on_green_data_truth_and_lineage_pass():
    assert C.gate_decision(0, {"status": "PASS"})[0] is True
    ok, why = C.gate_decision(1, {"status": "PASS"})
    assert ok is False and "data_truth --gate is not GREEN (exit 1)" in why
    ok, why = C.gate_decision(0, {"status": "UNVERIFIED", "why": "evidence is for another file version"})
    assert ok is False and "lineage is UNVERIFIED" in why and "another file version" in why
    ok, why = C.gate_decision(None, None)
    assert ok is False and "not run" in why and "missing" in why


def test_gate_cli_skip_writes_reason_and_step_output(tmp_path, monkeypatch):
    from src.integrity import training_lineage as TL
    monkeypatch.setattr(TL, "side_lineage", lambda root=None: {"status": "PASS"})
    gh = tmp_path / "gh_out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    assert C.main(["gate", "--dt-rc", "1", "--out", str(tmp_path)]) == 0
    g = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert g["run"] is False and "not GREEN" in g["reason"]
    assert gh.read_text(encoding="utf-8").strip() == "go=false"
    assert "SKIPPED" in C.message(None, g) and "not GREEN" in C.message(None, g)


# ── never installs ─────────────────────────────────────────────────────────────────────────────────────
def test_build_refuses_any_output_inside_the_checkout():
    with pytest.raises(SystemExit):
        C.build(ROOT / "models" / "side_models")
    with pytest.raises(SystemExit):
        C.build(ROOT / "tmp_candidate_out")


def test_candidate_module_never_writes_repo_paths():
    src = (ROOT / "src" / "side_models" / "candidate.py").read_text(encoding="utf-8")
    # every write goes under out_dir / inst_dir (the runner temp dir); none names a repo path
    for bad in ('ROOT / "models"', "ROOT / 'models'", 'ROOT / "config"', 'ROOT / "reports"'):
        assert bad not in src.replace('SERVING = ROOT / "models"', ""), bad


# ── held-out only ──────────────────────────────────────────────────────────────────────────────────────
def test_holdout_start_is_after_serving_data_and_inside_the_window():
    assert C.holdout_start("2026-11-09", "2026-11-23", 42) == "2026-11-10"      # serving data end bounds it
    assert C.holdout_start("2026-11-09", "2027-01-31", 42) == "2026-12-21"      # the 42-day window bounds it


def test_comparison_refuses_games_a_model_trained_on():
    serving = {"data_end": "2026-11-09", "models": {}}
    fit = {"models": {"corners": {"L": {"fit_to": "2026-11-12"}}}}
    with pytest.raises(ValueError):
        C.assert_held_out("2026-11-10", serving, fit)                        # compare_fit trained into the window
    with pytest.raises(ValueError):
        C.assert_held_out("2026-11-09", serving, {"models": {}})             # serving trained on the start day
    C.assert_held_out("2026-11-13", serving, fit)                            # both strictly before: fine


def test_metrics_score_only_held_out_games(monkeypatch):
    dates = ["2026-10-01", "2026-11-05", "2026-11-12", "2026-11-20", "2026-11-30"]
    fr = pd.DataFrame({"date": dates, "home_corners": [5, 6, 4, 7, 3], "away_corners": [4, 3, 5, 2, 6]})
    monkeypatch.setattr(C.T, "training_frame", lambda table, lg, fam, f, since: fr.copy())
    monkeypatch.setattr(C.M, "mu_of", lambda prm, f, side: np.full(len(f), prm["mu"]))
    monkeypatch.setattr(C.M, "p_over", lambda mu, alpha, line: np.clip(mu / 20.0, 0.01, 0.99))
    prm = {"total": {"mu": 10.0, "alpha": 0.1}, "home": {"mu": 5.0, "alpha": 0.1}, "away": {"mu": 5.0, "alpha": 0.1}}
    cov = {"corners": {"L": {"window_from": "2022-01-01"}}}
    serving = {"data_end": "2026-11-09", "coverage": cov, "models": {"corners": {"L": {"params": prm}}}}
    fit = {"coverage": cov, "models": {"corners": {"L": {"params": prm, "fit_to": "2026-11-09"}}}}
    cfg = {"features": {}, "families": {"corners": {"markets": {"corners_total": {"part": "total", "lines": [8.5]}}}}}
    m = C.heldout_metrics(pd.DataFrame(), serving, fit, "2026-11-10", "2026-11-23", cfg)
    assert list(m["n"]) == [2]                                   # only 11-12 and 11-20: not before start, not after until
    assert m["logloss_serving"].iloc[0] == m["logloss_candidate"].iloc[0]
    s = C.league_market_summary(m)
    assert list(s["n_games"]) == [2] and s["logloss_delta"].iloc[0] == 0.0


def test_message_carries_no_price_and_says_not_installed():
    s = {"candidate_model_version": "c", "candidate_data_end": "2026-11-23", "serving_model_version": "s",
         "heldout": {"start": "2026-11-10", "end": "2026-11-23"}, "heldout_games_max": 22, "cells": 58,
         "candidate_better_logloss": 26, "candidate_worse_logloss": 20, "mean_logloss_delta": -0.00024,
         "candidate_better_brier": 25, "candidate_worse_brier": 19, "lineage": {"status": "PASS"}}
    t = C.message(s, {"run": True})
    assert "NOT installed" in t and "awaiting owner approval" in t and "odds" not in t.lower()


def test_admin_board_view_reads_summary_and_heldout_table():
    from dashboard_data import side_candidate_view as V
    assert V.parse_summary(None) is None and V.parse_summary("not json") is None
    s = {"candidate_model_version": "c", "heldout": {"start": "a", "end": "b"}, "lineage": {"status": "PASS"}}
    lines = dict(V.candidate_lines(s))
    assert lines["Candidate"] == "c" and lines["Install"].startswith("awaiting owner approval")
    f = V.heldout_frame("family,league,market,n_games,logloss_delta\ncards,L,cards_total,3,-0.01\n")
    assert list(f.columns) == ["family", "league", "market", "n_games", "logloss_delta"]
    assert dict(V.serving_lines(None))["Serving"].startswith("unknown")
