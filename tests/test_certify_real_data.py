"""scripts/certify_real_data.py - the row-level real-data certification (owner 2026-11-23)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("certify_real_data", ROOT / "scripts" / "certify_real_data.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_verdict_never_passes_a_default_or_an_unverified_value():
    z = C._z()
    assert C._verdict(z) == C.REAL_100
    assert C._verdict({**z, C.MIS: 3}) == C.REAL_WITH_GAPS
    assert C._verdict({**z, C.OBS: 10, C.DEF: 1}) == C.NOT_REAL
    assert C._verdict({**z, C.OBS: 10, C.UNV: 1}) == C.NOT_REAL
    assert C._verdict({**z, C.OBS: 10}, measured=False) == C.UNVERIFIED


def test_untraced_team_feature_is_not_counted_real():
    assert C._team_class("home_scored_last5") in (C.OBS, C.DER)
    assert C._team_class("home_scored_last5__missing") == C.DER
    assert C._team_class("a_feature_nobody_traced") == "UNTRACED"


def test_archive_rows_are_checked_against_the_providers(monkeypatch, tmp_path):
    """A backtest_archive stat that API-Football contradicts, or that no provider holds, is never counted as
    agreeing; the later rows of both clubs are counted as reading it."""
    out = tmp_path / "output"
    out.mkdir()
    pd.DataFrame({"league": ["L"], "date": ["2022-10-23"], "home_team": ["A"], "away_team": ["B"],
                  "HS": [9.0], "AS": [5.0], "HST": [None], "AST": [None], "HC": [None], "AC": [None],
                  "HF": [None], "AF": [None]}).to_parquet(out / "af_history.parquet")
    pd.DataFrame({"league": ["X"], "date": ["2022-01-01"], "home_team": ["Y"], "away_team": ["Z"],
                  "home_shots": [1.0]}).to_parquet(out / "fd_history.parquet")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    feat = pd.DataFrame({
        "league": ["L"] * 3, "date": ["2022-10-23", "2022-10-30", "2022-11-06"],
        "home_team": ["A", "A", "C"], "away_team": ["B", "C", "B"],
        "row_source": ["backtest_archive", "fd_cache", "fd_cache"],
        "home_shots": [11.0, 1, 1], "away_shots": [5.0, 1, 1], "home_sot": [4.0, 1, 1], "away_sot": [None, 1, 1],
        "home_corners": [7.0, 1, 1], "away_corners": [None, 1, 1], "home_fouls": [None, 1, 1],
        "away_fouls": [None, 1, 1]})
    r = C._archive_rows(feat, feat)
    assert r["rows"] == 1
    assert r["stat_cells_agree_with_a_provider"] == 1          # away shots 5 = 5
    assert r["stat_cells_contradicted_by_a_provider"] == 1     # home shots 11 vs 9
    assert r["stat_cells_no_provider_holds"] == 2              # home SOT, home corners
    assert r["downstream_training_rows_reading_them"] == 2


def test_committed_certification_holds_no_price_value():
    p = ROOT / "output" / "real_data_certification.json"
    if not p.exists():
        return
    doc = json.loads(p.read_text(encoding="utf-8"))
    bad = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if any(t in lk for t in ("odds", "price", "stake", "probab")) and isinstance(v, (int, float)) \
                        and not isinstance(v, bool) and not lk.endswith(("_cells", "rows")):
                    bad.append(f"{path}/{k}")
                walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(doc)
    assert not bad, bad
    assert set(doc["classes"]) == set(C.CLASSES)


def test_unmeasured_repos_are_unverified_never_a_pass():
    r = C.external_repos()
    assert r["verdict"] == C.UNVERIFIED
    assert all(v["verdict"] == C.UNVERIFIED for v in r["repos"].values())
