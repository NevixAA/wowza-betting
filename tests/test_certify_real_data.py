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


def _audit(df, kind="provider", documented=True, wrong=frozenset(), private=False):
    ta = C._TableAudit("t", kind, documented, set(wrong))
    ta.add(df)
    return ta.finish(private=private)


def test_row_level_rules_fire_and_reclassify_cells():
    """Each named rule moves the cells it hits out of the observed / derived classes; missing stays missing."""
    n = 120
    df = pd.DataFrame({
        "fixture_id": range(n), "bookmaker": ["Bet365"] * n, "source": ["api_football:odds"] * n,
        "kickoff_utc": ["2026-11-01T15:00:00Z"] * n, "snapshot_ts": ["2026-11-01T12:00:00Z"] * n,
        "odds": [1.5 + i / 100 for i in range(n - 2)] + [1.0, None], "home_goals": [i % 5 for i in range(n - 1)] + [-1],
        "p_model": [i / 200 for i in range(n - 1)] + [1.2], "home_team": ["A"] * (n - 1) + ["unknown"],
        "corners_pg": [4.0] * n})
    df.loc[5, "snapshot_ts"] = "2026-11-01T15:30:00Z"                       # captured after kickoff
    r = _audit(df)
    rules = r["rules"]
    assert rules[C.R_PRICE]["rows"] == 1 and rules[C.R_NEG]["rows"] == 1 and rules[C.R_PROB]["rows"] == 1
    assert rules[C.R_SENT]["rows"] == 1
    assert rules[C.R_POSTKO]["rows"] == 1 and rules[C.R_POSTKO]["cells"] > 0
    assert rules[C.R_CONST]["columns"] == ["corners_pg"]
    assert r["cells"][C.MIS] == 1                                         # the missing price: kept missing
    assert r["verdict"] == C.NOT_REAL
    assert sum(r["cells"].values()) == n * df.shape[1]                    # every cell in exactly one class


def test_duplicates_identity_flags_and_documentation():
    df = pd.DataFrame({"fixture_key": ["k1", "k1", "k2"], "run_id": ["r", "r", "r"], "market": ["ou", "ou", "ou"],
                       "odds": [2.0, 2.1, 2.0], "bookmaker": ["v9_capture"] * 3,
                       "quality_flags": ["", "POST_KICKOFF_PRICE", "FEATURE_DEGRADED"], "model_x": [1.0, 2.0, 3.0]})
    r = _audit(df, kind="derived", documented=False, wrong={"POST_KICKOFF_PRICE"})
    assert r["rules"][C.R_DUP]["rows"] == 1 and r["rules"][C.R_DUP]["conflicting_rows"] == 1
    assert r["rules"][C.R_NOID]["rows"] == 2      # a synthetic book label is no identity (row 2: already WRONG)
    assert r["rules"][C.R_WRONG]["rows"] == 1 and r["rules"][C.R_IMPUTED]["rows"] == 1
    assert r["rules"][C.R_NODOC]["rows"] == 3
    clean = _audit(pd.DataFrame({"fixture_id": [1, 2], "source": ["api_football:x"] * 2, "home_goals": [1, None]}))
    assert clean["verdict"] == C.REAL_WITH_GAPS


def test_private_scope_writes_no_column_names():
    df = pd.DataFrame({"fixture_id": range(150), "bookmaker": ["B"] * 150, "odds": [2.0 + i / 100 for i in range(150)],
                       "home_corners": [3.0] * 150})
    r = _audit(df, private=True)
    assert "key_columns" not in r
    assert r["rules"][C.R_CONST]["columns"] == 1                           # a count, never a name


def test_external_scopes_without_checkouts_are_unverified(tmp_path):
    assert C.pro_scope(None)["verdict"] == C.UNVERIFIED
    assert C.v11_scope(tmp_path)["verdict"] == C.UNVERIFIED
    assert C.exec_scope(tmp_path)["verdict"] == C.UNVERIFIED


def test_exec_table_grouping():
    assert C._exec_table("data/odds/20261110-3.parquet") == "data/odds"
    assert C._exec_table("wowzaV9-Pro/main/data/season_2026_27/clv/dt=2026-11-01/run=x.parquet") == \
        "wowzaV9-Pro/main/data/season_2026_27/clv"
    assert C._exec_table("prediction_ledger/v11/v11_shadow/ou25/la_liga_2/2026/x.csv") == \
        "prediction_ledger/v11/v11_shadow/ou25"
    assert C._exec_table("wowza-betting/main/output/bets_ledger.csv") == "wowza-betting/main/output/bets_ledger"
