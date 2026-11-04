"""--fullstats (owner 2026-11-04): every cached API-Football /fixtures/statistics value for
af_history, 0 API calls, home/away by team id, a value the provider does not give stays missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STD = "Championship"


def _body(home_id, away_id):
    def side(tid, vals):
        return {"team": {"id": tid}, "statistics": [{"type": k, "value": v} for k, v in vals.items()]}
    home = {"Shots on Goal": 5, "Shots off Goal": 4, "Total Shots": 12, "Offsides": 2,
            "Ball Possession": "58%", "Total passes": 480, "Passes accurate": 401, "Passes %": "84%",
            "Red Cards": None, "expected_goals": "1.37", "Brand New Stat": 9}
    away = {"Shots on Goal": 1, "Shots off Goal": 3, "Total Shots": 6, "Offsides": None,
            "Ball Possession": "42%", "Total passes": 350, "Passes accurate": 270, "Passes %": "77%",
            "Red Cards": 1, "expected_goals": "0.41"}
    # AWAY first on purpose: home/away must come from the team ids, not the response order
    return {"errors": [], "response": [side(away_id, away), side(home_id, home)]}


@pytest.fixture()
def full(tmp_path, monkeypatch):
    import collect_af_history_gap as G
    cdir = tmp_path / "collect"
    cdir.mkdir()
    for name, val in (("STATS_STORE", cdir / "af_fixture_stats.parquet"),
                      ("FULL_FILL", cdir / "af_history_fullstats_fill.parquet"),
                      ("FULL_AUDIT", cdir / "af_fullstats_audit.json")):
        monkeypatch.setattr(G, name, val)
    fx = pd.DataFrame([
        {"fixture_id": 1001, "league_id": 40, "season": "2025", "date": "2026-03-07", "home_id": 1,
         "home_name": "Leeds", "away_id": 2, "away_name": "Hull City"},
        {"fixture_id": 1002, "league_id": 40, "season": "2025", "date": "2026-03-08", "home_id": 3,
         "home_name": "Stoke City", "away_id": 4, "away_name": "Millwall"},
    ])
    fx["_d"] = pd.to_datetime(fx["date"]).dt.normalize()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stats_1001.json").write_text(json.dumps(_body(1, 2)), encoding="utf-8")
    af = pd.DataFrame([
        {"league": STD, "date": pd.Timestamp("2026-03-07"), "home_team": "Leeds", "away_team": "Hull",
         "fixture_id": np.nan, "home_team_id": np.nan, "away_team_id": np.nan},
        # matched, but its answer was never cached -> counted, left missing, NO call
        {"league": STD, "date": pd.Timestamp("2026-03-08"), "home_team": "Stoke", "away_team": "Millwall",
         "fixture_id": np.nan, "home_team_id": np.nan, "away_team_id": np.nan},
    ])
    return G, af, fx, cache


def test_fullstats_places_every_value_by_team_id_and_never_calls(full, monkeypatch):
    G, af, fx, cache = full
    import collect_common as CC
    monkeypatch.setattr(CC.AFClient, "get", lambda *a, **k: pytest.fail("API call made"))
    st = G.run_fullstats(af=af, cache_dir=cache, fixtures=fx)
    fill = pd.read_parquet(G.FULL_FILL)
    assert len(fill) == 1 and st["calls"] == 0
    r = fill.iloc[0]
    assert (r["home_team"], r["away_team"], r["fixture_id"]) == ("Leeds", "Hull", 1001)
    assert (r["home_sot"], r["away_sot"]) == (5, 1)
    assert (r["home_shots_off"], r["away_shots_off"]) == (4, 3)
    assert (r["home_possession"], r["away_possession"]) == (58, 42)
    assert (r["home_passes"], r["home_passes_accurate"], r["home_pass_pct"]) == (480, 401, 84)
    assert r["home_xg"] == pytest.approx(1.37) and r["away_xg"] == pytest.approx(0.41)
    assert r["home_offsides"] == 2


def test_fullstats_null_or_absent_value_stays_missing_never_zero(full):
    G, af, fx, cache = full
    G.run_fullstats(af=af, cache_dir=cache, fixtures=fx)
    r = pd.read_parquet(G.FULL_FILL).iloc[0]
    assert pd.isna(r["home_red"]) and r["away_red"] == 1          # null -> missing, not 0
    assert pd.isna(r["away_offsides"])
    assert pd.isna(r["home_saves"]) and pd.isna(r["away_corners"])  # type absent -> missing


def test_fullstats_audit_counts_uncached_and_unknown_types(full):
    G, af, fx, cache = full
    st = G.run_fullstats(af=af, cache_dir=cache, fixtures=fx)
    assert st["per_league"][STD]["not_cached"] == 1
    assert st["unknown_provider_types"] == ["Brand New Stat"]
    assert "home_brand_new_stat" not in pd.read_parquet(G.FULL_FILL).columns
    audit = json.loads(G.FULL_AUDIT.read_text(encoding="utf-8"))
    assert audit["calls"] == 0


def test_fullstats_refuses_an_answer_without_both_clubs(full):
    G, af, fx, cache = full
    (cache / "stats_1001.json").write_text(json.dumps(_body(1, 99)), encoding="utf-8")
    st = G.run_fullstats(af=af, cache_dir=cache, fixtures=fx)
    assert st["per_league"][STD]["teams_not_in_answer"] == 1
    assert pd.read_parquet(G.FULL_FILL).empty


def test_workflow_runs_fullstats_with_no_api_key():
    y = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    i = y.index("collect_af_history_gap.py --fullstats")
    step = y[y.rindex("- name:", 0, i):i]
    assert "APIFOOTBALL_KEY" not in step and "always()" in step
