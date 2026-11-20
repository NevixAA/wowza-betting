"""Prediction ledger coverage (owner rule 2026-11-17, reconfirmed 2026-11-18): every model predicts every match
of every tracked league for every market it can. Pins that
  * every v9 team market registered for one track is registered for the other track too (the New-Format
    HT O/U 0.5 / 1.5 entries are active since the owner connected the New-Format HT models on 2026-11-20);
  * the per-run coverage check names every board fixture that got no row, with its reason, and finds no
    fixture missing an active market of its own track;
  * the check runs only behind the ledger flag (flag off: nothing written).
Synthetic boards only (tmp_path); no committed ledger is touched.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src import prediction_ledger as PL
from src import prediction_ledger_v9 as V

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def _r(**over):
    r = {"league": "Championship", "date": "2026-11-21", "kickoff_utc": "2026-11-21 15:00:00",
         "home_team": "Norwich City", "away_team": "Cardiff City", "_oa_event_id": "e1",
         "home_scored_last5": 1.2, "table_pos_gap": 3, "serving_recipe": "legacy_v9",
         "p_over25": 0.6, "model_type": "standard", "p_btts": 0.5, "p_over15": 0.8, "p_over35": 0.3,
         "p_ht_over05": 0.7, "p_ht_over15": 0.2, "generated_at": "2026-11-17T12:00:00Z", "git_sha": "abc",
         "model_version": "m", "model_version_btts": "b", "model_version_over15": "o15",
         "model_version_over35": "o35", "model_version_ht_over05": "h05", "model_version_ht_over15": "h15"}
    r.update(over)
    return r


def _board():
    return pd.DataFrame([
        _r(),
        _r(league="Norway Eliteserien", home_team="Molde", away_team="Brann", _oa_event_id="e2",
           model_type="new_format", p_ht_over05=0.66, p_ht_over15=0.3, model_version_ht_over05="h05nf",
           model_version_ht_over15="h15nf"),
        _r(home_team="Hull City", away_team="Derby County", _oa_event_id="e3", kickoff_utc=None),
        _r(home_team="Leeds United", away_team="Stoke City", _oa_event_id="e4", model_type="unknown"),
        _r(home_team="Millwall", away_team="Watford", _oa_event_id="e5", kickoff_utc="2026-11-17 11:00:00"),
    ])


def test_every_team_market_is_registered_for_both_tracks():
    ents = [e for e in PL.entries(repo="v9") if e.get("family") == "team"]
    by = {}
    for e in ents:
        by.setdefault(e["model_id"], {})[e["market"]] = e
    assert set(by["v9_standard"]) == set(by["v9_new_format"])
    for m in ("ht_ou05", "ht_ou15"):
        nf = by["v9_new_format"][m]
        assert nf["track"] == "new_format" and nf["settle"]["period"] == "ht"
        # owner 2026-11-20: the New-Format HT model serves New-Format rows (its own file)
        assert nf["status"] == "active" and nf["prob_col"] == by["v9_standard"][m]["prob_col"]
        assert nf["model_file"]["*"].endswith("_nf.pkl")
        assert nf["model_file"] != by["v9_standard"][m]["model_file"]


def test_coverage_names_every_dropped_fixture_and_misses_no_market():
    b = _board()
    cov: dict = {}
    rows = V.team_rows(b, now=NOW, coverage=cov)
    rep = V.coverage_report(b, rows, cov["dropped"])
    assert rep["board_fixtures"] == 5 and rep["fixtures_with_rows"] == 2
    why = {d["home_team"]: d["reason"] for d in rep["dropped"]}
    assert set(why) == {"Hull City", "Leeds United", "Millwall"}
    assert "kickoff_utc" in why["Hull City"]
    assert "not a v9 track" in why["Leeds United"]
    assert "after kickoff" in why["Millwall"]
    assert rep["missing_cells_n"] == 0
    # every active market of each track got rows; New-Format HT rows carry the New-Format HT versions
    d = pd.DataFrame(rows)
    active = rep["active_team_entries"]
    for mid, g in d.groupby("model_id"):
        assert set(g["market"]) == set(active[mid])
    nf_ht = d[(d["model_id"] == "v9_new_format") & d["market"].str.startswith("ht_")]
    assert set(nf_ht["model_version"]) == {"h05nf", "h15nf"} and (nf_ht["scored"] == 1).all()
    # tracked leagues with no board fixture are listed with a reason
    assert all(x["reason"] for x in rep["tracked_leagues_without_board_fixtures"])
    assert "Championship" not in {x["league"] for x in rep["tracked_leagues_without_board_fixtures"]}
    assert json.dumps(rep, default=str)                                  # serialisable, statistics only
    assert not any(k in json.dumps(rep, default=str) for k in ("odds_", "price", "stake"))


def test_coverage_is_written_only_behind_the_flag(tmp_path, monkeypatch):
    p = tmp_path / "predictions.csv"
    _board().to_csv(p, index=False)
    monkeypatch.delenv(PL.FLAG_ENV, raising=False)
    assert "skipped" in V.record_board(p, root=tmp_path / "L")
    assert not (tmp_path / "L").exists()
    st = V.record_board(p, root=tmp_path / "L", now=NOW, force=True)
    assert st["coverage"] == {"board_fixtures": 5, "fixtures_with_rows": 2, "dropped_n": 3, "missing_cells_n": 0}
    f = tmp_path / "L" / "v9" / "_coverage" / "team_board" / "2026-11.jsonl"
    rec = json.loads(f.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["dropped_n"] == 3
    assert PL.validate(tmp_path / "L") == []                           # the coverage file is no partition
