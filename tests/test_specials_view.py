"""Specials (research) page helpers: they only re-shape Pro's stored rows, never invent a number."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard_data import specials_view as V


def _row(**kw):
    r = {"row_id": "r1", "league": "A", "home_team": "H", "away_team": "W",
         "kickoff_utc": "2030-01-02T15:00:00Z", "logged_at": "2030-01-01T07:53:00Z", "n_league_history": 900,
         "cs_unusual_pick": "cs_4_3", "p_cs_unusual": 0.01, "base_cs_unusual": 0.002, "ratio_cs_unusual": 5.0}
    for i in range(7):
        for j in range(7):
            r[f"p_cs_{i}_{j}"], r[f"base_cs_{i}_{j}"] = 0.02, 0.02
    r["p_cs_other"], r["base_cs_other"] = 0.02, 0.02
    for k in list(V.BAND_LABELS) + list(V.HIGH) + list(V.LOW):
        r[f"p_{k}"], r[f"base_{k}"] = 0.10, 0.05
        r[f"ratio_{k}"] = 2.0
        r[f"cand_{k}"] = 0.0
    r["cand_cs_unusual"] = 0.0
    r.update(kw)
    return r


NOW = pd.Timestamp("2030-01-01T12:00:00Z")


def test_flagged_frame_lists_only_upcoming_flagged_rows_with_their_logged_numbers():
    log = pd.DataFrame([_row(cand_ge6=1.0, p_ge6=0.21, base_ge6=0.07, ratio_ge6=3.0),
                        _row(row_id="r2", kickoff_utc="2029-12-30T15:00:00Z", cand_ge6=1.0)])
    f = V.flagged_frame(log, NOW, V.HIGH)
    assert len(f) == 1 and f.iloc[0]["special"] == "6+ goals"
    assert f.iloc[0]["model"] == 0.21 and f.iloc[0]["league base"] == 0.07 and f.iloc[0]["ratio"] == 3.0


def test_unusual_score_row_names_the_score():
    log = pd.DataFrame([_row(cand_cs_unusual=1.0)])
    f = V.flagged_frame(log, NOW, ("cs_unusual",))
    assert f.iloc[0]["score"] == "4-3" and f.iloc[0]["ratio"] == 5.0


def test_grid_is_the_full_0_to_6_square():
    g = V.grid_frame(pd.Series(_row()))
    assert len(g) == 49 and set(g["score"]) >= {"0-0", "6-6", "4-3"}


def test_graded_frame_is_insufficient_below_the_floor():
    log = pd.DataFrame([_row(cand_nil_nil=1.0)])
    settled = pd.DataFrame([{"row_id": "r1", "y_nil_nil": 1}])
    g = V.graded_frame(log, settled, min_events=30)
    assert g.iloc[0]["events"] == 1 and g.iloc[0]["status"] == "INSUFFICIENT"


def test_empty_inputs_give_empty_frames_not_samples():
    assert V.flagged_frame(pd.DataFrame(), NOW, V.HIGH).empty
    assert V.graded_frame(pd.DataFrame(), pd.DataFrame(), 30).empty
    assert V.study_table({}).empty
    assert V.calibration_frame(pd.DataFrame(), "A").empty


def test_page_renders_with_stored_rows(monkeypatch, tmp_path):
    """The page over a Pro-shaped folder (rows written here, in the test) renders without an exception."""
    import json
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from dashboard_data import core
    d = tmp_path / "output" / "specials"
    d.mkdir(parents=True)
    ko = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame([_row(kickoff_utc=ko, cand_ge6=1.0, cand_nil_nil=1.0, cand_cs_unusual=1.0),
                  _row(row_id="r2", kickoff_utc="2026-01-01T15:00:00Z", cand_ge6=1.0)]).to_csv(
        d / "specials_log.csv", index=False)
    pd.DataFrame([{"row_id": "r2", "y_ge6": 0, "home_goals": 1, "away_goals": 1}]).to_csv(
        d / "specials_settled.csv", index=False)
    rule = {"version": "SPC-1", "candidate_quantile": 0.9, "min_candidate_events": 30}
    (d / "study_summary.json").write_text(json.dumps({"rule": rule, "rows": 10, "leagues": 1,
                                                     "skill": {}, "candidates": {}}), encoding="utf-8")
    pd.DataFrame([{"league": "A", "score": "0-0", "n": 10, "events": 1, "mean_p": 0.07, "actual": 0.1,
                   "base_mean": 0.08}]).to_csv(d / "study_calibration_grid.csv", index=False)
    monkeypatch.setitem(core.REPOS["pro"], "candidates", [tmp_path])
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "pages" / "23_🎲_Specials.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
