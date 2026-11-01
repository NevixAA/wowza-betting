"""R2-D props recipe scored in shadow: first-sight, never changes a tip (owner decision 2026-11-01).

    python -m pytest tests/test_props_r2_shadow.py -q
"""
from __future__ import annotations

import pandas as pd

from player_model import props_r2_shadow as S


def test_committed_challengers_load():
    ch = S.load_challengers()
    assert {"goals", "sot"} <= set(ch), sorted(ch)


def _rec(tmp_path, p_r2=0.3):
    class _P:   # stand-in payload: the recorder must only call predict_with_diagnostics
        pass
    r = S.Recorder(challengers={"goals": {"recipe": "x"}})
    import player_model.model as M
    orig = M.predict_with_diagnostics
    M.predict_with_diagnostics = lambda df, payload: pd.DataFrame(
        {"p": [p_r2] * len(df), "required_fail": [False] * len(df)}, index=df.index)
    return r, M, orig


def test_first_sight_only_and_no_tip_change(tmp_path):
    r, M, orig = _rec(tmp_path)
    try:
        feat = pd.DataFrame({"player_id": [1, 2], "player_name": ["A", "B"]})
        served = pd.Series([0.2, 0.4]).values
        r.add("goals", feat, served, league="Championship", match_date="2026-11-05", fixture_id=9)
        log = tmp_path / "s.csv"
        assert r.flush(log) == 2
        assert list(served) == [0.2, 0.4]                      # served probabilities untouched
        r2, _, _ = _rec(tmp_path, p_r2=0.9)
        r2.add("goals", feat, served, league="Championship", match_date="2026-11-05", fixture_id=9)
        assert r2.flush(log) == 0                               # second sight never rewrites
        d = pd.read_csv(log)
        assert d["p_r2"].tolist() == [0.3, 0.3]
    finally:
        M.predict_with_diagnostics = orig


def test_report_compares_both_recipes_on_settled_props(tmp_path):
    log, led = tmp_path / "s.csv", tmp_path / "l.csv"
    pd.DataFrame([{"logged_at": "x", "market": "goals", "fixture_id": "9", "player_id": "1", "player_name": "A",
                   "league": "L", "match_date": "2026-11-05", "kickoff_utc": "", "p_served": 0.2, "p_r2": 0.6,
                   "r2_required_fail": False, "r2_version": "v"}]).to_csv(log, index=False)
    pd.DataFrame([{"market": "goals", "player_id": "1", "match_date": "2026-11-05", "result": "WIN"}]).to_csv(led, index=False)
    rep = S.report(log, led)
    m = rep["by_market"]["goals"]
    assert rep["n"] == 1 and m["logloss_r2"] < m["logloss_served"]
