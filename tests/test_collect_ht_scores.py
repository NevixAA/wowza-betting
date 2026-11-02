"""HT data gap (Data Truth round 2): scripts/collect_ht_scores.py fetches API-Football half-time
scores for new-format league-seasons the history lacks and writes them BESIDE fd_history.

    python -m pytest -p no:cacheprovider tests/test_collect_ht_scores.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_ht_scores as H  # noqa: E402


def _fx(fid, date, home, away, ht=(1, 0), status="FT"):
    return {"fixture": {"id": fid, "date": f"{date}T19:30:00+00:00", "status": {"short": status}},
            "teams": {"home": {"id": fid * 10, "name": home}, "away": {"id": fid * 10 + 1, "name": away}},
            "score": {"halftime": {"home": ht[0] if ht else None, "away": ht[1] if ht else None},
                      "fulltime": {"home": 2, "away": 1}}}


def _hist():
    return pd.DataFrame([
        # lacking HT, same-day match
        {"league": "USA MLS", "date": "2026-08-01", "home_team": "Columbus Crew",
         "away_team": "Inter Miami", "ht_home_goals": np.nan, "ht_away_goals": np.nan},
        # lacking HT, provider dates the kick-off the next UTC day
        {"league": "USA MLS", "date": "2026-08-02", "home_team": "LA Galaxy",
         "away_team": "Seattle Sounders", "ht_home_goals": np.nan, "ht_away_goals": np.nan},
        # already has HT: never touched
        {"league": "USA MLS", "date": "2026-08-01", "home_team": "Orlando City",
         "away_team": "Chicago Fire", "ht_home_goals": 0.0, "ht_away_goals": 0.0},
        # lacking HT, but two provider fixtures fit (a duplicated record): refused
        {"league": "USA MLS", "date": "2026-08-09", "home_team": "Austin",
         "away_team": "Dallas", "ht_home_goals": np.nan, "ht_away_goals": np.nan},
        # a standard league row: out of scope
        {"league": "League One", "date": "2026-08-01", "home_team": "A", "away_team": "B",
         "ht_home_goals": np.nan, "ht_away_goals": np.nan},
    ])


class _FakeClient:
    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        return self.bodies.get((params["league"], params["season"]), {"response": []})

    def flush(self):
        pass


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    for name in ("SCORES", "FILL", "AUDIT", "JOURNAL"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    return tmp_path


def test_parse_keeps_finished_fixtures_and_never_invents_a_halftime_score():
    body = {"response": [_fx(1, "2026-08-01", "A", "B"), _fx(2, "2026-08-01", "C", "D", ht=None),
                         _fx(3, "2026-08-01", "E", "F", status="NS"),
                         _fx(4, "2026-08-01", "G", "H", status="PST")]}
    rows = H.parse(body, "USA MLS", 253, "2026")
    assert [r["fixture_id"] for r in rows] == [1, 2]
    assert (rows[0]["ht_home"], rows[0]["ht_away"]) == (1, 0)
    assert rows[1]["ht_home"] is None and rows[1]["ht_away"] is None


def test_plan_is_one_call_per_league_season_lacking_ht():
    lack = H.lacking_rows(_hist())
    assert set(lack["canon"]) == {"USA MLS"} and len(lack) == 3
    p = H.plan(lack, pd.DataFrame())
    assert [(x["league_id"], x["season"]) for x in p] == [(253, "2026")]
    # journaled after the latest lacking match: not fetched again
    j = pd.DataFrame([{"league_id": 253, "season": "2026", "fetched_at": "2026-09-01T00:00:00+00:00"}])
    assert H.plan(lack, j) == []


def test_run_fills_beside_the_history_with_ambiguity_refusal(paths):
    body = {"response": [_fx(11, "2026-08-01", "Columbus Crew", "Inter Miami CF", ht=(2, 1)),
                         _fx(12, "2026-08-03", "LA Galaxy", "Seattle Sounders", ht=(0, 1)),
                         _fx(13, "2026-08-01", "Orlando City", "Chicago Fire", ht=(3, 3)),
                         _fx(14, "2026-08-09", "Austin", "Dallas", ht=(1, 1)),
                         _fx(15, "2026-08-09", "Austin", "Dallas", ht=(0, 0))]}
    hist = _hist()
    before = hist.copy()
    client = _FakeClient({(253, "2026"): body})
    st = H.run(client=client, history=hist)
    assert client.calls == [("fixtures", {"league": 253, "season": "2026"})] and st["calls"] == 1
    fill = pd.read_parquet(H.FILL)
    got = {(r.home_team, r.away_team): (r.HTHG, r.HTAG, r.fixture_id, r.match_rule)
           for r in fill.itertuples()}
    assert got == {("Columbus Crew", "Inter Miami"): (2, 1, 11, "same_day"),
                   ("LA Galaxy", "Seattle Sounders"): (0, 1, 12, "adjacent_day")}
    pd.testing.assert_frame_equal(hist, before)                    # the history is not rewritten
    audit = json.loads(H.AUDIT.read_text(encoding="utf-8"))
    mls = audit["per_league"]["USA MLS"]
    assert mls["lacking"] == 3 and mls["filled"] == 2 and mls["refused_ambiguous"] == 1
    assert pd.read_parquet(H.SCORES)["fixture_id"].nunique() == 5


def test_dry_run_makes_no_call(paths):
    client = _FakeClient({})
    st = H.run(dry_run=True, client=client, history=_hist())
    assert st["planned_calls"] == 1 and client.calls == []


def test_a_fixture_claimed_by_two_history_rows_fills_neither():
    lack = H.lacking_rows(pd.DataFrame([
        {"league": "USA MLS", "date": d, "home_team": "Columbus Crew", "away_team": "Inter Miami",
         "ht_home_goals": np.nan, "ht_away_goals": np.nan} for d in ("2026-08-01", "2026-08-02")]))
    scores = pd.DataFrame(H.parse({"response": [_fx(11, "2026-08-01", "Columbus Crew", "Inter Miami")]},
                                  "USA MLS", 253, "2026"))
    fill, stats = H.match(lack, scores)
    assert fill.empty and stats["USA MLS"]["refused_fixture_claimed_twice"] == 2


def test_one_match_under_two_history_spellings_gets_the_score_on_both_rows():
    # the history carries MLS 2022 games twice: 'Austin' and 'Austin FC' on the same date
    lack = H.lacking_rows(pd.DataFrame([
        {"league": "USA MLS", "date": "2022-02-26", "home_team": h, "away_team": "FC Cincinnati",
         "ht_home_goals": np.nan, "ht_away_goals": np.nan} for h in ("Austin", "Austin FC")]))
    scores = pd.DataFrame(H.parse({"response": [_fx(816721, "2022-02-26", "Austin", "FC Cincinnati",
                                                    ht=(0, 1))]}, "USA MLS", 253, "2022"))
    fill, stats = H.match(lack, scores)
    assert sorted(fill["home_team"]) == ["Austin", "Austin FC"] and set(fill["HTAG"]) == {1}
    assert stats["USA MLS"]["history_rows_same_match_two_spellings"] == 2


def test_workflow_runs_the_ht_step():
    wf = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    assert "python scripts/collect_ht_scores.py" in wf and "contains(env.STEPS, 'htscores')" in wf
