"""Owner decision 2026-11-04 (fix round 8, props item 3): re-fetch /fixtures/players for every fixture
behind player_history and store the counts BESIDE it under the owner's null rule.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_player_refetch.py -q

No network: the API-Football client runs on a counting fake session; every output goes to tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_common as CC  # noqa: E402
import collect_player_refetch as RF  # noqa: E402
from collect_common import AFClient  # noqa: E402
from player_model import history_refetch as HR  # noqa: E402
from player_model import player_stat_nulls as PSN  # noqa: E402


class _Resp:
    def __init__(self, body, code=200):
        self.status_code, self._b, self.headers = code, body, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _Session:
    """Counts calls; /status is free. `bodies(fixture_id)` -> the answer (None -> HTTP 500)."""

    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, headers=None, params=None, timeout=None):
        ep = url.rsplit("/v3.football.api-sports.io/", 1)[-1]
        if ep.endswith("status"):
            return _Resp({"response": {"requests": {"current": 0, "limit_day": 75000}}})
        self.calls.append((ep, dict(params or {})))
        b = self.bodies(int(params["fixture"]))
        return _Resp(b) if b is not None else _Resp({}, 500)


def _client(tmp_path, bodies, cap=45_000, budget="budget.json"):
    return AFClient(key="test-not-a-key", session=_Session(bodies), budget_file=tmp_path / budget,
                    daily_cap=cap, pause_s=0)


def _fx():
    return pd.DataFrame([
        {"fixture_id": 100, "league_id": 39, "season": "2025", "date": "2025-08-16T14:00:00+00:00",
         "home_id": 1, "home_name": "Arsenal", "away_id": 2, "away_name": "Chelsea", "status": "FT"},
        {"fixture_id": 200, "league_id": 39, "season": "2025", "date": "2025-08-23T14:00:00+00:00",
         "home_id": 2, "home_name": "Chelsea", "away_id": 3, "away_name": "Everton", "status": "FT"},
        {"fixture_id": 300, "league_id": 39, "season": "2025", "date": "2025-08-30T14:00:00+00:00",
         "home_id": 3, "home_name": "Everton", "away_id": 1, "away_name": "Arsenal", "status": "FT"},
        {"fixture_id": 400, "league_id": 39, "season": "2025", "date": "2025-09-06T14:00:00+00:00",
         "home_id": 1, "home_name": "Arsenal", "away_id": 3, "away_name": "Everton", "status": "FT"},
    ])


def _h(fid, pid, date, home, away, team, is_home, **vals):
    base = {c: 0 for c in PSN.HISTORY_COUNT_COLS}
    base.update(rating=7.0, minutes=90)
    base.update(vals)
    return {"fixture_id": fid, "player_id": pid, "date": pd.Timestamp(date), "home_team": home,
            "away_team": away, "team": team, "is_home": is_home, "league": "Premier League", **base}


def _history():
    return pd.DataFrame([
        _h(100, 11, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1, goals=1, shots_total=3),
        _h(100, 12, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1),           # stored 0s (suspect)
        _h(100, 21, "2025-08-16", "Arsenal", "Chelsea", "Chelsea", 0),
        _h(100, 99, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1),           # not in the answer
        _h(100, 22, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1),           # wrong side
        _h(200, 21, "2025-08-24", "Chelsea", "Everton", "Chelsea", 1),           # date differs
        _h(300, 11, "2025-08-30", "Everton", "Fulham", "Arsenal", 0),            # clubs differ
        _h(400, 11, "2025-09-06", "Arsenal", "Everton", "Arsenal", 1),           # empty answer
        _h(555, 11, "2025-09-10", "England", "Wales", "England", 1),             # not in the lists
    ])


def _p(pid, minutes, **groups):
    return {"player": {"id": pid, "name": f"P{pid}"}, "statistics": [{"games": {"minutes": minutes,
                                                                                "rating": groups.pop("rating", None)},
                                                                      **groups}]}


ANSWER_100 = {"response": [
    {"team": {"id": 1, "name": "Arsenal"}, "players": [
        _p(11, 90, rating="7.4", goals={"total": 1, "assists": None, "conceded": 0, "saves": None},
           shots={"total": 3, "on": 2}, cards={"yellow": 0, "red": 0}, passes={"total": 40, "key": 1}),
        _p(12, 75, goals={"total": None, "assists": None, "conceded": 0, "saves": None},
           shots={"total": None, "on": None}, cards={"yellow": 0, "red": 0}, passes={"total": 20, "key": None}),
        _p(13, None, goals={"total": None})]},                                  # unused sub
    {"team": {"id": 2, "name": "Chelsea"}, "players": [
        _p(21, 90, goals={"total": None, "assists": None, "conceded": 1, "saves": None},
           shots={"total": 1, "on": None}, cards={"yellow": 1, "red": 0}, passes={"total": 50, "key": 2}),
        _p(22, 30, goals={"total": None}, shots={"total": None})]}]}


def _bodies(fid):
    return ANSWER_100 if fid == 100 else {"response": []}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cdir = tmp_path / "collect"
    for n in ("OUT", "JOURNAL", "AUDIT"):
        monkeypatch.setattr(RF, n, cdir / Path(getattr(RF, n)).name)
    monkeypatch.setattr(CC, "BUDGET_FILE", cdir / "af_budget.json")
    return tmp_path


def _rows(history=None, cups=None):
    return RF.source_rows(_history() if history is None else history,
                          pd.DataFrame() if cups is None else cups)


def test_plan_proves_fixture_identity_and_counts_every_refusal(env):
    p = RF.plan(_rows(), _fx(), pd.DataFrame())
    assert p["fixtures"] == 5 and p["proven_fixtures"] == 2 and p["planned_calls"] == 2
    assert p["refused_fixtures"] == {"fixture_not_in_lists": 1, "date_differs_from_list": 1,
                                     "clubs_unproven": 1}
    assert p["refused_rows"]["fixture_not_in_lists"] == 1
    assert p["todo"]["fixture_id"].tolist() == [400, 100]                     # newest first
    st = RF.run(dry_run=True, rows=_rows(), fx=_fx())
    assert st["planned_calls"] == 2 and st["calls"] == 0
    assert not RF.OUT.exists() and not RF.JOURNAL.exists()                    # a dry run writes nothing


def test_row_without_fixture_id_is_mapped_only_through_a_proven_list_fixture(env):
    h = _history()
    h.loc[1, "fixture_id"] = None                                             # 2025-08-16 Arsenal v Chelsea
    h.loc[8, "fixture_id"] = None                                             # England v Wales: no list fixture
    p = RF.plan(_rows(h), _fx(), pd.DataFrame())
    assert p["rows_without_fixture_id"] == 2 and p["rows_mapped_through_lists"] == 1
    assert p["refused_rows"]["no_fixture_id_unmapped"] == 1
    assert 100 in set(p["rows_frame"].loc[p["rows_frame"]["player_id"] == 12, "fixture_id"])


def test_run_applies_the_null_rule_keeps_originals_and_journals(env):
    client = _client(env, _bodies)
    st = RF.run(client=client, rows=_rows(), fx=_fx())
    assert st["calls"] == 2 and st["fetched"] == 2 and st["empty_answers"] == 1
    assert st["rows_refused"] == {"player_not_in_answer": 1, "side_unproven": 1, "empty_answer": 1}
    out = pd.read_parquet(RF.OUT).set_index("player_id")
    assert sorted(out.index) == [11, 12, 21]
    # 12 played 75', goals / shots null, the fixture has them for others -> 0; nobody has saves -> missing
    assert out.loc[12, "goals"] == 0 and out.loc[12, "shots_on_target"] == 0
    assert pd.isna(out.loc[12, "saves"]) and out.loc[12, "passes_total"] == 20
    assert out.loc[12, "orig_saves"] == 0 and pd.isna(out.loc[12, "rating"]) and out.loc[12, "orig_rating"] == 7.0
    assert set(out.loc[12, "filled_cols"].split(",")) >= {"goals", "shots_total", "shots_on_target"}
    assert out.loc[11, "goals"] == 1 and out.loc[11, "rating"] == 7.4
    assert pd.isna(out.loc[11, "assists"])                                    # nobody in the fixture has it
    assert out.loc[12, "key_passes"] == 0 and "key_passes" in out.loc[12, "filled_cols"]
    assert out.loc[21, "is_home"] == 0 and out.loc[21, "yellow_cards"] == 1
    j = pd.read_parquet(RF.JOURNAL).set_index("fixture_id")
    assert j.loc[100, "status"] == "ok" and j.loc[400, "status"] == "empty"
    a = json.loads(RF.AUDIT.read_text(encoding="utf-8"))
    assert a["stored_rows"] == 3 and a["journal_by_status"] == {"ok": 1, "empty": 1}
    f = a["fields"]["saves"]
    assert f["missing"] == 3 and f["orig_zero_now_missing"] == 3
    assert a["fields"]["goals"]["null_read_as_zero"] == 2 and a["fields"]["goals"]["sent_value"] == 1
    # resumable: both fixtures journaled (the empty answer too) -> nothing asked again
    again = _client(env, _bodies)
    st2 = RF.run(client=again, rows=_rows(), fx=_fx())
    assert st2["planned_calls"] == 0 and again.session.calls == []


def test_failed_call_records_nothing_and_budget_stop_is_clean(env):
    st = RF.run(client=_client(env, lambda fid: None), rows=_rows(), fx=_fx())
    assert st["fetched"] == 0 and not RF.JOURNAL.exists()
    assert RF.plan(_rows(), _fx(), pd.DataFrame())["planned_calls"] == 2       # asked again next run
    c = _client(env, _bodies, cap=1, budget="budget_capped.json")
    st = RF.run(client=c, rows=_rows(), fx=_fx())
    assert "budget_stop" in st and st["fetched"] == 1 and len(c.session.calls) == 1
    assert len(pd.read_parquet(RF.JOURNAL)) == 1


def test_cup_rows_need_their_team_id(env):
    cups = pd.DataFrame([dict(_h(100, 11, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1), team_id=1),
                         dict(_h(100, 21, "2025-08-16", "Arsenal", "Chelsea", "Chelsea", 0), team_id=7)])
    RF.run(client=_client(env, _bodies), rows=RF.source_rows(pd.DataFrame(), cups), fx=_fx())
    out = pd.read_parquet(RF.OUT)
    assert out["player_id"].tolist() == [11] and out["source"].tolist() == ["cup_rows"]
    assert pd.read_parquet(RF.JOURNAL)["n_refused"].sum() == 1


def test_reader_replaces_the_suspect_zero_and_marks_the_rest(env):
    RF.run(client=_client(env, _bodies), rows=_rows(), fx=_fx())
    h = _history()
    got = HR.with_refetched(h, pd.read_parquet(RF.OUT))
    assert len(got) == len(h) and (got.index == h.index).all()
    by = got.set_index(["fixture_id", "player_id"])
    assert by.loc[(100, 12), HR.STATUS_COL] == HR.REFETCHED
    assert pd.isna(by.loc[(100, 12), "saves"])                                # 0 -> missing (replaced)
    assert by.loc[(100, 12), "goals"] == 0 and pd.isna(by.loc[(100, 12), "rating"])
    assert by.loc[(400, 11), HR.STATUS_COL] == HR.UNVERIFIED and by.loc[(400, 11), "saves"] == 0
    assert by.loc[(100, 99), HR.STATUS_COL] == HR.UNVERIFIED
    assert h["saves"].eq(0).all()                                             # the input is not touched
    # no re-fetch stored yet -> nothing changes, every row unverified
    none = HR.with_refetched(h, pd.DataFrame())
    assert (none[HR.STATUS_COL] == HR.UNVERIFIED).all() and none["saves"].eq(0).all()


def test_predict_reads_history_through_the_refetch_reader():
    src = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    i = src.index("def mode_predict")
    body = src[i:src.index("\ndef ", i + 10)]
    assert body.index("with_cup_rows(pd.read_parquet(HISTORY_CACHE))") < body.index("history = with_refetched(history)")


def test_workflow_runs_playerrefetch_after_lineups_before_fullstats_and_strength():
    y = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    pos = {k: y.rindex(s) for k, s in (("lineups", "python scripts/collect_lineups.py"),
                                       ("refetch", "python scripts/collect_player_refetch.py"),
                                       ("fullstats", "python scripts/collect_af_history_gap.py --fullstats"),
                                       ("strength", "python -m src.strength.build"))}
    assert pos["lineups"] < pos["refetch"] < pos["fullstats"] < pos["strength"]
    step = y[y.rindex("- name:", 0, pos["refetch"]):pos["refetch"]]
    assert "contains(env.STEPS, 'playerrefetch')" in step and "APIFOOTBALL_KEY" in step
    assert "always()" not in step                                             # an API step: not after a stop
    lines = [ln for ln in y.splitlines() if ("default:" in ln or "description:" in ln) and "fixtures" in ln
             or ln.strip().startswith("STEPS:")]
    assert len(lines) == 3 and all("lineups,playerrefetch,strength" in ln for ln in lines)


def test_collector_writes_only_under_output_collect():
    for p in (RF.OUT, RF.JOURNAL, RF.AUDIT):
        assert Path(p).parent == ROOT / "output" / "collect"
    assert Path(HR.REFETCH_FILE) == ROOT / "output" / "collect" / "player_history_refetch.parquet"
