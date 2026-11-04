"""Owner decision 2026-11-04: injuries / lineups / coaches / transfers history from API-Football.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_collect_newdata.py -q

No network: the API-Football client is a counting fake (as tests/test_team6_round6.py); every
collector output goes to tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_af_lists as AL  # noqa: E402
import collect_coaches as CO  # noqa: E402
import collect_common as CC  # noqa: E402
import collect_injuries as INJ  # noqa: E402
import collect_lineups as LU  # noqa: E402
import collect_transfers as TR  # noqa: E402
from collect_common import AFClient  # noqa: E402

REG = 40          # Championship's api_football_id
CUP = 11          # a continental cup outside the registry


class _Resp:
    def __init__(self, body):
        self.status_code, self._b, self.headers = 200, body, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _Session:
    """Counts calls; /status is free."""

    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, headers=None, params=None, timeout=None):
        ep = url.rsplit("/v3.football.api-sports.io/", 1)[-1]
        if ep.endswith("status"):
            return _Resp({"response": {"requests": {"current": 0, "limit_day": 75000}}})
        self.calls.append((ep, dict(params or {})))
        return _Resp(self.bodies(ep, params))


def _fx():
    rows = [
        (1, REG, "2024", "2024-08-10", 10, 20, "FT"),
        (2, REG, "2024", "2024-08-17", 20, 30, "FT"),
        (3, REG, "2025", "2025-08-09", 30, 10, "FT"),
        (4, REG, "2025", "2025-08-16", 10, 40, "AET"),
        (9, CUP, "2025", "2025-09-01", 50, 10, "FT"),       # not a registry league
    ]
    return pd.DataFrame([{"fixture_id": f, "league_id": l, "season": s, "date": d, "home_id": h,
                          "away_id": a, "status": st} for f, l, s, d, h, a, st in rows])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cdir = tmp_path / "collect"
    monkeypatch.setattr(AL, "COVERAGE", cdir / "af_league_coverage.parquet")
    monkeypatch.setattr(AL, "registry_af_ids", lambda: {REG})
    for mod, names in ((INJ, ("ROWS", "SEASON_JOURNAL", "FIXTURE_JOURNAL")),
                       (LU, ("JOURNAL", "TEAMS", "PLAYERS")), (CO, ("JOURNAL", "CAREERS")),
                       (TR, ("JOURNAL", "ROWS"))):
        for n in names:
            monkeypatch.setattr(mod, n, cdir / Path(getattr(mod, n)).name)
    monkeypatch.setattr(CC, "BUDGET_FILE", cdir / "af_budget.json")
    fxp = tmp_path / "competition_fixtures.parquet"
    _fx().to_parquet(fxp)
    monkeypatch.setattr(AL, "FIXTURES", fxp)
    return tmp_path


def _client(tmp, bodies, cap=45_000):
    sess = _Session(bodies)
    return AFClient(key="k", session=sess, budget_file=tmp / "collect" / "af_budget.json",
                    daily_cap=cap, pause_s=0), sess


def _coverage(lineups=True, injuries=True):
    return {"errors": [], "response": [{"seasons": [
        {"year": 2024, "coverage": {"injuries": injuries, "fixtures": {"lineups": lineups}}},
        {"year": 2025, "coverage": {"injuries": True, "fixtures": {"lineups": True}}}]}]}


# ── shared plan ─────────────────────────────────────────────────────────────────────────────────

def test_team_ids_cover_every_fixture_list_team_newest_first(env):
    assert AL.team_ids() == [10, 50, 40, 30, 20]        # 10 and 50 last played 2025-09-01
    assert set(AL.registry_fixtures()["league_id"]) == {REG}


def test_coverage_flags_keep_a_missing_flag_missing():
    rows = AL.parse_league_coverage({"response": [{"seasons": [
        {"year": 2019, "coverage": {"fixtures": {"lineups": False}}}]}]}, REG)
    assert rows[0]["lineups"] is False and rows[0]["injuries"] is None    # not given -> None, not False


# ── 1. injuries ─────────────────────────────────────────────────────────────────────────────────

def _inj_entry(fid, pid, tid, season, league=REG, reason="Knee Injury"):
    return {"player": {"id": pid, "name": f"P{pid}", "type": "Missing Fixture", "reason": reason},
            "team": {"id": tid}, "fixture": {"id": fid, "date": "2025-08-09T14:00:00+00:00"},
            "league": {"id": league, "season": int(season)}}


def test_injuries_one_call_per_league_season_then_never_again(env):
    def bodies(ep, p):
        if ep == "leagues":
            return _coverage(injuries=False)
        if p.get("season") == "2025":
            return {"errors": [], "results": 2, "paging": {"current": 1, "total": 1},
                    "response": [_inj_entry(3, 7, 30, 2025, reason=None),
                                 _inj_entry(3, 8, 30, 2024)]}          # other season: not stored
        return {"errors": [], "results": 0, "paging": {"current": 1, "total": 1}, "response": []}
    cl, sess = _client(env, bodies)
    assert INJ.run(dry_run=True)["planned_calls_total"] == 1 + 2            # coverage + 2 seasons
    st = INJ.run(client=cl)
    assert [c[0] for c in sess.calls] == ["leagues", "injuries", "injuries"]
    assert sess.calls[1][1] == {"league": REG, "season": "2025"}          # newest season first
    rows = pd.read_parquet(INJ.ROWS)
    assert len(rows) == 1 and st["foreign_entries_skipped"] == 1
    r = rows.iloc[0]
    assert (r["player_id"], r["team_id"], r["fixture_id"], r["date"]) == (7, 30, 3, "2025-08-09")
    assert r["reason"] is None                                            # not given -> missing
    j = pd.read_parquet(INJ.SEASON_JOURNAL).set_index("season")
    assert j.loc["2024", "status"] == "empty"                              # a real empty answer
    assert not j.loc["2024", "coverage_injuries"]                         # flag kept for the record
    cl2, sess2 = _client(env, lambda ep, p: pytest.fail("re-fetched"))
    assert INJ.run(client=cl2)["calls"] == 0 and sess2.calls == []


def test_injuries_open_season_is_refreshed_only_when_new_fixtures_finished(env, tmp_path):
    ok = {"errors": [], "results": 0, "paging": {"current": 1, "total": 1}, "response": []}
    cl, _ = _client(env, lambda ep, p: _coverage() if ep == "leagues" else ok)
    INJ.run(client=cl)
    fx = _fx()
    fx.loc[len(fx)] = {"fixture_id": 5, "league_id": REG, "season": "2025", "date": "2025-08-23",
                       "home_id": 20, "away_id": 10, "status": "FT"}
    cl2, sess2 = _client(env, lambda ep, p: ok)
    INJ.run(client=cl2, fx=fx)
    assert sess2.calls == [("injuries", {"league": REG, "season": "2025"})]


def test_injuries_incomplete_answer_falls_back_to_each_fixture(env):
    def bodies(ep, p):
        if ep == "leagues":
            return _coverage()
        if "fixture" in p:
            return {"errors": [], "results": 1, "response": [_inj_entry(p["fixture"], 9, 10, 2025)]}
        if p.get("season") == "2025":                                     # paged: not complete
            return {"errors": [], "results": 1, "paging": {"current": 1, "total": 2},
                    "response": [_inj_entry(3, 7, 30, 2025)]}
        return {"errors": [], "results": 0, "paging": {"current": 1, "total": 1}, "response": []}
    cl, sess = _client(env, bodies)
    st = INJ.run(client=cl)
    assert st["incomplete_seasons"] == 1
    assert sorted(c[1]["fixture"] for c in sess.calls if "fixture" in c[1]) == [3, 4]
    rows = pd.read_parquet(INJ.ROWS)
    assert set(rows["source"]) == {"league_season", "fixture"}
    assert INJ.run(dry_run=True)["planned_fixture_fallback_calls"] == 0     # journaled


def test_injuries_completeness_rules():
    assert INJ.completeness({"results": 2, "response": [{}, {}], "paging": {"current": 1, "total": 1}})["complete"]
    assert not INJ.completeness({"results": 3, "response": [{}, {}]})["complete"]
    assert not INJ.completeness({"results": 1, "response": [{}], "paging": {"current": 1, "total": 3}})["complete"]


# ── 2. lineups ──────────────────────────────────────────────────────────────────────────────────

def _lineup(fid):
    def side(tid, coach):
        return {"team": {"id": tid}, "coach": {"id": coach, "name": "C"}, "formation": "4-3-3",
                "startXI": [{"player": {"id": tid * 100 + i, "name": "x", "number": i, "pos": "D",
                                        "grid": f"2:{i}"}} for i in range(1, 3)],
                "substitutes": [{"player": {"id": None, "name": "unknown", "number": None,
                                            "pos": None, "grid": None}}]}
    return {"errors": [], "response": [side(30, 5), side(10, None)]}


def test_lineups_newest_first_skips_seasons_without_coverage_and_journals(env):
    def bodies(ep, p):
        if ep == "leagues":
            return _coverage(lineups=False)
        return _lineup(p["fixture"]) if p["fixture"] == 4 else {"errors": [], "response": []}
    cl, sess = _client(env, bodies)
    assert LU.run(dry_run=True)["planned_calls_total_upper_bound"] == 1 + 4
    st = LU.run(client=cl)
    assert [c[1].get("fixture") for c in sess.calls[1:]] == [4, 3]        # 2024: provider has none
    assert st["skipped"] == {"no_provider_lineups": 2}
    t = pd.read_parquet(LU.TEAMS).set_index("team_id")
    assert t.loc[30, "coach_id"] == 5 and pd.isna(t.loc[10, "coach_id"])
    assert t.loc[30, "formation"] == "4-3-3" and t.loc[30, "n_start"] == 2
    p = pd.read_parquet(LU.PLAYERS)
    assert len(p) == 6 and p.loc[p["role"] == "sub", "player_id"].isna().all()   # no id -> missing
    assert p.loc[(p["team_id"] == 30) & (p["slot"] == 0) & (p["role"] == "start"), "grid"].item() == "2:1"
    j = pd.read_parquet(LU.JOURNAL).set_index("fixture_id")
    assert j.loc[3, "n_teams"] == 0                                        # empty answer journaled
    cl2, sess2 = _client(env, lambda ep, p: pytest.fail("re-fetched"))
    LU.run(client=cl2)
    assert sess2.calls == []


def test_lineups_budget_stop_is_resumable(env):
    cl, sess = _client(env, lambda ep, p: _coverage() if ep == "leagues" else _lineup(p["fixture"]), cap=3)
    st = LU.run(client=cl)
    assert "budget_stop" in st and st["fetched"] == 2
    assert sorted(pd.read_parquet(LU.JOURNAL)["fixture_id"]) == [3, 4]
    assert LU.run(dry_run=True)["planned_fixture_calls_upper_bound"] == 2


# ── 3. coaches / 4. transfers ───────────────────────────────────────────────────────────────────

def test_coaches_store_career_rows_by_provider_ids(env):
    body = {"errors": [], "response": [{"id": 77, "name": "Coach", "career": [
        {"team": {"id": 10}, "start": "2023-07-01", "end": None},
        {"team": {"id": 99}, "start": "2019-01-01", "end": "2023-06-30"},
        {"team": {"id": None}, "start": "2015-01-01", "end": "2018-01-01"}]}]}
    cl, sess = _client(env, lambda ep, p: body if p["team"] == 10 else {"errors": [], "response": []})
    assert CO.run(dry_run=True)["planned_calls"] == 5
    st = CO.run(client=cl)
    assert sess.calls[0] == ("coachs", {"team": 10}) and st["calls"] == 5
    c = pd.read_parquet(CO.CAREERS)
    assert len(c) == 2 and st["unplaceable_career_entries"] == 1           # no team id: counted
    assert pd.isna(c.set_index("team_id").loc[10, "end"])                  # current post: no end
    assert CO.run(dry_run=True)["planned_calls"] == 0


def test_transfers_keep_type_verbatim_and_dedupe_across_clubs(env):
    tr = {"date": "2024-07-01", "type": "Loan", "teams": {"in": {"id": 10}, "out": {"id": 20}}}
    body = {"errors": [], "response": [{"player": {"id": 5, "name": "X"}, "transfers": [
        tr, {"date": None, "type": "N/A", "teams": {"in": {"id": 10}, "out": {}}},
        {"date": "2022-01-15", "type": "N/A", "teams": {"in": {"id": 20}, "out": {}}}]}]}
    cl, sess = _client(env, lambda ep, p: body if p["team"] in (10, 20) else {"errors": [], "response": []})
    st = TR.run(client=cl)
    t = pd.read_parquet(TR.ROWS)
    assert len(t) == 2 and st["unplaceable_transfers"] == 2               # 2 clubs x 1 dateless
    assert set(t["type"]) == {"Loan", "N/A"}
    assert t.loc[t["date"] == "2022-01-15", "team_out_id"].isna().all()   # not given -> missing
    assert TR.run(dry_run=True)["planned_calls"] == 0


# ── workflow ────────────────────────────────────────────────────────────────────────────────────

def test_workflow_runs_the_four_steps_after_profiles_and_insidebox_lineups_last():
    y = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    pos = {k: y.rindex(f"python {s}") for k, s in (        # the step, not the header comment
        ("profiles", "scripts/collect_player_profiles.py"),
        ("insidebox", "scripts/collect_af_history_gap.py --insidebox"),
        ("injuries", "scripts/collect_injuries.py"), ("coaches", "scripts/collect_coaches.py"),
        ("transfers", "scripts/collect_transfers.py"), ("lineups", "scripts/collect_lineups.py"),
        ("fullstats", "scripts/collect_af_history_gap.py --fullstats"),
        ("strength", "-m src.strength.build"))}
    assert pos["profiles"] < pos["insidebox"] < pos["injuries"] < pos["coaches"] < pos["transfers"] \
        < pos["lineups"] < pos["fullstats"] < pos["strength"]
    for k in ("injuries", "coaches", "transfers", "lineups"):
        assert f"contains(env.STEPS, '{k}')" in y
    defaults = [ln for ln in y.splitlines() if "default:" in ln and "fixtures" in ln
                or ln.strip().startswith("STEPS:")]
    assert len(defaults) == 2
    for ln in defaults:
        # fix round 8 (owner 2026-11-04): the player history re-fetch runs right after lineups
        assert "profiles,injuries,coaches,transfers,lineups,playerrefetch,strength" in ln
    assert "COLLECT_AF_FLOOR" not in y and "COLLECT_AF_DAILY_CAP" not in y
    assert CC.FLOOR == 25_000 and CC.DAILY_CAP == 45_000


def test_collectors_write_only_under_output_collect():
    for path in (INJ.ROWS, INJ.SEASON_JOURNAL, INJ.FIXTURE_JOURNAL, LU.JOURNAL, LU.TEAMS, LU.PLAYERS,
                 CO.JOURNAL, CO.CAREERS, TR.JOURNAL, TR.ROWS, AL.COVERAGE):
        assert Path(path).parent == ROOT / "output" / "collect"
