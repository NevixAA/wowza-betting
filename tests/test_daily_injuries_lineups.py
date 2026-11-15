"""Owner decision 2026-11-15 ("Schedule daily"): the daily incremental injuries + lineups collection
(scripts/collect_daily_injuries_lineups.py, .github/workflows/daily_injuries_lineups.yml) and the
Data truth 8 freshness check.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_daily_injuries_lineups.py -q

No network: the API-Football client is a counting fake; every output goes to tmp_path.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_af_lists as AL  # noqa: E402
import collect_common as CC  # noqa: E402
import collect_competition_fixtures as CF  # noqa: E402
import collect_daily_injuries_lineups as D  # noqa: E402
import collect_injuries as INJ  # noqa: E402
import collect_lineups as LU  # noqa: E402
from collect_common import AFClient  # noqa: E402
from src.integrity import data_truth as DT  # noqa: E402

REG = 40
NOW = datetime(2026, 11, 15, 5, 17, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, body):
        self.status_code, self._b, self.headers = 200, body, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _Session:
    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, headers=None, params=None, timeout=None):
        ep = url.rsplit("/v3.football.api-sports.io/", 1)[-1]
        if ep.endswith("status"):
            return _Resp({"response": {"requests": {"current": 0, "limit_day": 75000}}})
        self.calls.append((ep, dict(params or {})))
        return _Resp(self.bodies(ep, params))


def _client(tmp, bodies):
    s = _Session(bodies)
    return AFClient(key="k", session=s, budget_file=tmp / "collect" / "af_budget_daily.json", pause_s=0), s


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cdir = tmp_path / "collect"
    cdir.mkdir()
    monkeypatch.setattr(AL, "COVERAGE", cdir / "af_league_coverage.parquet")
    monkeypatch.setattr(AL, "registry_af_ids", lambda: {REG})
    for mod, names in ((INJ, ("ROWS", "SEASON_JOURNAL", "FIXTURE_JOURNAL")), (LU, ("JOURNAL", "LINEUPS_DIR")),
                       (CF, ("JOURNAL", "UPCOMING", "OUT")), (D, ("PREMATCH_JOURNAL", "RUN_LOG", "BUDGET_FILE"))):
        for n in names:
            monkeypatch.setattr(mod, n, cdir / Path(getattr(mod, n)).name)
    monkeypatch.setattr(LU, "LEGACY", {k: cdir / v.name for k, v in LU.LEGACY.items()})
    monkeypatch.setattr(CC, "BUDGET_FILE", cdir / "af_budget.json")
    monkeypatch.setattr(AL, "FIXTURES", CF.OUT)
    # one open registry season, finished fixture 1 (lineups due), upcoming fixture 2 kicking off in 10 h
    pd.DataFrame([{"league_id": REG, "season": "2026", "season_start": "2026-08-01", "season_end": "2027-05-30",
                   "n_fixtures": 2, "n_finished": 1, "fetched_at": NOW.isoformat()}]).to_parquet(CF.JOURNAL)
    pd.DataFrame([{"fixture_id": 1, "league_id": REG, "league": "Championship", "league_name": "Championship",
                   "competition_type": "club_league", "season": "2026", "date": "2026-11-14", "home_id": 10,
                   "home_name": "A", "away_id": 20, "away_name": "B", "ft_home": 1, "ft_away": 0,
                   "status": "FT"}]).to_parquet(CF.OUT)
    ko = NOW + timedelta(hours=10)
    CF.replace_upcoming([{"fixture_id": 2, "league_id": REG, "season": "2026", "kickoff": ko.isoformat(),
                          "kickoff_ts": int(ko.timestamp()), "status": "NS", "home_id": 20, "away_id": 10,
                          "fetched_at": NOW.isoformat()}], REG, "2026")
    pd.DataFrame([{"league_id": REG, "season": "2026", "injuries": True, "lineups": True, "events": True,
                   "statistics_fixtures": True, "statistics_players": True,
                   "fetched_at": NOW.isoformat()}]).to_parquet(AL.COVERAGE)
    return tmp_path


def _inj(fid, pid, date="2026-11-14"):
    return {"league": {"id": REG, "season": 2026}, "player": {"id": pid, "name": f"P{pid}", "type": "Missing Fixture",
            "reason": "Knee Injury"}, "team": {"id": 10}, "fixture": {"id": fid, "date": f"{date}T15:00:00+00:00"}}


def _lineup():
    return [{"team": {"id": t}, "coach": {"id": 1, "name": "C"}, "formation": "4-4-2",
             "startXI": [{"player": {"id": t * 100 + i, "name": "x", "number": i, "pos": "M", "grid": "1:1"}}
                         for i in range(11)], "substitutes": []} for t in (10, 20)]


def _bodies(inj_season=True, inj_fixture=True, lineups=True):
    def f(ep, params):
        if ep == "injuries" and "league" in params:
            return {"errors": [], "results": 1 if inj_season else 0, "paging": {"current": 1, "total": 1},
                    "response": [_inj(1, 7)] if inj_season else []}
        if ep == "injuries":
            return {"errors": [], "results": 1 if inj_fixture else 0, "paging": {"current": 1, "total": 1},
                    "response": [_inj(2, 8, "2026-11-15")] if inj_fixture else []}
        if ep == "fixtures/lineups":
            return {"errors": [], "response": _lineup() if lineups else []}
        return {"errors": [], "response": []}
    return f


# ── fixture lists keep the upcoming fixtures ────────────────────────────────────────────────────

def test_upcoming_parsed_and_replaced_per_league_season(env):
    body = {"response": [
        {"fixture": {"id": 5, "date": "2026-11-16T15:00:00+00:00", "timestamp": 1794841200, "status": {"short": "NS"}},
         "teams": {"home": {"id": 1}, "away": {"id": 2}}},
        {"fixture": {"id": 6, "date": "2026-11-14T15:00:00+00:00", "timestamp": 1, "status": {"short": "FT"}},
         "teams": {"home": {"id": 1}, "away": {"id": 2}}}]}
    rows = CF.parse_upcoming(body, REG, "2026", "t")
    assert [r["fixture_id"] for r in rows] == [5] and rows[0]["kickoff_ts"] == 1794841200
    CF.replace_upcoming(rows, REG, "2026")
    up = pd.read_parquet(CF.UPCOMING)
    assert set(up["fixture_id"]) == {5}            # fixture 2 of the same league-season left with the answer


# ── fetch times ─────────────────────────────────────────────────────────────────────────────────

def test_injury_rows_keep_the_earliest_first_seen_at(env):
    r = {c: None for c in INJ.ROW_COLS}
    r.update(league_id=REG, season="2026", fixture_id=2, date="2026-11-15", team_id=10, player_id=8,
             reason="Knee", source="fixture_prematch", fetched_at="2026-11-14T05:00:00+00:00")
    INJ.store_rows([r])
    INJ.store_rows([{**r, "reason": "Knee Injury", "fetched_at": "2026-11-16T05:00:00+00:00"}])
    got = pd.read_parquet(INJ.ROWS)
    assert len(got) == 1 and got.loc[0, "reason"] == "Knee Injury"
    assert got.loc[0, "fetched_at"].startswith("2026-11-16") and got.loc[0, "first_seen_at"].startswith("2026-11-14")


def test_lineup_rows_carry_fetched_at_and_backfill_uses_the_journal(env):
    teams, players = LU.parse({"response": _lineup()}, 1, "2026-11-15T05:20:00+00:00")
    assert {p["fetched_at"] for p in players} == {"2026-11-15T05:20:00+00:00"} and teams[0]["fetched_at"]
    LU.LINEUPS_DIR.mkdir(parents=True)
    old = pd.DataFrame(players).drop(columns="fetched_at")
    old.to_parquet(LU.partition_path("players", REG, "2026"))
    pd.DataFrame([{"fixture_id": 1, "league_id": REG, "season": "2026", "n_teams": 2,
                   "fetched_at": "2026-11-08T01:00:00+00:00"}]).to_parquet(LU.JOURNAL)
    out = LU.backfill_fetched_at()
    assert out == {"files_rewritten": 1, "rows_filled": 22, "rows_left_none": 0}
    assert set(LU.read_lineups("players")["fetched_at"]) == {"2026-11-08T01:00:00+00:00"}


# ── the daily run ───────────────────────────────────────────────────────────────────────────────

def test_daily_run_collects_due_work_once(env):
    c, s = _client(env, _bodies())
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    assert st["failures"] == []
    eps = sorted((e, tuple(sorted(p.items()))) for e, p in s.calls)
    assert eps == [("fixtures/lineups", (("fixture", 1),)), ("injuries", (("fixture", 2),)),
                   ("injuries", (("league", REG), ("season", "2026")))]
    inj = pd.read_parquet(INJ.ROWS)
    assert set(inj["source"]) == {"league_season", "fixture_prematch"} and inj["first_seen_at"].notna().all()
    assert st["lineups"]["player_rows"] == 22 and st["lineups"]["per_league"][str(REG)]["fixtures"] == 1
    # a second run the same day asks nothing again (journals + REFRESH_H)
    c2, s2 = _client(env, _bodies())
    st2 = D.run(skip_fixtures=True, client=c2, now=NOW + timedelta(hours=1))
    assert s2.calls == [] and st2["failures"] == []
    assert len(D.RUN_LOG.read_text(encoding="utf-8").splitlines()) == 2


def test_fail_loud_when_due_work_returns_nothing(env):
    c, _ = _client(env, _bodies(inj_season=False, lineups=False))
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    msgs = " | ".join(st["failures"])
    assert "covered league-season(s) due, 0 entries" in msgs and "0 player rows" in msgs


def test_fail_loud_on_budget_stop_before_any_row(env, monkeypatch):
    c, s = _client(env, _bodies())
    c.daily_cap = 0
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    assert s.calls == [] and len(st["failures"]) == 3


def test_prematch_window_and_coverage(env):
    assert list(D.plan_prematch(NOW)["fixture_id"]) == [2]
    assert D.plan_prematch(NOW - timedelta(hours=30)).empty            # kickoff beyond the window
    pd.DataFrame([{"league_id": REG, "season": "2026", "injuries": False, "lineups": True,
                   "fetched_at": NOW.isoformat()}]).to_parquet(AL.COVERAGE)
    assert D.plan_prematch(NOW).empty                                 # no provider injuries coverage


# ── Data truth 8 ────────────────────────────────────────────────────────────────────────────────

def _truth_inputs(root: Path, lineup_journaled: bool, inj_read: datetime, list_read: datetime):
    pd.DataFrame([{"league_id": REG, "season": "2026", "season_start": "2026-08-01", "season_end": "2027-05-30",
                   "fetched_at": list_read.isoformat()}]).to_parquet(root / "competition_seasons.parquet")
    pd.DataFrame([{"fixture_id": 1, "league_id": REG, "season": "2026", "date": "2026-11-10"}]) \
        .to_parquet(root / "competition_fixtures.parquet")
    pd.DataFrame([{"league_id": REG, "season": "2026", "injuries": True, "lineups": True}]) \
        .to_parquet(root / "af_league_coverage.parquet")
    pd.DataFrame([{"league_id": REG, "season": "2026", "fetched_at": inj_read.isoformat()}]) \
        .to_parquet(root / "injuries_seasons.parquet")
    pd.DataFrame([{"fixture_id": 1 if lineup_journaled else 99, "fetched_at": inj_read.isoformat()}]) \
        .to_parquet(root / "lineups_fixtures.parquet")


def test_freshness_green_when_daily_collection_ran(tmp_path):
    _truth_inputs(tmp_path, True, NOW - timedelta(hours=2), NOW - timedelta(hours=2))
    r = DT.collect_freshness(NOW, root=tmp_path, registry_ids={REG})
    assert r["status"] == DT.GREEN, r


def test_freshness_red_when_stale_while_matches_are_played(tmp_path):
    _truth_inputs(tmp_path, False, NOW - timedelta(days=3), NOW - timedelta(days=3))
    r = DT.collect_freshness(NOW, root=tmp_path, registry_ids={REG})
    assert r["status"] == DT.RED
    assert "without lineups" in r["detail"] and "injuries 2026" in r["detail"] and "fixture list" in r["detail"]


def test_freshness_no_data_without_inputs(tmp_path):
    assert DT.collect_freshness(NOW, root=tmp_path, registry_ids={REG})["status"] == DT.GREY
