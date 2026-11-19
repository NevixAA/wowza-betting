"""Owner decision 2026-11-15 ("Schedule daily"): the daily incremental injuries + lineups collection
(scripts/collect_daily_injuries_lineups.py, .github/workflows/daily_injuries_lineups.yml) and the
Data truth 11 freshness check.

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


# ── lineups: empty answers re-asked while recent (2026-11-16) ──────────────────────────────────

def _journal(rows):
    pd.DataFrame([{"league_id": REG, "season": "2026", **r} for r in rows],
                 columns=LU.J_COLS).to_parquet(LU.JOURNAL)


def _clock(monkeypatch, when: datetime):
    monkeypatch.setattr(AL, "now_iso", lambda: when.isoformat())


def test_recent_empty_is_re_asked_and_a_filled_answer_replaces_it(env, monkeypatch):
    # fixture 1 (kickoff date 2026-11-14) came back empty 21 h ago: due a re-ask
    first = NOW - timedelta(hours=21)
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": first.isoformat()}])
    p = LU.plan(now=NOW)
    assert p["todo"].empty and list(p["retry"]["fixture_id"]) == [1]
    _clock(monkeypatch, NOW)
    c, s = _client(env, _bodies())
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    assert ("fixtures/lineups", {"fixture": 1}) in s.calls and st["failures"] == []
    lu = st["lineups"]
    assert (lu["due"], lu["retry_due"], lu["retry_calls"], lu["retry_filled"], lu["retry_still_empty"]) == (0, 1, 1, 1, 0)
    assert lu["retry_player_rows"] == 22 and lu["retry_filled_ids"] == [1]
    j = pd.read_parquet(LU.JOURNAL)
    assert len(j) == 1 and j.loc[0, "n_teams"] == 2 and j.loc[0, "fetched_at"] == NOW.isoformat()
    assert len(LU.read_lineups("players")) == 22 and len(LU.read_lineups("teams")) == 2
    a = pd.read_parquet(LU.attempts_path()).sort_values("fetched_at")
    assert list(a["ask"]) == ["journal", "retry"] and list(a["n_teams"]) == [0, 2]
    assert list(a["fetched_at"]) == [first.isoformat(), NOW.isoformat()] and set(a["kickoff_date"]) == {"2026-11-14"}
    # filled: never asked again
    assert LU.plan(now=NOW + timedelta(days=2))["retry"].empty
    # the run record carries the retry counters
    rec = D.RUN_LOG.read_text(encoding="utf-8").splitlines()[-1]
    assert '"retry_calls": 1' in rec and '"retry_filled": 1' in rec


def test_retry_that_stays_empty_does_not_fail_loud_and_is_spaced(env, monkeypatch):
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": (NOW - timedelta(hours=24)).isoformat()}])
    _clock(monkeypatch, NOW)
    c, s = _client(env, _bodies(lineups=False))
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    assert st["failures"] == []                                     # a still-empty re-ask is not a failure
    assert (st["lineups"]["retry_calls"], st["lineups"]["retry_still_empty"]) == (1, 1)
    j = pd.read_parquet(LU.JOURNAL)
    assert len(j) == 1 and j.loc[0, "n_teams"] == 0 and j.loc[0, "fetched_at"] == NOW.isoformat()
    assert len(pd.read_parquet(LU.attempts_path())) == 2 and LU.read_lineups("players").empty
    # spacing: not again within RETRY_SPACING_H of the last ask, again after it
    assert LU.plan(now=NOW + timedelta(hours=LU.RETRY_SPACING_H - 1))["retry"].empty
    assert list(LU.plan(now=NOW + timedelta(hours=LU.RETRY_SPACING_H))["retry"]["fixture_id"]) == [1]


def test_retry_spacing_after_a_fresh_empty(env):
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": (NOW - timedelta(hours=6)).isoformat()}])
    assert LU.plan(now=NOW)["retry"].empty                          # asked 6 h ago
    assert len(LU.plan(now=NOW + timedelta(hours=14))["retry"]) == 1   # 20 h after the ask


def test_old_empty_is_final(env):
    # last asked after kickoff date + 7 days (2026-11-21 00:00 UTC): final, whatever the spacing
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": "2026-11-21T00:00:00+00:00"}])
    assert LU.plan(now=datetime(2026, 11, 25, tzinfo=timezone.utc))["retry"].empty
    # the historic case: asked years after the match
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": "2026-11-09T00:00:00+00:00"}])
    old = pd.read_parquet(CF.OUT).assign(date="2018-09-01")
    old.to_parquet(CF.OUT)
    assert LU.plan(now=NOW)["retry"].empty
    # one last ask before the window closes still counts: asked 2026-11-20 23:00 -> re-asked once more
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": "2026-11-20T23:00:00+00:00"}])
    pd.read_parquet(CF.OUT).assign(date="2026-11-14").to_parquet(CF.OUT)
    assert len(LU.plan(now=datetime(2026, 11, 21, 19, tzinfo=timezone.utc))["retry"]) == 1


def test_empty_without_known_kickoff_is_final_and_logged(env, monkeypatch):
    _journal([{"fixture_id": 99, "n_teams": 0, "fetched_at": (NOW - timedelta(days=1)).isoformat()},
              {"fixture_id": 98, "n_teams": 0, "fetched_at": "2019-01-01T00:00:00+00:00"}])
    p = LU.plan(now=NOW)
    assert p["retry"].empty and p["retry_final_unknown_kickoff"] == [99]    # only the recent one is logged
    _clock(monkeypatch, NOW)
    pd.DataFrame([{"fixture_id": 1, "league_id": REG, "season": "2026", "n_teams": 2,
                   "fetched_at": NOW.isoformat()}, *pd.read_parquet(LU.JOURNAL).to_dict("records")]) \
        .to_parquet(LU.JOURNAL)
    c, s = _client(env, _bodies())
    st = D.run(skip_fixtures=True, client=c, now=NOW)
    assert not any(e == "fixtures/lineups" for e, _ in s.calls)
    assert st["lineups"]["retry_final_unknown_kickoff"] == [99] and st["failures"] == []


def test_retries_come_after_new_fixtures_and_respect_the_budget(env, monkeypatch):
    # fixture 1 never asked; fixture 3 (same date) came back empty a day ago
    fx = pd.read_parquet(CF.OUT)
    pd.concat([fx, fx.assign(fixture_id=3)], ignore_index=True).to_parquet(CF.OUT)
    _journal([{"fixture_id": 3, "n_teams": 0, "fetched_at": (NOW - timedelta(days=1)).isoformat()}])
    _clock(monkeypatch, NOW)
    c, s = _client(env, _bodies())
    c.daily_cap = 1                                                  # one call left today
    st = LU.run(client=c, now=NOW)
    assert s.calls == [("fixtures/lineups", {"fixture": 1})]         # the new fixture first
    assert st["budget_stop"] and st["retry_calls"] == 0 and st["planned_retry_calls_upper_bound"] == 1
    j = pd.read_parquet(LU.JOURNAL).set_index("fixture_id")
    assert j.loc[3, "n_teams"] == 0 and j.loc[1, "n_teams"] == 2
    # the retry is still due on the next run
    assert list(LU.plan(now=NOW + timedelta(hours=1))["retry"]["fixture_id"]) == [3]
    # --limit caps first asks and re-asks together
    c2, s2 = _client(env, _bodies())
    c2.daily_cap = 100
    _journal([{"fixture_id": 3, "n_teams": 0, "fetched_at": (NOW - timedelta(days=1)).isoformat()}])
    st2 = LU.run(limit=1, client=c2, now=NOW)
    assert len(s2.calls) == 1 and st2["retry_calls"] == 0


def test_dry_run_counts_the_retries(env):
    _journal([{"fixture_id": 1, "n_teams": 0, "fetched_at": (NOW - timedelta(days=1)).isoformat()}])
    out = D.dry_run(NOW)
    assert out["lineup_calls_before_list_refresh"] == 0 and out["lineup_retry_calls"] == 1


# ── two firings a day (2026-11-17, backup cron) ────────────────────────────────────────────────

def _lists_read_at(when: datetime):
    j = pd.read_parquet(CF.JOURNAL)
    j["fetched_at"] = when.isoformat()
    j.to_parquet(CF.JOURNAL)


def test_fixture_lists_fresh_window(env):
    _lists_read_at(NOW - timedelta(hours=17))
    assert D.fixture_lists_fresh(NOW) == (NOW - timedelta(hours=17)).isoformat()
    _lists_read_at(NOW - timedelta(hours=19))
    assert D.fixture_lists_fresh(NOW) is None                        # older than FIXTURE_REFRESH_H
    # an ended season is not re-read by step 1, so its old fetch does not make the lists stale
    j = pd.read_parquet(CF.JOURNAL)
    old = {**j.iloc[0].to_dict(), "season": "2025", "season_start": "2025-08-01", "season_end": "2026-05-30",
           "fetched_at": "2026-06-10T00:00:00+00:00"}
    pd.concat([j.assign(fetched_at=(NOW - timedelta(hours=2)).isoformat()), pd.DataFrame([old])])         .to_parquet(CF.JOURNAL)
    assert D.fixture_lists_fresh(NOW) == (NOW - timedelta(hours=2)).isoformat()
    # a season read BEFORE it ended is re-read once more: it counts
    pd.concat([j.assign(fetched_at=(NOW - timedelta(hours=2)).isoformat()),
               pd.DataFrame([{**old, "fetched_at": "2026-05-01T00:00:00+00:00"}])]).to_parquet(CF.JOURNAL)
    assert D.fixture_lists_fresh(NOW) is None
    CF.JOURNAL.unlink()
    assert D.fixture_lists_fresh(NOW) is None                        # no journal: never fresh


def test_second_firing_reuses_lists_and_asks_only_new_work(env, monkeypatch):
    """Primary at NOW, backup 12 h later (05:17 / 17:17 UTC): the backup re-reads no fixture list,
    no league-season, no pre-kickoff fixture and no lineup the primary stored; it asks only the
    fixture that entered ITS 26 h window, the daily budget counts both firings, and 'nothing due'
    is not a failure."""
    ko3 = NOW + timedelta(hours=33)              # outside the primary's window, inside the backup's
    up = pd.read_parquet(CF.UPCOMING).to_dict("records")
    CF.replace_upcoming(up + [{"fixture_id": 3, "league_id": REG, "season": "2026", "kickoff": ko3.isoformat(),
                               "kickoff_ts": int(ko3.timestamp()), "status": "NS", "home_id": 30, "away_id": 40,
                               "fetched_at": NOW.isoformat()}], REG, "2026")
    _lists_read_at(NOW - timedelta(hours=24))
    list_runs = []

    def fake_lists(client=None):
        list_runs.append(AL.now_iso())
        _lists_read_at(datetime.fromisoformat(AL.now_iso()))
        return {"fetched": 1, "rows": 0}
    monkeypatch.setattr(CF, "run", fake_lists)

    _clock(monkeypatch, NOW)
    c1, s1 = _client(env, _bodies())
    st1 = D.run(client=c1, now=NOW)
    assert st1["failures"] == [] and len(list_runs) == 1
    assert sorted((e, tuple(sorted(p.items()))) for e, p in s1.calls) == [
        ("fixtures/lineups", (("fixture", 1),)), ("injuries", (("fixture", 2),)),
        ("injuries", (("league", REG), ("season", "2026")))]

    later = NOW + timedelta(hours=12)
    _clock(monkeypatch, later)
    c2, s2 = _client(env, _bodies())
    st2 = D.run(client=c2, now=later)
    assert len(list_runs) == 1 and st2["fixtures"]["skipped"] == "fresh"
    assert s2.calls == [("injuries", {"fixture": 3})]
    assert st2["failures"] == [] and st2["injuries"]["season_calls_due"] == 0
    assert st2["lineups"]["due"] == 0 and st2["lineups"]["retry_due"] == 0
    assert c2.calls_today() == 4                 # af_budget_daily.json carries the first firing's 3 calls

    # a third firing right after: nothing at all is due and nothing fails
    c3, s3 = _client(env, _bodies())
    st3 = D.run(client=c3, now=later + timedelta(minutes=5))
    assert s3.calls == [] and st3["failures"] == []
    # the next day's firing (24 h after the primary) refreshes the lists and re-reads the season
    nxt = NOW + timedelta(hours=24)
    _clock(monkeypatch, nxt)
    c4, s4 = _client(env, _bodies())
    D.run(client=c4, now=nxt)
    assert len(list_runs) == 2 and ("injuries", {"league": REG, "season": "2026"}) in s4.calls


def test_dry_run_plans_no_list_calls_when_fresh(env):
    out = D.dry_run(NOW)                         # env journal read at NOW
    assert out["fixture_lists_fresh_since"] and out["fixture_lists_leagues_calls"] == 0
    assert out["fixture_lists_open_season_calls"] == 0
    _lists_read_at(NOW - timedelta(hours=20))
    out = D.dry_run(NOW)
    assert out["fixture_lists_fresh_since"] is None and out["fixture_lists_leagues_calls"] > 0


# ── Data truth 11 ───────────────────────────────────────────────────────────────────────────────

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


def test_freshness_is_rule_11_and_not_gated():
    red = {"check": "Data truth 11: injuries/lineups freshness", "status": DT.RED, "value": "stale", "detail": ""}
    assert not DT.is_gated(red["check"])                     # 'Data truth 11' is not 'Data truth 1'
    assert DT.is_gated("Data truth 1: one canonical name") and DT.is_gated("Data truth 7b: x")
    assert DT.is_gated("Data truth 13b: new provenance violations") and not DT.is_gated("Data truth 13a: x")
    # go-live merge: 10 = props features (gated, prep numbering), so the ungated example is rule 3 only
    assert DT.is_gated("Data truth 10b: x") and not DT.is_gated("Data truth 3: no missing data")
    assert not [f for f in DT.gate(checks=[red]) if "Data truth 11" in f]
    assert DT.collect_freshness.__name__ and any("Data truth 11" in line for line in
                                                 Path(DT.__file__).read_text(encoding="utf-8").splitlines())
