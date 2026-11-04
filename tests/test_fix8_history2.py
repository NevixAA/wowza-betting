"""fix8/history2 (2026-11-04) - follow-ups to the owner's history corrections of fix round 7:

1. the 90-minute score: a history row proven to be an AET / PEN fixture is served at the provider's
   score.fulltime (output/corrections/history_score_90min.csv, original kept), and the HT fill then
   accepts it;
2. FT-status score disagreements stay refused (UNIDENTIFIABLE); awarded results (AWD / WO) are flagged
   for exclusion (output/corrections/history_awarded_results.csv);
3. the duplicate rule covers every pair the loader's own de-duplication merges, the provider date
   deciding;
4. lineups are stored one file per league-season;
5. coaches / transfers are re-asked once after each transfer-window close, answers only added.

    python -m pytest -p no:cacheprovider tests/test_fix8_history2.py -q

No network: every API-Football client is a counting fake; every write goes to tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_history_corrections as B  # noqa: E402
import collect_af_lists as AL  # noqa: E402
import collect_coaches as CO  # noqa: E402
import collect_common as CC  # noqa: E402
import collect_ht_scores as H  # noqa: E402
import collect_lineups as LU  # noqa: E402
import collect_transfers as TR  # noqa: E402
from collect_common import AFClient, BudgetStop  # noqa: E402
from src import history_corrections as HC  # noqa: E402

KEY = ["league", "date", "home_team", "away_team"]
MLS = "USA MLS"


@pytest.fixture()
def no_corrections(tmp_path, monkeypatch):
    """The real committed lists must not leak into synthetic tests."""
    for name, f in (("CUP_EXCLUDED", "cup"), ("DUP_REMOVED", "dup"), ("AWARDED_EXCLUDED", "awd"),
                    ("SCORE_90", "score90")):
        monkeypatch.setattr(HC, name, tmp_path / f"none_{f}.csv")
    return tmp_path


def _row(date, home="Los Angeles FC", away="Philadelphia Union", hg=3.0, ag=3.0, league=MLS, **kw):
    r = {"league": league, "date": date, "home_team": home, "away_team": away, "home_goals": hg,
         "away_goals": ag, "ht_home_goals": np.nan, "ht_away_goals": np.nan}
    r.update(kw)
    return r


def _score(fid, date, home="Los Angeles FC", away="Philadelphia Union", ft=(2, 2), status="PEN",
           ht=(1, 0), league=MLS, league_id=253, season="2022", et=None):
    return {"fixture_id": fid, "league": league, "league_id": league_id, "season": season, "date": date,
            "kickoff": f"{date}T20:00:00+00:00", "home_id": fid * 10, "home_name": home,
            "away_id": fid * 10 + 1, "away_name": away, "ht_home": ht[0] if ht else None,
            "ht_away": ht[1] if ht else None, "status": status,
            "ft_home": ft[0] if ft else None, "ft_away": ft[1] if ft else None,
            "ft_source": H.FT_SRC_FETCH if ft else None,
            "et_home": et[0] if et else None, "et_away": et[1] if et else None,
            "pen_home": None, "pen_away": None}


def _pool(rows):
    return pd.DataFrame(rows, columns=H.SCORE_COLS)


def _hist(rows):
    h = pd.DataFrame(rows)
    h["date"] = pd.to_datetime(h["date"])
    return h


def _write_score90(path, hg=3, ag=3, ft90=(2, 2)):
    pd.DataFrame([{"league": MLS, "date": "2022-11-05", "home_team": "Los Angeles FC",
                   "away_team": "Philadelphia Union", "history_ft_home": hg, "history_ft_away": ag,
                   "ft90_home": ft90[0], "ft90_away": ft90[1], "fixture_id": 970025,
                   "provider_status": "PEN"}]).to_csv(path, index=False)


# ---- 1. 90-minute score --------------------------------------------------------------------------

def test_parse_keeps_extratime_penalty_and_awarded_status():
    def fx(fid, status, et=None, pen=None):
        return {"fixture": {"id": fid, "date": "2022-11-05T20:00:00+00:00", "status": {"short": status}},
                "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}},
                "score": {"halftime": {"home": 1, "away": 0}, "fulltime": {"home": 2, "away": 2},
                          "extratime": {"home": et[0], "away": et[1]} if et else {"home": None, "away": None},
                          "penalty": {"home": pen[0], "away": pen[1]} if pen else {"home": None, "away": None}}}
    rows = H.parse({"response": [fx(1, "PEN", et=(1, 1), pen=(3, 0)), fx(2, "AWD"), fx(3, "FT"),
                                 fx(4, "PST")]}, MLS, 253, "2022")
    got = {r["fixture_id"]: r for r in rows}
    assert set(got) == {1, 2, 3}                                     # postponed: not stored
    assert (got[1]["et_home"], got[1]["et_away"], got[1]["pen_home"], got[1]["pen_away"]) == (1, 1, 3, 0)
    assert got[3]["et_home"] is None and got[3]["pen_home"] is None    # not given -> missing, never 0
    assert got[2]["status"] == "AWD"


def test_awarded_fixture_never_gives_a_half_time_score_and_never_triggers_an_ft_reread():
    lack = H.lacking_rows(_hist([_row("2022-11-05", hg=3.0, ag=0.0)]), corrections=False)
    sc = _pool([_score(1, "2022-11-05", status="AWD", ft=None, ht=(0, 0))])
    fill, _ = H.match(lack, sc)
    assert fill.empty
    assert H.ft_plan(sc, pd.DataFrame()) == []


def test_apply_score_correction_serves_the_90_minute_score_and_relabels(no_corrections):
    _write_score90(HC.SCORE_90, hg=2, ag=1, ft90=(1, 1))
    df = pd.DataFrame([_row("2022-11-05", hg=2.0, ag=1.0, total_goals=3.0, over25=1.0, over35=0.0,
                            over15=1.0, btts=1.0, ftr="H"),
                       _row("2022-11-12", total_goals=6.0, ftr="D")]).assign(row_source="fd_cache")
    out = HC.apply_history_corrections(df)
    r = out.iloc[0]
    assert (r["home_goals"], r["away_goals"], r["total_goals"]) == (1.0, 1.0, 2.0)
    assert (r["over25"], r["over15"], r["btts"], r["ftr"]) == (0.0, 1.0, 1.0, "D")
    assert out.iloc[1]["home_goals"] == 3.0                          # another match: untouched
    assert out.attrs["history_corrections"]["score_rows_corrected_90min"] == 1
    assert df.iloc[0]["home_goals"] == 2.0                           # the input frame is not mutated


def test_score_correction_leaves_a_changed_row_and_non_history_rows(no_corrections):
    _write_score90(HC.SCORE_90, hg=4, ag=3)                          # the listing differs from the row
    out = HC.apply_history_corrections(pd.DataFrame([_row("2022-11-05")]).assign(row_source="fd_cache"))
    assert out.iloc[0]["home_goals"] == 3.0
    assert out.attrs["history_corrections"]["score_listed_rows_changed_since_listing"] == 1
    _write_score90(HC.SCORE_90)
    out = HC.apply_history_corrections(pd.DataFrame([_row("2022-11-05")]).assign(row_source="api_football"))
    assert out.iloc[0]["home_goals"] == 3.0                          # never an API-Football row


def test_score_list_is_not_a_removal_list(no_corrections):
    _write_score90(HC.SCORE_90)
    fill = pd.DataFrame([{"league": MLS, "date": "2022-11-05", "home_team": "Los Angeles FC",
                          "away_team": "Philadelphia Union"}])
    assert not HC.corrected_mask(fill).any()                         # the HT fill row is kept


def test_ht_fill_accepts_the_row_after_the_90_minute_correction(no_corrections):
    h = _hist([_row("2022-11-05")])
    sc = _pool([_score(970025, "2022-11-05")])
    fill, per = H.match(H.lacking_rows(h), sc)
    assert fill.empty and per[MLS]["refused_ft_mismatch"] == 1        # 3-3 history vs 2-2 provider
    _write_score90(HC.SCORE_90)
    fill, per = H.match(H.lacking_rows(h), sc)
    assert len(fill) == 1 and fill.iloc[0]["ft_check"] == "equal" and fill.iloc[0]["HTHG"] == 1


def test_score_corrections_rules():
    h = _hist([_row("2022-11-05"),                                                        # PEN, 3-3 vs 2-2
               _row("2023-05-29", "Guadalajara Chivas", "Tigres UANL", 2.0, 2.0, league="Mexico Liga MX"),
               _row("2023-06-01", "CFR Cluj", "U Craiova 1948", 0.0, 0.0, league="Romanian Superliga"),
               _row("2022-08-29", "Sligo Rovers", "Dundalk", 2.0, 0.0, league="Ireland Premier Division"),
               _row("2022-03-05", "Queretaro", "Atlas", 0.0, 3.0, league="Mexico Liga MX")])
    pool = _pool([_score(1, "2022-11-05"),
                  _score(2, "2023-05-29", "Guadalajara Chivas", "Tigres UANL", status="AET",
                         league="Mexico Liga MX", league_id=262),                        # already 90'
                  _score(3, "2023-06-01", "CFR 1907 Cluj", "U Craiova 1948", ft=(1, 0), status="AET",
                         league="Romanian Superliga", league_id=283),                     # history below
                  _score(4, "2022-08-29", "Sligo Rovers", "Dundalk", ft=(0, 3), status="FT",
                         league="Ireland Premier Division", league_id=357),
                  _score(5, "2022-03-05", "Club Queretaro", "Atlas", ft=None, status="AWD",
                         league="Mexico Liga MX", league_id=262)])
    prov, refused, awd, st = B.score_corrections(h, pool)
    assert list(prov["fixture_id"]) == [1]
    p = prov.iloc[0]
    assert (p["history_ft_home"], p["history_ft_away"], p["ft90_home"], p["ft90_away"]) == (3, 3, 2, 2)
    assert p["et_check"] == "extratime_not_stored" and p["found_by"] == "provider_status_PEN"
    why = dict(zip(refused["fixture_id"], refused["reason"]))
    assert why == {3: "history_below_90min_score", 4: "ft_status_scores_differ"}
    assert set(refused["verdict"]) == {"UNIDENTIFIABLE"}
    assert list(awd["fixture_id"]) == [5] and st["aet_pen_history_already_90min"] == 1


def test_score_corrections_refuse_a_fixture_claimed_from_two_dates():
    h = _hist([_row("2022-11-05"), _row("2022-11-06")])
    _, _, _, st = B.score_corrections(h, _pool([_score(1, "2022-11-05")]))
    assert st["claims_refused_fixture_claimed_from_two_dates"] == 2


def test_extratime_check_is_recorded_when_the_provider_gives_it():
    h = _hist([_row("2022-11-05")])
    prov, _, _, _ = B.score_corrections(h, _pool([_score(1, "2022-11-05", et=(1, 1))]))
    assert prov.iloc[0]["et_check"] == "equals_90_plus_extratime"


# ---- 2. awarded results --------------------------------------------------------------------------

def test_awarded_list_excludes_the_row(no_corrections):
    pd.DataFrame([{"league": "Mexico Liga MX", "date": "2022-03-05", "home_team": "Queretaro",
                   "away_team": "Atlas", "home_goals": 0.0, "away_goals": 3.0, "fixture_id": 5,
                   "provider_status": "AWD"}]).to_csv(HC.AWARDED_EXCLUDED, index=False)
    df = pd.DataFrame([_row("2022-03-05", "Queretaro", "Atlas", 0.0, 3.0, league="Mexico Liga MX"),
                       _row("2022-03-12", "Necaxa", "Queretaro", 1.0, 0.0, league="Mexico Liga MX")])
    out = HC.apply_history_corrections(df)
    assert list(out["home_team"]) == ["Necaxa"]
    assert out.attrs["history_corrections"]["awarded_rows_excluded"] == 1


# ---- 3. every pair the loader merges ---------------------------------------------------------------

def _pair_history():
    # both copies HOLD a half-time score, so round 7 (rows lacking HT) never saw them
    return _hist([_row("2024-06-08", "New England Revolution", "New York Red Bulls", 1.0, 0.0,
                       ht_home_goals=0.0, ht_away_goals=0.0),
                  _row("2024-06-09", "New England Revolution", "New York Red Bulls", 1.0, 0.0,
                       ht_home_goals=0.0, ht_away_goals=0.0, home_shots=12.0),
                  _row("2024-06-15", "Columbus Crew", "Inter Miami", 2.0, 1.0,
                       ht_home_goals=1.0, ht_away_goals=0.0),
                  _row("2024-06-16", "Columbus Crew", "Inter Miami", 2.0, 1.0,
                       ht_home_goals=1.0, ht_away_goals=0.0)])


def test_loader_pairs_are_decided_by_the_provider_date(no_corrections):
    h = _pair_history()
    pool = _pool([_score(1, "2024-06-08", "New England Revolution", "New York Red Bulls", ft=(1, 0),
                         status="FT", season="2024")])                # Columbus v Miami: no fixture stored
    empty = B._as_corrections(pd.DataFrame(), "duplicate_removed")
    rem, con, st = B.pair_removals(h, pool, empty, B._scores_frame(pd.DataFrame()))
    assert st["history_pairs_one_to_three_days_apart"] == 2 and st["pairs_proven"] == 1
    r = rem.iloc[0]
    assert (r["date"], r["kept_date"], r["fixture_id"], r["found_by"]) == ("2024-06-09", "2024-06-08", 1,
                                                                          "loader_merge_pair")
    assert r["home_shots"] == 12.0                                   # full original values kept
    # the loader's tie-break kept the more populated 06-09 copy: the provider date changes the survivor
    assert r["loader_survivor_was"] == "removed_copy" and st["survivor_changes_vs_loader_tiebreak"] == 1
    assert set(con["conflict"]) == {"a_copy_claims_no_provider_fixture"} and len(con) == 2


def test_loader_pair_with_different_fixtures_is_a_conflict(no_corrections):
    h = _pair_history().iloc[:2]
    pool = _pool([_score(1, "2024-06-08", "New England Revolution", "New York Red Bulls", ft=(1, 0),
                         status="FT", season="2024"),
                  _score(2, "2024-06-09", "New England Revolution", "New York Red Bulls", ft=(1, 0),
                         status="FT", season="2024")])
    rem, con, _ = B.pair_removals(h, pool, B._as_corrections(pd.DataFrame(), "x"),
                                  B._scores_frame(pd.DataFrame()))
    assert rem.empty and set(con["conflict"]) == {"copies_claim_different_fixtures"}


def test_builder_writes_found_by_and_the_new_lists(tmp_path, monkeypatch, no_corrections):
    for name in ("DUP_OUT", "DUP_CONFLICTS", "CUP_OUT", "CUP_UNPROVEN", "SCORE_OUT", "SCORE_REFUSED",
                 "AWARDED_OUT", "AUDIT"):
        monkeypatch.setattr(B, name, tmp_path / getattr(B, name).name)
    h = pd.concat([_pair_history(), _hist([_row("2022-11-05")])], ignore_index=True)
    monkeypatch.setattr(B, "load_history", lambda: h)
    monkeypatch.setattr(B, "provider_scores", lambda: _pool([
        _score(1, "2024-06-08", "New England Revolution", "New York Red Bulls", ft=(1, 0), status="FT",
               season="2024"),
        _score(970025, "2022-11-05")]))
    monkeypatch.setattr(B, "load_archive", lambda: pd.DataFrame())
    monkeypatch.setattr(B, "read_parquet", lambda p, columns=None: pd.DataFrame())
    audit = B.run()
    d = pd.read_csv(B.DUP_OUT)
    assert list(d["found_by"]) == ["loader_merge_pair"] and list(d["date"]) == ["2024-06-09"]
    s = pd.read_csv(B.SCORE_OUT)
    assert list(s["fixture_id"]) == [970025] and audit["score_90min"]["corrected_to_90min"] == 1
    assert B.AWARDED_OUT.exists() and pd.read_csv(B.AWARDED_OUT).empty
    assert audit["loader_merge_duplicates"]["removed_rows"] == 1


# ---- 4. lineups, one file per league-season -----------------------------------------------------------

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


def _client(tmp, bodies, cap=45_000, budget="af_budget.json"):
    sess = _Session(bodies)
    return AFClient(key="k", session=sess, budget_file=tmp / "collect" / budget,
                    daily_cap=cap, pause_s=0), sess


_CLOCK = {"now": "2026-08-01T10:00:00+00:00"}


def _fx():
    rows = [(1, 40, "2024", "2024-08-10", 10, 20, "FT"), (2, 40, "2024", "2024-08-17", 20, 30, "FT"),
            (3, 40, "2025", "2025-08-09", 30, 10, "FT"), (4, 41, "2025", "2025-08-16", 10, 40, "AET")]
    return pd.DataFrame([{"fixture_id": f, "league_id": lg, "season": s, "date": d, "home_id": h,
                          "away_id": a, "status": st} for f, lg, s, d, h, a, st in rows])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cdir = tmp_path / "collect"
    monkeypatch.setattr(AL, "COVERAGE", cdir / "af_league_coverage.parquet")
    monkeypatch.setattr(AL, "registry_af_ids", lambda: {40, 41})
    monkeypatch.setattr(AL, "_league_closes", lambda: {})
    monkeypatch.setattr(AL, "now_iso", lambda: _CLOCK["now"])     # the answers' fetched_at
    _CLOCK["now"] = "2026-08-01T10:00:00+00:00"
    monkeypatch.setattr(LU, "JOURNAL", cdir / "lineups_fixtures.parquet")
    monkeypatch.setattr(LU, "LINEUPS_DIR", cdir / "lineups")
    monkeypatch.setattr(LU, "LEGACY", {k: cdir / v.name for k, v in LU.LEGACY.items()})
    for mod in (CO, TR):
        monkeypatch.setattr(mod, "JOURNAL", cdir / Path(mod.JOURNAL).name)
    monkeypatch.setattr(CO, "CAREERS", cdir / "coach_careers.parquet")
    monkeypatch.setattr(TR, "ROWS", cdir / "transfers.parquet")
    monkeypatch.setattr(CC, "BUDGET_FILE", cdir / "af_budget.json")
    fxp = tmp_path / "competition_fixtures.parquet"
    _fx().to_parquet(fxp)
    monkeypatch.setattr(AL, "FIXTURES", fxp)
    return tmp_path


def _lineup(fid):
    def side(tid):
        return {"team": {"id": tid}, "coach": {"id": 5, "name": "C"}, "formation": "4-4-2",
                "startXI": [{"player": {"id": tid * 100 + i, "name": "x", "number": i, "pos": "M",
                                        "grid": f"3:{i}"}} for i in range(1, 12)],
                "substitutes": [{"player": {"id": tid * 1000, "name": "s", "number": 20, "pos": "F",
                                            "grid": None}}]}
    return {"errors": [], "response": [side(10 + fid), side(20 + fid)]}


def _cov():
    return {"errors": [], "response": [{"seasons": [
        {"year": 2024, "coverage": {"fixtures": {"lineups": True}}},
        {"year": 2025, "coverage": {"fixtures": {"lineups": True}}}]}]}


def test_lineups_are_stored_one_file_per_league_season(env):
    cl, _ = _client(env, lambda ep, p: _cov() if ep == "leagues" else _lineup(p["fixture"]))
    LU.run(client=cl)
    files = sorted(x.name for x in LU.LINEUPS_DIR.glob("*.parquet"))
    assert files == ["players_40_2024.parquet", "players_40_2025.parquet", "players_41_2025.parquet",
                     "teams_40_2024.parquet", "teams_40_2025.parquet", "teams_41_2025.parquet"]
    assert not LU.LEGACY["players"].exists() and not LU.LEGACY["teams"].exists()   # no single file
    assert len(pd.read_parquet(LU.partition_path("players", 40, "2024"))) == 2 * 2 * 12
    assert len(LU.read_lineups("players")) == 4 * 2 * 12
    assert set(LU.read_lineups("teams", league_id=40, season="2025")["fixture_id"]) == {3}
    assert sorted(pd.read_parquet(LU.JOURNAL)["fixture_id"]) == [1, 2, 3, 4]       # journal unchanged
    cl2, sess2 = _client(env, lambda ep, p: pytest.fail("re-fetched"))
    LU.run(client=cl2)
    assert sess2.calls == []


def test_read_lineups_includes_a_round7_single_file_partition_wins(env):
    cl, _ = _client(env, lambda ep, p: _cov() if ep == "leagues" else _lineup(p["fixture"]))
    LU.run(client=cl, limit=1)                                      # fixture 4 (newest) stored
    LU.LEGACY["teams"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fixture_id": 9, "team_id": 1, "formation": "3-5-2", "coach_id": None,
                   "coach_name": None, "n_start": 11, "n_subs": 7},
                  {"fixture_id": 4, "team_id": 14, "formation": "OLD", "coach_id": 5, "coach_name": "C",
                   "n_start": 11, "n_subs": 1}]).to_parquet(LU.LEGACY["teams"], index=False)
    t = LU.read_lineups("teams").set_index(["fixture_id", "team_id"])
    assert t.loc[(9, 1), "formation"] == "3-5-2"                     # the old store is still read
    assert t.loc[(4, 14), "formation"] == "4-4-2"                    # the partition row wins


def test_no_lineup_partition_can_approach_the_github_limit():
    # the largest registry league-season in the stored fixture lists, at the plan's ~40 rows a fixture
    fx = AL.fixture_list()
    if fx.empty:
        pytest.skip("no fixture list in this checkout")
    biggest = int(fx.groupby(["league_id", "season"]).size().max())
    assert biggest * 40 < 200_000                                    # rows; a 9-column parquet of this is a few MB


# ---- 5. coaches / transfers refresh after each transfer window -----------------------------------------

def test_last_close_and_team_close():
    assert AL.last_close(AL.WINDOW_CLOSES, "2026-11-04") == pd.Timestamp("2026-09-02")
    assert AL.last_close(AL.WINDOW_CLOSES, "2026-02-02") == pd.Timestamp("2025-09-02")
    assert AL.last_close(AL.WINDOW_CLOSES, "2026-02-03") == pd.Timestamp("2026-02-03")
    c = AL.team_last_close(_fx(), "2026-08-01", league_closes={41: ((7, 15),)})
    assert c[40] == pd.Timestamp("2026-07-15")                       # its league's own close is later
    assert c[20] == pd.Timestamp("2026-02-03")


def _coach_body(end=None):
    return {"errors": [], "response": [{"id": 77, "name": "Coach", "career": [
        {"team": {"id": 10}, "start": "2023-07-01", "end": end}]}]}


def test_coaches_initial_then_once_after_each_window_previous_answers_kept(env):
    cl, sess = _client(env, lambda ep, p: _coach_body() if p["team"] == 10 else {"errors": [], "response": []})
    st = CO.run(client=cl, today="2026-08-01")
    assert st["planned_initial"] == 4 and st["planned_refresh"] == 0 and st["calls"] == 4
    assert CO.run(dry_run=True, today="2026-08-31")["planned_calls"] == 0          # no close since
    st = CO.run(dry_run=True, today="2026-09-02")
    assert st["planned_refresh"] == 4 and st["planned_initial"] == 0
    _CLOCK["now"] = "2026-09-03T10:00:00+00:00"
    cl2, sess2 = _client(env, lambda ep, p: _coach_body("2026-08-20") if p["team"] == 10
                         else {"errors": [], "response": []})
    CO.run(client=cl2, today="2026-09-03")
    assert len(sess2.calls) == 4
    c = pd.read_parquet(CO.CAREERS)
    assert len(c) == 2                                                # the old answer is still there
    assert set(c["end"].fillna("open")) == {"open", "2026-08-20"}
    j = pd.read_parquet(CO.JOURNAL)
    assert len(j) == 8 and set(j["refresh_window"]) == {"initial", "2026-09-02"}
    assert CO.run(dry_run=True, today="2026-12-01")["planned_calls"] == 0          # asked once per window


def test_transfers_refresh_adds_only_changed_answers_and_is_budget_guarded(env):
    def body(t):
        return {"errors": [], "response": [{"player": {"id": 5, "name": "X"}, "transfers": [
            {"date": "2026-07-01", "type": t, "teams": {"in": {"id": 10}, "out": {"id": 20}}}]}]}
    cl, _ = _client(env, lambda ep, p: body("N/A"))
    TR.run(client=cl, today="2026-08-01")
    assert len(pd.read_parquet(TR.ROWS)) == 1                        # both clubs: one transfer
    _CLOCK["now"] = "2026-09-03T10:00:00+00:00"
    cl2, sess2 = _client(env, lambda ep, p: body("€ 2M"), cap=2, budget="day2.json")   # 2 calls allowed
    st = TR.run(client=cl2, today="2026-09-03")
    assert "budget_stop" in st and st["calls"] == 2
    t = pd.read_parquet(TR.ROWS)
    assert set(t["type"]) == {"N/A", "€ 2M"}                         # previous answer kept, new added
    assert TR.run(dry_run=True, today="2026-09-03")["planned_refresh"] == 2         # the rest stays due


def test_old_journal_without_refresh_window_still_counts_as_asked(env):
    CO.JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"team_id": [10, 20, 30, 40], "n_coaches": [1, 0, 0, 0],
                  "fetched_at": ["2026-11-04T08:00:00+00:00"] * 4}).to_parquet(CO.JOURNAL, index=False)
    assert CO.run(dry_run=True, today="2026-11-05")["planned_calls"] == 0
    assert CO.run(dry_run=True, today="2027-02-03")["planned_refresh"] == 4


# ---- committed lists -----------------------------------------------------------------------------

def test_committed_score_list_is_proven_against_the_history():
    p = ROOT / "output" / "corrections" / "history_score_90min.csv"
    if not p.exists():
        pytest.skip("no 90-minute list in this checkout")
    d = pd.read_csv(p)
    if d.empty:
        return
    h = pd.read_parquet(ROOT / "output" / "fd_history.parquet", columns=KEY + ["home_goals", "away_goals"])
    h["date"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
    hk = h.drop_duplicates(KEY).set_index(KEY)
    for r in d.itertuples(index=False):
        a = hk.loc[(r.league, r.date, r.home_team, r.away_team)]
        assert (a.home_goals, a.away_goals) == (r.history_ft_home, r.history_ft_away)   # original kept
        assert r.ft90_home <= r.history_ft_home and r.ft90_away <= r.history_ft_away
        assert (r.ft90_home, r.ft90_away) != (r.history_ft_home, r.history_ft_away)
        assert r.provider_status in ("AET", "PEN")


def test_committed_refused_scores_are_never_applied():
    p = ROOT / "output" / "corrections" / "history_score_refused.csv"
    if not p.exists():
        pytest.skip("no refused list in this checkout")
    d = pd.read_csv(p)
    assert set(d["verdict"]) <= {"UNIDENTIFIABLE"}
    s = HC.read_score_corrections()
    assert not set(map(tuple, d[KEY].values)) & set(map(tuple, s[KEY].astype(str).values))
