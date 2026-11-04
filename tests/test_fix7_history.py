"""Owner decisions of 2026-11-04 on the match history (fix7/history):

1. proven domestic-cup rows filed under a league are excluded from the league history
   (output/corrections/history_excluded_cup_rows.csv, applied by src/data_loader);
2. the same match filed twice one day apart: the copy on the provider's fixture date stays, the other
   is removed (output/corrections/history_removed_duplicates.csv, applied by src/data_loader);
3. the provider's score.fulltime is stored beside score.halftime in the HT fill, and a fill row whose
   provider FT differs from the history FT is refused.

    python -m pytest -p no:cacheprovider tests/test_fix7_history.py -q
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
import collect_ht_scores as H  # noqa: E402
from collect_common import BudgetStop  # noqa: E402
from src import history_corrections as HC  # noqa: E402

KEY = ["league", "date", "home_team", "away_team"]


def _fx(fid, kickoff, home, away, ht=(1, 0), ft=(2, 1), status="FT"):
    return {"fixture": {"id": fid, "date": kickoff, "status": {"short": status}},
            "teams": {"home": {"id": fid * 10, "name": home}, "away": {"id": fid * 10 + 1, "name": away}},
            "score": {"halftime": {"home": ht[0] if ht else None, "away": ht[1] if ht else None},
                      "fulltime": {"home": ft[0] if ft else None, "away": ft[1] if ft else None}}}


def _row(date, home="Columbus Crew", away="Inter Miami", hg=2.0, ag=1.0, league="USA MLS", **kw):
    r = {"league": league, "date": date, "home_team": home, "away_team": away, "home_goals": hg,
         "away_goals": ag, "ht_home_goals": np.nan, "ht_away_goals": np.nan}
    r.update(kw)
    return r


def _scores(rows):
    return pd.DataFrame(rows, columns=H.SCORE_COLS)


def _score(fid, date, home="Columbus Crew", away="Inter Miami", ht=(1, 0), ft=(2, 1), league="USA MLS",
           league_id=253, season="2026", kickoff=None, status="FT"):
    return {"fixture_id": fid, "league": league, "league_id": league_id, "season": season, "date": date,
            "kickoff": kickoff or f"{date}T23:30:00+00:00", "home_id": 1, "home_name": home, "away_id": 2,
            "away_name": away, "ht_home": ht[0], "ht_away": ht[1], "status": status,
            "ft_home": ft[0] if ft else None, "ft_away": ft[1] if ft else None,
            "ft_source": H.FT_SRC_FETCH if ft else None}


@pytest.fixture()
def no_corrections(tmp_path, monkeypatch):
    """The real committed lists must not leak into synthetic tests."""
    monkeypatch.setattr(HC, "CUP_EXCLUDED", tmp_path / "none_cup.csv")
    monkeypatch.setattr(HC, "DUP_REMOVED", tmp_path / "none_dup.csv")
    return tmp_path


@pytest.fixture()
def hpaths(tmp_path, monkeypatch, no_corrections):
    for name in ("SCORES", "FILL", "AUDIT", "JOURNAL", "FT_JOURNAL", "FT_MISMATCH"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    monkeypatch.setattr(H, "COMP_FIXTURES", tmp_path / "competition_fixtures.parquet")
    return tmp_path


class _Client:
    def __init__(self, bodies=None, stop_after=None):
        self.bodies, self.calls, self.stop_after = bodies or {}, [], stop_after

    def get(self, endpoint, params):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise BudgetStop("test budget")
        self.calls.append((endpoint, dict(params)))
        key = (endpoint, tuple(sorted(params.items())))
        return self.bodies.get(key, {"response": []})

    def flush(self):
        pass


# ---- 3. full-time score beside the half-time score -----------------------------------------------

def test_parse_stores_fulltime_and_never_invents_it():
    body = {"response": [_fx(1, "2026-08-01T19:30:00+00:00", "A", "B", ft=(3, 2)),
                         _fx(2, "2026-08-01T19:30:00+00:00", "C", "D", ft=None),
                         _fx(3, "2026-08-01T19:30:00+00:00", "E", "F", ft=(1, None))]}
    rows = H.parse(body, "USA MLS", 253, "2026")
    assert (rows[0]["ft_home"], rows[0]["ft_away"], rows[0]["ft_source"]) == (3, 2, H.FT_SRC_FETCH)
    assert rows[1]["ft_home"] is None and rows[1]["ft_away"] is None and rows[1]["ft_source"] is None
    # half a score is no score: both sides missing, never (1, 0) or (1, None)
    assert rows[2]["ft_home"] is None and rows[2]["ft_away"] is None


def test_ft_completed_by_fixture_id_and_league_only_never_overwritten():
    sc = pd.DataFrame([_score(1, "2026-08-01", ft=None), _score(2, "2026-08-01", ft=None),
                       _score(3, "2026-08-01", ft=(4, 4))])
    comp = pd.DataFrame({"fixture_id": [1, 2, 3], "league_id": [253, 999, 253],
                         "ft_home": [2, 5, 0], "ft_away": [0, 5, 0]})
    out, n = H.complete_ft_from_competition_fixtures(sc, comp)
    assert n == 1
    assert (out.loc[0, "ft_home"], out.loc[0, "ft_away"], out.loc[0, "ft_source"]) == (2, 0, H.FT_SRC_COMP)
    assert pd.isna(out.loc[1, "ft_home"])            # another league's id: never borrowed
    assert (out.loc[2, "ft_home"], out.loc[2, "ft_away"]) == (4, 4)    # stored value kept


def test_add_ft_rereads_only_lacking_league_seasons_resumably(hpaths):
    H.append_parquet(H.SCORES, [_score(1, "2026-08-01", ft=None), _score(2, "2025-08-01", season="2025", ft=(1, 1))],
                     ["fixture_id"], H.SCORE_COLS)
    body = {"response": [_fx(1, "2026-08-01T23:30:00+00:00", "Columbus Crew", "Inter Miami", ft=(2, 1))]}
    # a budget stop before the first call: nothing recorded, resumable
    st = H.add_ft(client=_Client(stop_after=0), fetch=True)
    assert st["ft_calls"] == 0 and "ft_budget_stop" in st and H.read_parquet(H.FT_JOURNAL).empty
    c = _Client({("fixtures", (("league", 253), ("season", "2026"))): body})
    st = H.add_ft(client=c, fetch=True)
    assert c.calls == [("fixtures", {"league": 253, "season": "2026"})]      # 2025 already has FT
    s = pd.read_parquet(H.SCORES).set_index("fixture_id")
    assert (s.loc[1, "ft_home"], s.loc[1, "ft_away"]) == (2, 1)
    c2 = _Client()
    H.add_ft(client=c2, fetch=True)
    assert c2.calls == []                                                    # journaled / complete
    # without --add-ft no call is ever made
    assert H.add_ft(client=_Client(stop_after=0), fetch=False)["ft_calls"] == 0


def test_fill_carries_provider_ft_and_refuses_a_fulltime_mismatch():
    lack = H.lacking_rows(pd.DataFrame([_row("2026-08-01", hg=2.0, ag=1.0),
                                        _row("2026-08-02", home="LA Galaxy", away="Seattle Sounders", hg=3.0, ag=3.0),
                                        _row("2026-08-03", home="Austin FC", away="FC Dallas", hg=1.0, ag=0.0)]),
                           corrections=False)
    sc = _scores([_score(1, "2026-08-01", ft=(2, 1)),
                  _score(2, "2026-08-02", home="LA Galaxy", away="Seattle Sounders", ft=(2, 2), status="PEN"),
                  _score(3, "2026-08-03", home="Austin FC", away="FC Dallas", ft=None)])
    fill, per = H.match(lack, sc)
    got = fill.set_index("fixture_id")
    assert set(got.index) == {1, 3}
    assert got.loc[1, "ft_check"] == "equal" and (got.loc[1, "provider_ft_home"], got.loc[1, "provider_ft_away"]) == (2, 1)
    assert got.loc[3, "ft_check"] == "provider_ft_missing" and pd.isna(got.loc[3, "provider_ft_home"])
    mm = fill.attrs["ft_mismatch"]
    assert len(mm) == 1 and mm.iloc[0]["status"] == "PEN" and mm.iloc[0]["history_ft_home"] == 3.0
    assert per["USA MLS"]["refused_ft_mismatch"] == 1


def test_run_writes_ft_columns_mismatch_list_and_upgrades_an_old_sidecar(hpaths):
    old = pd.DataFrame([{"league": "USA MLS", "date": pd.Timestamp("2020-01-01"), "home_team": "X",
                         "away_team": "Y", "HTHG": 0, "HTAG": 0, "fixture_id": 9, "af_date": "2020-01-01",
                         "match_rule": "same_day", "source": H.SOURCE, "matched_at": "t"}])
    old.to_parquet(H.FILL, index=False)                        # written before the FT columns existed
    H.append_parquet(H.SCORES, [_score(1, "2026-08-01", ft=None)], ["fixture_id"], H.SCORE_COLS)
    pd.DataFrame({"fixture_id": [1], "league_id": [253], "ft_home": [2], "ft_away": [1]}) \
        .to_parquet(H.COMP_FIXTURES, index=False)
    st = H.run(match_only=True, history=pd.DataFrame([_row("2026-08-01")]))
    f = pd.read_parquet(H.FILL)
    assert set(H.FILL_COLS) <= set(f.columns) and len(f) == 2
    new = f[f["fixture_id"] == 1].iloc[0]
    assert (new["provider_ft_home"], new["provider_ft_away"], new["ft_check"]) == (2, 1, "equal")
    assert st["ft_completed_from_competition_fixtures"] == 1 and st["refused_ft_mismatch"] == 0
    assert H.FT_MISMATCH.exists() and pd.read_csv(H.FT_MISMATCH).empty


def test_workflow_htscores_step_builds_corrections_and_adds_ft():
    wf = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    step = wf.split("name: 7. Half-time scores")[1].split("- name:")[0]
    assert "python scripts/build_history_corrections.py --fetch" in step
    assert "python scripts/collect_ht_scores.py --add-ft" in step
    assert step.index("build_history_corrections") < step.index("collect_ht_scores")
    assert "output/corrections/history_*.csv" in wf


# ---- 2. duplicates -----------------------------------------------------------------------------

def _twins(score_b=(2.0, 1.0)):
    return pd.DataFrame([_row("2026-08-01", hg=2.0, ag=1.0, odds_over25=np.nan),
                         _row("2026-08-02", hg=score_b[0], ag=score_b[1], odds_over25=1.9)])


def test_duplicate_twin_on_the_provider_date_stays_the_other_is_removed(no_corrections):
    h = _twins()
    h["date"] = pd.to_datetime(h["date"])
    sc = _scores([_score(1, "2026-08-01", kickoff="2026-08-01T23:30:00+00:00")])
    rem, con, st = B.duplicate_removals(h, sc)
    assert con.empty and len(rem) == 1
    r = rem.iloc[0]
    assert (r["date"], r["kept_date"], r["fixture_id"]) == ("2026-08-02", "2026-08-01", 1)
    assert r["odds_over25"] == 1.9                             # full original values kept
    # 23:30 UTC is 19:30 in New York: the kept copy is both the UTC and the local date
    assert r["provider_local_date"] == "2026-08-01" and not r["removed_copy_is_local_date"]
    assert st["timezone"]["kept_copy_is_provider_utc_date"] == 1


def test_duplicate_with_different_scores_is_a_conflict_left_alone(no_corrections):
    h = _twins(score_b=(3.0, 1.0))
    h["date"] = pd.to_datetime(h["date"])
    rem, con, st = B.duplicate_removals(h, _scores([_score(1, "2026-08-01")]))
    assert rem.empty and set(con["conflict"]) == {"score_differs_between_copies"} and len(con) == 2


def test_duplicate_with_no_copy_on_the_provider_date_is_a_conflict(no_corrections):
    h = pd.DataFrame([_row("2026-08-01"), _row("2026-08-03")])
    h["date"] = pd.to_datetime(h["date"])
    # provider date 08-02: each copy claims it on the adjacent day, neither is on the provider date
    rem, con, _ = B.duplicate_removals(h, _scores([_score(1, "2026-08-02")]))
    assert rem.empty and len(con) == 2
    assert set(con["conflict"]) <= {"no_copy_on_provider_date", "dates_2_days_apart"}


def _write_dup(path, removed_date="2026-08-02", hg=2.0, ag=1.0):
    pd.DataFrame([{"league": "USA MLS", "date": removed_date, "home_team": "Columbus Crew",
                   "away_team": "Inter Miami", "home_goals": hg, "away_goals": ag, "odds_over25": 1.9,
                   "kept_date": "2026-08-01", "kept_home_team": "Columbus Crew",
                   "kept_away_team": "Inter Miami", "fixture_id": 1}]).to_csv(path, index=False)


def test_apply_removes_the_listed_copy_and_fills_the_kept_copy(no_corrections):
    _write_dup(HC.DUP_REMOVED)
    df = _twins().assign(row_source="fd_cache")
    out = HC.apply_history_corrections(df)
    assert len(out) == 1 and out.iloc[0]["date"] == "2026-08-01"
    assert out.iloc[0]["odds_over25"] == 1.9                   # real value of the same proven match
    rep = out.attrs["history_corrections"]
    assert rep["duplicate_rows_removed"] == 1 and rep["cells_filled_from_removed_copy"] >= 1


def test_apply_leaves_a_changed_row_and_non_history_sources(no_corrections):
    _write_dup(HC.DUP_REMOVED, hg=5.0, ag=5.0)                 # listing differs from the row now
    df = _twins().assign(row_source="fd_cache")
    out = HC.apply_history_corrections(df)
    assert len(out) == 2 and out.attrs["history_corrections"]["listed_rows_changed_since_listing"] == 1
    _write_dup(HC.DUP_REMOVED)
    out = HC.apply_history_corrections(_twins().assign(row_source="api_football"))
    assert len(out) == 2                                       # never removes a non-history row


def test_apply_with_repeated_index_labels_drops_by_position(no_corrections):
    _write_dup(HC.DUP_REMOVED)
    df = pd.concat([_twins(), _twins().iloc[:1]]).assign(row_source="fd_cache")   # labels 0, 1, 0
    out = HC.apply_history_corrections(df)
    assert len(out) == 2 and (out["date"] == "2026-08-01").all()


def test_collector_fills_the_kept_twin_once_the_copy_is_removed(hpaths):
    h = _twins()
    H.append_parquet(H.SCORES, [_score(1, "2026-08-01")], ["fixture_id"], H.SCORE_COLS)
    H.run(match_only=True, history=h)
    assert pd.read_parquet(H.FILL).empty                       # claimed from two dates: refused
    _write_dup(HC.DUP_REMOVED)
    H.run(match_only=True, history=h)
    f = pd.read_parquet(H.FILL)
    assert len(f) == 1 and str(f.iloc[0]["date"])[:10] == "2026-08-01" and f.iloc[0]["ft_check"] == "equal"


def test_loader_identity_pipeline_applies_the_corrections(no_corrections):
    from src import data_loader as DL
    _write_dup(HC.DUP_REMOVED)
    df = _twins().assign(row_source="fd_cache")
    df["date"] = pd.to_datetime(df["date"])
    out = DL.one_club_one_match(df)
    # without the list the loader's own de-duplication keeps the more populated 08-02 copy; with it
    # the copy on the provider date stays
    assert len(out) == 1 and pd.Timestamp(out.iloc[0]["date"]) == pd.Timestamp("2026-08-01")


# ---- 1. cup rows -------------------------------------------------------------------------------

ARG = "Argentina Primera Division"


def _cup_body():
    return {"response": [
        {"league": {"id": 1032, "name": "Copa de la Liga Profesional", "type": "Cup"},
         "country": {"name": "Argentina"},
         "seasons": [{"year": 2023, "start": "2023-08-18", "end": "2023-12-16"}]},
        {"league": {"id": 128, "name": "Liga Profesional Argentina", "type": "League"},
         "country": {"name": "Argentina"}, "seasons": [{"year": 2023, "start": "2023-01-27", "end": "2023-07-29"}]},
        {"league": {"id": 77, "name": "Other", "type": "Cup"}, "country": {"name": "Uruguay"},
         "seasons": [{"year": 2023, "start": "2023-01-01", "end": "2023-12-31"}]}]}


def test_parse_cups_keeps_only_national_cups_of_the_country():
    rows = B.parse_cups(_cup_body(), "Argentina")
    assert {r["league_id"] for r in rows} == {1032}


def _cand(rows):
    c = pd.DataFrame(rows)
    c["date"] = pd.to_datetime(c["date"])
    c["canon"] = ARG
    return c


def _cupfx(fid, date, home, away, ft=(1, 0)):
    return {"fixture_id": fid, "league_id": 1032, "cup_name": "Copa de la Liga Profesional",
            "country": "Argentina", "season": "2023", "date": date, "kickoff": f"{date}T22:00:00+00:00",
            "home_id": 1, "home_name": home, "away_id": 2, "away_name": away,
            "ft_home": ft[0] if ft else None, "ft_away": ft[1] if ft else None, "status": "FT"}


def test_prove_cup_rows_by_date_clubs_and_score():
    cand = _cand([_row("2023-09-02", "River Plate", "Boca Juniors", 1.0, 0.0, league=ARG),
                  _row("2023-09-03", "Racing Club", "Independiente", 2.0, 2.0, league=ARG),   # FT differs
                  _row("2023-09-10", "Lanus", "Banfield", 0.0, 0.0, league=ARG)])             # no cup fixture
    fx = pd.DataFrame([_cupfx(5, "2023-09-03", "River Plate", "Boca Juniors"),             # adjacent day
                       _cupfx(6, "2023-09-03", "Racing Club", "Independiente", ft=(1, 1))])
    proven, unproven = B.prove_cup_rows(cand, fx)
    assert len(proven) == 1
    p = proven.iloc[0]
    assert (p["fixture_id"], p["cup_league_id"], p["match_rule"]) == (5, 1032, "adjacent_day")
    assert p["cup_name"] == "Copa de la Liga Profesional" and p["ft_check"] == "equal"
    why = dict(zip(unproven["home_team"], unproven["reason"]))
    assert why["Racing Club"] == "full_time_score_differs"
    assert why["Lanus"] == "no_cup_fixture_stored_within_pm1_day"


def test_prove_cup_rows_refuses_two_cup_fixtures():
    cand = _cand([_row("2023-09-02", "River Plate", "Boca Juniors", 1.0, 0.0, league=ARG)])
    fx = pd.DataFrame([_cupfx(5, "2023-09-02", "River Plate", "Boca Juniors"),
                       dict(_cupfx(7, "2023-09-02", "River Plate", "Boca Juniors"), league_id=130, cup_name="Copa Argentina")])
    proven, unproven = B.prove_cup_rows(cand, fx)
    assert proven.empty and unproven.iloc[0]["reason"] == "ambiguous_more_than_one_cup_fixture"


def test_unplaced_rows_are_those_without_a_league_fixture_within_one_day():
    h = pd.DataFrame([_row("2023-09-02", "River Plate", "Boca Juniors", league=ARG),
                      _row("2023-07-01", "River Plate", "Boca Juniors", league=ARG),
                      _row("2019-01-01", "River Plate", "Boca Juniors", league=ARG)])   # before the span
    h["date"] = pd.to_datetime(h["date"])
    fx = pd.DataFrame({"league": [ARG, ARG, ARG], "date": ["2023-06-30", "2023-10-01", "2022-01-01"]})
    u = B.unplaced_rows(h, fx)
    assert list(u["date"].dt.strftime("%Y-%m-%d")) == ["2023-09-02"]


def test_fetch_cups_is_budget_guarded_and_journaled(tmp_path, monkeypatch):
    for name in ("CUPS", "CUP_FIXTURES", "CUP_JOURNAL"):
        monkeypatch.setattr(B, name, tmp_path / getattr(B, name).name)
    cand = _cand([_row("2023-09-02", "River Plate", "Boca Juniors", league=ARG)])
    fixtures = {"response": [_fx(5, "2023-09-02T22:00:00+00:00", "River Plate", "Boca Juniors", ft=(1, 0))]}
    c = _Client({("leagues", (("country", "Argentina"), ("type", "cup"))): _cup_body(),
                 ("fixtures", (("league", 1032), ("season", "2023"))): fixtures})
    st = B.fetch_cups(cand, client=c)
    assert st["calls"] == 2 and [x[0] for x in c.calls] == ["leagues", "fixtures"]
    fx = pd.read_parquet(B.CUP_FIXTURES)
    assert list(fx["fixture_id"]) == [5] and fx.iloc[0]["cup_name"] == "Copa de la Liga Profesional"
    c2 = _Client()
    assert B.fetch_cups(cand, client=c2)["calls"] == 0 and c2.calls == []     # stored, season ended
    # MAX_CALLS caps the step even when the job budget would allow more
    monkeypatch.setattr(B, "MAX_CALLS", 0)
    for name in ("CUPS", "CUP_FIXTURES", "CUP_JOURNAL"):
        monkeypatch.setattr(B, name, tmp_path / ("x_" + getattr(B, name).name))
    st = B.fetch_cups(cand, client=_Client())
    assert st["calls"] == 0 and "budget_stop" in st


def test_apply_excludes_a_proven_cup_row(no_corrections):
    pd.DataFrame([{"league": ARG, "date": "2023-09-02", "home_team": "River Plate", "away_team": "Boca Juniors",
                   "home_goals": 1.0, "away_goals": 0.0, "cup_league_id": 1032, "fixture_id": 5}]) \
        .to_csv(HC.CUP_EXCLUDED, index=False)
    df = pd.DataFrame([_row("2023-09-02", "River Plate", "Boca Juniors", 1.0, 0.0, league=ARG),
                       _row("2023-09-09", "River Plate", "Boca Juniors", 1.0, 0.0, league=ARG)])
    out = HC.apply_history_corrections(df)
    assert list(out["date"]) == ["2023-09-09"] and out.attrs["history_corrections"]["cup_rows_excluded"] == 1


# ---- committed lists ---------------------------------------------------------------------------

def test_committed_duplicate_list_is_proven_against_the_history():
    """Every removed row is in fd_history with its recorded score; its kept twin is there too, on the
    provider date, with the same score; no removed key is also a kept key."""
    p = ROOT / "output" / "corrections" / "history_removed_duplicates.csv"
    if not p.exists():
        pytest.skip("no duplicate list in this checkout")
    d = pd.read_csv(p)
    if d.empty:
        return
    h = pd.read_parquet(ROOT / "output" / "fd_history.parquet", columns=KEY + ["home_goals", "away_goals"])
    h["date"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
    hk = h.drop_duplicates(KEY).set_index(KEY)
    for r in d.drop_duplicates(KEY).itertuples(index=False):
        a = hk.loc[(r.league, r.date, r.home_team, r.away_team)]
        b = hk.loc[(r.league, r.kept_date, r.kept_home_team, r.kept_away_team)]
        assert (a.home_goals, a.away_goals) == (r.home_goals, r.away_goals)
        assert (b.home_goals, b.away_goals) == (a.home_goals, a.away_goals)
        assert r.kept_date == r.provider_date and abs((pd.Timestamp(r.date) - pd.Timestamp(r.kept_date)).days) == 1
    removed = set(map(tuple, d[KEY].values))
    kept = set(zip(d["league"], d["kept_date"], d["kept_home_team"], d["kept_away_team"]))
    assert not removed & kept


def test_committed_cup_list_carries_evidence():
    p = ROOT / "output" / "corrections" / "history_excluded_cup_rows.csv"
    if not p.exists():
        pytest.skip("no cup list in this checkout")
    d = pd.read_csv(p)
    if d.empty:
        return
    assert d[["fixture_id", "cup_league_id", "cup_name", "provider_date"]].notna().all().all()
    from src import league_registry as LR
    league_ids = {int(e["api_football_id"]) for e in LR.leagues() if e.get("api_football_id") is not None}
    assert not d["cup_league_id"].astype(int).isin(league_ids).any()      # never a registry league
