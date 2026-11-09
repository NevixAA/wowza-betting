"""P7 (2026-11-09) missing-data collection: proven aliases, football-data raw files, and the
provider-proven extra fixtures for the player re-fetch. Synthetic data only, no network."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prove_af_history_aliases as PA  # noqa: E402

IDS = {"Test League": 999}


def _fx(rows):
    return pd.DataFrame([{"fixture_id": i, "league_id": 999, "date": d, "d": d, "home_name": h,
                          "away_name": a, "ft_home": hg, "ft_away": ag, "status": "FT"}
                         for i, (d, h, a, hg, ag) in enumerate(rows, 1)])


def _h(rows):
    return pd.DataFrame([{"canon": "Test League", "d": d, "home_team": h, "away_team": a, "hg": hg, "ag": ag}
                         for d, h, a, hg, ag in rows])


DATES = ["2025-01-04", "2025-01-11", "2025-01-18", "2025-01-25"]


def test_forced_opponent_on_three_dates_proves_one_league_scoped_alias():
    fx = _fx([(d, "Alpha United", "Gamma Rovers Athletic", 1, 0) for d in DATES[:3]])
    h = _h([(d, "Alpha United", "Gammaz", pd.NA, pd.NA) for d in DATES[:3]])
    r = PA.prove(h, fx, IDS)
    assert r["map"] == {"Test League": {"Gammaz": "Gamma Rovers Athletic"}}
    assert r["evidence"]["Test League"]["Gammaz"][0]["dates"] == 3


def test_two_dates_are_not_enough():
    fx = _fx([(d, "Alpha United", "Gamma Rovers Athletic", 1, 0) for d in DATES[:2]])
    h = _h([(d, "Alpha United", "Gammaz", pd.NA, pd.NA) for d in DATES[:2]])
    assert PA.prove(h, fx, IDS)["map"] == {}


def test_neither_side_resolved_is_never_evidence_even_with_a_matching_score():
    fx = _fx([(d, "Xeno Club", "Yotta Club", 2, 1) for d in DATES])
    h = _h([(d, "Xx", "Yy", 2, 1) for d in DATES])
    assert PA.prove(h, fx, IDS)["map"] == {}


def test_a_score_that_disagrees_with_the_forced_fixture_is_not_evidence():
    fx = _fx([(d, "Alpha United", "Gamma Rovers Athletic", 1, 0) for d in DATES[:3]])
    h = _h([(d, "Alpha United", "Gammaz", 3, 3) for d in DATES[:3]])
    r = PA.prove(h, fx, IDS)
    assert r["map"] == {} and r["refused"]["dates_with_score_conflict"] == 3


def test_a_date_the_history_club_plays_but_the_provider_name_does_not_refuses_the_pair():
    rows = [(d, "Alpha United", "Gamma Rovers Athletic", 1, 0) for d in DATES[:3]]
    rows.append((DATES[3], "Alpha United", "Delta FC", 0, 0))          # provider: Gamma absent
    fx = _fx(rows)
    h = _h([(d, "Alpha United", "Gammaz", pd.NA, pd.NA) for d in DATES[:3]]
           + [(DATES[3], "Beta Town", "Gammaz", pd.NA, pd.NA)])
    r = PA.prove(h, fx, IDS)
    assert "Gammaz" not in r["map"].get("Test League", {})


def test_a_provider_side_rename_becomes_date_ranged_segments():
    d1 = ["2023-01-07", "2023-01-14", "2023-01-21"]
    d2 = ["2024-08-03", "2024-08-10", "2024-08-17"]
    fx = _fx([(d, "Alpha United", "Jahn R", 1, 0) for d in d1] + [(d, "Alpha United", "SSV Jahn R", 1, 0) for d in d2])
    h = _h([(d, "Alpha United", "Regensburgx", pd.NA, pd.NA) for d in d1 + d2])
    m = PA.prove(h, fx, IDS)["map"]["Test League"]["Regensburgx"]
    assert isinstance(m, list) and len(m) == 2
    assert PA.lookup({"Test League": {"Regensburgx": m}}, "Test League", "Regensburgx", "2023-05-01") == "Jahn R"
    assert PA.lookup({"Test League": {"Regensburgx": m}}, "Test League", "Regensburgx", "2024-09-01") == "SSV Jahn R"


def test_an_alias_is_used_only_in_its_league_and_only_when_the_spelling_is_in_the_pool(monkeypatch):
    monkeypatch.setattr(PA, "_LOADED", {"Test League": {"Gammaz": "Gamma Rovers Athletic"}})
    none = lambda n, p: None                                            # noqa: E731
    assert PA.resolve_with_proven("Gammaz", ["Gamma Rovers Athletic"], "Test League", "2025-02-01", none) \
        == "Gamma Rovers Athletic"
    assert PA.resolve_with_proven("Gammaz", ["Gamma Rovers Athletic"], "Other League", "2025-02-01", none) is None
    assert PA.resolve_with_proven("Gammaz", ["Someone Else"], "Test League", "2025-02-01", none) is None


def test_the_committed_alias_file_is_league_scoped_with_evidence():
    p = ROOT / "config" / "af_history_proven_aliases.json"
    if not p.exists():
        pytest.skip("alias file not generated in this checkout")
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    for lg, m in d["map"].items():
        for name in m:
            ev = d["evidence"][lg][name]
            assert all(e["dates"] >= PA.MIN_DATES for e in ev)


# ── football-data raw files ──────────────────────────────────────────────────────────────────

import collect_fd_raw as FD  # noqa: E402


class _Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content = code, content


class _Sess:
    def __init__(self, table):
        self.table, self.calls = table, []

    def get(self, url, **k):
        self.calls.append(url)
        return self.table.get(url, _Resp(404))


CSV = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,Avg>2.5,Avg<2.5\nE2,05/08/2023,A,B,1,0,1.9,1.9\n"


def test_fd_raw_writes_new_files_and_never_replaces_with_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(FD, "OUT_DIR", tmp_path)
    monkeypatch.setattr(FD, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(FD, "plan", lambda now: [("League One", "E2", "2324"), ("League One", "E2", "2425")])
    u1, u2 = FD.URL.format(season="2324", code="E2"), FD.URL.format(season="2425", code="E2")
    st = FD.run(session=_Sess({u1: _Resp(200, CSV), u2: _Resp(503)}))
    assert st["written"] == 1 and st["failed"] == 1 and (tmp_path / "E2_2324.csv").read_bytes() == CSV
    st = FD.run(session=_Sess({u1: _Resp(200, b"<html>maintenance</html>")}))
    assert (tmp_path / "E2_2324.csv").read_bytes() == CSV and st["written"] == 0
    st = FD.run(session=_Sess({u1: _Resp(200, CSV)}))
    assert st["unchanged"] == 1


# ── extra fixtures for the player re-fetch ──────────────────────────────────────────────────

import collect_refetch_extra_fixtures as XF  # noqa: E402


def test_extra_fixtures_come_only_from_the_provider_answer():
    body = {"response": [{"fixture": {"id": 1, "date": "2025-10-10T22:00:00+00:00", "status": {"short": "FT"}},
                          "league": {"id": 34, "name": "WCQ CONMEBOL", "season": 2026},
                          "teams": {"home": {"id": 7, "name": "Uruguay"}, "away": {"id": 8, "name": "Dominican Rep"}},
                          "goals": {"home": 1, "away": 0}},
                         {"fixture": {}, "league": {}}]}
    rows = XF.parse(body, "2026-11-09T00:00:00Z")
    assert len(rows) == 1 and rows[0]["fixture_id"] == 1 and rows[0]["league_id"] == 34
    assert rows[0]["home_name"] == "Uruguay" and rows[0]["source"] == "api-football /fixtures?ids"


def test_extra_fixtures_are_not_in_the_shared_fixture_list():
    import collect_af_lists as AL
    import inspect
    assert "refetch_extra_fixtures" not in inspect.getsource(AL)


# ── item 6: a history row whose is_home disagrees with the provider ─────────────────────────

def test_is_home_is_corrected_only_when_the_rows_club_is_proven_on_the_provider_side():
    import collect_player_refetch as RF
    from test_player_refetch import ANSWER_100, _fx, _h
    fxrow = _fx().set_index("fixture_id").loc[100]
    rows = RF.source_rows(pd.DataFrame([
        # Chelsea's player stored as HOME while Chelsea is the list's AWAY club: proven -> corrected
        _h(100, 21, "2025-08-16", "Arsenal", "Chelsea", "Chelsea", 1),
        # an Arsenal row for a player the answer puts in Chelsea's block: the club is NOT on that side
        _h(100, 22, "2025-08-16", "Arsenal", "Chelsea", "Arsenal", 1),
    ]), pd.DataFrame())
    out, j = RF.parse_answer(ANSWER_100, 100, fxrow, rows, "2026-11-09T00:00:00Z")
    got = {r["player_id"]: r for r in out}
    assert 21 in got and got[21]["is_home"] == 0 and got[21]["is_home_orig"] == 1 and got[21]["is_home_corrected"]
    assert 22 not in got and j["refused"].get("side_unproven") == 1 and j["is_home_corrected"] == 1


# ── results sync from main ──────────────────────────────────────────────────────────────────

def _git(repo, *a):
    import subprocess
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                   capture_output=True)


def test_sync_takes_untouched_files_and_only_newer_rows_of_a_file_both_changed(tmp_path, monkeypatch):
    import sync_results_from_main as SY
    repo = tmp_path / "r"
    (repo / "output").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    base = pd.DataFrame({"league": ["L", "L"], "date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
                         "home_team": ["A", "C"], "away_team": ["B", "D"]})
    base.to_parquet(repo / "output" / "af_history.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(repo / "output" / "fd_history.parquet", index=False)
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "dt")
    # this branch removes a proven duplicate: keep A-B, drop C-D (its newest date)
    base.iloc[:1].to_parquet(repo / "output" / "af_history.parquet", index=False)
    _git(repo, "commit", "-qam", "dt repair")
    _git(repo, "checkout", "-q", "main")
    newer = pd.concat([base, pd.DataFrame({"league": ["L"], "date": pd.to_datetime(["2026-11-05"]),
                                           "home_team": ["E"], "away_team": ["F"]})], ignore_index=True)
    newer.to_parquet(repo / "output" / "af_history.parquet", index=False)
    pd.DataFrame({"x": [1, 2]}).to_parquet(repo / "output" / "fd_history.parquet", index=False)
    _git(repo, "commit", "-qam", "main results")
    _git(repo, "checkout", "-q", "dt")
    monkeypatch.setattr(SY, "ROOT", repo)
    monkeypatch.setattr(SY, "MAIN", "main")
    monkeypatch.setattr(SY, "WHOLE_DIRS", [])
    monkeypatch.setattr(SY, "SIDECAR", repo / "output" / "corrections" / "synced.csv")
    r = SY.run()
    assert r["take_main"] == ["output/fd_history.parquet"] and r["both_changed"] == ["output/af_history.parquet"]
    af = pd.read_parquet(repo / "output" / "af_history.parquet")
    pairs = set(zip(af.home_team, af.away_team))
    # main's NEW row is added; the row this branch REMOVED (C-D, the branch's newest date) never returns
    assert ("E", "F") in pairs and ("C", "D") not in pairs and len(af) == 2
    assert r["rows_appended"]["output/af_history.parquet"] == 1
    assert len(pd.read_parquet(repo / "output" / "fd_history.parquet")) == 2


# ── owner 2026-11-09: date formats and the one-day rule for re-fetch rows ──────────────────

def _plan_case(list_date, hist_date, home="Uruguay", away="Dominican Republic", lh="Uruguay", la="Dominican Republic"):
    import collect_player_refetch as RF
    from test_player_refetch import _h
    fx = pd.DataFrame([{"fixture_id": 900, "league_id": 10, "season": "2025", "date": list_date,
                        "home_id": 1, "home_name": lh, "away_id": 2, "away_name": la, "status": "FT"},
                       {"fixture_id": 901, "league_id": 39, "season": "2025", "date": "2025-10-01",
                        "home_id": 3, "home_name": "Arsenal", "away_id": 4, "away_name": "Chelsea", "status": "FT"}])
    rows = RF.source_rows(pd.DataFrame([_h(900, 7, hist_date, home, away, home, 1)]), pd.DataFrame())
    return RF.plan(rows, fx, pd.DataFrame())


def test_a_list_date_with_an_offset_is_read_in_utc_not_dropped():
    p = _plan_case("2025-10-10T22:00:00+00:00", "2025-10-10")
    assert p["proven_fixtures"] == 1 and p["refused_fixtures"] == {}


def test_one_day_apart_with_both_clubs_proven_is_accepted_and_recorded():
    p = _plan_case("2025-10-11T01:30:00+00:00", "2025-10-10")
    assert p["proven_fixtures"] == 1 and p["date_shift_accepted_fixtures"] == 1
    dc = p["date_corrections"]
    assert (dc["history_date"].iloc[0], dc["provider_date"].iloc[0]) == ("2025-10-10", "2025-10-11")


def test_one_day_apart_with_a_club_unproven_or_two_days_apart_is_refused():
    p = _plan_case("2025-10-11T01:30:00+00:00", "2025-10-10", lh="Paraguay")
    assert p["proven_fixtures"] == 0 and p["refused_fixtures"] == {"clubs_unproven": 1}
    p = _plan_case("2025-10-12T01:30:00+00:00", "2025-10-10")
    assert p["proven_fixtures"] == 0 and p["refused_fixtures"] == {"date_differs_from_list": 1}


def test_every_refetch_output_path_is_redirected_by_the_test_fixture():
    """A collector path the shared test fixture does not redirect is written into the repository by
    the tests (it happened with DATE_CORRECTIONS on 2026-11-09)."""
    import inspect
    import collect_player_refetch as RF
    import test_player_refetch as T
    src = inspect.getsource(T.env.__wrapped__ if hasattr(T.env, "__wrapped__") else T.env)
    for n in ("OUT", "JOURNAL", "AUDIT", "DATE_CORRECTIONS"):
        assert f'"{n}"' in src, n
    assert RF.DATE_CORRECTIONS.name == "player_history_date_corrections.csv"
