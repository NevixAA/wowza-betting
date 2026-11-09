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
