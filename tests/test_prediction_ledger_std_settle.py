"""v9_standard rows of the prediction ledger settle (bug 2026-11-22: 2,964 of 2,964 past-kickoff
v9_standard rows PENDING). The Standard leagues are in neither API-Football history bank, and
football-data uploads a round days after it is played; the settlement now also reads the API-Football
results banks written by scripts/collect_standard_team_cards.py, AFTER football-data / af_history.

Pins: a Standard row with a played result settles; football-data still wins when it has the match;
HT markets settle only from an HT score (a bank without one is passed over, never inferred);
ledger-only league rows (fixture_ref af:<id>) settle on the API-Football fixture id; nothing settles
before kickoff; an FT-only bank never serves an extra-time score; the collector stores score.fulltime
(90 minutes), never `goals`. All banks are tmp files; nothing under output/ is written.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import prediction_ledger as PL
from src import prediction_ledger_v9 as V
from src import team_names as TN

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 22, 18, 0, tzinfo=timezone.utc)
KO = "2026-11-21T15:00:00Z"


def _row(**kw):
    base = dict(repo="v9", model_id="v9_standard", model_version="standard_x_1", market="ou25", line=2.5,
                selection="OVER", fixture_ref="oa:abc", oddsapi_event_id="abc", league="Championship",
                season="2026", kickoff_utc=KO, home_team="Norwich City", away_team="Cardiff City",
                probability=0.55, snapshot="first_sight", predicted_at="2026-11-20T12:00:00Z",
                minutes_to_kickoff=1620)
    base.update(kw)
    row = PL.make_row(**base)
    row["league_slug"] = PL.LeagueSlugs().slug(row["league"])      # PL.append stamps it on a real write
    return row


def _results(rows):
    cols = ["fixture_id", "league", "league_id", "season", "match_date", "kickoff_utc", "status", "home_team",
            "away_team", "home_team_id", "away_team_id", "ft_home", "ft_away", "ht_home", "ht_away", "source",
            "fetched_at"]
    d = pd.DataFrame([dict(zip(["fixture_id", "league", "match_date", "status", "home_team", "away_team",
                                "ft_home", "ft_away", "ht_home", "ht_away"], r)) for r in rows]).reindex(columns=cols)
    for c in ("ft_home", "ft_away", "ht_home", "ht_away"):
        d[c] = pd.to_numeric(d[c]).astype("Int64")
    return d


def _cards(rows):
    d = pd.DataFrame([dict(zip(["fixture_id", "league", "match_date", "status", "home_team", "away_team",
                                "FTHG", "FTAG"], r)) for r in rows])
    for c in ("FTHG", "FTAG"):
        d[c] = pd.to_numeric(d[c]).astype("Int64")
    return d


@pytest.fixture
def banks(tmp_path, monkeypatch):
    """No committed bank is read: every path points into tmp_path (absent unless a test writes it)."""
    paths = {k: tmp_path / f"{k}.parquet" for k in ("FD_HISTORY", "AF_HISTORY", "AF_RESULTS", "AF_STD_CARDS")}
    for k, p in paths.items():
        monkeypatch.setattr(V, k, p)
    return paths


def _settle(row, now=NOW):
    return V.outcome_fn(now=now, team=V.TeamResults(), latest_ko={})(row)


def test_standard_row_with_a_played_result_settles(banks):
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, 1, 0)]).to_parquet(
        banks["AF_RESULTS"])
    out = _settle(_row())
    assert out["outcome_status"] == "SETTLED" and out["y"] == 1
    assert (out["home_goals"], out["away_goals"]) == (2, 1)
    assert out["result_source"] == "api-football-results"
    under = _settle(_row(selection="UNDER"))
    assert under["outcome_status"] == "SETTLED" and under["y"] == 0


def test_std_cards_bank_settles_ft_markets_but_never_ht_or_extra_time(banks):
    _cards([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 0, 0)]).to_parquet(
        banks["AF_STD_CARDS"])
    out = _settle(_row(market="btts", line=None, selection="NO"))
    assert out["outcome_status"] == "SETTLED" and out["y"] == 1
    assert out["result_source"] == "api-football-std-cards"
    # no HT score in that bank -> the HT market waits (never inferred from FT 0-0)
    ht = _settle(_row(market="ht_ou05", line=0.5, selection="UNDER"))
    assert ht["outcome_status"] == "PENDING"
    # after 72 h with still no HT score anywhere -> NO_RESULT with the reason
    late = _settle(_row(market="ht_ou05", line=0.5, selection="UNDER"),
                   now=datetime(2026, 11, 25, 16, tzinfo=timezone.utc))
    assert late["outcome_status"] == "NO_RESULT" and "HT score" in late["outcome_value"]
    # an AET row in the FT-only bank (its goals include extra time) is not a 90-minute score
    _cards([(1001, "Championship", "2026-11-21", "AET", "Norwich", "Cardiff", 3, 2)]).to_parquet(
        banks["AF_STD_CARDS"])
    assert _settle(_row())["outcome_status"] == "PENDING"


def test_ht_markets_settle_from_ht_scores(banks):
    _cards([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1)]).to_parquet(
        banks["AF_STD_CARDS"])
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, 1, 1)]).to_parquet(
        banks["AF_RESULTS"])
    o05 = _settle(_row(market="ht_ou05", line=0.5, selection="OVER"))
    o15 = _settle(_row(market="ht_ou15", line=1.5, selection="OVER"))
    assert o05["outcome_status"] == o15["outcome_status"] == "SETTLED"
    assert (o05["y"], o15["y"]) == (1, 1) and o05["outcome_value"] == "1-1"
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, 1, 0)]).to_parquet(
        banks["AF_RESULTS"])
    o15 = _settle(_row(market="ht_ou15", line=1.5, selection="OVER"))
    assert o15["outcome_status"] == "SETTLED" and o15["y"] == 0
    # a missing HT score in the results bank stays missing -> pending, never 0-0
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, None, None)]).to_parquet(
        banks["AF_RESULTS"])
    assert _settle(_row(market="ht_ou05", line=0.5, selection="OVER"))["outcome_status"] == "PENDING"


def test_ledger_only_league_rows_settle_on_the_fixture_id(banks):
    # National League (TRAINING_ONLY, ledger-only rows #63): fd-style names in the ledger, API-Football
    # spellings in the bank that no resolver joins - the fixture id does.
    _results([(555, "National League", "2026-11-21", "FT", "Boreham Wood FC", "Gateshead FC", 1, 1, 0, 1)]
             ).to_parquet(banks["AF_RESULTS"])
    row = _row(fixture_ref="af:555", fixture_id="555", oddsapi_event_id="", league="National League",
               home_team="Boreham Wood", away_team="Gateshead", market="btts", line=None, selection="YES")
    out = _settle(row)
    assert out["outcome_status"] == "SETTLED" and out["y"] == 1 and out["outcome_value"] == "1-1"
    ht = _settle({**row, "market": "ht_ou05", "line": 0.5, "selection": "OVER"})
    assert ht["outcome_status"] == "SETTLED" and ht["y"] == 1
    # the same id under another league never settles the row
    _results([(555, "Championship", "2026-11-21", "FT", "X", "Y", 1, 1, 0, 1)]).to_parquet(banks["AF_RESULTS"])
    assert _settle(row)["outcome_status"] == "PENDING"


def test_football_data_wins_when_it_has_the_match(banks, monkeypatch):
    from src import history_corrections as HC
    monkeypatch.setattr(HC, "serve_history", lambda d, tag=None: d)
    pd.DataFrame([{"league": "Championship", "date": pd.Timestamp("2026-11-21"), "home_team": "Norwich",
                   "away_team": "Cardiff", "home_goals": 0.0, "away_goals": 0.0, "ht_home_goals": 0.0,
                   "ht_away_goals": 0.0}]).to_parquet(banks["FD_HISTORY"])
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, 1, 0)]).to_parquet(
        banks["AF_RESULTS"])
    out = _settle(_row())
    assert out["outcome_status"] == "SETTLED" and out["result_source"] == "football-data" and out["y"] == 0


def test_no_settlement_before_kickoff(banks):
    _results([(1001, "Championship", "2026-11-21", "FT", "Norwich", "Cardiff", 2, 1, 1, 0)]).to_parquet(
        banks["AF_RESULTS"])
    before = datetime(2026, 11, 21, 14, 59, tzinfo=timezone.utc)
    assert _settle(_row(), now=before) == {"outcome_status": "PENDING"}


def test_hertha_berlin_resolves_for_results_only():
    assert TN.resolve_result("Hertha Berlin", ["Hertha", "Darmstadt"]) == "Hertha"
    assert TN.resolve_result("Hertha Berlin", ["Hertha BSC", "Darmstadt"]) == "Hertha BSC"
    assert TN.resolve_result("Hertha Berlin", ["Hertha", "Hertha BSC"]) is None     # one club, two names
    assert TN.resolve("Hertha Berlin", ["Hertha", "Darmstadt"]) is None            # predict-time untouched


def _tc():
    spec = importlib.util.spec_from_file_location("collect_standard_team_cards_ss",
                                                  ROOT / "scripts" / "collect_standard_team_cards.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fx(fid, lid, status, ft, ht, goals):
    return {"fixture": {"id": fid, "date": "2026-11-21T15:00:00+00:00", "status": {"short": status}},
            "league": {"id": lid, "season": 2026},
            "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}},
            "goals": {"home": goals[0], "away": goals[1]},
            "score": {"halftime": {"home": ht[0], "away": ht[1]}, "fulltime": {"home": ft[0], "away": ft[1]}}}


def test_collector_results_bank_uses_the_90_minute_score(tmp_path, monkeypatch):
    TC = _tc()
    monkeypatch.setattr(TC, "OUTPUT", tmp_path / "cards.parquet")
    monkeypatch.setattr(TC, "RESULTS_OUTPUT", tmp_path / "results.parquet")
    r = TC.parse_result(_fx(1, 40, "AET", (1, 1), (0, 1), (2, 1)), "Championship", "t")
    assert (r["ft_home"], r["ft_away"], r["ht_home"], r["ht_away"]) == (1, 1, 0, 1)
    assert TC.parse_result(_fx(2, 40, "2H", (None, None), (0, 0), (1, 0)), "Championship", "t") is None
    assert TC.parse_result(_fx(3, 40, "FT", (2, 0), (None, None), (2, 0)), "Championship", "t")["ht_home"] is None

    class _C:
        calls, stopped, max_calls = 0, "", 100

        def get(self, endpoint, params):
            self.calls += 1
            if endpoint == "fixtures" and params["date"] == "2026-11-21":
                return {"response": [_fx(10, 43, "FT", (3, 1), (1, 1), (3, 1)),
                                     _fx(11, 999, "FT", (1, 0), (0, 0), (1, 0)),
                                     _fx(12, 40, "NS", (None, None), (None, None), (None, None))]}
            return {"response": []}
    out = TC.run(today="2026-11-21", client=_C())
    d = pd.read_parquet(tmp_path / "results.parquet")
    assert d["fixture_id"].tolist() == [10] and d["league"].tolist() == ["National League"]
    assert out["results"] == 1 and list(d.columns) == TC.RESULTS_COLUMNS
    assert TC.run(today="2026-11-21", client=_C())["results"] == 0        # first write wins


def test_workflow_stages_the_results_bank():
    t = (ROOT / ".github" / "workflows" / "af_history_extend.yml").read_text(encoding="utf-8")
    assert "git add -f output/af_results_recent.parquet" in t
