"""Producers Data Truth (dt2/producers, 2026-11-02): ids written at the source, provider nulls kept
as unknown, club identity through src.team_names only, no typed defaults in the views.

    python -m pytest tests/test_producers_data_truth.py -q
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from player_model import api_football as AF


def _fx(fid, home, away, status="NS"):
    return {"fixture": {"id": fid, "status": {"short": status}},
            "teams": {"home": {"name": home}, "away": {"name": away}},
            "goals": {"home": 2, "away": 1}, "score": {"halftime": {"home": None, "away": None}}}


# ── api_football: public raw accessors, resolver-based fixture match, nulls kept ────────────────
def test_raw_accessors_return_the_provider_response(monkeypatch):
    calls = []

    def fake_get(endpoint, params, cache_hours=24):
        calls.append((endpoint, params))
        return {"response": [{"endpoint": endpoint}]}

    monkeypatch.setattr(AF, "_get", fake_get)
    assert AF.fixture_players_raw(7) == [{"endpoint": "/fixtures/players"}]
    assert AF.fixture_events_raw(7) == [{"endpoint": "/fixtures/events"}]
    assert AF.fixture_statistics_raw(7) == [{"endpoint": "/fixtures/statistics"}]
    assert AF.fixture_by_id_raw(7) == {"endpoint": "/fixtures"}
    assert ("/fixtures", {"id": 7}) in calls
    monkeypatch.setattr(AF, "_get", lambda *a, **k: None)
    assert AF.fixture_players_raw(7) == [] and AF.fixture_by_id_raw(7) is None


def test_match_fixture_resolves_clubs_within_the_league_date_pool():
    pool = [_fx(1, "Manchester United", "Liverpool"), _fx(2, "Arsenal", "Chelsea")]
    assert AF.match_fixture(pool, "Manchester United", "Liverpool")["fixture"]["id"] == 1
    assert AF.match_fixture(pool, "Arsenal", "Chelsea")["fixture"]["id"] == 2
    # A pair that crosses two fixtures is not a fixture.
    assert AF.match_fixture(pool, "Arsenal", "Liverpool") is None
    # A club that is not in the pool resolves to nothing (no substring / fuzzy fallback).
    assert AF.match_fixture(pool, "Real Madrid", "Chelsea") is None
    assert AF.match_fixture([], "Arsenal", "Chelsea") is None


def test_senior_side_never_matches_its_reserve_side():
    pool = [_fx(9, "Real Sociedad II", "Barcelona B")]
    assert AF.match_fixture(pool, "Real Sociedad", "Barcelona") is None


def test_fixture_status_and_goals_use_the_resolved_fixture(monkeypatch):
    monkeypatch.setattr(AF, "fixtures_on_date_raw",
                        lambda *a, **k: [_fx(5, "Arsenal", "Chelsea", status="PST")])
    assert AF.get_fixture_status(39, "2026", "2026-11-01", "Arsenal", "Chelsea") == ("PST", 5)
    assert AF.get_fixture_status(39, "2026", "2026-11-01", "Arsenal", "Everton") == ("NOT_FOUND", None)
    assert AF.get_fixture_goals(39, "2026", "2026-11-01", "Arsenal", "Chelsea") == (2, 1, None, None)
    assert AF.find_fixture_id(39, "2026", "2026-11-01", "Arsenal", "Chelsea") == 5


def test_player_stats_keep_provider_nulls_as_none(monkeypatch):
    raw = [{"team": {"name": "Arsenal"}, "players": [
        {"player": {"id": 11, "name": "A Player"},
         "statistics": [{"games": {"minutes": None, "substitute": None, "position": "F"},
                         "goals": {"total": None, "assists": 0}, "shots": {"total": 3, "on": None},
                         "cards": {"yellow": 1, "red": None}}]}]}]
    monkeypatch.setattr(AF, "fixture_players_raw", lambda *a, **k: raw)
    s = AF.get_fixture_player_stats(100)[0]
    assert s["minutes_played"] is None and s["started"] is None
    assert s["goals"] is None and s["shots_on_target"] is None and s["red_card"] is None
    assert s["assists"] == 0 and s["shots_total"] == 3 and s["yellow_card"] == 1


def test_update_results_needs_no_private_api_football_import():
    # The public accessors exist for it; the private client stays private to api_football.
    for name in ("fixture_players_raw", "fixture_events_raw", "fixture_statistics_raw",
                 "fixture_by_id_raw", "fixtures_on_date_raw"):
        assert callable(getattr(AF, name))


# ── props ledger: (player_id, fixture_id, market) ─────────────────────────────────────────────
def _tip(pid=11, fid=500, market="goals", name="A. Player", date="2026-11-08"):
    return {"tier": "SNIPER", "date": date, "league": "Premier League", "match": "Arsenal vs Chelsea",
            "player_id": pid, "fixture_id": fid, "player_name": name, "team": "Arsenal",
            "position": "F", "market": market, "model_prob": 0.4, "market_odds": 2.8, "ev": 0.1,
            "edge_rel": 0.1, "n_games": 20}


def test_props_ledger_refuses_a_tip_without_fixture_id_and_dedupes_on_ids(tmp_path):
    from player_model import ledger as PL
    f = tmp_path / "player_ledger.csv"
    n = PL.append_player_signals(pd.DataFrame([_tip(), _tip(fid=None), _tip(pid=None),
                                               _tip(name="Namesake", pid=12)]), ledger_file=f)
    assert n == 2
    led = pd.read_csv(f)
    assert set(led["player_id"]) == {11, 12} and set(led["fixture_id"]) == {500}
    # Same ids again (another spelling of the name): not a new row.
    assert PL.append_player_signals(pd.DataFrame([_tip(name="Alex Player")]), ledger_file=f) == 0
    # Same player, other market -> new row.
    assert PL.append_player_signals(pd.DataFrame([_tip(market="sot")]), ledger_file=f) == 1


def test_props_ledger_legacy_row_without_fixture_id_still_blocks_a_duplicate(tmp_path):
    from player_model import ledger as PL
    f = tmp_path / "player_ledger.csv"
    legacy = {c: "" for c in PL.PLAYER_LEDGER_COLS}
    legacy.update(match_date="2026-11-08", player_id="11", market="goals", player_name="A. Player")
    pd.DataFrame([legacy]).to_csv(f, index=False)
    assert PL.append_player_signals(pd.DataFrame([_tip()]), ledger_file=f) == 0


# ── odds history: player_id where the match is unambiguous ────────────────────────────────────
def test_odds_rows_get_the_tip_player_id_only_when_unambiguous():
    from player_model import odds_fetcher as OF
    tips = pd.DataFrame([
        {"date": "2026-11-08", "match": "Arsenal vs Chelsea", "player_id": 11, "player_name": "Bukayo Saka"},
        {"date": "2026-11-08", "match": "Arsenal vs Chelsea", "player_id": 12, "player_name": "K. Havertz"},
        {"date": "2026-11-08", "match": "Arsenal vs Chelsea", "player_id": 13, "player_name": "J. Silva"},
        {"date": "2026-11-08", "match": "Arsenal vs Chelsea", "player_id": 14, "player_name": "Joao Silva"},
    ])
    recs = [{"match_date": "2026-11-08", "match": "Arsenal vs Chelsea", "player": p, "market": "goals",
             "odds": 3.0} for p in ("bukayo saka", "kai havertz", "jose silva", "cole palmer")]
    out = OF.attach_player_ids(recs, tips)
    ids = {r["player"]: r["player_id"] for r in recs}
    assert ids["bukayo saka"] == 11            # exact name
    assert ids["kai havertz"] == 12            # initial + surname, one candidate
    assert ids["jose silva"] is None           # two Silvas with initial J -> refused
    assert ids["cole palmer"] is None          # no tip player in that fixture
    assert out == {"matched": 2, "ambiguous": 1, "unmatched": 1}


def test_odds_rows_of_another_fixture_never_take_its_ids():
    from player_model import odds_fetcher as OF
    tips = pd.DataFrame([{"date": "2026-11-08", "match": "Arsenal vs Chelsea", "player_id": 11,
                          "player_name": "Bukayo Saka"}])
    recs = [{"match_date": "2026-11-08", "match": "Everton vs Fulham", "player": "bukayo saka",
             "market": "goals", "odds": 3.0}]
    OF.attach_player_ids(recs, tips)
    assert recs[0]["player_id"] is None


def test_carry_player_ids_fills_only_from_the_same_quoted_player():
    from player_model import odds_fetcher as OF
    df = pd.DataFrame([
        {"match_date": "2026-11-08", "league": "PL", "match": "A vs B", "player": "x", "player_id": None},
        {"match_date": "2026-11-08", "league": "PL", "match": "A vs B", "player": "x", "player_id": 7},
        {"match_date": "2026-11-08", "league": "PL", "match": "A vs B", "player": "y", "player_id": None},
    ])
    out = OF._carry_player_ids(df)
    assert list(out["player_id"].astype(object).where(out["player_id"].notna(), None)) == [7, 7, None]


# ── market movement: prop curves by player_id ─────────────────────────────────────────────────
def test_prop_curve_is_read_by_player_id_only(tmp_path):
    from src.market_movement import CurveIndex
    rows = []
    for pid, name, odds in ((11, "bukayo saka", (3.0, 2.5)), (12, "kai havertz", (4.0, 4.5))):
        for t, o in zip(("10", "18"), odds):
            rows.append({"snapshot_date": "2026-11-08", "snapshot_ts": f"2026-11-08T{t}:00:00Z",
                         "match_date": "2026-11-08", "league": "Premier League",
                         "match": "Arsenal vs Chelsea", "player": name, "market": "goals",
                         "odds": o, "player_id": pid})
    pd.DataFrame(rows).to_csv(tmp_path / "player_prop_odds_history.csv", index=False)
    idx = CurveIndex(sources=[("player_prop_odds_history.csv", "props")], out_dir=tmp_path)
    c = idx.curve("Premier League", "Arsenal", "Chelsea", "2026-11-08", "goals", player_id=11)
    assert (c["open"], c["close"]) == (3.0, 2.5)
    assert idx.curve("Premier League", "Arsenal", "Chelsea", "2026-11-08", "goals", player_id="12")["close"] == 4.5
    # No id -> no curve (never a name match, never a team-mate's curve).
    assert idx.curve("Premier League", "Arsenal", "Chelsea", "2026-11-08", "goals") is None
    assert idx.curve("Premier League", "Arsenal", "Chelsea", "2026-11-08", "goals", player_id=99) is None


def test_enrich_leaves_clv_unknown_without_an_entry_price(tmp_path):
    from src.market_movement import CurveIndex, enrich
    pd.DataFrame([{"snapshot_ts": f"2026-11-08T1{t}:00:00Z", "match_date": "2026-11-08",
                   "league": "Premier League", "match": "Arsenal vs Chelsea", "market": "over25",
                   "odds": o} for t, o in ((0, 2.0), (8, 1.8))]).to_csv(tmp_path / "s.csv", index=False)
    idx = CurveIndex(sources=[("s.csv", "standard")], out_dir=tmp_path)
    d = pd.DataFrame([{"league": "Premier League", "home_team": "Arsenal", "away_team": "Chelsea",
                       "match_date": "2026-11-08", "odds": None}])
    out = enrich(d, lambda r: "over25", idx=idx, entry_col="odds")
    assert out.iloc[0]["mv_close"] == 1.8 and math.isnan(out.iloc[0]["mv_clv_pct"])


# ── clv_tracker: a missing stat is unknown, not zero ──────────────────────────────────────────
def test_clv_settle_stats_keep_missing_as_none():
    from player_model import clv_tracker as CT

    class R:
        goals, assists, shots_on_target, yellow_cards, red_cards = 1.0, float("nan"), None, 0, 0
    s = CT._settle_stats(R)
    assert s["goals"] == 1.0 and s["assists"] is None and s["sot"] is None and s["yc"] == 0.0
    assert CT._MARKET_NEEDS["sot2"] == ("sot",) and CT._MARKET_NEEDS["cards"] == ("yc", "rc")
    assert CT._id_str("1635606.0") == "1635606" and CT._id_str("") is None


# ── fantasy: unknown availability is not recommendable ────────────────────────────────────────
def test_unknown_availability_is_never_recommendable():
    from player_model.fpl_api import availability_unknown, not_recommendable
    av = pd.Series(["available", "unknown", None, "injured", "doubtful", "not in squad"])
    assert list(availability_unknown(av)) == [False, True, True, False, False, False]
    assert list(not_recommendable(av)) == [False, True, True, True, False, True]
    assert list(not_recommendable(av, also=("doubtful",))) == [False, True, True, True, True, True]


def test_fpl_players_df_keeps_missing_numbers_nan():
    from player_model import fpl_api as F
    b = {"teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
         "elements": [{"id": 9, "first_name": "A", "second_name": "B", "team": 1, "element_type": 4,
                       "now_cost": 65, "status": "a"}]}
    r = F.players_df(b).iloc[0]
    assert r["price"] == 6.5 and r["availability"] == "available"
    for c in ("owned_pct", "fpl_form", "fpl_ppg", "fpl_ep_next", "total_points", "ict_index"):
        assert math.isnan(r[c]), c


# ── fantasy residual: NaN-native inputs ───────────────────────────────────────────────────────
def test_residual_matrix_keeps_nan_and_linear_arm_predicts_nan_for_incomplete_rows():
    from player_model import fantasy_residual as R
    x = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.inf], "b": [0.5, 1.0, 1.5, 2.0]})
    m = R._matrix(x, ["a", "b"])
    assert np.isnan(m[2, 0]) and np.isnan(m[3, 0]) and m[0, 0] == 1.0
    assert list(R._complete(m)) == [True, True, False, False]
    assert list(R._observed_columns(np.array([[np.nan, 1.0], [np.nan, 2.0]]))) == [1]


def test_minutes_model_is_nan_native():
    import inspect
    from player_model import fantasy_residual as R
    src = inspect.getsource(R.MinutesModel.fit)
    assert "HistGradientBoostingClassifier" in src and "import LogisticRegression" not in src
    assert ".fillna(0" not in inspect.getsource(R.prepare)


# ── Model Lab: no typed minimum ───────────────────────────────────────────────────────────────
def test_model_lab_minimum_comes_from_the_registry_or_gate_config(tmp_path):
    from dashboard_data import model_lab as M
    ev = tmp_path / "ev.json"
    ex = tmp_path / "ex.json"
    gate = tmp_path / "gate.json"
    ev.write_text(json.dumps({"computed_at": "x", "results": {
        "E1|standard": {"settled_paired": 50}, "E2|newformat": {"settled_paired": 10},
        "E3|standard": {}}}), encoding="utf-8")
    ex.write_text(json.dumps({"experiments": [{"experiment_id": "E1", "n_min_per_track": 100},
                                              {"experiment_id": "E2"}, {"experiment_id": "E3",
                                                                        "n_min_per_track": 100}]}),
                  encoding="utf-8")
    gate.write_text(json.dumps({"prospective": {}}), encoding="utf-8")
    d, _ = M.prospective(ev, ex, gate)
    # E2 has no registered minimum and the gate config none either; E3 has no settled count.
    assert list(d["experiment"]) == ["E1"] and d.iloc[0]["required_n"] == 100
    gate.write_text(json.dumps({"prospective": {"min_n_per_track": 40}}), encoding="utf-8")
    d, _ = M.prospective(ev, ex, gate)
    assert d.set_index("experiment").loc["E2", "required_n"] == 40


# ── team side: fixture identity into predictions and bets_ledger ──────────────────────────────
def test_attach_fixture_ids_resolves_api_football_and_keeps_oddsapi_id_apart():
    from src.predict import attach_fixture_ids
    board = pd.DataFrame([
        {"league": "Premier League", "date": pd.Timestamp("2026-11-08"), "home_team": "Arsenal",
         "away_team": "Chelsea", "_oa_event_id": "abc123"},
        {"league": "Premier League", "date": pd.Timestamp("2026-11-08"), "home_team": "Everton",
         "away_team": "Fulham", "_oa_event_id": ""},
    ])
    out = attach_fixture_ids(board, fixtures_for=lambda lid, season, day: [_fx(777, "Arsenal", "Chelsea")])
    assert out.loc[0, "fixture_id"] == 777 and pd.isna(out.loc[1, "fixture_id"])
    assert out.loc[0, "oddsapi_event_id"] == "abc123" and pd.isna(out.loc[1, "oddsapi_event_id"])


def test_bets_ledger_carries_both_fixture_ids(tmp_path, monkeypatch):
    import config
    from src import ledger as L
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    monkeypatch.setattr(L, "_version_fields", lambda *a, **k: {k: "" for k in L.VERSION_FIELDS})
    bet = {"date": "2026-11-08", "league": "Premier League", "home_team": "Arsenal",
           "away_team": "Chelsea", "bet": "OVER", "odds_over25": 1.9, "odds_under25": 1.95,
           "best_edge": 0.05, "signal_tier": "VALUABLE", "model_type": "standard",
           "fixture_id": 777.0, "oddsapi_event_id": "abc123"}
    no_id = dict(bet, home_team="Everton", away_team="Fulham", fixture_id=None, oddsapi_event_id=None)
    L.append_tips(pd.DataFrame([bet, no_id]))
    led = pd.read_csv(tmp_path / "bets_ledger.csv", dtype=str, keep_default_na=False).set_index("home_team")
    assert led.loc["Arsenal", "fixture_id"] == "777" and led.loc["Arsenal", "oddsapi_event_id"] == "abc123"
    assert led.loc["Everton", "fixture_id"] == "" and led.loc["Everton", "oddsapi_event_id"] == ""
