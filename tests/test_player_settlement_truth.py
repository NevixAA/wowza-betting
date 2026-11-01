"""Player-prop settlement grades what happened, never a guess (Data Truth audit 16, 19, 23, 25).

    python -m pytest tests/test_player_settlement_truth.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import update_results as U


@pytest.mark.parametrize("market,stats,want", [
    ("sot", {"shots_on_target": 1}, "WIN"),
    ("sot2", {"shots_on_target": 1}, "LOSS"),
    ("sot2", {"shots_on_target": 2}, "WIN"),
    ("sot3", {"shots_on_target": 2}, "LOSS"),
    ("goals2", {"goals": 2}, "WIN"),
    ("cards", {"yellow_card": 0}, "LOSS"),
    ("assists", {"assists": 1}, "WIN"),
    ("corners", {"corners": 3}, ""),
])
def test_threshold_markets_settle(market, stats, want):
    assert U._resolve_player_market(market, stats) == want


def _ledger(tmp_path, monkeypatch, days_ago, league="Premier League", player="Bukayo Saka"):
    d = (datetime.utcnow().date() - timedelta(days=days_ago)).isoformat()
    p = tmp_path / "player_ledger.csv"
    pd.DataFrame([{"match_date": d, "league": league, "home_team": "Arsenal", "away_team": "Spurs",
                   "player_name": player, "market": "sot2", "market_odds": "2.5", "result": "",
                   "pnl": "", "resolved_date": ""}]).to_csv(p, index=False)
    monkeypatch.setattr(U, "PLAYER_LEDGER_FILE", p)
    monkeypatch.setenv("APIFOOTBALL_KEY", "x")
    return p


def _api(monkeypatch, stats, seen=None):
    import player_model.api_football as AF

    def fid(league_id, season, date_str, home, away):
        if seen is not None:
            seen.append(season)
        return 1
    monkeypatch.setattr(AF, "find_fixture_id", fid)
    monkeypatch.setattr(AF, "get_fixture_player_stats", lambda _id: stats)


def test_no_stats_yet_stays_pending(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].fillna("").iloc[0] == ""


def test_no_stats_after_a_week_is_void_with_reason(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=8)
    _api(monkeypatch, [])
    U.update_player_results()
    row = pd.read_csv(p, dtype=str).iloc[0]
    assert row["result"] == "VOID" and row["notes"] == "NO_STATS_7D"


def test_player_absent_from_published_stats_is_dnp(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [{"player_name": "Declan Rice", "minutes_played": 90, "shots_on_target": 1}])
    U.update_player_results()
    row = pd.read_csv(p, dtype=str).iloc[0]
    assert row["result"] == "VOID" and row["notes"] == "DNP"


def test_threshold_prop_settles(tmp_path, monkeypatch):
    p = _ledger(tmp_path, monkeypatch, days_ago=2)
    _api(monkeypatch, [{"player_name": "Bukayo Saka", "minutes_played": 90, "shots_on_target": 3}])
    U.update_player_results()
    assert pd.read_csv(p, dtype=str)["result"].iloc[0] == "WIN"


@pytest.mark.parametrize("today,want", [("2026-11-01", ["2627", "2526"]),
                                        ("2027-03-01", ["2627", "2526"]),
                                        ("2027-08-15", ["2728", "2627"])])
def test_fd_std_settlement_reads_the_live_season(today, want):
    assert U._fd_std_season_codes(today) == want


def test_season_comes_from_the_registry(tmp_path, monkeypatch):
    from src.season_resolver import resolve_api_season
    _ledger(tmp_path, monkeypatch, days_ago=2, league="Japan J-League")
    seen = []
    _api(monkeypatch, [], seen)
    U.update_player_results()
    d = (datetime.utcnow().date() - timedelta(days=2)).isoformat()
    assert seen == [resolve_api_season("Japan J-League", d)]
