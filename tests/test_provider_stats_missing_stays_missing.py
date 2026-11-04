"""The 'or 0' bug (owner 2026-11-04): a statistic API-Football did not send stays missing.

player_model/data_fetcher.py turned a missing corner count, total shots, possession and every player
count into 0 (`int(stats.get(...) or 0)`); feature_engineering then read a missing count as "did not
score" (`(NaN >= 1).astype(int)`), and the live scanner read a silent feed as 0 shots on target.
Each is None / NaN now, a recorded 0 is still 0, and nothing downstream casts the gap back to 0.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import data_fetcher as DF  # noqa: E402


def _stats(corners, shots, poss):
    return {"response": [{"team": {"name": "Home FC"},
                          "statistics": [{"type": "Corner Kicks", "value": corners},
                                         {"type": "Total Shots", "value": shots},
                                         {"type": "Ball Possession", "value": poss}]}]}


def test_count_and_pct_helpers():
    assert DF._count_or_none(None) is None and DF._count_or_none("") is None and DF._count_or_none("x") is None
    assert DF._count_or_none(0) == 0 and DF._count_or_none("7") == 7          # a recorded 0 is a real 0
    assert DF._pct_or_none(None) is None and DF._pct_or_none("%") is None
    assert DF._pct_or_none("54%") == 54.0 and DF._pct_or_none(0) == 0.0


def test_fixture_statistics_missing_stay_none_and_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(DF, "FIXTURE_STATS_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(DF, "_api_get", lambda ep, params: calls.append(ep) or _stats(None, None, None))
    got = DF._fetch_fixture_statistics(1)
    assert got == {"Home FC": {"corners": None, "shots_total": None, "possession": None}}
    assert json.loads((tmp_path / "stats_1.json").read_text())["schema"] == DF.FIXTURE_STATS_SCHEMA
    assert DF._fetch_fixture_statistics(1) == got and len(calls) == 1          # served from the v2 cache
    monkeypatch.setattr(DF, "_api_get", lambda ep, params: _stats(0, 11, "61%"))
    assert DF._fetch_fixture_statistics(2) == {"Home FC": {"corners": 0, "shots_total": 11, "possession": 61.0}}


def test_legacy_cache_zeros_are_refused_nonzero_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(DF, "FIXTURE_STATS_CACHE_DIR", tmp_path)
    (tmp_path / "stats_3.json").write_text(json.dumps({"A": {"corners": 0, "shots_total": 9, "possession": 0.0},
                                                       "B": {"corners": 4, "shots_total": 0, "possession": 47.0}}))
    monkeypatch.setattr(DF, "_api_get", lambda *a: (_ for _ in ()).throw(AssertionError("no API call")))
    assert DF._fetch_fixture_statistics(3) == {"A": {"corners": None, "shots_total": 9, "possession": None},
                                               "B": {"corners": 4, "shots_total": None, "possession": 47.0}}


def _player(stats):
    return {"player": {"id": 101, "name": "P One"}, "statistics": [stats]}


META = {"fixture_id": 9, "date": "2026-10-01", "home_team": "H", "away_team": "A", "team": "H", "is_home": 1,
        "team_corners": None, "referee": "", "season": "2026"}


def test_fixture_player_counts_missing_stay_none():
    r = DF._parse_fixture_player(_player({"games": {"minutes": 90, "substitute": False},
                                          "goals": {"total": None, "assists": 1, "conceded": 0},
                                          "shots": {"total": None, "on": None}, "cards": {"yellow": 0}}),
                                 META, "Premier League")
    assert r["goals"] is None and r["shots_total"] is None and r["shots_on_target"] is None
    assert r["assists"] == 1 and r["yellow_cards"] == 0 and r["goals_conceded"] == 0
    assert r["team_corners"] is None and r["passes_total"] is None and r["saves"] is None
    # no minutes recorded -> no row (never a 0-minute appearance)
    assert DF._parse_fixture_player(_player({"games": {"minutes": None}}), META, "Premier League") is None


def test_season_stats_missing_appearances_or_counts_are_none(tmp_path, monkeypatch):
    monkeypatch.setenv("APIFOOTBALL_KEY", "test-not-a-key")
    monkeypatch.setattr(DF, "_SEASON_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(DF, "CACHE_DIR", tmp_path)

    def _resp(games, goals, passes):
        return {"response": [{"player": {}, "statistics": [{"games": games, "goals": goals, "shots": {"total": 20},
                                                             "cards": {"yellow": 2, "red": 0}, "passes": passes}]}]}
    monkeypatch.setattr(DF, "_api_get", lambda *a: _resp({"appearences": None, "minutes": 900}, {"total": 3}, {}))
    r = DF.fetch_player_season_stats(1, 39, "2026")
    assert r["season_appearances"] is None and r["season_goals_pg"] is None and r["season_pass_accuracy"] is None
    monkeypatch.setattr(DF, "_api_get", lambda *a: _resp({"appearences": 10, "minutes": 900, "lineups": 8},
                                                           {"total": None}, {"accuracy": "81"}))
    r = DF.fetch_player_season_stats(2, 39, "2026")
    assert r["season_goals_pg"] is None and r["season_shots_pg"] == 2.0 and r["season_cards_pg"] == 0.2
    assert r["season_start_rate"] == 0.8 and math.isclose(r["season_pass_accuracy"], 0.81)
    assert r["season_dribble_pg"] is None                                    # not sent -> None, not 0


def test_season_cache_written_before_the_fix_is_not_served(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setenv("APIFOOTBALL_KEY", "test-not-a-key")
    monkeypatch.setattr(DF, "_SEASON_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(DF, "CACHE_DIR", tmp_path)
    (tmp_path / "ss_5_39_2026.json").write_text(json.dumps({"_cached_at": datetime.now().isoformat(),
                                                             "stats": {"season_goals_pg": 0.0}}))
    monkeypatch.setattr(DF, "_api_get", lambda *a: {"response": []})
    assert DF.fetch_player_season_stats(5, 39, "2026") == {}               # re-read, legacy 0 not served


def test_targets_stay_nan_where_the_count_is_missing():
    from player_model.feature_engineering import _label
    d = pd.DataFrame({"goals": [0, 1, 2, None]})
    t = _label(d, "goals", 1)
    assert t.iloc[:3].tolist() == [0.0, 1.0, 1.0] and pd.isna(t.iloc[3])
    assert _label(d, "goals", 2).iloc[:3].tolist() == [0.0, 0.0, 1.0]
    assert _label(d, "absent", 1).isna().all()


def test_live_scanner_silent_feed_uses_the_prior_only():
    from src import live_scanner as L
    prior_only = L._live_lambda_remaining(2.7, 30, None, 0, 0)
    assert math.isclose(prior_only, max(0.05, 2.7 / 90 * 60))
    assert L._live_lambda_remaining(2.7, 30, float("nan"), 0, 0) == prior_only
    assert L._live_lambda_remaining(2.7, 30, 0, 0, 0) < prior_only            # a real 0 SOT is evidence


def test_no_or_zero_on_provider_statistics_in_the_collector():
    src = (ROOT / "player_model" / "data_fetcher.py").read_text(encoding="utf-8")
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    hits = re.findall(r'int\(\s*\w+\.get\("[^"]+"\)\s*or 0\s*\)|float\([^()]*or "0%"', code)
    assert not hits, hits
