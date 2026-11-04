"""The 'or 0' bug (owner 2026-11-04): a statistic API-Football did not send stays missing.

player_model/data_fetcher.py turned a missing corner count, total shots, possession and every player
count into 0 (`int(stats.get(...) or 0)`); feature_engineering then read a missing count as "did not
score" (`(NaN >= 1).astype(int)`), and the live scanner read a silent feed as 0 shots on target.
Each is None / NaN now, a recorded 0 is still 0, and nothing downstream casts the gap back to 0.

Owner 2026-11-04 (fix round 8, refined in fix round 9): API-Football sends null, not 0, for a count a
player did not record in a match he played. A per-player null count is 0 when the player played
(minutes > 0) AND the provider returned player statistics for the fixture, even if nobody in the match
has that statistic; a fixture the provider did not answer, or a player with no / 0 minutes, stays
missing (player_model/player_stat_nulls.py, the one shared reading).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

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


def test_fixture_player_counts_without_the_fixture_answer_stay_none():
    """A caller that does not pass the fixture's answer (`answered`) cannot prove the provider
    covered the statistic: every null stays None (the fix-round-7 reading)."""
    r = DF._parse_fixture_player(_player({"games": {"minutes": 90, "substitute": False},
                                          "goals": {"total": None, "assists": 1, "conceded": 0},
                                          "shots": {"total": None, "on": None}, "cards": {"yellow": 0}}),
                                 META, "Premier League")
    assert r["goals"] is None and r["shots_total"] is None and r["shots_on_target"] is None
    assert r["assists"] == 1 and r["yellow_cards"] == 0 and r["goals_conceded"] == 0
    assert r["team_corners"] is None and r["passes_total"] is None and r["saves"] is None
    # no minutes recorded -> no row (never a 0-minute appearance)
    assert DF._parse_fixture_player(_player({"games": {"minutes": None}}), META, "Premier League") is None


# ── the owner's null rule (2026-11-04, fix round 8) ─────────────────────────────────────────────

def _entry(pid, minutes, **groups):
    return {"player": {"id": pid, "name": f"P{pid}"}, "statistics": [{"games": {"minutes": minutes}, **groups}]}


def _answer(*players):
    return [{"team": {"id": 1, "name": "H"}, "players": list(players)}]


def test_null_is_zero_for_a_player_who_played_in_a_fixture_the_provider_answered():
    """Owner 2026-11-04, refined in fix round 9: 0 EVEN IF nobody in the match has the statistic."""
    from player_model import player_stat_nulls as N
    scorer = _entry(1, 90, goals={"total": 1, "assists": None}, shots={"total": 3, "on": 2},
                    passes={"total": 40, "key": None, "accuracy": None}, penalty={"scored": None})
    quiet = _entry(2, 70, goals={"total": None, "assists": None}, shots={"total": None, "on": None},
                   passes={"total": 22, "key": None, "accuracy": None}, penalty={"scored": None})
    bench = _entry(3, None, goals={"total": None}, shots={"total": None, "on": None})
    zero_min = _entry(4, 0, goals={"total": None})
    ans = N.fixture_answered(_answer(scorer, quiet, bench, zero_min))
    assert ans is True
    q = N.player_counts(quiet["statistics"][0], ans)
    assert q["goals"] == 0 and q["shots_total"] == 0 and q["shots_on_target"] == 0   # played, answered
    assert q["passes_total"] == 22                                                    # sent value kept
    # nobody in the match has assists / key passes / a penalty: still 0 for a player who played
    assert q["assists"] == 0 and q["key_passes"] == 0 and q["penalty_scored"] == 0
    assert q["penalty_missed"] == 0 and q["saves"] == 0                               # absent group too
    assert all(v is None for v in N.player_counts(bench["statistics"][0], ans).values())     # no minutes
    assert all(v is None for v in N.player_counts(zero_min["statistics"][0], ans).values())  # 0 minutes
    # a fixture the provider did not answer: nothing is read as 0, even for a player with minutes
    assert N.fixture_answered([]) is False and N.fixture_answered(None) is False
    assert N.fixture_answered([{"team": {"id": 1}, "players": []}]) is False
    assert N.fixture_answered([{"team": {"id": 1}, "players": [{"player": {"id": 5}, "statistics": []}]}]) is False
    unanswered = N.player_counts(quiet["statistics"][0], False)
    assert unanswered["goals"] is None and unanswered["passes_total"] == 22          # sent value kept
    assert all(v is None for k, v in N.player_counts(quiet["statistics"][0], None).items() if k != "passes_total")
    # a bench entry alone is still an answer (its statistics block exists), but he did not play
    assert N.fixture_answered(_answer(bench)) is True
    # a non-null value that is not a count is missing, never 0; rating / accuracy are not counts
    assert N.read_count("n/a", 90, True) is None and N.read_count(None, 90, True) == 0
    assert N.read_count(None, 90, False) is None and N.read_count(0, None, False) == 0
    assert "rating" not in N.COUNT_FIELDS and "passes_accuracy" not in N.COUNT_FIELDS
    assert N.rating_of({"games": {"rating": None}}) is None and N.rating_of({"games": {"rating": "7.1"}}) == 7.1
    # the fix-round-8 per-statistic set is refused, not silently read as "answered"
    with pytest.raises(TypeError):
        N.player_counts(quiet["statistics"][0], frozenset({"goals"}))


def test_parsed_rows_carry_their_provenance():
    from player_model import player_stat_nulls as N
    quiet = _entry(2, 70, goals={"total": None})
    r = DF._parse_fixture_player(quiet, META, "Premier League", True)
    assert r[N.SOURCE_COL] == N.PARSED_WITH_RULE and r["goals"] == 0 and r["penalty_scored"] == 0
    r0 = DF._parse_fixture_player(quiet, META, "Premier League")
    assert r0[N.SOURCE_COL] == N.PARSED_NO_PROOF and r0["goals"] is None
    assert N.PARSED_WITH_RULE in N.VERIFIED_SOURCES and N.UNVERIFIED not in N.VERIFIED_SOURCES
    # a blank season is written as missing, never ""
    assert DF._parse_fixture_player(quiet, {**META, "season": ""}, "Friendlies", True)["season"] is None
    assert DF._parse_fixture_player(quiet, {k: v for k, v in META.items() if k != "season"}, "X", True)["season"] is None


def test_every_history_count_column_is_read_by_the_shared_rule():
    from player_model import player_stat_nulls as N
    stats = {"games": {"minutes": 90, "substitute": False}}
    full = {"goals": {"total": 1, "assists": 1, "conceded": 1, "saves": 1}, "shots": {"total": 1, "on": 1},
            "cards": {"yellow": 1, "red": 1}, "passes": {"total": 1, "key": 1}, "fouls": {"committed": 1, "drawn": 1},
            "duels": {"total": 1, "won": 1}, "offsides": 1, "tackles": {"total": 1, "interceptions": 1, "blocks": 1},
            "dribbles": {"attempts": 1, "success": 1, "past": 1},
            "penalty": {"scored": 1, "missed": 1, "won": 1, "saved": 1, "commited": 1}}
    other = _entry(9, 90, **full)
    ans = N.fixture_answered(_answer(other))
    assert ans is True
    r = DF._parse_fixture_player({"player": {"id": 5, "name": "Quiet"}, "statistics": [stats]}, META,
                                 "Premier League", ans)
    for col in N.HISTORY_COUNT_COLS:
        assert r[col] == 0, col                                              # played + answered -> 0
    assert r["rating"] is None and r["minutes"] == 90
    r9 = DF._parse_fixture_player(other, META, "Premier League", ans)
    assert all(r9[c] == 1 for c in N.HISTORY_COUNT_COLS)                     # sent values kept


def test_cup_collector_and_live_player_stats_use_the_rule(monkeypatch):
    from player_model import api_football as AF
    played = _entry(11, 80, goals={"total": None}, shots={"on": None}, cards={"yellow": None, "red": None})
    other = _entry(12, 90, goals={"total": 2}, shots={"on": 3}, cards={"yellow": 1, "red": 0})
    unused = _entry(13, None, goals={"total": None})
    raw = [{"team": {"id": 1, "name": "H"}, "players": [played, other, unused]}]
    monkeypatch.setattr(AF, "fixture_players_raw", lambda *a, **k: raw)
    s = {r["player_id"]: r for r in AF.get_fixture_player_stats(100)}
    assert s[11]["goals"] == 0 and s[11]["shots_on_target"] == 0 and s[11]["yellow_card"] == 0
    assert s[11]["red_card"] == 0 and s[11]["assists"] == 0                   # nobody has assists: 0
    assert s[13]["goals"] is None                                             # unused substitute
    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_cup_player_stats as CUP
    fix = pd.Series({"fixture_id": 77, "home_id": 1, "away_id": 2, "home_name": "H", "away_name": "A",
                     "date": "2026-10-01", "season": 2026, "league": "Champions League", "status": "FT"})
    rows = {r["player_id"]: r for r in CUP.parse({"response": raw}, fix)}
    assert rows[11]["goals"] == 0 and rows[11]["assists"] == 0 and 13 not in rows
    assert rows[11]["count_source"] == "provider_rule" and "count_source" in CUP.COLS
    # the provider did not answer the fixture: nothing is read as 0, nothing is stored
    assert CUP.parse({"response": []}, fix) == []


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
