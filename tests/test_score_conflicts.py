"""2026-11-15 go-live rehearsal R3: a football-data / API-Football score conflict is settled by API-Football's
goal events, beside the bank (scripts/build_score_conflicts.py, src.history_corrections.apply_score_conflicts),
and data truth rule 5 is RED on any served conflict without a settled decision."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_score_conflicts as BSC  # noqa: E402
from src import history_corrections as HC  # noqa: E402
from src.integrity import data_truth as DT  # noqa: E402
from src.integrity import fixture_claims as FC  # noqa: E402

IRL = "Ireland Premier Division"


def _probe(home_id, goals):
    ev = [{"time": {"elapsed": m, "extra": None}, "team": {"id": t, "name": str(t)}, "type": "Goal", "detail": d}
          for m, t, d in goals]
    return {"calls": [{"name": "fixture", "response": [{"teams": {"home": {"id": home_id}, "away": {"id": 2}}}]},
                      {"name": "events", "response": ev}]}


def test_events_score_counts_90_minutes_and_credits_own_goals_to_the_named_side():
    p = _probe(1, [(19, 1, "Normal Goal"), (33, 2, "Normal Goal"), (57, 1, "Own Goal"), (66, 2, "Penalty"),
                   (80, 1, "Missed Penalty"), (94, 2, "Normal Goal")])          # 94' = extra time
    h, a, _ = BSC.events_score(p)
    assert (h, a) == (2, 2)


def test_decisions():
    assert BSC.decide((2, 1), (1, 2), (1, 2)) == "serve_provider_90min"      # UCD v Drogheda
    assert BSC.decide((0, 4), (0, 3), (0, 3)) == "serve_provider_90min"      # an extra-time goal in history
    assert BSC.decide((0, 0), (0, 1), (0, 0)) == "keep_history"              # provider counted extra time
    assert BSC.decide((0, 3), (1, 2), (1, 2)) == "exclude_awarded"           # history holds the forfeit
    assert BSC.decide((1, 1), (2, 0), (0, 0)) == "UNRESOLVED"
    assert BSC.decide((1, 1), (2, 0), None) == "UNRESOLVED"


def _hist(rows):
    return pd.DataFrame([{"league": IRL, "date": pd.Timestamp(d), "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag, "over25": float(hg + ag > 2.5), "ftr": "H",
                          "row_source": src} for d, h, a, hg, ag, src in rows])


LIST = pd.DataFrame([
    {"league": IRL, "date": "2026-11-07", "home_team": "UC Dublin", "away_team": "Drogheda", "history_ft_home": 2,
     "history_ft_away": 1, "decision": "serve_provider_90min", "served_ft_home": 1, "served_ft_away": 2,
     "fixture_id": 1648307},
    {"league": IRL, "date": "2023-12-19", "home_team": "A", "away_team": "B", "history_ft_home": 0,
     "history_ft_away": 3, "decision": "exclude_awarded", "served_ft_home": None, "served_ft_away": None,
     "fixture_id": 2},
    {"league": IRL, "date": "2022-08-29", "home_team": "Sligo Rovers", "away_team": "Dundalk", "history_ft_home": 2,
     "history_ft_away": 0, "decision": "keep_history", "served_ft_home": 2, "served_ft_away": 0, "fixture_id": 3}])


def test_apply_serves_excludes_keeps_for_any_source():
    h = _hist([("2026-11-07", "UC Dublin", "Drogheda", 2, 1, "fd_cache"),
               ("2023-12-19", "A", "B", 0, 3, "backtest_archive"),
               ("2022-08-29", "Sligo Rovers", "Dundalk", 2, 0, "fd_cache"),
               ("2022-09-01", "X", "Y", 1, 1, "fd_cache")])
    out, rep = HC.apply_score_conflicts(h, LIST)
    assert len(out) == 3 and "A" not in set(out["home_team"])
    ucd = out[out["home_team"] == "UC Dublin"].iloc[0]
    assert (ucd["home_goals"], ucd["away_goals"], ucd["ftr"]) == (1.0, 2.0, "A")
    assert out[out["home_team"] == "Sligo Rovers"].iloc[0]["home_goals"] == 2
    assert rep["conflict_rows_served_provider_score"] == 1 and rep["conflict_rows_excluded_awarded"] == 1


def test_a_row_changed_since_listing_is_left_alone():
    h = _hist([("2026-11-07", "UC Dublin", "Drogheda", 3, 1, "fd_cache")])
    out, rep = HC.apply_score_conflicts(h, LIST)
    assert out.iloc[0]["home_goals"] == 3 and rep["conflict_listed_rows_changed_since_listing"] == 1


POOL = FC.provider_pool(ht=pd.DataFrame([
    {"fixture_id": 1648307, "league": IRL, "date": "2026-11-07", "home_name": "UC Dublin", "away_name": "Drogheda",
     "ft_home": 1, "ft_away": 2, "status": "FT"}]), comp=pd.DataFrame())


def _bank(hg, ag):
    return pd.DataFrame({"date": [pd.Timestamp("2026-11-07")], "league": [IRL], "season": ["2026"],
                         "home_team": ["UC Dublin"], "away_team": ["Drogheda"], "home_goals": [float(hg)],
                         "away_goals": [float(ag)]})


def test_rule5_red_on_an_unsettled_conflict_green_once_settled(monkeypatch):
    monkeypatch.setattr(HC, "read_score_conflicts", lambda path=None: pd.DataFrame(columns=list(LIST.columns)))
    c = DT.one_match({"fd_cache": _bank(2, 1)}, pool=POOL)
    assert c["status"] == DT.RED and "1 unsettled provider score conflict(s)" in c["value"]
    assert "provider fixture 1648307 1-2, served 2-1" in c["detail"]
    monkeypatch.setattr(HC, "read_score_conflicts", lambda path=None: LIST)
    c = DT.one_match({"fd_cache": _bank(2, 1)}, pool=POOL)
    assert c["status"] == DT.GREEN, c["value"]


def test_committed_list_settles_the_rehearsal_conflict_and_the_seven_refused_rows():
    d = pd.read_csv(HC.SCORE_CONFLICTS)
    assert set(d["decision"]) <= set(HC.CONFLICT_DECISIONS)
    ucd = d[d["fixture_id"] == 1648307].iloc[0]
    assert (ucd["home_team"], ucd["history_ft_home"], ucd["history_ft_away"], ucd["decision"],
            ucd["served_ft_home"], ucd["served_ft_away"]) == ("UC Dublin", 2, 1, "serve_provider_90min", 1, 2)
    refused = pd.read_csv(ROOT / "output" / "corrections" / "history_score_refused.csv")
    assert set(refused["fixture_id"]) <= set(d["fixture_id"])
    assert d["evidence"].str.startswith("API-Football /fixtures/events").all()
