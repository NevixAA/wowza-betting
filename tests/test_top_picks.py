"""Top picks (2026-10-18): three lists of 5, frozen once per day, graded from the ledgers."""
import numpy as np
import pandas as pd
import pytest

from src import top_picks as TP


def _ledgers(tmp_path, n_days=40, per_day=12, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-08-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    side_rows = []
    for d in days:
        for i in range(per_day):
            odds = float(rng.uniform(1.3, 2.6))
            win = rng.uniform() < 1 / odds
            side_rows.append({"match_date": d, "generated_at": f"{d}T06:00", "league": "L", "home_team": f"H{i}",
                              "away_team": f"A{i}", "market": ["btts", "over15", "over35"][i % 3], "side": "YES",
                              "odds": round(odds, 2), "edge_pct": 5.0, "model_prob": min(0.95, 1 / odds + 0.05),
                              "signal_tier": "SNIPER", "model_type": "new_format",
                              "result": "WIN" if win else "LOSS"})
    pd.DataFrame(side_rows).to_csv(tmp_path / "side_bets_ledger.csv", index=False)
    return list(days)


@pytest.fixture
def env(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(TP, "LOG", tmp_path / "top_picks_log.csv")
    return tmp_path


def test_lists_shape_and_rules(env):
    days = _ledgers(env)
    d = TP.todays_lists(days[-1])
    assert set(d["list"]) == {"BEST_BETS", "SAFEST", "VALUE"}
    assert (d.groupby("list").size() <= TP.TOP_K).all()
    safest = d[d["list"] == "SAFEST"]["odds"]
    day = TP.load_tips().query("day == @days[-1]")
    assert sorted(safest.tolist()) == sorted(day.nsmallest(5, "odds")["odds"].tolist())     # 5 shortest prices
    assert (d[d["list"] == "BEST_BETS"]["p_win"] >= TP.HIT_FLOOR).all()


def test_p_win_uses_only_earlier_days(env):
    days = _ledgers(env)
    tips = TP.load_tips()
    first = tips[tips["day"] == days[0]]
    assert TP.p_win(first, days[0], tips).isna().all()          # nothing earlier -> no model, no leak


def test_freeze_once_and_grade(env):
    days = _ledgers(env)
    a = TP.freeze_day(days[-1])
    b = TP.freeze_day(days[-1])                                  # second call must not rewrite
    assert len(a) == len(b) and len(pd.read_csv(TP.LOG)) == len(a)
    sb = TP.scoreboard()
    assert set(sb["list"]) <= {"BEST_BETS", "SAFEST", "VALUE"} and (sb["picks"] > 0).all()
    assert "/" in TP.day_line(days[-1])


def test_no_ledgers_is_empty(env):
    assert TP.todays_lists("2026-10-18").empty
    assert TP.freeze_day("2026-10-18").empty
    assert TP.scoreboard().empty
