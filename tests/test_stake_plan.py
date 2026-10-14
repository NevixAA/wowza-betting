"""Stake plan (wired 2026-10-14): policy applied per match DAY, output only, first-sight audit."""
import pandas as pd

from src import stake_plan as sp


def _bets():
    rows = []
    for day in ("2026-10-17", "2026-10-18"):
        for i in range(8):
            rows.append({"date": day, "league": f"L{i}", "home_team": f"H{i}", "away_team": f"A{i}",
                         "best_side": "OVER", "odds_over25": 2.2, "odds_under25": 1.7,
                         "signal_tier": "SNIPER", "model_type": "new_format"})
    return pd.DataFrame(rows)


def _side():
    return pd.DataFrame([{"match_date": "2026-10-17", "league": "Brazil Serie A", "home_team": "X", "away_team": "Y",
                          "market": "btts", "odds": 2.05, "signal_tier": "SNIPER", "model_type": "new_format",
                          "side": "YES"},
                         {"match_date": "2026-10-17", "league": "USA MLS", "home_team": "P", "away_team": "Q",
                          "market": "btts", "odds": 2.05, "signal_tier": "SNIPER", "model_type": "new_format",
                          "side": "YES"}])


def test_sized_per_day_and_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    from src.bankroll_policy import load_policy
    cap = load_policy()["portfolio"]["daily_cap"]
    p = sp.write(_bets(), _side())
    per_day = p.groupby("date")["final_fraction"].sum()
    assert len(per_day) == 2 and (per_day <= cap + 1e-9).all()
    # both days get stakes of their own (a week squeezed into one daily cap would starve them)
    assert (per_day > 0).all()
    assert p.loc[p["league"] == "USA MLS", "final_fraction"].sum() == 0          # paper league
    assert set(p["bankroll_basis"]) == {sp.BASIS}


def test_audit_first_sight(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    sp.write(_bets(), _side())
    n1 = len(pd.read_csv(tmp_path / "audit.csv"))
    sp.write(_bets(), _side())
    assert len(pd.read_csv(tmp_path / "audit.csv")) == n1


def test_never_raises():
    assert sp.write(pd.DataFrame({"nonsense": [1]}), None).empty
