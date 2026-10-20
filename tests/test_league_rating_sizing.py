"""League-rating sizing in the stake plan (owner 2026-10-18): 10 = today's stake, 1 = 20% of it."""
import pandas as pd
import pytest

from src.bankroll_policy import load_policy, rating_scale, size_day

SEL = pd.DataFrame([{"date": "2026-10-18", "fixture": "f1", "league": "Brazil Serie A", "model_type": "new_format",
                     "market": "BTTS", "side": "YES", "signal_tier": "SNIPER", "odds": 2.0},
                    {"date": "2026-10-18", "fixture": "f2", "league": "League One", "model_type": "new_format",
                     "market": "BTTS", "side": "YES", "signal_tier": "SNIPER", "odds": 2.0}])


def test_scale_bounds():
    assert rating_scale(10, 0.2) == 1.0
    assert rating_scale(1, 0.2) == pytest.approx(0.2)
    assert rating_scale(99, 0.2) == 1.0                         # never above today's stake
    assert rating_scale(None, 0.2) == pytest.approx(0.2)       # unrated -> minimum


def test_rated_stakes_scale_down_only():
    pol = load_policy()
    base = size_day(SEL, pol, 100.0, 100.0)                     # no ratings passed: unchanged (Pro simulator)
    rated = size_day(SEL, pol, 100.0, 100.0, league_ratings={"Brazil Serie A": 10, "League One": 1})
    b = base.set_index("fixture")["requested_fraction"]
    r = rated.set_index("fixture")["requested_fraction"]
    assert r["f1"] == pytest.approx(b["f1"]) and r["f2"] == pytest.approx(0.2 * b["f2"])
    assert "league_rating" in rated.columns and "league_rating" not in base.columns
    none = size_day(SEL, pol, 100.0, 100.0, league_ratings={})  # ratings unavailable -> every league minimum
    assert (none["requested_fraction"] <= base["requested_fraction"] * 0.2 + 1e-12).all()


def test_disabled_policy_ignores_ratings():
    pol = load_policy()
    pol = {**pol, "league_rating_sizing": {"enabled": False}}
    a = size_day(SEL, pol, 100.0, 100.0, league_ratings={"League One": 1})
    b = size_day(SEL, pol, 100.0, 100.0)
    assert a["requested_fraction"].tolist() == b["requested_fraction"].tolist()


def test_audit_append_keeps_existing_header(tmp_path, monkeypatch):
    from src import stake_plan as sp
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    monkeypatch.setattr(sp, "load_league_ratings", lambda: {"Brazil Serie A": 10})
    old_cols = ["policy_version", "fixture", "track", "final_fraction"]
    pd.DataFrame([{"policy_version": "x", "fixture": "old", "track": "t", "final_fraction": 0.01}]).to_csv(
        tmp_path / "audit.csv", index=False)
    bets = pd.DataFrame([{"date": "2026-10-18", "league": "Brazil Serie A", "home_team": "A", "away_team": "B",
                          "best_side": "OVER", "odds_over25": 2.1, "odds_under25": 1.8, "signal_tier": "SNIPER",
                          "model_type": "new_format"}])
    sp.write(bets, None)
    audit = pd.read_csv(tmp_path / "audit.csv")
    assert list(audit.columns) == old_cols and len(audit) == 2      # appended under the old header
