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
    assert rating_scale(None, 0.2) is None                     # unrated -> no scale (no evidence)


def test_rated_stakes_scale_down_only():
    pol = load_policy()
    base = size_day(SEL, pol, 100.0, 100.0)                     # no ratings passed: unchanged (Pro simulator)
    rated = size_day(SEL, pol, 100.0, 100.0, league_ratings={"Brazil Serie A": 10, "League One": 1})
    b = base.set_index("fixture")["requested_fraction"]
    r = rated.set_index("fixture")["requested_fraction"]
    assert r["f1"] == pytest.approx(b["f1"]) and r["f2"] == pytest.approx(0.2 * b["f2"])
    assert "league_rating" in rated.columns and "league_rating" not in base.columns
    # Ratings unavailable -> no league is rated -> NO stake (Data Truth 2026-11-01: an unrated league
    # used to be sized as if rated 1, an invented rating).
    none = size_day(SEL, pol, 100.0, 100.0, league_ratings={})
    assert (none["requested_fraction"] == 0).all() and set(none["reason"]) == {"NO_LEAGUE_RATING"}
    half = size_day(SEL, pol, 100.0, 100.0, league_ratings={"Brazil Serie A": 10})
    h = half.set_index("fixture")
    assert h.loc["f1", "requested_fraction"] > 0 and h.loc["f2", "reason"] == "NO_LEAGUE_RATING"


def test_policy_without_min_scale_stakes_nothing():
    """min_scale comes from the policy file only; no typed 0.2 stands in for it."""
    pol = load_policy()
    pol = {**pol, "league_rating_sizing": {"enabled": True}}
    r = size_day(SEL, pol, 100.0, 100.0, league_ratings={"Brazil Serie A": 10, "League One": 10})
    assert (r["requested_fraction"] == 0).all() and set(r["reason"]) == {"NO_RATING_MIN_SCALE"}


def test_no_league_threshold_evidence_means_no_stake():
    """Owner 2026-11-01: a tip tiered on a GLOBAL bar is never staked. Column absent -> not applied."""
    pol = load_policy()
    sel = SEL.assign(league_threshold_evidence=[True, False])
    r = size_day(sel, pol, 100.0, 100.0).set_index("fixture")
    assert r.loc["f2", "reason"] == "NO_LEAGUE_THRESHOLD_EVIDENCE" and r.loc["f2", "requested_fraction"] == 0
    assert r.loc["f1", "reason"] == "OK" and r.loc["f1", "requested_fraction"] > 0
    nan = size_day(SEL.assign(league_threshold_evidence=[float("nan"), None]), pol, 100.0, 100.0)
    assert set(nan["reason"]) == {"NO_LEAGUE_THRESHOLD_EVIDENCE"}       # unknown is not evidence


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
