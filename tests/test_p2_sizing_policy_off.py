"""Phase 2 / Part 9 (2026-11-03): pooled league-rating stake sizing stays OFF (owner 2026-11-02).

Every test here has a failure condition that a real regression would trigger:
  * the committed policy flips league_rating_sizing.enabled back on,
  * src/stake_plan.write loads Pro's pooled ratings or scales a stake by them while the flag is off,
  * size_day scales without both the flag AND an explicit ratings dict (Pro's simulator passes none),
  * a dashboard / Telegram caption tells the reader the pooled rating sizes stakes while it does not.
A power check (test_power_*) proves the stake_plan assertion DOES catch scaling when it happens.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_p2_sizing_policy_off.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import stake_plan as sp
from src.bankroll_policy import load_policy, size_day

ROOT = Path(__file__).resolve().parents[1]


def _bets():
    # O/U 2.5 Over SNIPER tips in distinct leagues/fixtures, new-format track (a REAL track today).
    return pd.DataFrame([{"date": "2026-11-03", "league": lg, "home_team": f"H{i}", "away_team": f"A{i}",
                          "best_side": "OVER", "odds_over25": 2.0, "odds_under25": 1.85,
                          "signal_tier": "SNIPER", "model_type": "new_format"}
                         for i, lg in enumerate(("Championship", "League One", "Brazil Serie A"))])


_LEAGUES = ("Championship", "League One", "Brazil Serie A")


def _staked(pol: dict) -> dict:
    """TEST ONLY: give the three test leagues a staked new_format O/U 2.5 Over cell, so the ratings mechanics are
    exercised whatever the month's PROMISING cells are (per-cell staking, owner 2026-11-09)."""
    import copy
    pol = copy.deepcopy(pol)
    for lg in _LEAGUES:
        pol["cell_staking"]["cells"][f"new_format|OU25_OVER|{lg}"] = {
            "state": "ACTIVE_LOW", "base_fraction": 0.005, "max_odds": 2.5,
            "tier_multipliers": {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}}
    return pol


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import src.betting as betting
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    # raising=False: origin/main predates league_threshold_evidence, and this file is also run there
    monkeypatch.setattr(betting, "league_threshold_evidence", lambda league, market="ou25": True, raising=False)
    import src.bankroll_policy as BPm
    real = BPm.load_policy
    monkeypatch.setattr(BPm, "load_policy", lambda *a, **k: _staked(real(*a, **k)))
    return monkeypatch


def _unscaled(plan: pd.DataFrame) -> pd.Series:
    ok = plan[plan["reason"] == "OK"]
    return (ok["requested_fraction"] - ok["base_fraction"] * ok["tier_multiplier"]
            * ok["league_multiplier"]).abs() < 1e-12


def test_committed_policy_keeps_pooled_rating_sizing_off():
    lrs = load_policy()["league_rating_sizing"]
    assert lrs["enabled"] is False                                   # owner 2026-11-02
    assert "out-of-sample" in lrs["note"]                            # the reason stays beside the flag


def test_stake_plan_never_reads_pooled_ratings_while_off(isolated):
    calls = []
    isolated.setattr(sp, "load_league_ratings", lambda *a, **k: calls.append(1) or {lg: 1 for lg in
                     ("Championship", "League One", "Brazil Serie A")})   # every league rated 1 = 20%
    plan = sp.write(_bets(), None)
    assert calls == []                                               # flag off -> ratings never loaded
    assert (plan["reason"] == "OK").sum() >= 1                       # something was actually sized
    assert _unscaled(plan).all()                                     # requested = base x tier x league
    assert "rating_scale" not in plan.columns and "league_rating" not in plan.columns


def test_power_stake_plan_check_catches_scaling_when_flag_is_on(isolated):
    """The same assertion fails when pooled sizing is (wrongly) on: proof the check above has teeth."""
    pol = _staked(load_policy())
    on = {**pol, "league_rating_sizing": {**pol["league_rating_sizing"], "enabled": True}}
    isolated.setattr("src.bankroll_policy.load_policy", lambda *a, **k: on)
    isolated.setattr(sp, "load_league_ratings", lambda *a, **k: {lg: 1 for lg in
                     ("Championship", "League One", "Brazil Serie A")})
    plan = sp.write(_bets(), None)
    assert (plan["reason"] == "OK").sum() >= 1
    assert not _unscaled(plan).all()                                  # rating 1 -> 0.2 x the stake


def test_size_day_needs_flag_and_explicit_ratings():
    """Pro's simulator (wowzaV9-Pro src/bankroll/simulate.py day_plans) calls size_day WITHOUT
    league_ratings, so even a policy with the flag on is unscaled there; the live stake plan only
    passes ratings while the flag is on. Both gates must hold."""
    pol = _staked(load_policy())
    sel = sp.selections(_bets(), None).assign(league_threshold_evidence=True)
    on = {**pol, "league_rating_sizing": {**pol["league_rating_sizing"], "enabled": True}}
    ratings = {lg: 1 for lg in sel["league"]}
    base = size_day(sel, pol, 100.0, 100.0)["requested_fraction"].tolist()
    assert size_day(sel, on, 100.0, 100.0)["requested_fraction"].tolist() == base         # no ratings
    assert size_day(sel, pol, 100.0, 100.0, league_ratings=ratings)["requested_fraction"].tolist() == base
    assert sum(base) > 0


# Presentation: the pooled rating may be SHOWN, but no caption may say it sizes stakes while it does not.
CLAIMS = ("the number the stake plan sizes by", "the stake plan's sizing", "the same source the stake plan sizes by",
          "the stake plan's source")
SURFACES = ["pages/1_\U0001f4ca_Dashboard.py", "pages/17_\U0001f3df️_Leagues.py",
            "telegram_bot/tips_table.py", "dashboard_data/leagues.py"]


@pytest.mark.parametrize("rel", SURFACES)
def test_no_surface_claims_the_rating_sizes_stakes_while_off(rel):
    assert load_policy()["league_rating_sizing"]["enabled"] is False
    text = " ".join((ROOT / rel).read_text(encoding="utf-8").split())   # collapse wrapped strings
    text = text.replace('" "', "")
    hits = [c for c in CLAIMS if c in text]
    assert not hits, f"{rel} still says the pooled rating sizes stakes: {hits}"


def test_telegram_footer_says_rating_is_display_only():
    text = (ROOT / "telegram_bot" / "tips_table.py").read_text(encoding="utf-8")
    assert "not used to size stakes" in text
