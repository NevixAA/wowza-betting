"""Per-cell staking (owner 2026-11-09 "Paper + PROMISING small"): src/bankroll_policy.stake_entry / size_day.

    python -m pytest tests/test_cell_staking.py -q
"""
from __future__ import annotations

import copy

import pandas as pd

from src import bankroll_policy as BP

ARG = "Argentina Primera Division"
PROMISING = {"standard|OU25_OVER|Serie B", "standard|OU25_UNDER|Scottish League Two", "standard|BTTS|La Liga 2",
             "standard|BTTS|Serie B", "new_format|OU25_OVER|Brazil Serie A", "new_format|OU25_OVER|Sweden Allsvenskan",
             "new_format|BTTS|Brazil Serie A", "new_format|BTTS|Ireland Premier Division",
             "new_format|BTTS|Norway Eliteserien"}


def _sel(league, model_type="new_format", market="BTTS", side="YES", odds=2.0, tier="SNIPER", fx=None):
    return {"date": "2026-11-21", "fixture": fx or f"{league} H v A", "league": league, "home_team": "H",
            "away_team": "A", "model_type": model_type, "market": market, "side": side, "odds": odds,
            "signal_tier": tier, "edge": 0.05}


def _size(rows, policy):
    return BP.size_day(pd.DataFrame(rows), policy, bankroll=100.0, peak=100.0)


def test_committed_policy_stakes_exactly_the_owner_cells():
    pol = BP.load_policy()
    cs = pol["cell_staking"]
    assert cs["enabled"] is True
    cells = cs["cells"]
    assert set(cells) == PROMISING | {f"new_format|BTTS|{ARG}"}
    for k in PROMISING:
        assert cells[k]["state"] == "ACTIVE_LOW" and cells[k]["base_fraction"] == 0.005
    a = cells[f"new_format|BTTS|{ARG}"]
    t = pol["tracks"]["new_format|BTTS"]
    assert (a["state"], a["base_fraction"], a["max_odds"]) == ("ACTIVE", 0.0125, 2.6) == (t["state"], t["base_fraction"], t["max_odds"])


def test_cell_without_entry_is_paper_never_inherited_from_its_track():
    pol = BP.load_policy()
    out = _size([_sel(ARG, fx="F1"), _sel("Japan J-League", fx="F2")], pol)
    arg, jpn = out.iloc[0], out.iloc[1]
    assert arg["reason"] == "OK" and arg["final_fraction"] > 0 and arg["deployment_state"] == "ACTIVE"
    assert jpn["reason"] == BP.NO_CELL_ENTRY_REASON and jpn["final_fraction"] == 0
    assert pol["tracks"]["new_format|BTTS"]["state"] == "ACTIVE"          # the track would have staked it


def test_promising_cell_stakes_at_active_low_and_caps_still_apply():
    pol = BP.load_policy()
    out = _size([_sel("Brazil Serie A", fx="F1"), _sel("Brazil Serie A", market="OU25", side="OVER", fx="F1")], pol)
    assert set(out["deployment_state"]) == {"ACTIVE_LOW"}
    assert (out["base_fraction"] == 0.005).all()
    assert out["final_fraction"].sum() <= pol["portfolio"]["fixture_cap"] + 1e-12


def test_promising_cell_respects_its_own_odds_cap():
    pol = BP.load_policy()
    out = _size([_sel("Norway Eliteserien", odds=2.6)], pol)              # the cell's own max 2.5
    assert out.iloc[0]["reason"] == "ODDS_ABOVE_TRACK_CAP" and out.iloc[0]["final_fraction"] == 0


def test_paper_league_and_drawdown_rules_still_apply():
    pol = copy.deepcopy(BP.load_policy())
    pol["paper_leagues"] = list(pol.get("paper_leagues", [])) + [ARG]
    assert _size([_sel(ARG)], pol).iloc[0]["reason"] == "PAPER_LEAGUE"
    pol = BP.load_policy()
    full = _size([_sel(ARG)], pol).iloc[0]["final_fraction"]
    dd = BP.size_day(pd.DataFrame([_sel(ARG)]), pol, bankroll=90.0, peak=100.0).iloc[0]["final_fraction"]
    assert 0 < dd < full                                                  # the drawdown throttle scales it


def test_switch_off_restores_per_track_staking():
    pol = copy.deepcopy(BP.load_policy())
    pol["cell_staking"]["enabled"] = False
    out = _size([_sel("Japan J-League")], pol)
    assert out.iloc[0]["reason"] == "OK" and out.iloc[0]["deployment_state"] == "ACTIVE"


def test_cell_tier_multipliers_keep_the_protected_league_wide_track_staked():
    pol = BP.load_policy()
    out = _size([_sel(ARG, tier="VALUABLE", fx="F1"), _sel("Norway Eliteserien", tier="VALUABLE", fx="F2")], pol)
    assert (out["reason"] == "OK").all()
    assert out.iloc[0]["requested_fraction"] == 0.0125 and out.iloc[1]["requested_fraction"] == 0.005
