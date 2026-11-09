"""Per-cell staking (owner 2026-11-09 "Paper + PROMISING small"): src/bankroll_policy.stake_entry / size_day.

    python -m pytest tests/test_cell_staking.py -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

from src import bankroll_policy as BP

ARG = "Argentina Primera Division"
ROOT = Path(__file__).resolve().parents[1]
#: TEST ONLY ACTIVE_LOW cells (the generator's format), so the mechanics tests do not depend on which cells the
#: month's verdicts make PROMISING (they change every month by design).
_LOW = {"state": "ACTIVE_LOW", "base_fraction": 0.005, "max_odds": 2.5,
        "tier_multipliers": {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}}


def _with_cells(pol, keys):
    pol = copy.deepcopy(pol)
    for k in keys:
        pol["cell_staking"]["cells"][k] = copy.deepcopy(_LOW)
    return pol


def _sel(league, model_type="new_format", market="BTTS", side="YES", odds=2.0, tier="SNIPER", fx=None):
    return {"date": "2026-11-21", "fixture": fx or f"{league} H v A", "league": league, "home_team": "H",
            "away_team": "A", "model_type": model_type, "market": market, "side": side, "odds": odds,
            "signal_tier": tier, "edge": 0.05}


def _size(rows, policy):
    return BP.size_day(pd.DataFrame(rows), policy, bankroll=100.0, peak=100.0)


def test_committed_policy_stakes_exactly_the_owner_cells():
    """The committed cells are exactly what scripts/cell_staking_from_proposal.py builds from the latest proposal:
    the PROMISING cells at ACTIVE_LOW plus the PROTECTED Argentina BTTS entry, minus owner-held paper tracks."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import cell_staking_from_proposal as G
    pol = BP.load_policy()
    cs = pol["cell_staking"]
    assert cs["enabled"] is True
    latest = json.loads((ROOT / "output" / "threshold_proposals" / "latest.json").read_text(encoding="utf-8"))
    prop = json.loads((ROOT / latest["path"]).read_text(encoding="utf-8"))
    assert G.build_cells(prop, cs["cells"], pol) == cs["cells"]
    for k, v in cs["cells"].items():
        if k != f"new_format|BTTS|{ARG}":
            assert v["state"] == "ACTIVE_LOW" and v["base_fraction"] == 0.005
            assert not G.held_by_owner(pol, k.rsplit("|", 1)[0])          # an owner paper hold is never lifted
    a = cs["cells"][f"new_format|BTTS|{ARG}"]
    t = pol["tracks"]["new_format|BTTS"]
    assert (a["state"], a["base_fraction"], a["max_odds"]) == ("ACTIVE", 0.0125, 2.6) == (t["state"], t["base_fraction"], t["max_odds"])


def test_owner_paper_hold_is_never_lifted_by_a_promising_cell():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cell_staking_from_proposal as G
    pol = BP.load_policy()
    prop = {"proposal_id": "TP-TEST", "entries": [{"proposed": {"bars_opt": {"under": {
        "track": "new_format", "mside": "over25_under", "league": "Sweden Allsvenskan", "verdict": "PROMISING",
        "odds_max": 3.5}}}}]}
    held = []
    assert G.held_by_owner(pol, "new_format|OU25_UNDER")                  # owner 2026-11-02 NF Unders paper
    assert "new_format|OU25_UNDER|Sweden Allsvenskan" not in G.build_cells(prop, {}, pol, held)
    assert held == ["new_format|OU25_UNDER|Sweden Allsvenskan"]


def test_cell_without_entry_is_paper_never_inherited_from_its_track():
    pol = BP.load_policy()
    out = _size([_sel(ARG, fx="F1"), _sel("Japan J-League", fx="F2")], pol)
    arg, jpn = out.iloc[0], out.iloc[1]
    assert arg["reason"] == "OK" and arg["final_fraction"] > 0 and arg["deployment_state"] == "ACTIVE"
    assert jpn["reason"] == BP.NO_CELL_ENTRY_REASON and jpn["final_fraction"] == 0
    assert pol["tracks"]["new_format|BTTS"]["state"] == "ACTIVE"          # the track would have staked it


def test_promising_cell_stakes_at_active_low_and_caps_still_apply():
    pol = _with_cells(BP.load_policy(), ["new_format|BTTS|Test League", "new_format|OU25_OVER|Test League"])
    out = _size([_sel("Test League", fx="F1"), _sel("Test League", market="OU25", side="OVER", fx="F1")], pol)
    assert set(out["deployment_state"]) == {"ACTIVE_LOW"}
    assert (out["base_fraction"] == 0.005).all()
    assert out["final_fraction"].sum() <= pol["portfolio"]["fixture_cap"] + 1e-12


def test_promising_cell_respects_its_own_odds_cap():
    pol = _with_cells(BP.load_policy(), ["new_format|BTTS|Test League"])
    out = _size([_sel("Test League", odds=2.6)], pol)                     # the cell's own max 2.5
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
    pol = _with_cells(BP.load_policy(), ["new_format|BTTS|Test League"])
    out = _size([_sel(ARG, tier="VALUABLE", fx="F1"), _sel("Test League", tier="VALUABLE", fx="F2")], pol)
    assert (out["reason"] == "OK").all()
    assert out.iloc[0]["requested_fraction"] == 0.0125 and out.iloc[1]["requested_fraction"] == 0.005
