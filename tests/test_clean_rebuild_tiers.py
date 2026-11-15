"""Owner 2026-11-11 ("Clean rebuild"): no SNIPER / MARKSMAN / VALUABLE on a future tip - team, props, stakes."""
from __future__ import annotations

import pandas as pd

from player_model import predict as PP
from src.bankroll_policy import load_policy, size_day
from src.tier_labels import LEGACY_TIERS, QUALIFIED


def test_props_tip_qualifies_at_the_lowest_bar_without_a_tier():
    # clears the former VALUABLE gates (ev >= 0.15, odds < 4 -> relative edge >= 0.12, GES >= 0.35)
    assert PP._classify_tier(0.20, 2.5, 0.15, 1.0, 0, 0.5, "assists", "M") == QUALIFIED
    # a former SNIPER-strength signal is the same QUALIFIED tip: no tier above the bar
    assert PP._classify_tier(0.60, 2.5, 0.40, 1.0, 3, 0.9, "goals", "F") == QUALIFIED
    assert PP._classify_tier(0.10, 2.5, 0.15, 1.0, 0, 0.5, "assists", "M") == "AVOID"     # below the bar
    # capped data-only combinations stay PAPER rows, never a tip
    assert PP._classify_tier(0.60, 2.5, 0.40, 1.0, 3, 0.9, "goals", "D") == "PAPER"
    assert PP._classify_tier(0.60, 2.5, 0.40, 1.0, 3, 0.9, "sot2", "F") == "PAPER"
    for args in ((0.2, 2.5, 0.15, 1.0, 0, 0.5, "assists", "M"), (0.6, 7.0, 0.5, 1.0, 3, 0.9, "sot", "F")):
        assert PP._classify_tier(*args) not in LEGACY_TIERS


def test_stakes_are_flat_per_cell_no_tier_multiplier():
    sel = pd.DataFrame([{"fixture": f"f{i}", "league": "Argentina Primera Division", "model_type": "new_format",
                         "market": "BTTS", "side": "YES", "signal_tier": t, "odds": 2.0}
                        for i, t in enumerate((QUALIFIED, "SNIPER", "MARKSMAN", "VALUABLE"))])
    out = size_day(sel, load_policy(), bankroll=100.0, peak=100.0)
    assert set(out["tier_multiplier"]) == {1.0}
    assert set(out["requested_fraction"]) == {0.0125}          # Argentina NF BTTS 1.25% flat (owner 2026-11-11)
    cells = load_policy()["cell_staking"]["cells"]
    assert not any("tier_multipliers" in v for v in cells.values())
