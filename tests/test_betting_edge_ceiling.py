"""Estate scan 2026-11-02 finding (e) + owner 2026-11-09 ("no global vars").

The documented SNIPER edge ceiling (config.EDGE_CEILING / LEAGUE_EDGE_CEILING) only ever applied to a
fixture tiered on the GLOBAL bar (no per-league threshold). Since 2026-11-09 serving has no global bar:
a league tiers on its own owner-applied bars or not at all (NO_LEAGUE_BAR). So the ceiling can no longer
apply anywhere, and an edge above the old ceiling in a league with its own bars is tiered by those bars.
"""
from __future__ import annotations

import config
from src import betting as B
from tests.own_bars import install


def test_no_global_tier_exists_for_a_ceiling_to_demote(monkeypatch):
    install(monkeypatch, [], markets=("ou25",))
    assert B._base_tier(config.EDGE_CEILING + 0.05, "OVER", "Any League") == "AVOID"


def test_own_bars_tier_an_edge_above_the_old_ceiling(monkeypatch):
    install(monkeypatch, ["L"], markets=("ou25",), bars=(0.12, 0.08, 0.04))
    monkeypatch.setattr(config, "SNIPER_EDGE_CEILING_ENFORCED", True, raising=False)
    assert B._base_tier(config.EDGE_CEILING + 0.05, "OVER", "L") == "SNIPER"


def test_the_ceiling_switch_is_gone_from_serving():
    assert not hasattr(B, "sniper_edge_ceiling_enforced")
