"""Estate scan 2026-11-02 finding (e): the documented SNIPER edge ceiling never fired, because the
SNIPER return came first and every ceiling is above the global SNIPER bar. The ceiling now applies
before SNIPER — only when the owner switches it on (SNIPER_EDGE_CEILING_ENFORCED); off, tiers are
byte-for-byte what they were. No threshold or ceiling value is touched."""
from __future__ import annotations

import pytest

import config
from src import betting as B


@pytest.fixture
def global_bar(monkeypatch):
    """A league tiered on the GLOBAL bar (no per-league threshold, no approved optimiser value)."""
    monkeypatch.setattr(B, "_STANDARD_TH_CACHE", {})
    return ""


def test_ceiling_downgrades_over_ceiling_edge_when_enforced(monkeypatch, global_bar):
    monkeypatch.setattr(config, "SNIPER_EDGE_CEILING_ENFORCED", True, raising=False)
    edge = config.EDGE_CEILING + 0.05
    assert edge >= config.SNIPER_THRESHOLD_OVER
    assert B._base_tier(edge, "OVER", global_bar) == "MARKSMAN"
    assert B._base_tier(edge, "UNDER", global_bar) == "MARKSMAN"
    # below the ceiling, above the SNIPER bar: still SNIPER
    assert B._base_tier(config.SNIPER_THRESHOLD_OVER + 0.001, "OVER", global_bar) == "SNIPER"


def test_ceiling_off_by_default_keeps_current_tiers(monkeypatch, global_bar):
    monkeypatch.setattr(config, "SNIPER_EDGE_CEILING_ENFORCED", None, raising=False)
    monkeypatch.delenv("WOWZA_SNIPER_EDGE_CEILING", raising=False)
    assert not B.sniper_edge_ceiling_enforced()
    assert B._base_tier(config.EDGE_CEILING + 0.05, "OVER", global_bar) == "SNIPER"


def test_ceiling_never_applies_to_per_league_thresholds(monkeypatch):
    monkeypatch.setattr(config, "SNIPER_EDGE_CEILING_ENFORCED", True, raising=False)
    monkeypatch.setattr(B, "_STANDARD_TH_CACHE", {})
    lg = next(iter(config.LEAGUE_SNIPER_THRESHOLDS))
    th = config.LEAGUE_SNIPER_THRESHOLDS[lg]
    assert B._base_tier(max(th, config.EDGE_CEILING) + 0.05, "OVER", lg) == "SNIPER"
