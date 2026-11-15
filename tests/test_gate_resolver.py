"""§21.1-3 + owner 2026-11-09 — the resolver must agree with production, and production has NO global bar.

The study published on 2026-10-05 reported Championship's SNIPER gate as 0.15 from config while
production ran 0.07 from an approved optimiser value. These tests exist so that cannot recur. Since
2026-11-09 (owner: "100% real data, no fakes, no bugs, no defaults, no global vars") every league x
market tiers ONLY on its own owner-applied sniper / marksman / valuable bars; without them it has no
gate at all (NO_LEAGUE_BAR), never a global stand-in.
"""
from __future__ import annotations

import json

import pytest

import config
import src.betting as B
import src.gate_resolver as G
import src.side_thresholds as ST
from src.betting import _base_tier
from src.gate_resolver import Gates, resolve_effective_tier_gates, regime_id

OWN = {"sniper_th": 0.12, "marksman_th": 0.09, "valuable_th": 0.05, "approved": False, "status": "UNPROVEN"}


@pytest.fixture
def bars_env(tmp_path, monkeypatch):
    """A models dir + applied ledger the test controls. Returns write(std, side, applied_ids)."""
    models = tmp_path / "models"
    models.mkdir()
    ledger = tmp_path / "threshold_applied.json"
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(ST, "APPLIED_LEDGER", ledger)

    def write(std=None, side=None, applied=()):
        (models / "best_params_standard.json").write_text(json.dumps(std or {}), encoding="utf-8")
        (models / "best_params_side_markets.json").write_text(json.dumps(side or {}), encoding="utf-8")
        ledger.write_text(json.dumps({"history": [{"entries": list(applied)}]}), encoding="utf-8")
        ST._APPLIED_CACHE.clear()
        B._STANDARD_TH_CACHE = None
    yield write
    ST._APPLIED_CACHE.clear()
    B._STANDARD_TH_CACHE = None


# ── 1. the resolver agrees with the real production function ──────────────────────────────────
def test_resolver_matches_base_tier_at_its_own_boundaries(bars_env):
    bars_env(std={"Serie B": OWN}, applied=["ou25|Serie B"])
    g = resolve_effective_tier_gates("Serie B", "ou25")
    assert g.own_bars and (g.sniper, g.marksman, g.valuable) == (0.12, 0.09, 0.05)
    assert _base_tier(g.sniper + 1e-6, league="Serie B") == "SNIPER"
    assert _base_tier(g.sniper - 1e-4, league="Serie B") == "MARKSMAN"
    assert _base_tier(g.marksman - 1e-4, league="Serie B") == "VALUABLE"
    assert _base_tier(g.valuable - 1e-4, league="Serie B") == "AVOID"


# ── 2. no own bars = no gate, never a global one ──────────────────────────────────────────────
@pytest.mark.parametrize("market", ["ou25", "btts", "over15", "over35"])
def test_a_league_without_own_bars_has_no_gate(bars_env, market):
    bars_env()
    g = resolve_effective_tier_gates("Serie B", market)
    assert not g.own_bars and (g.sniper, g.marksman, g.valuable) == (None, None, None)
    assert ST.NO_LEAGUE_BAR in g.sniper_source
    assert _base_tier(0.99, "OVER", "Serie B") == "AVOID"


def test_unapplied_bars_are_not_served_even_when_complete(bars_env):
    """A complete entry that src/threshold_apply did not write (the owner did not approve it) is no gate."""
    bars_env(std={"Serie B": OWN}, applied=[])
    g = resolve_effective_tier_gates("Serie B", "ou25")
    assert not g.own_bars and "not applied" in g.sniper_source


def test_pre_repair_params_without_valuable_are_not_served(bars_env):
    """The pre-2026-11-09 optimiser entries carry no valuable_th: no per-league VALUABLE = no own bars."""
    old = {"sniper_th": 0.14, "marksman_th": 0.12, "approved": True}
    bars_env(std={"Championship": old}, applied=["ou25|Championship"])
    g = resolve_effective_tier_gates("Championship", "ou25")
    assert not g.own_bars and "valuable_th" in g.sniper_source


def test_unordered_or_dropped_bars_are_not_served(bars_env):
    bars_env(std={"A": {**OWN, "valuable_th": 0.2}, "B": {**OWN, "drop": True}},
             applied=["ou25|A", "ou25|B"])
    assert not resolve_effective_tier_gates("A", "ou25").own_bars
    assert not resolve_effective_tier_gates("B", "ou25").own_bars


def test_unproven_owner_applied_bars_are_served(bars_env):
    """approved=False means not statistically PROVEN; the owner still applied them - they are the
    league's own real bars (owner 2026-11-08: every league has its own bars)."""
    bars_env(side={"btts": {"Serie B": OWN}}, applied=["btts|Serie B"])
    g = resolve_effective_tier_gates("Serie B", "btts")
    assert g.own_bars and g.valuable == 0.05 and "owner-applied" in g.valuable_source


# ── 3. VALUABLE is per league, per market ─────────────────────────────────────────────────────
def test_valuable_is_per_league_and_per_market(bars_env):
    bars_env(std={"Serie B": OWN},
             side={"btts": {"Serie B": {**OWN, "valuable_th": 0.03, "marksman_th": 0.06, "sniper_th": 0.08}}},
             applied=["ou25|Serie B", "btts|Serie B"])
    main = resolve_effective_tier_gates("Serie B", "ou25")
    btts = resolve_effective_tier_gates("Serie B", "btts")
    assert (main.valuable, btts.valuable) == (0.05, 0.03)


# ── 4. provenance and regime ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("market", ["ou25", "btts", "over15", "over35"])
def test_every_gate_names_its_source(bars_env, market):
    bars_env()
    g = resolve_effective_tier_gates("Championship", market)
    for s in (g.valuable_source, g.marksman_source, g.sniper_source):
        assert s and len(s) > 8


def test_regime_id_changes_when_an_entry_is_applied(bars_env):
    bars_env(std={"Serie B": OWN}, applied=[])
    a = regime_id()
    bars_env(std={"Serie B": OWN}, applied=["ou25|Serie B"])
    assert regime_id() != a


def test_the_resolver_never_writes_anything():
    """Read-only by construction — importing or calling it must not touch an artifact."""
    src = open(G.__file__, encoding="utf-8").read()
    for bad in ("write_text(", "to_csv(", "json.dump(", "open(", ".save("):
        assert bad not in src, f"gate_resolver performs a write: {bad}"


def test_gates_schema_keeps_the_old_fields():
    g = Gates("L", "ou25", "standard", None, None, None, "x" * 9, "x" * 9, "x" * 9)
    d = g.as_dict()
    for k in ("approved_optimizer_applied", "league_cap_applied", "edge_ceiling", "edge_ceiling_active", "own_bars"):
        assert k in d
