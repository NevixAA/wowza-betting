"""§21.1-3 — the resolver must agree with production, not with config.py.

The study published on 2026-10-05 reported Championship's SNIPER gate as 0.15 from config while
production runs 0.07 from an APPROVED optimiser value. These tests exist so that cannot recur.
"""
from __future__ import annotations

import json

import pytest

import config
from src.betting import _base_tier
from src.gate_resolver import (SIDE_VALUABLE_HARDCODED, Gates, resolve_effective_tier_gates,
                               regime_id)


def _std():
    f = config.MODELS_DIR / "best_params_standard.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


# ── 1. the resolver agrees with the real production function ──────────────────────────────────
@pytest.mark.parametrize("league", sorted(_std()))
def test_resolver_matches_base_tier_at_its_own_boundary(league):
    """The strongest available check: feed production's own _base_tier an edge just above and
    just below the resolved SNIPER gate and confirm it flips there."""
    g = resolve_effective_tier_gates(league, "ou25")
    assert _base_tier(g.sniper + 1e-6, league=league) == "SNIPER"
    below = _base_tier(g.sniper - 1e-4, league=league)
    assert below != "SNIPER", f"{league}: _base_tier still SNIPER below the resolved gate"


@pytest.mark.parametrize("league", sorted(_std()))
def test_resolver_matches_base_tier_at_the_marksman_boundary(league):
    g = resolve_effective_tier_gates(league, "ou25")
    if g.marksman >= g.sniper:
        pytest.skip("MARKSMAN disabled for this league (floor at or above SNIPER)")
    assert _base_tier(g.marksman + 1e-6, league=league) in ("MARKSMAN", "SNIPER")
    assert _base_tier(g.marksman - 1e-4, league=league) in ("VALUABLE", "AVOID")


# ── 2. an APPROVED optimiser value beats the hand-set table ───────────────────────────────────
def test_an_approved_optimizer_value_wins_over_the_config_table():
    approved = [l for l, v in _std().items()
                if v.get("approved") and v.get("sniper_th") is not None
                and l in config.LEAGUE_SNIPER_THRESHOLDS]
    if not approved:
        pytest.skip("no approved league also present in the config table")
    for l in approved:
        g = resolve_effective_tier_gates(l, "ou25")
        assert g.approved_optimizer_applied
        assert g.sniper == pytest.approx(float(_std()[l]["sniper_th"]))
        assert "best_params_standard" in g.sniper_source


def test_an_UNapproved_optimizer_value_is_never_deployed():
    """A not-approved threshold is merely least-bad and is OOS-negative. Production ignores it;
    so must any study."""
    unapproved = [l for l, v in _std().items() if v.get("approved") is False]
    if not unapproved:
        pytest.skip("no unapproved leagues in the artifact")
    for l in unapproved:
        g = resolve_effective_tier_gates(l, "ou25")
        assert not g.approved_optimizer_applied
        assert g.sniper != pytest.approx(float(_std()[l]["sniper_th"])) or \
            "best_params_standard" not in g.sniper_source


# ── 3. the league cap ─────────────────────────────────────────────────────────────────────────
def test_the_league_cap_is_reported_when_it_bites(monkeypatch):
    """LEAGUE_SNIPER_CAP does min(threshold, cap) over every calibrated league. A study that
    reads the calibrated table without the cap overstates every one of them."""
    import importlib
    monkeypatch.setenv("LEAGUE_SNIPER_CAP", "0.12")
    importlib.reload(config)
    import src.gate_resolver as gr
    importlib.reload(gr)
    capped = [l for l, v in config._LEAGUE_SNIPER_CALIBRATED.items()
              if config.LEAGUE_SNIPER_THRESHOLDS[l] < v
              and not (_std().get(l) or {}).get("approved")]
    for l in capped:
        g = gr.resolve_effective_tier_gates(l, "ou25")
        assert g.league_cap_applied and "CAPPED" in g.sniper_source
    monkeypatch.delenv("LEAGUE_SNIPER_CAP", raising=False)
    importlib.reload(config); importlib.reload(gr)


# ── 4. markets do NOT share a gate ────────────────────────────────────────────────────────────
def test_side_markets_use_a_different_valuable_bar_than_main_ou():
    """Hard-coded 0.04 in _generate_side_bets; main O/U follows config.VALUABLE_THRESHOLD.
    On a production run at 0.03 these genuinely differ, and a study that pools them is wrong."""
    main = resolve_effective_tier_gates("Championship", "ou25")
    btts = resolve_effective_tier_gates("Championship", "btts")
    assert btts.valuable == SIDE_VALUABLE_HARDCODED
    assert main.valuable == float(config.VALUABLE_THRESHOLD)
    assert "hard-coded" in btts.valuable_source


def test_one_league_can_carry_three_different_gates_across_markets():
    gs = {m: resolve_effective_tier_gates("La Liga 2", m) for m in ("ou25", "btts", "over15")}
    assert len({g.sniper for g in gs.values()}) > 1, \
        "markets resolved to identical gates — the market argument is being ignored"


# ── 5. provenance and regime ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("market", ["ou25", "btts", "over15", "over35"])
def test_every_threshold_names_its_source(market):
    g = resolve_effective_tier_gates("Championship", market)
    for s in (g.valuable_source, g.marksman_source, g.sniper_source):
        assert s and len(s) > 8


def test_regime_id_changes_when_the_environment_changes(monkeypatch):
    import importlib
    a = regime_id()
    monkeypatch.setenv("MARKSMAN_THRESHOLD", "0.99")
    importlib.reload(config)
    import src.gate_resolver as gr
    importlib.reload(gr)
    assert gr.regime_id() != a, "regime id ignored an env change that moves a real gate"
    monkeypatch.delenv("MARKSMAN_THRESHOLD", raising=False)
    importlib.reload(config); importlib.reload(gr)


def test_the_resolver_never_writes_anything():
    """Read-only by construction — importing or calling it must not touch an artifact."""
    import src.gate_resolver as gr
    src = open(gr.__file__, encoding="utf-8").read()
    for bad in ("write_text(", "to_csv(", "json.dump(", "open(", ".save("):
        assert bad not in src, f"gate_resolver performs a write: {bad}"
