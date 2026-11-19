"""Test helper: the bankroll policy with PER-TRACK staking (cell_staking disabled).

Since 2026-11-09 (owner "Paper + PROMISING small") the committed policy stakes per CELL
(config/bankroll_policy.json cell_staking; tests/test_cell_staking.py pins that live behaviour). The tests
that pin the TRACK mechanics underneath it - O/U side split, tier multipliers, rating sizing, contract guards,
the stake-plan write path - run on the same file with cell_staking.enabled = false, the switch that restores
per-track staking. It patches only load_policy; size_day and every cap are the production code.

Owner 2026-11-17 (final, "Yes, adopt"): every track and cell of the committed policy is PAPER and the only real
stake is the manual TOP-3 flat stake (top3_league.stake; tests/test_top3_flat_stake.py pins that). The mechanics
need a staked track, so the policy here is the one before that decision: each entry's `before_2026_11_17`
values are restored and top3_league dropped.
"""
from __future__ import annotations

def per_track_policy() -> dict:
    import json
    from src import bankroll_policy as BP
    with open(BP.POLICY_PATH, encoding="utf-8") as fh:
        p = json.load(fh)
    p.pop("top3_league", None)
    for t in list(p.get("tracks", {}).values()) + list((p.get("cell_staking") or {}).get("cells", {}).values()):
        t.update(t.pop("before_2026_11_17", {}))
        t.pop("staking_basis_2026_11_17", None)
    p.setdefault("cell_staking", {})["enabled"] = False
    return p


def install_per_track(monkeypatch, *modules) -> None:
    """Every load_policy the code (and the given test modules) can reach returns the per-track policy."""
    from src import bankroll_policy as BP
    monkeypatch.setattr(BP, "load_policy", lambda path=None: per_track_policy())
    for m in modules:
        if hasattr(m, "load_policy"):
            monkeypatch.setattr(m, "load_policy", lambda path=None: per_track_policy())
