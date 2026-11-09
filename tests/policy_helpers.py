"""Test helper: the bankroll policy with PER-TRACK staking (cell_staking disabled).

Since 2026-11-09 (owner "Paper + PROMISING small") the committed policy stakes per CELL
(config/bankroll_policy.json cell_staking; tests/test_cell_staking.py pins that live behaviour). The tests
that pin the TRACK mechanics underneath it - O/U side split, tier multipliers, rating sizing, contract guards,
the stake-plan write path - run on the same file with cell_staking.enabled = false, the switch that restores
per-track staking. It patches only load_policy; size_day and every cap are the production code.
"""
from __future__ import annotations

def per_track_policy() -> dict:
    import json
    from src import bankroll_policy as BP
    with open(BP.POLICY_PATH, encoding="utf-8") as fh:
        p = json.load(fh)
    p.setdefault("cell_staking", {})["enabled"] = False
    return p


def install_per_track(monkeypatch, *modules) -> None:
    """Every load_policy the code (and the given test modules) can reach returns the per-track policy."""
    from src import bankroll_policy as BP
    monkeypatch.setattr(BP, "load_policy", lambda path=None: per_track_policy())
    for m in modules:
        if hasattr(m, "load_policy"):
            monkeypatch.setattr(m, "load_policy", lambda path=None: per_track_policy())
