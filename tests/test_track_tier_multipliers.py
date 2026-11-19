"""Per-track tier multipliers (owner 2026-10-22): OU25 tiers staked equally, other tracks keep the ladder."""
import pandas as pd
import pytest

from src.bankroll_policy import load_policy, size_day


def _sel(market, tiers, side="OVER"):
    # one selection per fixture so the one-position-per-fixture rule never interferes
    return pd.DataFrame([{"fixture": f"f{i}", "league": "Argentina Primera Division", "model_type": "new_format",
                          "market": market, "side": side, "signal_tier": t, "odds": 2.0}
                         for i, t in enumerate(tiers)])


def test_ou25_tiers_are_staked_equally_and_valuable_is_staked():
    pol = load_policy()
    for side in ("OVER", "UNDER"):                      # per-side tracks since 2026-10-26, same settings
        out = size_day(_sel("OU25", ["SNIPER", "MARKSMAN", "VALUABLE"], side), pol, bankroll=100.0, peak=100.0)
        assert set(out["track"]) == {f"new_format|OU25_{side}"}
        mult = dict(zip(out["signal_tier"], out["tier_multiplier"]))
        assert mult == {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}
        req = dict(zip(out["signal_tier"], out["requested_fraction"]))
        assert req["VALUABLE"] == req["SNIPER"] == req["MARKSMAN"]
        if pol["tracks"][f"new_format|OU25_{side}"]["state"] == "PAPER":
            # owner 2026-11-02: new-format Unders are paper-only until the clean retrain - never staked
            assert (out["final_fraction"] == 0).all() and set(out["reason"]) == {"NO_REAL_STAKE_PAPER"}
        else:
            assert req["VALUABLE"] > 0


def test_no_tier_ladder_anywhere_owner_2026_11_11():
    """Owner 2026-11-11 ("Clean rebuild"): tier multipliers are retired from staking - every tip, whatever
    (historical) code it carries, stakes flat; the policy's ladders are never read."""
    pol = load_policy()
    out = size_day(_sel("BTTS", ["QUALIFIED", "SNIPER", "MARKSMAN", "VALUABLE"], side="YES"), pol,
                   bankroll=100.0, peak=100.0)
    assert set(out["tier_multiplier"]) == {1.0}
    assert "TIER_NOT_STAKED" not in set(out["reason"])
    pol2 = {**pol, "track_tier_multipliers": {}, "tier_multipliers": {"SNIPER": 9.0, "VALUABLE": 0.0}}
    out2 = size_day(_sel("OU25", ["SNIPER", "VALUABLE"]), pol2, bankroll=100.0, peak=100.0)
    assert set(out2["tier_multiplier"]) == {1.0}


@pytest.fixture(autouse=True)
def _per_track_policy(monkeypatch):
    """These tests pin the TRACK mechanics: per-track staking (tests/policy_helpers). The live per-cell
    staking (owner 2026-11-09) is pinned in tests/test_cell_staking.py."""
    import sys
    from tests.policy_helpers import install_per_track
    install_per_track(monkeypatch, sys.modules[__name__])
