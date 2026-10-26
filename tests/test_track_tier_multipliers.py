"""Per-track tier multipliers (owner 2026-10-22): OU25 tiers staked equally, other tracks keep the ladder."""
import pandas as pd

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
        assert req["VALUABLE"] > 0 and req["VALUABLE"] == req["SNIPER"] == req["MARKSMAN"]


def test_btts_keeps_the_global_ladder():
    pol = load_policy()
    out = size_day(_sel("BTTS", ["SNIPER", "MARKSMAN", "VALUABLE"]), pol, bankroll=100.0, peak=100.0)
    mult = dict(zip(out["signal_tier"], out["tier_multiplier"]))
    assert mult["SNIPER"] == pol["tier_multipliers"]["SNIPER"]
    assert mult["MARKSMAN"] == pol["tier_multipliers"]["MARKSMAN"]
    assert mult["VALUABLE"] == 0.0 and out.loc[out.signal_tier == "VALUABLE", "reason"].iloc[0] == "TIER_NOT_STAKED"


def test_no_override_means_global_ladder():
    pol = load_policy()
    pol = {**pol, "track_tier_multipliers": {}}
    out = size_day(_sel("OU25", ["SNIPER", "VALUABLE"]), pol, bankroll=100.0, peak=100.0)
    assert dict(zip(out["signal_tier"], out["tier_multiplier"]))["VALUABLE"] == 0.0
