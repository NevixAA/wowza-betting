"""O/U 2.5 bankroll-policy tracks split by SIDE (owner decision 2026-10-26).

new_format|OU25 and standard|OU25 were replaced by new_format|OU25_OVER, new_format|OU25_UNDER,
standard|OU25_OVER and standard|OU25_UNDER, each with EXACTLY its parent's settings, so the split is a
pure re-key: every stake is identical. An O/U 2.5 tip without a usable side is never staked.
"""
import copy

import pandas as pd
import pytest

import src.bankroll_policy as BP
from src.bankroll_policy import load_policy, size_day, split_track, track_key

SIDE_TRACKS = ("new_format|OU25_OVER", "new_format|OU25_UNDER", "standard|OU25_OVER", "standard|OU25_UNDER")
#: the parents' settings on the day of the split (config/bankroll_policy.json at 2026-10-26 before the split)
PARENTS = {"new_format": {"state": "ACTIVE_LOW", "base_fraction": 0.005, "max_odds": 2.5, "evidence_score": 1.0},
           "standard": {"state": "PAPER", "base_fraction": 0.0}}
EQUAL = {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}


def _row(fixture, market, side, mt="new_format", tier="SNIPER", odds=2.0, league="Brazil Serie A"):
    return {"date": "2026-10-26", "fixture": fixture, "league": league, "model_type": mt, "market": market,
            "side": side, "signal_tier": tier, "odds": odds}


def _board() -> pd.DataFrame:
    """A mixed day: both sides, both formats, every tier, odds inside/above caps, a paper league, a
    one-position-per-fixture clash and an opposite-direction pair."""
    rows = []
    for i, (side, tier, odds) in enumerate([("OVER", "SNIPER", 2.05), ("UNDER", "MARKSMAN", 1.95),
                                            ("UNDER", "VALUABLE", 2.4), ("OVER", "VALUABLE", 2.6),
                                            ("UNDER", "SNIPER", 3.1), ("OVER", "MARKSMAN", 1.8)]):
        rows.append(_row(f"nf{i}", "OU25", side, tier=tier, odds=odds))
        rows.append(_row(f"st{i}", "OU25", side, mt="standard", tier=tier, odds=odds, league="Championship"))
    rows.append(_row("mls", "OU25", "UNDER", league="USA MLS"))
    rows.append(_row("nf0", "BTTS", "YES", tier="SNIPER", odds=2.1))           # clashes with nf0 OU25 OVER
    rows.append(_row("nf1", "OVER15", "OVER", odds=1.6))                         # opposite to nf1 OU25 UNDER
    return pd.DataFrame(rows)


# ── the key ──────────────────────────────────────────────────────────────────────────────────────────
def test_track_key_splits_ou25_by_side_only():
    assert track_key("new_format", "OU25", "OVER") == "new_format|OU25_OVER"
    assert track_key("standard", "OU25", " under ") == "standard|OU25_UNDER"
    assert track_key("new_format", "BTTS", "YES") == "new_format|BTTS"
    assert track_key("new_format", "OVER15", "OVER") == "new_format|OVER15"
    for bad in (None, "", "YES", "nan", float("nan")):
        assert track_key("new_format", "OU25", bad) == "new_format|OU25_UNKNOWN_SIDE"
    for k in SIDE_TRACKS + ("new_format|BTTS",):
        mt, mk, sd = split_track(k)
        assert track_key(mt, mk, sd) == k
    assert split_track("player_props") == ("player_props", "", None)


# ── the policy file ──────────────────────────────────────────────────────────────────────────────────
def test_each_side_starts_with_exactly_its_parents_settings():
    pol = load_policy()
    assert "new_format|OU25" not in pol["tracks"] and "standard|OU25" not in pol["tracks"]
    assert "new_format|OU25" not in pol["track_tier_multipliers"]
    for k in SIDE_TRACKS:
        assert pol["tracks"][k] == PARENTS[k.split("|")[0]], k
        assert pol["track_tier_multipliers"][k] == EQUAL, k


# ── identical stakes ─────────────────────────────────────────────────────────────────────────────────
def _legacy(policy: dict) -> dict:
    """The same policy in the pre-split shape: one OU25 track per model_type (sides must agree)."""
    p = copy.deepcopy(policy)
    for sect in ("tracks", "track_tier_multipliers"):
        for mt in ("new_format", "standard"):
            o, u = p[sect].pop(f"{mt}|OU25_OVER"), p[sect].pop(f"{mt}|OU25_UNDER")
            assert o == u
            p[sect][f"{mt}|OU25"] = o
    return p


def test_split_is_a_pure_rekey_every_stake_identical(monkeypatch):
    pol = load_policy()
    after = size_day(_board(), pol, bankroll=100.0, peak=100.0, league_ratings={"Brazil Serie A": 7})
    monkeypatch.setattr(BP, "track_key", lambda mt, mk, sd=None: f"{mt}|{mk}")      # the pre-split reader
    before = size_day(_board(), _legacy(pol), bankroll=100.0, peak=100.0, league_ratings={"Brazil Serie A": 7})
    cols = ["deployment_state", "base_fraction", "tier_multiplier", "requested_fraction", "capped_fraction",
            "daily_scale", "final_fraction", "stake_amount", "reason"]
    pd.testing.assert_frame_equal(after[cols], before[cols])
    assert (after["final_fraction"] > 0).sum() >= 3                    # the board really stakes something
    ou = after["market"] == "OU25"
    assert (after.loc[ou, "track"] == after.loc[ou, "model_type"] + "|OU25_" + after.loc[ou, "side"]).all()
    assert (before.loc[ou, "track"] == before.loc[ou, "model_type"] + "|OU25").all()


def test_sides_are_independent_tracks():
    pol = load_policy()
    pol["tracks"]["new_format|OU25_UNDER"] = {"state": "PAPER", "base_fraction": 0.0}
    out = size_day(pd.DataFrame([_row("a", "OU25", "OVER"), _row("b", "OU25", "UNDER")]), pol, 100.0, 100.0)
    r = dict(zip(out["side"], out["reason"]))
    assert r == {"OVER": "OK", "UNDER": "NO_REAL_STAKE_PAPER"}


# ── fail closed ──────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("side", [None, "", "YES", "garbage"])
def test_unknown_side_gets_no_stake_with_a_reason(side):
    out = size_day(pd.DataFrame([_row("a", "OU25", side)]), load_policy(), 100.0, 100.0)
    r = out.iloc[0]
    assert r["track"] == "new_format|OU25_UNKNOWN_SIDE" and r["deployment_state"] == "UNSUPPORTED"
    assert r["reason"] == "NO_REAL_STAKE_UNKNOWN_SIDE" and r["final_fraction"] == 0 and r["stake_amount"] == 0


def test_missing_side_column_stakes_no_ou25_but_still_stakes_btts():
    sel = pd.DataFrame([_row("a", "OU25", "OVER"), _row("b", "BTTS", "YES", tier="SNIPER", odds=2.0)]).drop(columns="side")
    out = size_day(sel, load_policy(), 100.0, 100.0)
    r = dict(zip(out["market"], out["reason"]))
    assert r == {"OU25": "NO_REAL_STAKE_UNKNOWN_SIDE", "BTTS": "OK"}


def test_unknown_side_is_flagged_by_health_checks():
    from src.stake_plan import health_check
    from src.system_health import stake_mapping
    pol = load_policy()
    out = size_day(pd.DataFrame([_row("a", "OU25", None), _row("b", "OU25", "OVER")]), pol, 100.0, 100.0)
    assert any("no usable side" in p for p in health_check(out, pol))
    assert stake_mapping(out.assign(generated_at="x"))["status"] != "GREEN"
    ok = size_day(pd.DataFrame([_row("b", "OU25", "OVER")]), pol, 100.0, 100.0)
    assert health_check(ok, pol) == [] and stake_mapping(ok.assign(generated_at="x"))["status"] == "GREEN"


def test_stake_plan_contract_accepts_side_tracks_and_rejects_a_wrong_side():
    from src.contracts.schemas import _track_ok
    d = pd.DataFrame({"model_type": ["new_format", "new_format", "standard", "new_format"],
                      "market": ["OU25", "OU25", "OU25", "BTTS"], "side": ["OVER", "UNDER", "UNDER", "YES"],
                      "track": ["new_format|OU25_OVER", "new_format|OU25_OVER", "standard|OU25", "new_format|BTTS"]})
    # row 1 carries the OVER track on an UNDER tip; row 2 is a pre-split audit row (parent key, accepted)
    assert _track_ok(d).tolist() == [True, False, True, True]
