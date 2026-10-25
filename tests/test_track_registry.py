"""config/track_registry.json must agree with what production ACTUALLY applies (states, paper
leagues, blocked markets, real-money tracks). A policy or config change that is not regenerated into
the registry fails here, so the read-model cannot drift silently."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import build_track_registry as B  # noqa: E402
from src.bankroll_policy import REAL_STATES, size_day  # noqa: E402

REG = json.loads((ROOT / "config" / "track_registry.json").read_text(encoding="utf-8"))
POLICY = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
REGIMES = json.loads((ROOT / "config" / "league_market_regimes.json").read_text(encoding="utf-8"))
ELIG = {"ENABLED_LEAGUES": sorted(config.ENABLED_LEAGUES), "PAPER_LEAGUES": sorted(config.PAPER_LEAGUES),
        "PROVIDER_UNSUPPORTED": sorted(config.PROVIDER_UNSUPPORTED), "MIN_OVER_ODDS": config.MIN_OVER_ODDS,
        "MIN_UNDER_ODDS": config.MIN_UNDER_ODDS, "NF_MAX_OU_ODDS": config.NF_MAX_OU_ODDS}


def test_registry_covers_every_regime_entry_once():
    keys = [e["key"] for e in REG["entries"]]
    assert len(keys) == len(set(keys)) == REG["n_entries"]
    assert set(keys) == {e["key"] for e in REGIMES["entries"]}


def test_policy_version_matches_the_live_policy():
    assert REG["policy_version"] == POLICY["policy_version"]
    assert all(e["stake_policy"]["policy_version"] == POLICY["policy_version"] for e in REG["entries"])


@pytest.mark.parametrize("field", ["production_state", "track_state", "enabled", "paper_league", "sent",
                                   "real_money", "blocked_market", "provider_supported"])
def test_stored_production_fields_equal_a_fresh_derivation(field):
    for e in REG["entries"]:
        pv = B.production_view(e["league"], e["market"], e["side"], e["model_type"], POLICY, ELIG)
        assert e[field] == pv[field], (e["key"], field, e[field], pv[field])


def test_registry_agrees_with_bankroll_policy_size_day():
    """Size one SNIPER tip per entry through the production sizing function and compare."""
    for i, e in enumerate(REG["entries"]):
        caps = e["odds_caps"]
        lo = max(float(caps.get("track_min_odds") or 1.0), float(caps.get("min_side_odds") or 1.0))
        hi = min(float(caps.get("track_max_odds") or 9.0), float(caps["hard_max_odds"]) - 0.01)
        odds = round((lo + hi) / 2, 2) if hi > lo else lo
        sel = pd.DataFrame([{"date": "2026-10-25", "fixture": f"2026-10-25|H{i}|A{i}", "league": e["league"],
                             "model_type": e["model_type"], "market": e["market"], "side": e["side"],
                             "signal_tier": "SNIPER", "odds": odds}])
        r = size_day(sel, POLICY, bankroll=100.0, peak=100.0).iloc[0]
        assert r["deployment_state"] == e["track_state"], e["key"]
        staked_path = not (str(r["reason"]).startswith("NO_REAL_STAKE") or r["reason"] == "PAPER_LEAGUE")
        assert staked_path == e["real_money"], (e["key"], r["reason"], e["real_money"])
        if e["paper_league"] and e["track_state"] in REAL_STATES:
            assert r["reason"] == "PAPER_LEAGUE", e["key"]
        if e["blocked_market"]:
            assert r["reason"] == f"NO_REAL_STAKE_{e['track_state']}" and r["final_fraction"] == 0, e["key"]


def test_paper_leagues_match_production_and_are_not_sent():
    assert REG["global"]["paper_leagues"] == sorted(set(config.PAPER_LEAGUES) | set(POLICY.get("paper_leagues", [])))
    frame = pd.DataFrame({"league": [e["league"] for e in REG["entries"]], "key": [e["key"] for e in REG["entries"]]})
    kept = set(config.drop_paper_leagues(frame)["key"])
    assert kept == {e["key"] for e in REG["entries"] if e["sent"]}


def test_blocked_and_real_money_tracks_match_the_policy():
    tracks = POLICY["tracks"]
    assert REG["global"]["blocked_tracks"] == sorted(k for k, v in tracks.items()
                                                     if v["state"] in ("BLOCKED", "BLOCKED_FROM_REAL"))
    assert REG["global"]["real_money_tracks"] == sorted(k for k, v in tracks.items()
                                                        if v["state"] in REAL_STATES and v.get("base_fraction", 0) > 0)
    assert REG["global"]["player_props"] == tracks["player_props"]["state"] == "BLOCKED_FROM_REAL"
    for e in REG["entries"]:
        if e["market"] == "OVER35":
            assert e["blocked_market"] and not e["real_money"]


def test_selection_engine_is_legacy_in_production():
    assert REG["selection_engine_version"]["production"] == "legacy_raw_edge"
    assert all(e["selection_engine"]["production"] == "legacy_raw_edge" for e in REG["entries"])


def test_an_unregenerated_policy_change_would_be_caught():
    """The derivation test above is only useful if a real policy move changes the derivation."""
    import copy
    moved = copy.deepcopy(POLICY)
    moved["tracks"]["new_format|BTTS"]["state"] = "PAPER"
    moved["paper_leagues"] = moved.get("paper_leagues", []) + ["Championship"]
    diffs = [e["key"] for e in REG["entries"]
             if B.production_view(e["league"], e["market"], e["side"], e["model_type"], moved, ELIG)
             ["production_state"] != e["production_state"]]
    assert any("|BTTS|" in k for k in diffs) and any(k.startswith("Championship|") for k in diffs)


def test_build_is_deterministic():
    elig = dict(ELIG)
    a = B.render(B.build(REGIMES, POLICY, {}, elig, {}, {}, {}))
    b = B.render(B.build(REGIMES, POLICY, {}, elig, {}, {}, {}))
    assert a == b
