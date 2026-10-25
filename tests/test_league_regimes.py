"""Research regime registry (config/league_market_regimes.json): deterministic, complete, and its
production_state is what production actually applies (owner rule 2026-10-25)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.bankroll_policy import REAL_STATES, load_policy, size_day

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "config" / "league_market_regimes.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("build_league_regimes", ROOT / "scripts" / "build_league_regimes.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def _af(league="Argentina Primera Division", market="btts", side="YES", mt="new_format", n_days=10, per_day=5,
        win_every=2, odds=2.0, tier="SNIPER", p=0.6, start="2026-08-15"):
    rows = []
    for d in range(n_days):
        day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=7 * d, hours=15)
        for i in range(per_day):
            won = float(i % win_every == 0) if win_every else 0.0
            rows.append({"fx": f"{league}|{d}|{i}", "league": league, "model_type": mt, "market": market,
                         "side": side, "kick": day, "p_model": p, "odds": odds, "opp_odds": 1.9,
                         "p_market_fair": 0.5, "prod_tier": tier, "close_status": "VALID_PRE_KICKOFF_CLOSE",
                         "gap_h": 0.25, "clv": 1.0, "outcome_yes": won if side in ("OVER", "YES") else 1 - won,
                         "won": won, "pnl": odds - 1 if won else -1.0, "md": day.tz_convert(None).normalize()})
    return pd.DataFrame(rows)


def _build(af, policy=None, leagues=("Argentina Primera Division",)):
    return B.build_registry(af, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), policy or load_policy(),
                            "2026-10-25", leagues=list(leagues))


def _entry(reg, key):
    return next(e for e in reg["entries"] if e["key"] == key)


def test_generation_is_deterministic():
    af = pd.concat([_af(), _af(market="ou25", side="UNDER", win_every=3, odds=1.9)], ignore_index=True)
    a = json.dumps(_build(af), sort_keys=True)
    b = json.dumps(_build(af.sample(frac=1.0, random_state=7)), sort_keys=True)   # row order must not matter
    assert a == b


def test_every_tipped_league_market_has_an_entry():
    keys = {e["key"] for e in REG["entries"]}
    for lg in config.ENABLED_LEAGUES:
        mt = config.model_type_for_league(lg)
        for market, side, _a, _l in B.SPECS:
            assert f"{lg}|{market}|{side}|{mt}" in keys, (lg, market, side)
    assert REG["n_entries"] == len(REG["entries"]) == len(keys)
    assert sum(REG["regime_counts"].values()) == REG["n_entries"]


def test_protected_entries_are_owner_locked_and_only_they_are():
    for e in REG["entries"]:
        assert e["owner_locked"] is (e["research_regime"] == "PROTECTED"), e["key"]
        assert e["research_regime"] in ("PROTECTED", "REPAIR", "LEARNING")
        assert e["reason"]


def test_production_state_is_what_production_applies():
    policy = load_policy()
    tracks = policy["tracks"]
    assert tracks["standard|OU25"]["state"] == "PAPER"
    assert tracks["standard|OVER35"]["state"] == tracks["new_format|OVER35"]["state"] == "BLOCKED"
    assert tracks["player_props"]["state"] == "BLOCKED_FROM_REAL"
    assert REG["global_protections"]["player_props"] == "BLOCKED_FROM_REAL"
    sel = []
    for i, e in enumerate(REG["entries"]):
        if e["model_type"] == "standard" and e["market"] == "OU25":
            assert e["production_state"] == "PAPER"
        if e["market"] == "OVER35":
            assert e["production_state"] in ("BLOCKED", "PAPER_LEAGUE")
        if config.is_paper_league(e["league"]):
            assert e["production_state"] == "PAPER_LEAGUE"
        sel.append({"fixture": f"f{i}", "league": e["league"], "model_type": e["model_type"],
                    "market": e["market"], "signal_tier": "SNIPER", "odds": 1.80})
    # the production sizer itself: a SNIPER at 1.80 gets a real stake exactly where the registry says the
    # state is a real-money state
    out = size_day(pd.DataFrame(sel), policy, bankroll=1000.0, peak=1000.0)
    for e, (_, r) in zip(REG["entries"], out.iterrows()):
        assert (r["requested_fraction"] > 0) == (e["production_state"] in REAL_STATES), (e["key"], r["reason"])


def test_one_losing_week_never_moves_an_entry():
    af = _af(n_days=1, per_day=60, win_every=0)                 # 60 straight losses, ONE matchday
    e = _entry(_build(af), "Argentina Primera Division|BTTS|YES|new_format")
    assert e["research_regime"] == "LEARNING" and e["independent_matchdays"] == 1


def test_sustained_loss_is_repair_and_credible_staked_profit_is_protected():
    loss = _af(n_days=12, per_day=4, win_every=0, odds=1.9)
    e = _entry(_build(loss), "Argentina Primera Division|BTTS|YES|new_format")
    assert e["research_regime"] == "REPAIR" and e["recent_roi"]["ci90"][1] < 0
    assert e["repair_priority_terms"]["neg_roi"] > 0

    win = _af(n_days=12, per_day=4, win_every=1, odds=1.9)     # every bet wins on an ACTIVE track
    e = _entry(_build(win), "Argentina Primera Division|BTTS|YES|new_format")
    assert e["research_regime"] == "PROTECTED" and e["owner_locked"] is True

    # the same profit on a PAPER track is not protected: protection is for staked tracks only
    std = _af(league="Championship", market="btts", mt="standard", n_days=12, per_day=4, win_every=1, odds=1.9)
    e = _entry(_build(std, leagues=("Championship",)), "Championship|BTTS|YES|standard")
    assert e["production_state"] == "PAPER" and e["research_regime"] == "LEARNING"


def test_market_beating_model_is_repair_and_scored_separately():
    af = _af(n_days=12, per_day=10, win_every=2, tier="AVOID")     # no signals; 120 priced fixtures
    af["p_model"] = np.where(af["won"] == 1, 0.2, 0.8)             # confidently wrong
    e = _entry(_build(af), "Argentina Primera Division|BTTS|YES|new_format")
    assert e["evidence_n"] == 0 and e["research_regime"] == "REPAIR"
    assert e["repair_priority_terms"]["market_beats_model"] == pytest.approx(0.20)
    assert e["repair_priority_terms"]["neg_roi"] == 0


def test_registry_documents_its_rules():
    assert REG["rules"]["version"] == B.RULES_VERSION and REG["repair_priority"]["weights"] == B.WEIGHTS
    assert abs(sum(B.WEIGHTS.values()) - 1.0) < 1e-9
