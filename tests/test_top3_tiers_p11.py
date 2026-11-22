"""P11 three named tiers in the live TOP-3 (src/top3_league.py, tier mode) - synthetic data only.

Owner 2026-11-16 / 2026-11-18: T1 best value, T2 safest, T3 risky but valuable per league per gameweek; never
padded; every league its own parameters. The tier mode is used ONLY when config/top3_tiers_p11.json carries the
owner's approval; otherwise the v6 R2 TOP-3 stays (tests/test_top3_league.py).
"""
from __future__ import annotations

import json

import pandas as pd

from src import top3_league as T
from tests.test_top3_league import LEAGUE, _cand, _ctx, _frame, _freeze, _stats


def _cfg(gate="CAL", floor="REL", lams=None, status="APPROVED 2026-11-20", approved_at="2026-11-20T10:00:00Z"):
    lams = lams or {"over25_over": 0.5, "over25_under": 0.5, "btts_yes": 0.5, "over15_over": 0.5, "over35_over": 0.5}
    return {"status": status, "owner_approved_at": approved_at,
            "leagues": {f"standard|{LEAGUE}": {"gate": gate, "t3_floor": floor}},
            "calibration": {"cells": [{"track": "standard", "league": LEAGUE, "mside": m, "lam": v, "lam_se": 0.0,
                                       "lam_rows": 100} for m, v in lams.items()]}}


def _odds(row, o):
    row["odds"] = o
    return row


def _run(rows, cfg):
    return T.freeze_due(_frame(rows), set(), _ctx(), pd.Timestamp("2026-11-20T17:45:00Z"), 120.0, {}, {}, None,
                        T.tier_stats(cfg), None, cfg)


def test_load_tiers_needs_the_owner_approval(tmp_path):
    (tmp_path / "config").mkdir()
    f = tmp_path / "config" / "top3_tiers_p11.json"
    f.write_text(json.dumps(_cfg(status="PROPOSAL - AWAITING OWNER", approved_at=None)), encoding="utf-8")
    assert T.load_tiers(tmp_path) is None
    f.write_text(json.dumps(_cfg()), encoding="utf-8")
    assert T.load_tiers(tmp_path)["leagues"]
    assert T.load_tiers(tmp_path / "missing") is None


def test_three_named_tiers_value_safest_risky():
    # cal_ev = (fair + 0.5 (p - fair)) * odds - 1
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),       # p_cal 0.56, cal_ev +0.120 -> T1
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),     # p_cal 0.81, cal_ev +0.053 -> T2
            _odds(_cand(3, "OVER35", "OVER", pm=0.36, pf=0.30), 3.40),     # p_cal 0.33, cal_ev +0.122, longest odds
            _odds(_cand(4, "BTTS", "YES", pm=0.60, pf=0.55), 1.85)]        # p_cal 0.575, cal_ev +0.064
    picks, weeks = _run(rows, _cfg())
    got = dict(zip(picks["slot"], picks["fixture_id"]))
    assert got == {"T1_BEST_VALUE": "fx3", "T2_SAFEST": "fx2", "T3_RISKY_VALUE": "fx1"}
    assert list(picks["top3_rank"]) == [1, 2, 3]
    assert weeks["n_picks"].iloc[0] == 3


def test_show_all_fills_t3_from_the_rest_never_by_the_old_floor():
    """Owner 2026-11-22 "3 per league, stake by proof": the old t3_floor no longer leaves T3 empty - T3 is the
    longest-odds remaining candidate with a positive edge (here the lock row fx99, edge 0, is not positive, so
    fx3 at 1.85 takes it), and a priced candidate below the bar is still eligible (shown, never staked)."""
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),
            _odds(_cand(3, "BTTS", "YES", pm=0.60, pf=0.55), 1.85)]
    picks, _ = _run(rows, _cfg(floor=2.5))
    assert list(picks["slot"]) == ["T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE"]
    assert dict(zip(picks["slot"], picks["fixture_id"]))["T3_RISKY_VALUE"] == "fx3"


def test_cal_gate_unstakes_tips_without_calibrated_value_bar_gate_stakes_them():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 1.90)]   # raw EV +0.18; lam 0 -> cal_ev < 0
    lams = {"over25_over": 0.0}
    pol = {"top3_league": {"stake": {"rule": "FLAT", "fraction": 0.005, "execution": "MANUAL"}}}

    def run(cfg):
        return T.freeze_due(_frame(rows), set(), _ctx(), pd.Timestamp("2026-11-20T17:45:00Z"), 120.0, {}, pol,
                            None, T.tier_stats(cfg), None, cfg)
    picks, weeks = run(_cfg(gate="CAL", lams=lams))
    # CAL gate fails and, with lam 0, its calibrated p equals the fair price: not stakeable and (owner 2026-11-22
    # "Beat the fair price") never shown
    assert "fx1" not in set(picks["fixture_id"])
    ex = json.loads(weeks["excluded"].iloc[0])
    assert ex.get("TIER_NO_VALUE") == 1 and ex.get("NOT_ABOVE_FAIR_PRICE", 0) >= 1
    picks, _ = run(_cfg(gate="BAR", lams=lams))
    fx1 = picks[picks["fixture_id"] == "fx1"].iloc[0]
    assert fx1["stake_reason"] == T.STAKED and float(fx1["stake"]) == 0.005
    assert fx1["stake_effect"].startswith("REAL 0.5%")


def test_no_tier_config_keeps_the_v6_r2_top3():
    rows = [_cand(1, "OU25", "OVER", ev_lb=0.30), _cand(2, "BTTS", "YES", ev_lb=0.01)]
    picks, _ = _freeze(_frame(rows), _ctx(), stats=_stats({"btts_yes": 1.0, "over25_over": 0.2}))
    assert set(picks["slot"]) == {"TOP"}


def test_btts_book_fallback_only_when_set_and_only_after_bet365_pinnacle(monkeypatch):
    """P11 v2 (owner 2026-11-18 O1): BTTS falls back to William Hill (then 1xBet / Ladbrokes) only when neither
    Bet365 nor Pinnacle has a pair; other markets and the default stay Bet365 / Pinnacle only."""
    from src.bars_opt import oddsapi_prices as P
    ko = pd.Timestamp("2026-10-04T15:00:00Z")
    ts = ko - pd.Timedelta(hours=125)
    rows = [{"fixture_id": "f1", "market": "btts", "league": "L", "event_id": "e1", "bookmaker": "williamhill",
             "ts": ts, "ko": ko, "yes": 1.8, "no": 2.0},
            {"fixture_id": "f2", "market": "btts", "league": "L", "event_id": "e2", "bookmaker": "williamhill",
             "ts": ts, "ko": ko, "yes": 1.8, "no": 2.0},
            {"fixture_id": "f2", "market": "btts", "league": "L", "event_id": "e2", "bookmaker": "pinnacle",
             "ts": ts, "ko": ko, "yes": 1.9, "no": 1.95},
            {"fixture_id": "f3", "market": "over15", "league": "L", "event_id": "e3", "bookmaker": "williamhill",
             "ts": ts, "ko": ko, "yes": 1.3, "no": 3.4}]
    pairs = pd.DataFrame(rows)
    dates = pd.Series({"f1": "2026-10-04", "f2": "2026-10-04", "f3": "2026-10-04"})
    monkeypatch.setattr(P, "BOOK_FALLBACK_BY_MARKET", {})
    got = P.pick_pairs(pairs, dates, bet_role=True)
    assert set(got["fixture_id"]) == {"f2"} and set(got["bookmaker"]) == {"pinnacle"}
    monkeypatch.setattr(P, "BOOK_FALLBACK_BY_MARKET", {"btts": ("williamhill", "onexbet", "ladbrokes_uk")})
    got = P.pick_pairs(pairs, dates, bet_role=True).set_index("fixture_id")["bookmaker"].to_dict()
    assert got == {"f1": "williamhill", "f2": "pinnacle"}           # f3 (O1.5, William Hill only): no price
