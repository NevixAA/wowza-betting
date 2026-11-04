"""Stake-contract guards (Phase 2 / P1_STAKE_CONTRACT_RECONCILIATION, 2026-11-03).

The committed output/stake_audit.csv holds 110 rows with side "YES/OVER" and blank odds (written before
219c1848, 2026-10-22) and 78 rows the one-position rule flags. These tests make sure NO NEW production
row can repeat those shapes, and that the health checks show them instead of reading GREEN:

  1. every reason size_day can write satisfies the STAKE_PLAN_ROW / STAKE_AUDIT_ROW contract pattern
     (NO_LEAGUE_THRESHOLD_EVIDENCE, NO_LEAGUE_RATING, OPPOSITE_DIRECTION_ON_FIXTURE ... were outside it);
  2. a side that does not belong to its market (BTTS "YES/OVER", BTTS "OVER", blank) is never staked -
     before this, size_day ignored `side` on every non-O/U market and staked the row;
  3. the side-selection contract rejects such a side loudly (same discipline as a missing column);
  4. one position (fixture, market, side) can never carry two stakes in one run, whatever the policy's
     one_position_per_fixture flag says;
  5. stake_plan.health_check and system_health.stake_mapping report an invalid side and a duplicate
     staked position.
Every guard has a failing case below; none of them changes a stake on a valid tip.
"""
import copy
import re

import pandas as pd
import pytest

from src.bankroll_policy import load_policy, size_day
from src.contracts import SCHEMAS, validate
from src.contracts.schemas import STAKE_REASON_RX


def _row(fixture, market, side, mt="new_format", tier="SNIPER", odds=2.0, league="Brazil Serie A"):
    return {"date": "2026-11-03", "fixture": fixture, "league": league, "model_type": mt, "market": market,
            "side": side, "signal_tier": tier, "odds": odds}


def _pol(**portfolio):
    p = copy.deepcopy(load_policy())
    p["portfolio"].update(portfolio)
    return p


# ── 1. reasons vs contract ─────────────────────────────────────────────────────────────────────────
def _reasons_in_code() -> set[str]:
    import inspect
    import src.bankroll_policy as BP
    src = inspect.getsource(BP.size_day)
    lits = set(re.findall(r'reason = "([A-Z_]+)"', src)) | set(re.findall(r'"reason"\] = "([A-Z_]+)"', src))
    lits |= set(re.findall(r'"([A-Z_]+)" if min_scale is None else "([A-Z_]+)"', src)[0])
    return lits


def test_every_reason_size_day_writes_satisfies_the_contract():
    reasons = _reasons_in_code()
    # sanity: the scan found the reasons we know exist, so an empty result cannot pass vacuously
    assert {"OK", "PAPER_LEAGUE", "ONE_POSITION_PER_FIXTURE", "OPPOSITE_DIRECTION_ON_FIXTURE",
            "NO_LEAGUE_THRESHOLD_EVIDENCE", "NO_LEAGUE_RATING"} <= reasons
    rx = re.compile(STAKE_REASON_RX)
    bad = sorted(r for r in reasons if not rx.fullmatch(r))
    assert bad == [], f"reasons the stake contract would flag as violations: {bad}"


def test_a_global_threshold_row_is_contract_clean():
    """A real size_day row with NO_LEAGUE_THRESHOLD_EVIDENCE validates against STAKE_PLAN_ROW."""
    sel = pd.DataFrame([{**_row("2026-11-03|A|B", "BTTS", "YES"), "league_threshold_evidence": False}])
    out = size_day(sel, load_policy(), 100.0, 100.0)
    assert out.iloc[0]["reason"] == "NO_LEAGUE_THRESHOLD_EVIDENCE"
    v = [x for x in validate(out, SCHEMAS["STAKE_PLAN_ROW"]) if x["column"] == "reason"]
    assert v == []


def test_the_reason_pattern_still_rejects_garbage():
    rx = re.compile(STAKE_REASON_RX)
    for bad in ("", "ok", "SOMETHING_ELSE", "OK; PAPER", "STAKE"):
        assert not rx.fullmatch(bad), bad


# ── 2. invalid side never staked ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("market,side", [("BTTS", "YES/OVER"), ("BTTS", "OVER"), ("BTTS", None), ("BTTS", ""),
                                         ("OVER15", "YES/OVER"), ("OVER15", "YES"), ("OVER15", None)])
def test_a_side_that_does_not_belong_to_its_market_is_never_staked(market, side):
    out = size_day(pd.DataFrame([_row("2026-11-03|A|B", market, side, odds=1.6 if market == "OVER15" else 2.0)]),
                   load_policy(), 100.0, 100.0)
    r = out.iloc[0]
    assert r["final_fraction"] == 0 and r["stake_amount"] == 0
    assert r["reason"] == "NO_REAL_STAKE_INVALID_SIDE"


@pytest.mark.parametrize("market,side,odds", [("BTTS", "YES", 2.0), ("OVER15", "OVER", 1.6)])
def test_a_valid_side_is_still_staked(market, side, odds):
    """The guard must not touch a valid tip (no stake change)."""
    out = size_day(pd.DataFrame([_row("2026-11-03|A|B", market, side, odds=odds)]), load_policy(), 100.0, 100.0)
    assert out.iloc[0]["reason"] == "OK" and out.iloc[0]["final_fraction"] > 0


# ── 3. side-selection contract ─────────────────────────────────────────────────────────────────────
def _side_sel(side, market="btts"):
    return pd.DataFrame([{"date": "2026-11-03", "league": "Brazil Serie A", "home_team": "X", "away_team": "Y",
                          "model_type": "new_format", "market": market, "side": side, "signal_tier": "SNIPER",
                          "odds": 2.0, "model_prob": 0.6, "fair_prob": 0.5, "edge": 0.1, "ev": 0.2}])


@pytest.mark.parametrize("side,market", [("YES/OVER", "btts"), ("OVER", "btts"), ("", "btts"),
                                         ("YES", "over15"), ("YES/OVER", "over35")])
def test_side_selection_contract_rejects_a_side_outside_its_market(side, market):
    from src import stake_plan as sp
    with pytest.raises(sp.SelectionSchemaError):
        sp.check_side_selections(_side_sel(side, market))


@pytest.mark.parametrize("side,market", [("YES", "btts"), ("OVER", "over15"), ("OVER", "over35")])
def test_side_selection_contract_accepts_valid_sides(side, market):
    from src import stake_plan as sp
    sp.check_side_selections(_side_sel(side, market))            # does not raise


# ── 4. duplicate executable position ───────────────────────────────────────────────────────────────
def test_one_position_never_carries_two_stakes_even_with_the_policy_flag_off():
    pol = _pol(one_position_per_fixture=False)
    sel = pd.DataFrame([_row("2026-11-03|A|B", "BTTS", "YES"), _row("2026-11-03|A|B", "BTTS", "YES")])
    out = size_day(sel, pol, 100.0, 100.0)
    assert int((out["final_fraction"] > 0).sum()) == 1
    assert set(out["reason"]) == {"OK", "DUPLICATE_POSITION"}
    assert validate(out, SCHEMAS["STAKE_PLAN_ROW"]) == [] or all(
        v["column"] != "reason" for v in validate(out, SCHEMAS["STAKE_PLAN_ROW"]))


def test_flag_off_still_allows_two_different_positions():
    """The duplicate guard is about the SAME position; it does not enforce one-per-fixture."""
    pol = _pol(one_position_per_fixture=False)
    sel = pd.DataFrame([_row("2026-11-03|A|B", "BTTS", "YES"), _row("2026-11-03|A|B", "OVER15", "OVER", odds=1.6)])
    out = size_day(sel, pol, 100.0, 100.0)
    assert int((out["final_fraction"] > 0).sum()) == 2


def test_production_policy_allows_same_direction_positions_within_the_fixture_cap():
    """Owner 2026-11-04: several bets on one match are fine unless opposite (BTTS YES = OVER), so the
    production policy stakes BTTS YES + Over 1.5 together, bounded by fixture_cap as a total."""
    pol = load_policy()
    out = size_day(pd.DataFrame([_row("2026-11-03|A|B", "BTTS", "YES"),
                                 _row("2026-11-03|A|B", "OVER15", "OVER", odds=1.6)]), pol, 100.0, 100.0)
    assert int((out["final_fraction"] > 0).sum()) == 2
    assert "ONE_POSITION_PER_FIXTURE" not in set(out["reason"])
    assert float(out["final_fraction"].sum()) <= pol["portfolio"]["fixture_cap"] + 1e-9


# ── 5. health checks show it ───────────────────────────────────────────────────────────────────────
def _plan(rows):
    out = size_day(pd.DataFrame(rows), load_policy(), 100.0, 100.0)
    return out.assign(generated_at="2026-11-03 00:00:00")


def test_health_check_reports_an_invalid_side():
    from src.stake_plan import health_check
    p = _plan([_row("2026-11-03|A|B", "BTTS", "YES/OVER")])
    assert any("side" in x.lower() and "market" in x.lower() for x in health_check(p, load_policy()))


def test_health_check_reports_two_staked_rows_on_one_fixture():
    """A plan handed in from elsewhere (or a future regression in size_day) with two stakes on one fixture."""
    from src.stake_plan import health_check
    p = _plan([_row("2026-11-03|A|B", "BTTS", "YES"), _row("2026-11-03|C|D", "BTTS", "YES")])
    p.loc[1, "fixture"] = "2026-11-03|A|B"                        # forge a second stake on fixture A|B
    assert (p["final_fraction"] > 0).sum() == 2
    assert any("fixture" in x.lower() for x in health_check(p, load_policy()))


def test_health_check_clean_plan_is_empty():
    from src.stake_plan import health_check
    p = _plan([_row("2026-11-03|A|B", "BTTS", "YES"), _row("2026-11-03|C|D", "OVER15", "OVER", odds=1.6)])
    assert health_check(p, load_policy()) == []


def test_stake_mapping_is_not_green_on_an_invalid_side():
    from src.system_health import stake_mapping
    bad = _plan([_row("2026-11-03|A|B", "BTTS", "YES/OVER"), _row("2026-11-03|C|D", "BTTS", "YES")])
    assert stake_mapping(bad)["status"] != "GREEN"
    ok = _plan([_row("2026-11-03|C|D", "BTTS", "YES")])
    assert stake_mapping(ok)["status"] == "GREEN"
