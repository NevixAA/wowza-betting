"""src/staked.py - THE definition of "Staked" (owner 2026-11-04: "1 similar staked").

Staked = the bankroll policy recorded, at the time, a stake > 0 on a real-money track (stake_audit /
stake_plan; REAL_STATES; base_fraction > 0). A tier never proves a stake, a missing record is never
assumed, and every non-staked row carries a label that says why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import staked as S  # noqa: E402


def _rec(home, away="A", *, state="ACTIVE", base=0.0125, amount=1.0, market="BTTS", side="YES",
         model="new_format", date="2026-10-20", gen="2026-10-19 08:00:00", pv="pv1", reason="OK"):
    return dict(date=date, fixture=f"{date}|{home}|{away}", league="L", model_type=model, market=market, side=side,
                deployment_state=state, base_fraction=base, stake_amount=amount, final_fraction=amount / 100,
                reason=reason, policy_version=pv, generated_at=gen)


def _led(home, away="A", *, market="btts", side="YES", model="new_format", date="2026-10-20",
         gen="2026-10-19 09:00", tier="SNIPER", pnl=1.0):
    return dict(match_date=date, home_team=home, away_team=away, market=market, side=side, model_type=model,
                generated_at=gen, signal_tier=tier, pnl=pnl, source="live")


def test_qualifies_needs_real_state_positive_base_and_stake():
    m = S.qualifies(["ACTIVE", "PAPER", "ACTIVE_LOW", "ACTIVE", "BLOCKED", None, "probation"],
                    [0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.005],
                    [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.3])
    assert m.tolist() == [True, False, False, False, False, False, True]


def test_tier_never_decides_and_every_row_is_labelled():
    rec = S.records_from(pd.DataFrame([
        _rec("SNIPER_PAPER", state="PAPER", base=0.0, amount=0.0, reason="NO_REAL_STAKE_PAPER"),
        _rec("VALUABLE_STAKED", amount=0.6),
        _rec("FIRST", gen="2026-10-14 13:39:16", amount=0.0, reason="TIER_NOT_STAKED")]))
    led = pd.DataFrame([_led("SNIPER_PAPER"), _led("VALUABLE_STAKED", tier="VALUABLE"),
                        _led("OLD", gen="2026-10-01 10:00"),            # before the first recorded decision
                        _led("LATE", gen="2026-10-20 10:00"),           # after it, but nothing recorded
                        _led("FIRST")])
    lab = S.label_ledger(led, "side_bets_ledger.csv", rec)
    assert lab["stake_status"].tolist() == [S.NOT_STAKED, S.STAKED, S.PRE_POLICY, S.NO_RECORD, S.NOT_STAKED]
    assert lab.loc[0, "stake_reason"] == "NO_REAL_STAKE_PAPER" and lab.loc[1, "stake_units"] == 0.6
    assert lab.loc[[2, 3], "stake_units"].isna().all()                  # no record -> no stake, never 0
    assert S.is_staked(lab).tolist() == [False, True, False, False, False]


def test_restatement_counts_once_and_earliest_stake_is_the_stake():
    """The same bet logged under a renamed track / new policy version is ONE position; its stake is
    the first qualifying record. A later zero (withdrawn) keeps it staked and is counted."""
    rec = S.records_from(pd.DataFrame([
        _rec("H", market="OU25", side="UNDER", amount=0.5, gen="2026-10-18 06:00:00", pv="v1"),
        _rec("H", market="OU25", side="UNDER", amount=0.42, gen="2026-10-19 06:00:00", pv="v2"),
        _rec("H", market="OU25", side="UNDER", state="PAPER", base=0.0, amount=0.0, gen="2026-10-19 20:00:00",
             pv="v3", reason="NO_REAL_STAKE_PAPER")]))
    dec = S.decisions(rec)
    assert len(dec) == 1 and bool(dec.iloc[0]["staked"]) and dec.iloc[0]["stake_units"] == 0.5
    assert bool(dec.iloc[0]["withdrawn"]) and dec.iloc[0]["n_records"] == 3
    lab = S.label_ledger(pd.DataFrame([_led("H", market=None, side="UNDER")]), "bets_ledger.csv", rec)
    assert lab.loc[0, "stake_status"] == S.STAKED and lab.loc[0, "stake_units"] == 0.5


def test_sides_are_not_pooled_on_ou25_but_single_direction_markets_ignore_bad_sides():
    rec = S.records_from(pd.DataFrame([
        _rec("O", market="OU25", side="OVER", amount=0.4),
        _rec("B", market="BTTS", side="YES/OVER", model="unknown", state="UNSUPPORTED", base=0.0, amount=0.0,
             reason="NO_REAL_STAKE_UNSUPPORTED")]))
    led = pd.DataFrame([_led("O", side="UNDER"), _led("O", side="OVER"), _led("B", model="new_format")])
    lab = S.label(led.assign(market=["ou25", "ou25", "btts"]), rec)
    assert lab["stake_status"].tolist() == [S.NO_RECORD, S.STAKED, S.NOT_STAKED]


def test_model_type_fallback_refuses_ambiguity_and_never_crosses_a_known_model():
    rec = S.records_from(pd.DataFrame([
        _rec("X", market="OU25", side="OVER", model="new_format", amount=0.4),
        _rec("X", market="OU25", side="OVER", model="standard", amount=0.0, state="PAPER", base=0.0),
        _rec("Y", market="OU25", side="OVER", model="new_format", amount=0.4)]))
    led = pd.DataFrame([_led("X", market="ou25", side="OVER", model=None),          # two candidates: refused
                        _led("Y", market="ou25", side="OVER", model=None),          # one candidate: matched
                        _led("Y", market="ou25", side="OVER", model="standard")])   # known, different: refused
    lab = S.label(led, rec)
    assert lab["stake_status"].tolist() == [S.NO_RECORD, S.STAKED, S.NO_RECORD]


def test_ht_and_props_are_never_staked_and_no_records_prove_nothing():
    rec = S.records_from(pd.DataFrame([_rec("H")]))
    ht = pd.DataFrame([_led("H", market="ht_under15")])
    assert S.label_ledger(ht, "ht_ledger.csv", rec)["stake_status"].tolist() == [S.NOT_POLICY]
    assert S.label_ledger(ht, "player_ledger.csv", rec)["stake_status"].tolist() == [S.NOT_POLICY]
    empty = S.records_from()
    lab = S.label_ledger(pd.DataFrame([_led("H", gen="2020-01-01 00:00")]), "side_bets_ledger.csv", empty)
    assert lab["stake_status"].tolist() == [S.NO_RECORD]                  # not PRE_POLICY: start unknown


def test_plan_staked_on_reader_frames():
    plan = pd.DataFrame({"deployment_state": ["ACTIVE", "PAPER", "ACTIVE_LOW", "ACTIVE"],
                         "base_fraction": [0.0125, 0.0, 0.005, 0.0], "final_fraction": [0.01, 0.0, 0.004, 0.01],
                         "stake_amount": [1.0, 0.0, None, 1.0]})
    assert S.plan_staked(plan).tolist() == [True, False, True, False]
    assert S.plan_staked(pd.DataFrame({"final_fraction": [0.01]})).tolist() == [False]   # no state recorded
    assert S.plan_staked(pd.DataFrame()).empty


def test_load_records_reads_audit_and_plan(tmp_path):
    pd.DataFrame([_rec("A", gen="2026-10-14 13:39:16")]).to_csv(tmp_path / "stake_audit.csv", index=False)
    pd.DataFrame([_rec("B", gen="2026-11-01 06:00:00")]).to_csv(tmp_path / "stake_plan.csv", index=False)
    rec = S.load_records(tmp_path)
    assert sorted(rec["home"]) == ["A", "B"]
    assert str(S.policy_start(rec)) == "2026-10-14 13:39:16+00:00"
    assert S.load_records(tmp_path / "missing").empty


def test_mixed_timestamp_formats_still_prove_the_tip_time():
    """bets_ledger holds '2026-04-26 12:56' and one '2026-08-08T12:59:30Z' row; both are pre-policy."""
    rec = S.records_from(pd.DataFrame([_rec("Z", gen="2026-10-14 13:39:16")]))
    led = pd.DataFrame([_led("P", gen="2026-04-26 12:56"), _led("Q", gen="2026-08-08T12:59:30Z")])
    assert S.label_ledger(led, "side_bets_ledger.csv", rec)["stake_status"].tolist() == [S.PRE_POLICY] * 2


def test_committed_records_every_staked_row_qualifies():
    """On the committed stake files: every position labelled staked has a qualifying record."""
    import config
    rec = S.load_records(config.OUTPUT_DIR)
    if rec.empty:
        return
    dec = S.decisions(rec)
    st = dec[dec["staked"]]
    assert (st["stake_units"] > 0).all()
    q = rec[rec["qualifies"]]
    assert len(q) and (q["deployment_state"].isin(S.REAL_STATES)).all()
