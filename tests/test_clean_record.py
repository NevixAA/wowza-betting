"""The Clean Record split (owner 2026-11-11: "adjust to count from 0 after all the fixing").

THE rule (dashboard_data/clean_record.py): a tip counts in the Clean Record only when it was generated AND played
on or after config.CLEAN_RECORD_START (2026-11-21). A tip dated before the start never reaches a Clean Record
total. The SYNTHETIC post-go-live rows below exist ONLY inside these tests (a temporary output directory); no
committed ledger is touched.
"""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

import config
from dashboard_data import cells_view as CV
from dashboard_data import clean_record as CR
from src import units as U

ROOT = Path(__file__).resolve().parents[1]
STAT = re.compile(r"<span class='wz-stat-l'>(.*?)</span><span class='wz-stat-v'>(.*?)</span>")


# ── the one owner date ─────────────────────────────────────────────────────────────────────────────
def test_one_owner_set_start_date():
    j = json.loads((ROOT / "config" / "trustworthy_performance.json").read_text(encoding="utf-8"))
    assert config.CLEAN_RECORD_START == "2026-11-21" == j["clean_record_start_utc"][:10]
    assert "OWNER DECISION 2026-11-11" in j["_clean_record_start_utc"]
    # the props clean models start the same day (aligned, never a second date)
    assert config.PROPS_CLEAN_MODELS_LIVE_DATE == config.CLEAN_RECORD_START
    # the pre-repair comparison window opens at the old cutoff and ends at the start
    assert config.PERFORMANCE_CUTOFF_DATE < config.CLEAN_RECORD_START
    assert CR.start_date() == "2026-11-21" and CR.start_ts() == pd.Timestamp("2026-11-21", tz="UTC")


# ── the split ─────────────────────────────────────────────────────────────────────────────────────
def _rows():
    return pd.DataFrame({
        "gen": ["2026-11-21 00:00:00", "2026-11-22 08:00", "2026-11-20 23:59:59", "2026-11-20 10:00", None, None,
                "2026-11-22 08:00", "2026-10-01 08:00"],
        "day": ["2026-11-21", "2026-11-23", "2026-11-21", "2026-11-25", "2026-11-15", "2026-11-22",
                "2026-11-19", "2026-10-02"],
    })


def test_assign_needs_both_times_on_or_after_the_start():
    got = CR.assign(_rows(), "gen", "day").tolist()
    assert got == [CR.CLEAN, CR.CLEAN,
                   CR.BEFORE,      # generated one second before the start, played on it: old models
                   CR.BEFORE,      # generated before the start for a match after it: never Clean Record
                   CR.BEFORE,      # no generation time, played before the start
                   CR.UNPLACED,    # no generation time, played after the start: neither record
                   CR.UNPLACED,    # generated after the start for a match before it: impossible, never counted
                   CR.BEFORE]


def test_ledger_without_generation_time_can_only_be_before():
    d = _rows().drop(columns="gen")
    got = CR.assign(d, None, "day")
    assert set(got[pd.to_datetime(d["day"]) >= "2026-11-21"]) == {CR.UNPLACED}
    assert set(got[pd.to_datetime(d["day"]) < "2026-11-21"]) == {CR.BEFORE}


def _settled():
    """today_view.load_settled-shaped rows: a big pre-start winner (and a trap row generated the day before the
    start for a match after it) that must never reach the Clean Record."""
    return pd.DataFrame({
        "_gen": ["2026-11-01 06:00", "2026-11-20 10:00", "2026-11-21 09:00", "2026-11-21 09:00", "2026-11-22 07:00"],
        "_dt": pd.to_datetime(["2026-11-02", "2026-11-22", "2026-11-22", "2026-11-22", "2026-11-23"]),
        "_pnl": [9.0, 5.0, 0.9, -1.0, 1.1],
        "_staked": [True, True, True, False, False],
        "_mkt": ["O/U 2.5", "BTTS", "O/U 2.5", "BTTS", "Over 1.5"],
    })


def test_a_tip_dated_before_the_start_never_counts_in_clean_totals():
    s = CR.team_summaries(_settled())
    c, b = s[CR.CLEAN], s[CR.BEFORE]
    assert (c["all"]["n"], c["all"]["units"]) == (3, 1.0)
    assert (c["staked"]["n"], c["staked"]["units"]) == (1, 0.9)
    assert (b["all"]["n"], b["all"]["units"]) == (2, 14.0)          # the trap row is pre-repair
    assert c["all"]["wins"] == 2 and abs(c["all"]["hit"] - 2 / 3) < 1e-12
    assert abs(c["all"]["roi"] - 1.0 / 3) < 1e-12
    assert c["all"]["status"] == "NOT_ENOUGH_DATA"                  # 3 settled < the owner's 30


def test_records_are_never_mixed_in_one_figure():
    d = _settled()
    both = pd.concat([CR.split(d, "_gen", "_dt")[CR.CLEAN], CR.split(d, "_gen", "_dt")[CR.BEFORE]])
    with pytest.raises(CR.MixedRecordsError):
        CR.summary(both, pnl="_pnl", date="_dt", key="mixed")


def test_empty_clean_record_is_zero_not_an_old_number():
    pre = _settled().iloc[:2]
    s = CR.team_summaries(pre)
    assert s[CR.CLEAN]["all"]["n"] == 0 and s[CR.CLEAN]["all"]["units"] is None
    assert s[CR.BEFORE]["all"]["n"] == 2
    assert "No settled" in CR.empty_text() and "2026-11-21" in CR.empty_text()


def test_committed_ledgers_clean_rows_obey_the_rule():
    from dashboard_data import today_view
    sp = CR.team_split(today_view.load_settled())
    c = sp[CR.CLEAN]
    if len(c):
        assert (pd.to_datetime(c["_gen"], utc=True, format="mixed") >= CR.start_ts()).all()
        assert (pd.to_datetime(c["_dt"]) >= pd.Timestamp(CR.start_date())).all()
    if not CR.started():
        assert c.empty


def test_props_and_top3_follow_the_same_split():
    led = pd.DataFrame({"signal_date": ["2026-11-20", "2026-11-21", "2026-11-21"],
                        "match_date": ["2026-11-22", "2026-11-22", "2026-11-22"],
                        "tier": ["VALUABLE", "QUALIFIED", "AVOID"], "result": ["WIN", "LOSS", "WIN"],
                        "pnl": [0.8, -1.0, 0.5], "market_odds": [1.8, 2.0, 1.5]})
    s = CR.props_summaries(led)
    assert (s[CR.CLEAN]["n"], s[CR.CLEAN]["units"]) == (1, -1.0)     # AVOID is not a tip
    assert (s[CR.BEFORE]["n"], s[CR.BEFORE]["units"]) == (1, 0.8)
    t3 = pd.DataFrame({"issued_at": ["2026-11-20T10:00:00Z", "2026-11-21T10:00:00Z"],
                       "match_date": ["2026-11-22", "2026-11-22"], "flat_pnl": [1.0, -1.0]})
    s3 = CR.top3_summaries(t3)
    assert (s3[CR.CLEAN]["n"], s3[CR.CLEAN]["units"]) == (1, -1.0)


def test_comparison_table_has_the_same_metrics_side_by_side():
    s = CR.team_summaries(_settled())
    t = CR.comparison(s[CR.CLEAN]["all"], s[CR.BEFORE]["all"])
    assert t["Record"].tolist() == [CR.clean_label(), CR.before_label()]
    assert t["Settled"].tolist() == [3, 2] and t["Units (flat 1u)"].tolist() == [1.0, 14.0]


# ── the cells registry (approved proposal) ───────────────────────────────────────────────────────────
def test_cells_registry_is_the_approved_proposal():
    d = CV.load()
    f = CV.cells_frame(d)
    assert d["problems"] == []
    ap = json.loads((ROOT / "config" / "threshold_approvals.json").read_text(encoding="utf-8"))["approvals"]
    assert d["proposal"]["proposal_id"] == max(ap, key=lambda x: x["approved_at"])["proposal_id"]
    ev = d["evidence"]["league_market_status"]
    assert len(f) == len(ev) and not f.duplicated(["League", "Market", "Track"]).any()
    assert CV.counts(f)["PROVEN"] == sum(1 for x in ev if x["status"] == "PROVEN")
    # every staked cell is a policy cell entry with base_fraction > 0, and nothing else stakes
    pol = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))["cell_staking"]["cells"]
    want = {k for k, v in pol.items() if float(v.get("base_fraction") or 0) > 0}
    got = {f"{r['Track']}|{CV.MSIDE_TRACK[[k for k, v in CV.MSIDE_LABEL.items() if v == r['Market']][0]]}|"
           f"{r['League']}" for _, r in CV.staked(f).iterrows()}
    assert got == want
    off = f[f["Status"] == "OFF"]
    assert off["Qualifying bar"].isna().all() and (off["Stake %"] == 0).all()
    served = f[~f["Status"].isin(["OFF", CV.WITHDRAWN])]
    assert served["Qualifying bar"].notna().all()


def test_withdrawn_entries_serve_no_bar_and_say_why():
    """Owner 2026-11-11 ("Bet-time snapshot + re-run", "Strict, keep Argentina"): every side of a withdrawn
    approval entry is WITHDRAWN - paper, no bar (re-run pending), with the recorded reason; an exemption keeps
    its bar. Stakes come from config/bankroll_policy.json only."""
    d = CV.load()
    f = CV.cells_frame(d)
    assert d["problems"] == []
    wd, ex = CV.withdrawals(d["approval"])
    served = CV.served_cells(d["proposal"])
    want = {k for k, c in served.items() if c["_entry_id"] in wd}
    got = {(r["Track"], [m for m, lab in CV.MSIDE_LABEL.items() if lab == r["Market"]][0], r["League"])
           for _, r in f[f["Status"] == CV.WITHDRAWN].iterrows()}
    assert got == want and len(want) >= len(wd)
    w = f[f["Status"] == CV.WITHDRAWN]
    assert w["Qualifying bar"].isna().all() and w["TIER2 boundary"].isna().all()
    assert w["Basis / reason"].str.startswith(CV.WITHDRAWN_TEXT + ": ").all()
    assert w["Study status"].isin(["PROVEN", "PROMISING", "UNPROVEN"]).all()
    for eid in ex:                                                  # an exemption is never withdrawn
        assert eid not in wd
        ks = [k for k, c in served.items() if c["_entry_id"] == eid]
        rows = f[[(r["Track"], [m for m, lab in CV.MSIDE_LABEL.items() if lab == r["Market"]][0], r["League"]) in ks
                  for _, r in f.iterrows()]]
        assert len(rows) and rows["Qualifying bar"].notna().all()
        assert rows["Basis / reason"].str.contains("owner exemption").all()
    assert CV.counts(f)[CV.WITHDRAWN] == len(w)


def test_withdrawal_reading_on_a_synthetic_approval():
    data = {"approval": {"withdrawals": [{"entry_id": "btts|X League", "reason": "close used as bet price"}],
                         "exemptions": []},
            "proposal": {"entries": [{"entry_id": "btts|X League", "proposed": {"bars_opt": {"yes": {
                "track": "new_format", "mside": "btts_yes", "league": "X League", "verdict": "PROMISING",
                "valuable_th": 0.02, "marksman_th": 0.03, "measure": "EV", "mode": "MODEL", "odds_min": None,
                "odds_max": 2.5, "config_basis": "SELECTED", "bar_basis": "OWN_BETS"}}}}]},
            "evidence": {"league_market_status": [{"track": "new_format", "mside": "btts_yes", "league": "X League",
                                                   "status": "PROMISING"}]},
            "policy": {"cell_staking": {"enabled": True, "cells": {}}}, "problems": []}
    f = CV.cells_frame(data)
    r = f.iloc[0]
    assert r["Status"] == CV.WITHDRAWN and pd.isna(r["Qualifying bar"]) and r["Stake"] == "PAPER"
    assert r["Basis / reason"] == f"{CV.WITHDRAWN_TEXT}: close used as bet price" and r["Study status"] == "PROMISING"
    # a withdrawn cell the policy still stakes is a named problem, never hidden
    data["policy"]["cell_staking"]["cells"]["new_format|BTTS|X League"] = {"state": "ACTIVE", "base_fraction": 0.01}
    data["problems"] = []
    CV.cells_frame(data)
    assert any("still stakes" in p for p in data["problems"])


def test_cell_key_of_ledger_tips():
    assert CV.key_of("new_format", "BTTS", "YES", "X") == ("new_format", "btts_yes", "X")
    assert CV.key_of("standard", "O/U 2.5", "OVER", "X") == ("standard", "over25_over", "X")
    assert CV.key_of("standard", "OU25", "UNDER", "X") == ("standard", "over25_under", "X")
    assert CV.key_of("standard", "over15", "OVER", "X") == ("standard", "over15_over", "X")
    assert CV.key_of("standard", "OU25", None, "X")[1] is None           # no side: no cell, never a guess


# ── every results page on a SIMULATED post-go-live dataset (synthetic rows, tests only) ──────────────
SIM_FILES = ("bets_ledger.csv", "side_bets_ledger.csv", "ht_ledger.csv", "stake_audit.csv", "stake_plan.csv",
             "player_ledger.csv")
HOME, AWAY = "Sim Home FC", "Sim Away FC"


@pytest.fixture()
def golive_out(tmp_path, monkeypatch):
    """A copy of the committed ledgers plus synthetic rows after 2026-11-21: three Clean Record team tips (one
    staked through a recorded ACTIVE_LOW stake), one trap tip generated before the start for a match after it,
    and one clean props tip. config.OUTPUT_DIR points at the copy for the test only."""
    out = tmp_path / "output"
    out.mkdir()
    for f in SIM_FILES:
        if (config.OUTPUT_DIR / f).exists():
            shutil.copy(config.OUTPUT_DIR / f, out / f)
    b = pd.read_csv(out / "bets_ledger.csv", low_memory=False)
    base = {c: pd.NA for c in b.columns}
    rows = [
        {**base, "source": "live", "generated_at": "2026-11-21 09:00", "match_date": "2026-11-22", "league": "La Liga 2",
         "home_team": HOME, "away_team": AWAY, "side": "OVER", "odds": 1.9, "edge_pct": 4.0, "signal_tier": "QUALIFIED",
         "result": "WIN", "pnl": 0.9, "model_type": "standard"},
        {**base, "source": "live", "generated_at": "2026-11-21 09:00", "match_date": "2026-11-22", "league": "La Liga 2",
         "home_team": HOME + " B", "away_team": AWAY + " B", "side": "UNDER", "odds": 2.0, "edge_pct": 3.0,
         "signal_tier": "QUALIFIED", "result": "LOSS", "pnl": -1.0, "model_type": "standard"},
        {**base, "source": "live", "generated_at": "2026-11-22 07:00", "match_date": "2026-11-23", "league": "La Liga 2",
         "home_team": HOME + " C", "away_team": AWAY + " C", "side": "UNDER", "odds": 2.1, "edge_pct": 3.0,
         "signal_tier": "QUALIFIED", "result": "WIN", "pnl": 1.1, "model_type": "standard"},
        # the trap: generated BEFORE the start, played after it, a big winner - never in the Clean Record
        {**base, "source": "live", "generated_at": "2026-11-20 10:00", "match_date": "2026-11-22", "league": "La Liga 2",
         "home_team": HOME + " T", "away_team": AWAY + " T", "side": "OVER", "odds": 6.0, "edge_pct": 9.0,
         "signal_tier": "SNIPER", "result": "WIN", "pnl": 5.0, "model_type": "standard"},
    ]
    pd.concat([b, pd.DataFrame(rows)], ignore_index=True).to_csv(out / "bets_ledger.csv", index=False)
    a = pd.read_csv(out / "stake_audit.csv", low_memory=False)
    stake = {c: pd.NA for c in a.columns}
    stake.update({"date": "2026-11-22", "fixture": f"2026-11-22|{HOME}|{AWAY}", "league": "La Liga 2",
                  "model_type": "standard", "market": "OU25", "side": "OVER", "signal_tier": "QUALIFIED", "odds": 1.9,
                  "policy_version": a["policy_version"].dropna().iloc[-1], "track": "standard|OU25_OVER",
                  "deployment_state": "ACTIVE_LOW", "bankroll_before": 100.0, "base_fraction": 0.005,
                  "tier_multiplier": 1.0, "league_multiplier": 1.0, "drawdown_multiplier": 1.0,
                  "requested_fraction": 0.005, "capped_fraction": 0.005, "daily_scale": 1.0, "final_fraction": 0.005,
                  "stake_amount": 0.5, "reason": "OK", "generated_at": "2026-11-21 09:00:00"})
    pd.concat([a, pd.DataFrame([stake])], ignore_index=True).to_csv(out / "stake_audit.csv", index=False)
    p = pd.read_csv(out / "player_ledger.csv", low_memory=False)
    pr = {c: pd.NA for c in p.columns}
    pr.update({"signal_date": "2026-11-21", "match_date": "2026-11-22", "league": "La Liga 2", "home_team": HOME,
               "away_team": AWAY, "player_name": "Sim Player", "market": "shots_on_target", "tier": "QUALIFIED",
               "market_odds": 1.8, "result": "WIN", "pnl": 0.8})
    pd.concat([p, pd.DataFrame([pr])], ignore_index=True).to_csv(out / "player_ledger.csv", index=False)
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    import streamlit as st
    st.cache_data.clear()
    yield out
    st.cache_data.clear()


def _render(name: str):
    from streamlit.testing.v1 import AppTest
    page = next((ROOT / "pages").glob(name))
    at = AppTest.from_file(str(page), default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = {}
    for m in at.metric:
        metrics.setdefault(m.label, m.value)
    stats = {}
    for m in at.markdown:
        for k, v in STAT.findall(m.value):
            stats.setdefault(html.unescape(k), html.unescape(v))
    return at, metrics, stats


def test_simulated_dataset_splits_as_designed(golive_out):
    from dashboard_data import today_view
    s = CR.team_summaries(today_view.load_settled(golive_out))
    assert (s[CR.CLEAN]["all"]["n"], s[CR.CLEAN]["all"]["units"]) == (3, 1.0)
    assert (s[CR.CLEAN]["staked"]["n"], s[CR.CLEAN]["staked"]["units"]) == (1, 0.9)
    led = pd.read_csv(golive_out / "player_ledger.csv", low_memory=False)
    assert CR.props_summaries(led)[CR.CLEAN]["n"] == 1


def test_home_shows_only_the_clean_record_post_golive(golive_out):
    _, m, s = _render("00_*Home.py")
    assert m["Clean Record staked P/L (flat 1u)"] == U.fmt(0.9)
    assert s["Clean Record staked bets"] == "1"
    assert s["Clean Record all tips, paper + staked (flat 1u)"] == f"{U.fmt(1.0)} · n=3"
    # the pre-repair card never includes a Clean Record tip, and the trap (generated before the start) is there
    from dashboard_data import today_view
    before = CR.team_split(today_view.load_settled(golive_out))[CR.BEFORE]
    assert m["Staked P/L (flat 1u)"] == U.fmt(U.total(before.loc[before["_staked"], "_pnl"]))


def test_clean_record_page_and_tips_page_post_golive(golive_out):
    _, m, _ = _render("24_*Clean_Record.py")
    assert m["Clean Record units · All team tips, paper + staked"] == U.fmt(1.0)
    assert m["Clean Record units · Staked bets"] == U.fmt(0.9)
    assert m["Clean Record units · Player props"] == U.fmt(0.8)
    _, m, _ = _render("1_*Dashboard.py")
    assert m["Clean Record staked P/L (flat 1u)"] == U.fmt(0.9)
    assert m["Clean Record staked bets"] == "1"


def test_success_rates_defaults_to_the_clean_record_post_golive(golive_out):
    _, m, _ = _render("5_*Success_Rates.py")
    assert m["Settled observations"] == "1"          # the one staked Clean Record tip, never the trap


@pytest.mark.parametrize("page", ["17_*Leagues.py", "11_*Player_Props.py", "0_*Overview.py",
                                  "25_*Bars_and_Cells.py", "9_*Model_Performance.py"])
def test_other_results_pages_render_post_golive(golive_out, page):
    _render(page)


def test_props_page_shows_clean_record_post_golive_even_without_a_board(golive_out):
    _, m, s = _render("11_*Player_Props.py")
    assert s["Clean Record Settled"] == "1"                          # the clean prop tip only, never the old ones
    assert m["Clean Record Flat ROI (1u, paper)"] == "+80.0%"
    assert "Flat ROI (1u, paper)" in m                               # the pre-repair card, in its own section
