"""Dashboard truth (Phase 2 dashtruth, 2026-11-05): every page figure must match its INDEPENDENT recomputation.

scripts/dashboard_truth.py recomputes each headline figure from the committed files with plain pandas (no
dashboard module; src/staked.py is THE shared "Staked" definition). These tests fail when a dashboard
function stops agreeing with that recomputation:

  * synthetic fixtures: a small output/ with known answers, read by the dashboard functions the pages call
    (today_view, v9, leagues, production_view, truth_view) and by the reproducer;
  * the checker itself: a shown value that is off by more than its display rounding is a MISMATCH, an
    absent label is UNIDENTIFIABLE, never a pass;
  * smoke on committed data: Home and System Health rendered headlessly (AppTest) against this checkout's
    output/, every figure MATCH.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dt():
    spec = importlib.util.spec_from_file_location("dashboard_truth", ROOT / "scripts" / "dashboard_truth.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DT = _dt()
NOW = "2026-10-20T12:00:00+00:00"

# ── synthetic output/ ──────────────────────────────────────────────────────────────────────────
LED_COLS = ["source", "generated_at", "match_date", "league", "home_team", "away_team", "side", "odds",
            "edge_pct", "signal_tier", "result", "pnl", "model_type"]


def _ledgers(out: Path) -> None:
    rows = [  # O/U 2.5
        ("live", "2026-10-15 08:00", "2026-10-15", "Brazil Serie A", "A", "B", "OVER", 2.0, 8.0, "SNIPER", "WIN", 1.0, "new_format"),
        ("live", "2026-10-16 08:00", "2026-10-16", "Brazil Serie A", "C", "D", "UNDER", 1.9, 6.0, "MARKSMAN", "LOSS", -1.0, "new_format"),
        ("live", "2026-10-17 08:00", "2026-10-17", "Premier League", "E", "F", "OVER", 2.1, 7.0, "SNIPER", "LOSS", -1.0, "standard"),
        ("live", "2026-10-18 08:00", "2026-10-18", "Brazil Serie A", "G", "H", "OVER", 1.8, 9.0, "VALUABLE", "WIN", 0.8, "new_format"),
        ("live", "2026-10-01 08:00", "2026-10-01", "Brazil Serie A", "I", "J", "OVER", 2.2, 9.0, "SNIPER", "WIN", 1.2, "new_format"),
        ("live", "2026-10-15 09:00", "2026-10-15", "USA MLS", "K", "L", "OVER", 2.0, 9.0, "SNIPER", "WIN", 1.0, "new_format"),
        ("live", "2026-10-19 08:00", "2026-10-19", "Brazil Serie A", "M", "N", "OVER", 2.0, 9.0, "SNIPER", "", None, "new_format"),
    ]
    pd.DataFrame(rows, columns=LED_COLS).to_csv(out / "bets_ledger.csv", index=False)
    side = pd.DataFrame([
        ("live", "2026-10-16 08:00", "2026-10-16", "Brazil Serie A", "C", "D", "OVER", 2.5, 5.0, "SNIPER", "WIN", 1.5,
         "new_format", "btts", 0.45),
        ("live", "2026-10-18 08:00", "2026-10-18", "Brazil Serie A", "O", "P", "OVER", 1.5, 4.0, "MARKSMAN", "LOSS", -1.0,
         "new_format", "over15", 0.70)], columns=LED_COLS + ["market", "model_prob"])
    side.to_csv(out / "side_bets_ledger.csv", index=False)


def _plan(out: Path) -> None:
    def r(date, home, away, market, side, mt, state, amt, gen="2026-10-14 07:00"):
        return {"date": date, "fixture": f"{date}|{home}|{away}", "league": "Brazil Serie A", "track": mt,
                "market": market, "side": side, "signal_tier": "SNIPER", "odds": 2.0, "deployment_state": state,
                "base_fraction": 0.01 if amt else 0.0, "final_fraction": amt / 100, "stake_amount": amt,
                "reason": "OK" if amt else "paper track", "decision_state": "BET" if amt else "NO_BET",
                "policy_version": "p1", "model_type": mt, "generated_at": gen}
    audit = [r("2026-10-15", "A", "B", "OU25", "OVER", "new_format", "ACTIVE", 1.0),
             r("2026-10-16", "C", "D", "OU25", "UNDER", "new_format", "ACTIVE", 0.5),
             r("2026-10-17", "E", "F", "OU25", "OVER", "standard", "PAPER", 0.0),
             r("2026-10-18", "G", "H", "OU25", "OVER", "new_format", "ACTIVE", 0.75),
             r("2026-10-15", "K", "L", "OU25", "OVER", "new_format", "ACTIVE", 1.0),
             r("2026-10-16", "C", "D", "BTTS", "YES", "new_format", "ACTIVE", 1.25),
             r("2026-10-18", "O", "P", "OVER15", "OVER", "new_format", "ACTIVE_LOW", 0.5)]
    pd.DataFrame(audit).to_csv(out / "stake_audit.csv", index=False)
    plan = [r("2026-10-20", "Q", "R", "OU25", "OVER", "new_format", "ACTIVE", 1.0, "2026-10-20 06:00"),
            r("2026-10-21", "S", "T", "OU25", "UNDER", "new_format", "PAPER", 0.0, "2026-10-20 06:00"),
            r("2026-10-21", "U", "V", "BTTS", "YES", "new_format", "ACTIVE", 1.25, "2026-10-20 06:00"),
            r("2026-10-19", "W", "X", "OU25", "OVER", "new_format", "ACTIVE", 1.0, "2026-10-19 06:00")]
    pd.DataFrame(plan).to_csv(out / "stake_plan.csv", index=False)


def _truth_files(out: Path) -> None:
    (out / "corrections").mkdir()
    (out / "collect").mkdir()
    pd.DataFrame({"date": ["2022-05-29"] * 3, "reason": ["no_cup_fixture_stored_within_pm1_day"] * 2
                  + ["club_not_resolved_among_cup_fixtures"]}).to_csv(
        out / "corrections" / "history_cup_unproven_rows.csv", index=False)
    pd.DataFrame({"league": ["X"], "reason": ["ft_status_scores_differ"]}).to_csv(
        out / "corrections" / "history_score_refused.csv", index=False)
    (out / "collect" / "player_history_refetch_audit.json").write_text(json.dumps(
        {"generated_at": "2026-10-19T09:00:00+00:00", "plan": {"rows_to_verify": 1000, "refused_rows": {"x": 7}},
         "stored_rows": 250, "run": {"calls": 12, "budget_stop": "cap"}}), encoding="utf-8")
    (out / "collect" / "af_budget.json").write_text(json.dumps({"2026-10-18": 10, "2026-10-19": 0}), encoding="utf-8")
    (out / "data_truth_health.json").write_text(json.dumps(
        {"generated_at": "2026-10-20T06:00:00Z", "stoplight": "GREEN", "gate_reasons": [],
         "checks": [{"check": "Data truth 1: one canonical name", "status": "GREEN", "value": "0"},
                    {"check": "Data truth 3: no missing data", "status": "AMBER", "value": "gap"}]}), encoding="utf-8")


@pytest.fixture()
def out(tmp_path, monkeypatch):
    o = tmp_path / "output"
    o.mkdir()
    _ledgers(o)
    _plan(o)
    _truth_files(o)
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", o)
    return o


def _ctx():
    import config
    return {"now_utc": NOW, "today_local": "2026-10-20", "cutoff": str(config.PERFORMANCE_CUTOFF_DATE),
            "paper_leagues": sorted(config.PAPER_LEAGUES), "thresholds": {}}


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# ── the dashboard functions against the reproducer ────────────────────────────────────────────────
def test_synthetic_book_has_the_known_answer(out):
    v = DT.independent(out, _ctx())
    # staked = A-B O/U (+1.0), C-D O/U (-1.0), G-H O/U VALUABLE tier but staked (+0.8), C-D BTTS (+1.5),
    # O-P Over 1.5 (-1.0). Not: E-F (PAPER track), I-J (no record, before the policy), K-L (paper league),
    # M-N (unsettled).
    assert v["staked_n"] == 5 and _close(v["staked_pnl"], 1.3)
    assert _close(v["stake_units"], 1.0 + 0.5 + 0.75 + 1.25 + 0.5)
    assert v["not_staked_n"] == 2 and v["not_staked_pre_policy_n"] == 1
    assert v["plan_tips"] == 3 and v["plan_active"] == 2 and v["plan_up_ou"] == 1


def test_home_and_tips_book_matches_independent(out):
    from dashboard_data import today_view as T
    v = DT.independent(out, _ctx())
    p = T.performance(T.load_settled(out), today="2026-10-20")
    assert p["n"] == v["staked_n"]
    for k, fid in (("pnl", "staked_pnl"), ("stake_units", "stake_units"), ("stake_pnl", "stake_pnl"),
                   ("drawdown", "drawdown"), ("last7_pnl", "last7_pnl"), ("paper_pnl", "not_staked_pnl")):
        assert _close(p[k], v[fid]), (k, p[k], v[fid])
    assert _close(100 * p["roi"], v["staked_roi_pct"]) and _close(100 * p["win_rate"], v["staked_win_pct"])
    assert p["last7_n"] == v["last7_n"] and p["paper_n"] == v["not_staked_n"]
    assert p["paper_pre_policy_n"] == v["not_staked_pre_policy_n"]
    ex = p["by_market"].loc[p["by_market"]["Market"] != "O/U 2.5", "P/L"].sum()
    assert _close(ex, v["staked_pnl_ex_ou25"])


def test_overview_tracks_match_independent(out, monkeypatch):
    from dashboard_data import v9
    monkeypatch.setattr(v9, "OUT", out)
    v = DT.independent(out, _ctx())
    perf = v9.performance()
    for mt in ("new_format", "standard"):
        s = perf["by_track"].get(mt, {}).get("staked_only", {"n": 0, "pnl": 0.0})
        assert s["n"] == v[f"track_{mt}_n"], mt
        assert _close(s["pnl"], v[f"track_{mt}_pnl"]), mt
        if s["n"]:
            assert _close(100 * s["hit"], v[f"track_{mt}_hit_pct"], 0.01)
            assert _close(100 * s["roi"], v[f"track_{mt}_roi_pct"], 0.01)
            assert _close(100 * s["break_even"], v[f"track_{mt}_be_pct"], 0.01)


def test_leagues_totals_match_independent(out):
    from dashboard_data import leagues as L
    v = DT.independent(out, _ctx())
    t = L.totals(L.settled())
    assert t["tips"] == v["leagues_n"]
    assert _close(t["pl"], v["leagues_pnl"]) and _close(100 * t["hit"], v["leagues_hit_pct"])


def test_production_plan_counts_match_independent(out):
    from dashboard_data import production_view as P
    from dashboard_data import today_view as T
    from src.staked import plan_staked
    v = DT.independent(out, _ctx())
    plan = P.stake_plan(out / "stake_plan.csv", today="2026-10-20")
    assert len(plan) == v["plan_tips"] and int(plan_staked(plan).sum()) == v["plan_active"]
    assert plan["date"].nunique() == v["plan_days"]
    a = T.action(plan)
    assert (a["active"], a["paper"]) == (v["plan_active"], v["plan_paper"])
    assert (a["shadow_bet"], a["shadow_n"]) == (v["plan_shadow_bet"], v["plan_shadow_n"])
    assert T.upcoming_staked_ou(plan) == v["plan_up_ou"]


def test_production_plan_keeps_an_unrecorded_stake_missing(tmp_path):
    from dashboard_data import production_view as P
    sp = tmp_path / "stake_plan.csv"
    sp.write_text("date,fixture,deployment_state,final_fraction\n2026-10-25,a,ACTIVE,\n2026-10-25,b,ACTIVE,0.01\n",
                  encoding="utf-8")
    plan = P.stake_plan(sp).set_index("fixture")
    assert pd.isna(plan.loc["a", "stake_pct"]) and plan.loc["b", "stake_pct"] == 1.0


def test_calibration_matches_independent(out, monkeypatch):
    from dashboard_data import v9
    monkeypatch.setattr(v9, "OUT", out)
    v = DT.independent(out, _ctx())
    # v9.calibration needs >= min_bin rows; read the same rows at min_bin=1
    c = v9.calibration(min_bin=1)
    assert c["n"] == v["cal_n"]
    assert _close(c["claimed"], v["cal_claimed"], 1e-4) and _close(c["realised"], v["cal_realised"], 1e-4)


# ── the data truth panel ───────────────────────────────────────────────────────────────────────
def test_truth_panel_reads_recorded_values(out):
    from dashboard_data import truth_view as TV
    v = DT.independent(out, _ctx())
    p = TV.panel(out, now="2026-10-20T12:00:00Z")
    u = {r["item"]: r["value"] for r in p["unverified"]}
    assert u["Player rows still zeros_unverified"] == v["truth_zeros_unverified"] == 750
    assert u["Unproven cup rows (kept in history)"] == v["truth_cup_unproven"] == 3
    assert u["Refused scores (history vs provider)"] == v["truth_scores_refused"] == 1
    s = p["stoplight"]
    assert s["status"] == "GREEN" and s["basis"].startswith("gate verdict") and s["age_h"] == 6.0
    assert not s["stale"]
    assert p["collection"]["last_collection"] == "2026-10-19"      # audit generated_at; budget day 10-19 = 0 calls


def test_truth_panel_missing_inputs_are_no_data(tmp_path):
    from dashboard_data import truth_view as TV
    p = TV.panel(tmp_path, now="2026-10-20T12:00:00Z")
    assert p["stoplight"]["status"] == TV.NO_DATA
    assert all(r["value"] == TV.NO_DATA for r in p["unverified"])
    assert all(r["filled"] == TV.NO_DATA for r in p["collection"]["rows"])
    assert p["collection"]["last_collection"] == TV.NO_DATA


def test_truth_panel_without_gate_shows_worst_status_labelled(out):
    from dashboard_data import truth_view as TV
    d = json.loads((out / "data_truth_health.json").read_text(encoding="utf-8"))
    d.pop("stoplight")
    (out / "data_truth_health.json").write_text(json.dumps(d), encoding="utf-8")
    s = TV.stoplight(out, now="2026-10-25T12:00:00Z")
    assert s["status"] == "AMBER" and "no gate verdict" in s["basis"] and s["stale"]


# ── the checker itself ─────────────────────────────────────────────────────────────────────────
def _page(**kinds):
    return {"metrics": kinds.get("metrics", []), "stats": kinds.get("stats", []),
            "captions": kinds.get("captions", [])}


def test_compare_flags_a_drifted_figure_and_an_absent_label():
    rendered = {"pages": {"home": _page(metrics=[["Staked P/L (flat 1u)", "+1.31u"]],
                                        stats=[["Settled staked bets", "5"]])}}
    rows = {(r["page"], r["figure"]): r for r in DT.compare(rendered, {"staked_pnl": 1.30, "staked_n": 5,
                                                                        "staked_win_pct": 60.0})}
    home = DT.PAGES["home"]
    assert rows[(home, "staked_pnl")]["status"] == DT.MISMATCH          # 1.31 shown, 1.30 recomputed
    assert rows[(home, "staked_n")]["status"] == DT.MATCH
    assert rows[(home, "staked_win_pct")]["status"] == DT.UNIDENTIFIABLE  # not shown: never a pass
    rendered["pages"]["home"]["metrics"] = [["Staked P/L (flat 1u)", "+1.30u"]]
    rows = {(r["page"], r["figure"]): r for r in DT.compare(rendered, {"staked_pnl": 1.295})}
    assert rows[(home, "staked_pnl")]["status"] == DT.MATCH             # half a cent = display rounding


def test_compare_names_the_old_tier_basis():
    rendered = {"pages": {"home": _page(stats=[["Settled staked bets", "815"]])}}
    r = [x for x in DT.compare(rendered, {"staked_n": 107, "tier:staked_n": 815}) if x["figure"] == "staked_n"
         and x["page"] == DT.PAGES["home"]][0]
    assert r["status"] == DT.MISMATCH and "tier basis" in r["why"] and "REPRODUCES" in r["why"]


# ── smoke: committed data, rendered pages ───────────────────────────────────────────────────────
def test_committed_home_and_health_figures_match():
    pytest.importorskip("streamlit.testing.v1")
    res = DT.run(ROOT, pages=["home", "health"])
    bad = [r for r in res["figures"] if r["status"] != DT.MATCH]
    assert res["counts"][DT.MATCH] >= 20, res["counts"]
    assert not bad, [(r["page"], r["label"], r["shown"], r["independent"], r.get("why")) for r in bad]
