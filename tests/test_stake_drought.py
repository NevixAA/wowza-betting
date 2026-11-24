"""Stake drought monitor (src/stake_drought.py, owner rule 2026-11-24): window + PENDING from the rule start,
OFF_SEASON, cause attribution, idempotency, and the hard rule - it never writes config/ or models/."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import stake_drought as S

ROOT = Path(__file__).resolve().parents[1]
LG, MK, SD = "Testland League", "OU25", "OVER"
FAM = [(LG, MK, SD, t) for t in S.TIERS]


def _weeks(*rows):
    return pd.DataFrame([{"league_week_id": w, "league": LG, "freeze_ts": f, "excluded": json.dumps(ex or {})}
                         for w, f, ex in rows])


def _cand(fid, wk="Testland League|2026-W50", side=SD, p=0.6, state="TWO_SIDED_MULTI_BOOK", n_books=5, odds=2.0):
    return {"league_week_id": wk, "fixture_id": fid, "league": LG, "market": MK, "side": side,
            "p_model_raw": p, "price_quality_state": state, "n_books": n_books, "odds": odds,
            "kickoff_utc": "2026-12-12T15:00:00Z"}


def _pick(fid, tier, reason, at="2026-12-11T10:00:00Z", wk="Testland League|2026-W50", real=None):
    return {"selection_mode": "TOP3_LEAGUE_GW", "league_week_id": wk, "fixture_id": fid, "league": LG,
            "market": MK, "side": SD, "slot": tier, "stake_reason": reason,
            "track_is_real": (reason == "STAKED") if real is None else real, "issued_at": at}


def _cal(n, start="2026-12-01T15:00:00Z"):
    ko = pd.date_range(start, periods=n, freq="2D", tz="UTC")
    return [pd.DataFrame({"league": LG, "kickoff": ko, "fixture": [f"f{i}" for i in range(n)]})]


def _run(now, ledger=None, weeks=None, cands=None, cal=None, start=None):
    return S.evaluate_product(S.TEAM, FAM, pd.DataFrame(ledger or []),
                              weeks if weeks is not None else _weeks(),
                              pd.DataFrame(cands or [], columns=list(_cand("x").keys())),
                              cal if cal is not None else [], now, start)


# ── window and PENDING ──────────────────────────────────────────────────────────────────────────
def test_window_pending_until_30_days_after_rule_start():
    w = S.window("2026-12-20T23:00:00Z", None)
    assert w["pending"] and w["start"] == S.RULE_START and w["observed_days"] == 29
    w = S.window("2026-12-21T00:30:00Z", None)
    assert not w["pending"] and w["start"] == S.RULE_START and w["observed_days"] == 30
    w = S.window("2027-01-15T12:00:00Z", None)
    assert w["start"] == pd.Timestamp("2026-12-16T00:00:00Z") and not w["pending"]


def test_window_never_before_rule_start_and_product_start():
    early = pd.Timestamp("2026-11-20T13:00:00Z")          # a product start BEFORE the rule start
    assert S.window("2026-12-01", early)["observed_start"] == S.RULE_START
    late = pd.Timestamp("2026-11-23T00:00:00Z")            # props TOP-3 starts later -> later first flag
    assert S.window("2026-12-21T05:00:00Z", late)["pending"]
    assert not S.window("2026-12-23T05:00:00Z", late)["pending"]


def test_pending_family_never_flagged_even_with_nothing_staked():
    rows = _run("2026-12-05T08:00:00Z", cal=_cal(5))
    assert {r["state"] for r in rows} == {S.PENDING}
    assert all(r["dominant_cause"] == "" for r in rows)


def test_ok_when_staked_in_window_and_stake_before_rule_start_does_not_count():
    wk = _weeks(("Testland League|2026-W50", "2026-12-11T09:00:00Z", None))
    led = [_pick("a", "T1_BEST_VALUE", "STAKED")]
    rows = {r["tier"]: r for r in _run("2026-12-22T08:00:00Z", led, wk, [_cand("a")], _cal(5))}
    assert rows["T1_BEST_VALUE"]["state"] == S.OK and rows["T1_BEST_VALUE"]["days_since_last_stake"] == 11
    assert rows["T2_SAFEST"]["state"] == S.FLAG
    # a STAKED pick issued before 2026-11-21 is never counted
    old = [_pick("a", "T1_BEST_VALUE", "STAKED", at="2026-11-20T23:06:00Z")]
    r = {r["tier"]: r for r in _run("2026-12-22T08:00:00Z", old, wk, [_cand("a")], _cal(5))}
    assert r["T1_BEST_VALUE"]["state"] == S.FLAG and r["T1_BEST_VALUE"]["last_stake_at"] == ""


def test_staked_needs_track_is_real():
    wk = _weeks(("Testland League|2026-W50", "2026-12-11T09:00:00Z", None))
    led = [_pick("a", "T1_BEST_VALUE", "STAKED", real=False)]
    r = {r["tier"]: r for r in _run("2026-12-22T08:00:00Z", led, wk, [_cand("a")], _cal(5))}
    assert r["T1_BEST_VALUE"]["state"] == S.FLAG


def test_stake_outside_rolling_window_flags():
    wk = _weeks(("Testland League|2026-W48", "2026-11-25T09:00:00Z", None))
    led = [_pick("a", "T1_BEST_VALUE", "STAKED", at="2026-11-25T10:00:00Z", wk="Testland League|2026-W48")]
    r = {r["tier"]: r for r in _run("2027-01-10T08:00:00Z", led, wk, [], _cal(10, "2026-12-15T15:00:00Z"))}
    assert r["T1_BEST_VALUE"]["state"] == S.FLAG and r["T1_BEST_VALUE"]["days_since_last_stake"] == 46


# ── OFF_SEASON ──────────────────────────────────────────────────────────────────────────────────
def test_off_season_when_calendar_has_no_fixture_in_window():
    rows = _run("2026-12-22T08:00:00Z", cal=_cal(4, "2026-10-01T15:00:00Z"))   # all before the window
    assert {r["state"] for r in rows} == {S.OFF_SEASON}
    rows = _run("2026-12-22T08:00:00Z", cal=_cal(4, "2026-12-01T15:00:00Z"))
    assert {r["state"] for r in rows} == {S.FLAG}


# ── cause attribution ───────────────────────────────────────────────────────────────────────────
def test_causes_counted_and_dominant():
    wk = _weeks(("Testland League|2026-W50", "2026-12-11T09:00:00Z", None))
    cands = [_cand("nm", p=None), _cand("np", state="INSUFFICIENT"), _cand("b2", n_books=2),
             _cand("os", side="UNDER"), _cand("ot"), _cand("q1"), _cand("p1"), _cand("p2"), _cand("p3")]
    led = [_pick("ot", "T2_SAFEST", "STAKED"),
           _pick("p1", "T1_BEST_VALUE", "NO_STAKE_BELOW_BAR"), _pick("p2", "T1_BEST_VALUE", "NO_STAKE_BELOW_BAR"),
           _pick("p3", "T1_BEST_VALUE", "NO_STAKE_BELOW_BAR")]
    r = {x["tier"]: x for x in _run("2026-12-22T08:00:00Z", led, wk, cands, _cal(1, "2026-12-12T15:00:00Z"))}
    t1 = r["T1_BEST_VALUE"]
    assert t1["state"] == S.FLAG
    assert t1["n_no_model_prediction"] == 1
    assert t1["n_no_price_under_3_books"] == 2           # no two-sided price + only 2 books
    assert t1["n_model_took_other_side"] == 1
    assert t1["n_picked_other_tier"] == 1
    assert t1["n_not_picked"] == 1                         # q1: qualified, ranked out
    assert t1["n_no_stake_below_bar"] == 3
    assert t1["dominant_cause"] == "NO_STAKE_BELOW_BAR"
    assert r["T2_SAFEST"]["state"] == S.OK


def test_no_fixtures_cause_when_calendar_fixtures_never_reach_top3():
    wk = _weeks(("Testland League|2026-W50", "2026-12-11T09:00:00Z", None))
    r = _run("2026-12-22T08:00:00Z", [], wk, [], _cal(6))
    assert all(x["state"] == S.FLAG and x["dominant_cause"] == "NO_FIXTURES" and x["n_no_fixtures"] == 6
               for x in r)


def test_real_pick_gate_codes_are_attributed():
    wk = _weeks(("Testland League|2026-W50", "2026-12-11T09:00:00Z", {"DUPLICATE_PICK": 4, "NO_LEAGUE_BAR": 1}))
    led = [_pick("a", "T1_BEST_VALUE", "NO_STAKE_REAL_PICK_GATE")]
    r = {x["tier"]: x for x in _run("2026-12-22T08:00:00Z", led, wk, [_cand("a")], _cal(1, "2026-12-12T15:00:00Z"))}
    assert r["T1_BEST_VALUE"]["n_blocked_real_pick_gate"] == 5
    assert r["T1_BEST_VALUE"]["dominant_cause"] == "BLOCKED_REAL_PICK_GATE"


def test_props_unproven_pick_counts_as_no_approved_bar():
    fam = [("Prop League", "goals", "", t) for t in S.TIERS]
    led = pd.DataFrame([{"selection_mode": "PROPS_TOP3_LEAGUE_GW", "league_week_id": "Prop League|2026-W50",
                         "fixture_id": "1", "player_id": "9", "league": "Prop League", "market": "goals",
                         "role": "T1_BEST_VALUE", "stake": 0.0, "cell_status": "UNPROVEN",
                         "issued_at": "2026-12-11T10:00:00Z"}])
    cands = pd.DataFrame([{"league_week_id": "Prop League|2026-W50", "fixture_id": "1", "player_id": "9",
                           "league": "Prop League", "market": "goals", "value": 0.1},
                          {"league_week_id": "Prop League|2026-W50", "fixture_id": "1", "player_id": "8",
                           "league": "Prop League", "market": "goals", "value": -0.1}])
    weeks = pd.DataFrame([{"league_week_id": "Prop League|2026-W50", "league": "Prop League",
                           "freeze_ts": "2026-12-11T09:00:00Z", "excluded": "{}"}])
    rows = S.evaluate_product(S.PROPS, fam, led, weeks, cands, [], "2026-12-30T08:00:00Z",
                              pd.Timestamp("2026-11-23T00:00:00Z"))
    t1 = {r["tier"]: r for r in rows}["T1_BEST_VALUE"]
    assert t1["state"] == S.FLAG and t1["n_no_stake_no_approved_bar"] == 1 and t1["n_no_stake_no_value"] == 1


def test_universe_skips_paper_leagues_and_unseen_markets():
    reg = [{"league": "A", "market": "OU25", "side": "OVER", "enabled": True, "paper_league": False},
           {"league": "A", "market": "BTTS", "side": "YES", "enabled": True, "paper_league": False},
           {"league": "MLS", "market": "OU25", "side": "OVER", "enabled": True, "paper_league": True}]
    cands = pd.DataFrame({"market": ["OU25"], "side": ["OVER"]})
    u = S.team_universe(reg, cands, {"MLS"})
    assert u == [("A", "OU25", "OVER", t) for t in sorted(S.TIERS)]
    pu = S.props_universe(["P", "MLS"], pd.DataFrame({"market": ["goals", "sot"]}), {"MLS"})
    assert len(pu) == 6 and all(x[0] == "P" for x in pu)


# ── output: idempotent, and never a write to config/ or models/ ─────────────────────────────────
def _digest(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(d.rglob("*")) if p.is_file()}


def test_write_is_idempotent_and_touches_only_its_two_files(tmp_path, monkeypatch):
    inputs = {"registry": [{"league": LG, "market": MK, "side": SD, "enabled": True}], "paper": [],
              "prop_leagues": [], "team_start": None, "props_start": None,
              "team_ledger": pd.DataFrame(), "team_weeks": _weeks(), "team_cands": pd.DataFrame([_cand("a")]),
              "props_ledger": pd.DataFrame(), "props_weeks": pd.DataFrame(), "props_cands": pd.DataFrame(),
              "team_calendar": _cal(3), "props_calendar": [], "props_predictions": None, "bars_ctx": None}
    monkeypatch.setattr(S, "load_inputs", lambda od=None, root=None: inputs)
    a = S.write("2026-12-22T08:00:00Z", tmp_path)
    assert a["changed"] and sorted(p.name for p in tmp_path.iterdir()) == [S.CSV_NAME, S.JSON_NAME]
    before = _digest(tmp_path)
    b = S.write("2026-12-22T21:59:00Z", tmp_path)          # same day, same inputs -> same bytes, no write
    assert not b["changed"] and _digest(tmp_path) == before
    doc = json.loads((tmp_path / S.JSON_NAME).read_text(encoding="utf-8"))
    assert doc["counts"]["TOP3"][S.FLAG] == 3 and "odds" not in (tmp_path / S.CSV_NAME).read_text()


def test_never_writes_config_or_models(tmp_path):
    cfg, mdl = _digest(ROOT / "config"), _digest(ROOT / "models")
    S.write("2026-12-22T08:00:00Z", tmp_path, ROOT)          # the real loaders over the real serving config
    assert _digest(ROOT / "config") == cfg and _digest(ROOT / "models") == mdl
    assert sorted(p.name for p in tmp_path.iterdir()) == [S.CSV_NAME, S.JSON_NAME]


def test_module_states_the_hard_rule():
    doc = S.__doc__
    assert "NEVER changes bars, stakes, tiers or models" in doc and "FLAGS" in doc
    src = Path(S.__file__).read_text(encoding="utf-8")
    for bad in ("to_json(", "threshold_apply", "save_params", "write_params"):
        assert bad not in src


# ── visibility: system health (ungated AMBER) and the Admin Board frame ─────────────────────────
def test_system_health_check_is_amber_when_flagged_and_ungated():
    from src import system_health as SH
    fam = {"product": "TOP3", "league": LG, "market": MK, "side": SD, "tier": "T1_BEST_VALUE",
           "state": S.FLAG, "dominant_cause": "NO_FIXTURES"}
    c = SH.stake_drought_check({"total": {S.FLAG: 1, S.PENDING: 0, S.OFF_SEASON: 0, S.OK: 2}, "families": [fam]})
    assert c["check"] == "Stake drought" and c["status"] == SH.AMBER and c["gated"] is False
    c = SH.stake_drought_check({"total": {S.FLAG: 0, S.PENDING: 5, S.OFF_SEASON: 0, S.OK: 0}, "families": []})
    assert c["status"] == SH.GREEN
    assert SH.stake_drought_check({})["status"] == SH.GREY


def test_admin_board_frame_puts_flagged_first():
    from dashboard_data import admin_board as AB
    doc = {"families": [
        {"product": "TOP3", "league": "B", "market": "OU25", "side": "OVER", "tier": "T2_SAFEST", "state": S.OK,
         "days_since_last_stake": 3, "n_staked_window": 1},
        {"product": "TOP3", "league": "A", "market": "BTTS", "side": "YES", "tier": "T1_BEST_VALUE", "state": S.FLAG,
         "dominant_cause": "NO_STAKE_BELOW_BAR", "days_since_last_stake": None, "n_no_stake_below_bar": 4}]}
    f = AB.drought_frame(doc)
    assert list(f["State"]) == [S.FLAG, S.OK]
    assert f.loc[0, "Cause"] == "NO_STAKE_BELOW_BAR" and f.loc[0, "Days since last stake"] == "never"
    assert "NO_STAKE_BELOW_BAR 4" in f.loc[0, "Cause counts"]
