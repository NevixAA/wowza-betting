"""P1 2026-10-22 — near-kickoff odds capture, last-seen heartbeat and the CLV certifier.

Before: 34 of 449 settled shadow rows (7.6%) had a certified close. A capture ran inside T-30m for
~30% of fixtures, but the change-log archives write only price MOVES, so an unchanged near-kickoff
price left no proof; and delivered NEAR runs lived 40-50 minutes. These tests pin the schedule,
the heartbeat, the certifier using it (never post-kickoff, never beyond 30 minutes) and the guard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

import update_results as ur
from src import capture_schedule as cs
from src import clv_collection_health as cch
from src import odds_heartbeat as hb
from src import shadow_compare as sc
from src import shadow_settlement as ss

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
MATCH = "Leeds United vs Norwich City"


# ── adaptive schedule ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mins,interval", [
    (5, 0.0), (30, 0.0), (31, 4.0), (60, 4.0), (61, 14.0), (360, 14.0), (361, 55.0), (700, 55.0),
    (None, 0.0), (float("nan"), 0.0)])
def test_due_interval_tightens_towards_kickoff(mins, interval):
    assert cs.due_interval_min(mins) == interval


def test_pass_spacing_tightens_when_a_kickoff_is_imminent():
    assert cs.sleep_seconds(20) == 120 and cs.sleep_seconds(50) == 180
    assert cs.sleep_seconds(300) == 300 and cs.sleep_seconds(None) == 300


def test_loop_runs_its_base_length_even_with_nothing_near():
    assert cs.plan_next([600.0], elapsed_min=10) == (True, 300)


def test_loop_stops_after_base_when_no_kickoff_is_within_keepalive():
    cont, _ = cs.plan_next([600.0], elapsed_min=39)
    assert cont is False


def test_loop_stays_alive_past_base_for_a_kickoff_inside_keepalive():
    cont, sleep_s = cs.plan_next([80.0], elapsed_min=60)
    assert cont is True and sleep_s == 300
    cont, sleep_s = cs.plan_next([25.0], elapsed_min=100)
    assert cont is True and sleep_s == 120


def test_loop_never_passes_the_cap():
    assert cs.plan_next([10.0], elapsed_min=cs.CAP_MIN - 1)[0] is False


def test_started_matches_do_not_keep_the_loop_alive():
    assert cs.plan_next([-5.0, -60.0], elapsed_min=60)[0] is False


def test_a_kickoff_that_passes_during_the_sleep_does_not_keep_it_alive():
    # nearest kickoff 1 minute away, 2-minute sleep: nothing left to observe pre-kickoff
    assert cs.plan_next([1.0], elapsed_min=60)[0] is False


def test_inactive_scheduler_prices_everything_and_prints_no_plan(capsys):
    s = cs.PassScheduler(None)
    assert s.due(1, 10.0) and s.due(1, 10.0)
    s.close()
    assert "LOOP_CONTINUE" not in capsys.readouterr().out


def test_scheduler_skips_a_far_fixture_until_due_and_persists_across_passes(tmp_path, capsys):
    state = tmp_path / "s.json"
    s1 = cs.PassScheduler(state, now=1_000_000, loop_start_s=1_000_000)
    assert s1.due(7, 8.0)                       # 8h out, never fetched -> due
    s1.mark(7)
    assert s1.due(8, 0.3)                       # 18 min out -> due
    s1.mark(8)
    s1.close()
    out = capsys.readouterr().out
    assert "LOOP_CONTINUE=1" in out and "LOOP_SLEEP_S=120" in out
    s2 = cs.PassScheduler(state, now=1_000_000 + 300, loop_start_s=1_000_000)
    assert not s2.due(7, 8.0)                   # 5 minutes later: hourly fixture not due
    assert s2.due(8, 0.2)                       # inside 30 minutes: every pass
    s3 = cs.PassScheduler(state, now=1_000_000 + 56 * 60, loop_start_s=1_000_000)
    assert s3.due(7, 7.0)


def test_corrupt_state_means_everything_due(tmp_path):
    state = tmp_path / "s.json"
    state.write_text("{not json", encoding="utf-8")
    assert cs.PassScheduler(state, now=1).due(1, 10.0)


# ── heartbeat ────────────────────────────────────────────────────────────────────────────────
def _obs(ts, odds=1.90, ko="2026-10-18T14:00:00Z", market="over25", match=MATCH):
    return {"snapshot_ts": ts, "match_date": ko[:10], "kickoff_utc": ko,
            "league": "Championship", "match": match, "market": market, "odds": odds}


def test_heartbeat_records_an_unchanged_price_seen_again(tmp_path):
    hb.record("std_capture", [_obs("2026-10-18T11:00:00Z")], out_dir=tmp_path)
    hb.record("std_capture", [_obs("2026-10-18T13:52:00Z")], out_dir=tmp_path)   # same 1.90
    d = pd.read_csv(tmp_path / "std_capture" / "2026-10-18.csv")
    assert len(d) == 1 and d.loc[0, "snapshot_ts"] == "2026-10-18T13:52:00Z"
    assert d.loc[0, "odds"] == 1.90 and d.loc[0, "writer"] == "std_capture"


def test_heartbeat_never_records_a_post_kickoff_observation(tmp_path):
    hb.record("nf_capture", [_obs("2026-10-18T13:55:00Z")], out_dir=tmp_path)
    assert hb.record("nf_capture", [_obs("2026-10-18T14:00:00Z", 3.4),
                                    _obs("2026-10-18T14:10:00Z", 3.6)], out_dir=tmp_path) == 0
    d = pd.read_csv(tmp_path / "nf_capture" / "2026-10-18.csv")
    assert d.loc[0, "snapshot_ts"] == "2026-10-18T13:55:00Z" and d.loc[0, "odds"] == 1.90


def test_heartbeat_refuses_rows_without_a_kickoff(tmp_path):
    assert hb.record("nf_capture", [_obs("2026-10-18T13:55:00Z", ko="")], out_dir=tmp_path) == 0
    assert not (tmp_path / "nf_capture").exists()


def test_heartbeat_keeps_only_the_near_kickoff_window(tmp_path):
    assert hb.record("nf_capture", [_obs("2026-10-18T07:00:00Z")], out_dir=tmp_path) == 0


def test_an_older_observation_never_replaces_a_newer_one(tmp_path):
    hb.record("std_capture", [_obs("2026-10-18T13:50:00Z", 1.85)], out_dir=tmp_path)
    hb.record("std_capture", [_obs("2026-10-18T13:00:00Z", 1.95)], out_dir=tmp_path)
    d = pd.read_csv(tmp_path / "std_capture" / "2026-10-18.csv")
    assert d.loc[0, "odds"] == 1.85


def test_heartbeat_is_one_row_per_fixture_market(tmp_path):
    for m in range(5):
        hb.record("std_capture", [_obs(f"2026-10-18T13:{10 + m}:00Z"),
                                  _obs(f"2026-10-18T13:{10 + m}:00Z", 1.95, market="under25")],
                  out_dir=tmp_path)
    assert len(pd.read_csv(tmp_path / "std_capture" / "2026-10-18.csv")) == 2


def test_merge_folds_heartbeat_into_its_parent_and_keeps_orphans():
    parent = pd.DataFrame([_obs("2026-10-18T11:00:00Z")])
    beats = {"std_capture": pd.DataFrame([dict(_obs("2026-10-18T13:52:00Z"), writer="x")]),
             "other": pd.DataFrame([_obs("2026-10-18T13:40:00Z")])}
    out = hb.merge_into_archives([("standard_sidemarket_odds_history.csv", parent)], beats)
    assert [n for n, _ in out] == ["standard_sidemarket_odds_history.csv",
                                   "odds_heartbeat/other"]
    assert len(out[0][1]) == 2 and "writer" not in out[0][1].columns


# ── certifier ────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def shadow(tmp_path, monkeypatch):
    f = tmp_path / "v91_shadow.csv"
    monkeypatch.setattr(sc, "SHADOW_FILE", f)
    monkeypatch.setattr(ur, "SHADOW_FILE", f)
    monkeypatch.setattr(ur, "SHADOW_HEALTH_FILE", tmp_path / "shadow_settlement_health.json")
    monkeypatch.setattr(ur, "_shadow_identity", lambda: {})
    return f


def _row(**kw):
    r = {"fixture_id": "evt1", "market": "ou25", "logged_at": "2026-10-17T09:00:00+00:00",
         "match_date": "2026-10-18", "kickoff_utc": "2026-10-18 14:00:00",
         "league": "Championship", "model_type": "standard", "p_v9": 0.55, "v9_side": "OVER",
         "v9_tier": "VALUABLE", "v9_edge": 0.03, "entry_odds": 2.00,
         "home_team": "Leeds United", "away_team": "Norwich City"}
    r.update(kw)
    return r


def _ev():
    return {"home_team": "Leeds", "away_team": "Norwich", "home_score": 2, "away_score": 1,
            "date_str": "2026-10-18", "event_id": ""}


def _settle(archives, now=NOW):
    return ur.settle_shadow_predictions(providers=[("t", lambda lg: [_ev()])],
                                        archives=archives, now=now)


def _read(f):
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def test_heartbeat_turns_a_stale_change_log_close_into_a_certified_one(shadow, tmp_path):
    sc.append_shadow([_row()])
    changelog = pd.DataFrame([_obs("2026-10-18T11:00:00Z")])               # T-3h, then flat
    hb.record("std_capture", [_obs("2026-10-18T13:52:00Z")], out_dir=tmp_path / "hb")
    archives = hb.merge_into_archives([("standard_sidemarket_odds_history.csv", changelog)],
                                      hb.load(tmp_path / "hb"))
    _settle(archives)
    d = _read(shadow)
    assert d.loc[0, "close_status"] == "OK" and float(d.loc[0, "closing_odds"]) == 1.90
    assert float(d.loc[0, "close_minutes_to_kickoff"]) == 8.0
    assert float(d.loc[0, "clv"]) == round((2.00 / 1.90 - 1) * 100, 2)


def test_without_the_heartbeat_the_same_fixture_stays_stale(shadow):
    sc.append_shadow([_row()])
    _settle([("standard_sidemarket_odds_history.csv",
              pd.DataFrame([_obs("2026-10-18T11:00:00Z")]))])
    d = _read(shadow)
    assert d.loc[0, "close_status"] == "STALE_CLOSE_GT_30MIN" and d.loc[0, "clv"] == ""


def test_a_post_kickoff_heartbeat_row_is_never_a_close(shadow):
    """Even if a post-kickoff row reached an archive, the certifier ignores it."""
    sc.append_shadow([_row()])
    bad = pd.DataFrame([_obs("2026-10-18T11:00:00Z"), _obs("2026-10-18T14:05:00Z", 3.4)])
    _settle([("odds_heartbeat/x", bad)])
    d = _read(shadow)
    assert d.loc[0, "close_status"] == "STALE_CLOSE_GT_30MIN"
    assert float(d.loc[0, "last_pre_kickoff_odds"]) == 1.90


def test_a_stale_first_archive_does_not_hide_a_certified_later_one(shadow):
    sc.append_shadow([_row()])
    stale = pd.DataFrame([_obs("2026-10-18T09:00:00Z", 1.80)])
    near = pd.DataFrame([_obs("2026-10-18T13:45:00Z", 1.92)])
    _settle([("standard_odds_history.csv", stale), ("standard_sidemarket_odds_history.csv", near)])
    d = _read(shadow)
    assert d.loc[0, "close_status"] == "OK" and float(d.loc[0, "closing_odds"]) == 1.92
    assert d.loc[0, "close_source"] == "standard_sidemarket_odds_history.csv"


def test_when_no_archive_certifies_the_first_archive_stays_the_stale_reference(shadow):
    sc.append_shadow([_row()])
    a = pd.DataFrame([_obs("2026-10-18T09:00:00Z", 1.80)])
    b = pd.DataFrame([_obs("2026-10-18T12:00:00Z", 1.92)])
    _settle([("standard_odds_history.csv", a), ("standard_sidemarket_odds_history.csv", b)])
    d = _read(shadow)
    assert d.loc[0, "close_status"] == "STALE_CLOSE_GT_30MIN"
    assert float(d.loc[0, "last_pre_kickoff_odds"]) == 1.80


def test_settlement_waits_for_long_capture_runs_to_land_their_close(shadow):
    sc.append_shadow([_row()])
    out = _settle([], now=datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc))   # KO + 3h
    assert _read(shadow).loc[0, "result"] == ""
    assert out["pending"].get("AWAITING_CLOSE_CAPTURE") == 1
    assert ss.CLOSE_SETTLE_DELAY_MIN > cs.CAP_MIN + 15


def test_settlement_writes_the_clv_collection_health_artifact(shadow, tmp_path):
    sc.append_shadow([_row()])
    _settle([("standard_sidemarket_odds_history.csv",
              pd.DataFrame([_obs("2026-10-18T13:52:00Z")]))])
    h = json.loads((tmp_path / "clv_collection_health.json").read_text(encoding="utf-8"))
    assert h["generated_at"] == "2026-10-20T12:00:00Z"
    sh = h["shadow_all_time"]
    assert sh["eligible"] == 1 and sh["with_t30_close"] == 1 and sh["with_t10_close"] == 1
    assert sh["by_league"]["Championship"]["certified_t30_pct"] == 100.0


# ── health + guard ───────────────────────────────────────────────────────────────────────────
def _shadow_frame(n_ok, n_stale, n_none=0, ko="2026-10-24 14:00:00"):
    rows = []
    for i in range(n_ok + n_stale + n_none):
        st, mins = (("OK", "8.0") if i < n_ok else
                    ("STALE_CLOSE_GT_30MIN", "200.0") if i < n_ok + n_stale else ("NO_CLOSE", ""))
        rows.append({"result": "WIN", "close_status": st, "close_minutes_to_kickoff": mins,
                     "market": "ou25", "league": "L", "kickoff_utc": ko})
    return pd.DataFrame(rows)


EXP = {"measure_from": "2026-10-23T00:00:00Z", "min_certified_t30_pct": 35.0,
       "min_shadow_sample": 40, "min_collection_t30_pct": 40.0, "min_collection_sample": 200,
       "collection_window_days": 7, "max_health_age_hours": 30}


def test_health_counts_certified_stale_and_missing_closes():
    h = cch.build(_shadow_frame(10, 25, 5), [], now=datetime(2026, 10, 26, tzinfo=timezone.utc),
                  expectation=EXP)
    s = h["shadow_since_measure_from"]
    assert (s["eligible"], s["with_t30_close"], s["stale_close"], s["no_close"]) == (40, 10, 25, 5)
    assert s["certified_t30_pct"] == 25.0 and s["median_close_minutes_certified"] == 8.0
    assert h["check"]["status"] == "FAIL"


def test_guard_passes_above_the_floor_and_is_lenient_on_a_small_sample():
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)
    assert cch.build(_shadow_frame(30, 20), [], now=now, expectation=EXP)["check"]["status"] \
        == "INSUFFICIENT_SAMPLE"                    # collection sample 0 < 200, shadow passes
    assert cch.build(_shadow_frame(3, 20), [], now=now, expectation=EXP)["check"]["status"] \
        == "INSUFFICIENT_SAMPLE"                    # 23 rows < 40: never a FAIL


def test_rows_before_measure_from_are_not_judged():
    h = cch.build(_shadow_frame(0, 60, ko="2026-10-10 14:00:00"), [],
                  now=datetime(2026, 10, 26, tzinfo=timezone.utc), expectation=EXP)
    assert h["shadow_since_measure_from"]["eligible"] == 0 and h["check"]["status"] != "FAIL"


def test_collection_view_uses_only_pre_kickoff_observations():
    rows = []
    for i in range(250):
        m = f"H{i} vs A{i}"
        rows.append(_obs("2026-10-24T13:50:00Z" if i < 50 else "2026-10-24T10:00:00Z",
                         ko="2026-10-24T14:00:00Z", match=m))
        rows.append(_obs("2026-10-24T14:30:00Z", 4.0, ko="2026-10-24T14:00:00Z", match=m))
    h = cch.build(pd.DataFrame(), [("newformat_odds_history.csv", pd.DataFrame(rows))],
                  now=datetime(2026, 10, 26, tzinfo=timezone.utc), expectation=EXP)
    co = h["collection"]
    assert co["fixture_markets"] == 250 and co["t30_pct"] == 20.0 and co["t10_pct"] == 20.0
    assert co["by_track"]["new_format"]["fixtures"] == 250
    assert h["check"]["status"] == "FAIL"           # 20% < 40% on 250 fixture-markets


def _run_guard(tmp_path, h, now):
    hp, ep = tmp_path / "h.json", tmp_path / "e.json"
    hp.write_text(json.dumps(h), encoding="utf-8")
    ep.write_text(json.dumps(EXP), encoding="utf-8")
    from scripts.check_clv_collection_health import main
    return main(["--health", str(hp), "--expectation", str(ep), "--now", now])


def test_guard_script_exit_codes(tmp_path):
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)
    bad = cch.build(_shadow_frame(5, 45), [], now=now, expectation=EXP)
    good = cch.build(_shadow_frame(30, 20), [], now=now, expectation=EXP)
    assert _run_guard(tmp_path, bad, "2026-10-26T01:00:00+00:00") == 1
    assert _run_guard(tmp_path, good, "2026-10-26T01:00:00+00:00") == 0
    assert _run_guard(tmp_path, good, "2026-10-28T01:00:00+00:00") == 1   # 49h old -> frozen


def test_the_committed_expectation_is_honest_and_loadable():
    e = cch.load_expectation()
    assert e["measure_from"] and 7.6 < e["min_certified_t30_pct"] <= 60
    assert e["min_shadow_sample"] >= 30 and e["min_collection_sample"] >= 100


# ── wiring ───────────────────────────────────────────────────────────────────────────────────
SCRIPTS = {"std_capture": ROOT / "scripts" / "capture_std_sidemarket_odds_forward.py",
           "nf_capture": ROOT / "scripts" / "capture_nf_odds_forward.py"}
WFS = {"std_capture": ROOT / ".github" / "workflows" / "std_odds_capture.yml",
       "nf_capture": ROOT / ".github" / "workflows" / "nf_odds_capture.yml"}


@pytest.mark.parametrize("writer", sorted(SCRIPTS))
def test_capture_scripts_record_the_heartbeat_before_the_no_rows_return(writer):
    s = SCRIPTS[writer].read_text(encoding="utf-8")
    i = s.index(f'odds_heartbeat.record("{writer}", rows)')
    assert i < s.index("    if not rows:")
    assert "_SCHED.due(" in s and "_SCHED.mark(fid)" in s
    assert "finally:\n        _SCHED.close()" in s


@pytest.mark.parametrize("writer", sorted(WFS))
def test_capture_workflows_run_the_adaptive_loop_and_commit_the_heartbeat(writer):
    text = WFS[writer].read_text(encoding="utf-8")
    y = yaml.safe_load(text)
    job = next(iter(y["jobs"].values()))
    assert job["timeout-minutes"] >= cs.CAP_MIN + 15
    run = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "CAPTURE_STATE=" in run and "LOOP_START_S=" in run and "LOOP_CONTINUE" in run
    assert f"LOOP_CAP_MIN={int(cs.CAP_MIN)}" in run
    assert f"git add -f output/odds_heartbeat/{writer}" in run
    assert "git add -A" not in run


def test_update_results_runs_the_guard_last_and_commits_the_health_file():
    y = yaml.safe_load((ROOT / ".github" / "workflows" / "update_results.yml")
                       .read_text(encoding="utf-8"))
    steps = y["jobs"]["update"]["steps"]
    assert "check_clv_collection_health.py" in steps[-1]["run"] and steps[-1]["if"] == "always()"
    assert any("output/clv_collection_health.json" in str(s.get("run", ""))
               and "git add" in str(s.get("run", "")) for s in steps)
