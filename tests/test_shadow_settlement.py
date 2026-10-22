"""P0 2026-10-20 — the prospective shadow log must actually get settled.

641 rows were logged from 2026-10-04 and not one was ever settled: no code wrote the grading
columns, and the rows held no team names to join a result on. These tests pin the settlement.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline
import update_results as ur
from src import shadow_compare as sc
from src import shadow_settlement as ss

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


# ── fixtures ──────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def shadow(tmp_path, monkeypatch):
    f = tmp_path / "v91_shadow.csv"
    monkeypatch.setattr(sc, "SHADOW_FILE", f)
    monkeypatch.setattr(ur, "SHADOW_FILE", f)
    monkeypatch.setattr(ur, "SHADOW_HEALTH_FILE", tmp_path / "shadow_settlement_health.json")
    monkeypatch.setattr(ur, "_shadow_identity", lambda: {})
    return f


def _row(fid="evt1", side="OVER", ko="2026-10-18 14:00:00", logged="2026-10-17T09:00:00+00:00",
         home="Leeds United", away="Norwich City", entry=1.95, market="ou25", **kw):
    r = {"fixture_id": fid, "market": market, "logged_at": logged, "match_date": ko[:10],
         "kickoff_utc": ko, "league": "Championship", "model_type": "standard",
         "p_v9": 0.5512345678901234, "p_goal_distribution": 0.52, "p_market": 0.5,
         "v9_side": side, "v9_tier": "VALUABLE", "v9_edge": 0.0312, "entry_odds": entry,
         "production_valuable_gate": 0.03, "production_marksman_gate": 0.08,
         "production_sniper_gate": 0.12, "threshold_regime_id": "abc",
         "v9_model_sha": "43772ac8d639", "dataset_id": "e837b54a1bef",
         "home_team": home, "away_team": away}
    r.update(kw)
    return r


def _ev(home="Leeds", away="Norwich", hs=2, as_=1, date="2026-10-18", eid=""):
    return {"home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
            "date_str": date, "event_id": eid}


def _arch(rows):
    """Archive frame in the drift.py schema."""
    return pd.DataFrame([{"snapshot_date": ts[:10], "snapshot_ts": ts, "match_date": md,
                          "league": "Championship", "match": m, "market": mk, "odds": o}
                         for ts, md, m, mk, o in rows])


def _settle(events=None, archives=None, now=NOW, journal=None):
    events = events if events is not None else [_ev()]
    return ur.settle_shadow_predictions(
        providers=[("test", lambda lg: events)], archives=archives or [], now=now,
        journal=journal)


def _read(f):
    return pd.read_csv(f, dtype=str, keep_default_na=False)


# ── grading ───────────────────────────────────────────────────────────────────────────────────
def test_unsettled_over_row_becomes_win(shadow):
    sc.append_shadow([_row(side="OVER")])
    _settle([_ev(hs=2, as_=1)])
    d = _read(shadow)
    assert d.loc[0, "result"] == "WIN" and d.loc[0, "outcome"] == "OVER"
    assert d.loc[0, "home_goals"] == "2" and d.loc[0, "away_goals"] == "1"


def test_unsettled_over_row_becomes_loss(shadow):
    sc.append_shadow([_row(side="OVER")])
    _settle([_ev(hs=1, as_=1)])
    assert _read(shadow).loc[0, "result"] == "LOSS"


def test_under_grading(shadow):
    sc.append_shadow([_row("a", side="UNDER", home="Leeds United", away="Norwich City"),
                      _row("b", side="UNDER", home="Hull City", away="Stoke City")])
    _settle([_ev(hs=1, as_=1), _ev("Hull", "Stoke", 3, 0)])
    d = _read(shadow).set_index("fixture_id")
    assert d.loc["a", "result"] == "WIN" and d.loc["a", "outcome"] == "UNDER"
    assert d.loc["b", "result"] == "LOSS" and d.loc["b", "outcome"] == "OVER"


def test_over_grading_at_exact_boundary():
    assert ss.grade("ou25", "OVER", 2, 1) == ("OVER", "WIN")
    assert ss.grade("ou25", "OVER", 1, 1) == ("UNDER", "LOSS")
    assert ss.grade("ou25", "UNDER", 0, 2) == ("UNDER", "WIN")


def test_side_markets_grade_cleanly():
    assert ss.grade("btts", "YES", 1, 1) == ("YES", "WIN")
    assert ss.grade("btts", "NO", 2, 0) == ("NO", "WIN")
    assert ss.grade("btts", "BTTS_YES".replace("BTTS_", ""), 0, 0) == ("NO", "LOSS")
    assert ss.grade("over15", "OVER", 1, 1) == ("OVER", "WIN")
    assert ss.grade("over35", "OVER", 2, 1) == ("UNDER", "LOSS")
    assert ss.grade("ou35", "UNDER", 2, 1) == ("UNDER", "WIN")
    assert ss.grade("corners", "OVER", 9, 9) == (None, None)


def test_btts_row_settles_with_its_own_archive_key(shadow):
    sc.append_shadow([_row(market="btts", side="YES", entry=1.80)])
    arch = _arch([("2026-10-18T13:00:00", "2026-10-18", "Leeds United vs Norwich City",
                   "btts_yes", 1.70)])
    _settle([_ev(hs=1, as_=1)], archives=[("a", arch)])
    d = _read(shadow)
    # the price chosen is still the BTTS-Yes archive price; >30 min before kickoff it is not a close
    assert d.loc[0, "result"] == "WIN" and float(d.loc[0, "last_pre_kickoff_odds"]) == 1.70


# ── eligibility ───────────────────────────────────────────────────────────────────────────────
def test_future_fixture_gets_no_result(shadow):
    sc.append_shadow([_row(ko="2026-10-25 14:00:00", logged="2026-10-19T09:00:00+00:00")])
    out = _settle([_ev(date="2026-10-25")])
    assert _read(shadow).loc[0, "result"] == ""
    assert out["pending"].get("NOT_FINISHED") == 1


def test_in_progress_match_is_not_settled(shadow):
    sc.append_shadow([_row(ko="2026-10-20 11:00:00", logged="2026-10-19T09:00:00+00:00")])
    _settle([_ev(date="2026-10-20")])          # NOW is 12:00 — only 60 minutes after kickoff
    assert _read(shadow).loc[0, "result"] == ""


def test_a_row_logged_after_kickoff_is_never_settled(shadow):
    sc.append_shadow([_row(logged="2026-10-18T14:05:00+00:00")])
    out = _settle()
    assert _read(shadow).loc[0, "result"] == ""
    assert out["pending"].get("NOT_PROSPECTIVE") == 1


def test_no_result_found_leaves_the_row_open(shadow):
    sc.append_shadow([_row()])
    out = _settle([_ev("Hull", "Stoke")])
    assert _read(shadow).loc[0, "result"] == ""
    assert out["pending"].get("NO_RESULT_FOUND") == 1


# ── freeze ────────────────────────────────────────────────────────────────────────────────────
def _digest(path, cols):
    d = _read(path)
    return {c: hashlib.sha256("\x1f".join(d[c]).encode()).hexdigest() for c in cols
            if c in d.columns}


def test_prediction_time_fields_are_byte_identical_after_settlement(shadow):
    sc.append_shadow([_row("a"), _row("b", home="Hull City", away="Stoke City"),
                      _row("c", ko="2026-10-30 14:00:00")])
    before = _digest(shadow, sc.FROZEN)
    _settle([_ev(), _ev("Hull", "Stoke", 0, 0)])
    after = _digest(shadow, sc.FROZEN)
    assert before == after
    for c in ("p_v9", "p_market", "v9_side", "v9_tier", "v9_edge", "entry_odds",
              "production_sniper_gate", "threshold_regime_id", "v9_model_sha", "dataset_id"):
        assert c in sc.FROZEN


def test_apply_updates_refuses_to_touch_a_frozen_column(shadow):
    sc.append_shadow([_row()])
    d = _read(shadow)
    bad = [{"fixture_id": "evt1", "market": "ou25", "match_date": "2026-10-18",
            "fields": {"result": "WIN"}}]
    out, n = ss.apply_updates(d, bad)
    assert n == 1
    # a crafted update naming a frozen column is ignored, never written
    bad[0]["fields"] = {"p_v9": "0.99", "settle_status": "SETTLED"}
    out2, _ = ss.apply_updates(d, bad)
    assert out2.loc[0, "p_v9"] == d.loc[0, "p_v9"]


def test_grading_columns_are_still_never_written_at_prediction_time(shadow):
    p = pd.DataFrame({"_oa_event_id": ["e1"], "date": ["2026-10-30"], "league": ["Championship"],
                      "home_team": ["Leeds United"], "away_team": ["Norwich City"],
                      "p_over25": [0.55], "best_side": ["OVER"], "signal_tier": ["AVOID"],
                      "best_edge": [0.01], "odds_over25": [1.9], "odds_under25": [1.9]})
    pipeline._log_shadow(p)
    d = pd.read_csv(shadow)
    for c in sc.GRADING_ONLY:
        assert d[c].isna().all(), c
    assert d.loc[0, "home_team"] == "Leeds United", "identity must be logged at first sight"


# ── closing price ─────────────────────────────────────────────────────────────────────────────
def test_close_is_the_last_quote_before_kickoff(shadow):
    sc.append_shadow([_row(side="OVER", entry=2.00)])
    arch = _arch([
        ("2026-10-17T10:00:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 2.05),
        ("2026-10-18T13:40:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 1.90),
        ("2026-10-18T13:10:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 1.95),
    ])
    _settle(archives=[("standard_odds_history.csv", arch)])
    d = _read(shadow)
    assert float(d.loc[0, "closing_odds"]) == 1.90
    assert d.loc[0, "close_ts"] == "2026-10-18T13:40:00Z"
    assert float(d.loc[0, "close_minutes_to_kickoff"]) == 20.0
    assert d.loc[0, "close_band"] == "T-30m" and d.loc[0, "close_status"] == "OK"
    assert float(d.loc[0, "clv"]) == round((2.00 / 1.90 - 1) * 100, 2)


def test_a_post_kickoff_quote_is_rejected(shadow):
    sc.append_shadow([_row(side="OVER", entry=2.00)])
    arch = _arch([
        ("2026-10-18T12:00:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 1.95),
        ("2026-10-18T14:30:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 3.40),
    ])
    _settle(archives=[("a", arch)])
    assert float(_read(shadow).loc[0, "last_pre_kickoff_odds"]) == 1.95       # never the post-kickoff 3.40


def test_only_post_kickoff_quotes_means_no_close(shadow):
    sc.append_shadow([_row()])
    arch = _arch([("2026-10-18T14:30:00", "2026-10-18", "Leeds United vs Norwich City",
                   "over25", 3.40)])
    _settle(archives=[("a", arch)])
    d = _read(shadow)
    assert d.loc[0, "result"] == "WIN"
    assert d.loc[0, "closing_odds"] == "" and d.loc[0, "clv"] == ""
    assert d.loc[0, "close_status"] == "POST_KICKOFF_ONLY"


def test_missing_close_still_settles_the_result_with_nan_clv(shadow):
    sc.append_shadow([_row()])
    _settle(archives=[])
    d = pd.read_csv(shadow)
    assert d.loc[0, "result"] == "WIN"
    assert np.isnan(d.loc[0, "closing_odds"]) and np.isnan(d.loc[0, "clv"])
    assert d.loc[0, "close_status"] == "NO_CLOSE"


def test_the_close_never_comes_from_the_other_side(shadow):
    sc.append_shadow([_row(side="UNDER", entry=1.95)])
    arch = _arch([
        ("2026-10-18T13:00:00", "2026-10-18", "Leeds United vs Norwich City", "over25", 1.80),
        ("2026-10-18T12:00:00", "2026-10-18", "Leeds United vs Norwich City", "under25", 2.02),
    ])
    _settle([_ev(hs=0, as_=0)], archives=[("a", arch)])
    assert float(_read(shadow).loc[0, "last_pre_kickoff_odds"]) == 2.02


# ── idempotence ───────────────────────────────────────────────────────────────────────────────
def test_an_already_settled_row_is_not_regraded(shadow):
    sc.append_shadow([_row(result="LOSS", settle_status="SETTLED")])
    _settle([_ev(hs=3, as_=3)])
    assert _read(shadow).loc[0, "result"] == "LOSS"


def test_repeated_settlement_changes_nothing(shadow):
    sc.append_shadow([_row(), _row("b", home="Hull City", away="Stoke City")])
    _settle([_ev(), _ev("Hull", "Stoke", 1, 0)], now=NOW)
    first = shadow.read_bytes()
    out = _settle([_ev(), _ev("Hull", "Stoke", 1, 0)],
                  now=datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc))
    assert shadow.read_bytes() == first and out["settled"] == 0


def test_journal_reapply_onto_a_file_predict_has_appended_to(shadow, tmp_path):
    """The workflow's race: predict appends while this run settles. Re-applying the journal onto
    the newer file must keep predict's rows AND land the settlement."""
    sc.append_shadow([_row()])
    j = tmp_path / "journal.json"
    original = shadow.read_bytes()
    _settle(journal=str(j))
    shadow.write_bytes(original)                         # "reset to origin/main"
    sc.append_shadow([_row("new", ko="2026-10-30 14:00:00")])   # predict appended meanwhile
    assert ur.reapply_shadow_settlement(str(j)) == 1
    d = _read(shadow)
    assert list(d["fixture_id"]) == ["evt1", "new"]
    assert d.loc[0, "result"] == "WIN" and d.loc[1, "result"] == ""
    assert ur.reapply_shadow_settlement(str(j)) == 0     # idempotent


def test_untouched_rows_and_line_endings_survive_byte_for_byte(shadow):
    sc.append_shadow([_row(), _row("future", ko="2026-10-30 14:00:00")])
    raw = shadow.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    shadow.write_bytes(raw)                              # a CRLF working copy (autocrlf)
    future_line = [ln for ln in raw.split(b"\r\n") if ln.startswith(b"future")][0]
    _settle()
    out = shadow.read_bytes()
    assert b"\r\n" in out and b"\n" not in out.replace(b"\r\n", b"")
    assert any(ln.startswith(future_line) for ln in out.split(b"\r\n"))


# ── names ─────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ours,theirs", [
    ("Queens Park Rangers", "QPR"),
    ("1. FC Kaiserslautern", "Kaiserslautern"),
    ("Cádiz CF", "Cadiz"),
    ("Sheffield Wednesday", "Sheffield Weds"),
])
def test_team_name_variants_resolve(shadow, ours, theirs):
    sc.append_shadow([_row(home=ours, away="Norwich City")])
    _settle([_ev(theirs, "Norwich", 3, 1)])
    assert _read(shadow).loc[0, "result"] == "WIN"


def test_manchester_city_is_never_matched_to_united():
    ev = [_ev("Manchester United", "Leeds")]
    assert ss.match_event("Manchester City", "Leeds United", "2026-10-18", None, ev,
                          fallback_matcher=ur._names_match) is None


def test_two_candidate_events_are_refused_not_guessed():
    ev = [_ev("Real Madrid", "Getafe"), _ev("Real Sociedad", "Getafe")]
    assert ss.match_event("Real", "Getafe", "2026-10-18", None, ev,
                          fallback_matcher=ur._names_match) is None


def test_a_row_with_no_identity_is_resolved_by_oddsapi_event_id(shadow):
    sc.append_shadow([_row(fid="abc123", home="", away="")])
    _settle([_ev("Leeds United", "Norwich City", 2, 2, eid="abc123")])
    d = _read(shadow)
    assert d.loc[0, "result"] == "WIN"
    assert d.loc[0, "home_team"] == "Leeds United", "identity filled once from the event"


def test_oddsapi_is_not_asked_about_rows_older_than_its_window(shadow):
    sc.append_shadow([_row(ko="2026-10-10 14:00:00", logged="2026-10-09T09:00:00+00:00")])
    called = []
    ur.settle_shadow_predictions(
        providers=[("oddsapi", lambda lg: called.append(lg) or [], 3),
                   ("api-football", lambda lg: [_ev(date="2026-10-10")])],
        archives=[], now=NOW)
    assert called == [] and _read(shadow).loc[0, "result_source"] == "api-football"


# ── prospective window + workflow ─────────────────────────────────────────────────────────────
def test_prospective_window_sees_settled_rows(shadow):
    sc.append_shadow([_row()])
    assert len(sc.prospective_window("2026-10-01")) == 0
    _settle()
    w = sc.prospective_window("2026-10-01")
    assert len(w) == 1 and w.loc[0, "result"] == "WIN"


def test_end_to_end_prediction_to_prospective_window(shadow):
    """predict writes -> a result appears -> update_results settles -> the forward test sees it."""
    p = pd.DataFrame({"_oa_event_id": ["e77"], "date": ["2026-10-18"],
                      "kickoff_utc": ["2026-10-18 14:00:00"], "league": ["Championship"],
                      "home_team": ["Queens Park Rangers"], "away_team": ["Norwich City"],
                      "model_type": ["standard"], "p_over25": [0.58], "best_side": ["OVER"],
                      "signal_tier": ["VALUABLE"], "best_edge": [0.04],
                      "odds_over25": [2.02], "odds_under25": [1.85], "impl_prob_over": [0.49]})
    pipeline._log_shadow(p)
    # _log_shadow stamps logged_at=now; pin it before kickoff for a deterministic test
    d = pd.read_csv(shadow, dtype=str, keep_default_na=False)
    d["logged_at"] = "2026-10-17T08:00:00+00:00"
    d.to_csv(shadow, index=False)
    arch = _arch([("2026-10-18T13:45:00", "2026-10-18", "Queens Park Rangers vs Norwich City",
                   "over25", 1.96)])
    _settle([_ev("QPR", "Norwich", 2, 1)], archives=[("standard_odds_history.csv", arch)])
    w = sc.prospective_window("2026-10-04")
    assert len(w) == 1
    assert w.loc[0, "result"] == "WIN"
    assert w.loc[0, "closing_odds"] == pytest.approx(1.96)
    assert w.loc[0, "clv"] == pytest.approx(round((2.02 / 1.96 - 1) * 100, 2))


def test_health_artifact_is_written(shadow):
    import json
    sc.append_shadow([_row(), _row("f", ko="2026-10-30 14:00:00")])
    _settle()
    h = json.loads(ur.SHADOW_HEALTH_FILE.read_text(encoding="utf-8"))
    assert h["shadow_rows"] == 2 and h["settled_rows"] == 1
    assert h["settlement_pct_of_finished"] == 100.0 and h["last_settlement_ts"]


def _wf():
    import yaml
    p = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "update_results.yml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_workflow_commits_the_shadow_log_per_file():
    steps = _wf()["jobs"]["update"]["steps"]
    shadow_steps = [s for s in steps if "output/v91_shadow.csv" in str(s.get("run", ""))
                    and "git add" in str(s.get("run", ""))]
    assert shadow_steps, "update_results.yml must stage output/v91_shadow.csv"
    run = shadow_steps[-1]["run"]
    assert shadow_steps[-1].get("if") == "always()"
    assert "git add -A" not in run and "--shadow-reapply" in run and "git push" in run


def test_workflow_passes_the_journal_to_the_settling_run():
    steps = _wf()["jobs"]["update"]["steps"]
    upd = next(s for s in steps if "update_results.py --days" in str(s.get("run", "")))
    assert "SHADOW_SETTLEMENT_JOURNAL" in upd.get("env", {})


def test_settlement_runs_last_and_guarded_in_main():
    import inspect
    src = inspect.getsource(ur.main)
    assert src.rfind("settle_shadow_predictions") > src.rfind("update_live_signal_results")


# ── owner rule 2026-10-22: a close counts only within 30 minutes of kickoff ─────────────────────
def test_close_more_than_30_min_before_kickoff_is_not_a_close(shadow):
    sc.append_shadow([_row(side="OVER", entry=2.00)])                       # kickoff 14:00
    _settle(archives=[("a", _arch([("2026-10-18T12:00:00", "2026-10-18", "Leeds United vs Norwich City",
                                    "over25", 1.95)]))])
    d = _read(shadow)
    assert d.loc[0, "result"] in ("WIN", "LOSS")                             # result still settles
    assert d.loc[0, "closing_odds"] == "" and d.loc[0, "clv"] == ""
    assert d.loc[0, "close_status"] == "STALE_CLOSE_GT_30MIN"
    assert float(d.loc[0, "last_pre_kickoff_odds"]) == 1.95


def test_close_within_30_min_counts(shadow):
    sc.append_shadow([_row(side="OVER", entry=2.00)])
    _settle(archives=[("a", _arch([("2026-10-18T13:45:00", "2026-10-18", "Leeds United vs Norwich City",
                                    "over25", 1.90)]))])
    d = _read(shadow)
    assert float(d.loc[0, "closing_odds"]) == 1.90 and d.loc[0, "close_status"] == "OK"
    assert abs(float(d.loc[0, "clv"]) - (2.00 / 1.90 - 1) * 100) < 0.01


def test_rule_is_applied_to_rows_settled_before_it_existed_and_is_idempotent():
    import pandas as pd
    from src.shadow_settlement import enforce_close_rule, frozen_digest
    cols = list(sc.COLUMNS)
    row = {c: "" for c in cols}
    row.update({"fixture_id": "x", "market": "ou25", "match_date": "2026-10-18", "p_v9": "0.55",
                "result": "WIN", "closing_odds": "1.95", "clv": "2.56", "close_status": "OK",
                "close_minutes_to_kickoff": "319.0"})
    ok = dict(row, fixture_id="y", close_minutes_to_kickoff="12.0")
    df = pd.DataFrame([row, ok])[cols].astype(str)
    d0 = frozen_digest(df.fillna(""))
    out, n = enforce_close_rule(df)
    assert n == 1 and out.loc[0, "closing_odds"] == "" and out.loc[0, "last_pre_kickoff_odds"] == "1.95"
    assert out.loc[1, "closing_odds"] == "1.95"                              # a near-kickoff close stays
    assert frozen_digest(out) == d0
    out2, n2 = enforce_close_rule(out)
    assert n2 == 0 and out2.equals(out)
