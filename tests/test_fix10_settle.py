"""Fix round 10 'settle' (owner decisions 2026-11-04):

  1. prop settlement reads the provider player sheet with THE shared null rule
     (player_model.player_stat_nulls); the team 'Shots on Goal' read uses src.af_team_stats; a result
     the rule decided is recorded beside the ledger; the event feed cross-checks goals/assists/cards;
  2. national-team seasons: the provider's league.season, recorded beside the history;
  3. the status re-read covers every registry league in the collection workflow;
  4. the props predict job counts its median fills per feature per run (serving unchanged);
  5. 'Total passes' / 'Passes accurate' stay missing when null, even in an answered block.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_fix10_settle.py -q

No network: every provider answer is a fake; every output goes to tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import update_results as U  # noqa: E402
from src import af_team_stats as AFTS  # noqa: E402


# ── 1. settlement under the shared null rule ─────────────────────────────────────────────────────

def _pe(pid, minutes, **groups):
    st = {"games": {"minutes": minutes, "substitute": False}}
    st.update(groups)
    return {"player": {"id": pid, "name": f"P{pid}"}, "statistics": [st]}


def _fake_provider(monkeypatch, sheet, events=(), status="FT", team_stats=None):
    fixture = {"fixture": {"id": 900, "date": "2026-10-24T15:00:00+00:00", "status": {"short": status}},
               "goals": {"home": 1, "away": 0}, "score": {"fulltime": {"home": 1, "away": 0}}}
    calls = []

    def _af(ep, params, hours):
        calls.append(ep)
        if ep == "/fixtures/players":
            return sheet
        if ep == "/fixtures/events":
            return list(events)
        if ep == "/fixtures/statistics":
            return team_stats or []
        if ep == "/fixtures":
            return [fixture]
        raise AssertionError(ep)
    import src.league_registry as LR
    monkeypatch.setattr(LR, "prop_leagues", lambda *a, **k: {"Championship": 40})
    monkeypatch.setattr(U, "_af_response", _af)
    return calls


def _ledger(*rows):
    base = {"signal_date": "2026-10-23", "match_date": "2026-10-24", "league": "Championship",
            "home_team": "Cardiff City", "away_team": "Hull City", "tier": "AVOID",
            "market_odds": "2.5", "result": "", "pnl": "", "fixture_id": "900"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _sheet():
    return [{"team": {"id": 1}, "players": [
                _pe(7, 90, shots={"on": None}, goals={"total": None, "assists": None}, cards={"yellow": None, "red": None}),
                _pe(8, 90, shots={"on": 2}, goals={"total": 1, "assists": 0}, cards={"yellow": 1, "red": 0}),
                _pe(6, 0, shots={"on": None}, goals={"total": None})]},
            {"team": {"id": 2}, "players": [_pe(9, 90, shots={"on": 0})]}]


def test_null_count_of_a_player_who_played_settles_by_the_rule(monkeypatch):
    _fake_provider(monkeypatch, _sheet())
    led = _ledger({"player_name": "P7", "player_id": "7", "market": "sot"},
                  {"player_name": "P7", "player_id": "7", "market": "goals"},
                  {"player_name": "P7", "player_id": "7", "market": "cards"},
                  {"player_name": "P8", "player_id": "8", "market": "sot2"},
                  {"player_name": "P6", "player_id": "6", "market": "sot"})
    updates, _, skipped = U._grade_player_rows(led, "2026-11-04")
    by = {u["idx"]: u for u in updates}
    assert by[0]["result"] == "LOSS" and by[0]["rule_zero_stats"] == ["shots_on_target"]
    assert by[1]["result"] == "LOSS" and by[1]["rule_zero_stats"] == ["goals"]
    assert by[2]["result"] == "LOSS" and by[2]["rule_zero_stats"] == ["yellow_card", "red_card"]
    assert by[3]["result"] == "WIN" and by[3]["rule_zero_stats"] == []          # a sent value decides
    assert by[4]["result"] == "VOID" and "0 min" in by[4]["notes"]             # minutes 0: no rule zero
    assert not skipped


def test_an_unanswered_fixture_stays_pending(monkeypatch):
    sheet = [{"team": {"id": 1}, "players": [{"player": {"id": 7}, "statistics": []}]}]
    _fake_provider(monkeypatch, sheet)
    players = U._fixture_players(900)
    assert players["answered"] is False and players["by_id"][7]["shots_on_target"] is None
    updates, _, skipped = U._grade_player_rows(_ledger({"player_name": "P7", "player_id": "7", "market": "sot"}),
                                               "2026-11-04")
    assert not updates and skipped == {"appearance_unknown": 1}


def test_event_feed_naming_the_player_is_kept(monkeypatch):
    # COMPLETE feed (1 goal event, final score 1-0) names P7: the feed decides, as before fix 10
    ev = [{"type": "Goal", "detail": "Normal Goal", "time": {"elapsed": 30}, "player": {"id": 7},
           "assist": {"id": None}},
          {"type": "Card", "detail": "Yellow Card", "time": {"elapsed": 50}, "player": {"id": 7}}]
    _fake_provider(monkeypatch, _sheet(), events=ev)
    led = _ledger({"player_name": "P7", "player_id": "7", "market": "goals"},
                  {"player_name": "P7", "player_id": "7", "market": "sot"},
                  {"player_name": "P7", "player_id": "7", "market": "cards"},
                  {"player_name": "P7", "player_id": "7", "market": "assists"})
    updates, _, skipped = U._grade_player_rows(led, "2026-11-04")
    by = {u["idx"]: u for u in updates}
    assert by[0]["result"] == "WIN" and by[0]["rule_zero_stats"] == []          # from the feed
    assert by[1]["result"] == "LOSS" and by[1]["rule_zero_stats"] == ["shots_on_target"]
    assert by[2]["result"] == "WIN" and by[2]["rule_zero_stats"] == []          # yellow in the feed
    assert by[3]["result"] == "LOSS" and by[3]["rule_zero_stats"] == ["assists"]   # feed silent
    assert not skipped


def test_incomplete_feed_naming_the_player_keeps_the_row_pending(monkeypatch):
    # 2 goal events against a 1-0 score: not complete; it names P7 -> sheet and feed disagree
    ev = [{"type": "Goal", "detail": "Normal Goal", "time": {"elapsed": 30}, "player": {"id": 7},
           "assist": {"id": None}},
          {"type": "Goal", "detail": "Normal Goal", "time": {"elapsed": 60}, "player": {"id": 8},
           "assist": {"id": None}}]
    _fake_provider(monkeypatch, _sheet(), events=ev)
    updates, _, skipped = U._grade_player_rows(
        _ledger({"player_name": "P7", "player_id": "7", "market": "goals"},
                {"player_name": "P7", "player_id": "7", "market": "sot"}), "2026-11-04")
    assert skipped == {"rule_zero_contradicted_by_events": 1}
    assert [u["result"] for u in updates] == ["LOSS"]                            # sot has no event feed


def test_extra_time_bound_uses_the_rule(monkeypatch):
    # AET: goals / cards come from the regulation event feed; the sheet's shots-on-target (rule 0)
    # is the whole-match upper bound -> LOSS, recorded as a rule decision.
    ev = [{"type": "Goal", "detail": "Normal Goal", "time": {"elapsed": 20}, "player": {"id": 8},
           "assist": {"id": None}}]
    _fake_provider(monkeypatch, _sheet(), events=ev, status="AET")
    updates, _, skipped = U._grade_player_rows(
        _ledger({"player_name": "P7", "player_id": "7", "market": "sot"},
                {"player_name": "P7", "player_id": "7", "market": "goals"}), "2026-11-04")
    by = {u["idx"]: u for u in updates}
    assert by[0]["result"] == "LOSS" and by[0]["rule_zero_stats"] == ["shots_on_target"]
    assert by[1]["result"] == "LOSS" and by[1]["rule_zero_stats"] == []        # decided by the feed


def test_team_shots_on_goal_read_with_the_team_rule(monkeypatch):
    # the residual proof path (a stat the rule left None) reads 'Shots on Goal' via src.af_team_stats:
    # a null in an answered block is 0, an unanswered block is missing
    rec = {"player_id": 7, "team_id": 1, "goals": 0, "assists": 0, "yellow_card": 0, "red_card": 0,
           "shots_on_target": None}
    players = {"teams": {1: [rec, {"shots_on_target": 0}]}}
    answered = [{"team": {"id": 1}, "statistics": [{"type": "Shots on Goal", "value": None},
                                                     {"type": "Total Shots", "value": 4}]}]
    _fake_provider(monkeypatch, [], team_stats=answered)
    assert U._fill_null_stats(rec, 900, {}, players)["shots_on_target"] == 0
    unanswered = [{"team": {"id": 1}, "statistics": [{"type": "Shots on Goal", "value": None}]}]
    _fake_provider(monkeypatch, [], team_stats=unanswered)
    assert U._fill_null_stats(rec, 900, {}, players)["shots_on_target"] is None


def test_settler_reads_with_the_shared_helper_only():
    src = (ROOT / "update_results.py").read_text(encoding="utf-8")
    body = src[src.index("def _fixture_players"):src.index("def _goal_events_complete")]
    assert "PSN.player_counts(" in body and "PSN.fixture_answered(" in body
    assert 'shots.get("on")' not in body and 'goals.get("total")' not in body   # no private reading
    fill = src[src.index("def _fill_null_stats"):src.index("def _appearance")]
    assert 'AFTS.team_count(' in fill and 's.get("value")' not in fill


def test_null_rule_settlements_are_recorded_beside_the_ledger(tmp_path):
    led = _ledger({"player_name": "P7", "player_id": "7", "market": "sot"},
                  {"player_name": "P8", "player_id": "8", "market": "sot2"})
    ups = [{"idx": 0, "result": "LOSS", "pnl": -1.0, "resolved_date": "2026-11-04", "minutes": 90,
            "rule_zero_stats": ["shots_on_target"], "fixture_id": 900},
           {"idx": 1, "result": "WIN", "pnl": 1.5, "resolved_date": "2026-11-04", "minutes": 90,
            "rule_zero_stats": [], "fixture_id": 900}]
    dst = tmp_path / "rec.csv"
    assert U.record_null_rule_settlements("output/player_ledger.csv", led, ups, dst) == 1
    assert U.record_null_rule_settlements("output/player_ledger.csv", led, ups, dst) == 0   # no duplicate
    rec = pd.read_csv(dst, dtype=str, keep_default_na=False)
    assert list(rec.columns) == U.NULL_RULE_SETTLED_COLS and len(rec) == 1
    r = rec.iloc[0]
    assert (r["result_before"], r["result_after"], r["pnl_after"]) == ("", "LOSS", "-1.0")
    assert r["rule_zero_stats"] == "shots_on_target" and "null rule" in r["reason"]


def test_ci_settlement_reaches_every_pending_past_due_row():
    # update_player_results takes EVERY pending row dated before today (no --days window), so the
    # next CI run re-grades all pending past-due rows under the rule; the record is committed.
    src = (ROOT / "update_results.py").read_text(encoding="utf-8")
    body = src[src.index("def update_player_results"):src.index("def grade_restored_player_sidecar")]
    assert 'ledger["match_date"] < today_str' in body and "days" not in body.split("pending_mask")[1][:200]
    assert "record_null_rule_settlements(" in body
    for wf in ("update_results.yml", "daily_summary.yml"):
        y = (ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
        assert "output/corrections/player_ledger_null_rule_settled.csv" in y
    assert "player_ledger_null_rule_settled.csv merge=union" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


# ── 2. national-team seasons ─────────────────────────────────────────────────────────────────────

def test_provider_seasons_feed_the_profile_keys(tmp_path):
    from player_model import data_fetcher as DF
    import collect_player_profiles as PROF
    dst = tmp_path / "seasons.csv"
    assert DF.record_national_seasons([{"fixture_id": 3, "league": "Friendlies", "date": "2026-03-31",
                                        "season": "2026"},
                                       {"fixture_id": 4, "league": "Friendlies", "date": "2026-03-31",
                                        "season": None}], dst) == 1                  # no season: not recorded
    assert PROF.provider_seasons(dst) == {3: "2026"}
    h = pd.DataFrame([{"fixture_id": 3, "player_id": 12, "season": "", "league": "Friendlies", "date": "2026-03-31"},
                      {"fixture_id": 4, "player_id": 13, "season": "", "league": "Friendlies", "date": "2026-03-31"}])
    kf = PROF.key_frame(h, PROF.provider_seasons(dst))
    assert kf["season_key"].tolist() == ["2026", None]
    assert kf["season_source"].tolist() == ["provider_fixture", None]
    assert kf["problem"].tolist() == [None, "season_unresolved"]


def test_workflows_store_and_commit_the_provider_seasons():
    props = (ROOT / ".github" / "workflows" / "player_props.yml").read_text(encoding="utf-8")
    assert "output/collect/national_fixture_seasons.csv" in props
    dt = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    assert "--invalid-keys-out output/corrections/player_history_invalid_profile_keys.csv" in dt
    assert "git add -f output/corrections/player_history_invalid_profile_keys.csv" in dt


# ── 3. status re-read for all leagues ────────────────────────────────────────────────────────────

def test_status_reread_covers_all_leagues_in_the_collection_workflow():
    y = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    step = y[y.index("- name: 7. Half-time scores"):y.index("- name: 8.")]
    assert "collect_ht_scores.py --add-status --status-all-leagues" in step
    assert "APIFOOTBALL_KEY" in step
    src = (ROOT / "scripts" / "collect_ht_scores.py").read_text(encoding="utf-8")
    body = src[src.index("def add_status"):src.index("def run(")]
    # budget-guarded (AFClient, BudgetStop is a clean stop) and resumable (journal per league-season)
    assert "AFClient()" in body and "except BudgetStop" in body and "STATUS_JOURNAL" in body


# ── 4. props median-fill telemetry ───────────────────────────────────────────────────────────────

class _Const:
    """A NaN-native-free stand-in estimator: probability = mean of the row."""
    def predict_proba(self, X):
        p = np.clip(np.asarray(X, float).mean(axis=1) / 10.0, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def test_median_fill_is_counted_without_changing_serving(tmp_path):
    from player_model import model as M
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [2.0, 2.0, np.nan, 4.0],
                       "c": [np.nan] * 4})
    payload = {"market": "sot", "feature_cols": ["a", "b", "c", "d"], "models": {"lr": _Const()}}
    X_ref, _ = M._prep(df, payload["feature_cols"])
    M.median_fill_reset()
    p = M.predict_proba(df, payload)
    # serving identical to the bare legacy computation
    ref = pd.Series(M._ensemble_proba(payload["models"], X_ref), index=df.index)
    pd.testing.assert_series_equal(p, ref)
    rows = M.median_fill_rows("2026-11-04T00:00:00Z", "1")
    got = {r["feature"]: r["cells_filled"] for r in rows}
    assert got == {"a": 2, "b": 1}             # c / d have no value anywhere: left NaN, not a fill
    assert all(r["rows_scored"] == 4 and r["imputer"] == "batch_median_legacy" for r in rows)
    dst = tmp_path / "fill.csv"
    assert M.write_median_fill(dst, run_at="t1", run_id="1") == 2
    assert M.write_median_fill(dst, run_at="t2", run_id="2") == 2               # appended per run
    out = pd.read_csv(dst)
    assert list(out.columns) == M.MEDIAN_FILL_COLS and len(out) == 4
    M.median_fill_reset()
    assert M.median_fill_rows("t", "") == []


def test_predict_job_writes_and_commits_the_telemetry():
    src = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    body = src[src.index("def mode_predict"):]
    assert "median_fill_reset()" in body and "write_median_fill()" in body
    y = (ROOT / ".github" / "workflows" / "player_props.yml").read_text(encoding="utf-8")
    assert "output/props_median_fill.csv" in y


# ── 5. passes stay missing when null ─────────────────────────────────────────────────────────────

def test_pass_counts_stay_missing_even_in_an_answered_block():
    block = [{"type": "Shots on Goal", "value": 3}, {"type": "Red Cards", "value": None},
             {"type": "Total passes", "value": None}, {"type": "Passes accurate", "value": None},
             {"type": "Passes %", "value": None}]
    assert AFTS.answered(block)
    v = AFTS.team_values(block)
    assert v["Red Cards"] == 0.0                                    # an ordinary count: null -> 0
    assert v["Total passes"] is None and v["Passes accurate"] is None   # "not tracked": missing
    assert AFTS.team_count(block, "Total passes") is None
    # a SENT value is kept, a sent 0 included
    sent = [{"type": "Total passes", "value": 0}, {"type": "Passes accurate", "value": 401}]
    assert AFTS.team_values(sent) == {"Total passes": 0.0, "Passes accurate": 401.0}
    assert AFTS.PASS_COUNTS == {"Total passes", "Passes accurate"}
    assert not (AFTS.PASS_COUNTS & AFTS.COUNT_TYPES)
    doc = " ".join(AFTS.__doc__.split())
    assert "NEVER zero-filled" in doc and "not tracked" in doc and "2026-11-04" in doc
