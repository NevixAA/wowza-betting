"""Post-collection step 1 (2026-11-07): scripts/collection_complete.py - is the real-data collection
complete? - plus the sidelined invalid-key fix and the re-fetch proof (player_model.history_refetch).

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_collection_complete.py -q

Synthetic data only: no network, every file in tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_af_lists as AL  # noqa: E402
import collect_lineups as LU  # noqa: E402
import collect_sidelined as SD  # noqa: E402
import collection_complete as CCX  # noqa: E402
from player_model import history_refetch as HR  # noqa: E402
from player_model import player_stat_nulls as PSN  # noqa: E402

REG = 40


# ── the new-match rule and the verdict ─────────────────────────────────────────────────────────

def test_split_new_steady_vs_backlog():
    last = pd.Timestamp("2026-11-07 04:00")
    s, b = CCX.split_new(["2026-11-07", "2026-11-06", "2026-11-05", "2019-09-15"], last)
    assert (s, b) == (2, 2)                        # from the day before the last fetch: new matches


def test_split_new_without_a_journal_is_all_backlog():
    assert CCX.split_new(["2026-11-07"], None) == (0, 1)


def test_last_fetch_reads_the_newest_utc_time():
    j = pd.DataFrame({"fetched_at": ["2026-11-06T10:00:00+00:00", "2026-11-07T03:58:24+00:00", None]})
    assert CCX.last_fetch(j) == pd.Timestamp("2026-11-07 03:58:24")
    assert CCX.last_fetch(pd.DataFrame()) is None


def _r(name, backlog, steady):
    return CCX.result(name, backlog=backlog, steady=steady, recurring=True)


def test_verdict_complete_only_without_backlog_and_with_clean_audits():
    ok_audit = [{"file": "a.json", "check": "adds up", "ok": True, "detail": ""}]
    assert CCX.verdict([_r("lineups", 0, 12), _r("fixtures", 0, 101)], ok_audit)["complete"]
    v = CCX.verdict([_r("lineups", 5, 0), _r("playerrefetch", 31384, 0)], ok_audit)
    assert not v["complete"] and v["backlog_calls"] == 31389
    assert v["backlog_by_collector"] == {"lineups": 5, "playerrefetch": 31384}
    bad = [{"file": "a.json", "check": "adds up", "ok": False, "detail": ""}]
    assert not CCX.verdict([_r("lineups", 0, 0)], bad)["complete"]
    err = [_r("lineups", 0, 0) | {"error": "KeyError"}]
    assert not CCX.verdict(err, ok_audit)["complete"]


def test_result_counts_calls_left_as_backlog_plus_steady():
    r = CCX.result("x", backlog=3, steady=4, recurring=False, unreachable={"a": 2, "b": 0})
    assert r["calls_left"] == 7 and r["unreachable"] == {"a": 2}


# ── audit consistency on synthetic audits ──────────────────────────────────────────────────────

def _ht_audit(d: Path, lacking=10, filled=7, unmatched=2, mism=1, fill_rows=7, csv_rows=1):
    (d / "ht_fill_audit.json").write_text(json.dumps({
        "refused_ft_mismatch": mism, "filled_rows_total": fill_rows,
        "per_league": {"L": {"lacking": lacking, "filled": filled, "unmatched": unmatched,
                             "refused_ft_mismatch": mism}}}), encoding="utf-8")
    pd.DataFrame({"x": range(csv_rows)}).to_csv(d / "ht_fill_ft_mismatch.csv", index=False)
    pd.DataFrame({"x": range(fill_rows)}).to_parquet(d / "fd_history_ht_fill.parquet")


def test_audit_checks_pass_on_a_consistent_ht_audit(tmp_path):
    _ht_audit(tmp_path)
    out = CCX.audit_checks(tmp_path)
    assert out and all(a["ok"] for a in out)


def test_audit_checks_flag_counts_that_do_not_add_up(tmp_path):
    _ht_audit(tmp_path, lacking=11)                        # one history row unaccounted for
    bad = [a for a in CCX.audit_checks(tmp_path) if not a["ok"]]
    assert len(bad) == 1 and "lacking = filled" in bad[0]["check"]


def test_audit_checks_flag_an_audit_that_disagrees_with_its_file(tmp_path):
    _ht_audit(tmp_path, csv_rows=2)
    bad = [a["check"] for a in CCX.audit_checks(tmp_path) if not a["ok"]]
    assert any("ht_fill_ft_mismatch.csv" in c for c in bad)


def test_budget_override_day_is_reported_not_failed(tmp_path):
    (tmp_path / "af_budget.json").write_text(json.dumps({"2026-11-02": 53590, "2026-11-03": 45000}))
    out = CCX.audit_checks(tmp_path)
    assert all(a["ok"] for a in out)
    assert any("override" in a["check"] and a["detail"] == {"2026-11-02": 53590} for a in out)
    (tmp_path / "af_budget.json").write_text(json.dumps({"2026-11-02": 60001}))
    assert not all(a["ok"] for a in CCX.audit_checks(tmp_path))


def _refetch_files(d: Path, refused=1, n_rows_journal=2, audit_refused=1):
    rf = pd.DataFrame({"fixture_id": [1, 1], "player_id": [10, 11], "source": ["history", "history"],
                       "minutes_refetch": [90, 0]})
    rf.to_parquet(d / "player_history_refetch.parquet")
    pd.DataFrame({"fixture_id": [1], "status": ["ok"], "n_rows": [n_rows_journal], "n_refused": [refused]}) \
        .to_parquet(d / "player_refetch_fixtures.parquet")
    (d / "player_history_refetch_audit.json").write_text(json.dumps({
        "plan": {"rows": 5, "rows_by_source": {"history": 5}, "rows_verified_at_parse": 2, "rows_to_verify": 3,
                 "fixtures": 2, "proven_fixtures": 1, "refused_fixtures": {"fixture_not_in_lists": 1},
                 "journaled_fixtures": 0, "planned_calls": 1},
        "run": {"rows_stored": 2, "rows_refused": {"side_unproven": 1}},
        "stored_rows": 2, "stored_rows_by_source": {"history": 2}, "journal_fixtures": 1,
        "journal_by_status": {"ok": 1}, "rows_refused_in_answers": audit_refused,
        "fields": {"goals": {"rows": 1, "sent_value": 0, "sent_zero": 1, "null_read_as_zero": 0, "missing": 0,
                             "orig_zero_now_zero": 1, "orig_zero_now_missing": 0, "orig_zero_now_value": 0}}}))


def test_refetch_audit_consistent(tmp_path):
    _refetch_files(tmp_path)
    out = [a for a in CCX.audit_checks(tmp_path) if a["file"] == "player_history_refetch_audit.json"]
    assert len(out) == 6 and all(a["ok"] for a in out), [a for a in out if not a["ok"]]


def test_refetch_audit_journal_disagreeing_with_the_sidecar_fails(tmp_path):
    _refetch_files(tmp_path, n_rows_journal=3, audit_refused=2)
    bad = [a["check"] for a in CCX.audit_checks(tmp_path) if not a["ok"]]
    assert any(c.startswith("journal:") for c in bad)


def test_lineups_journal_against_partitions(tmp_path):
    pd.DataFrame({"fixture_id": [1, 2], "n_teams": [2, 0]}).to_parquet(tmp_path / "lineups_fixtures.parquet")
    (tmp_path / "lineups").mkdir()
    pd.DataFrame({"fixture_id": [1, 1], "team_id": [5, 6]}).to_parquet(tmp_path / "lineups" / "teams_40_2025.parquet")
    assert all(a["ok"] for a in CCX.audit_checks(tmp_path))
    pd.DataFrame({"fixture_id": [3], "team_id": [5]}).to_parquet(tmp_path / "lineups" / "teams_40_2024.parquet")
    assert not all(a["ok"] for a in CCX.audit_checks(tmp_path))


# ── a collector check on synthetic files (lineups: the collector's own plan) ───────────────────

@pytest.fixture()
def lineups_env(tmp_path, monkeypatch):
    cdir = tmp_path / "collect"
    cdir.mkdir()
    monkeypatch.setattr(AL, "COVERAGE", cdir / "af_league_coverage.parquet")
    monkeypatch.setattr(AL, "registry_af_ids", lambda: {REG})
    monkeypatch.setattr(LU, "JOURNAL", cdir / "lineups_fixtures.parquet")
    fx = pd.DataFrame([{"fixture_id": f, "league_id": REG, "season": s, "date": d, "home_id": 1, "away_id": 2,
                        "status": "FT"} for f, s, d in ((1, "2019", "2019-09-15"), (2, "2026", "2026-11-01"),
                                                       (3, "2026", "2026-11-06"), (4, "2026", "2026-11-07"),
                                                       (5, "2018", "2018-08-01"))])
    fxp = tmp_path / "competition_fixtures.parquet"
    fx.to_parquet(fxp)
    monkeypatch.setattr(AL, "FIXTURES", fxp)
    pd.DataFrame([{"league_id": REG, "season": s, "lineups": v, "fetched_at": "2026-11-03T00:00:00+00:00"}
                  for s, v in (("2019", True), ("2026", True), ("2018", False))]).to_parquet(AL.COVERAGE)
    pd.DataFrame([{"fixture_id": 2, "league_id": REG, "season": "2026", "n_teams": 0,
                   "fetched_at": "2026-11-07T03:58:00+00:00"}]).to_parquet(LU.JOURNAL)
    return tmp_path


def test_check_lineups_splits_backlog_steady_and_unreachable(lineups_env):
    r = CCX.check_lineups()
    # fixture 1 (2019, failed call) is backlog; 3 and 4 are new matches; 5 is a season without lineups
    assert (r["backlog"], r["steady_state"]) == (1, 2)
    assert r["unreachable"] == {"fixtures_season_without_provider_lineups": 1, "fixtures_provider_answer_empty": 1}
    assert r["recurring"] and r["detail"]["to_fetch_oldest"] == "2019-09-15"


# ── sidelined: an invalid player id is never sent (post-collection fix 2026-11-07) ─────────────

def test_sidelined_never_plans_player_id_zero():
    h = pd.DataFrame({"player_id": [0, 0, 7, 8, None], "date": ["2026-10-01"] * 5})
    have = pd.DataFrame({"player_id": [8], "n_spells": [0], "fetched_at": ["2026-11-02T00:00:00+00:00"]})
    assert SD.todo_players(h, have) == [7]
    assert SD.invalid_rows(h) == 3


# ── the re-fetch proof ──────────────────────────────────────────────────────────────────────────

def _before():
    # fixture 1: two re-fetched rows + one refused row; fixture 2: not journaled; row 5 parsed with the rule
    rows = [(1, 10, 0, 0, 0, 0, None), (1, 11, 0, 1, 2, 1, None), (1, 12, 0, 0, 0, 0, None),
            (2, 20, 1, 1, 1, 0, None), (3, 30, 0, 0, 0, 1, PSN.PARSED_WITH_RULE)]
    df = pd.DataFrame(rows, columns=["fixture_id", "player_id", "goals", "shots_on_target", "shots_total",
                                     "yellow_cards", PSN.SOURCE_COL])
    for c in HR.REPLACED:
        if c not in df.columns:
            df[c] = 0
    return df


def _refetch(before):
    rf = pd.DataFrame({"fixture_id": [1, 1], "player_id": [10, 11], "minutes_refetch": [90, 0]})
    new = {"goals": [1, None], "shots_on_target": [2, None], "shots_total": [3, None], "yellow_cards": [0, None]}
    for c in HR.REPLACED:
        rf[c] = new.get(c, [0, None])
        rf[f"orig_{c}"] = before.set_index(["fixture_id", "player_id"]).loc[[(1, 10), (1, 11)], c].to_numpy()
    return rf


def test_refetch_proof_counts_and_checks():
    b = _before()
    p = HR.refetch_proof(b, _refetch(b), journal=pd.DataFrame({"fixture_id": [1], "n_refused": [1]}))
    assert p["ok"] and p["rows_replaced"] == 2
    g = p["fields"]["goals"]
    assert (g["old0_now_value"], g["old0_now_missing"], g["applied_exactly"], g["orig_kept_equal_to_stored"]) \
        == (1, 1, 2, 2)
    sot = p["markets"]["sot"]
    assert (sot["0_to_1"], sot["label_now_missing"], sot["positives_before"], sot["positives_after"]) == (1, 1, 1, 1)
    assert p["markets"]["sot2"]["0_to_1"] == 1
    j = p["journaled_fixtures"]
    assert (j["rows_to_verify"], j["rows_replaced"], j["rows_refused"]) == (3, 2, 1)
    assert j["refused_rows_count_source"] == {PSN.UNVERIFIED: 1} and j["refused_rows_values_changed"] == 0
    assert p["not_refetched_rows"]["count_source"] == {PSN.UNVERIFIED: 2, PSN.PARSED_WITH_RULE: 1}
    assert not any(p["not_refetched_rows"]["values_changed"].values())


def test_refetch_proof_fails_when_the_original_is_not_the_stored_value():
    b = _before()
    rf = _refetch(b)
    rf.loc[0, "orig_goals"] = 5                            # a sidecar that does not hold the stored value
    assert not HR.refetch_proof(b, rf)["ok"]


def test_refetch_proof_fails_when_a_sidecar_row_has_no_history_row():
    b = _before()
    rf = pd.concat([_refetch(b), _refetch(b).assign(fixture_id=99)], ignore_index=True)
    p = HR.refetch_proof(b, rf)
    assert p["refetch_rows_without_history_row"] == 2 and not p["ok"]


def test_refetch_proof_never_overrides_a_row_parsed_with_the_rule():
    b = _before()
    rf = _refetch(b)
    extra = rf.iloc[[0]].assign(fixture_id=3, player_id=30)
    p = HR.refetch_proof(b, pd.concat([rf, extra], ignore_index=True))
    assert p["rows_replaced"] == 2 and p["count_source_after"][PSN.PARSED_WITH_RULE] == 1


# ── CLIs refuse to write under output/ ─────────────────────────────────────────────────────────

def test_collection_complete_refuses_json_under_output():
    with pytest.raises(SystemExit, match="refusing"):
        CCX.main(["--json", str(ROOT / "output" / "x.json"), "--steps", "fullstats"])


def test_refetch_proof_cli_refuses_out_under_output():
    with pytest.raises(SystemExit, match="refusing"):
        HR.main(["--dry-run", "--out", str(ROOT / "output" / "x.json")])
