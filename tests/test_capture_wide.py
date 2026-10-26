"""WIDE passes from inside the NEAR loop + FAR price refresh (owner-approved 2026-10-26).

Covers: the WIDE-due decision from a RECORDED timestamp (unknown age runs, fresh skips), the
per-day budget guards, FAR change-only recording of the engine's families (pair-level, plus the
FAR heartbeat), that nothing reaches the 5-minute predict loop, and the workflow wiring.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import src.book_quotes as bq
from src import capture_schedule as cs

ROOT = Path(__file__).resolve().parents[1]
WFS = {"std_capture": ROOT / ".github" / "workflows" / "std_odds_capture.yml",
       "nf_capture": ROOT / ".github" / "workflows" / "nf_odds_capture.yml"}
SCRIPTS = {"std_capture": ROOT / "scripts" / "capture_std_sidemarket_odds_forward.py",
           "nf_capture": ROOT / "scripts" / "capture_nf_odds_forward.py"}


def _t(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


NOW = _t("2026-10-26T12:00:00Z")


# ── WIDE due: recorded timestamp, never mtime ─────────────────────────────────────────────────
@pytest.mark.parametrize("state", [{}, {"last_wide_utc": "garbage"}, {"last_wide_utc": None},
                                   {"last_wide_utc": "2026-10-27T12:00:00Z"}])
def test_unknown_age_runs_now(state):
    """Missing, corrupt or future timestamp = unknown age = infinitely old."""
    ok, why = cs.wide_due(state, NOW)
    assert ok and why == "UNKNOWN_AGE"


def test_fresh_wide_is_skipped_and_stale_wide_runs():
    assert cs.wide_due({"last_wide_utc": "2026-10-26T07:00:00Z"}, NOW)[0] is False   # 5 h
    ok, why = cs.wide_due({"last_wide_utc": "2026-10-26T06:00:00Z"}, NOW)            # 6 h
    assert ok and why.startswith("DUE")


def test_missing_or_corrupt_state_file_is_unknown_age(tmp_path):
    assert cs.load_wide_state(tmp_path / "nope.json") == {}
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert cs.load_wide_state(p) == {}
    assert cs.wide_due(cs.load_wide_state(p), NOW)[0] is True


def test_the_due_decision_ignores_file_mtime(tmp_path):
    """A fresh checkout gives every file a new mtime. Only the recorded timestamp counts."""
    p = tmp_path / "s.json"
    cs.record_wide(p, _t("2026-10-26T01:00:00Z"), 400)    # file mtime = now, recorded 11 h ago
    assert cs.wide_due(cs.load_wide_state(p), NOW)[0] is True


def test_record_wide_counts_passes_and_calls_per_utc_day(tmp_path):
    p = tmp_path / "s.json"
    cs.record_wide(p, _t("2026-10-26T01:00:00Z"), 400)
    st = cs.record_wide(p, _t("2026-10-26T07:00:00Z"), 450)
    assert (st["passes_today"], st["calls_today"], st["last_pass_calls"]) == (2, 850, 450)
    assert st["last_wide_utc"] == "2026-10-26T07:00:00Z"
    st = cs.record_wide(p, _t("2026-10-27T00:30:00Z"), 300)       # new UTC day resets
    assert (st["day"], st["passes_today"], st["calls_today"]) == ("2026-10-27", 1, 300)


# ── budget guard ──────────────────────────────────────────────────────────────────────────────
def test_day_pass_cap_blocks_a_fifth_pass():
    st = {"last_wide_utc": "2026-10-26T05:00:00Z", "day": "2026-10-26", "passes_today": 4,
          "calls_today": 100, "last_pass_calls": 25}
    ok, why = cs.wide_due(st, NOW, max_per_day=4)
    assert not ok and why.startswith("DAY_PASS_CAP")


def test_day_call_cap_refuses_a_pass_that_would_cross_it():
    st = {"last_wide_utc": "2026-10-26T05:00:00Z", "day": "2026-10-26", "passes_today": 2,
          "calls_today": 1800, "last_pass_calls": 490}
    ok, why = cs.wide_due(st, NOW, max_calls_per_day=2200)
    assert not ok and why.startswith("DAY_CALL_CAP")
    assert cs.wide_due(st, NOW, max_calls_per_day=2400)[0] is True
    # yesterday's spend does not count against today
    assert cs.wide_due({**st, "day": "2026-10-25"}, NOW, max_calls_per_day=2200)[0] is True


def test_no_wide_pass_starts_near_the_loop_cap():
    ok, why = cs.wide_due({}, NOW, loop_elapsed_min=cs.CAP_MIN - 5, loop_cap_min=cs.CAP_MIN)
    assert not ok and why == "LOOP_TOO_LATE"
    assert cs.wide_due({}, NOW, loop_elapsed_min=30, loop_cap_min=cs.CAP_MIN)[0] is True


def test_a_due_wide_waits_while_a_close_is_forming_but_not_forever():
    st = {"last_wide_utc": "2026-10-26T05:30:00Z"}                       # 6.5 h old
    ok, why = cs.wide_due(st, NOW, next_ko_min=20)
    assert not ok and why.startswith("DEFER_CLOSE_FORMING")
    assert cs.wide_due(st, NOW, next_ko_min=120)[0] is True              # nothing imminent
    late = {"last_wide_utc": "2026-10-26T03:30:00Z"}                     # 8.5 h: past the cap
    assert cs.wide_due(late, NOW, next_ko_min=20)[0] is True
    assert cs.wide_due({}, NOW, next_ko_min=5)[0] is True                # unknown age runs


def test_cli_prints_machine_readable_lines(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LOOP_START_S", raising=False)
    assert cs.main(["wide-due", "--state", str(tmp_path / "none.json")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "WIDE_DUE=1" in out and "WIDE_REASON=UNKNOWN_AGE" in out


def test_record_wide_from_env_is_inert_without_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPTURE_WIDE_STATE", raising=False)
    cs.record_wide_from_env(NOW, 10)                 # no path -> no file, no raise
    monkeypatch.setenv("CAPTURE_WIDE_STATE", str(tmp_path / "w" / "s.json"))
    cs.record_wide_from_env(NOW, 10)
    assert json.loads((tmp_path / "w" / "s.json").read_text(encoding="utf-8"))["calls_today"] == 10


# ── FAR change-only recording of the engine's families ────────────────────────────────────────
def _row(ts, side, odds, mins, market="ou", line=2.5, band="FAR", book=8, fid=1):
    return {"snapshot_ts": ts, "fixture_id": fid, "kickoff_utc": "2026-10-30T18:00:00Z",
            "minutes_to_kickoff": mins, "ladder_band": band, "league": "L",
            "model_type": "standard", "home_team": "A", "away_team": "B", "bookmaker_id": book,
            "bookmaker": "Bet365", "market": market, "side": side, "line": line, "odds": odds,
            "source": "t"}


def _pair(ts, over, under, mins, **kw):
    return [_row(ts, "over", over, mins, **kw), _row(ts, "under", under, mins, **kw)]


@pytest.fixture
def quotes(tmp_path, monkeypatch):
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    monkeypatch.setattr(bq, "LEGACY_FILE", tmp_path / "none.csv")
    monkeypatch.setattr(bq, "FAR_HEARTBEAT_H", 9.0)
    return tmp_path


def test_far_unchanged_inside_the_heartbeat_window_writes_nothing(quotes):
    assert bq.append_quotes(_pair("2026-10-26T00:00:00Z", 1.30, 3.40, 5000)) == 2
    assert bq.append_quotes(_pair("2026-10-26T06:00:00Z", 1.30, 3.40, 4640)) == 0


def test_a_far_move_writes_both_sides_of_the_pair(quotes):
    """edge_af_quotes pairs only same-snapshot rows: the moved side alone would make the book
    one-sided until its partner happened to move too."""
    bq.append_quotes(_pair("2026-10-26T00:00:00Z", 1.30, 3.40, 5000))
    assert bq.append_quotes(_pair("2026-10-26T06:00:00Z", 1.33, 3.40, 4640)) == 2
    d = bq.load_quotes()
    last = d[d.snapshot_ts == "2026-10-26T06:00:00Z"].set_index("side")["obs_reason"]
    assert last.to_dict() == {"over": "change", "under": "heartbeat"}
    assert list(d[d.snapshot_ts == "2026-10-26T00:00:00Z"]["obs_reason"]) == ["open", "open"]


def test_far_heartbeat_rewrites_an_unchanged_pair_once_it_is_old(quotes):
    bq.append_quotes(_pair("2026-10-26T00:00:00Z", 1.30, 3.40, 5000))
    assert bq.append_quotes(_pair("2026-10-26T08:59:00Z", 1.30, 3.40, 4461)) == 0
    assert bq.append_quotes(_pair("2026-10-26T09:00:00Z", 1.30, 3.40, 4460)) == 2
    d = bq.load_quotes()
    assert set(d[d.snapshot_ts == "2026-10-26T09:00:00Z"]["obs_reason"]) == {"heartbeat"}


def test_far_heartbeat_can_be_switched_off(quotes, monkeypatch):
    monkeypatch.setattr(bq, "FAR_HEARTBEAT_H", 0.0)
    bq.append_quotes(_pair("2026-10-26T00:00:00Z", 1.30, 3.40, 5000))
    assert bq.append_quotes(_pair("2026-10-27T00:00:00Z", 1.30, 3.40, 3560)) == 0


def test_btts_is_tracked_and_half_time_is_not(quotes):
    yes_no = [_row("2026-10-26T00:00:00Z", "yes", 1.80, 5000, market="btts", line=None),
              _row("2026-10-26T00:00:00Z", "no", 2.00, 5000, market="btts", line=None)]
    ht = _pair("2026-10-26T00:00:00Z", 3.25, 1.30, 5000, market="ht_ou", line=1.5)
    assert bq.append_quotes(yes_no + ht) == 4
    moved = [{**r, "snapshot_ts": "2026-10-26T06:00:00Z", "odds": r["odds"] + 0.1}
             for r in yes_no + ht]
    assert bq.append_quotes(moved) == 2                # BTTS pair only; HT stays open-only
    d = bq.load_quotes()
    assert set(d[d.snapshot_ts == "2026-10-26T06:00:00Z"]["market"]) == {"btts"}


def test_the_engine_reads_a_far_refresh_as_a_fresh_two_sided_pair(quotes):
    """End to end with src/edge_af_quotes: after a heartbeat the book is a same-snapshot pair
    whose timestamp is the refresh, not the 9-hour-old open."""
    import pandas as pd
    from src import edge_af_quotes as EAF
    bq.append_quotes(_pair("2026-10-26T00:00:00Z", 1.30, 3.40, 5000, line=1.5))
    bq.append_quotes(_pair("2026-10-26T09:00:00Z", 1.30, 3.40, 4460, line=1.5))
    q = bq.load_quotes()
    q = q.assign(_fam="OU15", _ts=pd.to_datetime(q.snapshot_ts, utc=True),
                 _side=q.side.str.upper())
    (fid, fam, quote, kind), = EAF._pairs_for(q)
    assert kind == "pair" and str(quote.ts) == "2026-10-26 09:00:00+00:00"


# ── nothing in the 5-minute predict loop ──────────────────────────────────────────────────────
def test_predict_workflow_makes_no_capture_calls():
    text = (ROOT / ".github" / "workflows" / "predict.yml").read_text(encoding="utf-8")
    for needle in ("capture_std_sidemarket_odds_forward", "capture_nf_odds_forward",
                   "capture_schedule", "CAPTURE_WIDE"):
        assert needle not in text, needle


def test_book_quotes_and_capture_schedule_make_no_api_calls():
    for mod in (ROOT / "src" / "book_quotes.py", ROOT / "src" / "capture_schedule.py"):
        s = mod.read_text(encoding="utf-8")
        for bad in ("import requests", "from requests", "import urllib", "import http",
                    "from player_model", "import player_model"):
            assert bad not in s, (mod.name, bad)


# ── capture scripts + workflows ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("writer", sorted(SCRIPTS))
def test_scripts_record_wide_passes_only_and_count_calls(writer):
    s = SCRIPTS[writer].read_text(encoding="utf-8")
    assert "_ODDS_CALLS += 1" in s
    i = s.index("_ODDS_CALLS += 1")
    assert i < s.index('r = requests.get(f"{_BASE}/odds"', i), "count before the request"
    tail = s[s.index('if __name__ == "__main__":'):]
    assert "if MAX_HOURS is None and _KEY:" in tail and "record_wide_from_env(" in tail


@pytest.mark.parametrize("writer", sorted(WFS))
def test_workflow_parses_runs_wide_inside_the_loop_and_stages_per_file(writer):
    text = WFS[writer].read_text(encoding="utf-8")
    y = yaml.safe_load(text)
    job = next(iter(y["jobs"].values()))
    steps = job["steps"]
    cap = next(s for s in steps if "CAPTURE_BOOK_QUOTES" in s.get("env", {}))
    run = cap["run"]
    loop = run[run.index("while : ; do"):]
    assert loop.index("wide_if_due") < loop.index("--max-hours"), "WIDE check precedes NEAR pass"
    assert "python -m src.capture_schedule wide-due --state" in run
    assert '--next-ko-min "$LAST_KO"' in run and "s/^NEXT_KO_MIN=" in loop
    assert "\x01" not in text
    assert f'CAPTURE_WIDE_STATE="output/capture_wide/{writer}.json"' in run
    env = cap["env"]
    assert float(env["CAPTURE_WIDE_EVERY_H"]) == 6
    assert int(env["CAPTURE_WIDE_MAX_PER_DAY"]) <= 4
    assert int(env["CAPTURE_WIDE_MAX_CALLS_PER_DAY"]) <= 2500
    commit = "\n".join(s.get("run", "") for s in steps if "git commit" in s.get("run", ""))
    assert f"git add -f output/capture_wide/{writer}.json" in commit
    assert "git add -A" not in commit and "git add ." not in commit
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "main"
    # the loop cap plus one late WIDE pass and the commit must stay inside the timeout
    assert job["timeout-minutes"] >= cs.CAP_MIN + 15


def test_std_near_now_covers_04_to_08_utc():
    y = yaml.safe_load(WFS["std_capture"].read_text(encoding="utf-8"))
    crons = [c["cron"] for c in y[True]["schedule"]]
    assert "23,52 4-8 * * *" in crons
    wide = [c for c in crons if c == "5 0,6,12,18 * * *"]
    assert wide, "the WIDE cron string is matched verbatim by the run step"
