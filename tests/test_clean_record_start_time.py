"""Go-live rehearsal defect 6 (2026-11-15): the Clean Record split on the start DATE, so a tip generated on the
start day BEFORE the go-live merge time counted as Clean Record. The split now uses the full
clean_record_start_utc timestamp (a date alone = 00:00 UTC, as before). Synthetic config in tmp_path only."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from dashboard_data import clean_record as CR


@pytest.fixture
def start_at(tmp_path, monkeypatch):
    def _set(value):
        p = tmp_path / "trustworthy_performance.json"
        p.write_text(json.dumps({"clean_record_start_utc": value}), encoding="utf-8")
        monkeypatch.setattr(CR, "TP_CONFIG", p)
    return _set


def test_start_ts_is_the_full_timestamp(start_at):
    start_at("2026-11-21T09:30:00Z")
    assert CR.start_ts() == pd.Timestamp("2026-11-21T09:30:00Z")
    assert not CR.started("2026-11-21T09:29:59Z") and CR.started("2026-11-21T09:30:00Z")


def test_a_date_alone_is_midnight_utc(start_at):
    start_at("2026-11-21")
    assert CR.start_ts() == pd.Timestamp("2026-11-21", tz="UTC")


def test_row_generated_on_the_start_day_before_the_start_time_is_not_clean(start_at):
    start_at("2026-11-21T09:30:00Z")
    d = pd.DataFrame({
        "gen": ["2026-11-21 06:00", "2026-11-21 09:30", "2026-11-21 12:00", "2026-11-21", "2026-11-22"],
        "day": ["2026-11-21", "2026-11-22", "2026-11-21", "2026-11-22", "2026-11-23"],
    })
    got = CR.assign(d, "gen", "day").tolist()
    assert got == [CR.BEFORE,       # same day, before the merge time: old system
                   CR.CLEAN,        # exactly at the start
                   CR.CLEAN,        # after the start, match dated the start day: played on it
                   CR.UNPLACED,     # a date without a time on the start day cannot be placed
                   CR.CLEAN]


def test_midnight_start_keeps_the_date_behaviour(start_at):
    start_at("2026-11-21T00:00:00Z")
    d = pd.DataFrame({"gen": ["2026-11-21", "2026-11-20 23:59"], "day": ["2026-11-22", "2026-11-22"]})
    assert CR.assign(d, "gen", "day").tolist() == [CR.CLEAN, CR.BEFORE]
