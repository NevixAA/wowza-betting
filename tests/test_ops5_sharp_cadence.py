"""Sharp snapshot cadence for Pro 1X2 (owner: keep the 12h max price age, snapshot more often).

No network: requests.get is replaced. The schedule test reads .github/workflows/sharp_tracker.yml.
"""
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src import sharp_tracker as st
from src.integrity import oddsapi_quota as Q

ROOT = Path(__file__).resolve().parents[1]


class _Resp:
    def __init__(self, events, last="1", remaining="50000", status=200):
        self.status_code = status
        self.headers = {"x-requests-last": last, "x-requests-remaining": remaining}
        self._events = events

    def json(self):
        return self._events


def _event(eid, market, ko):
    if market == "h2h":
        outs = [{"name": "H", "price": 2.0}, {"name": "A", "price": 3.5}, {"name": "Draw", "price": 3.2}]
    else:
        outs = [{"name": "Over", "price": 1.9, "point": 2.5}, {"name": "Under", "price": 1.9, "point": 2.5}]
    return {"id": eid, "home_team": "H", "away_team": "A", "commence_time": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmakers": [{"markets": [{"key": market, "outcomes": outs}]}] * 5}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(st, "SHARP_HISTORY_DIR", tmp_path / "sharp_history")
    monkeypatch.setattr(st, "SHARP_HISTORY_FILE", tmp_path / "sharp_history.json")
    monkeypatch.setattr(st, "SHARP_TIPS_FILE", tmp_path / "sharp_tips.csv")
    monkeypatch.setattr(st, "append_sharp_signals", lambda tips: None)
    monkeypatch.setattr(config, "ENABLED_LEAGUES", ["Serie B", "Championship"])
    monkeypatch.setattr(config, "ODDS_API_SPORT_KEYS", {"Serie B": "soccer_italy_serie_b",
                                                        "Championship": "soccer_efl_champ"})
    return tmp_path


def _usage_line(caplog):
    lines = [r.getMessage() for r in caplog.records if "[oddsapi_usage]" in r.getMessage()]
    assert len(lines) == 1
    return lines[0]


def test_a_fetch_logs_its_credit_cost_for_the_quota_monitor(sandbox, monkeypatch, caplog):
    ko = datetime.utcnow() + timedelta(days=1)
    calls = []

    def get(url, params, timeout):
        calls.append(params["markets"])
        return _Resp([_event("e1", params["markets"], ko)])
    monkeypatch.setattr(st.requests, "get", get)
    caplog.set_level(logging.INFO)
    st.run()
    assert len(calls) == 4                                        # 2 leagues x 2 markets
    line = _usage_line(caplog)
    rec = json.loads(line.split("[oddsapi_usage] ", 1)[1])
    assert rec["source"] == "sharp_tracker" and rec["status"] == "FETCHED"
    assert rec["credits"] == 4 and rec["credits_by_market"] == {"totals": 2, "h2h": 2}
    assert rec["max_credits_per_snapshot"] == 4 and rec["requests_remaining"] == 50000
    assert rec["h2h_freshness"]["upcoming_fixtures"] == 1 and rec["h2h_freshness"]["older_than_12h"] == 0
    # the integrity monitor (src/integrity/oddsapi_quota) now counts sharp_tracker's credits
    parsed = Q.parse_log_text("2026-11-02T10:23:00.0000000Z " + line, "sharp_tracker.yml", 1)
    assert Q.build_health([parsed], 24)["by_workflow"]["sharp_tracker.yml"]["credits"] == 4


def test_a_fire_soon_after_the_last_snapshot_costs_nothing(sandbox, monkeypatch, caplog):
    now = datetime(2026, 11, 2, 10, 23)
    st._save_history({"e1_h2h": {"league": "Serie B", "home": "H", "away": "A", "date": "2026-11-03 18:00",
                                 "market": "h2h", "opening": {},
                                 "snapshots": [{"at": "2026-11-02 09:30", "odds": {}, "n_books": 5}]}})
    monkeypatch.setattr(st.requests, "get", lambda *a, **k: pytest.fail("no OddsAPI call inside the interval"))
    caplog.set_level(logging.INFO)
    assert st.run(now=now) is None
    rec = json.loads(_usage_line(caplog).split("[oddsapi_usage] ", 1)[1])
    assert rec["status"] == "SKIPPED_FRESH" and rec["credits"] == 0


def test_quota_floor_stops_the_run(sandbox, monkeypatch, caplog):
    ko = datetime.utcnow() + timedelta(days=1)
    calls = []

    def get(url, params, timeout):
        calls.append(1)
        return _Resp([_event("e1", params["markets"], ko)], remaining=str(st.CREDIT_FLOOR - 1))
    monkeypatch.setattr(st.requests, "get", get)
    caplog.set_level(logging.INFO)
    st.run()
    assert len(calls) == 1                                        # the first answer shows the floor -> stop
    rec = json.loads(_usage_line(caplog).split("[oddsapi_usage] ", 1)[1])
    assert rec["status"] == "STOPPED_QUOTA_FLOOR" and "floor" in rec["stopped"]


def _cron_hours():
    text = (ROOT / ".github" / "workflows" / "sharp_tracker.yml").read_text(encoding="utf-8")
    slots = []
    for minute, hour in re.findall(r"cron:\s*'(\S+)\s+(\S+)\s+\*\s+\*\s+\*'", text):
        hours = range(24) if hour == "*" else [int(h) for h in hour.split(",")]
        slots += [h * 60 + int(minute) for h in hours]           # minutes after 00:00 UTC
    return sorted(slots)


def test_schedule_keeps_prices_under_12h_and_inside_the_credit_budget():
    slots = _cron_hours()
    gaps = [(b - a) for a, b in zip(slots, slots[1:] + [slots[0] + 1440])]
    assert max(gaps) <= 60                                        # a slot every hour, overnight included
    # Even if GitHub delivers only every 4th slot (it delivered ~3 of 8 before), the newest price at any
    # moment is under 12h old, with room for a delivery delay of several hours.
    assert 4 * max(gaps) + 6 * 60 < 12 * 60
    # Paid fetches per day with every slot delivered, under the min-interval guard.
    fetches, last = 0, None
    for t in [(s + 1440 * d) / 60 for d in range(2) for s in slots]:
        if last is None or t - last >= st.MIN_INTERVAL_H:
            fetches += int(t >= 24)
            last = t
    per = st.max_credits_per_snapshot(19)
    assert per == 38 and fetches == 12 and fetches * per == 456
