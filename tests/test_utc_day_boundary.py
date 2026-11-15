"""Day decisions use the UTC day, also when the machine's local date is already tomorrow (fix13).

Found 2026-11-07 00:xx Israel time (2026-11-06 22:xx UTC): three tests failed on a laptop and passed
in CI. The notifier filtered "today's" tips with a naive local datetime.now(), so between local
midnight and UTC midnight every tip dated the UTC day was dropped as "already played"; and the
rolling-store tests computed their expected days with local date.today() while the store uses UTC.

Each test pins the clock at 23:30 UTC on 2026-11-06 with a local zone 2 hours ahead (local is
2026-11-07 01:30), i.e. exactly the window the bug lived in. The fake clock answers naive now()
with the LOCAL time, so any code that falls back to the local clock picks the wrong day and fails.
"""
from __future__ import annotations

import datetime as _real_dt
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import config

UTC_NOW = _real_dt.datetime(2026, 11, 6, 23, 30, tzinfo=timezone.utc)
LOCAL_TZ = timezone(timedelta(hours=2))                    # Israel winter time
UTC_DAY, LOCAL_DAY = "2026-11-06", "2026-11-07"
ROOT = Path(__file__).resolve().parents[1]


class FakeDatetime(_real_dt.datetime):
    """datetime whose clock is UTC_NOW; naive now() returns the LOCAL wall clock (UTC+2)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return UTC_NOW.astimezone(LOCAL_TZ).replace(tzinfo=None)
        return UTC_NOW.astimezone(tz)

    @classmethod
    def today(cls):
        return cls.now()

    @classmethod
    def utcnow(cls):
        return UTC_NOW.replace(tzinfo=None)


def test_the_fake_clock_is_in_the_bug_window():
    assert FakeDatetime.now().strftime("%Y-%m-%d") == LOCAL_DAY
    assert FakeDatetime.now(timezone.utc).strftime("%Y-%m-%d") == UTC_DAY


def test_utc_clock_helpers_give_the_utc_day(monkeypatch):
    from src import utc_clock as C
    monkeypatch.setattr(C, "datetime", FakeDatetime)
    assert C.utc_today_str() == UTC_DAY
    assert str(C.utc_today()) == UTC_DAY
    assert C.utc_today_ts() == pd.Timestamp(UTC_DAY)
    assert C.utc_now() == _real_dt.datetime(2026, 11, 6, 23, 30)      # naive UTC wall clock


def test_telegram_still_sends_the_utc_days_tip_after_local_midnight(tmp_path, monkeypatch):
    from src import utc_clock as C
    from telegram_bot import notifier as N
    monkeypatch.setattr(C, "datetime", FakeDatetime)
    monkeypatch.setattr(N, "datetime", FakeDatetime)              # a naive now() here would be LOCAL
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    notified, sent = set(), []
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: notified)
    monkeypatch.setattr(N, "_save_notified", lambda keys, *a, **k: notified.update(keys))
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    pd.DataFrame([{"date": UTC_DAY, "league": "Japan J-League", "home_team": "Mito", "away_team": "Shimizu",
                   "bet": "UNDER", "best_side": "UNDER", "odds_over25": 2.45, "odds_under25": 1.89,
                   "best_edge": 0.05, "signal_tier": "VALUABLE", "drift_signal": "",
                   "model_type": "new_format"}]).to_csv(tmp_path / "bets.csv", index=False)
    assert N.notify_new_snipers() == 1
    assert f"{UTC_DAY}|Mito|Shimizu|UNDER" in notified


def test_rolling_store_cutoff_is_the_utc_day(tmp_path, monkeypatch):
    import src.rolling_store as rs
    monkeypatch.setattr(rs, "datetime", FakeDatetime)
    live = tmp_path / "book_odds_snapshots.csv"
    day = lambda off: (pd.Timestamp(UTC_DAY) + pd.Timedelta(days=off)).strftime("%Y-%m-%d")
    rows = [{"snapshot_ts": f"{day(-5)}T10:00:00Z", "league": "L", "match_date": day(off),
             "kickoff_utc": f"{day(off)}T19:00:00Z", "match": f"A{off} vs B", "bookmaker": "bk",
             "market": "OU25", "side": "OVER", "odds": 1.91} for off in (-3, -2)]
    pd.DataFrame(rows).to_csv(live, index=False, lineterminator="\n")
    s = rs.roll(live, keep_days=2)
    # UTC day 11-06, keep 2 days -> cutoff 11-04: only 11-03 moves. A local-day cutoff (11-05)
    # would also have archived 11-04.
    assert s["moved"] == 1
    archived = sorted(p.stem for p in (tmp_path / "book_odds_snapshots_archive").glob("*.csv"))
    assert archived == [day(-3)]


def test_prop_quotes_day_file_is_the_utc_day(tmp_path, monkeypatch):
    from player_model import odds_fetcher as of
    monkeypatch.setattr(_real_dt, "datetime", FakeDatetime)      # _append_quotes imports it locally
    monkeypatch.setattr(of, "_QUOTES_DIR", tmp_path)
    of._append_quotes([{"match_date": "2026-11-08", "league": "L", "match": "X vs Y", "player": "p",
                        "market": "goals", "side": "yes", "point": None, "odds": 4.0,
                        "bookmaker": "bk", "source": "oddsapi"}])
    assert [p.name for p in tmp_path.glob("*.csv")] == [f"{UTC_DAY}.csv"]


def test_drift_purge_uses_the_utc_day(monkeypatch):
    from src import drift as D
    from src import utc_clock as C
    monkeypatch.setattr(C, "datetime", FakeDatetime)
    # cutoff_days=10 from the UTC day 11-06 -> 10-27 is kept; from the local day 11-07 it would go.
    hist = {"A | B | 2026-10-27": {}, "C | D | 2026-10-26": {}}
    monkeypatch.setattr(D, "_load", lambda: hist)
    monkeypatch.setattr(D, "_save", lambda h: None)
    D.purge_old(cutoff_days=10)
    assert "A | B | 2026-10-27" in hist and "C | D | 2026-10-26" not in hist


@pytest.mark.parametrize("rel", ["telegram_bot/notifier.py", "src/live_scanner.py", "src/drift.py",
                                 "player_model/predict.py", "player_model/pipeline.py",
                                 "dashboard_data/today_view.py"])
def test_day_deciding_code_has_no_naive_local_today(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for bad in ('datetime.now().strftime("%Y-%m-%d")', "date.today()", "pd.Timestamp.now().normalize()"):
        assert bad not in src, f"{rel}: {bad} decides a day on the LOCAL clock; use src/utc_clock.py"
