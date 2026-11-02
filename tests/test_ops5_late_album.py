"""Late daily album (owner rule, Data Truth round 5).

The album for tip day T is on time until T's rollover (10:00 Israel wall clock: 07:00 UTC in summer,
08:00 UTC in winter). After that it is sent labelled 'late' and lists only matches whose kickoff is known
and after the send time. Nothing here reaches Telegram (tests/conftest.py guard; send_album is mocked).
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from telegram_bot import daily_pack as dp


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


@pytest.mark.parametrize("now, late", [
    (_utc(2026, 11, 2, 7, 59), False),     # winter: 09:59 Israel
    (_utc(2026, 11, 2, 8, 0), True),       # winter: 10:00 Israel = rollover
    (_utc(2026, 7, 1, 6, 59), False),      # summer: 09:59 Israel
    (_utc(2026, 7, 1, 7, 0), True),        # summer: 10:00 Israel
    (_utc(2026, 10, 25, 7, 30), False),    # the DST-end day: 09:30 Israel (UTC+2 from 02:00)
    (_utc(2026, 11, 2, 22, 30), True),     # an evening fallback send for the same UTC date
])
def test_album_is_late_follows_the_israeli_rollover(now, late):
    assert dp.album_is_late(now) is late


def _tip_rows():
    return pd.DataFrame([
        {"league": "Championship", "date": "2026-11-02", "home_team": "A", "away_team": "B",
         "kickoff_utc": "2026-11-02T08:30:00Z", "tier": "SNIPER", "market": "O/U 2.5 OVER", "odds": 1.9,
         "edge": 5.0, "tip_day": "2026-11-02", "notify_key": "k1"},          # already started at 09:00
        {"league": "Championship", "date": "2026-11-02", "home_team": "C", "away_team": "D",
         "kickoff_utc": "2026-11-02T15:00:00Z", "tier": "SNIPER", "market": "O/U 2.5 OVER", "odds": 1.9,
         "edge": 4.0, "tip_day": "2026-11-02", "notify_key": "k2"},          # still to play
        {"league": "Championship", "date": "2026-11-02", "home_team": "E", "away_team": "F",
         "kickoff_utc": None, "tier": "SNIPER", "market": "O/U 2.5 OVER", "odds": 1.9,
         "edge": 3.0, "tip_day": "2026-11-02", "notify_key": "k3"},          # kickoff unknown
    ])


@pytest.fixture
def pack(monkeypatch):
    """build_pack with every image stubbed; records what the tips image and the sections were given."""
    seen = {}
    monkeypatch.setattr(dp, "todays_rows", lambda day: _tip_rows())
    monkeypatch.setattr(dp, "render_pngs", lambda rows, title: seen.update(tips=rows, title=title) or [b"png"])
    monkeypatch.setattr(dp, "top_picks_image", lambda today, freeze=False, now=None: seen.update(tp_now=now) or [])
    monkeypatch.setattr(dp, "props_image", lambda today, freeze=False, cutoff=None: seen.update(props=cutoff) or [])
    monkeypatch.setattr(dp, "bet_builder_image", lambda today, cutoff=None: seen.update(bb=cutoff) or [])
    monkeypatch.setattr(dp, "sharp_image", lambda today, cutoff=None: seen.update(sharp=cutoff) or [])
    for name in ("top_picks_results_image", "results_image", "props_results_image", "record_image",
                 "one_x_two_image"):
        monkeypatch.setattr(dp, name, lambda *a, **k: [])
    monkeypatch.setattr(dp, "scoreboard_images", lambda: [])
    return seen


def test_late_album_is_labelled_and_lists_no_started_match(pack):
    now = _utc(2026, 11, 2, 9, 0)                                       # 11:00 Israel - late
    pngs, caption = dp.build_pack(now)
    assert "(late)" in caption and "LATE" in caption
    assert pack["title"].endswith("(late)")
    assert list(pack["tips"]["notify_key"]) == ["k2"]                   # started and unknown-kickoff left out
    assert "today's tips (1)" in caption
    assert pack["props"] == now and pack["bb"] == now and pack["sharp"] == now and pack["tp_now"] == now


def test_on_time_album_is_unchanged(pack):
    pngs, caption = dp.build_pack(_utc(2026, 11, 2, 6, 30))            # 08:30 Israel - on time
    assert "late" not in caption.lower()
    assert "(late)" not in pack["title"]
    assert list(pack["tips"]["notify_key"]) == ["k1", "k2", "k3"]
    assert pack["props"] is None and pack["bb"] is None and pack["sharp"] is None and pack["tp_now"] is None


def test_section_filters_drop_started_and_unknown(monkeypatch):
    now = _utc(2026, 11, 2, 9, 0)
    drawn = {}
    monkeypatch.setattr(dp, "render_table", lambda title, cols, rows, **k: drawn.update({title: rows}) or [b"p"])

    # sharp: kickoff comes from the sharp history the tips were built from
    s = pd.DataFrame([
        {"date": "2026-11-02", "league": "Serie B", "match": "X vs Y", "market": "1X2 HOME (X)",
         "opening_odds": 2.0, "current_odds": 1.8, "drift_pct": -10.0, "signal": "STRONG"},
        {"date": "2026-11-02", "league": "Serie B", "match": "P vs Q", "market": "1X2 HOME (P)",
         "opening_odds": 2.0, "current_odds": 1.8, "drift_pct": -10.0, "signal": "STRONG"},
        {"date": "2026-11-02", "league": "Serie B", "match": "M vs N", "market": "1X2 HOME (M)",
         "opening_odds": 2.0, "current_odds": 1.8, "drift_pct": -10.0, "signal": "STRONG"}])
    hist = {"1_h2h": {"league": "Serie B", "home": "X", "away": "Y", "date": "2026-11-02 08:00"},
            "2_h2h": {"league": "Serie B", "home": "P", "away": "Q", "date": "2026-11-02 19:30"}}
    monkeypatch.setattr(dp, "_sharp_frame", lambda: s.copy())
    dp.sharp_image("2026-11-02", cutoff=now, history=hist)
    rows = next(v for k, v in drawn.items() if k.startswith("Sharp"))
    assert [r["cells"][1] for r in rows] == ["P vs Q"]
    drawn.clear()
    dp.sharp_image("2026-11-02", history=hist)                          # on time: all three
    assert len(next(v for k, v in drawn.items() if k.startswith("Sharp"))) == 3

    # Bet Builder: kickoff looked up in v9's own fixture kickoffs
    bb = pd.DataFrame([{"match_date": "2026-11-02", "league": "La Liga 2", "match": "R vs S",
                        "legs": "Over 3.5 goals (30%)", "joint_probability": 0.3, "fair_odds": 3.3},
                       {"match_date": "2026-11-02", "league": "La Liga 2", "match": "T vs U",
                        "legs": "Over 3.5 goals (30%)", "joint_probability": 0.3, "fair_odds": 3.3}])
    ko = pd.DataFrame([{"league": "La Liga 2", "home_team": "R", "away_team": "S", "match_date": "2026-11-02",
                        "kickoff_utc": "2026-11-02T08:00:00Z"},
                       {"league": "La Liga 2", "home_team": "T", "away_team": "U", "match_date": "2026-11-02",
                        "kickoff_utc": "2026-11-02T20:00:00Z"}])
    monkeypatch.setattr(dp, "_pro_bet_builder", lambda: bb.copy())
    drawn.clear()
    dp.bet_builder_image("2026-11-02", cutoff=now, ko=ko)
    rows = [r for v in drawn.values() for r in v]
    assert [r["band"].split(" · ")[0] for r in rows if "band" in r] == ["T vs U"]


def test_props_late_filter(monkeypatch):
    now = _utc(2026, 11, 2, 9, 0)
    t = pd.DataFrame([
        {"date": "2026-11-02", "kickoff_utc": "2026-11-02T08:00:00Z", "player_id": 1, "player_name": "Early",
         "match": "A v B", "market": "goals", "model_prob": 0.4, "market_odds": 3.0, "ev": 0.2},
        {"date": "2026-11-02", "kickoff_utc": "2026-11-02T18:00:00Z", "player_id": 2, "player_name": "Later",
         "match": "C v D", "market": "goals", "model_prob": 0.4, "market_odds": 3.0, "ev": 0.2}])
    monkeypatch.setattr(dp, "_read", lambda name, **k: t.copy())
    drawn = {}
    monkeypatch.setattr(dp, "render_table", lambda title, cols, rows, **k: drawn.update(rows=rows) or [b"p"])
    dp.props_image("2026-11-02", cutoff=now)
    names = [r["cells"][1] for r in drawn["rows"] if "cells" in r]
    assert names == ["Later"]


def test_send_daily_pack_late_without_sending(pack, monkeypatch):
    """The real send path at 11:00 Israel with only the transport mocked: the album goes out labelled
    late, without the started match, and nothing reaches Telegram (conftest guard)."""
    import datetime as real
    import types
    from telegram_bot import notifier as nt

    class _Clock(real.datetime):
        @classmethod
        def now(cls, tz=None):
            return real.datetime(2026, 11, 2, 9, 0, tzinfo=real.timezone.utc)
    monkeypatch.setattr(dp, "_dt", types.SimpleNamespace(datetime=_Clock, timezone=real.timezone,
                                                         timedelta=real.timedelta))
    sent = {}
    monkeypatch.setattr(nt, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(nt, "_load_notified", lambda: set())
    monkeypatch.setattr(nt, "_save_notified", lambda s: sent.update(keys=set(s)))
    monkeypatch.setattr(dp, "send_album", lambda tok, chat, pngs, cap: sent.update(caption=cap) or True)
    assert dp.send_daily_pack() is True
    assert "(late)" in sent["caption"] and "LATE" in sent["caption"]
    assert list(pack["tips"]["notify_key"]) == ["k2"]
    assert "DIGEST|2026-11-02" in sent["keys"]
