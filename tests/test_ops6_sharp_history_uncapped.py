"""ops6 item 3: sharp price history keeps EVERY snapshot (owner) - no 50-per-fixture cap - and stays
month-partitioned (a settled month's partition is never rewritten). No network."""
import json
from datetime import datetime, timedelta

from src import sharp_tracker as st


def _ev(eid, date, at, price):
    return {"id": eid, "market": "h2h", "league": "Serie B", "home": "H", "away": "A", "date": date,
            "fetched_at": at, "odds_home": price, "odds_away": 3.5, "odds_draw": 3.2, "n_books": 5}


def test_every_snapshot_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SHARP_HISTORY_DIR", tmp_path / "sharp_history")
    monkeypatch.setattr(st, "SHARP_HISTORY_FILE", tmp_path / "sharp_history.json")
    t0 = datetime(2026, 11, 1, 8)
    h = {}
    for i in range(120):
        h = st._update_history(h, [_ev("e1", "2026-11-20", (t0 + timedelta(hours=i)).isoformat(), 2.0 + i / 1000)])
    snaps = h["e1_h2h"]["snapshots"]
    assert len(snaps) == 120
    assert snaps[0]["odds"]["odds_home"] == 2.0            # the earliest snapshot survives
    st._save_history(h)
    back = st._load_history()
    assert len(back["e1_h2h"]["snapshots"]) == 120


def test_partitioned_by_month_and_settled_month_untouched(tmp_path, monkeypatch):
    d = tmp_path / "sharp_history"
    monkeypatch.setattr(st, "SHARP_HISTORY_DIR", d)
    monkeypatch.setattr(st, "SHARP_HISTORY_FILE", tmp_path / "sharp_history.json")
    h = st._update_history({}, [_ev("old", "2026-10-03", "2026-10-02T10:00:00", 2.1),
                                _ev("new", "2026-11-20", "2026-11-02T10:00:00", 2.2)])
    st._save_history(h)
    assert sorted(p.name for p in d.glob("*.json")) == ["2026-10.json", "2026-11.json"]
    oct_bytes = (d / "2026-10.json").read_bytes()
    h = st._update_history(st._load_history(), [_ev("new", "2026-11-20", "2026-11-02T14:00:00", 2.25)])
    st._save_history(h)
    assert (d / "2026-10.json").read_bytes() == oct_bytes
    assert len(json.loads((d / "2026-11.json").read_text(encoding="utf-8"))["new_h2h"]["snapshots"]) == 2
