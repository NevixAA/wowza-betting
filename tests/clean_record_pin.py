"""Pin the Clean Record start for the tests that check the split MECHANICS on synthetic rows dated around
2026-11-21 (owner 2026-11-18: the committed start becomes the go-live merge instant, so these tests must not
depend on the committed value). The committed value itself is checked for consistency, not pinned, by
test_clean_record.test_one_owner_set_start_date and test_dashboard_records.test_no_second_start_date."""
from __future__ import annotations

import json

import pytest

PINNED_UTC = "2026-11-21T00:00:00Z"


def pin_start(tmp_path, monkeypatch, utc: str) -> str:
    """Point every reader of the Clean Record start (clean_record, season, config) at `utc` for one test."""
    import config
    from dashboard_data import clean_record as CR
    from dashboard_data import season
    cfg = json.loads(CR.TP_CONFIG.read_text(encoding="utf-8"))
    cfg["clean_record_start_utc"] = utc
    p = tmp_path / "_clean_record_pin" / "trustworthy_performance_pinned.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(CR, "TP_CONFIG", p)
    monkeypatch.setattr(season, "TP_CONFIG", p)
    monkeypatch.setattr(season, "OWNER_DECISION", utc[:10])
    monkeypatch.setattr(config, "CLEAN_RECORD_START", utc[:10])
    return utc


@pytest.fixture()
def pinned_start(tmp_path, monkeypatch):
    yield pin_start(tmp_path, monkeypatch, PINNED_UTC)
