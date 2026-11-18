"""Pin the Clean Record start for the tests that check the split MECHANICS on synthetic rows dated around
2026-11-21 (owner 2026-11-18: the committed start becomes the go-live merge instant, so these tests must not
depend on the committed value). The committed value itself is checked for consistency, not pinned, by
test_clean_record.test_one_owner_set_start_date and test_dashboard_records.test_no_second_start_date."""
from __future__ import annotations

import json

import pytest

PINNED_UTC = "2026-11-21T00:00:00Z"


@pytest.fixture()
def pinned_start(tmp_path, monkeypatch):
    import config
    from dashboard_data import clean_record as CR
    from dashboard_data import season
    cfg = json.loads(CR.TP_CONFIG.read_text(encoding="utf-8"))
    cfg["clean_record_start_utc"] = PINNED_UTC
    p = tmp_path / "trustworthy_performance_pinned.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(CR, "TP_CONFIG", p)
    monkeypatch.setattr(season, "TP_CONFIG", p)
    monkeypatch.setattr(season, "OWNER_DECISION", PINNED_UTC[:10])
    monkeypatch.setattr(config, "CLEAN_RECORD_START", PINNED_UTC[:10])
    yield PINNED_UTC
