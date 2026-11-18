"""scripts/golive/set_clean_record_start.py (owner 2026-11-18, "count them": the Clean Record starts at the go-live
merge instant). On tmp copies only: both places change together, nothing else in the files moves, and a value
that is not a UTC instant is refused."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("scr", ROOT / "scripts" / "golive" / "set_clean_record_start.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg, sea = tmp_path / "tp.json", tmp_path / "season.py"
    shutil.copy(m.CFG, cfg)
    shutil.copy(m.SEASON, sea)
    monkeypatch.setattr(m, "CFG", cfg)
    monkeypatch.setattr(m, "SEASON", sea)
    return m, cfg, sea


def test_sets_both_places_and_nothing_else(tmp_path, monkeypatch):
    m, cfg, sea = _mod(tmp_path, monkeypatch)
    before_c, before_s = cfg.read_text(encoding="utf-8"), sea.read_text(encoding="utf-8")
    assert m.main(["--at", "2026-11-19T06:30:00Z"]) == 0
    j = json.loads(cfg.read_text(encoding="utf-8"))
    assert j["clean_record_start_utc"] == "2026-11-19T06:30:00Z"
    assert "2026-11-18" in j["_owner_confirmed"] and "count them" in j["_owner_confirmed"]
    assert 'OWNER_DECISION = "2026-11-19"' in sea.read_text(encoding="utf-8")
    changed_c = [a for a, b in zip(before_c.splitlines(), cfg.read_text(encoding="utf-8").splitlines()) if a != b]
    changed_s = [a for a, b in zip(before_s.splitlines(), sea.read_text(encoding="utf-8").splitlines()) if a != b]
    assert len(changed_c) == 2 and len(changed_s) == 1
    assert len(before_c.splitlines()) == len(cfg.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("bad", ["2026-11-19", "2026-11-19 06:30:00", "2026-11-19T06:30:00+02:00", ""])
def test_refuses_a_value_that_is_not_a_utc_instant(tmp_path, monkeypatch, bad):
    m, cfg, _ = _mod(tmp_path, monkeypatch)
    before = cfg.read_bytes()
    with pytest.raises(SystemExit):
        m.main(["--at", bad])
    assert cfg.read_bytes() == before
