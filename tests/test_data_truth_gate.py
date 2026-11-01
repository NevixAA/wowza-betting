"""Data truth stoplight: the ratchet may only go down (owner 2026-11-01, docs/STOPLIGHT.md).

    python -m pytest tests/test_data_truth_gate.py -q
"""
from __future__ import annotations

import json

from src.integrity import data_truth as DT

GREEN_CHECKS = [{"check": f"Data truth {i}: x", "status": "GREEN", "value": "", "detail": ""} for i in (1, 2, 3, 4)]


def _setup(tmp_path, monkeypatch, n_entries, ceiling):
    b, r = tmp_path / "baseline.json", tmp_path / "ratchet.json"
    b.write_text(json.dumps({"entries": [{"file": "f.py", "code": f"x{i}"} for i in range(n_entries)]}))
    r.write_text(json.dumps({"max_known_defaults": ceiling}))
    monkeypatch.setattr(DT, "BASELINE", b)
    monkeypatch.setattr(DT, "RATCHET", r)


def test_green_when_within_ceiling(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, 3, 3)
    assert DT.gate(previous_ceiling=3, checks=GREEN_CHECKS) == []


def test_baseline_above_ceiling_is_red(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, 4, 3)
    assert any("ratchet ceiling" in f for f in DT.gate(checks=GREEN_CHECKS))


def test_raising_the_ceiling_is_red(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, 3, 4)
    assert any("may only go down" in f for f in DT.gate(previous_ceiling=3, checks=GREEN_CHECKS))


def test_lowering_the_ceiling_is_green(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, 2, 2)
    assert DT.gate(previous_ceiling=3, checks=GREEN_CHECKS) == []


def test_red_rule_blocks_but_missing_data_does_not(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, 0, 0)
    red3 = [dict(c, status="RED") if c["check"].startswith("Data truth 3") else c for c in GREEN_CHECKS]
    assert DT.gate(checks=red3) == []                      # operational: watched by system health
    red2 = [dict(c, status="RED") if c["check"].startswith("Data truth 2") else c for c in GREEN_CHECKS]
    assert DT.gate(checks=red2)


def test_committed_ratchet_holds():
    """The committed baseline is within the committed ceiling (what CI enforces)."""
    assert len(DT._baseline()["entries"]) <= DT._ratchet()["max_known_defaults"]
