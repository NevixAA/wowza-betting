"""Owner P0 item 6 (2026-11-22): the synthetic props ROI file is quarantined.

output/player_props_calibration.json holds ROI against synthetic implied prices (base rates). It is
kept for audit with an in-file INVALID_FOR_REAL_BETTING_ROI marker; src/integrity/synthetic_roi is its
only reader and refuses every betting-performance purpose.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.integrity import synthetic_roi as S

ROOT = Path(__file__).resolve().parents[1]
FILE = "output/player_props_calibration.json"


def test_file_kept_with_in_file_marker():
    d = json.loads((ROOT / FILE).read_text(encoding="utf-8"))
    q = d[S.MARKER_KEY]
    assert q["status"] == S.MARKER == "INVALID_FOR_REAL_BETTING_ROI"
    assert q["usable_for"] == ["audit"]
    assert set(S.REFUSED_PURPOSES) <= set(q["refused_for"])
    # the audit content is kept (not deleted, not rewritten)
    assert {"goals", "assists", "sot2", "sot3"} <= set(d)


def test_every_quarantined_file_carries_the_marker():
    for rel in S.QUARANTINED:
        assert S.load_for_audit(rel)[S.MARKER_KEY]["status"] == S.MARKER


@pytest.mark.parametrize("purpose", S.REFUSED_PURPOSES)
def test_loader_refuses_every_real_purpose(purpose):
    with pytest.raises(S.SyntheticROIError):
        S.load_quarantined(FILE, purpose)
    with pytest.raises(S.SyntheticROIError):
        S.assert_not_quarantined(ROOT / FILE, purpose)


def test_audit_read_needs_the_marker(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"goals": {"roi_%": 1.0}}), encoding="utf-8")
    with pytest.raises(S.SyntheticROIError):
        S.load_for_audit(p)


def _code_files():
    skip = ("tests/", "_archive/", "reports/", "docs/", ".git/", ".claude/")
    for p in ROOT.rglob("*"):
        if p.suffix not in (".py", ".yml", ".yaml", ".sh") or not p.is_file():
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith(skip) or "/." in rel or rel == "src/integrity/synthetic_roi.py":
            continue
        yield rel, p


def test_no_code_reads_the_synthetic_roi_file():
    """No dashboard page, Telegram path, promotion gate, staking, edge validation or workflow names it."""
    hits = [rel for rel, p in _code_files()
            if re.search(r"player_props_calibration", p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == [], f"only src/integrity/synthetic_roi.py may read the quarantined file: {hits}"


def test_no_code_reads_synthetic_roi_columns_from_a_props_calibration_source():
    """The file's ROI keys never appear beside a props-calibration reader (dashboard / Telegram)."""
    for rel, p in _code_files():
        if not rel.startswith(("pages/", "dashboard_data/", "telegram_bot/", "app.py", "dashboard_ui.py")):
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        assert not ("props_calibration" in t and ("roi_%" in t or "implied_odds" in t)), rel
