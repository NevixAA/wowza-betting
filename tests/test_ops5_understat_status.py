"""Understat collector records what the source did (run 41440689838: 60 of 60 URLs failed DNS and the
run ended 'nothing to map' / 'nothing new to commit' with no status). No network: the session is fake."""
import json
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_understat_shots as C  # noqa: E402


class _DeadSession:
    """Every request fails the way the GitHub runner did."""
    def __init__(self):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        raise requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='understat.com', port=443): Max retries exceeded "
            "(Caused by NameResolutionError(\"Failed to resolve 'understat.com'\"))")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "COLLECT_DIR", tmp_path)
    monkeypatch.setattr(C, "STATUS", tmp_path / "understat_status.json")
    monkeypatch.setattr(C, "STATUS_LOG", tmp_path / "understat_status_log.jsonl")
    monkeypatch.setattr(C, "build_maps", lambda: {})
    monkeypatch.setattr(C, "fetch_leagues", lambda s: pytest.fail("no page request after a failed preflight"))
    return tmp_path


def test_unreachable_source_is_recorded_not_silent(sandbox, monkeypatch, capsys):
    dead = _DeadSession()
    monkeypatch.setattr(requests, "Session", lambda: dead)
    assert C.main(["--max-matches", "10"]) == 0
    doc = json.loads((sandbox / "understat_status.json").read_text(encoding="utf-8"))
    assert doc["status"] == "SOURCE_UNREACHABLE"
    assert doc["requests"] == 1 and doc["ok"] == 0 and doc["unreachable"] == 1
    assert "NameResolutionError" in doc["sample_error"] or "Failed to resolve" in doc["sample_error"]
    assert dead.calls == 1                                   # stops after one probe, not 60 DNS time-outs
    assert len((sandbox / "understat_status_log.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert "::error title=Understat source::Understat SOURCE_UNREACHABLE" in capsys.readouterr().out


@pytest.mark.parametrize("stats, n_lg, n_m, want", [
    ({"requests": 60, "ok": 0, "unreachable": 60, "timeout": 0}, 0, 0, "SOURCE_UNREACHABLE"),
    ({"requests": 10, "ok": 0, "unreachable": 2, "http_error": 8}, 0, 0, "SOURCE_FAILED"),
    ({"requests": 10, "ok": 10, "pages_without_matches": 10}, 0, 0, "SOURCE_EMPTY"),
    ({"requests": 10, "ok": 7, "unreachable": 3}, 120, 5, "PARTIAL"),
    ({"requests": 10, "ok": 10}, 120, 5, "OK"),
    ({"requests": 1, "ok": 1}, 0, 0, "OK"),                 # nothing left to ask
])
def test_source_status_words(stats, n_lg, n_m, want):
    assert C.source_status(stats, n_lg, n_m) == want
