"""Phase 2 Part 10 (cross-repository contracts): the dashboard must never show a P&L read from another
repo when that payload is schema-incompatible or cannot reproduce its own P&L, and a failing
cross-repo read must not be swallowed into "no data yet".

Failing condition (each test): the adapter passes an incompatible / non-reproducible Pro payload
through to the page, or a raising Pro read is indistinguishable from a missing artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard_data import pro

ROOT = Path(__file__).resolve().parents[1]


def _good() -> dict:
    # Mirrors the shape Pro's src/pipelines/tip_scoreboard.py writes (Pro 78580b42, output/tip_scoreboard.json).
    return {"generated_at": "2026-11-01T18:00:53Z",
            "one_x_two": {"sent": 51, "sent_excl_paper": 45, "pending": 6,
                          "headline": {"settled": 39, "won": 10, "hit_rate": 0.2564, "break_even": 0.3008,
                                       "model_claimed": 0.412, "avg_odds": 3.78, "units": -6.4,
                                       "roi": -0.1641, "roi_ci90": [-0.4985, 0.2342],
                                       "negative_ev_when_sent": 1},
                          "last_10": []},
            "bet_builder": {"sent": 0}}


def _serve(monkeypatch, tmp_path, payload) -> None:
    f = tmp_path / "tip_scoreboard.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pro, "_f", lambda rel: f if rel == "output/tip_scoreboard.json" else None)


def test_compatible_scoreboard_passes_unchanged(monkeypatch, tmp_path):
    _serve(monkeypatch, tmp_path, _good())
    j = pro.tip_scoreboard()
    assert "schema_error" not in j
    assert j["one_x_two"]["headline"]["units"] == -6.4


@pytest.mark.parametrize("mutate, why", [
    (lambda j: j["one_x_two"]["headline"].pop("units"), "units"),
    (lambda j: j["one_x_two"]["headline"].update(units="-6.4u"), "units"),
    (lambda j: j["one_x_two"]["headline"].update(won=40), "won"),
    # units must equal roi x settled (Pro computes roi = mean of the same flat-1u units): -0.1641*39 = -6.4
    (lambda j: j["one_x_two"]["headline"].update(units=+6.4), "reproduce"),
    (lambda j: j["one_x_two"]["headline"].update(hit_rate=0.9), "reproduce"),
    (lambda j: j["one_x_two"].update(headline=[1, 2]), "headline"),
])
def test_incompatible_scoreboard_is_rejected(monkeypatch, tmp_path, mutate, why):
    j = _good()
    mutate(j)
    _serve(monkeypatch, tmp_path, j)
    out = pro.tip_scoreboard()
    assert "schema_error" in out, f"incompatible payload ({why}) reached the page"
    assert why in out["schema_error"]
    # nothing that could be rendered as P&L survives
    assert "one_x_two" not in out and "bet_builder" not in out


def test_unsettled_headline_is_still_compatible(monkeypatch, tmp_path):
    j = _good()
    j["one_x_two"]["headline"] = {"settled": 0}
    _serve(monkeypatch, tmp_path, j)
    assert "schema_error" not in pro.tip_scoreboard()


def test_pro_page_renders_schema_mismatch_not_no_data():
    page = next((ROOT / "pages").glob("7_*Pro.py")).read_text(encoding="utf-8")
    assert 'sb.get("schema_error")' in page and "SCHEMA_MISMATCH" in page


def test_load_all_records_a_raising_read(monkeypatch):
    from dashboard_data import overview_view
    from dashboard_data import research_view as RV
    monkeypatch.setattr(overview_view, "prefetch_pro_listings", lambda: None)   # no network in a test

    class FakePro:
        def tip_scoreboard(self):
            raise ValueError("boom")

    monkeypatch.setattr(RV, "CALLS", ("tip_scoreboard",), raising=False)
    out = RV.load_all(FakePro(), max_workers=1)
    assert out.get("tip_scoreboard") == {}
    assert "tip_scoreboard" in (out.get("_errors") or {}), "a raising cross-repo read was swallowed silently"
