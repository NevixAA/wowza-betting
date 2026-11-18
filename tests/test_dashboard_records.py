"""Owner 2026-11-17: Bars page, Fixes page, Clean Record (from 21.11) as its own page, old pages labelled
"Season to 20.11 (pre-repair models)" and never changed.

Every new page is rendered headless (AppTest) on the committed data AND on empty data (its file missing).
"""
import json
import re
import shutil
from pathlib import Path

import pytest

from tests.clean_record_pin import pinned_start  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "reports" / "dashboard" / "bars_statistics.json"
LOG = ROOT / "reports" / "dashboard" / "repairs_log.json"


def _render(name: str):
    from streamlit.testing.v1 import AppTest
    page = next((ROOT / "pages").glob(name))
    at = AppTest.from_file(str(page), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title] \
        + [s.value for s in at.subheader] + [e.label for e in at.expander]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(autouse=True)
def _admin(monkeypatch, request):
    # the record NAMES are checked on the start pinned at 2026-11-21 (tests/clean_record_pin.py); the committed
    # value (owner 2026-11-18: the go-live merge instant) is checked for consistency by test_no_second_start_date
    if request.node.name != "test_no_second_start_date":
        request.getfixturevalue("pinned_start")
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


# ── the record names and the one start date ──────────────────────────────────────────────────────────
def test_labels_follow_the_start_date():
    from dashboard_data import season
    assert season.season_label() == "Season to 20.11 (pre-repair models)"
    assert season.clean_label() == "Clean Record (from 21.11)"
    assert season.starts_text() == "starts 21.11"


def test_no_second_start_date():
    """The owner decision used for NAMES only must equal the configured start wherever the tree has one."""
    from dashboard_data import season
    cfg = season.configured_start()
    if cfg is not None:
        assert cfg == season.OWNER_DECISION, (cfg, season.OWNER_DECISION)


def test_old_pages_get_the_season_label_only_without_the_split():
    from dashboard_data import season
    for p in season.PRE_REPAIR_PAGES:
        assert (ROOT / "pages" / p).exists(), p
    assert season.page_label(str(ROOT / "pages" / "26_📏_Bars.py")) is None
    assert season.page_label(None) is None
    want = None if season.split_in_tree() else "Season to 20.11 (pre-repair models)"
    assert season.page_label(str(ROOT / "pages" / "5_🎯_Success_Rates.py")) == want


def test_page_header_draws_the_label_on_a_pre_repair_page(tmp_path):
    """dashboard_ui.page_header finds the calling page's file and labels it (no page file is edited)."""
    from dashboard_data import season
    if season.split_in_tree():
        pytest.skip("the go-live tree labels its sections itself")
    pages = tmp_path / "pages"
    pages.mkdir()
    page = pages / "5_🎯_Success_Rates.py"
    page.write_text(f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nimport dashboard_ui as ui\n"
                    "ui.page_header('Success Rates', 'x')\n", encoding="utf-8")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception
    assert "Season to 20.11 (pre-repair models)" in _text(at)
    other = pages / "26_📏_Bars.py"
    other.write_text(page.read_text(encoding="utf-8"), encoding="utf-8")
    at = AppTest.from_file(str(other), default_timeout=60).run()
    assert "Season to 20.11" not in _text(at)


# ── statistics only ─────────────────────────────────────────────────────────────────────────────────
PRICE_KEYS = re.compile(r"(^|_)(odds|price|close|closing|entry_odds|market_odds|bet_odds|snapshot)($|_)", re.I)


def _keys(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(x, list):
        for v in x:
            _keys(v, out)
    return out


def test_bars_statistics_hold_no_price():
    d = json.loads(STATS.read_text(encoding="utf-8"))
    keys = _keys(d, set())
    # the only odds-named fields are a cell's odds RANGE rule (a filter, not a price)
    assert {k for k in keys if PRICE_KEYS.search(k)} <= {"odds_min", "odds_max"}
    ids = {p["proposal_id"]: p for p in d["proposals"]}
    assert set(ids) == {"TP-2026-11-15-bade2c0b", "TP-2026-11-16-500964af"}
    assert ids["TP-2026-11-15-bade2c0b"]["state"] == "APPROVED"
    assert ids["TP-2026-11-16-500964af"]["state"] == "AWAITING_OWNER"
    for p in d["proposals"]:
        assert len(p["source"]["commit"]) == 40
        assert sum(p["counts"].values()) == len(p["cells"])
        assert {c["status"] for c in p["cells"]} <= {"PROVEN", "PROMISING", "UNPROVEN", "OFF"}
        for c in p["cells"]:
            if c["status"] == "OFF":
                assert c["stake_state"] == "OFF" and not c["stake_fraction"]


def test_stake_plan_has_no_staked_cell_only_the_top3_flat_stake():
    """Owner 2026-11-17 ("Yes, adopt"): no cell stakes on its own; every TOP-3 tip carries 0.5% flat, placed manually
    (was: the 3 PROMISING cells, 2026-11-16)."""
    from dashboard_data import bars_stats_view as B
    d = B.load()["data"]
    assert len(B.stake_plan_frame(d)) == 0
    plan = d["stake_plan"]
    assert plan["decided"] == "2026-11-17" and "every TOP-3 tip" in plan["rule"] and "0.5%" in plan["rule"]
    assert "paper" not in plan["rule"].lower()
    for p in d["proposals"]:
        assert not any(c["stake_fraction"] for c in p["cells"])


def test_repairs_log_is_cited_and_newest_first():
    d = json.loads(LOG.read_text(encoding="utf-8"))
    dates = [i["date"] for i in d["items"]]
    assert dates == sorted(dates, reverse=True)
    for i in d["items"]:
        assert i["refs"], i["title"]
        assert i["what_was_wrong"] and i["what_changed"] and i["where"]


# ── the pages render, with real data and with empty data ─────────────────────────────────────────────
def test_bars_page_renders_with_the_committed_statistics():
    at = _render("26_*Bars.py")
    t = _text(at)
    assert "AWAITING OWNER APPROVAL ON NOV 20" in t and "APPROVED BY THE OWNER" in t
    assert "TP-2026-11-15-bade2c0b" in t and "TP-2026-11-16-500964af" in t
    assert "Season to 20.11" not in t


def test_fixes_page_renders_with_the_committed_log():
    at = _render("27_*Fixes.py")
    t = _text(at)
    d = json.loads(LOG.read_text(encoding="utf-8"))
    for i in d["items"]:
        assert i["title"] in t


@pytest.mark.parametrize("page,src", [("26_*Bars.py", STATS), ("27_*Fixes.py", LOG)])
def test_new_pages_render_without_their_file(tmp_path, page, src):
    bak = tmp_path / src.name
    shutil.move(src, bak)
    try:
        at = _render(page)
        assert "ERROR" in _text(at) and "cannot be read" in _text(at)
    finally:
        shutil.move(bak, src)


def test_clean_record_page_before_the_split_is_an_honest_empty_state():
    from dashboard_data import season
    at = _render("24_*Clean_Record.py")
    t = _text(at)
    assert "Clean Record (from 21.11)" in t
    if not season.split_in_tree():
        assert "starts 21.11" in t.lower() or "STARTS 21.11" in t
        assert "Not enough data yet" in t
        assert not at.metric                                     # no number before the clean system exists


def test_calibration_variants_are_shown_as_not_adopted():
    d = json.loads(STATS.read_text(encoding="utf-8"))
    (na,) = d["not_adopted"]
    assert na["proposal_id"] == "TP-2026-11-16-9cef8a02" and na["state"] == "NOT_ADOPTED"
    assert len(na["commit"]) == 40
    assert all(r["PROVEN"] == 0 for r in na["counts"])
    for r in na["counts"]:                                       # 194 cells per variant (HT cells OFF included)
        assert r["PROVEN"] + r["PROMISING"] + r["UNPROVEN"] + r["OFF"] == 194, r
    t = _text(_render("26_*Bars.py"))
    assert "TESTED · NOT ADOPTED" in t and "TP-2026-11-16-9cef8a02" in t


def test_started_follows_the_start_date():
    import datetime as dt
    from dashboard_data import season
    s = season.start_date()
    assert not season.started(s - dt.timedelta(days=1))
    assert season.started(s) and season.started(s + dt.timedelta(days=30))


def test_clean_record_page_after_the_start_without_the_split_shows_no_figure(monkeypatch):
    from dashboard_data import season
    if season.split_in_tree():
        pytest.skip("the go-live tree computes the record")
    monkeypatch.setattr(season, "started", lambda today=None: True)   # the page reads this module
    at = _render("24_*Clean_Record.py")
    t = _text(at)
    assert "does not carry the clean-model split yet" in t
    assert not at.metric
