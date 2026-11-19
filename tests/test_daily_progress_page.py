"""pages/24_📬_Daily_Progress.py (owner 2026-11-15): the daily progress report on the dashboard.

Pins: ADMIN only and fail closed when login is dormant; the report is fetched from the private
wowza-exec repo (never mirrored, never in the manifest's data); every line is shown as written,
alerts marked; a missing token, an HTTP error or a wrong schema shows an error and NO numbers; the
token never reaches the page; a stale report is labelled. Rendered headlessly with AppTest, network
replaced by a fake urlopen.
"""
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/24_📬_Daily_Progress.py"
TOKEN = "github_pat_TESTTOKEN_never_shown_0123456789"


def _report(generated="2026-11-15T09:57:51Z"):
    return {
        "schema": "wowza-daily-progress/1",
        "subject": "Wowza daily progress 2026-11-15 - go-live in 6 days, 1 failed run(s), 2 item(s) to check",
        "report_date_israel": "2026-11-15", "generated_at_utc": generated,
        "end_date": "2026-11-21", "go_live_date": "2026-11-21", "n_alerts": 2,
        "sections": [
            {"title": "1. Models", "lines": [{"text": "Clean team models: 7 of 8 registry entries (current_champions.json)",
                                              "url": None, "level": 0, "alert": False}]},
            {"title": "2. Bars", "lines": [{"text": "new_format|BTTS|Argentina Primera Division: ACTIVE 1.25%",
                                            "url": None, "level": 1, "alert": False}]},
            {"title": "3. Results", "lines": [{"text": "Yesterday 2026-11-14 (old models - not in Clean Record; ...): 22 settled",
                                               "url": None, "level": 0, "alert": False}]},
            {"title": "4. Data collection", "lines": [{"text": "OddsAPI credits remaining: 2,037,623",
                                                       "url": None, "level": 0, "alert": False},
                                                      {"text": "Tests and gates on data-truth-100: RED",
                                                       "url": "https://github.com/NevixAA/wowza-betting/actions/runs/1",
                                                       "level": 0, "alert": True}]},
            {"title": "5. Problems (failed GitHub Actions runs, last 24 h)", "lines": [
                {"text": "wowza-exec: Collect - failure", "url": "https://github.com/NevixAA/wowza-exec/actions/runs/2",
                 "level": 0, "alert": True}]},
        ],
        "note": "OddsAPI data: counts and credits only (licence: no redistribution of prices).",
    }


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(files, seen):
    def urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization")))
        assert req.full_url.startswith("https://api.github.com/repos/NevixAA/wowza-exec/contents/")
        assert req.full_url.endswith("?ref=progress-report")
        name = req.full_url.split("/contents/")[1].split("?")[0]
        if name not in files:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        body = files[name]
        return _Resp(body if isinstance(body, bytes) else json.dumps(body).encode())
    return urlopen


def _render(monkeypatch, files, tier="admin", token=TOKEN, login=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import dashboard_access
    st.cache_data.clear()                     # the page caches fetches for 10 min; each test is fresh
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(files, seen))
    if tier is None:
        monkeypatch.delenv("WOWZA_TIER_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("WOWZA_TIER_OVERRIDE", tier)
    if login is not None:
        monkeypatch.setattr(dashboard_access, "login_configured", lambda: login)
    if token:
        monkeypatch.setenv("WOWZA_EXEC_REPORT_TOKEN", token)
    else:
        monkeypatch.delenv("WOWZA_EXEC_REPORT_TOKEN", raising=False)
    at = AppTest.from_file(str(ROOT / PAGE), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at, seen


def _all_text(at) -> str:
    parts = []
    for kind in ("title", "subheader", "markdown", "caption", "error", "warning", "info", "expander"):
        for el in getattr(at, kind, []):
            parts.append(str(getattr(el, "value", "")) + " " + str(getattr(el, "label", "")))
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def test_admin_sees_every_section_and_line(monkeypatch):
    at, seen = _render(monkeypatch, {"report.json": _report(), "history": [{"name": "2026-11-15.json"}]})
    txt = _all_text(at)
    for t in ("1. Models", "2. Bars", "3. Results", "4. Data collection", "5. Problems"):
        assert t in txt
    assert "old models - not in Clean Record" in txt
    assert "OddsAPI credits remaining: 2,037,623" in txt
    assert "🔴 Tests and gates on data" in txt                     # alert marked
    assert "[link](https://github.com/NevixAA/wowza-exec/actions/runs/2)" in txt
    assert {m.label: m.value for m in at.metric}["Items to check"] == "2"
    assert {m.label: m.value for m in at.metric}["Failed runs (24 h)"] == "1"
    assert TOKEN not in txt
    assert all(auth == f"Bearer {TOKEN}" for _, auth in seen)
    assert at.selectbox[0].options == ["Latest", "2026-11-15"]


def test_markdown_in_repo_text_is_literal(monkeypatch):
    r = _report()
    r["sections"][0]["lines"][0]["text"] = "evil [x](https://evil.example) *bold* $x$"
    r["sections"][1]["lines"][0]["url"] = "https://evil.example/run"
    at, _ = _render(monkeypatch, {"report.json": r})
    txt = _all_text(at)
    assert "\\[x\\]" in txt and "\\*bold\\*" in txt and "\\$x\\$" in txt
    assert "[link](https://evil.example" not in txt                # a non-GitHub url gets no link


@pytest.mark.parametrize("tier", ["free", "pro"])
def test_free_and_pro_never_load_the_report(monkeypatch, tier):
    at, seen = _render(monkeypatch, {"report.json": _report()}, tier=tier)
    assert seen == []
    assert "2,037,623" not in _all_text(at)


def test_dormant_login_fails_closed_even_though_the_tier_is_admin(monkeypatch):
    """No [auth] -> dashboard_access says 'admin' for everyone; this page still refuses."""
    at, seen = _render(monkeypatch, {"report.json": _report()}, tier=None, login=False)
    assert seen == []
    assert "Login is not configured" in _all_text(at)


def test_missing_token_or_http_error_shows_no_numbers(monkeypatch):
    at, seen = _render(monkeypatch, {"report.json": _report()}, token="")
    txt = _all_text(at)
    assert seen == [] and "no read token configured" in txt and "2,037,623" not in txt
    at, _ = _render(monkeypatch, {})
    txt = _all_text(at)
    assert "HTTP 404" in txt and "Nothing is shown in its place" in txt and TOKEN not in txt


def test_wrong_schema_is_refused(monkeypatch):
    r = _report()
    r["schema"] = "something-else/9"
    at, _ = _render(monkeypatch, {"report.json": r})
    txt = _all_text(at)
    assert "expected 'wowza-daily-progress/1'" in txt and "2,037,623" not in txt


def test_old_report_is_labelled_stale(monkeypatch):
    at, _ = _render(monkeypatch, {"report.json": _report(generated="2026-11-01T04:20:00Z")})
    txt = _all_text(at)
    assert "STALE" in txt or "The daily report ended" in txt


def test_page_is_admin_only_and_never_in_the_mirrored_data():
    import dashboard_access as A
    assert {p: n for p, _t, _i, n in A.PAGES}[PAGE] == "admin"
    man = json.loads((ROOT / "deploy" / "dashboard_manifest.json").read_text(encoding="utf-8"))
    data = [e["path"] if isinstance(e, dict) else e for e in man.get("data", [])]
    assert not [p for p in data if "daily_progress" in p.lower() or "progress-report" in p.lower()
                   or Path(p).name == "report.json"]
    src = (ROOT / PAGE).read_text(encoding="utf-8")
    assert "raw.githubusercontent" not in src and "github_pat_" not in src
