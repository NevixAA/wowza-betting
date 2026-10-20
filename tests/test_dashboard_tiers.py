"""Free / Pro / Admin tiers (2026-10-18): who sees which page, fail-closed defaults, the free 2 tips."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _access():
    import dashboard_access
    return dashboard_access


def test_dormant_until_login_is_configured(monkeypatch):
    """No [auth] secrets -> the app is exactly as before tiers existed (everything visible), so merging
    the tiers can never lock the owner into Free (owner 2026-10-20)."""
    A = _access()
    monkeypatch.delenv("WOWZA_TIER_OVERRIDE", raising=False)
    monkeypatch.setattr(A, "login_configured", lambda: False)
    assert A.current_tier() == "admin"


def test_with_login_unknown_viewer_is_free(monkeypatch):
    A = _access()
    monkeypatch.delenv("WOWZA_TIER_OVERRIDE", raising=False)
    monkeypatch.setattr(A, "login_configured", lambda: True)
    monkeypatch.setattr(A, "user_email", lambda: None)
    assert A.current_tier() == "free"                      # login on, not logged in -> free
    monkeypatch.setattr(A, "user_email", lambda: "stranger@example.com")
    monkeypatch.setattr(A, "_emails", lambda key: {"owner@example.com"} if key == "admins" else set())
    assert A.current_tier() == "free"                      # logged in, on neither list -> free
    monkeypatch.setattr(A, "user_email", lambda: "owner@example.com")
    assert A.current_tier() == "admin"
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "superuser")   # unknown override value ignored
    monkeypatch.setattr(A, "user_email", lambda: None)
    assert A.current_tier() == "free"


def test_page_sets_nest():
    A = _access()
    free = {p for p, *_ , need in A.PAGES if A.allows("free", need)}
    pro = {p for p, *_ , need in A.PAGES if A.allows("pro", need)}
    admin = {p for p, *_ , need in A.PAGES if A.allows("admin", need)}
    assert free < pro < admin
    assert free == {"pages/free_tips.py", "pages/5_🎯_Success_Rates.py", "pages/account.py"}
    assert {"pages/1_📊_Dashboard.py", "pages/16_💰_Bankroll.py", "pages/12_💼_Portfolio.py"} <= pro
    assert "pages/7_🔬_Pro.py" not in pro                     # research stays admin-only
    assert all((ROOT / p).exists() for p, *_ in A.PAGES)
    assert A.WINDOWS["free"] == [7] and A.WINDOWS["pro"] == [7, 14, 30, 60, 90]


def test_gated_pages_guard_themselves():
    A = _access()
    for p, *_, need in A.PAGES:
        if need == "free":
            continue
        src = (ROOT / p).read_text(encoding="utf-8")
        assert f'require("{need}")' in src or f'_require_tier("{need}")' in src, p


def test_free_tips_top_two_by_evidence():
    pick = _access().free_tip_pick
    plan = pd.DataFrame([
        {"date": "2026-10-18", "fixture": "2026-10-18|A|B", "league": "L", "track": "new_format|OU25", "market": "OU25",
         "side": "OVER", "signal_tier": "SNIPER", "odds": 2.1, "final_fraction": 0.0055},
        {"date": "2026-10-18", "fixture": "2026-10-18|C|D", "league": "L", "track": "new_format|BTTS", "market": "BTTS",
         "side": "YES", "signal_tier": "MARKSMAN", "odds": 2.0, "final_fraction": 0.0112},
        {"date": "2026-10-18", "fixture": "2026-10-18|E|F", "league": "L", "track": "new_format|OVER15", "market": "OVER15",
         "side": "OVER", "signal_tier": "SNIPER", "odds": 1.6, "final_fraction": 0.0055},
        {"date": "2026-10-18", "fixture": "2026-10-18|G|H", "league": "USA MLS", "track": "new_format|BTTS", "market": "BTTS",
         "side": "YES", "signal_tier": "SNIPER", "odds": 2.0, "final_fraction": 0.0},       # paper: never free-tipped
        {"date": "2026-10-19", "fixture": "2026-10-19|I|J", "league": "L", "track": "new_format|BTTS", "market": "BTTS",
         "side": "YES", "signal_tier": "SNIPER", "odds": 2.0, "final_fraction": 0.0137},
    ])
    policy = {"tracks": {"new_format|BTTS": {"evidence_score": 3.0}, "new_format|OVER15": {"evidence_score": 1.5},
                         "new_format|OU25": {"evidence_score": 1.0}}}
    tips, day, n = pick(plan, policy, pd.Timestamp("2026-10-18"))
    assert day == "2026-10-18" and n == 3 and len(tips) == 2
    assert tips["fixture"].tolist() == ["2026-10-18|C|D", "2026-10-18|E|F"]      # BTTS, then OVER15
    tips, day, _ = pick(plan, policy, pd.Timestamp("2026-10-19"))
    assert day == "2026-10-19" and len(tips) == 1
    assert pick(plan.iloc[0:0], policy, pd.Timestamp("2026-10-18"))[0].empty


@pytest.mark.parametrize("tier,first_title", [("free", "Today's free tips"), ("pro", None), ("admin", None)])
def test_router_runs_per_tier(monkeypatch, tier, first_title):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", tier)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    if first_title:
        assert any(first_title in t.value for t in at.title)
