"""Sectioned navigation (owner architecture brief 2026-10-25, section 30).

The sidebar is grouped by function (System Health, Production, League Research, Model Lab, Market
Intelligence, Bankroll, Fantasy, V11 Research, Account). These tests pin three things:
  1. every page in dashboard_access.PAGES sits in exactly one section, and every page file is routed;
  2. the regrouping did not change WHO sees WHAT: the pages each tier gets (path, title, icon,
     default) are identical to the snapshot taken before the change, plus new pages that are
     admin-only;
  3. new architecture pages render headlessly and stay VIEWERS (small artifacts only).
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# pages_for(tier) BEFORE 2026-10-25 (st.Page recorded as (path, title, icon, default)).
BEFORE = {
    'free': [
        ('pages/free_tips.py', "Today's free tips", '🎁', True),
        ('pages/5_🎯_Success_Rates.py', 'Success Rates', '✅', False),
        ('pages/account.py', 'Account', '👤', False),
    ],
    'pro': [
        ('pages/free_tips.py', "Today's free tips", '🎁', False),
        ('pages/5_🎯_Success_Rates.py', 'Success Rates', '✅', False),
        ('pages/00_🎯_Home.py', 'Home', '🎯', True),
        ('pages/1_📊_Dashboard.py', 'Tips', '📊', False),
        ('pages/17_🏟️_Leagues.py', 'Leagues', '🏟️', False),
        ('pages/11_👤_Player_Props.py', 'Player Props', '👤', False),
        ('pages/13_⚽_Fantasy.py', 'Fantasy', '⚽', False),
        ('pages/14_🧩_Bet_Builder.py', 'Bet Builder', '🧩', False),
        ('pages/6_⚡_Live.py', 'Live', '⚡', False),
        ('pages/12_💼_Portfolio.py', 'Portfolio', '💼', False),
        ('pages/16_💰_Bankroll.py', 'Bankroll', '💰', False),
        ('pages/account.py', 'Account', '👤', False),
    ],
    'admin': [
        ('pages/free_tips.py', "Today's free tips", '🎁', False),
        ('pages/5_🎯_Success_Rates.py', 'Success Rates', '✅', False),
        ('pages/00_🎯_Home.py', 'Home', '🎯', True),
        ('pages/1_📊_Dashboard.py', 'Tips', '📊', False),
        ('pages/17_🏟️_Leagues.py', 'Leagues', '🏟️', False),
        ('pages/11_👤_Player_Props.py', 'Player Props', '👤', False),
        ('pages/13_⚽_Fantasy.py', 'Fantasy', '⚽', False),
        ('pages/14_🧩_Bet_Builder.py', 'Bet Builder', '🧩', False),
        ('pages/6_⚡_Live.py', 'Live', '⚡', False),
        ('pages/12_💼_Portfolio.py', 'Portfolio', '💼', False),
        ('pages/16_💰_Bankroll.py', 'Bankroll', '💰', False),
        ('pages/0_🏠_Overview.py', 'Overview', '🏠', False),
        ('pages/9_🎓_Model_Performance.py', 'Model Performance', '🎓', False),
        ('pages/4_ℹ️_Model_Info.py', 'Model Info', 'ℹ️', False),
        ('pages/7_🔬_Pro.py', 'Pro research', '🔬', False),
        ('pages/8_🛰️_V11_Market.py', 'V11 Market', '🛰️', False),
        ('pages/10_🤖_Agent_Analysis.py', 'Agent Analysis', '🤖', False),
        ('pages/15_🔭_League_Scout.py', 'League Scout', '🔭', False),
        ('pages/18_🔧_League_Research.py', 'League Repair Lab', '🔧', False),
        ('pages/account.py', 'Account', '👤', False),
    ],
}

# Display-label changes the owner asked for (visual-redesign brief 2026-10-25). File names unchanged.
RELABELLED = {"pages/7_🔬_Pro.py": ("Pro research", "Research"),
              "pages/8_🛰️_V11_Market.py": ("V11 Market", "Market Lab")}

# The owner's grouping by user intent (2026-10-25).
EXPECTED_SECTION = {
    "pages/00_🎯_Home.py": "Today", "pages/free_tips.py": "Today", "pages/1_📊_Dashboard.py": "Today",
    "pages/6_⚡_Live.py": "Today",
    "pages/17_🏟️_Leagues.py": "Performance", "pages/12_💼_Portfolio.py": "Performance",
    "pages/16_💰_Bankroll.py": "Performance", "pages/5_🎯_Success_Rates.py": "Performance",
    "pages/13_⚽_Fantasy.py": "Tools", "pages/11_👤_Player_Props.py": "Tools", "pages/14_🧩_Bet_Builder.py": "Tools",
    "pages/18_🔧_League_Research.py": "Research Lab", "pages/9_🎓_Model_Performance.py": "Research Lab",
    "pages/7_🔬_Pro.py": "Research Lab", "pages/8_🛰️_V11_Market.py": "Research Lab",
    "pages/15_🔭_League_Scout.py": "Research Lab", "pages/10_🤖_Agent_Analysis.py": "Research Lab",
    "pages/0_🏠_Overview.py": "System", "pages/4_ℹ️_Model_Info.py": "System", "pages/account.py": "System",
}

# Architecture pages added 2026-10-25: (path, section). All admin-only.
NEW_PAGES = []


def _access():
    import dashboard_access
    return dashboard_access


class _Rec:
    """Stand-in for st.Page: records what pages_for asked for without a Streamlit runtime."""
    def __init__(self, path, title=None, icon=None, default=False, **kw):
        self.path, self.title, self.icon, self.default = path, title, icon, bool(default)


def _nav(monkeypatch, tier):
    A = _access()
    monkeypatch.setattr(A.st, "Page", _Rec)
    return A.pages_for(tier)


def test_every_page_in_exactly_one_known_section():
    A = _access()
    paths = [p for p, *_ in A.PAGES]
    assert len(paths) == len(set(paths)), "a page is listed twice in PAGES"
    assert set(paths) == set(A.SECTION_OF), "PAGES and SECTION_OF disagree"
    assert set(A.SECTION_OF.values()) <= set(A.NAV_SECTIONS)
    assert A.NAV_SECTIONS == ["Today", "Performance", "Tools", "Research Lab", "System"]
    for path, section in EXPECTED_SECTION.items():
        assert A.section_of(path) == section, path


def test_every_page_file_is_routed():
    """No page file is orphaned: every pages/*.py (not *.disabled) is in navigation."""
    A = _access()
    files = {f"pages/{p.name}" for p in (ROOT / "pages").glob("*.py")}
    assert files == {p for p, *_ in A.PAGES}


@pytest.mark.parametrize("tier", ["free", "pro", "admin"])
def test_navigation_sections_and_tiers_unchanged(monkeypatch, tier):
    A = _access()
    nav = _nav(monkeypatch, tier)
    assert list(nav) == [s for s in A.NAV_SECTIONS if s in nav], "sections out of order"
    assert all(nav.values()), "an empty section was emitted"
    got = [(p.path, p.title, p.icon, p.default) for ps in nav.values() for p in ps]
    assert len(got) == len({g[0] for g in got}), "a page appears in two sections"
    for s, ps in nav.items():
        assert all(A.section_of(p.path) == s for p in ps)
    new = {p for p, _ in NEW_PAGES}
    old_got = sorted(g for g in got if g[0] not in new)
    before = sorted((p, RELABELLED[p][1] if p in RELABELLED else t, i, d) for p, t, i, d in BEFORE[tier])
    assert old_got == before, "tier visibility (or icon/default/unapproved label) of an existing page changed"
    new_got = {g[0] for g in got if g[0] in new}
    assert new_got == (new if tier == "admin" else set()), "new pages must be admin-only"
    assert sum(g[3] for g in got) == 1                              # exactly one default page


def test_new_pages_are_admin_and_in_their_section():
    A = _access()
    need = {p: n for p, _t, _i, n in A.PAGES}
    for path, section in NEW_PAGES:
        assert need[path] == "admin", path
        assert A.section_of(path) == section, path
        src = (ROOT / path).read_text(encoding="utf-8")
        assert '_require_tier("admin")' in src, path


# ── new pages stay viewers ────────────────────────────────────────────────────────────────────
FORBIDDEN_IMPORTS = re.compile(
    r"^(src\.(model|model_validation|promotion_gate|predict|backtest|data_loader|feature_engineering|"
    r"research_lock|system_health|edge_engine_shadow|market_edge|stake_plan)|pipeline|retrain|"
    r"player_model|sklearn|lightgbm|xgboost|subprocess|git)(\.|$)")


def _view_sources(path):
    """The page plus the dashboard_data helper modules it imports."""
    src = (ROOT / path).read_text(encoding="utf-8")
    out = [(path, src)]
    for m in re.findall(r"from dashboard_data import ([\w, ]+)", src):
        for name in [x.strip() for x in m.split(",")]:
            f = ROOT / "dashboard_data" / f"{name}.py"
            if f.exists():
                out.append((f"dashboard_data/{name}.py", f.read_text(encoding="utf-8")))
    return out


@pytest.mark.parametrize("path", [p for p, _ in NEW_PAGES] or ["<none>"])
def test_new_pages_read_only_small_artifacts(path):
    if path == "<none>":
        pytest.skip("no new pages yet")
    for name, src in _view_sources(path):
        assert "read_parquet" not in src, f"{name}: a new page must not read a parquet"
        assert "player_history" not in src, f"{name}: player_history is never read at render time"
        assert ".pkl" not in src and "joblib" not in src, f"{name}: no model artifacts at render"
        assert "getmtime" not in src and "st_mtime" not in src, f"{name}: freshness from recorded timestamps only"
        for node in ast.walk(ast.parse(src)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                assert not FORBIDDEN_IMPORTS.match(m), f"{name}: imports {m}"
        if name.startswith("pages/"):
            assert "st.cache_data" in src, f"{name}: reads must be cached"


@pytest.mark.parametrize("path", [p for p, _ in NEW_PAGES] or ["<none>"])
def test_new_pages_render(monkeypatch, path):
    if path == "<none>":
        pytest.skip("no new pages yet")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    at = AppTest.from_file(str(ROOT / path), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page rendered no title"
