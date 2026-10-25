"""The dashboard has ONE design system (visual redesign 2026-10-25, docs/DASHBOARD_DESIGN_SYSTEM_2026_10.md).

Pins: the semantic token set, the theme file mirroring it, status chips that never rely on colour
alone, the standard number formats, and that dashboard_data/ui.py is only a re-export (no second set
of helpers creeping back).
"""
import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TOKENS = {"BACKGROUND", "SURFACE", "SURFACE_ELEVATED", "SURFACE_MUTED", "BORDER", "BORDER_STRONG",
                   "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED", "PRIMARY", "PRIMARY_SOFT", "POSITIVE",
                   "NEGATIVE", "WARNING", "INFO", "LIVE", "PAPER", "RESEARCH", "BLOCKED", "NO_DATA", "V9",
                   "PRO", "V11", "FANTASY"}

# config.toml key -> token it must equal, in BOTH [theme.dark] and [theme.light]
THEME_MAP = {"primaryColor": "PRIMARY", "backgroundColor": "BACKGROUND", "secondaryBackgroundColor": "SURFACE_MUTED",
             "textColor": "TEXT_PRIMARY", "borderColor": "BORDER", "dataframeBorderColor": "BORDER",
             "greenColor": "POSITIVE", "redColor": "NEGATIVE", "orangeColor": "WARNING", "violetColor": "RESEARCH",
             "blueColor": "PRIMARY", "grayColor": "NO_DATA"}


def _ui():
    import dashboard_ui
    return dashboard_ui


def test_token_sets_are_complete_and_identical_in_shape():
    ui = _ui()
    assert REQUIRED_TOKENS <= set(ui.DARK), REQUIRED_TOKENS - set(ui.DARK)
    assert set(ui.DARK) == set(ui.LIGHT)
    for v in list(ui.DARK.values()) + list(ui.LIGHT.values()):
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?", v), v


def test_semantic_aliases_mean_what_the_brief_says():
    ui = _ui()
    for t in (ui.DARK, ui.LIGHT):
        assert t["LIVE"] == t["POSITIVE"] and t["PAPER"] == t["WARNING"] and t["BLOCKED"] == t["NEGATIVE"]
        assert t["RESEARCH"] not in (t["POSITIVE"], t["NEGATIVE"], t["WARNING"])


def test_theme_is_dark_only():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert "light" not in cfg, "owner brief 2026-10-25: one consistent dark theme"


@pytest.mark.parametrize("mode", ["dark"])
def test_theme_file_mirrors_tokens(mode):
    ui = _ui()
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert cfg["base"] == "dark"
    tok = ui.DARK if mode == "dark" else ui.LIGHT
    for key, name in THEME_MAP.items():
        assert cfg[mode][key].upper() == tok[name].upper(), f"[theme.{mode}] {key} != {name}"
    assert cfg["chartCategoricalColors"] == ui.CATEGORICAL


def test_every_status_has_icon_and_word():
    ui = _ui()
    for k in ("ACTIVE", "PAPER", "BLOCKED", "PROTECTED", "REPAIR", "LEARNING", "LIVE", "RESEARCH", "SHADOW",
              "QUALIFIED", "OBSERVE", "NO BET", "NO_DATA", "NOT_CONNECTED", "STALE", "ERROR"):
        icon, word, tone = ui.status_meta(k)
        assert icon.strip() and word.strip(), k
        chip = ui.status_chip(k)
        assert word in chip and icon in chip, chip
    assert "🛡" in ui.status_chip("PROTECTED") and ":red-" not in ui.status_chip("PROTECTED")


def test_legacy_tiers_are_neutral_never_red():
    ui = _ui()
    for t in ("SNIPER", "MARKSMAN", "VALUABLE"):
        assert ui.status_chip(t).startswith(":gray-badge["), t


def test_empty_states_are_four_different_claims():
    ui = _ui()
    words = {ui.status_meta(k)[1] for k in ("NO_DATA", "NOT_CONNECTED", "STALE", "ERROR")}
    assert words == {"NO DATA", "FEED NOT CONNECTED", "STALE", "ERROR"}


def test_number_formats():
    ui = _ui()
    assert ui.fmt_roi(0.084) == "+8.4%"
    assert ui.fmt_units(12.4) == "+12.40u" and ui.fmt_units(-3) == "-3.00u"
    assert ui.fmt_odds(1.94) == "1.94"
    assert ui.fmt_prob(0.63) == "63%"
    assert ui.fmt_clv(1.2) == "+1.2%"
    assert ui.fmt_xpts(6.42) == "6.42"
    assert ui.fmt_money(8.5) == "£8.5m"
    assert ui.pct(0.55, 20) == "55.0% (n=20)"
    assert ui.fmt_roi(None) == ui.fmt_odds(float("nan")) == "—"


def test_small_sample_marker_carries_n():
    ui = _ui()
    assert ui.sample_marker(12) and "n=12" in ui.sample_marker(12)
    assert ui.sample_marker(500) == ""
    assert "n=45" in ui.evidence_badge("TRUE_FORWARD", 45, span="6 GW", state="ACCRUING")


def test_data_ui_is_only_a_reexport():
    """No second helper set: dashboard_data/ui.py may define `table` and nothing else."""
    tree = ast.parse((ROOT / "dashboard_data" / "ui.py").read_text(encoding="utf-8"))
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
    assert defs == ["table"] and not assigns
    from dashboard_data import ui as data_ui
    ui = _ui()
    for name in ("pct", "units", "points", "kpi", "state_banner", "freshness_chip", "absent", "pnl_bars",
                 "compare_bars", "cumulative_line", "polarity", "repo_colour", "NEUTRAL", "is_dark"):
        assert getattr(data_ui, name) is getattr(ui, name), name


def test_redesigned_pages_invent_no_colours_and_no_dividers():
    for name in ("00_🎯_Home.py", "1_📊_Dashboard.py"):
        src = (ROOT / "pages" / name).read_text(encoding="utf-8")
        assert not re.search(r"#[0-9A-Fa-f]{6}\b", src), f"{name} hardcodes a hex colour"
        assert "<style" not in src, f"{name} carries its own CSS"
        assert "st.divider()" not in src and 'st.markdown("---")' not in src, f"{name} uses dividers"

