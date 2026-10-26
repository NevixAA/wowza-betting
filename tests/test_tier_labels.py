"""Tier display labels (owner decision 2026-10-26): SNIPER / MARKSMAN / VALUABLE are shown to people as
Big gap / Medium gap / Small gap, from ONE mapping (src/tier_labels.py). Display only.

Pinned here:
  1. the mapping and its helpers (labels, pass-through, display_frame never touches numbers);
  2. STATIC: no display module spells an old name inside a display string (a bare code literal such as
     "SNIPER" in a filter or dict key is data and is allowed);
  3. Telegram on committed data: every picture string and every message carries no old name, while
     the notified.json dedup keys and the ledgers stay exactly as they were (codes, untouched files).
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest

from src import tier_labels as TL

ROOT = Path(__file__).resolve().parents[1]
CODE = re.compile(r"\b(SNIPER|MARKSMAN|VALUABLE)\b")
LABEL_WORDS = re.compile(r"Big gap|Medium gap|Small gap|No tip", re.I)


# ── 1. mapping ────────────────────────────────────────────────────────────────────────────────────
def test_mapping_is_the_owner_decision():
    assert TL.TIER_LABEL == {"SNIPER": "Big gap", "MARKSMAN": "Medium gap", "VALUABLE": "Small gap",
                             "AVOID": "No tip"}
    assert TL.GAP_EXPLAINER == ("Gap = how far our probability is from the bookmaker's price. "
                                "It is not a confidence rating.")
    assert TL.STAKED_LABEL == "Big + Medium gap" and TL.PAPER_TIER_LABEL == "Small gap"
    assert not CODE.search(TL.GAP_EXPLAINER + TL.STAKED_LABEL + "".join(TL.TIER_LABEL.values()))


def test_helpers():
    assert TL.tier_label("SNIPER") == "Big gap" and TL.tier_label("marksman") == "Medium gap"
    assert TL.tier_label("VALUABLE", short=True) == "Small"
    assert TL.tier_label("PAPER") == "PAPER" and TL.tier_label(None) == ""
    assert TL.tier_tag("MARKSMAN").endswith("Medium gap")
    assert TL.relabel_text("Staked (SNIPER + MARKSMAN); VALUABLE paper") == \
        "Staked (Big gap + Medium gap); Small gap paper"
    assert TL.relabel_text("SNIPER_THRESHOLD") == "SNIPER_THRESHOLD"      # identifiers are not words
    assert TL.relabel_text(3.5) == 3.5


def test_display_frame_maps_text_never_numbers_and_never_mutates():
    df = pd.DataFrame({"tier": ["SNIPER", "VALUABLE", "PAPER"], "pnl": [1.5, -1.0, 0.25],
                       "SNIPER now": [3, 4, 5]})
    out = TL.display_frame(df)
    assert list(out["tier"]) == ["Big gap", "Small gap", "PAPER"]
    assert list(out.columns) == ["tier", "pnl", "Big gap now"]
    assert out["pnl"].tolist() == df["pnl"].tolist() and out["Big gap now"].tolist() == [3, 4, 5]
    assert list(df["tier"]) == ["SNIPER", "VALUABLE", "PAPER"]             # caller's frame untouched


# ── 2. static: no hard-coded old names in display strings ──────────────────────────────────────────
TELEGRAM_MODULES = [ROOT / "telegram_bot" / "notifier.py", ROOT / "telegram_bot" / "daily_pack.py",
                    ROOT / "telegram_bot" / "tips_table.py"]
DASHBOARD_MODULES = ([ROOT / "dashboard_ui.py"] + sorted((ROOT / "pages").glob("*.py"))
                     + sorted((ROOT / "dashboard_data").glob("*.py")))
DISPLAY_MODULES = sorted(TELEGRAM_MODULES + DASHBOARD_MODULES)


def _docstring_ids(tree) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def display_string_violations(path: Path) -> list[str]:
    """String literals that spell an old tier name AND are more than the bare code.

    A literal that IS a code ("SNIPER") is data - a filter value, a dict key, a policy key. Anything
    longer that contains one ("Staked (SNIPER + MARKSMAN)", f"... {x} VALUABLE ...") is text a person
    reads, and must come from src/tier_labels.py instead. Docstrings and comments are not display."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            for v in node.values:
                if isinstance(v, ast.Constant):
                    skip.add(id(v))
            if CODE.search(parts):
                out.append(f"{path.name}:{node.lineno}: f-string {parts[:90]!r}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            s = node.value
            if CODE.search(s) and s.strip() not in TL.INTERNAL_CODES:
                out.append(f"{path.name}:{node.lineno}: {s[:90]!r}")
    return out


@pytest.mark.parametrize("path", DISPLAY_MODULES, ids=lambda p: p.name)
def test_no_hard_coded_tier_names_in_display_strings(path):
    bad = [v for v in display_string_violations(path) if not _allowed(v)]
    assert not bad, "display strings must use src/tier_labels.py:\n" + "\n".join(bad)


#: Literals that mention a code but are not shown to a person (log lines, data keys with context).
ALLOWED = ()


def _allowed(v: str) -> bool:
    return any(a in v for a in ALLOWED)


def test_the_static_check_catches_display_strings(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""Docstring SNIPER is fine."""\nA = ("SNIPER", "MARKSMAN")\nB = "Staked (SNIPER + MARKSMAN)"\n'
                 'C = f"{n} VALUABLE tips"\n# comment VALUABLE\nD = "SNIPER_THRESHOLD"\n', encoding="utf-8")
    bad = display_string_violations(f)
    assert len(bad) == 2 and ":3:" in bad[0] and ":4:" in bad[1], bad


def test_display_modules_use_the_mapping():
    for p in TELEGRAM_MODULES + [ROOT / "dashboard_ui.py"]:
        assert "tier_labels" in p.read_text(encoding="utf-8"), f"{p.name} does not import src/tier_labels"


def test_pages_render_tables_through_the_relabelling_helpers():
    """Pages and view modules never call st.dataframe directly: ui.table / ui.compact_table / ui.dataframe
    run every table through tier_labels.display_frame, so a tier code in a cell is shown as its label."""
    bad = [f"{p.name}:{i}" for p in DASHBOARD_MODULES if p.name != "dashboard_ui.py"
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if "st.dataframe(" in line and not line.lstrip().startswith("#")]
    assert not bad, bad


def test_slim_deployment_ships_the_mapping():
    import json
    m = json.loads((ROOT / "deploy" / "dashboard_manifest.json").read_text(encoding="utf-8"))
    assert "src/tier_labels.py" in {e["path"] for e in m["code"]}


# ── 3. Telegram on committed data ───────────────────────────────────────────────────────────────────
def _hash(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _captured_drawing(monkeypatch):
    from PIL import ImageDraw
    drawn: list[str] = []
    orig = ImageDraw.ImageDraw.text

    def _text(self, xy, text, *a, **k):
        drawn.append(str(text))
        return orig(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _text)
    return drawn


def test_telegram_pictures_show_labels_not_codes(monkeypatch):
    import datetime as dt
    from telegram_bot import daily_pack as DP, tips_table as TT
    drawn = _captured_drawing(monkeypatch)
    led = pd.read_csv(ROOT / "output" / "bets_ledger.csv", usecols=["match_date"]) \
        if (ROOT / "output" / "bets_ledger.csv").exists() else pd.DataFrame()
    if led.empty:
        pytest.skip("no ledger in this checkout")
    days = sorted(led["match_date"].astype(str).str[:10].unique())[-6:]
    today = days[-1]
    imgs = []
    for d in days:
        imgs += DP.results_image(d)
    imgs += TT.render_pngs(TT.tips_rows(days[0], today), "tips")
    imgs += DP.record_image(today) + DP.week_images(today) + DP.last_week_leagues_image(today) + \
        DP.leagues_image(today)
    assert imgs, "nothing rendered"
    joined = "\n".join(drawn)
    assert not CODE.search(joined), [s for s in drawn if CODE.search(s)][:5]
    assert LABEL_WORDS.search(joined)                       # the labels are actually drawn
    assert DP.build_pack(dt.datetime.now(dt.timezone.utc), freeze=False)[1]  # caption still builds


def test_telegram_messages_labels_codes_and_keys_unchanged(monkeypatch, tmp_path):
    """A notify dry run (sends captured, no network, no notified.json write): messages carry labels,
    never codes; every dedup key it records is the code-free fixture key it always was; the ledgers
    and notified.json on disk are byte-identical afterwards."""
    from telegram_bot import notifier as N
    watched = [ROOT / "output" / f for f in ("bets.csv", "bets_ledger.csv", "side_bets_ledger.csv",
                                            "player_ledger.csv", "side_bets.csv")] + \
              [N.NOTIFIED_FILE, N.PLAYER_NOTIFIED_FILE, N.PLAYER_KICKOFF_FILE]
    before = {p: _hash(p) for p in watched}
    sent: list[str] = []
    saved: set[str] = set()
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "0")
    monkeypatch.setenv("TIPS_INSTANT_ALERTS", "1")
    monkeypatch.setattr(N, "_send_one", lambda token, chat_id, text: sent.append(text) or True)
    monkeypatch.setattr(N, "_load_notified", lambda path=None: set())
    monkeypatch.setattr(N, "_save_notified", lambda keys, path=None: saved.update(keys))
    monkeypatch.setattr(N, "PLAYER_KICKOFF_FILE", tmp_path / "kickoff.json")
    monkeypatch.setattr(N.requests, "post", lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
    for fn in (N.notify_new_snipers, N.notify_side_bets, N.notify_daily_digest, N.notify_weekly_summary,
               N.notify_props_daily_digest, N.notify_props_weekly_summary, N.notify_player_props):
        fn()
    text = "\n".join(sent)
    assert not CODE.search(text), [m[:120] for m in sent if CODE.search(m)][:3]
    if not sent:
        pytest.skip("nothing to send on this checkout's data")
    for k in saved:                                 # dedup keys: fixture identity, never a tier word
        assert not LABEL_WORDS.search(k) and not CODE.search(k), k
    o_u = [k for k in saved if k.count("|") == 3 and k.split("|")[0][:2] == "20"]
    assert all(k.split("|")[3] in ("OVER", "UNDER") for k in o_u)
    assert {p: _hash(p) for p in watched} == before


# ── 4. dashboard pages on committed data ───────────────────────────────────────────────────────────
#: Every page that shows a tier somewhere (cards, chips, captions, tables, filters, selectors).
TIER_PAGES = ["00_🎯_Home.py", "0_🏠_Overview.py", "1_📊_Dashboard.py", "10_🤖_Agent_Analysis.py",
              "11_👤_Player_Props.py", "12_💼_Portfolio.py", "16_💰_Bankroll.py", "17_🏟️_Leagues.py",
              "21_📡_Market_Intelligence.py", "22_🏭_Production.py", "4_ℹ️_Model_Info.py",
              "5_🎯_Success_Rates.py", "7_🔬_Pro.py", "9_🎓_Model_Performance.py", "free_tips.py"]
PAGE_CODE = re.compile(r"\b(SNIPER|MARKSMAN|VALUABLE)\b|\b(Sniper|Marksman)\b")


def _page_strings(at) -> list[str]:
    """Every string a person can read on the rendered page. A multiselect's VALUE is the internal key
    (its options are drawn through format_func), so it is the one thing not read here."""
    out: list[str] = []

    def walk(node):
        kind = type(node).__name__
        for attr in ("label", "value", "body", "options", "help", "placeholder"):
            if kind == "Multiselect" and attr == "value":
                continue
            try:
                v = getattr(node, attr)
            except Exception:                                   # noqa: BLE001
                continue
            if isinstance(v, pd.DataFrame):
                out.extend(map(str, v.columns))
                out.extend(str(x) for x in v.to_numpy().ravel().tolist())
            elif isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out.extend(str(x) for x in v)
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            for k in ch:
                walk(ch[k])

    walk(at._tree)
    return out


@pytest.mark.parametrize("page", TIER_PAGES)
def test_dashboard_pages_show_labels_not_codes(page, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from dashboard_data import core
    path = ROOT / "pages" / page
    monkeypatch.setattr(core, "RAW_BASE", {})                   # Pro / V11 unreachable: no network
    for k in ("pro", "v11"):
        monkeypatch.setitem(core.REPOS, k, {**core.REPOS[k], "candidates": []})
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    st.cache_data.clear()
    at = AppTest.from_file(str(path.resolve()), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    left = [s[:140] for s in _page_strings(at) if PAGE_CODE.search(s)]
    assert not left, left[:5]
