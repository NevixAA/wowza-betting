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
DISPLAY_MODULES = sorted(
    [ROOT / "telegram_bot" / "notifier.py", ROOT / "telegram_bot" / "daily_pack.py",
     ROOT / "telegram_bot" / "tips_table.py"])


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


def test_display_modules_use_the_mapping():
    for p in DISPLAY_MODULES:
        assert "tier_labels" in p.read_text(encoding="utf-8"), f"{p.name} does not import src/tier_labels"


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
