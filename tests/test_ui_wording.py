"""Owner 2026-11-17 ("Show stake on TOP-3"): the word "paper" never reaches a user - dashboard or Telegram.

dashboard_words relabels at the display layer (Streamlit calls, PIL text in pictures, every Telegram send);
internal states (PAPER in the policy, ledgers, code) are unchanged. scripts/ui_wording_scan.py renders every page
headless and lists any occurrence (the go-live check); these tests pin the mechanism."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dashboard_words as W  # noqa: E402


def test_relabel_text_keeps_case_and_never_leaves_the_word():
    assert W.relabel_text("PAPER") == "NO STAKE"
    assert W.relabel_text("Paper league") == "No stake (league)"
    assert W.relabel_text("a paper track") == "a no-stake track"
    assert W.relabel_text("PAPER_LEAGUE") == "NO STAKE (LEAGUE)"
    assert W.relabel_text("paper-only research") == "no stake research"
    assert W.relabel_text("staked") == "staked" and W.relabel_text(None) is None and W.relabel_text(3) == 3
    for s in ("PAPER", "Paper", "paper_1x2", "PaPeR", "paper-only", "PAPER LEAGUE"):
        assert "paper" not in W.relabel_text(s).lower()


def test_relabel_frame_columns_index_and_cells_not_numbers():
    df = pd.DataFrame({"State": ["PAPER", "ACTIVE"], "Paper P/L": [1.5, -2.0]}, index=["paper row", "x"])
    out = W.relabel_obj(df)
    assert list(out.columns) == ["State", "No stake P/L"]
    assert list(out.index) == ["no stake row", "x"]
    assert out["State"].tolist() == ["NO STAKE", "ACTIVE"] and out["No stake P/L"].tolist() == [1.5, -2.0]
    assert df["State"].tolist() == ["PAPER", "ACTIVE"]                  # the caller's frame is untouched


def test_streamlit_is_patched_once_and_option_values_stay_raw():
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator
    assert W.install() and W.install()                                  # idempotent
    assert getattr(DeltaGenerator.markdown, "__wowza_words__", False)
    assert getattr(st.markdown, "__wowza_words__", False) or getattr(getattr(st.markdown, "__func__", None),
                                                                     "__wowza_words__", False)
    seen = {}

    def fake(label, options, **k):
        seen.update(label=label, shown=[k["format_func"](o) for o in options], options=list(options))
        return options[0]
    wrapped = W._wrap_options(fake)
    assert wrapped("Paper state", ["PAPER", "ACTIVE"]) == "PAPER"  # the program gets the raw value
    assert seen["label"] == "No stake state" and seen["shown"] == ["NO STAKE", "ACTIVE"]


def test_every_page_installs_the_wording_layer():
    import dashboard_access
    pages = ["app.py"] + [p for p, *_ in dashboard_access.PAGES]
    for p in pages:
        src = (ROOT / p).read_text(encoding="utf-8")
        assert "dashboard_access" in src, p                             # importing it installs dashboard_words
    assert "_words.install()" in (ROOT / "dashboard_access.py").read_text(encoding="utf-8")


def test_telegram_sends_never_carry_the_word(monkeypatch):
    from telegram_bot import notifier as N
    sent = []
    monkeypatch.setattr(N, "_send_one", lambda token, chat, text: sent.append(text) or True)
    assert N._send("t", "c", "📄 <b>PAPER</b> — paper track")
    assert sent and all("paper" not in s.lower() for s in sent)
    for f in ("telegram_bot/tips_table.py", "telegram_bot/daily_pack.py"):
        assert "relabel_text(caption)" in (ROOT / f).read_text(encoding="utf-8"), f


def test_pictures_never_carry_the_word():
    from PIL import Image, ImageDraw
    assert W.install_pil()
    img = Image.new("RGB", (200, 40))
    d = ImageDraw.Draw(img)
    drawn = []
    orig = W.relabel_text

    def spy(s):
        r = orig(s)
        drawn.append(r)
        return r
    W.relabel_text = spy
    try:
        d.text((0, 0), "Paper only", fill="white")
        assert d.textlength("PAPER") == d.textlength("NO STAKE")         # measured on the text actually drawn
    finally:
        W.relabel_text = orig
    assert drawn and all("paper" not in x.lower() for x in drawn if isinstance(x, str))
