"""Scan every dashboard page's RENDERED text for words the owner retired from user-facing output.

Owner 2026-11-17 ("Show stake on TOP-3"): TOP-3 tips show "0.5% stake"; other predictions carry NO stake label;
the word "paper" never appears in anything a user sees. This renders app.py and every page of
dashboard_access.PAGES headless (streamlit AppTest, admin tier, every tab is rendered by Streamlit), collects
every visible string - titles, markdown, captions, metrics, alerts, tab / expander labels, widget labels and
options, and every cell, column and index label of every table - and lists each occurrence.

    python scripts/ui_wording_scan.py [--word paper] [--json out.json] [--page <path>]

Exit 0 = no occurrence; 1 = occurrences listed. Read-only: no file is written apart from --json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_TEXT_KINDS = ("title", "header", "subheader", "markdown", "caption", "text", "code", "latex", "info", "warning",
               "error", "success", "exception", "metric", "tabs", "expander", "selectbox", "radio", "multiselect",
               "checkbox", "toggle", "button", "text_input", "number_input", "slider", "select_slider", "json",
               "table", "dataframe", "sidebar")


_OPTION_WIDGETS = ("selectbox", "radio", "multiselect", "select_slider")


def _strings(el, kind: str = "") -> list[str]:
    """Every user-visible string of one element (best effort over the AppTest element types). An option widget's
    `.value` is the program-side value, not what is shown (the shown text is its formatted `options`)."""
    out = []
    attrs = ("value", "body", "label", "help", "placeholder", "delta", "options")
    if kind in _OPTION_WIDGETS:
        attrs = tuple(a for a in attrs if a != "value")
    for attr in attrs:
        try:
            v = getattr(el, attr)
        except Exception:                                              # noqa: BLE001
            continue
        if v is None:
            continue
        if hasattr(v, "to_dict") and hasattr(v, "columns"):           # a DataFrame (dataframe / table value)
            try:
                out += [str(c) for c in v.columns]
                out += [str(i) for i in v.index]
                out += [str(x) for x in v.astype(str).to_numpy().ravel().tolist()]
            except Exception:                                          # noqa: BLE001
                pass
        elif isinstance(v, (list, tuple, set)):
            out += [str(x) for x in v]
        else:
            out.append(str(v))
    return out


def scan_page(page: str, word: re.Pattern) -> list[dict]:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / page), default_timeout=240)
    at.run()
    hits = []
    seen = set()
    for kind in _TEXT_KINDS:
        try:
            els = list(at.get(kind))
        except Exception:                                              # noqa: BLE001
            continue
        for el in els:
            for s in _strings(el, kind):
                for m in word.finditer(s):
                    snip = s[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                    key = (kind, snip)
                    if key not in seen:
                        seen.add(key)
                        hits.append({"page": page, "kind": kind, "text": snip})
    for e in at.exception:
        hits.append({"page": page, "kind": "EXCEPTION", "text": str(e.value)[:200]})
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="paper")
    ap.add_argument("--json")
    ap.add_argument("--page", action="append")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ["WOWZA_TIER_OVERRIDE"] = "admin"
    import dashboard_access
    pages = a.page or (["app.py"] + [p for p, *_ in dashboard_access.PAGES])
    word = re.compile(re.escape(a.word), re.IGNORECASE)
    hits = []
    for p in pages:
        try:
            h = scan_page(p, word)
        except Exception as e:                                         # noqa: BLE001
            h = [{"page": p, "kind": "RUN_FAILED", "text": f"{type(e).__name__}: {e}"[:200]}]
        hits += h
        print(f"{p}: {sum(1 for x in h if x['kind'] not in ('EXCEPTION', 'RUN_FAILED'))} hit(s)"
              + (f", {sum(1 for x in h if x['kind'] in ('EXCEPTION', 'RUN_FAILED'))} exception(s)"
                 if any(x['kind'] in ('EXCEPTION', 'RUN_FAILED') for x in h) else ""), flush=True)
    for x in hits:
        print(f"  {x['page']} [{x['kind']}] {x['text']}")
    if a.json:
        Path(a.json).write_text(json.dumps(hits, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if any(x["kind"] not in ("EXCEPTION",) for x in hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
