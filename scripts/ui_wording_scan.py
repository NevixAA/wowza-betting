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


# ── Telegram (owner 2026-11-19: an HT tip said "Paper only"; the page scan above never saw Telegram) ──────
#: every module that builds Telegram text or pictures. Their user-facing string literals are scanned statically:
#: a message is assembled from these literals, so a retired word in one of them can reach a user.
TELEGRAM_SOURCES = ("telegram_bot/*.py", "src/top3_league.py", "src/props_top3.py", "src/live_scanner.py",
                    "src/top_picks.py")
#: dashboard sources scanned STATICALLY as well as rendered (owner 2026-11-22: the Player Props page said "PAPER" /
#: "Paper research" in its source and only the display relabel hid it; a literal there now fails the scan)
STATIC_PAGE_SOURCES = ("pages/11_*Player_Props.py", "dashboard_data/props_view.py",
                       "dashboard_data/props_top3_view.py")
#: (file, exact literal) the owner kept on purpose. EMPTY: owner 2026-11-17 "no paper label on tips" and
#: dashboard_words ("the word paper never appears in anything a user sees") leave no user-facing exception.
TELEGRAM_ALLOW: frozenset = frozenset()
_CODE_CALLS = {"endswith", "startswith", "get", "replace", "isin", "drop_paper_leagues", "is_paper_league", "pop",
               "setdefault", "split", "rsplit", "partition"}


def _docstring_ids(tree) -> set:
    import ast
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            out.add(id(n.body[0].value))
    return out


def scan_telegram_sources(word: re.Pattern, root: Path | None = None, sources=TELEGRAM_SOURCES) -> list[dict]:
    """Every user-facing string literal of the Telegram builders that contains `word`. Skipped as CODE, not text:
    docstrings and comments, a bare identifier / state token ('PAPER', 'paper_1x2'), a literal compared against
    (==, in, a .endswith(...) / .get(...) argument, a subscript or dict key)."""
    import ast
    r = Path(root) if root else ROOT
    files = sorted({p for g in sources for p in r.glob(g)})
    hits = []
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = _docstring_ids(tree)
        parent = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parent[id(c)] = n
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or id(n) in docs:
                continue
            if not word.search(n.value) or re.fullmatch(r"\w+", n.value.strip()):
                continue
            p = parent.get(id(n))
            if isinstance(p, ast.JoinedStr):
                p = parent.get(id(p), p)
            code = (isinstance(p, (ast.Compare, ast.Subscript))
                    or (isinstance(p, ast.Dict) and any(k is n for k in p.keys))
                    or (isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr in _CODE_CALLS))
            rel = f.relative_to(r).as_posix()
            if code or (rel, n.value) in TELEGRAM_ALLOW:
                continue
            hits.append({"page": rel, "kind": f"{'telegram' if sources is TELEGRAM_SOURCES else 'source'} literal "
                                              f"L{n.lineno}", "text": n.value[:160]})
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="paper")
    ap.add_argument("--json")
    ap.add_argument("--page", action="append")
    ap.add_argument("--telegram", action="store_true", help="scan only the Telegram message builders (static)")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    _w = re.compile(r"\b" + re.escape(a.word) + r"\b", re.IGNORECASE)
    tg = scan_telegram_sources(_w)
    print(f"telegram builders: {len(tg)} hit(s)", flush=True)
    st_hits = scan_telegram_sources(_w, sources=STATIC_PAGE_SOURCES)
    print(f"static page sources: {len(st_hits)} hit(s)", flush=True)
    tg += st_hits
    for x in tg:
        print(f"  {x['page']} [{x['kind']}] {x['text']}")
    if a.telegram:
        if a.json:
            Path(a.json).write_text(json.dumps(tg, indent=1, ensure_ascii=False), encoding="utf-8")
        return 1 if tg else 0
    os.environ["WOWZA_TIER_OVERRIDE"] = "admin"
    import dashboard_access
    pages = a.page or (["app.py"] + [p for p, *_ in dashboard_access.PAGES])
    word = re.compile(re.escape(a.word), re.IGNORECASE)
    hits = list(tg)
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
