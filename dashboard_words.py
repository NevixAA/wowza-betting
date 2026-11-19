"""User-facing wording rules, applied where every dashboard page and every Telegram message passes.

Owner 2026-11-17 ("Show stake on TOP-3"): from 2026-11-21 every TOP-3 tip shows "0.5% stake"; every other
prediction carries NO stake label at all; the word "paper" never appears in anything a user sees (the dashboard
and Telegram). Internally (code, ledgers, policy states such as PAPER) nothing changes: a state is relabelled only
when it is SHOWN.

    relabel_text("PAPER track")  -> "NO STAKE track"
    relabel_obj(DataFrame)       -> a copy with every column / index label and string cell relabelled
    install()                    -> Streamlit's text / table / widget calls relabel what they render (idempotent)

Widget OPTIONS keep their values (code compares them); only what is displayed goes through format_func.
tests/test_ui_wording.py and scripts/ui_wording_scan.py (every page rendered headless) hold the rule.
"""
from __future__ import annotations

import functools
import re

RETIRED_WORD = "paper"

# longest first; the case of the original is kept (PAPER -> NO STAKE, Paper -> No stake, paper -> no stake)
_RULES = [
    (re.compile(r"PAPER[_ ]LEAGUE", re.IGNORECASE), "no stake (league)"),
    (re.compile(r"paper[- ]only", re.IGNORECASE), "no stake"),
    (re.compile(r"paper[- ]track", re.IGNORECASE), "no-stake track"),
    (re.compile(r"paper", re.IGNORECASE), "no stake"),
]


def _case(src: str, rep: str) -> str:
    if src.isupper():
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def relabel_text(s):
    if not isinstance(s, str) or RETIRED_WORD not in s.lower():
        return s
    for rx, rep in _RULES:
        s = rx.sub(lambda m: _case(m.group(0), rep), s)
    return s


def _relabel_frame(df):
    import pandas as pd
    out = df.copy()
    try:
        out.columns = [relabel_text(c) if isinstance(c, str) else c for c in out.columns]
        if not pd.api.types.is_numeric_dtype(out.index) and not isinstance(out.index, pd.MultiIndex):
            out.index = [relabel_text(i) if isinstance(i, str) else i for i in out.index]
        for c in out.columns:
            col = out[c]
            if col.dtype == object or str(col.dtype) in ("string", "str") or isinstance(col.dtype,
                                                                                         pd.CategoricalDtype):
                out[c] = col.map(lambda v: relabel_text(v) if isinstance(v, str) else v)
    except Exception:                                                  # noqa: BLE001 - display only
        return df
    return out


def relabel_obj(x):
    """Strings, lists / tuples / dicts of them, pandas frames / series and Stylers (values only)."""
    try:
        import pandas as pd
    except ImportError:                                                # pragma: no cover
        pd = None
    if isinstance(x, str):
        return relabel_text(x)
    if pd is not None:
        if isinstance(x, pd.DataFrame):
            return _relabel_frame(x)
        if isinstance(x, pd.Series):
            return x.map(lambda v: relabel_text(v) if isinstance(v, str) else v).rename(relabel_text(x.name)
                                                                                        if isinstance(x.name, str)
                                                                                        else x.name)
        Styler = getattr(getattr(pd.io.formats, "style", None), "Styler", None)
        if Styler is not None and isinstance(x, Styler):
            try:                                                       # same shape and labels: styles stay valid
                d = x.data
                for c in d.columns:
                    if d[c].dtype == object:
                        d[c] = d[c].map(lambda v: relabel_text(v) if isinstance(v, str) else v)
            except Exception:                                          # noqa: BLE001
                pass
            return x
    if isinstance(x, list):
        return [relabel_obj(v) for v in x]
    if isinstance(x, tuple):
        return tuple(relabel_obj(v) for v in x)
    if isinstance(x, dict):
        return {k: relabel_obj(v) for k, v in x.items()}
    return x


# ── Streamlit ─────────────────────────────────────────────────────────────────────────────────────
#: name -> (positional args relabelled, keyword args relabelled)
_TEXT = {n: ((0,), ("help",)) for n in ("markdown", "caption", "text", "title", "header", "subheader", "info",
                                        "warning", "error", "success", "toast", "code", "latex")}
_TEXT.update({
    "write": ("all", ()),
    "metric": ((0, 1, 2), ("label", "value", "delta", "help")),
    "dataframe": ((0,), ("data", "column_config", "column_order")),
    "data_editor": ((0,), ("data", "column_config", "column_order")),
    "table": ((0,), ("data",)),
    "tabs": ((0,), ("tabs",)),
    "expander": ((0,), ("label",)),
    "popover": ((0,), ("label", "help")),
    "button": ((0,), ("label", "help")),
    "download_button": ((0,), ("label", "help")),
    "link_button": ((0,), ("label", "help")),
    "checkbox": ((0,), ("label", "help")),
    "toggle": ((0,), ("label", "help")),
    "text_input": ((0,), ("label", "help", "placeholder")),
    "number_input": ((0,), ("label", "help")),
    "slider": ((0,), ("label", "help")),
    "date_input": ((0,), ("label", "help")),
})
#: option widgets: the label is relabelled and what each option SHOWS goes through format_func
_OPTIONS = ("selectbox", "radio", "multiselect", "pills", "segmented_control", "select_slider")


def _wrap_text(fn, pos, kws):
    @functools.wraps(fn)
    def w(*a, **k):
        a = list(a)
        # a bound DeltaGenerator method receives `self` first
        off = 1 if a and type(a[0]).__name__ == "DeltaGenerator" else 0
        if pos == "all":
            a = a[:off] + [relabel_obj(v) for v in a[off:]]
        else:
            for i in pos:
                if off + i < len(a):
                    a[off + i] = relabel_obj(a[off + i])
        for kw in kws:
            if kw in k:
                v = relabel_obj(k[kw])
                if kw == "column_config" and isinstance(v, dict):   # keys name columns: renamed with them
                    v = {relabel_text(c) if isinstance(c, str) else c: x for c, x in v.items()}
                k[kw] = v
        return fn(*a, **k)
    w.__wowza_words__ = True
    return w


def _wrap_options(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        a = list(a)
        off = 1 if a and type(a[0]).__name__ == "DeltaGenerator" else 0
        if off < len(a):
            a[off] = relabel_obj(a[off])
        for kw in ("label", "help", "placeholder"):
            if kw in k:
                k[kw] = relabel_obj(k[kw])
        base = k.get("format_func") or str
        k["format_func"] = lambda o, _b=base: relabel_text(str(_b(o)))
        return fn(*a, **k)
    w.__wowza_words__ = True
    return w


def install() -> bool:
    """Patch Streamlit once (DeltaGenerator methods and the module-level st.* aliases). Never raises."""
    try:
        import streamlit as st
        from streamlit.delta_generator import DeltaGenerator as DG
    except Exception:                                                  # noqa: BLE001
        return False
    if getattr(DG, "__wowza_words_installed__", False):
        return True
    for name, (pos, kws) in _TEXT.items():
        f = getattr(DG, name, None)
        if f is not None and not getattr(f, "__wowza_words__", False):
            setattr(DG, name, _wrap_text(f, pos, kws))
    for name in _OPTIONS:
        f = getattr(DG, name, None)
        if f is not None and not getattr(f, "__wowza_words__", False):
            setattr(DG, name, _wrap_options(f))
    main = getattr(st, "_main", None)
    for name in list(_TEXT) + list(_OPTIONS):
        if main is not None and hasattr(main, name) and hasattr(st, name):
            try:
                setattr(st, name, getattr(main, name))
            except Exception:                                          # noqa: BLE001
                pass
    DG.__wowza_words_installed__ = True
    install_pil()                                                      # pictures shown on a page
    return True


# ── images (Telegram albums / tips table, PIL) ───────────────────────────────────────────────────
def install_pil() -> bool:
    """Relabel the text PIL draws and measures (ImageDraw.text / textlength / textbbox), so a picture can never
    carry the retired word and the layout is measured on the text actually drawn. Idempotent; never raises."""
    try:
        from PIL import ImageDraw
    except Exception:                                                  # noqa: BLE001
        return False
    D = ImageDraw.ImageDraw
    if getattr(D, "__wowza_words_installed__", False):
        return True
    for name in ("text", "textlength", "textbbox", "multiline_text", "multiline_textbbox"):
        f = getattr(D, name, None)
        if f is None:
            continue

        def w(self, *a, _f=f, _n=name, **k):
            a = list(a)
            i = 0 if _n == "textlength" else 1                         # text(xy, text, ...) / textlength(text, ...)
            if len(a) > i:
                a[i] = relabel_text(a[i])
            if "text" in k:
                k["text"] = relabel_text(k["text"])
            return _f(self, *a, **k)
        setattr(D, name, w)
    D.__wowza_words_installed__ = True
    return True
