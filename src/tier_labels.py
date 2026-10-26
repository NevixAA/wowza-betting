"""What a PERSON sees for a signal tier. The one place tier names are spelled for display.

Owner decision 2026-10-26: the tiers measure the SIZE of the gap between our probability and the
bookmaker's price, not confidence. Measured this season a bigger gap does not mean better results on
O/U 2.5 (SNIPER -4.2% vs VALUABLE +2.2% new-format; every standard tier negative); only new-format
BTTS shows it (SNIPER +25.7%). So the names that sounded like accuracy ("sniper", "marksman") are
shown as gap sizes:

    SNIPER   -> Big gap
    MARKSMAN -> Medium gap
    VALUABLE -> Small gap
    AVOID    -> No tip

DISPLAY ONLY. The internal codes SNIPER / MARKSMAN / VALUABLE / AVOID are unchanged everywhere they
are stored or computed on: ledgers and CSVs (signal_tier, tier), thresholds and env vars
(MARKSMAN_THRESHOLD ...), bankroll policy multiplier keys, Telegram dedup keys in notified.json, Pro
interfaces and model training. History stays comparable; only the words a person reads change.

Every Telegram message / picture, dashboard page and human-readable report takes its tier words from
here. tests/test_tier_labels.py fails if a display module spells an old name in a display string.
"""
from __future__ import annotations

import re

#: Internal code -> what a person reads.
TIER_LABEL: dict[str, str] = {
    "SNIPER": "Big gap",
    "MARKSMAN": "Medium gap",
    "VALUABLE": "Small gap",
    "AVOID": "No tip",
}

#: For narrow table columns.
TIER_LABEL_SHORT: dict[str, str] = {
    "SNIPER": "Big",
    "MARKSMAN": "Medium",
    "VALUABLE": "Small",
    "AVOID": "No tip",
}

#: Telegram icon per tier: bars that show the gap SIZE (the old target / gun / gem icons implied
#: accuracy). PAPER is the props paper tier.
TIER_ICON: dict[str, str] = {
    "SNIPER": "▰▰▰",
    "MARKSMAN": "▰▰▱",
    "VALUABLE": "▰▱▱",
    "PAPER": "📄",
}

#: The one-line explanation shown wherever the tiers are explained to a person.
GAP_EXPLAINER = ("Gap = how far our probability is from the bookmaker's price. "
                 "It is not a confidence rating.")

#: The staked tiers (SNIPER + MARKSMAN) and the paper tier (VALUABLE), as a person reads them.
STAKED_LABEL = "Big + Medium gap"
PAPER_TIER_LABEL = TIER_LABEL["VALUABLE"]

#: The old names, for tests and for the glossary line. Never use these in display strings.
INTERNAL_CODES = ("SNIPER", "MARKSMAN", "VALUABLE")

#: For readers of older documents: the internal code behind each label.
GLOSSARY = ("Internal codes (unchanged in stored data): Big gap = SNIPER, Medium gap = MARKSMAN, "
            "Small gap = VALUABLE.")

_CODE_RE = re.compile(r"\b(SNIPER|MARKSMAN|VALUABLE)\b")
_HAS_CODE = re.compile(r"\b(?:SNIPER|MARKSMAN|VALUABLE)\b")


def tier_label(code, short: bool = False) -> str:
    """The display word for a tier code. Unknown values (PAPER, WATCH, blanks) pass through as text."""
    if code is None:
        return ""
    s = str(code)
    key = s.strip().upper()
    table = TIER_LABEL_SHORT if short else TIER_LABEL
    return table.get(key, s)


def tier_tag(code) -> str:
    """Icon + label, e.g. '▰▰▰ Big gap' (Telegram lines)."""
    icon = TIER_ICON.get(str(code).strip().upper(), "📌")
    return f"{icon} {tier_label(code)}"


def relabel_text(text):
    """Replace every whole-word tier code in a display string with its label. Non-strings pass
    through untouched. Underscored identifiers (SNIPER_THRESHOLD) are not words and are kept."""
    if not isinstance(text, str):
        return text
    return _CODE_RE.sub(lambda m: TIER_LABEL[m.group(1)], text)


def display_frame(df):
    """A copy of a table for DISPLAY: tier codes in headers and text cells shown as labels.

    Numbers are never touched (only object/string columns are mapped), and the caller's frame is not
    modified, so nothing computed downstream sees a label instead of a code."""
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty and not len(df.columns):
        return df
    cols = [relabel_text(c) if isinstance(c, str) else c for c in df.columns]
    hits = [c for c in df.columns
            if (df[c].dtype == object or str(df[c].dtype) in ("string", "category", "str"))
            and df[c].astype(str).str.contains(_HAS_CODE).any()]
    if not hits and cols == list(df.columns):
        return df
    out = df.copy()
    for c in hits:
        out[c] = out[c].map(relabel_text) if str(out[c].dtype) != "category" else \
            out[c].astype(object).map(relabel_text)
    out.columns = cols
    return out
