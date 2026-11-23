"""Short, unique display label of a prediction-ledger model_version (owner 2026-11-23: "why duplicates?").

    version_label("props_goals_props_clean_harness_5de94714")   # 'clean · 5de94714'
    version_label("props_goals_legacy_1592478c")                # 'legacy · 1592478c'
    version_label("btts_20261119T134019Z_44c67049")             # 'trained 2026-11-19 · 44c67049'

Display only: the full model_version stays the key everywhere. The trailing content hash keeps two versions of one
recipe apart. Pure, no file read (the slim dashboard tree imports it).
"""
from __future__ import annotations

import re

#: recipe words in a version id -> label (first match wins; checked before the training date)
_RECIPES = (("standard_repaired", "repaired"), ("legacy", "legacy"), ("props_clean_harness", "clean"),
            ("clean_hgb_nan_native", "clean HGB"), ("clean_recipe_retrained", "clean retrained"))


def version_label(version) -> str:
    s = str(version or "")
    if not s:
        return ""
    h = re.search(r"_([0-9a-f]{8,12})$", s)
    hsh = h.group(1)[:8] if h else ""
    base = next((lab for pat, lab in _RECIPES if pat in s), None)
    if base is None:
        m = re.search(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", s)
        d = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        sv = re.search(r"_(\d+\.\d+\.\d+)_", s)
        base = (f"trained {m.group(1)}-{m.group(2)}-{m.group(3)}" if m else f"trained {d.group(1)}" if d
                else f"v{sv.group(1)}" if sv else (s[: -len(h.group(0))] if h else s))
    return f"{base} · {hsh}" if hsh else base
