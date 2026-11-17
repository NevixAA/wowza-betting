"""The two records' NAMES on the dashboard: "Season to 20.11 (pre-repair models)" and "Clean Record (from 21.11)".

Owner 2026-11-17: "we need to start from 0 counting again (do not delete the old history of this season's
results); we need a comparison later. But after 21.11 it's a new page."

THE START DATE. `config/trustworthy_performance.json` -> clean_record_start_utc is THE
value (owner decision 2026-11-11, written by runbook step U6 at the go-live merge; dashboard_data/clean_record.py
splits every figure on it). This module reads that value whenever the tree has it. A tree without it (main before
the go-live merge, where the file does not exist yet, or a branch where it is still null) falls back to
OWNER_DECISION below, which only NAMES the record; no figure is computed from it. tests/test_dashboard_records.py
fails if the configured value and OWNER_DECISION ever disagree, so there is never a second, different date.

WHICH PAGES ARE "SEASON TO 20.11". PRE_REPAIR_PAGES lists the results pages that existed before the repairs. Until
the Clean Record split ships (dashboard_data/clean_record.py arrives with the go-live merge), every figure on them is
pre-repair, so dashboard_ui.page_header labels each one "Season to 20.11 (pre-repair models)". Once the split is in
the tree the pages label their own sections, and the page-wide label switches off. Nothing on those pages changes:
no row is dropped, filtered or rewritten.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP_CONFIG = ROOT / "config" / "trustworthy_performance.json"

#: Owner decision 2026-11-11 (restated 2026-11-17): the Clean Record counts from Sat 2026-11-21. Used only
#: when config/trustworthy_performance.json has no clean_record_start_utc; a test pins the two together.
OWNER_DECISION = "2026-11-21"

#: Results pages of the season before the repairs (file names under pages/).
PRE_REPAIR_PAGES = ("00_🎯_Home.py", "1_📊_Dashboard.py", "17_🏟️_Leagues.py", "5_🎯_Success_Rates.py",
                    "12_💼_Portfolio.py", "16_💰_Bankroll.py", "9_🎓_Model_Performance.py", "0_🏠_Overview.py",
                    "11_👤_Player_Props.py", "free_tips.py")


def configured_start() -> str | None:
    """'YYYY-MM-DD' of clean_record_start_utc in config/trustworthy_performance.json, or None (absent / null)."""
    try:
        v = json.loads(TP_CONFIG.read_text(encoding="utf-8")).get("clean_record_start_utc")
    except (OSError, ValueError):
        return None
    return str(v)[:10] if v else None


def start_date() -> dt.date:
    """The first day of the Clean Record (configured value, else the owner decision)."""
    return dt.date.fromisoformat(configured_start() or OWNER_DECISION)


def _ddmm(d: dt.date) -> str:
    return f"{d.day}.{d.month}"


def season_label() -> str:
    """'Season to 20.11 (pre-repair models)': the last day before the Clean Record start."""
    return f"Season to {_ddmm(start_date() - dt.timedelta(days=1))} (pre-repair models)"


def clean_label() -> str:
    """'Clean Record (from 21.11)'."""
    return f"Clean Record (from {_ddmm(start_date())})"


def started(today: dt.date | None = None) -> bool:
    """True from the Clean Record start day (UTC) on."""
    return (today or dt.datetime.now(dt.timezone.utc).date()) >= start_date()


def starts_text() -> str:
    return f"starts {_ddmm(start_date())}"


def split_in_tree() -> bool:
    """True when the tree carries the Clean Record split (dashboard_data/clean_record.py, the go-live merge)."""
    return importlib.util.find_spec("dashboard_data.clean_record") is not None


def page_label(page_file: str | None) -> str | None:
    """The page-wide record label for a page file, or None. Only pre-repair results pages get one, and only
    while the tree has no Clean Record split (then every figure on them is pre-repair)."""
    if not page_file or Path(page_file).name not in PRE_REPAIR_PAGES or split_in_tree():
        return None
    return season_label()
