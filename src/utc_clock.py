"""The repo's one clock for deciding a DAY: UTC.

Every stored day in this repo is a UTC calendar day: fixture `date`/`match_date`, ledger rows,
snapshot and quote day-files, digest keys. GitHub runners are UTC, so a naive `datetime.now()`
or `date.today()` happens to agree there. On a laptop in Israel (UTC+2/+3) it does not: between
local midnight and UTC midnight the local date is one day AHEAD, so "today's" filters drop the
UTC day's fixtures and day files get filed under the wrong date (found 2026-11-07, fix13).

Use these helpers wherever code means "today" or "now" in the stored-data sense. Naive values
returned here are UTC wall-clock times (the repo convention for naive timestamps, see
src/shadow_settlement.py), so they compare directly with naive UTC columns.
Display-only clocks (an "HH:MM sent at" footer) are not covered and are left alone.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd


def utc_now() -> datetime:
    """Current UTC time as a NAIVE datetime (UTC wall clock)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_today() -> date:
    """Current UTC calendar date."""
    return datetime.now(timezone.utc).date()


def utc_today_str() -> str:
    """Current UTC calendar date as YYYY-MM-DD."""
    return utc_today().strftime("%Y-%m-%d")


def utc_today_ts() -> pd.Timestamp:
    """Current UTC calendar date as a naive, normalized pandas Timestamp (midnight UTC)."""
    return pd.Timestamp(utc_today())
