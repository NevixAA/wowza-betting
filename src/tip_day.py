"""The TIP DAY a fixture belongs to - which morning album carries it (owner 2026-10-25).

    from src.tip_day import add_tip_day, tip_day_of

The album goes out on the Israeli morning (~06:00 UTC = 08:00-09:00 Asia/Jerusalem). Fixtures are
filed under the UTC date of kickoff, so a match kicking off before the album - J-League at 05:00-07:00
UTC, Liga MX / MLS / Argentina / Brazil at 00:00-03:00 UTC - was listed in the album sent AFTER it had
started or finished. 166 of 1,067 tipped fixtures (2026-10) kicked off before 08:00 UTC; ~95% of those
tips already existed when the PREVIOUS morning's album went out.

A tip day therefore runs from ROLLOVER_HOUR (10:00 Israel time, DST-safe because it is computed in the
Israeli zone) to 10:00 the next day: every match that kicks off before the next album is in today's.

Kickoff times come from predictions.csv (upcoming fixtures) and v91_shadow.csv (every fixture's
kickoff, recorded at first sight - so played matches keep theirs). A row with no known kickoff keeps its
match date, i.e. exactly the old behaviour. Display / grouping only: no tip, price, tier or stake changes.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import config

TZ = ZoneInfo(os.getenv("TIP_DAY_TZ", "Asia/Jerusalem"))
ROLLOVER_HOUR = int(os.getenv("TIP_DAY_ROLLOVER_HOUR", "10"))
_KEY = ["league", "home_team", "away_team"]


def tip_day_of(kickoff_utc, match_date: str | None = None) -> str:
    """'YYYY-MM-DD' tip day for one kickoff; falls back to the match date when kickoff is unknown."""
    ko = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    if pd.isna(ko):
        return str(match_date or "")[:10]
    local = ko.tz_convert(TZ)
    return (local - pd.Timedelta(hours=ROLLOVER_HOUR)).strftime("%Y-%m-%d")


def today(now: datetime | None = None) -> str:
    """The tip day a morning album sent at `now` carries: the Israeli calendar date. The album's send
    window (05:00-11:00 UTC) is the same date in Israel and in UTC, so this equals the UTC date there."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    return now.tz_convert(TZ).strftime("%Y-%m-%d")


def kickoffs() -> pd.DataFrame:
    """league, match_date, home_team, away_team, kickoff_utc - upcoming (predictions.csv) + played
    (v91_shadow.csv, first-sight kickoff). One row per fixture."""
    parts = []
    for name, date_col in (("predictions.csv", "date"), ("v91_shadow.csv", "match_date")):
        p = config.OUTPUT_DIR / name
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=lambda c: c in {*_KEY, date_col, "kickoff_utc"}, low_memory=False)
        except Exception:                                               # noqa: BLE001 - grouping must never fail
            continue
        if "kickoff_utc" not in d.columns or date_col not in d.columns:
            continue
        parts.append(d.rename(columns={date_col: "match_date"}))
    if not parts:
        return pd.DataFrame(columns=[*_KEY, "match_date", "kickoff_utc"])
    d = pd.concat(parts, ignore_index=True)
    d["match_date"] = d["match_date"].astype(str).str[:10]
    d = d[pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce").notna()]
    return d.drop_duplicates([*_KEY, "match_date"], keep="first")


def add_tip_day(df: pd.DataFrame, date_col: str = "match_date", ko: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return df with `kickoff_utc` (filled where known) and `tip_day`. Never drops or reorders rows."""
    out = df.copy()
    md = out[date_col].astype(str).str[:10]
    k = ko if ko is not None else kickoffs()
    have = out["kickoff_utc"] if "kickoff_utc" in out.columns else pd.Series(pd.NA, index=out.index)
    if len(k) and set(_KEY) <= set(out.columns):
        look = k.set_index([*_KEY, "match_date"])["kickoff_utc"]
        idx = pd.MultiIndex.from_arrays([out[c].astype(str) for c in _KEY] + [md])
        look.index = pd.MultiIndex.from_arrays([look.index.get_level_values(i).astype(str) for i in range(4)])
        found = pd.Series(look.reindex(idx).to_numpy(), index=out.index)
        have = have.where(pd.to_datetime(have, utc=True, errors="coerce").notna(), found)
    out["kickoff_utc"] = have
    out["tip_day"] = [tip_day_of(kv, m) for kv, m in zip(out["kickoff_utc"], md)]
    return out
