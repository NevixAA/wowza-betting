"""Rolling live file + immutable per-match-date archive, for the append-only odds logs.

WHY (owner-approved 2026-10-22). Three append-only CSVs were rewritten whole on every run:
player_prop_odds_history.csv (59 MB, committed 75x/week), book_odds_snapshots.csv (37 MB, 45x) and
prop_quotes/2026-10.csv (16 MB, 75x). Git stores a complete new blob per commit, so the repo grew by
gigabytes a week and Streamlit Cloud began timing out cloning it. Same failure, same cure as
player_history.parquet: stop rewriting rows that can never change again.

LAYOUT. The live file stays at its CURRENT PATH (every raw-URL / V9_LOCAL reader keeps working)
but only holds rows whose MATCH DATE is recent or upcoming (match_date >= today - keep_days). Rows
of matches already played move ONCE into  <stem>_archive/<YYYY-MM-DD>.csv  beside it, keyed by
MATCH date (not capture time). That is what keeps the writers' dedupe and every close lookup
intact: an upcoming match always has its COMPLETE quote history in the live file, however early
its first quote was captured.

    roll(live)      -> move finished match-dates into the archive. Idempotent, never raises.
    read_all(live)  -> archive parts + live, exact duplicates dropped. For full-history readers.

Values are moved as TEXT (dtype=str, no NA parsing) so nothing is re-formatted on the way into the
archive, and every file is written with "\\n" line endings regardless of the OS that ran it
(pandas otherwise uses os.linesep, i.e. CRLF on a Windows checkout -- see the .gitattributes note).
Rows with an unparseable match date are never archived.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE_SUFFIX = "_archive"
DEFAULT_KEEP_DAYS = 2
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def archive_dir_for(live_path) -> Path:
    p = Path(live_path)
    return p.parent / f"{p.stem}{ARCHIVE_SUFFIX}"


def _read_text(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)


def _write_atomic(df: pd.DataFrame, p: Path) -> None:
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, p)


def _merge(prev: pd.DataFrame, new: pd.DataFrame, dedupe_subset) -> pd.DataFrame:
    cols = list(prev.columns) + [c for c in new.columns if c not in prev.columns]
    out = pd.concat([prev, new], ignore_index=True).reindex(columns=cols).fillna("")
    subset = [c for c in (dedupe_subset or []) if c in out.columns] or None
    return out.drop_duplicates(subset=subset, keep="first")


def roll(live_path, archive_dir=None, date_col: str = "match_date",
         keep_days: int = DEFAULT_KEEP_DAYS, today=None, dedupe_subset=None) -> dict:
    """Move rows whose `date_col` is older than today - keep_days into archive_dir/<date>.csv.

    Idempotent: a second call finds nothing to move. Re-running after a partial failure is safe
    (archive parts are written before the live file shrinks, and merging into an existing part
    drops duplicates). `dedupe_subset` is applied when merging into a part; None = exact rows.
    NEVER RAISES -- a failed roll must not cost a capture; the live file is simply left larger.
    """
    live = Path(live_path)
    adir = Path(archive_dir) if archive_dir else archive_dir_for(live)
    stats = {"moved": 0, "kept": 0, "parts": 0}
    try:
        if not live.exists():
            return stats
        d = _read_text(live)
        if d.empty or date_col not in d.columns:
            stats["kept"] = len(d)
            return stats
        today = pd.Timestamp(today or datetime.now(timezone.utc).date())
        cutoff = (today - timedelta(days=int(keep_days))).strftime("%Y-%m-%d")
        md = d[date_col].astype(str).str[:10]
        old = md.str.match(_DATE) & (md < cutoff)
        stats["kept"] = int((~old).sum())
        if not old.any():
            return stats
        adir.mkdir(parents=True, exist_ok=True)
        for day, g in d[old].groupby(md[old], sort=True):
            part = adir / f"{day}.csv"
            if part.exists():
                g = _merge(_read_text(part), g, dedupe_subset)
            _write_atomic(g, part)
            stats["parts"] += 1
        _write_atomic(d[~old], live)
        stats["moved"] = int(old.sum())
        log.info(f"[rolling_store] {live.name}: archived {stats['moved']:,} row(s) into "
                 f"{stats['parts']} part(s); live keeps {stats['kept']:,}")
    except Exception as e:                                          # noqa: BLE001
        log.warning(f"[rolling_store] roll of {live} skipped ({type(e).__name__}: {e})")
        stats["error"] = f"{type(e).__name__}: {e}"
    return stats


def archive_parts(live_path, archive_dir=None, dates: Iterable[str] | None = None) -> list[Path]:
    adir = Path(archive_dir) if archive_dir else archive_dir_for(live_path)
    if not adir.exists():
        return []
    parts = sorted(p for p in adir.glob("*.csv") if _DATE.match(p.stem))
    if dates is not None:
        want = {str(x)[:10] for x in dates}
        parts = [p for p in parts if p.stem in want]
    return parts


def read_all(live_path, archive_dir=None, dates: Iterable[str] | None = None,
             **read_kw) -> pd.DataFrame:
    """Archive parts (optionally only `dates`) + the live file, exact duplicate rows dropped.

    `read_kw` go to pd.read_csv for every part (usecols, dtype, ...). Returns an empty frame if
    nothing exists. Row order: archive parts by match date, then the live file.
    """
    live = Path(live_path)
    frames = [pd.read_csv(p, **read_kw) for p in archive_parts(live, archive_dir, dates)]
    if live.exists():
        frames.append(pd.read_csv(live, **read_kw))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates().reset_index(drop=True)
