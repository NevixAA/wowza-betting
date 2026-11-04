"""
history_corrections.py - the owner's corrections to the match history, applied in the READ only.

Owner decisions of 2026-11-04 on output/fd_history.parquet (football-data.co.uk). The history file is
never rewritten; each correction is a row of a sidecar list BESIDE it, with the original values kept,
and the loader (src/data_loader.one_club_one_match and the local-workbook path) and the HT collector
(scripts/collect_ht_scores.py) apply the lists when they read the history.

  output/corrections/history_excluded_cup_rows.csv
      "Cups are not in Wowza tips; keep cups only as player data." History rows filed under a league
      that are PROVEN to be a domestic-cup fixture (Argentina's Copa de la Liga Profesional, Copa
      Argentina, ...): the provider has no fixture of the league within +-1 day of the row, and exactly
      one fixture of a national cup of the league's country on that day (or the adjacent day) has
      both clubs, resolved home->home and away->away by src.team_names.resolve_result, and the same
      full-time score where both hold one. Evidence per row: provider fixture id, cup league id and
      name, season, kick-off, provider club names and score. A row that cannot be proven stays IN.
  output/corrections/history_removed_duplicates.csv
      "Duplicates: remove." The same match filed twice, one day apart (one copy dated by the local
      kick-off date, the other by the UTC date). Removed only when the twin is proven the same
      fixture: both rows claim the SAME provider fixture id, both clubs resolve onto that fixture's
      clubs, and the full-time scores agree where both hold one. The provider's fixture date (UTC)
      decides which copy stays. The removed row's full original values and the kept row's key are
      kept in the list.

Both are built by scripts/build_history_corrections.py (no call for the duplicates; the cup fixture
lists via collect_common.AFClient, budget-guarded). A list that is absent means "no correction".

APPLYING. A listed row is removed only when the frame holds a row with the SAME key (league, date,
home_team, away_team - the history's own spelling) AND the same full-time score as recorded (a
history row that changed since the list was built is not touched, it is counted). Only history rows
are eligible (row_source football_data / fd_cache, or a frame without row_source); an API-Football
or archive row is never removed by a history correction. Before a duplicate copy is removed, an
EMPTY cell of the kept row is filled from it - the real value of the same proven match, which is
what the loader's one-match de-duplication did with that copy before (never a default).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = ROOT / "output" / "corrections"
CUP_EXCLUDED = CORRECTIONS_DIR / "history_excluded_cup_rows.csv"
DUP_REMOVED = CORRECTIONS_DIR / "history_removed_duplicates.csv"
KEY = ["league", "date", "home_team", "away_team"]
HISTORY_SOURCES = {"football_data", "fd_cache"}
#: never copied from a removed duplicate onto the kept row (identity, provenance, derived labels)
_NO_FILL = set(KEY) | {"row_source", "season", "ftr"}


def _day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _keys(df: pd.DataFrame, cols=KEY) -> list[tuple]:
    return list(zip(df[cols[0]].astype(str), _day(df[cols[1]]),
                    df[cols[2]].astype(str), df[cols[3]].astype(str)))


def _ft(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return None
        return int(a), int(b)
    except (TypeError, ValueError):
        return None


def read_corrections(cup_path: Path | None = None, dup_path: Path | None = None) -> pd.DataFrame:
    """Both lists as one frame: KEY + home_goals, away_goals (the recorded original FT),
    correction ('cup_excluded' / 'duplicate_removed') and, for a duplicate, kept_date /
    kept_home_team / kept_away_team. Empty when neither list exists."""
    parts = []
    for path, kind in ((cup_path or CUP_EXCLUDED, "cup_excluded"), (dup_path or DUP_REMOVED, "duplicate_removed")):
        p = Path(path)
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
        except (OSError, ValueError, pd.errors.EmptyDataError) as e:
            log.error(f"[history-corrections] {p.name} unreadable ({e}) - NOT applied")
            continue
        if d.empty:
            continue
        need = set(KEY) | {"home_goals", "away_goals"}
        if not need <= set(d.columns):
            log.error(f"[history-corrections] {p.name} lacks {sorted(need - set(d.columns))} - NOT applied")
            continue
        d = d.assign(correction=kind)
        for c in ("kept_date", "kept_home_team", "kept_away_team"):
            if c not in d.columns:
                d[c] = None
        parts.append(d[KEY + ["home_goals", "away_goals", "correction",
                              "kept_date", "kept_home_team", "kept_away_team"]])
    if not parts:
        return pd.DataFrame(columns=KEY + ["home_goals", "away_goals", "correction",
                                           "kept_date", "kept_home_team", "kept_away_team"])
    return pd.concat(parts, ignore_index=True)


def corrected_mask(df: pd.DataFrame, corr: pd.DataFrame | None = None) -> pd.Series:
    """True for rows of `df` whose KEY is a listed correction (score not compared: for sidecars
    keyed on the history's key, e.g. the HT fill, which hold no history FT)."""
    corr = read_corrections() if corr is None else corr
    if df is None or df.empty or corr.empty or not set(KEY) <= set(df.columns):
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    listed = set(_keys(corr))
    return pd.Series([k in listed for k in _keys(df)], index=df.index, dtype=bool)


def apply_history_corrections(df: pd.DataFrame, corr: pd.DataFrame | None = None,
                              source_col: str = "row_source", tag: str = "history") -> pd.DataFrame:
    """`df` without the listed rows (see the module docstring); the kept copy of a removed duplicate
    takes the removed copy's values in its EMPTY cells. Counts in df.attrs['history_corrections']."""
    corr = read_corrections() if corr is None else corr
    if df is None or df.empty or corr.empty or not set(KEY) <= set(df.columns):
        return df
    eligible = (df[source_col].isin(HISTORY_SOURCES) if source_col in df.columns
                else pd.Series(True, index=df.index))
    rec = {}
    for r, k in zip(corr.itertuples(index=False), _keys(corr)):
        rec.setdefault(k, []).append(r)
    keys = _keys(df)
    ok_l = eligible.tolist()
    hg = df["home_goals"].tolist() if "home_goals" in df.columns else [None] * len(df)
    ag = df["away_goals"].tolist() if "away_goals" in df.columns else [None] * len(df)
    # POSITIONS, not index labels: a concatenated frame can repeat a label
    drop, changed, filled, n_cup = [], 0, 0, 0
    fill_from: list[tuple[int, tuple]] = []          # (position to remove, kept key)
    for p, (k, ok) in enumerate(zip(keys, ok_l)):
        if not ok or k not in rec:
            continue
        ft_row = _ft(hg[p], ag[p])
        hit = next((r for r in rec[k] if _ft(r.home_goals, r.away_goals) == ft_row), None)
        if hit is None:
            changed += 1                             # the row is not the one proven: left alone
            continue
        drop.append(p)
        n_cup += hit.correction == "cup_excluded"
        if hit.correction == "duplicate_removed" and isinstance(hit.kept_date, str):
            fill_from.append((p, (k[0], str(hit.kept_date)[:10], str(hit.kept_home_team),
                                  str(hit.kept_away_team))))
    attrs = dict(getattr(df, "attrs", {}) or {})
    if fill_from:
        at = {}
        for p, (k, ok) in enumerate(zip(keys, ok_l)):
            if ok:
                at.setdefault(k, []).append(p)
        drop_set = set(drop)
        df = df.copy()
        cols = [df.columns.get_loc(c) for c in df.columns if c not in _NO_FILL and not str(c).startswith("_")]

        def _empty(v):
            return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT

        for p, kk in fill_from:
            for q in at.get(kk, []):
                if q in drop_set:
                    continue
                for ci in cols:
                    a, b = df.iat[q, ci], df.iat[p, ci]
                    if _empty(a) and not _empty(b):
                        df.iat[q, ci] = b
                        filled += 1
    keep = np.ones(len(df), dtype=bool)
    keep[drop] = False
    out = df[keep]
    rep = {"rows_removed": len(drop), "cup_rows_excluded": n_cup,
           "duplicate_rows_removed": len(drop) - n_cup, "cells_filled_from_removed_copy": filled,
           "listed_rows_changed_since_listing": changed}
    out.attrs = attrs
    out.attrs["history_corrections"] = rep
    if drop or changed:
        log.info(f"[{tag}] history corrections (owner 2026-11-04): {n_cup} proven cup row(s) excluded, "
                 f"{len(drop) - n_cup} duplicate row(s) removed, {filled} empty cell(s) of the kept copy "
                 f"filled from the removed one; {changed} listed row(s) differ from the listing (left in)")
    return out


def drop_corrected_rows(history: pd.DataFrame, tag: str = "history") -> pd.DataFrame:
    """apply_history_corrections for a raw history frame (no row_source column)."""
    return apply_history_corrections(history, tag=tag)
