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

Follow-ups of fix8/history2 (2026-11-04, same owner decisions):

  output/corrections/history_removed_duplicates.csv (extended)
      The duplicate rule now covers EVERY same-match pair the loader's own one-match de-duplication
      (src.integrity.duplicate_matches) merges between two history rows one day apart, not only the
      rows that lacked a half-time score. Same proof (one provider fixture id claimed by both copies,
      both clubs resolved, same FT); the provider date decides the survivor instead of the loader's
      "most populated, then earliest" tie-break. Column found_by tells the two apart.
  output/corrections/history_score_90min.csv
      "Settlement is on the 90-minute score." A history row proven to be a provider fixture the
      provider finished AET or PEN, whose full-time score is not the provider's score.fulltime (the
      90-minute score) but is >= it on both sides (extra time / a shoot-out only ever add goals), is
      SERVED with the provider's 90-minute score. The original history score is kept in the list
      (history_ft_home / history_ft_away) and in fd_history.parquet, which is never rewritten.
  output/corrections/history_awarded_results.csv
      "Awarded results are not real played scores." A history row proven to be a provider fixture
      with status AWD (technical loss) or WO (walk over) is excluded from the model history like a
      cup row. Empty until the provider's awarded fixtures are stored (collect_ht_scores keeps them
      from fix8/history2 on).

fix15/dates (owner decision 2026-11-07, "use the provider's date"):

  output/corrections/history_date_corrections.csv
      A SINGLE history row dated one day off the provider's own date for the same proven fixture
      (late Americas kick-offs, 23:xx UTC, filed under the next UK calendar day; no twin, so the
      duplicate rule never saw them) is SERVED on the provider's date. Proven only when the row
      claims exactly one provider fixture of the stored fixture lists (same league, both clubs
      resolved home->home / away->away), no other history row claims that fixture, the full-time
      scores agree where both hold one, the dates are exactly one day apart, and no other history
      row of the same league and clubs sits on the provider date. The list keeps the original
      date (history_date), the provider date, the provider kick-off UTC, the fixture id and the
      evidence. Applied by the LOADER only (apply_date_corrections, after the lists above and
      before any club renaming, enrichment or feature), so rest days and the rolling-window order
      use the provider's date; the collectors keep reading the history's own key. A sidecar keyed on
      the history's own key (the HT fill) is moved onto the corrected date (rekey_dated_sidecar).

fix16/readers: every DIRECT reader of output/fd_history.parquet that uses its results (the strength
rating, the edge replay, the HT graders of update_results, the dashboard pages, the price provenance
check, the join replay) applies all of the above through serve_history, the loader's order. The
builders and collectors of these lists and the raw-bank audits keep reading the raw file.

All are built by scripts/build_history_corrections.py (no call for the duplicates, scores, dates and
awarded rows; the cup fixture lists via collect_common.AFClient, budget-guarded). A list that is
absent means "no correction".

APPLYING. A listed row is removed only when the frame holds a row with the SAME key (league, date,
home_team, away_team - the history's own spelling) AND the same full-time score as recorded (a
history row that changed since the list was built is not touched, it is counted). Only history rows
are eligible (row_source football_data / fd_cache, or a frame without row_source); an API-Football
or archive row is never removed by a history correction. Before a duplicate copy is removed, an
EMPTY cell of the kept row is filled from it - the real value of the same proven match, which is
what the loader's one-match de-duplication did with that copy before (never a default).
A 90-minute score correction is applied AFTER the removals, on the same terms: same key, the row's
FT equal to the recorded history FT (else counted, left alone), history rows only. The labels built
from the score before this step (total_goals, over15/25/35, btts, and ftr where the row holds one)
are recomputed on the corrected rows only.
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
AWARDED_EXCLUDED = CORRECTIONS_DIR / "history_awarded_results.csv"
SCORE_90 = CORRECTIONS_DIR / "history_score_90min.csv"
DATE_CORR = CORRECTIONS_DIR / "history_date_corrections.csv"
#: the columns apply_date_corrections needs (the list carries the evidence beside them)
DATE_COLS = ["league", "history_date", "home_team", "away_team", "home_goals", "away_goals",
             "provider_date", "fixture_id"]
SCORE_COLS =["history_ft_home", "history_ft_away", "ft90_home", "ft90_away"]
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


def read_corrections(cup_path: Path | None = None, dup_path: Path | None = None,
                     awarded_path: Path | None = None) -> pd.DataFrame:
    """The REMOVAL lists as one frame: KEY + home_goals, away_goals (the recorded original FT),
    correction ('cup_excluded' / 'duplicate_removed' / 'awarded_excluded') and, for a duplicate,
    kept_date / kept_home_team / kept_away_team. Empty when no list exists. The 90-minute score list
    is not a removal (read_score_corrections)."""
    parts = []
    for path, kind in ((cup_path or CUP_EXCLUDED, "cup_excluded"), (dup_path or DUP_REMOVED, "duplicate_removed"),
                       (awarded_path or AWARDED_EXCLUDED, "awarded_excluded")):
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


def read_score_corrections(path: Path | None = None) -> pd.DataFrame:
    """The 90-minute score list: KEY + history_ft_home/away (the original, as recorded) + ft90_home/
    away (the provider's score.fulltime). Empty when absent or unreadable (logged)."""
    p = Path(path or SCORE_90)
    empty = pd.DataFrame(columns=KEY + SCORE_COLS)
    if not p.exists():
        return empty
    try:
        d = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        log.error(f"[history-corrections] {p.name} unreadable ({e}) - NOT applied")
        return empty
    if d.empty:
        return empty
    if not set(KEY + SCORE_COLS) <= set(d.columns):
        log.error(f"[history-corrections] {p.name} lacks {sorted(set(KEY + SCORE_COLS) - set(d.columns))} "
                  f"- NOT applied")
        return empty
    return d


def _relabel(df: pd.DataFrame, pos: list[int]) -> None:
    """Recompute the score labels of the rows at positions `pos` (in place) from home/away_goals."""
    if not pos:
        return
    hg = pd.to_numeric(df["home_goals"].iloc[pos], errors="coerce").to_numpy(dtype=float)
    ag = pd.to_numeric(df["away_goals"].iloc[pos], errors="coerce").to_numpy(dtype=float)
    tot = hg + ag
    lab = {"total_goals": tot, "over15": (tot > 1.5).astype(float), "over25": (tot > 2.5).astype(float),
           "over35": (tot > 3.5).astype(float), "btts": ((hg > 0) & (ag > 0)).astype(float)}
    for c, v in lab.items():
        if c in df.columns:
            df.iloc[pos, df.columns.get_loc(c)] = v
    if "ftr" in df.columns:
        ci = df.columns.get_loc("ftr")
        for p, h, a in zip(pos, hg, ag):
            if isinstance(df.iat[p, ci], str):        # a row without ftr keeps none (never added)
                df.iat[p, ci] = "H" if h > a else "A" if h < a else "D"


def apply_score_corrections(df: pd.DataFrame, scores: pd.DataFrame | None = None,
                            source_col: str = "row_source", tag: str = "history") -> tuple[pd.DataFrame, dict]:
    """(df with the listed history rows served at the provider's 90-minute score, counts)."""
    scores = read_score_corrections() if scores is None else scores
    rep = {"score_rows_corrected_90min": 0, "score_listed_rows_changed_since_listing": 0}
    if df is None or df.empty or scores.empty or not set(KEY) <= set(df.columns) \
            or not {"home_goals", "away_goals"} <= set(df.columns):
        return df, rep
    eligible = (df[source_col].isin(HISTORY_SOURCES).tolist() if source_col in df.columns
                else [True] * len(df))
    rec = {}
    for r, k in zip(scores.itertuples(index=False), _keys(scores)):
        rec[k] = (_ft(r.history_ft_home, r.history_ft_away), _ft(r.ft90_home, r.ft90_away))
    hg, ag = df["home_goals"].tolist(), df["away_goals"].tolist()
    hit = []
    for p, (k, ok) in enumerate(zip(_keys(df), eligible)):
        if not ok or k not in rec:
            continue
        orig, ft90 = rec[k]
        if ft90 is None or _ft(hg[p], ag[p]) != orig:
            rep["score_listed_rows_changed_since_listing"] += 1      # not the row proven: left alone
            continue
        hit.append((p, ft90))
    if hit:
        df = df.copy()
        hc, ac = df.columns.get_loc("home_goals"), df.columns.get_loc("away_goals")
        for p, (h, a) in hit:
            df.iat[p, hc], df.iat[p, ac] = float(h), float(a)
        _relabel(df, [p for p, _ in hit])
        rep["score_rows_corrected_90min"] = len(hit)
        log.info(f"[{tag}] 90-minute score (owner rule): {len(hit)} history row(s) of AET/PEN fixtures "
                 f"served at the provider's score.fulltime (originals kept in {SCORE_90.name})")
    return df, rep


def apply_history_corrections(df: pd.DataFrame, corr: pd.DataFrame | None = None,
                              source_col: str = "row_source", tag: str = "history",
                              scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """`df` without the listed rows (see the module docstring); the kept copy of a removed duplicate
    takes the removed copy's values in its EMPTY cells; then the 90-minute score list. Counts in
    df.attrs['history_corrections']."""
    corr = read_corrections() if corr is None else corr
    if df is None or df.empty or not set(KEY) <= set(df.columns):
        return df
    attrs = dict(getattr(df, "attrs", {}) or {})
    out, rep = _apply_removals(df, corr, source_col, tag) if not corr.empty else (df, {})
    out, srep = apply_score_corrections(out, scores, source_col, tag)
    if not rep and not any(srep.values()):
        return out
    out.attrs = attrs
    out.attrs["history_corrections"] = {**rep, **srep}
    return out


def _apply_removals(df: pd.DataFrame, corr: pd.DataFrame, source_col: str, tag: str) -> tuple[pd.DataFrame, dict]:
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
    drop, changed, filled, n_cup, n_awd = [], 0, 0, 0, 0
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
        n_awd += hit.correction == "awarded_excluded"
        if hit.correction == "duplicate_removed" and isinstance(hit.kept_date, str):
            fill_from.append((p, (k[0], str(hit.kept_date)[:10], str(hit.kept_home_team),
                                  str(hit.kept_away_team))))
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
    n_dup = len(drop) - n_cup - n_awd
    rep = {"rows_removed": len(drop), "cup_rows_excluded": n_cup, "awarded_rows_excluded": n_awd,
           "duplicate_rows_removed": n_dup, "cells_filled_from_removed_copy": filled,
           "listed_rows_changed_since_listing": changed}
    if drop or changed:
        log.info(f"[{tag}] history corrections (owner 2026-11-04): {n_cup} proven cup row(s) and {n_awd} "
                 f"awarded result(s) excluded, {n_dup} duplicate row(s) removed, {filled} empty cell(s) "
                 f"of the kept copy filled from the removed one; {changed} listed row(s) differ from the "
                 f"listing (left in)")
    return out, rep


def read_date_corrections(path: Path | None = None) -> pd.DataFrame:
    """The date list (DATE_COLS + evidence), dates as YYYY-MM-DD. Empty when absent or unreadable
    (logged: an unreadable list is NOT applied)."""
    p = Path(path or DATE_CORR)
    empty = pd.DataFrame(columns=DATE_COLS)
    if not p.exists():
        return empty
    try:
        d = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        log.error(f"[history-corrections] {p.name} unreadable ({e}) - NOT applied")
        return empty
    if d.empty:
        return empty
    if not set(DATE_COLS) <= set(d.columns):
        log.error(f"[history-corrections] {p.name} lacks {sorted(set(DATE_COLS) - set(d.columns))} - NOT applied")
        return empty
    d = d.copy()
    d["history_date"] = _day(d["history_date"]).to_numpy()
    d["provider_date"] = _day(d["provider_date"]).to_numpy()
    return d


def _date_map(dates: pd.DataFrame) -> dict:
    """{(league, history_date, home, away): (recorded FT or None, provider_date)}."""
    out = {}
    for r, k in zip(dates.itertuples(index=False),
                    _keys(dates, ["league", "history_date", "home_team", "away_team"])):
        if isinstance(r.provider_date, str) and r.provider_date:
            out[k] = (_ft(r.home_goals, r.away_goals), r.provider_date)
    return out


def apply_date_corrections(df: pd.DataFrame, dates: pd.DataFrame | None = None,
                           source_col: str = "row_source", tag: str = "history") -> pd.DataFrame:
    """`df` with every listed history row served on the provider's date (module docstring,
    fix15/dates). A row is re-dated only when its key (the history's own league, date and spelling)
    is listed AND its full-time score equals the recorded one (a missing score equals a missing
    score); else it is counted and left alone. History rows only (row_source football_data /
    fd_cache, or a frame without row_source). Counts per league in df.attrs['history_date_corrections']."""
    dates = read_date_corrections() if dates is None else dates
    if df is None or df.empty or dates is None or dates.empty or not set(KEY) <= set(df.columns):
        return df
    rec = _date_map(dates)
    eligible = (df[source_col].isin(HISTORY_SOURCES).tolist() if source_col in df.columns
                else [True] * len(df))
    hg = df["home_goals"].tolist() if "home_goals" in df.columns else [None] * len(df)
    ag = df["away_goals"].tolist() if "away_goals" in df.columns else [None] * len(df)
    hit, changed, per = [], 0, {}
    for p, (k, ok) in enumerate(zip(_keys(df), eligible)):
        if not ok or k not in rec:
            continue
        ft, pdate = rec[k]
        lg = per.setdefault(k[0], {"rows_redated": 0, "listed_rows_changed_since_listing": 0})
        if _ft(hg[p], ag[p]) != ft:
            changed += 1
            lg["listed_rows_changed_since_listing"] += 1
            continue
        hit.append((p, pdate))
        lg["rows_redated"] += 1
    if not hit and not changed:
        return df
    attrs = dict(getattr(df, "attrs", {}) or {})
    out = df.copy()
    ci = out.columns.get_loc("date")
    is_dt = pd.api.types.is_datetime64_any_dtype(out["date"])
    for p, pdate in hit:
        out.iat[p, ci] = pd.Timestamp(pdate) if is_dt else pdate
    out.attrs = attrs
    out.attrs["history_date_corrections"] = {"per_league": per, "rows_redated": len(hit),
                                             "listed_rows_changed_since_listing": changed}
    log.info(f"[{tag}] provider date (owner 2026-11-07): {len(hit)} history row(s) dated one day off "
             f"the provider's date for the same proven fixture served on the provider's date "
             f"(originals kept in {DATE_CORR.name}); {changed} listed row(s) differ from the listing "
             f"(left alone)")
    return out


def rekey_dated_sidecar(fill: pd.DataFrame, dates: pd.DataFrame | None = None,
                        tag: str = "sidecar") -> pd.DataFrame:
    """A sidecar keyed on the history's OWN key (league, date, home_team, away_team - e.g.
    output/collect/fd_history_ht_fill.parquet) with its rows on a re-dated history key moved onto the
    provider's date, so it still reaches the row apply_date_corrections re-dated. A row whose new key
    is already held by another sidecar row is left where it is (counted): one answer per key."""
    dates = read_date_corrections() if dates is None else dates
    if fill is None or fill.empty or dates is None or dates.empty or not set(KEY) <= set(fill.columns):
        return fill
    rec = _date_map(dates)
    keys = _keys(fill)
    present = set(keys)
    moved, held = [], 0
    for i, k in enumerate(keys):
        if k not in rec:
            continue
        nk = (k[0], rec[k][1], k[2], k[3])
        if nk in present:
            held += 1
            continue
        moved.append((i, rec[k][1]))
        present.add(nk)
    if not moved and not held:
        return fill
    attrs = dict(getattr(fill, "attrs", {}) or {})
    out = fill.copy()
    ci = out.columns.get_loc("date")
    is_dt = pd.api.types.is_datetime64_any_dtype(out["date"])
    for i, pdate in moved:
        out.iat[i, ci] = pd.Timestamp(pdate) if is_dt else pdate
    out.attrs = attrs
    out.attrs["dated_sidecar_rekey"] = {"moved_to_provider_date": len(moved), "left_new_key_held": held}
    log.info(f"[{tag}] provider date (owner 2026-11-07): {len(moved)} sidecar row(s) moved onto the "
             f"re-dated history key, {held} left (the new key already answered)")
    return out


def redated_fixture_ids(dates: pd.DataFrame | None = None) -> dict:
    """{provider fixture id: provider date} of the re-dated history rows (src/af_history_corrections
    no longer bridges the af_history joins of these fixtures: the served row is on the provider date)."""
    dates = read_date_corrections() if dates is None else dates
    if dates is None or dates.empty:
        return {}
    f = pd.to_numeric(dates["fixture_id"], errors="coerce")
    return {int(a): str(b) for a, b in zip(f, dates["provider_date"]) if a == a and isinstance(b, str)}


def drop_corrected_rows(history: pd.DataFrame, tag: str = "history") -> pd.DataFrame:
    """apply_history_corrections for a raw history frame (no row_source column)."""
    return apply_history_corrections(history, tag=tag)


#: the columns a direct reader of fd_history must hold for serve_history to recognise a listed row
#: (the key and the recorded full-time score)
SERVE_COLS = KEY + ["home_goals", "away_goals"]


def serve_history(history: pd.DataFrame, tag: str = "history", source_col: str = "row_source") -> pd.DataFrame:
    """fd_history as the LOADER serves it (fix16/readers): every owner correction, in the loader's
    order (src.data_loader.identity_before_one_match) - the removal lists (proven cup rows, awarded
    results, duplicate copies; the kept copy's empty cells filled from the removed one) and the
    90-minute scores (apply_history_corrections), then the provider dates (apply_date_corrections).
    For a DIRECT reader of output/fd_history.parquet; it must read at least SERVE_COLS, else a listed
    row is not recognised and is counted as changed (never removed / re-dated on a guess). The frame
    must hold history rows only, or a row_source column (only football_data / fd_cache rows are
    eligible). Counts in attrs['history_corrections'] and attrs['history_date_corrections']."""
    if history is None or history.empty:
        return history
    out = apply_history_corrections(history, source_col=source_col, tag=tag)
    return apply_date_corrections(out, source_col=source_col, tag=tag)
