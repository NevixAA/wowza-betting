"""
af_history_corrections.py - the owner's duplicate rule for output/af_history.parquet, applied in the
READ only (owner decisions 2026-11-04; fix14/afdupes 2026-11-07).

THE DEFECT. The API-Football history holds the SAME provider fixture on TWO dates one day apart in
the leagues that play across midnight UTC (USA MLS, Brazil Serie A, Mexico Liga MX, Argentina
Primera Division): one copy dated by the provider's UTC date, the other by the local kick-off date
(mostly a row the old loader bank wrote back - src/af_history_provenance). Example: Liga MX Leon v
Toluca, fixture 1551013, kick-off 2026-10-24T23:00Z, stored on 2026-10-24 (provider row, FT 1-1,
16 shots) and on 2026-10-25 (written back, 27 shots). The fixture matcher of
scripts/collect_af_history_gap.py refused every such fixture as "claimed by two dates" (2,268 rows
in --insidebox, 3,029 in --fullstats), while --insidebox, which only matches the rows lacking
data, placed fills on whichever copy happened to lack them.

THE RULE (the fd_history rule of src/history_corrections, same owner decision). The provider's
date decides. A copy is removed only when its twin is PROVEN to be the same fixture: both copies
claim the SAME provider fixture id, both clubs resolve onto that fixture's clubs (by the provider's
team ids where the row stores them, else src.team_names.resolve_result on the provider's list of
the day), the copies are exactly one day apart, and the full-time score agrees between the copies
and with the provider wherever both hold one. The copy on the provider's fixture date (UTC) stays.
Anything else is a CONFLICT: counted, listed, left alone. scripts/build_af_history_corrections.py
builds the lists from committed data (0 API calls):

  output/corrections/af_history_removed_duplicates.csv   the removed copies with ALL their original
      af_history values, the kept copy's key (kept_date / kept_home_team / kept_away_team) and the
      evidence (provider fixture id, provider date and kick-off UTC, local date, provider clubs and
      score, how each copy was matched, which copy was written back).
  output/corrections/af_history_duplicate_conflicts.csv   the pairs that could not be proven.

af_history.parquet is never rewritten.

APPLYING (apply_af_history_corrections, every reader of af_history: src/data_loader, the gap
collector, src/strength/results, src/research/edge_replay). A listed row is removed only when the
frame holds a row with the SAME key (league, date, home_team, away_team - af_history's own
spelling) AND the same recorded identity cells (FT score, fixture id, shots, shots on target; a
missing cell equals a missing cell) - a row that changed since the list was built is counted and
left in. The kept row must be present too, else nothing is removed (counted).

Values of the removed copy reach the kept copy only when they are PROVIDER values:
  * a written-back removed copy (no fixture id, no FT score) gives nothing: the loader never serves
    a written-back statistic (src/af_history_provenance), and a removal must not launder one in;
  * a provider removed copy fills the EMPTY cells of the kept copy ("removed copies fill empty cells
    of the kept copy", as the fd rule). When the kept copy is itself written back, its own statistic
    cells were never served (unproven) - they are cleared first, so the provider copy's values are
    what the kept row serves and nothing unproven becomes served by the fill;
  * where both copies are provider rows holding DIFFERENT values of one statistic (16 vs 27 shots),
    the kept cell is made MISSING - never averaged, never the kept copy's value by default. The
    provider's own fixture-statistics answer then decides: the loader's --fullstats / --insidebox
    sidecar fills (nan_only, by the same proven fixture) put the provider value there when held;
    otherwise the cell stays missing.

THE SIDECARS (rekey_sidecar). output/collect/af_history_stats_fill.parquet and
af_history_fullstats_fill.parquet are keyed on af_history's key. A sidecar row keyed on a removed
copy, carrying the SAME provider fixture id the pair was proven by, is moved onto the kept copy's
key (it is that fixture's answer); when the kept key already holds a row of the sidecar, the moved
row is dropped (counted) - one answer per fixture. A sidecar row on a removed key with another
fixture id is left where it is (counted). The collector rewrites both sidecars on the kept keys on
its next run (scripts/collect_af_history_gap.py applies the same two functions).

THE SERVED ROW'S DATE (add_join_aliases, src/data_loader only). Measured 2026-11-07: the served
frame's row of these matches is the football-data row, dated by the UK calendar day - the SAME day
as the removed copy (the old loader bank wrote that copy back FROM the football-data row), one day
after the provider's date (Leon v Toluca: provider 2026-10-24 23:00 UTC, fd_history 2026-10-25).
The loader joins af_history and the sidecars onto the served frame by (league, date, clubs), so a
kept copy on the provider date alone would reach NO served row and every value placed on the
removed copy's key until now (404 --insidebox rows) would be lost. So, for the loader's joins only,
the kept copy's values (after the rule above) are ALSO offered under the removed copy's key - the
same proven fixture, one row per key, never a second match: af_history and the collectors still
hold one row per fixture, and the served frame holds one row per match (data truth rule 5). Re-dating
the served row to the provider date is the football-data side's decision, not made here.

fix15/dates (owner 2026-11-07, "use the provider's date"): the football-data side now makes it -
src.history_corrections.apply_date_corrections serves those football-data rows on the provider's
date (output/corrections/history_date_corrections.csv), so the kept copy joins them directly and the
alias of such a pair is no longer added (redundant; the fixture id and provider date must agree). The
alias stays for a two-date pair whose served row is not re-dated.

SAME-DAY SPELLING DUPLICATES (fix15/dates, found_by af_same_day_spelling): the same list also removes
the extra copies of one provider fixture held on the provider's date under different club spellings
('Club Leon' / 'Leon'), proven as above (same provider fixture id, both clubs, same score). The kept
copy is on the same date, so plan_fill, the sidecar re-key and the identity check apply unchanged and
no join alias is added.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = ROOT / "output" / "corrections"
AF_DUP_REMOVED = CORRECTIONS_DIR / "af_history_removed_duplicates.csv"
AF_DUP_CONFLICTS = CORRECTIONS_DIR / "af_history_duplicate_conflicts.csv"
KEY = ["league", "date", "home_team", "away_team"]
KEPT = ["kept_date", "kept_home_team", "kept_away_team"]
#: the recorded cells a listed row must still hold to be the row that was proven
SIGNATURE = ["FTHG", "FTAG", "fixture_id", "HS", "AS", "HST", "AST"]
#: statistic cells: a conflict between two provider copies makes the kept cell missing
STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
             "HXG", "AXG", "HIB", "AIB"]
#: cells a provider removed copy may fill on the kept copy (identity: league, date, clubs, season
#: are never copied; FTR only with the score it belongs to)
FILL_COLS = ["FTHG", "FTAG", "FTR", "fixture_id", "home_team_id", "away_team_id"] + STAT_COLS


def _day(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), errors="coerce").dt.strftime("%Y-%m-%d")


def _keys(df: pd.DataFrame, cols=KEY) -> list[tuple]:
    return list(zip(df[cols[0]].astype(str), _day(df[cols[1]]).tolist(),
                    df[cols[2]].astype(str), df[cols[3]].astype(str)))


def _empty(v) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    try:
        return bool(isinstance(v, float) and np.isnan(v))
    except TypeError:
        return False


def _num(v):
    if _empty(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _sig(row: dict) -> tuple:
    return tuple(_num(row.get(c)) for c in SIGNATURE)


def read_af_corrections(path: Path | None = None) -> pd.DataFrame:
    """The removed-duplicates list (KEY + SIGNATURE + KEPT + provider_fixture_id), date as
    YYYY-MM-DD. Empty when absent or unreadable (logged: an unreadable list is NOT applied)."""
    p = Path(path or AF_DUP_REMOVED)
    cols = KEY + SIGNATURE + KEPT + ["provider_fixture_id"]
    empty = pd.DataFrame(columns=cols)
    if not p.exists():
        return empty
    try:
        d = pd.read_csv(p, low_memory=False)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        log.error(f"[af-history-corrections] {p.name} unreadable ({e}) - NOT applied")
        return empty
    if d.empty:
        return empty
    if not set(cols) <= set(d.columns):
        log.error(f"[af-history-corrections] {p.name} lacks {sorted(set(cols) - set(d.columns))} "
                  f"- NOT applied")
        return empty
    d = d[cols].copy()
    d["date"] = _day(d["date"]).to_numpy()
    d["kept_date"] = _day(d["kept_date"]).to_numpy()
    return d


def _written_back(row: dict) -> bool:
    """src.af_history_provenance.written_back_mask for one row."""
    no_fx = _empty(row.get("fixture_id"))
    no_ft = _empty(row.get("FTHG")) or _empty(row.get("FTAG"))
    return no_fx and no_ft


def plan_fill(kept: dict, removed: dict) -> tuple[dict, list, list, list]:
    """(new values for the kept row, cells filled, cells made missing, unproven cells cleared) for
    one proven pair - the rule of the module docstring. Pure: used by the loader and recorded by the
    build script as evidence."""
    new: dict = {}
    filled, missing, cleared = [], [], []
    if _written_back(removed):
        return new, filled, missing, cleared
    cur = dict(kept)
    if _written_back(kept):
        for c in STAT_COLS:
            if c in cur and not _empty(cur[c]):
                cur[c] = np.nan
                new[c] = np.nan
                cleared.append(c)
    for c in FILL_COLS:
        if c not in removed:
            continue
        a, b = cur.get(c, np.nan), removed[c]
        if _empty(b):
            continue
        if _empty(a):
            new[c] = b
            filled.append(c)
            if c in cleared:
                cleared.remove(c)
        elif c in STAT_COLS and _num(a) != _num(b):
            new[c] = np.nan
            missing.append(c)
    return new, filled, missing, cleared


def apply_af_history_corrections(af: pd.DataFrame, corr: pd.DataFrame | None = None,
                                 tag: str = "af_history") -> pd.DataFrame:
    """`af` (af_history's own columns) without the listed removed copies, the kept copies corrected
    by plan_fill. Counts per league in out.attrs['af_history_corrections']."""
    corr = read_af_corrections() if corr is None else corr
    if af is None or af.empty or corr is None or corr.empty or not set(KEY) <= set(af.columns):
        return af
    keys = _keys(af)
    pos = {k: i for i, k in enumerate(keys)}                  # af_history's key is unique
    recs = af.to_dict("records")
    drop, changed, kept_absent = [], 0, 0
    per: dict = {}
    updates: dict = {}
    for r, k in zip(corr.to_dict("records"), _keys(corr)):
        p = pos.get(k)
        if p is None:
            continue
        lg = per.setdefault(k[0], {"rows_removed": 0, "cells_filled": 0, "cells_made_missing": 0,
                                   "unproven_cells_cleared": 0, "listed_rows_changed_since_listing": 0,
                                   "kept_row_absent": 0})
        if _sig(recs[p]) != _sig(r):
            changed += 1
            lg["listed_rows_changed_since_listing"] += 1
            continue
        kk = (k[0], str(r["kept_date"])[:10], str(r["kept_home_team"]), str(r["kept_away_team"]))
        q = pos.get(kk)
        if q is None or q == p:
            kept_absent += 1
            lg["kept_row_absent"] += 1
            continue
        kept_now = {**recs[q], **updates.get(q, {})}
        new, filled, missing, cleared = plan_fill(kept_now, recs[p])
        if new:
            updates.setdefault(q, {}).update(new)
        drop.append(p)
        lg["rows_removed"] += 1
        lg["cells_filled"] += len(filled)
        lg["cells_made_missing"] += len(missing)
        lg["unproven_cells_cleared"] += len(cleared)
    if not drop and not changed and not kept_absent:
        return af
    attrs = dict(getattr(af, "attrs", {}) or {})
    out = af.copy()
    for q, new in updates.items():
        for c, v in new.items():
            if c in out.columns:
                out.iat[q, out.columns.get_loc(c)] = v
    keep = np.ones(len(out), dtype=bool)
    keep[drop] = False
    out = out[keep]
    out.attrs = attrs
    tot = {k2: sum(v[k2] for v in per.values()) for k2 in next(iter(per.values()))} if per else {}
    out.attrs["af_history_corrections"] = {"per_league": per, "total": tot}
    log.info(f"[{tag}] af_history duplicates (owner 2026-11-04, provider date decides): "
             f"{len(drop)} proven copy(ies) removed, {tot['cells_filled']} empty cell(s) of the "
             f"kept copy filled from a provider copy, {tot['cells_made_missing']} conflicting "
             f"cell(s) made missing, {tot['unproven_cells_cleared']} unproven written-back "
             f"cell(s) cleared; {changed} listed row(s) changed since listing, {kept_absent} with the "
             f"kept row absent (left in)")
    return out


def rekey_sidecar(fill: pd.DataFrame, corr: pd.DataFrame | None = None,
                  tag: str = "af_sidecar") -> pd.DataFrame:
    """A sidecar keyed on af_history's key, its rows on a removed copy's key moved onto the kept
    copy's key (same provider fixture id only; see the module docstring). Counts in
    out.attrs['af_sidecar_rekey']."""
    corr = read_af_corrections() if corr is None else corr
    if fill is None or fill.empty or corr is None or corr.empty or not set(KEY) <= set(fill.columns):
        return fill
    rec = {}
    for r, k in zip(corr.to_dict("records"), _keys(corr)):
        fid = _num(r.get("provider_fixture_id"))
        rec[k] = ((k[0], str(r["kept_date"])[:10], str(r["kept_home_team"]), str(r["kept_away_team"])),
                  None if fid is None else int(fid))
    keys = _keys(fill)
    present = set(keys)
    fids = (pd.to_numeric(fill["fixture_id"], errors="coerce").tolist() if "fixture_id" in fill.columns
            else [None] * len(fill))
    moved, dropped, other = [], [], 0
    for i, k in enumerate(keys):
        if k not in rec:
            continue
        kk, fid = rec[k]
        f = fids[i]
        if fid is not None and f is not None and f == f and int(f) != fid:
            other += 1
            continue
        if kk in present:
            dropped.append(i)
        else:
            moved.append((i, kk))
            present.add(kk)
    if not moved and not dropped and not other:
        return fill
    attrs = dict(getattr(fill, "attrs", {}) or {})
    out = fill.copy()
    date_is_dt = pd.api.types.is_datetime64_any_dtype(out["date"])
    for i, kk in moved:
        out.iat[i, out.columns.get_loc("date")] = pd.Timestamp(kk[1]) if date_is_dt else kk[1]
        out.iat[i, out.columns.get_loc("home_team")] = kk[2]
        out.iat[i, out.columns.get_loc("away_team")] = kk[3]
    keep = np.ones(len(out), dtype=bool)
    keep[dropped] = False
    out = out[keep]
    out.attrs = attrs
    out.attrs["af_sidecar_rekey"] = {"moved_to_kept_key": len(moved), "dropped_kept_key_held": len(dropped),
                                     "left_other_fixture": other}
    log.info(f"[{tag}] af_history duplicates: {len(moved)} sidecar row(s) moved onto the kept copy's "
             f"key, {len(dropped)} dropped (kept key already answered), {other} left (another fixture)")
    return out


def follow_redated_fixtures(frame: pd.DataFrame, redated: dict | None = None,
                            tag: str = "af_redated") -> pd.DataFrame:
    """For the LOADER's joins only (fix15/dates): a row of `frame` (af_history, or a sidecar keyed on
    af_history's key) that carries the provider fixture id of a football-data row the loader re-dates
    onto the provider's date (src.history_corrections.redated_fixture_ids) and is dated ONE day off
    that provider date is offered on the provider date - the same proven fixture (its own fixture id),
    so the provider statistics it holds still reach the served row (measured 2026-11-07: 3 served
    matches, e.g. CF Montreal v Atlanta Utd 2024-10-23, fixture 1312300, whose only provider values
    are the --insidebox / --fullstats answers keyed on af_history's football-data-dated copy). One row
    per key: when the provider-date key is already held, the row is left where it is (counted).
    Counts in out.attrs['af_redated_follow']."""
    if redated is None:
        try:
            from src.history_corrections import redated_fixture_ids
            redated = redated_fixture_ids()
        except Exception as e:                                         # noqa: BLE001
            log.error(f"[{tag}] history date list unreadable ({type(e).__name__}: {e}) - nothing moved")
            redated = {}
    if (frame is None or frame.empty or not redated or "fixture_id" not in frame.columns
            or not set(KEY) <= set(frame.columns)):
        return frame
    keys = _keys(frame)
    present = set(keys)
    fids = pd.to_numeric(frame["fixture_id"], errors="coerce").tolist()
    moved, held = [], 0
    for i, (k, f) in enumerate(zip(keys, fids)):
        if f != f or int(f) not in redated:
            continue
        pdate = redated[int(f)]
        if k[1] == pdate or abs((pd.Timestamp(k[1]) - pd.Timestamp(pdate)).days) != 1:
            continue
        nk = (k[0], pdate, k[2], k[3])
        if nk in present:
            held += 1
            continue
        moved.append((i, pdate))
        present.add(nk)
    if not moved and not held:
        return frame
    attrs = dict(getattr(frame, "attrs", {}) or {})
    out = frame.copy()
    ci = out.columns.get_loc("date")
    is_dt = pd.api.types.is_datetime64_any_dtype(out["date"])
    for i, pdate in moved:
        out.iat[i, ci] = pd.Timestamp(pdate) if is_dt else pdate
    out.attrs = attrs
    out.attrs["af_redated_follow"] = {"moved_to_provider_date": len(moved), "left_key_held": held}
    log.info(f"[{tag}] provider date (fix15): {len(moved)} row(s) of a re-dated fixture (same fixture id) "
             f"offered on the provider's date, {held} left (key already held)")
    return out


def add_join_aliases(frame: pd.DataFrame, corr: pd.DataFrame | None = None,
                     tag: str = "af_join", redated: dict | None = None) -> pd.DataFrame:
    """`frame` (af_history after apply_af_history_corrections, or a sidecar after rekey_sidecar) plus,
    for every listed TWO-DATE pair whose kept row is in it and whose removed key is NOT, a copy of the
    kept row under the removed copy's key - so the loader's (league, date, clubs) join reaches the
    served row whichever of the two days it is dated by (module docstring). For the loader's joins
    ONLY: the collectors and the result readers never call it. Not for a pair whose served
    football-data row the loader re-dates onto the kept copy's (provider) date - `redated`, {fixture
    id: provider date} of src.history_corrections.redated_fixture_ids (fix15/dates) - nor for a
    same-day spelling duplicate. Counts in out.attrs['af_join_aliases']."""
    corr = read_af_corrections() if corr is None else corr
    if frame is None or frame.empty or corr is None or corr.empty or not set(KEY) <= set(frame.columns):
        return frame
    if redated is None:
        try:
            from src.history_corrections import redated_fixture_ids
            redated = redated_fixture_ids()
        except Exception as e:                                         # noqa: BLE001
            log.error(f"[{tag}] history date list unreadable ({type(e).__name__}: {e}) - every alias kept")
            redated = {}
    keys = _keys(frame)
    pos = {}
    for i, k in enumerate(keys):
        pos.setdefault(k, i)
    present = set(keys)
    picks, new_keys = [], []
    n_same_day, n_redated = 0, 0
    for r, k in zip(corr.to_dict("records"), _keys(corr)):
        kk = (k[0], str(r["kept_date"])[:10], str(r["kept_home_team"]), str(r["kept_away_team"]))
        if kk[1] == k[1]:
            n_same_day += 1          # a spelling duplicate (fix15): no other date to bridge
            continue
        fid = _num(r.get("provider_fixture_id"))
        if fid is not None and not isinstance(fid, str) and redated.get(int(fid)) == kk[1]:
            n_redated += 1           # the served row is re-dated onto the provider date: joins directly
            continue
        if kk not in pos or k in present:
            continue
        picks.append(pos[kk])
        new_keys.append(k)
        present.add(k)
    if not picks:
        return frame
    attrs = dict(getattr(frame, "attrs", {}) or {})
    add = frame.iloc[picks].copy()
    date_is_dt = pd.api.types.is_datetime64_any_dtype(frame["date"])
    add["date"] = [pd.Timestamp(k[1]) if date_is_dt else k[1] for k in new_keys]
    add["home_team"] = [k[2] for k in new_keys]
    add["away_team"] = [k[3] for k in new_keys]
    out = pd.concat([frame, add], ignore_index=True)
    out.attrs = attrs
    out.attrs["af_join_aliases"] = {"aliases_added": len(picks), "skipped_served_row_redated": n_redated,
                                    "skipped_same_day_spelling": n_same_day}
    log.info(f"[{tag}] af_history duplicates: the kept copy of {len(picks)} proven fixture(s) also "
             f"joinable under the removed copy's key (the served row's date)")
    return out
