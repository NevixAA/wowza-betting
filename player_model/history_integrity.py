"""player_history integrity rules: the frozen window, point-in-time injury features, and the
push guard against one writer clobbering another (owner-approved 2026-10-26).

Three defects, measured 2026-10-25/26 on the real history and its git snapshots:

1. THE DAILY COLLECT REWROTE THE PAST. Every night `--mode collect` re-derived and overwrote
   feature values on rows up to ~120 days old (opp_strength_index, referee_*, n_prev_games,
   season_appearances ...) because build_features recomputes the whole fetched live-season
   window and the merge let every fresh non-null value win. Seven month parts changed a day.
   Raw match facts did NOT need that freedom: across three consecutive collects (10-17->10-18,
   10-18->10-19, 10-19->10-25) every change to a raw per-match column (minutes, goals, shots,
   passes, duels, cards, saves, rating, target_*) was on a row <= 14 days old. The only "raw"
   changes on older rows were `referee` flipping between two spellings of the same name
   ("P. Simani" <-> "Peiman Simani, Finland") and `team_corners` oscillating 0 <-> n on the
   same J-League fixtures as the 2026 and 2027 seasons are both collected -- source-order
   instability, not corrections. So after the window NOTHING changes; the mechanism for a raw
   fact exception exists (RAW_FACTS_MUTABLE_AFTER_FREEZE) but is empty on that evidence.

2. INJURY FEATURES CARRIED THE FUTURE. `enrich-sidelined-live` computed
   days_since_last_injury / chronic_injury_risk / return_from_injury_flag as of TODAY and wrote
   that one value onto EVERY row of every active player, back to 2022 -- 355,770 rows changed
   in one weekly refresh. A 2023 training row therefore knew about injuries that happened in
   2026. Even the full backfill (`enrich-sidelined`) counted injuries that STARTED after the
   match in chronic_injury_risk. Now every value is computed as of the row's own match date
   (injury_features_asof) and the refresh only writes rows inside INJURY_REFRESH_WINDOW_DAYS.

3. A PUSH RACE DROPPED ROWS. The injury refresh and the collect committed within a minute on
   10-24 and `git pull --rebase -X theirs` let the older copy of a part win, dropping 764 rows
   until the next collect re-added them. guard_cli() refuses to push a commit whose history is
   missing any (fixture_id, player_id) that origin already had.
"""
from __future__ import annotations

import datetime as dt
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["fixture_id", "player_id"]

# ── Rule 1: the frozen window ────────────────────────────────────────────────────────────────
# A row whose match date is more than FROZEN_AFTER_DAYS before today is IMMUTABLE to a collect.
# New rows are always added, whatever their age; rows inside the window may still be refreshed
# (late result corrections, data arriving after the match).
FROZEN_AFTER_DAYS = 14
# Columns a collect may still correct on a frozen row (raw facts only, never derived features).
# EMPTY on the evidence in the module docstring: no raw fact changed after 14 days.
RAW_FACTS_MUTABLE_AFTER_FREEZE: tuple[str, ...] = ()

# ── Rule 2: point-in-time injury features ─────────────────────────────────────────────────────
# Every column derived from the /sidelined history. The collect never produces them (they are
# owned by the injury refresh), so a collect must never overwrite them either.
INJURY_COLUMNS = ("chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag")
# The injury refresh writes only rows whose match is at most this many days old.
INJURY_REFRESH_WINDOW_DAYS = 60
# The pre-existing sentinel: "no sidelined spell ended before this match" (config.py: 365 =
# healthy). Kept so the feature means what the serving models were trained on.
NO_INJURY_DAYS = 365.0
RETURN_WINDOW_DAYS = 21
CHRONIC_LOOKBACK_DAYS = 365


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _as_day(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def _parse_day(s) -> dt.date | None:
    if s is None:
        return None
    try:
        s = str(s)[:10]
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        return None


# ── merge (collect) ───────────────────────────────────────────────────────────────────────────

def merge_history(existing: pd.DataFrame, fresh: pd.DataFrame, today=None,
                  frozen_after_days: int = FROZEN_AFTER_DAYS,
                  mutable_after_freeze: tuple[str, ...] = RAW_FACTS_MUTABLE_AFTER_FREEZE,
                  ) -> tuple[pd.DataFrame, dict]:
    """Union `fresh` (what a collect just fetched and derived) into `existing`.

      * a key only in `existing` is carried through untouched (a collect never removes a row);
      * a key only in `fresh` is ADDED, whatever its date;
      * a key in both whose stored match date is inside the window (>= today - frozen_after_days)
        is refreshed: the fresh value wins wherever it is non-null, the stored one fills gaps
        (combine_first semantics, as before);
      * a key in both whose stored match date is OLDER is FROZEN: the stored row is kept
        exactly, except columns named in `mutable_after_freeze`, which take a non-null fresh
        value. A row with an unparseable stored date counts as frozen.

    Injury columns in `fresh` are blanked first: a collect does not compute them (build_features
    only inserts the 365/0/0 defaults when the column is absent), so letting them through would
    overwrite real point-in-time values with an invented default -- which it did, nightly.
    """
    today = _utc_today() if today is None else _as_day(today)
    cutoff = today - pd.Timedelta(days=int(frozen_after_days))
    fresh = fresh.copy()
    for c in INJURY_COLUMNS:
        if c in fresh.columns:
            fresh[c] = np.nan

    a = existing.drop_duplicates(subset=KEY, keep="last").set_index(KEY)
    b = fresh.drop_duplicates(subset=KEY, keep="last").set_index(KEY)
    both = a.index.intersection(b.index)
    stats = {"existing": len(a), "fetched": len(b), "added": 0, "refreshed": 0, "frozen": 0,
             "frozen_raw_updates": 0}
    parts = [a.drop(index=both), b.drop(index=both)] if len(both) else [a, b]
    if len(both):
        d = pd.to_datetime(a.loc[both, "date"], errors="coerce") if "date" in a.columns \
            else pd.Series(pd.NaT, index=both)
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_convert("UTC").dt.tz_localize(None)
        live_mask = (d >= cutoff).to_numpy()
        live, frozen = both[live_mask], both[~live_mask]
        if len(live):
            parts.append(b.loc[live].combine_first(a.loc[live]))
        if len(frozen):
            keep = a.loc[frozen].copy()
            for c in mutable_after_freeze:
                if c in b.columns:
                    new = b.loc[frozen, c]
                    if c in keep.columns:
                        changed = new.notna() & ~((new == keep[c]) | (new.isna() & keep[c].isna()))
                        stats["frozen_raw_updates"] += int(changed.sum())
                        keep[c] = new.where(new.notna(), keep[c])
                    else:
                        keep[c] = new
            parts.append(keep)
        stats["refreshed"], stats["frozen"] = int(len(live)), int(len(frozen))
    # existing first, so the stored column order is kept; sort_index reproduces the
    # combine_first row order the old merge produced and downstream steps have been reading.
    merged = pd.concat([p for p in parts if len(p)]).sort_index().reset_index()
    stats["added"] = int(len(merged) - len(a))
    return merged, stats


# ── injury features as of the match date ─────────────────────────────────────────────────────

def injury_evidence(entries: list[dict], match_date) -> dict:
    """The sidelined events a row's injury features are allowed to see: only spells that
    STARTED before the match. Returns the last end strictly before the match and the starts in
    [match - 365d, match)."""
    m = _as_day(match_date).date()
    year_ago = m - dt.timedelta(days=CHRONIC_LOOKBACK_DAYS)
    last_end, starts = None, []
    for e in entries or []:
        s, f = _parse_day(e.get("start")), _parse_day(e.get("end"))
        if s is not None and s >= m:
            continue                      # began on/after the match: the future
        if f is not None and f < m and (last_end is None or f > last_end):
            last_end = f
        if s is not None and year_ago <= s < m:
            starts.append(s)
    return {"match_date": m, "last_end": last_end, "starts": sorted(starts)}


def injury_features_asof(entries: list[dict], match_date) -> dict:
    """chronic_injury_risk / days_since_last_injury / return_from_injury_flag for ONE row, as of
    its own match date. `entries` is the player's /sidelined list ({start, end} ISO dates); an
    empty list is a real answer ("no recorded spell"), and gives the 0 / 365 / 0 the models were
    trained on. A FAILED fetch must not reach here (it is not an empty list -- see
    data_fetcher.fetch_all_player_sidelined)."""
    ev = injury_evidence(entries, match_date)
    days = (float((ev["match_date"] - ev["last_end"]).days) if ev["last_end"] is not None
            else NO_INJURY_DAYS)
    return {"chronic_injury_risk": float(len(ev["starts"])),
            "days_since_last_injury": days,
            "return_from_injury_flag": float(0 < days <= RETURN_WINDOW_DAYS)}


def apply_injury_pit(df: pd.DataFrame, sidelined_map: dict, row_mask=None) -> int:
    """Write point-in-time injury features IN PLACE on the rows selected by `row_mask` whose
    player has an entry in `sidelined_map` (players whose fetch failed are absent and their
    rows are left exactly as they were). Returns the number of rows written."""
    for c in INJURY_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    mask = pd.Series(True, index=df.index) if row_mask is None else pd.Series(row_mask, index=df.index)
    pid = pd.to_numeric(df["player_id"], errors="coerce")
    mask &= pid.isin(list(sidelined_map.keys()))
    dates = pd.to_datetime(df["date"], errors="coerce")
    mask &= dates.notna()
    idx = df.index[mask.to_numpy()]
    if not len(idx):
        return 0
    out = {c: np.empty(len(idx)) for c in INJURY_COLUMNS}
    cache: dict = {}
    for i, (p, d) in enumerate(zip(pid.loc[idx].astype("int64").to_numpy(), dates.loc[idx])):
        k = (int(p), d.normalize())
        f = cache.get(k)
        if f is None:
            f = cache[k] = injury_features_asof(sidelined_map[int(p)], d)
        for c in INJURY_COLUMNS:
            out[c][i] = f[c]
    for c in INJURY_COLUMNS:
        df.loc[idx, c] = out[c]
    return int(len(idx))


def reconstruct_from_stamp(match_date, stamp_date, chronic, days) -> tuple[float | None, float | None]:
    """Recover the point-in-time value of a row from a value the OLD refresh computed as of a
    later `stamp_date` D (>= match date M), without the raw /sidelined history.

    The old refresh wrote days = D - L, where L is the last spell end before D (or the 365
    sentinel when there was none), and chronic = number of spells starting in [D - 365, D].

      days_since at M:  sentinel at D -> no spell ended before D, so none before M -> 365.
                        L < M        -> no end in (L, D) ⊇ (L, M), so the last end before M is
                                        L -> M - L.
                        L >= M       -> the last end before M is unknown -> None.
      chronic at M:     chronic(D) == 0 and (no L, or L < M - 365) -> 0. A spell starting in
                        [M-365, M) would either start inside [D-365, D] (counted, but 0 were)
                        or end before D, i.e. by L < M-365, before it started. The one
                        exception is a spell open for over a year across D, ignored.
                        Anything else -> None.

    Returns (days, chronic); None means "not recoverable". Exact for days, under the stated
    assumption for chronic -- verified against injury_features_asof in tests."""
    m, d = _as_day(match_date), _as_day(stamp_date)
    if d < m or days is None or not np.isfinite(days):
        return None, None
    sentinel = float(days) == NO_INJURY_DAYS
    last_end = None if sentinel else d - pd.Timedelta(days=float(days))
    if last_end is None:
        out_days = NO_INJURY_DAYS
    elif last_end < m:
        out_days = float((m - last_end).days)
    else:
        out_days = None
    out_chronic = None
    if chronic is not None and np.isfinite(chronic) and float(chronic) == 0.0:
        if last_end is None or last_end < m - pd.Timedelta(days=CHRONIC_LOOKBACK_DAYS):
            out_chronic = 0.0
    return out_days, out_chronic


# ── push guard ────────────────────────────────────────────────────────────────────────────────

def lost_keys(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Keys present in `before` and missing from `after`."""
    b = before[KEY].drop_duplicates()
    a = after[KEY].drop_duplicates()
    m = b.merge(a, on=KEY, how="left", indicator=True)
    return m.loc[m["_merge"] == "left_only", KEY].reset_index(drop=True)


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def guard_cli(ref: str = "origin/main", root: str = "player_history.parquet") -> int:
    """Run in a workflow AFTER `git pull --rebase` and BEFORE `git push`. Every month part
    that differs between `ref` (what origin already has) and the working tree is read from
    `ref`; if any of its (fixture_id, player_id) keys is absent from the working tree's history
    the push would delete rows another writer committed, so this exits 1 instead.

    With the shared concurrency group both writers start from a fresh checkout and this should
    never fire; it is the tripwire that makes the 2026-10-24 failure loud instead of silent."""
    changed = [p for p in _git("diff", "--name-only", ref, "--", root).split() if p.endswith(".parquet")]
    in_ref = set(_git("ls-tree", "-r", "--name-only", ref, "--", root).split())
    changed = [p for p in changed if p in in_ref]
    if not changed:
        print(f"[history-guard] no part differs from {ref}; nothing to check")
        return 0
    if not Path(root).exists():
        print(f"::error title=history guard::{root} missing but {ref} has it")
        return 1
    have = pd.read_parquet(root, columns=KEY)
    total_lost = 0
    for p in changed:
        before = pd.read_parquet(io.BytesIO(_git("show", f"{ref}:{p}", binary=True)), columns=KEY)
        lost = lost_keys(before, have)
        if len(lost):
            total_lost += len(lost)
            print(f"::error title=history guard::{p}: {len(lost)} row(s) on {ref} would be "
                  f"dropped by this push")
    if total_lost:
        print(f"[history-guard] REFUSING to push: {total_lost} committed row(s) would be lost")
        return 1
    print(f"[history-guard] ok: {len(changed)} changed part(s), no committed row lost")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "guard":
        sys.exit(guard_cli(*(sys.argv[2:3] or ["origin/main"])))
    print("usage: python -m player_model.history_integrity guard [ref]")
    sys.exit(2)
