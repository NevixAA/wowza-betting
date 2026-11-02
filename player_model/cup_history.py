"""
cup_history.py - UEFA club cup player rows beside player_history (owner question 2026-11-02).

Props tips exist for Champions League / Europa League / Conference League fixtures (873 for 3-5 Nov
2026; 2,193 in the ledger since August) while player_history.parquet held ZERO cup rows, so a
player's form, minutes and rest days ignored every cup appearance.

SOURCE. scripts/collect_cup_player_stats.py (data_truth_collect.yml step 'cupstats') calls
API-Football /fixtures/players for every UEFA cup fixture of the collected fixture list and writes
the rows to output/collect/cup_player_rows.parquet - player_history's raw columns (the parser is
data_fetcher._parse_fixture_player, the one the league collect uses), league = the registry cup
name, plus team_id / opponent_id (API-Football ids) and fixture_status (FT / AET / PEN).

HOW THEY ARE USED.
  * Serving (pipeline.mode_predict): with_cup_rows() adds the collected rows the history does not
    already hold, so form / minutes / rest days and the opponent tables see cup matches at once.
  * Training data: pipeline --mode merge-cups appends them to player_history with features built
    in the FULL history context (one build_features pass over history + cup rows; only the cup
    rows are written; every stored row stays exactly as it is). A cup fixture decided after extra
    time (AET / PEN) is history for the features but is NOT appended as a training row: its stats
    include extra time and no source splits them at 90 minutes (props settle on 90 minutes).
"""
from __future__ import annotations

import pandas as pd

from . import config

CUP_ROWS_FILE = config.OUTPUT_DIR / "collect" / "cup_player_rows.parquet"
# Sidecar of the same collector (fix6/props6): per cup fixture the referee (/fixtures) and both
# teams' corners (/fixtures/statistics). Joined at read time; the player rows file is never rewritten.
CUP_META_FILE = config.OUTPUT_DIR / "collect" / "cup_fixture_meta.parquet"
EXTRA_TIME = {"AET", "PEN"}


def _read(p) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as e:                                              # noqa: BLE001
        print(f"[cups] {p.name} unreadable ({e}): not used")
        return pd.DataFrame()


def with_fixture_meta(rows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """team_corners (the row's own team, by API-Football team id) and referee from the sidecar where
    the row has none. A value the row carries wins; a fixture the sidecar lacks stays missing."""
    if rows.empty or meta.empty or "fixture_id" not in rows.columns:
        return rows
    m = meta.assign(fixture_id=pd.to_numeric(meta["fixture_id"], errors="coerce")).drop_duplicates(
        "fixture_id", keep="last").set_index("fixture_id")
    fid = pd.to_numeric(rows["fixture_id"], errors="coerce")
    out = rows.copy()
    tid = pd.to_numeric(out.get("team_id"), errors="coerce") if "team_id" in out.columns else None
    if tid is not None:
        home = fid.map(m["home_id"]).astype(float)
        away = fid.map(m["away_id"]).astype(float)
        corners = pd.Series(float("nan"), index=out.index)
        corners = corners.where(~(tid == home), fid.map(m["home_corners"]).astype(float))
        corners = corners.where(~(tid == away), fid.map(m["away_corners"]).astype(float))
        own = pd.to_numeric(out["team_corners"], errors="coerce") if "team_corners" in out.columns \
            else pd.Series(float("nan"), index=out.index)
        out["team_corners"] = own.where(own.notna(), corners)
    ref = fid.map(m["referee"])
    own_ref = out["referee"] if "referee" in out.columns else pd.Series(None, index=out.index, dtype=object)
    has = own_ref.notna() & (own_ref.astype(str).str.strip() != "")
    out["referee"] = own_ref.where(has, ref)
    return out


def cup_rows(path=None, meta_path=None) -> pd.DataFrame:
    """The collected cup rows (empty frame when none were collected yet), with the sidecar's corners
    and referee joined (with_fixture_meta)."""
    rows = _read(CUP_ROWS_FILE if path is None else path)
    if rows.empty:
        return rows
    return with_fixture_meta(rows, _read(CUP_META_FILE if meta_path is None else meta_path))


def new_cup_rows(history: pd.DataFrame, cups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collected cup rows whose (fixture_id, player_id) the history does not hold."""
    cups = cup_rows() if cups is None else cups
    if cups.empty:
        return cups
    if history is None or history.empty or not {"fixture_id", "player_id"} <= set(history.columns):
        return cups.copy()
    held = pd.MultiIndex.from_arrays([pd.to_numeric(history["fixture_id"], errors="coerce"),
                                      pd.to_numeric(history["player_id"], errors="coerce")])
    k = pd.MultiIndex.from_arrays([pd.to_numeric(cups["fixture_id"], errors="coerce"),
                                   pd.to_numeric(cups["player_id"], errors="coerce")])
    return cups[~k.isin(held)].copy()


def with_cup_rows(history: pd.DataFrame, cups: pd.DataFrame | None = None) -> pd.DataFrame:
    """history + the collected cup rows it does not hold (date-typed, sorted by date). Columns the
    cup rows do not carry (derived features) are NaN on them; serving reads raw columns only."""
    add = new_cup_rows(history, cups)
    if add.empty:
        return history
    add = add.assign(date=pd.to_datetime(add["date"], errors="coerce"))
    out = pd.concat([history, add], ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    print(f"[cups] +{len(add):,} UEFA cup player row(s) "
          f"({add['league'].value_counts().to_dict()}) beside player_history")
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def training_rows(cups: pd.DataFrame) -> pd.DataFrame:
    """The cup rows that may become TRAINING rows: 90-minute results only (see module docstring)."""
    if cups.empty or "fixture_status" not in cups.columns:
        return cups
    return cups[~cups["fixture_status"].isin(EXTRA_TIME)]
