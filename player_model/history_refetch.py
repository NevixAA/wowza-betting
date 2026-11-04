"""
history_refetch.py - the props pipeline's reader of the re-fetched player counts (owner 2026-11-04,
fix round 8, props item 3).

WHY. player_history.parquet and the UEFA cup rows were parsed by `int(x or 0)`: every per-player
count holds 0 where API-Football sent null, so a real 0 and a missing value cannot be told apart in
those files. scripts/collect_player_refetch.py re-reads /fixtures/players per fixture and stores the
counts under the owner's null rule (player_model.player_stat_nulls) BESIDE the history, in
output/collect/player_history_refetch.parquet, with the original values kept there as orig_<col>.

THE RULE HERE. For a row whose (fixture_id, player_id) was re-fetched with player and fixture proven,
the re-fetched value REPLACES the stored one - including a missing value replacing a suspect 0. This
is not a nan-only fill on purpose: the stored 0 is the suspect value, so filling only blanks would
keep every wrong 0. A row the collect parsed from a raw provider answer WITH the rule (count_source
'provider_rule', written by player_model.data_fetcher._parse_fixture_player since fix round 9,
2026-11-04) is verified as stored: it keeps its values and its 'provider_rule' label and is never
re-fetched. Every other row - written by the old `or 0` parser before the fix - keeps its stored
values and is marked 'zeros_unverified' in `count_source`, so the Phase 2 props retrain can exclude
it (owner 2026-11-04). Nothing is written back: player_history.parquet is never rewritten here.

NOT HERE. Stored derived columns (rolling per-game rates, target_* labels) were computed from the
old values; serving recomputes its rolling features from the raw rows, and the Phase 2 retrain must
rebuild features and labels from these raw counts. This reader changes raw counts only.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .player_stat_nulls import (HISTORY_COUNT_COLS, PARSED_WITH_RULE, REFETCHED, SOURCE_COL,
                                UNVERIFIED)

REFETCH_FILE = config.OUTPUT_DIR / "collect" / "player_history_refetch.parquet"
#: the replaced columns: every per-player count player_history stores + rating (not a count; an
#: unrated appearance was stored as 0 by the same bug)
REPLACED = list(HISTORY_COUNT_COLS) + ["rating"]
STATUS_COL = SOURCE_COL


def load_refetch(path=None) -> pd.DataFrame:
    p = REFETCH_FILE if path is None else path
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as e:                                              # noqa: BLE001
        print(f"[refetch] {p.name} unreadable ({e}): not used, every row stays {UNVERIFIED}")
        return pd.DataFrame()


def with_refetched(history: pd.DataFrame, refetch: pd.DataFrame | None = None) -> pd.DataFrame:
    """history with the re-fetched counts in place of the stored ones where (fixture_id, player_id)
    was re-fetched, and `count_source` on every row: 'refetched' | 'provider_rule' (parsed with the
    rule, kept as stored) | 'zeros_unverified'."""
    if history is None or history.empty:
        return history
    rf = load_refetch() if refetch is None else refetch
    out = history.copy()
    stored = out[STATUS_COL] if STATUS_COL in out.columns else pd.Series(None, index=out.index, dtype=object)
    ruled = stored.eq(PARSED_WITH_RULE)
    out[STATUS_COL] = UNVERIFIED
    out.loc[ruled, STATUS_COL] = PARSED_WITH_RULE
    if rf is None or rf.empty or not {"fixture_id", "player_id"} <= set(out.columns):
        return out
    rf = rf.assign(fixture_id=pd.to_numeric(rf["fixture_id"], errors="coerce"),
                   player_id=pd.to_numeric(rf["player_id"], errors="coerce"))
    rf = rf.dropna(subset=["fixture_id", "player_id"]).drop_duplicates(["fixture_id", "player_id"], keep="last")
    rf = rf.set_index(["fixture_id", "player_id"])
    key = pd.MultiIndex.from_arrays([pd.to_numeric(out["fixture_id"], errors="coerce"),
                                     pd.to_numeric(out["player_id"], errors="coerce")])
    # a row parsed with the rule is the newer reading of the same answer: a re-fetch never overrides it
    hit = key.isin(rf.index) & ~ruled.to_numpy()
    if not hit.any():
        return out
    sel = rf.reindex(key[hit])
    for c in REPLACED:
        if c not in rf.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns \
            else pd.Series(float("nan"), index=out.index)
        col.loc[hit] = pd.to_numeric(sel[c], errors="coerce").astype(float).to_numpy()
        out[c] = col
    out.loc[hit, STATUS_COL] = REFETCHED
    print(f"[refetch] {int(hit.sum()):,} of {len(out):,} row(s) carry re-fetched counts; "
          f"{int((out[STATUS_COL] == PARSED_WITH_RULE).sum()):,} parsed with the rule; "
          f"{int((out[STATUS_COL] == UNVERIFIED).sum()):,} marked {UNVERIFIED}")
    return out
