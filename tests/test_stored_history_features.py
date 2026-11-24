"""Stored player_history carries every computable props feature, and a collect builds new rows in the
stored-history context (owner 2026-11-24, after Pro props learner run 45358726547).

    python -m pytest tests/test_stored_history_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import config as C
from player_model import pipeline as P

HISTORY = Path(__file__).resolve().parents[1] / "player_history.parquet"

#: PENDING THE OWNER-APPROVED REBUILD (draft PR feat/prev-season-rebuild, 2026-11-24). These nine
#: are computable (build_features gives ~505k of 893k stored rows a value) but the stored rows were
#: never rebuilt after props6 2/5 (43d4eebdf) added them. Ratchet: an entry that holds a value is a
#: test failure until it is removed here; nothing may be added without the owner.
PENDING_REBUILD = frozenset({
    "prev_season_goals_pg", "prev_season_assists_pg", "prev_season_shots_pg", "prev_season_sot_pg",
    "prev_season_cards_pg", "prev_season_minutes_pg", "prev_season_start_rate",
    "prev_season_appearances", "prev_season_shot_accuracy",
})


@pytest.fixture(scope="module")
def stored():
    if not HISTORY.exists():
        pytest.skip("no player_history.parquet in this checkout")
    import pyarrow.parquet as pq
    have = set(pq.read_schema(next(HISTORY.glob("*.parquet")) if HISTORY.is_dir() else HISTORY).names)
    cols = [c for c in C.PLAYER_FEATURE_COLS if c in have]
    return pd.read_parquet(HISTORY, columns=cols), have


def test_every_props_feature_is_a_stored_column(stored):
    _, have = stored
    assert not [c for c in C.PLAYER_FEATURE_COLS if c not in have]


def test_no_computable_props_feature_is_all_missing_in_the_stored_history(stored):
    df, _ = stored
    empty = sorted(c for c in df.columns if not df[c].notna().any())
    unregistered = [c for c in empty if c not in PENDING_REBUILD]
    assert not unregistered, (f"{len(unregistered)} props feature(s) empty on every stored row: "
                              f"{unregistered} - rebuild the stored column, never register it here")


def test_pending_rebuild_ratchet(stored):
    df, _ = stored
    filled = sorted(c for c in PENDING_REBUILD if c in df.columns and df[c].notna().any())
    assert not filled, f"rebuilt, remove from PENDING_REBUILD: {filled}"
    assert PENDING_REBUILD <= set(C.PLAYER_FEATURE_COLS)


# ── the collect builds in the stored-history context ─────────────────────────────────────────────

def _rows(player_id, season, dates, fixture0, team="A", opp="B"):
    out = []
    for i, d in enumerate(dates):
        out.append({"fixture_id": fixture0 + i, "player_id": player_id, "player_name": f"P{player_id}",
                    "date": pd.Timestamp(d), "season": season, "league": "Premier League",
                    "team": team, "opponent": opp, "home_team": team, "away_team": opp, "is_home": 1,
                    "position": "F", "minutes": 90, "started": 1, "goals": i % 2, "assists": 0,
                    "shots_total": 3, "shots_on_target": 1, "yellow_cards": 0, "red_cards": 0})
    return out


def test_collect_features_see_the_stored_previous_season(monkeypatch):
    """A batch holding only this season's rows gets prev_season_* from the STORED previous season,
    exactly as a build over history + batch (serving's path) gives them."""
    prev = _rows(7, "2024", pd.date_range("2024-09-01", periods=8, freq="7D"), 1000)
    cur = _rows(7, "2025", pd.date_range("2025-09-01", periods=3, freq="7D"), 2000)
    existing = P.build_features(prev + cur[:1])
    batch = cur[1:]
    alone = P.build_features(batch)
    ctx = P.build_collected_rows(batch, existing)
    # serving's path: build_features over the STORED rows + the new ones (build_upcoming_features)
    full = P.build_features(pd.concat([existing, pd.DataFrame(batch)], ignore_index=True))
    key = ["fixture_id", "player_id"]
    assert sorted(ctx["fixture_id"]) == [2001, 2002]
    if not alone.empty:
        assert alone["prev_season_goals_pg"].isna().all()           # the old batch-only build
    want = full[full["fixture_id"].isin([2001, 2002])].sort_values(key)["prev_season_goals_pg"].to_numpy()
    got = ctx.sort_values(key)["prev_season_goals_pg"].to_numpy()
    assert np.isfinite(got).all() and np.allclose(got, want)
    # the stored previous season holds 7 rows (the first appearance is never stored), 4 goals
    assert np.allclose(got, 4 / 7)


def test_collect_context_uses_fresh_raw_for_a_refetched_row():
    prev = _rows(7, "2024", pd.date_range("2024-09-01", periods=8, freq="7D"), 1000)
    cur = _rows(7, "2025", pd.date_range("2025-09-01", periods=3, freq="7D"), 2000)
    existing = P.build_features(prev + cur)
    refetch = [dict(cur[2], goals=5)]
    out = P.build_collected_rows(refetch, existing)
    assert out["fixture_id"].tolist() == [2002] and int(out["goals"].iloc[0]) == 5


def test_collect_without_stored_history_builds_the_batch():
    batch = _rows(7, "2025", pd.date_range("2025-09-01", periods=3, freq="7D"), 2000)
    a, b = P.build_collected_rows(batch, None), P.build_features(batch)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
