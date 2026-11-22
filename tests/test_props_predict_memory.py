"""Props predict memory (2026-11-22): the serving feature build no longer copies the ~1M-row history twice and
build_features takes the one concatenated frame without a third copy. The props predict step had peaked at
~15 GB on a 16 GB hosted runner and was killed. Values must be identical to the old build."""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import player_model.feature_engineering as FE

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "player_history.parquet"


def _old_build(upcoming, history, n=None):
    """build_upcoming_features before 2026-11-22, verbatim (the reference)."""
    up = FE.upcoming_rows(upcoming)
    n = n or FE.config.ROLLING_N
    h = history.copy()
    h[FE.UPCOMING_COL] = False
    if "fixture_id" in h.columns and "fixture_id" in up.columns:
        h = h[~pd.to_numeric(h["fixture_id"], errors="coerce").isin(pd.to_numeric(up["fixture_id"], errors="coerce"))]
    frame = FE.build_features(pd.concat([h, up], ignore_index=True, sort=False), n)
    out = frame[frame[FE.UPCOMING_COL].astype(bool)].copy()
    out["position"] = out["position_prematch"]
    out["n_games"] = pd.to_numeric(out["n_prev_games"], errors="coerce").clip(upper=n)
    out["data_source"] = "match_level"
    return out.reset_index(drop=True)


def _slice():
    parts = sorted(HIST.glob("*.parquet")) if HIST.is_dir() else []
    if len(parts) < 3:
        pytest.skip("player_history.parquet parts not in this checkout")
    return pd.concat([pd.read_parquet(p) for p in parts[-3:]], ignore_index=True)


def test_serving_features_identical_to_the_old_build_and_history_untouched():
    hist = _slice()
    snap = hist.copy()
    last = hist.sort_values("date").groupby("player_id").tail(1).sort_values("player_id").head(300)
    fids = sorted(pd.to_numeric(hist["fixture_id"], errors="coerce").dropna().astype(int).unique())[-3:]
    up = [{"player_id": r["player_id"], "player_name": r.get("player_name", ""),
           # some upcoming rows reuse a fixture id the history holds (that history fixture is dropped)
           "fixture_id": int(fids[i % 3]) if i % 10 == 0 else 990000000 + i % 50, "date": "2026-11-24",
           "league": r.get("league", ""), "team": r.get("team", ""), "home_team": r.get("team", ""),
           "away_team": "X", "is_home": 1.0, "referee": None, "lineup_position": None}
          for i, r in enumerate(last.to_dict("records"))]
    new = FE.build_upcoming_features(up, hist)
    old = _old_build(up, hist)
    assert len(new) == len(old) > 0
    pd.testing.assert_frame_equal(new.reindex(columns=sorted(new.columns)), old.reindex(columns=sorted(old.columns)),
                                  check_dtype=False)
    pd.testing.assert_frame_equal(hist, snap)               # the caller's history is never modified


def test_no_extra_history_copies():
    src = inspect.getsource(FE.build_upcoming_features)
    assert "history.copy()" not in src and "copy=False" in src
    assert "copy: bool = True" in inspect.getsource(FE.build_features)   # every other caller still copies
