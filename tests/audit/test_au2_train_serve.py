"""AU2 forensic audit - targeted train/serve regression tests (read-only; synthetic data only).

Tests marked xfail(strict=True) assert the CORRECT behaviour and currently FAIL on the audited
commit fb0e9697e because of a defect recorded in reports/02_ML_TRAIN_SERVE_FORENSICS.md. When the
defect is fixed the test starts passing, strict xfail turns that into a failure, and the marker must
be removed - so a fix cannot land silently and a regression cannot return silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _history(n_rounds: int = 12) -> pd.DataFrame:
    """Two-league synthetic history, weekly rounds, deterministic scores, HT scores present."""
    rng = np.random.default_rng(7)
    rows = []
    for lg, teams in (("Championship", ["A1", "B1", "C1", "D1"]), ("League One", ["A2", "B2", "C2", "D2"])):
        for k in range(n_rounds):
            d = pd.Timestamp("2026-08-01") + pd.Timedelta(days=7 * k + (0 if lg == "Championship" else 1))
            pairs = [(teams[(k + i) % 4], teams[(k + i + 2) % 4]) for i in range(2)]
            for h, a in pairs:
                hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 3))
                rows.append({"date": d, "league": lg, "season": "2026/27", "home_team": h, "away_team": a,
                             "home_goals": hg, "away_goals": ag, "ht_home_goals": min(hg, 1),
                             "ht_away_goals": min(ag, 1), "home_shots": 10, "away_shots": 8,
                             "home_sot": 4, "away_sot": 3, "home_corners": 5, "away_corners": 4,
                             "home_fouls": 11, "away_fouls": 12, "odds_over25": 1.9, "odds_under25": 1.95})
    h = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    h["total_goals"] = h["home_goals"] + h["away_goals"]
    h["over25"] = (h["total_goals"] > 2.5).astype(float)
    return h


def _board() -> pd.DataFrame:
    """An upcoming board deliberately NOT sorted by (date, league, home, away) - production boards
    are grouped by league then date (the audited 2026-11-01 board is not date-sorted)."""
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-11-03"), "league": "League One", "home_team": "A2", "away_team": "B2"},
        {"date": pd.Timestamp("2026-11-01"), "league": "Championship", "home_team": "C1", "away_team": "D1"},
        {"date": pd.Timestamp("2026-11-02"), "league": "Championship", "home_team": "A1", "away_team": "B1"},
        {"date": pd.Timestamp("2026-11-01"), "league": "League One", "home_team": "C2", "away_team": "D2"},
    ]).assign(odds_over25=1.9, odds_under25=1.95, best_over25=1.95, best_under25=2.0)


# ── AU2-B01: HT features assigned by position onto an unsorted board ─────────────────────────

# AU2-B01 FIXED on data-truth-100 (attach_ht_features keeps board order); xfail removed 2026-11-02.
def test_ht_features_follow_board_order():
    from src.ht_features import attach_ht_features
    hist, board = _history(), _board()
    ht = attach_ht_features(hist, board)
    # what src/predict.py does: feat[c] = _ht[c].to_numpy()  -> requires identical row order
    assert list(ht["home_team"]) == list(board["home_team"])


def test_ht_features_key_aligned_are_well_defined():
    """Control: aligned by fixture key, every board row receives its own team's HT features."""
    from src.ht_features import attach_ht_features
    hist, board = _history(), _board()
    ht = attach_ht_features(hist, board).set_index(["league", "home_team", "away_team"])
    for r in board.itertuples():
        assert (r.league, r.home_team, r.away_team) in ht.index


# ── AU2-B02: legacy serving rest days positional tail ────────────────────────────────────────

def _rest_by_key(hist: pd.DataFrame, board: pd.DataFrame) -> list:
    last = {}
    for r in hist.sort_values("date").itertuples():
        last[r.home_team] = r.date
        last[r.away_team] = r.date
    return [(r.date - last[r.home_team]).days for r in board.itertuples()]


# AU2-B02 FIXED on data-truth-100 (_add_rest_days keeps input order); xfail removed 2026-11-02.
def test_legacy_serving_rest_days_belong_to_their_fixture():
    from src.feature_engineering import build_upcoming_features
    hist, board = _history(), _board()
    f = build_upcoming_features(board, hist)
    assert list(f["home_rest_days"].astype(float)) == [float(x) for x in _rest_by_key(hist, board)]


# ── Control: the repaired builder is identical in training and serving (canary) ─────────────

def test_repaired_builder_train_equals_serve():
    from src.features_repaired import (build_features_repaired, build_upcoming_features_repaired,
                                       STANDARD_SERVING_FEATURES)
    hist = _history()
    last = hist["date"].max()
    target = hist[hist["date"] == last]
    tr = build_features_repaired(hist, price_mode="none", team_scope="league")
    tr = tr[tr["date"] == last].sort_values(["league", "home_team"]).reset_index(drop=True)
    sv = build_upcoming_features_repaired(target[["date", "league", "home_team", "away_team"]], hist,
                                          price_mode="none", cutoff=last, team_scope="league")
    sv = sv.sort_values(["league", "home_team"]).reset_index(drop=True)
    cols = [c for c in STANDARD_SERVING_FEATURES if c != "bookmaker_overround"]
    pd.testing.assert_frame_equal(tr[cols].astype(float), sv[cols].astype(float), check_exact=False, atol=1e-9)


def test_repaired_builder_rest_days_are_per_fixture_on_unsorted_board():
    from src.features_repaired import build_upcoming_features_repaired
    hist, board = _history(), _board()
    f = build_upcoming_features_repaired(board, hist, price_mode="none", team_scope="league")
    assert list(f["home_team"]) == list(board["home_team"])           # input order preserved
    assert list(f["home_rest_days"].astype(float)) == [float(x) for x in _rest_by_key(hist, board)]


# ── AU2-B03: full-time targets are not masked when goals are missing ─────────────────────────

# AU2-B03 FIXED on data-truth-100 (src/data_loader.py:1524 masks with .where(_ftn)); xfail removed 2026-11-02.
def test_fulltime_targets_stay_missing_without_goals():
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    # the half-time targets use .where(_htn); the full-time ones must too
    assert 'out["over25"] = (out["total_goals"] > 2.5).astype(float).where' in src.replace("      ", " ")


def test_nan_goal_comparison_is_zero_not_nan():
    """Documents the mechanism the xfail above depends on (pure pandas semantics)."""
    s = pd.Series([np.nan, 3.0])
    assert list((s > 2.5).astype(float)) == [0.0, 1.0]


# ── AU2-B05: props serving rest_days is the PREVIOUS match's rest ───────────────────────────

# AU2-B05 FIXED on data-truth-100: without a fixture_date the serving rest_days is NaN (missing stays
# missing), with fixture_date 2026-10-25 it is 7.0; xfail removed 2026-11-02.
def test_props_serving_rest_days_is_days_since_last_match():
    from player_model.feature_engineering import build_upcoming_features
    base = {"player_name": "X", "team": "A1", "opponent": "B1", "league": "Championship", "position": "F",
            "minutes": 90, "started": 1, "goals": 0, "assists": 0, "shots_total": 2, "shots_on_target": 1,
            "yellow_cards": 0, "key_passes": 1, "rating": 7.0, "is_home": 1}
    hist = pd.DataFrame([
        # fixture_id: the team table is keyed (fixture_id, team) since the league-keyed rewrite
        dict(base, player_id=1, fixture_id=101, date=pd.Timestamp("2026-10-01"), rest_days=6.0),
        dict(base, player_id=1, fixture_id=102, date=pd.Timestamp("2026-10-15"), rest_days=14.0),
        dict(base, player_id=1, fixture_id=103, date=pd.Timestamp("2026-10-18"), rest_days=3.0),
    ])
    # the serving row of the upcoming fixture on 2026-10-25 (identity only, no result)
    up = {"player_id": 1, "player_name": "X", "fixture_id": 104, "date": "2026-10-25",
          "league": "Championship", "team": "A1", "home_team": "A1", "away_team": "B1", "is_home": 1}
    f = build_upcoming_features([up], hist)
    # serving = training's definition: 7 days since the last match (2026-10-18), not a copied 3.0
    assert float(f["rest_days"].iloc[0]) == 7.0
