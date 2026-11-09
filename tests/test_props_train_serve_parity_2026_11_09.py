"""Train/serve identity for the props features (owner 2026-11-09): serving rows are built by
build_features itself, the position is pre-match evidence only, a sum over unrecorded counts is
missing, and a provider id carrying two players is excluded. Synthetic data only; the real-data
parity is scripts/props_train_serve_parity.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model import identity_conflicts as IC  # noqa: E402

FX0 = 9_100_000


def _history(n_rounds: int = 12) -> pd.DataFrame:
    """Two clubs, four players each, weekly fixtures, every count recorded, a referee per fixture."""
    rng = np.random.default_rng(11)
    rows = []
    d0 = pd.Timestamp("2025-08-02")
    for r in range(n_rounds):
        home, away = ("Arsenal", "Chelsea") if r % 2 == 0 else ("Chelsea", "Arsenal")
        for team in ("Arsenal", "Chelsea"):
            for k, pos in enumerate(("F", "M", "D", "G")):
                pid = (1 if team == "Arsenal" else 2) * 100 + k
                rows.append({"fixture_id": FX0 + r, "player_id": pid, "player_name": f"P{pid}",
                             "date": d0 + pd.Timedelta(days=7 * r), "league": "Premier League",
                             "season": "2025", "home_team": home, "away_team": away, "team": team,
                             "is_home": int(team == home), "referee": "M. Oliver" if r % 3 else "A. Taylor",
                             "position": pos if not (pid == 100 and r == 6) else "M",
                             "minutes": 90, "started": 1,
                             "goals": int(rng.integers(0, 2)), "assists": int(rng.integers(0, 2)),
                             "shots_total": int(rng.integers(0, 5)), "shots_on_target": int(rng.integers(0, 3)),
                             "yellow_cards": int(rng.integers(0, 2)), "key_passes": int(rng.integers(0, 3)),
                             "rating": float(rng.uniform(6, 8)), "fouls_committed": int(rng.integers(0, 3)),
                             "fouls_drawn": int(rng.integers(0, 3)), "duels_total": int(rng.integers(2, 9)),
                             "duels_won": int(rng.integers(0, 2)), "team_corners": int(rng.integers(2, 9)),
                             "red_cards": 0, "passes_total": int(rng.integers(10, 50)), "offsides": 0,
                             "tackles_total": int(rng.integers(0, 4)), "interceptions": int(rng.integers(0, 3)),
                             "dribbles_attempted": 2, "dribbles_success": int(rng.integers(0, 3)),
                             "dribbles_past": 0, "goals_conceded": int(rng.integers(0, 3)),
                             "saves": int(rng.integers(0, 4)), "penalty_scored": int(pid == 100 and r == 4),
                             "penalty_missed": 0, "penalty_won": 0})
    return pd.DataFrame(rows)


def _serving_rows(raw: pd.DataFrame, fid: int) -> list[dict]:
    """What predict passes for a fixture: identity only, no result."""
    fr = raw[raw["fixture_id"] == fid]
    return [{k: r[k] for k in ("player_id", "player_name", "fixture_id", "date", "league", "team",
                               "home_team", "away_team", "is_home", "referee")} for _, r in fr.iterrows()]


@pytest.fixture(autouse=True)
def _no_collected_lineups(monkeypatch):
    monkeypatch.setattr(FE, "collected_lineup_positions",
                        lambda: pd.DataFrame(columns=["fixture_id", "player_id", "pos"]))


def test_serving_rows_equal_the_training_rows_exactly():
    raw = _history()
    last = FX0 + 11
    full = FE.build_features(raw)
    hist = raw[raw["fixture_id"] != last]
    sv = FE.build_upcoming_features(_serving_rows(raw, last), hist)
    tr = full[full["fixture_id"] == last].sort_values("player_id").reset_index(drop=True)
    sv = sv.sort_values("player_id").reset_index(drop=True)
    assert len(sv) == len(tr) == 8
    cols = [c for c in config.PLAYER_FEATURE_COLS if c in tr.columns]
    assert len(cols) > 80
    bad = []
    for c in cols:
        a, b = pd.to_numeric(sv[c], errors="coerce"), pd.to_numeric(tr[c], errors="coerce")
        if not ((a.isna() & b.isna()) | (a == b)).all():
            bad.append(c)
    assert bad == []                                       # identical, not within tolerance
    assert (sv["position_source"] == tr["position_source"]).all()


def test_position_is_pre_match_evidence_only():
    raw = _history()
    full = FE.build_features(raw)
    r = full[(full["player_id"] == 100)].sort_values("date").reset_index(drop=True)
    # round 6: the match-stats position is M, but pre-match the latest earlier position was F
    six = r[r["fixture_id"] == FX0 + 6].iloc[0]
    assert six["position_prematch"] == "F" and six["position_source"] == FE.POSITION_FROM_LATEST
    assert six["pos_forward"] == 1.0 and six["pos_midfielder"] == 0.0
    # round 7: the latest earlier position is now M (round 6)
    seven = r[r["fixture_id"] == FX0 + 7].iloc[0]
    assert seven["position_prematch"] == "M"


def test_announced_lineup_wins_and_unknown_is_missing(monkeypatch):
    raw = _history()
    monkeypatch.setattr(FE, "collected_lineup_positions", lambda: pd.DataFrame(
        {"fixture_id": [float(FX0 + 5)], "player_id": [101.0], "pos": ["F"]}))
    full = FE.build_features(raw)
    row = full[(full["player_id"] == 101) & (full["fixture_id"] == FX0 + 5)].iloc[0]
    assert row["position_prematch"] == "F" and row["position_source"] == FE.POSITION_FROM_LINEUP
    raw2 = raw.copy()
    raw2.loc[raw2["player_id"] == 102, "position"] = None   # never a known position
    f2 = FE.build_features(raw2)
    u = f2[f2["player_id"] == 102]
    assert (u["position_source"] == FE.POSITION_NONE).all()
    assert u[["pos_forward", "pos_midfielder", "pos_defender"]].isna().all().all()   # never 0/0/0


def test_live_lineup_position_is_used_for_the_serving_row():
    raw = _history()
    last = FX0 + 11
    rows = _serving_rows(raw, last)
    for r in rows:
        if r["player_id"] == 103:
            r["lineup_position"] = "D"                     # a keeper announced as a defender
    sv = FE.build_upcoming_features(rows, raw[raw["fixture_id"] != last])
    x = sv[sv["player_id"] == 103].iloc[0]
    assert x["position"] == "D" and x["position_source"] == FE.POSITION_FROM_LINEUP and x["pos_defender"] == 1.0


def test_a_sum_over_unrecorded_counts_is_missing_not_zero():
    raw = _history(4)
    raw.loc[raw["fixture_id"] == FX0 + 1, ["goals", "shots_on_target"]] = np.nan
    d = raw.copy()
    d["_tkey"], d["_okey"] = FE._team_keys(d)
    t = FE.team_match_table(d)
    one = t[t["fixture_id"] == FX0 + 1]
    assert one["sot_scored"].isna().all()                  # was 0.0


def test_penalty_prior_does_not_read_the_rows_own_count():
    raw = _history()
    a = FE.build_features(raw)
    raw2 = raw.copy()
    raw2.loc[(raw2["player_id"] == 100) & (raw2["fixture_id"] == FX0 + 8), ["penalty_scored", "penalty_missed"]] = np.nan
    b = FE.build_features(raw2)
    k = (a["player_id"] == 100) & (a["fixture_id"] == FX0 + 8)
    x, y = a.loc[k, "penalty_conversion_rate"].iloc[0], b.loc[k, "penalty_conversion_rate"].iloc[0]
    assert (np.isnan(x) and np.isnan(y)) or x == y


def test_same_day_rows_have_one_order():
    raw = _history(4)
    dup = raw[raw["fixture_id"] == FX0 + 3].copy()
    dup["fixture_id"] = FX0 + 50
    both = pd.concat([raw, dup], ignore_index=True)
    a = FE.build_features(both)
    b = FE.build_features(both.sample(frac=1.0, random_state=3))
    cols = ["fixture_id", "player_id", "goals_pg", "sot_pg", "n_prev_games"]
    pd.testing.assert_frame_equal(a[cols].sort_values(cols[:2]).reset_index(drop=True),
                                  b[cols].sort_values(cols[:2]).reset_index(drop=True))


def test_identity_conflict_needs_both_conditions(tmp_path):
    h = pd.DataFrame([
        {"player_id": 7, "fixture_id": 1, "date": "2026-09-14", "team": "FCSB", "player_name": "J. Paulo"},
        {"player_id": 7, "fixture_id": 2, "date": "2026-09-14", "team": "Otelul", "player_name": "J. Paulino"},
        {"player_id": 8, "fixture_id": 3, "date": "2025-11-19", "team": "Uruguay", "player_name": "F. Torres"},
        {"player_id": 8, "fixture_id": 4, "date": "2025-11-19", "team": "Palmeiras", "player_name": "F. Torres"},
        {"player_id": 9, "fixture_id": 5, "date": "2026-01-01", "team": "A", "player_name": "X"}])
    tr = pd.DataFrame([{"player_id": 7, "player_name": "João Paulo"}, {"player_id": 7, "player_name": "João Paulino"},
                       {"player_id": 8, "player_name": "F. Torres"}])
    found = IC.find(h, tr)
    assert found["player_id"].tolist() == [7]              # 8: one name -> one real player
    p = tmp_path / "c.csv"
    found.to_csv(p, index=False)
    kept = IC.apply(h, p)
    assert set(kept["player_id"]) == {8, 9}


def test_the_committed_correction_holds_the_provider_conflict():
    assert IC.conflicted_ids() == {142016}


def test_serving_wiring():
    pred = (ROOT / "player_model" / "predict.py").read_text(encoding="utf-8")
    assert "_feats = build_upcoming_features(_all_rows, history_df)" in pred      # ONE build per run
    assert '"referee": _ref_name' in pred and '"lineup_position": lineup_pos.get(' in pred
    pipe = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    assert "fixture_referees=_build_fixture_referees()" in pipe
    assert "history = source_frame()" in pipe
    harness = (ROOT / "player_model" / "props_harness.py").read_text(encoding="utf-8")
    assert "_drop_conflicted(raw)" in harness
