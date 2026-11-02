"""props_truth (round-2 data-truth findings, 2026-11-02): each test fails on data-truth-100 40d7dda9."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from player_model import config  # noqa: E402


# ── V3: placeholder profile features leave the props training feature set ─────────────────────────
def test_profile_placeholder_features_withheld_from_props_training():
    fake = {"age", "age_peak_delta", "height_cm", "height_aerial_interaction",
            "height_diff_vs_opp_def", "height_advantage_score", "opp_def_mean_height"}
    assert not fake & set(config.PLAYER_FEATURE_COLS)


def test_props_contract_covers_feature_list():
    from player_model import props_contract as PC
    c = PC.load_contract()
    assert set(c["features"]) == set(config.PLAYER_FEATURE_COLS)


# ── World Cup sentinel: the cap is never written as the probability ───────────────────────────────
def test_capped_prob_refuses_instead_of_writing_the_cap():
    from player_model.predict import capped_prob
    assert capped_prob(0.71, 0.5) is None           # was min(0.71, 0.5) = 0.5, a sentinel
    assert capped_prob(0.31, 0.5) == 0.31
    assert capped_prob(0.97, 1.0) is None           # absolute 0.95 ceiling likewise


# ── Ghost tips: current-club rule at tip time and at grading time ─────────────────────────────────
def test_club_conflict_tip_time():
    from player_model.club_rule import club_conflict
    d = pd.Timestamp("2026-08-23")
    assert club_conflict("Atletico Madrid", "Inter Miami", pd.Timestamp("2026-08-16"), d) is True
    assert club_conflict("Inter Miami", "Inter Miami CF", pd.Timestamp("2026-08-16"), d) is False
    assert club_conflict("Atletico Madrid", "Inter Miami", pd.Timestamp("2026-05-01"), d) is False
    assert club_conflict("Atletico Madrid", None, None, d) is False


def test_clv_settle_never_grades_off_another_clubs_fixture(tmp_path, monkeypatch):
    from player_model import clv_tracker
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"bet_id": "p|2026-08-23|Rodrigo De Paul|sot", "odds_bet": 2.0,
                   "result": "", "pnl": ""}]).to_csv(tmp_path / "clv_records.csv", index=False)
    pd.DataFrame([{"player_name": "Rodrigo De Paul", "market": "sot", "match_date": "2026-08-23",
                   "player_id": "2472", "fixture_id": "", "home_team": "Atletico Madrid",
                   "away_team": "Villarreal"}]).to_csv(tmp_path / "player_ledger.csv", index=False)
    hist = pd.DataFrame([{"player_name": "Rodrigo De Paul", "date": pd.Timestamp("2026-08-23"),
                          "goals": 0, "assists": 0, "shots_on_target": 1, "yellow_cards": 0,
                          "red_cards": 0, "home_team": "Inter Miami", "away_team": "Toronto FC",
                          "player_id": 2472, "fixture_id": 1490440}])
    pq = tmp_path / "ph.parquet"
    hist.to_parquet(pq)
    assert clv_tracker.settle_results(parquet_path=pq) == 0
    out = pd.read_csv(tmp_path / "clv_records.csv")
    assert out["result"].isna().all()
    # the same row of the tipped fixture still grades
    hist.assign(home_team="Atletico Madrid", away_team="Villarreal").to_parquet(pq)
    assert clv_tracker.settle_results(parquet_path=pq) == 1


# ── player_history producers ──────────────────────────────────────────────────────────────────────
def test_stable_player_id_rejects_zero():
    from player_model.data_fetcher import _stable_player_id
    assert _stable_player_id(0, "Felipe Sanchez") is None
    assert _stable_player_id(2472, "Rodrigo De Paul") == 2472


def test_fixture_side_by_team_id_then_identity():
    from player_model.data_fetcher import fixture_side
    fx = {"teams": {"home": {"id": 2287, "name": "CF Pachuca"}, "away": {"id": 2291, "name": "Puebla"}}}
    assert fixture_side({"team": {"id": 2287, "name": "Pachuca"}}, fx, "CF Pachuca", "Puebla") is True
    assert fixture_side({"team": {"id": 2291, "name": "Puebla"}}, fx, "CF Pachuca", "Puebla") is False
    # no ids: club identity, never `==` on the two endpoints' spellings
    assert fixture_side({"team": {"name": "Pachuca"}}, {}, "CF Pachuca", "Puebla") is True
    assert fixture_side({"team": {"name": "Somewhere Else"}}, {}, "CF Pachuca", "Puebla") is None


def test_minutes_above_match_maximum_read_as_missing():
    from player_model.feature_engineering import minutes_ok
    out = minutes_ok(pd.Series([90, 121, 174]))
    assert out.iloc[0] == 90 and out.iloc[1] == 121 and np.isnan(out.iloc[2])


def test_prev_games_full_counts_whole_stored_history():
    from player_model.history_integrity import prev_games_full
    df = pd.DataFrame({"player_id": [7, 7, 7, 7, 0],
                       "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-02-01", "2024-02-08",
                                               "2024-01-01"]),
                       "fixture_id": [1, 2, 3, 4, 5],
                       "n_prev_games": [1, 2, 1, 2, 3]})     # rows 3-4 truncated by a later batch
    assert prev_games_full(df).tolist()[:4] == [1, 2, 3, 4]
    assert np.isnan(prev_games_full(df).iloc[4])


def test_merge_history_recounts_fresh_rows_not_frozen_ones():
    from player_model.history_integrity import merge_history
    ex = pd.DataFrame({"fixture_id": [1, 2], "player_id": [7, 7],
                       "date": pd.to_datetime(["2026-01-01", "2026-01-08"]), "n_prev_games": [1, 2]})
    fresh = pd.DataFrame({"fixture_id": [3], "player_id": [7], "date": pd.to_datetime(["2026-10-30"]),
                          "n_prev_games": [1]})                 # batch-local count
    m, st = merge_history(ex, fresh, today="2026-11-02")
    assert m.set_index("fixture_id").loc[3, "n_prev_games"] == 3
    assert m.set_index("fixture_id").loc[[1, 2], "n_prev_games"].tolist() == [1, 2]


# ── odds history: a book's last price is real ─────────────────────────────────────────────────────
def test_odds_history_keeps_each_book_and_returns_to_a_price(tmp_path, monkeypatch):
    from player_model import odds_fetcher as OF
    import src.rolling_store as RS
    f = tmp_path / "player_prop_odds_history.csv"
    monkeypatch.setattr(OF, "_ODDS_HISTORY_FILE", f)
    monkeypatch.setattr(RS, "roll", lambda *a, **k: None)
    base = {"match_date": "2099-01-01", "league": "La Liga", "match": "A vs B", "player": "X",
            "market": "sot", "source": "api_football"}

    class _Clock:
        n = 0
    import datetime as _dt
    real = _dt.datetime

    def _run(rows):
        _Clock.n += 1

        class _D(real):
            @classmethod
            def now(cls, tz=None):
                return real(2026, 11, 2, 10, _Clock.n, tzinfo=tz)
        monkeypatch.setattr(_dt, "datetime", _D)
        try:
            OF._append_odds_history(rows)
        finally:
            monkeypatch.setattr(_dt, "datetime", real)

    _run([{**base, "odds": 2.0, "bookmaker": "Bet365"}, {**base, "odds": 2.0, "bookmaker": "Unibet"}])
    _run([{**base, "odds": 2.2, "bookmaker": "Bet365"}, {**base, "odds": 2.0, "bookmaker": "Unibet"}])
    _run([{**base, "odds": 2.0, "bookmaker": "Bet365"}, {**base, "odds": 2.0, "bookmaker": "Unibet"}])
    h = pd.read_csv(f)
    b365 = h[h["bookmaker"] == "Bet365"]
    assert b365["odds"].tolist() == [2.0, 2.2, 2.0]           # the return to 2.0 is kept
    assert h[h["bookmaker"] == "Unibet"]["odds"].tolist() == [2.0]   # unchanged quotes collapse


# ── rebuild script (no network) ───────────────────────────────────────────────────────────────────
def test_rebuild_player_history_derived_columns():
    import rebuild_player_history as RB
    df = pd.DataFrame({
        "fixture_id": [1, 2, 3, 4], "player_id": [7, 7, 0, 9], "player_name": ["A", "A", "B", "C"],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01"]),
        "league": ["La Liga 2"] * 3 + ["Mexico Liga MX"],
        "team": ["Eibar", "Eibar", "Eibar", "Pachuca"], "home_team": ["Eibar", "Eibar", "Eibar", "CF Pachuca"],
        "away_team": ["Leganes", "Leganes", "Leganes", "Puebla"], "is_home": [1, 1, 1, 0],
        "opponent": ["Leganes", "Leganes", "Leganes", "CF Pachuca"], "minutes": [90, 90, 90, 174],
        "n_prev_games": [1, 1, 1, 1], "aerial_won_rate": [0.5] * 4, "opp_def_mean_height": [180.0] * 4,
        "age": [25.0] * 4, "height_cm": [180.0] * 4, "weight_kg": [75.0] * 4,
        "height_diff_vs_opp_def": [0.0] * 4, "height_advantage_score": [0.0] * 4,
        "height_aerial_interaction": [0.25] * 4, "age_peak_delta": [0.0] * 4,
        "league_tier": [1, 1, 1, 3], "league_quality": [1.0] * 4, "player_career_avg_quality": [1.0] * 4,
        "context_quality_discount": [1.0] * 4, "quality_mismatch_goals": [1.0] * 4,
        "quality_mismatch_sot": [1.0] * 4})
    out, counts, chg = RB.rebuild(df)
    assert out[["age", "height_cm", "weight_kg", "age_peak_delta", "height_aerial_interaction"]].isna().all().all()
    assert out["league_tier"].isna().all()
    assert out.loc[3, "is_home"] == 1 and out.loc[3, "opponent"] == "Puebla"
    assert out["n_prev_games"].tolist()[:2] == [1, 2] and np.isnan(out.loc[2, "n_prev_games"])
    assert df.loc[3, "minutes"] == 174 and out.loc[3, "minutes"] == 174        # raw untouched
    assert counts["own_club_as_opponent_before"] == 1 and counts["own_club_as_opponent_after"] == 0
    assert set(chg["column"]) == {"is_home", "opponent", "n_prev_games"}
