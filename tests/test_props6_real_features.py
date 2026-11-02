"""fix6/props6 (owner rules 2026-11-02: no fake data, no proxies). Each test fails on
origin/data-truth-100 769db5be.

  1. set-piece proxies purged (train AND serve), typed clip bounds / constants gone from measured
     features, typed-weight formulas withheld from the training feature list;
  2. prev_season_* and shot_accuracy_l10 (real values, explicit minimum samples, one definition);
  3. rest_days on the COMPLETE match history (league + collected cup rows), train and serve;
  4. the cup collector stores team corners (/fixtures/statistics) and the referee (/fixtures);
  5. merge-cups writes only the stored history's columns and carries the collected corners/referee.

Everything offline (synthetic frames, fake HTTP session).

    python -m pytest -p no:cacheprovider tests/test_props6_real_features.py -q
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from player_model import config as C  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model.setpiece import features as SPF  # noqa: E402
from src.strength import lookup  # noqa: E402

IDS = {"Arsenal": 42, "Chelsea": 49, "Everton": 45, "Fulham": 36}
PURGED = ("sp_taker_score", "sp_receiver_score", "set_piece_threat_score", "set_piece_corner_matchup",
          "defender_sot_role_index", "defender_sp_edge", "defender_sot_edge",
          "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge", "opp_sp_goals_conceded_pg",
          "career_sp_assist_rate", "sp_goals_pg", "aerial_height_sp_composite",
          "sp_goal_probability_composite", "sp_assist_pg", "career_sp_goals_rate", "sp_goals_share")


def _player_rows(fid, date, home, away, season, shots=2, sot=1, minutes=90, league="Premier League"):
    rows = []
    for team, ih in ((home, 1), (away, 0)):
        for j, pos in enumerate(("G", "D", "M", "F")):
            rows.append({
                "fixture_id": fid, "player_id": IDS[team] * 100 + j, "player_name": f"{team}-{pos}",
                "date": date, "league": league, "home_team": home, "away_team": away, "team": team,
                "is_home": ih, "referee": "Ref A", "season": season, "position": pos,
                "minutes": minutes, "started": 1, "goals": 1 if (pos == "F" and ih) else 0,
                "assists": 0, "shots_total": shots, "shots_on_target": sot if pos == "F" else 0,
                "yellow_cards": 0, "key_passes": 1, "rating": 7.0, "fouls_committed": 1,
                "fouls_drawn": 1, "duels_total": 4, "duels_won": 2, "team_corners": 5,
                "red_cards": 0, "passes_total": 30, "offsides": 0, "tackles_total": 1,
                "interceptions": 1, "dribbles_attempted": 1, "dribbles_success": 1,
                "dribbles_past": 0, "goals_conceded": 0, "saves": 2 if pos == "G" else 0,
                "penalty_scored": 0, "penalty_missed": 0, "penalty_won": 0, "penalty_saved": 0,
                "sp_goal": 1 if (pos == "D" and ih) else 0, "headed_goal": 0, "fk_goal": 0,
                "sp_assist": 0, "fk_assist": 0,
                "team_id": IDS[team], "opponent_id": IDS[away] if ih else IDS[home]})
    return rows


def _season_rows(season: str, start: str, rounds: int, fid0: int, **kw):
    rows = []
    for r in range(rounds):
        d = pd.Timestamp(start) + pd.Timedelta(days=7 * r)
        pairs = [("Arsenal", "Chelsea"), ("Everton", "Fulham")] if r % 2 == 0 else \
                [("Chelsea", "Everton"), ("Fulham", "Arsenal")]
        for k, (h, a) in enumerate(pairs):
            rows += _player_rows(fid0 + 10 * r + k, d, h, a, season, **kw)
    return rows


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """No strength tables, no collected data, no Understat: only the rows a test passes."""
    sdir = tmp_path / "strength"
    sdir.mkdir()
    monkeypatch.setattr(lookup, "DIR", sdir)
    monkeypatch.setattr(FE, "COLLECT_DIR", tmp_path)
    monkeypatch.setattr(SPF, "DIR", tmp_path)
    from player_model import cup_history as CH
    monkeypatch.setattr(CH, "CUP_ROWS_FILE", tmp_path / "cup_player_rows.parquet")

    def _clear():
        from player_model import team_identity as TI
        TI._CACHE.clear()
        lookup._LOADED.clear()
        FE._COLLECTED.clear()
        FE._SERVE_CACHE.clear()
    _clear()
    yield tmp_path
    _clear()


# ── 1. set-piece proxies purged ──────────────────────────────────────────────────────────────────

def test_setpiece_proxies_are_not_built_for_training_or_serving(iso):
    rows = _season_rows("2024", "2024-08-10", 8, 7_000_000)
    b = FE.build_features(rows)
    assert not [c for c in PURGED if c in b.columns]
    up = [{"player_id": IDS["Arsenal"] * 100 + 1, "team": "Arsenal", "opponent": "Chelsea", "is_home": 1,
           "position": "D", "fixture_league": "Premier League", "fixture_date": "2024-10-20"}]
    out = FE.build_upcoming_features(up, b, {}, {}, FE._build_history_index(b))
    assert len(out) == 1 and not [c for c in PURGED if c in out.columns]
    # the Understat set-piece features are the only set-piece columns (NaN: no Understat data)
    assert all(c in out.columns and math.isnan(out[c].iloc[0]) for c in SPF.SETPIECE_COLUMNS)


def test_setpiece_proxies_never_in_the_training_feature_list_or_contract():
    from player_model import props_contract as PC
    assert not set(PURGED) & set(C.PLAYER_FEATURE_COLS)
    assert set(PURGED) <= set(C.SETPIECE_PROXIES_REMOVED)
    assert not set(PURGED) & set(PC.load_contract()["features"])


def test_no_set_piece_lazy_factor():
    from player_model import predict as P
    feat = {"set_piece_threat_score": 9.0, "pos_defender": 1, "pos_midfielder": 0,
            "opp_sot_conceded_pg": 1.0, "minutes_pg": 90}
    assert "SET_PIECE" not in P._detect_lazy_factors({}, feat, "sot")
    assert "set_piece_threat_score" not in inspect.getsource(P)


def _one_row(**vals) -> pd.DataFrame:
    df = pd.DataFrame([vals])
    FE._ensure(df, FE._COMPOSITE_INPUTS)
    return FE._add_composites(df)


def test_measured_ratios_are_not_clipped_to_typed_bounds():
    f = _one_row(opp_goals_conceded_pg=4.0, lg_goals_per_team=1.0, goals_pg=2.0, sot_pg=1.5,
                 opp_def_player_rating_pg=8.0, lg_def_rating=1.0, opp_top_def_rating=8.0,
                 lg_def_rating_top=1.0, opp_def_aerial_win_rate=0.6, lg_def_aerial=0.5,
                 offsides_pg=3.0, pos_forward=1, pos_midfielder=0, pos_defender=0,
                 fouls_per90=1.0, dribbles_pg=0.0, dribble_success_rate=1.0, opp_mid_cards_pg=1.0,
                 referee_yellows_pg=0.1, lg_yellows_per_match=4.0).iloc[0]
    assert f["opp_strength_index"] == pytest.approx(4.0)            # was clipped to 3.0
    assert f["quality_adj_goals_pg"] == pytest.approx(8.0)          # was clipped to 5.0
    assert f["opp_def_player_quality"] == pytest.approx(64 * 1.2)   # was clipped to 4.0
    assert f["offside_aggressiveness"] == pytest.approx(3.0)        # was clipped to 2.0
    assert f["shooting_efficiency_index"] == pytest.approx(2.0 / 1.5)   # was clipped to 1.0
    assert f["card_clash_index"] == pytest.approx(1.0 * 1.0 * 0.1 / 4.0)  # ratio floor 0.1 gone


def test_foul_committer_ratio_is_the_exact_share(iso):
    rows = _season_rows("2024", "2024-08-10", 4, 7_100_000)
    b = FE.build_features(rows)
    # every row: 1 committed, 1 drawn -> exactly 0.5 (was 1 / 2.01)
    assert (b["foul_committer_ratio"].dropna() == 0.5).all() and b["foul_committer_ratio"].notna().any()


def test_typed_weight_formulas_are_withheld_from_training():
    typed = {"penalty_threat_score", "card_risk_index", "midfield_engine_score",
             "progressive_carrier_score", "disciplinary_pressure_index", "foul_magnet_score",
             "defensive_vulnerability_index", "creative_playmaker_score", "opp_mid_discipline"}
    assert not typed & set(C.PLAYER_FEATURE_COLS)
    assert typed <= set(C.TYPED_FORMULA_FEATURES_WITHHELD)
