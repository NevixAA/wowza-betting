"""dt/features (Data Truth 2026-11-01): one focused test per behaviour change in the features scope.

Every site below used to INVENT a value (typed prior, global default, 0 for missing, positional or
name-only match). Each test pins the honest behaviour: the value is computed from real data or it
is explicitly missing (NaN / None / skipped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── feature_engineering ──────────────────────────────────────────────────────────────────────────
def test_referee_unknown_is_nan_and_missing_fouls_do_not_drag_the_mean():
    from src.feature_engineering import _compute_referee_stats
    h = pd.DataFrame({"referee": ["A"] * 4 + ["B"],
                      "home_fouls": [10, 10, 10, np.nan, 12],
                      "away_fouls": [10, 10, 10, 10, 12]})
    stats, glob = _compute_referee_stats(h)
    assert stats == {"A": 20.0}                         # the NaN-foul row is excluded, not a 0
    assert glob == pytest.approx(21.0)
    assert np.isnan(_compute_referee_stats(pd.DataFrame({"x": [1]}))[1])   # no column -> NaN, not 22.0


def test_build_features_referee_unknown_gets_nan():
    from src.feature_engineering import build_features
    from test_standard_forensic_repair import _frame
    raw = _frame()
    raw["referee"] = np.where(np.arange(len(raw)) % 2 == 0, "Ref One", None)
    f = build_features(raw)
    assert f.loc[f["referee"].isna(), "referee_foul_avg"].isna().all()


def test_rest_days_are_keyed_by_league():
    from src.feature_engineering import _add_rest_days
    d = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-10"]),
                      "league": ["L1", "L2", "L1"],
                      "home_team": ["United", "United", "United"],
                      "away_team": ["X", "Y", "Z"]})
    r = _add_rest_days(d)
    # the L2 'United' on 01-05 neither has history nor resets L1 United's clock
    assert np.isnan(r["home_rest_days"].iloc[1]) and r["home_rest_days"].iloc[2] == 9


def test_legacy_ht_strength_uses_the_real_league_average_not_065():
    from src.feature_engineering import build_features
    from test_standard_forensic_repair import _frame
    raw = _frame()
    raw["ht_home_goals"] = (raw["home_goals"] // 2).astype(float)
    raw["ht_away_goals"] = (raw["away_goals"] // 2).astype(float)
    f = build_features(raw)
    k = f["home_ht_attack_str"].notna()
    assert k.any()
    assert np.allclose(f.loc[k, "home_ht_attack_str"],
                       f.loc[k, "home_ht_scored_last5"] / (f.loc[k, "league_avg_goals"] / 2))


# ── features_repaired ────────────────────────────────────────────────────────────────────────────
def test_unresolved_board_name_is_flagged_not_replaced_by_the_raw_spelling():
    from src.features_repaired import build_upcoming_features_repaired
    from test_standard_forensic_repair import _frame, STD_A
    raw = _frame()
    teams = sorted(raw.loc[raw.league == STD_A, "home_team"].unique())
    up = pd.DataFrame({"date": raw["date"].max() + pd.Timedelta(days=7), "league": STD_A,
                       "home_team": [teams[0], "Completely Unknown Athletic"],
                       "away_team": [teams[1], teams[2]]})
    f = build_upcoming_features_repaired(up, raw)
    assert f["name_unresolved"].tolist() == [False, True]
    assert pd.isna(f["history_team_home"].iloc[1])           # missing, not the raw spelling
    assert np.isnan(f["home_scored_last5"].iloc[1]) and np.isnan(f["away_scored_last5"].iloc[1])
    assert f["home_scored_last5"].iloc[0] == f["home_scored_last5"].iloc[0]   # resolved row featurised


def test_asof_price_class_is_missing_when_the_frame_does_not_state_it():
    from src.features_repaired import build_features_repaired
    from test_standard_forensic_repair import _frame
    f = build_features_repaired(_frame(), price_mode="asof")
    assert f["_price_class"].isna().all()                 # never a claimed "ENTRY_ASOF"


# ── ht_features ─────────────────────────────────────────────────────────────────────────────────
def test_ht_priors_come_from_pooled_history_and_teams_are_league_keyed():
    from src.ht_features import build_ht_features
    d = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15"]),
        "league": ["L1", "L1", "L2"],
        "home_team": ["Same", "Same", "Same"], "away_team": ["B", "C", "D"],
        "ht_home_goals": [2, 0, np.nan], "ht_away_goals": [0, 0, np.nan],
        "home_goals": [3, 1, np.nan], "away_goals": [1, 0, np.nan]})
    f = build_ht_features(d)
    assert f["ht_lambda_total"].iloc[:1].isna().all()     # nothing played anywhere yet -> NaN
    assert f["ht_home_n"].iloc[1] > 0 and f["ht_home_n"].iloc[2] == 0   # L2 'Same' has no history
    assert f["ht_league_base"].iloc[1] == pytest.approx(2.0)   # pooled as-of means 2.0 + 0.0


# ── backtest ────────────────────────────────────────────────────────────────────────────────────
def test_side_market_fair_prob_needs_the_two_sided_price():
    from src.backtest import _two_sided_fair
    from src.edge_engine import power_devig
    fair = _two_sided_fair(pd.Series([1.80, 1.80]), pd.Series([2.05, np.nan]))
    assert fair.iloc[0] == pytest.approx(power_devig(1.80, 2.05))
    assert np.isnan(fair.iloc[1])                          # one-sided -> not priced, no typed 1.08
    import src.backtest as B
    assert not hasattr(B, "_DEFAULT_ODDS") and not hasattr(B, "_OVERROUND")


def test_threshold_walk_forward_orders_derived_seasons():
    from src.backtest import _derived_season
    d = pd.DataFrame({"date": pd.to_datetime(["2024-09-01", "2025-03-01", "2025-09-01", "bad"],
                                             errors="coerce"),
                      "league": ["Championship"] * 4})
    s = _derived_season(d)
    assert s.iloc[:3].tolist() == ["2024/25", "2024/25", "2025/26"] and pd.isna(s.iloc[3])


# ── live_scanner ────────────────────────────────────────────────────────────────────────────────
def _game(**kw):
    g = {"fixture_id": None, "league": "Championship", "home_team": "Leeds", "away_team": "Hull",
         "elapsed_mins": 75, "home_goals": 0, "away_goals": 0, "total_goals": 0}
    g.update(kw)
    return g


def test_live_no_prematch_probability_means_no_signal():
    from src import live_scanner as LS
    preds = pd.DataFrame({"league": ["Championship"], "home_team": ["Leeds"], "away_team": ["Hull"],
                          "p_over25": [np.nan]})
    assert LS._detect_signals([_game()], preds) == []
    wc = _game(league="World Cup 2026", home_team="Brazil", away_team="Spain")
    assert LS._detect_signals([wc], preds) == []           # no typed 0.45 for national teams


def test_live_no_clock_or_score_means_no_signal():
    from src import live_scanner as LS
    preds = pd.DataFrame({"league": ["Championship"], "home_team": ["Leeds"], "away_team": ["Hull"],
                          "p_over25": [0.30], "home_attack_str": [0.8], "away_attack_str": [0.8]})
    assert LS._detect_signals([_game(elapsed_mins=None)], preds) == []
    assert LS._detect_signals([_game(home_goals=None, total_goals=None)], preds) == []


def test_live_prediction_match_is_league_scoped():
    from src import live_scanner as LS
    preds = pd.DataFrame({"league": ["League One"], "home_team": ["Leeds"], "away_team": ["Hull"],
                          "p_over25": [0.4]})
    assert LS._match_prediction(preds, "Leeds", "Hull", "Championship") is None
    assert LS._match_prediction(preds, "Leeds", "Hull", "League One")["p_over25"] == 0.4
    assert LS._match_prediction(preds, "Leeds", "Hull", None) is None


# ── api_football / api_football_ou ──────────────────────────────────────────────────────────────
def test_unknown_formation_is_nan():
    from src.api_football_ou import _formation_attack_score
    assert np.isnan(_formation_attack_score("")) and np.isnan(_formation_attack_score("3-3-3-1"))
    assert _formation_attack_score("4-3-3") == 0.85


def test_team_season_rates_are_none_when_no_games_played(monkeypatch):
    import src.api_football_ou as AF
    monkeypatch.setattr(AF, "_KEY", "k")
    monkeypatch.setattr(AF, "_season_for", lambda *a, **k: "2026")
    monkeypatch.setattr(AF, "_season_known", lambda s: True)
    monkeypatch.setattr(AF, "_resolve_team_id", lambda *a, **k: 7)
    monkeypatch.setattr(AF.config, "API_FOOTBALL_IDS", {"Championship": 40})
    monkeypatch.setattr(AF, "_get", lambda *a, **k: {"response": {
        "fixtures": {"played": {"home": 0, "away": 2}, "wins": {"home": 0, "away": 1}},
        "goals": {}, "clean_sheet": {"home": 0, "away": 1}, "form": ""}})
    s = AF.fetch_team_season_stats("Leeds", "Championship")
    assert s["cs_rate_h"] is None and s["wins_rate_h"] is None   # was 0 / max(0,1) = 0.0
    assert s["cs_rate_a"] == 0.5 and s["form_win_rate"] is None


def test_upcoming_team_stats_are_keyed_by_league(monkeypatch):
    import src.api_football_ou as AF
    monkeypatch.setattr(AF, "_KEY", "k")
    monkeypatch.setattr(AF, "fetch_team_season_stats",
                        lambda team, lg, *a, **k: {"goals_for_h": 1.0 if lg == "A" else 2.0})
    up = pd.DataFrame({"league": ["A", "B"], "home_team": ["Same", "Same"], "away_team": ["x", "y"],
                       "date": ["2026-11-02", "2026-11-02"]})
    r = AF.fetch_upcoming_team_stats(up)
    assert r[("A", "Same")]["goals_for_h"] == 1.0 and r[("B", "Same")]["goals_for_h"] == 2.0


def test_h2h_skips_null_scores_and_coach_unknown_is_nan(monkeypatch):
    import src.api_football_ou as AF
    monkeypatch.setattr(AF, "_KEY", "k")
    monkeypatch.setattr(AF.config, "API_FOOTBALL_IDS", {"Championship": 40})
    monkeypatch.setattr(AF, "_season_for", lambda *a, **k: "2026")
    monkeypatch.setattr(AF, "_season_known", lambda s: True)
    monkeypatch.setattr(AF, "_resolve_team_id", lambda name, *a: {"H": 1, "A": 2}.get(name))
    monkeypatch.setattr(AF, "_get", lambda ep, *a, **k: {"response": [
        {"goals": {"home": None, "away": None}, "teams": {"home": {"id": 1}}},
        {"goals": {"home": 2, "away": 1}, "teams": {"home": {"id": 1, "winner": True}}}]}
        if ep == "fixtures/headtohead" else {"response": []})
    up = pd.DataFrame({"league": ["Championship"], "home_team": ["H"], "away_team": ["A"],
                       "date": ["2026-11-02"]})
    h = AF.fetch_h2h_features(up)[0]
    assert h["h2h_n"] == 1.0 and h["h2h_avg_goals"] == 3.0    # the postponed one is not a 0-0
    c = AF.fetch_coach_features(up)[0]
    assert all(np.isnan(v) for v in c.values())               # no 180 days / "not caretaker"


def test_possession_absent_is_none():
    from src.api_football_ou import _pct
    assert _pct(None) is None and _pct("55%") == 55.0 and _pct("") is None


def test_standings_lookup_refuses_a_shared_word_match():
    from src.api_football import lookup_team_standing, _per_game
    smap = {"Manchester United": {"rank": 3}}
    assert lookup_team_standing(smap, "Manchester City") is None
    assert lookup_team_standing(smap, "Manchester United") == {"rank": 3}
    assert _per_game(3, 0) is None and _per_game(None, 4) is None and _per_game(6, 3) == 2.0


def test_injuries_match_by_team_id():
    from src.api_football import count_attacking_injuries
    inj = [{"team_id": 1, "team_name": "Manchester United", "position": "Attacker"},
           {"team_id": 2, "team_name": "Manchester City", "position": "Attacker"}]
    assert count_attacking_injuries(inj, 2) == 1 and count_attacking_injuries(inj, None) == 0


# ── data_loader ─────────────────────────────────────────────────────────────────────────────────
def test_same_club_pairs_come_from_the_registry_not_code():
    import src.data_loader as DL
    assert not hasattr(DL, "_SAME_CLUB_NO_OVERLAP") and not hasattr(DL, "_SEASON_MAP")
    assert DL._safe_joins() is True                           # no positional legacy rollback
