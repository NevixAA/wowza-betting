"""Props Data Truth (dt/props, 2026-11-01): no typed feature value, measured opponent / keeper /
penalty / referee / league context with explicit minimum samples, one definition for training and
serving, and ids instead of names.

    python -m pytest tests/test_props_data_truth.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from player_model import config as C
from player_model import feature_engineering as FE

LEAGUES = {"Premier League": ["Arsenal", "Chelsea", "Everton", "Fulham"],
           "Serie A": ["Inter", "Milan", "Roma", "Lazio"]}
# Per-league scoring so league normalisers must differ (Serie A low-scoring here).
GOALS = {"Premier League": 2, "Serie A": 0}


def _history(rounds: int = 24) -> list[dict]:
    rows, fid = [], 1000
    for lg, teams in LEAGUES.items():
        d0 = pd.Timestamp("2025-08-01")
        for r in range(rounds):
            # every team plays once per round (two fixtures per league per round)
            pairs = [(teams[0], teams[1]), (teams[2], teams[3])] if r % 2 == 0 else \
                    [(teams[0], teams[2]), (teams[1], teams[3])]
            for k, (h, a) in enumerate(pairs):
                fid += 1
                date = d0 + pd.Timedelta(days=7 * r)
                for side, team in (("h", h), ("a", a)):
                    base = abs(hash((lg, team))) % 1000 * 10
                    for pos, j in (("G", 0), ("D", 1), ("D", 2), ("M", 3), ("M", 4), ("F", 5)):
                        rows.append({
                            "fixture_id": fid, "player_id": base + j, "player_name": f"{team}-{pos}{j}",
                            "date": date, "league": lg, "home_team": h, "away_team": a, "team": team,
                            "is_home": int(side == "h"), "referee": f"Ref {lg[0]}{k}", "season": "2025",
                            "position": pos, "minutes": 90, "started": 1,
                            "goals": GOALS[lg] if pos == "F" and side == "h" else 0,
                            "assists": 0, "shots_total": 3 if pos in "FM" else 0,
                            "shots_on_target": 2 if pos == "F" else 0,
                            "yellow_cards": 1 if (pos == "M" and team in (teams[1], teams[3])) else 0,
                            "key_passes": 1, "rating": 7.0 + 0.1 * j,
                            "fouls_committed": 2 if pos == "M" else 1, "fouls_drawn": 1,
                            "duels_total": 4, "duels_won": 2, "team_corners": 6 if side == "h" else 3,
                            "red_cards": 0, "passes_total": 30, "offsides": 0, "tackles_total": 1,
                            "interceptions": 1, "dribbles_attempted": 1, "dribbles_success": 1,
                            "dribbles_past": 0,
                            "goals_conceded": (GOALS[lg] if side == "a" else 0) if pos == "G" else 0,
                            "saves": 2 if pos == "G" else 0,
                            "penalty_scored": 1 if (pos == "F" and r < 4) else 0,
                            "penalty_missed": 1 if (pos == "F" and r == 4) else 0,
                            "penalty_won": 0, "penalty_saved": 0,
                            "height_cm": 180.0, "weight_kg": 75.0, "age": 25.0,   # old placeholders
                        })
    return rows


@pytest.fixture(scope="module")
def built():
    return FE.build_features(_history())


def test_rolling_helper_matches_the_old_lambda():
    df = pd.DataFrame({"p": [1, 1, 1, 1, 2, 2], "x": [1.0, np.nan, 3.0, 5.0, 2.0, 4.0]})
    g = df.groupby("p", group_keys=False)
    old = g["x"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    pd.testing.assert_series_equal(FE._roll(g, "x", 3, 2), old, check_names=False)
    old_e = g["x"].transform(lambda s: s.shift(1).expanding().mean())
    pd.testing.assert_series_equal(FE._expand(g, "x"), old_e, check_names=False)


def test_no_typed_opponent_values_below_the_minimum_sample(built):
    # A team's first MIN_OPP_MATCHES fixtures have no measured history: NaN, never 1.3 / 4.5 / 0.72.
    early = built[built["date"] < pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * C.MIN_OPP_MATCHES)]
    for c in ("opp_goals_conceded_pg", "opp_sot_conceded_pg", "opp_gk_save_rate",
              "opp_def_fouls_pg", "opp_mid_cards_pg", "opp_strength_index"):
        assert early[c].isna().all(), c
    late = built[built["date"] >= pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * (C.MIN_OPP_MATCHES + 1))]
    assert late["opp_goals_conceded_pg"].notna().all()
    for typed in (1.3, 4.5):
        assert not (built["opp_goals_conceded_pg"] == typed).any()


def test_opponent_concede_rate_is_the_measured_one(built):
    # In Serie A nobody scores (GOALS = 0): every measured concede rate there is exactly 0.
    sa = built[(built["league"] == "Serie A") & built["opp_goals_conceded_pg"].notna()]
    assert len(sa) and (sa["opp_goals_conceded_pg"] == 0).all()
    pl = built[(built["league"] == "Premier League") & built["opp_goals_conceded_pg"].notna()]
    assert (pl["opp_goals_conceded_pg"] > 0).any()


def test_league_normalisers_are_per_league_not_global(built):
    lg = built.dropna(subset=["lg_goals_per_team"]).groupby("league")["lg_goals_per_team"].last()
    assert lg["Serie A"] == 0.0 and lg["Premier League"] > 0.5    # each league's OWN average
    lc = built.dropna(subset=["lg_corners_per_team"]).groupby("league")["lg_corners_per_team"].last()
    assert lc.notna().all()


def test_opp_mid_discipline_is_the_opponents_own_midfielders(built):
    # Only teams[1]/teams[3] midfielders get booked. A player facing Chelsea sees Chelsea's (1.0).
    x = built[(built["opponent"] == "Chelsea") & built["opp_mid_cards_pg"].notna()]
    assert len(x) and (x["opp_mid_cards_pg"] == 1.0).all()
    y = built[(built["opponent"] == "Arsenal") & built["opp_mid_cards_pg"].notna()]
    assert (y["opp_mid_cards_pg"] == 0.0).all()


def test_keeper_save_rate_from_the_keepers_stats(built):
    # Inter's keeper: 2 saves, 0 conceded every match -> 1.0. No typed 0.72 anywhere.
    x = built[(built["opponent"] == "Inter") & built["opp_gk_save_rate"].notna()]
    assert len(x) and (x["opp_gk_save_rate"] == 1.0).all()
    assert not (built["opp_gk_save_rate"] == 0.72).any()


def test_penalty_conversion_from_the_players_record(built):
    f = built[built["position"] == "F"].sort_values("date")
    one = f[f["player_id"] == f["player_id"].iloc[0]].reset_index(drop=True)
    # attempts before match k = k (one per match for r < 5): NaN until MIN_PEN_ATTEMPTS
    assert one.loc[: C.MIN_PEN_ATTEMPTS - 2, "penalty_conversion_rate"].isna().all()
    assert one.loc[C.MIN_PEN_ATTEMPTS - 1, "penalty_conversion_rate"] == 1.0
    assert one["penalty_conversion_rate"].iloc[-1] == pytest.approx(4 / 5)
    assert not (built["penalty_conversion_rate"] == 0.75).any()


def test_placeholder_profiles_rest_and_rating_are_missing_not_typed(built):
    for c in ("height_cm", "age", "age_peak_delta", "opp_def_mean_height", "height_diff_vs_opp_def"):
        assert built[c].isna().all(), c
    first = built.sort_values("date").groupby("player_id").head(1)
    # first kept row is a player's 2nd match: rest_days measured (7), not the typed 6.0
    assert (first["rest_days"] == 7).all()
    for c in ("league_tier", "league_quality", "player_career_avg_quality", "context_quality_discount"):
        assert built[c].isna().all(), c


def test_serving_uses_the_training_definitions(built):
    hist = built
    nxt = pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * 24)
    up = hist[hist["team"] == "Arsenal"].drop_duplicates("player_id", keep="last").copy()
    up["opponent"] = "Chelsea FC"                      # a spelling variant: resolved, not compared
    up["is_home"] = 1.0
    up["fixture_league"] = "Premier League"
    up["fixture_date"] = str(nxt.date())
    ref = FE.referee_profile_from_history(hist, "Ref P0", "Premier League")
    out = FE.build_upcoming_features(up.to_dict("records"), hist, ref, {}, FE._build_history_index(hist))
    assert len(out) == len(up)
    t = FE.team_match_table(hist)
    chelsea = t[t["_tkey"].str.startswith("Chelsea|")].sort_values("date").tail(C.ROLLING_N)
    assert out["opp_goals_conceded_pg"].iloc[0] == pytest.approx(chelsea["goals_conceded"].mean())
    assert out["opp_mid_cards_pg"].iloc[0] == pytest.approx(1.0)
    assert (out["rest_days"] == 7).all()
    assert out["referee_yellows_pg"].notna().all() and ref["n_games"] >= C.MIN_REF_MATCHES
    # every model feature exists at serve time; typed defaults are gone
    assert not [c for c in C.PLAYER_FEATURE_COLS if c not in out.columns]
    unknown = FE.build_upcoming_features(
        [dict(up.iloc[0], opponent="Nobody United")], hist, {}, {}, FE._build_history_index(hist))
    assert math.isnan(unknown["opp_goals_conceded_pg"].iloc[0])
    assert math.isnan(unknown["referee_yellows_pg"].iloc[0])


def test_player_without_id_is_skipped():
    hist = FE.build_features(_history(6))
    row = dict(hist.iloc[-1], player_id=None, opponent="Chelsea", is_home=1.0)
    assert FE.build_upcoming_features([row], hist).empty


def test_ges_is_nan_when_an_input_is_missing():
    ok = {"goals_pg": 0.4, "shots_pg": 3.0, "minutes_pg": 85.0, "pos_forward": 1, "sot_rate": 0.5}
    assert FE.compute_ges(ok, opp_weakness=1.2) == FE.compute_ges(ok, opp_weakness=1.2)
    assert math.isnan(FE.compute_ges(ok))                         # no opponent rate -> NaN
    assert math.isnan(FE.compute_ges({**ok, "goals_pg": None}, opp_weakness=1.2))


def test_tier_is_avoid_without_a_ges_or_a_price():
    from player_model import predict as P
    assert math.isnan(P._devig_fair_prob(None)) and math.isnan(P._devig_fair_prob(1.0))
    assert P._classify_tier(1.0, 4.0, 1.0, 1.0, 3, float("nan"), "goals") == "AVOID"
    assert P._classify_tier(float("nan"), 4.0, float("nan"), 1.0, 3, 0.9, "cards") == "AVOID"


def test_no_synthetic_assist_price():
    from player_model import predict as P
    tips = pd.DataFrame([{"market": "assists", "model_prob": 0.4, "confidence": 0.6, "ges": None,
                          "lazy_factors": "", "position": "M", "market_odds": np.nan, "tier": "AVOID",
                          "ev": np.nan, "data_source": "match_level"}])
    out = P.enrich_no_odds_markets(tips.copy())
    assert out["market_odds"].isna().all() and (out["tier"] == "AVOID").all() and out["ev"].isna().all()


def test_missing_input_is_scored_by_the_nan_native_model_only():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from player_model import model as M
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = (X["a"] > 0).astype(int)
    lr = LogisticRegression().fit(X, y)
    hgb = HistGradientBoostingClassifier(max_iter=20).fit(X, y)
    Xq = pd.DataFrame({"a": [1.0, np.nan], "b": [0.0, 0.0]})
    p = M._ensemble_proba({"logistic": lr, "gradient_boost": hgb}, Xq)
    assert p[0] == pytest.approx((lr.predict_proba(Xq.iloc[[0]])[0, 1]
                                  + hgb.predict_proba(Xq.iloc[[0]])[0, 1]) / 2)
    assert p[1] == pytest.approx(hgb.predict_proba(Xq.iloc[[1]])[0, 1])
    assert np.isnan(M._ensemble_proba({"logistic": lr}, Xq)[1])          # nobody can score it


def test_ledger_dedupes_on_ids_and_refuses_undated(tmp_path):
    from player_model import ledger as L
    f = tmp_path / "player_ledger.csv"
    base = {"tier": "VALUABLE", "match": "A vs B", "market": "goals", "date": "2026-11-07",
            "fixture_id": 77, "league": "Premier League"}
    tips = pd.DataFrame([{**base, "player_id": 1, "player_name": "J. Silva"},
                         {**base, "player_id": 2, "player_name": "J. Silva"},     # namesake
                         {**base, "player_id": 3, "player_name": "X", "date": None}])
    assert L.append_player_signals(tips, f) == 2                    # both Silvas, undated refused
    assert L.append_player_signals(tips, f) == 0


def test_api_ids_not_names():
    from player_model import data_fetcher as DF
    assert DF._stable_player_id(None, "Somebody") is None
    assert DF._stable_player_id("123", "Somebody") == 123
    assert DF._sp_classified([{"detail": "Normal Goal"}]) is False
    assert DF._sp_classified([{"detail": "Penalty"}, {"detail": "Header"}]) is True
    assert DF._sp_classified([]) is True and DF._sp_classified(None) is False


def test_fpl_missing_status_is_unknown():
    from player_model import fpl_api as F
    b = {"teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
         "elements": [{"id": 9, "first_name": "A", "second_name": "B", "team": 1, "element_type": 4}]}
    r = F.players_df(b).iloc[0]
    assert r["status"] == "unknown" and r["availability"] == "unknown" and math.isnan(r["owned_pct"])
