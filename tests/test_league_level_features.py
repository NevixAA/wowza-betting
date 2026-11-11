"""As-of league LEVEL features (owner 2026-11-11, "fix before final refit").

The Phase 2 champion predicted the same BTTS level in every league because no input carried the
league's own scoring level. src.feature_engineering.LEAGUE_LEVEL_FEATURES are the league's own
rates over its matches on STRICTLY EARLIER DATES: no same-day result, no other league, no pooled
fallback (a league's first date is NaN), COVID seasons never in the aggregate, and serving
(league_level_asof over the history strictly before the fixture) equals training (build_features).

    python -m pytest tests/test_league_level_features.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import feature_engineering as FE

COLS = list(FE.LEAGUE_LEVEL_FEATURES)


def _m(date, league, home, away, hg, ag, hthg=0, htag=0):
    tg = hg + ag
    ht = hthg + htag
    return {"date": pd.Timestamp(date), "league": league, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "total_goals": tg, "over25": float(tg > 2.5),
            "over15": float(tg > 1.5), "over35": float(tg > 3.5), "btts": float(hg > 0 and ag > 0),
            "ht_home_goals": hthg, "ht_away_goals": htag, "ht_total_goals": ht,
            "ht_over05": float(ht >= 1), "ht_over15": float(ht >= 2),
            "home_shots": 10, "away_shots": 8, "home_sot": 5, "away_sot": 4, "home_corners": 5,
            "away_corners": 4, "home_fouls": 11, "away_fouls": 13, "odds_over25": 1.9, "odds_under25": 1.95}


@pytest.fixture()
def no_covid(monkeypatch):
    """Synthetic leagues have no season rule; the COVID mask is exercised in its own test."""
    monkeypatch.setattr(config, "covid_season_mask", lambda df, *a, **k: pd.Series(False, index=df.index))


def _hist():
    rows = []
    d0 = pd.Timestamp("2025-08-02")
    for i in range(10):
        d = d0 + pd.Timedelta(days=7 * i)
        rows.append(_m(d, "Serie B", f"A{i}", f"B{i}", 2, 1, 1, 0))      # btts, 3 goals
        rows.append(_m(d, "Serie B", f"C{i}", f"D{i}", 0, 0))            # no btts, 0 goals
        rows.append(_m(d, "Ligue 2", f"E{i}", f"F{i}", 3, 2, 1, 1))      # btts, 5 goals
    return pd.DataFrame(rows)


def test_training_values_are_strictly_earlier_dates_of_the_own_league(no_covid):
    h = _hist()
    # a same-day extra Serie B match that must NOT feed its own date's rows
    extra = _m(h["date"].max(), "Serie B", "S", "T", 4, 4, 2, 2)
    f = FE.build_features(pd.concat([h, pd.DataFrame([extra])], ignore_index=True))
    first = f[f["date"] == f["date"].min()]
    assert first[COLS].isna().all().all()                                # no history -> NaN, no fallback
    last_sb = f[(f["date"] == f["date"].max()) & (f["league"] == "Serie B")]
    assert np.allclose(last_sb["league_asof_btts_rate"], 0.5)            # 9 dates of (1, 0), not the 4-4
    assert np.allclose(last_sb["league_asof_goals_pg"], 1.5)
    assert np.allclose(last_sb["league_asof_over25_rate"], 0.5)
    assert np.allclose(last_sb["league_asof_ht_over05_rate"], 0.5)
    last_l2 = f[(f["date"] == f["date"].max()) & (f["league"] == "Ligue 2")]
    assert np.allclose(last_l2["league_asof_goals_pg"], 5.0)             # never Serie B's numbers
    assert np.allclose(last_l2["league_asof_ht_over15_rate"], 1.0)


def test_serving_equals_training_for_the_next_fixture(no_covid):
    h = _hist()
    nxt = _m(h["date"].max() + pd.Timedelta(days=7), "Serie B", "A0", "B0", 1, 1)
    train = FE.build_features(pd.concat([h, pd.DataFrame([nxt])], ignore_index=True))
    tr = train[(train["date"] == nxt["date"])].iloc[0]
    sv = FE.league_level_asof(pd.DataFrame([{k: nxt[k] for k in ("date", "league")}]), h).iloc[0]
    for c in COLS:
        assert tr[c] == pytest.approx(sv[c], abs=1e-12), c
    # a fixture in a league with no history at all stays NaN at serving too
    nf = FE.league_level_asof(pd.DataFrame([{"date": nxt["date"], "league": "Unknown"}]), h)
    assert nf[COLS].isna().all().all()


def test_serving_ignores_history_on_or_after_the_fixture_date(no_covid):
    h = _hist()
    d = h["date"].max()
    a = FE.league_level_asof(pd.DataFrame([{"date": d, "league": "Serie B"}]), h)
    b = FE.league_level_asof(pd.DataFrame([{"date": d, "league": "Serie B"}]), h[h["date"] < d])
    pd.testing.assert_frame_equal(a, b)


def test_a_missing_result_column_is_nan_never_invented(no_covid):
    h = _hist().drop(columns=["ht_over05", "ht_over15"])
    f = FE.add_league_level_features(h)
    assert f["league_asof_ht_over05_rate"].isna().all()
    assert f["league_asof_btts_rate"].notna().any()


def test_covid_season_rows_never_feed_the_aggregate(monkeypatch):
    h = _hist()
    covid = h["date"] < h["date"].min() + pd.Timedelta(days=30)         # first 5 dates "COVID"
    h.loc[covid & (h["league"] == "Serie B"), ["home_goals", "away_goals", "total_goals", "btts"]] = [5, 5, 10, 1.0]
    monkeypatch.setattr(config, "EXCLUDE_COVID_SEASONS", True)
    monkeypatch.setattr(config, "covid_season_mask",
                        lambda df, *a, **k: pd.Series(pd.to_datetime(df["date"]).to_numpy()
                                                      < np.datetime64(h["date"].min() + pd.Timedelta(days=30)),
                                                      index=df.index))
    d = h["date"].max()
    sv = FE.league_level_asof(pd.DataFrame([{"date": d, "league": "Serie B"}]), h).iloc[0]
    assert sv["league_asof_btts_rate"] == pytest.approx(0.5)              # only the non-COVID dates
    tr = FE.add_league_level_features(h)
    row = tr[(tr["date"] == d) & (tr["league"] == "Serie B")].iloc[0]
    assert row["league_asof_btts_rate"] == pytest.approx(sv["league_asof_btts_rate"])
