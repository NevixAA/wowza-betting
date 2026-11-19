"""Data Truth 2026-11-01 (serve scope): config / pipeline / serving sites that used to invent a value."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config


# ── COVID seasons: per league from its season rule, not two typed labels ─────────────────────
def test_covid_mask_matches_old_labels_for_start_year_leagues():
    d = pd.DataFrame({"league": ["Championship"] * 4,
                      "date": ["2019-06-30", "2019-08-10", "2021-05-20", "2021-08-10"]})
    assert config.covid_season_mask(d).tolist() == [False, True, True, False]


def test_covid_mask_catches_calendar_year_2020():
    """The label set {"2019/20", "2020/21"} never matched a calendar league's "2020"."""
    d = pd.DataFrame({"league": ["Norway Eliteserien"] * 3,
                      "date": ["2019-09-01", "2020-09-01", "2021-09-01"]})
    assert config.covid_season_mask(d).tolist() == [False, True, False]


def test_covid_mask_never_flags_a_fixed_tournament_season():
    assert config.covid_seasons_of("World Cup 2026") == set()


def test_covid_mask_keeps_unresolvable_rows_and_preserves_the_index():
    d = pd.DataFrame({"league": ["No Such League", "Championship"], "date": ["2020-05-01", None]},
                     index=[7, 7])
    m = config.covid_season_mask(d)
    assert list(m.index) == [7, 7] and not m.any()


# ── league-keyed tables are validated against the registry ──────────────────────────────────
def test_league_tables_are_registry_keyed():
    from src import league_registry as LR
    for tbl in (config.LEAGUE_EDGE_CEILING, config.LEAGUE_SNIPER_THRESHOLDS,
                config.LEAGUE_MARKSMAN_THRESHOLDS, config.ARTIFICIAL_PITCH_LEAGUES):
        assert all(LR.canonical_name(k) == k for k in tbl)
    with pytest.raises(LR.RegistryError):
        config._registry_keyed({"Narnia Premier": 0.1}, "TEST")


# ── auto-onboarded league: a timezone is required, never "UTC" ──────────────────────────────
def test_auto_league_without_timezone_is_skipped(tmp_path, monkeypatch, caplog):
    f = tmp_path / "auto_leagues.json"
    base = {"api_football_id": 999001, "current_season": "2026", "odds_api_sport_key": "soccer_x"}
    f.write_text(json.dumps({"leagues": [{**base, "name": "Test NoTZ League"},
                                         {**base, "name": "Test TZ League", "api_football_id": 999002,
                                          "timezone": "Europe/Oslo"}]}), encoding="utf-8")
    monkeypatch.setattr(config, "AUTO_LEAGUES_FILE", f)
    for attr in ("NEW_FORMAT_LEAGUES", "API_FOOTBALL_ONLY_LEAGUES", "ENABLED_LEAGUES"):
        monkeypatch.setattr(config, attr, set(getattr(config, attr)))
    for attr in ("API_FOOTBALL_EXTRA_SEASONS", "API_FOOTBALL_IDS", "API_FOOTBALL_SEASONS",
                 "ODDS_API_SPORT_KEYS", "LEAGUE_TIMEZONES"):
        monkeypatch.setattr(config, attr, dict(getattr(config, attr)))
    with caplog.at_level("WARNING"):
        added = config._merge_auto_leagues()
    assert added == ["Test TZ League"] and config.LEAGUE_TIMEZONES["Test TZ League"] == "Europe/Oslo"
    assert "Test NoTZ League" not in config.LEAGUE_TIMEZONES and "no timezone" in caplog.text


# ── side markets: no empty side ─────────────────────────────────────────────────────────────
def test_side_market_without_a_defined_side_is_skipped():
    import pipeline
    preds = pd.DataFrame([{"date": "2026-10-23", "league": "Brazil Serie A", "home_team": "A",
                           "away_team": "B", "p_btts2": 0.7, "odds_btts2": 2.0}])
    assert pipeline._generate_side_bets(preds, {"btts2": None}).empty


# ── repaired Standard payload must STATE what it is ─────────────────────────────────────────
def _payload(**research):
    rs = {"track": "standard", "target": "over25", "price_mode": "none", "team_scope": "league"}
    rs.update(research)
    return {"models": {"m": 1}, "feature_cols": ["x"], "imputer": {"v": 1}, "recipe": "v9_repaired",
            "target": "over25", "research": {k: v for k, v in rs.items() if v is not None},
            "identity_alias_map": {}}


def test_repaired_payload_with_every_fact_validates():
    from src.standard_serving import validate_payload
    validate_payload(_payload())


@pytest.mark.parametrize("missing", ["target", "price_mode", "team_scope"])
def test_repaired_payload_missing_a_fact_is_refused(missing):
    from src.standard_serving import validate_payload
    with pytest.raises(ValueError):
        validate_payload(_payload(**{missing: None}))


# ── allocation: no price -> no Kelly size (no assumed even money) ───────────────────────────
def test_allocation_cell_without_odds_is_not_sized():
    from src import allocation as A
    n = max(A.MIN_N, 10) + 20
    d = pd.DataFrame({"match_date": pd.date_range("2026-08-10", periods=n, freq="D", tz="UTC"),
                      "pnl": np.where(np.arange(n) % 4 == 0, -1.0, 1.0), "odds": np.nan,
                      "result": np.where(np.arange(n) % 4 == 0, "LOSS", "WIN")})
    c = A._cell(d, "Championship", "ou25", "standard")
    assert c.fraction == 0.0 and c.state != A.ALLOCATABLE


# ── legacy trainer: a weight/row mismatch is an error, not uniform weights ───────────────────
def test_legacy_train_refuses_mismatched_weights():
    from src import model as M
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "x": rng.normal(size=n),
                       "over25": rng.integers(0, 2, n).astype(float)})
    with pytest.raises(ValueError):
        M.train(df, "over25", sample_weight=np.ones(n - 1), feature_cols=["x"])
