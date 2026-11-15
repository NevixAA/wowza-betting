# Feature provenance registry (REAL DATA ONLY, 2026-11-15)

Machine-readable source: `config/feature_provenance.json` (rebuild: `python -m src.integrity.real_data_provenance --write`). Every list below is read from the **serialized payloads** `src/predict.py` loads, not from config. `train cov` = share of the payload's base-training rows with a value (its own train-only imputer record). `serve cov (tree)` = share of the committed board's fixtures (`output/predictions.csv`, 2026-11-02 .. 2026-11-09, 174 fixtures). `serve cov (main)` = the same on origin/main's live board (2026-11-15 .. 2026-11-22, 128 fixtures, read 2026-11-15). `__missing` indicators are built at scoring time from their base column and are listed once per payload below.

Classes: SOURCE_VERIFIED / DERIVED_VERIFIED (inputs real, point-in-time, traceable) | VALID_MODEL_ESTIMATE (eligible: explicit train-median fill + learned missing indicator on most training rows) | MISSING_SOURCE | FABRICATED_VALUE | TIME_LEAKAGE | TRAIN_SERVE_MISMATCH | UNVERIFIED.

## Feature definitions (team)

| feature | source type | provider | raw columns | transformation | at |
|---|---|---|---|---|---|
| `home_scored_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `scored`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:140` |
| `home_conceded_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `conceded`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:141` |
| `home_over25_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `over25`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:142` |
| `home_corners_pg_roll` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_corners, away_corners | team-centric `corners`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:145` |
| `home_fouls_pg_roll` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_fouls, away_fouls | team-centric `fouls`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:146` |
| `home_xg_last5` | rolling team form | API-Football /fixtures/statistics (api_football cache + output/collect/af_history_stats_fill.parquet) | home_xg, away_xg | team-centric `xg`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:147` |
| `home_insidebox_last5` | rolling team form | API-Football /fixtures/statistics (api_football cache + output/collect/af_history_stats_fill.parquet) | home_insidebox, away_insidebox | team-centric `insidebox`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:148` |
| `home_ht_scored_last5` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_scored`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:149` |
| `home_ht_conceded_last5` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_conceded`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:150` |
| `home_ht_over05_rate` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_over05_flag`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:151` |
| `home_ht_over15_rate` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_over15_flag`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:152` |
| `home_has_ht_history` | data-availability indicator | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | 1.0 when the form window holds >= 1 real half-time score, else 0.0 (a fact about the window, not a stand-in value) | `src/feature_engineering.py:181` |
| `home_attack_str` | derived ratio | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | home_scored_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:560` |
| `home_defense_str` | derived ratio | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | home_conceded_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:562` |
| `home_ht_attack_str` | derived ratio | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | home_ht_scored_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:553` |
| `home_rest_days` | derived calendar | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date | days since the team's previous match in the same league (no previous -> NaN) | `src/feature_engineering.py:338` |
| `away_scored_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `scored`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:140` |
| `away_conceded_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `conceded`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:141` |
| `away_over25_last5` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | team-centric `over25`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:142` |
| `away_corners_pg_roll` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_corners, away_corners | team-centric `corners`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:145` |
| `away_fouls_pg_roll` | rolling team form | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_fouls, away_fouls | team-centric `fouls`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:146` |
| `away_xg_last5` | rolling team form | API-Football /fixtures/statistics (api_football cache + output/collect/af_history_stats_fill.parquet) | home_xg, away_xg | team-centric `xg`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:147` |
| `away_insidebox_last5` | rolling team form | API-Football /fixtures/statistics (api_football cache + output/collect/af_history_stats_fill.parquet) | home_insidebox, away_insidebox | team-centric `insidebox`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:148` |
| `away_ht_scored_last5` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_scored`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:149` |
| `away_ht_conceded_last5` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_conceded`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:150` |
| `away_ht_over05_rate` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_over05_flag`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:151` |
| `away_ht_over15_rate` | rolling team form | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | team-centric `ht_over15_flag`, shift(1), mean of the last ROLLING_N matches per (league, team) - _rolling / _latest_form (one function, train = serve) | `src/feature_engineering.py:152` |
| `away_has_ht_history` | data-availability indicator | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | 1.0 when the form window holds >= 1 real half-time score, else 0.0 (a fact about the window, not a stand-in value) | `src/feature_engineering.py:181` |
| `away_attack_str` | derived ratio | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | away_scored_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:561` |
| `away_defense_str` | derived ratio | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | away_conceded_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:563` |
| `away_ht_attack_str` | derived ratio | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | away_ht_scored_last5 / (league average goals on strictly earlier dates / 2) | `src/feature_engineering.py:554` |
| `away_rest_days` | derived calendar | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date | days since the team's previous match in the same league (no previous -> NaN) | `src/feature_engineering.py:338` |
| `home_season_goals_h` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | home matches of the fixture's (league, derived season): expanding mean of scored (home_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:383` |
| `home_season_conceded_h` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | home matches of the fixture's (league, derived season): expanding mean of conceded (away_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:384` |
| `home_cs_rate_h` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | home matches of the fixture's (league, derived season): expanding mean of clean sheets (away_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:385` |
| `away_season_goals_a` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | away matches of the fixture's (league, derived season): expanding mean of scored (away_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:389` |
| `away_season_conceded_a` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | away matches of the fixture's (league, derived season): expanding mean of conceded (home_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:390` |
| `away_cs_rate_a` | season-to-date venue stat | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_goals, away_goals | away matches of the fixture's (league, derived season): expanding mean of clean sheets (home_goals), shift(1). Serving: the same history rows; API-Football /teams/statistics is display-only (af_* columns) since 2026-11-15 | `src/feature_engineering.py:391` |
| `combined_sot_ratio` | derived ratio | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_shots, away_shots, home_sot, away_sot | mean of each side's rolling SoT / rolling shots (no shot history -> NaN) | `src/feature_engineering.py:584` |
| `combined_ht_goals_avg` | derived sum | football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader) | ht_home_goals, ht_away_goals | home_ht_scored_last5 + away_ht_scored_last5 (either unknown -> NaN) | `src/feature_engineering.py:551` |
| `p_over25_poisson_dc` | derived model estimate | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | home_goals, away_goals | Dixon-Coles P(over 2.5) from the two rolling goals-for rates as lambdas, each clipped to [0.3, 4.0] (no form -> NaN, never the market price) | `src/feature_engineering.py:608` |
| `home_advantage` | derived rate | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_team, home_goals, away_goals | share of the home team's last ROLLING_N home matches in which it scored more than the league's average away goals on dates strictly before THAT match (training's rule; serving reads the same flags since 2026-11-15) | `src/feature_engineering.py:319` |
| `away_home_adv_factor` | derived rate | football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format) | date, league, home_team, home_goals, away_goals | share of the away team's last ROLLING_N home matches in which it scored more than the league's average away goals on dates strictly before THAT match (training's rule; serving reads the same flags since 2026-11-15) | `src/feature_engineering.py:319` |

Point-in-time: every definition above is verified by `tests/test_real_data_pipeline_trace.py` (`test_training_features_are_point_in_time`: rewriting later results never moves an earlier row; `test_serving_equals_training_for_the_next_match`: serving == training for the next match on 12 drifting leagues).

## Standard O/U 2.5 [standard] - active

`models/model_v9_standard.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.1%: the other rows are the explicit train-median fill + a learned missing indicator | 17.1% | 98.6% | 98.6% | train-only median 1.226 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.0%: the other rows are the explicit train-median fill + a learned missing indicator | 17.0% | 98.6% | 97.2% | train-only median 1.272 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 100.0% | train-only median 7.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 97.2% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949344 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.969006 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.982979 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.964706 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 0.347804 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.420529 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.429243 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 52.7% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 98.7% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.5% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.111111 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.409091 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 0.3125 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 0.222222 + `__missing` indicator, clipped to the training range. | yes |

## Standard O/U 2.5 (repaired rollback) [standard] - standby (rollback)

`models/model_v9_standard_repaired.pkl`: 41 features (41 base + 0 `__missing` indicators, DERIVED_VERIFIED). **Serving guard: 1.0 imputer fills 4 feature(s) with no training values with the constant 0.0 (home_xg_last5, away_xg_last5, home_insidebox_last5, away_insidebox_last5)**

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | NO |
| `away_scored_last5` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | NO |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | NO |
| `away_conceded_last5` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | NO |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `away_over25_last5` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.95374 + `__missing` indicator, clipped to the training range. | NO |
| `away_attack_str` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 0.971193 + `__missing` indicator, clipped to the training range. | NO |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.985714 + `__missing` indicator, clipped to the training range. | NO |
| `away_defense_str` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 0.966927 + `__missing` indicator, clipped to the training range. | NO |
| `home_advantage` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `away_home_adv_factor` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | NO |
| `away_rest_days` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | NO |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.9% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | NO |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.4% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | NO |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.9% | 98.6% | 98.6% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | NO |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.4% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | NO |
| `combined_sot_ratio` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.348672 + `__missing` indicator, clipped to the training range. | NO |
| `home_xg_last5` | FABRICATED_VALUE - no training values: filled with the constant 0.0 on every training row and at serving | 0.0% | 98.6% | 98.6% | constant 0.0 (1.0 imputer) | NO |
| `away_xg_last5` | FABRICATED_VALUE - no training values: filled with the constant 0.0 on every training row and at serving | 0.0% | 98.6% | 97.2% | constant 0.0 (1.0 imputer) | NO |
| `home_insidebox_last5` | FABRICATED_VALUE - no training values: filled with the constant 0.0 on every training row and at serving | 0.0% | 98.6% | 100.0% | constant 0.0 (1.0 imputer) | NO |
| `away_insidebox_last5` | FABRICATED_VALUE - no training values: filled with the constant 0.0 on every training row and at serving | 0.0% | 98.6% | 97.2% | constant 0.0 (1.0 imputer) | NO |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.388889 + `__missing` indicator, clipped to the training range. | NO |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.117647 + `__missing` indicator, clipped to the training range. | NO |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.133333 + `__missing` indicator, clipped to the training range. | NO |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.409091 + `__missing` indicator, clipped to the training range. | NO |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 0.307692 + `__missing` indicator, clipped to the training range. | NO |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 0.222222 + `__missing` indicator, clipped to the training range. | NO |
| `home_ht_scored_last5` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | NO |
| `away_ht_scored_last5` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | NO |
| `home_ht_conceded_last5` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | NO |
| `away_ht_conceded_last5` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | NO |
| `home_ht_attack_str` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 98.6% | 98.6% | train-only median 0.426493 + `__missing` indicator, clipped to the training range. | NO |
| `away_ht_attack_str` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 97.2% | 97.2% | train-only median 0.445733 + `__missing` indicator, clipped to the training range. | NO |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 100.0% | 95.8% | 95.8% | train-only median 0.0 + `__missing` indicator, clipped to the training range. | NO |
| `home_ht_over05_rate` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | NO |
| `away_ht_over05_rate` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | NO |
| `home_ht_over15_rate` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `away_ht_over15_rate` | VALID_MODEL_ESTIMATE - training coverage 37.6%: the other rows are the explicit train-median fill + a learned missing indicator | 37.6% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | NO |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 99.4% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | NO |

## New Format O/U 2.5 [new_format] - active

`models/model_v9_newformat.pkl`: 56 features (28 base + 28 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.5 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.959064 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.972182 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.974545 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.95743 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 94.3% | 95.1% | 100.0% | train-only median 0.342199 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 95.1% | 100.0% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.0% | 98.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.6% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.076923 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.071429 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 0.307692 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 0.2 + `__missing` indicator, clipped to the training range. | yes |

## BTTS [standard] - active

`models/model_v9_btts.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.1%: the other rows are the explicit train-median fill + a learned missing indicator | 17.1% | 98.6% | 98.6% | train-only median 1.226 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.0%: the other rows are the explicit train-median fill + a learned missing indicator | 17.0% | 98.6% | 97.2% | train-only median 1.272 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 100.0% | train-only median 7.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 97.2% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949344 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.969006 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.982979 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.964706 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 0.347804 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.420529 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.429243 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 52.7% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 98.7% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.5% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.111111 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.409091 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 0.3125 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 0.222222 + `__missing` indicator, clipped to the training range. | yes |

## BTTS [new_format] - active

`models/model_v9_btts.pkl`: 56 features (28 base + 28 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.5 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.959064 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.972182 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.974545 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.95743 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 94.3% | 95.1% | 100.0% | train-only median 0.342199 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 95.1% | 100.0% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.0% | 98.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.6% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.076923 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.071429 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 0.307692 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 0.2 + `__missing` indicator, clipped to the training range. | yes |

## Over 1.5 [standard] - active

`models/model_v9_over15.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.1%: the other rows are the explicit train-median fill + a learned missing indicator | 17.1% | 98.6% | 98.6% | train-only median 1.226 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.0%: the other rows are the explicit train-median fill + a learned missing indicator | 17.0% | 98.6% | 97.2% | train-only median 1.272 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 100.0% | train-only median 7.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 97.2% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949344 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.969006 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.982979 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.964706 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 0.347804 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.420529 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.429243 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 52.7% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 98.7% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.5% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.111111 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.409091 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 0.3125 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 0.222222 + `__missing` indicator, clipped to the training range. | yes |

## Over 1.5 [new_format] - active

`models/model_v9_over15.pkl`: 56 features (28 base + 28 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.5 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.959064 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.972182 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.974545 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.95743 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 94.3% | 95.1% | 100.0% | train-only median 0.342199 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 95.1% | 100.0% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.0% | 98.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.6% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.076923 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.071429 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 0.307692 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 0.2 + `__missing` indicator, clipped to the training range. | yes |

## Over 3.5 [standard] - active

`models/model_v9_over35.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.8% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.1%: the other rows are the explicit train-median fill + a learned missing indicator | 17.1% | 98.6% | 98.6% | train-only median 1.226 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 17.0%: the other rows are the explicit train-median fill + a learned missing indicator | 17.0% | 98.6% | 97.2% | train-only median 1.272 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 100.0% | train-only median 7.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.5% | 98.6% | 97.2% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949344 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.969006 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.982979 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.964706 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.8% | 98.6% | 98.6% | train-only median 0.347804 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 98.6% | 98.6% | train-only median 0.420529 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 52.8% | 97.2% | 97.2% | train-only median 0.429243 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 52.7% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 98.7% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.5% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.111111 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 1.409091 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.4% | 100.0% | 100.0% | train-only median 0.3125 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.4% | 98.6% | 98.6% | train-only median 0.222222 + `__missing` indicator, clipped to the training range. | yes |

## Over 3.5 [new_format] - active

`models/model_v9_over35.pkl`: 56 features (28 base + 28 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.5 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 94.8% | 92.2% | 76.8% (GAP) | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 94.7% | 92.2% | 83.9% | train-only median 12.2 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.959064 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.972182 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 0.974545 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 0.95743 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 94.3% | 95.1% | 100.0% | train-only median 0.342199 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 95.1% | 100.0% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.0% | 98.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.6% | 97.1% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 98.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 97.1% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.076923 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 1.071429 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 94.2% | 98.0% | 100.0% | train-only median 0.307692 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 94.1% | 97.1% | 100.0% | train-only median 0.2 + `__missing` indicator, clipped to the training range. | yes |

## HT Over 0.5 [standard] - active

`models/model_ht_over05.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.7% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.7% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 37.1%: the other rows are the explicit train-median fill + a learned missing indicator | 37.1% | 98.6% | 98.6% | train-only median 1.252 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 37.1%: the other rows are the explicit train-median fill + a learned missing indicator | 37.1% | 98.6% | 97.2% | train-only median 1.29 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.7% | 98.6% | 100.0% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.7% | 98.6% | 97.2% | train-only median 8.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949348 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.974291 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.987026 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.970906 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 0.347551 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.424454 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.429526 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 99.1% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.6% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.9% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.142857 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.428571 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 0.3 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 0.214286 + `__missing` indicator, clipped to the training range. | yes |

## HT Over 1.5 [standard] - active

`models/model_ht_over15.pkl`: 86 features (43 base + 43 `__missing` indicators, DERIVED_VERIFIED).

| feature | class | train cov | serve cov (tree) | serve cov (main) | missing policy | eligible |
|---|---|---|---|---|---|---|
| `home_scored_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_scored_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_conceded_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `away_conceded_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 1.2 + `__missing` indicator, clipped to the training range. REQUIRED: a fixture missing it gets no probability at serving | yes |
| `home_over25_last5` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_over25_last5` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_corners_pg_roll` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_corners_pg_roll` | DERIVED_VERIFIED | 90.7% | 97.2% | 97.2% | train-only median 4.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_fouls_pg_roll` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 12.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_fouls_pg_roll` | DERIVED_VERIFIED | 90.7% | 97.2% | 97.2% | train-only median 11.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 37.1%: the other rows are the explicit train-median fill + a learned missing indicator | 37.1% | 98.6% | 98.6% | train-only median 1.252 + `__missing` indicator, clipped to the training range. | yes |
| `away_xg_last5` | VALID_MODEL_ESTIMATE - training coverage 37.1%: the other rows are the explicit train-median fill + a learned missing indicator | 37.1% | 98.6% | 97.2% | train-only median 1.29 + `__missing` indicator, clipped to the training range. | yes |
| `home_insidebox_last5` | DERIVED_VERIFIED | 81.7% | 98.6% | 100.0% | train-only median 7.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_insidebox_last5` | DERIVED_VERIFIED | 81.7% | 98.6% | 97.2% | train-only median 8.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_scored_last5` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_scored_last5` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_conceded_last5` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_conceded_last5` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.6 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over05_rate` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over05_rate` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.8 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_over15_rate` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_over15_rate` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_has_ht_history` | DERIVED_VERIFIED | 100.0% | - | - | train-only median 1.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_attack_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.949348 + `__missing` indicator, clipped to the training range. | yes |
| `away_attack_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.974291 + `__missing` indicator, clipped to the training range. | yes |
| `home_defense_str` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 0.987026 + `__missing` indicator, clipped to the training range. | yes |
| `away_defense_str` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.970906 + `__missing` indicator, clipped to the training range. | yes |
| `combined_sot_ratio` | DERIVED_VERIFIED | 90.7% | 98.6% | 98.6% | train-only median 0.347551 + `__missing` indicator, clipped to the training range. | yes |
| `home_ht_attack_str` | DERIVED_VERIFIED | 99.2% | 98.6% | 98.6% | train-only median 0.424454 + `__missing` indicator, clipped to the training range. | yes |
| `away_ht_attack_str` | DERIVED_VERIFIED | 99.2% | 97.2% | 97.2% | train-only median 0.429526 + `__missing` indicator, clipped to the training range. | yes |
| `combined_ht_goals_avg` | DERIVED_VERIFIED | 99.1% | 95.8% | 95.8% | train-only median 1.2 + `__missing` indicator, clipped to the training range. | yes |
| `p_over25_poisson_dc` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 0.48157 + `__missing` indicator, clipped to the training range. | yes |
| `home_advantage` | DERIVED_VERIFIED | 99.6% | 100.0% | 100.0% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_home_adv_factor` | DERIVED_VERIFIED | 99.9% | 98.6% | 98.6% | train-only median 0.4 + `__missing` indicator, clipped to the training range. | yes |
| `home_rest_days` | DERIVED_VERIFIED | 100.0% | 100.0% | 100.0% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `away_rest_days` | DERIVED_VERIFIED | 100.0% | 98.6% | 98.6% | train-only median 7.0 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_goals_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.4 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_goals_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.125 + `__missing` indicator, clipped to the training range. | yes |
| `home_season_conceded_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 1.142857 + `__missing` indicator, clipped to the training range. | yes |
| `away_season_conceded_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 1.428571 + `__missing` indicator, clipped to the training range. | yes |
| `home_cs_rate_h` | DERIVED_VERIFIED | 95.0% | 100.0% | 100.0% | train-only median 0.3 + `__missing` indicator, clipped to the training range. | yes |
| `away_cs_rate_a` | DERIVED_VERIFIED | 95.0% | 98.6% | 98.6% | train-only median 0.214286 + `__missing` indicator, clipped to the training range. | yes |

## Fantasy inputs

No fantasy model is serialized: V2 fits `LinearResidual` + `MinutesModel` at run time on FINAL pre-deadline FPL snapshot gameweeks; V1 is formula-based on the props payloads.

| input | source | at | class | missing policy |
|---|---|---|---|---|
| fpl_ep_next / price / form / ownership | FPL bootstrap-static | `player_model/fpl_api.py:226` | SOURCE_VERIFIED | NaN |
| xg90 / xa90 / bps90 (V2) | FPL snapshot, shrunk to the same-snapshot position rate | `player_model/fantasy_panel.py:279` | VALID_MODEL_ESTIMATE | position with no minutes -> NaN |
| team gf / ga / xg / xga (V2) | FPL snapshot team totals, 4-game prior to the league mean | `player_model/fantasy_panel.py:111` | VALID_MODEL_ESTIMATE | a team with games but no xG stays NaN (no league-mean fill, 2026-11-15) |
| P(0) / P(1-59) / P(60) (V2 MinutesModel) | fitted at run time on FINAL pre-deadline gameweeks | `player_model/fantasy_residual.py:626` | VALID_MODEL_ESTIMATE | NaN-aware boosting |
| residual training rows (V2) | FINAL gameweeks only, as the walk-forward validates (2026-11-15) | `player_model/fantasy_v2.py:151` | DERIVED_VERIFIED | - |
| xpts_3gw / xpts_5gw (V2) | FPL fixtures + ep_next | `player_model/fantasy_v2.py:266` | VALID_MODEL_ESTIMATE | no fixture list for the club, or negative ep_next -> NaN (2026-11-15) |
| p_goal / p_assist / p_sot2 (V1) | props payloads models/model_player_{goals,assists,sot2}.pkl | `player_model/fantasy.py:458` | UNVERIFIED | props provenance PENDING (fix/props-data-audit) |
| def_actions_pg (V1 bonus input) | API-Football tackles + interceptions + duels won | `player_model/fantasy.py:228` | DERIVED_VERIFIED | any missing component -> NaN (2026-11-15; was read as 0) |
| injured / doubtful (V1) | FPL bootstrap status via output/fpl_af_crosswalk.csv | `player_model/fantasy.py:496` | SOURCE_VERIFIED | unmatched player -> unknown (NA), no match at all -> no board (2026-11-15) |
| fixture_adj_pts (V1) | FPL FDR + home flag | `player_model/fantasy.py:719` | VALID_MODEL_ESTIMATE | unknown FDR -> NaN (2026-11-15; was x1.0) |

Unfitted coefficients (conventions, labelled on the page; not observations): V1 FDR->xG `0.5 + 0.42*FDR`, CS blend 0.6/0.4, band 0.3-3.5, bonus weights x0.66, SoT2 weight 0.45, FDR step 0.10, home edge 0.05, appearance 2-if-minutes>=60, P(start) 0.6/0.4 blend; V2 xG/goals 0.5/0.5 blend, clips, position shares, `start_rate + 0.25` long-run play rate, half-life 1 GW. Each is a candidate for a fit on the settled FPL ledger (see the verdict).

