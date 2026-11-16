# Defaults reconciliation, props and fantasy (P0 "real data only", 2026-11-15)

## The real count today

`config/data_truth_baseline.json` holds **0 entries** at `fix/props-data-audit`. The 35 entries the
owner refers to (27 in `player_model/feature_engineering.py`, 3 in `fantasy.py`, 5 in
`league_quality.py`) were in the baseline until commit `0704142d9` (the data-truth-100 merge of the
7 v9 fix branches: "known-defaults baseline 35 -> 0, ratchet ceiling 0"). The list below is
recovered from that commit's parent (`git show 0704142d9^1:config/data_truth_baseline.json`). Every
one of the 35 code sites was checked against today's source: **0 remain** (the exact line is gone
from the file). Data truth 2 on this branch: `0 new default(s); 0 known (baseline); 193 allow-listed
(counters/display)`, `STOPLIGHT GREEN`. None of the 193 allow-list entries is in
feature_engineering.py, fantasy.py or league_quality.py. No entry was added to the baseline or the
allow-list by this work.

Classes: OBSERVED_ZERO / LEARNED_IMPUTATION / VALID_DOMAIN_CONSTANT / MODEL_PRIOR /
DISPLAY_FALLBACK / FABRICATED_OBSERVATION / UNKNOWN.

## The 35 historic entries

| # | file | code (historic) | class | replaced by (today) | removed in |
|---|---|---|---|---|---|
| 1 | feature_engineering | `rating ... .fillna(7.0)` | FABRICATED_OBSERVATION | rating_pg = mean of RATED earlier matches (rating > 0), NaN without one | 0ba311cd3 |
| 2 | feature_engineering | `rest_days ... .fillna(6.0).clip(1, 30)` | FABRICATED_OBSERVATION | `rest_days_before`: days since the last PLAYED day in any competition held, NaN without one, uncapped | 0ba311cd3 |
| 3 | feature_engineering | `_pen_conv = scored/total .fillna(0.75)` | FABRICATED_OBSERVATION | `_penalty_asof`: own record (season profile + per-match), NaN below MIN_PEN_ATTEMPTS | 0ba311cd3 |
| 4 | feature_engineering | `opp_goals_conceded_pg.fillna(1.3)` | FABRICATED_OBSERVATION | `_TEAM_ROLL` measured rolling, NaN below MIN_OPP_MATCHES | 0ba311cd3 |
| 5 | feature_engineering | `opp_sot_conceded_pg.fillna(4.5)` | FABRICATED_OBSERVATION | measured rolling, NaN below the minimum | 0ba311cd3 |
| 6 | feature_engineering | `team_goals_pg_roll.fillna(1.3)` | FABRICATED_OBSERVATION | measured own-team rolling (side fixed 2026-11-11) | 0ba311cd3 |
| 7 | feature_engineering | `... .fillna(0.50)` (defender duel rate) | FABRICATED_OBSERVATION | `team_match_table` def_aerial = duels won / total of measured defenders | 0ba311cd3 |
| 8 | feature_engineering | `opp_def_aerial_win_rate.fillna(0.50)` | FABRICATED_OBSERVATION | measured rolling, NaN below the minimum | 0ba311cd3 |
| 9 | feature_engineering | `opp_def_fouls_pg.fillna(1.50)` | FABRICATED_OBSERVATION | measured rolling | 0ba311cd3 |
| 10 | feature_engineering | `opp_def_cards_pg.fillna(0.25)` | FABRICATED_OBSERVATION | measured rolling | 0ba311cd3 |
| 11 | feature_engineering | `opp_def_mean_height.fillna(181.0)` | FABRICATED_OBSERVATION | mean of the defenders' COLLECTED heights, NaN when none known | 0ba311cd3 |
| 12 | feature_engineering | `height_cm.fillna(180.0) - opp_def_mean_height` | FABRICATED_OBSERVATION | collected profile height, NaN when unprofiled | 0ba311cd3 |
| 13 | feature_engineering | `opp_sp_goals_conceded_pg.fillna(0.12)` | FABRICATED_OBSERVATION | feature removed (UNSUPPORTED_FEATURES_REMOVED.md) | 36959d8df |
| 14 | feature_engineering | `... .fillna(1.3)` (league goals mean) | FABRICATED_OBSERVATION | `_league_asof`: the league's own prior mean, NaN below MIN_LEAGUE_MATCHES | 0ba311cd3 |
| 15 | feature_engineering | `_lg_mean.fillna(1.3).replace(0, 1.3)` | FABRICATED_OBSERVATION | as 14 | 0ba311cd3 |
| 16 | feature_engineering | `_opp_conc_roll.fillna(1.3) / _lg_mean` | FABRICATED_OBSERVATION | opp_strength_index = measured / measured, NaN otherwise | 0ba311cd3 |
| 17 | feature_engineering | `opp_gk_save_rate.fillna(0.72)` | FABRICATED_OBSERVATION | measured keeper rolling (goals_conceded checked against the score, 2026-11-11) | 0ba311cd3 |
| 18 | feature_engineering | `opp_gk_saves_pg.fillna(3.0)` | FABRICATED_OBSERVATION | measured rolling | 0ba311cd3 |
| 19 | feature_engineering | `opp_mid_fouls_pg ... .fillna(2.0)` | FABRICATED_OBSERVATION | opponent's OWN midfielders, measured | 0ba311cd3 |
| 20 | feature_engineering | `df.get("opp_mid_cards_pg", 0.4) ... .fillna(0.4)` | FABRICATED_OBSERVATION | measured | 0ba311cd3 |
| 21 | feature_engineering | `opp_sp_goals_conceded_pg.fillna(0.12) / 0.12` | FABRICATED_OBSERVATION | removed with the set-piece features | 36959d8df |
| 22 | feature_engineering | `_aer = ...get(..., 0.50).fillna(0.50)` | FABRICATED_OBSERVATION | opp_def_player_quality = measured ratios vs the league's measured means | 0ba311cd3 |
| 23 | feature_engineering | `_rat = ...get(..., 6.8).fillna(6.8)` | FABRICATED_OBSERVATION | as 22 (lg_def_rating measured) | 0ba311cd3 |
| 24 | feature_engineering | `_top = ...get(..., 7.0).fillna(7.0)` | FABRICATED_OBSERVATION | as 22 | 0ba311cd3 |
| 25 | feature_engineering | `_car = ...get("player_career_avg_quality", 0.65).fillna(0.65)` | FABRICATED_OBSERVATION | legacy quality columns NaN by construction, out of the feature list (LEGACY_QUALITY_FEATURES_REMOVED) | 0ba311cd3 |
| 26 | feature_engineering | `penalty_conversion_rate = (...).fillna(0.75).mean()` (serving) | FABRICATED_OBSERVATION | serving = build_features itself (2026-11-09), the same `_penalty_asof` | 0ba311cd3 |
| 27 | feature_engineering | `_awr = won / total.replace(0, nan) .fillna(0.5)` | FABRICATED_OBSERVATION | measured, NaN when no duel recorded | 0ba311cd3 |
| 28 | fantasy | `minutes_pg ... .fillna(60)` | FABRICATED_OBSERVATION | `mins = to_numeric(minutes_pg)`: NaN stays NaN (fantasy.py:447, 468) | 4edc1c4b5 |
| 29 | fantasy | `position.map(FPL_GOAL_PTS).fillna(5.0)` | FABRICATED_OBSERVATION | `goal_pts = position.map(FPL_GOAL_PTS)` (fantasy.py:559): unknown position -> NaN, no attack points | 4edc1c4b5 |
| 30 | fantasy | `p_start = (0.6·sr + 0.4·ssr).clip(0,1).fillna(0.6)` | FABRICATED_OBSERVATION (the 0.6 fill) | the fill is gone (fantasy.py:649). **The 0.6 / 0.4 blend weights remain: MODEL_PRIOR**, documented here; no observation is invented (each input fills only from the other measured rate) | 4edc1c4b5 |
| 31 | league_quality | `expanding().mean().shift(1).fillna(0.65)` | FABRICATED_OBSERVATION | measured career strength (Elo odds), NaN without history | 0ba311cd3 |
| 32 | league_quality | `rating ... .fillna(6.8)` | FABRICATED_OBSERVATION | unrated -> excluded | 0ba311cd3 |
| 33 | league_quality | `opp_def_player_rating_pg.fillna(6.8)` | FABRICATED_OBSERVATION | measured rolling | 0ba311cd3 |
| 34 | league_quality | `opp_top_def_rating.fillna(7.0)` | FABRICATED_OBSERVATION | measured rolling | 0ba311cd3 |
| 35 | league_quality | `_aerial.fillna(0.50) / 0.50` | FABRICATED_OBSERVATION | measured / league measured | 0ba311cd3 |

Result: 34 FABRICATED_OBSERVATION removed (value NaN), 1 FABRICATED fill removed with a MODEL_PRIOR
blend weight remaining (#30). UNKNOWN: 0.

## Where a fabricated value still reached a model, and what was done (2026-11-15)

Removing a default from the code does not clean an existing artifact:

* **Legacy payloads goals2 / goals3 / sot3 / sot4** were trained before 2026-11-01 on frames that held
  these defaults, and scored live with batch medians. They are now **refused** (no probability):
  they list fabricated set-piece features (`c41165a49`). They were already hidden.
* **Promoted clean models (sot, sot2, goals, assists, cards)** were trained on 2026-11-11 (run
  43035218594, code 6fd79496) by the clean harness, which builds every feature from raw rows with the
  code after `0ba311cd3`. They did not see a single one of the 35 defaults. They did see the four
  data defects fixed on this branch (provider-drifted `started`, stale /sidelined injuries, the shots
  definition change, dribbles_past / penalty_won null read as 0). That is why they are compared with a
  clean candidate in CLEAN_MODEL_BACKTEST.md.
* **Live batch median**: every legacy payload filled missing cells with `X.median()` of the scoring
  batch until `c41165a49`. Nothing is filled now (PROPS_IMPUTER_MIGRATION.md).

## Remaining persisted imputation (LEARNED_IMPUTATION, allowed)

* cards champion (`recipe_retrained`): fit-block medians persisted in the artifact
  (FIT_BLOCK_MEDIAN_PERSISTED). Train-fitted, never refitted live. The next retrain serves
  `hgb_nan_native` for cards (plan.json `candidate_by_market`, `af27fa808`).
* R2 `PropsImputer` (research challengers only, refused today because they list set-piece features).

## Tests

`tests/test_props_real_data_only_2026_11_15.py::test_no_fillna_with_an_invented_value_in_feature_code`
(numeric fillna literals other than 0 in the feature and scoring code),
`::test_the_scans_catch_adversarial_code` (feeds `fillna(6.0)`, `fillna(1.3)`, `df[x] = 0.0` and
`X.fillna(X.median())` to the scans), Data truth 2 (`tests/test_data_truth_checks.py`,
`tests/test_fix9_defaults.py`).
