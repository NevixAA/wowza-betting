# Final data truth certification (owner deep-repair brief 2026-11-22, items 15-16)

Owner priority, 2026-11-23: **"the most important achievement is 100% real data for every row in our data set,
for all the models."** This certification is built around that sentence. A cell says CERTIFIED only where a
committed artifact proves it; a scope that was not measured says UNVERIFIED. No price, odds, probability or stake
value appears here.

| | |
|---|---|
| Repo / base | NevixAA/wowza-betting, main @ `6de281c97` (merged into this branch; payload parity + full suite measured on `5a6727f42`, no model / src / config change between them except #86) (inputs merged: #74, #75, #76, #78, #81, #82, #83, #84, #85, #86) |
| Other repos (main) | wowzaV9-Pro `17d6faabc` · wowza_v11 `4b336541f` · wowza-dashboard `eda50a6b2` · wowza-exec `82f6d7d9f` |
| Row-level evidence | `output/real_data_certification.json` (`scripts/certify_real_data.py`, generated `2026-11-23T10:51:17Z` on `38ee262ac` (branch merge of main `6de281c97`)) |
| Companion audit | `reports/POST_GOLIVE_DEEP_AUDIT.md` (defects, per-repo health, the 12 closing answers) |
| Evidence notation | `path@blob` = git blob (first 9 hex) of that file on the PR head; Pro / v11 files cite their repo |

## 1. Verdict on the owner's priority

**100% real data for every row: NOT ACHIEVED.** It is achieved for most scopes and blocked in two by named rows.

| Scope | 100% real? | What blocks it (rows) |
|---|---|---|
| History banks (`fd_history` 66,484 rows, `af_history` 93,553 rows) | **YES** - 976,839 provider-observed cells, 0 defaults; 1,504,923 cells missing and kept missing | - |
| Standard team training (O/U 2.5, BTTS, O1.5, O3.5 Standard part) | **NO** | 21 of 49,407 training rows: 2 rows come from `output/backtest_all_leagues.csv` (an internal backtest output, not a provider file) - Belgian First Division A 2022-10-23 Standard v Anderlecht and 2022-11-12 Charleroi v Mechelen; of their 16 stat cells 1 agrees with a provider, 7 contradict API-Football, 8 are held by no provider - plus 19 later rows of the same 4 clubs whose rolling windows read them (D1). 34 UNVERIFIED_VALUE cells per payload |
| Standard HT training (HT O0.5 / O1.5) | **NO** | the same 19 downstream rows (the 2 archive rows have no HT score, so they are not HT training rows); 19 UNVERIFIED_VALUE cells per payload |
| New-Format team training (all 6 NF parts) | **YES at cell level** (0 defaults, 0 unverified in the payload records) | row-level frame not saved (lineage keeps keys + hash `9ed237ebc04aee09`). The backtest archive holds **0** NF-league rows (17 leagues, all Standard: `backtest_archive_scope`), so D1 cannot reach NF |
| Serving board (`output/predictions.csv`, 161 fixtures) | **YES** - 0 defaults; 36 missing inputs kept missing | - |
| Props (966,274 player rows -> 13 certified payloads) | **YES** for every row a model trains on | 1,882 `zeros_unverified` rows are context only, never a label; 4,045 population rows fail `all_prior_verified` and are excluded. The STORED `player_history.parquet` still holds legacy placeholders age 25 / 180 cm / 75 kg / age_peak_delta 0 on ~888k of 891,912 rows; no model reads them (D4) |
| Corners / cards side models | **YES** for served leagues (learning ledger only) | single-source leagues verified by a sanity rule, not a second provider (17,043 fixtures per family); 908 corners / 798 cards fixtures unverified -> NaN, never a training row |
| Fantasy inputs | **YES** for the FPL inputs (2,631 of 2,631 FINAL player-gameweeks = FPL `event/live`) | 21 DOMAIN_PRIOR constants (3 marked FIXED) in the projection formula (conventions, not observations); 0 open MISSING_DATA_FALLBACK reachable (`props_batch_median` unreachable per the props certification); V1 props-frame parity not certified |
| Ledgers (bets, side, HT, player, TOP-3, v91 shadow, prediction ledger) | **YES** - 0 synthetic labels, 0 prices <= 1 | 69 HT and 39,953 player rows carry no price (kept missing, never a bet); 96 of 351,882 prediction-ledger rows unscored, every one with its reason; bookmaker + capture time not recorded on team / player ledgers |
| Pro research, v11 research, exec stores | **UNVERIFIED** (row level not measured here) | reconciliation evidence only (identity, schema, chronology) |

## 2. Method

`scripts/certify_real_data.py` (read-only, counts only) puts every value of every dataset that feeds a model into
exactly one class:

| Class | Meaning |
|---|---|
| PROVIDER_OBSERVED | a provider's own number (football-data, API-Football, FPL, OddsAPI) |
| VERIFIED_DERIVED | computed from observed values by documented, tested code (rolling form, rates, `__missing` indicators, the provider null rule, a model probability on those inputs) |
| CORRECTED_BESIDE_ORIGINAL | a provider value replaced by a proven provider value; the original kept beside it |
| MISSING_KEPT_MISSING | no observation: NaN, the row excluded, or no probability |
| DEFAULT_OR_SYNTHETIC | any default, placeholder, constant fill or synthetic value stored or read as an observation |
| UNVERIFIED_VALUE | a value no provider confirms (none holds it, or one contradicts it) |

A scope is REAL_100 only with 0 DEFAULT_OR_SYNTHETIC and 0 UNVERIFIED_VALUE (`REAL_100_WITH_MISSING_KEPT_MISSING`
when some cells are honestly missing). Model-side fills (train-only medians, NaN-native trees, fitted base rates) are
**training-fitted imputations**, listed in section 5; they are never counted as data. Sources reused:
`src/integrity/real_data_provenance.py` (TEAM_FEATURES classes), the payloads' own imputer records,
`player_model/props_harness.py` (`source_frame`, `population`), `models/side_models/side_models.json`,
`player_model/fantasy_defaults.py`, `src/integrity/synthetic_roi.py` labels, `src/integrity/data_truth.py`
(serving inputs per track), `reports/lineage/training_lineage_evidence.json`.

Reproduce: `python scripts/certify_real_data.py --write --std-frame <golive/std_retrain_1611/feat_new.parquet>`
(the saved Standard training frame, hash `ebe47c041a7f595f`, kept locally since the 2026-11-19 install; without it
the Standard row-level block is skipped and only the payload records are counted). Tests:
`tests/test_certify_real_data.py`.

## 3. Row-level counts per dataset

| Dataset | Rows | obs | derived | corrected | missing | default/synthetic | unverified | Verdict |
|---|---|---|---|---|---|---|---|---|
| `output/fd_history.parquet` | 66,484 | 511,672 | 0 | 0 | 286,136 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| `output/af_history.parquet` | 93,553 | 465,167 | 0 | 0 | 1,218,787 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Standard O/U 2.5 [standard] `model_v9_standard.pkl` | 49,407 | 0 | 3,232,403 | 0 | 166,541 | 0 | 34 | NOT_REAL |
| New Format O/U 2.5 [new_format] `model_v9_newformat.pkl` | 26,846 | 0 | 1,189,065 | 0 | 13,535 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| BTTS [standard] `model_v9_btts.pkl` | 49,407 | 0 | 3,232,403 | 0 | 166,541 | 0 | 34 | NOT_REAL |
| BTTS [new_format] `model_v9_btts.pkl` | 26,846 | 0 | 1,189,065 | 0 | 13,535 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Over 1.5 [standard] `model_v9_over15.pkl` | 49,407 | 0 | 3,232,403 | 0 | 166,541 | 0 | 34 | NOT_REAL |
| Over 1.5 [new_format] `model_v9_over15.pkl` | 26,846 | 0 | 1,189,065 | 0 | 13,535 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Over 3.5 [standard] `model_v9_over35.pkl` | 49,407 | 0 | 3,232,403 | 0 | 166,541 | 0 | 34 | NOT_REAL |
| Over 3.5 [new_format] `model_v9_over35.pkl` | 26,846 | 0 | 1,189,065 | 0 | 13,535 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| HT Over 0.5 [standard] `model_ht_over05.pkl` | 49,256 | 0 | 3,143,681 | 0 | 165,900 | 0 | 19 | NOT_REAL |
| HT Over 1.5 [standard] `model_ht_over15.pkl` | 49,256 | 0 | 3,143,681 | 0 | 165,900 | 0 | 19 | NOT_REAL |
| New Format HT Over 0.5 [new_format] `model_v9_ht_over05_nf.pkl` | 26,759 | 0 | 1,182,828 | 0 | 13,444 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| New Format HT Over 1.5 [new_format] `model_v9_ht_over15_nf.pkl` | 26,759 | 0 | 1,182,828 | 0 | 13,444 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Serving board `output/predictions.csv` (all served inputs) | 161 | 0 | 35,372 | 0 | 36 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Props source rows (`props_harness.source_frame`) | 966,274 | 959,081 | 5,311 | 0 | 1,882 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Side model corners fixtures (18 leagues) | 30,966 | 30,058 | 0 | 0 | 908 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Side model cards fixtures (17 leagues) | 28,943 | 28,145 | 0 | 0 | 798 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| `bets_ledger.csv` (price presence) | 1,943 | 1,943 | 0 | 0 | 0 | 0 | 0 | REAL_100 |
| `side_bets_ledger.csv` (price presence) | 795 | 795 | 0 | 0 | 0 | 0 | 0 | REAL_100 |
| `ht_ledger.csv` (price presence) | 84 | 15 | 0 | 0 | 69 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| `player_ledger.csv` (price presence) | 47,426 | 7,473 | 0 | 0 | 39,953 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| `top3_league_ledger.csv` (price presence) | 3 | 3 | 0 | 0 | 0 | 0 | 0 | REAL_100 |
| `props_top3_ledger.csv` (price presence) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | REAL_100 |
| `v91_shadow.csv` (price presence) | 1,394 | 1,394 | 0 | 0 | 0 | 0 | 0 | REAL_100 |
| `prediction_ledger (v9, all partitions)` (probability) | 351,882 | 0 | 351,786 | 0 | 96 | 0 | 0 | REAL_100_WITH_MISSING_KEPT_MISSING |
| Fantasy (FPL inputs) | 2,959 OOS rows | see section 1 | | | | | | REAL_100_WITH_MISSING_KEPT_MISSING |
| wowzaV9-Pro (grid / research inputs) | not measured | | | | | | | UNVERIFIED |
| wowza_v11 (research inputs) | not measured | | | | | | | UNVERIFIED |
| wowza-exec (private price + ledger stores) | not measured | | | | | | | UNVERIFIED |

Standard O/U 2.5 training rows (saved frame `ebe47c041a7f595f`) by row source: football_data_raw 29,454, fd_cache 19,624, provider_supplement 327, backtest_archive 2. Backtest archive file: 40,060 rows, tracks {"standard": {"leagues": 17, "rows": 40060}}.

## 4. Every default / placeholder / synthetic / unverified value found

| # | Dataset | Column(s) | Rows / cells | Cause | Reaches a model? |
|---|---|---|---|---|---|
| D1 | Standard training frame (`row_source = backtest_archive`) | home/away shots, SOT, corners, fouls of 2 matches (Belgian First Division A 2022-10-23 Standard v Anderlecht; 2022-11-12 Charleroi v Mechelen) | 2 rows (16 stat cells: 1 agrees with a provider, 7 contradicted by API-Football, 8 held by no provider) + 19 downstream rows | `src/data_loader._load_backtest_archive` appends `output/backtest_all_leagues.csv`; for these two matches football-data has the score but no stats, so the archive row (with stats) survives the identity pipeline | **YES**: the 4 Standard O/U-family payloads (row + windows), the 2 Standard HT payloads (windows only). Serving today: no (2022 windows) |
| D4 | `player_history.parquet` (stored) | age / height_cm / weight_kg / age_peak_delta | 887,907 / 888,192 / 888,142 / 887,907 of 891,912 rows | legacy `build_features` placeholders written into the stored file | **NO**: `player_model/feature_engineering._PROFILE_PLACEHOLDERS` reads them as missing; profiles come from `output/collect/player_profiles.parquet`; props certification: no served feature has one value on > 99.9% of training rows |
| - | Team payloads | - | 0 FABRICATED_VALUE / 0 untraced features in the 12 active team payload parts | the retired `model_v9_standard_repaired.pkl` (4 fabricated features) is refused by `owner_install` and serves nothing | - |
| - | Ledgers | price columns | 0 synthetic labels, 0 prices <= 1 | `output/player_props_calibration.json` stays quarantined (INVALID_FOR_REAL_BETTING_ROI) | - |
| - | Fantasy | `props_batch_median` (register entry not marked FIXED) | 0 | legacy props `_prep`: unreachable in production per the props certification (AST call graph + runtime probe: 0 calls) | NO |

## 5. Training-fitted imputations (allowed: identified, versioned, validated)

| Scope | Imputation | Version | Cells filled (fit block) | Validation |
|---|---|---|---|---|
| Standard O/U 2.5 [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_standard.pkl) | 166,541 (28 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| New Format O/U 2.5 [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_newformat.pkl) | 13,535 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| BTTS [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_btts.pkl) | 166,541 (28 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| BTTS [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_btts.pkl) | 13,535 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| Over 1.5 [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_over15.pkl) | 166,541 (28 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| Over 1.5 [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_over15.pkl) | 13,535 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| Over 3.5 [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_over35.pkl) | 166,541 (28 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| Over 3.5 [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_over35.pkl) | 13,535 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| HT Over 0.5 [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_ht_over05.pkl) | 165,900 (23 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| HT Over 1.5 [standard] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_ht_over15.pkl) | 165,900 (23 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| New Format HT Over 0.5 [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_ht_over05_nf.pkl) | 13,444 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| New Format HT Over 1.5 [new_format] | train-only median per feature + `__missing` indicator, clipped to the training range; a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard) | v9-train-only-imputer-1.1 (model_v9_ht_over15_nf.pkl) | 13,444 (13 features) | tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; scripts/team_train_serve_parity.py (feature parity) |
| Props goals / assists / sot / sot2 | none (HistGradientBoosting NaN-native) | props_clean_hgb_nan_native | 0 | parity 6,316 rows, 0 mismatched features, bit-identical probabilities |
| Props cards + recipe challengers | fit-block medians for 128 features, persisted in the payload | props_cards_clean_recipe_retrained_bfd5e1cc | per payload record | batch invariance (diff <= 5.6e-17), serving-vs-training parity |
| Rare props goals2/3, sot3/4 | unseen player -> fitted window market rate (beta prior, method of moments) | player_base_rate | - | holdout replay 100% identical |
| Fantasy V2 | MinutesModel NaN-aware boosting; team-strength prior; `p_start` mix 0.227 (measured, CI 0.213-0.241) | fitted at run time on FINAL pre-deadline gameweeks | - | walk-forward chronological, byte-identical rebuild (P1_FANTASY_VALIDATION) |

Low coverage, named: Standard `home/away_xg_last5` is observed on **7.5%** of the fit block (the rest is the
train median + indicator), `home/away_insidebox_last5` on **44.9%**. These are VALID_MODEL_ESTIMATE in the sense of
`real_data_provenance` (most training cells are the explicit fitted fill), not observations.

## 6. Acceptance matrix

Verdicts: CERTIFIED · CERTIFIED_WITH_DOCUMENTED_LIMITATIONS (CWDL) · NOT_CERTIFIED · RESEARCH_ONLY · UNVERIFIED.
No row is CERTIFIED: every scope carries at least one named limitation.

| Scope | Real Data | Train-Serve Parity | Integrity | Deployment | Verdict |
|---|---|---|---|---|---|
| **Standard O/U 2.5** | NOT_REAL: D1, 21 rows / 34 cells UNVERIFIED_VALUE (`output/real_data_certification.json`@0c2ce95d0) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity WARN (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `67ee3666`, tips + TOP-3 (`models/registry/current_champions.json`@3a9d28ea3) | NOT_CERTIFIED |
| **NF O/U 2.5** | REAL_100 w/ missing kept missing; 0 archive rows can reach NF (`output/real_data_certification.json`@0c2ce95d0) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `303c9a05`, tips + TOP-3 | CWDL (row frame unsaved; D3; price provenance WARN) |
| **BTTS** | Std part NOT_REAL: D1, 21 rows / 34 cells UNVERIFIED_VALUE; NF part REAL_100 w/ missing | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `44c67049` (Std + NF parts) | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Over 1.5** | as BTTS | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `ca01b7d3` | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Over 3.5** | as BTTS | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `e8b12d74` | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Standard HT** | NOT_REAL: D1 downstream, 19 rows / 19 cells (`output/real_data_certification.json`@0c2ce95d0) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`); #78: 0 of 2,604 cells on the installed HT payloads (`reports/P0_HT_FEATURE_PARITY.md`@f68386ed4) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `0776cea1` / `aa4c62ce` (PR #78), HT ledger | NOT_CERTIFIED |
| **NF HT** | REAL_100 w/ missing kept missing; 0 archive rows can reach NF (`output/real_data_certification.json`@0c2ce95d0) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity WARN (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `ae89ea10` / `ead4bb74`, prediction ledger only (no tips) | RESEARCH_ONLY |
| **Goals props** | REAL_100 w/ missing: 934,014 all-prior-verified rows; D4 stored placeholders unread (`output/real_data_certification.json`@0c2ce95d0) | 6,316 rows, 0 mismatched features, bit-identical (`output/props_real_data_provenance.json`@6df83d4b3) | VERIFIED 13/13 sha256 (`reports/P0_PROPS_PRODUCTION_CERTIFICATION.md`@af661fca1) | production `5de94714`; ledger lacks book + capture time | CWDL |
| **Assists props** | as Goals | as Goals | as Goals | production `abf39bb5` | CWDL |
| **SOT props** | as Goals | as Goals | as Goals | production `571b102c` / sot2 `5b5fad60` | CWDL |
| **Cards props** | as Goals | as Goals; batch invariance <= 5.6e-17 | as Goals; fit-block medians (128 features) persisted | production `bfd5e1cc` | CWDL |
| **Rare props** | as Goals (player_id + window rate only) | holdout replay identical | as Goals | research picks only, max 3/GW | RESEARCH_ONLY |
| **Corners side model** | REAL_100 w/ missing: 13,015 two-source + 17,043 single-source sane; 908 NaN (`output/real_data_certification.json`@0c2ce95d0) | same scorer for train / serve; features strictly earlier dates (`tests/test_side_models.py`@f1226d94a) | truth rule per league (`models/side_models/side_models.json`@df093eeda) | prediction ledger only (learning) | RESEARCH_ONLY |
| **Cards side model** | REAL_100 w/ missing: 11,102 two-source + 17,043 single-source sane; 798 NaN | as Corners | as Corners | prediction ledger only (learning) | RESEARCH_ONLY |
| **Fantasy V1** | FPL inputs 2,631/2,631 = event/live (`reports/P1_FANTASY_VALIDATION.md`@683237986) | UNVERIFIED: props frame parity not run | MAE +0.613 vs FPL (CI +0.526..+0.739) | dashboard, V1_NOTE flag | NOT_CERTIFIED |
| **Fantasy V2** | as V1; 21 DOMAIN_PRIOR conventions named | walk-forward chronological, byte-identical rebuild (`output/fantasy_v2_paired_walkforward.json`@5ba1b6e7f) | 0 chronology violations | shadow (`FANTASY_V2_RESIDUAL_MODE: shadow`) | RESEARCH_ONLY |
| **Pro research** | UNVERIFIED row level; registry reconciled as-of (`reports/P1_PRO_RECONCILIATION.md`@5bdff8722) | n/a | weekly audit FAIL 2026-11-22 19:49Z (pre-fix code) (Pro `output/weekly_audit.json`@e0483c7f1) | research | RESEARCH_ONLY |
| **Price trader** | UNVERIFIED (EXECUTABLE_PRICE_DATA_SUFFICIENT=NO) | n/a | all edge flags NO (Pro `output/price_trader/price_trader_verdict.json`@3c18e4af2) | REAL_TRADING_ENABLED=NO | RESEARCH_ONLY |
| **V11 research** | UNVERIFIED row level; 0 dup keys, 0 post-kickoff rows (`reports/P1_V11_RECONCILIATION.md`@2eff3e907) | identity parity with v9 (team_names byte copy) | v11 health PASS (v11 `output/v11_research_health.json`@3d609995c) | shadow ledger | RESEARCH_ONLY |
| **Dashboard** | reads committed artifacts only; missing = 404, never fake | byte-level rebuild 0 diffs (`reports/P1_DASHBOARD_SYNC_VERIFICATION.md`@28e4894f7) | four health indicators (`reports/P0_HEALTH_STATUS_CORRECTION.md`@9716919cf); known inf/NaN page exception | synced hourly (dashboard `deploy/BUILD_INFO.json`@c074d31ce) | CWDL |
| **Execution repo** | UNVERIFIED (private stores, row level not measured) | n/a | CI green (5 success, rest skipped by gate) | private dual-write stores | UNVERIFIED |

## 7. What has to happen for 100%

1. **D1** - drop the backtest-archive stat cells no provider confirms (or `WOWZA_USE_BACKTEST_ARCHIVE=0`) and refit the
   6 Standard payloads on the cleaned frame. Owner decision: it is a retrain (research lock / "retrain only when
   wired"). Expected movement is tiny (21 of 49,407 rows), but it is the only path to REAL_100 on Standard.
2. **Parity monitor reference (D3)** - `scripts/integrity_feature_parity.py` builds its training side without the
   football-data season files and HT fills, and reports FAIL on tracks the cell-level parity check passes (0 of
   2,666 cells differ in this audit). Point it at the same loader path as the trainers so production model
   integrity can reach PASS.
3. **D4** - rewrite the stored placeholders in `player_history.parquet` to NaN (cosmetic for models, required for
   "every row real").
4. **Ledgers** - record bookmaker and capture time on team / player tips, and a certified close on TOP-3 and tips.
5. **Fantasy V1** - a serving-vs-training parity run on the `player_history` frame V1 scores.
6. **Pro / v11 / exec** - a row-level provenance pass on their stores (not done here).
