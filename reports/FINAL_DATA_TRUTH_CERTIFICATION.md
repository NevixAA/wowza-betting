# Final data truth certification (owner deep-repair brief 2026-11-22, items 15-16)

Owner priority, 2026-11-23: **"the most important achievement is 100% real data for every row in our data set,
for all the models."** This certification is built around that sentence. A cell says CERTIFIED only where a
committed artifact proves it; a scope that was not measured says UNVERIFIED. No price, odds, probability or stake
value appears here.

| | |
|---|---|
| Repo / base | NevixAA/wowza-betting, main @ `4fb3bf4ea` (merged into this branch; payload parity + full suite measured on `5a6727f42`, no model / src / config change between them except #86, #87, #88; certification re-run after #87 + #88: every count identical) (inputs merged: #74, #75, #76, #78, #81, #82, #83, #84, #85, #86, #87, #88) |
| Other repos (main) | wowzaV9-Pro `17d6faabc` · wowza_v11 `4b336541f` · wowza-dashboard `eda50a6b2` · wowza-exec `82f6d7d9f` |
| Row-level evidence | `output/real_data_certification.json` (`scripts/certify_real_data.py`, generated `2026-11-23T12:06:25Z` on the merge of main `4fb3bf4ea` into `20eccb97d` (the file's `code_sha` records the pre-merge head `20eccb97d`); counts identical to the `6de281c97` run) |
| Other-repo row-level audit (update 2026-11-23 15:22Z) | `output/real_data_certification.json` `external_repos` + `fantasy_v1_parity` (`scripts/certify_real_data.py --pro-root --v11-root --exec-root`, run on `ff66ec4d3` + this branch): wowzaV9-Pro `8505aefac` (origin/main), wowza_v11 `3ba97e606` (origin/main), wowza-exec data branches af-prices-live `ff962fe31`, af-prices-archive `e48e95ccb`, odds-history `2ea844590`, oddsapi-book-close `2e179f20d`, af-side-odds `6596504c8`, prediction-ledger-v11 `f90125e28`, sharp-paper `259e1deaa`, specials-prices `1060ad110` (exec: counts and verdicts only) |
| Companion audit | `reports/POST_GOLIVE_DEEP_AUDIT.md` (defects, per-repo health, the 12 closing answers) |
| Evidence notation | `path@blob` = git blob (first 9 hex) of that file on the PR head; Pro / v11 files cite their repo |

## 1. Verdict on the owner's priority

**100% real data for every row: NOT ACHIEVED.** It is achieved for most v9 scopes and blocked by named rows in
Standard training, the Fantasy V1 serving frame (parity measured: FAILED) and the Pro / v11 / exec stores (measured
row level on 2026-11-23: NOT_REAL, section 3b). No scope is UNVERIFIED any more.

| Scope | 100% real? | What blocks it (rows) |
|---|---|---|
| History banks (`fd_history` 66,484 rows, `af_history` 93,553 rows) | **YES** - 976,839 provider-observed cells, 0 defaults; 1,504,923 cells missing and kept missing | - |
| Standard team training (O/U 2.5, BTTS, O1.5, O3.5 Standard part) | **NO** | 21 of 49,407 training rows: 2 rows come from `output/backtest_all_leagues.csv` (an internal backtest output, not a provider file) - Belgian First Division A 2022-10-23 Standard v Anderlecht and 2022-11-12 Charleroi v Mechelen; of their 16 stat cells 1 agrees with a provider, 7 contradict API-Football, 8 are held by no provider - plus 19 later rows of the same 4 clubs whose rolling windows read them (D1). 34 UNVERIFIED_VALUE cells per payload |
| Standard HT training (HT O0.5 / O1.5) | **NO** | the same 19 downstream rows (the 2 archive rows have no HT score, so they are not HT training rows); 19 UNVERIFIED_VALUE cells per payload |
| New-Format team training (all 6 NF parts) | **YES at cell level** (0 defaults, 0 unverified in the payload records) | row-level frame not saved (lineage keeps keys + hash `9ed237ebc04aee09`). The backtest archive holds **0** NF-league rows (17 leagues, all Standard: `backtest_archive_scope`), so D1 cannot reach NF |
| Serving board (`output/predictions.csv`, 161 fixtures) | **YES** - 0 defaults; 36 missing inputs kept missing | - |
| Props (966,274 player rows -> 13 certified payloads) | **YES** for every row a model trains on | 1,882 `zeros_unverified` rows are context only, never a label; 4,045 population rows fail `all_prior_verified` and are excluded. The STORED `player_history.parquet` still holds legacy placeholders age 25 / 180 cm / 75 kg / age_peak_delta 0 on ~888k of 891,912 rows; no model reads them (D4) |
| Corners / cards side models | **YES** for served leagues (learning ledger only) | single-source leagues verified by a sanity rule, not a second provider (17,043 fixtures per family); 908 corners / 798 cards fixtures unverified -> NaN, never a training row |
| Fantasy inputs | **YES** for the FPL inputs (2,631 of 2,631 FINAL player-gameweeks = FPL `event/live`) | 21 DOMAIN_PRIOR constants (3 marked FIXED) in the projection formula (conventions, not observations); 0 open MISSING_DATA_FALLBACK reachable (`props_batch_median` unreachable per the props certification) |
| Fantasy V1 train/serve parity (props payloads on `player_history`) | **NO - PARITY_FAILED (measured)** | 733 paired players x 128 features per payload (goals / assists / sot2): 67,075 of 93,824 cells differ (124 of 128 features; only `is_home` and the 3 position flags agree); 733 of 733 rows get a different probability on each market. V1 feeds the payloads the STORED `player_history` feature columns (legacy build), the payloads trained on `props_harness.build_frame`. 5 serving players have no training-frame row |
| Ledgers (bets, side, HT, player, TOP-3, v91 shadow, prediction ledger) | **YES** - 0 synthetic labels, 0 prices <= 1 | 69 HT and 39,953 player rows carry no price (kept missing, never a bet); 96 of 351,882 prediction-ledger rows unscored, every one with its reason; bookmaker + capture time not recorded on team / player ledgers before #87; #87 records `entry_bookmaker` on `bets_ledger` + `v91_shadow` from now on (older rows blank, never back-filled); capture time and player ledgers still not recorded |
| wowzaV9-Pro season store (26 tables, 14.5M rows; 20 tables feed a Pro model / ledger) | **NO - NOT_REAL (measured)** | 15 of 26 tables REAL (with missing kept missing); 11 NOT_REAL. Tables feeding models / ledgers: 13,127 default + 2,745,961 unverified of 133.4M cells. Main causes: prices copied from v9 ledgers with no bookmaker on the row (`settlements`, `clv`, `player_props`, `market_snapshots` `v9_capture` rows), duplicate keys (`settlements` 21,355 rows, `player_props` 8,112, `combo_settlements` 2,240, `scout_odds` 32), v9 constant fills in `settlements_backfill` / `feature_snapshots`, 5,543,005 `calculation_sha = unknown` placeholders in `movement_observations` (section 3b) |
| wowza_v11 inputs + output ledgers (28 tables, 484k rows) | **NO - NOT_REAL (measured)** | 40 prices <= 1.0, 1,106 shadow rows captured after kickoff (1,104 flagged by v11's own `is_post_kickoff`), prices without a bookmaker on v9 `bets_ledger` / `predictions` and v11 shadow / microstructure tables; v11 research aggregates and corrections are REAL. football-data CSVs v11 fetches at run time are not stored anywhere: not measurable |
| wowza-exec private stores (8 data branches, 77.7M rows) | **NO - NOT_REAL (measured, counts only)** | REAL: `oddsapi-book-close`, `af-side-odds`, `sharp-paper`. NOT_REAL: `odds-history` (181,218 historical snapshots after kickoff, 19,283 duplicate keys), `prediction-ledger-v11` (12 duplicate keys), `specials-prices` (`specials_log`: 285 rows of prices without a book, 1 constant column), `af-prices-live` / `af-prices-archive` (mirrors of public research / ledger files: 281 / 235 tables carry a named defect) |

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
| wowzaV9-Pro season store, all 26 tables (section 3b) | 14,539,687 | 123,647,641 | 273,868,223 | 860 | 45,371,964 | 6,124,413 | 13,979,811 | NOT_REAL |
| wowzaV9-Pro, the 20 tables that feed a Pro model / ledger | 5,641,829 | 40,827,171 | 77,708,733 | 860 | 12,130,475 | 13,127 | 2,745,961 | NOT_REAL |
| wowza_v11 inputs + output ledgers (28 tables) | 484,154 | 2,578,282 | 10,378,912 | 176,493 | 3,370,802 | 40 | 574,637 | NOT_REAL |
| wowza-exec, 8 private data branches (counts only) | 77,662,599 | 889,035,528 | 639,806,280 | 0 | 156,664,732 | 28,071,789 | 74,714,776 | NOT_REAL |
| Fantasy V1 serving frame vs props training frame (3 payloads x 733 rows x 128 features) | 733 | 26,749 identical of 93,824 per payload | | | | | 67,075 differing per payload | PARITY_FAILED |

Re-run 2026-11-23 15:22Z (no `--std-frame`: the Standard saved-frame counts of the 12:06Z run on the same payloads are
carried forward, marked `carried_forward_from`; every Standard / NF / HT / serving / side / ledger count is identical).
Props moved with new `player_history` rows only: source rows 967,783 (provider_rule 6,820, zeros_unverified 1,882),
population 939,562, all-prior-verified 935,351, excluded 4,211; stored placeholders height 888,263 / weight 888,240 of
893,422 rows (verdict unchanged).

Standard O/U 2.5 training rows (saved frame `ebe47c041a7f595f`) by row source: football_data_raw 29,454, fd_cache 19,624, provider_supplement 327, backtest_archive 2. Backtest archive file: 40,060 rows, tracks {"standard": {"leagues": 17, "rows": 40060}}.

## 3b. Other repos, row level (owner 2026-11-23 "Yes, do all")

Every cell of every table gets one class; a table is REAL only with 0 default and 0 unverified cells. Rules
(each named in the JSON with rows / cells / columns; a rule that cannot run on a table is written NOT_CHECKABLE):

| Rule | Class of the cells it hits | Test |
|---|---|---|
| `sentinel_or_placeholder_value` | DEFAULT_OR_SYNTHETIC | -99 / -999 / -9999 / 999 / 9999 / 99999, or n/a / none / null / unknown / tbd / placeholder / default in an identity or provenance column |
| `constant_column_where_variation_expected` | DEFAULT_OR_SYNTHETIC | a numeric value column (>= 100 values; not bool / version / line / stake / `has_*`) with one value on every row; `anchor_probability` (v11, a sample median broadcast by design) exempt and named |
| `decimal_price_le_1` | DEFAULT_OR_SYNTHETIC | a decimal price <= 1.0 |
| `probability_outside_0_1` | DEFAULT_OR_SYNTHETIC | a probability column outside [0, 1] |
| `negative_count` | DEFAULT_OR_SYNTHETIC | a count column < 0 |
| `flagged_synthetic_default_fallback_imputed` | DEFAULT_OR_SYNTHETIC (derived cells of the row) | the row's own columns / `quality_flags` / notes say synthetic, default, fallback, imputed, placeholder, degraded or no form |
| `duplicate_key` | UNVERIFIED_VALUE | a row repeating the key (every identity / capture / run / source column, needs a row id such as `fixture_key`) of an earlier row |
| `prematch_capture_after_kickoff` | UNVERIFIED_VALUE | capture time after kickoff in a pre-match table (Pro tables without kickoff join `fixtures`), or `minutes_to_kickoff` / `minutes_close_before_kickoff` < 0 in any non-live table |
| `flagged_wrong_by_own_columns` | UNVERIFIED_VALUE | `quality_flags` holds a flag Pro's `src/quality.py` registers as WRONG |
| `no_provider_or_source_identity` | UNVERIFIED_VALUE | a price whose row names no real bookmaker (an across-books median / best is counted VERIFIED_DERIVED; a provider feed without a book layer, e.g. API-Football `/odds/live`, counts its source), or a provider-table row with no source |
| `no_documented_derivation` | UNVERIFIED_VALUE | a derived table no code / README of its repo names |

Missing values are MISSING_KEPT_MISSING (never a defect). `no_provider_or_source_identity` cells are prices that are
real quotes from SOME book but the row does not say which one: they cannot be traced, so they are not counted as
provider-observed. They are not invented prices.

**wowzaV9-Pro** (`data/season_2026_27`, origin/main `8505aefac`). "Feeds" = a model / ledger module of Pro's `src/`
(scout model + promotion, shadow, bet_builder, train_*, btts_eval, prediction_ledger, combo, bankroll, price_trader,
news_impact, ...) names the table; monitoring, importer and migration modules are not counted.

| Table | Feeds a Pro model / ledger | Rows | Default | Unverified | Verdict | Rules fired (rows) |
|---|---|---|---|---|---|---|
| `fixtures` | yes | 89,030 | 0 | 0 | REAL_100 | - |
| `team_match_stats` | yes | 6,862 | 0 | 0 | REAL_100 w/ missing | - |
| `team_news` | yes | 680 | 0 | 0 | REAL_100 w/ missing | - |
| `model_snapshots` | yes | 435,474 | 0 | 0 | REAL_100 w/ missing | - |
| `live_odds_snapshots` | yes | 349,317 | 0 | 0 | REAL_100 w/ missing | - |
| `scout_results` / `scout_results_ft90` / `scout_fixture_status` | yes | 247,204 / 430 / 824 | 0 | 0 | REAL_100 w/ missing | - (ft90: 860 corrected-beside-original cells) |
| `combo_candidates` / `combo_legs` / `combo_dependencies` | yes | 109,090 / 293,422 / 1,809 | 0 | 0 | REAL_100 w/ missing | - |
| `data_quality` | yes | 8,064 | 0 | 0 | REAL_100 w/ missing | - |
| `scout_odds` | yes | 716,342 | 0 | 506 | NOT_REAL | duplicate_key 32 (all conflicting) |
| `scout_model` | yes | 7,518 | 250 | 0 | NOT_REAL | flagged_synthetic_default_fallback_imputed 50 |
| `combo_settlements` | yes | 67,277 | 0 | 46,212 | NOT_REAL | duplicate_key 2,240 (all conflicting) |
| `market_snapshots` | yes | 466,405 | 8 | 188,629 | NOT_REAL | no_provider_or_source_identity 188,029 (`bookmaker = v9_capture`); flagged_wrong 38 (BTTS first-half mislabel); post-kickoff 12; price <= 1: 8 |
| `player_props` | yes | 1,871,306 | 0 | 541,631 | NOT_REAL | no_provider_or_source_identity 352,280 (`market_odds` without a book); duplicate_key 8,112 (7,662 conflicting: two old fixture keys under one v2 key) |
| `clv` | yes | 464,869 | 0 | 737,851 | NOT_REAL | no_provider_or_source_identity 464,869 (`odds_bet` / `odds_close` without a book) |
| `settlements` | yes | 505,128 | 16 | 1,227,840 | NOT_REAL | no_provider_or_source_identity 504,102; duplicate_key 21,355 (same fixture / market / side / run / source, different model probability); flagged_wrong 1,026; price <= 1: 16 |
| `settlements_backfill` | yes | 778 | 12,853 | 3,292 | NOT_REAL | constant columns 12,406 cells (v9 `h2h_*`, `api_implied_*`, coach, formation fills); flagged no-form 7; prices without a book 778 |
| `signals` | no | 89,030 | 0 | 0 | REAL_100 | - |
| `live_signals` | no | 18,238 | 0 | 0 | REAL_100 w/ missing | - |
| `scout_specials_quotes` | no | 2,658,194 | 0 | 0 | REAL_100 w/ missing | - |
| `book_odds_snapshots` | no (studies only) | 465,818 | 8 | 0 | NOT_REAL | price <= 1: 8 |
| `feature_snapshots` | no | 89,030 | 568,257 | 93,096 | NOT_REAL | `feature_degraded` / no-form rows 10,306 (431,731 cells); `key_attacker_absent_home/away` constant (136,526 cells); prices without a book 73,183; post-kickoff 3 |
| `movement_observations` | no (v11 import, weekly audit) | 5,577,548 | 5,543,021 | 11,140,754 | NOT_REAL | `calculation_sha = unknown` 5,543,005; prices without a book 5,570,385; price <= 1: 16 |

**wowza_v11** (origin/main `3ba97e606`). Inputs found in v11's code: v9 `output/predictions.csv`, `bets_ledger.csv`,
`book_odds_snapshots.csv`, `af_history.parquet` (measured on this checkout) and football-data season CSVs fetched at run
time by `src/results.py` (stored nowhere: not measurable). Outputs: every `output/*.csv` and `output/corrections/*.csv`.

| Table | Rows | Default | Unverified | Verdict | Rules fired (rows) |
|---|---|---|---|---|---|
| input `af_history.parquet` | 93,553 | 0 | 0 | REAL_100 w/ missing | - |
| input `book_odds_snapshots.csv` | 31,933 | 2 | 0 | NOT_REAL | price <= 1: 2 |
| input `bets_ledger.csv` | 1,943 | 1 | 5,538 | NOT_REAL | `closing_odds` <= 1: 1; prices without a book 1,943 |
| input `predictions.csv` | 161 | 0 | 1,015 | NOT_REAL | prices without a book 161 |
| `v11_shadow_log.csv` | 2,622 | 1 | 21,153 | NOT_REAL | post-kickoff 482 (481 flagged `is_post_kickoff` by v11); prices without a book 2,622; price <= 1: 1 |
| `v11_shadow_snapshots.csv` | 89,777 | 4 | 200,144 | NOT_REAL | post-kickoff 624 (623 flagged by v11); prices without a book 89,777; price <= 1: 4 |
| `v11_graded.csv` | 2,622 | 0 | 1,307 | NOT_REAL | `live_first_odds` without a book |
| `v11_market_movement_detail.csv` / `v11_market_microstructure.csv` | 86,385 each | 16 each | 172,740 each | NOT_REAL | `close_odds` <= 1: 16; entry / close prices without a book |
| 14 research aggregates + 5 corrections (incl. `v11_p_market_restated.csv`, 176,493 corrected-beside-original cells) | 88,773 | 0 | 0 | REAL_100 (w/ missing) | - |

**wowza-exec** (PRIVATE: counts and verdicts only; no column, value, price or row is written).

| Branch @ head | Tables | Rows | Default | Unverified | Verdict | Rules fired |
|---|---|---|---|---|---|---|
| `oddsapi-book-close` @`2e179f20d` | 2 | 129 | 0 | 0 | REAL_100 | - |
| `af-side-odds` @`6596504c8` | 1 | 484,826 | 0 | 0 | REAL_100 w/ missing | - |
| `sharp-paper` @`259e1deaa` | 2 | 87 | 0 | 0 | REAL_100 w/ missing | - |
| `prediction-ledger-v11` @`f90125e28` | 3 | 1,298 | 0 | 318 | NOT_REAL | duplicate_key 12 (all conflicting) |
| `specials-prices` @`1060ad110` | 2 | 339,600 | 602 | 3,095 | NOT_REAL | `data/special_prices` REAL; `specials_log` prices without a book 285, 1 constant column. The pre-dedupe original (561,562 rows) is kept beside the correction and reported apart (not in the verdict) |
| `odds-history` @`2ea844590` | 4 | 32,195,242 | 0 | 3,333,510 | NOT_REAL | historical snapshots after kickoff 181,218; duplicate_key 19,283 (8,242 conflicting); events / match / units ledgers REAL |
| `af-prices-live` @`ff962fe31` | 310 | 24,286,482 | 14,144,711 | 37,008,911 | NOT_REAL | 281 tables with a defect: prices without a book 9.0M rows / 258 tables; `calculation_sha = unknown` 5.55M (the Pro mirror); synthetic / fallback labels 169,011 rows / 27 tables (public `backtest_results_*` mirrors); duplicate_key 517,121 / 38; post-kickoff 122,433 / 10; probability outside [0,1] 24,760 / 9; price <= 1: 2,637 / 24 |
| `af-prices-archive` @`e48e95ccb` (immutable seed of af-prices-live) | 262 | 20,354,935 | 13,926,476 | 34,368,942 | NOT_REAL | the same mirrors at seed time (235 tables with a defect) |

## 4. Every default / placeholder / synthetic / unverified value found

| # | Dataset | Column(s) | Rows / cells | Cause | Reaches a model? |
|---|---|---|---|---|---|
| D1 | Standard training frame (`row_source = backtest_archive`) | home/away shots, SOT, corners, fouls of 2 matches (Belgian First Division A 2022-10-23 Standard v Anderlecht; 2022-11-12 Charleroi v Mechelen) | 2 rows (16 stat cells: 1 agrees with a provider, 7 contradicted by API-Football, 8 held by no provider) + 19 downstream rows | `src/data_loader._load_backtest_archive` appends `output/backtest_all_leagues.csv`; for these two matches football-data has the score but no stats, so the archive row (with stats) survives the identity pipeline | **YES**: the 4 Standard O/U-family payloads (row + windows), the 2 Standard HT payloads (windows only). Serving today: no (2022 windows) |
| D4 | `player_history.parquet` (stored) | age / height_cm / weight_kg / age_peak_delta | 887,907 / 888,192 / 888,142 / 887,907 of 891,912 rows | legacy `build_features` placeholders written into the stored file | **NO**: `player_model/feature_engineering._PROFILE_PLACEHOLDERS` reads them as missing; profiles come from `output/collect/player_profiles.parquet`; props certification: no served feature has one value on > 99.9% of training rows |
| - | Team payloads | - | 0 FABRICATED_VALUE / 0 untraced features in the 12 active team payload parts | the retired `model_v9_standard_repaired.pkl` (4 fabricated features) is refused by `owner_install` and serves nothing | - |
| - | Ledgers | price columns | 0 synthetic labels, 0 prices <= 1 | `output/player_props_calibration.json` stays quarantined (INVALID_FOR_REAL_BETTING_ROI) | - |
| - | Fantasy | `props_batch_median` (register entry not marked FIXED) | 0 | legacy props `_prep`: unreachable in production per the props certification (AST call graph + runtime probe: 0 calls) | NO |
| D5 | Fantasy V1 serving frame | 124 of 128 props features | 67,075 of 93,824 cells per payload (733 players) | V1 scores the stored `player_history` columns, the payloads trained on `props_harness.build_frame` (`fantasy_v1_parity`) | **YES**: V1 goals / assists / sot2 probabilities (dashboard V1, V2 WOWZA arm / `wowza_gap`) |
| D6 | wowzaV9-Pro tables feeding models / ledgers | prices, keys, v9 fills (section 3b) | 13,127 default + 2,745,961 unverified cells | prices without a bookmaker, duplicate keys, v9 constant fills, own WRONG flags | **YES** (Pro research models / ledgers only; no v9 tip reads them) |
| D7 | wowza_v11 inputs + ledgers | prices, post-kickoff rows | 40 default + 574,637 unverified cells | section 3b | v11 shadow (research) |
| D8 | wowza-exec private stores | counts only | 28.1M default + 74.7M unverified cells over 8 branches | section 3b; `oddsapi-book-close`, `af-side-odds`, `sharp-paper` clean | stores only (v9 attach reads `odds-history` / `af-side-odds`; their defects are named) |

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
| **Standard O/U 2.5** | NOT_REAL: D1, 21 rows / 34 cells UNVERIFIED_VALUE (`output/real_data_certification.json`@f587248be) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity WARN (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `67ee3666`, tips + TOP-3 (`models/registry/current_champions.json`@3a9d28ea3) | NOT_CERTIFIED |
| **NF O/U 2.5** | REAL_100 w/ missing kept missing; 0 archive rows can reach NF (`output/real_data_certification.json`@f587248be) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `303c9a05`, tips + TOP-3 | CWDL (row frame unsaved; D3; price provenance WARN) |
| **BTTS** | Std part NOT_REAL: D1, 21 rows / 34 cells UNVERIFIED_VALUE; NF part REAL_100 w/ missing | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `44c67049` (Std + NF parts) | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Over 1.5** | as BTTS | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `ca01b7d3` | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Over 3.5** | as BTTS | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `e8b12d74` | NOT_CERTIFIED (Std part) / CWDL (NF part) |
| **Standard HT** | NOT_REAL: D1 downstream, 19 rows / 19 cells (`output/real_data_certification.json`@f587248be) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`); #78: 0 of 2,604 cells on the installed HT payloads (`reports/P0_HT_FEATURE_PARITY.md`@f68386ed4) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity FAIL (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `0776cea1` / `aa4c62ce` (PR #78), HT ledger | NOT_CERTIFIED |
| **NF HT** | REAL_100 w/ missing kept missing; 0 archive rows can reach NF (`output/real_data_certification.json`@f587248be) | cell-level PASS: 0 of 2,666 cells, 62/62 probabilities identical (this audit, `scripts/team_train_serve_parity.py` on `5a6727f42`) | lineage PASS (`reports/lineage/training_lineage_evidence.json`@a59759051); model integrity WARN (D3 monitor) (`output/model_integrity_health.json`@e8954a3c2) | active `ae89ea10` / `ead4bb74`, prediction ledger only (no tips) | RESEARCH_ONLY |
| **Goals props** | REAL_100 w/ missing: 934,014 all-prior-verified rows; D4 stored placeholders unread (`output/real_data_certification.json`@f587248be) | 6,316 rows, 0 mismatched features, bit-identical (`output/props_real_data_provenance.json`@6df83d4b3) | VERIFIED 13/13 sha256 (`reports/P0_PROPS_PRODUCTION_CERTIFICATION.md`@af661fca1) | production `5de94714`; ledger lacks book + capture time | CWDL |
| **Assists props** | as Goals | as Goals | as Goals | production `abf39bb5` | CWDL |
| **SOT props** | as Goals | as Goals | as Goals | production `571b102c` / sot2 `5b5fad60` | CWDL |
| **Cards props** | as Goals | as Goals; batch invariance <= 5.6e-17 | as Goals; fit-block medians (128 features) persisted | production `bfd5e1cc` | CWDL |
| **Rare props** | as Goals (player_id + window rate only) | holdout replay identical | as Goals | research picks only, max 3/GW | RESEARCH_ONLY |
| **Corners side model** | REAL_100 w/ missing: 13,015 two-source + 17,043 single-source sane; 908 NaN (`output/real_data_certification.json`@f587248be) | same scorer for train / serve; features strictly earlier dates (`tests/test_side_models.py`@f1226d94a) | truth rule per league (`models/side_models/side_models.json`@df093eeda) | prediction ledger only (learning) | RESEARCH_ONLY |
| **Cards side model** | REAL_100 w/ missing: 11,102 two-source + 17,043 single-source sane; 798 NaN | as Corners | as Corners | prediction ledger only (learning) | RESEARCH_ONLY |
| **Fantasy V1** | FPL inputs 2,631/2,631 = event/live (`reports/P1_FANTASY_VALIDATION.md`@683237986) | **FAILED (measured)**: 67,075 of 93,824 cells differ per payload, 733/733 probabilities differ (`output/real_data_certification.json` `fantasy_v1_parity`) | MAE +0.613 vs FPL (CI +0.526..+0.739) | dashboard, V1_NOTE flag | NOT_CERTIFIED |
| **Fantasy V2** | as V1; 21 DOMAIN_PRIOR conventions named | walk-forward chronological, byte-identical rebuild (`output/fantasy_v2_paired_walkforward.json`@28774db3c) | 0 chronology violations | shadow (`FANTASY_V2_RESIDUAL_MODE: shadow`); #88 adds a pre-registered promotion rule (`config/fantasy_v2_promotion_rule.json`) - eligibility changes nothing served | RESEARCH_ONLY |
| **Pro research** | NOT_REAL row level (measured): 20 model / ledger tables, 13,127 default + 2,745,961 unverified cells (section 3b); registry reconciled as-of (`reports/P1_PRO_RECONCILIATION.md`@5bdff8722) | n/a | weekly audit FAIL 2026-11-22 19:49Z (pre-fix code) (Pro `output/weekly_audit.json`@e0483c7f1) | research | RESEARCH_ONLY |
| **Price trader** | UNVERIFIED (EXECUTABLE_PRICE_DATA_SUFFICIENT=NO) | n/a | all edge flags NO (Pro `output/price_trader/price_trader_verdict.json`@3c18e4af2) | REAL_TRADING_ENABLED=NO | RESEARCH_ONLY |
| **V11 research** | NOT_REAL row level (measured): 40 default + 574,637 unverified cells; 1,106 shadow rows captured after kickoff (flagged by v11 itself), prices without a book (section 3b); ledger reconciliation 0 dup keys (`reports/P1_V11_RECONCILIATION.md`@2eff3e907) | identity parity with v9 (team_names byte copy) | v11 health PASS (v11 `output/v11_research_health.json`@3d609995c) | shadow ledger | RESEARCH_ONLY |
| **Dashboard** | reads committed artifacts only; missing = 404, never fake | byte-level rebuild 0 diffs (`reports/P1_DASHBOARD_SYNC_VERIFICATION.md`@28e4894f7) | four health indicators (`reports/P0_HEALTH_STATUS_CORRECTION.md`@9716919cf); known inf/NaN page exception | synced hourly (dashboard `deploy/BUILD_INFO.json`@c074d31ce) | CWDL |
| **Execution repo** | NOT_REAL row level (measured, counts only): 3 of 8 data branches REAL (`oddsapi-book-close`, `af-side-odds`, `sharp-paper`); `odds-history` 181,218 post-kickoff snapshots + 19,283 duplicate keys; mirrors in `af-prices-live` / `-archive` carry the public files' defects (section 3b) | n/a | CI green (5 success, rest skipped by gate) | private dual-write stores | NOT_CERTIFIED |

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
4. **Ledgers** - record bookmaker and capture time on team / player tips, and a certified close on TOP-3 and tips. (#87: entry bookmaker now recorded on `bets_ledger` + `v91_shadow`, new rows only; capture time, player ledger and the certified close remain open.)
5. **Fantasy V1** - parity measured and FAILED (D5): score V1 on the props serving build
   (`feature_engineering.build_upcoming_features`, the path the props payloads are certified on) instead of the stored
   `player_history` columns, then re-run `fantasy_v1_parity` to 0 differing cells. Code change in the fantasy path:
   owner decision (it changes V1 numbers on the dashboard).
6. **Pro / v11 / exec** - measured (section 3b). To reach REAL: record the bookmaker on every copied price (v9 ledgers ->
   Pro `settlements` / `clv` / `player_props`, v11 shadow), drop or key-disambiguate duplicate rows (Pro `settlements`,
   `player_props`, `combo_settlements`, `scout_odds`; exec `prediction-ledger-v11`, `odds-history`), stop storing v9
   constant fills (`settlements_backfill`, `feature_snapshots`), write the real `calculation_sha` in
   `movement_observations`, and exclude post-kickoff snapshots from pre-match stores (v11 shadow, `odds-history`).
   Each is a data repair in its own repo, not done here (read-only).
