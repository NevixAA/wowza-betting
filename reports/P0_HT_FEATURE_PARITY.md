# P0 item 4: half-time feature parity (home/away_has_ht_history)

Owner 2026-11-22: P0 deep-repair list, item 4, "Repair half-time feature parity". Retrain decision, same day: "Yes, retrain the 2 HT models".
Counts and probabilities only. No prices.

| | |
|---|---|
| Repo | NevixAA/wowza-betting, branch `retrain/std-ht-drop-has-ht-history` (base main `ce044147d`) |
| Commits | `f9cfba733` config + harness (per-market exclusion) · `cbc00c134` install (NOT merged) · `300011fe2` track registry · `5e6d6aedb` HT atomic fix + test + evidence |
| Owner approval needed | **Yes, for the install.** The PR stays unmerged until the owner approves the comparison below. |

## 1. Root cause

`has_ht_history` is 1.0 when the team's last 5 league matches before the fixture include at least one real half-time score.

| Candidate cause | Verdict | Evidence |
|---|---|---|
| Genuine improved coverage (PR #59 HT backfill) | **YES. This is the cause.** | Old frame: Standard 2021/22 and 2022/23 had 95 HT-eligible rows each, with 24-28% flag=0. Of the 318 zeros, 315 were windows of 5 `backtest_archive` rows with no HT score; 1 was mixed `backtest_archive` and `fd_cache`, and 2 were `fd_cache` gaps. PR #59 replaced those rows with football-data rows that carry HT. New frame: 0 home zeros in 49,446 Standard HT-eligible rows. |
| Structural degeneracy | YES (contributing) | Training requires form (`require_present`: home/away_scored_last5), so every eligible row has at least 1 earlier match. With full HT coverage, a 0 is possible only when the window holds no HT at all. |
| Wrong historical (training) feature engineering | No | Independent rebuild vs training frame: 0 of 77,327 (new) and 0 of 59,462 (old) rows differ per side. |
| Wrong live (serving) feature engineering | No | 1,468 fixtures, 40 dates, all 32 leagues: 0 serving mismatches. Live board, 151 fixtures: 0 mismatches. Installed payload parity: 0 of 2,604 cells. |
| Missingness-marker implementation | **One latent defect, fixed** | A match with only one HT side recorded fed `ht_scored/ht_conceded_last5` while the flag said "no HT score". Caught by the new test. 0 such rows in the committed history (0 of 93,383), so no trained or served value changed. |
| League / season joins | No | The window is per (league, club). The other-league case is in the test. Board names are provider spellings (172 of 302 board cells renamed through the identity layer); after resolution, 0 mismatches. |
| Caches / stale artifacts | No | The installed refit reproduces bit-for-bit from frame `ebe47c041a7f595f` (only code_sha differs). |
| Train-vs-serve transform difference | No (one known, by design) | Serving reads the full history; training drops COVID seasons. 172 sample cells on COVID-season dates (2019/20, 2020/21) have no training row; serving == independent value on all of them. |

"Constant live values where training varied": the live board is 1.0 on 302 of 302 cells. That is correct: every board club has 5 league matches with HT, so the independent rebuild is also 1.0 on all 302. Older payloads varied only because their training rows included the HT-less `backtest_archive` windows.

## 2. Flag share across payloads (training rows, imputer stats)

| Payload | home =1.0 | away =1.0 | DT3 |
|---|---|---|---|
| ht_over05 65646406 (2026-11-18, pre-#59) | 99.05% | 99.02% | green |
| ht_over05/15 5a76d680 / e9a12ce1 (served, 2016/17 refit) | **100% (min = max)** | 99.997% (one 0) | **RED** |
| ht_over05/15 0776cea1 / aa4c62ce (this retrain) | not an input | 99.997% | **GREEN** |
| Standard O/U 2.5, btts, over15, over35 (served) | 99.95% | 99.95% | green |
| new-format over25 / HT nf (served) | 99.9% / 99.99% | 99.9% / 99.995% | green |

## 3. Fix

1. **Retrain.** Only the two Standard HT models drop the single input the DT3 rule flags (`home_has_ht_history`). The change is recorded in `config/phase2_retrain.json` `features.market_exclude`, with the owner's words.
   - `away_has_ht_history` is not flagged (min 0, max 1) and stays.
   - Everything else is unchanged: champion_recipe, frame `ebe47c041a7f595f`, Standard training 2016-08-05..2026-11-01, data end 2026-11-08, 49,393 rows, 84 inputs (was 86).
   - Thresholds are unchanged.
2. **Code.** `src/feature_engineering.py` now treats a half-time score as atomic. The function is shared by training and serving.

Files changed:
- `config/phase2_retrain.json`
- `src/retrain_harness/protocol.py`
- `src/retrain_harness/runner.py`
- `scripts/phase2_refit_serving.py`
- `src/feature_engineering.py`
- `tests/test_phase2_retrain_harness.py`
- `tests/test_ht_feature_parity.py`
- `scripts/audit/ht_feature_parity_evidence.py`
- `reports/ht_feature_parity/*`
- install: `config/promotion_overrides.json`, `models/model_ht_over05.pkl`, `models/model_ht_over15.pkl`, registry and lineage files, `config/track_registry.json`

## 4. Before / after evidence (chronological validation)

The holdout uses the protocol of the 2016/17 retrain comparison (`std_retrain_1611/cmp`). Each recipe is fitted on Standard rows dated before 2026-04-01 and scored once on 2026-04-01..2026-11-08. Both use the same frame and the same rows: 46,449 training, 2,944 holdout.

| Market | Group | n | LL old | LL new | Brier old / new | AUC old / new | LL base rate (2y) | AUC base |
|---|---|---|---|---|---|---|---|---|
| HT O0.5 | All Standard | 2,944 | 0.58659 | 0.58659 | 0.19870 / 0.19870 | 0.5302 / 0.5302 | 0.58694 | 0.5303 |
| HT O0.5 | EFL | 747 | 0.59639 | 0.59639 | 0.20323 / 0.20323 | 0.5126 / 0.5126 | 0.59890 | 0.4961 |
| HT O1.5 | All Standard | 2,944 | 0.65258 | 0.65258 | 0.23003 / 0.23004 | 0.5427 / 0.5426 | 0.65441 | 0.5189 |
| HT O1.5 | EFL | 747 | 0.64607 | 0.64607 | 0.22685 / 0.22685 | 0.5205 / 0.5205 | 0.64835 | 0.4504 |

- **The models are the same.** New minus old log loss is 0.00000 (95% CI [0.00000, 0.00001]) overall, in EFL and in each EFL league. The dropped flag carried no information.
- Holdout probability moves: HT O0.5 mean 0.001 pp, max 0.011 pp; HT O1.5 mean 0.000 pp, max 0.186 pp.
- **The model beats the trailing league base rate only slightly.** HT O1.5 overall: -0.00183 LL [-0.00366, +0.00014], P = 0.97. HT O0.5 overall is level with it.
- **Current board** (generated 2026-11-22 14:54 UTC): 94 Standard fixtures in 7 leagues, 2026-11-22..29.
  - HT O0.5: mean 0.013 pp, max 0.087 pp.
  - HT O1.5: mean 0.000 pp, max 0.001 pp.
  - 0 fixtures move by more than 1 pp.
- **Installed payloads, train/serve parity** (`scripts/team_train_serve_parity.py`): 0 of 2,604 cells differ (62 fixtures, 2026-11-14..20, 42 raw inputs). 62/62 probabilities are identical on both payloads.
- **Data truth 3 served-feature spread:** GREEN on all 21 served payloads (was RED on these two).

Row-level files in `reports/ht_feature_parity/`. Columns: fixture_id, league, kickoff, feature, raw_source, expected_value, training_value, serving_value, mismatch_reason.
- `ht_flag_frame_rows.csv`: every HT-eligible flag=0 row in both frames, with its window provenance. There are no mismatch rows: 0 exist.
- `ht_flag_serving_sample.csv`: 2,936 cells.
- `ht_flag_live_board.csv`: 302 cells.
- `flag_share_by_track_season_*.csv`, `summary.json`.
- fixture_id is the API-Football id where the source carries one, else date|league|home|away.

## 5. Tests

- `tests/test_ht_feature_parity.py`: 9 tests. They compare an independent rebuild, training and serving on 7 hostile histories. Before the atomic fix, the half-HT case failed; all 9 pass after it.
- `tests/test_p2_parity_adversarial.py`: pass, 39 together with the new file.
- `tests/test_phase2_retrain_harness.py`: 50 passed. One new test: only the two Standard HT markets lose the flag.
- CI on Linux runs the full suite on the PR.

## 6. Remaining blockers / notes

- **Real HT gaps remain in the new frame.** There are 8 HT-eligible rows with flag 0, all correct, each a window with no HT score in the source:
  - Standard: Turkish Super Lig 2023-08-13 (5 football_data_raw rows from 2022/23), Bundesliga 2 2025-05-27, Serie B 2026-08-21 (5 fd_cache rows from 2025/26).
  - New-format: Finland x2, Japan, Romania, China.
  - These are missing source data, not a feature bug. Listed in `ht_flag_frame_rows.csv`.
- **The flag is near-constant in every other served payload** (99.9-99.95%). DT3 does not flag those payloads because its rule is min == max or missing > 99.9%, and the rule was not changed. Dropping the flag there needs a retrain, which is an owner decision and was not done.
- **The training/serving window around COVID seasons differs by design** (training excludes them). It only matters for fixtures right after a COVID season, never for today's board.
- LightGBM pickles vary in bytes between identical fits; predictions are identical (max diff 0.0). The install sha256 is the one listed in the override stage.

## 7. Production impact

- Until install: none. Main still serves 5a76d680 / e9a12ce1, and DT3 stays RED.
- After install: DT3 GREEN on both HT payloads. HT probabilities move by at most 0.19 pp on the holdout and 0.09 pp on the current board.
- No other model, threshold, bar, tier or stake changes.
- The feature-engineering fix changes 0 values on the committed data.
