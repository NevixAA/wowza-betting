# P0 - Clean retrain plan and harness (Phase 2 part 4)

Branch `phase2/retrainharness`, from `origin/data-truth-100` at 97e48c6a. Date: 2026-11-06.

**This is a build only.** No model was trained on real data. No backtest or walk-forward was run on real data, nothing about edge, ROI or profit was computed, and no threshold was changed. Every component was tested on small synthetic frames. The only real-data step was the dry run, which counts rows and lists columns.

## Verdict

- **Ready:** the harness, the pre-registered protocol (`config/phase2_retrain.json`), the dry-run CLI and a dispatch-only workflow. The workflow is gated twice: by the repository variable and by the owner's holdout approval in the config.
- **Waiting for data:** collection ends 2026-11-08. The dry run must be repeated on the final committed data. The counts below come from today's commit and will move.
- **Waiting for the owner:** the holdout dates, adopting the thresholds (reused from the promotion gate), the league-size rule, and the open questions in section 8. Until `holdout.approved_by` is set **and** `PHASE2_RETRAIN_ENABLED` is `true`, the code refuses to fit. A test checks this.

## 1. What was built

| File | What it is |
|---|---|
| `config/phase2_retrain.json` | The pre-registered protocol: holdout dates, fold sizes, tracks, markets, feature groups, candidates and their hyperparameters, comparison rules and thresholds. Its sha256 is recorded in every run. |
| `src/retrain_harness/protocol.py` | Config checks, fit guards, run-folder guard, eligible rows, feature list, walk-forward folds, final holdout, the dry-run plan, and the offline committed-data loader |
| `src/retrain_harness/candidates.py` | The six candidates (section 3) |
| `src/retrain_harness/evaluate.py` | Log loss, Brier, ECE, calibration table and line, CLV, and the paired match-date bootstrap with Holm correction |
| `src/retrain_harness/runner.py` | The walk-forward and holdout stages. Everything goes to one run folder |
| `scripts/phase2_retrain.py` | CLI: `--dry-run` (plan, row counts, features; no fitting) or a guarded run |
| `.github/workflows/phase2_retrain.yml` | `workflow_dispatch` only. Job `if: vars.PHASE2_RETRAIN_ENABLED == 'true'`, `contents: read`, one job per track. It uploads the run folder as an artifact, commits nothing, and fails if a tracked file changed |
| `tests/test_phase2_retrain_harness.py` | 35 synthetic tests (section 9) |

## 2. Data

The run uses the loader's served frame, built from the **committed** data: `src.data_loader.load_all_matches`, with the football-data download turned off and the network blocked. Then come COVID exclusion (per league, `config.covid_season_mask`) and `src.feature_engineering.build_features` (point-in-time, `shift(1)` windows, the parity-validated form block). Every loader correction applies because the loader applies it:

- identity and one-match-one-row;
- history corrections, including the Argentina cup rows;
- written-back af_history statistics are not served;
- impossible shot counts are set to missing;
- O/U pairs with a negative margin are flagged.

The loader's `training_coverage.json` report is not rewritten. The frame hash is recorded in each run manifest.

Dry run on today's commit:

| | Value |
|---|---|
| Loaded rows | 70,744 |
| COVID rows excluded | 12,271 |
| Feature rows | 58,473 (2019-02-15 .. 2026-10-30) |
| Frame hash | `a4cb912462c730d2` |
| Written-back cells not served (home shots / home xG / home inside-box) | 11,978 / 9,394 / 10,887 blanked, of which 6,808 / 6,718 / 9,460 were refilled from a provider source; the rest stay missing |
| O/U 2.5 pairs flagged "not a price" (negative margin) | 434 (the market candidates read these as missing) |

Eligible row = the track's league (via `src.model.track_of`) + the market's label + rolling form present on both sides (`home_scored_last5`, `away_scored_last5`). This is the blind-fixture rule: serving never tips such a fixture, so it is neither trained on nor scored.

## 3. Candidates (per track, per market)

| Candidate | What it is | Missing values |
|---|---|---|
| `champion_recipe` | `src.model.train_repaired`, as the scheduled retrain calls it (`RETRAIN_RECIPE`): LR + GB (+ LightGBM), Platt on its calibration block, meta-logistic on its meta block, derived recency weights. Retrained on the repaired frame, one track at a time | A missing indicator for every feature, plus the 1.1 train-only imputer stored in the payload. A feature with no training value is **dropped**, never filled |
| `logistic` | Standardised logistic regression (C 0.5) | Median imputer fitted on the training window only, a missing indicator for **every** feature, pickled with the model. Columns with no training value are dropped |
| `hgb` | `HistGradientBoostingClassifier` (300 iterations, lr 0.05, 31 leaves, min leaf 40, L2 1.0, no early stopping) | **NaN-native**: nothing is filled at train or serve time |
| `dixon_coles` | Per-league score matrix, using `src.goal_distribution`'s matrix and tau. Strengths come from the last 730 days of the training window, shrunk toward the league mean by 25 matches. **rho is fitted by maximum likelihood** on the training window, within [-0.25, 0.25]. Every goal market (and HT, from half-time goals) comes from one matrix, so the markets are coherent | A club or league with no training match gets **no probability**. The research class's 1.0 strength and 1.4 / 1.1 goal fallbacks are not used |
| `market_only` | Power de-vig (`src.market_anchor.power_devig`) of a **two-sided** pair | One-sided or flagged pair: no probability |
| `model_market` | Logistic stack of [logit p_hgb, logit p_market]. The base is fitted on the training window minus the validation block. **The stack is fitted on the validation block only** | No two-sided price: no probability (no fallback to the model alone) |

`src/poisson.py` (`dixon_coles_p_over25`, rho fixed at -0.08) only computes a feature. It is not a fitted model, so the harness does not use it as a candidate. It already feeds `p_over25_poisson_dc` through `build_features`.

**Price class.** The committed frame has one O/U 2.5 pair per match. For football-data rows it is closing-preferred (AvgC first, entry columns as fallback). For new-format rows it comes from captured snapshots. So `market_only` is a **closing-market benchmark**, and `model_market` built on it is marked `deployable: false` in every output, because a closing price is not known at bet time. Price-derived columns (`bookmaker_overround`, `implied_prob_*`) are **never** model inputs.

## 4. Protocol

- **Splits are on whole dates** (one matchday = one date with an eligible row in the track):
  - expanding walk-forward;
  - the first test block starts after 365 matchdays;
  - blocks are 90 matchdays;
  - the last 60 matchdays of each training window form the validation block, used only for the model+market stack.
- **Untouched final holdout: 2026-07-01 .. 2026-11-08 (PROPOSED).** No fold contains a holdout row. The holdout stage fits on the whole walk-forward region and scores the holdout once. It writes `holdout_opened.json` before scoring and refuses a second holdout stage in the same run folder. Rows after 2026-11-08 are not used.
- **Metrics:** log loss, Brier, ECE (10 bins), a calibration table, and calibration intercept and slope. These are reported per stage, track, market and candidate, and per league.
- **CLV:** counted only where both a real entry price and a real closing price exist. Those columns are attached only with `--fd-raw` (football-data non-C vs C columns, Standard O/U 2.5). A side counts when p_side × entry > 1, and the value is entry / close − 1. Without both prices, n is 0. Nothing is approximated.
- **Comparison:** the per-row log-loss difference of each candidate against `champion_recipe`, on **common rows only**. The test is a one-sided bootstrap over match dates, with p = (k+1)/(B+1). **Holm** correction runs across the whole family: every (stage, track, market, candidate) comparison in the run.
  - `BETTER_THAN_REFERENCE` needs all of these:
    - Holm p < 0.05;
    - log-loss improvement ≥ 0.001;
    - n ≥ 1,000 common rows;
    - ≥ 20 matchdays;
    - Brier not worse by more than 0.0001.
  - Otherwise the verdict is `NOT_SHOWN_BETTER`, `INSUFFICIENT_N` or `NOT_COMPARABLE`.
  - These thresholds are **reused unchanged from the promotion gate PG-1.0** (`config/promotion_gate.json`, declared 2026-10-25, before any candidate). A test fails if the two files disagree.
  - Leagues under 150 rows are marked in the per-league table and never dropped.
- **Outputs** go to `<out>/<run_id>/`:
  - `manifest.json`: config and its sha, code sha, frame hash, errors and notes;
  - `predictions.parquet`, `metrics.json`, `per_league.csv`, `comparison.csv`;
  - `models/` (holdout-stage models).

  The folder may not be in `models/`, `output/`, `config/` or `src/`. Nothing calls the promotion gate or writes the registry.
- **Promotion is never automatic.** A verdict is a statement about forecasting quality only. The owner reads the artifact and decides.

## 5. Dry-run row counts (committed data, 2026-11-06)

`python scripts/phase2_retrain.py --dry-run`. These are counts only: nothing was fitted.

| Track / market | Eligible | Walk-forward region | Holdout | WF matchdays | Holdout matchdays | Folds | Rows scored by WF | Two-sided price, WF / holdout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard O/U 2.5, BTTS, O1.5, O3.5 | 31,114 | 29,522 | 1,592 | 1,329 | 71 | 11 | 21,217 | O/U 2.5: 29,420 / 1,592; others: 0 (one-sided) |
| standard HT O0.5, HT O1.5 | 19,335 | 17,743 | 1,592 | 771 | 71 | 5 | 9,305 | no price |
| new_format O/U 2.5, BTTS, O1.5, O3.5 | 26,706 | 25,015 | 1,691 | 1,914 | 111 | 18 | 21,692 | O/U 2.5: 2,339 / 946; others: 0 |
| new_format HT O0.5, HT O1.5 | 26,432 | 24,824 | 1,608 | 1,913 | 111 | 18 | 21,502 | no price |

Features: standard 43; new_format 28. New_format leaves out the xG, inside-box and HT block, the same exclusion list as the serving champion (`_NF_DROP`). No column is created when it is absent.

Per league, O/U 2.5 (walk-forward region / holdout). **Bold** = holdout ≥ 150 rows (`min_league_n`).

| Standard | WF | Holdout | | New format | WF | Holdout |
|---|---:|---:|---|---|---:|---:|
| Belgian First Division A | 1,533 | 78 | | Argentina Primera Division | 2,578 | **210** |
| Bundesliga 2 | 1,503 | 79 | | Austrian Bundesliga | 963 | 60 |
| Championship | 2,740 | 140 | | Brazil Serie A | 2,437 | **150** |
| Dutch Eredivisie | 1,511 | 89 | | China Super League | 1,552 | 104 |
| Greek Super League | 1,179 | 53 | | Denmark Superliga | 954 | 72 |
| La Liga 2 | 2,294 | 118 | | Finland Veikkausliiga | 1,083 | 90 |
| League One | 2,737 | 141 | | Ireland Premier Division | 1,188 | 62 |
| League Two | 2,743 | 149 | | Japan J-League | 2,189 | 118 |
| Ligue 2 | 1,719 | 89 | | K-League 1 | 1,230 | 65 |
| National League | 2,679 | **173** | | Mexico Liga MX | 1,690 | 124 |
| Portuguese Primeira Liga | 1,511 | 80 | | Norway Eliteserien | 1,519 | 103 |
| Scottish Championship | 883 | 58 | | Romanian Superliga | 1,583 | 103 |
| Scottish League One | 885 | 57 | | Saudi Pro League | 1,379 | 41 |
| Scottish League Two | 888 | 55 | | Sweden Allsvenskan | 1,513 | 127 |
| Scottish Premiership | 1,132 | 66 | | USA MLS | 3,157 | **262** |
| Serie B | 1,889 | 88 | | | | |
| Turkish Super Lig | 1,696 | 79 | | | | |

In the HT markets, standard has the same holdout counts as above; the walk-forward region is smaller because older seasons have no half-time score. New_format HT counts are within a few rows of O/U, except China (78 holdout), Japan (96) and Romania (68). Full per-league counts for every market are in the dry-run JSON (`--json-out`).

**Which leagues have enough data.** Every league has at least 883 walk-forward rows. On the proposed holdout, only 1 of 17 standard and 3 of 15 new-format leagues reach 150 holdout rows. Each track as a whole passes the 1,000-row minimum (1,592 and 1,691). So per-league holdout numbers on these dates are mostly below the gate's readable size.

**Holdout start alternatives** (same end, 2026-11-08). These are counts only:

| Holdout start | Standard O/U rows (days) | Standard leagues ≥150 | NF O/U rows (days) | NF leagues ≥150 | WF folds std / NF |
|---|---:|---:|---:|---:|---:|
| 2026-01-01 | 4,589 (209) | 14 / 17 | 3,764 (262) | 15 / 15 | 10 / 16 |
| 2026-04-01 | 2,649 (132) | 6 / 17 | 2,889 (180) | 10 / 15 | 11 / 17 |
| **2026-07-01 (proposed)** | 1,592 (71) | 1 / 17 | 1,691 (111) | 3 / 15 | 11 / 18 |

## 6. Imputation

This follows `reports/P9_MODEL_IMPUTATION.md`, options 1 and 2:

- no batch median anywhere;
- no 0.0 fill;
- `hgb` and the `model_market` base are NaN-native;
- `logistic` and `champion_recipe` use a missing indicator per feature, plus an imputer fitted on the training window only and saved with the model;
- a feature with no training value is dropped and listed in the run manifest.

Tests check that a row's probability is identical whether it is scored alone or inside a batch (batch invariance), for `logistic`, `hgb` and `champion_recipe`.

## 7. Known limits (pinned, not hidden)

1. **The holdout is untouched by these candidates, not by people.** The forward ledgers since 2026-08 cover the proposed holdout period, and their results have been read. An earlier start would include even more of that period.
2. **`champion_recipe` receives the test rows** because `train_repaired` takes its holdout through `holdout_start`, which is how the promotion gate calls it. It uses their labels only for the metrics it logs. A test flips every test label and checks that the predictions do not change.
3. **The historical O/U price is closing-preferred and not labelled per row.** That is why market-only is a benchmark and model+market is not deployable on it. A deployable model+market needs an entry-class price; today that means `--fd-raw`, Standard O/U 2.5 only. New-format snapshot timing is not labelled in the frame.
4. **BTTS, Over 1.5 and Over 3.5 have only one side priced** in the frame, and HT markets have no price. So those markets have no market-only or model+market candidate, and no CLV.
5. **Under 2.5** is 1 − P(Over 2.5). Its log loss, Brier and calibration are the same numbers as Over, so they are reported once. CLV is reported for each side.
6. **The `context` feature group** (rest days, home advantage, season venue stats) is checked for parity by distribution only, not row by row. The `form` group is row-validated (P0 parity report).
7. **Dixon-Coles strengths are fixed within a test block.** They are as of the training cutoff and are not updated match by match inside the block.
8. **Compute is unmeasured.** The fold counts are 11 (standard) and 18 (new_format) × 6 markets × 6 candidates, plus the holdout stage. Runtime on Actions is not known until a run, and each track job has a 360-minute timeout. `--only` and `--stage` can split a run.

## 8. What the owner has to decide

| # | Decision | Proposed | Why it matters |
|---|---|---|---|
| 1 | **Holdout dates** (fixed before any run; sign `holdout.approved_by` / `approved_at`) | 2026-07-01 .. 2026-11-08 | Later start: newer and more like serving, but small per league (section 5). Earlier start: readable per league, but more of the period already seen in forward results |
| 2 | **Comparison thresholds** | PG-1.0 values reused: min effect 0.001 log loss, n ≥ 1,000, ≥ 20 matchdays, Brier +0.0001, Holm at 0.05, leagues ≥ 150 | Reused, not chosen here. Confirm, or declare new values **before** the first run |
| 3 | **Which leagues are judged per league** | Only leagues with ≥ 150 holdout rows | On the proposed dates that is National League, Argentina, Brazil and MLS. Every other league appears in the per-league table but is marked too small |
| 4 | **New-format feature set** | The champion's exclusion (no xG, inside-box or HT block) | The HT fill now brings new-format HT coverage to about 88.6% on data-truth-100 (PHASE2_CURRENT_BASELINE). Bring the HT block back for new_format, or keep parity with the serving champion? |
| 5 | **Context features** (rest days, home advantage, season venue) | Included | Their parity is checked by distribution only, not row by row |
| 6 | **Price for model+market and CLV** | Loader price (closing-preferred) = benchmark, not deployable | A deployable blend and real CLV need entry-class prices: provide football-data raw files (`--fd-raw`) for Standard, and decide what counts as an entry price for new_format |
| 7 | **Walk-forward sizes** | 365 matchdays to start, 90-matchday blocks, 60-matchday validation | They set the fold counts above and the compute cost |
| 8 | **Turn it on** | — | After decisions 1-7 and the end of collection (2026-11-08): set the repository variable `PHASE2_RETRAIN_ENABLED=true`, then dispatch `phase2_retrain.yml` |

## 9. Tests (synthetic only)

`python -m pytest tests/test_phase2_retrain_harness.py -q`: **35 passed** in about 31 s. All fits use a generated frame of a few hundred matches. The tests cover:

- **Config and guards:**
  - the committed config is valid and unapproved;
  - its thresholds equal PG-1.0;
  - fitting is refused without approval, without the variable, or with it set to `false`;
  - a refused run creates nothing;
  - the CLI refuses before it loads any data;
  - production folders are refused as run folders.
- **Features:** a missing column is never created, and a price is never a model input.
- **Splits:**
  - cuts fall on whole dates and windows expand;
  - each row is tested at most once;
  - no fold touches the holdout;
  - the holdout trains only on earlier rows.
- **Plan:** the dry-run plan counts rows and fits nothing (every `fit` is patched to fail).
- **Candidates:**
  - batch invariance (logistic, hgb, champion recipe);
  - a feature with no training value is dropped;
  - HGB is NaN-native;
  - the champion recipe ignores the test labels and carries the 1.1 imputer and indicators;
  - Dixon-Coles is coherent (O1.5 ≥ O2.5 ≥ O3.5), its rho stays in bounds, an unseen club or league gets NaN, and HT reads half-time goals;
  - market-only de-vigs two-sided pairs only (power method checked);
  - the model+market stack moves with the validation labels and not with the test labels, and gives NaN without a price.
- **Evaluation:**
  - known log loss and Brier values;
  - the calibration line is recovered;
  - CLV needs real entry and closing prices;
  - Holm values are correct;
  - all four verdicts are reachable.
- **End to end:**
  - a run writes only its run folder;
  - the serving `models/*.pkl` are unchanged;
  - the holdout is refused a second time;
  - the outputs contain no claim words.
- **Workflow:** dispatch only, gated, read-only, no commit or push.
- **Code:** the data-truth default scanner finds 0 hits in the new files.

## 10. How to run

```
python scripts/phase2_retrain.py --dry-run [--json-out <dir outside models/ and output/>/plan.json]
# after owner approval, in GitHub Actions only:
gh workflow run phase2_retrain.yml -f stage=walk_forward    # then stage=holdout, once
```
