# P0 - Clean retrain plan and harness (Phase 2 part 4)

Branch `phase2/retrainharness`, from `origin/data-truth-100` at 97e48c6a. Date: 2026-11-06. Updated 2026-11-07 on branch `prep/retrain-decisions` with the owner's decisions, the new_format HT variant and a new dry run.

**This is a build only.** No model was trained on real data. No backtest or walk-forward was run on real data, nothing about edge, ROI or profit was computed, and no threshold was changed. Every component was tested on small synthetic frames. The only real-data step was the dry run, which counts rows and lists columns.

## Owner decisions (2026-11-07)

| Decision | Value | Where |
|---|---|---|
| Holdout | **2026-04-01 .. 2026-11-08, approved** (`approved_by` "Nevo (owner)", `approved_at` 2026-11-07T07:03:25Z) | `config/phase2_retrain.json` `holdout` |
| Comparison rules | **PG-1.0 reused unchanged** (min effect 0.001 log loss, n ≥ 1,000, ≥ 20 matchdays, Brier +0.0001, Holm at 0.05, leagues ≥ 150). A test fails if they differ from `config/promotion_gate.json` | `comparison` |
| New-format HT block | **Added as a candidate variant** (`owner_decisions.new_format_ht_variant = true`), never forced in: the holdout decides | section 3a |

## Verdict

- **Ready:** the harness, the pre-registered protocol (`config/phase2_retrain.json`), the dry-run CLI and a dispatch-only workflow. The workflow is gated twice: by the repository variable and by the owner's holdout approval in the config. The holdout is now approved, so the repository variable `PHASE2_RETRAIN_ENABLED` is the only gate left.
- **Waiting for data:** collection ends 2026-11-08. The dry run must be repeated on the final committed data. The counts below come from the 2026-11-07 commit and will move.
- **Waiting for the owner:** the remaining questions in section 8 (context features, the price for model+market and CLV, walk-forward sizes) and switching the variable on. Until `PHASE2_RETRAIN_ENABLED` is `true`, the code refuses to fit; a fixture copy of the config with `approved_by` emptied is still refused. Tests check both.

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
| `tests/test_phase2_retrain_harness.py` | 45 synthetic tests (section 9) |

## 2. Data

The run uses the loader's served frame, built from the **committed** data: `src.data_loader.load_all_matches`, with the football-data download turned off and the network blocked. Then come COVID exclusion (per league, `config.covid_season_mask`) and `src.feature_engineering.build_features` (point-in-time, `shift(1)` windows, the parity-validated form block). Every loader correction applies because the loader applies it:

- identity and one-match-one-row;
- history corrections, including the Argentina cup rows;
- written-back af_history statistics are not served;
- impossible shot counts are set to missing;
- O/U pairs with a negative margin are flagged.

The loader's `training_coverage.json` report is not rewritten. The frame hash is recorded in each run manifest.

Dry run on the 2026-11-07 commit (`prep/retrain-decisions` at 17139dcc6, run on a scratch copy of the tree):

| | Value |
|---|---|
| Loaded rows | 70,744 |
| COVID rows excluded | 12,271 |
| Feature rows | 58,473 (2019-02-15 .. 2026-10-30) |
| Frame hash | `a4cb912462c730d2` |
| Written-back cells not served (home shots / home xG / home inside-box) | 11,787 / 9,597 / 10,998 blanked, of which 6,596 / 6,900 / 9,619 were refilled from a provider source; the rest stay missing |
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

### 3a. The new_format HT variant (owner 2026-11-07)

With `owner_decisions.new_format_ht_variant = true`, every new_format candidate that reads features (`champion_recipe`, `logistic`, `hgb`, `model_market`) is trained and scored twice:

- **`<name>`**: the champion exclusion list (`features.track_exclude.new_format`, the serving `_NF_DROP`), 28 features;
- **`<name>+ht`**: the same list with the **HT block restored**, 39 features. The HT block is the 11 HT columns of the exclusion list (`home/away_ht_scored_last5`, `home/away_ht_conceded_last5`, `home/away_ht_over05_rate`, `home/away_ht_over15_rate`, `home/away_ht_attack_str`, `combined_ht_goals_avg`) plus `home/away_has_ht_history`. xG and inside-box stay excluded.

`market_only` and `dixon_coles` read no features and run once. Both variants are separate candidates in the one Holm family, each compared against `champion_recipe` (so new_format has 9 comparisons per stage and market instead of 5). Standard is unchanged.

`home/away_has_ht_history` is **not** in the champion exclusion list, so it is already in the 28-feature list; restoring it changes nothing, and the two lists differ by exactly the 11 HT columns.

**Refusal, never a silent skip.** If the flag is on and the variant cannot run, fitting is refused before anything is written: no HT column in the exclusion list (the variant would equal the champion list), an HT block column outside the included feature groups, an HT block column missing from the feature frame, the new_format track not configured, or no enabled feature candidate. The flag must be `true` or `false`. Code: `src/retrain_harness/protocol.py` (`ht_block`, `ht_variant_blockers`, `feature_sets`, `candidate_runs`), `runner.py`.

`src/poisson.py` (`dixon_coles_p_over25`, rho fixed at -0.08) only computes a feature. It is not a fitted model, so the harness does not use it as a candidate. It already feeds `p_over25_poisson_dc` through `build_features`.

**Price class.** The committed frame has one O/U 2.5 pair per match. For football-data rows it is closing-preferred (AvgC first, entry columns as fallback). For new-format rows it comes from captured snapshots. So `market_only` is a **closing-market benchmark**, and `model_market` built on it is marked `deployable: false` in every output, because a closing price is not known at bet time. Price-derived columns (`bookmaker_overround`, `implied_prob_*`) are **never** model inputs.

## 4. Protocol

- **Splits are on whole dates** (one matchday = one date with an eligible row in the track):
  - expanding walk-forward;
  - the first test block starts after 365 matchdays;
  - blocks are 90 matchdays;
  - the last 60 matchdays of each training window form the validation block, used only for the model+market stack.
- **Untouched final holdout: 2026-04-01 .. 2026-11-08 (approved by the owner 2026-11-07; proposed earlier: 2026-07-01).** No fold contains a holdout row. The holdout stage fits on the whole walk-forward region and scores the holdout once. It writes `holdout_opened.json` before scoring and refuses a second holdout stage in the same run folder. Rows after 2026-11-08 are not used.
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

## 5. Dry-run row counts (committed data, 2026-11-07, approved holdout 2026-04-01 .. 2026-11-08)

`python scripts/phase2_retrain.py --dry-run --json-out <scratch>/plan.json`, run on a scratch copy of the tree (the repository's `output/` was not written). These are counts only: nothing was fitted. Config sha256 `3a4fd24a89345fe1…`; the dry run printed `approved: YES`.

| Track / market | Eligible | Walk-forward region | Holdout | WF matchdays | Holdout matchdays | Folds | Rows scored by WF | Two-sided price, WF / holdout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard O/U 2.5, BTTS, O1.5, O3.5 | 31,114 | 28,465 | 2,649 | 1,268 | 132 | 11 | 20,160 | O/U 2.5: 28,393 / 2,619; others: 0 (one-sided) |
| standard HT O0.5, HT O1.5 | 19,335 | 16,716 | 2,619 | 722 | 120 | 4 | 8,278 | no price |
| new_format O/U 2.5, BTTS, O1.5, O3.5 | 26,706 | 23,817 | 2,889 | 1,845 | 180 | 17 | 20,494 | O/U 2.5: 1,701 / 1,684; others: 0 |
| new_format HT O0.5, HT O1.5 | 26,432 | 23,626 | 2,806 | 1,844 | 180 | 17 | 20,304 | no price |

No eligible row falls after 2026-11-08 (the frame ends 2026-10-30), so 0 rows are unused.

Features: standard 43; new_format 28 (the champion exclusion list: no xG, inside-box or HT block, as `_NF_DROP`); new_format `+ht` variant 39. No column is created when it is absent.

**HT variant rows.** The `+ht` variant trains and scores the same rows as its base (eligibility never reads a feature). Rows on which all 11 restored HT values are present:

| new_format market | WF region | of which HT block complete | Holdout | of which HT block complete |
|---|---:|---:|---:|---:|
| O/U 2.5, BTTS, O1.5, O3.5 | 23,817 | 23,658 | 2,889 | 2,860 |
| HT O0.5, HT O1.5 | 23,626 | 23,616 | 2,806 | 2,806 |

Per league, walk-forward region / holdout. **Bold** = holdout ≥ 150 rows (`min_league_n`).

| Standard | O/U 2.5 WF | O/U 2.5 holdout | HT O0.5 holdout | | New format | O/U 2.5 WF | O/U 2.5 holdout | HT O0.5 holdout |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Belgian First Division A | 1,462 | 149 | 149 | | Argentina Primera Division | 2,488 | **300** | **300** |
| Bundesliga 2 | 1,439 | 143 | 142 | | Austrian Bundesliga | 915 | 108 | 108 |
| Championship | 2,651 | **229** | **225** | | Brazil Serie A | 2,337 | **250** | **250** |
| Dutch Eredivisie | 1,457 | 143 | 143 | | China Super League | 1,448 | **208** | **182** |
| Greek Super League | 1,125 | 107 | 107 | | Denmark Superliga | 906 | 120 | 120 |
| La Liga 2 | 2,181 | **231** | **225** | | Finland Veikkausliiga | 1,001 | **172** | **172** |
| League One | 2,646 | **232** | **227** | | Ireland Premier Division | 1,115 | 135 | 135 |
| League Two | 2,661 | **231** | **227** | | Japan J-League | 2,095 | **212** | **190** |
| Ligue 2 | 1,665 | 143 | 143 | | K-League 1 | 1,169 | 126 | 126 |
| National League | 2,615 | **237** | **237** | | Mexico Liga MX | 1,630 | **184** | **184** |
| Portuguese Primeira Liga | 1,446 | 145 | 145 | | Norway Eliteserien | 1,443 | **179** | **179** |
| Scottish Championship | 856 | 85 | 85 | | Romanian Superliga | 1,518 | **168** | 133 |
| Scottish League One | 859 | 83 | 83 | | Saudi Pro League | 1,307 | 113 | 113 |
| Scottish League Two | 863 | 80 | 80 | | Sweden Allsvenskan | 1,432 | **208** | **208** |
| Scottish Premiership | 1,090 | 108 | 108 | | USA MLS | 3,013 | **406** | **406** |
| Serie B | 1,818 | **159** | 149 | | | | | |
| Turkish Super Lig | 1,631 | 144 | 144 | | | | | |

BTTS, O1.5 and O3.5 have the same rows as O/U 2.5, and HT O1.5 the same as HT O0.5. Full per-league counts for every market are in the dry-run JSON (`--json-out`).

**Which leagues have enough data.** Leagues at or above 150 holdout rows:

| Market | Standard | New format |
|---|---:|---:|
| O/U 2.5, BTTS, O1.5, O3.5 | **6 / 17** | **10 / 15** |
| HT O0.5, HT O1.5 | **5 / 17** (Serie B drops to 149) | **9 / 15** (Romania drops to 133) |

Belgian First Division A has 149 holdout rows, one short. Every league has at least 856 walk-forward rows, and each track as a whole passes the 1,000-row minimum (2,649 and 2,889). These counts match the 2026-04-01 row of the alternatives table in the 2026-11-06 version of this report.

**Holdout start alternatives** (same end, 2026-11-08; counted 2026-11-06, before the decision):

| Holdout start | Standard O/U rows (days) | Standard leagues ≥150 | NF O/U rows (days) | NF leagues ≥150 | WF folds std / NF |
|---|---:|---:|---:|---:|---:|
| 2026-01-01 | 4,589 (209) | 14 / 17 | 3,764 (262) | 15 / 15 | 10 / 16 |
| **2026-04-01 (approved)** | 2,649 (132) | 6 / 17 | 2,889 (180) | 10 / 15 | 11 / 17 |
| 2026-07-01 (proposed earlier) | 1,592 (71) | 1 / 17 | 1,691 (111) | 3 / 15 | 11 / 18 |

## 6. Imputation

This follows `reports/P9_MODEL_IMPUTATION.md`, options 1 and 2:

- no batch median anywhere;
- no 0.0 fill;
- `hgb` and the `model_market` base are NaN-native;
- `logistic` and `champion_recipe` use a missing indicator per feature, plus an imputer fitted on the training window only and saved with the model;
- a feature with no training value is dropped and listed in the run manifest.

Tests check that a row's probability is identical whether it is scored alone or inside a batch (batch invariance), for `logistic`, `hgb` and `champion_recipe`.

## 7. Known limits (pinned, not hidden)

1. **The holdout is untouched by these candidates, not by people.** The forward ledgers since 2026-08 cover part of the approved holdout period (2026-04-01 .. 2026-11-08), and their results have been read.
2. **`champion_recipe` receives the test rows** because `train_repaired` takes its holdout through `holdout_start`, which is how the promotion gate calls it. It uses their labels only for the metrics it logs. A test flips every test label and checks that the predictions do not change.
3. **The historical O/U price is closing-preferred and not labelled per row.** That is why market-only is a benchmark and model+market is not deployable on it. A deployable model+market needs an entry-class price; today that means `--fd-raw`, Standard O/U 2.5 only. New-format snapshot timing is not labelled in the frame.
4. **BTTS, Over 1.5 and Over 3.5 have only one side priced** in the frame, and HT markets have no price. So those markets have no market-only or model+market candidate, and no CLV.
5. **Under 2.5** is 1 − P(Over 2.5). Its log loss, Brier and calibration are the same numbers as Over, so they are reported once. CLV is reported for each side.
6. **The `context` feature group** (rest days, home advantage, season venue stats) is checked for parity by distribution only, not row by row. The `form` group is row-validated (P0 parity report).
7. **Dixon-Coles strengths are fixed within a test block.** They are as of the training cutoff and are not updated match by match inside the block.
8. **Compute is unmeasured.** The fold counts are 11 (standard; 4 for the HT markets) × 6 markets × 6 candidates and 17 (new_format) × 6 markets × 10 candidates (6 plus the 4 `+ht` variants), plus the holdout stage. Runtime on Actions is not known until a run, and each track job has a 360-minute timeout. `--only` and `--stage` can split a run.

## 8. What the owner has to decide

| # | Decision | Status / proposed | Why it matters |
|---|---|---|---|
| 1 | **Holdout dates** | **DECIDED 2026-11-07: 2026-04-01 .. 2026-11-08, approved** | 6 / 17 standard and 10 / 15 new-format leagues reach 150 holdout rows on O/U 2.5 (section 5) |
| 2 | **Comparison thresholds** | **DECIDED 2026-11-07: PG-1.0 reused unchanged** (min effect 0.001 log loss, n ≥ 1,000, ≥ 20 matchdays, Brier +0.0001, Holm at 0.05, leagues ≥ 150) | A test fails if the config and `config/promotion_gate.json` disagree |
| 3 | **Which leagues are judged per league** | Only leagues with ≥ 150 holdout rows (`min_league_n` of PG-1.0) | On the approved dates: standard Championship, La Liga 2, League One, League Two, National League, Serie B; new format Argentina, Brazil, China, Finland, Japan, Mexico, Norway, Romania, Sweden, MLS (O/U 2.5). Every other league appears in the per-league table, marked too small |
| 4 | **New-format feature set** | **DECIDED 2026-11-07: both.** The champion exclusion list, plus a `+ht` variant with the HT block restored (section 3a) | The holdout decides between them; nothing is forced in |
| 5 | **Context features** (rest days, home advantage, season venue) | Included | Their parity is checked by distribution only, not row by row |
| 6 | **Price for model+market and CLV** | Loader price (closing-preferred) = benchmark, not deployable | A deployable blend and real CLV need entry-class prices: provide football-data raw files (`--fd-raw`) for Standard, and decide what counts as an entry price for new_format |
| 7 | **Walk-forward sizes** | 365 matchdays to start, 90-matchday blocks, 60-matchday validation | They set the fold counts above and the compute cost |
| 8 | **Turn it on** | — | After decisions 5-7 and the end of collection (2026-11-08): repeat the dry run on the final data, set the repository variable `PHASE2_RETRAIN_ENABLED=true`, then dispatch `phase2_retrain.yml` |

## 9. Tests (synthetic only)

2026-11-07 (`prep/retrain-decisions`): `tests/test_phase2_retrain_harness.py` **45 passed** (35 before; the approval tests now use fixture configs, plus the committed-decisions test and the HT-variant tests). `tests/test_props_harness.py` checks the committed `feature_rule = all_prior_verified` and that `--run` still refuses while holdout dates, `min_verified_share` or markets are null. Full suite: **2,825 passed**, 36 skipped, 7 xfailed (baseline 2,810). `python -m src.integrity.data_truth --gate`: STOPLIGHT GREEN (0 new defaults). `scripts/dashboard_manifest.py --check`: no missing code or data. The dry run ran on a scratch copy of the tree; `output/` files touched by the checks were restored. All fits use a generated frame of a few hundred matches. The tests cover:

- **Config and guards:**
  - the committed config holds the owner's decisions: holdout 2026-04-01 .. 2026-11-08, approval set, `new_format_ht_variant` true, and only the repository variable still blocks fitting;
  - its thresholds still equal PG-1.0 (`gate_version` PG-1.0), reused and not changed;
  - on fixture copies of the config: fitting is refused when `approved_by` (or `approved_at`) is empty, without the variable, or with it set to `false`;
  - a refused run creates nothing;
  - the CLI refuses before it loads any data (unapproved fixture config; committed config without the variable; a variant that cannot run);
  - production folders are refused as run folders.
- **HT variant (synthetic):**
  - `+ht` restores the HT block only; xG stays excluded and no price enters; standard and flag-off configs get no variant;
  - each feature candidate appears as `<name>` and `<name>+ht`; `market_only` and `dixon_coles` once;
  - the dry-run plan and printout list both feature lists, equal row counts and the HT-complete counts;
  - fitting is refused (nothing written) when the frame lacks an HT column, the exclusion list holds no HT column, an HT column is outside the feature groups, or new_format is not configured; a non-boolean flag is a config error;
  - end to end: both variants are scored in both stages, `hgb` and `hgb+ht` differ, every variant is compared against `champion_recipe` in one Holm family, and each has its own model file.
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
