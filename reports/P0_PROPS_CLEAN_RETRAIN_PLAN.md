# P0 - Props clean retrain: harness and plan (Phase 2, key `propsharness`)

Built 2026-11-06 on branch `phase2/propsharness` (from `origin/data-truth-100` 97e48c6a). **Build only.**
Nothing was trained, backtested, scored or priced on real data. No edge, ROI or CLV figure was computed,
and no threshold changed. Every number below is a **row count** on the committed data, produced by the
dry-run (`reports/p0_props_harness/dry_run.json`) and recounted a second way by an independent script.
Both methods gave the same counts.

## Verdict

- **The harness is ready to run once the owner decides.** It is gated three times. The workflow is
  dispatch-only and its job needs `vars.PHASE2_RETRAIN_ENABLED == 'true'`. The harness refuses `--run`
  unless `PHASE2_RETRAIN_ENABLED=true` and every owner decision in
  `config/experiments/props_clean_retrain/plan.json` is set. It also stops before any fit when the
  verified share is below the owner's minimum.
- **Today, 0 training rows qualify.** No history row is verified yet: 949,250 rows are `zeros_unverified`
  and the only 1,306 `provider_rule` rows are cup rows from November. 1,289 of those cup rows are in the
  population, and all of them fall after the proposed holdout.
- **After the re-fetch, at most 99.35% of in-scope rows can qualify** (label-only rule). That is an
  upper bound. The re-fetch has not started; it begins after lineups (run-day 6) and should finish on
  2026-11-08.
- **CLV against real two-sided prices is possible for anytime scorer only.** The quote log holds a real
  two-sided closing price for **71 player-markets**, all `goals` (Pinnacle Yes/No via OddsAPI, 38
  fixtures, 2026-10-14 to 2026-11-01). SOT, SOT 2+, assists and cards have **0**, so their CLV would rest
  on the assumed margin. The harness reports those rows separately and flags them.

## 1. What was built

| File | What |
|---|---|
| `player_model/props_harness.py` | Data assembly, verified-row selection (3 feature rules), splits, the 3 candidates, metrics, the gated `run()`, the `dry_run()` and the CLI |
| `player_model/props_harness_clv.py` | Opening price, closing two-sided fair price (power de-vig, median over books), the assumed-margin fallback (flagged), CLV rows and the per-market summary |
| `config/experiments/props_clean_retrain/plan.json` | `owner_decisions` (all `null`), `proposed` (shown by the dry-run only, never used to train), the walk-forward settings, the candidates, `hand_adjustment: "none"` |
| `.github/workflows/props_clean_retrain.yml` | `workflow_dispatch` only. The job is gated on `vars.PHASE2_RETRAIN_ENABLED == 'true'` with `contents: read`, so it commits nothing. It runs the harness tests, then the dry run, then the gated run, and uploads `output/props_harness/` and `models/challengers/props_clean/` as a run artifact |
| `player_model/model.py` | `recipe_estimators()` added. `train()` now builds its two estimators there. This is a refactor only: same classes, same hyperparameters, same keys |
| `tests/test_props_harness.py` | 32 tests, synthetic data only |
| `reports/p0_props_harness/dry_run.json` | The dry run on the committed data |

The harness never writes a serving file (`models/model_player_*.pkl`) and never calls the props promotion
gate. It writes only to `models/challengers/props_clean/` and `output/props_harness/`.

## 2. Data rules

- **Rows:** `player_history.parquet` plus the collected UEFA cup rows it does not hold
  (`cup_history.new_cup_rows`). The re-fetched counts are applied through `history_refetch.with_refetched`.
  Cup fixtures decided after extra time (AET/PEN) are kept as context but are never training rows.
- **Population:** the rows `build_features` trains on. That means a real player id, at least 10 minutes,
  and at least one earlier such row for the player.
- **Verified:** `count_source` is `refetched` or `provider_rule`. `zeros_unverified` rows are context
  only, never training or evaluation rows.
- **Feature rule (owner decision).** A verified row's rolling features come from the player's earlier
  rows, and those may still be unverified. The three rules are:
  - `label_only`: only the row itself must be verified.
  - `window_verified`: the row and the player's previous `ROLLING_N` (5) rows must be verified.
  - `all_prior_verified`: the row and every earlier row of the player must be verified.
- **Features and labels are rebuilt from the raw counts** by `feature_engineering.build_features`. The
  stored derived columns are not read. Labels use the same `_label` rule, so a missing count gives a
  missing label, and that row is dropped for that market.
- **Real profiles.** The 7 profile features (`age`, `age_peak_delta`, `height_cm`,
  `height_aerial_interaction`, `opp_def_mean_height`, `height_diff_vs_opp_def`,
  `height_advantage_score`) are **re-admitted**. `build_features` attaches the collected profiles and
  leaves NaN where none is known. The old placeholders (25 / 180 / 75) are read as missing. This closes
  the P9 train/serve gap. Coverage on the population: real height on 844,560 of 922,479 rows (91.6%)
  and birth date on 898,679 (97.4%), from 20,914 players with a collected profile.

## 3. Targets, candidates, protocol

- **Targets (markets):**
  - `sot` (SOT 1+, `target_sot`)
  - `sot2` (SOT 2+)
  - `goals` (anytime scorer)
  - `assists`
  - `cards` (`yellow_cards >= 1`, as `target_cards`)
- **Candidates**, trained on the same rows and windows:
  1. `recipe_retrained`: the current recipe (logistic + HistGradientBoosting, each Platt-calibrated,
     then averaged), with an imputer fitted on the **fit block only** and saved in the payload. A
     feature with no fit-block value is dropped, not filled. The PFC-1.0 contract's `SEMANTIC_ZERO`
     0.0 fill is **not** used: under the null rule a missing count is unknown.
  2. `hgb_nan_native`: the recipe's HistGradientBoosting alone. Nothing is filled at train or serve
     time.
  3. `player_base_rate`: the player's own rate in the training window, shrunk toward the window's
     market rate. The prior strength is fitted on that window (method of moments). A new player gets
     the market rate.
- **Windows:**
  - The calibration block is the last matchdays of each training window. The fit block holds at most
    85% of the rows, which is the recipe's ratio. The cut falls on a date, so no matchday is in both
    blocks.
  - **Walk-forward by matchday** (one calendar match date across all leagues): 8 folds of 7 matchdays
    before the holdout. Each fold trains on all earlier matchdays. Fold dates are fixed on the calendar
    of all population rows, so they do not move as rows become verified.
  - **Holdout:** one fit on everything before `holdout_start`, scored once on
    [`holdout_start`, `holdout_end`].
- **Metrics per market and candidate:**
  - log loss, Brier and ECE (10 bins)
  - calibration intercept and slope, plus a reliability table
  - on the holdout, log loss minus `player_base_rate` with a week-block bootstrap 95% interval
- **CLV:**
  - The open is the first Yes/Over price seen (best over books).
  - The close is the last answer on or before the match day that holds a **real pair**
    (`fair_basis = two_sided_devig`).
  - Selection: the model's own probability × the opening price > 1. No threshold is applied.
  - `clv = open price × closing fair probability − 1`.
  - Without a pair, the fair price is the assumed overround (`assumed_overround`, `assumed_margin = true`).
    Those rows are summarised **apart** and never pooled with two-sided rows.
  - On two-sided rows, the model's log loss is reported beside the closing fair price's log loss on
    the same rows.
- **Hand adjustment: none.** Owner decision 2026-11-05. The harness scores with each candidate's own
  probability. A test checks the code never calls `hand_adjusted_prob` and that the scored probability
  equals the candidate's own output.

## 4. Dry run on the committed data (counts only)

Source: 950,556 rows (history 876,130 + cup rows 74,426), dated 2022-08-12 to 2026-11-05. Population
922,479 rows. Of these, 920,934 are in scope, meaning on or before the proposed holdout end. Holdout
shown = **PROPOSED** 2026-09-01 to 2026-10-31 (not an owner decision).

**Rows per split** (the same for all 5 markets, because the pre-fix parser stored no blank count):

| View | Rule | pre-holdout | holdout | after holdout | Share of in-scope rows |
|---|---|---:|---:|---:|---:|
| All rows (verified or not) | - | 866,548 | 54,386 | 1,545 | - |
| Verified **today** | label_only | 0 | 0 | 1,289 | 0% |
| Verified **today** | window / all_prior | 0 | 0 | 0 | 0% |
| Verified **after the re-fetch** (upper bound) | label_only | 860,535 | 54,386 | 1,545 | 99.35% |
| Verified **after the re-fetch** (upper bound) | window_verified | 854,084 | 51,930 | 1,435 | 98.38% |
| Verified **after the re-fetch** (upper bound) | all_prior_verified | 848,548 | 48,057 | 1,133 | 97.36% |

- **What "upper bound" means.** Every not-yet-verified row of a fixture that the re-fetch plan has
  proven is counted as verified. The re-fetch plan proves 32,724 of 32,971 fixtures. The other 247
  fixtures (6,354 rows) are not in the fixture lists. The real figure will be lower for three reasons:
  a player the answer cannot prove stays unverified, an empty answer stays unverified, and a re-fetched
  count can become missing, which drops that row's label for the market.
- **The 6,013 in-scope rows that cannot be verified** are all in the refused fixtures.
- **Walk-forward test blocks** run from 2026-07-06 to 2026-08-31, with 1,336 matchdays before the
  holdout. Verified test rows per fold after the re-fetch (upper bound):
  - label_only: 793 / 1,389 / 2,306 / 2,179 / 2,862 / 4,645 / 6,441 / 7,213 (27,828 in total)
  - all_prior_verified: 25,504 in total

  The early-July folds are small because most European leagues are between seasons.

**Two-sided prices (quote log, 281,657 quotes):**

| | Count |
|---|---:|
| Player-markets with a Yes/Over price | 122,874 |
| Player-markets with a real two-sided closing price | 71 (all `goals`, 38 fixtures, 2026-10-14 to 2026-11-01) |
| Two-sided: sot / sot2 / assists / cards | 0 / 0 / 0 / 0 |

P2 section 7 counted 139 pairs. That count is per book per answer. The 71 here is per player-market at
the close, using only answers dated on or before the match day.

**Second-way check.** An independent script recounted the population, the today and after-refetch
verified rows and the shares from the raw parquet files. It used a different method: duplicate-based
first appearance, and fixture-list membership instead of the plan's proofs. Every number above matched
exactly.

## 5. What is ready and what waits

| Item | State |
|---|---|
| Harness modules, CLI dry-run, gated workflow, tests | **Ready** (32 synthetic tests pass) |
| Verified history rows | **Waits for data.** The player re-fetch has 32,724 calls planned and 0 made; it is due to finish 2026-11-08. Rerun `python -m player_model.props_harness --dry-run` afterwards for the real verified counts |
| Owner decisions (section 6) | **Waits for the owner.** All 5 are `null` in `plan.json` |
| Training run | **Waits for owner approval.** It needs the decisions plus the repository variable `PHASE2_RETRAIN_ENABLED=true`, then a manual dispatch of "Props clean retrain (Phase 2, owner-gated)" |
| Run time on the full data | **Unmeasured.** Nothing was run on real data. The job timeout is 360 min. The run is 9 fits × 3 candidates × the number of markets kept, on up to about 900k rows, plus one `build_features` pass over 950k rows |
| Cutover to serving | Not part of this harness. Any promotion stays an owner decision through the props gate |

**Known limits:**
- The feature rules cover only the player's **own** earlier rows. Opponent and team tables are built
  from other players' rows, so they can still include unverified counts. After the re-fetch this is at
  most the 6,013 refused in-scope rows plus whatever the re-fetch refuses.
- The quote log stores a price only the first time it is seen, so a pair whose one side did not move
  is missing there.
- Kickoff times are not in the log, so "close" means the last answer dated on or before the match day.

## 6. Owner decisions needed

1. **Holdout dates.** The proposal is 2026-09-01 to 2026-10-31: 54,386 rows, all of them in proven
   fixtures. Once set, these dates must not move. The holdout has to end on or before 2026-11-01 for
   the 71 two-sided `goals` prices to fall inside it.
2. **Minimum verified share before training may start.** No proposal. The measured upper bound after
   the re-fetch is 99.35% (label_only), 98.38% (window) or 97.36% (all_prior). The real share will be
   known after the 2026-11-08 dry run.
3. **Markets to keep.** The proposal is sot, sot2, goals, assists and cards. Only `goals` has real
   two-sided prices. For the other four, CLV can only be reported on the assumed margin, so their
   evaluation would be log loss, Brier and calibration only.
4. **Feature rule** (needed because a verified row's features come from earlier rows): `label_only`,
   `window_verified` or `all_prior_verified`. The proposal is `all_prior_verified`, the strictest. It
   costs about 2 points of share against `label_only`.
