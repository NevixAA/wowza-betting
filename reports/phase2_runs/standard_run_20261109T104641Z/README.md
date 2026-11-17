# Phase 2 clean retrain - standard - run_20261109T104641Z (RE-RUN, leak fixed)

Evidence only; no pickles or predictions (public repository; reproducible from the committed data at
the code commit below).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `4a4bd06cf58ba6012c6393568e2ec5c4d1b662f6` |
| Config sha256 | `3a4fd24a89345fe195bd4934fd7a4227f8c27988fc187e1545c451c908e24e92` |
| Data frame | frame_hash `4638ab5ee15f3a7f` (committed data, network blocked). The hash covers the match rows (date, league, teams, goals): the SAME data as the pre-fix run; the features differ by code commit |
| Where it ran | LOCALLY on the owner's PC (Windows, 32 GB RAM), single-threaded (OMP_NUM_THREADS=1; the CPU was shared with other jobs), in a detached worktree at the commit below: `PHASE2_RETRAIN_ENABLED=true python scripts/phase2_retrain.py --out <dir> --stage all --tracks standard`. RE-RUN by owner decision 2026-11-09 ("Yes, fix leak + refit"): the training features changed (same-day leak fixed, commit 2d1d900b) and the harness no longer loses folds to all-missing columns (commit 4a4bd06c). Local because the comparison step peaks at about 10 GB, which killed the 16 GB GitHub runner twice on 2026-11-08. |
| Started / finished | 2026-11-09T10:46:41+00:00 / 2026-11-09T11:17:51+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) at 2026-11-07T07:03:25Z |
| Replaces | `standard_run_20261108T190758Z` (pre-fix features) |

**Verdict changes against the pre-fix run: 0 of 60.** champion_recipe holds for every market.
NOT_FITTED folds: 0; ERROR folds: 0 (coverage.csv: folds and rows per
stage × market × candidate). model_market on markets without a two-sided price is fitted with no
stack and scores no row (its note says NOT_FITTED and why); its comparison is NOT_COMPARABLE.

## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | dixon_coles | 2199 | +0.00489 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2649 | +0.01577 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2649 | +0.00101 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | dixon_coles | 2170 | +0.00039 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2619 | +0.02490 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2619 | -0.00045 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | dixon_coles | 2170 | -0.00009 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2619 | +0.01929 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2619 | +0.00022 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | dixon_coles | 2199 | +0.00301 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2649 | +0.01089 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2649 | +0.00202 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over25 | dixon_coles | 2199 | +0.00510 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2649 | +0.01379 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2649 | -0.00025 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 2619 | -0.01461 | 0.020 | BETTER_THAN_REFERENCE |
| over25 | model_market | 2619 | -0.01256 | 0.020 | BETTER_THAN_REFERENCE |
| over35 | dixon_coles | 2199 | +0.00309 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2649 | +0.01679 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2649 | -0.00116 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |
