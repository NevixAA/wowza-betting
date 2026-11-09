# Phase 2 clean retrain - new_format - run_20261109T112024Z (RE-RUN, leak fixed)

Evidence only; no pickles or predictions (public repository; reproducible from the committed data at
the code commit below).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `4a4bd06cf58ba6012c6393568e2ec5c4d1b662f6` |
| Config sha256 | `3a4fd24a89345fe195bd4934fd7a4227f8c27988fc187e1545c451c908e24e92` |
| Data frame | frame_hash `4638ab5ee15f3a7f` (committed data, network blocked). The hash covers the match rows (date, league, teams, goals): the SAME data as the pre-fix run; the features differ by code commit |
| Where it ran | LOCALLY on the owner's PC (Windows, 32 GB RAM), single-threaded (OMP_NUM_THREADS=1; the CPU was shared with other jobs), in a detached worktree at the commit below: `PHASE2_RETRAIN_ENABLED=true python scripts/phase2_retrain.py --out <dir> --stage all --tracks new_format`. RE-RUN by owner decision 2026-11-09 ("Yes, fix leak + refit"): the training features changed (same-day leak fixed, commit 2d1d900b) and the harness no longer loses folds to all-missing columns (commit 4a4bd06c). Local because the comparison step peaks at about 10 GB, which killed the 16 GB GitHub runner twice on 2026-11-08. |
| Started / finished | 2026-11-09T11:20:24+00:00 / 2026-11-09T12:24:40+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) at 2026-11-07T07:03:25Z |
| Replaces | `new_format_run_20261108T214340Z` (pre-fix features) |

**Verdict changes against the pre-fix run: 0 of 108.** champion_recipe holds for every market.
NOT_FITTED folds: 0; ERROR folds: 0 (coverage.csv: folds and rows per
stage × market × candidate). model_market on markets without a two-sided price is fitted with no
stack and scores no row (its note says NOT_FITTED and why); its comparison is NOT_COMPARABLE.

## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | champion_recipe+ht | 2889 | +0.00018 | 1.000 | NOT_SHOWN_BETTER |
| btts | dixon_coles | 2819 | +0.00464 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2889 | +0.00884 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb+ht | 2889 | +0.00922 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2889 | -0.00028 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic+ht | 2889 | +0.00002 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| btts | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | champion_recipe+ht | 2806 | +0.00022 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | dixon_coles | 2764 | +0.00443 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2806 | +0.01044 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb+ht | 2806 | +0.00888 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2806 | -0.00098 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic+ht | 2806 | -0.00079 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | champion_recipe+ht | 2806 | -0.00023 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | dixon_coles | 2764 | +0.00116 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2806 | +0.01425 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb+ht | 2806 | +0.01197 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2806 | -0.00437 | 0.039 | BETTER_THAN_REFERENCE |
| ht_over15 | logistic+ht | 2806 | -0.00438 | 0.039 | BETTER_THAN_REFERENCE |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over15 | champion_recipe+ht | 2889 | -0.00003 | 1.000 | NOT_SHOWN_BETTER |
| over15 | dixon_coles | 2819 | +0.00653 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2889 | +0.00635 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb+ht | 2889 | +0.00537 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2889 | -0.00229 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic+ht | 2889 | -0.00230 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over25 | champion_recipe+ht | 2889 | +0.00098 | 1.000 | NOT_SHOWN_BETTER |
| over25 | dixon_coles | 2819 | +0.00782 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2889 | +0.01749 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb+ht | 2889 | +0.01104 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2889 | -0.00141 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic+ht | 2889 | -0.00046 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 1761 | -0.01044 | 0.204 | NOT_SHOWN_BETTER |
| over25 | model_market | 1761 | -0.00672 | 0.234 | NOT_SHOWN_BETTER |
| over25 | model_market+ht | 1761 | -0.00705 | 0.234 | NOT_SHOWN_BETTER |
| over35 | champion_recipe+ht | 2889 | +0.00327 | 1.000 | NOT_SHOWN_BETTER |
| over35 | dixon_coles | 2819 | +0.00610 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2889 | +0.01523 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb+ht | 2889 | +0.01670 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2889 | +0.00046 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic+ht | 2889 | +0.00057 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
