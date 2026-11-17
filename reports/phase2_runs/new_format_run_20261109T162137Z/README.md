# Phase 2 clean retrain - new_format - run_20261109T162137Z (RE-RUN on the MERGED frame)

Owner 2026-11-09, "Yes, re-run": the Phase 2 comparison on the merged (improved) data-truth-100 frame
(P7 collection + results sync). Evidence only; no pickles or predictions (public repository).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `a29fdb1ef5f60ca096a3140a919d8124f598b399` |
| Data frame | frame_hash `8d973b3f2f1f04f7` (71014 loaded rows; previous runs: `4638ab5ee15f3a7f`, 70,744) |
| Where it ran | LOCALLY on the owner's PC, single-threaded (OMP/OPENBLAS/MKL_NUM_THREADS=1), detached worktree at the commit above; the comparison step peaks at about 10 GB |
| Started / finished | 2026-11-09T16:21:37+00:00 / 2026-11-09T17:27:00+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) |
| Compared with | `new_format_run_20261109T112024Z` (the same code path before the merge) |

**Verdict changes against `new_format_run_20261109T112024Z`: 4 of 108.** NOT_FITTED 0, ERROR 0.

| Stage | Market | Candidate | Before the merge | On the merged frame |
|---|---|---|---|---|
| holdout | ht_over15 | logistic | BETTER_THAN_REFERENCE (-0.00437) | NOT_SHOWN_BETTER (-0.00329, p_holm 0.204, n 2986) |
| holdout | ht_over15 | logistic+ht | BETTER_THAN_REFERENCE (-0.00438) | NOT_SHOWN_BETTER (-0.00327, p_holm 0.520, n 2986) |
| holdout | over25 | model_market+ht | NOT_SHOWN_BETTER (-0.00705) | BETTER_THAN_REFERENCE (-0.00750, p_holm 0.039, n 1866) |
| walk_forward | over15 | logistic | BETTER_THAN_REFERENCE (-0.00341) | NOT_SHOWN_BETTER (-0.00297, p_holm 0.144, n 20494) |

## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | champion_recipe+ht | 2995 | +0.00008 | 1.000 | NOT_SHOWN_BETTER |
| btts | dixon_coles | 2920 | +0.00524 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2995 | +0.01127 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb+ht | 2995 | +0.01463 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2995 | +0.00006 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic+ht | 2995 | +0.00006 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| btts | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | champion_recipe+ht | 2986 | +0.00039 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | dixon_coles | 2911 | +0.00567 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2986 | +0.01082 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb+ht | 2986 | +0.01048 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2986 | +0.00002 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic+ht | 2986 | +0.00031 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | champion_recipe+ht | 2986 | +0.00085 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | dixon_coles | 2911 | +0.00185 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2986 | +0.01232 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb+ht | 2986 | +0.01441 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2986 | -0.00329 | 0.204 | NOT_SHOWN_BETTER |
| ht_over15 | logistic+ht | 2986 | -0.00327 | 0.520 | NOT_SHOWN_BETTER |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over15 | champion_recipe+ht | 2995 | -0.00044 | 1.000 | NOT_SHOWN_BETTER |
| over15 | dixon_coles | 2920 | +0.00755 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2995 | +0.00912 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb+ht | 2995 | +0.00669 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2995 | -0.00182 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic+ht | 2995 | -0.00172 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over25 | champion_recipe+ht | 2995 | +0.00059 | 1.000 | NOT_SHOWN_BETTER |
| over25 | dixon_coles | 2920 | +0.01062 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2995 | +0.01474 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb+ht | 2995 | +0.01291 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2995 | +0.00010 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic+ht | 2995 | +0.00057 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 1866 | -0.00910 | 0.076 | NOT_SHOWN_BETTER |
| over25 | model_market | 1866 | -0.00317 | 1.000 | NOT_SHOWN_BETTER |
| over25 | model_market+ht | 1866 | -0.00750 | 0.039 | BETTER_THAN_REFERENCE |
| over35 | champion_recipe+ht | 2995 | +0.00068 | 1.000 | NOT_SHOWN_BETTER |
| over35 | dixon_coles | 2920 | +0.00642 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2995 | +0.01417 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb+ht | 2995 | +0.01245 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2995 | +0.00007 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic+ht | 2995 | +0.00031 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
