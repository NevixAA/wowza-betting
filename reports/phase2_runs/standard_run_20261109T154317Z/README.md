# Phase 2 clean retrain - standard - run_20261109T154317Z (RE-RUN on the MERGED frame)

Owner 2026-11-09, "Yes, re-run": the Phase 2 comparison on the merged (improved) data-truth-100 frame
(P7 collection + results sync). Evidence only; no pickles or predictions (public repository).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `a29fdb1ef5f60ca096a3140a919d8124f598b399` |
| Data frame | frame_hash `8d973b3f2f1f04f7` (71014 loaded rows; previous runs: `4638ab5ee15f3a7f`, 70,744) |
| Where it ran | LOCALLY on the owner's PC, single-threaded (OMP/OPENBLAS/MKL_NUM_THREADS=1), detached worktree at the commit above; the comparison step peaks at about 10 GB |
| Started / finished | 2026-11-09T15:43:17+00:00 / 2026-11-09T16:19:09+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) |
| Compared with | `standard_run_20261109T104641Z` (the same code path before the merge) |

**Verdict changes against `standard_run_20261109T104641Z`: 0 of 60.** NOT_FITTED 0, ERROR 0.

| Stage | Market | Candidate | Before the merge | On the merged frame |
|---|---|---|---|---|
| - | - | - | none | none |

## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | dixon_coles | 2313 | +0.00479 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2813 | +0.01935 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2813 | +0.00150 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | dixon_coles | 2284 | +0.00075 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2783 | +0.02179 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2783 | -0.00051 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | dixon_coles | 2284 | +0.00117 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2783 | +0.01888 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2783 | +0.00044 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | dixon_coles | 2313 | +0.00396 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2813 | +0.00969 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2813 | +0.00195 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over25 | dixon_coles | 2313 | +0.00566 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2813 | +0.01001 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2813 | -0.00014 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 2783 | -0.01380 | 0.020 | BETTER_THAN_REFERENCE |
| over25 | model_market | 2783 | -0.01232 | 0.020 | BETTER_THAN_REFERENCE |
| over35 | dixon_coles | 2313 | +0.00313 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2813 | +0.01667 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2813 | -0.00105 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |
