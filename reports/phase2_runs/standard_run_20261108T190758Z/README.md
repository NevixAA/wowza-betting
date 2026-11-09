# Phase 2 clean retrain - standard - run_20261108T190758Z

Evidence only. The model pickles and predictions.parquet are NOT committed (the repository is public;
the predictions are reproducible from the committed data at the code commit below).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `804799a104bb11ae7e748629eaca178ca40031dc` |
| Config sha256 | `3a4fd24a89345fe195bd4934fd7a4227f8c27988fc187e1545c451c908e24e92` |
| Data | committed data via `src.data_loader.load_all_matches`, network blocked (config `data.source = committed`) |
| Data frame | frame_hash `4638ab5ee15f3a7f` |
| Where it ran | GitHub Actions, run 42512834177 (workflow phase2_retrain.yml, job retrain (standard), attempt 1). |
| Started / finished | 2026-11-08T19:07:58+00:00 / 2026-11-08T19:40:07+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) at 2026-11-07T07:03:25Z |


## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

Negative delta = the candidate has a lower log loss than the current recipe retrained on clean data.

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | dixon_coles | 2199 | +0.00493 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2649 | +0.01743 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2649 | +0.00108 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | dixon_coles | 2170 | +0.00000 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2619 | +0.02166 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2619 | -0.00073 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | dixon_coles | 2170 | -0.00004 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2619 | +0.02007 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2619 | +0.00035 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | dixon_coles | 2199 | +0.00277 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2649 | +0.00890 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2649 | +0.00196 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over25 | dixon_coles | 2199 | +0.00504 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2649 | +0.01629 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2649 | -0.00024 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 2619 | -0.01456 | 0.020 | BETTER_THAN_REFERENCE |
| over25 | model_market | 2619 | -0.01285 | 0.020 | BETTER_THAN_REFERENCE |
| over35 | dixon_coles | 2199 | +0.00287 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2649 | +0.01584 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2649 | -0.00144 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |

## Summary

champion_recipe (the current design, retrained on the repaired data) is kept for every market: no
other model is shown better, except where a row above says BETTER_THAN_REFERENCE. market_only and
model_market use the loader's CLOSING-preferred O/U 2.5 price, a benchmark that is not bettable.
