# Phase 2 clean retrain - new_format - run_20261108T214340Z

Evidence only. The model pickles and predictions.parquet are NOT committed (the repository is public;
the predictions are reproducible from the committed data at the code commit below).

| | |
|---|---|
| Harness | phase2-retrain-1.0 |
| Code commit | `804799a104bb11ae7e748629eaca178ca40031dc` |
| Config sha256 | `3a4fd24a89345fe195bd4934fd7a4227f8c27988fc187e1545c451c908e24e92` |
| Data | committed data via `src.data_loader.load_all_matches`, network blocked (config `data.source = committed`) |
| Data frame | frame_hash `4638ab5ee15f3a7f` |
| Where it ran | LOCALLY on the owner's PC (Windows, 32 GB RAM): `PHASE2_RETRAIN_ENABLED=true python scripts/phase2_retrain.py --out <dir> --stage all --tracks new_format` in a detached worktree at the same commit. The GitHub job (run 42512834177, job retrain (new_format)) was stopped twice inside 'Fit and score' (20:10 and 21:37 UTC) with no log and no artifact; the local run showed the comparison step peaking at about 10 GB of memory, the likely cause on a 16 GB runner. |
| Started / finished | 2026-11-08T21:43:40+00:00 / 2026-11-08T23:45:40+00:00 |
| Holdout | 2026-04-01 .. 2026-11-08, approved by Nevo (owner) at 2026-11-07T07:03:25Z |
| Local run archive sha256 | `ed14069c1a5b5e35cb0a15aa9778a4868018facddbe6bb3253ca9be273d68f09` |

## Holdout verdicts (reference = champion_recipe, Holm across the run's family)

Negative delta = the candidate has a lower log loss than the current recipe retrained on clean data.

| Market | Candidate | Rows | Delta log loss | p (Holm) | Verdict |
|---|---|---|---|---|---|
| btts | champion_recipe+ht | 2889 | +0.00065 | 1.000 | NOT_SHOWN_BETTER |
| btts | dixon_coles | 2819 | +0.00482 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb | 2889 | +0.00939 | 1.000 | NOT_SHOWN_BETTER |
| btts | hgb+ht | 2889 | +0.01364 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic | 2889 | -0.00011 | 1.000 | NOT_SHOWN_BETTER |
| btts | logistic+ht | 2889 | +0.00019 | 1.000 | NOT_SHOWN_BETTER |
| btts | market_only | 0 |  |  | NOT_COMPARABLE |
| btts | model_market | 0 |  |  | NOT_COMPARABLE |
| btts | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | champion_recipe+ht | 2806 | +0.00047 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | dixon_coles | 2764 | +0.00434 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb | 2806 | +0.00781 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | hgb+ht | 2806 | +0.00901 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic | 2806 | -0.00113 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | logistic+ht | 2806 | -0.00096 | 1.000 | NOT_SHOWN_BETTER |
| ht_over05 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over05 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | champion_recipe+ht | 2806 | +0.00002 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | dixon_coles | 2764 | +0.00118 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb | 2806 | +0.01294 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | hgb+ht | 2806 | +0.01190 | 1.000 | NOT_SHOWN_BETTER |
| ht_over15 | logistic | 2806 | -0.00427 | 0.039 | BETTER_THAN_REFERENCE |
| ht_over15 | logistic+ht | 2806 | -0.00433 | 0.039 | BETTER_THAN_REFERENCE |
| ht_over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| ht_over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over15 | champion_recipe+ht | 2889 | +0.00006 | 1.000 | NOT_SHOWN_BETTER |
| over15 | dixon_coles | 2819 | +0.00628 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb | 2889 | +0.00352 | 1.000 | NOT_SHOWN_BETTER |
| over15 | hgb+ht | 2889 | +0.00651 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic | 2889 | -0.00256 | 1.000 | NOT_SHOWN_BETTER |
| over15 | logistic+ht | 2889 | -0.00256 | 1.000 | NOT_SHOWN_BETTER |
| over15 | market_only | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market | 0 |  |  | NOT_COMPARABLE |
| over15 | model_market+ht | 0 |  |  | NOT_COMPARABLE |
| over25 | champion_recipe+ht | 2889 | +0.00010 | 1.000 | NOT_SHOWN_BETTER |
| over25 | dixon_coles | 2819 | +0.00778 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb | 2889 | +0.01234 | 1.000 | NOT_SHOWN_BETTER |
| over25 | hgb+ht | 2889 | +0.00965 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic | 2889 | -0.00140 | 1.000 | NOT_SHOWN_BETTER |
| over25 | logistic+ht | 2889 | -0.00041 | 1.000 | NOT_SHOWN_BETTER |
| over25 | market_only | 1761 | -0.01054 | 0.204 | NOT_SHOWN_BETTER |
| over25 | model_market | 1761 | -0.00658 | 0.335 | NOT_SHOWN_BETTER |
| over25 | model_market+ht | 1761 | -0.00670 | 0.455 | NOT_SHOWN_BETTER |
| over35 | champion_recipe+ht | 2889 | +0.00272 | 1.000 | NOT_SHOWN_BETTER |
| over35 | dixon_coles | 2819 | +0.00531 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb | 2889 | +0.01263 | 1.000 | NOT_SHOWN_BETTER |
| over35 | hgb+ht | 2889 | +0.01621 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic | 2889 | -0.00033 | 1.000 | NOT_SHOWN_BETTER |
| over35 | logistic+ht | 2889 | -0.00022 | 1.000 | NOT_SHOWN_BETTER |
| over35 | market_only | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market | 0 |  |  | NOT_COMPARABLE |
| over35 | model_market+ht | 0 |  |  | NOT_COMPARABLE |

## Summary

champion_recipe (the current design, retrained on the repaired data) is kept for every market: no
other model is shown better, except where a row above says BETTER_THAN_REFERENCE. market_only and
model_market use the loader's CLOSING-preferred O/U 2.5 price, a benchmark that is not bettable.
