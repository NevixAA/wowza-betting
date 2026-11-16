# Props imputer migration: no batch statistic, explicit failure (P0 "real data only", 2026-11-15)

Commit `c41165a49` on `fix/props-data-audit`. File: `player_model/model.py`.

## Root cause

`model._prep`, the legacy preprocessing behind every pre-R2 props payload:

```python
X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())          # the median of the batch being SCORED
```

* At serving time the "batch" is one run's upcoming players, so a row's probability depended on
  which other players were in the run. A live statistic reached the model.
* In training (`train(recipe="legacy")`) the batch was the whole frame, holdout included: holdout
  statistics leaked into the fit.
* Before 2026-11-01 an absent column was also created as 0.0 (an invented observation).

## What changed

| path | before | after |
|---|---|---|
| `_prep` | numeric matrix, `fillna(X.median())` | the raw matrix: every cell numeric, missing and ABSENT columns NaN, **nothing filled** |
| legacy payload scoring | batch-median-filled matrix -> logistic + HGB | raw matrix -> HGB (NaN-native) on incomplete rows, logistic only on complete rows (`_ensemble_proba`); a row no estimator can score gets no probability |
| clean-harness payload (sot, sot2, goals, assists: `hgb_nan_native`; cards: `recipe_retrained`) | the harness model's own `predict` | unchanged (NaN-native, or the medians persisted in the artifact from its fit block) |
| repaired R2 payload | persisted `PropsImputer` (fit block only) | unchanged |
| any payload listing an unsupported feature (`model.UNSUPPORTED_FEATURES` = `config.SETPIECE_PROXIES_REMOVED`) | scored | **refused**: p NaN for every row, `imputer_version = REFUSED_UNSUPPORTED_FEATURES` |
| a row missing a `REQUIRED_FAIL_ROW` input (props_contract: n_prev_games, minutes_pg, starter_rate, the position flags, the core rates goals/assists/shots/sot/cards per game) | scored (legacy filled it; clean models scored it NaN-native) | **no probability**, `required_fail = True`, every recipe |
| `train(recipe="legacy")` | trained with whole-frame medians | raises (retired); props train only through `props_harness` or `recipe="props_r2_repaired"` |
| median-fill telemetry | counted batch-median fills | records none (no fill happens) |

The contract classes (`props_contract.py`) are unchanged in meaning: REQUIRED_FAIL_ROW -> no
probability; SEMANTIC_ZERO -> 0 (red_cards_pg, offsides_pg only; penalties_won_pg left it on
2026-11-15 because a null penalty.won is now "not published"); IMPUTABLE / OPTIONAL_DEGRADED -> the
fit-block median of a persisted imputer, or NaN for a NaN-native model.

## Proof (tests/test_props_real_data_only_2026_11_15.py, run on the SERVED artifacts)

| property | test | result |
|---|---|---|
| identical regardless of batch composition and row order | `test_predictions_do_not_depend_on_batch_composition_or_order`: each scorable served model, 60 rows with 20% missing cells, scored as one batch, reversed, one row at a time, and mixed with 30 unrelated rows multiplied by 50 | equal to 1e-12, every market |
| required missing inputs fail explicitly | `test_a_missing_required_input_fails_explicitly`: a missing and an ABSENT required column | p NaN, required_fail True; the other rows are scored |
| no holdout or live data fits a statistic | `test_no_statistic_of_holdout_or_live_rows_reaches_a_fitted_imputer`: harness medians = fit-block medians; adding 50 extreme live rows moves no probability. `tests/test_props_r2_integrity.py::test_imputer_uses_fit_block_statistics_only` | equal |
| nothing filled, no zero column | `test_prep_fills_nothing_and_never_creates_a_zero_column` | pass |
| no batch median in the scoring code | `test_no_live_batch_median_in_the_scoring_path` (AST: no `.median()` / `.fillna()` call in `_prep`, `predict_with_diagnostics`, `_predict_recipe`, `predict_proba`) | pass |
| legacy payloads are batch-invariant now | `tests/test_props_r2_integrity.py::test_serving_legacy_payload_never_fills_from_the_batch`, `::test_legacy_path_is_batch_invariant_now` | pass |
| legacy training cannot run | `test_legacy_training_cannot_run`, `tests/test_props_r2_integrity.py::test_legacy_recipe_training_is_retired` | raises |

Full suite on the branch: 3179 passed, 44 skipped, 6 xfailed, 0 failed. The run's exit status 1 is
the conftest hygiene guard. It flags `src/cloudbet/__pycache__/*.pyc`: those compiled files are
committed on the base branch, and Python rewrites them on every import. This branch did not cause
it; the files were restored and none was committed.

## Serving impact

* sot / sot2 / goals / assists / cards (clean models): the same probabilities, except rows missing
  a required input, which now get none. With the new `started` from lineups, a player whose last 5
  matches had no collected lineup has no starter_rate, so he gets no probability.
* goals2 / goals3 / sot3 / sot4 (legacy): refused, because they list fabricated set-piece features.
  They were already hidden from tips and the board.
* Parity (`scripts/props_train_serve_parity.py`, 2026-11-01..08, 80 fixtures, 2,231 player-fixtures,
  145 features): 0 features and 0 probabilities differ on the five served models.

## Limitations

* The cards champion still fills with its persisted fit-block medians until the next retrain; the
  plan names hgb_nan_native for cards (`candidate_by_market`).
* The rows `required_fail` drops are counted in the diagnostics, not yet on the dashboard.
