# P9 - Model imputation: what the serving median fills fill, and the Phase 2 options

fix9/defaults, 2026-11-04. Owner rule for this round: **do not change serving now** - a change to
the imputers changes the frozen models' predictions. This report says what each fill does, measures
it on the committed live board, and lists the options for the Phase 2 clean retrain. No code path
in this report was changed; `src/imputer.py`'s two `from_dict` defaults are allow-listed in
`config/data_truth_allowlist.json` with a pointer here.

## 1. The three fills

| Site | Used by (serving) | What it fills | With what |
|---|---|---|---|
| `src/model.py` `_prep` line 140 (`imputer=None`, LEGACY) | O/U 2.5 **new_format** (`model_v9_newformat.pkl`), **BTTS / Over 1.5 / Over 3.5** (`model_v9_btts/over15/over35.pkl`), **HT Over 0.5 / 1.5** (`model_ht_over05/15.pkl`) | every NaN cell of a feature column in the frame being scored | the **median of the scoring batch** (`X.median()`); a column with no value anywhere in the batch gets **0.0** (`.fillna(0.0)`) |
| `src/imputer.py` `apply_imputer` (manifest `v9-train-only-imputer-1.0`) | O/U 2.5 **standard** (`model_v9_standard_repaired.pkl`, recipe `v9_repaired`) | every NaN cell of its 41 features | the feature's **training median** (base-train block only, persisted in the payload); a feature with no training value (`home/away_xg_last5`, `home/away_insidebox_last5`) gets `fill_all_missing` = **0.0**; values are clipped to the training range |
| `player_model/model.py` `_prep` line 43 | all 9 player-prop payloads (`model_player_*.pkl`, recipe `props_legacy_batch_median`) | every NaN cell of the 120-138 features | the **median of the scoring batch**; an all-NaN column stays NaN and is scored only by the NaN-native HistGradientBoosting member (`_ensemble_proba`) |

Batches at serving (`src/predict.py`): the standard O/U model scores the standard rows; the
new-format O/U model the new-format rows; the three side-market models score **standard + new_format
rows in one batch** (`side_mask = std_mask | nf_mask`); the HT models the standard rows. A batch
median therefore depends on which other fixtures are on the board (P0 bug 1, "batch dependence").

## 2. Measured on the committed live board

Board: `output/predictions.csv` as committed on data-truth-100 (174 fixtures, 72 standard + 102
new_format, match dates 2026-11-02..2026-11-09, `generated_at` 2026-11-02T05:59:14Z - the newest
committed board on 2026-11-04). Feature values in that file are the pre-imputation values (`_prep`
fills a copy). Every feature column of every payload is present in the file, so nothing below is
estimated.

| Model (batch) | Rows | Feature cells | Cells filled | Share | Rows with >= 1 fill | Fill used |
|---|---:|---:|---:|---:|---:|---|
| O/U 2.5 standard, v9_repaired (std) | 72 | 2,952 | **40** | 1.36% | 3 | training median (36), constant 0.0 (4) |
| O/U 2.5 new_format, legacy (nf) | 102 | 2,754 | **92** | 3.34% | 12 | batch median |
| BTTS, legacy (std+nf) | 174 | 7,308 | **252** | 3.45% | 18 | batch median |
| Over 1.5, legacy (std+nf) | 174 | 7,308 | **252** | 3.45% | 18 | batch median |
| Over 3.5, legacy (std+nf) | 174 | 7,308 | **252** | 3.45% | 18 | batch median |
| HT Over 0.5, legacy (std) | 72 | 4,680 | **40** | 0.85% | 3 | batch median |
| HT Over 1.5, legacy (std) | 72 | 4,680 | **40** | 0.85% | 3 | batch median |

- **Total: 968 filled cells** on this board (40 by the training imputer, 928 by a batch median).
- **4 cells were filled with the constant 0.0**: one standard row lacks `home/away_xg_last5` and
  `home/away_insidebox_last5`, the four features of the repaired standard payload that have no
  training median, so the 1.0 manifest's `fill_all_missing` (0.0) filled them (the other 36 of its 40
  cells took a training median). The legacy 0.0 path (a column missing on every row of a batch) filled
  **0** cells today.
- Most-filled columns: the half-time form block (`combined_ht_goals_avg` 18 rows in the side batch;
  `*_ht_*` 11 rows each) and, on new_format, `home/away_fouls_pg_roll` and `home/away_corners_pg_roll`
  (8 rows each).
- Of the filled rows, the blind fixtures (no rolling form; 6 on this board, 1 standard + 5
  new_format) get **no tip** by the blind-fixture guard. Non-blind filled rows: standard O/U 2
  (Bundesliga 2, Serie B; 1 of them carries an O/U tier), new_format O/U 7 (China 5, Austria 1,
  Argentina 1; 0 tiered), side-market batch 12.
- **Player props: UNIDENTIFIABLE on committed data.** The props feature frame is built at predict
  time and is not committed (`output/player_tips.csv` carries no feature values). The last
  measurement on record is `output/player_props_integrity_health.json` (2026-11-01): the legacy
  batch median is not batch-invariant (one injected NaN in `kp_per90` moved 16 of 120 rows, max
  |dp| 0.0126). Props are paper-only (never staked).

How to reproduce: for each payload, `feature_cols` against `output/predictions.csv` restricted to the
serving batch above; count `isna()` cells; a column all-NaN in the batch is the 0.0 case.

## 3. Options for the Phase 2 clean retrain

Phase 2 retrains only on data that is 100% real (owner rule; no retrain before then). Each option
removes the invented value; they differ in what the model learns from "missing".

1. **NaN-native model.** HistGradientBoosting (already the props ensemble member and the FPL minutes
   model) routes a missing value down its own learned branch; nothing is filled at train or serve. A
   row's probability depends only on that row. Cost: the logistic / calibrated members cannot take
   NaN, so either the ensemble becomes trees-only or the linear members score complete rows only
   (the props `_ensemble_proba` rule). Best fit for features that are missing for structural reasons
   (HT form in leagues without HT history, xG for standard leagues).
2. **Explicit missing indicators.** Keep a fill but add `<feature>__missing` columns
   (`src/imputer.MISSING_SUFFIX`, the 1.1 manifest's cold-start ablation D), and fill with the
   TRAINING median only (never the batch). The model can learn that "unknown" differs from "typical";
   the filled value itself carries no information. Doubles the width of the affected blocks; needs the
   1.1 manifest (`all_missing_policy="drop"` or `"error"`, never `"zero"`).
3. **Exclude rows.** Train and serve only rows with every required input (the blind-fixture guard
   generalised): no fill at all, no tip on an incomplete row. Cleanest semantics, but it shrinks the
   new-format and side-market coverage (12 of 102 new_format rows and 18 of 174 side rows on this
   board would get no probability) and must be applied identically at train time.

The repaired standard recipe already removes the batch dependence (training medians, persisted);
the remaining issue there is the 0.0 for the four features with no training value (4 cells on this
board), which option 1 or 2 resolves. The legacy new_format, side-market and HT payloads carry both issues (batch median
and the 0.0 all-missing fallback) until their Phase 2 retrain.
