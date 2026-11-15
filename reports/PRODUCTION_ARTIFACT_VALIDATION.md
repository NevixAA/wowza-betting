# Production artifact validation: props section (P0 "real data only", 2026-11-15)

Every served props artifact (`config.MODEL_FILES`), loaded with the production loader
(`props_gate._HarnessUnpickler`) at `fix/props-data-audit`. Each file's sha256 equals its
registry champion (`models/registry/props/props_champions.json`; also asserted by
`tests/test_props_r2_integrity.py::test_serving_files_are_the_registered_champions`). All 8 load:
none is UNVERIFIED. Nothing was promoted, replaced or rewritten.

| market | file | sha256 (16) | registry version | training window (fit .. calibration end) | candidate | imputer | features | unsupported features listed | features withheld from the next retrain | status | behaviour today |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sot | models/model_player_sot.pkl | 571b102c4214f619 | props_sot_clean_hgb_nan_native_571b102c | 2022-08-18 .. 2026-08-31 (run 43035218594, code 6fd79496, 2026-11-11) | hgb_nan_native | NONE_NAN_NATIVE | 128 | none | 20 | ACTIVE (paper) | scored NaN-native; required missing -> none |
| sot2 | models/model_player_sot2.pkl | 5b5fad604a126baa | props_sot2_clean_hgb_nan_native_5b5fad60 | same run | hgb_nan_native | NONE_NAN_NATIVE | 128 | none | 20 | ACTIVE (paper) | same |
| goals | models/model_player_goals.pkl | 5de94714b722dbc3 | props_goals_clean_hgb_nan_native_5de94714 | same run | hgb_nan_native | NONE_NAN_NATIVE | 128 | none | 20 | ACTIVE (paper) | same |
| assists | models/model_player_assists.pkl | abf39bb5abbb2d60 | props_assists_clean_hgb_nan_native_abf39bb5 | same run | hgb_nan_native | NONE_NAN_NATIVE | 128 | none | 20 | ACTIVE (paper) | same |
| cards | models/model_player_cards.pkl | bfd5e1cc51835a63 | props_cards_clean_recipe_retrained_bfd5e1cc | same run | recipe_retrained | FIT_BLOCK_MEDIAN_PERSISTED (persisted fit-block medians) | 128 | none | 20 | ACTIVE (paper) | persisted medians, never refitted; required missing -> none. Next retrain: hgb_nan_native |
| goals2 | models/model_player_goals2.pkl | 426e3bb5a6a6f783 | props_goals2_legacy_426e3bb5 | before 2026-11-01 (legacy, no window recorded) | legacy | batch median until 2026-11-15 | 138 | sp_taker_score, sp_receiver_score, set_piece_threat_score, set_piece_corner_matchup, defender_sot_role_index | 36 | HIDDEN (pending clean retrain) | **REFUSED**: no probability |
| sot3 | models/model_player_sot3.pkl | 805e72e2f0dc44b9 | props_sot3_legacy_805e72e2 | before 2026-11-01 | legacy | as goals2 | 138 | as goals2 | 36 | HIDDEN | **REFUSED** |
| sot4 | models/model_player_sot4.pkl | 960f5a9ba70fd47c | props_sot4_legacy_960f5a9b | before 2026-11-01 | legacy | as goals2 | 138 | as goals2 | 36 | HIDDEN | **REFUSED** |

* "features withheld from the next retrain" = the artifact's features that are now in
  SHOTS_DEFINITION_WITHHELD, PROFILE_CONSTANT_FEATURES_WITHHELD, TYPED_FORMULA_FEATURES_WITHHELD or
  LEGACY_QUALITY_FEATURES_REMOVED. Serving still computes them, so these artifacts keep working, but
  they are not in the clean schema (125 features).
* The five clean models never saw a typed default (DEFAULTS_RECONCILIATION.md). They were trained
  before the 2026-11-15 data fixes, so they learned from: the provider `started` flag (correct
  until 2026-08; the drift falls in their calibration and holdout months), /sidelined injury
  features that went stale from 2025-10, shots_total on the old definition, and dribbles_past /
  penalty_won with Oct-Nov provider nulls read as 0. Today they are SERVED the corrected inputs.
  The re-fit candidates are in CLEAN_MODEL_BACKTEST.md.
* `models/model_player_goals3.pkl` exists but is not in `config.MODEL_FILES`: no serving path reads
  it. It lists set_piece_threat_score and set_piece_corner_matchup.
* Research challengers `models/challengers/props_r2/*.pkl` (8 files): load OK, list 5 set-piece
  features, so they are refused. Never served.
* scikit-learn warns that the files were pickled with 1.9.0 and are loaded with 1.9.1 locally. The
  parity check below gives identical probabilities.

## Production-path inference with missing inputs

`tests/test_props_real_data_only_2026_11_15.py::test_production_path_with_missing_inputs_never_invents_an_observation`
loads each scorable served artifact and scores rows with half of the inputs missing through
`model.predict_with_diagnostics`, the function `predict_proba` and the live pipeline use:

* `hgb_nan_native` (sot, sot2, goals, assists): the model receives NaN. Its probability equals the
  harness model's own `predict` on the NaN matrix, bit for bit. Nothing is filled.
* `recipe_retrained` (cards): the only fill is the fit-block median persisted in the artifact
  (`state.fill_values`), a subset of its own feature list.
* any row missing a required input: no probability (`required_fail`).
* `::test_a_served_payload_with_an_unsupported_feature_gets_no_probability`: goals2 / sot3 / sot4
  give no probability on any input; the five clean models give one.
* `::test_predictions_do_not_depend_on_batch_composition_or_order`: identical alone, in any order
  and next to unrelated rows.

Train/serve parity on real fixtures (`scripts/props_train_serve_parity.py`, sampled fixtures
2026-11-01..08, 80 fixtures, 2,231 player-fixtures, 145 features = every feature of the 5 served
models plus every feature of the clean schema): **0 features and 0 probabilities differ** (commit
`c37952a08`). Re-run at the branch head `d051ee5c7` (after the scoring change `c41165a49` and the
registry league ids `afe2d2669`), same sample, frame rebuilt from scratch: **0 features and 0
probabilities differ** on all five served models (max |diff| 0.0).

## Limitations

* Rows that `required_fail` drops are not yet shown on the dashboard.
* The legacy hidden markets have no model until a clean retrain is promoted (an owner decision).
