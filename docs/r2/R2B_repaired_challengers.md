# R2-B — Repaired challengers for NF / side markets / HT (2026-11-01)

Workstream R2-B of the WOWZA model-integrity repair (owner brief 2026-11-01, findings 1, 2, 3, 4,
5, 13, 16, 19). **Research challengers only.** Nothing in this branch changes what serves, any
threshold, tier, stake or Telegram output. No promotion. Lifecycle stage of every experiment:
`RETROSPECTIVE_REPLAY`.

* v9 branch `r2/repaired-challengers` (NevixAA/wowza-betting)
* Pro branch `r2/repaired-challengers` (NevixAA/wowzaV9-Pro) — registry, CI benchmark, collector

## Defects addressed

| # | Defect | Where | Severity | Reachable in production today | Repair (challenger path) | Tests |
|---|---|---|---|---|---|---|
| 1 | `train()` defaults to `legacy_v9`: batch-median imputation, meta fitted and scored on its own test block, weights misaligned; payloads carry no recipe / version / provenance | `src/model.py` | P0 for every non-Standard team model | YES — NF O/U 2.5, BTTS, O1.5, O3.5, HT 0.5/1.5 all train and serve legacy | One parameterised architecture `train_repaired(recipe=nf_/side_/ht_v9_repaired)`; full payload metadata (recipe, model/feature/identity/imputer versions, cutoff, start/end, rows, frame hash, feature list, block definition, code SHA, recency, metrics, provenance) | `test_r2_payload_metadata_is_complete`, `test_unknown_recipe_and_legacy_options_refused` |
| 2 | Batch-median imputation: a fixture's probability depends on the other rows in the call; holdout imputed with its own medians; an all-missing column becomes 0.0 silently | `src/model._prep` (legacy), `src/imputer` 1.0 `fill_all_missing` | P0 | YES (all legacy payloads). The Standard repaired payload uses the 1.0 train-only imputer, but it still fills its no-training-values xG columns with 0.0 | Imputer **1.1**: fitted on BASE only, persisted, used for calibration/meta/holdout/live; features with no training values are DROPPED and recorded (`dropped_all_missing_features`); `apply_imputer` raises instead of inventing a value unless the feature is a declared semantic zero; a repaired payload without its imputer is refused at scoring | `test_batch_invariance_single_reorder_and_unrelated_rows`, `test_upcoming_features_do_not_depend_on_the_board`, `test_fully_missing_feature_is_dropped_and_recorded`, `test_r2_imputer_refuses_to_invent_a_value_for_a_no_training_feature`, `test_repaired_payload_without_imputer_is_refused`; legacy defect pinned by `test_documents_legacy_batch_dependence`, `test_v1_manifest_unchanged_documents_legacy_zero_fill` |
| 3 | No untouched holdout: legacy meta-logistic fitted on the rows it is scored on; no persisted block boundaries | `src/model.train` | P0 for every reported legacy metric | YES (every legacy retrain's metrics and the promotion gate's legacy numbers) | BASE → CALIBRATION → META → UNTOUCHED HOLDOUT; holdout boundary fixed OUTSIDE the trainer (`holdout_start`) so every arm is scored on identical rows; whole-date cuts, disjointness asserted; boundaries + `holdout_independence` persisted. Same-day league aggregates were already excluded (`features_repaired._league_asof_mean` uses strictly earlier dates) | `test_blocks_are_chronological_disjoint_and_persisted`, `test_calibrator_and_meta_never_read_holdout_labels` (flip every holdout label → base, Platt, meta coefficients and imputer bit-identical); legacy pinned by `test_documents_legacy_meta_reads_its_holdout_labels` |
| 4 | Weights indexed by a reset index — rows reorder, weights do not | `src/model.train` | P1 | YES (legacy NF/side) | Weight attached as `_sample_weight` before any filter/sort, re-derived from the original row position after the sort and compared (`AssertionError` on any mismatch); length mismatch is an error | `test_weights_stay_with_fixture_after_shuffle_and_sort` (unique weights, shuffled input, spy on the fitted weights), `test_weight_length_mismatch_is_an_error` |
| 5 | Hand-typed `TRAINING_DECAY_WEIGHTS` froze (no 2025/26, 2026/27 keys; calendar-year labels never match) | `config.py`, `pipeline._get_weights` | P1 | YES (legacy NF/side) | `src/recency.py`: candidates FROZEN before evaluation — `season_age` (PRIMARY, = the Standard repaired derived weights), `exp_half_life` (365 d), `bounded_linear` (4 y, floor 0.25), `uniform`; `as_of` must be a training date; function, params, per-season effective weight and ESS persisted | `test_recency_candidates_are_frozen_and_persisted`, `test_season_age_matches_the_standard_derived_weights` |
| 13 | Source regime (football-data vs backtest archive) predictable from missingness | data | P1 (research) | Training frames only | Ablations FULL / MODERN_ONLY / MODERN_ARCHIVE / RECENT_2 / RECENT_3 / RECENCY_WEIGHTED (+ the two other frozen recency functions), each scored on the identical holdout; source composition by season, missing rate by source, missingness→source 5-fold AUC, target rate by source, metrics by holdout season | CI `regimes` jobs |
| 16 | Standard cold-start priors typed in (HA 0.55/0.45, away goals 1.10, SOT 0.35/0.33, HT 0) | `src/features_repaired.py` | P2 (research) | YES (Standard repaired serving) — **not changed** | `cold_start="nan"` option (default `"current"` byte-identical) + imputer options → ablation A current / B NaN + train median / C train-fitted league prior / D global prior + missing indicators. RESEARCH ONLY | `test_cold_start_nan_only_changes_the_cold_start_columns`, `test_league_prior_and_missing_indicator_options`; CI `coldstart` job |
| 19 | Side models pooled over Standard + New Format | `pipeline.mode_train` | P1 | YES (BTTS / O1.5 / O3.5 serve pooled) | Pooled, Standard-only and NF-only challengers on one holdout boundary, scored per track, never pooled | `test_track_specific_side_rows_never_mix`; CI `core` side jobs |
| id | NF duplicate identities (e.g. China Super League Shandong Luneng / Shandong Taishan, Henan Jianye / Henan Songshan Longmen) | history | P1 | YES (legacy NF features are built on raw names) | The committed evidence-backed alias map already carries 20 China Super League aliases incl. both pairs; challengers use it + same-day derivation. New `identity_repair.overlap_audit` lists every same-day-evidence pair with active ranges, overlap and decision (`ALIASED_EXPLICIT` / `ALIASED_DERIVED` / `CANDIDATE_REVIEW` / `WEAK_EVIDENCE` / `TWO_CLUBS`); nothing is auto-aliased | `test_nf_identity_overlap_audit_china_duplicates`; CI `identity` job → `identity_health.json` |

## What did NOT change (non-regression proof)

* **Standard OU2.5 serving (`v9_repaired`)**: the serving payload is untouched, its 1.0 imputer
  path is unchanged, and `build_features_repaired` / `build_upcoming_features_repaired` default
  output hashes are identical to main (`8f307dd607177087` / `36b5961632bcd579` on the synthetic
  suite frame). CI `parity` job replays real data (board + last-30-day replay + every legacy
  serving payload) under main and under the branch and requires byte-identical files.
* **Every committed serving payload** (Standard legacy, Standard repaired, NF, BTTS, O1.5, O3.5,
  HT 0.5, HT 1.5) scores bit-identically to main's code: `test_serving_payload_scores_identically_to_main`.
* **Training**: `legacy_v9` and `v9_repaired` produce the same probabilities, blocks and metadata
  keys as main 4e9c933 (`tests/fixtures/r2b_training_golden_main_4e9c933.json`).
* `python -m src.research_lock --check`: `protected_tracks_modified=False`.

## Architecture (one recipe family)

```
history (v9 loader, COVID excluded, rows < cutoff)
  -> identity: committed alias map + same-day derivation, merged map frozen   (src/identity_repair)
  -> as-of features, league-scoped, train/serve one function, no price          (src/features_repaired)
     [HT: + src/ht_features run PER LEAGUE]
  -> holdout = rows >= HOLDOUT_START (newest ~20%, whole date)  -- identical for every arm
     pre-holdout: BASE 80% | CALIBRATION 10% | META 10% (whole dates, disjoint)
  -> imputer 1.1 fit on BASE; all-missing features dropped                      (src/imputer)
  -> LR + GB + LightGBM on BASE (weights = frozen recency, travelling with rows)
  -> Platt on CALIBRATION, meta-logistic on META, every metric on HOLDOUT       (src/model)
  -> payload: models + imputer + full metadata
```

NF features follow the Nov-1 parity semantics by construction: `home_advantage` is the rolling
training rule for train AND serve (one function), no market/overround feature enters the football
model (finding 15), and serving resolves club names league-scoped (`_resolve_names`).

## Results (RETROSPECTIVE, CI)

PENDING - Pro CI run 41297555994 (workflow pro_r2_repaired_challengers.yml, branch r2/repaired-challengers) was still running when this doc was committed. Its collector writes output/studies/r2_repaired_challengers/{verdict.csv, regimes.csv, coldstart.csv, summary.json, identity_overlap_audit.csv} and the results block of registry/r2/r2b_repaired_challengers.json on the Pro branch. Copy the verdict table here after review.

## Open risks

* Results not yet verified: the first CI run had not finished at hand-off; any job failure is an infra or code failure, not a model result.
* The champion holdout number is optimistic (legacy meta fitted on the holdout); read CHAMPION_BASEMEAN next to it.
* Market metrics depend on the forward price capture only (genuine two-sided entry prices); expect small n, and NOT_EVALUABLE for HT.
* The source tag (MODERN vs ARCHIVE) is reconstructed by reloading with WOWZA_USE_BACKTEST_ARCHIVE=0 and matching canonical keys; rows the unifier names differently in the two loads count as ARCHIVE.
* The serving-parity job reloads football-data twice; a source update between the two loads would show as a false mismatch.
* Standard-repaired retrains still use imputer 1.0 (zero-fill of all-missing xG), deliberately unchanged here; moving it to 1.1 is a Standard model change that needs owner approval.
