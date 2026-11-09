# Team-model promotion review after the Phase 2 clean retrain (2026-11-09)

Owner request, 2026-11-09: "why didn't push new format?". Goal: install the clean-retrained team
models (both tracks) into serving for go-live. The first pass found four blockers (findings 1-3 and
the parity section below). The owner decided them the same day; the section "Owner decisions and
what was done" records the result.

## Owner decisions (2026-11-09) and what was done

| Decision | Owner's words | Done on this branch |
|---|---|---|
| A | "the models that worked before we reach 100% clean data are irrelevant" | `predict.yml` `STANDARD_OU25_SERVING_RECIPE` `v9_repaired` → `legacy_v9`: Standard O/U 2.5 serves `models/model_v9_standard.pkl`, the clean recipe, scored on `build_upcoming_features` (the builder it was trained with; no identity map added to the model). The 2026-10-28 `v9_repaired` model is archived (decision log stage ARCHIVE); its file, champion pointer and stored artifact stay as the rollback. `src/integrity/model_health.py` reads the payload that actually serves and recognises the clean recipes |
| B + D | "Yes, fix leak + refit." | Leak fixed (`src/feature_engineering._league_prior_dates_mean`; commit `2d1d900b`). It also covered `_add_home_advantage`'s `league_avg_away_goals`, the same row-order pattern. Serving's away-goals reference no longer falls back to an all-league mean. Parity after the fix: **0 of 6,837 cells differ, 159/159 probabilities identical** on both tracks. The Phase 2 comparison is re-run locally on both tracks with the fix; then `scripts/phase2_refit_serving.py` refits `champion_recipe` per track × market on all clean data through 2026-11-08 |
| C | "Override for this switch." | `src/promotion_gate.owner_clean_switch`: a ONE-TIME override of PG-1.0's prospective-period rule, bound to the exact sha256 of each refit artifact listed in `config/promotion_overrides.json`, each installable once, recorded as stage OWNER_CLEAN_SWITCH / PROMOTE with the owner's words. The two-period rule stays in force for later promotions |

Also fixed (coordinator, "no bugs"): 11 walk-forward fits of the first Standard run (hgb on
over25/btts/over15/over35 folds 0-1, model_market on over25 folds 0-2) crashed with "window shape
cannot be larger than input array shape". **Cause:** sklearn 1.9's HistGradientBoosting binning
raises on any column with no non-missing value; Standard training windows in 2021-2022 have no xG
and no half-time history at all (13 such columns in fold 0). HGB now drops such columns and records
them (the champion recipe's rule), a candidate that truly cannot fit is recorded NOT_FITTED with the
reason, any other exception fails the run after its reports are written, and every run writes
`coverage.csv` (commit `4a4bd06c`). The holdout stage never failed, so no holdout verdict depended on
the missing folds; the walk-forward comparisons for hgb and model_market ran on fewer folds.

## Results (2026-11-09)

**1. Re-run with the leak fix and the harness fix** (local, code `4a4bd06c`, the same data frame
`4638ab5ee15f3a7f`, the same approved holdout; `reports/phase2_runs/standard_run_20261109T104641Z`,
`new_format_run_20261109T112024Z`):

| Track | Comparisons | Verdict changes vs the pre-fix run | NOT_FITTED | ERROR | Holdout BETTER_THAN_REFERENCE |
|---|---|---|---|---|---|
| Standard | 60 | **0** | 0 | 0 | market_only and model_market on O/U 2.5 (closing price: a benchmark, not bettable) |
| New-Format | 108 | **0** | 0 | 0 | logistic and logistic+ht on HT Over 1.5 (no price; not tipped) |

champion_recipe holds for every market on both tracks; no `+ht` variant improves anything.

Coverage of the formerly failing candidates (Standard walk-forward):
- **hgb:** now scored in all 4 folds, 20,160 rows on every market (before: 2 of 4 folds, 16,281 rows).
- **model_market on O/U 2.5:** 20,088 rows (before: 14,024).

No verdict depended on the missing folds: every verdict is unchanged, and the holdout stage never lost a fit.

**2. Refit** (`scripts/phase2_refit_serving.py`, code `406c1260`): champion_recipe per track × market.
- **Training window:** Standard 2021-07-30 → 2026-10-25 (HT models from 2023-07-28); New-Format 2019-02-22 → 2026-10-25.
- **The recipe's own untouched tail:** 2026-10-26 → 11-08, 44-47 rows. Under the declared rule two ISO weeks were needed to reach 20 rows; the last week alone had fewer (the international break).
- **Dropped columns:** none.

**3. Parity** (`scripts/team_train_serve_parity.py`, the last 6 match dates, 159 fixtures, all 43 raw inputs, all 12 refit models):
- After the leak fix every cell matched within 1e-9, but three Standard models (O1.5, O2.5, O3.5) still differed on one fixture each, by up to 0.36 pp.
  - Cause: `away_xg_last5` 1.104 vs 1.1040000000000003. Serving's plain mean of the last 5 values sums in a different order than training's rolling sum, and the 1-ulp difference flipped a tree split.
  - Fix: `_latest_form` now runs training's own `_rolling` on a row appended after each team's matches (commit `63057046`).
- **Final: 0 of 6,837 cells differ, 159/159 probabilities identical for all 12 models.**

**4. Installed through the gate** (`owner_clean_switch`, override `clean-data-switch-2026-11-09`):

| Model type | New champion | sha256 (16) | Replaces (kept as the rollback) |
|---|---|---|---|
| standard | standard_20261109T125211Z_254623a0 | 254623a04a51bdc3 | standard_20261018T080423Z_64313878 |
| newformat | newformat_20261109T125212Z_4268cf9f | 4268cf9fd694c34c | newformat_20261018T080423Z_9eb6b4b4 |
| btts (track-split) | btts_20261109T125212Z_5a7f72b4 | 5a7f72b4bf4ff952 | btts_20261005T101905Z_238e3419 |
| over15 (track-split) | over15_20261109T125212Z_a343092b | a343092b9f76a708 | over15_20261018T080423Z_da4743a2 |
| over35 (track-split) | over35_20261109T125212Z_c05e41a4 | c05e41a45e55f5c9 | over35_20261005T101905Z_f546ce72 |
| ht_over05 | ht_over05_20261109T125212Z_bd7223e7 | bd7223e7cc5311bd | ht_over05_20261018T080423Z_359b491a |
| ht_over15 | ht_over15_20261109T125212Z_df6dc412 | df6dc41205a6dce9 | ht_over15_20261004T093828Z_6f370f63 |

`standard_repaired` (v9_repaired, 2026-10-28) is archived: decision-log stage ARCHIVE; its pointer, file and stored artifact are kept.

**5. Consequences handled:**
- **H-MODEL-01 shadow challengers RETIRED.** They were registered against the pre-switch champions, so their comparison basis is gone. Artifacts kept and verified. The challenger-coverage health check reads GREY ("no active challenger") instead of a false RED.
- **AU1-B05 resolved.** The batch-dependent New-Format probability, a strict xfail, now passes as a guard.
- **NF calibration restarts.** The shift's fit window starts at the new champion's install time (n = 0 until 150 settled rows).
- **Track registry regenerated.** Only the champion fields change.

**Open (owner only):** the research lock (`python -m src.research_lock --check`) reports the
predict.yml switch change `STANDARD_OU25_SERVING_RECIPE "v9_repaired" -> "legacy_v9"`. Re-approval
records the owner's name and is not done by an agent:
`python -m src.research_lock --approve --by "Nevo (owner)" --note "2026-11-09 decision A: Standard O/U 2.5 serves the clean recipe"`.
Until then that check and the 5 tests that read the lock stay red, and the branch is not merged.

## What this branch changes

| Path | Change |
|---|---|
| `reports/phase2_runs/standard_run_20261108T190758Z/` | Evidence of the Standard run (GitHub run 42512834177): comparison, metrics, per-league, manifest, holdout record, README |
| `reports/phase2_runs/new_format_run_20261108T214340Z/` | Evidence of the New-Format run, run LOCALLY (the GitHub 16 GB runner died twice in the comparison step, which peaks at about 10 GB). Same code `804799a1`, same data frame (`frame_hash 4638ab5ee15f3a7f` on both runs), same approved holdout |
| `src/nf_calibration.py` | The NF O/U 2.5 logit shift now fits only on rows logged since the SERVING champion was installed (`models/registry/current_champions.json` `newformat.champion_since`), and fits nothing when that date can't be read |
| `tests/test_nf_calibration.py` | Tests use their own registry file; 4 new tests (rows from a previous champion ignored, rows from the new one used, unreadable registry means no shift, committed registry readable) |
| `scripts/team_train_serve_parity.py` | Train/serve parity of every raw input of a team payload, with the probability impact |

No serving model file, registry entry, promotion record, threshold or stake changed.

## Finding 1: Standard O/U 2.5's serving model was not in the retrain

Standard O/U 2.5 is served by `models/model_v9_standard_repaired.pkl` (`STANDARD_OU25_SERVING_RECIPE:
"v9_repaired"` in predict.yml; owner switch 2026-10-28). It is a FROZEN model type
(`src/model_registry.FROZEN_MODEL_TYPES`), built by the research recipe `v9_repaired`:
- 41 features from `src/features_repaired.build_upcoming_features_repaired`;
- a frozen identity alias map inside the payload.

The Phase 2 harness's `champion_recipe` for Standard O/U 2.5 is a different model: the
scheduled-retrain recipe `std_v9_repaired` (86 inputs from `src/feature_engineering.build_features`),
which writes `model_v9_standard.pkl`. That file is NOT read while the switch is `v9_repaired`.

So:
- **"The current recipe wins" on Standard O/U 2.5 compares against a model that doesn't serve.** The model that serves was never scored by the harness.
- **Its payload can't be served on the repaired path.** It has no `identity_alias_map` and no `research` block, so `src/standard_serving.validate_payload` refuses it. Under the fail-closed policy, installing it there would remove every Standard O/U 2.5 tip.

**Owner decision A:** what serves Standard O/U 2.5 after go-live?
- (a) Keep the frozen `v9_repaired` model and add it as the reference candidate in a Phase 2 run (the harness then judges the clean recipe against what really serves).
- (b) Retrain `v9_repaired` itself on the clean data.
- (c) Switch Standard O/U 2.5 back to the scheduled recipe (`legacy_v9` switch) with the clean model.

## Finding 2: the retrained models stop at 2026-03-31

Every harness pickle was fitted for the HOLDOUT stage, on data up to 2026-03-31 (Standard from
2021-07-30, New-Format from 2019-02-22). Serving them would leave out the seven most recent months of
clean data (Apr 1 – Nov 8), the period the holdout validated.

**Owner decision B:** the usual route is a refit of the SAME validated recipe on all clean data
through 2026-11-08 for serving. That refit has no untouched holdout of its own; its evidence is the
holdout result of the identical recipe. The alternative is to serve the Mar-31 models as validated.

## Finding 3: the promotion gate can't install them today

`src/promotion_gate.py` (PG-1.0, owner 2026-10-25: "Retraining may be frequent. Promotion must be
rare.") promotes only after:
- Period 1: an untouched holdout against the serving champion;
- Period 2: n ≥ 1,000 prospective first-sight rows logged after the challenger was frozen.

A Period-1 pass only PARKS a challenger (SHADOW_MORE). The owner override `owner_install` is limited
to FROZEN types (today only `standard_repaired`). Installing `newformat`, `btts`, `over15`, `over35`,
`ht_over05` or `ht_over15` before Nov 21 therefore needs one of:
- (a) an owner-approved extension of `owner_install` to these types for the clean-data cutover, recorded in the decisions log (a gate rule change);
- (b) submitting them as challengers now (Period 1), shadowing them from go-live, and promoting after 1,000 settled prospective rows per track (weeks to months).

**Owner decision C:** (a) or (b).

## What is compatible once those decisions are made

| Market | Serving file | Track routing in serving | Harness payload |
|---|---|---|---|
| NF O/U 2.5 | `model_v9_newformat.pkl` | New-Format rows only | `nf_v9_repaired`, 28 inputs; same serving format (`_predict_flat`) |
| BTTS / O1.5 / O3.5 | `model_v9_btts/over15/over35.pkl`: ONE file for both tracks | `src.model._predict_by_track`: top level = Standard, `tracks["new_format"]` = New-Format (the `train_by_track` format) | Two per-track `side_v9_repaired` payloads. They fit the supported track-split format directly; this is not a hand merge |
| HT O0.5 / O1.5 | `model_ht_over05/15.pkl` | Standard rows only (predict.py scores HT on `std_mask`) | Standard `ht_v9_repaired`. The New-Format HT models (including logistic for ht_over15, Holm p 0.039) have NO serving path |
| Standard O/U 2.5 | `model_v9_standard_repaired.pkl` (frozen) | Standard rows | **Not compatible** (finding 1) |

## Calibration layers

`src/nf_calibration.py` fits a logit shift on settled New-Format O/U 2.5 rows logged since
2026-11-01T17:00Z: the OLD model's predictions (27 rows in the committed log, below the n ≥ 150 apply
bar). After any NF promotion, that shift would correct the old model's error on the new model.

The per-row model stamp in `output/v91_shadow.csv` (`v9_model_sha`) can't separate models: it hashes
file size + mtime over all `models/*.pkl` and changes with every checkout (5 values on 27 rows).
**Fix on this branch:** the fit window starts at the later of the fixed-pipeline start and the
serving champion's `champion_since`, and nothing is fitted when that date is unreadable. Behaviour
today is unchanged (the current champion has been serving since 2026-10-25).

Other layers: `src/standard_serving.apply` is the Standard repaired scorer, not a calibration. The
Platt/meta blocks live inside each payload and are trained with it. No other post-model shift exists
in `src/predict.py`.

## Train/serve parity

`python scripts/team_train_serve_parity.py <out> <4 harness payloads> 6` compares, on the last 6 match
dates of the committed history (2026-10-25..10-30; 159 fixtures whose clubs play once that day):
- training: `build_features`, the harness frame;
- live: `build_upcoming_features` + `attach_ht_features`, what predict.py runs.

It covers all 43 raw inputs of the payloads (the `__missing` indicators derive from them) and then
scores both frames with each payload.

**Cells:** 628 of about 6,800 differ. Every one of them is in the strength block:

| Column | Cells differing | Relative difference (mean / max) |
|---|---|---|
| home_attack_str, away_attack_str | 107, 107 | 0.02% / 0.11% |
| home_defense_str, away_defense_str | 106, 107 | 0.02% / 0.11% |
| home_ht_attack_str, away_ht_attack_str | 102, 99 | 0.02% / 0.11% |

Every rolling-form input, HT history, rest days, season stats, home-advantage input and the Poisson-DC
probability matches exactly.

**Probability (served blend):**

| Payload | Rows | Mean abs diff | Max abs diff | Rows > 1 point | Rows exact |
|---|---|---|---|---|---|
| over25 / standard | 159 | 0.004 pp | 0.22 pp | 0 | 52 |
| over25 / new_format | 159 | 0.015 pp | 1.09 pp | 1 | 52 |
| btts / standard | 159 | 0.001 pp | 0.05 pp | 0 | 52 |
| btts / new_format | 159 | 0.010 pp | 0.50 pp | 0 | 52 |

**Cause: a same-day leak on the TRAINING side.** The strengths divide by the league's average goals:
- Training: `_league_expanding_avg` (`src/feature_engineering.py:217`) takes the expanding mean "up to (not including) each match" in ROW order, so league matches played EARLIER ON THE SAME DAY count.
- Live: `build_upcoming_features` uses all history strictly before the date, the correct rule (other same-day results are unknown before kickoff).

The live builder is right; the fix belongs in training (expanding mean over earlier DATES only). That
changes every model's inputs, so per the brief it is NOT changed here. **Owner decision D:** fix the
training leak before the refit of decision B (recommended: the refit happens anyway, so the fixed
features cost nothing extra), then re-run this parity check (target: 0 differing cells).

Note: the Standard O/U 2.5 model that SERVES (`v9_repaired`) uses a different builder
(`src/features_repaired`) and was not part of this check (finding 1).
