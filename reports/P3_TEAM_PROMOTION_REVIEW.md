# Team-model promotion review after the Phase 2 clean retrain (2026-11-09)

Owner request, 2026-11-09: "why didn't push new format?". Goal: install the clean-retrained team
models (both tracks) into serving for go-live. **Result: nothing was installed.** Three findings
block an honest promotion, and each needs an owner decision. This branch carries the run evidence,
the calibration fix that any promotion needs, and a train/serve parity tool.

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
