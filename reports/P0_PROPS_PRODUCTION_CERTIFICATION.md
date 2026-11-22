# P0 item 5 - Props production certification (2026-11-22)

Repo: NevixAA/wowza-betting. Branch `p0/props-cert-synthetic-roi`, based on origin/main `2a86ab87a`
(merged up to `9e607e24c` before the final suite). Evidence comes from the real serialized artifacts on
the branch (they are byte-identical to main: same sha256) and from the GitHub Actions artifact of run
43035218594.

## Verdict

**VERIFIED, 13 of 13 installed props artifacts.** Data truth 13a no longer reads "props PENDING":
`props_status` returns VERIFIED only while every installed artifact still hashes to what was certified.

| Role | Market | Version | Candidate | Features | Cutoff |
|---|---|---|---|---|---|
| production | goals | props_goals_clean_hgb_nan_native_5de94714 | hgb_nan_native | 128 | 2026-08-31 |
| production | assists | props_assists_clean_hgb_nan_native_abf39bb5 | hgb_nan_native | 128 | 2026-08-31 |
| production | sot | props_sot_clean_hgb_nan_native_571b102c | hgb_nan_native | 128 | 2026-08-31 |
| production | sot2 | props_sot2_clean_hgb_nan_native_5b5fad60 | hgb_nan_native | 128 | 2026-08-31 |
| production | cards | props_cards_clean_recipe_retrained_bfd5e1cc | recipe_retrained | 128 | 2026-08-31 |
| rare served (research picks only) | goals2 / goals3 / sot3 / sot4 | 81b5b864 / 89f588a0 / f11626d9 / e0d074b8 | player_base_rate | player_id only | 2026-08-31 |
| research challenger | goals2 / goals3 / sot3 / sot4 | e218cb6c / 62e5b2bb / a3646f1d / 8ed4b805 | hgb / recipe / hgb / recipe | 128 | 2026-08-31 |

## Root cause of PENDING

`real_data_provenance.props_status` waited for `output/props_real_data_provenance.json`, a file nothing
produced. The committed `output/player_props_integrity_health.json` was written before the clean
promotions. There was also a latent defect: `props_status` returned REPORTED whenever a file existed,
whatever it said. A failing or stale file would have made 13a GREEN.

## What is verified per artifact (`src/integrity/props_certification.py`)

| Check | How |
|---|---|
| exact sha256 | sha256 of the decompressed bytes = version manifest = pointer (or challenger.json) = version suffix |
| raw input provenance | owner feature rule `all_prior_verified`, floors 0.95 (training) / 0.85 (holdout). Run 43035218594 (code 6fd79496): verified share 0.9979 training / 0.9884 holdout. The payload's harness artifact sha256 equals the run report. Offline check: the downloaded harness pickles were re-hashed, and each installed model gives probabilities **bit-identical** to the run's own model on 1,200 real rows. |
| feature schema | 128 features = `config.PLAYER_FEATURE_COLS`, all in the feature contract, 0 withheld / proxy / removed. The schema hash equals the run's (59e3dd18). Base-rate payloads read no feature, only `player_id`. No served feature holds one value on more than 99.9% of training rows. |
| training window and cutoff | fit 2022-08-18 to 2026-01-09; calibration 2026-01-10 to 2026-08-31 (= cutoff in the manifest and pointer); holdout 2026-09-01 to 2026-10-31 |
| chronological validation | 8 of 8 walk-forward folds per market, each trained strictly before its test week, plus the untouched holdout (n 55,907; cards 55,908) |
| historical coverage | 734,045 fit + 130,673 calibration rows (= manifest) |
| missing-data handling | hgb: NaN-native, nothing filled. cards / recipe challengers: persisted fit-block medians for 128 features. Base rate: an unseen player gets the fitted window market rate. |
| no batch statistics | runtime probe on real player_history rows. Each of 25 players is scored alone, with 10 others, with 1,000 others, in a shuffled batch, and beside 200 unrelated rows with 40% of inputs missing. hgb and base-rate models are bit-identical. cards (recipe) differs by at most 5.6e-17 on 25 rows, the float rounding of its logistic dot product. Tolerance is 1e-12. |
| identity | base-rate keys are positive API-Football player ids. Global identity tests are present (stable id, fixture side, club conflict, id conflict). |
| train/serve recipe parity | Serving calls the harness model's own `predict()`. Pickled sklearn 1.9.1 = CI runtime 1.9.1 (checked in the player_props run log). The committed parity report of run 43035218594 shows 6,316 serving-vs-training rows, 0 mismatched features and 0 non-identical probabilities in all 5 ML markets. |
| permission | production = owner-approved champion. Rare = champion of a hidden market, research picks only. Challenger = SHADOW / SHADOW_RESEARCH, not gate-qualified, never the serving bytes. |

## Legacy `_prep()` (batch median): not reachable in production

- **Call graph** (AST, every non-test module that imports `player_model.model`): `_prep` is called only at
  model.py lines 255 (`train(recipe="legacy")`), 443 and 468 (the legacy branch of
  `predict_with_diagnostics` and `predict_proba`). That branch runs only for a payload that has neither the
  clean recipe nor the repaired recipe. `train()` is called only by `scripts/props_r2_challengers.py`
  (research replay; `save_model` raises). No production module calls it, and the weekly
  `pipeline.mode_train` is retired.
- **Runtime test**: `_prep` was replaced by a function that raises. Through the production entry points it
  scored all 9 serving payloads and the 8 R2 shadow payloads that `predict.py` also scores. `_prep` was
  called 0 times.
- **Residual risk (not fixed, needs owner approval)**: `props_gate.rollback_props` can still re-install a
  legacy registry version. The certification would then go FAILED and STALE, but no code refuses the
  legacy branch. Recommended: refuse a non-clean payload in `predict_proba` for serving.

## Disclosed findings (not in the verdict; owner review)

1. **Mixed-dtype sensitivity of the cards model (recipe_retrained).** `player_history` stores features as
   float32. `RecipeRetrained._fill` fills NaN inside the input column's dtype. When float64 rows sit next to
   float32 targets, the targets' fill values are no longer rounded to float32. The probability then moves by
   up to 6.7e-5 (sot4 challenger: 2.9e-14). This is not a batch statistic, and the dtype is uniform inside
   one production feature frame. The serving-vs-training parity report is exact. A fix (cast to float64
   before filling) would change cards serving by up to about 1e-4 against how the holdout was scored, so it
   needs an owner decision.
2. **Holdout replay on today's data.** The run's scored holdout was re-scored by the installed artifacts on
   the holdout frame rebuilt today with `props_harness.build_frame`:
   - Base-rate markets: 100% identical.
   - ML markets: 98.7-99.4% of rows within 1e-9.
   - The rest moved by up to 0.094 (cards). Their inputs changed after 2026-11-11 (history corrections and
     re-fetches since).

   The holdout metrics describe the 2026-11-11 data snapshot, not today's history.
3. **Fantasy V1** (`p_goal` / `p_assist` / `p_sot2`) scores these certified payloads on the fantasy frame. Its
   provenance row stays UNVERIFIED (depends_on_props) because parity of that frame was not certified here.

## Files changed

- `src/integrity/props_certification.py` (new): the certification, run-evidence extraction, offline identity and
  holdout replay.
- `src/integrity/real_data_provenance.py`: `props_status` re-hashes the artifacts and returns
  VERIFIED / FAILED / STALE / PENDING / UNREADABLE. 13a is GREEN only on VERIFIED.
- `output/props_real_data_provenance.json` (new, generated, no prices).
- `reports/p0_props_cert/run43035218594_evidence.json` (new): the run's shas, verified shares, windows, folds and
  holdout counts. No probabilities, no prices, no CLV.
- `tests/test_props_feature_provenance.py`, `tests/test_props_batch_invariance.py` (new).

## Before / after

| | Before | After |
|---|---|---|
| 13a props | PENDING (AMBER) | VERIFIED (13/13) |
| A failing or stale props file | read REPORTED (GREEN) | FAILED / STALE (AMBER) |
| Batch invariance of the installed artifacts | not tested | tested per artifact, real rows |
| `_prep` reachability | assumed | proven by AST call graph + runtime |

## Regenerate

```
python -m src.integrity.props_certification --write
# offline extras (run artifact expires 2026-12-11):
gh run download 43035218594 -R NevixAA/wowza-betting -n props-clean-retrain -D <dir>
python -m src.integrity.props_certification --write --run-artifact <dir> [--holdout-frame <parquet>]
```

Production impact: none. No model, serving file, pointer, stake or tip changes.
Owner approval: not needed for this PR. It is needed for finding 1 and for the rollback guard.
