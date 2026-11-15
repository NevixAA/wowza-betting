# Merge notes for `feat/team-info-features` (from `fix/real-data-provenance`, 2026-11-15)

`fix/real-data-provenance` is merged into data-truth-100. `git merge-tree` of the two branch heads
(feat/team-info-features 411dda3c0 and this branch) reports **no conflict**. The regions below are
what changed, so a rebase or merge can be checked by eye.

## src/feature_engineering.py

| Region | Change | Your hunks |
|---|---|---|
| New `home_advantage_flags(hist)` (after `_add_home_advantage`, before `_add_rest_days`) | training's own home-advantage flags, read by serving | yours insert `LEAGUE_LEVEL_FEATURES` / `add_league_level_features` / `league_level_asof` after `_league_expanding_avg`: different place |
| New module constant `SEASON_VENUE_AF_COLUMNS` (just above `def build_upcoming_features`) | API-Football `/teams/statistics` field to display column | - |
| `build_upcoming_features`, after `_parity = os.getenv("NF_FEATURE_PARITY"...)` | `_ha_latest = home_advantage_flags(hist_league)` | - |
| `build_upcoming_features`, the per-row home-advantage block (was `lg_avg_away` / `_no_ref` / `ht_home_rows` / `at_home_rows`) | replaced by `_ha(team)` reading `_ha_latest` | yours add `league_level_asof` after the HT-strength block: near, not overlapping |
| `build_upcoming_features`, rest days (`combined = pd.concat([_h, _up])`) | computed once per distinct fixture (`_fkey` / `_upk` / `_first` / `_rd`) | - |
| `build_upcoming_features`, "Phase 3: /teams/statistics" block | no longer writes the six season-venue model features; fills `af_*` display columns | - |

## src/model.py

| Region | Change | Your hunks |
|---|---|---|
| New `UnservablePayload`, `serving_refusal(payload)` before `predict_proba` | real-data serving guard | - |
| `predict_proba(..., serving=False)`, `_predict_by_track(..., serving=False)`, `_predict_flat(..., serving=False)` | the new keyword is passed through. With `serving=True`: refuse payloads with no imputer, a 1.0 imputer filling a no-training-values feature, or a feature absent from the manifest; NaN for rows missing a REQUIRED (core form) input | yours add the `calibration_input == "logit"` transform inside the `meta_model` try block of `_predict_flat`. Mine adds lines **before** `base_models = ...` (the `degraded` mask) and **after** the ensemble (NaN on degraded rows). Both survive the merge. |

**What it means for your payloads:**
- Every `feature_cols` entry must be in the payload's imputer manifest (`train_repaired` does that).
- Payloads must carry `imputer`.

## src/predict.py

| Region | Change | Your hunks |
|---|---|---|
| New `SERVING_REFUSALS` + `_serve_proba(frame, payload, what)` before `def predict_upcoming` | every live scoring call goes through `predict_proba(serving=True)`; a refused payload scores nothing (NaN, ERROR log) | yours add the `team_info_features.attach_for_payloads` call after `build_upcoming_features`: separate lines |
| The five `predict_proba(...)` calls (Standard, NF, side markets, HT 0.5 / 1.5) | now `_serve_proba(...)` | - |

## New gate you will meet: Data truth 8b (gated)

`config/feature_provenance.json` lists every feature of every served payload. When a payload that
reads your new features (team info, `league_asof_*`) is installed, those features are **UNVERIFIED**
until they are traced. That makes **8b RED** (a new violation) until both of these are done:

1. Add each feature to `src/integrity/real_data_provenance.TEAM_FEATURES`: source type, provider,
   raw columns, an anchor `(file, regex)` for its transformation, class.
2. Run `python -m src.integrity.real_data_provenance --write`.

`tests/test_real_data_pipeline_trace.py::test_serving_equals_training_for_the_next_match` checks
train/serve parity for every TEAM_FEATURES entry, so add your features there once they are traced.

## Also changed on this branch (not in your files)

- `config.py`: `STANDARD_OU25_SERVING_RECIPES = ("legacy_v9",)`. v9_repaired is RETIRED by the owner
  (2026-11-15).
- `src/standard_serving.py`: the repaired branch was removed.
- Registry and lineage: the `standard_repaired` pointer is RETIRED.
- `src/integrity/data_truth.py`: rule 8 hook (8a / 8b / 8c), and `"Data truth 8b"` added to `GATED`.
