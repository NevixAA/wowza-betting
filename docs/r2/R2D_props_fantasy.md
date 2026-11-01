# R2-D — Props preprocessing integrity and Fantasy defaults (findings 20, 21, 22)

Branch `r2/props-integrity` (not merged). Workstream R2-D of the 2026-11-01 model-integrity repair.
Player props remain PAPER permanently; nothing here stakes, promotes or changes a serving file.

## 1. Defects

| # | Defect | Files | Severity | Reachable in production today |
|---|---|---|---|---|
| 20a | `_prep` imputes NaN with the median of the frame being processed. In TRAINING that frame is the whole history, so holdout (and calibration) statistics enter the fit. | `player_model/model.py::_prep`, `train` | P0 (integrity) | Yes, every weekly props retrain (`player_props_train.yml`) and every PG-PROPS candidate. |
| 20b | At PREDICTION the frame is one fixture's players (`predict.py` scores `feat_df` per fixture), so a player's probability depends on who else is in the batch. | `model.predict_proba`, `player_model/predict.py:692`, `fantasy.py:227`, `fantasy_features.py:115`, `props_gate.py:777,884` | P0 | Yes. Measured on the 8 serving payloads (`output/player_props_integrity_health.json`): single-row vs batch differs by up to **1.29pp** (goals 1.26pp, sot3 1.29pp, cards 0.97pp, assists 0.93pp) when 10 of 120 rows carry a NaN. |
| 21 | A feature column absent from the scoring frame is filled with 0.0 for every row (log warning only). | `model._prep` | P1 | Latent: the live builder `build_upcoming_features` currently emits all 138 contract features (checked statically), so it fires only if a feature is added to `config.PLAYER_FEATURE_COLS` without the live builder. Many features are instead given inline constants in the builder (e.g. `opp_gk_save_rate` 0.72, `league_tier` 3) — see §6. |
| 21b | No row records how many inputs were missing or degraded, or which imputer/contract produced it. | `model.py` | P2 | Yes. |
| — | Serving props champions (registered 2026-10-25) were trained on pre-2026-10-26 `player_history` with leaked (today-stamped) injury values. | data | known | Yes; recorded per track in the health file (`training_data_note`). Both R2 arms train on the repaired data, so the comparison isolates preprocessing. |

## 2. Repair

* **Feature contract** `player_model/props_feature_contract.json` (`PFC-1.0`, sha-256 in file), generator and
  loader `player_model/props_contract.py`. 138 features: 12 `REQUIRED_FAIL_ROW` (venue, history depth,
  minutes, starter rate, position encodings, the five core rates), 3 `SEMANTIC_ZERO` (red cards, penalties won,
  offsides — provider omits events that did not happen), 40 `OPTIONAL_DEGRADED` (injury, referee, season stats,
  profile, league/opponent quality, opponent GK/defender detail, set-piece roles), 83 `IMPUTABLE`
  (own-history derived). A test fails if the file drifts from the feature list or is edited without re-hashing.
* **Persisted imputer** `PropsImputer` (`PROPS-IMPUTER-1.0`): fitted on the FIT block only
  (rows `0 .. cal_split`, i.e. the first 68% chronologically); fill = fit-block median (0 for SEMANTIC_ZERO);
  a feature with no fit-block data is DROPPED and listed (`dropped_unfittable`), never given an invented constant.
  `transform` is a pure per-row function: batch-invariant by construction. Required-feature missing →
  the row gets **no probability** (NaN). Every scored row carries `n_missing`, `n_degraded`,
  `n_required_missing`, `required_fail`, `imputer_version`, `contract_version`, `recipe`
  (`model.predict_with_diagnostics`).
* **Recipes.** `model.train(recipe="legacy")` is the DEFAULT and unchanged — the weekly retrain and PG-PROPS
  produce exactly what they did. `recipe="props_r2_repaired"` is the challenger: same split bounds, estimators,
  hyper-parameters; REQUIRED_FAIL rows excluded from fit/calibration. Its payload adds `recipe`, `imputer`
  (state), `imputer_version`, `contract_version`, `blocks` (index and date ranges of fit / calibration /
  holdout), `required_fail_excluded`, `training_rows`. A legacy payload gains no keys, so its pickle bytes and
  registry sha are unchanged.
* **Dispatch.** `model.predict_proba` routes on `payload["recipe"]`; a payload without the key (all serving
  props models) runs the original `_prep` path bit-for-bit — proven against all 8 serving files.
* **Challenger replay** `scripts/props_r2_challengers.py`, workflow `.github/workflows/props_r2_challengers.yml`
  (GitHub Actions only; triggers on push to this branch or dispatch). Writes `output/props_r2/props_r2_challengers.json`,
  challenger payloads `models/challengers/props_r2/props_r2_<market>.pkl` (deliberately NOT `model_player_*`;
  the PG-PROPS static guard forbids writing serving names), and uploads the per-row scored holdout
  (`props_r2_holdout_scored.csv.gz`, with the integrity columns) as a run artifact.
* **Experiments** `config/experiments/props_r2/PROPS-R2-<market>.json`, frozen and committed BEFORE the first
  result: hypothesis, baseline, blocks, versions, metrics (LL, Brier, ECE first; AUC secondary; no market
  metrics — paper), frozen pass criteria (upper 95% week-block CI of ΔLL ≤ +0.001 and ECE not worse by > 0.005),
  promotion: none automatic.
* **Health** `player_model/props_integrity_health.py` → `output/player_props_integrity_health.json`: per track
  recipe, serving version, artifact-vs-registry sha, imputer/contract versions, missing-feature policy, holdout
  independence, chronology, live batch-invariance probe, injury-leak note, challenger result, PASS/WARN/FAIL;
  plus the Fantasy defaults section.

## 3. Tests (`tests/test_props_r2_integrity.py`, 20 test cases)

Contract covers exactly the feature list and is hashed; unknown feature refused; **legacy parity on all 8 serving
payloads** (NaNs injected + an absent column, equality to a frozen copy of the pre-R2 code); serving files match
the registry sha; legacy payload keys unchanged; legacy batch dependence characterised (documents the defect);
repaired payload carries versions and ordered date blocks; fill values equal fit-block medians and do not move
when calibration/holdout rows are perturbed ×50; single row == same row in a batch == reversed batch == batch with
40 unrelated extreme rows (atol 1e-12, BLAS summation order only); absent column → fit median, not 0 (0 only for
SEMANTIC_ZERO), counted missing/degraded; required feature missing → NaN probability; pickle round-trip; Fantasy
register classified and descriptive. Also passes: `test_props_promotion_gate.py` (22, including the static
"only the gate writes serving files" guard), `test_fantasy_v2.py` except the known pre-existing failure.

## 4. Before/after preprocessing — PROPS-R2 retrospective replay (GitHub Actions)

Run: workflow `props_r2_challengers.yml`, run id 41297477426, code `1970aaa23`, results committed by CI in
`175efa820` (`output/props_r2/props_r2_challengers.json`). Data: `player_history.parquet` at that commit,
875,853 rows, 2022-08-12 .. 2026-10-31. 21.8 min for 8 markets × 2 arms.
Blocks (goals; others identical except sot4): fit 2022-08-12..2025-08-23 (595,579 rows) | calibration 2025-08-23..2026-01-10 |
holdout 2026-01-10..2026-10-31. Evidence class RETROSPECTIVE. Both arms on the same rows; no REQUIRED_FAIL rows occurred,
no feature was unfittable.

| market | holdout n | events | LL legacy | LL repaired | ΔLL [95% week-block CI] | Brier leg / rep | ECE10 leg / rep | calib slope leg / rep | AUC leg / rep | legacy as-served shift: rows moved / max | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goals | 175,171 | 14,631 | 0.26233 | 0.26233 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.07254 / 0.07254 | 0.0189 / 0.0189 | 0.905 / 0.905 | 0.7422 / 0.7422 | 698 / 5.18pp | RETROSPECTIVE_PASS (non-inferior) |
| goals2 | 175,171 | 1,367 | 0.04326 | 0.04326 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.00775 / 0.00775 | 0.0062 / 0.0062 | 1.154 / 1.154 | 0.8407 / 0.8407 | 692 / 0.14pp | RETROSPECTIVE_PASS (non-inferior) |
| assists | 175,171 | 10,720 | 0.22023 | 0.22023 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.05641 / 0.05641 | 0.0112 / 0.0112 | 0.934 / 0.934 | 0.6830 / 0.6830 | 698 / 5.03pp | RETROSPECTIVE_PASS (non-inferior) |
| sot | 175,171 | 39,819 | 0.47651 | 0.47651 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.15475 / 0.15475 | 0.0266 / 0.0266 | 0.934 / 0.934 | 0.7346 / 0.7346 | 698 / 1.63pp | RETROSPECTIVE_PASS (non-inferior) |
| sot2 | 175,171 | 9,442 | 0.18593 | 0.18593 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.04867 / 0.04867 | 0.0184 / 0.0184 | 0.874 / 0.874 | 0.8016 / 0.8016 | 698 / 6.83pp | RETROSPECTIVE_PASS (non-inferior) |
| sot3 | 175,171 | 2,183 | 0.06114 | 0.06114 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.01227 / 0.01227 | 0.0081 / 0.0081 | 1.001 / 1.001 | 0.8518 / 0.8518 | 698 / 1.29pp | RETROSPECTIVE_PASS (non-inferior) |
| sot4 | 174,090 | 464 | 0.01846 | 0.01846 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.00266 / 0.00266 | 0.0032 / 0.0032 | 0.961 / 0.961 | 0.8881 / 0.8881 | 326 / 0.01pp | RETROSPECTIVE_PASS (non-inferior) |
| cards | 175,171 | 22,330 | 0.37237 | 0.37237 | +0.0e+00 [+0.0e+00, +0.0e+00] | 0.10926 / 0.10926 | 0.0081 / 0.0081 | 1.017 / 1.017 | 0.6202 / 0.6202 | 698 / 1.09pp | RETROSPECTIVE_PASS (non-inferior) |

**Reading.** On the historical holdout the two recipes are indistinguishable to 5 decimals in every metric. That is
expected and is itself the finding: in `player_history` only the three injury columns are ever missing (~15% of holdout
rows), and their whole-frame median equals their fit-block median (365 / 0 / 0). So the *training-time* leak (20a)
is real in mechanism but has **no measurable effect on today's data**; it would bite as soon as a feature with
drifting missingness appears. The *serving-time* defect (20b) is what moves numbers: re-scoring the same holdout per
fixture (as `predict.py` does) with the legacy payload shifts **~700 rows per market by up to 6.8pp (sot2), 5.2pp
(goals), 5.0pp (assists)**; aggregate LL is unchanged because those rows are few. The probe on the serving payloads
(health file) shows the same: up to 1.29pp single-vs-batch. The repaired recipe removes both by construction at
zero accuracy cost.

## 5. Production behaviour

None changed. Serving props payloads keep the legacy path (bit-identical, tested). `train()` default stays legacy, so
weekly PG-PROPS candidates are unchanged. `fantasy_health.json` gains an additive `defaults_audit` block.
`python -m src.research_lock --check`: protected_tracks_modified=False.


## 6. Fantasy defaults (finding 22)

Register: `player_model/fantasy_defaults.py` (`FANTASY-DEFAULTS-1.0`, 34 entries, descriptive only), exposed in
`output/fantasy_health.json["defaults_audit"]` (additive; `status` unaffected) and in the props integrity health
file, with fire counts read from the committed outputs. **Fantasy V2 behaviour is unchanged.**

| Default | Where | Class | Reach | Note |
|---|---|---|---|---|
| minutes 60 | fantasy.py:229 | MISSING_DATA_FALLBACK | V1 | dead in practice (rows filtered to ≥45 min) |
| p_start 0.6 | fantasy.py:354 | MISSING_DATA_FALLBACK | V1→V2 | lands on the 0.60 captain floor, unflagged |
| position MID | fantasy.py:302-303 | MISSING_DATA_FALLBACK | V1 | only when FPL matching fails (`fpl_matched`) |
| goal points 5 | fantasy.py:306 | MISSING_DATA_FALLBACK | V1 | dead after alias |
| GKP goal points 6 | fantasy.py:33,35; fantasy_panel.py:244 | OBSERVED (rule) | V1→V2 | **wrong: FPL game_config says 10** |
| referee 3.5 yellows | feature_engineering.py:312-319,1423 | MISSING_DATA_FALLBACK | props cards features | |
| heuristic opp xG 0.5+0.42·FDR | fantasy.py:323 | DOMAIN_PRIOR | V1→V2 | unfitted, averaged over 5 fixtures |
| CS λ 0.6/0.4 blend | fantasy.py:333 | DOMAIN_PRIOR | V1→V2 | |
| DC thresholds 5/4.5/4 | fantasy.py:41 | DOMAIN_PRIOR | V1→V2 | **suspected bug** (FPL: DEF ≥10 CBIT, MID/FWD ≥12) |
| bonus ×0.66 | fantasy.py:172 | TRAIN_FITTED_PRIOR | V1→V2 | fitted with different inputs than production |
| SOT2 weight 0.45 | fantasy.py:309 | TRAIN_FITTED_PRIOR | V1→V2 | lstsq, hard-coded |
| missing props model → p=0 | fantasy.py:227 | MISSING_DATA_FALLBACK | V1→V2 | silent, HIGH impact if it fires |
| props batch median | model.py `_prep` | MISSING_DATA_FALLBACK | V1→V2 | finding 20 (legacy only) |
| V2 p_start = P60 + 0.35·P(1-59) | fantasy_residual.py:708 | DOMAIN_PRIOR | V2 | unfitted; drives Confidence/Risk and captain pools |
| V2 attack/defence shares | fantasy_v2.py:67-68 | DOMAIN_PRIOR | V2 | |
| V2 no-ppg → 2.0 pts/fixture | fantasy_v2.py:320 | MISSING_DATA_FALLBACK | V2 | phantom 3/5-GW points |
| V2 unknown team → league mean λ | fantasy_panel.py:143 | MISSING_DATA_FALLBACK | V2 | unknown looks average |
| V2 chance NaN → 100/0 by status | fantasy_panel.py:229 | MISSING_DATA_FALLBACK | V2 | FPL convention |
| V2 residual features NaN → 0 | fantasy_residual.py:84 | MISSING_DATA_FALLBACK | V2 (shadow) | |
| FPL status unknown → available | fpl_api.py:229 | MISSING_DATA_FALLBACK | V1 | more confident on unknown |
| seasons ('2026','2025') | fantasy.py:693,712 | MISSING_DATA_FALLBACK | V1 | hard-coded season |

Full list (34) with importance and detectability in the module.

**Suspected correctness bugs — reported, NOT changed (owner decision):**
1. **DC thresholds (HIGH):** `fantasy.py:41` uses 5/4.5/4 actions including duels_won; FPL 2025/26 rule is DEF ≥10 CBIT,
   MID/FWD ≥12 CBIRT. Today's `fantasy_tips.csv` mean dc_pts FWD 1.29 > DEF 1.28; in `fpl_bootstrap.json` 0% of FWDs
   reach 12. Feeds `wowza_uncond`/`wowza_gap` (V2 shadow residual + WOWZA benchmark arm); plausibly much of the
   documented +1.0 Wowza bias. FPL now publishes `defensive_contribution` directly.
2. **GKP goal = 6, should be 10** (LOW): four places.
3. **Bonus receives CS points (0–4) where it expects a probability** (MED): `fantasy.py:345`.
4. Status "n" not treated as out in V1 (latent; V2 is correct).
5. V1 "next GW" numbers are 5-fixture averages but settled against one GW (MED).
6. V2 p_start constant 0.35 unfitted though `actual_starts` exists (MED).
7. `fantasy_health` status pinned at AGING (official_squads hard-coded UNKNOWN_AGE).
8. Hard-coded API-Football seasons in `fantasy.py` (MED from mid-2027).
9. V2 ranking gate has no CI although its docstring promises one (LOW).

## 7. Open risks

* Serving props payloads remain legacy (batch-median) until an owner-approved retrain; health stays WARN.
* The repaired recipe is not yet scored live (no SHADOW first-sight path); next lifecycle step is a shadow
  scorer in `player_props.yml` writing the per-row integrity columns.
* `train(recipe=...)` default is legacy, so the weekly PG-PROPS candidates still carry the leak; switching the
  default is an architecture change for the owner.
* Feature-builder inline constants (`feature_engineering.py`, e.g. GK save rate 0.72, league tier 3, referee 3.5)
  are upstream "invented values" the contract cannot see (they arrive as non-missing). Not changed here because
  they alter serving features; listed for the default-variable audit.
* Block boundaries are row-count based (`split_bounds`, shared with PG-PROPS): 2025-08-23 and 2026-01-10 each
  appear on both sides of a boundary (same-day rows split across fit/calibration and calibration/holdout). Props
  features are per-player as-of values, so no league-wide same-day aggregate is known to leak, but a date-aligned
  boundary would be cleaner; not changed because the gate's holdout definition depends on it.
* Historical replay cannot show the serving defect in aggregate metrics (few rows affected); the live effect should
  be measured on first-sight rows once a shadow scorer exists.
