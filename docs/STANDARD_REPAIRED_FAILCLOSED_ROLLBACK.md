# Standard O/U 2.5 serving: fail CLOSED instead of falling back to legacy (R2 finding 17)

Owner-facing note, written BEFORE the code change (R2 working rule). Branch
`r2/registry-season-failclosed`. Nothing here changes a threshold, a tier rule, a stake or the
serving recipe that is deployed today (`STANDARD_OU25_SERVING_RECIPE: "v9_repaired"` in
`.github/workflows/predict.yml` stays exactly as it is).

## What changes

| Situation | Before (main) | After (this branch) |
|---|---|---|
| `v9_repaired` scores normally | repaired p_over25, `serving_recipe=v9_repaired` | **unchanged** |
| Repaired scoring FAILS (file missing, payload invalid, feature build error, row misalignment, non-finite probability) | every Standard row silently re-scored by the **legacy** model, stamped `legacy_v9_fallback`; legacy tips still sent | every Standard row gets **no O/U 2.5 probability** (`p_over25` empty), so **no O/U 2.5 tip** for Standard fixtures; rows stamped `serving_recipe=v9_repaired_failed`; ERROR in the log; `output/standard_serving_health.json` says `FAIL`; system health turns RED |
| `STANDARD_OU25_SERVING_RECIPE` set to an unknown value (typo such as `v9_repaird`) | silently treated as `legacy_v9` | **configuration error**: `pipeline.py --mode predict` stops before calling OddsAPI, the workflow run goes red, the error names the bad value and the two valid ones |
| `STANDARD_OU25_SERVING_RECIPE: "legacy_v9"` (explicit owner rollback) | legacy model serves Standard O/U 2.5 | **unchanged** - still works exactly as before |
| Line deleted / variable unset or empty | legacy_v9 | **unchanged** (legacy_v9), but now logged as a WARNING so it cannot happen unnoticed |

What is NOT affected by a repaired-scoring failure: New Format O/U 2.5, the BTTS / O1.5 / O3.5
side models and the HT models keep scoring and tipping exactly as before (they never read the
repaired path). Only Standard O/U 2.5 tips stop.

Why: a fallback to legacy means the system quietly serves a model the owner moved away from
(2026-10-28 "switch Standard now"), and the tips look normal. The only trace was one column value.
A failure must be visible and must not produce bets from a model nobody chose.

## How to roll back

There are two independent rollbacks. Use the one that matches what you want.

1. **Serve the legacy Standard model on purpose** (the old owner rollback, still supported).
   In `.github/workflows/predict.yml`, step "Run predict pipeline", change

   ```yaml
   STANDARD_OU25_SERVING_RECIPE: "v9_repaired"
   ```
   to
   ```yaml
   STANDARD_OU25_SERVING_RECIPE: "legacy_v9"
   ```
   Commit, push. The next predict run serves `models/model_v9_standard.pkl` for Standard O/U 2.5.
   Only these two spellings are accepted; anything else stops predict with a configuration error.

2. **Undo this fail-closed behaviour itself** (go back to the silent legacy fallback). Revert the
   R2-A commit that touched `src/standard_serving.py` (`git revert <sha>`; the SHA is listed in
   `docs/r2/R2A_registry_season_failclosed.md`). There is deliberately no environment switch for
   it: a switch that silently re-enables a fallback is the defect being removed.

## How to tell it fired

* `output/predictions.csv`: Standard rows with `serving_recipe = v9_repaired_failed` and empty
  `p_over25`; their `bet` is `AVOID`.
* `output/standard_serving_health.json`: `status: FAIL`, the exception type and message, the
  number of Standard fixtures left unscored, the time.
* `src/system_health.py` "Standard OU25 serving" check: RED.
* The predict log line `STANDARD_OU25_SERVING_RECIPE=v9_repaired but repaired scoring FAILED ...
  NO probability, NO bet (fail closed)`.

## What to do when it fires

Do not switch to legacy reflexively. First read the error in `standard_serving_health.json`:
a missing or corrupt `models/model_v9_standard_repaired.pkl` is fixed by restoring that file (it
is installed only by `src/promotion_gate.owner_install`); a feature-build error is a code bug to
fix. Rollback 1 is the owner's decision when Standard tips must resume before the fix lands.
