# Model Champion High-Water Mark

Owner request 2026-10-18. **Retraining is an experiment. Promotion is a privilege.**

More data keeps producing new models. A new model only replaces the one serving production when it is
convincingly better on the same fixtures. Weekly results can still swing, because football does. What
can no longer regress is the **validated quality of the model in production**.

```
CURRENT CHAMPION ──(new data)──> TRAIN CHALLENGER ──> SAME-ROWS COMPARISON
      worse or tied   -> KEEP_CHAMPION   (challenger REJECTED, metrics kept)
      maybe better    -> SHADOW_MORE     (challenger parked, scored live next to the champion)
      clearly better  -> PROMOTE         (new champion, new high-water mark)
```

## What production looked like before (audit)

| Finding | Evidence |
|---|---|
| Retrains overwrote `models/model_v9_<x>.pkl` in place | `pipeline._train_one` → `save_models` |
| No model had an identity | `provenance.model_sha` hashed size+mtime of *all* pkls; changes every CI run |
| No ledger said which model made a bet | `bets_ledger.csv` / `side_bets_ledger.csv` carry `model_type` only |
| Rule since 2026-10-04 blocked *worse*, but promoted *any* improvement | 2026-10-18: 5 of 7 models promoted, every gain's 95% CI included zero |
| Before 2026-10-04, worse models shipped inside a 0.030 tolerance | 2026-09-24 newformat 0.59537 → 0.59656 (+0.00119) promoted |
| The live rule judged the wrong number | Mean of base-model log losses, not the served ensemble. 2026-10-11 newformat was blocked while the served ensemble was clearly better (CI −0.0052 to −0.0011) |
| Shadow log results were never graded | `v91_shadow.csv`: 572 rows, 0 results |

Replaying the 112 decisions recorded in `retrain_log.json` under the new gate gives
`output/model_lineage/legacy_gate_replay.csv`:

| Old rule did | New gate says | Count |
|---|---|---:|
| promoted | KEEP_CHAMPION: worse or tied | 36 |
| promoted | SHADOW_MORE | 3 |
| promoted | cannot tell: no CI was recorded before 2026-10-05 | 59 |
| blocked | KEEP_CHAMPION | 13 |
| blocked | would have PROMOTED (2026-10-11 newformat) | 1 |

## The gate (`src/champion_gate.py`)

The challenger and the champion are both scored on the **same holdout rows**, with the same labels and
the same scoring function. Both are scored on their **served probabilities**.

**PROMOTE** requires every one of these:

* Log-loss improvement of at least **0.001**. That is above the measured meta-stack bias (0.0006–0.0014)
  and about the CI half-width on our holdouts.
* 95% block-bootstrap CI on the improvement entirely below zero.
* Improvement confirmed on the **later half** of the holdout on its own (the second confirmation
  period).
* Brier score not worse (by more than 0.0001).
* ECE not worse by more than 0.01.
* No league with at least 150 rows losing more than 0.02 log loss.
* Prediction canary passes: the spread has not collapsed and predictions are not pinned at 0 or 1.

**KEEP_CHAMPION** results from any of these:

* The challenger is worse or exactly tied.
* The CI does not even admit a material gain.
* The canary fails.
* **Anything prevents the evaluation**: missing data, a schema mismatch, n < 1000, an exception, or
  rows that are not the same. Fail closed.

**SHADOW_MORE** means the challenger is better on the point estimate and its CI still admits a material
gain, but at least one PROMOTE check fails.

**Evaluation layers:**

| Layer | What it is | Used to promote? |
|---|---|---|
| B. Rolling fresh OOS | The newest chronological block, unseen by the challenger. It moves forward with every retrain, so no single holdout is optimised against forever. | Yes |
| C. Second confirmation | The later half of that block, scored on its own. | Yes |
| D. Prospective shadow | `shadow_h2h.csv`: champion and challenger on the same future fixtures. | Measured by Pro |
| A. Frozen historical benchmark | **Deliberately not used.** Any fixed past slice is inside both models' training data, so it measures memorisation. | No |

**Escape hatch.** An owner-approved architecture change (new feature family, new target) cannot be
scored fairly against an incompatible champion. For that case, set `CHAMPION_FORCE_PROMOTE=<model_type>`
for one run. The override is logged as `OWNER_OVERRIDE` and still requires the canary to pass.

### Known tension

On 2026-10-04, 1,568 simulated comparisons found that no promotion rule beat having no gate, and that
retraining beats freezing. On 2026-10-18 the owner chose stability over that small average edge.
SHADOW_MORE and the head-to-head log exist so that the cost of this choice is **measured**, not assumed.
Pro reports it every week.

## Identity and artifacts (`src/model_registry.py`)

* **Version** is `<model_type>_<trained_at UTC>_<sha256[:8]>`, for example
  `standard_20261018T080423Z_64313878`.
* **Identity is the bytes.** Predict hashes the file it loaded and looks the version up. If that hash
  is not registered, the row is stamped `UNREGISTERED_<type>_<sha12>`. A version is never guessed.
* **Seven independent champions:** standard, newformat, btts, over15, over35, ht_over05 and ht_over15.
  Promoting one never moves another's bar. Player models are not covered yet.

| Path | Contents |
|---|---|
| `models/model_v9_<x>.pkl` | The serving file. Same path and format as before, so predict is unchanged. |
| `models/registry/current_champions.json` | The champion pointer for each model type. |
| `models/registry/<type>/versions/<v>.json` | Immutable manifest: type, version, batch, trained_at, training range/cutoff/rows, feature schema hash, commit, data fingerprint, artifact hash, parent champion, validation metrics, verdict, role. |
| `models/registry/<type>/champions/<v>.pkl.gz` | Immutable bytes of every version that served. |
| `models/registry/<type>/challenger.pkl.gz` | The one parked SHADOW_MORE challenger. |

**Atomic promotion** runs in a fixed order:

1. Immutable copy of the bytes.
2. Manifest.
3. Serving file (write, then rename).
4. Pointer.
5. History row.

**Rollback** is `reg.rollback("<type>", "<version>")`. It restores the exact bytes; no retraining is
needed.

## Attribution

Each new prediction or ledger row gets new columns. They are added at the end of the file, so
existing readers are unaffected:

| File | New columns |
|---|---|
| `predictions.csv` | `model_version`, `retrain_batch_id`, `training_cutoff`, `prediction_role` for the O/U model, plus `model_version_btts`, `model_version_over15`, `model_version_over35`, `model_version_ht_over05` and `model_version_ht_over15` |
| `bets_ledger.csv`, `side_bets_ledger.csv`, `ht_ledger.csv` | `model_version`, `retrain_batch_id`, `training_cutoff`, `prediction_role` |

How the version is assigned:

* **Which model:** an O/U row gets the model that actually scored it, including predict's two
  fallbacks onto the standard model.
* **Which moment:** the version recorded at the first sighting of a tip stays put. A later tier upgrade
  does not rewrite it.
* **Paper and blocked rows** are recorded too, because the ledger keeps every tier and every league.
* **Rows written before 2026-10-18** stay blank, which means `UNKNOWN_LEGACY`.

## Lineage records (`output/model_lineage/`, all append-only)

| File | Written by | One row per |
|---|---|---|
| `retrain_batches.csv` | retrain | Champion-vs-challenger decision: the full audit record (§52) |
| `champion_history.csv` | retrain | Champion era. `end_at` of an era = `start_at` of the next row for that type |
| `high_water_mark.json` | retrain | Model type: historical bar (at promotion) + current-regime bar (latest holdout) |
| `shadow_h2h.csv` | predict | Fixture × model type × challenger, first sight |
| `legacy_gate_replay.csv` | backfill | Pre-2026-10-18 decision, replayed |

**Legacy reconstruction** (`scripts/backfill_model_lineage.py`, run once):

* All 126 committed serving artifacts were registered from git. Identity is the SHA-256 of the committed
  bytes, timed at commit (deploy) time, and each carries `reconstructed: true`.
* Rejected challengers from before 2026-10-18 were never kept, so they stay `UNKNOWN_LEGACY`.

## What did not change

* Prediction logic, features, thresholds, tiers, staking, BET_ENABLED and league collection.
* Telegram tips and the serving file format.

**Retrain cadence.** Unchanged: Sundays.

**Model promotion vs. betting.** Promotion never enables betting, and betting eligibility never decides
which data trains a model.

## Results analysis lives in Pro

`wowzaV9-Pro: src/analytics/model_lineage.py` grades `shadow_h2h.csv`, attributes ledger bets to
champion eras, and produces:

* champion live performance;
* retrain-batch future results;
* the weekly per-version snapshot;
* churn;
* whether promotions improved later results.

Champion health (HEALTHY / WATCH / DEGRADED / STALE) is computed there too.
