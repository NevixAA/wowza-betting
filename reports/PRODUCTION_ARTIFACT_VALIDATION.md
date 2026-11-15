# Production artifact validation - team and fantasy (REAL DATA ONLY, 2026-11-15)

Branch `fix/real-data-provenance` (base data-truth-100 5573b01d0). Machine-readable evidence:
`reports/real_data_evidence/production_artifact_validation.json` (rebuild:
`python -m src.integrity.artifact_validation --write`). The props payloads are validated by the props
agent (`fix/props-data-audit`); they appear here only as fantasy dependencies.

## How each artifact is validated

1. Loaded with `src.model.load_models` (the loader `src/predict.py` uses), sha256 of the file.
2. Metadata read from the payload itself: recipe and version, training window and cutoff, code sha,
   feature schema (count + sha256 of the ordered list), imputer version and fit window.
3. **Serving guard** `src.model.serving_refusal` (new): refuses a payload with no persisted
   train-only imputer (batch-median imputation), a 1.0 imputer that fills a feature with no training
   values with a constant, or a feature absent from the imputer manifest.
4. **Production-path inference probe** (`src.integrity.artifact_validation.inference_probe`):
   `predict_proba(serving=True)` - the call `src/predict.py` now makes - on three rows built from the
   payload's own training distribution:
   - `complete`: every input at its training median -> must be a probability in (0, 1);
   - `optional_gap`: every non-required input missing -> must be a probability (explicit imputation
     + `__missing` indicators);
   - `core_gap`: a required core-form input missing -> must be **no probability** (NaN);
   - each row scored alone must equal the batch (batch invariance).
   SAFE only when all four hold; a refused payload is BLOCKED (it serves nothing); one that cannot
   be loaded or probed is UNVERIFIED.

## Team artifacts

| Model | Status | Path | sha256 | Version / recipe | Training window (cutoff) | Features (schema) | Imputer (fit window) | Unsupported referenced | Probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Standard O/U 2.5 | active (`STANDARD_OU25_SERVING_RECIPE=legacy_v9` serves this file) | models/model_v9_standard.pkl | 4025384d...3fc3b9 | r2-repaired-1.0 / std_v9_repaired | 2021-07-30 .. 2026-11-01 (2026-11-01) | 86 = 43 + 43 indicators (2d37b07b4f8fa149) | 1.1 (2021-07-30 .. 2025-10-17) | none | 0.517 / 0.505 / NaN, batch-invariant | **SAFE** |
| Standard O/U 2.5 retired v9_repaired | **RETIRED 2026-11-15** (kept for audit) | models/model_v9_standard_repaired.pkl | 1f5805e6...8a9f1a74 | v9_repaired-1.0 / v9_repaired | 2021-07-30 .. 2026-10-26 (2026-10-28) | 41 (4b5dfddba0ea13f6) | **1.0** (2021-07-30 .. 2024-11-22) | home/away_xg_last5, home/away_insidebox_last5 (FABRICATED_VALUE: 0% training coverage, filled 0.0) | refused | **BLOCKED** |
| New Format O/U 2.5 | active | models/model_v9_newformat.pkl | 9198e5bd...35faa7f6 | r2-repaired-1.0 / nf_v9_repaired | 2019-02-22 .. 2026-11-01 (2026-11-01) | 56 = 28 + 28 (e16255c4dd9bca9f) | 1.1 (2019-02-22 .. 2025-08-23) | none | 0.506 / 0.507 / NaN | **SAFE** |
| BTTS | active, track-split | models/model_v9_btts.pkl | d7cd2762...01d7a7e1 | r2-repaired-1.0 / side_v9_repaired | 2021-07-30 .. 2026-11-01 | 86 standard + 56 new_format | 1.1 both tracks | none | std 0.547 / 0.532 / NaN; NF 0.553 / 0.578 / NaN | **SAFE** |
| Over 1.5 | active, track-split | models/model_v9_over15.pkl | 2af54e42...0b9c82 | r2-repaired-1.0 / side_v9_repaired | 2021-07-30 .. 2026-11-01 | 86 + 56 | 1.1 | none | std 0.768 / 0.759 / NaN; NF 0.770 / 0.794 / NaN | **SAFE** |
| Over 3.5 | active, track-split | models/model_v9_over35.pkl | ffeaa631...650d62 | r2-repaired-1.0 / side_v9_repaired | 2021-07-30 .. 2026-11-01 | 86 + 56 | 1.1 | none | std 0.290 / 0.275 / NaN; NF 0.290 / 0.265 / NaN | **SAFE** |
| HT Over 0.5 | active (Standard fixtures) | models/model_ht_over05.pkl | de69dff4...29623bde | r2-repaired-1.0 / ht_v9_repaired | 2023-07-28 .. 2026-11-01 | 86 | 1.1 (2023-07-28 .. 2026-02-09) | none | 0.740 / 0.721 / NaN | **SAFE** |
| HT Over 1.5 | active (Standard fixtures) | models/model_ht_over15.pkl | 68f5c41c...7ed7811 | r2-repaired-1.0 / ht_v9_repaired | 2023-07-28 .. 2026-11-01 | 86 | 1.1 | none | 0.364 / 0.328 / NaN | **SAFE** |

All active team payloads were trained at code sha a29fdb1ef (2026-11-09), which already contains the
league-average same-day leak fix (2d1d900bf). No challenger team payload exists under
`models/challengers/` (only `props_r2`); there are no paper team artifacts.

**Missing-data behaviour (every active payload):** train-only medians from the base-training block
(1.1 manifest), a learned `__missing` indicator per feature, clip to the training range. A 1.1
manifest raises on a feature with no training values. Since this branch the live call also gives a
fixture **no probability** when a required core-form input (home/away scored/conceded last 5) is
missing: its inputs would be the training medians, not that fixture's football. Before, such a row
was scored and only the betting layer's no-form guard forced AVOID.

**Preprocessing recipe:** four disjoint date blocks (base / calibration / meta / holdout, 0.8 / 0.1 /
0.1 before the 2026-11-02 holdout); imputer and base models fit on base, Platt calibrators on
calibration, meta-logistic on meta. The holdout has only 27 rows (2026-11-02 .. 11-04/06), so the
holdout metrics stored in the payloads are not evidence of quality. The quality evidence is the
Phase 2 harness run (`reports/phase2_runs`).

### v9_repaired payload: RETIRED by the owner (2026-11-15)

`models/model_v9_standard_repaired.pkl` was the owner's 2026-10-28 Standard switch and is the
documented rollback (`STANDARD_OU25_SERVING_RECIPE=v9_repaired`):
- Its 1.0 imputer recorded xG and inside-box shots with **no training values** (the 2026-10-28
  frame did not carry them).
- It filled them with 0.0 on every training row, so the model learned "xG = 0".
- Live boards carry real xG on 98.6% of Standard fixtures, which the model would read as out of
  distribution.

The serving guard now refuses it, so the rollback fails closed (no Standard probability, system
health RED via `standard_serving_health`) instead of serving fabricated inputs.

**Owner, 2026-11-15: "fix what you need" - RETIRED.** `STANDARD_OU25_SERVING_RECIPE` accepts only
`legacy_v9`; `v9_repaired` is refused like a typo. `owner_install` refuses the type and
`train_standard_repaired.yml` stops at its first step. The registry pointer carries `status: RETIRED`
with the date and the owner's words, and the decision log has a `RETIRE` row. The file is kept for
audit (`docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md`). The active Standard payload is already
the clean retrain.

## Fantasy

No fantasy model is serialized:
- **V2** (`player_model/fantasy_v2.project`) fits `LinearResidual` and `MinutesModel` at run time on
  the archived pre-deadline FPL snapshots. This branch fits them on **FINAL gameweeks only**, which is
  what the walk-forward validates. All archived snapshot rows were fetched before their deadline
  (measured: 100% of rows `fetched_at < deadline`, fantasy audit 2026-11-15).
- **V1** (`player_model/fantasy.py`) is formula-based on three props payloads.

| Dependency | sha256 | Present | Notes |
|---|---|---|---|
| models/model_player_goals.pkl | 5de94714...4110038 | yes | props payload; provenance PENDING (props agent) |
| models/model_player_assists.pkl | abf39bb5...810c499b | yes | props payload; PENDING |
| models/model_player_sot2.pkl | 5b5fad60...b2621734 | yes | props payload; PENDING |
| output/fpl_af_crosswalk.csv | 5caaf4f8...52c893ac | yes | 606 id pairs, no date-of-birth-verified pair. Not on main, and no workflow rebuilds it. If it is missing, V1 now publishes nothing (F1) instead of everyone as fit. |
| output/fantasy_fpl_snapshots.csv | 764dffe2...f720cb3 | yes | V2 training panel (pre-deadline only) |
| output/fantasy_projection_log.csv | 623dada5...7562fd | yes | WOWZA arm. On main it has no `fpl_id` column, so `wowza_uncond` is missing on 82% of main's V2 rows. |

**Verdict: PENDING_PROPS.** The fantasy-owned inputs pass. The V1 board depends on props payloads
whose provenance belongs to the props agent and is not yet reported.

The production `fantasy_refresh.yml` runs **main's** fantasy code, about 1,680 lines older than this
branch. Main still carries these typed fills:
- `minutes_pg.fillna(60)`;
- `p_start.fillna(0.6)`;
- goal points `fillna(5)`;
- FDR `or 3`.

The measured clean-sheet probability on main has 5 distinct values, 4 of them the pure-FDR formula.
These fills go away when data-truth-100 (with this branch) reaches main.
