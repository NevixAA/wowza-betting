# REAL DATA ONLY - final audit, team and fantasy models (2026-11-15)

Owner P0 (2026-11-15): *no football observation may be invented, silently approximated or presented
as real when the data is unavailable. Derived features are fine when every input is valid,
point-in-time and traceable.*

- **Scope:** the team models (Standard O/U 2.5 including `src/standard_serving`, New Format O/U 2.5,
  BTTS, Over 1.5, Over 3.5, HT 0.5/1.5), the fantasy models, the provenance registry, the health gate
  and production artifact validation.
- **Out of scope:** the props models (another agent, `fix/props-data-audit`) and the team all-data
  features (another agent, `feat/team-info-features`).
- **Branch:** `fix/real-data-provenance`, base data-truth-100 5573b01d0.

Companion reports:
- `FEATURE_PROVENANCE_REGISTRY.md` (every feature of every served payload);
- `PRODUCTION_ARTIFACT_VALIDATION.md`;
- `CLEAN_MODEL_BACKTEST.md` (team section);
- `REAL_DATA_FINAL_VERDICT.md`.

## 1. Method

1. **Feature lists come from the serialized payloads** (`src.model.load_models` on the files
   `src/predict.py` loads), never from config. There are 43 distinct base features across the team
   payloads, plus one `__missing` indicator per feature on the R2 payloads.
2. **Every base feature traced** from provider to model: raw columns, transformation (file:line,
   located at build time from an anchor), point-in-time rule, missing policy
   (`src/integrity/real_data_provenance.TEAM_FEATURES`).
3. **Coverage measured, never assumed:**
   - training coverage = 1 - `missing_rate` on the base-training block, from each payload's own
     train-only imputer record;
   - serving coverage = share of the board's fixtures of that track with a value (the committed
     board and origin/main's live board of 2026-11-15, 128 fixtures).
4. **Preprocessing checked:**
   - `src/model._prep`: legacy batch median, all-missing column becomes 0.0;
   - `src/imputer`: 1.0 `fill_all_missing = 0.0`, 1.1 fail-closed, clip to training range;
   - `_CalibratedModel`: only a probability clip to [0.001, 0.999];
   - `src/predict` and `src/standard_serving` scoring calls;
   - every serving-time enrichment that writes into the feature frame.
5. **Adversarial pipeline trace tests** (`tests/test_real_data_pipeline_trace.py`, 45 tests): see §4.
6. **Fantasy:** a read-only audit of the production path (`fantasy_refresh.yml`: V1 `fantasy.py`, V2
   `fantasy_v2.py` / `fantasy_panel.py` / `fantasy_residual.py`), with coverage measured on
   origin/main's outputs.

## 2. Findings and fixes - team

| # | Finding | Class | Reached a served model? | Fix |
|---|---|---|---|---|
| T1 | `build_upcoming_features` **overwrote** the six season-venue model features (`home_season_goals_h`, `home_season_conceded_h`, `home_cs_rate_h`, `away_season_goals_a`, `away_season_conceded_a`, `away_cs_rate_a`) with API-Football `/teams/statistics` averages, on every served payload. The models learned them from the history banks (`_add_season_venue_stats`). The offline parity script blocks the network, so it never saw it. | TRAIN_SERVE_MISMATCH | **Yes**, since 2026-08-16 (`APIFOOTBALL_KEY` in predict.yml) | Provider figures go to display columns `af_*` (`SEASON_VENUE_AF_COLUMNS`); the model features stay training's derivation |
| T2 | Serving `home_advantage` / `away_home_adv_factor` compared a club's last five home goals with **today's** full-history league away-goal average. Training compares each match with the average on dates **strictly before that match**. 5 of 25 drifting synthetic leagues disagreed. | TRAIN_SERVE_MISMATCH | **Yes**, every payload | `home_advantage_flags()` builds training's own flags; serving reads the last n |
| T3 | A fixture listed twice on the board was its own "previous match": the second copy got **0 rest days**. | FABRICATED_VALUE (row-level) | Only on a duplicated board fixture | Rest days are computed once per distinct fixture and mapped to every copy |
| T4 | The legacy `_prep` path fills from the **scoring batch's medians** (an all-missing column becomes 0.0). `_predict_flat` refused it only for repaired recipes, so any payload without `recipe` / `imputer` would be scored that way. | TRAIN_SERVE_MISMATCH | No (every served payload has an imputer) | `predict_proba(serving=True)` refuses a payload without a persisted imputer; `src/predict.py` scores every market through `_serve_proba` (fail closed: NaN, ERROR log, `SERVING_REFUSALS`) |
| T5 | The **rollback** payload `model_v9_standard_repaired.pkl` (1.0 imputer) filled `home/away_xg_last5` and `home/away_insidebox_last5` (0% training coverage) with **0.0 on every training row**; live boards carry real xG on 98.6% of Standard fixtures. | FABRICATED_VALUE | No (standby); it would on a rollback | `serving_refusal` refuses a 1.0 manifest with no-training-values features; then the owner **RETIRED** it (2026-11-15, "fix what you need"): `v9_repaired` is no longer a serving recipe, `owner_install` refuses the type, the registry says RETIRED, the file is kept for audit |
| T6 | A fixture missing its core form (home/away scored/conceded last 5) was still **scored**, with training medians in place of its football. Only the betting layer's no-form guard stopped the tip. | VALID_MODEL_ESTIMATE presented as a probability | Yes: 1 Standard fixture on main's 2026-11-15 board | Serving gives such a fixture **no probability** (`p = NaN`): no edge, no tip, no number on the board |
| T7 | `src/standard_serving.score_fixtures` raised on any non-finite probability, so T6's honest NaN would have failed the whole Standard board. | - | - | Non-finite is allowed only on the `serving_no_form` rows |

No time leakage was found. `test_training_features_are_point_in_time` rewrites every result from
match 60 on and proves rows 0-60 unchanged for all traced features. The 2026-11-09 same-day league
average fix (2d1d900bf) is already inside the trained code (a29fdb1ef).

**Coverage notes (eligible, explicit, but worth the owner's attention):**
- Standard xG: 17% training coverage vs 98% serving coverage, VALID_MODEL_ESTIMATE; a candidate
  ablation is proposed in `CLEAN_MODEL_BACKTEST.md`.
- HT form: 53% training coverage on Standard.
- New Format corners and fouls: **77% serving vs 95% training coverage** on main's board, a collection
  gap in Austria, Sweden, Norway, Japan, Mexico and Argentina. Those fixtures are scored with the
  explicit train-median fill + `__missing = 1`. Data truth 8a reports it as a serving coverage gap
  (AMBER), never as GREEN.

## 3. Findings and fixes - fantasy

All file:line references are the branch before the fix.

| # | Finding | Class | Fix |
|---|---|---|---|
| F1 | `fantasy.py:493-494, 540-542`: with 0 FPL matches (crosswalk missing) the V1 board kept **every** player with `injured=False` / `doubtful=False`, so departed or unavailable players were published as fit. | FABRICATED_VALUE | 0 matches: no board, ERROR log |
| F2 | `fantasy.py:228`: `mean().sum(axis=1)` read a missing defensive-action component as 0. | FABRICATED_VALUE (latent) | `sum(min_count=len(cols))`; incomplete means no figure |
| F4 | `fantasy.py:719-721`: an unknown FDR gave x1.0, published as "fixture-adjusted". | FABRICATED_VALUE | Unknown FDR: fixture-adjusted points NaN |
| F6 | `fantasy_panel.py:138`: `raw.fillna(mu)`, so a team **with games** but unknown xG got the league mean weighted by its games. | FABRICATED_VALUE (latent) | That team stays NaN; only a team with no game yet is the prior alone |
| F7 | `fantasy_v2.py:321`: a club with no fixture rows (cache missing) scored 0 for GW2-5, labelled BLANK. | FABRICATED_VALUE | No fixture rows: 3/5-GW points NaN, no BLANK label |
| F8 | `fantasy_v2.py:222`: the served residual fit used FINAL + PROVISIONAL gameweeks; the walk-forward validates on FINAL only. | TRAIN_SERVE_MISMATCH | FINAL only |
| F11 | `fantasy_v2.py:333`: a negative FPL `ep_next` was extrapolated into GW2-5 (down to -6.15). | FABRICATED_VALUE | Negative `ep_next`: later-gameweek points NaN |

**Not changed, documented as conventions** (unfitted coefficients, labelled "convention" on the
page; each is a fit candidate on the settled FPL ledger):

V1:
- appearance points "2 if minutes_pg >= 60" (F3);
- P(start) 0.6/0.4 blend of starts per appearance (F5; it overstates P(start) for rotation players
  because non-appearances are invisible in the source);
- bonus x0.66 and SoT2 x0.45 (F9), calibrated on legacy props payloads;
- FDR step 0.10, home edge 0.05.

V2:
- `start_rate + 0.25` long-run play rate (F10);
- position shares, clips and half-life.

The V1 board also depends on three props payloads, whose provenance is the props agent's (PENDING).

**Production gap (not fixable on this branch):** `fantasy_refresh.yml` runs **main**, whose fantasy
code is about 1,680 lines older and still carries:
- `minutes_pg.fillna(60)`;
- `p_start.fillna(0.6)`;
- goal points `fillna(5)`;
- FDR `or 3`;
- a pure-FDR clean-sheet probability (5 distinct values on main's board, 4 of them the FDR formula).

These go away only when data-truth-100 reaches main.

## 4. Pipeline trace tests (`tests/test_real_data_pipeline_trace.py`)

**Path covered:** raw provider rows -> history -> `build_features` / `build_upcoming_features` ->
train-only imputer -> payload -> `predict_proba(serving=True)` -> `src/predict._serve_proba`.

**Adversarial cases (each must be blocked, or reach inference as an explicit NaN + indicator):**

| Case | Test |
|---|---|
| Provider missing values | `test_provider_missing_values_stay_missing` |
| Stale or leaked results on a fixture row | `test_outcome_columns_on_a_fixture_are_ignored` |
| Wrong league join (same-named club elsewhere) | `test_same_named_club_in_another_league_never_joins` |
| Aliases | `test_alias_through_the_identity_map`, `test_unknown_or_ambiguous_board_name_gets_no_form` |
| Wrong opponent | `test_wrong_opponent_swap_swaps_the_sides` |
| Invalid timestamps | `test_invalid_fixture_timestamp_gives_missing_season_stats` |
| Impossible ranges | `test_impossible_values_are_clipped_to_the_training_range` |
| Duplicates (found T3) | `test_duplicate_fixture_rows_score_identically` |
| Unsupported competitions | `test_unsupported_competition_gets_no_probability` |
| Train/serve mismatch: parity on 12 drifting leagues | `test_serving_equals_training_for_the_next_match` |
| Train/serve mismatch: API-Football override | `test_api_football_team_statistics_never_overwrite_model_features` |
| Train/serve mismatch: refusals | batch-median, zero-filled and absent-from-manifest refusal tests |
| Core-form gap gets no probability | `test_missing_core_form_gets_no_probability_and_others_are_unchanged` |
| Batch invariance | `test_serving_scores_are_batch_invariant` |

**Production artifacts, through the real loader** (`test_active_production_payloads_pass_the_serving_guard`,
`test_repaired_rollback_payload_is_refused`) and the registry and health
(`test_registry_covers_every_active_payload_feature`, `test_health_reports_three_separate_lines_...`,
`test_a_new_fabricated_feature_turns_8b_red`).

Fantasy regression tests: `tests/test_real_data_fantasy.py` (F1, F2, F4, F6, F7, F8, F11).

## 5. Health gate (`src/integrity/data_truth.py` + system health)

Data truth 8 is three lines, never one combined green:

| Line | Meaning | Today (this branch) | Gated |
|---|---|---|---|
| 8a active production violations | fabricated, unsupported, mismatched or refused inputs in a payload that scores live fixtures; serving coverage gaps; props PENDING | AMBER: 0 violations, props PENDING (plus the NF corners gap on main's board) | no (system health, like rule 3) |
| 8b new provenance violations | a violation, or an active feature, that `config/feature_provenance.json` does not hold | GREEN: 0 | **yes** (push/PR stoplight) |
| 8c registered historical exceptions | documented; not reaching a served model | AMBER: 7 (3 documented + the retired payload's 4 fabricated features) | no |

Per production model (active only): real-data verified X, eligible learned/imputed Y, unsupported
active Z, fabricated active N, PIT violations N, train/serve mismatches N.

| Model [track] | Verified | Estimate | Unsupported | Fabricated | PIT | Mismatch |
|---|---|---|---|---|---|---|
| Standard O/U 2.5 [standard] | 84 | 2 | 0 | 0 | 0 | 0 |
| New Format O/U 2.5 [new_format] | 56 | 0 | 0 | 0 | 0 | 0 |
| BTTS / Over 1.5 / Over 3.5 [standard] | 84 | 2 | 0 | 0 | 0 | 0 |
| BTTS / Over 1.5 / Over 3.5 [new_format] | 56 | 0 | 0 | 0 | 0 | 0 |
| HT Over 0.5 / 1.5 [standard] | 84 | 2 | 0 | 0 | 0 | 0 |
| Fantasy [fpl] | 4 | 5 | 1 (props dependency, PENDING) | 0 | 0 | 0 |

"Verified" counts the `__missing` indicators, which are derived from the base column at scoring time.
The props row is filled from the props agent's output once it exists (`output/props_real_data_provenance.json`);
until then it reads PENDING and 8a cannot be GREEN.

## 6. Limitations

- The live size of T1 cannot be measured offline. The board now carries the provider's figures
  (`af_*`) beside the model inputs, so the next predict run shows the difference per fixture.
- T2's real-data effect was not re-measured on the full history: running the loader in a worktree
  can rewrite committed banks, which the owner's stage-explicit-paths rule warns about.
- Serving coverage in the committed registry is measured on the tree's committed board
  (2026-11-02 .. 11-09). The registry report also shows origin/main's board.
- The fantasy fixes land on data-truth-100. Production fantasy keeps main's older code until the
  merge.
