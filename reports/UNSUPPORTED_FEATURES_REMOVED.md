# Unsupported props features: removed from every active path (P0 "real data only", 2026-11-15)

Branch `fix/props-data-audit` (v9). Scope: the 9 set-piece features named by the owner, plus the
features withheld by the 2026-11-15 props audit. Nothing here retrains or promotes a model.

## Verdict per set-piece feature

All 9 are **EXCLUDED_FROM_ACTIVE_MODEL**. None is SOURCE_VERIFIED: no provider endpoint we hold
records who took a set piece, who received it, or which goals came from one.

| feature | how it was built (last version, `git show 36959d8df^:player_model/feature_engineering.py`) | invented part | upstream that would be needed | in the training list | built today | in a served artifact |
|---|---|---|---|---|---|---|
| sp_taker_score | `career_sp_assist_rate/0.15*0.60 + fk_goals_pg/0.05*0.25 + (0.45·M + 0.30·F + 0.05·D)*0.15`, clipped | typed weights, a typed position prior, typed normalisers | per-assist set-piece origin (API-Football goal events say "Normal Goal" only: reports/DIAG_AF_SETPIECE_EVENTS.md) | no | no | goals2, sot3, sot4 (hidden); every R2 challenger |
| sp_receiver_score | `(1 - sp_taker_score) · aerial_won_rate · (D + 0.6·M)` | built on sp_taker_score; "aerial" is ALL duels | aerial-duel counts by origin | no | no | goals2, sot3, sot4; R2 |
| set_piece_threat_score | `aerial_won_rate · corners_rel · sp_rel · (D + 0.7·M)`. The first version (8d777be12, 2026-06-10) used a typed **0.30 set-piece rate** and corners / 6.0 | typed 0.30 rate (first version), typed position weights, clips | set-piece goals conceded by the opponent (no source) | no | no | goals2, goals3, sot3, sot4; R2 |
| set_piece_corner_matchup | `set_piece_threat_score · corners_rel` | inherits the above | as above | no | no | goals2, goals3, sot3, sot4; R2 |
| defender_sot_role_index | `D · aerial_won_rate · sp_receiver_score · corners_rel`; an unknown role read as 0 (no D flag means 0) | missing role -> 0, typed clips | lineup role (exists now), set-piece role (does not) | no | no | goals2, sot3, sot4; R2 |
| defender_sp_edge | `sp_receiver_score · (height_adv+2)/4 · sp_rel · corners_rel`, clip 0..4 | typed scale and clips | as above | no | no | none |
| sp_threat_vs_weak_sp_defense | `sp_goals_pg · sp_rel · (1 - sp_taker_score)` | sp_goals_pg is ANY header or free-kick goal (not a set piece) | set-piece goal classification | no | no | none |
| sp_taker_assist_edge | `sp_taker_score · career_sp_assist_rate/0.10 · corners_rel` | typed normalisers | set-piece assist origin | no | no | none |
| opp_sp_goals_conceded_pg | opponent's rolling "sp goals" conceded; was `fillna(0.12)` | the 0.12 default; "sp goal" = any header / free kick | set-piece goal classification | no | no | none |

Historical availability: computed for every row until 2026-11-02, almost entirely from typed
constants and NaN-turned-0 inputs. Since 2026-11-02 none of them has been built for training or for
serving (`feature_engineering.REMOVED_SETPIECE_FEATURES`, `config.SETPIECE_PROXIES_REMOVED`), so
prediction-time availability, missingness by league and train/serve parity do not apply: they are
absent. The only set-piece features with a real source are the Understat per-shot `us_sp_*`
columns (NaN outside the Understat leagues).

## Where the 9 were still referenced, and what changed

| place | before 2026-11-15 | now |
|---|---|---|
| `config.PLAYER_FEATURE_COLS` (training list) | absent since 2026-11-02 | absent; a test refuses re-entry |
| `props_feature_contract.json` | absent | absent: never IMPUTABLE, simply not in the schema |
| `feature_engineering._add_composites` | not built | not built (test: never a column, not even NaN) |
| fantasy (`fantasy.py`, `fantasy_features.py`, `fantasy_v2.py`), `predict.py` | not referenced | not referenced (test scans the source) |
| served legacy payloads goals2 / goals3 / sot3 / sot4 (`models/model_player_*.pkl`) | list 2-5 of them; serving fed NaN, then filled the cell with the median of the scoring batch | **refused**: `model.predict_with_diagnostics` returns no probability for a payload listing any `UNSUPPORTED_FEATURES` (imputer_version `REFUSED_UNSUPPORTED_FEATURES`). These markets were already hidden (`config.HIDDEN_PENDING_CLEAN_RETRAIN`) |
| R2 challengers (`models/challengers/props_r2/*.pkl`) | list 5 of them | refused on the same rule (they are research artifacts, never served) |
| promoted clean models sot / sot2 / goals / assists / cards | do not list any | unchanged |

## Other features withheld from the next retrain (2026-11-15 audit)

| set | features | why | file |
|---|---|---|---|
| `PROFILE_CONSTANT_FEATURES_WITHHELD` | age_peak_delta, height_advantage_score, height_aerial_interaction | typed peak age 27, /10 with clip ±2, 160-200 cm clip | `player_model/config.py` |
| `SHOTS_DEFINITION_WITHHELD` (17) | shots_pg, career_shots_pg, shots_home_pg, shots_away_pg, season_shots_pg, prev_season_shots_pg, sot_rate, shot_accuracy_rate, shot_accuracy_l10, prev_season_shot_accuracy, box_actions_per90, sot_quality_score, opp_adjusted_shot_threat, clinical_vs_gk, volume_shot_penetration, box_dominance_vs_gk, box_threat_vs_leaky_defense | the provider changed the meaning of shots.total in Sep-Nov 2026; replaced in training by the 17 `*_lgrel` counterparts (each match's shots relative to its league's own as-of level) | `player_model/config.py`, `feature_engineering.league_shots_level` |

Withheld features are still computed for the served payloads that list them: removing them from
config does not change a serialized model. Each served artifact's list is in
PRODUCTION_ARTIFACT_VALIDATION.md. The next clean retrain trains without them (candidate runs in
CLEAN_MODEL_BACKTEST.md).

## Evidence

* Commits: `af27fa808` (withheld sets, lgrel features), `c41165a49` (unsupported payloads refused).
* Tests: `tests/test_props_real_data_only_2026_11_15.py::test_set_piece_features_are_excluded_everywhere_active`,
  `::test_a_served_payload_with_an_unsupported_feature_gets_no_probability` (every served file),
  `::test_unsupported_features_cannot_re_enter` (adversarial payload);
  `tests/test_props_data_audit_2026_11_15.py::test_profile_constant_composites_are_withheld_from_training`,
  `::test_every_shots_feature_has_its_league_relative_counterpart_in_training`.

## Limitations

* goals2 / goals3 / sot3 / sot4 have no probability until a clean retrain promotes models for them
  (owner-gated; they were already hidden from tips and the board).
* `models/model_player_goals3.pkl` is not in `config.MODEL_FILES`. It is not on any serving path,
  and it lists 2 of the 9 features.
