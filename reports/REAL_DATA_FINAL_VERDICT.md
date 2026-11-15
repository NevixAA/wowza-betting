# REAL DATA ONLY - final verdict (2026-11-15)

Branch `fix/real-data-provenance` (base data-truth-100 5573b01d0). No promotion, no retrain of a
served model, no change to bets, outcomes, bankroll, exposure or staking. Details are in
`REAL_DATA_FINAL_AUDIT.md`, `FEATURE_PROVENANCE_REGISTRY.md`, `PRODUCTION_ARTIFACT_VALIDATION.md` and
`CLEAN_MODEL_BACKTEST.md`.

| Component | Unsupported active features | Fabricated observations | Train/serve parity | Status |
|---|---|---|---|---|
| Player props | pending (props agent, `fix/props-data-audit`) | pending | pending | **PENDING**: no verdict delivered yet; Data truth 8a stays AMBER until it is |
| Standard O/U 2.5 | 0 of 86 | 0 (fixed at serving: API-Football venue override, point-in-time home advantage, duplicate-fixture rest days). The rollback payload has 4 fabricated features and is BLOCKED. | parity on 12/12 drifting synthetic leagues for every traced feature; artifact probe SAFE | **GREEN** (active); rollback **BLOCKED**, owner decision |
| New Format O/U 2.5 | 0 of 56 | 0 (same serving fixes) | parity 12/12; probe SAFE | **GREEN**. Serving coverage gap: corners and fouls 77% vs 95% in training (6 leagues), explicit fill + indicator, AMBER |
| BTTS | 0 of 86 (std) + 56 (NF) | 0 | parity 12/12; probe SAFE on both tracks | **GREEN** (NF corners/fouls gap as above) |
| Over 1.5 | 0 of 86 + 56 | 0 | parity 12/12; probe SAFE | **GREEN** (same gap) |
| Over 3.5 | 0 of 86 + 56 | 0 | parity 12/12; probe SAFE | **GREEN** (same gap) |
| Half-time models (HT 0.5 / 1.5) | 0 of 86 each | 0 | parity 12/12; probe SAFE | **GREEN** |
| Fantasy | 0 fantasy-owned. The V1 board reads 3 props payloads (pending props agent). | 0 after F1, F2, F4, F6, F7, F11 (this branch); main still runs the older code with typed fills until data-truth-100 merges | F8 fixed: served fit = validated fit (FINAL gameweeks) | **AMBER**: PENDING_PROPS, plus production on main's older code |

"Unsupported" counts MISSING_SOURCE and UNVERIFIED features in an active payload. Fabricated counts
FABRICATED_VALUE features. Two eligible VALID_MODEL_ESTIMATE features per Standard-track payload
(home/away xG: 17% training coverage, 37% for HT) are explicit, learned estimates, not fabrications.
An xG ablation is proposed in `CLEAN_MODEL_BACKTEST.md`.

**System status: not GREEN.**
- Team: unsupported active = 0, fabricated active = 0, critical mismatches = 0.
- Props: PENDING, so Data truth 8a reads AMBER and the overall system cannot be called green.
- 8b (new violations, gated): GREEN.
- 8c: 7 registered exceptions.

## Fixes

| Fix | Root cause | File | Commit | Tests | Data source | Before -> after | Limitations |
|---|---|---|---|---|---|---|---|
| T1 season-venue stats | Serving overwrote six model features with API-Football `/teams/statistics`; the models learned them from the history banks | `src/feature_engineering.py` (`SEASON_VENUE_AF_COLUMNS`) | 0615abd20 | `test_api_football_team_statistics_never_overwrite_model_features` | football-data / af_history banks (training's derivation); the provider's figures are kept as `af_*` | the provider's numbers fed the model live -> the history derivation, as trained | live size not measurable offline; the `af_*` columns measure it from the next run |
| T2 home advantage | Serving used today's league away-goal average; training uses the point-in-time average | `src/feature_engineering.py` (`home_advantage_flags`) | 0615abd20 | `test_serving_equals_training_for_the_next_match` (12 seeds), `test_home_advantage_uses_point_in_time_league_average` | history banks | 5/25 synthetic leagues mismatched (0.2-0.4) -> 0/25 | real-history size not re-measured (loader run avoided) |
| T3 duplicate-fixture rest days | `_add_rest_days` saw the first copy of a duplicated fixture as a played match | `src/feature_engineering.py` | 0615abd20 | `test_duplicate_fixture_rows_score_identically` | history banks | second copy 0 days -> same real value as the first copy | - |
| T4 batch-median path | `_prep` without an imputer fills from the scoring batch (all-missing -> 0.0); only repaired recipes were refused | `src/model.py` (`serving_refusal`, `predict_proba(serving=True)`), `src/predict.py` (`_serve_proba`) | 39f72c74f | `test_serving_refuses_a_payload_without_imputer`, `test_predict_serve_helper_fails_closed` | - | possible -> refused (fail closed, ERROR, `SERVING_REFUSALS`) | research and backtest scoring unchanged (`serving=False`) |
| T5 rollback payload | 1.0 imputer filled xG and inside-box (0% training coverage) with 0.0 | `src/model.py`, `src/standard_serving.py` | 39f72c74f | `test_repaired_rollback_payload_is_refused`, `test_serving_refuses_a_zero_filled_feature_with_no_training_values` | - | would serve fabricated inputs on rollback -> refused; rollback fails closed | rollback unavailable until retired or retrained (owner) |
| T6/T7 no-form fixtures | A fixture without core form was scored on training medians | `src/model.py`, `src/standard_serving.py` | 39f72c74f | `test_missing_core_form_gets_no_probability_and_others_are_unchanged`, `tests/test_standard_ou25_switch.py` | - | a probability (median inputs) -> no probability | such a fixture shows no model number on the board |
| F1 | 0 FPL matches published every player as fit | `player_model/fantasy.py` | dcace028d | `test_f1_zero_fpl_matches_publish_nothing` | FPL bootstrap via crosswalk | all kept with injured=False -> no board | the stale-CSV fallback (old board stays) is pre-existing |
| F2 | missing action component read as 0 | `player_model/fantasy.py` | dcace028d | `test_f2_...` | API-Football player stats | 5.0 (one component 0) -> no figure | source has no `blocks` column (3 actions, documented) |
| F4 | unknown FDR -> x1.0 | `player_model/fantasy.py` | dcace028d | `test_f4_...` | FPL fixtures | "adjusted" unadjusted value -> NaN | - |
| F6 | `raw.fillna(league mean)` for a team with games | `player_model/fantasy_panel.py` | dcace028d | `test_f6_...` | FPL snapshot | league mean x games -> NaN | - |
| F7 / F11 | no fixture rows -> 0 and BLANK; negative ep_next extrapolated | `player_model/fantasy_v2.py` | dcace028d | `test_f7_f11_...` | FPL fixtures / ep_next | 0 / down to -6.15 -> NaN | - |
| F8 | served fit used PROVISIONAL gameweeks; the walk-forward validates on FINAL only | `player_model/fantasy_v2.py` | dcace028d | `test_f8_...` | FPL snapshots | FINAL + PROVISIONAL -> FINAL | - |
| Registry, gate, validation | No machine-readable provenance, one combined status, no production-path probe | `src/integrity/real_data_provenance.py`, `src/integrity/artifact_validation.py`, `src/integrity/data_truth.py`, `config/feature_provenance.json` | b41a8c009 | `tests/test_real_data_pipeline_trace.py` (45) | serialized payloads, imputer records, board | - -> 8a / 8b / 8c, per-model counts, SAFE / BLOCKED probe | serving coverage measured on the board in the tree |

## Unresolved

| Issue | Blocker | Affected track |
|---|---|---|
| Props provenance not reported | the props agent's outputs (`fix/props-data-audit`) have not landed; 8a reads `output/props_real_data_provenance.json` when it exists | Player props, Fantasy V1 (p_goal / p_assist / p_sot2) |
| Rollback payload `model_v9_standard_repaired.pkl` fabricated xG / inside-box | owner decision: retire it, or retrain it clean (no promotion without the owner) | Standard O/U 2.5 rollback |
| Production fantasy runs main's older code (`fillna(60)`, `fillna(0.6)`, `fillna(5)`, FDR `or 3`, pure-FDR clean sheets; `fpl_id` missing from the projection log, so `wowza_uncond` is missing on 82% of rows) | data-truth-100 is not merged to main | Fantasy |
| `output/fpl_af_crosswalk.csv` is on data-truth-100 only, and no workflow rebuilds it | owner or ops: add a rebuild step | Fantasy V1 |
| New Format corners and fouls served on 77% of fixtures vs 95% in training (Austria, Sweden, Norway, Japan, Mexico, Argentina) | collection gap (API-Football statistics for those leagues) | NF O/U 2.5, BTTS / O1.5 / O3.5 NF track |
| Standard xG learned from 17% of training rows (37% HT) | candidate ablation in the Phase 2 harness; deferred to avoid colliding with `feat/team-info-features` edits to the harness and config | Standard O/U 2.5, side markets (std), HT |
| Fantasy unfitted conventions (F3, F5, F9, F10, V1/V2 coefficients) | need a fit on the settled FPL ledger (enough gameweeks) | Fantasy |
| Data truth 3 RED on the committed board: `home/away_has_ht_history` not written to `output/predictions.csv` | pre-existing; the indicator is computed at scoring but not exported to the board (operational rule, not gated) | all team tracks (reporting only) |
