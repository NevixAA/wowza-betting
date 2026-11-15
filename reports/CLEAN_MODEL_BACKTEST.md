# Clean model backtest - team section (REAL DATA ONLY, 2026-11-15)

## Decision: no team feature had to be removed, so no clean candidate was trained

The owner's rule: *a feature that is FABRICATED or UNSUPPORTED is removed from the active schema,
and clean candidates are trained with the Phase 2 harness and compared chronologically against the
champion, with an ablation.*

The audit (`reports/FEATURE_PROVENANCE_REGISTRY.md`, `config/feature_provenance.json`) read the
feature lists from the **serialized payloads** and found:

| Active payload | Features | FABRICATED | UNSUPPORTED (missing source / unverified) | TIME_LEAKAGE | TRAIN_SERVE_MISMATCH after this branch |
|---|---|---|---|---|---|
| Standard O/U 2.5 | 86 | 0 | 0 | 0 | 0 |
| New Format O/U 2.5 | 56 | 0 | 0 | 0 | 0 |
| BTTS / Over 1.5 / Over 3.5 (std 86, NF 56 each) | 142 each | 0 | 0 | 0 | 0 |
| HT Over 0.5 / HT Over 1.5 | 86 each | 0 | 0 | 0 | 0 |

The three violations found were all at **serving**, and the payloads were trained on the correct
(training) derivation:

1. Season-venue stats were overwritten by API-Football `/teams/statistics`.
2. `home_advantage` / `away_home_adv_factor` used today's league average instead of the
   point-in-time average.
3. Duplicated board fixtures got 0 rest days.

All three were fixed in the serving code, so serving now produces exactly what the models learned.
Retraining would not change what the payloads learned. The owner's point that a config change is not
enough holds the other way round here: the serialized models already learned the right derivation,
and it was serving that drifted from it.

The one fabricated feature set is in the **retired** v9_repaired payload
(`models/model_v9_standard_repaired.pkl`: xG and inside-box shots filled with 0.0 on every training
row). Its clean replacement is the active Standard champion (`std_v9_repaired`, Phase 2 harness,
trained 2026-11-09 with real xG where the provider has it). The owner RETIRED it on
2026-11-15 ("fix what you need"); no retrain of it is proposed.

## Two VALID_MODEL_ESTIMATE features (eligible, but candidates for an ablation)

| Feature (Standard payloads) | Training coverage | Serving coverage (main board 2026-11-15) |
|---|---|---|
| home/away_xg_last5 (Standard O/U 2.5, BTTS, O1.5, O3.5) | 17.0-17.1% | 97.2-98.6% |
| home/away_xg_last5 (HT models, trained from 2023-07) | 37.1% | 97.2-98.6% |

83% of the Standard base-training rows carry the explicit train-median fill plus `__missing = 1`,
while live fixtures almost always carry a real value. That is an explicit, learned estimate (not an
invented observation), but the model learned its xG response from 17% of its rows. A candidate
ablation (Standard champion recipe without xG/inside-box, Phase 2 harness, chronological holdout
2026-04-01 .. 2026-11-08) is the right test.

It was **not run on this branch**, for three reasons:
- `config/phase2_retrain.json` and the harness are being changed by the team-info agent
  (`feat/team-info-features`, `src/retrain_harness/*`, `config/phase2_retrain.json`);
- a second editor on the same files would collide;
- the owner rule says no production retrain until the architecture is wired and the go-live
  rehearsal is green.

Recommended as a Phase 2 candidate arm (`drop: [home_xg_last5, away_xg_last5, home_insidebox_last5,
away_insidebox_last5]`, same holdout, same metrics: log loss, Brier, AUC, calibration, coverage, per
league and per odds band).

## Measured effect of the serving fixes

- **Home advantage:** on 25 synthetic drifting leagues, serving and training disagreed on 5 of 25 (20%)
  next matches before the fix (deviations of 0.2-0.4 on a 0-1 rate) and on 0 of 25 after
  (`tests/test_real_data_pipeline_trace.py::test_serving_equals_training_for_the_next_match`, 12
  seeds in CI).
  - The offline parity check (`scripts/team_train_serve_parity.py`) did not flag this.
  - On real data the effect is bounded by how far a league's away-goal average has moved since each
    of a club's last five home matches.
- **Season-venue stats:** the API-Football override fired only with the network on, so the offline
  parity check could never see it.
  - Its live size is not measurable offline.
  - The board now carries the provider's figures beside the model inputs (`af_home_season_goals_h`
    ... `af_away_cs_rate_a`), so the next predict run measures the difference directly.
- **Duplicate fixtures:** a fixture listed twice got 0 rest days on its second copy; now both copies
  carry the same, real value.

No serialized artifact, holdout, threshold or bankroll setting was changed. No promotion.
