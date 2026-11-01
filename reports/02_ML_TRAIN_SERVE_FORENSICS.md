# 02 — ML train/serve forensics

Audit team AU2, 2026-11-01. Read-only with respect to production. Nothing here changed a model, ledger,
output or workflow.

| item | value |
|---|---|
| v9 commit audited | `fb0e9697ec8556d4dd3df57f31f8ed5415a5012d` (origin/main) |
| Pro commit (shadow data only) | `e0ccc27b98f57c60ee28eedba81812e9b5c5c25b` |
| Environment | Windows, Python 3.11.17, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6. The repaired payload was trained with sklearn 1.9.1 and pandas 3.1.0. `requirements.txt` pins sklearn `>=1.9,<1.10`, while pandas and numpy are unpinned. |
| History used | v9's own `load_all_matches()` with the football-data download disabled, so it uses committed caches only: 85,467 rows, 2019-02-15 .. 2026-10-30 (`au2_common.load_history_offline`). |
| Scripts | `scripts/audit/au2_payload_inventory.py`, `au2_canary.py`, `au2_props_canary.py`, `au2_live_board.py`, `au2_restdays_impact.py`, `au2_history_lag.py`, `au2_schema_table.py`, `au2_ledger_immutability.py` |
| Tests | `tests/audit/test_au2_train_serve.py`: 4 controls pass and 4 strict xfails document live defects. |
| Machine outputs | `reports/au2_data/*.json|csv`. The per-feature table for every team family is `schema_table.csv`. Per-feature canary agreement is in `canary_feature_agreement.csv` and `props_canary_feature_agreement.csv`. |

## Summary

| family (serving artifact) | target OK | train builder = serve builder? (canary, identical history) | served p reproducible from persisted row | imputation | biggest live defects |
|---|---|---|---|---|---|
| **Standard OU2.5, repaired** (`model_v9_standard_repaired.pkl`, 41 f) — SERVING since 2026-10-28 | yes¹ | **YES: 41/41 features exact, p identical on 295/295 fixtures (max 0.000 pp)** | not from the row, because inputs are not persisted. A rebuild from history reproduces it within 0.03 pp. | persisted train-only imputer, batch-invariant | serving history lags results (AU2-L01) |
| Standard OU2.5, legacy (`model_v9_standard.pkl`, 42 f) — rollback; still scored and logged as `p_champion_raw` | yes¹ | **NO: 35/42 features disagree, p within 0.5 pp on 17% of fixtures, median 1.7 pp, p90 4.5 pp, max 9.3 pp** | yes | batch median | B02, B04, B07, overround shift 3.5 SD |
| New Format OU2.5 (`model_v9_newformat.pkl`, 27 f) | yes¹ | **NO: 20/27 disagree, 12% within 0.5 pp, median 2.5 pp, max 16.4 pp** | yes (84/88 within 1e-6, max 0.05 pp) | batch median | B02, B04, B07 |
| BTTS (`model_v9_btts.pkl`, 42 f, std+NF) | yes¹ | NO: 35/42 disagree, median 1.6–1.8 pp | yes (158/158) | batch median over the mixed Std+NF board | B02, B04, overround shift 4.0 SD |
| Over 1.5 (`model_v9_over15.pkl`) | yes¹ | NO: 35/42 disagree, median 1.7 pp | yes (146/158 exact, max 0.31 pp) | batch median, mixed | same |
| Over 3.5 (`model_v9_over35.pkl`) | yes¹ | NO: 35/42 disagree, median 1.4–1.9 pp | yes (156/158) | batch median, mixed | same |
| HT 0.5 / HT 1.5 (`model_ht_over05/15.pkl`, 65 f) | yes | NO: 42 shared legacy features as above. The 23 HT-specific features are close but not equal (different history input). | yes, but from **misaligned** features | batch median | **B01: HT features land on the wrong fixture** |
| Player goals / assists / SOT / cards (+ goals2, sot2–4; 138 shared f) | yes² | **NO: 122/138 features disagree.** Within 0.5 pp: goals 42%, assists 52%, sot 18%, cards 22%. Serving p is systematically lower (sot 24.1% → 21.9%). | n/a (not persisted) | batch median; absent column → 0.0 | B05, B08 |

¹ One fixture in history has no score, and its FT targets are fabricated 0.0. That is a latent defect (AU2-B03), see A.1.
² Labels match raw stats. The `cards` target counts yellow cards only, and that may differ from bookmaker settlement (A.3).

The repaired Standard OU2.5 recipe is the only model family where training and serving produce the same input row. In
every other family the legacy split between `build_features` (vectorised, training) and `build_upcoming_features`
(per-fixture loop, serving) leaves different numbers in at least 35 of 42 features. The deltas are not noise. They are
systematic: different windows, different scopes, positional joins and different price sources.

---

## A. Target verification

### A.1 Team targets (`src/data_loader.py:1412-1433`, CI path; `:1643-1658`, local xlsx path)

Checked on all 85,467 loaded rows:

* `over25 = total_goals > 2.5`, `over15 > 1.5`, `over35 > 3.5`, `btts = home>0 & away>0`. **0 mismatches** against
  recomputation on every row that has both scores. There are no inversions, the thresholds are correct, and home/away
  are not swapped (HT goals never exceed FT goals on any row; no negative or non-integer goals).
* `ht_over05 / ht_over15`: 0 mismatches on the 22,351 rows with a half-time score. They are NaN on 100% of rows without one.
  These targets were masked correctly after the 2026-09-28 fix.
* **AU2-B03.** The FT targets are **not** masked. `(NaN > 2.5).astype(float)` is `0.0`. A fixture with no score becomes
  an UNDER, an Over-1.5 NO, an Over-3.5 NO and a BTTS NO. This is the same defect class that was fixed for HT on
  2026-09-28. On the audited data it hits **1 row** (Ligue 2, 2025-12-05, Bastia v Red Star; all four labels 0.0).
  That row is eligible for the legacy `valid` frame, the repaired `track_rows` frame and the side-market frames, because
  `dropna(subset=["over25", ...])` cannot remove a 0.0. The live impact is negligible at 1 row. The code path is active
  and will fabricate a label for any future unscored fixture the loader keeps.
* **Duplicate fixtures in the legacy training frame.** 3,844 (date, league, home team) groups appear twice with two
  spellings of the away club, the same score in 100% of groups and often different `season` labels ("2023" vs
  "2023/24"). That is 7,688 rows, 9.0% of history: Standard 3,228 rows, New Format 4,460. The legacy models
  (NF, BTTS, O1.5, O3.5, HT, legacy Std) train on both copies, and the two spellings form two separate team histories.
  The repaired Standard recipe removes them before features (`identity_repair`: 9,683 duplicate rows removed at its
  training). This is already known in the estate ("14% duplicates, fix awaits owner"). I confirm it is still active
  for every legacy model.
* Settlement-side checks: `edge_under = (1 − p_over25) − 1/odds_under25` is recomputed exactly on all 158 board rows
  (max error 2e-16), so there is no O/U side inversion at the edge stage. Ledger P/L recomputed from odds and result matches the
  ledger on every settled Standard row (report 03).

### A.2 HT targets
Covered in A.1. HT models train on `valid.dropna(subset=["ht_over05", "home_ht_over05_rate"])`, so missing HT labels stay
missing.

### A.3 Player targets (`player_model/feature_engineering.py:1075-1082`)

On 876,130 `player_history` rows:

* 0 NaN in goals, assists, shots_on_target or yellow_cards.
* 0 label mismatches for goals≥1, goals≥2, sot≥1, sot≥3, cards≥1 and assists≥1.
* `target_sot4` is empty on 5,403 rows that predate the target, and `train()` drops them, so they are not imputed.
* **Population mismatch.** Training rows are appearances of at least 10 minutes (`minutes >= 10`, and `n_prev_games >= 1` in
  `train()`). Serving scores every eligible squad player before line-ups are known. The model is conditional on
  playing 10 or more minutes, and the market settles on a different condition (typically "plays at all", otherwise void).
* **Label definition.** `target_cards = yellow_cards >= 1`. 2,982 of 5,391 red-card rows have no yellow and are labelled
  "not carded". Whether that matches bookmaker settlement for "to be carded" is not verified here. That belongs to
  phase 6 (SETTLEMENT_UNVERIFIED until shown).

## B. Training vs production feature schema

### B.1 Payload inventory (`payload_inventory.json`)

| payload | sha256[:10] | recipe | features | feature-order hash | imputer | meta blender | provenance inside payload |
|---|---|---|---|---|---|---|---|
| standard_legacy | 6431387884 | legacy | 42 | bf1214ff751d28ab | none (batch median) | yes | none: keys are feature_cols, metrics, models, target |
| standard_repaired | 1f5805e63c | v9_repaired | 41 | 74e7abb665d7fe3e | **persisted** | yes | training cutoff, frame hash, alias map sha, v9 commit, library versions |
| newformat | 9eb6b4b486 | legacy | 27 | 39cdb8cb29379098 | none | yes | none |
| btts / over15 / over35 | 238e3419f4 / da4743a266 / f546ce723b | legacy | 42 | bf1214ff751d28ab | none | yes | none |
| ht_over05 / ht_over15 | 359b491a45 / 6f370f6304 | legacy | 65 (42 + 23 HT) | 47da0e665dbd7b8d | none | yes | none |
| player × 8 | — | legacy | 138 | 626c14e5d9e81485 | none | no | none: keys are feature_cols, market, metrics, models |

The legacy payloads persist no training cutoff, data fingerprint or schema hash. That identity lives only in
`models/registry` (`model_registry.stamp_for` matches the file's sha256 against the manifest). The order of the
feature columns is persisted (`feature_cols`) and `_prep` selects by name, so a column-**order** mismatch is not
possible on any path. A **missing** column behaves differently by family:

* Team legacy `_prep` silently drops a missing feature (`[c for c in feat_cols if c in df.columns]`). sklearn then
  refuses the frame (feature-count/name check), so the failure is loud. Every legacy feature is present on the live board.
* The props `_prep` fills an absent column with **0.0** and logs a warning (`player_model/model.py:30-37`). It is not loud.

### B.2 Canary: training builder vs serving builder on identical history

Method (`au2_canary.py`). Every Standard and New Format fixture **played** 2026-10-17 .. 2026-10-30 that is in history
(295 Standard, 257 NF) was built twice:

* **TRAIN.** `build_features(history)` row (legacy), or `build_features_repaired(canonical history)` row (repaired).
* **SERVE.** `build_upcoming_features(fixture, history[date < match day])`, or
  `build_upcoming_features_repaired(fixture, canonical history, cutoff = match day)`. The repaired path is exactly what
  `standard_serving.score_fixtures` runs.

Both builders get the same COVID-filtered history and the same club spelling, so name resolution is held fixed. Its
effect is measured on the live board in B.3. Odds are the history odds for both builders.

**Tolerances.** Features: EXACT ≤ 1e-9, CLOSE ≤ 1e-6·max(1,|x|). Both-NaN counts as agreement and one-sided NaN as a
mismatch. Verdict per feature: AGREE when ≥ 99.9% of fixtures are CLOSE, MOSTLY_AGREE when ≥ 95%, otherwise DISAGREE.
Probability: PASS when |p_train − p_serve| ≤ 0.5 pp.

| family | features AGREE / total | p PASS rate | median / p90 / max abs Δp | mean p train → serve |
|---|---|---|---|---|
| **STD OU2.5 repaired** | **41 / 41** | **100%** | **0.000 / 0.000 / 0.000 pp** | 0.5199 → 0.5199 |
| STD OU2.5 legacy | 7 / 42 | 16.9% | 1.71 / 4.53 / 9.29 pp | 0.5386 → 0.5346 |
| NF OU2.5 | 7 / 27 | 12.5% | 2.45 / 6.46 / 16.44 pp | 0.5111 → 0.5195 |
| BTTS (Std rows / NF rows) | 7 / 42 | 16.6% / 11.3% | 1.64 / 4.46 / 10.53 pp (Std) | 0.5534 → 0.5486 |
| O1.5 (Std / NF) | 7 / 42 | 17.6% / 15.2% | 1.66 / 4.27 / 10.95 pp (Std) | 0.7663 → 0.7613 |
| O3.5 (Std / NF) | 7 / 42 | 19.3% / 14.0% | 1.44 / 5.20 / 13.79 pp (Std) | 0.3029 → 0.2989 |
| HT-specific 23 features, key-aligned | 0 / 23 (ht_h2h_n 99%) | — | Δ small: median 0.002–0.009 per feature | — |
| HT 23 features, **positional (as served) vs key-aligned** | 0 / 23, 2.4% of rows agree | — | ht_lambda_total median Δ 0.148 | — |

The only legacy features that agree are the six season venue stats and `bookmaker_overround`. The overround agrees in
the canary only because both builders were handed the same price. On the live board it is the most-shifted feature
(B.3). Causes of the disagreement in the legacy builders, confirmed per feature in `canary_feature_agreement.csv`:

| feature group | training (`build_features`) | serving (`build_upcoming_features`) | canary agreement |
|---|---|---|---|
| `*_scored/conceded/over25_last5`, shots, SOT, corners, fouls, xG, inside-box, HT form | last 5 rows of a team-centric frame keyed by name **across leagues**, sorted (team, league, date), so a club that changed league has a later-league match inside its window | `list(last 5 home) + list(last 5 away)`, then `[-5:]`, i.e. mostly the last five **away** matches; `away_ht_*` not league-scoped | 14–39% |
| `*_attack/defense_str`, `*_ht_attack_str` | as-of expanding league average; HT strength divided by `league_avg_goals` **before** it is computed (0.65 constant) | all-time league average | 1–6% |
| `home_advantage`, `away_home_adv_factor` | per-team rolling, aligned via a league-grouped expanding mean (documented misalignment) | recomputed per fixture (NF_FEATURE_PARITY=1 since 2026-11-01) | 17–20% |
| `home/away_rest_days` | days since the team's previous row, all leagues | **AU2-B02**: positional tail of a date-sorted `tail(1000)` concat | 26–30% (0% missing in canary) |
| `p_over25_poisson_dc` | DC on training form | DC on serving form | 10% |
| `combined_sot_ratio` | per-team rolling ratio | same formula, other windows | 0% |
| season venue stats | expanding within (league, season label, team) | latest season rows, same name | 100% |

**Batch invariance.** The repaired payload is batch-invariant by construction (0 rows change when the board is split
in two). Legacy batch-median imputation is small when features are mostly present. In the canary the max change was
0.06 pp. On the live board, where rest days are 32–66% missing, re-scoring the Standard rows of the side models in a
Standard-only batch (instead of production's mixed Std+NF batch) moves BTTS by up to 0.99 pp (5 rows > 0.5 pp), O1.5 by
up to 1.49 pp (17 rows) and O3.5 by up to 1.06 pp (4 rows). So **a Standard fixture's BTTS/O1.5/O3.5 probability
depends on which New Format fixtures share the board** (AU2-B04).

### B.3 Live board (2026-11-01T10:59:39Z, 158 rows: 70 Standard, 88 NF)

**Reproducibility of served probabilities from the persisted feature row** (`live_board_forensics.json`):

* BTTS: 158/158 rows reproduce within 1e-6.
* HT 0.5 / 1.5: 70/70.
* O3.5: 156/158.
* O1.5: 146/158, max 0.31 pp.
* NF OU2.5: 84/88, max 0.05 pp.

The small residuals are consistent with float round-trip through CSV, which affects the batch medians. The persisted
row is what those models scored. The **repaired Standard probability cannot be reproduced from `predictions.csv`**,
because its 41 inputs are built inside `standard_serving` and never written. The `home_rest_days` etc. columns on
Standard rows are the *legacy* frame, which the serving Standard model does not read (AU2-P01). Rebuilding from the
committed history reproduces it within 0.03 pp.

**Rest days: verification of the reported "~60% missing live vs 0% in training".**

| | Standard rows | NF rows |
|---|---|---|
| persisted `home_rest_days` missing (legacy frame) | **65.7%** (away 54.3%) | 31.8% (away 43.2%) |
| re-running the production code reproduces the missingness pattern | 100% | 100% |
| same quantity computed by fixture key, raw OddsAPI name | 52.9% | 50.0% |
| by fixture key, league-scoped name resolution | **1.4%** | 4.5% |
| **repaired builder (what Standard OU2.5 actually serves on)** | **1.4%** | n/a |

* Finding **confirmed** for the legacy frame. In training it is 0%: the scaler mean is learned on complete rest days.
  The legacy rest-days code has two independent defects:
  * No name resolution: an OddsAPI spelling never matches history.
  * A positional `tail(len(df))` of a date-sorted concat. On this board 157 of 158 rows take the rest days of a
    different fixture, because the board is not date-sorted.
* Finding **does not apply** to the Standard O/U 2.5 model that serves since 2026-10-28. Its repaired builder resolves
  names league-scoped and computes rest days per fixture (1.4% missing = 1 unresolved club).
* It **does apply** to every legacy-frame model still serving: NF OU2.5, BTTS, O1.5, O3.5 and HT. It also applies to the
  legacy Standard probability logged as `p_champion_raw`.
* Probability effect of fixing **only** rest days on this board (`restdays_impact.json`): median |Δp| is 0.17–0.38 pp
  and p90 is 0.9–1.9 pp. The maximum is 12–16 pp, on rows whose resolved last match is months old (off-season leagues).
  For NF OU2.5, 6 of 88 rows cross the 3% VALUABLE edge floor and 2 change best side. This is a counterfactual on one
  board, not an ROI estimate.

**Overround feature.** `bookmaker_overround` is a live input of legacy Std, BTTS, O1.5, O3.5 and HT. Its live median is 0.085 (one
book). The training scaler mean is 0.060–0.062 with SD 0.008. That is a **3.5–4.1 training-SD shift on every Standard
row** (`schema_table.csv`). Training read football-data closing averages, while live reads one bookmaker's current price
(documented as P0 bug 5 in `features_repaired.py`). The repaired Standard model dropped the feature. NF rows use the
best-of-books overround since today's NF_FEATURE_PARITY.

**HT features land on the wrong fixture (AU2-B01, new).** `src/ht_features.attach_ht_features` returns the upcoming
rows sorted by (date, league, home, away). `src/predict.py:501` assigns them with `feat[c] = _ht[c].to_numpy()`,
i.e. by position, onto a board ordered by league, then date. On the audited board:

* The persisted `ht_*` columns equal the **positional** output on 100% of rows.
* They equal the fixture's own key-aligned features on only 6–35% of rows (`ht_lambda_total` 6.3%, `ht_home_n` 34.8%).
* Example: Derby County at home shows `ht_home_n` 9.26 on 11-03 and 0.00 on 11-06.
* Effect on Standard rows: HT 0.5 median 0.61 pp, max 1.72 pp, 19/70 rows > 1 pp. HT 1.5 median 1.05 pp, max 3.59 pp,
  36/70 rows > 1 pp.

It has been active since commit `e8b9f8b2d` (2026-09-29). The HT ledger has had **0 rows since 2026-09-29**, so no
HT tip or P/L was affected. The scope is `predictions.csv` `p_ht_*`, the dashboard, and the HT rows of
`shadow_h2h.csv` (254 fixtures logged).

**Serving history lag (AU2-L01).** On the audited board, 48 of 70 Standard fixtures (39 with both teams) have a team
whose most recent league match had already kicked off but was absent from the history that scored it. That includes
24/24 Championship fixtures, whose history ends 2026-10-24, and 58 played fixtures are missing in all. Training never
has this gap. This is a data-publication delay, not code. It affects every family including repaired (median
repaired `home_rest_days` 11 days on this board), and nothing flags it.

### B.4 Player props canary (`au2_props_canary.py`)

The sample is 400 random club player-matches, 2026-10-10 .. 2026-10-30. 393 were compared and 7 had no prior history.

* The TRAIN row is the stored `player_history` row, which is exactly what `mode_train` reads.
* The SERVE row is `build_upcoming_features([player's latest prior history row with the fixture's is_home/opponent],
  history[date < match], {}, {})`. That is the production call. `match_contexts` only ever carries World Cup h2h odds
  (`_build_match_contexts` reads `worldcup_history.json`), and referee profiles need an API key.

| market | p PASS (≤0.5 pp) | median / p90 abs Δp | mean p train → serve |
|---|---|---|---|
| goals | 42% | 0.73 / 6.82 pp | 0.0912 → 0.0842 |
| assists | 52% | 0.45 / 2.89 pp | 0.0598 → 0.0531 |
| sot | 18% | 2.41 / 14.56 pp | 0.2408 → 0.2190 |
| sot2 / sot3 / sot4 | 60% / 91% / 100% | 0.28 / 0.02 / 0.00 pp | 0.063→0.045 / 0.014→0.010 / equal |
| cards | 22% | 1.32 / 5.14 pp | 0.1336 → 0.1187 |
| goals2 | 97% | 0.01 pp | 0.0072 → 0.0067 |

122 of 138 features DISAGREE. The serving probabilities are lower than training for every market with signal. The
largest gaps are scale or definition mismatches, not noise:

* `opp_mid_discipline`: train median 4.76 vs serve 0.82.
* `opp_sot_conceded_vs_mid_pg`: 2.00 vs 0.33.
* `card_clash_index`: 0.87 vs 0.14.
* `age_peak_delta`: 0 vs 2.0.
* `referee_strictness`: 0.16 vs 0.00. This is a constant default at serve time.

Also:

* All 5 opponent-quality features agree on 0% of rows.
* **AU2-B05**: the props serving `rest_days` is copied from the player's last history row, i.e. the rest **before
  his previous match**. Training uses days since the previous match.

Props are paper-only (invariant 2), so this changes no money. It does mean every props evaluation that scores live
predictions is evaluating a different input distribution from the one the model was validated on.

### B.5 Scaler reuse, imputer refits, version, leakage

* **Scalers.** Every team and props payload persists its fitted StandardScaler inside the calibrated
  pipeline. Serving reuses it and never refits. The training scaler mean/SD for each feature is in `schema_table.csv`.
* **Imputers.** Legacy team `train()` imputes the train and test splits **separately**, each with its own medians
  (`src/model.py:185-186`), so no holdout statistic reaches the fit. Serving then re-imputes from the **scoring
  batch** (AU2-B04), which is a refit at every run. Legacy props `train()` calls `_prep(df)` on the whole frame
  **before** the chronological split (`player_model/model.py`), so holdout medians inform the imputation of the fit
  rows. That is a mild leakage into the props holdout metrics. The repaired Standard and props-R2 challenger recipes
  fit on the base block only.
* **Versions.** The repaired payload was trained under sklearn 1.9.1 and pandas 3.1.0. Serving CI installs unpinned pandas,
  but the 0.03 pp rebuild agreement (B.3) shows no version effect today.
* **Leakage in features.** No feature uses information from the fixture itself: the canary rows are as-of. The only
  price feature (legacy overround) is a closing average in training but an entry price at serving. That is a
  train/serve shift, not leakage into serving.

### B.6 Existing test suite on the audited commit

`python -m pytest tests -q` gives 1509 passed, 27 skipped, 4 xfailed (the AU2 strict xfails) and **4 failed**. All 4
failures are on the brief's known pre-existing list: test_dashboard_staked_basis, test_fantasy_v2 archive,
test_promotion_gate passing_all_criteria and test_shadow_settlement health artifact. Output is in `au2_data/existing_tests.txt`.

## C. Provenance per prediction

| field | predictions.csv | bets_ledger.csv | shadow_h2h / experiments | assessment |
|---|---|---|---|---|
| model version (artifact sha) | `model_version` + 5 side/HT version columns | `model_version` (since 2026-10-18; NaN before) | `champion_version` | OK, with one exception: AU2-B06 mislabels post-switch rows |
| training cutoff | `training_cutoff` | `training_cutoff` | `challenger_training_cutoff` | OK |
| serving recipe | `serving_recipe` (Std rows) | **absent** (inferable from model_version) | model_type `standard_repaired` | OK via version |
| code commit | `git_sha` | absent | `code_commit` | ledger gap |
| scoring timestamp | `generated_at` | `generated_at` (minute precision) | `prediction_ts` / `logged_at` | OK |
| data fingerprint (history as of scoring) | **absent** | absent | `dataset_id` (experiments only) | **gap**: AU2-L01 makes it material, because no row says which matches its features saw |
| feature schema hash | absent | absent | `feature_schema` (experiments only) | gap |
| calibration version | absent (inside artifact) | absent | — | covered by the artifact sha |
| model inputs | legacy frame persisted; **repaired inputs not persisted** | — | — | AU2-P01 |
| `model_sha` | size+mtime digest | — | — | still present and still meaningless in CI (mtime reset on checkout). CLAUDE.md already documents this. `model_version` supersedes it. |

**History is not rewritten now, but was before 2026-10-26** (`ledger_immutability.json`, all 89 committed versions of
`bets_ledger.csv` since 2026-10-18):

* 0 rows disappeared.
* 0 settlement fields were overwritten once set.
* 44 live rows had `edge_pct` and `signal_tier` rewritten in place. All rewrites happened before the owner fix
  `dfcdf289d` ("first logged tier stands", 2026-10-26 12:00 UTC), and none since.
* Rows generated before that time may carry an upgraded tier or later edge, not their first-sight value. Phase 6
  owns the full reconstruction.

## Bugs and limitations (Phase-11 fields)

| ID | Sev | Repo | File / line | Root cause | Affected dates | Affected rows | Production impact | Probability impact | Selection impact | ROI impact | Status | Class | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU2-B01 | P2 | v9 | `src/predict.py:498-501` with `src/ht_features.py:200-242` | `attach_ht_features` returns date/league-sorted rows, assigned positionally to an unsorted board | 2026-09-29 → now | every Standard row's `p_ht_over05/15` on every board; 65–94% of rows get another fixture's HT features. | HT probabilities, dashboard, `shadow_h2h` HT rows (254 fixtures). No HT tip since 09-29 (HT ledger 0 rows). | median 0.61 / 1.05 pp, max 1.7 / 3.6 pp | none on money; HT tips not produced | NOT IDENTIFIABLE (no HT bets) | ACTIVE | true production bug (paper market) | merge `_ht` back on (date, league, home, away), or have `attach_ht_features` preserve input order | `tests/audit/test_au2_train_serve.py::test_ht_features_follow_board_order` (strict xfail) |
| AU2-B02 | P1 | v9 | `src/feature_engineering.py` `build_upcoming_features`, lines 785-793 (`tail(1000)` concat + `.tail(len(df))`) | positional tail of a date-sorted concat, no name resolution | since initial commit (2026-05-21) | all legacy-frame rows. Live board: 157/158 rows positional mismatch; Std 66% / NF 32% missing vs 0% in training | NF OU2.5 (serving), BTTS/O1.5/O3.5 (serving, side tips), HT; legacy Std until 2026-10-28 | median 0.2–0.4 pp, p90 0.9–1.9 pp, max 12–16 pp | NF OU2.5 board: 6/88 rows cross the 3% edge floor, 2 flip side | NOT IDENTIFIABLE without a regime-split replay | ACTIVE (not for repaired Std OU2.5) | true production bug | compute rest days per fixture by key with league-scoped `resolve` (as `features_repaired` does) | `test_legacy_serving_rest_days_belong_to_their_fixture` (strict xfail) |
| AU2-B03 | P3 | v9 | `src/data_loader.py:1413-1416`, `:1644-1649` | FT targets not masked: `(NaN>2.5).astype(float)=0` | latent, any unscored fixture | 1 row now (Ligue 2 2025-12-05 Bastia v Red Star) | training labels of every model incl. repaired | negligible at n=1 | none measurable | none measurable | ACTIVE (latent) | true bug, data-coverage-limited | `.where(total_goals.notna())` like the HT targets | `test_fulltime_targets_stay_missing_without_goals` (strict xfail) |
| AU2-B04 | P2 | v9 | `src/model.py::_prep` (imputer=None); `src/predict.py` side-model call on `std_mask | nf_mask` | legacy batch-median imputation; side models scored on a mixed Std+NF batch | all legacy serving | all legacy rows | NF, BTTS, O1.5, O3.5, HT | Std rows moved up to 0.99 / 1.49 / 1.06 pp by NF board composition | edge-floor crossings possible | NOT IDENTIFIABLE | ACTIVE (repaired Std fixed) | true production bug (known as "P0 bug 1"; fixed only for repaired Std) | persist a train-only imputer per payload (the repaired path exists) | none added (repo has `test_standard_forensic_repair::test_01b_documents_legacy_batch_variance`) |
| AU2-B05 | P3 | v9 | `player_model/feature_engineering.py:1509` (`"rest_days": p.get("rest_days", 6.0)`) | serving copies the last history row's rest, not days since that row | all props predictions | all props rows | props (paper-only) | part of the props canary drift | paper only | none (paper) | ACTIVE | research-relevant flaw (paper market) | compute `(fixture date − last match date)` clipped 1..30 | `test_props_serving_rest_days_is_days_since_last_match` (strict xfail) |
| AU2-B06 | P2 | v9 | `src/shadow_experiments.py:233-237` (`champion_version` from the legacy stamp, `p_served` from the served column) → `pipeline._log_h2h` | post-switch rows labelled with the legacy `champion_version` carry the repaired `p_served` | 2026-10-28 12:48 → now | 42 rows in `shadow_experiments.csv` and `shadow_h2h.csv` | lineage / PG-1.0 prospective evidence for H-MODEL-01 | none on serving | none on tips | contaminates champion-vs-challenger comparison | ACTIVE | true bug (evaluation integrity) | stamp `champion_version` from the model that produced `p_served` (`ou_model_for_row` with `serving_recipe`) or log `p_champion_raw` as the champion | none added (data check in `au2_phase3.py`) |
| AU2-B07 | P1 | v9 | `src/feature_engineering.py` build_features vs build_upcoming_features (legacy) | two implementations of the same 42 features disagree on 35 | since initial commit | every legacy-frame prediction | NF OU2.5, BTTS, O1.5, O3.5, HT serving | median 1.4–2.5 pp, p90 4.3–6.5 pp, max 9–18 pp | NOT IDENTIFIABLE here | NOT IDENTIFIABLE | ACTIVE for legacy families; FIXED for Std OU2.5 (repaired, 2026-10-28) | true production bug | extend the repaired single-builder recipe to the other families (challenger, owner approval) | canary `au2_canary.py`; control test `test_repaired_builder_train_equals_serve` passes |
| AU2-B08 | P3 | v9 | `player_model/feature_engineering.py::build_upcoming_features` vs `build_features`; `player_model/model.py:30-38` | 122/138 props features disagree train vs serve; absent column → 0.0 | all props predictions | all props rows | props (paper) | median 0.4–2.4 pp; sot p90 14.6 pp; serving p lower on every market | paper only | none (paper) | ACTIVE | research-relevant flaw (paper market) | single builder for train and serve (as R2-D started for the imputer) | canary `au2_props_canary.py` |
| AU2-P01 | P3 | v9 | `src/standard_serving.py::score_fixtures` | repaired Std inputs are never persisted; `predictions.csv` Std rows show legacy-frame features the serving model does not read | 2026-10-28 → now | 70–89 Std rows per board | provenance / dashboards / health checks read the wrong frame (`feature_parity_health.json` compares repaired training against legacy live columns and reports the rest-days FAIL for the repaired model) | none | none | none | ACTIVE | provenance gap | write the repaired feature row (or its hash) next to each Standard prediction | none |
| AU2-L01 | P2 | data | football-data publication delay; no freshness guard | serving history ends before the latest played round | every board | 48/70 Std board rows today | all families incl. repaired | stale form / inflated rest days (median 11 d) | NOT IDENTIFIABLE | NOT IDENTIFIABLE | ACTIVE | data-coverage limitation | record history max date per league on each prediction and flag stale fixtures (no selection change) | `au2_history_lag.py` |

## Open questions

1. Owner decision: should the repaired single-builder recipe be extended to NF and the side markets? That is the only
   intervention shown here to remove the train/serve split. Its forward value is unproven.
2. Bookmaker "to be carded" settlement versus `target_cards` (yellow only). This is for phase 6.
3. Whether `feature_parity_health.json` should compare the repaired training frame against repaired serving features.
   Today its Standard OU2.5 FAIL on rest days is about the legacy frame, not the model that serves.
