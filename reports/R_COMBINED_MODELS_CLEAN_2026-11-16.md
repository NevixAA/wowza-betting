# Combined models on verified data only (COMBINED-MODELS-DEV-2)

Date: 2026-11-16. Branch `research/combined-models-clean`. **Research only.** Nothing changes in serving, nothing is promoted, no tip is made, and no API-Football or OddsAPI call was made. The owner decides any use.

* Code: `scripts/research_combined_models_clean.py`. Tests: `tests/test_research_combined_models_clean.py`.
* Pre-registration: `config/experiments/combined_models/protocol_dev2.json`.
  * Committed in `76bacb180` before any effect was computed (sha256 `9a1e2dac...`).
  * Amendment A1 was committed in `d1f23f50e`, also before any effect (sha256 `fafb970e79c783fa9ec8fae2caf2b90c3c867f31eb00225e0cbc7110c5bc4de1`). It covers the props reproduction check (section 1).
* Forward protocol: `config/experiments/combined_models/forward_protocol_fwd2.json` (COMBINED-MODELS-FWD-2).
* Result tables: `reports/combined_models_clean_data/`. They hold counts and estimates only: no price, no market probability and no player row. The OddsAPI-derived values stay in the private local work folder.

Owner's request (2026-11-16): "for every model, try to use all the models' data to find if some model gets extra boosts with the help of the other models", "but only on 100% verified data".

## In plain language

1. **No model gets a real boost from the other models on verified data.**
   * 1,679 per-league cells were registered (team, props and Pro targets × every other model × residual test and stack) and 593 had enough data to read.
   * After Holm over the whole family, **0 genuine cells survive.**
   * One cell passes the rule on paper: Pro "6+ goals" × market, Argentina. It is a statistical artefact: **no 6-goal match happened in that league in the scored months.** With no events, the "effect" is just the gap between Pro's own predictions and the bootstrap spread collapses to near zero. It is not carried forward.
2. **Stacking the models does not beat a model on its own.** The test fits each target together with a helper on earlier months, then scores it on 2026-04..10. Compared with the same target recalibrated alone, no per-league stack wins after Holm.
   * Pooled over leagues (context only), the largest gains are about 0.003-0.005 in log loss: team BTTS + market, and Pro "0-1 goals" + market.
   * For props, every stack gains about 0.0002 or less, and several stacks are worse because they overfit.
3. **The 11-11 side-λ effect was NOT caused by the two props bugs.**
   * Both bugs are fixed in the clean model. `team_goals_pg_roll` now tracks the player's own side (Spearman 0.35 with the own side's λ, −0.02 with the opponent's; on 11-11 it was −0.03 / 0.35). Data truth 7a (keeper conceded = score) is GREEN.
   * Re-run on DEV-1's own months with the clean model, the effect has almost the same size: pooled SoT +1.0, SoT 2+ +0.7, anytime scorer +0.8 pp (11-11: +1.3 / +0.7 / +0.9). Serie B anytime scorer: +2.0 pp (11-11: +1.8).
   * On the fresh months 2026-04..10 it is still there pooled (SoT +1.6 pp, CI [+0.9, +2.3]). It is **not** significant in any single league after Holm. The Serie B anytime-scorer cell drops to +1.2 pp, CI [−0.6, +2.9].
   * In log-loss terms it is tiny: the best props × Pro stack gains 0.00024 pooled for SoT. **It is a small, real calibration pattern of the props model, not a betting edge that survives per league.**
4. **What the residual tests do show is mostly the targets' own calibration, not help from other models.**
   * For team and Pro targets, outcomes run higher where Pro or the market say "more" than the target (pooled +5 to +15 pp between the extreme thirds).
   * A target recalibrated on its own earlier months absorbs most of that pattern, and the stacks then add nothing significant.
   * Pro's Dixon-Coles specials probabilities are clearly miscalibrated on these months: recalibrating them alone improves log loss by 0.003-0.011.
5. **What could not be tested:**
   * **Props as a helper** for team or Pro tips. A props row exists only for a player who appeared in the match, and props tips need the confirmed lineup. Using props at the team tip time (125-153 h before kickoff) would leak the lineup.
   * The **side models** (corners, cards, specials). The branch `research/side-models` on origin holds only a data-truth report and no `docs/` note with prediction paths, so their cells were **not run** and nothing was substituted.
   * **wowza_v11.** Its inputs are unverified, so it is excluded (section 1).

## 1. Verification (before any effect was computed)

`work/verification.json` is copied to `reports/combined_models_clean_data/verification.json`.

### Inputs used

| input | source | frame / data hash | Data Truth / provenance | train/serve parity | missing-value policy | status |
|---|---|---|---|---|---|---|
| Team Standard champion_recipe | run_20261115T190707Z (local, single-threaded), code d7ded2ef5 (feat/team-info-features); evidence README on 0a648a6b4. GitHub run 43706983618 is the earlier Standard run on frame f75ea2d98e98ad22 and is not used | frame **974e04bc77c4d438**, 71,583 loaded rows | 43 features, all registered and eligible in `config/feature_provenance.json` (data-truth-100); 0 fold errors | team_train_serve_parity: 125 fixtures, 72 raw columns, 0 mismatched cells, 0.0 pp (evidence README) | form required on both sides (blind fixtures not scored); train-only imputer with missing indicators persisted in the payload (registered "eligible learned / imputed"); a feature with no training value is dropped (xG in fold 0) | **PASS** |
| Team New-Format champion_recipe | **re-run today**, run_20261116T142331Z, code 0a648a6b4, single-threaded. The 11-15 run was on the older frame f75ea2d98e98ad22. | frame **974e04bc77c4d438** | 28 features, all eligible; 0 fold errors | as Standard | as Standard | **PASS**. The 94,188 champion_recipe predictions equal the 11-15 run's to 0.0 (the 2 rows that changed are in other tracks) |
| Props clean-schema candidate | GitHub run 43745808793, code d051ee5c7, hgb_nan_native, 125 features (sha 3f80d914...), rule all_prior_verified | frame **4f6c603d00c14567** (recomputed on the cached run frame) | branch gate: Data truth 7a, 7b, 7c, 7e GREEN; 7d, 7f, 7g AMBER (no failure) | props parity 0 features / 0 probabilities on 2,231 player-fixtures (36f1609e3) | NaN-native: no value filled anywhere; provider nulls stay missing (7c / 7e) | **PASS under amendment A1** (below) |
| Pro Dixon-Coles (SPC-1) | wowzaV9-Pro data-truth-100 7344ddb1, code unchanged; history exported by Pro's own exporter from v9 0a648a6b4 | 71,582 rows + 1 without a score = 71,583 = the team frame; parquet sha256 b60cad4b... | scores only (Data truth 5 GREEN: one match, one row) | n/a (the tracker uses the same code) | match without a score dropped and counted; no fallback λ for a club or league without history | **PASS** |
| Market (sharp / consensus) | team harness `market_bet`: power de-vigged two-sided **BET-time** price, OddsAPI tier BT, role BET, 125 h (Standard) / 153 h (NF) before kickoff | private file, local only | price provenance as in the all-data test | n/a | missing without a two-sided pair (never filled) | **PASS** (statistics only in the repo) |

**Props amendment A1.** The pre-registered check asked the 2026-09-01 refit to equal the run's holdout probabilities bit for bit. **It did not** (max |diff| 0.06-0.22, mean 0.002-0.004). Before any effect was computed, the cause was found:
* The run's own pickle, scored here on the cached frame, reproduces the run's CSV to 3e-16. Frame, features and scoring are therefore identical.
* The training window is identical (sot: 864,114 rows, calibration cut 2026-01-10).
* Local fits are deterministic: 1 vs 4 threads differ by 3e-16.
* HistGradientBoosting bins each feature from a 200,000-row subsample, and that subsample depends on row order. Shuffling the same training rows moves the probabilities by the same amount as runner vs local (max 0.054 vs 0.057, mean 0.0043 vs 0.0040), at equal log loss.

The amended rule checks four things per market on September: the pickle reproduces the CSV, every row is matched, 0 label disagreements, and refit vs run has |log-loss difference| ≤ 0.0005 with correlation ≥ 0.99. All five markets pass:

| market | rows | corr | log-loss refit − run |
|---|---|---|---|
| sot | 24,690 | 0.9992 | +0.000004 |
| sot2 | 24,690 | 0.9979 | −0.000072 |
| goals | 24,690 | 0.9977 | +0.000209 |
| assists | 24,690 | 0.9924 | −0.000061 |
| cards | 24,690 | 0.9967 | +0.000039 |

So the props probabilities here are **local monthly refits of the clean recipe on the run's verified frame (2025-01..2026-10).** They are equivalent to the runner's fit, not bit-identical to it.

**Data Truth stoplight.**

| where | result |
|---|---|
| data-truth-100 head 258abac30 | **GREEN**. 8a AMBER (0 active violations; props PENDING), **8b GREEN** (0 new provenance violations), 8c AMBER (7 registered historical exceptions, none inside the harness frames: they concern served payloads) |
| this branch, after the fix below | **GREEN** |

* Data truth 3 is RED on both: the live board lacks `has_ht_history`. It is serving-only, not gated, and not in any harness frame.
* This branch was RED on rule 2 before the fix: one `fillna(0)` on a "priced" count line in DEV-1's `price_inventory`. It is replaced by an exact crosstab count.
* data-truth-100 has since removed one more duplicate row (71,582 loaded vs the team frame's 71,583). The team runs predate it by one match; this is disclosed, not corrected.

### Inputs excluded (never approximated)

| excluded | reason |
|---|---|
| Pre-fix props pickles and outputs: the served champion's DEV-1 recipe refits (`cm-data/work/props_wf.parquet`), run 42855107089 payloads | fitted before the keeper `goals_conceded` / `team_goals_pg_roll` fix (f5fb6fe2d) |
| Served / legacy team payloads, v9_repaired | trained on unrepaired data or retired (registered 8c exceptions) |
| **wowza_v11** (`v11_shadow_log.csv`) | `p_model_over` is the live v9 board's served payload (pre-repair; it carries the nf-hist-af-override serving exception through 2026-11-15); its price is the latest pre-kickoff snapshot (not a proven bet-time price); O/U 2.5 only and starts in October (no learning window) |
| Props bookmaker prices (`player_prop_odds_history`) | they start 2026-10-14 (18 days of the scored window); "first price seen" is not a proven capture before the tip; CLV uses closing prices |
| Any closing price as a bet price | none used; no ROI computed |
| Props as a helper of team / Pro targets | lineup leak at the team tip time (protocol tip_time_rule) |
| Side models (corners / cards / specials) | **PENDING INPUT, NOT RUN**: origin/research/side-models (4e165327e) adds only `reports/R_SIDE_DATA_TRUTH_LIVE_2026-11-16.md` and its mismatches CSV; no `docs/` note with prediction paths |
| DEV-1 Pro λs (`cm-data/work/dc_wf.parquet`) | exported from the older v9 8c66793a3 data; re-run on the team frame's data instead |

Rows with defaults or fills: the props candidate fills nothing, Pro uses scores only, and the market is missing without a pair. The team champion_recipe imputes missing inputs with its train-only imputer and missing indicators. The provenance registry classifies this as eligible learned imputation, and the pre-registration accepted it on that basis. It is the one place where a model input is filled, and it is named here.

## 2. Design (as registered)

* **Scored window: 2026-04-01..2026-10-31 only.** No combined-model test had scored these months.
  * The team × market cells are a **re-look**: the 11-15 all-data test already opened this team holdout and compared champion_recipe with model_market_bet.
  * Stacks, recalibrations and tercile cutoffs learn only from 2025-01-01 up to the month before the scored month, refitted every month and per league.
* **Targets:**
  * team champion_recipe O1.5 / O2.5 / O3.5 / BTTS (Standard + NF);
  * props sot / sot2 / goals / assists / cards;
  * Pro Dixon-Coles 6+ goals / 0-1 goals / 0-0.
* **Helpers** (only what exists at the target's tip time):
  * team targets: Pro's same-market probability and the BET-time market;
  * Pro targets: the team-implied and market-implied same event;
  * props targets: team P(O1.5..BTTS), own and opponent Pro λ, market P(O2.5);
  * plus a stack of all of a target's helpers.
* **Residual test:** top vs bottom tercile of the helper's disagreement with the target (props: DEV-1's conditioners). Fixture-cluster bootstrap, 2,000 draws. Readable with ≥ 30 fixtures per group.
* **Stack test:** per-league logistic on [logit target + helper], compared with the **target recalibrated alone** on the same rows. One-sided gain in out-of-sample log loss. Readable with ≥ 300 training fixtures, ≥ 150 scored fixtures and ≥ 30 / 10 events.
* **Holm** over all 1,679 cells; unreadable cells enter at p = 1. Pooled rows are context only.

## 3. Target × helper results (per league, Holm over the whole family)

Full table: `reports/combined_models_clean_data/tests.csv`. In the "best cell" column, the gain is in log loss; residuals are in pp.

| target | helper | test | cells / readable | Holm-surviving | positive (readable) | raw p < 0.05 with the CI in the stated direction | best cell (effect, 95% CI, Holm p) |
|---|---|---|---|---|---|---|---|
| team | Pro | residual | 128 / 94 | 0 | 64 | 7 | - |
| team | Pro | stack | 128 / 48 | 0 | 27 | 1 | USA MLS O3.5: +0.0020 [+0.0003, +0.0037], H 1.00 |
| team | market | residual | 104 / 37 | 0 | 33 | 10 | USA MLS O2.5: +22.0 pp [+10.5, +33.4], H 0.33 |
| team | market | stack | 104 / 18 | 0 | 15 | 3 | USA MLS O2.5: +0.016 [+0.001, +0.031], H 1.00 |
| team | all | stack | 104 / 14 | 0 | 11 | 2 | USA MLS BTTS: +0.008 [+0.000, +0.016], H 1.00 |
| Pro | team | residual | 96 / 64 | 0 | 59 | 23 | Romania 0-0: +14.0 pp [+6.9, +21.6], H 0.35 |
| Pro | team | stack | 96 / 26 | 0 | 9 | 1 | - |
| Pro | market | residual | 78 / 27 | **1 (degenerate)** | 26 | 11 | Argentina 6+: +2.8 pp [+2.3, +3.2], H 1e-31, **0 events** |
| Pro | market | stack | 78 / 5 | 0 | 4 | 1 | USA MLS 0-1 goals: +0.015 [+0.004, +0.025], H 1.00 |
| Pro | all | stack | 78 / 5 | 0 | 3 | 1 | USA MLS 0-1 goals: +0.012 [+0.001, +0.024], H 1.00 |
| props | team | residual | 100 / 20 | 0 | 6 | 3 | Denmark cards: −5.1 pp [−7.7, −2.6], H 0.14 |
| props | team | stack | 100 / 55 | 0 | 11 | 0 | - |
| props | Pro | residual | 100 / 95 | 0 | 72 | 14 | Norway SoT: +4.7 pp [+1.6, +7.9], H 1.00 |
| props | Pro | stack | 100 / 45 | 0 | 22 | 1 | Romania SoT: +0.0009 [+0.0001, +0.0017], H 1.00 |
| props | market | residual | 95 / 5 | 0 | 5 | 1 | - |
| props | market | stack | 95 / 20 | 0 | 4 | 0 | - |
| props | all | stack | 95 / 15 | 0 | 3 | 0 | - |
| **total** | | | **1,679 / 593** | **0 genuine (1 degenerate)** | | | |

Most unreadable cells are stacks. Seven scored months rarely give a league 150 scored fixtures with a bet-time price, and props residuals need ≥ 30 fixtures in each third of an earlier-month cutoff. More than half of the family is unreadable because the window is short, not because of an effect.

### Surviving cells

**None with real outcome variance.** The single registered pass:

| league | target × helper | test | effect | 95% CI | Holm p | stack gain | why it is not a finding |
|---|---|---|---|---|---|---|---|
| Argentina Primera Division | Pro P(6+) × market | residual | +2.76 pp | [+2.32, +3.22] | 1.1e-31 | stack unreadable (3 six-goal matches in all 306 scored Argentina fixtures) | **0 six-goal matches** in the top (65), middle (65) and bottom (58) groups. The residual is −p, so the "effect" is Pro's own 0.8% vs 3.6% gap. The registered residual test lacked an event minimum. |

### Pooled context (never a decision)

**Stack gains, log loss vs the recalibrated target:**

| target | Pro | market | all helpers |
|---|---|---|---|
| team O1.5 | +0.0012 [−0.0010, +0.0032] | +0.0026 [−0.0016, +0.0071] | +0.0032 [−0.0018, +0.0084] |
| team O2.5 | +0.0011 [−0.0014, +0.0036] | +0.0010 [−0.0039, +0.0061] | +0.0007 [−0.0050, +0.0065] |
| team O3.5 | −0.0003 | +0.0003 | −0.0017 |
| team BTTS | −0.0008 | +0.0034 [−0.0010, +0.0074] | +0.0016 |
| Pro 0-1 goals | +0.0013 (team) | **+0.0046 [+0.0004, +0.0089]** | +0.0022 |
| Pro 6+ / 0-0 | −0.0014 / −0.0022 (team) | +0.0006 / +0.0012 | −0.0026 / −0.0017 |
| props SoT | **+0.00024 [+0.00007, +0.00042]** (Pro) | +0.00002 | −0.00025 |
| props SoT 2+ / goals / assists / cards | −0.00006 / +0.00001 / +0.00004 / 0.00000 (Pro) | −0.00015 / −0.00014 / −0.00007 / −0.00008 | −0.00095 / −0.00046 / −0.00027 / −0.00031 |
| props, team as helper (all markets) | | | −0.00017 to −0.00029 |

**Calibration of the targets themselves** (recalibrated alone vs raw):
* Pro specials improve by +0.003 to +0.011 log loss: Pro's Dixon-Coles tails are miscalibrated on these months.
* Props improve by about +0.0002 (cards get slightly worse, −0.0003), i.e. they are well calibrated.
* The team holdout gets *worse* when recalibrated on its walk-forward months (−0.0003 to −0.004).

**GBM stack-all** (context): mean gain vs the recalibrated target is −0.012 for Pro, −0.011 for team and +0.004 for props. Shallow trees overfit these small per-league samples.

**Residual context, pooled, 2026-04..10:**
* Team targets: +2.6 to +13.2 pp between the extreme disagreement thirds, every CI above 0.
* Pro targets: +4.6 to +14.8 pp.
* These patterns go away once the target is recalibrated (the stack gains above), so they mostly measure the target's own slope. They are not information the target lacks.

## 4. Comparison with 11-11 (did the side-λ effect survive the bug fixes?)

| measure | 11-11 (buggy props, DEV-1 months) | clean props, DEV-1 months (context re-look) | clean props, fresh months 2026-04..10 |
|---|---|---|---|
| pooled SoT | +1.3 [+0.9, +1.7] | +1.0 [+0.5, +1.5] | +1.6 [+0.9, +2.3] |
| pooled SoT 2+ | +0.7 [+0.5, +0.9] | +0.7 [+0.5, +1.0] | +0.6 [+0.2, +1.0] |
| pooled anytime scorer | +0.9 [+0.7, +1.2] | +0.8 [+0.5, +1.1] | +0.8 [+0.3, +1.2] |
| pooled assists / cards | - | - | +0.9 [+0.5, +1.3] / −0.6 [−1.2, −0.1] |
| Serie B anytime scorer | **+1.8 [+0.9, +2.7], Holm 0.019** | +2.0 [+0.9, +3.0] | +1.2 [−0.6, +2.9], **Holm 1.00** |
| per-league cells with CI > 0 | 1 of 57 after Holm | SoT 3 / 19, SoT 2+ 7 / 19, goals 5 / 19 (raw) | 12 of 95 (raw), 0 after Holm |

**Verdict.**
* The effect **survived the bug fixes**. 11-11's explanation, that a missing own-team attack feature caused it, is not supported: the feature is now fixed and correct (Spearman 0.35 with the own side's λ) and the effect stays the same size.
* It remains a small pooled residual: the props model slightly underrates players of sides that Pro expects to score more.
* It **does not survive per league on fresh data**, Serie B included.
* Adding Pro's λ to the props model improves log loss by about 0.0002 at most.
* `team_momentum_forward_matchup` is still in the clean schema and still carries almost nothing (Spearman 0.03 / 0.004). This is a candidate simplification for the props owner, not changed here.

## 5. Forward paper protocol COMBINED-MODELS-FWD-2

`config/experiments/combined_models/forward_protocol_fwd2.json`. **It registers no forward test, because no cell survived with a real outcome.**
* The degenerate Argentina cell is listed with the reason it is not carried forward.
* FWD-1 is void: its props champion was replaced, and its Serie B cell did not survive.
* The protocol states what would reopen the question: a new development protocol (DEV-3) declared before its results, for example:
  * more scored months once the clean props candidate and the team models are live;
  * the side models once their prediction paths are published and verified.

## Holm accounting and what is not claimed

* **Family:** 1,679 cells: team 568, Pro 426, props 685. 593 readable.
* **Not in the family:** the pooled rows, the GBM stacks, the recalibration comparison, the DEV-1-month re-look and the props feature-side check.
* **Configurations tried outside the registration:** none. Amendment A1 replaced a verification threshold before any effect existed. The residual test's missing event minimum is reported as a weakness of the registration, not patched after the fact.
* **Not claimed:** nothing here is PROVEN; no development row is priced as a bet; and no ROI, CLV or tip follows from it.
