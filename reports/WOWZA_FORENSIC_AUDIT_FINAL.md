# WOWZA forensic audit — final verdict (2026-11-01)

**Question:** does Wowza genuinely lose to bookmakers, or is some of the reported loss caused by implementation or evaluation defects?

**Short answer:** both. Selection, settlement and the main-market prices are sound. The losses on Standard O/U 2.5 (old model) are real. But for every model except the repaired Standard O/U 2.5, the inputs fed to the model live did not match the inputs it was trained on. Their August–October results therefore measure a broken pipeline, not what a correctly built model can do. Separately, several "positive" claims came from contaminated evidence: props CLV, the Over/Under split, and New Format BTTS as model skill. No track has a confirmed edge today.

**Scope and method:** read-only audit of v9 `origin/main` **fb0e9697ec8556d4dd3df57f31f8ed5415a5012d**, plus Pro `e0ccc27b98f5`, v11 `36ed1cd71ecb` and the slim dashboard `4811bdf37be4`. Five independent teams worked on branches `audit/au1`–`audit/au5`, consolidated on `audit/final`. Live evidence counts from **2026-09-01**, because August 2026 had a known bug. All ROI figures are flat 1 unit at first-sight prices, settled bets only, with matchday-block bootstrap 95% CIs unless stated otherwise.

| Phase | Report |
|---|---|
| 0 | `00_SYSTEM_EXECUTION_MAP.md` |
| 1 | `01_DEFAULTS_AND_FALLBACK_BUGS.md` |
| 2 | `02_ML_TRAIN_SERVE_FORENSICS.md` |
| 3 | `03_STANDARD_REPAIRED_MODEL_VERDICT.md` |
| 4 | `04_UNDER25_FORENSIC_INVESTIGATION.md` |
| 5 | `05_BOOKMAKER_PRICE_INTEGRITY.md` |
| 6 | `06_SETTLEMENT_AND_ROI_RECONCILIATION.md` |
| 7 | `07_CONFIGURATION_REGIME_AUDIT.md` |
| 8 | `08_PLAYER_PROPS_AND_CLV_AUDIT.md` |
| 9 | `09_POSITIVE_EDGE_CANDIDATES.md` |
| 10 | `10_INDEPENDENT_BACKTEST.md` + `10_backtest_mismatches.csv` |

---

## 1. Can Wowza genuinely beat bookmakers? — market by market

| Track | Verdict | Valid bets | ROI | 95% CI | Model version | Status | Main limitation |
|---|---|---|---|---|---|---|---|
| Standard O/U 2.5, legacy model (09-01 → 10-28) | **NEGATIVE PERFORMANCE** for the model as served | 315 | −10.4% | −21.3 to +2.4 | legacy (unversioned until 10-18) | retired 10-28 | Did worse than picking a random side (−7.9%); its features mismatched training on 35 of 42 |
| Standard O/U 2.5, repaired model (10-28 →) | **INSUFFICIENT DATA** — "INSUFFICIENT FORWARD EVIDENCE — CANNOT DECLARE REPAIRED MODEL UNPROFITABLE." | 4 settled of 23 | — | — | standard_repaired_…1f5805e6 | production | Needs about 150 settled; break-even hit rate 45.5% at mean odds 2.20 |
| New Format O/U 2.5 | **INVALID EVALUATION** as a test of model skill; **NO DEMONSTRATED EDGE** as served | 490 | −2.5% | −15.0 to +9.0 | legacy | production | Train/serve mismatch on 20 of 27 features (Δp up to 16.4 pp); downward base-rate bias; parity fixes landed 11-01 |
| New Format BTTS | **PROMISING / UNCONFIRMED**: a league regime, not model skill | 224 | +9.7% | −2.6 to +22.8 | legacy | production | 105% of the profit is Argentina, where blind YES returns +28.6%; decaying (Aug +49.7%, Sep +21.9%, Oct −4.4%) |
| New Format BTTS, excluding Argentina | **NO DEMONSTRATED EDGE** | — | −5.3% (from 09-01) | — | legacy | — | — |
| Standard BTTS | **NO DEMONSTRATED EDGE** | 146 | +1.7% | spans 0 | legacy | production | Same train/serve mismatch as above |
| New Format Over 1.5 | **NO DEMONSTRATED EDGE** (selection) | 78 | +10.9% | −5.3 to +28.0 | legacy | production | 71 of 76 bets are Argentina; equals a random priced subset (p = 0.26); selects 92.5% of priced boards |
| Over 3.5 | **INSUFFICIENT DATA** | 19 | −18.6% | spans 0 | legacy | production | — |
| Half-time markets | **INVALID EVALUATION / INSUFFICIENT DATA** | 0 staked since 09-29 | — | — | legacy | production | Features assigned to the wrong fixtures since 09-29 (AU2-B01) |
| Player props | **NEGATIVE PERFORMANCE**, paper only | 2,024 verified-entry | −28.1% | −33.3 to −23.0 | legacy | paper | Clean CLV −0.97% (−1.80 to −0.11); model adds nothing to price; settlement unverified (matched by name) |
| All side-market backtests | **INVALID EVALUATION** | — | — | — | — | — | Synthetic fill prices (O1.5 22,441 of 22,516) or closing-like prices; stale relative to the current loader |
| Serie B BTTS YES (league regime) | **PROMISING / UNCONFIRMED** | registered forward test from 11-01 | — | — | none (blind) | research | Discovery was 1 of 14 cells |

**Overall:**
- No track meets CONFIRMED POSITIVE EDGE.
- Standard O/U (old model) is genuinely negative.
- For New Format, the side markets and half-time, the honest status is **not yet testable**. Their pipelines were broken while their results were collected, so the losses neither prove nor disprove a correctly built model.
- Props have no betting edge on clean data, which confirms the existing paper-only rule.

### Claimed vs verified

| Claim | Where it came from | Independently verified | Corrected |
|---|---|---|---|
| NF BTTS +18.9% (142 bets, CI +1.8 to +35.0) | SNIPER+MARKSMAN by **ledger** tier | Reproduces exactly | First-sight tier from 09-01: +23.0%, n 70, CI −5.3 to +49.7. All tiers: +8.0% to +9.7%, CI spans 0. Argentina is 105% of the profit |
| Props CLV +1.03%, "genuine edge" | `src/clv_capture.py:113`: sign of an odds-weighted mean, no CI | Reproduces (n 4,125; beat-close 30.3%) | Clean CLV −0.97% (CI −1.80 to −0.11); log CLV −1.68%; 128 namesake-priced records averaged +52.5% |
| OVER +6.4% vs UNDER −12.5% | Not found as stated; closest is SNIPER+MARKSMAN by upgraded ledger tier | 09-01+: OVER +8.3%, UNDER −13.0% on upgraded tiers | All tiers from 09-01: OVER −4.6% (n 258), UNDER −5.0% (n 587) |
| Live O/U CLV +5.76% | Ledger as stored | Reproduces | 168 closes were taken from mis-parsed rows; without them +1.01%. From 09-01: NF −0.35%, Standard +0.18%, both zero within CI |
| "The bookmaker's price is as good as anything we can build" | Logit / league-level / anchored studies, 11-01 | Not valid | Tested on pipelines with the train/serve defects found here. Withdrawn |

---

## 2. Top 10 confirmed bugs (full Phase-11 rows are in the phase reports)

| # | ID | Sev | Status | File / location | Root cause | Measured effect |
|---|---|---|---|---|---|---|
| 1 | AU2-B07 | P1 | ACTIVE (all except repaired Std O/U) | legacy `build_features` vs `build_upcoming_features` | Two different feature builders for training and serving | Canary on identical history: 35 of 42 features disagree (Std / side), 20 of 27 (NF), 122 of 138 (props). Δp median 1.4–2.5 pp, max 16.4 pp (NF). Repaired Std: 0 of 41, exact |
| 2 | AU1-B05 / AU2-B04 | P1 | ACTIVE (legacy payloads) | `src/model.py` legacy `_prep` | Missing values filled from the median of the scoring batch | NF fixture alone vs in batch: ΔP(over) median 1.7 pp, max 17.9 pp. Props: single vs batch up to 1.29 pp (6.8 pp historically) |
| 3 | R2-C finding 12 | P1 | ACTIVE; fix is behind a flag (OFF) | `src/data_loader.py` af_history join | Joined on home+away+date with no league, then copied back by position | 85,454 history rows misaligned since 2019 (shots, xG, BTTS odds). Changes serving features on NF / side; 4 tips differ each way on 11-01 |
| 4 | AU1-B03 / AU2-B02 | P1 | ACTIVE (legacy builders) | `src/feature_engineering.py:788-793` | Rest days written back by position after a sort; last 1,000 rows only; no name resolution | 47.5–60% missing live vs ~0% in training; 157 of 158 board rows had another fixture's rest days. Δp median 0.2–0.4 pp, p90 up to 1.9 pp; 6 of 88 NF rows cross the 3% floor |
| 5 | AU2-B01 | P2 | ACTIVE since 09-29 | `src/predict.py:501` | HT features assigned by position; `attach_ht_features` returns rows sorted | Only 6–35% of rows get their own HT features; HT probabilities move up to 3.6 pp. No HT tips since, so no P/L effect |
| 6 | AU4-P8-01 + AU4-P5-01 | P1 | ACTIVE code | `src/clv_capture.py:113`; `update_results.py:261-286` | CLV verdict is the sign of a mean with no CI; ledger closes read mis-mapped NF parser-window rows | Props "genuine edge" verdict is false (clean −0.97%); live O/U CLV inflated from +1.01% to +5.76% |
| 7 | AU3-06-D / -E, AU2 cards | P2 (paper) | ACTIVE | `update_results.py` player settlement | Players matched by name with a 5-character fallback; no player/fixture id; sot2/sot3 never settle; cards target counts yellows only | 12 groups settled WIN under one spelling and LOSS under another; ~691 false VOIDs; 1,420 sot2/3 rows unsettleable; 2,982 straight reds labelled "not carded". Props marked SETTLEMENT_UNVERIFIED |
| 8 | AU3-06-A / B-AU5-02 / AU1-B15 | P2 | HISTORICAL (fixed 10-26) | `src/ledger.py` before `dfcdf289d` | Ledger upgraded tier and odds in place after the tip was sent | 627 tiers and 92 side odds rewritten; side P&L overstated by +2.25 to +2.35 u; tier-level ROI and H-BTTS-01 invalid before 10-26 |
| 9 | AU3-06-B | P2 (paper) | ACTIVE in file | `src/stake_plan.py` stake audit + track rename `ab23726d5` | Audit keyed by track; renaming NF O/U to OU25_UNDER sized one tip twice | 31 fixtures; 10.22 u of the 100 u paper bankroll allocated twice. Ledgers, ROI and tips unaffected |
| 10 | AU1-B04 / AU1-B13 | P2 | ACTIVE (research input) | committed side backtests | Backtests stale relative to the loader and priced near close; O1.5 per-league thresholds certified on the 1.40 fill were live 08-02 → 09-22 | Side thresholds certified on 1,181 real BTTS prices instead of 6,352 available |

Also confirmed:
- **B-AU5-01 (historical):** manual commit `075eeb313` (06-22) deleted 23 live rows; the June record is missing +5.60 u.
- **B-AU5-05 (active):** 32 played fixtures never settled; +9.37 u unbooked.
- **AU2-L01 (active):** serving history lags results; Championship ends 10-24, and 48 of 70 Std board rows are missing a team's latest match, unflagged.
- **AU2-B06 (active):** 42 PG-1.0 lineage rows since the switch carry the legacy label with the repaired probability.
- **AU1-B01:** the edge ceiling is dead code (labels only).
- **AU1-B09:** v7 `best_params.json` is read from outside the repo, so the both-losing guard is off in CI.

---

## 3. Which earlier loss conclusions are valid?

- **Settlement is correct.**
  - 2,024 bets were settled by both ledger and audit (AU3) with 0 WIN/LOSS mismatches.
  - 2,001 more were checked by the independent engine (AU5): 0 label and 0 P&L differences.
  - Ledger P&L equals the recomputed figure on all 602 settled side rows.
- **Selection logic is correct.** The independent engine (no production imports) re-selected 100% of the 1,530 production bets from 09-01 under the thresholds in force at the time.
- **O/U entry prices were real and available at decision time.** All 1,126 live O/U rows since 08-19 were standing at a book; the price was the best one 48.6% of the time, with a median gap to the best of 0.50%.
- **The legacy Standard O/U model lost money**, and did worse than a random side.
- **The model is overconfident** by about 9 pp on Under tips (claimed 0.504, realised 0.414, market 0.402). The vigged price beats the model's Brier score on its own selected bets.
- **The tier ladder carries no information.** It is inverted on NF O/U (SNIPER hit 34.5% vs VALUABLE 49.2%), and edge does not rank results (Spearman −0.07).
- **Props have no betting edge,** now shown on clean CLV and verified-entry ROI, not just prediction metrics.
- **Synthetic side prices never entered a bet's ROI.**

## 4. Which earlier conclusions are invalid or premature?

- **"NF and side-market models have no skill":** premature. Their train/serve inputs were mismatched (bug 1), imputation depended on the batch (bug 2), shot and xG history was misaligned (bug 3), and rest days were wrong (bug 4).
- **"Props show POSITIVE CLV — genuine edge":** invalid. It came from an odds-weighted mean, borrowed namesake prices and no CI.
- **"NF BTTS +18.9%" as model skill:** invalid as stated. It relies on ledger-upgraded tiers and the Argentina league regime, and it is decaying.
- **"OVER +6.4% / UNDER −12.5%":** not reproducible as stated; it is an upgraded-tier artefact.
- **Stored live O/U CLV +5.76%:** invalid, because of corrupted closes.
- **Any side-market backtest ROI, and O1.5 thresholds certified on 1.40:** invalid.
- **"The repaired Standard model is unprofitable":** not supportable; there are 4 settled bets.
- **The lead's 11-01 statement "the bookmaker's price is as good as anything we can build":** withdrawn (see the table in section 1).

## 5. Which profitable tracks deserve forward validation?

| Candidate | Evidence so far | What is needed |
|---|---|---|
| **Argentina BTTS YES** | League regime: blind YES +28.6%; the model adds nothing within the league (tipped minus not-tipped −0.8 pp) | A preregistered forward test with a blind-YES control. H-BTTS-01 needs a versioned amendment to first-sight tiers |
| **Serie B BTTS YES** | Registered 11-01 with a blind control and a picked-vs-not-picked comparison | — |
| **Brazil BTTS** | +17.7%, n 39 | INSUFFICIENT DATA; register before counting |
| **NF Over 1.5, Argentina** | 71 of 76 bets are Argentina | Same treatment as Argentina BTTS |
| **Props VALUABLE clean CLV +3.69%** (CI +1.08 to +6.51, n 145) | Picked among about 15 subgroups; verified ROI is negative | A preregistration candidate only, not evidence |
| **Repaired Standard O/U, and repaired NF / side once built** | — | Forward accrual to at least 150 settled bets per track, attributed by model version |

## 6. What should be repaired first?

1. **Extend the repaired recipe (one shared feature builder plus a persisted imputer) to NF O/U, the side markets and HT.**
   - Removes bugs 1, 2, 4 and 5 at once.
   - The R2-B challenger infrastructure exists and is CI-verified: serving parity byte-identical, 15 benchmark jobs green.
   - Route: shadow first, then owner review.
2. **Turn on the safe af join (`WOWZA_SAFE_ENRICH_JOINS`) as part of that retrain.** The current models were trained on misaligned history.
3. **Fix the CLV evaluation:**
   - a robust metric with a CI instead of `mean > 0`;
   - mark the 168 parser-window closes invalid;
   - pre-kickoff and same-book closes for props.
4. **Settle player props by player id and fixture id;** support sot2/sot3; settle cards as the bookmaker does.
5. **Result data freshness:** settle the 32 unsettled fixtures, and add a health flag when serving history lags results.
6. **Research evidence hygiene:**
   - amend H-BTTS-01/02 to first-sight tiers;
   - fix the PG-1.0 lineage labels (AU2-B06);
   - re-key the stake audit by fixture.

## 7. What must not be changed yet

- **Repaired Standard O/U serving:** keep it, and let it accrue evidence.
- **Thresholds:** do not tune SNIPER / MARKSMAN / VALUABLE thresholds or add direction or odds filters (for example "no Under above 2.5") from these historical subsets. They are descriptive, and any such rule needs untouched forward data.
- **Props stay paper.**
- **No real staking on Argentina / NF BTTS** until a preregistered forward test confirms it.
- **The `/1.08` side-edge rule and `WOWZA_PRICE_MAPPING_FILTER`:** owner decisions. Run the de-vig shadow longer first.
- **Historical ledgers:** do not rewrite. Report first-sight values instead.

## 8. Tests, commands and artifacts

**Branches:**
- `audit/au1` `c70878766`, `06f3c9eb8`
- `audit/au2` `73b113213`
- `audit/au3` `80277107f`
- `audit/au4` `6392fb5b4`
- `audit/au5` `a78c2dc93`, `299d45eee`
- consolidated `audit/final`

**Tests** (`tests/audit/`, all on `audit/final`): **58 passed, 15 strict xfail.**
- The strict xfails encode the correct behaviour for active defects: AU1 (7), AU2 (4) and AU4 (4). Each will turn red when its defect is fixed, which signals that the xfail marker should be removed.

| File | Tests |
|---|---|
| `test_au1_findings.py` | 4 guards + 7 xfail |
| `test_au2_train_serve.py` | 4 controls + 4 xfail |
| `test_au3_settlement.py` | 22 adversarial |
| `test_au4_props_clv.py` | 8 + 4 xfail |
| `test_independent_backtest_adversarial.py` | 20 adversarial: missing odds, swapped teams, namesakes, inverted Under, wrong line, duplicates, same player name on two fixtures, missing stats, post-kickoff quote, missing artifact, wrong feature order |

**Full v9 suite** on the audited SHA: 1,505–1,509 passed. The only failures are the 4 known pre-existing ones (dashboard_staked_basis, fantasy_v2 archive, promotion_gate, shadow_settlement).

**Commands:**
- `python -m pytest tests/audit -q`
- Re-run any phase with `python scripts/audit/au<N>_*.py`. The scripts write `reports/au<N>_data/`.
- Independent backtest: `python -m scripts.audit.independent_backtest.extract`, then `python -m scripts.audit.independent_backtest.run`.

**Not identifiable / blockers:**
- football-data.co.uk was unreachable from the laptop (TLS), so outcomes come only from committed sources.
- Actual-stake ROI before 10-14 has no stake log.
- Selections before 06-25 depend on an uncommitted guard file.
- Decision-time price availability before 08-19 cannot be checked (no per-book data).
- The ROI impact of bugs 1–4 requires re-serving historical boards with corrected features; that is part of repair step 1.
