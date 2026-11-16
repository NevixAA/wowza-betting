# P9 - Per-league bars re-run on the CLEAN models + TOP-3 v7 tiers (PROPOSALS ONLY)

Branch `bars/v5-clean-models` (off `prep/bars-oddsapi-hist` 8f1e06b9d). Owner 2026-11-16: "so why wait? do it now with
the fixed models". **Nothing is installed, approved or switched on.** `output/threshold_proposals/bars_oddsapi_hist/latest.json`
still points at TP-2026-11-15-bade2c0b; `config/threshold_approvals.json`, `config/bankroll_policy.json` and every
flag are unchanged. The owner decides.

* **New bars proposal: TP-2026-11-16-500964af** (`output/threshold_proposals/bars_oddsapi_hist/2026-11_500964af.json`,
  hash `500964af2c2ba3d16460f0caa5cafd23fad1980878d7e483144f3baf2f4babc9`, 64 entries, AWAITING_OWNER), evidence
  `evidence_2026-11_v5clean.json`, fresh holdout `TP-FH-2026-11-16-fae49848`.
* **New TOP-3 protocol proposal: top3-tiers/7** (`config/top3_tiers_protocol_v7.json`, sha256_lf
  `2670e6bf0209b7fe2243d0fc16dae3db3d2d9d7fa732d9d2e01b6aa33ff47cac`), frozen in commit 2d266f705 BEFORE the development fit;
  development thresholds (`output/research/top3_tiers_v7/t3_thresholds.json`, sha256_lf `d62b4b86...`) frozen in 824d902cf
  BEFORE the one holdout evaluation (`holdout_evaluation.json`).

Prices are PRIVATE (OddsAPI history, no redistribution): only counts and statistics are here. Priced universes and per-pick
frames stay in `C:\Users\nevo\dev\bars-oddsapi-local\v5clean` (local).

## 1. Input verification

| Check | Result |
|---|---|
| Recipe | `champion_recipe` ONLY (no info / league / logitcal / market variants), harness `phase2-retrain-1.0+info`, `config/phase2_info.json`, same holdout 2026-04-01..2026-11-08 (owner-approved 2026-11-07), folds, PG-1.0 |
| Frame (both tracks) | **974e04bc77c4d438** = the latest data-truth-100 (258abac30, after 5573b01d0) - dry run on feat/team-info-features 0a648a6b4 + origin/data-truth-100 (merge 5245d3493, no conflict) gives the same hash |
| Standard predictions | run_20261115T190707Z (code d7ded2ef5, frame 974e04bc77c4d438; backup `scratch-backup-2026-11-16/runs3`) |
| New-Format predictions | **re-run** run_20261116T153303Z on the latest frame (code 5245d3493, `--only champion_recipe --tracks new_format`, 1 thread, network blocked by the harness, no API call). Replaces the older f75ea2d98e98ad22 NF run |
| Out-of-sample, every row | walk-forward: 0 rows outside their fold's test block on 8 track x market sets (Standard 11 folds, NF 17 folds); each fold fits on dates <= train_end < test_start (`protocol.fold_frames`). Holdout: 2,857 Standard / 3,024 NF rows per market, all dated 2026-04-01..2026-11-08, models fitted on the walk-forward region only (dates < 2026-04-01); 0 walk-forward rows in the holdout window |
| Train/serve parity | `scripts/team_train_serve_parity.py` on the 8 clean payloads, last 6 dates (2026-11-01..06, 125 fixtures): **0 mismatched cells** (43 Standard / 28 NF raw columns), probability difference **0.0 pp** max on every payload |
| Data Truth stoplight | `python -m src.integrity.data_truth --gate` on this branch: **STOPLIGHT GREEN** (rules 1, 2, 4, 5, 6, 7, 8, 10a-c GREEN; rule 9 NO_DATA - Clean Record start not set). Rule 3 (operational, not gated) is RED: upcoming board fixtures 100% missing `has_ht_history` (72 Standard / 102 NF fixtures) - a live-board item, not training data |
| Identity map | loader `IDENTITY_ALIASES` regenerated on the latest frame: 133 aliases, identical to the v5 input |
| Bet prices | v5 exactly: tier **BT** only, `bet_role_only: true`; integrity 18,823 Standard + 16,410 NF history rows checked, **0 snapshot_after_kickoff**, 0 fixture / league / kickoff failures; football-data PRE-MATCH via the certified join (crosswalk sha 075889af...); own backfill stripped (6,029 Std / 6,715 NF pairs). Closing (B1, football-data closing columns) read only for CLV. Store = the v5 store commit 80f9de3 |

## 2. The v5 study re-run (protocol unchanged, sha256 4e5708fd...)

Same steps as v5 (runbook R7 2-3 + `study --month 2026-11 --tag v5clean`). The fresh-holdout windows are **identical to v5**
(the "seen" set is proven from git and did not change): Standard O1.5 / O3.5 2025-03-01 onwards, NF O2.5 / U2.5 / O3.5
2024-09-01..2025-06-22, NF O1.5 ..2025-06-27. Holm family 19 (v5: 17). **Honesty note:** these fresh rows were already
scored once (v3 and v5); this is a second scoring of the same outcomes with new predictions, labelled "re-test after a
price-data fix (outcomes previously seen)" - never untouched proof.

### Counts

| | PROVEN | PROMISING | UNPROVEN | OFF | cells |
|---|---|---|---|---|---|
| TP-2026-11-15-bade2c0b (approved) | 0 | 5 | 45 | 144 | 194 |
| **TP-2026-11-16-500964af (clean)** | **0** | **3** | **49** | **108 + 34 HT not re-measured = 142** | 194 |

The 34 Standard HT cells (ht_over05 / ht_over15) have no clean-model prediction (the clean champion_recipe harness scores
O1.5 / O2.5 / O3.5 / BTTS only); they were OFF (INSUFFICIENT_SELECTION_EVIDENCE) in bade2c0b and stay OFF, not re-measured.

| Track | Market-side | PROVEN | PROMISING | UNPROVEN | OFF |
|---|---|---|---|---|---|
| new_format | btts_yes | 0 | 0 | 4 | 11 |
| new_format | over15_over | 0 | 0 | 9 | 6 |
| new_format | over25_over | 0 | 1 | 9 | 5 |
| new_format | over25_under | 0 | 0 | 10 | 5 |
| new_format | over35_over | 0 | 1 | 4 | 10 |
| standard | btts_yes | 0 | 0 | 0 | 17 |
| standard | over15_over | 0 | 0 | 4 | 13 |
| standard | over25_over | 0 | 1 | 4 | 12 |
| standard | over25_under | 0 | 0 | 0 | 17 |
| standard | over35_over | 0 | 0 | 5 | 12 |

### Every cell that changed status (8 of 160 re-measured)

Price coverage is unchanged for every changed cell (same priced rows old / new), so every change is a model-probability
change (mean |dp| on the cell's priced rows 0.3-0.7 pp) moving a selection or a fresh-holdout result across a rule line.
The old proposal used the Nov-10 rebuilt champion_recipe (frame 58f10476), not the served pre-repair models, which is why
the probability moves are small.

| Cell | Old -> New | Why |
|---|---|---|
| NF U2.5 Austrian Bundesliga | PROMISING -> UNPROVEN | fresh holdout ROI +0.4% (43) -> -1.8% (40): no longer positive (MODEL_PROB, mean abs dp 0.62 pp) |
| NF U2.5 Japan J-League | PROMISING -> UNPROVEN | fresh holdout +33.8% (42, CI lo +0.1%) -> +29.7% (42, CI [-3.9%, +63.2%]); CLV +0.08% -> -0.24%: the PROMISING CLV condition fails (dp 0.62 pp) |
| NF O1.5 Denmark Superliga | OFF -> UNPROVEN | the selection now switches Denmark on (fresh test: 17 bets, -2.4%) (dp 0.29 pp) |
| NF O2.5 Norway Eliteserien | OFF -> UNPROVEN | switched on in selection; fresh test 61 bets, -3.9% (dp 0.69 pp) |
| NF U2.5 Denmark Superliga | OFF -> UNPROVEN | serving selection now finds own value (NO_SELECTION_EVIDENCE) (dp 0.63 pp) |
| NF O3.5 Romanian Superliga | OFF -> UNPROVEN | serving selection now finds own value on 50 priced rows (dp 0.45 pp) |
| Std U2.5 Belgian First Division A | UNPROVEN -> OFF | selection finds no positive bar (OFF_NO_VALUE) (dp 0.49 pp) |
| Std O3.5 Ligue 2 | UNPROVEN -> OFF | the fresh selection no longer switches Ligue 2 on (OFF_NO_OWN_ESTIMATE) (dp 0.30 pp) |

20 cells keep their status with a bar or configuration move (e.g. NF O2.5 fresh cells now `LEARNED|EDGE|min1.4` instead
of `LEARNED|EV`; La Liga 2 O2.5 qualifying bar unchanged at 0.13). Full list: `scripts/bars_status_diff.py` output.

### The 5 PROMISING cells and the overrides

| Cell | bade2c0b | clean | Stake consequence |
|---|---|---|---|
| Std O2.5 La Liga 2 | PROMISING (selection, certified FD prices) | PROMISING (bar 0.13 unchanged) | stays ACTIVE_LOW 0.5% |
| NF O2.5 Finland | PROMISING (+12.5%, 46) | PROMISING (+12.0%, 45, CLV +0.3%) | stays ACTIVE_LOW 0.5% |
| NF O3.5 Finland | PROMISING (+8.1%, 46) | PROMISING (identical) | owner block override ("Lift for Finland only") still applies, 0.5% |
| NF U2.5 Austria | PROMISING (PAPER, NF-Under rule) | UNPROVEN | none (was PAPER) |
| NF U2.5 Japan | PROMISING (PAPER, NF-Under rule) | UNPROVEN | none (was PAPER) |

Argentina NF BTTS: UNPROVEN, LEAGUE_WIDE in both; the owner's 0.5% league rule (max odds 2.6) is unchanged. So the
**staked set would not change** (La Liga 2 O2.5, Finland NF O2.5, Finland NF O3.5, Argentina NF BTTS, all 0.5%).

## 3. Expected TOP-3 tips per league per gameweek (holdout 2026-04-01..2026-11-08, weeks with fixtures, never padded)

v6 R2 (live rule) = tips clearing the clean cell bars, ranked by R2; v7 = the three named tiers. Statuses from
TP-2026-11-16-500964af; source parameters and calibration fitted on rows < 2026-04-01 only.

| League | Weeks w/ fixtures | v7 weeks 3/2/1/0 tips | v7 tips/wk | v6 R2 tips/wk | main causes (< 3 tips) |
|---|---|---|---|---|---|
| NF Argentina | 24 | 0/0/0/24 | 0.00 | 2.38 | NO_POSITIVE_EV 16, NO_PRICE 8 |
| NF Austria | 21 | 0/9/8/4 | 1.24 | 1.48 | NO_POSITIVE_EV 11, NO_PRICE 6, T3_NO_THRESHOLD 4 |
| NF Brazil | 25 | 0/0/2/23 | 0.08 | 2.76 | NO_POSITIVE_EV 19, NO_PRICE 6 |
| NF China | 25 | 0/0/0/25 | 0.00 | 0.00 | ALL_CELLS_OFF 25 |
| NF Denmark | 22 | 0/14/3/5 | 1.41 | 2.64 | T3_NO_THRESHOLD 12, NO_POSITIVE_EV 6, NO_PRICE 4 |
| NF Finland | 28 | 18/7/0/3 | 2.43 | 2.68 | T3_BAND_EMPTY 7, NO_PRICE 3 |
| NF Ireland | 28 | 0/0/0/28 | 0.00 | 0.00 | ALL_CELLS_OFF 28 |
| NF Japan | 21 | 0/5/5/11 | 0.71 | 1.38 | NO_PRICE 12, NO_POSITIVE_EV 7 |
| NF K-League 1 | 16 | 0/0/0/16 | 0.00 | 1.19 | NO_PRICE 11, NO_POSITIVE_EV 5 |
| NF Mexico | 26 | 0/0/2/24 | 0.08 | 1.12 | NO_POSITIVE_EV 18, NO_PRICE 8 |
| NF Norway | 22 | 0/0/0/22 | 0.00 | 2.32 | NO_POSITIVE_EV 12, NO_PRICE 10 |
| NF Romania | 26 | 0/0/0/26 | 0.00 | 0.77 | NO_PRICE 17, NO_CALIBRATION 9 |
| NF Saudi | 12 | 0/0/0/12 | 0.00 | 0.75 | NO_PRICE 7, NO_CALIBRATION 5 |
| NF Sweden | 26 | 0/0/0/26 | 0.00 | 2.12 | NO_POSITIVE_EV 17, NO_PRICE 9 |
| NF USA MLS | 26 | 0/0/0/26 | 0.00 | 2.19 | NO_POSITIVE_EV 23, NO_PRICE 3 |
| Std Belgium | 20 | 0/0/4/16 | 0.20 | 0.55 | NO_POSITIVE_EV 14, NO_PRICE 6 |
| Std Bundesliga 2 | 19 | 0/0/0/19 | 0.00 | 0.00 | ALL_CELLS_OFF |
| Std Championship | 19 | 0/0/0/19 | 0.00 | 0.47 | NO_POSITIVE_EV 14, NO_PRICE 5 |
| Std Eredivisie | 18 | 0/4/5/9 | 0.72 | 2.50 | NO_POSITIVE_EV 14, T3_NO_THRESHOLD 3 |
| Std Greece | 18 | 0/0/0/18 | 0.00 | 0.72 | NO_POSITIVE_EV 17 |
| Std La Liga 2 | 25 | 0/0/0/25 | 0.00 | 0.68 | NO_POSITIVE_EV 23 |
| Std League One | 21 | 0/0/0/21 | 0.00 | 0.29 | NO_POSITIVE_EV 20 |
| Std League Two | 21 | 0/0/0/21 | 0.00 | 0.00 | ALL_CELLS_OFF |
| Std Ligue 2 | 20 | 0/0/0/20 | 0.00 | 0.75 | NO_POSITIVE_EV 18 |
| Std National League | 18 | 0/0/0/18 | 0.00 | 0.00 | ALL_CELLS_OFF |
| Std Portugal | 21 | 0/0/0/21 | 0.00 | 0.00 | NO_PRICE 15, NO_CALIBRATION 5 |
| Std Scottish Champ / L1 / L2 / Prem | 19-20 each | 0/0/0/all | 0.00 | 0.00 | ALL_CELLS_OFF |
| Std Serie B | 18 | 1/7/3/7 | 1.11 | 1.50 | NO_POSITIVE_EV 9, T3_BAND_EMPTY 5, NO_PRICE 3 |
| Std Turkey | 18 | 0/0/0/18 | 0.00 | 0.00 | ALL_CELLS_OFF |

These are HISTORICAL rates on the price coverage of the holdout (own captures + BT history). From 2026-11-21 the live
capture prices every served fixture at bet time, so NO_PRICE weeks should shrink in proportion to live-capture coverage.

## 4. TOP-3 v7 (three named tiers) - protocol, fit, holdout

**Rules (frozen):** pool = holdout rows of non-OFF MODEL cells with a two-sided BET-role price, in the cell's odds range,
calibrated EV > 0 (v6 R2 calibration: p_cal = fair + lam (p_used - fair), lam per cell from development rows, >= 40,
shrunk inside the league). T1 = highest cal_ev; T2 = highest p_cal; T3 = highest cal_ev with odds >= the league's
odds_min and cal_ev >= its margin. Distinct fixture x market, one direction per fixture, never padded. Stake by proof:
STAKE only on PROVEN / PROMISING cells (PROMISING = 0.5%), otherwise PAPER. LEAGUE_WIDE cells stay the own list.

**Development fit (< 2026-04-01):** 21,761 scored rows, 2,186 with cal_ev > 0, in 11 leagues; 38 of 44 non-OFF model cells
have lam (median 0.26; 12 of 38 at <= 0.05, i.e. the dev data says the model adds almost nothing over the market there).
T3 thresholds (grid, >= 40 dev T3 picks, no global fallback): **Finland NF odds >= 2.75, margin 0.08 (44 dev picks,
+27.8%)**; **Serie B odds >= 3.00, margin 0.00 (42, +9.1%)**. The other 9 leagues have NO T3 threshold.

**Holdout (2026-04-01..2026-11-08, once, descriptive):**

| Tier | Bets | Hit | ROI | 95% CI | Mean odds | CLV (n) |
|---|---|---|---|---|---|---|
| T1 best value | 97 | 46.4% | -16.7% | [-36.0%, +2.2%] | 1.95 | +0.97% (97) |
| T2 safest | 65 | 63.1% | -9.4% | [-27.6%, +8.9%] | 1.58 | -0.08% (65) |
| T3 risky value | 19 | 10.5% | -67.4% | [-111.7%, -18.0%] | 3.13 | -1.23% (19) |
| **v7 combined** | **181** | 48.6% | **-19.4%** | [-32.3%, -6.4%] | 1.94 | +0.36% (181) |
| v7 STAKE (PROMISING cells) | 43 | 25.6% | -46.5% | [-76.7%, -16.3%] | 2.39 | +0.38% |
| v7 PAPER | 138 | 55.8% | -10.9% | [-25.1%, +3.3%] | 1.80 | +0.36% |
| **v6 R2 (live rule), same rows** | **702** | 50.0% | **-8.5%** | [-17.8%, +1.0%] | 2.00 | -0.35% (701) |

Per league (bets / ROI; combined with hit, CI, mean odds, CLV):

| League | T1 | T2 | T3 | v7 combined | v6 R2 |
|---|---|---|---|---|---|
| NF Austria | 17 / -26.7% | 9 / -51.7% | 0 | 26 / 27% / -35.4% [-77%, +7%] / 2.51 / -1.0% | 31 / -31.0% |
| NF Brazil | 2 / -100% | 0 | 0 | 2 / 0% / -100% / 2.24 | 69 / -1.5% |
| NF Denmark | 17 / -2.1% | 14 / +23.2% | 0 | 31 / 68% / +9.3% [-18%, +36%] / 1.64 / +1.6% | 58 / +3.8% |
| NF Finland | 25 / -32.8% | 25 / -12.0% | 18 / -65.6% | 68 / 44% / -33.8% [-55%, -13%] / 1.94 / -0.1% | 75 / -31.6% |
| NF Japan | 10 / -11.7% | 5 / -12.4% | 0 | 15 / 40% / -11.9% [-79%, +55%] / 2.15 / +1.9% | 29 / -18.0% |
| NF Mexico | 2 / +30.5% | 0 | 0 | 2 / 100% / +30.5% / 1.31 | 29 / -26.6% |
| Std Belgium | 4 / -33.2% | 0 | 0 | 4 / 50% / -33.2% / 1.33 | 11 / +7.4% |
| Std Eredivisie | 9 / +15.9% | 4 / -50.5% | 0 | 13 / 54% / -4.5% [-55%, +47%] / 1.83 / -1.4% | 45 / -4.2% |
| Std Serie B | 11 / -5.4% | 8 / +12.0% | 1 / -100% | 20 / 65% / -3.1% [-44%, +38%] / 1.77 / +1.7% | 27 / +0.1% |

v6 R2 also picks in 12 leagues where v7 has none (no calibrated positive EV): Argentina 57 / +5.0%, Sweden 55 / +1.9%,
USA MLS 57 / -47.6%, Norway 51 / -15.7%, K-League 19 / -18.5%, Romania 20 / +12.0%, Saudi 9 / +41.7%, La Liga 2 17 /
+49.0%, Greece 13 / +29.6%, Ligue 2 15 / +1.5%, Championship 9 / +19.8%, League One 6 / +12.5%.

**Reading.** No tier and no league is positive with a CI above zero; v7 is worse than v6 R2 on these rows (-19.4% vs
-8.5%) and covers far fewer leagues, because the calibration slopes are small: once the model is shrunk toward the fair
price, most rows carry no positive EV after the margin. T3 (long odds, the only two leagues with a threshold) lost 18 of
19. This is a descriptive holdout whose outcomes were read before by other rules; it is not a test and not a pass.

### Shortfall (holdout, 32 leagues x 32 weeks)

| | 3 tips | 2 | 1 | 0 | league-weeks |
|---|---|---|---|---|---|
| all | 19 | 46 | 32 | 927 | 1,024 |
| weeks with fixtures | 19 (2.8%) | 46 (6.7%) | 32 (4.7%) | 585 (85.8%) | 682 |

Causes of the 663 league-weeks with fixtures and < 3 tips: NO_POSITIVE_EV 263, ALL_CELLS_OFF 207, NO_PRICE 139,
T3_NO_THRESHOLD 21, NO_CALIBRATION 19, T3_BAND_EMPTY 12, NO_MODEL 2 (plus 342 NO_FIXTURES weeks: breaks / off-season).
Slot level (fixture x non-OFF cell, 8,637 slots): 3,086 no BET-role price, 294 no calibration, 45 no model, 4,735 priced
without positive calibrated EV (includes rows outside the cell's odds range), 477 value slots.

### Fix list (data / collection causes only; listed, not built)

| # | Cause | Where | Fix | Cost |
|---|---|---|---|---|
| 1 | NO_PRICE (139 weeks, 3,086 slots) | NF Romania 372 slots, Sweden 371, Finland 283, Norway 263, USA 266, Denmark 193, Brazil 172, Japan 160, Saudi 129, Argentina 127, K-League 83; Std Championship 189, Portugal 114 | (a) from 2026-11-21 the live bet-time capture must cover every served league x market (O1.5, O3.5, BTTS, U2.5 included), not only O/U 2.5 - check the capture list against the 44 non-OFF model cells; (b) history: extend tier BT to the uncovered leagues / markets | (a) code + a capped test run, no extra credits beyond the live plan; (b) OddsAPI historical credits per event x market x snapshot - size it on the 3,086 missing slots against the OddsAPI 5M plan floor before running |
| 2 | Portugal has no BT history at all | Std Portugal 114 slots, 41 no-calibration | collect BT for Portugal (branch prep/portugal-oddsapi-key exists) | one league of BT history credits |
| 3 | NO_CALIBRATION (19 weeks, 294 slots) | NF Romania 156, Saudi 97, Std Portugal 41 | follows from 1 and 2: these cells have < 40 priced development rows | none beyond 1-2 |
| 4 | BTTS has no history price | all Standard and NF BTTS cells (tier BT holds no Pinnacle BTTS pair; Bet365 is outside BT's eu region) | request BTTS from a region with a BTTS pair (uk) in a BT re-pull | credits for BTTS x the covered events |
| 5 | HT families not re-measured | 34 Standard HT cells | add HT markets to the clean harness run and HT to BT | one harness run (local, ~1 h) + credits |
| 6 | NO_MODEL (2 weeks, 45 slots) | promoted / first-match fixtures without rolling form | by design (blind fixture, never tipped) - no fix | - |

ALL_CELLS_OFF (China, Ireland, Bundesliga 2, League Two, National League, 4 Scottish, Turkey) and NO_POSITIVE_EV are
model / bars outcomes, not data gaps.

## 5. Caveat and the Nov 20 confirm-diff

The 2026-11-19/20 final refit adds about one more week of data (same recipe, same frame rules). Proposal: no full re-study;
on Nov 20, after R6, rebuild the universes on the refit predictions (runbook R7 steps 2-3), then

```
python scripts/bars_status_diff.py --old-evidence output/threshold_proposals/bars_oddsapi_hist/evidence_2026-11_v5clean.json \
    --new-evidence <R7 re-check evidence> --old-universe <this run>/priced --new-universe <refit>/priced --out <diff.json>
python scripts/bars_opt_oddsapi.py recheck --approved-evidence <the approved evidence> --universe <refit>/own_bettime \
    --study-cells <refit>/selstudy/cells.json --month 2026-11
```

which reports the prediction difference per family (mean / p95 |dp|) and every cell whose verdict would move. Today's
prediction difference between the bade2c0b inputs (frame 58f10476) and this run (974e04bc) is mean |dp| 0.08-0.96 pp
(p95 0.2-2.5 pp) per family; a one-week refit should be of that order or smaller. Any status change goes back to the owner.

## Reproduce

Runs: `C:\Users\nevo\dev\v5clean-runs` (NF run, dry-run plan, parity, OOS check `verify_oos.py`, pipeline), bars work dir
`C:\Users\nevo\dev\bars-oddsapi-local\v5clean` (u, priced, own_bettime, selstudy, study, top3v7). Store
`C:\Users\nevo\dev\exec-store` at 80f9de3.
