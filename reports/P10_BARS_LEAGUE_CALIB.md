# P10 - Per-league bars on the per-league CALIBRATION variants of the team models (PROPOSAL ONLY)

Branch `bars/v5-league-calib` (off `bars/v5-clean-models` 5f481d3b3). Owner 2026-11-16: run the per-league bars study for
the calibration variant and see whether it creates proven edges in more leagues than the clean champion. **Nothing is
installed, approved, flagged or merged.** `latest.json`, `config/threshold_approvals.json`, `config/bankroll_policy.json`
and every flag are unchanged.

## Plain-language summary

* **No. The calibration variants do not create a single PROVEN edge in any league - the same as the clean champion
  (0).** By the pre-registered rule ("better" = more leagues with a PROVEN cell), neither variant is better.
* What they do change: the bars selection switches **more cells on** (non-OFF cells: champion 52, variant A 75,
  variant B 63) and finds **a few more PROMISING cells** (champion 3 in 2 leagues; A 5 in 4 leagues; B 7 in 5 leagues).
  None is close to proof: every combined-Holm p is 1.0, and the PROMISING cells are thin (A's Eredivisie O3.5 rests on
  13 bets; Argentina NF U2.5 on +0.2% over 133 bets).
* Both variants **drop the champion's La Liga 2 O2.5 PROMISING cell** (one of today's staked 0.5% cells) to UNPROVEN.
* TOP-3 under the live v6 R2 rule: A gives more tips (800 vs 702 on the holdout) and a less negative result (-5.4% vs
  -8.5%), but both 95% intervals include zero, and these outcomes had been seen before. League-weeks with 0 tips fall
  from 56% to 50%. That is because more cells are switched on, not because there is proven value.
* The calibration fix does what it was built to do on the **Standard** track: the within-league calibration slope moves
  toward 1 in 49 of 68 league x market cells (median |slope - 1| 1.49 -> 0.74), and the flattened Standard league levels
  are repaired (cross-league level slope BTTS 4.46 -> 0.43, O1.5 2.39 -> 0.94). On New-Format it changes little
  (median |slope - 1| 0.86 -> 0.82). The mean absolute league-level gap barely moves (Std 4.35 -> 4.24 pp, NF 3.94 -> 3.94 pp).
* **Recommendation: keep the clean champion bars (TP-2026-11-16-500964af) as the basis.** The variant-A proposal
  **TP-2026-11-16-9cef8a02** is filed for the owner as evidence only. Approving it would change the staked set
  (La Liga 2 O2.5 out; Serie B O1.5, Eredivisie O3.5 and Argentina NF U2.5 in, all PROMISING), with no proof behind it.

## 0. Pre-registration

`config/bars_league_calib_protocol_p10.json`, sha256 `fd15aba47d1c2a83c44691f38a8edfcfaffda3859c7b72ca42c3f7e3de27c626`,
committed in bdafebfab BEFORE any result of this study was computed. It covers: variant A = `champion_recipe+league+logitcal`
(primary), B = `champion_recipe+logitcal`; the v5 protocol, cells and rules unchanged; per league, never global; ONE Holm
family over both variants plus the champion's 19 fresh cells; and the "better than the champion" rule with a per-cell
comparison. **Honesty:** the holdout 2026-04-01..2026-11-08 and the fresh-holdout rows had already been scored by other
rules (v3, v5, P9, the all-data study). This is a second look at outcomes already seen, labelled "re-test after a
price-data fix (outcomes previously seen)". It is not untouched proof.

## 1. Verification

| Check | Result |
|---|---|
| Frame | **974e04bc77c4d438**. Dry run of the harness on feat/team-info-features 0a648a6b4 + the LATEST origin/data-truth-100 **7d74553a4** (local merge 826881ac1, not pushed): the hash is unchanged by 7bcb54449 (af writeback preservation; the restored provider rows are kept beside the data, not loaded) |
| Predictions reused (allowed by the protocol: same frame, out-of-sample) | Standard run_20261115T190707Z (code d7ded2ef5), New-Format run_20261116T153303Z (code 5245d3493). Both manifests: frame 974e04bc77c4d438, config sha256 41648ecf... = the dry run. Both runs already held the two variants, so nothing was re-fitted and no API call was made |
| Out-of-sample, every row (2 variants x 2 tracks x 4 markets) | walk-forward: **0** rows outside their fold's test block (Standard 11 folds, NF 17); holdout 2,857 Std / 3,024 NF rows per market, all in 2026-04-01..2026-11-08, from models fitted on dates < 2026-04-01; **0** walk-forward rows in the holdout window (`oos.json`) |
| Train/serve parity of the 16 variant payloads | `scripts/team_train_serve_parity.py`, last 6 dates (2026-11-01..06, 125 fixtures): **0 mismatched cells** (Standard 50 raw columns incl. the league block, NF 33), probability difference **0.0 pp** max on all 16 payloads, 0 NaN rows |
| Data Truth stoplight | `python -m src.integrity.data_truth --gate` on this branch: section 7 |
| Bet prices | identical to P9: tier BT only, `bet_role_only`; 18,823 Std + 16,410 NF history rows checked, **0 integrity failures**, 0 snapshots after kickoff; football-data pre-match via the certified join (crosswalk 075889af...); own backfill stripped; B1 closing only for CLV; store 80f9de3; identity map = the P9 map (same frame). Price coverage is the same as P9 row for row (Std 31,511 priced of 93,512, NF 9,776 of 94,188) |
| Fresh-holdout windows | identical to P9 (the seen set is proven from git and unchanged) |
| Replay check | the v6 R2 TOP-3 replay (`scripts/bars_league_calib_p10.py top3`) on the champion inputs reproduces P9 exactly: 702 bets, -8.5% [-17.8%, +1.0%], and the same per-league tips/week |

## 2. Cell counts vs the clean champion (194 cells; the 34 Standard HT cells stay OFF, not re-measured, as in P9)

| | PROVEN | PROMISING | UNPROVEN | OFF | leagues with >= 1 PROVEN | leagues with >= 1 PROMISING |
|---|---|---|---|---|---|---|
| Clean champion TP-2026-11-16-500964af | 0 | 3 | 49 | 108 + 34 = 142 | 0 | 2 (Finland NF, La Liga 2) |
| **A league + logit cal, TP-2026-11-16-9cef8a02** | **0** | **5** | **70** | **85 + 34 = 119** | **0** | 4 (Finland NF, Argentina NF, Serie B, Eredivisie) |
| B logit cal alone (research, no proposal filed) | 0 | 7 | 56 | 97 + 34 = 131 | 0 | 5 (Finland NF, Argentina NF, Serie B, Greece, Portugal) |

Per family, PROMISING / UNPROVEN / OFF:

| Track | Market-side | Champion | A | B |
|---|---|---|---|---|
| NF | BTTS | 0/4/11 | 0/4/11 | 0/4/11 |
| NF | O1.5 | 0/9/6 | 0/10/5 | 0/8/7 |
| NF | O2.5 | 1/9/5 | 1/8/6 | 1/9/5 |
| NF | U2.5 | 0/10/5 | 1/8/6 | 1/11/3 |
| NF | O3.5 | 1/4/10 | 1/9/5 | 1/3/11 |
| Std | BTTS | 0/0/17 | 0/10/7 | 0/5/12 |
| Std | O1.5 | 0/4/13 | 1/8/8 | 1/6/10 |
| Std | O2.5 | 1/4/12 | 0/6/11 | 3/3/11 |
| Std | U2.5 | 0/0/17 | 0/1/16 | 0/1/16 |
| Std | O3.5 | 0/5/12 | 1/6/10 | 0/6/11 |

PROMISING cells:

| Cell | Champion | A | B | Fresh holdout (A) |
|---|---|---|---|---|
| NF O2.5 Finland | PROMISING | PROMISING (config now LEARNED/EV) | PROMISING | 46 bets, +12.5%, CI [-18.3%, +40.1%], CLV +0.28% |
| NF O3.5 Finland | PROMISING | PROMISING | PROMISING | 46, +8.1%, [-30.0%, +46.1%], CLV +0.06% |
| Std O2.5 La Liga 2 | PROMISING (selection) | **UNPROVEN** | **UNPROVEN** | - (selection basis) |
| NF U2.5 Argentina | OFF | PROMISING | PROMISING | 133, +0.2%, [-12.3%, +12.7%], CLV +0.95% |
| Std O1.5 Serie B | UNPROVEN | PROMISING | PROMISING | 61, +6.2%, [-9.2%, +21.7%], CLV +0.40% |
| Std O3.5 Eredivisie | UNPROVEN | PROMISING | UNPROVEN | **13 bets**, +9.4%, [-71.5%, +90.3%], CLV +3.5% |
| Std O2.5 Greece / Portugal / Serie B | UNPROVEN / OFF / UNPROVEN | OFF / UNPROVEN / UNPROVEN | PROMISING x3 (selection basis) | - |

### Combined Holm (protocol "multiplicity")

The family has 71 fresh-holdout cells: the champion's 19, A's 28 and B's 24. All 71 have a p-value. **0 PROVEN.** Every
combined Holm-adjusted p is 1.0. The smallest in-run p was the champion's NF U2.5 Japan (0.41), and it is 1.0 in the
combined family. The only fresh cell with >= 150 bets, A's Std O3.5 Serie B (157 bets), has ROI +3.2% with CI
[-30.4%, +36.9%]. Several New-Format cells give IDENTICAL fresh results in all three runs (e.g. NF O2.5 Denmark: 82 bets,
+2.1%). Their bars select the same bets whatever the model, so those results say nothing about the variants
(`holm_combined.json`).

## 3. Every cell that differs, and why

Price coverage is identical for every changed cell (the same priced rows for champion and variant), so **every change comes
from a change in model probability** (reason MODEL_PROB). The variants move probabilities much more than a refit does:
mean |dp| per family is 1.8-3.7 pp (p95 5-9 pp), against 0.3-0.7 pp between P9 and the approved bars. Same status but a
different bar or configuration: A 27 cells, B 21 (`status_diff_champion_vs_*.json`). Fresh-holdout figures in brackets
are the variant's own test on the fresh rows.

### Champion -> variant A: 42 status changes

| Cell | Champion -> A | Why | config A | mean abs dp |
|---|---|---|---|---|
| NF O1.5 Romanian Superliga | UNPROVEN -> OFF | no own estimate (selection keeps it off) | - | 1.7 pp |
| NF O2.5 Norway Eliteserien | UNPROVEN -> OFF | selection finds no positive bar | - | 4.5 pp |
| NF U2.5 Denmark Superliga | UNPROVEN -> OFF | selection finds no positive bar | - | 2.3 pp |
| NF U2.5 Norway Eliteserien | UNPROVEN -> OFF | selection finds no positive bar | - | 4.5 pp |
| NF U2.5 Saudi Pro League | UNPROVEN -> OFF | selection finds no positive bar | - | 2.6 pp |
| NF U2.5 USA MLS | UNPROVEN -> OFF | selection finds no positive bar | - | 2.0 pp |
| NF O3.5 Romanian Superliga | UNPROVEN -> OFF | no own estimate (selection keeps it off) | - | 1.1 pp |
| Std O2.5 Greek Super League | UNPROVEN -> OFF | selection finds no positive bar | - | 2.5 pp |
| Std O2.5 La Liga 2 | PROMISING -> UNPROVEN | selected on pre-holdout rows | RAW/EV/max3.5/min1.4 | 2.9 pp |
| NF O1.5 Ireland Premier Division | OFF -> UNPROVEN | tested on the fresh holdout (50 bets, ROI +2.2%, CLV -1.23%) | LEARNED/EDGE/max-/min- | 2.2 pp |
| NF O1.5 USA MLS | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.4 pp |
| NF U2.5 Argentina Primera Division | OFF -> PROMISING | tested on the fresh holdout (133 bets, ROI +0.2%, CLV +0.95%) | LEARNED/EDGE/max-/min1.4 | 2.9 pp |
| NF U2.5 Ireland Premier Division | OFF -> UNPROVEN | tested on the fresh holdout (65 bets, ROI -9.0%, CLV -1.71%) | LEARNED/EDGE/max-/min1.4 | 2.2 pp |
| NF U2.5 K-League 1 | OFF -> UNPROVEN | tested on the fresh holdout (81 bets, ROI +1.3%, CLV -0.44%) | LEARNED/EDGE/max-/min1.4 | 1.8 pp |
| NF O3.5 Austrian Bundesliga | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.4 pp |
| NF O3.5 Brazil Serie A | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.5 pp |
| NF O3.5 K-League 1 | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 1.9 pp |
| NF O3.5 Mexico Liga MX | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 1.8 pp |
| NF O3.5 Saudi Pro League | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.6 pp |
| NF O3.5 USA MLS | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 1.9 pp |
| Std BTTS Belgian First Division A | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.5 pp |
| Std BTTS Bundesliga 2 | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 5.9 pp |
| Std BTTS Championship | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.2 pp |
| Std BTTS Dutch Eredivisie | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 6.3 pp |
| Std BTTS La Liga 2 | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 5.2 pp |
| Std BTTS Portuguese Primeira Liga | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.2 pp |
| Std BTTS Scottish Championship | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.8 pp |
| Std BTTS Scottish League One | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.6 pp |
| Std BTTS Scottish Premiership | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.8 pp |
| Std BTTS Serie B | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.1 pp |
| Std O1.5 Bundesliga 2 | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.7 pp |
| Std O1.5 Dutch Eredivisie | OFF -> UNPROVEN | tested on the fresh holdout (25 bets, ROI +7.8%, CLV -1.43%) | RAW/EV/max-/min- | 3.5 pp |
| Std O1.5 La Liga 2 | OFF -> UNPROVEN | tested on the fresh holdout (18 bets, ROI +4.4%, CLV -1.92%) | RAW/EV/max-/min- | 4.1 pp |
| Std O1.5 League One | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.2 pp |
| Std O1.5 Scottish Championship | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 1.8 pp |
| Std O1.5 Serie B | UNPROVEN -> PROMISING | tested on the fresh holdout (61 bets, ROI +6.2%, CLV +0.40%) | RAW/EV/max-/min- | 3.0 pp |
| Std O2.5 Championship | OFF -> UNPROVEN | selected on pre-holdout rows | RAW/EV/max3.5/min1.4 | 1.9 pp |
| Std O2.5 Portuguese Primeira Liga | OFF -> UNPROVEN | selected on pre-holdout rows | RAW/EV/max3.5/min1.4 | 2.3 pp |
| Std U2.5 Dutch Eredivisie | OFF -> UNPROVEN | selected on pre-holdout rows | RAW/EDGE/max2.5/min- | 2.8 pp |
| Std O3.5 Dutch Eredivisie | UNPROVEN -> PROMISING | tested on the fresh holdout (13 bets, ROI +9.4%, CLV +3.54%) | RAW/EDGE/max-/min- | 3.5 pp |
| Std O3.5 League One | OFF -> UNPROVEN | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 1.8 pp |
| Std O3.5 Ligue 2 | OFF -> UNPROVEN | tested on the fresh holdout (16 bets, ROI -18.6%, CLV -7.02%) | RAW/EDGE/max-/min- | 2.6 pp |

### Champion -> variant B: 27 status changes

| Cell | Champion -> B | Why | config B | mean abs dp |
|---|---|---|---|---|
| NF O1.5 Romanian Superliga | UNPROVEN -> OFF | no own estimate (selection keeps it off) | - | 0.9 pp |
| NF O1.5 Saudi Pro League | UNPROVEN -> OFF | no own estimate (selection keeps it off) | - | 1.4 pp |
| NF U2.5 USA MLS | UNPROVEN -> OFF | selection finds no positive bar | - | 1.9 pp |
| NF O3.5 Romanian Superliga | UNPROVEN -> OFF | selection finds no positive bar | - | 1.0 pp |
| Std O2.5 La Liga 2 | PROMISING -> UNPROVEN | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 2.0 pp |
| Std O2.5 League One | UNPROVEN -> OFF | selection finds no positive bar | - | 1.6 pp |
| Std O3.5 Serie B | UNPROVEN -> OFF | no own estimate (selection keeps it off) | - | 2.1 pp |
| NF O1.5 Ireland Premier Division | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 1.6 pp |
| NF U2.5 Argentina Primera Division | OFF -> PROMISING | tested on the fresh holdout (133 bets, ROI +0.2%, CLV +0.95%) | LEARNED/EDGE/max-/min1.4 | 2.8 pp |
| NF U2.5 Ireland Premier Division | OFF -> UNPROVEN | tested on the fresh holdout (65 bets, ROI -9.0%, CLV -1.71%) | LEARNED/EDGE/max-/min1.4 | 1.3 pp |
| NF U2.5 K-League 1 | OFF -> UNPROVEN | tested on the fresh holdout (84 bets, ROI -1.9%, CLV -0.52%) | LEARNED/EDGE/max-/min1.4 | 1.5 pp |
| Std BTTS Belgian First Division A | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 4.1 pp |
| Std BTTS Dutch Eredivisie | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 8.0 pp |
| Std BTTS La Liga 2 | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 4.8 pp |
| Std BTTS Scottish Premiership | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 4.0 pp |
| Std BTTS Serie B | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 3.1 pp |
| Std O1.5 Dutch Eredivisie | OFF -> UNPROVEN | tested on the fresh holdout (28 bets, ROI -0.2%, CLV -0.41%) | RAW/EV/max-/min- | 3.5 pp |
| Std O1.5 La Liga 2 | OFF -> UNPROVEN | tested on the fresh holdout (93 bets, ROI -9.0%, CLV -1.43%) | RAW/EV/max-/min- | 3.3 pp |
| Std O1.5 League One | OFF -> UNPROVEN | switched on, no selection evidence | RAW/EDGE/max-/min- | 2.1 pp |
| Std O1.5 Serie B | UNPROVEN -> PROMISING | tested on the fresh holdout (81 bets, ROI +2.3%, CLV +0.27%) | RAW/EV/max-/min- | 2.4 pp |
| Std O2.5 Championship | OFF -> UNPROVEN | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 1.8 pp |
| Std O2.5 Greek Super League | UNPROVEN -> PROMISING | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 2.3 pp |
| Std O2.5 Portuguese Primeira Liga | OFF -> PROMISING | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 2.2 pp |
| Std O2.5 Serie B | UNPROVEN -> PROMISING | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 1.9 pp |
| Std U2.5 Scottish League Two | OFF -> UNPROVEN | selected on pre-holdout rows | RAW/EV/max2.5/min- | 1.5 pp |
| Std O3.5 League One | OFF -> UNPROVEN | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 1.9 pp |
| Std O3.5 Turkish Super Lig | OFF -> UNPROVEN | selected on pre-holdout rows | LEARNED/EDGE/max-/min- | 2.2 pp |


## 4. Expected TOP-3 tips per league per gameweek (v6 R2 rule, never padded) + calibration per league

Holdout 2026-04-01..2026-11-08, weeks with fixtures. C = clean champion, A = league + logit cal, B = logit cal. Statuses from each run's own proposal; source parameters and lam fitted on rows < 2026-04-01 only. "A v6 picks" = A's TOP-3 picks in the league on the holdout (bets, ROI; outcomes previously seen). Calibration columns: median within-league slope over the 4 markets, and mean |mean p - observed| (section 6). These are historical rates on the holdout's price coverage; from 2026-11-21 the live capture prices every served fixture, so NO_PRICE weeks should shrink.

| League | Wks | tips/wk C / A / B | A weeks 3/2/1/0 | A causes (< 3 tips) | A v6 picks (ROI) | cal slope C / A | mean abs gap C / A (pp) | lam C / A |
|---|---|---|---|---|---|---|---|---|
| NF Argentina Primera Division | 24 | 2.38 / 1.92 / 2.38 | 13/2/3/6 | NO_PRICE 8, NO_TIP_CLEARS_BAR 3 | 46 (-3.0%) | 1.15 / 0.96 | 2.0 / 2.8 | 0.07 / 1.00 |
| NF Austrian Bundesliga | 21 | 1.48 / 2.48 / 2.48 | 17/0/1/3 | NO_PRICE 4 | 52 (-20.1%) | 0.43 / 0.67 | 3.6 / 3.3 | 1.00 / 0.79 |
| NF Brazil Serie A | 25 | 2.76 / 2.64 / 2.72 | 19/4/1/1 | NO_PRICE 4, NO_TIP_CLEARS_BAR 2 | 66 (+1.0%) | -0.18 / -0.48 | 4.2 / 5.5 | 0.20 / 0.08 |
| NF China Super League | 25 | 0.00 / 0.00 / 0.00 | 0/0/0/25 | ALL_CELLS_OFF 25 | 0 | 0.54 / 0.33 | 3.9 / 2.3 | - / - |
| NF Denmark Superliga | 22 | 2.64 / 2.46 / 2.59 | 17/1/1/3 | NO_PRICE 4, NO_TIP_CLEARS_BAR 1 | 54 (+5.9%) | -0.84 / -0.56 | 4.9 / 2.8 | 0.37 / 1.00 |
| NF Finland Veikkausliiga | 28 | 2.68 / 2.68 / 2.64 | 25/0/0/3 | NO_PRICE 3 | 75 (-32.7%) | -0.29 / -0.57 | 5.6 / 7.1 | 0.94 / 0.98 |
| NF Ireland Premier Division | 28 | 0.00 / 2.25 / 2.25 | 18/2/5/3 | NO_PRICE 7, NO_TIP_CLEARS_BAR 3 | 63 (-8.1%) | 1.37 / 1.29 | 8.0 / 9.4 | - / 1.00 |
| NF Japan J-League | 21 | 1.38 / 2.10 / 2.10 | 10/7/0/4 | NO_PRICE 11 | 44 (-20.5%) | -0.96 / -0.46 | 3.1 / 4.0 | 0.40 / 1.00 |
| NF K-League 1 | 16 | 1.19 / 1.81 / 1.56 | 8/1/3/4 | NO_PRICE 8 | 29 (-20.7%) | -0.36 / 0.03 | 1.4 / 1.5 | 0.00 / 0.01 |
| NF Mexico Liga MX | 26 | 1.11 / 0.54 / 1.08 | 1/1/9/15 | NO_TIP_CLEARS_BAR 17, NO_PRICE 8 | 14 (+1.0%) | 0.16 / 0.15 | 3.2 / 2.5 | 0.83 / 0.73 |
| NF Norway Eliteserien | 22 | 2.32 / 2.18 / 2.32 | 14/2/2/4 | NO_PRICE 8 | 48 (-29.4%) | 1.76 / 1.44 | 4.0 / 3.3 | 0.33 / 0.97 |
| NF Romanian Superliga | 26 | 0.77 / 0.27 / 0.31 | 1/0/4/21 | NO_PRICE 18, NO_TIP_CLEARS_BAR 7 | 7 (+62.1%) | 1.13 / 0.95 | 3.2 / 2.6 | - / - |
| NF Saudi Pro League | 12 | 0.75 / 1.17 / 0.50 | 3/2/1/6 | NO_PRICE 7, NO_TIP_CLEARS_BAR 2 | 14 (+6.9%) | 0.66 / 0.79 | 2.5 / 3.3 | - / - |
| NF Sweden Allsvenskan | 26 | 2.12 / 1.31 / 1.58 | 6/5/6/9 | NO_TIP_CLEARS_BAR 11, NO_PRICE 9 | 34 (+14.6%) | 0.34 / 0.60 | 3.2 / 4.3 | 0.00 / 0.12 |
| NF USA MLS | 26 | 2.19 / 1.27 / 0.04 | 5/5/8/8 | NO_TIP_CLEARS_BAR 18, NO_PRICE 3 | 33 (-16.0%) | 0.51 / 0.50 | 6.1 / 4.5 | 0.00 / 0.06 |
| Std Belgian First Division A | 20 | 0.55 / 1.60 / 1.10 | 7/3/5/5 | NO_PRICE 13 | 32 (+7.2%) | 1.38 / 0.25 | 5.1 / 4.1 | 0.65 / 0.83 |
| Std Bundesliga 2 | 19 | 0.00 / 0.47 / 0.00 | 1/1/4/13 | NO_PRICE 10, NO_TIP_CLEARS_BAR 7, NO_MODEL 1 | 9 (+16.0%) | 0.22 / 0.32 | 1.0 / 2.8 | - / 0.01 |
| Std Championship | 19 | 0.47 / 0.63 / 0.79 | 0/3/6/10 | NO_TIP_CLEARS_BAR 14, NO_PRICE 5 | 12 (-3.2%) | -0.47 / -0.29 | 5.1 / 5.4 | 0.31 / 0.17 |
| Std Dutch Eredivisie | 18 | 2.50 / 1.28 / 0.56 | 4/3/5/6 | NO_TIP_CLEARS_BAR 13, NO_PRICE 1 | 23 (+8.9%) | 0.44 / -0.00 | 12.1 / 9.1 | 1.00 / 0.00 |
| Std Greek Super League | 18 | 0.72 / 0.00 / 2.06 | 0/0/0/18 | ALL_CELLS_OFF 18 | 0 | 0.36 / 0.64 | 1.4 / 2.3 | 0.10 / - |
| Std La Liga 2 | 25 | 0.68 / 0.96 / 2.84 | 5/3/3/14 | NO_TIP_CLEARS_BAR 18, NO_PRICE 2 | 24 (+38.0%) | 1.80 / 0.87 | 2.3 / 4.8 | 0.23 / 0.01 |
| Std League One | 21 | 0.29 / 1.76 / 1.33 | 8/4/5/4 | NO_TIP_CLEARS_BAR 8, NO_PRICE 5 | 37 (-8.6%) | 1.70 / 0.71 | 1.3 / 1.6 | 0.00 / 0.00 |
| Std League Two | 21 | 0.00 / 0.00 / 0.00 | 0/0/0/21 | ALL_CELLS_OFF 21 | 0 | 0.47 / 0.20 | 2.2 / 2.9 | - / - |
| Std Ligue 2 | 20 | 0.75 / 0.25 / 1.45 | 1/0/2/17 | NO_TIP_CLEARS_BAR 16, NO_PRICE 2, NO_MODEL 1 | 5 (-45.8%) | 1.46 / 0.59 | 2.7 / 3.9 | 0.18 / 0.13 |
| Std National League | 18 | 0.00 / 0.00 / 0.00 | 0/0/0/18 | ALL_CELLS_OFF 18 | 0 | 2.03 / 1.11 | 5.0 / 3.8 | - / - |
| Std Portuguese Primeira Liga | 21 | 0.00 / 0.95 / 2.29 | 6/0/2/13 | NO_PRICE 12, NO_TIP_CLEARS_BAR 2, NO_MODEL 1 | 20 (-5.0%) | 1.89 / 0.83 | 3.1 / 3.0 | - / 0.00 |
| Std Scottish Championship | 20 | 0.00 / 0.55 / 0.00 | 2/1/3/14 | NO_PRICE 17, NO_TIP_CLEARS_BAR 1 | 11 (+9.2%) | -1.15 / -0.42 | 8.8 / 7.3 | - / - |
| Std Scottish League One | 20 | 0.00 / 0.15 / 0.00 | 0/1/1/18 | NO_PRICE 19, NO_TIP_CLEARS_BAR 1 | 3 (+17.7%) | -1.91 / -0.39 | 4.9 / 5.7 | - / - |
| Std Scottish League Two | 19 | 0.00 / 0.00 / 0.95 | 0/0/0/19 | ALL_CELLS_OFF 19 | 0 | -1.86 / -1.44 | 5.5 / 4.9 | - / - |
| Std Scottish Premiership | 19 | 0.00 / 0.58 / 0.26 | 1/2/4/12 | NO_PRICE 15, NO_TIP_CLEARS_BAR 3 | 11 (+9.6%) | 2.63 / 0.54 | 4.6 / 2.9 | - / - |
| Std Serie B | 18 | 1.50 / 1.89 / 2.44 | 9/2/3/4 | NO_TIP_CLEARS_BAR 6, NO_PRICE 3 | 34 (+22.9%) | 0.59 / 0.47 | 4.0 / 3.7 | 0.73 / 0.51 |
| Std Turkish Super Lig | 18 | 0.00 / 0.00 / 0.00 | 0/0/0/18 | ALL_CELLS_OFF 18 | 0 | 2.75 / 1.33 | 4.6 / 4.0 | - / - |

## 5. Share of league-weeks with 0 / 1 / 2 / 3 tips, and why (v6 R2, holdout, 32 leagues)

Rule: tips that clear their cell's bars, TOP-3 by R2 per league-week, one direction per fixture, never padded.
LEAGUE_WIDE cells (Argentina NF BTTS) are the league's own list and stay outside the TOP-3, as in P9. All three columns
come from the same code.

| League-weeks with fixtures (682) | 3 tips | 2 | 1 | 0 |
|---|---|---|---|---|
| Clean champion | 173 (25.4%) | 54 (7.9%) | 75 (11.0%) | 380 (55.7%) |
| **A** | **201 (29.5%)** | **55 (8.1%)** | **87 (12.8%)** | **339 (49.7%)** |
| B | 233 (34.2%) | 62 (9.1%) | 79 (11.6%) | 308 (45.2%) |

| Cause of < 3 tips (weeks with fixtures) | Champion | A | B |
|---|---|---|---|
| ALL_CELLS_OFF (every cell of the league OFF) | 207 | 119 | 123 |
| NO_TIP_CLEARS_BAR (priced, but no row clears the cell's bar) | 170 | 153 | 172 |
| NO_PRICE (no BET-role two-sided price) | 126 | 206 | 149 |
| NO_MODEL (blind fixture, no rolling form) | 3 | 3 | 3 |
| RULE_CONFLICT (3+ tip slots, < 3 after distinct-fixture / one-direction) | 3 | 0 | 2 |
| **TOP-3 v6 R2 picks on the holdout** | 702, hit 50.0%, **-8.5%** [-17.8%, +1.0%], CLV -0.35% | 800, hit 53.5%, **-5.4%** [-13.4%, +2.7%], CLV +0.44% | 902, hit 52.8%, **-5.9%** [-12.6%, +1.0%], CLV -0.20% |

Slots (fixture x non-OFF model cell): champion 8,637 (3,086 without a price), A 13,008 (4,846 without a price), B 10,465
(3,783 without a price). A switches on cells in leagues and markets with thin BT coverage (Standard BTTS, Scottish leagues,
NF O3.5), so NO_PRICE grows while ALL_CELLS_OFF shrinks. The P9 fix list (live capture of every served league x market
from 2026-11-21, BT for Portugal, BTTS history) applies unchanged.

Note: P9's shortfall table (85.8% zero-tip weeks) is for the v7 three-tier rule; the figures above are for the v6 R2 live
rule, as the owner asked here.

## 6. Calibration slope and league-level gap

Holdout rows, deployable, out-of-sample (`slopes.json`; per league x market in the file, per league in the table in
section 4 = the median slope over the league's 4 markets and the mean |gap|).

| Measure | Champion | A (league + logit cal) | B (logit cal) |
|---|---|---|---|
| Std: league x market cells with the within-league slope closer to 1 than the champion | - | 49 of 68 | 47 of 68 |
| Std: median \|within-league slope - 1\| | 1.49 | 0.74 | 0.74 |
| NF: cells closer to 1 | - | 37 of 60 | 39 of 60 |
| NF: median \|within-league slope - 1\| | 0.86 | 0.82 | 0.81 |
| Std: mean \|league-level gap\| (mean p - observed) | 4.35 pp | 4.24 pp | 4.07 pp |
| NF: mean \|league-level gap\| | 3.94 pp | 3.94 pp | 3.64 pp |
| v6 R2 lam vs the fair price, median over cells (P9: 0.26) | 0.26 (38 cells) | 0.16 (52 cells; 14 at <= 0.05) | 0.57 (47 cells) |

Cross-league level slope (1 = each league's own level carried, > 1 = flattened, < 1 = over-spread), holdout:

| Track | Market | Champion | A | B |
|---|---|---|---|---|
| Std | BTTS | 4.46 | 0.43 | 0.45 |
| Std | O1.5 | 2.39 | 0.94 | 0.97 |
| Std | O2.5 | 3.34 | 1.96 | 1.83 |
| Std | O3.5 | 3.45 | 1.22 | 1.25 |
| NF | BTTS | 1.63 | 0.73 | 0.91 |
| NF | O1.5 | 1.96 | 1.39 | 1.64 |
| NF | O2.5 | 1.06 | 0.83 | 0.90 |
| NF | O3.5 | 1.14 | 0.90 | 0.91 |

Reading: the logit calibration (alone or with the league block) un-flattens the Standard league levels and brings the
Standard within-league slopes toward 1. It overshoots on BTTS (now over-spread) and does little on New-Format. Better
calibration of the probability does not, on its own, create a priced edge over the market: A's lam (how much of the model's
departure from the fair price survives on development rows) is lower than the champion's (median 0.16 vs 0.26).
Per-league holdout samples are 150-300 rows per market, so single-league slopes are noisy (negative slopes appear in both
models).

## 7. Checks before push

| Check | Result |
|---|---|
| `python -m src.integrity.data_truth --gate` | **STOPLIGHT GREEN** (rules 1, 2, 4, 5, 6, 7, 8, 10a-c GREEN; rule 9 NO_DATA - Clean Record start not set). Rule 3 (operational, not gated) RED as in P9: upcoming board fixtures 100% missing `has_ht_history` (72 Std / 102 NF) - a live-board item, not training data |
| `python scripts/dashboard_manifest.py --check` | exit 0: missing code [], missing data [] |
| `python -m pytest -q -p no:cacheprovider tests` | 3,444 passed, 44 skipped, 6 xfailed, **5 failed**. All 5 fail identically on the base `bars/v5-clean-models` 5f481d3b3 (checked), so none comes from this branch: `test_research_lock` x2 (the research-lock snapshot predates the owner-approved 2026-11-15 policy / threshold changes; re-signing it needs the owner), `test_team_frame_health` x1 and `test_threshold_proposals` x2 (the same approved-state drift). This branch changes no config, no served gate and no code path that they cover |

## 8. Proposal (owner decides; nothing applied)

* **TP-2026-11-16-9cef8a02** (`output/threshold_proposals/bars_oddsapi_hist/2026-11_9cef8a02.json`, proposal hash
  `9cef8a02c4585781270549f83c4d2405b96a2f12811b8e261aa16ec455b1415c`, 77 entries, AWAITING_OWNER), evidence
  `evidence_2026-11_p10leaguecal.json`, fresh verdicts `TP-FH-2026-11-16-dbf7a6eb`. Built from variant A by registration.
* Variant B: evidence `evidence_2026-11_p10logitcal.json` and fresh verdicts `TP-FH-2026-11-16-57023c95` are kept as
  research. The study tool also wrote a B proposal (TP-2026-11-16-8b1edfd0). The protocol allows only one proposal, from
  A, so that file and its candidates were deleted before commit.
* `latest.json` still points at TP-2026-11-15-bade2c0b (the tool's rewrite was discarded).
* **Recommendation: do not approve TP-2026-11-16-9cef8a02 for staking.** It proves nothing more than the champion (0 PROVEN
  in both), and it would swap a staked PROMISING cell (La Liga 2 O2.5) for three thin ones. If the owner wants the variant's
  larger UNPROVEN coverage for the PAPER track only, that is a separate decision. The honest test of A vs the champion is
  the live Clean Record from 2026-11-21, if the owner wants the variant run in paper beside the champion.

## Reproduce

Runs: `C:\Users\nevo\dev\v5lc-runs` (dry run `dryrun.log` / `dryrun.json`, parity, `verify_oos.py`, `pipeline.sh`, `all.sh`);
bars work dir `C:\Users\nevo\dev\bars-oddsapi-local\v5lc\{A,B}` (u, priced, own_bettime, selstudy, study) and `top3_*`
(local, prices private). Statistics committed under `output/research/bars_league_calib_p10/`. Script
`scripts/bars_league_calib_p10.py` (top3 / slopes / holm). Store `C:\Users\nevo\dev\exec-store` at 80f9de3.
