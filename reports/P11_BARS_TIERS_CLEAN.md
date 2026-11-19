# P11: best bars and T1 / T2 / T3 on the clean models

**Proposal for the owner's decision on 2026-11-20.**

On 2026-11-18 the owner asked: "I want the best optimized bars and T1, T2, T3 best that we can achieve right now on the 100% truth data retrained models."

Nothing in this report is approved, installed or switched on.

The work is on branch `bars/v6-tiers-clean-1118`:

| Item | Where | Status |
|---|---|---|
| Bars proposal TP-2026-11-18-42a409c5 | `output/threshold_proposals/bars_p11/2026-11.json`, hash 42a409c5…a297b9, 54 entries | Awaiting the owner |
| Evidence for the bars | `evidence_2026-11_p11.json` | — |
| Tier config | `config/top3_tiers_p11.json` | Proposal |
| Protocol | `config/p11_bars_tiers_protocol.json` | Committed in efc32cc94, before any fit |

The fit was committed in ae1b124d5 before the single validation run.

## In plain words

**What changes.** v6 takes the 3 tips with the highest calibrated value in each league each gameweek. P11 uses the same bars, but fills three named slots:

- **T1, best value:** the tip with the highest calibrated value.
- **T2, safest:** among the remaining value tips, the one with the highest calibrated chance of winning.
- **T3, risky value:** the longest-odds value tip, or the best value tip above that league's odds floor.

An empty slot stays empty. Slots are never padded.

**Validation (1 Sep – 8 Nov):** +5.6% over 128 bets.
- Hit rate 53%.
- 95% confidence interval −12.7% to +23.8%.

On the same matches:
- v6 lost 3.5% with the P11 bars, and 2.2% with bade2c0b.
- P9's v7 lost 19.4%.

**This is not proof.** No tier, league or cell is PROVEN. The proof will come from the live tips placed from 21.11.

**The gap is coverage.** Only 12 leagues produce tips. Of 257 league-weeks, 29 had 3 tips and 200 had none. Most empty weeks are leagues where the bars found no value (139 league-weeks).

## 1. Inputs

**Models**
- The champion_recipe holdout harness at commit 694995f90.
- Data frame 9ed237ebc04aee09, the same as refit_1118b.
- No leakage: 0 out-of-sample violations across the 12 track × market sets.

**Prices.** Only prices that existed at bet time are used:
- our own bet-time captures, with the backfill stripped out;
- football-data pre-match prices, through the certified join;
- OddsAPI history, sets BT, BTU and OUU, taken 125 hours before kickoff for Standard and 153 hours for New Format. Bet365 is used first, then Pinnacle.

The price join was checked on 36,663 history rows, with 0 failures.

**Time windows**
- Fit: everything before 2026-09-01.
- Validation: 09-01 to 11-08, run once.
- P9 had already looked at those outcomes, so the validation window counts as "outcomes previously seen".

**Coverage changes**
- Portugal now has prices.
- BTTS cannot be tested. Every bet-time BTTS quote in the history comes from William Hill, 1xBet, Ladbrokes or similar books. None comes from Bet365 or Pinnacle, so the book rule drops them all. See decision O1.

## 2. Method

**Bars.** The study code (`src/bars_opt/study.py`) is unchanged:
- each league gets its own bars;
- the configuration for each family is chosen by inner validation;
- the final bars are scored once on the validation window;
- Holm correction runs over a family of 15 tests.

**Tiers**
- **Pool:** tips from model cells that clear their own bar. League-wide cells stay on their own line.
- **Calibration:** the calibrated probability is fair + λ × (model probability − fair). λ is fitted only on rows from before the period being scored.
- **Per-league choices.** Each league picks two settings on its fit data:
  - the gate: CAL (calibrated value > 0) or BAR (the cell's own value > 0, with calibration used only for ranking);
  - the T3 odds floor: REL, 2.25, 2.5, 2.75 or 3.0. REL means T3 is simply the longest-odds tip left.

  Each league's settings are chosen by mean minus one standard error, which needs at least 40 fit picks. A league without that many gets CAL with REL and is labelled UNFITTED.

**Why v7 had so many empty weeks, and how P11 fixes each cause**

| v7 problem | P11 fix |
|---|---|
| Required calibrated value > 0 in every league. The small calibration slopes removed most tips. | Each league chooses CAL or BAR. 5 of 14 chose BAR. |
| Had a T3 threshold in only 2 leagues. | The REL rule gives every league a T3. |
| Portugal had no prices. | Portugal is now priced. |

## 3. Results

**By tier**

| Tier | Fit, inner out-of-sample | Apr–Aug | Validation, 1 Sep – 8 Nov |
|---|---|---|---|
| T1 | 550 bets, +3.8% | 107 bets, −1.3% | 57 bets, hit 54.4%, +11.7% [−16.6%, +40.0%], average odds 2.21, CLV +1.1% |
| T2 | 420 bets, −1.5% | 81 bets, +16.9% | 42 bets, hit 57.1%, +5.1% [−28.7%, +38.9%], average odds 1.90, CLV +0.2% |
| T3 | 300 bets, −5.4% | 60 bets, +11.5% | 29 bets, hit 44.8%, −5.8% [−46.3%, +34.7%], average odds 2.33, CLV +1.4% |
| All tiers | 1,270 bets, −0.2% | 248 bets, +7.7% | 128 bets, hit 53.1%, +5.6% [−12.7%, +23.8%], CLV +0.9% |

**Other rules on the same validation rows**
- v6 with the P11 bars: 157 bets, −3.5% [−20.6%, +13.5%].
- v6 with the approved bade2c0b bars: 341 bets, −2.2% [−14.5%, +10.1%]. Its cells were fitted on data that includes the validation months, which favours it slightly.
- P9's v7: 181 bets, −19.4%.

**PROVEN and PROMISING counts**

| | PROVEN | PROMISING | UNPROVEN | OFF |
|---|---|---|---|---|
| Bars study | 0 (Holm p = 1.0 on every tested cell) | 5 | 14 | 175 |
| Serving proposal | 0 | 5 | 27 | 162 |
| Approved bade2c0b, for comparison | 0 | 5 | 45 | 144 |

- No tier is PROVEN either.
- The 5 PROMISING cells:
  - New Format O1.5 Argentina;
  - New Format U2.5 Japan;
  - New Format O3.5 Norway;
  - Standard O1.5 Serie B;
  - Standard O2.5 La Liga 2.
- Compared with bade2c0b:
  - 23 cells are on in both;
  - 27 are on only in bade2c0b;
  - 9 are on only in P11.
- Three cells that were PROMISING in bade2c0b are not PROMISING in P11:
  - New Format O2.5 Finland;
  - New Format O3.5 Finland;
  - New Format U2.5 Austria.

**By league, validation window**

Tips per gameweek counts only weeks with fixtures. Each tier column shows bets, then ROI.

| League | Gate; T3 rule | Tips per GW | T1 | T2 | T3 | All | Fit |
|---|---|---|---|---|---|---|---|
| NF Argentina | BAR; REL | 2.4 | 9, +4% | 8, −6% | 7, +40% | 24, +11% | 108, +9% |
| NF Austria | CAL; REL | 2.38 | 7, −63% | 6, −63% | 6, +32% | 19, −33% | 204, +4% |
| NF Norway | CAL; REL | 2.12 | 7, +48% | 6, +24% | 4, +90% | 17, +49% | 97, −13% |
| NF Finland | CAL; REL | 2.0 | 6, +20% | 5, +45% | 3, −100% | 14, +3% | 142, −1% |
| NF Japan | BAR; REL | 1.75 | 5, −5% | 5, +37% | 4, −49% | 14, −3% | 158, 0% |
| NF Denmark | CAL; floor 2.25 | 1.62 | 6, −26% | 4, −37% | 3, −100% | 13, −47% | 156, −2% |
| NF China | CAL; REL | 1.0 | | | | 7, +27% | |
| Std Serie B | CAL; REL | 0.75 | | | | 6, +1% | 110, +11% |
| NF Brazil | BAR; REL | 0.67 | | | | 6, +17% | |
| Std La Liga 2 | BAR; REL | 0.5 | | | | 5, +60% | |
| NF Saudi | CAL; REL | 2.0 (one week only) | | | | 2 bets | |
| Std Greece | BAR; REL | 0.12 | | | | 1 bet | |

- Every other league had 0 tips.
- Seven leagues were switched on only by the serving refit: Ireland, Sweden, USA, Romania, Saudi, Belgium and League One. They get the default CAL / REL rule and are labelled NO_FIT_POOL (not validated).

**Leagues where v6 with bade2c0b tipped but P11 does not**

| League | Bets | ROI |
|---|---|---|
| Mexico | 24 | +7% |
| Sweden | 24 | −58% |
| USA | 27 | +3% |
| Romania | 9 | +18% |
| Championship | 22 | −10% |
| Eredivisie | 21 | +36% |
| Portugal | 21 | +8% |
| Ligue 2 | 4 | +5% |
| Belgium | 3 | +32% |

## 4. Why most league-weeks have fewer than 3 tips

| Cause | League-weeks | What it means and what fixes it |
|---|---|---|
| All cells off | 139 | No edge was measured in these leagues. Only live Clean Record history can change this. Never pad. |
| No value tip | 49 | Genuine. |
| Too few value tips | 27 | Genuine. |
| No price | 12 | 295 of 1,423 slots had no bet-time price. Live capture must cover O1.5, O3.5 and U2.5 for every cell that is on; decision O1 also helps. |
| T3 empty | 1 | — |

The leagues with every cell off are Bundesliga 2, Championship, Eredivisie, League Two, Portugal, the 4 Scottish leagues, Turkey, Mexico, Sweden, USA, Ireland, Romania and Belgium.

## 5. Owner decisions

- **O1: BTTS prices.** Accept William Hill, then 1xBet or Ladbrokes, as the BTTS price when neither Bet365 nor Pinnacle quotes. The data is already bought. This needs a protocol amendment and a re-run of about 30 minutes.
- **O2: adopt P11.** Approve TP-2026-11-18-42a409c5 and the tier config in place of bade2c0b and v6. Stakes stay at 0.5% flat, placed by hand.
- **O3: props.** Not included yet. Props need the clean props commit and the props holdout predictions.

## 6. Install steps (after approval)

1. Merge the branch. Then run `pytest tests/test_top3_tiers_p11.py tests/test_top3_league.py`.
   - Tier mode turns on only when `config/top3_tiers_p11.json` has status APPROVED and `owner_approved_at` set. Until then v6 runs unchanged.
2. Record the approval, by id and hash, in `config/threshold_approvals.json`.
3. Apply the bars. Do not apply bade2c0b.
   - Dry run: `python -m src.threshold_apply --proposal output/threshold_proposals/bars_p11/2026-11.json --dry-run`
   - Then run the same command without `--dry-run`.
   - Commit `models/best_params_*.json`, `config/threshold_applied.json` and `config/research_lock_snapshot.json`.
   - If the production parameters differ on the go-live branch, rebuild with `p11_bars_tiers.py propose` and re-approve the new id.
4. Set the tier config status to APPROVED, fill `owner_approved_at`, and commit. Live TOP-3 then switches to T1 / T2 / T3.
5. Leave U3c unchanged. Add the new id to `approved_proposals` in `config/top3_paper_tracks.json`, for U3e and U3f.
6. U3d needs no extra step.
7. **Rollback:**
   - revert the tier config commit to return to v6;
   - revert the apply commit to restore the previous bars.

## Reproduce

1. Run `scripts/bars_opt.py universe` for both tracks.
2. Run `scripts/p11_bars_tiers.py` with `prepare`, then `bars`, then `tiers-fit`.
3. Run `validate`, once only.
4. Run `propose`.

The runs are in `C:\Users\nevo\dev\p11-runs`. Local work, which includes prices and is never published, is in `C:\Users\nevo\dev\p11-work`.
