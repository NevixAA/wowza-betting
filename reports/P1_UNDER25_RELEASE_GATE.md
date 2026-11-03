# P1 — New-format Under 2.5 release gate

Phase 2, part 5. Independent QA audit, 2026-11-03. Branch `phase2/under` (from `origin/data-truth-100` e07a4591).
The NF Under state was **not changed** anywhere. Nothing was sent, placed, retrained or backtested.

## Verdict

**The bankroll policy is safe, but other routes still treated NF Unders as bets. Three are now fixed and two need your decision.**

1. The policy really does give NF Unders zero stake, on both `main` and `data-truth-100`. I ran the real sizing code on a test board to prove it, and the same board gets a stake when only the track is switched back on.
2. **The v9 Cloudbet bot ignored the policy.** It would have staked an NF Under at 4.5% of the bankroll (45.0 on a 1,000 test bankroll, on `main`). No workflow runs it, so it only runs when someone starts it by hand. Fixed on this branch.
3. **Telegram and the dashboard Tips table presented NF Unders as normal tips.** Telegram sent them as "BIG GAP TIP" with no paper label. The dashboard showed "equal stake" next to "PAPER". Both fixed on this branch.
4. **Still open (your decision):** the "Staked" headline figures and top picks still include NF Unders.
5. **Why Unders lost:** the NF model under-rates goals by about 10.6 points, and the bigger the edge it claims, the worse the result. Every settled tip was predicted before the 2026-11-01 and 2026-11-02 serving fixes, so whether the P0-1 fix helps is **UNIDENTIFIABLE** today (0 settled fixtures since the fix). This does not show that Unders can never work.

| Check | main 12bd71a3 | data-truth-100 e07a4591 | phase2/under (this branch) |
|---|---|---|---|
| Policy file: `new_format\|OU25_UNDER` = PAPER, base_fraction 0.0 | PASS | PASS | PASS |
| `stake_plan.write` real path (evaluate_value, then generate_bets, then sizing): NF Under SNIPER gets 0, reason `NO_REAL_STAKE_PAPER`; NF Over gets 0.005 (control) | PASS | PASS | PASS |
| Same board with the track flipped to ACTIVE_LOW in memory: Under is staked (proves the check can fail) | PASS (0.005) | PASS | PASS |
| `size_day`: 27 Under variants (3 side spellings x 3 tiers x 3 prices) all zero | n/a | PASS | PASS |
| v9 Cloudbet bot (`src/cloudbet/bot.py`, manual, DRY default) refuses an NF Under | **FAIL** (stakes 45.0) | **FAIL** | **FIXED** |
| Telegram instant tip says PAPER / not staked | **FAIL** | **FAIL** | **FIXED** |
| Dashboard Tips table "Stake note" | **FAIL** ("equal stake") | **FAIL** | **FIXED** ("paper track · not staked") |
| Dashboard free tier offers only staked tips | PASS | PASS | PASS |
| Ledger still records NF Under tips (forward tracking) | PASS (57 rows for matches on or after 11-02) | PASS | PASS (test) |
| Track registry, research lock and policy agree (13 NF Under entries, all PAPER, real_money false) | PASS | PASS | PASS |
| wowza-exec can place real money on an NF Under | No (PAPER_ONLY, real caps 0) | No | n/a (exec not changed) |
| "Staked" P/L headlines (Home, daily pack, Telegram summaries) leave out NF Unders since 11-02 | **No** (open P1) | **No** | **No** (open P1) |
| Top picks lists leave out NF Unders | **No** (open P2) | **No** | **No** (open P2) |

Sources: `tests/test_p2_nf_under_paper_e2e.py` (13 tests). Before the fixes, the 3 FAIL rows failed and the other 9 tests passed. After the fixes, all 13 pass. For main, I ran the same scenario on the 12bd71a3 code with a scratch script, because main has no `betting.league_threshold_evidence`, so the test fixture cannot patch it there.

## Owner actions

| # | Priority | Issue | What you decide |
|---|---|---|---|
| 1 | P1 | The v9 Cloudbet bot would stake NF Unders (and every standard-model tip, all PAPER) at its flat 4.5%. No workflow runs it. | Merge `phase2/under` (fix plus test) to main after review. |
| 2 | P1 | "Staked" totals count NF Under SNIPER/MARKSMAN rows settled after 2026-11-02 as staked, because THE staked definition (`dashboard_data/today_view.py`, owner 2026-11-02) is based on tier only. This covers Home "upcoming staked O/U", the daily pack "Staked" line and the Telegram summaries. | Should "staked" also need a real-money track on the match date? For example, NF Unders count as staked up to 2026-11-01 and as paper from 2026-11-02. That changes headline numbers, so it is your call. |
| 3 | P1 | Telegram instant tips: the label now appears on **every** tip whose track is not real money. That includes all standard-model O/U tips (standard tracks have been PAPER since before this audit), not only NF Unders. | Confirm the wider label, or limit it to NF Unders. |
| 4 | P2 | Top picks (`src/top_picks.py`) can list NF Unders under "Best bets / Safest / Value", the "5 tips a person should take". wowza-exec also uses these lists as a ranking tie-break. | Leave NF Unders out of the lists (still learn from them), or label them paper. |
| 5 | P2 | wowza-exec sets the state per league x market x model, with no **side**. Its DRY rehearsal (`cloudbet_shadow.yml`, every 3 h) can size NF Unders at the simulation caps. If `new_format/ou25` is ever moved to a money state, Unders would go with Overs. Based on reading the code; I did not run it. | Add a side to exec cells (or read v9's track state) before any exec cell leaves PAPER. |
| 6 | P3 | 7 fixtures hold both an NF Under and an NF Over tip in `bets_ledger.csv` (older rows from before the flip guard). | None needed for the gate; leave them to the settlement round. |

## Fixes made (each one had a failing test first)

| File | Change | Test |
|---|---|---|
| `src/cloudbet/bot.py` | `_drop_without_real_track`: a candidate is placed only if its `bankroll_policy.track_key(model_type, market, side)` track is in REAL_STATES with base_fraction > 0. If the policy can't be read, nothing is placed. | `test_cloudbet_bot_never_stakes_nf_under` (failed, now passes) + control `..._stakes_nf_under_when_track_is_real` |
| `telegram_bot/notifier.py` | `_paper_track_note`: an instant O/U tip on a non-real track gets "📄 PAPER — not staked (track …: PAPER)". The tip is still sent, as the owner rule says ("shown and recorded"). | `test_telegram_instant_tip_labels_nf_under_as_paper` (failed, now passes) + control `..._label_follows_the_policy_not_the_side` |
| `dashboard_data/tips_view.py` | `stake_label(tier, market, production_state)`: PAPER / BLOCKED / UNSUPPORTED becomes "paper track · not staked". The old 2-argument call is unchanged (the existing test still passes). | `test_dashboard_tips_table_shows_nf_under_as_paper` (failed, now passes) |

Affected existing tests pass: `test_cloudbet_bot`, `test_cloudbet_selection`, `test_home_tips_values`, `test_telegram_quiet`, `test_tier_labels`, `test_first_tier_stands`, `test_drift_tier_removed`, `test_data_truth_delivery`, `test_ou25_side_tracks`, `test_track_tier_multipliers`. That is 203 + 32 passed. I did not run the full suite (laptop rule). The dashboard repo picks up the Tips fix only after main is merged, because it syncs hourly from v9 main.

## Why NF Unders underperformed

**Inputs.** `output/bets_ledger.csv` and `output/v91_shadow.csv` at **main 12bd71a3** (the most recent committed results), plus the sidecar `output/corrections/clv_first_half_contaminated.csv` at **data-truth-100 e07a4591**. Script: `scripts/p2_under_analysis.py`. Every number is in `reports/p2_under_data/under_analysis.json`.

**Method.** Flat 1 unit per tip. Source `live`, model `new_format`, match date on or after 2026-08-10. Hit-rate CIs are Wilson 95%. ROI, CLV, bias and log-loss CIs use a 10,000-resample bootstrap (seed 20261103). No value is imputed.

**How the model's probability was derived.** The model probability on a tip is `1/odds + edge`. This is the pipeline's own definition: `src/betting.py:297-298` computes edge against the raw implied probability, without removing the bookmaker margin.

### The record

| NF Under tips (settled) | n | Hit (95% CI) | Break-even hit | Model said | Units | ROI (95% CI) |
|---|---|---|---|---|---|---|
| All tiers | 626 | 40.6% (36.8–44.5) | 42.9% | 51.7% | −41.40 | −6.6% (−15.7 to +2.4) |
| Staked tiers (SNIPER + MARKSMAN) | 412 | 37.1% (32.6–41.9) | 41.6% | 52.7% | −51.01 | −12.4% (−23.3 to −1.0) |
| Staked tiers, without paper league USA MLS | 292 | 39.7% (34.3–45.4) | 43.0% | 54.0% | −26.72 | −9.2% (−22.3 to +4.2) |
| Staked tiers, USA MLS only (paper since 10-07) | 120 | 30.8% | 38.1% | 49.4% | −24.29 | −20.2% (−41.7 to +1.8) |
| SNIPER | 165 | 32.7% | 40.1% | 55.4% | −31.10 | −18.9% (−37.0 to −0.0) |
| MARKSMAN | 247 | 40.1% | 42.6% | 50.9% | −19.91 | −8.1% (−22.1 to +6.5) |
| VALUABLE (paper tier) | 214 | 47.2% | 45.3% | 49.8% | +9.61 | +4.5% (−10.6 to +19.4) |
| *Control: NF Over, all tiers* | 155 | 45.8% | 42.7% | 48.0% | +9.99 | +6.5% (−12.2 to +25.5) |

**Cross-checks.**
- Booked P/L (−41.40u) equals P/L recomputed from odds and result (−41.40u). 0 rows differ.
- The policy's "−51u staked since 2026-08-10" figure is reproduced independently (−51.01u, n 412).

### What explains it

1. **Probability bias (main cause).** This uses all 491 settled NF fixtures in the shadow log, tipped or not. Each fixture's first-sight prediction is kept.
   - Actual over-2.5 rate: **58.5%** (54.0–62.7). The model said **47.9%**, a bias of **−10.6 pp** (−14.8 to −6.2). The market said 60.5% (its implied probability, margin not removed).
   - Model log loss is 0.6927 against the market's 0.6588. The difference is +0.034 (95% CI +0.007 to +0.060), so **the model is worse than the market**. Brier score: 0.2498 against 0.2333.
   - The model leans Under on 61.3% of fixtures; the market does on 16.7%.
   - On the tipped Unders, the model's P(under) exceeded the actual hit rate by **+11.1 pp** (7.3–14.9).
2. **Calibration by band.** The model under-rates overs in each probability band from 0.40 to 0.55:
   - 0.40–0.45: −10.8 pp
   - 0.45–0.50: −12.3 pp
   - 0.50–0.55: −15.0 pp
   - Every one of these CIs excludes 0.
3. **The edge points the wrong way (odds selection).** The bigger the claimed Under edge, the worse the result:
   - Edge ≥ 12%: hit 33.7%, while the model said 55.8%, ROI −16.9%.
   - Rank correlation between edge and winning: −0.086.
   - The tier ladder is inverted: SNIPER is worst and VALUABLE is best. A large "edge" on an Under mostly measures how far the model sits below the market on goals, which is the bias in point 1, not real value.
4. **Price bands.** The loss sits in long-priced Unders:
   - Odds ≥ 2.50: 215 tips, hit 28.8% against 35.6% needed, −42.18u, ROI −19.6% (−36.5 to −1.9).
   - 1.80–2.00 (+9.3%, n 78) and 2.20–2.50 (+3.1%, n 201): CIs include 0.
   - At ≥ 2.50 the market strongly expects goals and the model disagrees most.
5. **Leagues.** Model bias by league (shadow log, leagues with n ≥ 20):
   - Japan J-League: −21.6 pp (−34.0 to −8.7).
   - USA MLS: −16.9 pp (−25.9 to −7.5).
   - Norway: −15.3 pp (−30.2 to +0.4).
   - Argentina (−4.7) and Brazil (−0.9): close to calibrated.
   - Finland Under tips: +35.9% ROI (4.6–66.8, n 38). This is one of 11 leagues tested, so it is **not** evidence after allowing for multiple comparisons.
6. **CLV.** Mean −0.77% (−1.37 to −0.18), n 618, 34.5% positive. Prices moved slightly against our Unders after tipping. This is small next to the probability bias. 0 rows were affected by the first-half-price sidecar (all 127 of its NF Under rows predate 2026-08-10).
7. **Outdated rolling features (P0-1, fixed 2026-11-02).** Serving built "last 5" from a team's last 5 *away* games (`tests/test_form_train_serve_parity.py`). It reached main in 820bb6cd at 2026-11-02 08:34 UTC.
   - **All 626 settled tips and all 491 settled shadow fixtures were predicted before the 2026-11-01 17:00 UTC serving fixes.** 0 settled rows exist after either fix: 6 ledger and 7 shadow rows are pending.
   - The gap between model and market (no result needed) moved as follows:

     | Period | Gap | n |
     |---|---|---|
     | Before the 11-01 fixes | −12.5 pp (−13.2 to −11.8) | 564 |
     | Between the 11-01 fixes and P0-1 | −13.6 pp | 27 |
     | After P0-1 | −8.3 pp (−13.2 to −2.9) | 7 |

   - **The fix's effect on results is UNIDENTIFIABLE** (no settled fixtures). The early gap is narrower, but n = 7.
8. **Fixture-data errors.** No sign that data errors caused the loss:
   - 0 P/L mismatches, 0 settled rows with a future date, 0 missing prices, 0 duplicate fixtures.
   - Ledger result against the shadow log's score: 54 fixtures joined, **0 disagree**.
   - The 7 both-sides fixtures are listed as P3 above.
   - By model version: the current champion `newformat_20261018T080423Z_9eb6b4b4` has 108 settled Unders, ROI −18.0% (−39.3 to +4.0). The other 518 have a blank version (logged before versioning).

**What this does and does not show.** The NF model's goal level was off, and the Under side is where that bias turns into bets. This is a calibration and serving problem, not proof that Unders cannot be beaten:
- VALUABLE Unders and the 1.80–2.50 price range were not negative.
- Two large leagues are close to calibrated.

### Release protocol (measurable; the state stays PAPER until every condition holds)

Sample sizes come from `scripts/p2_under_sample_size.py`, saved in `reports/p2_under_data/sample_size.json`. Each was computed in closed form and checked by simulation.

| Step | Condition |
|---|---|
| 0. Preconditions | A clean-data NF champion is promoted through `src/promotion_gate.py` (P0_NEW_FORMAT_CLEAN_RETRAIN). Train/serve parity tests are green. A **start timestamp is frozen** in a committed file. Only predictions first logged after it count. No backfill and no replays. |
| 1. Calibration, all NF fixtures (tipped or not), paper leagues excluded | At least **584** settled fixtures, so the 95% CI on bias is ±4 pp (about 34 days at 17.0 fixtures a day). Mean (model P(over) − actual) must have a 95% CI that **includes 0** and a point estimate of at most **3 pp** in size. |
| 2. Versus market | Model minus market log loss: point estimate ≤ 0 and upper 95% CI ≤ +0.01. Today it is +0.034, CI +0.007 to +0.060. |
| 3. Under tips (every tier that would be staked) | At least **600** settled tips (ROI CI ±9.5 pp at the observed mean price of 2.395; about 106 days at 5.6 a day, without MLS). Required: ROI point > 0 **and** lower 95% CI > −5%; hit rate ≥ break-even; mean CLV ≥ 0. |
| 4. Edge is real | Rank correlation between edge and winning ≥ 0, and SNIPER ROI ≥ VALUABLE ROI − 5 pp. Today it is −0.086, and the ladder is inverted. |
| 5. Per league | A league is released only with ≥ 30 settled Under tips and a calibration-bias CI that includes 0. All other leagues stay paper. |
| 6. Looks and decision | At most 2 pre-registered looks, with 97.5% CIs (Bonferroni). Release is an owner-approved PR from PAPER to PROBATION with base_fraction ≤ 0.005 (the previous value). It is never automatic; Pro's allocator already only proposes `PAPER_PROMOTION_REVIEW`. |
| Stop rule | If the ROI upper 95% CI is below 0 at 600 tips, the track stays PAPER and the model goes back to research. |

## What I inspected

| Repo | main | data-truth-100 |
|---|---|---|
| wowza-betting | 12bd71a331610478021c029997b02d4869525ace | e07a45915ba53608dbcfe71bccb3191a3edcbed5 |
| wowza-exec | 56670768db0fa9e4fbfa02ab031542a83a889687 | a18bf544624915cd191fcdcd85f0b18769139ad2 |
| wowzaV9-Pro | f4175eea31684cd3bb40336194e276944f90056a | 78580b4286f3748f049b1a77a822b4b47a2b7a0a |
| wowza_v11 | e31e1824b65c984fdbbf4907437cae6f024c1cc5 | 27acd24ae2d39ee626a2a13c5afac1479c17c139 |
| wowza-dashboard | aa4e6a26f2e9f49f81cd1267c7b86741dfa5f626 | ff53514e70eb91a5787cf42e4fe3bbda078c678c |

Other routes checked by reading the code:
- **wowza_v11:** no stake or send path for NF O/U.
- **Pro:** `src/combo/notify.py` sends paper combos only ("Pro sends no stakes"). `src/bankroll/allocator.py` never moves a PAPER track to money.
- **No workflow runs the v9 Cloudbet bot** on either branch (grep of `.github/workflows`).
- **wowza-exec on both branches:** `deployment_mode` is PAPER_ONLY, `real_betting_allowed` is false, every real cap is 0.0 and `default_cell_state` is PAPER, so real money is impossible.
