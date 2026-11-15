# P1: Stake contract reconciliation

*Phase 2, key `contracts`, 2026-11-03. Independent QA recount. No figure here is copied from an earlier report.*

## Verdict

- **No contract violation in the stake files ever put money on a wrong bet.** The 110 rows with side `YES/OVER` and blank odds all had a **zero stake**.
- Of the 78 rows the "one position per fixture" rule flags, **74 are not extra bets**. 39 are the first bet on a fixture and 35 are the same bet logged again after a rename or a policy change.
- **4 rows are a genuine second bet on a fixture**, placed in a later run. They need **your decision**: the policy file says one bet per fixture, but a note in the contract baseline says the owner allowed same-direction bets.
- **Nothing executes these stakes.** wowza-exec has no reader of the stake files. The ledgers' ROI is flat 1-unit and does not use these stakes.
- **I found a gap that could produce new wrong rows, and closed it.** Before this branch, a BTTS or Over-1.5 tip with an invalid side (such as `YES/OVER` or blank) **would have been staked**, and the health check would still read GREEN. Four reasons the code already writes would also have been reported as false contract violations.
- **The fixes change no stake on any real tip.** Replaying the current plan (75 tips) gives identical stakes. Replaying all 815 audit rows changes only the zero-stake reason label on the 110 `YES/OVER` rows.

| What | Count | Real money effect | Reached planner | Reached execution | Reached settlement / ROI |
|---|---|---|---|---|---|
| Side `YES/OVER` (not a valid side) | 110 rows | none, all zero stake | yes, rejected as UNSUPPORTED | no | the tips are settled in `side_bets_ledger` (flat 1u), but the stake row adds nothing to any ROI |
| Blank odds | 110 rows (the same rows) | none, all zero stake | yes | no | same as above |
| One position per fixture: first bet on the fixture | 39 rows | the stake itself is legitimate | yes | no | ledger only (flat 1u) |
| One position per fixture: same bet logged again (restatement) | 35 rows (31 after the 2026-10-26 O/U rename, 4 after the 2026-11-02 policy change) | none if counted once; double-counted only if someone adds up audit stakes | yes | no | no production consumer adds up audit stakes (see 4.3) |
| One position per fixture: genuine second bet | 4 rows on 4 fixtures, 2.78 units | 2 still standing when placed; 2 after the first bet was withdrawn | yes | no | San Lorenzo: both bets settled LOSS; the other three fixtures are still pending |
| Any other stake-contract check (stake without reason OK, stake on a non-real state, wrong track, duplicate key, stake at odds of 3.00 or more) | **0** | none | | | |

Sources: `reports/p2_data/stake_contract_numbers.json` and `output/corrections/stake_audit_reconciled.csv`. Both are built by `scripts/audit/p2_stake_contract_reconcile.py` from wowza-betting **origin/main `12bd71a3`**. That commit holds the newest predict output: `stake_audit.csv`, 815 rows, generated 2026-10-14 13:39:16 to 2026-11-03 01:13:38 UTC.

## What you need to decide

| # | Priority | Decision |
|---|---|---|
| 1 | **P1** | **More than one bet on a fixture across runs.** `config/bankroll_policy.json` sets `one_position_per_fixture: true`, and the code enforces it only within a single predict run. `config/contract_baseline.json` (2026-10-26) says: "Owner 2026-10-25: several same-direction positions on one fixture are fine". Two of the four cases are a BTTS bet plus an O/U 2.5 Over bet, which have no shared direction. Choose one: (a) one bet per fixture across runs as well (a small code change, which changes stakes, so it needs your OK); or (b) several bets allowed, which means writing that rule into the policy file and the contract. I made **no** change here, because either way it changes stakes or the rules. |
| 2 | P2 | **Merge `phase2/contracts` into data-truth-100**, and later main. It contains the guards and tests described below. |
| 3 | P2 | **wowza-dashboard carries its own copy** of `src/contracts/schemas.py`, `src/stake_plan.py`, `src/bankroll_policy.py` and `src/system_health.py` (origin/main `aa4e6a26`, unchanged on its data-truth-100). It does not get these fixes, so it needs the same sync (cross-repo task P1_CROSS_REPOSITORY_CONTRACTS). |
| 4 | P3 | The health check's list of known historical violations (`config/contract_baseline.json`) matches by **count**, not by row date. A new violation could hide behind an old one only if the historical count shrank, which cannot happen in an append-only file. Today the one-position rule shows 78 rows against an acknowledged 2, so it **is** reported (AMBER). Making the list date-aware is a hardening step, not urgent. |

---

## Details

### 1. What I inspected

| Item | Repo / ref | SHA |
|---|---|---|
| Stake artifacts (newest; predict.yml commits them to production) | wowza-betting origin/main | `12bd71a331610478021c029997b02d4869525ace` |
| Same artifacts on the program branch, for comparison | wowza-betting origin/data-truth-100 | `e07a45915ba53608dbcfe71bccb3191a3edcbed5` |
| Code (contract, planner, health), base for the fixes | wowza-betting origin/data-truth-100 | `e07a4591` |
| Execution consumer check | wowza-exec origin/main | `56670768` |
| Pro simulator check | wowzaV9-Pro origin/main | `f4175eea` |
| Dashboard copy check | wowza-dashboard origin/main | `aa4e6a26` |

The data-truth-100 copy of `stake_audit.csv` has 726 rows (last row written 2026-11-02 05:59:19). It lags main by 89 rows, all of them under the 2026-11-02 policy version. The same script gives the same violation classes there: 110 bad sides, 66 one-position rows, 31 restatements and 2 second bets (`reports/p2_data/stake_contract_numbers_origin_data-truth-100.json`).

### 2. Recount

**The repo's own contract** (`src/contracts`, STAKE_AUDIT_ROW) on origin/main `stake_audit.csv`:
- `side` not allowed: 110 rows
- `odds` blank: 110 rows
- rule `one_position_per_fixture`: 78 rows

**STAKE_PLAN_ROW** on origin/main `stake_plan.csv` (75 rows, generated 2026-11-03 01:13:38): **0 violations**. The current plan has 12 staked rows, no fixture with 2 or more stakes, no invalid side and no blank odds.

**Independent row checks** (written in the reconcile script, not taken from the contract), on 815 rows:

| Check | Rows |
|---|---|
| side not valid for its market | 110 |
| odds blank | 110 |
| stake > 0 with reason other than OK | 0 |
| stake > 0 on a state that is not real-money | 0 |
| track not equal to `track_key(model_type, market, side)` (or the pre-split parent key) | 0 |
| duplicate (policy_version, fixture, track) | 0 |
| stake > 0 at odds of 3.00 or more | 0 |
| fixture with more than one staked row (all staked rows on it) | 78 rows / 39 fixtures |

There are 188 distinct rows with at least one violation. The two sets do not overlap: the 110 bad-side rows are exactly the 110 blank-odds rows, and none of them is staked.

**Regimes**, cut at the commit that defines each one (UTC, origin/main):

| Regime | From | Defining commit | Audit rows |
|---|---|---|---|
| R0 pre side-fix | start | (before) `219c1848` 2026-10-22 10:45:28: side tips reach the policy with real model_type/odds | 350 |
| R1 side fixed | 2026-10-22 10:45:28 | `219c1848` | 118 |
| R2 direction guard | 2026-10-25 12:17:16 | `61462e71`: one goal direction per fixture | 28 |
| R3 O/U side split | 2026-10-26 11:48:13 | `ab23726d`: O/U 2.5 tracks split by side | 230 |
| R4 policy 2026-11-02 | policy_version `...+nf-under-paper-2026-11-02+league-rating-off-2026-11-02` | `26ded833`, `820bb6cd` | 89 |

### 3. Invalid side `YES/OVER` and blank odds (110 rows, the same rows)

| Field | Value |
|---|---|
| Markets | BTTS 89, OVER15 18, OVER35 3 |
| Tiers | VALUABLE 89, SNIPER 13, MARKSMAN 8 |
| Written (tip time = audit `generated_at`, first sight) | 2026-10-14 13:39:16 to 2026-10-22 10:07:00, **all in R0**, before `219c1848` |
| State / reason then | `UNSUPPORTED` / `NO_REAL_STAKE_UNSUPPORTED` on all 110 (track `unknown\|BTTS` etc., because the producer also lost model_type) |
| Stake | 0 on all 110 |
| In the current stake plan | 0 |
| Execution | none: wowza-exec origin/main has no reader of stake_plan or stake_audit (its Cloudbet shadow reads `side_bets_ledger.csv`, which is a tip, not a stake) |
| Settlement | all 110 tips have a `side_bets_ledger` row: side YES 89, OVER 21; settled 56 WIN, 54 LOSS (flat 1u ledger, which does not depend on this stake row) |
| ROI | no stake-weighted ROI is affected (zero stake). `src/research/league_weekly._stake_lookup` splits `YES/OVER` into YES and OVER keys, but with `actually_bet = False`, so nothing changes. |
| Canonical value in the derived view | side = the ledger row's side (YES or OVER). The join key is (date, home, away, market); every key has exactly one ledger side, so confidence is HIGH. Odds = the ledger price, 1.40 to 2.48. That is the ledger's own first-sight price, not necessarily what the planner saw, so confidence is MEDIUM. |

**This is a historical audit artifact only.** It can no longer occur through the side producer: `219c1848` restored side and odds, and this branch now rejects an invalid side (see section 5).

### 4. One position per fixture (78 rows, 39 fixtures)

The contract rule flags **every** staked row on a fixture that has more than one staked row anywhere in the append-only file, across all runs and all policy versions. Each row is classified by the reconcile script:

| Class | Rows | Regimes | What it is |
|---|---|---|---|
| FIRST_STAKED_POSITION | 39 | R0 3, R1 19, R2 10, R3 7 | The first stake on the fixture. Legitimate. |
| TRACK_RENAME_RESTATEMENT | 31 | R3 31 | The same bet (fixture, market, side), first logged under `new_format\|OU25`, then under `new_format\|OU25_OVER/UNDER` after the 2026-10-26 split. The audit is first-sight per (policy_version, fixture, track), so a renamed track is a new key. |
| POLICY_VERSION_RESTATEMENT | 4 | R4 4 | The same bet logged again under the 2026-11-02 policy version (the key includes policy_version). |
| SECOND_POSITION_LATER_RUN | 2 | R1 1, R3 1 | A different bet on the same fixture, staked by a later run while the first was still standing. |
| SECOND_POSITION_AFTER_FIRST_WITHDRAWN | 2 | R4 2 | A different bet staked after the first bet's latest decision had become zero. |

There are **no same-run violations**: `size_day` step 7 works within a run. There are also **no opposite-direction pairs**: every second bet pairs BTTS with a goal line, and BTTS has no direction.

Of the 35 restatements, 15 have identical odds and stake. The other 20 were re-sized because the odds moved or, in R4, because league-rating sizing was switched off. They sum to 12.52 units. The ledger outcomes of the restated bets are 12 WIN, 18 LOSS and 5 pending; each is **one** ledger row per bet.

**4.1 The four genuine second bets** (full rows in the derived CSV):

| audit_row | Fixture | Second bet (odds, stake u, tip time UTC) | First bet (audit_row, odds, stake u, tip time) | Regime / state | Status |
|---|---|---|---|---|---|
| 375 | 2026-10-24 San Lorenzo v Newells Old Boys | BTTS YES 2.25, 0.79, 2026-10-22 13:45:31 | 354 OVER15 OVER 1.58, 0.34, 2026-10-22 11:00:25 | R1, ACTIVE | both settled LOSS. Already acknowledged in `config/contract_baseline.json` (max 2 rows). |
| 678 | 2026-11-07 Shimizu S Pulse v Tokyo Verdy | O/U 2.5 OVER 2.25, 0.32, 2026-11-01 09:27:27 | 673 BTTS YES 2.00, 0.73, 2026-11-01 05:55:37 | R3, ACTIVE_LOW | pending; neither bet is in the current plan |
| 741 | 2026-11-04 Flamengo v Grêmio | BTTS YES 2.02, 1.38, 2026-11-02 15:44:24 | 621 O/U 2.5 UNDER 2.50, 0.50, 2026-10-29 00:23:17 | R4, ACTIVE | the UNDER bet became PAPER (stake 0) under the 2026-11-02 policy in the same run. BTTS is in the current plan. |
| 782 | 2026-11-08 Instituto de Córdoba v Newells Old Boys | O/U 2.5 OVER 2.43, 0.29, 2026-11-02 15:44:24 | 708 BTTS YES 2.10, 0.85, 2026-11-01 17:03:44 | R4, ACTIVE_LOW | in the same run, the BTTS tip dropped to VALUABLE (TIER_NOT_STAKED, stake 0) |

The stakes total 2.78 units (column `stake_amount`, notional 100-unit bankroll).

**4.2 Is anything double-executed?** No. The stake plan is a recommendation file and nothing places it (see section 3). `stake_plan.csv` holds **one** row per tip per run, and the current plan has **0** fixtures with two stakes. The duplicates exist only in the append-only audit.

**4.3 Is anything double-counted in ROI?** Not in production:
- the ledgers carry flat-stake P&L with no stake column;
- `league_weekly._stake_lookup` takes the **max** stake per (fixture, market, side), not the sum;
- the dashboard reads `stake_plan.csv`, not the audit;
- Pro's simulator re-sizes from the ledgers itself.

Anyone who **sums** `stake_audit.stake_amount` would over-count by the 12.52 restated units. An example is the audit script `scripts/audit/independent_backtest/run.py` (`actual_stake_audit`), which is research only. In that case, use `stake_audit_reconciled.csv` and count each restatement once.

### 5. Fixes on `phase2/contracts` (failing test first)

`tests/test_stake_contract_guards.py` has 27 cases. **18 failed on the untouched base (`e07a4591`)**, and all 27 pass after the fixes.

| # | Gap found | Failing case on base | Fix |
|---|---|---|---|
| 1 | The contract's reason pattern did not include 4 zero-stake reasons the code already writes: `NO_LEAGUE_THRESHOLD_EVIDENCE`, `NO_LEAGUE_RATING`, `NO_RATING_MIN_SCALE` and `OPPOSITE_DIRECTION_ON_FIXTURE`. Once data-truth-100 reaches main, every global-threshold tip would have shown as a **false violation**. That noise is how real violations get waved through. | `test_every_reason_size_day_writes_satisfies_the_contract` scans `size_day`'s source, so a future reason cannot drift either. `test_a_global_threshold_row_is_contract_clean` | `src/contracts/schemas.py` `STAKE_REASON_RX`: the 4 reasons plus `DUPLICATE_POSITION` added. `test_the_reason_pattern_still_rejects_garbage` keeps a meaningful failure condition. |
| 2 | `size_day` ignored `side` on every market except O/U 2.5. A BTTS / OVER15 tip with side `YES/OVER`, `OVER`, blank or None **was staked**. | 7 parametrised cases | `bankroll_policy.side_valid` + reason `NO_REAL_STAKE_INVALID_SIDE` (stake 0). A caller that sends **no** side column (none in production) is not judged, which matches the existing `league_threshold_evidence` convention. A valid side is still staked (`test_a_valid_side_is_still_staked`). |
| 3 | The side-selection contract checked columns, model_type and odds, but not whether the side belongs to its market. | 5 cases | `stake_plan.check_side_selections` raises `SelectionSchemaError`, which makes a broken side a loud failure, the same discipline as a missing column. Valid sides pass. |
| 4 | With `one_position_per_fixture` off, the same (fixture, market, side) could hold **two stakes in one run**. | `test_one_position_never_carries_two_stakes_even_with_the_policy_flag_off` | `size_day` step 6c `DUPLICATE_POSITION`. It acts only when the flag is off (production has it on, and step 7 already covers that case), and it keeps step 7's ranking. Two *different* bets with the flag off are untouched (`test_flag_off_still_allows_two_different_positions`). |
| 5 | Health checks read **GREEN** on an invalid side, and `stake_plan.health_check` did not report two stakes on one fixture or position. | `test_stake_mapping_is_not_green_on_an_invalid_side`, `test_health_check_reports_an_invalid_side`, `test_health_check_reports_two_staked_rows_on_one_fixture` | `system_health.stake_mapping` counts an invalid side as unmapped. `stake_plan.health_check` reports invalid sides, positions with more than one stake and (while the policy says one per fixture) fixtures with more than one stake. A clean plan still returns `[]` (`test_health_check_clean_plan_is_empty`), so the check is not always-red. |

**No stake change** (`scripts/audit/p2_stake_guard_no_change.py`, output `reports/p2_data/stake_guard_no_change.json`). The script re-sizes production's own selections with the base `bankroll_policy.py` and with the patched one, under the committed policy:
- `stake_plan.csv`: 75 tips, 12 staked under both, fraction sum 0.077501 under both, **0 rows differ**;
- `stake_audit.csv` selections: 815 rows, 29 staked under both, sum 0.206999 under both. 110 rows differ, and **all of them are the `YES/OVER` rows**, whose zero-stake reason changes from `NO_REAL_STAKE_UNSUPPORTED` to `NO_REAL_STAKE_INVALID_SIDE`.

Pro's simulator (`wowzaV9-Pro/src/bankroll/data.py`) always sends side YES for BTTS and OVER for the over lines, so the guard does not change its sizing.

**Focused regression run** (not the full suite): 11 files covering stake guards, O/U side tracks, stake plan, side stake contract, direction guard, contracts, system health, league-rating sizing, drift tier, market_edge_v2 and league_weekly. Result: **194 passed**.

**What this does not do.** It does not enforce one position per fixture **across runs** (your decision 1). The O/U 2.5 path still writes a side-less O/U tip as a zero-stake row on track `*_UNKNOWN_SIDE`, by the existing design (tests in `test_ou25_side_tracks.py`). That row is flagged by `health_check`, by `stake_mapping` (non-GREEN) and by the contract's side check, so it is never hidden.

### 6. The derived view

`output/corrections/stake_audit_reconciled.csv` has 298 rows: one per (audit row, violation), covering 188 audit rows. It carries:
- the original stake row fields;
- `regime`, `in_current_stake_plan`, `ledger_match`, `ledger_result` and `reached_execution`;
- `violation`, `field`, `original_value`, `canonical_value`, `correction_reason`, `provenance` and `confidence`.

Confidence counts: side_invalid HIGH 110, odds_blank MEDIUM 110, one_position HIGH 74, one_position OWNER_DECISION 4. Nothing is UNRESOLVED. The audit note is `output/corrections/README_stake_audit_reconciled_2026-11-03.md`. `output/stake_audit.csv` and the ledgers are unchanged.

### 7. Reproduce

```
python scripts/audit/p2_stake_contract_reconcile.py --ref origin/main
python scripts/audit/p2_stake_contract_reconcile.py --ref origin/data-truth-100
python scripts/audit/p2_stake_guard_no_change.py --ref origin/main --base origin/data-truth-100
python -m pytest tests/test_stake_contract_guards.py -q
```
