# Phase 2 - current baseline (recounted 2026-11-03)

Independent recount. Every number below was recomputed today from the committed files and the code at the
SHAs listed. Nothing was copied from an earlier report. Scripts and raw outputs: `reports/phase2_baseline/`.

## Verdict

- **Both branches report WARN, with 0 RED.** data-truth-100: 20 GREEN, 2 AMBER. main (production): 20 GREEN, 1 AMBER.
- **The data-truth-100 fixes are real.** Known silent defaults went from 35 on main to 0. The scan patterns are the same on both branches, and an injected default still turns the check RED.
- **Production has one new data gap that health does not flag.** Since 2026-11-02 15:44 UTC, 99% of new-format fixtures on the live board have no half-time form. 11 of the 42 features used by the BTTS and Over 1.5 models are affected, and new-format BTTS is the only ACTIVE track. "Data truth 3" still shows GREEN because it checks Standard fixtures only.
- **The half-time history fill is on data-truth-100 only.** It covers about 87% of the missing new-format rows. It does not yet reach production.
- **Forward results are unchanged and still not validated.** Over the same settled bets, the two branches give identical results. New-format BTTS is the only cell whose interval excludes zero. It is retrospective, so it is **not** confirmed edge.
- **In all 5 repos, every workflow's latest completed main run succeeded, with two exceptions.** "Backtest" is active but last failed on 2026-07-13. "Pre-season Retrain" is disabled and was last cancelled on 2026-08-01. 7 research or one-off workflows have never run on main.

## The brief's table

"Previous" = production `origin/main` today. "Current" = `origin/data-truth-100`. Both columns were recounted today.

| Item | Previous (main 12bd71a3) | Current (data-truth-100 e07a4591) | Verified? | Remaining issue |
|---|---|---|---|---|
| Training/serving parity (form features) | Fix 1ea46192 is on main | Same fix | **Partly.** `test_form_train_serve_parity.py` passes on both. A replay on frozen inputs was not run here: UNIDENTIFIABLE in this audit (owned by the P0 parity report) | The fix exposed missing new-format HT history (see next row) |
| HT coverage, all tracks (live board) | Standard: HT form missing on 1.5%. **New-format: missing on 99% (98/99 fixtures)** | The committed board snapshot (main, 2026-11-02 05:59 UTC) shows NF 9.8% missing. That snapshot predates 1ea46192 on the board, so it is not a current measurement | **Yes.** Measured across 12 main board commits. NF went from 10% to 99% missing between 05:59 and 15:44 UTC on 2026-11-02 | 11 of 42 features in the BTTS / Over 1.5 / Over 3.5 / Standard / HT models are >90% missing on NF fixtures. The health check covers only Standard fixtures |
| HT history (banks) | fd_history NF rows with HT: 9.5% (4,011 / 42,084); since 2026-07-01: 0% | Same bank, plus a sidecar `output/collect/fd_history_ht_fill.parquet`: 33,259 rows filled of 38,073 lacking (87.4%); about 88.6% NF coverage when combined. Since July: 1,826 of 2,188 | **Yes** (counts recomputed from the sidecar and the bank) | 3,616 unmatched, 1,198 refused (fixture claimed twice) and 8,220 two-spelling rows. Not merged to main |
| Rest days | Standard 0% / 1.5% missing (home/away); NF 2.0% / 3.0% | Standard 0% / 1.4%; NF 2.0% / 2.9% | **Yes** | None material |
| Stake contracts | `stake_audit.csv`: 815 rows. **78 rows** break one-position-per-fixture (baseline allows 2). 35 staked rows repeat date, fixture, market and side. 110 frozen pre-fix rows (known) | 726 rows (older snapshot); 66 rows break the rule | **Yes** | 39 fixtures carry more than one staked position. One mixes Under 2.5 and BTTS YES (Flamengo v Gremio, 2026-11-04). The contract check is AMBER on both branches |
| NF O/U Unders PAPER | Current `stake_plan.csv`: 25 NF Under rows, **0 staked** | The committed plan is older than the policy change: 29 rows, 11 staked under `balanced-2026-10-13` | **Yes** on main | None on main. The data-truth-100 output snapshot is stale, so do not read it as live |
| Pooled league-rating sizing | `league_rating_sizing.enabled = false` | Same, plus a "no rating = no stake" note | **Yes** | None |
| Props defaults | No sidecar on main | `props_sentinel_prob.csv`: 376 tips whose probability was a cap (0.4: 198, 0.5: 168, 0.8: 10), P/L -117.84u. Staked-tier props P/L as stored is -231.34u; excluding the listed rows it is -113.50u | **Yes**. 376 rows, value split and P/L recomputed from the sidecar | `player_model/fantasy_residual.py` still has typed constants (0.35, 0.92, 0.01) that the defaults scanner does not cover |
| Known-defaults baseline + ratchet | 35 entries, ceiling 35 | **0 entries, ceiling 0**. The scanner finds 0 hits with identical patterns | **Yes**. An injected `fillna(1.5)` turns the check RED and the gate fails. A raised ceiling also fails the gate | The scanner reads 20 files only |
| CLV integrity | Certified CLV coverage 69.5% of 387 (since 2026-10-23); 35.8% all-time | 70.7% of 365; 35.5% all-time. Sidecars quarantine 170 + 6 first-half closes | Read from `clv_collection_health.json` (not recomputed from raw snapshots) | About 64% of shadow rows since 2026-08 have a stale close (535 of 829). Main has no corrections directory, so production displays do not apply the quarantine |
| Model promotion | Promotion gate GREEN: 56 decisions, last 2026-11-01 17:13Z | Same; `current_champions.json` identical | **Yes** (file diff = 0) | 6 of 8 champions are `LEGACY_GIT` with a blank training cutoff. `standard_repaired` is `OWNER_INSTALL`, installed before any forward evidence. `newformat` is a ROLLBACK |
| Corrected forward ROI (O/U and side markets, staked tiers, flat 1u, since 2026-08-10) | 1,828 settled. NF BTTS n143 +19.8%; NF O/U 2.5 n468 -7.7%; STD O/U 2.5 n158 -16.0%; NF Over 1.5 n111 +7.1%; STD BTTS n34 -12.2% | **Identical** | **Yes, two ways**: the ledger `pnl` and P/L rebuilt from odds and result agree exactly (max difference 0) | None of these is validated. NF BTTS iid 95% CI [+2.1%, +37.6%] is retrospective and tier-selected. Sharp ledger: +22.22u booked versus **-159.14u** at the signal-time price (matched three ways) |

## 1. What was inspected

| Repo | main | data-truth-100 | DT ahead / main ahead (commits) |
|---|---|---|---|
| wowza-betting (v9) | 12bd71a3316 | e07a45915ba | 140 / 43 |
| wowzaV9-Pro | f4175eea316 | 78580b4286f | 18 / 49 |
| wowza_v11 | e31e1824b65 | 27acd24ae2d | 9 / 2 |
| wowza-dashboard | aa4e6a26f2e | ff53514e70e | 2 / 2 |
| wowza-exec | 56670768db0 | a18bf544624 | 10 / 0 |

Fetched 2026-11-03 06:2x UTC. The health runs used `now = 2026-11-03T12:00Z`, which is about 5.5 h ahead of the run time. Every age-based value is therefore overstated by about 5.5 h. No status changes because of it: the DT edge-shadow artifact is still more than 24 h old at the real time. The Pro inputs came from Pro worktrees at the SHAs above (`PRO_LOCAL`).

**Important:** the committed `output/` files on data-truth-100 are a snapshot of production from about 2026-11-02 06:00 UTC, taken when main was merged in. Wherever a number depends on live output files, main is the current one.

## 2. System health (src/system_health.build_artifact, offline)

| | GREEN | AMBER | RED | NO_DATA | overall |
|---|---|---|---|---|---|
| main | 20 | 1 | 0 | 0 | WARN |
| data-truth-100 | 20 | 2 | 0 | 0 | WARN |

data-truth-100 has one more check than main: "Data truth 5: one match, one row". It is GREEN: 105,999 bank rows become 71,146 after the loader's identity pipeline, which removes 34,853 copies, and 0 duplicate pairs remain.

AMBER on both: **Data contracts**. There are 2 new violations, in `v91_shadow.csv` (`p_v9_served` column absent, about 1,020 rows) and `stake_audit.csv` (one-position rule, 66 on DT / 78 on main against a baseline of 2). The 6 known violations are covered by `config/contract_baseline.json` (the file lists 7 entries).
AMBER on DT only: **Edge-engine shadow**. Its artifact is about 30 h old in the snapshot, while on main it is about 11 h old and GREEN.

Data truth rules (DT / main): 1 naming GREEN (0 unknown leagues; blind board fixtures 3.4% / 3.6%). 2 defaults GREEN (0 new; 0 / 35 known). 3 missing GREEN on both, but see the HT gap above. 4 league relations GREEN on both. 5 one match GREEN (DT only). Full output: `phase2_baseline/health_*.json`.

**A GREEN with a blind spot.** Data truth 3 checks `home_rest_days` and `home_ht_over05_rate` on Standard fixtures only. So it shows GREEN while 99% of new-format board fixtures have no HT form. The check is accurate for what it measures, but it does not cover the new-format track.

Focused tests: 9 files, covering parity, integrity parity, data truth gate and checks, contracts, side stake contract, track registry, promotion gate and system health. **108 passed on data-truth-100 and 107 passed on main.** The full suite was not run (laptop rule).

## 3. Bankroll policy, track registry, track evidence

- Policy `balanced-2026-10-13+nf-under-paper-2026-11-02+league-rating-off-2026-11-02` (identical on both branches apart from one note). States: ACTIVE 1 (NF BTTS), ACTIVE_LOW 2 (NF O/U 2.5 Over, NF Over 1.5), PAPER 5, BLOCKED 2, BLOCKED_FROM_REAL 1 (player props). Status `WIRED_PAPER`: Cloudbet is PAPER_ONLY.
- Track registry: 100 entries and 0 drifted fields on both branches. On data-truth-100 only the policy hash changed (`a63a5edf...`).
- `output/track_evidence.json` was generated for 2026-11-01, so it is 2 days behind the ledger. For example, NF O/U 2.5 shows n=452 there against n=468 recounted. Statuses: NF BTTS VALIDATABLE; all others NEUTRAL or INSUFFICIENT. CONFIRMED is not reachable by design.
- Forward candidate progress (Pro): 48/1200 on main, 35/1200 in the DT snapshot.
- Milestone "Data Truth 100%": **open, 21 open issues / 0 closed**. The fixes sit on data-truth-100 and none is merged to main.

## 4. GitHub Actions, latest completed run on main per workflow

Source: `gh` API, 2026-11-03. Full list: `phase2_baseline/gh_latest_main_run_per_workflow.txt`.

| Repo | Workflows | Latest = success | Not success / never ran on main |
|---|---|---|---|
| wowza-betting | 34 (5 disabled) | 28 | Backtest: **failure 2026-07-13** (active, stale). Pre-season Retrain: cancelled 2026-08-01 (disabled). Never on main: Data Truth Collect (runs on data-truth-100, last 3 runs success), Diagnostic set-piece, Props R2 challengers, Understat probe |
| wowzaV9-Pro | 32 (2 disabled) | 29 | Never on main: League-Level + Market-Anchor benchmark, Logit Calibration benchmark, R2-B Repaired Challengers (research-branch workflows) |
| wowza_v11 | 1 | 1 | none |
| wowza-dashboard | 1 | 1 | none |
| wowza-exec | 1 | 1 | none |

The two data-truth-100 runs in wowza-betting both succeeded. The system-health "GitHub workflow health" check was GREEN on both branches (from the committed `workflow_health.json`). It covers 3 repos only, not dashboard or exec.

## 5. Other repos

Pro, v11 and exec have data-truth **tests** on data-truth-100 but no runtime health check (open issue #9). The dashboard carries copies of v9's `system_health.py` / `data_truth.py`, which it syncs from v9 main.

## Open issues (priority, owner action)

| P | Issue | Owner action |
|---|---|---|
| P0 | NF board: HT form 99% missing in production since 2026-11-02 15:44 UTC. It feeds NF BTTS (ACTIVE) and NF Over 1.5 (ACTIVE_LOW) | Decide whether to merge the HT fill (data-truth-100) or hold NF BTTS / Over 1.5 stakes. I changed no stake or state |
| P1 | Data truth 3 checks rest days / HT on Standard fixtures only, so it shows GREEN while the NF gap exists | Approve extending rule 3 to every track, with a test that fails on today's main board |
| P1 | `stake_audit.csv` one-position rule: 78 rows against an allowed 2. There are 35 repeated staked rows and 1 fixture with Under 2.5 + BTTS YES | Confirm whether Under 2.5 + BTTS YES counts as an opposite direction. P1_STAKE_CONTRACT_RECONCILIATION should explain the 78 |
| P1 | 21 Data Truth issues open. data-truth-100 is 140 commits ahead and 43 behind main. Corrections and sidecars are not in production | Plan the merge after Phase 2 validation |
| P1 | Champions: 6 of 8 are LEGACY_GIT with no training cutoff | Clean retrain (P0_NEW_FORMAT_CLEAN_RETRAIN) |
| P2 | `fantasy_residual.py` typed constants are outside the defaults scanner's 20 files | Add the fantasy/props files to `SCAN_FILES` (ratchet would then need a baseline with reasons) |
| P2 | Certified CLV all-time is about 36%, with 535 of 829 shadow closes stale | No action needed. Since 2026-10-23 it is about 70% |
| P3 | "Backtest" workflow: active, last main run failed 2026-07-13 | Disable it or fix it after the milestone closes |
| P3 | `track_evidence.json` is 2 days behind the ledger | None (the daily job refreshes it) |

No code was changed and no test was added in this step. Nothing was retrained, backtested, sent or staked.
