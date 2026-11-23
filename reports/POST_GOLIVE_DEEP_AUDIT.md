# Post-go-live deep audit (owner deep-repair brief 2026-11-22, item 14)

Go-live: Clean Record start 2026-11-19T06:41:48Z. This audit closes the deep-repair brief: it reads every input
report on main, re-measures what can be re-measured, and lists what is still open. Certification and the
acceptance matrix are in `reports/FINAL_DATA_TRUTH_CERTIFICATION.md`; row-level counts in
`output/real_data_certification.json`. Counts only - no price, odds, probability or stake value.

## Repos and SHAs

| Repo | main (at final refresh) | main CI, last 24 h |
|---|---|---|
| NevixAA/wowza-betting | `5a6727f42` (Merge #84) | 36 success / 1 failure (Tests and gates 05:36Z on `73c18f399`, fixed by #85: next run on `e4a3a9847` success) |
| NevixAA/wowzaV9-Pro | `17d6faabc` | 38 success / 2 failures (Specials 11-22 15:42Z, Player V3 11-22 17:34Z; both green on their next run) |
| NevixAA/wowza_v11 | `4b336541f` | 4 / 4 success |
| NevixAA/wowza-dashboard | `eda50a6b2` (BUILD_INFO v9_head_at_build `a1e459f3f`, built 09:11:46Z) | 6 / 6 success |
| NevixAA/wowza-exec (private, read via gh) | `82f6d7d9f` | 5 success, 3 skipped (gated off), 0 failures |

## Inputs read (all on main)

| Item | PR | Merge | Report / evidence |
|---|---|---|---|
| P0-2 four health indicators | #75 | `eea3383d0` | `reports/P0_HEALTH_STATUS_CORRECTION.md` |
| P0-3 join integrity + active lineage | #75 | `eea3383d0` | `reports/P0_JOIN_LINEAGE_VERIFICATION.md`, `reports/lineage/training_lineage_evidence.json` |
| P0-4 HT feature parity + HT retrain install | #78 | `697afc61c` | `reports/P0_HT_FEATURE_PARITY.md` |
| P0-5 props certification | #74 | `300bf660d` | `reports/P0_PROPS_PRODUCTION_CERTIFICATION.md`, `output/props_real_data_provenance.json` |
| P0-6 synthetic ROI quarantine | #74 | `300bf660d` | `reports/P0_SYNTHETIC_ROI_QUARANTINE.md` |
| P1-7 price / CLV coverage | #82 (+ Pro #19) | `73c18f399` | `reports/P1_PRICE_CLV_COVERAGE.md` |
| P1-8 Clean Record evaluation | #84 | `5a6727f42` | `reports/P1_CLEAN_RECORD_EVALUATION.md`, `output/clean_record_evaluation.json` |
| P1-9/10/12 Pro, v11, dashboard reconciliation | #83, Pro #18, v11 #4 | `15cdbcbc8` | `reports/P1_PRO_RECONCILIATION.md`, `P1_V11_RECONCILIATION.md`, `P1_DASHBOARD_SYNC_VERIFICATION.md` |
| P1-11 Fantasy validation | #81 | `cf53f2cd7` | `reports/P1_FANTASY_VALIDATION.md` |
| Standard ledger settlement | #76 | `5df8750be` | `tests/test_prediction_ledger_std_settle.py` |
| Main red fix | #85 | `e4a3a9847` | time/data-dependent tests, NaN bar max, coherence gate off the notifier |

## New findings in this audit

| # | Finding | Evidence | Status |
|---|---|---|---|
| D1 | 2 Standard training rows come from `output/backtest_all_leagues.csv` (`row_source = backtest_archive`), an internal backtest output, not a provider file. Their stats disagree with API-Football (Standard v Anderlecht 2022-10-23; Charleroi v Mechelen 2022-11-12: shots and SOT differ on both sides); their corners and fouls exist in no provider file. football-data has the scores but no stats for these two matches, so the archive row survives the identity pipeline. 19 later training rows of the 4 clubs read these stats in their rolling corners / fouls / SOT-ratio windows. The archive holds only Standard leagues, so no NF payload can read it. | `output/real_data_certification.json` -> `team_training.payloads[*].row_level_saved_frame.backtest_archive_rows`, `team_training.backtest_archive_scope`; `src/data_loader.py` `_load_backtest_archive` | **Open - owner decision** (fix = drop unconfirmed archive stat cells or `WOWZA_USE_BACKTEST_ARCHIVE=0`, then refit the 6 Standard payloads). 21 of 49,407 rows. No serving impact today (2022 windows). |
| D2 | `config/feature_provenance.json` was stale against the installed payloads (last written 2026-11-20, before #78): Standard xG training coverage said 17% (now 7.5%), inside-box 81% (now 44.9%), serving coverage 98.6% (now 96.8-100%), plus shifted source line references. No workflow regenerates it, and 13b only checks NEW violations, so staleness was invisible. | `python -m src.integrity.real_data_provenance --write` on main `5a6727f42` vs the committed registry | **Fixed in this PR**: registry regenerated; inside-box reclassified DERIVED_VERIFIED -> VALID_MODEL_ESTIMATE on the Standard payloads; 13a GREEN, 13b 0 new violations, 13c AMBER (7 registered) unchanged |
| D3 | The feature parity monitor (`scripts/integrity_feature_parity.py`) builds its training side from `base_history()` without the football-data season files (PR #59) and HT fills, so its reference differs from the real training frame. It reports parity FAIL, which keeps production model integrity at FAIL. The cell-level check (`scripts/team_train_serve_parity.py`, training builder vs serving builder on the same matches) passes: 0 of 2,666 raw-input cells differ (62 fixtures, 2026-11-14..20, 43 raw columns) and 62 / 62 probabilities are identical on each of the 9 active payload files, run on main `5a6727f42`. | `output/feature_parity_health.json`; `output/model_integrity_health.json` | **Open** - measurement defect, not a model defect. Fix: build the monitor's training side through the trainers' loader path. |
| D4 | `player_history.parquet` still stores the legacy placeholders age 25 / 180 cm / 75 kg / age_peak_delta 0 on ~888k of 891,912 rows. No model reads them (the feature builder treats them as missing). | `output/real_data_certification.json` -> `props.stored_placeholders_in_player_history` | **Open** - stored-data cleanliness; rewrite to NaN in a data commit |
| D5 | Corners / cards: single-source leagues (17,043 fixtures per family) are verified by API-Football + a sanity rule, not a second provider. | `models/side_models/side_models.json` coverage | Documented limitation (owner rule 2026-11-18) |
| D6 | Team / player ledgers store a price but not the bookmaker or capture time; 427 of 510 certified closes since 10-23 are cross-book (Bet365 close vs OddsAPI entry). | `reports/P0_SYNTHETIC_ROI_QUARANTINE.md`, `reports/P1_PRICE_CLV_COVERAGE.md` | Open (owner): record book + quote time per tip |

## Per-repo health (now)

Four indicators (PR #75), from the committed artifacts at the final refresh:

| Indicator | Status | Source |
|---|---|---|
| CI / change-control gate | GREEN (`data_truth --gate` and `research_lock --check` on this branch, below) | local run on `5a6727f42` + this PR |
| Operational data quality | **RED**: Data truth 9 "2 v91_shadow fixture(s) logged twice" (ungated); 13c AMBER (7 registered exceptions). Data truth 3 is GREEN since #78 | `output/data_truth_health.json` 2026-11-23T08:39:19Z, `output/system_health.json` 08:35:46Z (31 GREEN / 1 RED / 1 AMBER / 1 NO_DATA) |
| Production model integrity | **FAIL** (D3 parity monitor; artifact predates #78) | `output/model_integrity_health.json` 2026-11-22T20:03:05Z |
| Full-system certification | **NOT CERTIFIED** | follows from the two rows above |

Data truth 9 at the final refresh: **RED - 2 v91_shadow fixtures logged twice (`output/data_truth_health.json` 08:39:19Z); its fix had NOT merged to main at this refresh (another agent owns it)**

| Repo | Health |
|---|---|
| wowza-betting | CI green after #85; operational RED on Data truth 9 only; model integrity FAIL (D3 measurement) |
| wowzaV9-Pro | collectors green; weekly audit FAIL recorded 2026-11-22 19:49Z on pre-fix code (2 FAIL: near-kickoff per-book snapshot coverage - a different measure from the v9 certified close, see P1_PRICE_CLV_COVERAGE section 1; registry reconciliation, fixed by Pro #18). The next scheduled audit had not committed at this refresh |
| wowza_v11 | V11 Collect green; `v11_research_health.json` PASS; team_names byte-identical to v9 |
| wowza-dashboard | Sync from v9 green; byte-level rebuild 0 differences at `61e86987` (P1-12); known inf/NaN column-config exception on the Dashboard page (not fixed) |
| wowza-exec | green; private stores read-only here |

## The 12 closing questions

1. **Defects fixed (this brief, P0/P1 items 1-16).** Health stoplight conflated CI gate with data quality (#75);
   legacy join replay failed every track (#75); props PENDING + latent REPORTED pass (#74); synthetic props ROI
   unlabelled (#74); Standard HT `home_has_ht_history` constant 1.0 on 100% of training rows, retrained + installed
   `0776cea1` / `aa4c62ce` (#78) + latent half-recorded HT score marker (#78); Standard prediction-ledger settlement
   (#76); NEAR capture chaining for CLV closes (#82, Pro #19); Pro registry as-of reconciliation (Pro #18); v11 stale
   `team_names.py` (v11 #4); Fantasy pooled table compared different players, register text (#81); main red tests
   (#85). In this PR: **D2** stale `config/feature_provenance.json` regenerated; the row-level certification script.
2. **Already correct (verified, not changed).** 7 / 7 active serving joins; training lineage of all 12 team parts
   (PASS); cell-level train/serve parity of all 9 active team payload files (0 of 2,666 raw-input cells differ (62 fixtures, 2026-11-14..20, 43 raw columns) and 62 / 62 probabilities are identical on each of the 9 active payload files, run on main `5a6727f42`); history banks 0
   defaults; prediction ledger 0 unscored rows without a reason; ledgers 0 synthetic labels / 0 prices <= 1; side
   model truth rule; Fantasy FPL inputs 2,631 / 2,631 = `event/live`; dashboard byte-level sync.
3. **Files / SHAs (this PR).** `scripts/certify_real_data.py`, `tests/test_certify_real_data.py`,
   `output/real_data_certification.json`, `config/feature_provenance.json`, this report and
   `reports/FINAL_DATA_TRUTH_CERTIFICATION.md`; base main `5a6727f42`; PR head and merge SHAs are in the PR.
4. **Tests.** SUITE_PLACEHOLDER
5. **Per-repo health.** Table above: all 5 repos have green CI on main now; v9 operational data RED on Data truth 9
   only; Pro weekly audit FAIL recorded on pre-fix code.
6. **Active model integrity verdict.** Lineage PASS on all 12 team parts (evidence 2026-11-22; the 2 Standard HT
   parts were reinstalled by #78 from the same frame `ebe47c041a7f595f`, whose refit without the exclusion
   reproduces the old payloads exactly); cell-level parity 0 differing cells; but
   the production indicator is **FAIL** (parity monitor reference defect D3; the artifact predates #78). Real data:
   the 6 Standard parts are **NOT_REAL** (D1, 21 / 19 rows); the 6 NF parts are REAL_100 with missing kept missing.
7. **Props provenance verdict.** **VERIFIED 13 / 13** installed artifacts (sha256, schema, chronology, parity
   6,316 rows bit-identical, batch invariance). Row level: 959,081 provider-observed + 5,311 provider-rule rows;
   1,882 zeros_unverified kept as context; 4,045 excluded by `all_prior_verified`; 934,014 training-eligible. Stored
   placeholders in `player_history.parquet` (D4) are not read by any model.
8. **Fantasy results.** FPL inputs real (2,631 / 2,631). Paired walk-forward (968 identical rows, GW5-GW10): V2
   LINEAR MAE -0.036 vs FPL (90% CI -0.085 to -0.008) but captain regret worse (73 vs 66); Wowza V1 MAE +0.613 (CI
   +0.526 to +0.739). V2 stays shadow; V1 frame parity uncertified.
9. **Verified real-price coverage.** Certified close (strictly pre-kickoff, <= 30 min, odds > 1) on prospective
   shadow rows: 525 / 736 = **71.3%** since 2026-10-23 (all-time 561 / 1,200 = 46.8%, mixes the pre-fix regime);
   7-day collection 1,838 fixture-markets, **89.9%** observed inside T-30m (`output/clv_collection_health.json`
   2026-11-23T03:58Z, guard PASS). 427 of 510 certified closes (to 11-22) are cross-book. Ledger price presence:
   bets 1,943 / 1,943, side 795 / 795, TOP-3 3 / 3, HT 15 / 84, player 7,473 / 47,426; no team / player ledger
   records the bookmaker or capture time.
10. **Unresolved blockers.** D1 (owner: refit Standard without unconfirmed archive cells); D3 (parity monitor
    reference); D4 (stored placeholders); D6 (book + capture time per tip); Data truth 9 (2 duplicated v91_shadow
    fixtures, fix in flight by another agent); Fantasy V1 frame parity; row-level provenance of Pro / v11 / exec;
    Pro weekly audit near-kickoff snapshot coverage.
11. **Statistically validated edge?** **NO PROVEN EDGE YET.** Clean Record (from 2026-11-19T06:41:48Z): 19 settled
    forward bets (TOP-3 staked 2, O/U 2.5 tips 5, side tips 10, HT 1, props 1); 12 / 12 forward cells
    NOT_ENOUGH_DATA (minimums 30 settled, 8 matchdays, 150 certified CLV); 2,127 cells in
    `output/clean_record_evaluation_cells.csv`. Price trader: every edge flag NO.
12. **100% real-data certification achieved?** **NO overall.** Per scope: history banks YES; NF team training YES
    (cell level); serving board YES; props training rows YES; side models YES (served leagues); Fantasy FPL inputs
    YES; ledgers YES (no synthetic / invalid price; missing kept missing); **Standard team training NO** (D1: 21
    rows, 34 unverified cells per payload); **Standard HT training NO** (D1: 19 rows); Pro / v11 / exec
    **UNVERIFIED** (not measured row by row). Stored `player_history.parquet` holds placeholders no model reads (D4).
