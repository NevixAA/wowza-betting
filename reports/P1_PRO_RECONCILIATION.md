# P1 item 9 - Pro reconciliation (2026-11-22)

## Repo and SHAs
- Repo: NevixAA/wowzaV9-Pro
- Base: `1a413050cecd5814794113c91b814a634543ae5b` (main at start)
- Fix: PR #18, branch `p1/registry-reconcile-asof`, commit `a7f98128`, merged as `4e96caab014f1140de590ad193753ffd8beea06c`
- v9 used for local tests: wowza-betting main `61e869877be7bb3c2e152660fc1907d0335615a0`

## Price-trader conclusions (preserved, not touched)
`output/price_trader/price_trader_verdict.json` (calculation_version price_trader-1.0.0, generated 2026-11-22 15:45 UTC):
MOVEMENT_SIGNAL_EXISTS=NO, MOVEMENT_BEATS_PLACEBO=NO, EXECUTABLE_PRICE_DATA_SUFFICIENT=NO,
FULL_GREEN_PROFITABLE_AFTER_COSTS=NO, TRADING_BEATS_HOLD=NO, EDGE_SURVIVES_SLIPPAGE=NO, EDGE_SURVIVES_COMMISSION=NO,
PROSPECTIVE_CONFIRMATION=INSUFFICIENT_DATA, REAL_TRADING_ENABLED=NO, CURRENT_STAKING_CHANGED=NO,
MAX_SUPPORTED_STAKE_BY_LIQUIDITY=THEORETICAL_ONLY. There is no new evidence. The cost, slippage, liquidity, limit, placebo
and prospective requirements are unchanged.

## Weekly audit (dispatched once)
Run 45062671382 (workflow_dispatch, 19:46 UTC, success). It recorded `output/weekly_audit.json` at 19:49 UTC on the pre-fix code:
worst=FAIL, PASS 28 / INFO 10 / WARN 3 / FAIL 2.

| Check | Before (run 45062671382) | After the fix (same store, local run of the merged code) |
|---|---|---|
| registry reconciliation | FAIL: 1/26 disagree, +927 rows (live_odds_snapshots 347,873 vs 348,800) | PASS: registry exact at its own generated_at 18:05:00Z |
| registry written since registry | (not measured) | WARN: 927 rows ingested after the registry, live_odds_snapshots +927 |
| registry populated tables | WARN: EMPTY combo_price_snapshots | PASS 25/26; INFO "empty by design: combo_price_snapshots" (SOURCE_REQUIRED, with reason) |
| movement single calculation version | WARN: 1.0.0, unversioned | PASS after snapshot_id dedupe (1.0.0); INFO raw split 1.0.0=5,552,496 / unversioned=25,052 |
| snapshot coverage main O/U near-kickoff | FAIL: T-1h 30%, T-30m 19%, T-10m 10% | unchanged (out of scope, see below) |
| clv schema coverage by level | WARN: CLEAN 32.5% | unchanged |

The next scheduled audit (Mon 06:15 UTC) will run the merged code on CI. This report does not claim that result before it exists.

## Root causes and fixes
1. **Registry/warehouse count mismatch.** Only `pro_collect` refreshes `system_registry.json`, but `pro_live_odds`,
   `pro_scout(_near)` and `pro_bet_builder` append to the store between collect runs. Measured on 1a413050: the registry
   matched every partition ingested up to its `generated_at`. All 927 extra rows were ingested later by Pro Live Odds.
   The 2026-11-16 FAIL (4 tables, scout_odds +231) fits the same cause because scout writes between collects, but it was
   not re-measured as-of. A second cause: in `pro_collect.yml` the registry
   refreshed **before** `Import builder evidence`, which appends combo rows in the same run.
   Fix: `registry.reconcile_as_of()` reads `ingested_at` from parquet footer statistics. It reports *wrong at its own
   timestamp* as FAIL. It reports *written since* as WARN, listing tables and counts. A partition without statistics
   also FAILs. The registry step now runs after the combo import. Nothing is rewritten.
2. **Empty combo_price_snapshots.** This is correct. `src/schemas.py` declares the table SOURCE_REQUIRED: no bookmaker
   same-game-builder price is collected in any repo, and multiplying singles does not give an executable price. The audit
   now shows the table as INFO "empty by design" with that reason instead of an unexplained WARN. It is still listed.
   Pro staking and combo ROI stay NOT MEASURED.
3. **Mixed calculation-version metadata.** One CI ingest on 2026-08-24 (gh32701766518-1) stamped 25,052
   `movement_observations` rows `unversioned` because v11 was not checked out. The loader bug was fixed earlier in
   `src/importers/v11_research.py::_calc_version`. Every one of those snapshot_ids was later re-ingested under 1.0.0, so
   after the mandatory dedupe the archive holds one version. The audit now gives its verdict on the deduped set and keeps
   the raw split visible. The append-only archive was not rewritten.
4. **Collector reconciliation failures (last 24h).**
   - `Pro Specials collect` 15:42 UTC failed: `special_prices.csv` was 106.5 MB, over GitHub's 100 MB limit. Another
     change moved the prices to the private store, and the 16:28 run succeeded. The rows logged in the failed run were
     not persisted.
   - `Pro Player V3 Forward Test` failed 11-20, 11-21 and 11-22 17:34 because the runner lost communication (memory
     starvation). PR #17 added swap, and the 18:47 run succeeded.
   - `collect_health.json` (18:03) shows `failures: []`.
5. **Near-kickoff snapshots inadequate.** T-1h 30%, T-30m 19% and T-10m 10% are below the floor. This is unchanged and
   is **read-only here**: capture schedules belong to the price/CLV coverage work running in parallel.

## Files changed
`src/pipelines/registry.py`, `src/monitoring/weekly_audit.py`, `.github/workflows/pro_collect.yml`,
`tests/test_registry_asof.py` (new, 4 tests).

## Tests
Local, before push: pytest **602 passed** (V9_LOCAL = fresh v9 main 61e86987, V11_LOCAL = v11 main). `python -m src.pro_tests`
reported all checks passed. actionlint was clean. PR #18 CI `tests` passed. Merge only after green.

## Blockers
- Near-kickoff coverage FAIL needs the capture-schedule owner.
- The registry is still refreshed only by pro_collect, so between collects it can be up to one collect cycle behind.
  This is now reported as WARN with counts. Refreshing it from every writer would add a shared-file push conflict.

## Production impact
None. These are audit and monitoring changes only. No collector, model, selection, staking or trading behaviour changed.

## Owner approval
Not needed for these changes. A decision is needed only if the registry must be refreshed by every writer instead of
reporting the lag.
