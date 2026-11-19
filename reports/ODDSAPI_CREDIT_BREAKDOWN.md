# OddsAPI credit breakdown (report only)

As of 2026-11-02T16:00 UTC. Built by `scripts/report_oddsapi_credits.py` from `reports/oddsapi_credit_inputs_2026-11-02.json`; every input there is tagged *sourced* (a named log line or file) or *estimate* (chosen by this report). **No schedule was changed** - this is the owner's upgrade-vs-cut decision input only. Figures are rounded to 10 credits; the window is short (one day), so treat them as +/-20%.

## Answer

- Account burn at the current GitHub delivery: **~2590 credits/day** (remaining 97,049 -> 94,603 between 2026-11-01T17:03:13 and 2026-11-02T15:43:56).
- 94,603 credits left -> **~37 days** at that burn; the next reset is ~28 days away (assumed 2026-12-01T00:00), so the month is covered **only while GitHub keeps delivering predict about 5 times a day** (break-even below).
- What the account can afford to the reset: **~3340/day**.
- Upcoming, not yet in the measured burn: the sharp tracker's new hourly cadence (ops5 guard, up to 13 snapshots/day) projects to ~470/day against ~80/day measured.

## Credits per day by feature

Steady window 2026-11-01T17:05 -> 2026-11-02T16:00 UTC (23h, 24 runs; starts after the BTTS fetch-clock fix).

| Feature | Credits/day | Share | How measured |
|---|---:|---:|---|
| BTTS + alternate totals (predict) | 1980 | 79% | [btts] / [oddsapi_usage] credit count in each predict log |
| Player-prop odds (player_props, prop_odds_snapshot) | 190 | 8% | account remaining before/after each run's event calls |
| Totals board (predict) | 100 | 4% | estimate: 19 leagues x 1 credit per run (not logged) |
| Scores (update_results, daily_summary) | 90 | 4% | estimate: 'Fetching scores' lines x 2 credits |
| Exchange capture (exchange_odds_capture) | 90 | 3% | [exchange] credits line |
| Sharp snapshots (sharp_tracker) | 80 | 3% | leagues snapshotted x 2 markets (not logged before ops5) |
| **Attributed** | **2530** | | |
| Unattributed (account burn minus in-window attributed) | 150 | | see cross-check |

Exchange capture fires once a day and its run (2026-11-01 16:53, 87 credits) fell just before the window, so it is counted at its measured cost x 1/day; it is not part of the account drop below.

### By workflow (delivered vs requested)

| Workflow | Runs delivered/day | Requested/day (cron) | Credits/run | Credits/day |
|---|---:|---|---:|---:|
| predict.yml | 5.2 | fri_sun 80, mon_thu 40 | 398 | 2080 |
| player_props.yml | 7.3 | fri_sun 14, mon_thu 6 | 20 | 140 |
| sharp_tracker.yml | 2.1 | all 24 | 36 | 80 |
| update_results.yml | 2.1 | all 9 | 31 | 60 |
| prop_odds_snapshot.yml | 2.1 | all 6 | 22 | 50 |
| daily_summary.yml | 3.1 | all 5 | 8 | 30 |
| live_scanner.yml | 3.1 | all 8 | 0 | 0 |

GitHub delivers a small fraction of the requested predict fires. The BTTS 2h refresh clock suppresses nothing at this spacing (each run re-buys the whole ~175-fixture board at 2 credits each), so **predict's cost scales one-for-one with how many runs GitHub delivers**.

## Days left

- Remaining 94,603 at 2026-11-02T15:43:56 (sourced: predict.yml run 41474004000 log, [oddsapi_usage] requests_remaining).
- Burn since the assumed 2026-11-01 reset: ~3260/day (includes the 2026-11-01 morning, when GitHub delivered predict more often); steady burn ~2590/day.
- Days left at steady burn: ~37; days to reset: ~28.
- With the sharp projection added: ~2980/day -> ~32 days.
- `output/oddsapi_quota_health.json` (2026-11-01T18:58:31Z) reported 3700/day burn and 1335/day logged: its window still contained the pre-fix runs and its per-workflow table counts only BTTS and exchange (props, sharp, scores and the totals board print no credit line), which is why 'logged' was far below 'burn'.

## Break-even

Predict costs ~398 credits per delivered run (BTTS + alt totals + totals board). Everything else costs ~440/day. The month is covered while predict runs per day stay below **~7.3** (currently ~5.2); once the sharp tracker runs at its new cadence, below **~6.3**. Each extra delivered predict run per day costs ~11,300 credits to the reset. The conclusion is robust to the estimated totals-board and scores costs (together under 10% of burn); it is NOT robust to GitHub delivery, which nobody here controls.

## Options (no change made)

| Option | Saves/day (approx.) | Coverage cost |
|---|---:|---|
| BTTS one region (`BTTS_REGIONS=uk`) | ~990 | half the books in the BTTS consensus; fewer fixtures reach the min-books bar |
| Raise `BTTS_REFRESH_HOURS` 2 -> 6 (near-kickoff refetch unchanged) | 0 now, caps growth | none at today's ~4h spacing; limits cost if delivery improves |
| Alternate totals off (`ALT_TOTALS_MAX_CALLS=0`) | ~180 | Over 1.5 / 3.5 side tips lose their price again (owner decision 2026-11-01 option B) |
| Player-prop odds one region | ~100 | fewer prop books; props are paper-only |
| Share one prop-odds cache key between player_props and prop_odds_snapshot | not quantified (needs per-event logs) | none: stops the two workflows re-buying the same event inside the 2h TTL |
| Keep sharp at ~3 snapshots/day (revert ops5 cadence) | ~390 vs the projection | Pro 1X2 price age back to 12-14h every morning |
| Exchange capture off | ~90 | no exchange/lay research rows |
| Upgrade the plan | - | none; plan prices are not in the repo (owner input needed) |

## Cross-check

Route 1 (sum of the in-window runs' feature costs): ~2440/day. Route 2 (account remaining drop over the same runs): ~2590/day. Gap ~150/day (6%) - consumers that print no credit line (scores calls whose count is estimated, predict's totals board, any manual/dispatch run). The two routes agree within ~10%, so the per-feature split is usable for the decision.

## Inputs that most need checking

- Monthly allowance 100,000 and a calendar-month reset: *estimate* (code default + the 0 -> 99,694 jump on 2026-11-01). The plan page is not in the repo.
- Totals board 19 credits/run and scores 2 credits/call: *estimate* (not logged).
- One day of steady data, a Sunday-Monday: weekday boards are smaller, weekend delivery differs.
