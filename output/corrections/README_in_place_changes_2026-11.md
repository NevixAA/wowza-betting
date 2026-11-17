# In-place ledger changes, 2026-11-01..02: audit record

Rule (owner, CLAUDE_PROMPT_2): corrected records go beside the original, never silently overwrite
a committed ledger. The changes below were made in place before that rule was stated. These files
restore the original values beside the ledgers. The ledgers themselves are not touched again.

## output/player_ledger.csv: `player_ledger_in_place_changes_2026-11-01_02.csv`

One row per changed field: the ledger key, the commit, the reason, `value_before` (the original)
and `value_after`.

| Commit | Date (UTC) | What | Results changed |
|---|---|---|---|
| 34267dd2 | 2026-11-01 17:14 | regrade of props wrongly voided as "did not play" | VOID→LOSS 2,883, VOID→WIN 1,451 |
| 55a69de4 | 2026-11-01 17:31 | nightly DNP recheck, trial run | VOID→LOSS 8, VOID→WIN 1 |
| bd956cd7 | 2026-11-02 06:53 | nightly DNP recheck | VOID→LOSS 103, VOID→WIN 40 |
| c94164b7 | 2026-11-02 10:18 | nightly DNP recheck | VOID→LOSS 301, VOID→WIN 133 |

That is 4,920 results changed in total. The other field changes are annotations only, such as
`notes` = DNP_CONFIRMED, NO_STATS or "DNP (0 min)" on rows that stayed VOID, and the `pnl`,
`is_played` and `resolved_date` values that follow each changed result.

Why the results were changed: the original VOIDs were written on "did not play" without evidence.
The player-stats lookup failed on name and season mismatches, so a player who played was recorded
as not having played. Each correction was checked against API-Football fixture player stats.

The nightly recheck (`.github/workflows/oneoff_dnp_recheck.yml`) was paused on 2026-11-02 because
it writes in place. It may resume once it writes its corrections to a sidecar like this one.

## output/bets_ledger.csv: `bets_ledger_removed_backtest_rows_2026-11-01.csv`

Commit 7709b922 removed 4,115 rows with `source=backtest`, kept here in full with their original
values. These were 2020-2025 walk-forward simulation bets written on 2026-05-21, never settled,
with 646 duplicate keys. They were simulated rows inside the live production record. The 1,819 live
rows were not changed: they are byte-identical before and after.
