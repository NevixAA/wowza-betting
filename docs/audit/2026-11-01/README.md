# A-to-Z audit, 2026-11-01

Owner rule: every model, every script and every data file in the 5 Wowza repos must be free of
bugs and hold only true, reliable data. No defaults, no naming mismatches, no league mismatches.
Tracked by the GitHub milestone **Data Truth 100%**.

| File | What it is |
|---|---|
| `code_inventory.csv` | 399 code sites across the 5 repos that invent a value (A), guess a name (B) or hard-code a league (C). Each row has the exact line, what it affects and the honest fix. |
| `data_profile.csv` | Issues found in the 7,702 git-tracked data files (duplicates, unknown leagues, unresolved clubs, impossible odds/dates/probabilities, unsettled rows). Produced by `scripts/audit_data_files.py`. |

## Code inventory (399)

| Repo | A invented | B naming | C league |
|---|---|---|---|
| wowza-betting | 162 | 57 | 42 |
| wowza-dashboard | 35 | 10 | 6 |
| wowzaV9-Pro | 31 | 13 | 6 |
| wowza_v11 | 10 | 7 | 6 |
| wowza-exec | 5 | 7 | 2 |

## Order of work (no shortcuts)

1. Production data integrity (ledgers, banks, backtest inputs).
2. Code: all 399 sites fixed. A missing value stays missing, or comes from a real source.
3. Real-data collection for every value that had a constant (props features, appearances, heights,
   ratings). Quota needs go to the owner with exact call counts.
4. One retrain of every model on the clean data, plus validation that each model learns properly,
   per league (a model whose per-league outputs look alike is treated as a bug).
5. One backtest. Thresholds are set per league x market. There are no global thresholds: a league
   without enough evidence gets no staked tips until it has that evidence.
6. Re-enable the scheduled jobs and close the milestone. The ratchet stays on.
