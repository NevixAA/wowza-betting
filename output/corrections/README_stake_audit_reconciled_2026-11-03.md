# stake_audit_reconciled.csv (derived, 2026-11-03)

A **derived sidecar** of `output/stake_audit.csv`. The original is append-only and has **not** been changed.

- Built by `scripts/audit/p2_stake_contract_reconcile.py --ref origin/main`, from the stake_audit.csv, stake_plan.csv, bets_ledger.csv and side_bets_ledger.csv committed at wowza-betting origin/main `12bd71a3`.
- It has one row per (stake_audit row, violation): 298 rows covering 188 audit rows. `audit_row` is the 0-based data row, and `csv_line` = `audit_row + 2`.
- Columns: `original_value`, `canonical_value`, `correction_reason`, `provenance` and `confidence`. Confidence is HIGH, MEDIUM, OWNER_DECISION or UNRESOLVED.
- `side_invalid` (110 rows): `YES/OVER` is mapped to the side that the tip's own `side_bets_ledger` row states. Confidence is HIGH.
- `odds_blank` (110 rows): the canonical value is the ledger price of the same tip, recorded at the ledger's own first-sight time. That may not be the price the stake planner saw, so confidence is MEDIUM. These rows had a zero stake, so the price changes no stake.
- `one_position_per_fixture` (78 rows): each row is classified as FIRST_STAKED_POSITION, TRACK_RENAME_RESTATEMENT, POLICY_VERSION_RESTATEMENT or SECOND_POSITION_*. A restatement is the same bet recorded again, not a second bet. The four SECOND_POSITION rows need an owner decision (see the report).

Full write-up: `reports/P1_STAKE_CONTRACT_RECONCILIATION.md`.
