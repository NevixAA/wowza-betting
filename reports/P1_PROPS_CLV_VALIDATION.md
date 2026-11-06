# P1 - Props CLV validation (Phase 2 part 7)

2026-11-06. Branch `phase2/perfclv`, built on `origin/data-truth-100` at `97e48c6a4`. This is an **integrity
audit** of the prop CLV records that exist. It is not a backtest. It does not compute a CLV mean, an ROI or
an edge, and it makes no claim about one. It counts records by status.

Reproduce (read-only on output/, about 50 s):

```
python scripts/props_clv_integrity.py      # -> reports/p1_props_clv_data/{clv_record_status.csv, status_by_market_month.csv, summary.json}
python -m pytest -q tests/test_props_clv_integrity.py
```

## Verdict

- **8,471 records. 4,280 have a stored close and a stored CLV value. Only 3,389 of them pass the identity and timing checks.**
  The other **891 (20.8% of the closed records) carry a CLV value they should not have**: 537 IDENTITY_UNPROVEN,
  226 TIMING_UNPROVEN, 93 CLOSE_AFTER_KICKOFF, 35 ENTRY_AFTER_KICKOFF.
- **Same-book CLV is provable for 871 records only**, all from October and November, because per-book quotes start
  on 2026-10-14. The other 2,518 valid records are **VALID_CROSS_BOOK**: the close is not proven to be the entry
  book's own price. Any CLV figure from them is a cross-book or best-of-books comparison, not same-book CLV.
- **Every committed close was written by production code (main) with the old rule**: the last history row for
  (date, player, market), with no kickoff filter and no book filter. The best-of-books close and its
  `clv_basis=best_vs_best` label exist on data-truth-100 only. **0** committed records carry that label.
- **The writer is fixed on this branch.** A record that fails identity or timing now gets its status and **no**
  CLV value. The close is taken from quotes strictly before kickoff (AU4-P8-03), and the edge verdict line in
  `clv_report` is removed (AU4-P8-01).

## Inputs

| Input | Rows / range | Note |
|---|---|---|
| `output/clv_records.csv` | 8,471 records, match dates 2026-07-14 to 2026-11-08 | last written 2026-11-02 01:56 UTC (the main snapshot on data-truth-100) |
| `output/player_prop_odds_history.csv` + `_archive/` | 629,862 rows, match dates 2026-06-28 to 2026-11-14 | `src.rolling_store.read_all` |
| `output/prop_quotes/*.csv` | 281,657 rows, match dates 2026-10-14 to 2026-11-14 | per book, both sides; only Over / Yes used |
| `reports/au4_data/prop_kickoff_index.csv` + `output/player_tips.csv` | 1,729 fixtures with one kickoff; **19 refused** (two kickoffs recorded) | no price file stores a kickoff |
| `output/player_ledger.csv` | provider `player_id` attached to 8,409 of 8,471 records | namesake check |

## How each record is checked

Shared code: `player_model/clv_integrity.py`. The audit uses `classify_stored`, and the writer uses `decide_close`
with the same rules. The checks run in order, and the first one that fails decides the status.

1. **Close recorded?** If not, the status is NO_CLOSE.
2. **Identity, player.** The record's name is normalised (`odds_fetcher._norm`) and matched inside the **same
   fixture** (`odds_fetcher.same_fixture`, both clubs). If a name maps to two provider ids in `player_ledger` on
   that date, it is a namesake and the status is IDENTITY_UNPROVEN (3 records). Provider ids cannot be compared
   for the quotes themselves: the committed history and quote files carry no `player_id` (the column was added on
   data-truth-100 on 2026-11-02). Name plus fixture is the strongest identity the files support.
3. **Identity, market, line and side.** The market string carries the line (`sot` = 1+, `sot2` = 2+). The history
   holds the Over / Yes side only, which is the side of every prop tip (`side = over` on all 8,471). From
   prop_quotes, only `yes` / `over` rows are read.
4. **Entry proven.** The entry price must appear for this player, market and fixture **at or before** `ts_bet`.
   This catches the borrowed-price records (AU4-P8-02).
5. **Close traced.** The stored close must appear for the same player, market and fixture.
6. **Kickoff.** One recorded kickoff is required. A kickoff recorded with two values is refused.
7. **Timing.** The tip must be logged before kickoff. The close price must first be quoted **before** kickoff.
8. **Book.** VALID_SAME_BOOK requires one named book that quoted the entry price by `ts_bet` **and** whose last
   quote before kickoff equals the stored close (`same_book_close`, on prop_quotes). Otherwise the status is
   VALID_CROSS_BOOK.

## Counts per status

| Status | Records | Reason (count) |
|---|---|---|
| VALID_SAME_BOOK | 871 | entry book's own last pre-kickoff quote = stored close |
| VALID_CROSS_BOOK | 2,518 | no per-book quotes for the fixture: 2,304; quotes exist but no book quoted the entry price and then closed at the stored close: 214 |
| IDENTITY_UNPROVEN | 537 | no history rows for this player / market / fixture: 409; entry price never quoted for it: 114; entry price quoted only after the tip time: 12; close price never quoted for it: 2 |
| CLOSE_AFTER_KICKOFF | 93 | close price first quoted at or after kickoff |
| ENTRY_AFTER_KICKOFF | 35 | tip logged at or after kickoff |
| TIMING_UNPROVEN | 226 | no recorded kickoff: 219; two kickoffs recorded: 4; close price quoted both before and after kickoff: 3 |
| NO_CLOSE | 4,191 | match played but never closed: 3,611; match on or after the file's last entry (2026-11-02): 580 |
| **Total** | **8,471** | |

The tables were checked a second way: 871 + 2,518 + 891 = 4,280 closed records, which equals the 4,280 rows with a
stored `odds_close`. The status-by-market and status-by-month tables each sum to 8,471.

The 409 "no history rows for this fixture" records were closed by production from a row of the same name and
date in **another** fixture, or from a fixture spelling that `same_fixture` does not accept. The production
close did not check the fixture until 2026-10-25.

## Per market

| Market | SAME_BOOK | CROSS_BOOK | ID_UNPROVEN | CLOSE_AFTER_KO | ENTRY_AFTER_KO | TIMING_UNPROVEN | NO_CLOSE | Total |
|---|---|---|---|---|---|---|---|---|
| sot | 511 | 1,499 | 315 | 37 | 31 | 103 | 2,136 | 4,632 |
| goals | 213 | 541 | 131 | 32 | 4 | 107 | 1,342 | 2,370 |
| cards | 83 | 332 | 71 | 13 | 0 | 6 | 463 | 968 |
| sot2 | 54 | 115 | 14 | 10 | 0 | 10 | 223 | 426 |
| assists | 10 | 30 | 6 | 0 | 0 | 0 | 23 | 69 |
| sot3 | 0 | 1 | 0 | 1 | 0 | 0 | 4 | 6 |

## Per month (match date)

| Month | SAME_BOOK | CROSS_BOOK | ID_UNPROVEN | CLOSE_AFTER_KO | ENTRY_AFTER_KO | TIMING_UNPROVEN | NO_CLOSE | Total |
|---|---|---|---|---|---|---|---|---|
| 2026-07 | 0 | 20 | 0 | 0 | 0 | 0 | 5 | 25 |
| 2026-08 | 0 | 737 | 109 | 24 | 9 | 19 | 33 | 931 |
| 2026-09 | 0 | 1,318 | 220 | 26 | 4 | 85 | 82 | 1,735 |
| 2026-10 | 784 | 418 | 181 | 30 | 22 | 119 | 3,202 | 4,756 |
| 2026-11 | 87 | 25 | 27 | 13 | 0 | 3 | 869 | 1,024 |

The month x market x status cells are in `p1_props_clv_data/status_by_market_month.csv`, and each record's status
and reason are in `clv_record_status.csv`. By tier: AVOID 7,797 (817 same-book), VALUABLE 450 (43 same-book),
PAPER 216 (11), MARKSMAN 8 (0).

**October has 3,202 played-but-never-closed records.** That is 67% of the month. They are a gap in the CLV
sample, not a status problem. A CLV figure read from October describes the third of the slate that did close.
Why production stopped closing them is UNIDENTIFIED in this audit; it needs the production close_out logs.

## Is "closing" best-of-books?

- **Committed records (main):** no. The close was the **last history row of any single book**, and the history
  kept one row per **price** (dedup key `match_date, league, match, player, market, odds` on main). A price row
  therefore names only the **first** book that quoted that price. Entry is best-of-books, but the close is one
  book. The comparison is cross-book in general, which is why the audit decides same-book on prop_quotes and not
  on the history.
- **data-truth-100:** entry and close are both best-of-books (`close_best_of_books`, 2026-11-02). That is a
  consistent basis, but it is **still not same-book CLV**. The writer labels such records VALID_CROSS_BOOK
  unless the best close is the entry book's own quote.
- **Limit common to both files:** quotes are stored the first time a (book, price) is seen. A book that went
  A, then B, then back to A keeps no row for the return to A. Its "last quote" is the latest recorded change, not
  necessarily its price at kickoff. How many records this affects cannot be identified from the stored files.

## Writer fix (this branch)

| File | Change |
|---|---|
| `player_model/clv_integrity.py` (new) | statuses, identity and timing rules, shared by the writer and the audit |
| `player_model/clv_tracker.py` | `close_out`: kickoff from the record (new `kickoff_utc`), else the caller's map, else the board; close only from quotes **strictly before kickoff**; entry must be proven; a failing record gets `clv_status` and no CLV; final statuses are not revisited, while TIMING_UNPROVEN is retried. `log_new_tips` writes the tip's `kickoff_utc` |
| `src/clv_capture.py` | columns `kickoff_utc`, `clv_status`, `clv_status_reason`; `mark_status()`; `capture_close(..., status, reason)`; `clv_report` returns `by_status`, and its verdict is descriptive only (it said "POSITIVE CLV - genuine edge signal" whenever the mean odds ratio was above 0) |
| `src/contracts/schemas.py` | CLV_RECORD: the three optional columns plus the rule `clv_only_when_valid` |
| `scripts/props_clv_integrity.py` (new) | this audit |
| tests | `tests/test_props_clv_integrity.py` (new, synthetic). AU4 strict-xfail tests for P8-01 and P8-03 now pass, so their markers are removed. The close_out tests in four existing test files now give the record a kickoff and a proven entry |

Committed rows are **not rewritten**. Their status is kept beside the file in `p1_props_clv_data/clv_record_status.csv`.
Consumers that read `clv_pct` from `clv_records.csv` should join on `bet_id` and keep VALID_* rows only. No
consumer does that yet.

## Open questions for the owner

1. For a future prop CLV readout, should only **VALID_SAME_BOOK** count, or VALID_CROSS_BOOK too with a label?
   This audit recommends same-book only. That leaves 871 records today, all from 2026-10-14 on.
2. The 3,611 played-but-never-closed records (3,202 in October) are a gap in collection. Should they be
   investigated before any props CLV is read?
