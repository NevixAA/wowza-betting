# 05 — Bookmaker price integrity (AU4)

Audit team AU4, 2026-11-01. Read-only. Audited code and committed data:

| repo | ref | SHA |
|---|---|---|
| v9 `NevixAA/wowza-betting` | origin/main | `fb0e9697ec8556d4dd3df57f31f8ed5415a5012d` (worktree `v9_audit_au4`, branch `audit/au4`) |
| Pro `NevixAA/wowzaV9-Pro` | origin/main | `e0ccc27b98f57c60ee28eedba81812e9b5c5c25b` (read via `git show`, not checked out) |
| v11 `NevixAA/wowza_v11` | origin/main | `36ed1cd71ecbeb837a1704980df72fbd7ab089ae` (not audited in depth; see Limitations) |

Every number below is produced by a script in `scripts/audit/` from committed files on that SHA.
Outputs are in `reports/au4_data/`. Commands (run from the worktree root, v9 venv):

```
python scripts/audit/au4_kickoff_index.py      # prop kickoff index from git history of player_tips.csv
python scripts/audit/au4_price_integrity.py    # -> price_integrity.json (+ csv)
python scripts/audit/au4_ledger_prices.py      # -> ledger_prices.json, ledger_close_provenance.csv,
                                               #    ledger_decision_time_availability.csv
python -m pytest tests/audit/test_au4_props_clv.py -q   # 8 passed, 4 xfailed (strict)
```

All heavy runs took `r2_brief/HEAVY.lock` (`au4_common.heavy_lock`).

## 1. Summary

* **Raw capture is clean where it records book and kickoff.** The two per-book, per-side feeds
  (`book_quotes/` from API-Football, 754,014 rows; `book_odds_snapshots` from OddsAPI, 380,330 rows)
  have **0 post-kickoff rows**, 0 exact duplicates, and 0 conflicting prices for the same key and
  timestamp. In `book_quotes`, O/U lines are ordered correctly on every one of 67,186 same-book,
  same-snapshot groups (Over 1.5 < 2.5 < 3.5, Under reversed). Same-book same-time overround:
  O/U median 1.069, BTTS 1.080, HT O/U 1.078. None is below 1.00.
* **Pairing across books makes fake arbitrages.** Best Over plus best Under from different books at
  the same instant gives an overround below 1.00 in 172 of 45,768 `book_quotes` O/U pairs and 178 of
  40,388 OddsAPI pairs. Production only does this for the BTTS entry price (P5-03), where the effect
  is immaterial.
* **The real damage is downstream, in the ledgers' closing prices.** 168 live new-format O/U 2.5
  bets (match dates 2026-07-16 to 2026-08-10) carry a `closing_odds` taken from
  `newformat_odds_history` rows captured before the 2026-08-10 parser fix. Those rows hold first-half
  or team-total prices under the full-match label. Their mean stored CLV is **+41.8%**. They raise
  the live bets_ledger mean CLV from **+1.01% to +5.76%** (P5-01).
* **Entry prices were real and available at decision time.** All 1,126 live O/U 2.5 ledger rows since
  2026-08-19 had their price standing at one or more books at `generated_at`. None was above the
  best standing price. The ledger price was the best available in 48.6% of rows; the median gap to
  the best price was 0.50% (mean 2.21%).
* **Historical (backtest) prices are a different object from forward prices.** Standard O/U 2.5
  backtests price at football-data `AvgC>2.5`/`AvgC<2.5`, the cross-book **average closing** price.
  That is a non-executable consensus close used as if it were the entry price. Side-market
  backtests fill missing prices with constants: O1.5 1.40 on 22,441 of 22,516 rows, O3.5 2.60 on
  22,439, BTTS 1.85 on 21,335. **0 placed backtest bets used a synthetic price**, which reproduces
  R2-C.
* **Player-prop prices** are the weakest data in the estate; see report 08. The history keeps no
  book for OddsAPI rows and no kickoff time. It has 2,420 post-kickoff rows (0.69% of those with a
  known kickoff), 2,639 placeholder prices at or below 1.05 (2,120 of them "to be carded"), and a
  dedup key that drops reverted prices. 99.94% of prop quotes are one-sided.

## 2. Inventory and classification of every price file

Fields: F=fixture id, L=league, T=teams, D=match date, K=UTC kickoff, B=book, M=market, S=side,
N=line/threshold, O=odds, TS=snapshot time. "-" = not stored.

| file | rows | F L T D K B M S N O TS | book identity | post-KO rows (kickoff known) | classification |
|---|---|---|---|---|---|
| `book_quotes/*.csv` (AF, 10-04→11-01) | 754,014 | F L T - K B M S N O TS | per book (9) | **0** / 754,014 | REAL, LIVE pre-match, per-book; executable at quoted book |
| `book_odds_snapshots(+_archive)` (OddsAPI eu, 08-19→) | 380,330 | - L T D K B M S - O TS | per book (25, incl. exchanges) | **0** / 380,330 | REAL, LIVE pre-match per-book; exchange rows (matchbook, betfair_ex) are back prices before commission |
| `standard_odds_history.csv` (predict, 08-09→) | 10,656 | - L T D - - M - - O TS | **not recorded**: first OddsAPI book listing each price | 17 / 9,686 (joined kickoff) | REAL, LIVE, single unnamed book per snapshot |
| `newformat_odds_dense.csv` (predict, 08-15→) | 16,136 | - L T D - - M - - O TS | not recorded (as above) | 27 / 15,516 | REAL, LIVE, single unnamed book |
| `standard_sidemarket_odds_history.csv` | 66,158 | - L T D K - M - - O TS | forward = Bet365 (AF id 8); backfill = OddsAPI historical main line | 0 / 51,391 | forward REAL/LIVE; 12,272 backfill rows (`T00:00:00Z`) = HISTORICAL; 271 BTTS-yes > 3.20 = first-half (mis-mapped) |
| `newformat_odds_history.csv` | 64,353 | - L T D K - M - - O TS | as above | 0 / 40,102 (10,799 forward rows have no kickoff) | forward REAL/LIVE; 13,452 backfill HISTORICAL; **4,074 forward O/U rows before 2026-08-10T08Z mis-mapped**; 933 BTTS-yes > 3.20 mis-mapped |
| `exchange_quotes/2026-10.csv` | 6,810 | F L T - K B M S N O TS | per exchange (4) | 0 / 6,810 | REAL, LIVE exchange back + lay; executable minus commission |
| `prop_quotes/*.csv` (10-14→) | 271,645 | - L T D - B M S N O TS | per book (8) | 1,537 / 172,619 (git kickoff index) | REAL per-book, **one-sided (99.94%)**; post-KO rows are in-play |
| `player_prop_odds_history(+_archive)` | 620,714 | - L T D - B* M - - O TS | *OddsAPI rows (257,233) = **best of books, no book**; AF rows = per book | 2,420 / 349,933 | REAL but mixed best-of-books and per-book; not a clean close source (report 08) |
| `predictions.csv` `odds_*` | rolling | - L T D K - M - - O - | first-listed book (O/U 2.5); BTTS = **median Yes / median No across books** | n/a | REAL LIVE entry inputs; BTTS fair prob = mixed-book de-vig (P5-03) |
| `bets_ledger.csv` `odds` / `closing_odds` | 5,921 | ledger | entry = first-listed book; close = last row of an archive (different source possible) | — | entry REAL executable; **close mixed-provenance, 168 invalid** (P5-01) |
| `side_bets_ledger.csv` `odds` / `closing_odds` | 652 | ledger | entry = OddsAPI (BTTS median); close = AF Bet365 or OddsAPI archive | — | entry REAL; close cross-source; 7 first-half BTTS closes, 5 parser-window O1.5 closes (P5-02) |
| `backtest_results_standard/newformat.csv` | 19,381 / 5,508 | hist | football-data `AvgC>2.5` first (else MaxC, B365C, PC, Avg, ...) / OddsAPI main-line backfill | — | **HISTORICAL CLOSING, NON-EXECUTABLE CONSENSUS** (average of books' closing prices used as the entry) |
| `backtest_results_over15/over35/btts.csv` | 22,516 each | hist | — | — | **SYNTHETIC** on 99.7% / 99.7% / 94.8% of rows (`price_synthetic`); 0 placed rows synthetic |
| `side_edge_devig_shadow.csv`, `edge_engine_shadow.csv` | 83 / — | research | — | — | RESEARCH ESTIMATE (shadow, not executed) |

## 3. Two-sided markets: same book, same time

The rule in this audit is that an overround or a de-vig may only combine an Over and an Under (or a
Yes and a No) from **the same fixture, market, line, book and snapshot**.

| source | pairing | pairs | overround median | p01 / p99 | < 1.00 | > 1.15 |
|---|---|---|---|---|---|---|
| book_quotes O/U | same book, same ts | 207,064 (99.2% of rows paired) | 1.069 | 1.034 / 1.108 | 0 | 0 |
| book_quotes O/U | **best of books** | 45,768 | 1.040 | 1.009 / 1.091 | **172** | — |
| book_quotes BTTS | same book, same ts | 69,556 | 1.080 | — | 0 | 0 |
| book_quotes BTTS | best of books | — | — | — | 27 | — |
| book_quotes HT O/U | same book, same ts | 97,318 | 1.078 | 1.047 / 1.103 | 0 | 0 |
| book_odds_snapshots O/U | same book, same ts | 106,266 | 1.070 | 1.007 / 1.139 | 0 | **36** |
| book_odds_snapshots O/U | best of books | 40,388 | 1.054 | — | **178** | — |
| book_odds_snapshots BTTS | same book, same ts | 63,546 | 1.080 | 1.005 / 1.134 | 230 | 369 |
| standard_odds_history O/U 2.5 | same ts, book unknown | 4,647 | 1.077 | 1.031 / 1.139 | 0 | 0 |
| newformat_odds_dense O/U 2.5 | same ts, book unknown | 5,837 | 1.067 | 1.030 / 1.139 | 0 | 0 |
| std side history (Bet365) O/U 1.5 / 2.5 / 3.5 / BTTS | same ts | 3,542 / 3,791 / 3,854 / 3,515 | 1.064 / 1.059 / 1.067 / 1.075 | — | 0 | 0 |
| NF history (Bet365) O/U 2.5 pre-fix / post-fix | same ts | 662 / 3,828 | 1.067 / 1.061 | — | 0 | 0 |

Readings:
* The pre-fix NF rows have a **normal-looking overround** (1.067). The mis-mapping cannot be seen
  from the pair: both sides came from the same wrong market. It shows in the line order instead:
  2.88% of NF same-snapshot Over groups violate Over 1.5 < 2.5 < 3.5 (111 of 3,854), and the
  post-fix share is **0 of 3,249**. The standard side file has 16 violations, all pre-fix; post-fix
  is 0 of 3,230. This confirms independently that R2's window `2026-08-10T08:00Z` is the boundary.
* The OddsAPI same-book pairs above 1.15 and the BTTS pairs below 1.00 come from **exchange "books"**.
  Example: `matchbook` OU25 Over 1.80 / Under 1.12, overround 1.45 (`price_bookodds_overround_gt_115.csv`).
  A consensus, best-price or de-vig over `book_odds_snapshots` must drop exchanges or adjust them for
  commission. Pro's `src/market/consensus.build_consensus` de-vigs each book independently before
  taking the median and quarantines invalid books. That is correct, and this report found no
  cross-book pairing in Pro.
* `standard_odds_history` and `newformat_odds_dense` store an over/under pair **without the book**.
  `src/predict.py` takes the first book that lists Over 2.5 and, separately, the first that lists
  Under 2.5, so the pair is same-book only when the first book quotes both sides. The overround
  distribution (no pair below 1.03) is consistent with same-book pairs, but the files cannot prove
  it.

## 4. Market and line mappings

| check | result |
|---|---|
| book_quotes O/U line ordering, same book + ts (Over) | 0 violations / 67,186 groups |
| book_quotes O/U line ordering (Under) | 0 / 67,456 |
| NF history Over ordering, all forward / post-fix | 111 / 3,854 (2.88%) → **0 / 3,249** |
| std side history Over ordering, all forward / post-fix | 16 / 3,290 → **0 / 3,230** |
| BTTS Yes > 3.20 (first-half signature) | book_quotes 6 / 69,751; std side 271; NF history 933 |
| BTTS labels | book_quotes and OddsAPI carry explicit Yes/No per book; same-book Yes/No overround is normal (median 1.080), so Yes and No are not swapped |
| prop thresholds: point → market (`sot`=0.5, `sot2`=1.5, `sot3`=2.5, `sot4`=3.5) | 0 mismatches / 63,884 SOT rows |
| prop SOT ladder sot < sot2 < sot3 < sot4, same player + book + ts | 75 violations / 20,239 groups (0.37%) in prop_quotes; 88 / 43,858 in history |
| prop placeholder prices ≤ 1.05 | prop_quotes 8,054 (cards 4,058, goals 3,961); history 2,639 (cards 2,120) |

R2's realised-outcome check (`output/price_provenance_health.json`) agrees on line meaning: forward
valid O/U 2.5 has implied 0.551 vs realised 0.551. AU4 did not re-derive realised outcomes for the
team files; this report's line-order test is an independent second method that does not need results.

## 5. Was the price available at decision time? (bets_ledger, live O/U 2.5)

Method (`au4_ledger_prices.py` §C): for every live O/U ledger row with `generated_at` from
2026-08-19 10:00 UTC onwards (the start of `book_odds_snapshots`), take each book's last observed
price at or before `generated_at` (+59 s for minute resolution), using LOCF because the snapshots
are a change-log. Fixture, date and side are matched exactly; both files use OddsAPI names.

| measure | value |
|---|---|
| rows | 1,126 (all found in book snapshots) |
| ledger price standing at one or more books | **1,126 / 1,126** |
| ledger price above the best standing price | **0** |
| books standing at decision time | median 4, mean 6.3, max 18 |
| ledger price = best standing price | 48.6% |
| best-price gap (best / ledger − 1) | median 0.50%, mean 2.21%, p99 13.7% |
| age of the matching standing quote | median 1.4 min |

Interpretation: entry prices are real and were on offer. They are the first-listed OddsAPI book's
price, not the best price. Line-shopping would have added a mean of about 2.2% in odds. That is an
execution choice, not a defect. It means the ledger ROI understates what best-price execution would
have achieved, and it is not a reason to believe the ledger is inflated.

## 6. Closing prices

### 6.1 bets_ledger (`update_results._closing_for_market`)

Production close = the last row, by `snapshot_ts`, of the **first** archive (order: NF dense, NF
history, std history, std side) that has the fixture, matched by exact or fuzzy team name. AU4
reproduced the production close value for **1,081 / 1,081** live rows it could locate.

Provenance of the stored `closing_odds` (`ledger_close_provenance.csv`):

| source | model_type | close label | n |
|---|---|---|---|
| live | new_format | VALID_ARCHIVE | 754 |
| live | new_format | **INVALID_NF_PARSER_WINDOW** | **168** |
| live | new_format | NOT_FOUND (JSON-sourced, since purged) | 69 |
| live | standard | VALID_ARCHIVE | 427 |
| live | standard | NOT_FOUND | 22 |
| live | unknown | NOT_FOUND | 3 |
| backtest | standard | VALID_ARCHIVE | 4 |

The 168 invalid closes: match dates 2026-07-16 → 2026-08-10, 127 UNDER and 41 OVER. Stored CLV mean
+41.8%, median +66.5%, range −66.9% to +132.6%. Examples: UNDER 2.30 closing 1.25 (CLV +84%);
OVER 2.32 closing 7.00 (CLV −66.9%). The model's direction and the bets' results are unaffected
(94 LOSS / 74 WIN, settled on goals). Only CLV is corrupted.

| live bets_ledger CLV (clv_pct, %) | n | mean | median | fixture-block 95% CI |
|---|---|---|---|---|
| as stored | 1,443 | **+5.76** | 0.00 | +4.45 … +7.22 |
| excluding the 168 invalid closes | 1,275 | **+1.01** | 0.00 | +0.32 … +1.79 |
| excluding invalid, NF, match ≥ 2026-09-01 | 531 | −0.35 | 0.00 | −1.02 … +0.30 |
| excluding invalid, standard, match ≥ 2026-09-01 | 346 | +0.18 | 0.00 | −0.49 … +0.84 |

The residual +1.01% before September is carried by a few outliers that reach +287%, for example a
2026-07-05 China SL OVER at 6.00. Even after the invalid closes are removed, live O/U CLV in the
valid-evidence period (from 2026-09-01) is **indistinguishable from zero** on both tracks.

**Close staleness.** The archives are change-logs: a row is written only when the price changes,
and the newer `book_quotes` heartbeat does not exist here. The production close row was written a
median 240 minutes before kickoff (78% more than 60 min, 41% more than 6 h, 0.09% after kickoff).
predict observes each fixture every 15–30 min until kickoff, so this is the last *change*, not
necessarily a stale price. The archive cannot prove the price still stood at kickoff, so a CLV
built on it is "CLV vs last observed change", not vs a certified close.

### 6.2 side_bets_ledger

| market | close label | n |
|---|---|---|
| btts | VALID_ARCHIVE | 358 |
| btts | **INVALID_FIRST_HALF_BTTS** (close > 3.20) | 7 |
| btts | NOT_FOUND | 15 |
| over15 | VALID_ARCHIVE | 129 |
| over15 | **INVALID_NF_PARSER_WINDOW** | 5 |
| over35 | VALID_ARCHIVE | 24 |

Side entry prices come from OddsAPI: the BTTS median across books, and O1.5/O3.5 from the first
listed book. Closes come from an API-Football Bet365 archive in at least 372 of 523 matched rows. **Side
CLV is cross-source by construction**: it measures OddsAPI-at-entry against Bet365-at-close.

### 6.3 Player props

Covered in report 08. In brief: the prop close is the last history row for (date, player, market)
regardless of book or kickoff. 82 of 3,878 closes with a known kickoff are post-kickoff rows.

## 7. Post-kickoff and stale quotes

| file | post-KO rows | note |
|---|---|---|
| book_quotes, book_odds_snapshots, std side history, NF history, exchange_quotes | **0** | kickoff filter applied at capture |
| newformat_odds_dense | 27 / 15,516 | 0.2–14 min after kickoff. `drift.snapshot` stamps `datetime.now()` at **write** time, after the OddsAPI fetch, BTTS per-event calls and enrichment. The price was fetched before kickoff; the timestamp is late (P5-04). |
| standard_odds_history | 17 / 9,686 | median 90 min after the joined kickoff; consistent with rescheduled kickoffs (the joined kickoff is the first seen). Not enough evidence of a defect. |
| prop_quotes | 1,537 / 172,619 | in-play prices (report 08) |
| player_prop_odds_history | 2,420 / 349,933 | in-play prices; plus stamping from a 2-hour response cache (P8-06) |

## 8. Bugs (Phase-11 fields)

| Bug ID | Sev | Repo | File/line | Root cause | Affected dates | Affected rows | Production impact | Probability impact | Selection impact | ROI impact (before/after, same sample) | Status | Smallest safe fix | Regression test | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU4-P5-01 | P1 | v9 | `update_results.py:261-286` (`_load_archives`/`_closing_for_market`) | Closing lookup reads raw archives without `price_provenance.label_market_mapping`, so forward NF O/U rows from before the 2026-08-10 parser fix (first-half / team-total markets under full-match keys) become closes | match 2026-07-16 → 08-10 (code path still unguarded) | 168 bets_ledger live NF rows; 5 side O1.5 | `closing_odds`/`clv_pct` wrong in the ledger; feeds every CLV reader (dashboard, `market_edge` CLV gates, `edge_engine_shadow`) | none | none (CLV not used for O/U selection) | P&L none. Live mean CLV +5.76% → +1.01% (same 1,443/1,275 rows) | HISTORICAL data, ACTIVE code (no new rows affected since parser fixed) | apply `label_market_mapping` in `_load_archives` and drop `market_mapping_valid==False`; expose a derived `clv_pct_valid` column, never rewrite the ledger | `tests/audit/test_au4_props_clv.py::test_ledger_close_ignores_nf_parser_window_rows` (xfail strict) | true production bug (evaluation) |
| AU4-P5-02 | P3 | v9 | same + `update_side_market_results` | Same mechanism for side markets; also BTTS-yes > 3.20 first-half rows not filtered at close | 2026-08 | 7 BTTS, 5 O1.5 side closes | side CLV | none | none | none on P&L | HISTORICAL | as P5-01 | as P5-01 | true production bug (evaluation) |
| AU4-P5-03 | P3 | v9 | `src/btts_odds.py:292-300` | BTTS entry `odds_btts` = median Yes across books, `odds_btts_no` = median No across books; `btts_fair_yes` de-vigs those mixed-book medians | since BTTS per-event fetch | all BTTS predictions | the fair prob is mixed-book | median 0.00 pp vs per-book median de-vig; abs diff > 1 pp in 57 / 14,899 snapshots | not measured (diff too small to matter) | not identifiable (immaterial) | ACTIVE | de-vig each book's own pair, then take the median (Pro `build_consensus` already does this) | — | suspicious, immaterial |
| AU4-P5-04 | P3 | v9 | `src/drift.py:216` (`ts = datetime.now()` in `snapshot`) | Archive timestamp is write time, not quote time; no `Z` suffix | all dense/std history | 27 dense + (≤17) std rows appear post-KO | wrong timestamps only | none | none | none | ACTIVE | stamp the fetch start time (as `book_odds_snapshots` already does with `_snap_ts`) | — | true production bug (data timestamp) |
| AU4-P5-05 | P3 | v9 | `src/predict.py:182-197`, `src/drift.py:126` | O/U price kept from the first-listed book with no book id; entry and close can be different books | all | all ledger O/U rows | CLV mixes books; ledger is not best price (best in 48.6%; mean gap 2.21%) | none | none | ledger ROI understates best-price execution | ACTIVE | record `bookmaker` with the price; compute CLV same-book | — | intentional design / data-coverage limitation |
| AU4-P5-06 | P3 | v9 | `src/predict.py:150` (per-book capture includes exchanges) | Exchange back prices (matchbook, betfair_ex) mixed with bookmakers, no commission and no liquidity filter | 08-19 → | 36 O/U pairs > 1.15; BTTS 369 > 1.15, 230 < 1.00 | any best-of-books or consensus reader | research only | none in production | n/a | ACTIVE (research) | flag `is_exchange`; exclude or commission-adjust in consensus/best price | — | research-only flaw |

## 9. Verdict inputs

* Forward entry prices used by production O/U selection are **REAL and executable** at the quoted
  (first-listed) book at decision time. There is no evidence of synthetic, post-kickoff or stale
  entry prices in the live O/U ledger.
* Live O/U **CLV as stored is INVALID EVALUATION** for 168 NF rows. With them removed, CLV from
  2026-09-01 is zero within CI for both tracks (NF −0.35% [−1.02, +0.30]; standard +0.18%
  [−0.49, +0.84]). That is no positive CLV evidence for the main market.
* Backtest ROI for the standard O/U track is measured at **average closing odds**. It is not a
  timestamp-valid entry price, so the absolute backtest ROI level is not executable evidence. The
  direction of the bias depends on line movement and is NOT IDENTIFIABLE from the committed data,
  because there are no historical opening prices.
* Side-market backtests on synthetic prices are diagnostic only. AU4 confirms that no placed
  backtest bet used one.

## 10. Limitations and blockers

* v11 market research was not audited beyond confirming that Pro's consensus de-vigs per book; the
  v11 local clone is 46 commits behind and was read only via `git show`.
* `book_quotes` starts 2026-10-04 and `book_odds_snapshots` 2026-08-19, so decision-time
  availability before 2026-08-19 is NOT IDENTIFIABLE.
* The "NOT_FOUND" closes (94) came from `odds_history_v9.json` entries that have since been purged
  (10-day purge), so they cannot be re-verified.
* Team-name matching uses AU4's own token matcher (`au4_clv_forensic.same_fixture`), which is
  independent of production. It reproduced production's close on 1,081/1,081 rows.
