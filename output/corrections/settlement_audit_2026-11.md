# Settlement corrections, 2026-11-02 (round 2, branch fix4/settlement)

These are corrected records written BESIDE the committed ledgers. No ledger row was rewritten. The
original values stay in the ledgers, and each sidecar holds the original and the restated value side by side.

## sharp_ledger_restated.csv (round 2 V1)

* **Source:** `output/sharp_ledger.csv` at origin/data-truth-100 40d7dda9.
* **Regenerate:** `python update_results.py --restate-sharp` (offline).
* **Rows:** 2,010. That is every settled WIN/LOSS row booked before this fix (`pnl_basis` blank).
* **Why:** every sharp signal is a price drop (opening_odds > current_odds on 100% of settled rows), so the opening price was already gone when the signal fired.
* **current_odds is the signal-time price.** It is frozen: `player_model.ledger.append_sharp_signals` writes it once and never updates a key. 0 of 2,302 keys changed opening_odds, current_odds or signal_date across all 678 commits of the file.
* **P/L:** +22.22u (+1.1% ROI) as booked, against -159.14u (-7.9% ROI) at the signal-time price. The two figures come from two independent computations: the sidecar writer and a separate script that walks the history.
* **Going forward:** `update_sharp_results` books P/L at current_odds and writes `pnl_basis=signal_price`. A row with no signal-time price gets its result graded and its P/L left empty.
* **Readers:** the Telegram weekly summary and the yesterday block read `pnl` as booked. Restating them is an owner decision; nothing was sent.

## player_ledger_card_regrade_2026-11.csv (round 2 P2, straight red card)

* **Regenerate:** `python update_results.py --restate-sharp`. The same command writes both sidecars.
* **Rows:** 36 settled `cards` rows that are LOSS in player_ledger. On the player's recorded cards (`player_history.parquet`, player_id + date) each shows a straight red and no yellow, and clv_records grades that a WIN.
* **The rule:** `update_results.card_market_result` is now the one rule. A yellow OR a red card wins, matching `clv_tracker._MARKET_HIT['cards']`. A test holds both graders to it.
* **Scope:** only rows whose grade the rule itself changes are listed. 9 more rows where player_history and the ledger disagree on the stats themselves are left to finding P1-5.
* **Impact:** all 36 are AVOID and unpriced, so P/L is 0.00u before and after. Only the cards hit-rate changes: these rows go from LOSS to WIN.

## player_ledger_restored_2026-11.csv / player_ledger_relogged_first_sight_2026-11.csv

See `player_ledger_restored_2026-11.md`. The restored file holds 111 rows lost in 6 bot commits ("auto: player props"), with their first-sight values. The re-logged file holds 57 rows that were logged again; 24 of them came back with different first-sight values. `output/player_ledger.csv` is not modified.

## player_ledger_null_rule_settled.csv (fix round 10 'settle', owner 2026-11-04)

* **Written by:** CI settlement (`update_results.py`, workflows update_results / daily_summary). It is appended each time a result is settled because the shared null rule read a provider null as 0. No row is ever rewritten, and the file is never built on a laptop because it needs the API.
* **The rule:** `player_model/player_stat_nulls.py`. A null per-player count is 0 when the player played (minutes > 0) AND the provider returned player statistics for the fixture. Otherwise it stays missing and the row stays pending. Settlement (`update_results._fixture_players`) now reads the sheet with this helper and keeps no copy of its own. The team 'Shots on Goal' read uses `src/af_team_stats.py`.
* **What changed:** before, a null shots-on-target settled LOSS only when the team total reconciled, and otherwise stayed pending for ever. A null goal, assist or card settled only through a complete event feed. Those rows now settle as LOSS on the sheet. Only a LOSS can come from the rule, since a 0 never wins.
* **Columns:** the ledger key, `result_before` / `pnl_before` (blank, because the row was pending), `result_after` / `pnl_after`, `rule_zero_stats` (the stats the rule read as 0), `minutes` and `reason`.
* **Event feed, kept:** for goals, assists and cards the feed is still read where the sheet's null would decide the result. If the feed does not name the player, the rule's 0 stands. If a complete feed (its goals add up to the final score) names him, the feed decides, as before, and the row is not a rule decision. If an incomplete feed names him, the two provider records disagree, so the row stays pending (`rule_zero_contradicted_by_events`).
* **Restored rows:** `player_ledger_restored_2026-11.csv` is already the beside-record. A rule decision there is marked in `graded_notes`.
* **Committed rows affected:** on 2026-11-04, 1,971 player_ledger rows were pending past their match date (743 of them more than 7 days). 235 have no player_id and stay pending (never matched by name). The number the rule will settle depends on the provider sheets, so it is UNIDENTIFIABLE offline. The file counts it once CI runs.
