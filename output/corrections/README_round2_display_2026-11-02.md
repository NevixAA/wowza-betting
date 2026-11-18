# Round-2 display corrections (fix4/sharp_clv, 2026-11-02)

Sidecars only. No ledger, archive or log row was changed or deleted; every original value stays in its
file. Consumers read the sidecars through `src/clv_corrections.py` (one place). Rebuild the sidecars with
`python -m src.clv_corrections build`, which only reads the ledgers.

| Sidecar | Rows | What it marks | Effect on displays |
|---|---|---|---|
| `clv_first_half_contaminated.csv` | 170 | `bets_ledger.csv` rows whose `closing_odds` is a first-half price stored as the full-match close (all new_format, \|clv_pct\| > 50, match dates 2026-07-05..08-09) | Their CLV is left out of every CLV consumer: the Dashboard CLV tab, the evidence panels and the sharp confirm/disagree split in the Telegram digests. P/L is unchanged. New-format mean CLV goes from +7.94% as stored to -0.14%. |
| `clv_first_half_contaminated_archive_rows.csv` | 179 | The `newformat_odds_history.csv` snapshots (file line numbers, taken 2026-07-13..08-09) that supplied 163 of those closes: same date, league and market, resolver-matched clubs, and price equal to the stored close. The other 7 closes came from a source that has since rolled off. | QUARANTINED: listed here and left in the archive. |
| `ht_ledger_unpriced.csv` | 47 | Settled half-time bets with no market price, settled at the model's own fair odds | Left out of every HT P/L display (weekly summary, Success Rates). HT P/L on 61 settled bets goes from -6.59u to +5.57u on the 14 priced bets. The unpriced bets' -12.16u is not shown. |
| `top_picks_frozen_after_kickoff.csv` | 25 of 219 | Top picks frozen after their match had kicked off (2026-10-20, 21, 22, 24, 25) | These picks are graded EXCLUDED. The original result is kept as `result_stored`. `freeze_day` now refuses a pick whose match has kicked off. |
| `sharp_ledger_restated.csv` | (written by the settlement work) | The sharp record restated at the signal-time price | The Telegram weekly summary and the daily results show only this restated record. While the file is absent, they show **UNDER RESTATEMENT** and no P/L. The booked figure was +22.22u (opening_odds) against -159.14u at current_odds on 2,010 settled rows. |
