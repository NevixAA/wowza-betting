# Side-bets first-half CLV quarantine (fix5/data5, 2026-11-02)

Sidecars only. `side_bets_ledger.csv` and the odds archives are not changed; every original value stays
in its file. Consumers read the sidecar through `src/clv_corrections.py`
(`exclude_contaminated_clv` / `contaminated_mask`; a frame with a `market` column is matched against this
sidecar on match_date, league, home_team, away_team, market and side). Rebuild with
`python -m src.clv_corrections build-side`, which only reads the ledger and the archives.

| Sidecar | Rows | What it marks | Effect on displays |
|---|---|---|---|
| `side_clv_first_half_contaminated.csv` | 6 | `side_bets_ledger.csv` rows whose `closing_odds` is a first-half BTTS price stored as the full-match close: \|clv_pct\| > 50, BTTS YES, close 6.0-8.0 (above the 3.20 full-match ceiling in `price_provenance.BTTS_YES_FULLMATCH_MAX`). All were signalled on 2026-08-22, before the BTTS parser fix 8d9962c68 (2026-08-23). 5 new_format and 1 standard (La Liga 2 Oviedo v Leganes). The next largest stored \|clv_pct\| is 47.4. | Their CLV is left out of every CLV consumer that reads through `exclude_contaminated_clv`. P/L is unchanged. Side-bets mean CLV goes from -0.47% as stored (556 rows) to +0.25% (550 rows). |
| `side_clv_first_half_contaminated_archive_rows.csv` | 7 | The archive snapshots that supplied all 6 closes. This is the same evidence check as the 170 bets_ledger rows: same match date, canonical league and market key (`btts_yes`), resolver-matched clubs, and a price equal to the stored close. 6 rows are in `newformat_odds_history.csv` and 1 is in `standard_sidemarket_odds_history.csv`. | QUARANTINED: listed here and left in the archive. |
