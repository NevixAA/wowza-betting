# Special-market quotes: coverage per league

Generated 2026-11-10 11:30 UTC by `scripts/report_specials_coverage.py` from `output/specials_quotes/` and
`output/book_quotes/`, captures since 2026-11-10. Source: the std / nf odds captures' existing
/odds answers (zero extra API calls). See `src/specials_quotes.py`.

Total: 129 fixtures captured, 88 with Exact Score quoted, 29076 special rows, 10 of 32 covered leagues with Exact Score so far.

## Covered leagues

| league | track | fixtures_captured | fixtures_with_specials | fixtures_with_exact_score | exact_score_bookmakers | specials_rows |
|---|---|---|---|---|---|---|
| Argentina Primera Division | new_format | 0 | 0 | 0 | 0 | 0 |
| Austrian Bundesliga | new_format | 12 | 0 | 0 | 0 | 0 |
| Belgian First Division A | standard | 0 | 0 | 0 | 0 | 0 |
| Brazil Serie A | new_format | 11 | 0 | 0 | 0 | 0 |
| Bundesliga 2 | standard | 0 | 0 | 0 | 0 | 0 |
| Championship | standard | 0 | 0 | 0 | 0 | 0 |
| China Super League | new_format | 0 | 0 | 0 | 0 | 0 |
| Denmark Superliga | new_format | 0 | 0 | 0 | 0 | 0 |
| Dutch Eredivisie | standard | 9 | 9 | 9 | 5 | 2699 |
| Finland Veikkausliiga | new_format | 0 | 0 | 0 | 0 | 0 |
| Greek Super League | standard | 7 | 7 | 7 | 5 | 1810 |
| Ireland Premier Division | new_format | 0 | 0 | 0 | 0 | 0 |
| Japan J-League | new_format | 0 | 0 | 0 | 0 | 0 |
| K-League 1 | new_format | 0 | 0 | 0 | 0 | 0 |
| La Liga 2 | standard | 8 | 8 | 8 | 8 | 3067 |
| League One | standard | 12 | 12 | 12 | 8 | 4055 |
| League Two | standard | 12 | 12 | 12 | 8 | 4699 |
| Ligue 2 | standard | 0 | 0 | 0 | 0 | 0 |
| Mexico Liga MX | new_format | 10 | 0 | 0 | 0 | 0 |
| National League | standard | 12 | 12 | 12 | 7 | 4534 |
| Norway Eliteserien | new_format | 0 | 0 | 0 | 0 | 0 |
| Portuguese Primeira Liga | standard | 0 | 0 | 0 | 0 | 0 |
| Romanian Superliga | new_format | 8 | 0 | 0 | 0 | 0 |
| Saudi Pro League | new_format | 0 | 0 | 0 | 0 | 0 |
| Scottish Championship | standard | 4 | 4 | 4 | 8 | 1562 |
| Scottish League One | standard | 5 | 5 | 5 | 6 | 1598 |
| Scottish League Two | standard | 5 | 5 | 5 | 4 | 1312 |
| Scottish Premiership | standard | 14 | 14 | 14 | 6 | 3740 |
| Serie B | standard | 0 | 0 | 0 | 0 | 0 |
| Sweden Allsvenskan | new_format | 0 | 0 | 0 | 0 | 0 |
| Turkish Super Lig | standard | 0 | 0 | 0 | 0 | 0 |
| USA MLS | new_format | 0 | 0 | 0 | 0 | 0 |

## Registry leagues no existing capture reaches

These are COLLECTION_ONLY registry leagues: neither the std nor the new-format capture lists their
fixtures. No call was added for them; the owner decides. Fixtures a day come from the last 365 days
of `output/strength/competition_fixtures.parquet` (v9 data-truth-100);
calls a day = fixtures a day x 1 (one pre-kickoff snapshot each) or x 17.8 (the measured /odds calls per fixture of the existing captures since 2026-11-03).
Either option also needs fixture-list calls (one shared /fixtures?date call a day, or the captures'
per-league list calls).

| league | api_football_id | production_state | fixtures_per_day | calls_per_day_one_snapshot | calls_per_day_capture_cadence |
|---|---|---|---|---|---|
| World Cup 2026 | 1 | COLLECTION_ONLY |  |  |  |
| Champions League | 2 | COLLECTION_ONLY | 0.76 | 0.8 | 13.5 |
| Europa League | 3 | COLLECTION_ONLY | 0.74 | 0.7 | 13.2 |
| Conference League | 848 | COLLECTION_ONLY | 1.13 | 1.1 | 20.1 |
| Premier League | 39 | COLLECTION_ONLY | 1.01 | 1.0 | 18.0 |
| La Liga | 140 | COLLECTION_ONLY | 1.04 | 1.0 | 18.5 |
| Serie A | 135 | COLLECTION_ONLY | 1.04 | 1.0 | 18.5 |
| Bundesliga | 78 | COLLECTION_ONLY | 0.82 | 0.8 | 14.6 |
| Ligue 1 | 61 | COLLECTION_ONLY | 0.79 | 0.8 | 14.1 |
| Swiss Super League | 207 | COLLECTION_ONLY | 0.63 | 0.6 | 11.2 |
| Czech Liga | 345 | COLLECTION_ONLY | 0.74 | 0.7 | 13.2 |
| Croatia HNL | 210 | COLLECTION_ONLY | 0.49 | 0.5 | 8.7 |
| Serbia Super Liga | 286 | COLLECTION_ONLY | 0.78 | 0.8 | 13.9 |
| Ukraine Premier League | 333 | COLLECTION_ONLY | 0.64 | 0.6 | 11.4 |
| Poland Ekstraklasa | 106 | COLLECTION_ONLY | 0.82 | 0.8 | 14.6 |
| Cyprus First Division | 318 | COLLECTION_ONLY | 0.64 | 0.6 | 11.4 |

Totals (leagues with a known fixture rate): 12 calls a day for one snapshot per fixture; 215 calls a day at the existing capture cadence.
