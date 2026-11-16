# Side markets: data truth and live collection audit (2026-11-16)

Scope: the markets we do not model yet. These are match CORNERS (total and per team), CARDS (total and per team, yellow and red) and SPECIALS (correct score, a side scoring 4+, high/low match totals).
This audit is read-only. Nothing was built or enabled, no API-Football or OddsAPI call was made, and no workflow was triggered.
It contains statistics and counts only, with no prices from wowza-exec, OddsAPI or API-Football.
The fixture-level mismatch list is in `reports/R_SIDE_DATA_TRUTH_LIVE_2026-11-16_mismatches.csv`. It holds match statistics only.

## Plain-language summary

**Rule:** a market stays research-only until it has both verified history and live collection. Today no side market passes that rule.

1. **Bookmaker prices for corners and cards do not exist anywhere in our 5 repos.**
   - There is no history and no live capture, in any league.
   - The API-Football `/odds` answers that std/nf odds capture already receive include corners and cards bets. The capture code drops them on purpose (`_OU_EXCLUDE = ("Corner", "Card", ...)` in `scripts/capture_std_sidemarket_odds_forward.py` and `scripts/capture_nf_odds_forward.py`).
   - OddsAPI history in exec holds 11 market keys. None is a corners or cards market.
   - Verdict: **FAILED / NOT COLLECTED** in every league.
2. **Correct score and "a side scores 4+" have no history at all.** Live capture started on 2026-11-10, so we hold 6 days of quotes.
   - The OddsAPI correct-score probe (`ledger/probe_s.json`) returned no correct-score book in any league.
   - API-Football keeps no odds history.
   - Verdict: history **FAILED**, live **LIVE-PARTIAL**. 32 of 47 registry club leagues have quotes. The other 15 have none: the top 5, the 3 UEFA cups, Serbia, Ukraine, Saudi, China, Finland, Ireland and Portugal.
3. **High/low match totals (0.5 / 4.5 / 5.5 / 6.5) are the one specials market with real history.**
   - Exec OddsAPI history covers 29,631 events in 27 registry leagues from 2023-05 to 2026-11. That is about 89-99% of fixtures in 25 leagues, with up to 3 snapshots (T-48h, bet, close).
   - Verdict: history **PARTIAL**, live **LIVE-PARTIAL**.
4. **Match corners and cards are well covered for the 17 football-data leagues and the new-format leagues. They are missing for the COLLECTION_ONLY leagues, and they are only partly live.**
   - **Corners** (2022-2026 seasons): football-data and API-Football agree on 97.6% of the 20,642 fixtures that have both. 6 leagues are **VERIFIED**.
   - **Cards:** the two sources agree on yellow cards in only 93.7% of fixtures. The gap is mostly a counting convention: on teams with a red card, API-Football is often +2. Only Bundesliga 2 passes as VERIFIED.
   - The 15 COLLECTION_ONLY club competitions (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, the UEFA cups and others) hold corners only for the current season, from Pro, and **no team cards at all**. Both are **FAILED**.
   - National League corners are FAILED (7.8% coverage).
5. **Scores (the settlement data for specials) are in good shape.**
   - 100% of the 66,577 fixtures in the universe have a full-time score.
   - football-data agrees with API-Football on 99.98% of 25,759 fixtures in fd_raw, and on about 99.9% in fd_history.
   - 47 fd_history mismatches are listed; 17 of them are already in the project's correction files.
   - Scores are VERIFIED in 24 leagues. The rest are PARTIAL because they have one source or an unresolved mismatch.
6. **Live match statistics land on `main` and in Pro. They do not land in the data-truth-100 truth files.**
   - The repaired corners/cards files (`af_history_fullstats_fill.parquet`, `fd_raw/`) stop at 2026-11-09. They are only refreshed by the dispatch-only `data_truth_collect.yml`.
   - No scheduled job collects **team cards for the 17 standard leagues**:
     - `fd_history` drops the HY/AY/HR/AR columns;
     - Pro `team_stats` drops "Yellow Cards" and "Red Cards" from the answer it already fetches;
     - `af_history_extend` covers only the 15 new-format and AF-only leagues.
7. **Timing.** These figures are from actually delivered runs, not crons.
   - For upcoming fixtures, specials quotes were present at the tip moment (125 h for Standard, 153 h for New-Format) in **161 of 229 fixtures (70%)**.
   - The close is reached (last pre-kickoff quote at T-45m or later) by Pro's near loop in 26 of 26 fixtures. v9's std/nf captures reach it in only 13 of 63; their median last quote is 201 minutes before kickoff.
   - The cause: std_odds_capture has 52 cron firings a day but GitHub delivered only 4-9 runs a day, each a ~38-minute loop.

## Method, denominators and verdict rules

- **Denominator:**
  - Leagues: the 53 leagues of `config/leagues.json` (registry LR-1.2) on data-truth-100.
  - Fixtures: `output/strength/competition_fixtures.parquet`, the API-Football fixture list with statuses FT/AET/PEN. It holds 119,480 fixtures in 47 registry club competitions.
  - Seasons are API-Football seasons, and the verdict window is 2022-2026: 66,577 fixtures.
  - The 6 national-team competitions are outside this universe, so they are FAILED for every family.
- **Sources joined by `fixture_id`:**
  - API-Football `/fixtures/statistics`:
    - the fullstats fill;
    - the af_history provider rows.
  - football-data raw CSVs (`fd_raw`). These were joined through the fullstats crosswalk (18,970 rows), plus a same-date name-similarity match (6,792 rows); 10 rows stayed unmatched.
  - Pro `team_match_stats` (corners).
  - Player rows (`player_history`, sum of player cards per team). This is a check only, not an independent team count.
- **Impossible-value rules:**
  - a negative value;
  - match corners > 30;
  - match cards > 22;
  - a team red count > 5;
  - a team yellow count > 16;
  - a score total > 15.
  Rows where every corner and card value is 0 are flagged as suspected zero-fills.
- **VERIFIED** means all of the following hold. Missing any one gives PARTIAL; coverage under 50% gives FAILED.
  - every season in the window has ≥ 98% coverage;
  - an independent second source exists on ≥ 80% of fixtures;
  - that source agrees exactly on ≥ 95% (for cards, both yellow and red);
  - 0 impossible or flagged rows;
  - 0 duplicates.
  - For scores, the second source is fd_history, and there must be no unresolved mismatch.
- **No duplicates found:**
  - competition_fixtures: 0 duplicate `fixture_id`;
  - fullstats fill: 0;
  - fd_raw: 0 duplicate (league, date, home, away);
  - fd_history: 0 duplicate keys;
  - Pro team_match_stats: 0;
  - specials sidecars: 0 duplicate (fixture, book, bet, selection, capture, captured_at);
  - exec high/low rows: 0.

### Provenance

| Source | Repo / branch / commit | Fetch time in the data |
|---|---|---|
| `output/collect/af_history_fullstats_fill.parquet` (corners, yellow, red; source = "api-football /fixtures/statistics") | wowza-betting data-truth-100, `8b4ae90a7` (2026-11-09 16:11 UTC) | `matched_at` 2026-11-09T16:10:53Z (all 61,792 rows) |
| `output/collect/fd_raw/*.csv` + `manifest.json` (17 codes × 2022/23-2026/27, with sha256 per file) | data-truth-100, `2db0dfc27` (2026-11-09 15:44 UTC) | `fetched_at` 2026-11-09T13:12:45Z |
| `output/af_history.parquet` (provider rows with `fixture_id`: 2,592) | data-truth-100, `5caf4ab68`; main, `eec7123d2` (2026-11-15 13:27 UTC) | none (match date only); the main copy has no `fixture_id` column |
| `output/strength/competition_fixtures.parquet` | data-truth-100, `f9256e68e` (2026-11-15 23:14 UTC) | none |
| `output/fd_history.parquet` | data-truth-100, `42814ff2b`; main, `ebe731dcb` (2026-11-10 16:25 UTC) | none |
| `player_history.parquet` (partitioned) | main, `7696840eb` (2026-11-16 12:14 UTC) | none |
| Pro `data/season_2026_27/team_match_stats` | wowzaV9-Pro main, `7b4a5ff3` (2026-11-15 21:18 UTC) | `ingested_at` per row; latest 2026-11-15T21:15:17Z |
| v9 `output/specials_quotes/2026-11/*` (14 files) | wowza-betting main, `e6c7b9ebe` (2026-11-16 13:33 UTC) | `captured_at` + `provider_update` per row |
| Pro `scout_specials_quotes/2026-11/*` (15 files) | Pro main, `205c520f` (2026-11-16 11:37 UTC) | `captured_at` per row |
| Pro `output/specials/special_prices.csv` (re-reads the v9 sidecar; own calls only for uncovered fixtures) | Pro data-truth-100, `7344ddb1` (2026-11-15 15:06 UTC) | `captured_at` |
| exec `odds-history` (OddsAPI history, PRIVATE) | wowza-exec odds-history, `2ea8445` (2026-11-16 11:23 UTC) | `snapshot_ts`, `requested_at`; ledger `ledger/runs/*.json` |

### Defaults and fills found

- **Rule-derived zeros, not defaults.** The fullstats fill reads a provider `null` count in an answered team block as 0, under owner rule 2026-11-04 in `src/af_team_stats.py`. This affects red cards in about 49k matches, corners rarely, and yellows in 1,896.
  - The older af_history provider rows kept those as NaN, so their reds "agree" only 23% with the fill (1,382 NaN vs 0).
  - This is a parsing difference, not a data conflict.
- **Suspected false zeros.** 13 fixtures have every AF corner and card value at 0; 5 of them are in the window.
  - Two of them can be checked against football-data, and both are contradicted:
    - League Two `1039980` (fd: 11-4 corners);
    - Ligue 2 `1212691`.
  - They are flagged and block VERIFIED for their league.
- **Player-level fill.** `player_model/data_fetcher.py` writes `team_corners = 0` when statistics fail (docs/audit code inventory, row 311). So `player_history.team_corners` is not used as a corners source.
- **Pro dates.** Pro `team_match_stats` has 3 rows dated in the future: Swiss `1555431` (2026-11-28) and `1555438` (2026-12-06), and Ukraine `1565640` (2027-05-08). All 3 carry scores and stats ingested in October, and none is an FT fixture in the universe. These are likely abandoned or rescheduled matches, and the stats belong to a match that does not count.
- **Corners rule breach.** One fd_raw row breaks the corners rule: Scottish Championship 2024-08-02, 17 + 15 = 32 corners. It is plausible but needs a check.

## A. TRUTH

### A1. Match data (corners, cards, scores) per registry league, AF seasons 2022-2026

Column guide:
- `corn%` and `card%`: share of fixtures with both team values from any provider source.
- `corn_agree%`, `yel_agree%` and `red_agree%`: exact match of football-data against API-Football, on fixtures that have both.
- `fx`: fixtures in the window.

Headline numbers:
- fd vs AF on corners: 20,642 fixtures compared, 97.64% exact.
- fd vs AF on yellows: 93.70% exact; on reds: 99.58%.
- fd vs AF on scores: 25,759 fixtures compared, 99.98%.
- AF vs Pro corners: 4,570 fixtures, 98.12%. This is the same provider fetched later; the provider revises statistics after the match.
- AF vs player-row yellow sums: 22,671 fixtures, 87.9%.
- Impossible values: AF 0, fd 1, scores 0.
- AET/PEN fixtures in the window: 424. Specials must settle these on the 90-minute score; `output/corrections/history_score_90min.csv` holds 6 such rows.

| league | state | fx | corners | corn% | corn_agree% | corners_note | cards | card% | yel_agree% | red_agree% | cards_note | scores | scores_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Championship | ENABLED | 2408 | VERIFIED | 99.7 | 96.5 |  | PARTIAL | 99.7 | 93.1 | 100.0 | yellow agree 93.1% | VERIFIED |  |
| League One | ENABLED | 2391 | VERIFIED | 99.7 | 96.8 |  | PARTIAL | 99.5 | 92.5 | 99.9 | worst season 96.9%; yellow agree 92.5% | VERIFIED |  |
| League Two | ENABLED | 2402 | PARTIAL | 99.7 | 96.7 | 1 flagged row(s) | PARTIAL | 99.3 | 91.2 | 99.8 | worst season 94.3%; yellow agree 91.2%; 1 flagged row(s) | VERIFIED |  |
| Bundesliga 2 | ENABLED | 1331 | VERIFIED | 99.8 | 99.1 |  | VERIFIED | 99.8 | 97.1 | 99.8 |  | VERIFIED |  |
| Ligue 2 | ENABLED | 1488 | PARTIAL | 99.3 | 98.1 | 1 flagged row(s) | PARTIAL | 99.3 | 94.7 | 99.6 | yellow agree 94.7%; 1 flagged row(s) | VERIFIED |  |
| La Liga 2 | ENABLED | 2022 | VERIFIED | 99.7 | 98.2 |  | PARTIAL | 99.1 | 93.4 | 99.1 | worst season 92.0%; yellow agree 93.4% | VERIFIED |  |
| Serie B | ENABLED | 1669 | PARTIAL | 99.3 | 98.3 | worst season 97.4% | PARTIAL | 99.3 | 97.4 | 99.6 | worst season 97.4% | VERIFIED |  |
| Greek Super League | TRAINING_ONLY | 1021 | VERIFIED | 100.0 | 96.0 |  | PARTIAL | 100.0 | 92.3 | 99.5 | yellow agree 92.3% | PARTIAL | 2 unresolved fd-vs-AF score mismatch(es) |
| Denmark Superliga | ENABLED | 856 | PARTIAL | 99.9 | - | single source | PARTIAL | 99.9 | - | - | single source | VERIFIED |  |
| Austrian Bundesliga | ENABLED | 852 | PARTIAL | 100.0 | - | single source | PARTIAL | 100.0 | - | - | single source | VERIFIED |  |
| Romanian Superliga | ENABLED_PROVIDER_UNSUPPORTED | 1402 | PARTIAL | 92.3 | - | worst season 72.0%; single source; 1 flagged row(s) | PARTIAL | 92.2 | - | - | worst season 72.0%; single source; 1 flagged row(s) | VERIFIED |  |
| Sweden Allsvenskan | ENABLED | 1192 | PARTIAL | 100.0 | - | single source | PARTIAL | 100.0 | - | - | single source | VERIFIED |  |
| Norway Eliteserien | ENABLED | 1175 | PARTIAL | 100.0 | - | single source; 1 flagged row(s) | PARTIAL | 99.9 | - | - | single source; 1 flagged row(s) | VERIFIED |  |
| Finland Veikkausliiga | ENABLED | 863 | PARTIAL | 51.9 | - | worst season 0.0%; single source | PARTIAL | 50.8 | - | - | worst season 0.0%; single source | PARTIAL | 2 unresolved fd-vs-AF score mismatch(es) |
| Ireland Premier Division | ENABLED | 904 | PARTIAL | 66.4 | - | worst season 0.0%; single source | PARTIAL | 66.3 | - | - | worst season 0.0%; single source | VERIFIED |  |
| Argentina Primera Division | ENABLED | 2342 | PARTIAL | 100.0 | - | single source | PARTIAL | 99.7 | - | - | single source | PARTIAL | 2 unresolved fd-vs-AF score mismatch(es) |
| Brazil Serie A | ENABLED | 1861 | PARTIAL | 99.9 | - | single source | PARTIAL | 99.8 | - | - | single source | PARTIAL | 5 unresolved fd-vs-AF score mismatch(es) |
| Japan J-League | ENABLED | 1712 | PARTIAL | 97.0 | - | worst season 74.0%; single source | PARTIAL | 96.7 | - | - | worst season 74.0%; single source | VERIFIED |  |
| Mexico Liga MX | ENABLED | 1503 | PARTIAL | 99.9 | - | single source; 1 flagged row(s) | PARTIAL | 99.6 | - | - | worst season 96.5%; single source; 1 flagged row(s) | PARTIAL | 3 unresolved fd-vs-AF score mismatch(es) |
| China Super League | ENABLED | 1264 | PARTIAL | 99.3 | - | worst season 97.0%; single source | PARTIAL | 98.7 | - | - | worst season 96.7%; single source | PARTIAL | 2 unresolved fd-vs-AF score mismatch(es) |
| USA MLS | PAPER | 2582 | PARTIAL | 100.0 | - | single source | PARTIAL | 99.7 | - | - | single source | PARTIAL | 13 unresolved fd-vs-AF score mismatch(es) |
| Saudi Pro League | TRAINING_ONLY | 1262 | PARTIAL | 99.9 | - | single source | PARTIAL | 99.9 | - | - | single source | VERIFIED |  |
| K-League 1 | TRAINING_ONLY | 1138 | PARTIAL | 74.5 | - | worst season 31.0%; single source | PARTIAL | 73.3 | - | - | worst season 31.0%; single source | VERIFIED |  |
| World Cup 2026 | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition, outside the universe | FAILED | - | - | - | national-team competition | FAILED | outside the universe |
| Champions League | COLLECTION_ONLY | 1150 | FAILED | 13.0 | - | 13.0% of fixtures (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Europa League | COLLECTION_ONLY | 1042 | FAILED | 10.7 | - | 10.7% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Conference League | COLLECTION_ONLY | 1962 | FAILED | 8.8 | - | 8.8% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Premier League | COLLECTION_ONLY | 1620 | FAILED | 6.2 | - | 6.2% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| La Liga | COLLECTION_ONLY | 1639 | FAILED | 7.3 | - | 7.3% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Serie A | COLLECTION_ONLY | 1631 | FAILED | 6.7 | - | 6.7% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Bundesliga | COLLECTION_ONLY | 1313 | FAILED | 6.2 | - | 6.2% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Ligue 1 | COLLECTION_ONLY | 1393 | FAILED | 6.5 | - | 6.5% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Dutch Eredivisie | TRAINING_ONLY | 1383 | PARTIAL | 96.2 | 98.6 | worst season 94.1% | PARTIAL | 96.2 | 94.7 | 99.8 | worst season 94.1%; yellow agree 94.7% | VERIFIED |  |
| Portuguese Primeira Liga | TRAINING_ONLY | 1330 | VERIFIED | 99.4 | 98.3 |  | PARTIAL | 99.4 | 93.6 | 98.6 | yellow agree 93.6% | VERIFIED |  |
| Belgian First Division A | TRAINING_ONLY | 1378 | PARTIAL | 97.1 | 97.3 | worst season 92.7% | PARTIAL | 97.1 | 94.1 | 99.8 | worst season 92.7%; yellow agree 94.1% | VERIFIED |  |
| Scottish Premiership | TRAINING_ONLY | 1011 | PARTIAL | 97.6 | 97.6 | worst season 97.4% | PARTIAL | 97.6 | 92.7 | 99.4 | worst season 97.4%; yellow agree 92.7% | VERIFIED |  |
| Turkish Super Lig | TRAINING_ONLY | 1468 | PARTIAL | 98.1 | 99.0 | worst season 91.8% | PARTIAL | 98.1 | 93.3 | 99.3 | worst season 91.8%; yellow agree 93.3% | VERIFIED |  |
| Swiss Super League | COLLECTION_ONLY | 950 | FAILED | 8.2 | - | 8.2% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Czech Liga | COLLECTION_ONLY | 1231 | FAILED | 8.9 | - | 8.9% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Croatia HNL | COLLECTION_ONLY | 784 | FAILED | 8.2 | - | 8.2% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Serbia Super Liga | COLLECTION_ONLY | 1300 | FAILED | 8.0 | - | 8.0% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Ukraine Premier League | COLLECTION_ONLY | 1067 | FAILED | 7.1 | - | 7.1% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Poland Ekstraklasa | COLLECTION_ONLY | 1348 | FAILED | 9.2 | - | 9.2% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Cyprus First Division | COLLECTION_ONLY | 1078 | FAILED | 2.3 | - | 2.3% (Pro, current season only) | FAILED | 0.0 | - | - | no team cards in any source | PARTIAL | single source (AF) |
| Scottish Championship | TRAINING_ONLY | 817 | PARTIAL | 96.5 | - | worst season 94.5%; single source (fd); 1 flagged row(s) | PARTIAL | 96.5 | - | - | worst season 94.5%; single source (fd); 1 flagged row(s) | VERIFIED |  |
| National League | TRAINING_ONLY | 2426 | FAILED | 7.8 | - | fd has corners only from 2026/27; no AF stats | PARTIAL | 98.7 | - | - | worst season 94.5%; single source (fd) | PARTIAL | 1 unresolved fd-vs-AF score mismatch(es) |
| Scottish League One | TRAINING_ONLY | 818 | PARTIAL | 96.3 | - | worst season 93.2%; single source (fd) | PARTIAL | 96.5 | - | - | worst season 93.2%; single source (fd) | VERIFIED |  |
| Scottish League Two | TRAINING_ONLY | 798 | PARTIAL | 98.4 | - | worst season 92.9%; single source (fd) | PARTIAL | 98.4 | - | - | worst season 92.9%; single source (fd) | VERIFIED |  |
| Friendlies | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition | FAILED | - | - | - | national-team competition | FAILED | outside the universe |
| World Cup - Qualification Europe | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition | FAILED | - | - | - | national-team competition | FAILED | outside the universe |
| World Cup - Qualification CONCACAF | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition | FAILED | - | - | - | national-team competition | FAILED | outside the universe |
| World Cup - Qualification Intercontinental Play-offs | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition | FAILED | - | - | - | national-team competition | FAILED | outside the universe |
| Africa Cup of Nations | COLLECTION_ONLY | 0 | FAILED | - | - | national-team competition | FAILED | - | - | - | national-team competition | FAILED | outside the universe |

Verdict counts across the 53 leagues:

| Family | VERIFIED | PARTIAL | FAILED |
|---|---|---|---|
| Corners | 6 | 25 | 22 |
| Cards | 1 | 31 | 21 |
| Scores | 24 | 23 | 6 |

Where the gaps are:
- **Missing seasons.** These leagues have whole or partial seasons with no statistics in the fullstats fill:
  - Finland: 2024 has 0%, 2025 has 39%.
  - Ireland: 2022 has 0%, 2023 has 43%.
  - K-League: 2024 has 31%, 2025 has 47%.
  - Japan: 2026 has 74%.
  - Romania: 2022 has 72%.
  - The fill audit lists them as `lacking`, `not_cached` and `teams_not_in_answer`.
- **Current season in the truth branch.** The current season (2026) of the standard leagues is 80-87% covered by AF. The fill stopped on 2026-11-09, and the league's AF rows end on 2026-10-25..28.
- **Yellow-card disagreement.**
  - It is mostly a convention issue.
  - At team level, AF minus fd is 0 in 39,854 cases.
  - When the team had a red card, the difference is +2 in 471 cases and +1 in 111.
  - With no red card, it is +1 in 435 and -1 in 221.
  - A rule for second-yellow dismissals is an owner decision; no API call is needed.
- **Corner disagreement.** It is mostly ±1: AF minus fd is -1 in 352 team rows and +1 in 110.
- **Mismatch list.** All 1,790 fd-vs-AF corner, card and score mismatch fixtures are in the CSV. So are the 47 fd_history score mismatches: 17 are already in `output/corrections/history_*.csv` and 30 are unresolved. Some of the 30 may be errors of my date + name-similarity join, so each needs a fixture-level check.

### A2. Markets (bookmaker prices) per registry league

Where I looked:
- v9 `book_quotes`: bets ou / btts / ht_ou only.
- v9 `standard_sidemarket_odds_history.csv`: 15 market keys, all goals, BTTS or 1X2.
- v9 `specials_quotes`: 4 bets.
- v9 `exchange_quotes`.
- Pro: `scout_odds` (ou15/25/35, btts), `book_odds_snapshots`, `market_snapshots`, `live_odds_snapshots` (in-play only) and `scout_specials_quotes`.
- exec: `odds-history` (11 OddsAPI market keys), sharp paper (totals 1.5/2.5/3.5) and Cloudbet (team markets ou25/btts/over15/over35).
- v11 collects no side market.

What is held, and from when:
- **Corners O/U, cards O/U, team corners and team cards:** 0 rows anywhere. The only card market held is player-level (`player_to_receive_card`, 42,708 rows in exec), which is not this subject.
- **Correct score, side 4+ and high/low:** live only, since 2026-11-10.
  - v9 sidecar: 235,287 rows on 301 fixtures in 27 registry leagues.
  - Pro scout sidecar: 1,033,144 rows on 949 fixtures in 129 provider leagues; 107,321 of those rows are in 15 registry leagues.
  - Quality checks: 0 rows after kickoff, 0 invalid prices (≤ 1.0 or non-numeric), 0 duplicates.
- **High/low history:** exec OddsAPI `alternate_totals` at lines 0.5 / 4.5 / 5.5 / 6.5. That is 924,702 rows and 29,631 events from 2023-05-03 to 2026-11-08, with 0 invalid prices and 0 duplicates.

The verdict for correct score and side 4+ is FAILED in every league because no history exists. Live coverage is shown separately in the last columns:
- "played": fixtures that kicked off between 2026-11-10 and 2026-11-15 (international break, 54 fixtures);
- "upcoming": fixtures kicking off between 2026-11-16 14:00 and 2026-11-23 14:00.

| league | corners O/U | cards O/U | correct score | side 4+ | hi/lo totals | hi/lo hist events (vs fixtures %) | live CS fixtures | live CS cov played/upcoming % | srcs |
|---|---|---|---|---|---|---|---|---|---|
| Championship | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1850 (99.0) | 12 | - / 100 | v9 |
| League One | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1832 (99.2) | 24 | 100 / 100 | v9 |
| League Two | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1844 (99.5) | 24 | 100 / 100 | v9 |
| Bundesliga 2 | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1048 (98.8) | 9 | - / 100 | v9 |
| Ligue 2 | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1130 (97.6) | 9 | - / 100 | v9 |
| La Liga 2 | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1576 (99.0) | 17 | 100 / 54 | v9 |
| Serie B | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1298 (98.4) | 7 | - / 70 | v9 |
| Greek Super League | FAILED | FAILED | FAILED | FAILED | PARTIAL | 762 (95.5) | 7 | - / 100 | pro+v9 |
| Denmark Superliga | FAILED | FAILED | FAILED | FAILED | PARTIAL | 686 (98.8) | 6 | - / 100 | v9 |
| Austrian Bundesliga | FAILED | FAILED | FAILED | FAILED | PARTIAL | 689 (99.9) | 12 | - / 100 | v9 |
| Romanian Superliga | FAILED | FAILED | FAILED | FAILED | FAILED | - | 8 | - / 100 | v9 |
| Sweden Allsvenskan | FAILED | FAILED | FAILED | FAILED | PARTIAL | 898 (98.7) | 8 | - / 86 | v9 |
| Norway Eliteserien | FAILED | FAILED | FAILED | FAILED | PARTIAL | 889 (98.8) | 8 | - / 100 | v9 |
| Finland Veikkausliiga | FAILED | FAILED | FAILED | FAILED | PARTIAL | 655 (97.5) | 0 | - / - | - |
| Ireland Premier Division | FAILED | FAILED | FAILED | FAILED | PARTIAL | 622 (94.5) | 0 | - / - | - |
| Argentina Primera Division | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1490 (95.1) | 2 | - / - | v9 |
| Brazil Serie A | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1434 (99.0) | 11 | 100 / 60 | v9 |
| Japan J-League | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1196 (90.8) | 10 | - / 100 | v9 |
| Mexico Liga MX | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1155 (98.2) | 10 | 100 / 100 | v9 |
| China Super League | FAILED | FAILED | FAILED | FAILED | PARTIAL | 876 (94.4) | 0 | - / - | - |
| USA MLS | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1937 (99.1) | 8 | - / - | v9 |
| Saudi Pro League | FAILED | FAILED | FAILED | FAILED | FAILED | 147 (13.8) | 0 | - / 0 | - |
| K-League 1 | FAILED | FAILED | FAILED | FAILED | PARTIAL | 755 (89.2) | 6 | - / 100 | pro+v9 |
| World Cup 2026 | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - | - |
| Champions League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / - | - |
| Europa League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / - | - |
| Conference League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / - | - |
| Premier League | FAILED | FAILED | FAILED | FAILED | FAILED | events held, no hi/lo lines | 0 | - / 0 | - |
| La Liga | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / 0 | - |
| Serie A | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / 0 | - |
| Bundesliga | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / 0 | - |
| Ligue 1 | FAILED | FAILED | FAILED | FAILED | FAILED | events held, no hi/lo lines | 0 | - / 0 | - |
| Dutch Eredivisie | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1046 (94.0) | 18 | - / 100 | pro+v9 |
| Portuguese Primeira Liga | FAILED | FAILED | FAILED | FAILED | PARTIAL | 852 (80.4) | 0 | - / - | - |
| Belgian First Division A | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1023 (95.5) | 9 | - / 100 | pro+v9 |
| Scottish Premiership | FAILED | FAILED | FAILED | FAILED | PARTIAL | 782 (96.2) | 14 | - / 100 | pro+v9 |
| Turkish Super Lig | FAILED | FAILED | FAILED | FAILED | PARTIAL | 1159 (98.1) | 9 | - / 100 | pro+v9 |
| Swiss Super League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 6 | - / 100 | pro |
| Czech Liga | FAILED | FAILED | FAILED | FAILED | FAILED | - | 4 | - / 50 | pro |
| Croatia HNL | FAILED | FAILED | FAILED | FAILED | FAILED | - | 5 | - / 100 | pro |
| Serbia Super Liga | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - / 0 | - |
| Ukraine Premier League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | 0 / 0 | - |
| Poland Ekstraklasa | FAILED | FAILED | FAILED | FAILED | FAILED | - | 9 | - / 100 | pro |
| Cyprus First Division | FAILED | FAILED | FAILED | FAILED | FAILED | - | 3 | - / 50 | pro |
| Scottish Championship | FAILED | FAILED | FAILED | FAILED | FAILED | - | 9 | 100 / 100 | pro+v9 |
| National League | FAILED | FAILED | FAILED | FAILED | FAILED | - | 24 | 100 / 100 | pro+v9 |
| Scottish League One | FAILED | FAILED | FAILED | FAILED | FAILED | - | 10 | 100 / 100 | pro+v9 |
| Scottish League Two | FAILED | FAILED | FAILED | FAILED | FAILED | - | 10 | 100 / 100 | pro+v9 |
| Friendlies, WCQ ×3, AFCON | FAILED | FAILED | FAILED | FAILED | FAILED | - | 0 | - | - |

Notes on the high/low history:
- "vs fixtures %" is an event-count ratio: OddsAPI events with high/low lines divided by universe fixtures in the same date span. It is not a fixture-by-fixture join, which is why the best possible verdict here is PARTIAL.
- Romania has no OddsAPI sport key.
- The exchange capture adds high/low exchange quotes for Brazil Serie A only: 21 events, 527 rows, last on 2026-11-14.

### A3. Movements: more than one timestamped pre-kickoff snapshot per fixture-market

- **Corners and cards:** FAILED in every league, because there are no prices at all.
- **Correct score and side 4+:** FAILED as history in every league. The forward sidecars show movement:
  - v9: 54.3% of 38,752 fixture-markets have more than one pre-kickoff snapshot;
  - Pro: 78.4% of 120,100.
- **High/low history:** PARTIAL. 94.7% of exec events hold at least 2 labels (T-48h, bet, close).

| league | corners/cards | CS / side 4+ history | hi/lo history | hi/lo hist >1 snapshot % | live CS >1 snapshot % | live hi/lo >1 % | live side4+ >1 % | close <=45m % (played) |
|---|---|---|---|---|---|---|---|---|
| Championship | FAILED | FAILED | PARTIAL | 78.1 | 9.4 | 24.0 | 33.3 | - |
| League One | FAILED | FAILED | PARTIAL | 74.1 | 61.0 | 98.4 | 95.8 | 0.0 |
| League Two | FAILED | FAILED | PARTIAL | 73.1 | 61.6 | 79.2 | 87.5 | 16.7 |
| Bundesliga 2 | FAILED | FAILED | PARTIAL | 82.9 | 22.2 | 22.2 | 22.2 | - |
| Ligue 2 | FAILED | FAILED | PARTIAL | 75.8 | 100.0 | 100.0 | 100.0 | - |
| La Liga 2 | FAILED | FAILED | PARTIAL | 65.7 | 74.6 | 85.1 | 100.0 | 80.0 |
| Serie B | FAILED | FAILED | PARTIAL | 72.6 | 0.0 | 0.0 | 0.0 | - |
| Greek Super League | FAILED | FAILED | PARTIAL | 68.0 | 100.0 | 100.0 | 100.0 | - |
| Denmark Superliga | FAILED | FAILED | PARTIAL | 80.5 | 25.1 | 50.0 | 25.0 | - |
| Austrian Bundesliga | FAILED | FAILED | PARTIAL | 62.4 | 71.9 | 100.0 | 83.3 | - |
| Romanian Superliga | FAILED | FAILED | - | - | 34.8 | 100.0 | 100.0 | - |
| Sweden Allsvenskan | FAILED | FAILED | PARTIAL | 82.9 | 5.4 | 6.2 | 12.5 | - |
| Norway Eliteserien | FAILED | FAILED | PARTIAL | 86.8 | 32.7 | 57.1 | 100.0 | - |
| Finland Veikkausliiga | FAILED | FAILED | PARTIAL | 87.5 | - | - | - | - |
| Ireland Premier Division | FAILED | FAILED | PARTIAL | 85.0 | - | - | - | - |
| Argentina Primera Division | FAILED | FAILED | PARTIAL | 52.1 | 0.0 | 0.0 | 0.0 | - |
| Brazil Serie A | FAILED | FAILED | PARTIAL | 70.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| Japan J-League | FAILED | FAILED | PARTIAL | 80.8 | 0.0 | 0.0 | 0.0 | - |
| Mexico Liga MX | FAILED | FAILED | PARTIAL | 84.9 | 100.0 | 100.0 | 100.0 | 0.0 |
| China Super League | FAILED | FAILED | PARTIAL | 73.5 | - | - | - | - |
| USA MLS | FAILED | FAILED | PARTIAL | 77.5 | 57.3 | 51.6 | 68.8 | - |
| Saudi Pro League | FAILED | FAILED | PARTIAL (13.8% of fixtures) | 48.7 | - | - | - | - |
| K-League 1 | FAILED | FAILED | PARTIAL | 55.8 | 66.7 | 63.6 | 66.7 | - |
| Dutch Eredivisie | FAILED | FAILED | PARTIAL | 84.7 | 100.0 | 100.0 | 100.0 | - |
| Portuguese Primeira Liga | FAILED | FAILED | FAILED (1 label only) | 0.0 | - | - | - | - |
| Belgian First Division A | FAILED | FAILED | PARTIAL | 77.0 | 100.0 | 100.0 | 100.0 | - |
| Scottish Premiership | FAILED | FAILED | PARTIAL | 69.9 | 100.0 | 100.0 | 100.0 | - |
| Turkish Super Lig | FAILED | FAILED | PARTIAL | 77.6 | 44.4 | 44.4 | 44.4 | - |
| Swiss / Czech / Croatia / Cyprus / Poland (Pro scout only) | FAILED | FAILED | - | - | 0.0-32.5 | 0.0-97.2 | 0.0-77.8 | - |
| Scottish Championship | FAILED | FAILED | - | - | 100.0 | 100.0 | 100.0 | 100.0 |
| Scottish League One | FAILED | FAILED | - | - | 50.0 | 50.0 | 50.0 | 100.0 |
| Scottish League Two | FAILED | FAILED | - | - | 50.0 | 50.0 | 50.0 | 100.0 |
| National League | FAILED | FAILED | - | - | 100.0 | 100.0 | 100.0 | 100.0 |
| Top 5, UEFA cups, Serbia, Ukraine, national-team competitions | FAILED | FAILED | - | - | - | - | - | - |

## B. LIVE COLLECTION

The table below lists every scheduled workflow in the 5 repos that writes any of the three families today. Run figures come from `gh run list`. The "last row" date is read from the data, not from the run status.

Enablement:
- Repo variables found: wowza-betting has `DAILY_COLLECT_ENABLED=true` (set 2026-11-15 23:15) and `PHASE2_RETRAIN_ENABLED=true`; Pro has `PRO_SPECIALS_ENABLED=true` (set 2026-11-09).
- v11, exec and dashboard have no variables.
- Workflows with no `if:` guard on the job always run.

Delivered runs per day are counted over 2026-11-02..16.

| # | Workflow (repo / branch) | Cron | Enabled | Last 10 runs | Delivered runs/day (vs cron) | Family written | Last row in data | Rows per day, last 14 days | Leagues vs registry | Timing: tip / close |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `af_history_extend.yml` (wowza-betting / main) | `0 6 * * *` | yes, no guard | 10/10 success | 1/day, delivered 12:26-15:08 UTC (6-9 h late) | match corners, yellow and red per team, scores (`af_history.parquet` on **main**, no `fixture_id`) | match date 2026-11-14 (Brazil, Mexico); commit 2026-11-15 13:27 | by match date: 11-02 14, 11-03 4, 11-04 3, 11-05 5, 11-06 11, 11-07 37, 11-08 51, 11-09 4, 11-14 3 (international break 11-10..14); commits on 10 of 15 days | 15 of 53 (13 new-format + Saudi + K-League) | post-match; n/a |
| 2 | `predict.yml` (main), via the loader's football-data refresh of `fd_history.parquet` | 4 crons: 40/day Mon-Thu, 80/day Fri-Sun | yes, no guard | 10/10 success | 2-6/day delivered | match corners (no cards: the loader drops HY/AY/HR/AR), scores | fd_history commit 2026-11-10 16:25; corners last match 2026-11-09 (La Liga 2), 11-08 (most) | corner rows by match date: 11-02 11, 11-03 13, 11-04 7, 11-06 15, 11-07 61, 11-08 33, 11-09 2 | corners for 17 football-data leagues | post-match |
| 3 | `pro_team_stats.yml` (Pro / main) | `35 4,16 * * *` | yes | 10/10 success | 1-3/day | match corners (no cards: `STAT_MAP` excludes them), scores | `ingested_at` 2026-11-15T21:15Z, match date 2026-11-15 | by ingest day: 11-02 90, 11-03 31, 11-04 18, 11-05 41, 11-06 32, 11-07 112, 11-08 147, 11-09 11, 11-10 8, 11-12 1, 11-13 1, 11-14 20, 11-15 10 | 44 of 53, current season only; not National League or the Scottish lower 3 | post-match |
| 4 | `player_history_extend.yml` (main) | `40 3 * * *` | yes | 10/10 success | 1/day, delivered 10:40-12:06 | player cards (a team sum is a check only: 87.9% equal to AF team yellows) | match date 2026-11-15; commit 2026-11-16 12:14 | fixtures by match date: 11-02 25, 11-03 16, 11-04 7, 11-05 5, 11-06 26, 11-07 82, 11-08 79, 11-09 6, 11-13 1, 11-14 23, 11-15 4 | 25 leagues active (31 total) | post-match |
| 5 | `daily_injuries_lineups.yml` (main, writes data-truth-100) | `17 5 * * *` | `vars.DAILY_COLLECT_ENABLED == 'true'` | 2/2 success (1 dispatch, 1 schedule) | first scheduled run delivered 2026-11-16 13:40 (8 h 23 m late) | **scores** for all registry club leagues (`competition_fixtures.parquet`); no corners or cards | universe max match date 2026-11-15; commits 11-02..09 by data_truth_collect, then **none 11-10..14**, then 11-15 | 1 commit/day when it runs | 47 of 53 (all club competitions) | post-match |
| 6 | `data_truth_collect.yml` (data-truth-100) | **none (dispatch only)** | n/a | not scheduled | 0 | the truth files: `fdraw` (fd cards and corners), `fullstats` (AF corners and cards) | fd_raw fetched 2026-11-09T13:12Z; fullstats matched 2026-11-09T16:10Z | 0 since 11-09 | 17 fd + 28 AF leagues | n/a |
| 7 | `std_odds_capture.yml` (main) | 52 firings/day (`5 0,6,12,18`; `23,52 *`) | yes, `CAPTURE_BOOK_QUOTES=1` | 10/10 success | **4-9/day delivered**, each a ~38-minute loop (11-14: 7) | specials sidecar `specials_quotes/*_std.csv`: correct score, O/U 0.5/4.5/5.5/6.5, team O/U 3.5. **Corners and cards bets are dropped by `_OU_EXCLUDE`** | captured_at 2026-11-16 13:33Z | v9 sidecar total (std + nf) by capture day: 11-10 38,620; 11-11 4,495; 11-12 38,444; 11-13 9,318; 11-14 68,196; 11-15 48,789; 11-16 27,425 | 16 leagues (std track incl. National League and the Scottish four) | tip: 161/229 (70%) of fixtures whose tip moment passed had a quote by then (both sidecars); close: v9 13/63 fixtures with a quote ≤ 45 m, median last quote 201 m before kickoff |
| 8 | `nf_odds_capture.yml` (main) | 52 firings/day | yes | 10/10 success | 3-8/day delivered, ~37 minutes each | specials sidecar `*_nf.csv` | captured_at 2026-11-16 07:24Z | (included in #7) | 11 new-format leagues (incl. K-League); none for Finland, Ireland, China or Saudi | as #7 |
| 9 | `pro_scout.yml` + `pro_scout_near.yml` (Pro / main) | `50 */2` (12/day); `5 */3` (8/day) | yes | 10/10 success each (1 failure in 14 days for scout) | scout 3-4/day; near 2-4/day, ~330-minute loops | specials sidecar `scout_specials_quotes` (same 4 bets; zero extra calls) | captured_at 2026-11-16 11:34Z | 11-10 77,227; 11-11 59,558; 11-12 81,889; 11-13 132,956; 11-14 421,442; 11-15 248,490; 11-16 11,582 (all leagues; 10.4% are registry leagues) | 15 registry leagues (incl. Swiss, Czech, Croatia, Cyprus, Poland, National League, Scottish) | close: 26/26 registry fixtures ≤ 45 m, median 7.7 m |
| 10 | `pro_specials_collect.yml` (Pro / main → Pro data-truth-100) | `53 7 * * *` | `vars.PRO_SPECIALS_ENABLED == 'true'` | 7/7 success | 1/day, delivered 12:23-16:25 | re-reads the v9 sidecar; ≤ 250 own AF calls (1-3 fixtures/day) | commit 2026-11-15 15:06 (no 11-16 run yet at audit time) | special_price_runs: 11-11 16,776; 11-12 31,961; 11-13 38,322; 11-14 47,524; 11-15 5,761 rows | 1-2 leagues per run | not a timing source |
| 11 | `exchange_odds_capture.yml` (main) | `35 10 * * *` | yes | 10/10 success | 1/day, delivered 16:36-21:01 | exchange high/low lines 0.5-6.5 | 2026-11-14 16:43Z | 11-02 48, 11-03 96, 11-04 95, 11-05 48, 11-13 32, 11-14 32 | Brazil Serie A only | 1 snapshot a day |
| 12 | `oddsapi_history_collect.yml` (exec / main → odds-history) | **none (dispatch only)** | n/a | last 10: 9 success, 1 failure (11-15 13:24); 14 days: 14 success, 4 failure, 2 cancelled | dispatch | high/low **history** (no live) | odds-history `2ea8445` 2026-11-16 11:23Z; latest kickoff 2026-11-08 | n/a (historical) | 27 registry leagues with hi/lo lines | T-48h / bet / close labels |
| - | `sharp_paper_capture.yml` (exec) | `23 */2 * * *` | scheduled runs **skipped** by guard | 3 skipped, 2 success | 0 | totals 1.5/2.5/3.5 only, not side markets | - | - | - | - |

### Live-collection verdict per family

| Family | Verdict | Why |
|---|---|---|
| Match corners | **LIVE-PARTIAL** | Live on main (#1, 15 leagues) and in Pro (#3, 44 leagues, current season), plus fd corners via predict (#2, 17 leagues). Nothing writes into the data-truth-100 truth files: #6 is dispatch-only and last ran 2026-11-09. National League corners come only from fd from 2026/27. |
| Match cards (team) | **LIVE-PARTIAL** | Only #1 (15 new-format and AF-only leagues) collects team cards. **The 17 standard leagues and the 15 COLLECTION_ONLY leagues get no team-card collection.** #4 gives player cards only. |
| Scores (specials settlement) | **LIVE-OK** | #5 writes AF full-time scores for all 47 club competitions daily (first scheduled delivery 2026-11-16, 8 h late; a gap 11-10..14 before it was enabled). #2 adds fd scores. |
| Corners / cards prices | **NOT COLLECTED** | 0 rows anywhere. The answers that hold them are fetched and then discarded (#7, #8, #9). |
| Correct score, side 4+ | **LIVE-PARTIAL** | #7, #8 and #9 cover 32 of 47 club competitions. Not covered: the top 5, UEFA cups, Serbia, Ukraine, Saudi, China, Finland, Ireland and Portugal (no quotes in the window). Tip reach is 70%; the close is weak in v9 (13/63) and strong in Pro (26/26). |
| High/low totals | **LIVE-PARTIAL** | Same live capture as correct score. History (#12) is dispatch-only and ends at kickoff 2026-11-08. |
| Movements (all side markets) | **LIVE-PARTIAL** for specials, **NOT COLLECTED** for corners/cards | Multiple snapshots exist (v9 54.3%, Pro 78.4% of fixture-markets), but v9 misses closes because GitHub delivers only 4-9 of 52 daily firings. |

## C. GAPS and what closing them would take (proposals only)

**The rule:** without verified history AND live collection, a market stays research-only. Every item below is a proposal; nothing was built.

Budgets and costs used below:
- **API-Football:** 75,000 calls/day (Ultra, `api_usage_log.csv`). The highest usage seen on each of 2026-11-10..16 was 1,814-11,726, against 43,447-66,500 on 11-02..08 during backfills. The daily collector caps itself at 1,500 calls/day.
- **OddsAPI:** the exec ledger shows `remaining_end` 663,517 (run `43816556202-1-110224`, 2026-11-16 11:23Z) and `floor` 290,000, so 373,517 credits are spendable above the floor.
- **Actions minutes:** wowza-betting and Pro are public, so standard runners cost no minutes. exec is private: the odds-history collector used about 3,423 minutes in the last 14 days (20 runs, median 245 min).
- Fixture rates over the last 365 days of `competition_fixtures`: 16.1 fixtures/day in the standard leagues, 12.1 new-format, 12.1 COLLECTION_ONLY.

| # | Gap (family × market × league) | What would close it | API calls | Actions minutes | Licence |
|---|---|---|---|---|---|
| G1 | **Prices: corners O/U, cards O/U, team corners and team cards × all 53 leagues, live** | Extend the specials keep-list (`src/specials_quotes.py` and Pro `src/scout/specials.py`, kept identical by the existing parity test) with the exact full-time corners and cards bet names. The names must first be read from one stored `/odds` answer; none is committed today. | **0 extra**: the same `/odds` answers std/nf/scout already fetch | 0 extra (same runs); sidecar rows grow, size not measured | AF prices are written to **public** repos today (v9 `specials_quotes`, Pro scout sidecar). Check that the API-Football terms allow publishing bookmaker odds; if not, move the sidecars to private storage, as with exec odds-history. |
| G2 | **Prices: corners/cards history × 27 OddsAPI leagues** | OddsAPI historical event odds for corners and cards markets. **Assumption, not verified by our ledger:** the keys are `alternate_totals_corners` / `alternate_totals_cards`, and the price is 10 credits per market per region per snapshot. A 1-event probe is needed first and needs owner approval. | 29,631 events: **2 markets × 3 snapshots = 1,777,860 credits (476% of spendable)**; 2 markets × close = 592,620 (159%); **1 market (corners) × close = 296,310 (79%, fits)**. Break-even: 6,225 events at 2×3, or 18,675 events at 2×1. | exec (private): about the same order as the hi/lo history runs | OddsAPI: internal storage and training only, no redistribution (exec README) |
| G3 | **Correct score and side 4+ history × all leagues** | Cannot be bought: the OddsAPI correct-score probe returned no books in any league, and API-Football keeps no odds history. History can only accrue from the live sidecars. Side 4+ (team totals) on OddsAPI is unverified and would need a 1-event probe. | 0 (accrual) | 0 | as G1 |
| G4 | **Specials live × top 5, UEFA cups, Swiss, Czech, Croatia, Serbia, Ukraine, Poland, Cyprus (COLLECTION_ONLY), Saudi, China, Finland, Ireland, Portugal** | Add these leagues to the scout or std capture fixture lists. Pro scout already reaches 5 of them. | `reports/SPECIALS_QUOTES_COVERAGE.md`: 12 calls/day for 1 snapshot per fixture, **215 calls/day** at capture cadence (0.3% of 75k) | 0 (public) | as G1 |
| G5 | **Close (T-30m) for v9 std/nf specials × the 18 std leagues** | GitHub delivers only 4-9 of 52 daily firings. Either treat Pro `scout_near` (26/26 closes) as the close source for every registry league it watches, or extend its watch list to all registry leagues. | as G4 | Pro scout_near already runs ~330-minute loops (public, 0 cost) | - |
| G6 | **Team cards live × 17 standard leagues + 15 COLLECTION_ONLY + all Pro leagues** | (a) Add "Yellow Cards" and "Red Cards" to Pro `src/pipelines/team_stats.py` `STAT_MAP`: they are in the answer Pro already fetches, so **0 extra calls**. (b) Or add the standard leagues to `scripts/backfill_af_history.py` `LEAGUES`, or schedule the `fullstats` step. | (a) 0; (b) about **16.1 calls/day** (one `/fixtures/statistics` per finished standard fixture) | about 3-5 min/run, public | AF |
| G7 | **Live stats do not reach data-truth-100** (truth files frozen at 2026-11-09) | Schedule the dispatch-only `data_truth_collect.yml` steps `fdraw` + `fullstats`, or sync main's `af_history` / Pro `team_match_stats` into the truth branch after verification. | fdraw: 0 AF calls (football-data CSVs); fullstats: 0 calls when cached, else 1 per fixture | small, public | football-data: free CSVs (cite the site) |
| G8 | **Corners/cards history × COLLECTION_ONLY + National League + Scottish lower 3 + holes** (Finland 2024/25, Ireland 2022/23, K-League 2024/25, Japan 2026, Romania 2022) | (a) football-data mmz CSVs for E0, SP1, I1, D1 and F1 (corners and cards, **0 API calls**). (b) An AF `/fixtures/statistics` backfill for every window fixture still without AF stats. | (b) **26,171 calls** in total: National League 2,426, the UEFA cups 4,154, the top 5 7,596, other COLLECTION_ONLY 8,941, Finland 426, Ireland 305, K-League 304, rest < 110 each. That is about 1 day of headroom, or 18 days at the collector's 1,500/day cap. | public, small | AF; football-data |
| G9 | **Yellow-card convention × the 13 two-source leagues** (agreement 91-97%) | Owner decision on how second-yellow dismissals are counted (AF +2 on teams with a red), then re-score agreement. | 0 | 0 | - |
| G10 | **Flagged rows** | Check against a second source and correct or exclude: 13 all-zero AF fixtures (2 already contradicted by fd), 1 Scottish Championship row with 32 corners, 3 future-dated Pro stats rows, and 30 unresolved fd-vs-AF score mismatches (list in the CSV). | ≤ 47 AF calls, if refetched | 0 | - |
| G11 | **Ownership of `af_history_extend` writes on main without `fixture_id`** | Store the provider `fixture_id` in main's `af_history.parquet` (data-truth-100 already has the column). Without it, a main row cannot be proven against the provider. | 0 | 0 | - |

### What each side market needs before it can leave research-only

| Market | History | Live | Earliest possible |
|---|---|---|---|
| Corners O/U (match/team) | FAILED: no prices; match stats VERIFIED in 6 leagues only | NOT COLLECTED | G1 (0 calls) + G2 (credits) or accrual, plus G6/G7 for the match truth |
| Cards O/U (match/team) | FAILED: no prices; match stats VERIFIED in 1 league | NOT COLLECTED | G1 + G6 + G9, then accrual or G2 |
| Correct score | FAILED: no history exists to buy | LIVE-PARTIAL since 2026-11-10 | accrual only (Clean Record milestone length); G4 + G5 |
| Side scores 4+ | FAILED | LIVE-PARTIAL | accrual, or an OddsAPI team-totals probe (G3) |
| High/low totals | PARTIAL: 25 leagues at about 89-99% of fixtures, 3 labels | LIVE-PARTIAL | fixture-level join of the exec history to `fixture_id`, then G4 + G5 |

## Reproduction

The audit scripts are not committed (read-only audit), and the exec data was read only from a local scratch copy. The numbers come from:
- `competition_fixtures`, `af_history_fullstats_fill`, `af_history` (data-truth-100 and main), `fd_raw/` + manifest, `fd_history`, `player_history` (main) and Pro `team_match_stats`;
- the v9 main `specials_quotes`, Pro `scout_specials_quotes` and v9 `exchange_quotes`;
- exec `odds-history` (columns other than price, plus a validity count on price);
- `gh run list -L 400` and `gh variable list` for the 5 repos.

The data-truth-100 data was read at `258abac30`, v9 main at `a414cf983`, Pro main at `0ff931db` and exec odds-history at `2ea8445`.
