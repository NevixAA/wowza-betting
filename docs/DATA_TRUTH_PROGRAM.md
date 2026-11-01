# Data Truth Program (owner goal, 2026-11-01)

**Goal:** every number the five Wowza repos produce comes from true data. Four rules, each enforced
by an automated check that turns CI red, not by a one-off clean-up:

| # | Rule | Enforced by (target) |
|---|---|---|
| 1 | No mismatched labels / naming | fixed `v9 28ef297b` |
| 2 | No silent defaults | fixed `v9 28ef297b` |
| 3 | No missing data | fixed `v9 f83f8d9c` |
| 4 | No wrong league relations | fixed `v9 fc6b8bab` |

"100% bug-free" cannot be proven for a codebase this size. What this program delivers is: every
known violation fixed or fail-closed, and a red check the moment a new one appears.

## Repos in scope

| repo | role |
|---|---|
| wowza-betting (v9) | production predict, tips, stakes, props, fantasy, dashboard source |
| wowzaV9-Pro | research, collectors, Prediction Lab |
| wowza_v11 | grading / registry |
| wowza-dashboard | generated slim deploy of the v9 dashboard |
| wowza-exec | execution (Cloudbet shadow) |

## Phases

1. **Audit** — one auditor per rule across all five repos, every finding adversarially verified
   (workflow `wowza-data-truth-audit`). Output: the findings register below.
2. **Fix by impact** — production tips/stakes first, then settlement/grading, training data,
   dashboard, research. One commit per fix, each with a test; production-changing fixes keep a
   rollback switch.
3. **Enforce** — add the four CI checks (`src/integrity/*`, extended) and wire them into
   `system_health.json`, so a regression is a RED check, not a rediscovery.
4. **Retrain where the data changed** — the R2-C safe joins and price filter change training data;
   models are retrained and pass the promotion gate before serving.

## Done so far

| date | item | commit |
|---|---|---|
| 2026-11-01 | Props slate: every fixture in the 4-day window (was next 5 per league) | v9 `fc978db1` |
| 2026-11-01 | Side markets priced on two-sided de-vig; one-sided fixtures get no tip | v9 `41da473b` |
| 2026-11-01 | System health counts only workflows still failing on main | v9 `a39243cd` |
| 2026-11-01 | Pro loads v9 config with v9's own `src` (Team Stats / Studies ImportError) | Pro `a94364da` |
| 2026-11-01 | Rest days and HT features land on their own fixture (positional misalignment) | v9 `28ef297b` |
| 2026-11-01 | Side markets: blind-fixture guard, same rule as O/U 2.5 | v9 `f83f8d9c` |
| 2026-11-01 | Dutch-as-Norway / Bundesliga-2-as-Denmark rows removed; safe joins + price filter ON | v9 `fc6b8bab` |

## Findings register

Status 2026-11-01 evening: 55 of 56 fixed (each with a test that fails on the old code), 1 deferred (#42). Training-data fixes take effect at gated retrains; the promotion gate decides what serves.

Audit 2026-11-01 (workflow `wowza-data-truth-audit`): 59 candidates, 56 confirmed by an adversarial verifier, 3 refuted. Several overlap (props seasons appear three times). Ranked by impact, then severity.

| # | sev | impact | rule | repo | where | defect | status |
|---|---|---|---|---|---|---|---|
| 1 | HIGH | tips/stakes | defaults | wowza-betting | `src/feature_engineering.py:786` | build_upcoming_features assigns home/away_rest_days positionally from a date-sorted frame (combined_rd.tail(len(df))), so live fixtures receive another fixture's rest days, or NaN, and these feed the NF O/U 2.5, BTTS/... | fixed `v9 28ef297b` |
| 2 | HIGH | tips/stakes | defaults | wowza-betting | `src/predict.py:497` | attach_ht_features returns upcoming rows re-sorted by (date, league, home, away), but predict assigns them to feat by position, so the HT feature set, and therefore p_ht_over05/p_ht_over15, belongs to a different fixt... | fixed `v9 28ef297b` |
| 3 | HIGH | tips/stakes | defaults | wowza-betting | `pipeline.py:119` | _generate_side_bets has no blind-fixture guard: BTTS/O1.5/O3.5 tips are tiered, staked and top-picked on fixtures with no rolling-form history, where the form features are median-imputed, while the O/U 2.5 path forces... | fixed `v9 f83f8d9c` |
| 4 | HIGH | tips/stakes | leagues | wowza-betting | `src/data_loader.py:1406` | load_all_matches appends output/backtest_all_leagues.csv. That file stores every Dutch Eredivisie fixture a second time as 'Norway Eliteserien' and every Bundesliga 2 fixture a second time as 'Denmark 1st Div', so 45%... | fixed `v9 fc6b8bab` |
| 5 | HIGH | tips/stakes | leagues | wowza-betting | `src/data_loader.py:951` | _enrich_with_api_shots still serves the legacy (home, away, date) merge without a league key and with a positional zip over af_history. af_history still contains the Dutch-as-Norway and Bundesliga 2-as-Denmark 1st Div... | fixed `v9 fc6b8bab` |
| 6 | MEDIUM | tips/stakes | naming | wowza-betting | `src/feature_engineering.py:602` | Predict-time form lookup resolves OddsAPI club names with `team_names.resolve`, which does not consult `_RESULT_ALIASES`. Clubs whose only history spelling differs from OddsAPI therefore get no rolling form, and the b... | fixed `v9 11efa358` |
| 7 | MEDIUM | tips/stakes | defaults | wowza-betting | `src/feature_engineering.py:777` | combined_ht_goals_avg treats a missing team HT history as 0 HT goals (fillna(0) on each side), so a fixture with one unknown side gets a low, plausible-looking value instead of NaN; this feeds the Standard legacy, BTT... | fixed `v9 2acd9861` |
| 8 | MEDIUM | tips/stakes | missing | wowza-betting | `src/feature_engineering.py:664` | The serving-time features home/away_ht_over05_rate and ht_over15_rate match teams by exact OddsAPI name and skip the league-scoped name resolver, so they are NaN for most Standard fixtures and get imputed. The models ... | fixed `v9 11efa358` |
| 9 | MEDIUM | tips/stakes | missing | wowza-betting | `pipeline.py:119` | Over 1.5 and Over 3.5 side tips need odds_over15 and odds_over35 on the predictions row. Those columns come only from the OddsAPI bulk totals response, which rarely quotes those lines. Rows without a price are dropped... | fixed `v9 72f4e69d (targeted, owner option B)` |
| 10 | MEDIUM | tips/stakes | missing | wowza-betting | `src/feature_engineering.py:787` | Serving-time rest days use only the last 1,000 history rows across all leagues and raw OddsAPI names, so 64% of live Standard fixtures have no home_rest_days and are imputed. This is still unfixed on main. | fixed `v9 28ef297b` |
| 11 | LOW | tips/stakes | naming | wowza-exec | `src/bot.py:118` | wowza-exec joins v9's top_picks_log by `day/home/away/market` using the candidate's UTC match or kickoff date, but v9 writes `day` as the TIP DAY (local album day, src/tip_day). Late-evening Americas fixtures get top_... | fixed `exec 5667076` |
| 12 | LOW | tips/stakes | defaults | wowza-betting | `src/feature_engineering.py:807` | The legacy train/serve builders invent cold-start values that serve NF O/U 2.5 and side/HT models: SOT ratio 0.35/0.33, home_advantage 0.55, away_home_adv_factor 0.45, league away-goal average 1.1. Nothing marks a row... | fixed `v9 2acd9861` |
| 13 | LOW | tips/stakes | defaults | wowzaV9-Pro | `src/studies/league_results.py:32` | League ratings 1-10, which scale v9 stakes, use a hard-coded SEASON_FROM='2026-08-01' as the 'season' window, so from the 2027/28 season the rating silently pools two seasons, and for calendar-year leagues it has neve... | fixed `Pro 4207c163` |
| 14 | LOW | tips/stakes | missing | wowza-betting | `src/predict.py:122` | A league whose OddsAPI call returns non-200 or raises is logged at DEBUG and skipped. The health check alerts only when every league fails, so one league can drop off the tip board with no alert and no record of which... | fixed `v9 94ba6bd4` |
| 15 | HIGH | settlement | missing | wowza-betting | `update_results.py:99` | Sharp-ledger settlement for the five football-data 'std' leagues reads a hard-coded 2025/26 CSV URL and has no API-Football or OddsAPI fallback, so no sharp tip in League One, League Two, La Liga 2, Bundesliga 2 or Li... | fixed `v9 64d8b7f8` |
| 16 | HIGH | settlement | missing | wowza-betting | `update_results.py:595` | When API-Football returns no player stats for a finished fixture, every prop on that fixture is graded VOID 'DNP' permanently. The cause can be an error, an empty body or stats not yet published. The rows are never re... | fixed `v9 cbf29c9d` |
| 17 | MEDIUM | settlement | naming | wowza-betting | `update_results.py:420` | Production settlement of bets_ledger, side_bets_ledger, sharp and live-signal rows matches club names with the substring/first-word `_names_match`, not `src.team_names.resolve_result`, so the API-Football/football-dat... | fixed `v9 30f8d95f` |
| 18 | MEDIUM | settlement | naming | wowza-betting | `update_results.py:279` | The closing-price lookup for CLV joins ledger names (OddsAPI spelling) to archive names (API-Football spelling) with `_norm` equality and then `_names_match`. `_norm` does not fold 'ø' and the matcher has no alias tab... | fixed `v9 30f8d95f` |
| 19 | MEDIUM | settlement | naming | wowza-betting | `update_results.py:456` | The player-prop settler only understands market labels goals/sot/cards/assists, while the props producer writes sot2/sot3 (and can write goals2/sot4). `_resolve_player_market` returns '' for those labels, so none of t... | fixed `v9 cbf29c9d` |
| 20 | MEDIUM | settlement | naming | wowza_v11 | `src/results.py:293` | v11 grades v9 board fixtures (OddsAPI names) against football-data and af_history names with its own containment-only `resolve`, which has no alias table. Many recurring clubs never resolve, so their fixtures are ungr... | fixed `v9 073847fe + v11 c341aa3` |
| 21 | MEDIUM | settlement | defaults | wowza-betting | `update_results.py:121` | fetch_scores_fd filters New Format football-data results to seasons containing '2025' or '2026', so from the 2027 calendar seasons every NF result is dropped and settlement silently falls back to OddsAPI (3-day window... | fixed `v9 64d8b7f8` |
| 22 | MEDIUM | settlement | missing | wowza-betting | `update_results.py:370` | 52 live O/U 2.5 tips with match dates from 2026-05-17 to 2026-10-09 are still unsettled. The last-resort API-Football fallback looks up results in today's season, not the match's season, so a tip that misses its prima... | fixed `v9 46f4b0f9` |
| 23 | MEDIUM | settlement | missing | wowza-betting | `update_results.py:554` | Prop settlement hard-codes the European rule (month < 7 means yr-1) for the API-Football season, which ignores the canonical season resolver. Japan J-League props (season 2027 since 2026-07) have never settled, and ca... | fixed `v9 cbf29c9d` |
| 24 | MEDIUM | settlement | missing | wowza-betting | `update_results.py:1069` | HT tips are graded only against half-time scores in the committed fd_history and af_ht_history parquets, with no API fallback. Fixtures missing from those files stay pending indefinitely while the O/U ledger moves on. | fixed `v9 9b207966` |
| 25 | MEDIUM | settlement | leagues | wowza-betting | `update_results.py:554` | update_player_results derives the API-Football season as yr-1 for Jan-Jun and yr otherwise for every club league. J1 (end_year from 2026-07) is therefore looked up in season 2026, and from 2027-01-01 calendar-year lea... | fixed `v9 cbf29c9d` |
| 26 | LOW | settlement | naming | wowza-betting | `src/ledger.py:460` | The side-market producer emits `side` (YES/OVER) and SIDE_LEDGER_COLS declares it, but `append_side_market_tips` builds each row without a 'side' key, so every side_bets_ledger row has side=NaN. Readers that filter on... | fixed `v9 8174ccba` |
| 27 | LOW | settlement | defaults | wowza-betting | `update_results.py:778` | Sharp and side-market settlement replace an unparseable entry price with odds 1.0, so a WIN is booked as P/L 0.0 with no flag. | fixed `v9 8174ccba` |
| 28 | LOW | settlement | missing | wowza-betting | `src/ledger.py:460` | side_bets_ledger.csv has a `side` column, added so a settled BTTS row can be graded unambiguously, but append_side_market_tips never writes it, so it is 100% null. training_cutoff is also 100% null. | fixed `v9 8174ccba` |
| 29 | LOW | settlement | leagues | wowza-betting | `update_results.py:99` | The primary settlement source for the 7 football-data Standard leagues is hard-coded to the finished 2025/26 file (mmz4281/2526). A 2026/27 fixture can never be found there; it settles only through the 3-day OddsAPI w... | fixed `v9 64d8b7f8` |
| 30 | LOW | settlement | leagues | wowza-betting | `update_results.py:370` | The API-Football last-resort settlement fetches only today's season per league (resolve_api_season(league) with no date). A tip from the previous campaign can never be settled once the league rolls over. | fixed `v9 46f4b0f9` |
| 31 | MEDIUM | training | naming | wowza-betting | `src/data_loader.py:880` | af_history.parquet still files Bundesliga 2 fixtures under the non-registry label 'Denmark 1st Div' (and Dutch Eredivisie fixtures under 'Norway Eliteserien'). The default legacy enrichment join has no league key, so ... | fixed `v9 fc6b8bab` |
| 32 | MEDIUM | training | defaults | wowza-betting | `src/data_loader.py:481` | The football-data NF loader keeps only seasons matching '2022/2023/2024/2025/2026' and invents season '2025/26' when the Season column is absent, so from 2027 NF history (training data and live form features) stops up... | fixed `v9 003ab021` |
| 33 | MEDIUM | training | defaults | wowza-betting | `pipeline.py:685` | Legacy training weights fall back to DEFAULT_DECAY_WEIGHT 1.0 for every season label missing from the hand-typed table, so the two newest seasons and all calendar-year seasons (including COVID 2020) get the same weigh... | fixed `v9 d8eef1ac` |
| 34 | MEDIUM | training | defaults | wowzaV9-Pro | `src/pipelines/team_stats.py:233` | The Pro team-statistics collector is hard-coded to API-Football season '2026' (function default, CLI default and workflow default), so leagues whose live season has another key are not collected. | fixed `Pro 62fbcbae + 18654c8a` |
| 35 | MEDIUM | training | missing | wowza-betting | `src/api_football_ou.py:348` | fetch_league_fixture_results caches the CURRENT season's finished-fixture list for 8,760 hours. That file is restored across CI runs by actions/cache, so the history for API-Football-only leagues stops at the first fe... | fixed `v9 a5fc69b3` |
| 36 | MEDIUM | training | missing | wowza-betting | `src/data_loader.py:481` | New-format football-data files are filtered by a literal list of season years. From the 2027 calendar seasons (Feb-Mar 2027) onward, those rows will be dropped silently from training history and from result settlement. | fixed `v9 003ab021` |
| 37 | MEDIUM | training | leagues | wowza-betting | `src/feature_engineering.py:227` | In the history frame, split-season Standard leagues carry duplicate fixtures labelled with a calendar year ('2024', '2025', '2026') next to the correct 'YYYY/YY' copies. _add_season_venue_stats groups by (league, seas... | fixed `v9 0e16b292` |
| 38 | MEDIUM | training | leagues | wowza-betting | `player_model/config.py:427` | PROP_SEASONS applies the European start-year rule to every league. J1 is '2026' while the live season is 2027, and calendar-year leagues will be one year behind from 2027-01-01 to 06-30. Props fixture and stats fetche... | fixed `v9 63848c10` |
| 39 | LOW | training | naming | wowza-betting | `src/data_loader.py:141` | `_nf_real_odds` keys the new-format price archive by (snapshot_date, league, raw 'Home vs Away'), but the training loader looks it up by (match date, league, football-data 'Home vs Away'). Prices captured before match... | fixed `v9 64f5fe0e` |
| 40 | LOW | training | naming | wowza-betting | `player_model/league_quality.py:140` | `_get_tier` maps a league name to a tier by a two-way substring scan in dict order, so second divisions and foreign leagues pick up the top-flight key ('La Liga 2'->'La Liga', 'Brazil Serie A'->'Serie A', 'Bundesliga ... | fixed `v9 d8eef1ac` |
| 41 | LOW | training | defaults | wowza-betting | `player_model/config.py:422` | Props seasons still use the European start-year rule for every league (J1 is '2026' today while its live API season is 2027) and fall back to a literal '2025' in eight call sites. | fixed `v9 63848c10` |
| 42 | LOW | training | defaults | wowza-betting | `player_model/feature_engineering.py:639` | The props feature builder fills opponent and context features with invented constants that arrive as non-missing, so the PFC-1.0 contract and PropsImputer cannot see or count them. | deferred: do with the R2-D props recipe switch (legacy batch-median imputation would make NaN worse) |
| 43 | LOW | training | missing | wowza-betting | `src/data_loader.py:435` | In the CI history download, a league-season file that returns non-200 is skipped with `continue` and no log. Only exceptions are logged, so a 404 or 503 removes that league-season from training and serving history wit... | fixed `v9 003ab021` |
| 44 | LOW | training | missing | wowza-betting | `src/data_loader.py:1591` | The 'HT extra' download of big-5 leagues is pinned to the 2025/26 file and labels it season '2025/26', so the HT training history never gets the current season. | fixed `v9 003ab021` |
| 45 | LOW | training | leagues | wowza-betting | `src/data_loader.py:401` | The CI training and serving downloader keeps its own league-to-football-data-code tables. They omit registry leagues that have fd codes (Romanian Superliga ROM, Scottish League One SC2, Scottish League Two SC3), so th... | fixed `v9 003ab021` |
| 46 | LOW | training | leagues | wowza-betting | `src/data_loader.py:482` | Several loaders still filter seasons by literal years or treat J1 as a fixed season type. From 2027-01-01, calendar-year NF seasons (and from 2027-07 '2027/2028' labels) will silently drop out of CI history and settle... | fixed `v9 003ab021` |
| 47 | LOW | dashboard | defaults | wowza-betting | `src/api_football_ou.py:490` | key_attacker_absent_home/away is written as 0.0 ('nobody important is out') whenever the fixture id or injury context is missing, which defeats the NaN = not fetched intent stated in feature_engineering. | fixed `v9 d8eef1ac` |
| 48 | LOW | dashboard | missing | wowza-betting | `src/ledger.py:107` | The model-attribution field training_cutoff is almost always empty in the ledgers, even on rows that carry a model_version, so lineage analyses cannot tell which training window produced a tip. | fixed `v9 8c746bb3 (labelled, not invented)` |
| 49 | MEDIUM | research | leagues | wowzaV9-Pro | `src/scout/registry.py:49` | Pro's league registry reads v9 config.py with ast.literal_eval. Since R2-A, v9 derives those tables from config/leagues.json, so every list parses as empty. The weekly job therefore rewrote registry/wowza_league_regis... | fixed `Pro 8159b044` |
| 50 | MEDIUM | research | leagues | wowzaV9-Pro | `src/pipelines/team_stats.py:317` | Team match statistics are collected for the literal season '2026' for every league (CLI default and pro_team_stats.yml:95). J1's live autumn-spring season (API-Football 2027) is never collected, and every league will ... | fixed `Pro 62fbcbae + 18654c8a` |
| 51 | LOW | research | defaults | wowzaV9-Pro | `src/pipelines/team_news.py:174` | team_news fetches /injuries with one European start-year season for every league, so J1 (live season 2027) and every calendar-year league in Jan-Jun query the wrong season and record no injuries. | fixed `Pro 62fbcbae + 18654c8a` |
| 52 | LOW | research | defaults | wowza-betting | `player_model/predict.py:184` | The props confidence score that gates tiers is 45% hard-coded: recency is a literal 0.8 on every row, model agreement a literal 1.0, and p_start defaults to 0.8 when absent. | fixed `v9 d0c1fdf9 (honest score; empty gate off, owner decision)` |
| 53 | LOW | research | missing | wowza_v11 | `scripts/v11_shadow.py:142` | v11_shadow's _decide never sets 'reason' except when the model probability is missing, so v11_reason is 100% null in the snapshot history. Every row is NO_BET, and the record never says why. | fixed `v11 d083014` |
| 54 | LOW | research | leagues | wowzaV9-Pro | `src/pipelines/team_news.py:174` | Injuries are requested with one European start-year season for all leagues. For J1 this is season 2026, the finished spring tournament, so live J1 fixtures never receive injury counts. From 2027-01-01 the same will ha... | fixed `Pro 62fbcbae + 18654c8a` |
| 55 | LOW | research | leagues | wowzaV9-Pro | `src/pipelines/team_stats.py:220` | Pro's team_match_stats files the same league under two names. Rows before 2026-09-06 carry the API-Football provider names ('Belgian Pro League', 'Eredivisie', 'Primeira Liga'); later rows carry the canonical names. E... | fixed `Pro 25afc7c7` |
| 56 | LOW | research | leagues | wowzaV9-Pro | `registry/league_season_calendar.json:1` | Pro's season calendar (byte-copied to v11 src/league_season_calendar.json) still disagrees with v9 config/leagues.json for historical eras. v9 is the wrong side: it treats Argentina before 2021 as calendar_year and ha... | fixed `v9 9c86edc6` |
