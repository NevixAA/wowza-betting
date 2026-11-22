# P1 item 10 - v11 reconciliation (2026-11-22)

## Repo and SHAs
- Repo: NevixAA/wowza_v11. Base `c0d2961ea9a709ba64ef9350f5b4ecdd673c2218`. Fix: PR #4, branch `p1/identity-parity`, commit
  `76a3a098`, merged as `85c956f09caa440eaed63746544aa83a6ba30d92`. Post-merge Tests on main: success.
- Compared against v9 main `61e869877be7`, Pro main `1a413050cecd` (ledger as of that commit), and the v11 private ledger
  in wowza-exec `prediction-ledger-v11` @ `ef5b5d1410cf` (read via gh/git with read access; counts and keys only, no
  probabilities or prices printed).

## Collection and freshness (preserved)
V11 Collect has been green on every run since 2026-11-20, most recently 16:27 UTC. `v11_research_health.json` overall=PASS,
calculation_version 1.0.0. Nothing in the collection or freshness path was changed.

## Root cause found and fixed
**Stale vendored `src/team_names.py`.** At 18:06 UTC today v9 added the `hertha berlin` result alias (9323c5e14, Standard
ledger settlement). v11's copy lacked it. The self-check `src/team_names.py == v9 src/team_names.py` passed at 16:30 only
because it ran before the v9 change. The next V11 Collect would have gone red, and v11 could not resolve that club's
results. Fix: re-copied v9 main's file byte-for-byte.

## Cross-repo ledger reconciliation (real rows)
| | v9 | Pro | v11 (private) |
|---|---|---|---|
| rows / fixtures | 338,848 / 833 | 33,149 / 321 | 596 / 267 |
| schema (31 columns) | identical | identical | identical |
| duplicate keys (KEY_COLS) | 0 | 0 | 0 |
| rows with predicted_at >= kickoff | 0 | 0 | 0 |
| minutes_to_kickoff vs own timestamps (>1 min off) | 0 | 0 | 0 |
| empty fixture_ref | 0 | 0 | 0 |
| fixture_ref prefixes | af 331,408 / oa 7,440 | oa 24,178 / af 8,640 / nk 331 | oa 596 |

- **Shared fixture identity:** across all three repos, every `oa:` fixture_ref has exactly one oddsapi_event_id, league_slug,
  season, kickoff_utc, home_team and away_team, both within each repo and across repos (0 conflicts on every column).
- **Coverage:** all 267 v11 fixtures are in v9. 175 are in Pro. The 92 missing from Pro all kick off 2026-11-25..29, after
  Pro's latest evaluated window. Pro's bet builder has not reached them yet. This is a horizon difference, not an identity break.
- **Market/line matching:** v11 `ou25` uses line 2.5 with selections OVER/UNDER and settle rule total_goals_over/ft, the same
  as v9_standard and v9_new_format ou25. Selections and lines match on every sampled fixture.
- **Prediction versions:** v11 has one model_version (v11_shadow_bc9877bf) over 8 runs. v9 ou25 carries v9_standard
  {standard_20261118..50a974bb, standard_20261119..67ee3666, standard_repaired_20261028..1f5805e6} and v9_new_format
  {newformat_20261018..9eb6b4b4, newformat_20261118..303c9a05}. These are distinct keys by design (version is in the key).
- **Track label:** v11 rows carry track=`board` (its contract entry). v9 rows carry standard or new_format. Cross-repo joins
  must take the track from v9 or the league registry, not from v11's column. This is recorded, not changed: it is a contract value.
- **Pre-kickoff eligibility:** 0 rows at or after kickoff in any repo. The v11 writer refused 1 post-kickoff fixture in the
  last run (`run_not_pre_kickoff: 1`). 2 v11 rows are unscored (`odds rejected`), with probability blank and reason recorded.
- **Settlement consistency:** of 267 shared fixtures, v9 has settled 93 (API-Football) and v11 has settled 0. v11 settles
  only from football-data.co.uk and v9's af_history. Locally, the latest football-data result for these leagues is
  2026-11-07..14, so 11-20+ fixtures cannot settle yet (V11 Collect 16:29: "graded 0/119; 119 unmatched", mostly "no result
  within 1 day"). After 72h the view labels such rows NO_RESULT. The view is rebuilt from the raw ledger every run, so it
  becomes SETTLED once the source publishes. On v9 ou25 settled rows, y agrees with the goals in 604/604.
  **Consequence:** v9-vs-v11 comparisons must join on fixtures settled in both repos.
- **De-vig:** v11 `v11_p_market` is a multi-book consensus (power de-vig, median, gated on n_books), not the proportional
  de-vig of the single displayed pair. It cannot be re-derived from public columns, so only ranges were verified:
  0 values outside (0,1). The per-book quotes are private. In `v11_market_movement_detail.csv`,
  `clv_pct` = 100 x (entry_odds/close_odds - 1) on 86,092/86,092 rows (a PERCENT). Entry and close fair probabilities are
  all in [0,1]. 0 entries are at or after kickoff.
- **Research calc versions:** v11 movement, microstructure, research_state and momentum_control are all 1.0.0. Pro's
  deduped movement archive holds only 1.0.0 (see P1_PRO_RECONCILIATION.md).
- **Fair-probability movement is not executable profit:** `market_move_pp`/fair probabilities describe de-vigged
  consensus movement. `clv_pct` uses the side's real entry and close odds before costs. The Pro price-trader verdict
  (EXECUTABLE_PRICE_DATA_SUFFICIENT=NO, FULL_GREEN_PROFITABLE_AFTER_COSTS=NO) still governs any profit claim.

### Finding left for owner approval (not changed)
In `v11_market_movement_detail.csv`, 1,829 rows (64 fixtures, 2.1%; entries 2026-08-17..11-17) have a close-snapshot
kickoff (close_ts + close_minutes_to_kickoff) that differs from the row's kickoff_ts by more than 5 min. These look like
reschedules or kickoff updates. 1,761 of them are clv_quality=OK, and 401 have close_ts after the row's kickoff_ts. The
cause: `quality_flags` judges the close against the close snapshot's own kickoff. Proposed fix: a KICKOFF_CHANGED flag.
That changes eligibility, so it needs a movement CALC_VERSION bump (and a new Pro archive version), and therefore owner
approval.

## Sample: 32 real fixtures, 16 leagues, both tracks (present in v9, Pro and v11)
Rows per repo are ledger rows for that fixture (all markets). "ids agree" = one event id, league, season, kickoff and teams
across all three repos. "pre-KO" = every row in every repo written before kickoff.

| league | track | kickoff UTC | fixture | ref | v9 rows | Pro rows | v11 rows | ids agree | pre-KO |
|---|---|---|---|---|---|---|---|---|---|
| argentina_primera_division | new_format | 2026-11-20T23:00 | Velez Sarsfield BA v Belgrano de Cordoba | `oa:09a8ac0e` | 26 | 139 | 2 | yes | yes |
| argentina_primera_division | new_format | 2026-11-23T20:00 | Gimnasia La Plata v Newells Old Boys | `oa:08d2a1ac` | 26 | 72 | 2 | yes | yes |
| austrian_bundesliga | new_format | 2026-11-21T16:00 | WSG Tirol v Austria Lustenau | `oa:1921e07c` | 38 | 165 | 2 | yes | yes |
| austrian_bundesliga | new_format | 2026-11-22T16:00 | Austria Wien v RB Salzburg | `oa:5b642036` | 38 | 74 | 2 | yes | yes |
| brazil_serie_a | new_format | 2026-11-21T19:30 | Palmeiras v Flamengo | `oa:1d642bb0` | 26 | 158 | 2 | yes | yes |
| brazil_serie_a | new_format | 2026-11-24T21:00 | Chapecoense v Palmeiras | `oa:0c21d6b1` | 26 | 72 | 2 | yes | yes |
| denmark_superliga | new_format | 2026-11-20T18:00 | AGF Aarhus v Lyngby | `oa:02d5f406` | 38 | 208 | 4 | yes | yes |
| denmark_superliga | new_format | 2026-11-23T18:00 | Brondby IF v FC Midtjylland | `oa:7f611997` | 26 | 72 | 2 | yes | yes |
| japan_j_league | new_format | 2026-11-20T10:00 | Kashiwa Reysol v JEF United Chiba | `oa:c030e04e` | 38 | 156 | 4 | yes | yes |
| japan_j_league | new_format | 2026-11-25T10:30 | Urawa Red Diamonds v Cerezo Osaka | `oa:c6b1486b` | 12 | 72 | 2 | yes | yes |
| mexico_liga_mx | new_format | 2026-11-21T01:00 | Puebla v Atlético San Luis | `oa:40846f0c` | 38 | 166 | 4 | yes | yes |
| mexico_liga_mx | new_format | 2026-11-23T01:10 | Querétaro v Necaxa | `oa:3030284e` | 26 | 72 | 2 | yes | yes |
| norway_eliteserien | new_format | 2026-11-20T18:00 | Molde v Bodø/Glimt | `oa:3d4b217d` | 38 | 206 | 4 | yes | yes |
| norway_eliteserien | new_format | 2026-11-22T18:15 | Sarpsborg FK v Lillestrom | `oa:6c5dd9f6` | 38 | 146 | 4 | yes | yes |
| sweden_allsvenskan | new_format | 2026-11-21T12:00 | Östersunds FK v Varbergs BoIS | `oa:186c5a01` | 22 | 139 | 2 | yes | yes |
| sweden_allsvenskan | new_format | 2026-11-23T18:00 | Örgryte IS v Kalmar FF | `oa:83a9063e` | 26 | 72 | 2 | yes | yes |
| usa_mls | new_format | 2026-11-21T19:00 | New England Revolution v Philadelphia Union | `oa:b7ca8ba1` | 26 | 174 | 2 | yes | yes |
| usa_mls | new_format | 2026-11-25T03:00 | San Jose Earthquakes v Houston Dynamo | `oa:8149dd94` | 26 | 72 | 2 | yes | yes |
| bundesliga_2 | standard | 2026-11-20T17:30 | SV Darmstadt 98 v 1. FC Heidenheim | `oa:ef781f08` | 48 | 168 | 4 | yes | yes |
| bundesliga_2 | standard | 2026-11-22T12:30 | Karlsruher SC v VfL Osnabrück | `oa:1773ac46` | 36 | 72 | 2 | yes | yes |
| championship | standard | 2026-11-21T12:30 | Norwich City v Cardiff City | `oa:bed1f200` | 36 | 170 | 2 | yes | yes |
| championship | standard | 2026-11-24T20:00 | Wolverhampton Wanderers v Derby County | `oa:018856cb` | 36 | 76 | 2 | yes | yes |
| la_liga_2 | standard | 2026-11-20T19:30 | Andorra CF v Leganés | `oa:fabb494c` | 36 | 243 | 2 | yes | yes |
| la_liga_2 | standard | 2026-11-23T19:30 | AD Ceuta FC v Almería | `oa:f024fef3` | 36 | 72 | 2 | yes | yes |
| league_one | standard | 2026-11-20T20:00 | Barnsley v Sheffield Wednesday | `oa:deb5feb1` | 36 | 165 | 2 | yes | yes |
| league_one | standard | 2026-11-24T19:45 | Blackpool v Leicester City | `oa:58d85828` | 12 | 72 | 2 | yes | yes |
| league_two | standard | 2026-11-21T12:30 | Gillingham v Crewe Alexandra | `oa:5eda3cdc` | 36 | 141 | 2 | yes | yes |
| league_two | standard | 2026-11-24T19:45 | Port Vale v Northampton Town | `oa:eaa91165` | 12 | 72 | 2 | yes | yes |
| ligue_2 | standard | 2026-11-20T19:00 | Annecy FC v Clermont | `oa:dade9855` | 36 | 209 | 4 | yes | yes |
| ligue_2 | standard | 2026-11-23T19:45 | Sochaux v USL Dunkerque | `oa:87ff23cf` | 36 | 60 | 2 | yes | yes |
| serie_b | standard | 2026-11-20T19:30 | Hellas Verona v Cesena FC | `oa:0cdc0425` | 36 | 240 | 2 | yes | yes |
| serie_b | standard | 2026-11-24T19:00 | Palermo v Südtirol | `oa:83a91302` | 36 | 74 | 2 | yes | yes |

All 32: identity agrees, ou25 line 2.5 and OVER/UNDER selections match between v9 and v11, and every row is pre-kickoff.

## Files changed
`src/team_names.py` (re-vendored), `tests/test_v9_pro_v11_identity_parity.py` (new, 13 tests), `.github/workflows/tests.yml`
(sparse read-only Pro checkout as PRO_LOCAL).

## Tests
Local before push: `scripts/v11_tests.py` all checks passed. pytest 54 passed, 2 skipped (V9_LOCAL = v9 main 61e86987,
PRO_LOCAL = Pro main). actionlint was clean. PR CI passed, and post-merge Tests on main passed.
`test_team_names_identical_to_v9` fails on the old copy (4-line Hertha alias difference).

## Blockers
- v11 ledger settlement lags until football-data.co.uk publishes (external).
- KICKOFF_CHANGED eligibility change awaits owner approval.

## Production impact
None. v11 is research and shadow only. The alias re-vendor lets v11 resolve one more club, matching v9.

## Owner approval
Needed only for (a) the KICKOFF_CHANGED flag plus movement calc-version bump, and (b) whether v11 ledger settlement may
also read v9's API-Football settled results, which would mean faster settlement but a different source.
