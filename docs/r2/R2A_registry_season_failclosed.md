# R2-A: league registry, season resolver, fail-closed routing and serving (findings 6, 7, 8, 17)

Branch `r2/registry-season-failclosed` (v9). Not merged. Base: origin/main `4e9c93357` (2026-11-01).

| commit | content |
|---|---|
| `4f431f935` | owner rollback doc written FIRST: `docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md` |
| `246b2cc4e` | canonical league registry + season resolver; every `API_SEASON` call site moved |
| `746ba0f87` | predict fails closed (unknown league, missing NF artifact, repaired Standard failure); invalid recipe = configuration error |
| (next commit) | `scripts/r2a_failclosed_replay.py`, this file |

## Defects

| # | defect | files | severity | reachable in production today? |
|---|---|---|---|---|
| 6a | `config.API_SEASON = os.getenv("API_SEASON", "2025")`: one generic season, defaulting to a FINISHED season, used by `api_football_ou.fetch_team_season_stats`, `fetch_upcoming_team_stats`, `resolve_upcoming_fixture_ids`, `fetch_h2h_features` whenever no season was passed (always), and as the fallback in `feature_engineering` standings and `update_results._af_completed` | `config.py`, `src/api_football_ou.py`, `src/feature_engineering.py`, `update_results.py` | HIGH (wrong-but-plausible data: last season's team stats overwrite real model inputs `home_season_goals_h` etc.) | **Dormant in predict** (predict.yml has no `APIFOOTBALL_KEY`, so these enrichments return `{}` before the season is used). Reachable in any job that runs `build_upcoming_features` with the key. |
| 6b | `API_FOOTBALL_SEASONS.get(league, "2025")` / `.get(league, str(now.year))` fallbacks: any league missing from the hand table got a guessed year | `src/api_football_ou.py` (season round, coach), `src/data_loader.py` (shot enrichment), `src/feature_engineering.py` (injury context), `scripts/capture_nf_odds_forward.py`, `scripts/capture_std_sidemarket_odds_forward.py`, `scripts/warm_nf_shot_cache.py`, `scripts/backfill_af_history.py` | MEDIUM | Yes for the 5 training-only Standard leagues with API ids but no table entry (Portuguese, Belgian, Dutch, Scottish Premiership, Turkish) under `WOWZA_FULL_ENRICH=1` (backtest/backtest_matrix): fetched season **2025** (finished) instead of 2026. Capture scripts guessed `now.year` = right today, wrong every Jan-Jun for European leagues. |
| 6c | Hand-rolled season table: owner rolled it manually on 2026-08-05; J1's switch to an autumn-spring calendar (live season keyed 2027) was patched by hand 2026-10-22 | `config.py` | MEDIUM | Next failure would have been 2027-01-01 / 2027-07-01 rollovers. |
| 7a | Ten hand-maintained per-league tables in `config.py`; "Danish Superliga" in the season table vs "Denmark Superliga" everywhere else (Denmark on a finished season for months, fixed 10-22); "Austria Bundesliga" and "Austrian Bundesliga" as two keys; "World Cup" and "World Cup 2026" as two sport-key entries | `config.py` | MEDIUM | The duplicate keys were inert (no consumer read them) but the next divergence had no test to catch it. |
| 8 | `predict.py` scored a league in neither format list with the **Standard** model ("Fallback (unknown league)"), and New Format fixtures with the **Standard** model when the NF artifact was missing | `src/predict.py`, mirrored in `src/model_registry.ou_model_for` | HIGH (silent wrong-model tips) | Unknown-league branch: **not reachable today** (every `ENABLED_LEAGUES` entry, auto-onboarded leagues included, is in one of the two sets). NF branch: reachable whenever `models/model_v9_newformat.pkl` is missing. |
| 17 | Repaired Standard O/U 2.5 failure re-scored every Standard row with the **legacy** model and kept tipping (stamped `legacy_v9_fallback`); an invalid `STANDARD_OU25_SERVING_RECIPE` silently became `legacy_v9` | `src/standard_serving.py`, `config.py` | HIGH (production silently reverts to the model the owner moved away from) | Yes: any missing/corrupt repaired artifact or feature-build exception; any typo in predict.yml. |

## Repairs

**Registry (finding 7).** `config/leagues.json` (`LR-1.0-2026-11-01`, 48 leagues, 3 aliases) holds per
league: id, canonical name, aliases, model type, production state (ENABLED / ENABLED_PROVIDER_UNSUPPORTED
/ PAPER / TRAINING_ONLY / COLLECTION_ONLY), enabled, paper, country, timezone, football-data code,
API-Football id, OddsAPI sport key, Sofascore id, API-Football-only flag + extra seasons, provider names,
season rule, provider-unsupported evidence, ROI notes. `src/league_registry.py` validates (duplicate
names/ids, alias = canonical name, alias with two targets, provider id shared by two leagues, enabled
without model type, bad season rule) and derives every config table; `config.py` keeps all names, so
no consumer changed. Aliases resolve through `LeagueMap` lookups but are never iterated. Resolution is
exact (trim only). `output/league_registry_health.json`: validation, config consistency per table,
coverage gaps, and the last predict routing. The auto-onboarding overlay (`config/auto_leagues.json`)
still merges after derivation and is reported as non-registry keys.

**Seasons (finding 6).** `src/season_resolver.resolve_api_season(league, kickoff_date=None,
provider="api_football")`: rules `start_year(7)` (European), `calendar_year`, `end_year(7)`, `fixed`,
and dated `eras` (J1: calendar year until 2026-07-01, end-year after). Unknown league / provider /
date -> `SEASON_UNKNOWN`; callers skip the enrichment and `record_degradation()`. `config.API_FOOTBALL_SEASONS`
is derived from it (identical to the hand table on 2026-11-01, plus 20 leagues that used to fall back).
`config.API_SEASON` and `player_model.config.API_SEASON` are the `SEASON_UNKNOWN` sentinel.
`output/season_resolution_health.json`: resolved season per league, FAIL if an active league is unknown.

Call-site audit (every site that used `API_SEASON` or a literal/guessed season fallback), all now on the
resolver: `api_football_ou` x6 (team stats per fixture date, fixture ids per league min date, h2h per
fixture date, season round, coach), `feature_engineering` x2 (injury context, standings per fixture
date), `data_loader._enrich_with_api_shots`, `update_results` x2 (AF results fallback; live-signal
grading by the FIXTURE's date), `capture_nf_odds_forward`, `capture_std_sidemarket_odds_forward`,
`warm_nf_shot_cache`, `backfill_af_history`. A test greps the repo so no new site can read
`API_SEASON` or default a season to a literal year.

**Fail-closed routing (finding 8).** Unknown league -> `model_type=unknown`, `p_over25` empty, no side
probabilities, `bet=AVOID`, ERROR log, `last_predict_routing.unknown_league_fixtures` in the registry
health file (status WARN). Missing NF artifact -> NF rows unscored, routing status FAIL. The provenance
mirror stamps such rows `NONE_FAIL_CLOSED`.

**Fail-closed serving (finding 17).** Repaired failure -> Standard `p_over25` NaN, `serving_recipe=v9_repaired_failed`,
no O/U 2.5 tip, `output/standard_serving_health.json` FAIL, system health RED. NF, side and HT tips
unaffected. Invalid recipe -> `ServingConfigError` in `predict_upcoming` before OddsAPI is called;
`import config` never raises. `legacy_v9` (explicit or unset) behaves exactly as before.

## Tests

| file | what it proves |
|---|---|
| `tests/test_league_registry.py` | every derived table equals the pre-registry snapshot (`tests/fixtures/config_league_tables_pre_registry_2026_11_01.json`) minus the 3 alias keys; enabled iteration order unchanged; aliases resolve but are not keys; no fuzzy match; validation catches duplicates / alias cycles / two-target aliases / provider-id collapse / enabled-without-model; invalid registry refuses to load; player_model `PROP_LEAGUES` ids agree with the registry; health file |
| `tests/test_season_resolver.py` | July / January rollovers, J1 eras (2025 calendar, 2026 spring, 2027/2028 end-year), date types, SEASON_UNKNOWN for unknown league / provider / date / NaT, aliases, auto-league overlay, health, repo-wide grep for `API_SEASON` readers and literal-year fallbacks, `api_football_ou` never calls the API on SEASON_UNKNOWN |
| `tests/test_predict_fail_closed.py` | 14 near-name variants each get no probability / no bet while neighbours are scored; unknown never reaches `generate_bets`; missing NF artifact leaves NF unscored (not Standard-scored); fallback code gone; health written by the pipeline hook |
| `tests/test_standard_ou25_switch.py` (updated) | repaired file missing / scorer exception -> NaN, `v9_repaired_failed`, AVOID, health FAIL, NF unchanged; invalid recipe raises before odds are fetched; rollback env still byte-identical |

## Offline replay of the committed board (RETROSPECTIVE, before vs after)

`scripts/r2a_failclosed_replay.py`: the board of `output/predictions.csv` at `4e9c93357` (153 fixtures:
67 Standard, 86 New Format), one frozen history frame (85,467 rows) for both trees, predict.yml's deployed
env (`STANDARD_OU25_SERVING_RECIPE=v9_repaired`, thresholds), OddsAPI stubbed, writers stubbed, network
blocked. BEFORE = a detached worktree at `4e9c93357`; AFTER = this branch. Tips = the post-guard send set
(O/U 2.5 + side markets).

| scenario | predictions | tips sent | result |
|---|---|---|---|
| **board** (production today) | **byte-identical** (hash `c5373a3abd8f4d70`) | **byte-identical**, 53 = 53 (hash `c10831923c1d29b2`) | no production behaviour change for any known league |
| near_names (+4 copies relabelled `Austria Bundesliga`, `championship`, `Serie B `, `Premier League`) | only the 4 injected rows differ: BEFORE scored them with the Standard model (p 0.507-0.529), AFTER `p_over25` empty, `model_type=unknown` | identical (53); none of the 4 tipped before either (edge below threshold) | the silent route existed; now refused |
| repaired_missing (repaired artifact unavailable) | 67 Standard rows: BEFORE legacy re-score (`legacy_v9_fallback`), AFTER NaN + `v9_repaired_failed` | BEFORE 49, AFTER 32: the 17 removed tips are all Standard O/U 2.5 legacy-fallback tips (12 VALUABLE, 5 MARKSMAN: Serie B 6, Bundesliga 2 3, Championship 3, La Liga 2 3, Ligue 2 2); NF and side tips unchanged | intended (finding 17) |
| nf_missing (NF artifact not loaded) | 86 NF rows: BEFORE Standard-model probabilities, AFTER NaN | BEFORE 50, AFTER 24: the 26 removed tips are NF O/U 2.5 tips the Standard model produced (21 VALUABLE, 4 MARKSMAN, 1 SNIPER); side tips unchanged | intended (finding 8) |

No fixture on the live board is in an unknown or mis-routed league, so no live fixture is blocked.

## Production behaviour changes (if merged)

1. None on today's board (proved above). Thresholds, tiers, stakes, Telegram logic and the deployed recipe are untouched.
2. A repaired Standard scoring failure now stops Standard O/U 2.5 tips (no legacy fallback) and turns health RED.
3. A typo in `STANDARD_OU25_SERVING_RECIPE` now stops predict (red run) before OddsAPI is called.
4. A missing NF artifact now stops NF O/U 2.5 tips instead of tipping off the Standard model.
5. An unknown league fixture (none exists today) gets no probability.
6. Predict writes and commits three new health files: `output/league_registry_health.json`,
   `output/season_resolution_health.json`, `output/standard_serving_health.json`.
7. Non-predict jobs with `APIFOOTBALL_KEY`: seasons come from the resolver. Identical today for every
   league that had a table entry. Changed: the 5 training-only Standard leagues under `WOWZA_FULL_ENRICH=1`
   (backtest, backtest_matrix) now fetch season 2026 (+2025) instead of 2025 (+2024); capture scripts
   stop guessing `now.year` for European leagues in Jan-Jun; `update_results` live-signal grading uses the
   fixture's own season; at the 1 July / 1 January rollovers seasons roll by themselves.
8. Provenance stamps `NONE_FAIL_CLOSED` (instead of the Standard model's version) on rows no model scored.

## Open risks / not done

* **Dashboard slim repo**: `config.py` now imports `src/league_registry.py`, `src/season_resolver.py` and reads
  `config/leagues.json`. All three are added to `deploy/dashboard_manifest.json` (leagues.json as an ASSET,
  shipped with the code) by hand-patch, not a full `--write` trace. After merge run
  `python scripts/dashboard_manifest.py --check` and rebuild/push the slim repo, or the dashboard fails to import config.
* **Props seasons (same defect class, not in this scope)**: `player_model/config.PROP_SEASONS` applies the
  European start-year rule to EVERY league, so J1 is "2026" today (live season 2027) and calendar-year
  leagues (MLS, Brazil, Argentina, Nordics, China, Ireland) will be one year behind from 2027-01-01 to
  06-30. Fallbacks to literal "2025": `player_model/pipeline.py:546,586,776`, `player_model/predict.py:345,424`,
  `player_model/fantasy.py:679`, `telegram_bot/notifier.py:214`; `str(now.year)`: `scripts/snapshot_prop_odds.py:49`;
  hard-coded `"season": "2025"` in `player_model/data_fetcher.py:1218` (player profile);
  `data_fetcher._CALENDAR_YEAR_LEAGUES` still lists J1 as calendar-year (masked by the additive `api_current_season`).
  Recommended: derive PROP_SEASONS from `src.season_resolver` (owner/props workstream decision).
* **Cross-repo**: v11 and Pro still carry their own league lists (finding 28). `config/leagues.json` is ready
  to be consumed (raw URL) with the same equality tests; not wired here.
* Auto-onboarded leagues keep a STATIC season from `config/auto_leagues.json` (reported WARN in the season
  health) until the onboarding writes a `season_rule`.
* `model_type_for_league` stays exact-canonical: a registered alias is NOT a model route. Intentional (fail closed).
