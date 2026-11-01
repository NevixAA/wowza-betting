# R2-E: Prediction Lab identity, season calendar, research run status, v11 registry, dashboard skew

Workstream R2-E of the WOWZA model-integrity repair (owner brief 2026-11-01), findings 11, 26-32.
Branch `r2/lab-v11-dashboard` in three repos (Pro, v11, v9). Nothing here is merged to main, and
none of it changes a Standard threshold, a stake, a served model or a PAPER/BLOCKED market.

| repo | what changed | production effect if merged |
|---|---|---|
| Pro `NevixAA/wowzaV9-Pro` | Prediction Lab data v2 (canonical identity, league-aware seasons), season calendar service, research run-status classes | none (research only) |
| v11 `NevixAA/wowza_v11` | canonical league registry, derived NF seasons, strict result date window, missing `p_model` fails closed | v11 grades more fixtures (Saudi, K-League); never bets |
| v9 `NevixAA/wowza-betting` | dashboard code/data skew status, dormant `edge_engine.Candidate` no 0.5 default, this doc | dashboard shows build/data SHAs and warns on skew; no model, tip or stake change |

## 1. Finding 11: canonical identity in the Prediction Lab (Pro)

**Before:** `src/prediction_lab/data.py` grouped every rolling feature by the raw team string, and
joined `af_history` on `(date, home, away)` with no league. One club under two source spellings was
two teams, so its matches were counted twice and its rolling history was split in half.

**After (data version `lab-data/2.0`, identity version `pro-identity-1.0+identity/1.0`):**

* `src/data/canonical_identity.py` ports v9's two identity stages in v9's order:
  1. `data_loader._canonicalise_clubs`: within one league, spellings with the same affix-free key
     are merged, the most recently seen spelling wins, score conflicts are dropped and the richest
     row is kept;
  2. `identity_repair.canonicalise_and_dedupe(frame, frozen_map, derive=False)`: the frozen,
     evidence-backed identity/1.0 alias map (142 aliases, 18 leagues, league-scoped, ambiguity
     refused). Copies whose scores disagree are quarantined.
* Equivalence is proven against v9's code, not just claimed. `tests/test_lab_identity.py` loads
  v9's own `identity_repair.py` and extracts `_canonicalise_clubs` from v9's `data_loader.py`. It
  then checks the output matches row for row on a synthetic frame and on real `fd_history` rows
  (Sweden, Bundesliga 2, League Two and Ireland).
* The map is a byte-identical copy at `registry/identity/club_aliases_identity_1.0.json`. Its hash
  is pinned. An edited map raises `IdentityMapError`, and cycles, chains and self-aliases are
  rejected.
* Pro's `src/data/club_resolver.py` is **not** equivalent and is not used here. It infers
  OddsAPI-to-football-data matches from token subsets. That is vocabulary inference, not
  duplicate-match evidence.
* `af_history` is joined on `(league, date, canonical home, canonical away)` with
  `validate="many_to_one"`, and the row count must stay the same. 9,918 duplicate `af` keys were
  dropped before the join.
* Outputs now go to `output/pure_prediction/data_v2/` and `docs/pure_prediction_data_v2/`. The
  data 1.0 files are never overwritten. Every table carries `data_version`, `identity_version` and
  `season_calendar_version`.

**Reconciliation** (`python -m src.prediction_lab.data --reconcile --write`, v9 data at origin/main
2026-11-01, written to `output/pure_prediction/data_v2/identity_reconciliation.json` with a
per-league CSV):

| measure | before (1.0, raw) | after (2.0, canonical) |
|---|---|---|
| fixtures | 63,317 | 51,624 |
| unique (league, team) | 1,033 | 837 |
| same-day team collisions (a club in 2+ fixtures on one day) | 8,166 | **1** |
| collapsed clubs (alias pair that meets itself) | - | **0** |
| stage-1 spellings unified / rows removed / score conflicts | - | 104 / 2,039 / 1 |
| stage-2 cells renamed / duplicate rows removed / rows quarantined | - | 14,928 / 9,648 / 4 |
| rolling history per team, median (mean) | 111 (122.6) | 109 (123.4) |
| af card coverage | 0.08% | 0.19% |
| season unknown rows | - | 0 |

* The one remaining collision is Ireland, 2024-11-16: `Bray` and `Bray Wanderers`. identity/1.0
  rejected this pair because it had only 1 evidence date. It is reported, not guessed.
* The 4 quarantined rows are two fixtures with conflicting scores:
  * Queretaro v Atlas, 2022-03-05: 0-3 vs 0-1;
  * Olympiakos v Panathinaikos, 2023-10-22: 0-3 vs 1-1.
* Median history falls in the English leagues (Championship 121 to 104) because duplicates
  inflated it. It rises in split leagues (Argentina 191 to 267, Mexico 209 to 275, Austria 171 to
  234).
* 696 (league, name) pairs are outside the alias map. They are single-spelling clubs, kept as-is.
* The leakage audit on data v2 is CLEAN at all 4 cutoffs, with 0 of 568 tripwires.
* The `same_day_same_score_pairs_upper_bound` field (14,703 before, 7,166 after) also counts
  genuinely different matches that ended with the same score on the same day. It is an upper
  bound, not a duplicate count.

## 2. Finding 26: league-aware season calendar (Pro)

* `registry/league_season_calendar.json` (`season-calendar/1.0`) and `src/data/season_calendar.py`
  define three season types:
  * winter: 1 July boundary, label `2026/27`;
  * calendar: 1 January boundary, label `2026`;
  * dated special campaigns.
* The special campaigns are evidenced by the month-by-month distribution in `fd_history`:
  * Brazil 2020: Aug 2020 to Feb 2021 (COVID);
  * China 2021: ended Jan 2022;
  * Argentina 2020/21: Copa Maradona;
  * Japan 2026: spring special tournament, then autumn-spring with an END-year API-Football id
    (`2027`).
* An unknown league, or a near-name variant, returns `SEASON_UNKNOWN` with a label of None. No
  label is ever guessed.
* `test_api_season_matches_v9_config_today` checks the calendar against v9 `API_FOOTBALL_SEASONS`
  on 2026-11-01. It agrees for every league v9 lists.
  * **Found:** 5 Standard training leagues have no entry in v9 and fall back to
    `API_SEASON="2025"` there: Belgian First Division A, Dutch Eredivisie, Portuguese Primeira
    Liga, Scottish Premiership and Turkish Super Lig. The calendar resolves them to `2026`. This
    is finding 6, owned by the registry workstream.
* The lab keeps `season_label` as what it always was: a Jul-Jun chronological bucket used by folds
  and other callers.
  * `league_season`, `season_type` and `season_status` are new columns.
  * 23,141 of 51,624 rows have a league season that differs from the bucket.
  * The EARLY family and `season_stage` now use the league season.
* **EARLY stays inactive.** `EARLY_STATUS = "INACTIVE_PENDING_REBUILD"`, and a test fails if EARLY
  enters `FOOTBALL_FAMILIES`.
* TODO: read v9 `config/leagues.json` once `r2/registry-season-failclosed` lands. That branch had
  no commits when this workstream ran.

## 3. Finding 27: research workflow hygiene (Pro)

* `src/monitoring/run_status.py` classifies each run as INFRA_FAILURE, TEST_FAILURE, DATA_FAILURE,
  MODEL_FAILURE or SUCCESS.
  * The first failing class wins.
  * A cancelled step counts as infra, and so does a run where nothing ran.
  * A model that runs and loses is SUCCESS, because its verdict lives in its own output.
* Each run writes `output/run_status/<workflow>.json` and the step summary, both with
  `is_scientific_verdict: false`. The JSON is also uploaded as an artifact.
* Workflows covered:
  * `pro_prediction_lab.yml`: now installs `requirements.txt` (which includes pytest), runs the lab
    contract tests, builds data v2 and the reconciliation, and commits the `data_v2` paths;
  * `pro_research.yml`: gets an explicit Install step and a Tests step;
  * `pro_studies.yml`.

## 4. Findings 28-31: v11

* **Registry (28).** `src/league_registry.py` is now the single table.
  * `results.STD_LEAGUES` and `NEW_LEAGUES` are derived from it.
  * **Saudi Pro League and K-League 1** previously had no result source. They are now graded from
    v9's public committed `output/af_history.parquet` (read via `V9_LOCAL`, or raw HTTP), because
    football-data publishes neither league.
  * `Austria Bundesliga` and `Danish Superliga` are aliases, never keys.
  * `scripts/v11_tests.py` asserts equality with v9 `config.py` (`STANDARD_FORMAT_LEAGUES`,
    `NEW_FORMAT_LEAGUES` and `FOOTBALL_DATA_LEAGUES`, with aliases collapsed). It reads v9 from
    `V9_LOCAL` in CI, from a sibling checkout, or from public raw.
* **NF seasons (29).** `parse_new` no longer filters on `("2026", "2027")`.
  * It keeps the current and previous campaign, derived from season type and date. Labels are
    matched exactly, not by substring.
  * Rollover tests cover 1 January, 1 July, the Japan 2026 change, a 2028 date and an unknown
    league (which keeps nothing).
  * **Also fixed:** `grade_fixtures` fell back to ANY date when no result was within one day. A
    fixture whose result was not yet published could therefore be graded with an earlier meeting
    of the same two clubs. The window is now strict.
* **Missing model probability (30).** The `p_model=0.5` default is gone.
  * `edge_engine.Candidate.p_model` is now None.
  * `classify()` and `v11_shadow._decide` return `NO_BET` with reason `MISSING_MODEL_PROBABILITY`
    for None, NaN or out-of-range values.
  * `v11_residual` skips NaN rows instead of blending NaN.
* **Uncertainty defaults (31).** These stay as they were:
  * v11 `FALLBACK_N_EFF`, `FALLBACK_ECE`, `TAU` and the lineup and stale penalties;
  * the `ece=0.02` default in `edge_engine.Candidate`.

  They are conservative: each one widens the lower bound, so none of them can create confidence.
  They are documented in `src/evidence.py` and `src/edge_engine.py`.

### Every API that filled 0.5 (inventory)

| location | what | action |
|---|---|---|
| v11 `src/edge_engine.py` `Candidate.p_model = 0.5` | decision input | **fixed** (None, then NO_BET MISSING_MODEL_PROBABILITY) |
| v11 `scripts/v11_shadow.py::_decide` | decision input | **fixed** (explicit guard; the caller already skipped NaN) |
| v11 `scripts/v11_residual.py` | research blend | **fixed** (skip NaN) |
| v9 `src/edge_engine.py` `Candidate.p_model = 0.5` | dormant decision API (only `power_devig` and `proportional_devig` are imported, by `clv_capture`) | **fixed** (same as v11) |
| v9 `src/market_edge.py` (the serving shadow engine via `edge_engine_shadow`) | decision input | already fails closed (`MISSING_MODEL`); pinned by a new test |
| v9 `src/feature_engineering.py:535,998` `p_over25_poisson_dc = implied_prob_over.fillna(0.5)` | model FEATURE, fallback branch only when the Dixon-Coles helper is unavailable | listed, not changed (Standard production feature; changing it needs the challenger path) |
| v9 `player_model/predict.py:597-598` `home/away_win_prob` default 0.5 | props opponent-weakness multiplier | listed (props are paper-only; R2-D props workstream) |
| v9 `player_model/fantasy_residual.py:71` `home.fillna(0.5)` | fantasy feature | listed (finding 22, fantasy defaults) |
| v9 `player_model/feature_engineering.py:1309` duels ratio `fillna(0.5)` | props feature | listed (R2-D) |

## 5. Finding 32: dashboard code/data skew (v9)

* `dashboard_data/remote_mirror.py`:
  * `DASHBOARD_SCHEMA_VERSION = 1`.
  * `compatibility()` and `status()["compat"]` expose the dashboard build id, the source v9 SHA,
    the manifest commit, the mirrored data SHA, the data-side manifest commit, the build, code and
    data schema versions, and the last successful and last attempted sync.
  * Possible statuses: COMPATIBLE, STALE_BUILD, INCOMPATIBLE, UNKNOWN, NOT_SYNCED,
    BUILD_INFO_MISSING, and LOCAL_CHECKOUT (full checkout).
  * Only COMPATIBLE and LOCAL_CHECKOUT are `healthy`. An unknown or incompatible schema is never
    shown as healthy.
  * `sync()` downloads v9's own `deploy/dashboard_manifest.json` at the pinned data SHA into
    `deploy/.v9_manifest_at_data.json`. It does this on a full sync, or when the compare API says
    the manifest changed. It never overwrites the build's manifest, and it never counts toward
    sync failure.
* `scripts/build_dashboard_repo.py` writes `dashboard_build_id` (a content digest of the shipped
  code) and `schema_version` into `deploy/BUILD_INFO.json`. It refuses to build when the manifest
  schema does not match the code.
* `scripts/dashboard_manifest.py` writes `schema_version` into the manifest.
* `app.py`:
  * admin sidebar: shows build, source SHA, data SHA, both schema versions and the status, with a
    warning when unhealthy;
  * other tiers: see a generic notice only on INCOMPATIBLE.
* The manifest was regenerated with `python scripts/dashboard_manifest.py --write`.
* The slim repo was built locally but **not pushed**. Pushing it is left to the lead.
* Deploy order matters:
  1. Merge v9 first, so v9 main's manifest carries `schema_version: 1`.
  2. Then push the slim build.

  If the slim build goes first, it reports INCOMPATIBLE (data schema None) until v9 main has the
  new manifest. That is the intended fail-safe direction.

## Tests

| repo | tests |
|---|---|
| Pro | `tests/test_lab_identity.py` (14), `tests/test_season_calendar.py` (21), `tests/test_run_status.py` (12); `test_prediction_lab_paths`, `test_regression_harness`, `test_studies`, `test_requirements_cover_imports` still pass |
| v11 | `scripts/v11_tests.py` (the repo's CI harness), with new R2 sections for the registry, the v9 equality checks, seasons and rollover, the af results, the strict date window and missing p_model; all checks pass |
| v9 | `tests/test_dashboard_compat.py` (9), `tests/test_edge_engine_missing_probability.py` (8); existing dashboard manifest and mirror tests still pass |

## Open risks

* identity/1.0 covers 18 leagues. Leagues outside the map rely on stage 1 only. The single
  remaining collision (Bray) needs owner review before it becomes an alias. A future map is a new
  `IDENTITY_VERSION`, never an edit to this one.
* `season_stage` (a BASE feature) uses a min/max over the whole season group. That uses the season
  END date, which is a mild look-ahead. It was already present in 1.0 and is noted here, not fixed.
* Data v2 changes default feature values (`season_stage`, identity), so lab results under 2.0 are
  not comparable to 1.0. They are published side by side.
* The scout regression harness calls `load_fixtures()`, so it now gets canonical identity too.
  Scout candidate leagues have no calendar entry and keep the legacy bucket, by design.
* Saudi Pro League and K-League 1 grading depends on v9 continuing to commit `af_history.parquet`.
  `parse_af_history` degrades to "nothing graded" if the file is missing.
* The data-side manifest compatibility check uses path plus byte size for code entries. An edit that
  keeps the same size would not show as STALE_BUILD. The schema version is the hard gate.
