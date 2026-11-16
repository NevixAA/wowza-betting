# API-Football prices: move from the public repos to the private store

Status: PHASE 1 (inventory) and PHASE 2 (copy, no deletes) done 2026-11-16, for classes A, B and O. PHASE 3 is a plan only.
Nothing in any public repo was deleted or changed, and no live reader or writer was touched.
This document contains no prices: only paths, counts, sizes and dates.

* **Why.** API-Football's terms (https://www.api-football.com/terms) grant no licence to publish its data and forbid resale. The four repos below are PUBLIC.
* **Owner decisions (2026-11-16).** Class A (raw price archives) and class B (prices inside our own tip, settlement and ledger files) both move private. Models must not be damaged: the parity gate in section 5 is a hard gate. Class O (OddsAPI-only raw archives) moves too, together with A and B. Git history: **kept for now**; a history purge is deferred and decided after go-live (section 6).
* **Repos and commits inventoried** (all PUBLIC, checked with `gh repo view`):

| Repo | Branch | Commit |
|---|---|---|
| NevixAA/wowza-betting | main | `74c2dd886cb7a9a5ab6d8e2fc56ac69248447a9d` |
| NevixAA/wowza-betting | data-truth-100 | `258abac30e055080008d7d108f33ed7cd28fec58` |
| NevixAA/wowza-betting | prep/bars-oddsapi-hist | `8f1e06b9dd087e597deaee9fbf87b39fd715ce0a` |
| NevixAA/wowzaV9-Pro | main | `b73c5bc3621b4f3e75ec41fd48b7dbd3478534e1` |
| NevixAA/wowza_v11 | main | `3a050052e7b638272b7aa3c20b8db259629c836c` |
| NevixAA/wowza-dashboard | main | `7c01765c6d623259620a529e82d0730151f0bca6` |

`wowza-dashboard` holds no price files (0 files with a price column). It does READ class B files at runtime from v9's public raw URLs (section 4.4).

## 1. Classes

| Class | Meaning | Action |
|---|---|---|
| **A** | Raw bookmaker price archives from API-Football, or mixed API-Football + OddsAPI where a `source` column says which | Copied private (done). Move private at cut-over. |
| **B** | Per-row prices inside our own records: ledgers, tips, stake plans, shadows, settlements, research and backtest frames | Copied private (done, owner decision). At cut-over, the public side keeps a price-free view. |
| **C** | Derived statistics: counts, CLV %, ROI, coverage, band averages (`avg_odds`, `median_clv` and similar) | Stay public. Not listed row by row; the rule is in section 1.1. |
| **O** | Raw archives from **OddsAPI only** (no API-Football rows) | Copied private (done; owner decision 2026-11-16, "move them too"). Switches over with A and B on Nov 22-24, behind the same parity gate. |
| **F** | football-data.co.uk history (`output/fd_history.parquet`; also the raw `output/collect/fd_raw/*.csv` downloads on data-truth-100 and prep, whose bookmaker columns use football-data names such as `B365H`) | Not copied. A free public source with its own terms; an owner decision. |

### 1.1 How a file was classified (reproducible)

* Every CSV, Parquet and price JSON file at the recorded commits was opened and its column NAMES read. A file counts as price-bearing when one column is a per-row price, for example `odds`, `price`, `odds_over25`, `entry_odds`, `closing_odds`, `market_odds`, `implied_prob_over` or `api_implied_*`.
* Aggregates were excluded: `avg_*`, `mean_*` and `median_*` columns, odds bands, CLV and coverage. A file with only those columns is class C.
* Fantasy (FPL) `price` columns are FPL player prices, not bookmaker prices, and are excluded.
* Provider comes from the writer's code (`v3.football.api-sports.io` means API-Football; `api.the-odds-api.com` means OddsAPI). Where a `source` or `odds_source` column exists, its row counts are shown.
* `newformat_odds_history.csv` (from 2025-06) and `standard_sidemarket_odds_history.csv` (from 2024-08) have no per-row source column. Their only writers today are the API-Football Bet365 capture scripts, so they are class A in full. Rows written before those scripts existed are not labelled. Treat their provider as "API-Football per current writer".

## 2. Inventory - class A (raw price archives: move private)

| Repo @ branch | Path | Dataset | Provider | Files | MB | Rows | Date range | Source mix (rows) | Writer | Readers (workflows / code / dashboard / tests) |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| wowza-betting @ main | `output/book_quotes/2026-1*` | book_quotes | API-Football (source=api_football) | 44 | 205.3 | 1,283,556 | 2026-10-04 .. 2026-11-16 | api_football: 1,283,556 | nf_odds_capture.yml + std_odds_capture.yml -> scripts/capture_{nf_odds,std_sidemarket_odds}_forward.py via src/book_quotes.py | wf: nf_odds_capture.yml, std_odds_capture.yml; code (13): scripts/capture_nf_odds_forward.py, scripts/capture_std_sidemarket_odds_forward.py, scripts/edge_engine_replay.py, scripts/report_specials_coverage.py, scripts/v91_readiness.py, src/book_quotes.py, src/cloudbet/feed.py, src/clv_types.py ...; dashboard (2): market_intel.py, 21_📡_Market_Intelligence.py; tests: 7 |
| wowza-betting @ main | `output/newformat_odds_history.csv` | newformat_odds_history | API-Football Bet365 (nf_odds_capture) | 1 | 8.1 | 70,046 | 2025-06-23 .. 2026-11-16 | - | nf_odds_capture.yml -> scripts/capture_nf_odds_forward.py | wf: nf_odds_capture.yml; code (12): scripts/capture_nf_odds_forward.py, scripts/integrity_af_join_replay.py, scripts/integrity_price_provenance.py, scripts/train_repaired_challengers.py, src/data_loader.py, src/drift.py, src/feature_engineering.py, src/integrity/price_provenance.py ...; dashboard (0): ; tests: 2 |
| wowza-betting @ main | `output/odds_heartbeat/` | odds_heartbeat | API-Football (capture heartbeat) | 40 | 1.4 | 11,702 | 2026-10-22 .. 2026-11-16 | - | nf_odds_capture.yml + std_odds_capture.yml -> src/odds_heartbeat.py | wf: nf_odds_capture.yml, std_odds_capture.yml; code (5): scripts/capture_nf_odds_forward.py, scripts/capture_std_sidemarket_odds_forward.py, src/capture_schedule.py, src/odds_heartbeat.py, update_results.py; dashboard (0): ; tests: 3 |
| wowza-betting @ main | `output/player_prop_odds_history*` | player_prop_odds_history | mixed: API-Football + OddsAPI (source column) | 103 | 88.8 | 712,893 | 2026-06-28 .. 2026-11-16 | api_football: 429,253, oddsapi: 283,640 | player_props.yml + prop_odds_snapshot.yml -> player_model/odds_fetcher.py, scripts/snapshot_prop_odds.py; src/rolling_store.py rolls to _archive/ | wf: player_props.yml, prop_odds_snapshot.yml; code (7): player_model/clv_tracker.py, player_model/odds_fetcher.py, player_model/predict.py, player_model/prop_odds_af.py, scripts/snapshot_prop_odds.py, src/market_movement.py, src/rolling_store.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/prop_quotes/2026-1*` | prop_quotes | mixed: API-Football + OddsAPI (source column) | 27 | 46.8 | 378,751 | 2026-10-14 .. 2026-11-16 | api_football: 240,864, oddsapi: 137,887 | player_props.yml + prop_odds_snapshot.yml -> player_model/odds_fetcher.py (+ prop_odds_af.py for API-Football) | wf: player_props.yml, prop_odds_snapshot.yml; code (4): player_model/odds_fetcher.py, player_model/prop_odds_af.py, scripts/capture_exchange_odds.py, src/rolling_store.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/specials_quotes/2026-11/2026-11-1*` | specials_quotes | API-Football | 14 | 56.6 | 251,356 | 2026-11-10 .. 2026-11-16 | - | nf_odds_capture.yml + std_odds_capture.yml -> capture scripts via src/specials_quotes.py | wf: nf_odds_capture.yml, std_odds_capture.yml; code (5): scripts/capture_nf_odds_forward.py, scripts/capture_std_sidemarket_odds_forward.py, scripts/report_specials_coverage.py, src/side_odds_quotes.py, src/specials_quotes.py; dashboard (0): ; tests: 2 |
| wowza-betting @ main | `output/standard_sidemarket_odds_history.csv` | standard_sidemarket_odds_history | API-Football Bet365 (std_odds_capture) | 1 | 8.5 | 76,323 | 2024-08-10 .. 2026-11-16 | - | std_odds_capture.yml -> scripts/capture_std_sidemarket_odds_forward.py | wf: std_odds_capture.yml; code (12): scripts/capture_std_sidemarket_odds_forward.py, scripts/ht_backtest.py, scripts/integrity_af_join_replay.py, scripts/integrity_price_provenance.py, scripts/train_repaired_challengers.py, src/data_loader.py, src/integrity/price_provenance.py, src/ledger.py ...; dashboard (0): ; tests: 2 |
| wowzaV9-Pro @ main | `data/season_2026_27/live_odds_snapshots/dt=2026-*` | live_odds_snapshots | API-Football (live /odds/live) | 258 | 6.1 | 337,825 | 2026-08-30 .. 2026-11-16 | api_football:odds/live: 337,825 | pro_live_odds.yml -> src/pipelines/live_odds.py | wf: pro_live_odds.yml; code (6): config/pro_config.py, src/data/season_store.py, src/monitoring/manifest.py, src/monitoring/scheduler.py, src/pipelines/live_odds.py, src/schemas.py; dashboard (0): ; tests: 1 |
| wowzaV9-Pro @ main | `data/season_2026_27/market_snapshots/dt=2026-*` | market_snapshots | mixed: imported from v9 predictions + capture histories (odds_source) | 643 | 11.2 | 440,964 | 2024-08-10 .. 2026-11-23 | REAL: 440,964, v9:predictions#2: 264,435, v9:odds_histories#0: 176,529 | pro_collect.yml -> src/importers/current_wowza.py (imports v9 predictions + capture histories) | wf: -; code (15): config/pro_config.py, src/architecture/asof.py, src/data/season_store.py, src/importers/current_wowza.py, src/monitoring/manifest.py, src/monitoring/scheduler.py, src/pipelines/data_quality.py, src/pipelines/paper_1x2.py ...; dashboard (0): ; tests: 1 |
| wowzaV9-Pro @ main | `data/season_2026_27/scout_odds/dt=2026-1*` | scout_odds | API-Football | 1,921 | 22.8 | 622,568 | 2026-10-08 .. 2026-11-16 | api_football:scout: 622,568 | pro_scout.yml + pro_scout_near.yml -> src/scout/collect.py | wf: pro_scout.yml, pro_scout_near.yml; code (8): config/pro_config.py, src/data/season_store.py, src/monitoring/collection_policy.py, src/pipelines/data_quality.py, src/scout/collect.py, src/scout/hypotheses.py, src/scout/promotion.py, src/scout/report.py; dashboard (0): ; tests: 4 |
| wowzaV9-Pro @ main | `data/season_2026_27/scout_specials_quotes/2026-11/2026-11-1*` | scout_specials_quotes | API-Football | 15 | 286.9 | 1,078,929 | 2026-11-10 .. 2026-11-16 | - | pro_scout.yml + pro_scout_near.yml -> src/scout/specials.py | wf: pro_scout.yml, pro_scout_near.yml; code (1): src/scout/specials.py; dashboard (0): ; tests: 1 |
| wowzaV9-Pro @ main | `docs/league_scout_af_odds_probe_2026_10_08.csv` | league_scout_af_odds_probe | API-Football probe | 1 | 0.0 | 292 | - | - | one-off probe 2026-10-08 (no workflow) | wf: -; code (1): src/scout/leagues.py; dashboard (0): ; tests: 0 |

## 3. Inventory - class B (prices inside our own records: owner decided they move private)

"Readers" are every file at the same commit whose code names the dataset (`git grep` on .py/.yml/.sh). The "dashboard" count covers `pages/` and `dashboard_data/`.

| Repo @ branch | Path | Dataset | Provider | Files | MB | Rows | Date range | Source mix (rows) | Writer | Readers (workflows / code / dashboard / tests) |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| wowza-betting @ main | `docs/quality_tiers_2026_10/seed_first_sight_tips.csv` | - | own record | 1 | 0.3 | 1,555 | - | - | see workflows in readers column | wf: -; code (1): src/research/quality_tiers.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/backtest_all_leagues.csv` | - | own record | 1 | 25.9 | 40,060 | 2019-09-28 .. 2026-04-23 | - | see workflows in readers column | wf: -; code (3): scripts/integrity_af_join_replay.py, src/data_loader.py, src/integrity/data_truth.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/backtest_results_btts.csv` | - | own record | 1 | 14.3 | 19,669 | 2020-10-20 .. 2026-10-28 | - | see workflows in readers column | wf: backtest.yml; code (1): scripts/gate_study.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/backtest_results_newformat.csv` | - | own record | 1 | 3.0 | 3,260 | 2025-08-09 .. 2026-10-29 | - | see workflows in readers column | wf: backtest.yml, retrain.yml; code (3): retrain.py, scripts/gate_study.py, scripts/league_thresholds.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/backtest_results_over15.csv` | - | own record | 1 | 14.0 | 19,669 | 2020-10-20 .. 2026-10-28 | - | see workflows in readers column | wf: backtest.yml; code (2): scripts/gate_study.py, src/market_edge.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/backtest_results_over35.csv` | - | own record | 1 | 14.0 | 19,669 | 2020-10-20 .. 2026-10-28 | - | see workflows in readers column | wf: backtest.yml; code (1): scripts/gate_study.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/backtest_results_standard.csv` | - | own record | 1 | 17.9 | 19,416 | 2020-10-20 .. 2026-10-28 | - | see workflows in readers column | wf: backtest.yml, backtest_matrix.yml, retrain.yml; code (4): retrain.py, scheduler/git_push_outputs.py, scripts/gate_study.py, scripts/league_thresholds.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/bets.csv` | - | own record | 1 | 0.0 | 42 | 2026-11-18 .. 2026-11-23 | - | see workflows in readers column | wf: backtest_matrix.yml, daily_summary.yml, nf_odds_capture.yml, player_props.yml, predict.yml, sharp_move_alert.yml, std_odds_capture.yml, train_standard_repaired.yml, update_results.yml; code (82): config.py, dashboard_ui.py, pipeline.py, player_model/clv_tracker.py, player_model/config.py, player_model/fantasy_distribution.py, player_model/fantasy_fixture_difficulty.py, player_model/odds_fetcher.py ...; dashboard (29): bankroll_view.py, evidence_panels.py, evidence_view.py, leagues.py, market_intel.py ...; tests: 43 |
| wowza-betting @ main | `output/bets_ledger.csv` | - | own record | 1 | 0.4 | 1,925 | 2026-04-26 .. 2026-11-23 | live: 1,925 | see workflows in readers column | wf: daily_summary.yml, predict.yml, sharp_move_alert.yml, train_standard_repaired.yml, update_results.yml; code (31): pipeline.py, scheduler/git_push_outputs.py, scripts/allocation_registry.py, scripts/backfill_clv.py, scripts/build_league_regimes.py, scripts/edge_engine_replay.py, scripts/fix_ledger_tiers.py, scripts/gate_study.py ...; dashboard (7): evidence_panels.py, leagues.py, today_view.py, v9.py, 12_💼_Portfolio.py ...; tests: 18 |
| wowza-betting @ main | `output/clv_records.csv` | - | own record | 1 | 1.6 | 10,522 | - | - | see workflows in readers column | wf: player_props.yml, prop_odds_snapshot.yml; code (7): player_model/clv_tracker.py, player_model/predict.py, scripts/snapshot_prop_odds.py, src/clv_capture.py, src/contracts/schemas.py, src/system_health.py, telegram_bot/notifier.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/direction_conflicts.csv` | - | own record | 1 | 0.0 | 7 | 2026-10-29 .. 2026-11-07 | - | see workflows in readers column | wf: predict.yml; code (2): pipeline.py, src/direction_guard.py; dashboard (2): market_intel.py, 21_📡_Market_Intelligence.py; tests: 1 |
| wowza-betting @ main | `output/drift_shadow.csv` | - | own record | 1 | 0.1 | 279 | 2026-10-26 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml; code (5): config.py, pipeline.py, scripts/standard_switch_preview.py, src/betting.py, src/drift.py; dashboard (0): ; tests: 2 |
| wowza-betting @ main | `output/edge_engine_backtest.csv` | - | own record | 1 | 2.5 | 3,945 | 2026-08-15 .. 2026-11-01 | - | see workflows in readers column | wf: -; code (2): scripts/edge_engine_replay.py, src/research/edge_replay.py; dashboard (2): market_intel.py, 21_📡_Market_Intelligence.py; tests: 1 |
| wowza-betting @ main | `output/edge_engine_shadow.csv` | - | own record | 1 | 1.5 | 2,468 | 2026-10-25 .. 2026-11-23 | ROW: 1,256, (blank): 875, MEDIAN_VALID_BOOKS: 337 | see workflows in readers column | wf: predict.yml; code (13): pipeline.py, scripts/edge_engine_replay.py, src/btts_odds.py, src/contracts/schemas.py, src/edge_af_quotes.py, src/edge_engine_shadow.py, src/ledger.py, src/market_edge.py ...; dashboard (6): evidence_panels.py, evidence_view.py, market_intel.py, tips_view.py, 1_📊_Dashboard.py ...; tests: 10 |
| wowza-betting @ main | `output/ht_ledger.csv` | - | own record | 1 | 0.0 | 80 | 2026-09-25 .. 2026-10-04 | live: 80 | see workflows in readers column | wf: daily_summary.yml, predict.yml, update_results.yml; code (10): pipeline.py, scripts/backfill_clv.py, scripts/market_movement_backfill.py, scripts/market_movement_report.py, scripts/movement_segments.py, src/ledger.py, src/market_movement.py, src/system_health.py ...; dashboard (3): today_view.py, 1_📊_Dashboard.py, 5_🎯_Success_Rates.py; tests: 2 |
| wowza-betting @ main | `output/ht_observations.csv` | - | own record | 1 | 0.1 | 565 | 2026-10-02 .. 2026-11-22 | - | see workflows in readers column | wf: predict.yml, update_results.yml; code (3): pipeline.py, src/ledger.py, update_results.py; dashboard (0): ; tests: 2 |
| wowza-betting @ main | `output/league_weekly_candidates.csv` | - | own record | 1 | 1.3 | 2,468 | 2026-10-25 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml; code (2): src/research/league_weekly.py, src/system_health.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/league_weekly_research.csv` | - | own record | 1 | 0.7 | 896 | 2026-10-25 .. 2026-11-15 | - | see workflows in readers column | wf: predict.yml, update_results.yml; code (4): src/contracts/schemas.py, src/research/league_weekly.py, src/research/league_weekly_eval.py, src/system_health.py; dashboard (1): league_research.py; tests: 2 |
| wowza-betting @ main | `output/ledger_tier_seen.csv` | - | own record | 1 | 0.0 | 69 | 2026-10-26 .. 2026-11-21 | live: 69 | see workflows in readers column | wf: predict.yml; code (1): src/ledger.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/player_ledger.csv` | - | own record | 1 | 6.6 | 39,617 | 2026-06-14 .. 2026-11-23 | - | see workflows in readers column | wf: daily_summary.yml, oneoff_dnp_recheck.yml, player_props.yml, update_results.yml; code (11): player_model/clv_tracker.py, player_model/ledger.py, player_model/pipeline.py, player_model/predict.py, player_model/props_r2_shadow.py, scripts/market_movement_backfill.py, scripts/market_movement_report.py, scripts/movement_segments.py ...; dashboard (4): props_view.py, 11_👤_Player_Props.py, 12_💼_Portfolio.py, 5_🎯_Success_Rates.py; tests: 5 |
| wowza-betting @ main | `output/player_tips.csv` | - | own record | 1 | 0.2 | 1,152 | 2026-11-16 .. 2026-11-23 | - | see workflows in readers column | wf: player_props.yml; code (8): player_model/fantasy.py, player_model/odds_fetcher.py, player_model/pipeline.py, player_model/predict.py, src/cloudbet/candidates.py, src/integrity/data_truth.py, telegram_bot/daily_pack.py, telegram_bot/notifier.py; dashboard (2): props_view.py, 11_👤_Player_Props.py; tests: 1 |
| wowza-betting @ main | `output/predictions.csv` | - | own record | 1 | 0.3 | 145 | 2026-11-16 .. 2026-11-23 | - | see workflows in readers column | wf: phase2_retrain.yml, player_props.yml, predict.yml, std_odds_capture.yml, train_standard_repaired.yml; code (50): config.py, pipeline.py, player_model/config.py, player_model/fantasy_log.py, player_model/fantasy_residual.py, player_model/feature_engineering.py, player_model/pipeline.py, player_model/predict.py ...; dashboard (7): live_view.py, today_view.py, v9.py, 0_🏠_Overview.py, 10_🤖_Agent_Analysis.py ...; tests: 14 |
| wowza-betting @ main | `output/props_shown_log.csv` | - | own record | 1 | 0.0 | 378 | 2026-10-26 .. 2026-11-16 | - | see workflows in readers column | wf: daily_summary.yml; code (3): src/contracts/schemas.py, src/system_health.py, telegram_bot/daily_pack.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/quality_tier_shadow.csv` | - | own record | 1 | 0.1 | 412 | 2026-10-26 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml; code (1): src/research/quality_tiers.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/sharp_ledger.csv` | - | own record | 1 | 0.3 | 2,457 | 2026-06-14 .. 2026-11-24 | - | see workflows in readers column | wf: daily_summary.yml, sharp_tracker.yml, update_results.yml, worldcup.yml; code (3): player_model/ledger.py, telegram_bot/notifier.py, update_results.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/sharp_tips.csv` | - | own record | 1 | 0.0 | 47 | 2026-11-18 .. 2026-11-24 | - | see workflows in readers column | wf: predict.yml, sharp_tracker.yml; code (5): scheduler/git_push_outputs.py, src/sharp_tracker.py, src/sharp_tracker_v1.py, telegram_bot/daily_pack.py, telegram_bot/notifier.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/side_bets_ledger.csv` | - | own record | 1 | 0.2 | 752 | 2026-08-09 .. 2026-11-23 | live: 752 | see workflows in readers column | wf: daily_summary.yml, predict.yml, train_standard_repaired.yml, update_results.yml; code (25): pipeline.py, scripts/allocation_registry.py, scripts/backfill_clv.py, scripts/build_league_regimes.py, scripts/edge_engine_replay.py, scripts/gate_study.py, scripts/integrity_side_edge_shadow.py, scripts/market_movement_backfill.py ...; dashboard (7): evidence_panels.py, leagues.py, today_view.py, v9.py, 12_💼_Portfolio.py ...; tests: 15 |
| wowza-betting @ main | `output/side_edge_devig_shadow.csv` | - | own record | 1 | 0.3 | 620 | 2026-11-01 .. 2026-11-22 | - | see workflows in readers column | wf: integrity_health.yml; code (2): scripts/integrity_side_edge_shadow.py, src/integrity/side_edge_devig.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/stake_audit.csv` | - | own record | 1 | 0.3 | 969 | 2026-10-14 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml; code (6): pipeline.py, src/bankroll_policy.py, src/contracts/schemas.py, src/research/league_weekly.py, src/stake_plan.py, src/system_health.py; dashboard (0): ; tests: 7 |
| wowza-betting @ main | `output/stake_plan.csv` | - | own record | 1 | 0.0 | 63 | 2026-11-18 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml; code (13): dashboard_access.py, dashboard_ui.py, pipeline.py, src/artifact_freshness.py, src/bankroll_policy.py, src/cloudbet/bot.py, src/contracts/schemas.py, src/contracts/shadow.py ...; dashboard (10): bankroll_view.py, leagues.py, production_view.py, tips_view.py, today_view.py ...; tests: 18 |
| wowza-betting @ main | `output/top_picks_log.csv` | - | own record | 1 | 0.0 | 322 | - | - | see workflows in readers column | wf: daily_summary.yml; code (4): src/contracts/schemas.py, src/system_health.py, src/top_picks.py, telegram_bot/daily_pack.py; dashboard (0): ; tests: 1 |
| wowza-betting @ main | `output/v91_shadow.csv` | - | own record | 1 | 0.5 | 1,227 | 2026-10-04 .. 2026-11-23 | - | see workflows in readers column | wf: predict.yml, update_results.yml; code (14): pipeline.py, scripts/register_shadow_experiment.py, src/capture_schedule.py, src/clv_collection_health.py, src/contracts/schemas.py, src/nf_calibration.py, src/promotion_gate.py, src/research/league_weekly.py ...; dashboard (0): ; tests: 7 |
| wowza-betting @ main | `output/weekly_league_report.json` | - | own record | 1 | 0.3 | 5 | - | - | see workflows in readers column | wf: predict.yml, update_results.yml; code (1): src/research/league_weekly.py; dashboard (2): league_research_view.py, 18_🔧_League_Research.py; tests: 2 |
| wowza-betting @ main | `output/worldcup_tips.csv` | - | own record | 1 | 0.0 | 0 | - | - | see workflows in readers column | wf: predict.yml, worldcup.yml; code (3): scheduler/git_push_outputs.py, telegram_bot/notifier.py, worldcup/tracker.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/_backfill_fixtures.parquet` | - | own record | 1 | 0.3 | 935 | 2026-05-21 .. 2026-08-25 | - | see workflows in readers column | wf: pro_backfill_results.yml; code (1): src/pipelines/backfill_history.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/_backfill_joined.parquet` | - | own record | 1 | 0.2 | 514 | 2026-05-21 .. 2026-08-18 | - | see workflows in readers column | wf: pro_backfill_results.yml; code (1): src/pipelines/backfill_history.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/season_2026_27/clv/dt=2026-*` | - | own record | 93 | 10.6 | 371,308 | - | v9:clv#0: 371,308 | see workflows in readers column | wf: pro_collect.yml; code (47): config/pro_config.py, src/analytics/core.py, src/analytics/deep_wowza_snapshot.py, src/analytics/longshot_forward.py, src/analytics/longshot_plan.py, src/analytics/model_lineage.py, src/bankroll/allocator.py, src/bankroll/data.py ...; dashboard (0): ; tests: 13 |
| wowzaV9-Pro @ main | `data/season_2026_27/combo_candidates/dt=2026-*` | - | own record | 127 | 6.8 | 96,254 | 2026-09-02 .. 2026-11-15 | pro:bet_builder/canonical: 96,254 | see workflows in readers column | wf: pro_bet_builder.yml; code (5): config/pro_config.py, src/combo/canonical.py, src/data/season_store.py, src/monitoring/manifest.py, src/schemas.py; dashboard (0): ; tests: 2 |
| wowzaV9-Pro @ main | `data/season_2026_27/combo_legs/dt=2026-*` | - | own record | 127 | 5.7 | 265,509 | 2026-09-02 .. 2026-11-15 | pro:bet_builder/canonical: 265,509 | see workflows in readers column | wf: pro_bet_builder.yml; code (5): config/pro_config.py, src/combo/canonical.py, src/data/season_store.py, src/monitoring/manifest.py, src/schemas.py; dashboard (0): ; tests: 2 |
| wowzaV9-Pro @ main | `data/season_2026_27/combo_settlements/dt=2026-*` | - | own record | 123 | 3.6 | 53,057 | 2026-08-17 .. 2026-11-18 | pro:bet_builder/canonical: 53,057 | see workflows in readers column | wf: pro_bet_builder.yml; code (5): config/pro_config.py, src/combo/canonical.py, src/data/season_store.py, src/monitoring/manifest.py, src/schemas.py; dashboard (0): ; tests: 2 |
| wowzaV9-Pro @ main | `data/season_2026_27/feature_snapshots/dt=2026-*` | - | own record | 552 | 72.0 | 84,773 | 2026-08-17 .. 2026-11-23 | v9:predictions#3: 84,773 | see workflows in readers column | wf: -; code (7): config/pro_config.py, src/data/season_store.py, src/importers/current_wowza.py, src/monitoring/manifest.py, src/pipelines/data_quality.py, src/pipelines/xg_value.py, src/quality.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/season_2026_27/movement_observations/dt=2026-*` | - | own record | 87 | 196.2 | 5,398,263 | 2026-08-23 .. 2026-08-23 | v9:movement_observations#0: 5,398,263 | see workflows in readers column | wf: -; code (7): config/pro_config.py, src/data/season_store.py, src/importers/v11_research.py, src/monitoring/manifest.py, src/monitoring/weekly_audit.py, src/pipelines/data_quality.py, src/quality.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/season_2026_27/player_props/dt=2026-*` | - | own record | 93 | 21.4 | 1,522,613 | 2026-06-14 .. 2026-11-23 | v9:player_props#0: 1,522,613 | see workflows in readers column | wf: pro_backfill_results.yml, pro_bet_builder.yml, pro_research.yml, pro_weekly_audit.yml; code (18): config/pro_config.py, src/bankroll/allocator.py, src/combo/builder.py, src/combo/match_picture.py, src/combo/player_dependency.py, src/combo/settle.py, src/contract.py, src/data/season_store.py ...; dashboard (0): ; tests: 1 |
| wowzaV9-Pro @ main | `data/season_2026_27/settlements/dt=2026-*` | - | own record | 188 | 19.6 | 487,840 | 2020-09-19 .. 2026-11-23 | v9:ledgers#0: 449,669, v9:ledgers#1: 38,171 | see workflows in readers column | wf: pro_bet_builder.yml, pro_research.yml; code (26): config/pro_config.py, src/bankroll/run.py, src/combo/canonical.py, src/combo/events.py, src/data/season_store.py, src/importers/current_wowza.py, src/monitoring/manifest.py, src/monitoring/ml_learning_health.py ...; dashboard (0): ; tests: 3 |
| wowzaV9-Pro @ main | `data/season_2026_27/settlements_backfill/dt=2026-05-21/run=backfill-20260819T*` | - | own record | 2 | 0.4 | 778 | 2026-05-21 .. 2026-08-18 | v9:git_backfill: 778 | see workflows in readers column | wf: -; code (11): config/pro_config.py, src/data/season_store.py, src/monitoring/manifest.py, src/pipelines/backfill_history.py, src/pipelines/bet_builder.py, src/pipelines/btts_eval.py, src/pipelines/data_quality.py, src/pipelines/paper_1x2.py ...; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `experiments/valuable_2026_10/all_fixture_settled.parquet` | - | own record | 1 | 0.2 | 3,032 | - | ALL_FIXTURE: 3,032 | see workflows in readers column | wf: -; code (2): src/studies/edge_reassessment.py, src/studies/learned_tiers.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/clv_enriched.csv` | - | own record | 1 | 65.0 | 371,308 | 2026-08-17 .. 2026-11-22 | - | see workflows in readers column | wf: pro_collect.yml; code (2): src/market/clv_schema.py, src/monitoring/manifest.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/paper_1x2.csv` | - | own record | 1 | 0.1 | 394 | 2026-09-05 .. 2026-11-15 | - | see workflows in readers column | wf: pro_paper_1x2.yml, pro_team_stats.yml, pro_tip_scoreboard.yml; code (2): src/pipelines/paper_1x2.py, src/pipelines/tip_scoreboard.py; dashboard (0): ; tests: 1 |
| wowzaV9-Pro @ main | `output/player_v3/consensus/two_sided_pairs.csv` | - | own record | 1 | 0.0 | 234 | 2026-10-14 .. 2026-11-08 | - | see workflows in readers column | wf: pro_player_v3_forward.yml; code (1): src/player_v3/consensus.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/player_v3/player_market_residual.csv` | - | own record | 1 | 0.0 | 20 | - | - | see workflows in readers column | wf: -; code (2): src/player_v3/market.py, src/player_v3/report.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/price_trader/paper_trades.csv` | - | own record | 1 | 0.1 | 581 | 2026-10-16 .. 2026-11-22 | - | see workflows in readers column | wf: -; code (1): src/price_trader/paper.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/price_trader/price_paths.parquet` | - | own record | 1 | 0.4 | 18,807 | - | - | see workflows in readers column | wf: -; code (3): src/price_trader/lab.py, src/price_trader/paper.py, src/price_trader/research.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/price_trader/trades.parquet` | - | own record | 1 | 0.1 | 1,837 | 2026-08-21 .. 2026-11-22 | live: 1,837 | see workflows in readers column | wf: pro_price_trader.yml; code (4): src/price_trader/lab.py, src/price_trader/paper.py, src/price_trader/paths.py, src/price_trader/research.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/forward_first_sight.csv` | - | own record | 1 | 0.6 | 2,823 | 2026-08-17 .. 2026-11-15 | - | see workflows in readers column | wf: pro_standard_repair.yml; code (2): src/studies/standard_repair/forward.py, src/studies/standard_repair/pre_repair.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/forward_shadow.csv` | - | own record | 1 | 0.2 | 318 | 2026-10-28 .. 2026-11-08 | - | see workflows in readers column | wf: pro_standard_repair_shadow.yml; code (3): src/monitoring/research_index.py, src/studies/standard_repair/shadow.py, src/studies/standard_repair/shadow_registry.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/forward_shadow_settled.csv` | - | own record | 1 | 0.0 | 303 | 2026-10-29 .. 2026-11-01 | - | see workflows in readers column | wf: pro_standard_repair_shadow.yml; code (1): src/studies/standard_repair/shadow.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/forward_shadow_sightings/2026-1*` | - | own record | 5 | 0.9 | 3,735 | - | - | see workflows in readers column | wf: -; code (0): ; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/pre_repair/forward_first_sight.csv` | - | own record | 1 | 0.5 | 2,499 | 2026-08-17 .. 2026-11-01 | - | see workflows in readers column | wf: pro_standard_repair.yml; code (2): src/studies/standard_repair/forward.py, src/studies/standard_repair/pre_repair.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/standard_repair/top5_panel_repair.csv` | - | own record | 1 | 5.0 | 13,360 | 2024-07-26 .. 2026-10-26 | - | see workflows in readers column | wf: pro_standard_repair.yml, pro_standard_repair_benchmark.yml; code (1): src/studies/standard_repair/benchmark_out.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/studies/btts_contradiction_first_logged.csv` | - | own record | 1 | 0.1 | 459 | 2026-08-22 .. 2026-11-01 | live: 459 | see workflows in readers column | wf: -; code (1): src/studies/btts_contradiction.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/studies/learned_tiers/walk_forward_rows.csv` | - | own record | 1 | 0.4 | 1,955 | - | SNAPSHOT_HISTORY: 1,681, PRO_SHADOW: 274 | see workflows in readers column | wf: pro_learned_tiers.yml; code (1): src/studies/learned_tiers.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/studies/ou_challenger_forward.csv` | - | own record | 1 | 0.3 | 1,174 | 2026-10-08 .. 2026-11-23 | - | see workflows in readers column | wf: pro_studies.yml; code (2): src/studies/ou_challenger.py, src/studies/report.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/studies/weather/match_venues.csv` | - | own record | 1 | 4.3 | 31,556 | 2023-07-01 .. 2026-10-16 | - | see workflows in readers column | wf: -; code (1): src/studies/weather_surprise.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `output/valuable_shadow/valuable_shadow.csv` | - | own record | 1 | 0.9 | 1,782 | 2026-10-21 .. 2026-11-22 | - | see workflows in readers column | wf: pro_valuable_shadow.yml; code (5): src/studies/btts_league_regime.py, src/studies/candidate_forward.py, src/studies/learned_tiers.py, src/studies/valuable/report.py, src/studies/valuable/shadow.py; dashboard (0): ; tests: 0 |
| wowza_v11 @ main | `output/v11_market_microstructure.csv` | - | own record | 1 | 52.8 | 83,730 | 2026-11-16 .. 2026-11-16 | - | see workflows in readers column | wf: v11_collect.yml; code (2): scripts/v11_microstructure.py, src/research_state.py; dashboard (0): ; tests: 0 |
| wowza_v11 @ main | `output/v11_market_movement_detail.csv` | - | own record | 1 | 33.8 | 83,730 | - | - | see workflows in readers column | wf: v11_collect.yml; code (4): scripts/v11_market_movement.py, scripts/v11_microstructure.py, scripts/v11_tests.py, src/research_state.py; dashboard (0): ; tests: 0 |
| wowza_v11 @ main | `output/v11_shadow_log.csv` | - | own record | 1 | 0.9 | 2,463 | 2026-08-10 .. 2026-11-16 | exchange: 1,486, cross_book_median: 743, (blank): 170, single_book: 64 | see workflows in readers column | wf: v11_collect.yml; code (4): scripts/v11_grade.py, scripts/v11_residual.py, scripts/v11_shadow.py, src/research_state.py; dashboard (0): ; tests: 0 |
| wowza_v11 @ main | `output/v11_shadow_snapshots.csv` | - | own record | 1 | 30.6 | 86,951 | 2026-08-10 .. 2026-11-16 | cross_book_median: 50,531, exchange: 24,123, (blank): 9,150, single_book: 3,147 | see workflows in readers column | wf: v11_collect.yml; code (6): scripts/v11_fit_evidence.py, scripts/v11_market_movement.py, scripts/v11_microstructure.py, scripts/v11_momentum_control.py, scripts/v11_shadow.py, src/research_state.py; dashboard (0): ; tests: 0 |

### 3.1 Class B items the owner should still look at

The owner has decided that B moves private. These are the items where that decision has side effects worth a second look:

* **Settlement ledgers** (`bets_ledger.csv`, `side_bets_ledger.csv`, `ht_ledger.csv`, `player_ledger.csv`, Pro `settlements/`, Pro `clv/`): these are the system of record for P/L and CLV. After the cut-over, the private copy becomes authoritative. The public side gets new price-free views (new files), and the originals are never rewritten.
* **Tips files read by the public dashboard** (`predictions.csv`, `bets.csv`, `stake_plan.csv`, `player_tips.csv`, `top_picks_log.csv`, `v91_shadow.csv`, `edge_engine_*`, `league_weekly_research.csv`): the dashboard shows "odds taken" today. After the cut-over it shows only what the price-free view carries (result, stake, P/L, ROI, CLV %). If the owner wants prices on the dashboard, it must read the private store with a token (section 4.4).
* **Backtest and research frames** (`backtest_results_*.csv`, `backtest_all_leagues.csv`, Pro `feature_snapshots/` with `implied_prob_*` / `api_implied_*`, Pro `_backfill_*.parquet`, v11 movement files): they mix football-data.co.uk history with API-Football backfill and OddsAPI prices. Moving them private is safe for models only if the training loaders read the private copy (section 5).

## 4. Other findings

### 4.1 Class O (OddsAPI-only raw archives: copied, moves with A/B) and class F (not copied)

Other OddsAPI items checked and left as they are:
* `sharp_ledger.csv`, `sharp_tips.csv`, `worldcup_tips.csv` and `corrections/sharp_ledger_restated.csv` are our own records priced from OddsAPI. They are class B and were already copied.
* `output/threshold_proposals/bars_oddsapi_hist/*`, `output/collect/oddsapi_history_{probe,plan}.json` and `output/oddsapi_quota_health.json` (prep branch / main) hold thresholds, bands (`odds_min` / `odds_max`), counts and credit plans, not per-row prices. They are class C.
* The OddsAPI history itself already lives only in private wowza-exec (`odds-history`).

| Repo @ branch | Path | Dataset | Provider | Files | MB | Rows | Date range | Source mix (rows) | Writer | Readers (workflows / code / dashboard / tests) |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| wowza-betting @ main | `odds_history_v9.json` | odds_history_v9.json | OddsAPI (predict) | 1 | 0.2 | 339 | - | - | predict.yml -> src/predict.py / src/drift.py (OddsAPI) | wf: predict.yml, train_standard_repaired.yml; code (5): config.py, src/drift.py, src/ledger.py, src/predict.py, update_results.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/book_odds_snapshots*` | book_odds_snapshots | OddsAPI (predict) | 81 | 51.8 | 431,608 | 2026-08-19 .. 2026-11-16 | - | predict.yml -> src/predict.py + src/btts_odds.py (OddsAPI); src/rolling_store.py rolls to _archive/ | wf: predict.yml; code (11): scripts/edge_engine_replay.py, scripts/integrity_side_edge_shadow.py, scripts/nf_parity_preview.py, src/btts_odds.py, src/contracts/schemas.py, src/edge_af_quotes.py, src/edge_engine_shadow.py, src/integrity/side_edge_devig.py ...; dashboard (0): ; tests: 3 |
| wowza-betting @ main | `output/exchange_quotes/2026-1*` | exchange_quotes | OddsAPI exchanges | 2 | 2.9 | 14,291 | 2026-10-20 .. 2026-11-15 | - | exchange_odds_capture.yml -> scripts/capture_exchange_odds.py (OddsAPI) | wf: exchange_odds_capture.yml; code (1): scripts/capture_exchange_odds.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/newformat_odds_dense.csv` | newformat_odds_dense | OddsAPI (predict drift archive) | 1 | 1.8 | 17,866 | 2026-08-15 .. 2026-11-16 | - | predict.yml -> src/drift.py (OddsAPI payload) | wf: nf_odds_capture.yml, predict.yml; code (3): src/drift.py, src/market_movement.py, update_results.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/sharp_history/2026-*` | sharp_history | OddsAPI (sharp_tracker) | 6 | 35.3 | 5,995 | - | - | sharp_tracker.yml -> src/sharp_tracker.py (OddsAPI) | wf: predict.yml, sharp_tracker.yml; code (3): scheduler/git_push_outputs.py, src/sharp_tracker.py, src/sharp_tracker_v1.py; dashboard (0): ; tests: 0 |
| wowza-betting @ main | `output/standard_odds_history.csv` | standard_odds_history | OddsAPI (predict drift archive) | 1 | 1.1 | 12,118 | 2026-08-09 .. 2026-11-16 | - | predict.yml -> src/drift.py (OddsAPI payload) | wf: predict.yml; code (7): scripts/integrity_af_join_replay.py, scripts/integrity_price_provenance.py, src/data_loader.py, src/drift.py, src/market_movement.py, telegram_bot/notifier.py, update_results.py; dashboard (0): ; tests: 3 |
| wowza-betting @ main | `output/worldcup_history.json` | worldcup_history | OddsAPI (worldcup tracker) | 1 | 1.7 | 204 | - | - | worldcup.yml -> worldcup/tracker.py (OddsAPI) | wf: predict.yml, worldcup.yml; code (3): player_model/pipeline.py, scheduler/git_push_outputs.py, worldcup/tracker.py; dashboard (0): ; tests: 0 |
| wowzaV9-Pro @ main | `data/season_2026_27/book_odds_snapshots/dt=2026-*` | pro_book_odds_snapshots | OddsAPI (imported from v9 book_odds_snapshots.csv; source=v9:book_odds on every row) | 302 | 8.3 | 431,608 | 2026-08-19 .. 2026-11-16 | v9:book_odds#0: 431,608, REAL: 431,608 | pro_collect.yml -> src/importers/current_wowza.py (imports v9 book_odds_snapshots / book_quotes) | wf: pro_collect.yml, pro_standard_repair_shadow.yml; code (22): config/pro_config.py, src/architecture/canonical_market.py, src/contract.py, src/data/season_store.py, src/data/v9_source.py, src/importers/current_wowza.py, src/monitoring/manifest.py, src/monitoring/snapshot_coverage.py ...; dashboard (0): ; tests: 4 |

| Repo @ branch | Path | Dataset | Provider | Files | MB | Rows | Date range | Source mix (rows) | Writer | Readers (workflows / code / dashboard / tests) |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| wowza-betting @ main | `output/fd_history.parquet` | fd_history | football-data.co.uk | 1 | 0.7 | 66,421 | 2019-02-15 .. 2026-11-09 | - | predict.yml / retrain.yml -> src/data_loader.py (football-data.co.uk download) | wf: backtest_matrix.yml, predict.yml, retrain.yml, train_standard_repaired.yml; code (12): player_model/fantasy_fixture_difficulty.py, scripts/audit_data_files.py, scripts/collect_standard_team_cards.py, scripts/dashboard_manifest.py, scripts/integrity_af_join_replay.py, scripts/integrity_feature_parity.py, scripts/integrity_price_provenance.py, src/data_loader.py ...; dashboard (1): 6_⚡_Live.py; tests: 2 |

### 4.2 Other branches and history

| Repo @ branch | Class | Files with price columns | MB | Rows | Files not byte-identical to main |
|---|---|---:|---:|---:|---:|
| wowza-betting @ data-truth-100 | A | 158 | 254.8 | 1,840,228 | 6 |
| wowza-betting @ data-truth-100 | B | 73 | 119.4 | 229,223 | 63 |
| wowza-betting @ data-truth-100 | C | 1 | 0.0 | 58 | 0 |
| wowza-betting @ data-truth-100 | F | 1 | 0.7 | 66,213 | 1 |
| wowza-betting @ data-truth-100 | O | 82 | 86.1 | 428,671 | 6 |
| wowza-betting @ main | A | 230 | 415.5 | 2,784,627 | - |
| wowza-betting @ main | B | 34 | 106.7 | 194,730 | - |
| wowza-betting @ main | C | 1 | 0.0 | 58 | - |
| wowza-betting @ main | F | 1 | 0.7 | 66,421 | - |
| wowza-betting @ main | O | 93 | 94.8 | 482,421 | - |
| wowza-betting @ prep/bars-oddsapi-hist | A | 158 | 254.8 | 1,840,228 | 6 |
| wowza-betting @ prep/bars-oddsapi-hist | B | 83 | 120.1 | 232,672 | 73 |
| wowza-betting @ prep/bars-oddsapi-hist | C | 1 | 0.0 | 58 | 0 |
| wowza-betting @ prep/bars-oddsapi-hist | F | 1 | 0.7 | 66,213 | 1 |
| wowza-betting @ prep/bars-oddsapi-hist | O | 82 | 86.1 | 428,671 | 6 |
| wowzaV9-Pro @ main | A | 2,838 | 327.0 | 2,480,578 | - |
| wowzaV9-Pro @ main | B | 1,417 | 415.9 | 8,738,021 | - |
| wowzaV9-Pro @ main | C | 3 | 0.0 | 21 | - |
| wowzaV9-Pro @ main | O | 302 | 8.3 | 431,608 | - |
| wowza_v11 @ main | B | 4 | 118.1 | 256,874 | - |

* The table above covers the branches named in the brief. **Every one of the 135 remote branches of wowza-betting, all 26 of wowzaV9-Pro and all 11 of wowza_v11 carry price files.** Moving the files on `main` does not remove them from those branches. Stale branches must be deleted, or the history purge or privacy option (section 6) must be used.
* Git **history** keeps every committed version: wowza-betting has 10,083 commits, wowzaV9-Pro 2,718 and wowza_v11 749. A move on HEAD does not remove anything from history.
* **wowza-betting has 1 public fork.** Neither a history purge nor making the repo private removes data from an existing fork. When a public repo is made private, GitHub leaves its public forks public in a separate network.

### 4.3 Models and features that read API-Football prices (parity gate scope)

| Consumer | What it reads | Path today | How it gets the identical value after the cut-over |
|---|---|---|---|
| All seven v9 team pickles (`model_v9_standard`, `_newformat`, `_btts`, `_over15`, `_over35`, `model_ht_over05`, `_over15`): feature `bookmaker_overround` | `odds_over25` / `odds_under25` in the training frame | Standard: `fd_history.parquet` (class F). New-format and side markets: `newformat_odds_history.csv`, `standard_sidemarket_odds_history.csv` (A), plus `af_odds_history.parquet` when present (not committed) | Same bytes from the private store via one path resolver (`PRICE_STORE_DIR`). `src/feature_engineering.py` lines ~510 and ~949 run unchanged. |
| `ou25_pair_is_price` (data-truth-100 `src/data_loader.py` ~740, `retrain_harness/candidates.py`) | Same O/U 2.5 pair | As above | Derived from the same columns, so identical by construction. The frame-hash gate proves it. |
| `implied_prob_over/under`, `api_implied_*`, `api_overround` (`src/integrity/price_provenance.PRICE_DERIVED_FEATURES`) | 1/odds of the same columns | As above | Identical by construction. Proven by frame hash. |
| `src/data_loader.py` fills `odds_btts`, `odds_over15`, `odds_over35` (lines ~104, ~865) | `newformat_odds_history.csv`, `standard_sidemarket_odds_history.csv` | A | Resolver. |
| `scripts/train_repaired_challengers.py` (first pre-kickoff two-sided quote) | Same two files | A | Resolver. |
| `src/market_movement.CurveIndex`, `src/drift.py` (movement and drift signals) | `standard_sidemarket_odds_history`, `standard_odds_history` (O), `newformat_odds_history`, `newformat_odds_dense` (O), `player_prop_odds_history` | A + O | Resolver. Class O moves with A/B, so these files resolve to the store as well. |
| Props (`player_model/predict.py`, `clv_tracker.py`): `market_odds` and CLV | `player_prop_odds_history.csv` (+ `_archive/`), `prop_quotes/` | A | Resolver. Props parity (`scripts/props_train_serve_parity.py`). |
| Edge engine and TOP-3 / stake plan inputs (`src/edge_af_quotes.py`, `src/edge_engine_shadow.py`, `src/stake_plan.py`) | `book_quotes/` | A | Resolver. Check the `league_weekly_*` / `stake_plan` outputs for equality. |
| CLV certification (`src/shadow_settlement.py`, `src/clv_types.py`, `update_results.py`) | `book_quotes/`, capture histories, heartbeats | A | Resolver. Settlement and CLV recompute must show no difference. |
| Specials (`src/specials_quotes.py`, `scripts/report_specials_coverage.py`) | `specials_quotes/` | A | Resolver. |
| Pro scout model and promotion (`src/scout/*`) | `scout_odds/`, `scout_specials_quotes/` | A | Pro resolver on `data/season_2026_27/`. |
| Pro paper / shadow / studies (`paper_1x2`, `shadow`, `btts_contradiction`, `price_trader/paths`) | `market_snapshots/` (A), `book_odds_snapshots/` (O: OddsAPI imported from v9) | A + O | Pro resolver. |
| Pro `xg_value`, data quality | `feature_snapshots/` (B) | B | Pro resolver. |
| Class O readers: closing prices and CLV fallback (`update_results.py`, `src/ledger.py` via `odds_history_v9.json` / `standard_odds_history.csv` / `newformat_odds_dense.csv`), BTTS book odds (`src/btts_odds.py`), edge replay (`book_odds_snapshots`), sharp and World Cup trackers, exchange capture, Pro `book_odds_snapshots` (canonical market, price_trader paths, snapshot coverage) | Class O files | O | Resolver. Covered by gate items 3-5 (settlement, CLV, bars/TOP-3 inputs). |
| v11 shadow and movement research | v9 `predictions.csv` (raw URL), own `v11_shadow_*` | B | Token-authenticated read or price-free view (section 7). |

### 4.4 Cross-repo and public-URL readers (break if a source goes private)

* **Dashboard:** `wowza-dashboard/dashboard_data/remote_mirror.py` downloads the v9 files listed in `deploy/dashboard_manifest.json` from `raw.githubusercontent.com/NevixAA/wowza-betting` with no token. 81 data files are mirrored. These class B files are among them: `bets.csv`, `bets_ledger.csv`, `side_bets_ledger.csv`, `ht_ledger.csv`, `player_ledger.csv`, `player_tips.csv`, `predictions.csv`, `stake_plan.csv`, `top_picks_log.csv`, `v91_shadow.csv`, `edge_engine_backtest.csv`, `edge_engine_shadow.csv` and `league_weekly_research.csv`; also `fd_history.parquet` (F).
* **Dashboard pages** read Pro and v11 through raw and API URLs (`dashboard_data/core.py`).
* `src/stake_plan.py`, `telegram_bot/daily_pack.py` and `scripts/auto_onboard.py` read raw Pro files.
* **Pro** reads v9 output over raw HTTP (`src/data/v9_source.py`). `pro_valuable_shadow.yml` and `monthly_bankroll_policy.yml` clone `wowza-betting.git` with no token. `src/bankroll/data.py` and `src/analytics/deep_wowza_snapshot.py` read raw v11.
* **v11** `config.py` and `scripts/v11_tests.py` read raw v9.

## 5. HARD GATE: no damage to the models (parity proof)

The cut-over changes only WHERE files are read from, never their bytes. The gate proves that on the same commit inputs, in two layers. **Any difference blocks the cut-over.**

**Layer 1: bytes.** For every class A, B and O file, compare sha256 of the public file at commit X with sha256 of the private store file at the store commit Y that carries it. The Phase 2 manifest is the first instance of this check: all files matched (section 8). Rerun it at the cut-over commit.

**Layer 2: behaviour.** Run each item twice on the same code commit and the same pinned data commits: once with `PRICE_STORE_DIR` unset (public paths, today's behaviour) and once with `PRICE_STORE_DIR` set to the private checkout. Zero tolerated differences:

| # | What | How it is compared |
|---|---|---|
| 1 | Training frames: Standard and New-Format team models (`retrain_harness.protocol.load_committed_frame` for every target, including O1.5/O3.5/BTTS/HT), the props frame, Pro `feature_snapshots` import | Frame hash = sha256 over sorted column names + dtypes + `pandas.util.hash_pandas_object(df, index=True)`. Must be equal. |
| 2 | Features and predictions on real fixtures | `scripts/team_train_serve_parity.py` and `scripts/props_train_serve_parity.py` (data-truth-100) outputs, compared byte for byte excluding run timestamps. Plus an offline `predict` on a frozen fixture list: `predictions.csv` equal on every column. |
| 3 | Settlement | Re-grade (dry-run `update_results.py`) on frozen results. Ledgers equal on result, stake, P/L and ROI. |
| 4 | CLV | `clv_pct`, close status and certification counts equal (`shadow_settlement`, `clv_collection_health.json`). |
| 5 | Bars / TOP-3 inputs | `league_weekly_candidates.csv`, `league_weekly_research.csv`, `stake_plan.csv`, `weekly_league_report.json`, threshold-proposal inputs: equal. |
| 6 | Pro | `scout` model outputs, `paper_1x2`, `bankroll` evidence: equal. |

* **Pinned inputs.** The live files grow every few minutes. The gate therefore pins v9/Pro/v11 to one commit, and the store to the commit that `dual-write` produced at the same moment. Rows appended after the pin are excluded on both sides.
* **Who runs it.** A `workflow_dispatch` job (`price_store_parity.yml`) with read-only store access. It writes counts and hashes only, never prices. The job is red on any difference.

## 6. History: three options

| | Option 1: move data, keep history | Option 2: move data + purge history | Option 3: make the public repos PRIVATE |
|---|---|---|---|
| Hides current files | Yes (at cut-over) | Yes | Yes, immediately |
| Hides history and other branches | **No**: 10,083 + 2,718 + 749 commits and all 172 branches keep the files | Yes, after `git filter-repo` on every branch and a force-push. GitHub keeps unreachable commits viewable by SHA until it garbage-collects them (GitHub Support can purge). | Yes |
| Hides the existing fork (1 on wowza-betting) | No | No | No (public forks stay public) |
| Destructive | No | **Yes**: rewrites every SHA. All 100+ local worktrees and clones must be re-cloned. Open PRs and branch references break. `registry/` and `models/registry` entries that record commit SHAs point at commits that no longer exist. | No (reversible), but anything already copied stays copied |
| Live pipeline risk | Low (section 7 order) | High during the rewrite: all writers must pause, about 1-3 h estimated | **High**: see the Actions cost below. Every unauthenticated raw/clone reader in section 4.4 breaks until it is given a token. |
| GitHub Actions cost | None (public minutes are free) | None | **About 86,900 billable minutes a month**, against 3,000 included + $15 budget. See below. |
| Streamlit dashboard | Unchanged (reads class C / price-free views) | Unchanged | Community Cloud can deploy from a private repo once private-repo access is granted (estimate: recalled, check). The runtime mirror needs a read token in Streamlit secrets. |

**Option 3 cost.** Inputs, code and results: `docs/af_prices_private_migration/actions_minutes_*.{json,py}`.

| Input | Value | Provenance |
|---|---|---|
| wowza-betting run minutes, 2026-10-17..11-15 | 61,958 | measured from run history (per run ceil(end - start)) |
| wowzaV9-Pro | 28,673 | measured |
| wowza-dashboard | 57 | measured |
| wowza_v11 | 410 | measured |
| Billable / run-estimate ratio (job-level sample) | 0.89 / 1.09 / 0.93 / 0.99 | measured on 80 / 80 / 50 / 80 random runs |
| Included private minutes | 3,000 / month | stated by owner |
| Budget | $15 / month | stated by owner |
| Private usage so far in November | about 2,100 by 11-16, so about 3,940 / month extrapolated | stated by owner; extrapolation derived |
| Linux runner price | $0.006 / min (high case $0.008) | **estimate**: recalled list price after GitHub's 2026-01-01 cut, not checked |

* **Result.** Private billable minutes would be about **86,500 a month** for betting + Pro + dashboard. Adding the 3,940 already used makes about 90,400. At $0.006/min the overage is about **$520 a month**; at $0.008/min it is about **$700 a month**.
* **What $15 buys.** The budget covers only about 2,500 extra minutes: 5,500 at most including the 3,000 included minutes. At the measured rate, betting + Pro use about 2,900 minutes a day, so the cap is reached in roughly the first day of a month. **After that every workflow stops: red runs, missed captures, no settlement.**
* **Usage cut needed** to fit 3,000 + $15: about 98%.
* **Biggest consumers:** Live Scanner (18,400 min), New-Format Odds Capture (14,200), Standard Side-Market Odds Capture (13,000), Pro League Scout closing prices (23,500) and Player Props (7,100).
* **Cross-check.** The raw run-duration total (90,688) and the job-level-corrected total (86,478) differ by 4.6%, which passes.
* **Break-even.** Option 3 fits the budget only at about **$0.00017/min**, about 35x below any GitHub-hosted price. Only self-hosted runners or a much higher budget make Option 3 workable. The conclusion does not depend on which list price is right.

**Owner decision 2026-11-16: keep history for now.** A history purge (Option 2) is deferred and decided after go-live. Nothing in this plan rewrites history.

**Recommendation.**
1. Option 1 at the cut-over (section 7). It is the only option that is safe for the live pipeline around go-live.
2. Delete the stale branches that carry price files once each has been archived, which is cheap and non-destructive for `main`.
3. Then the owner decides between Option 2 (one planned maintenance window; re-clone everything) and accepting that history stays public.
4. Option 3 is not affordable on GitHub-hosted runners at today's workflow volume.

## 7. Cut-over plan (not executed)

### 7.1 Design

* **Live store.** A new orphan branch `af-prices-live` in NevixAA/wowza-exec, seeded with the latest class A, B and O files at their ORIGINAL repo-relative paths under `<repo>/`. The Phase 2 branch `af-prices-archive` stays an immutable snapshot.
* **Writers** (capture workflows, predict, update_results, player_props, Pro scout / live_odds / collect / bet_builder, v11 collect; for class O also `exchange_odds_capture.yml`, `sharp_tracker.yml`, `worldcup.yml` and the OddsAPI parts of `predict.yml`) open the store with a generalised `.github/scripts/price_store.sh open|push <producer>`. This is the same pattern as `side_odds_store.sh`: a write deploy key in a secret, cloned under `$RUNNER_TEMP`, counts-only logs, retry `pull --rebase` on push.
* **Writers append only.** Each writer appends to producer-owned day files, so two workflows never race on one file.
* **Readers** in workflows open the store read-only (shallow, sparse checkout of the paths they need) and get `PRICE_STORE_DIR`. One resolver per repo, for example `src/price_store.py: path(rel)`, returns `$PRICE_STORE_DIR/<repo>/<rel>` when set and the public path otherwise. Every reader in sections 2-4 switches to it.
* **Settlement, ROI and CLV** read prices only from the store.
* **Public ledgers.** Each ledger gets a new price-free view (for example `output/public/bets_ledger_view.csv`: tip, market, side, result, stake, P/L, ROI, CLV %). It is written beside the original, never by rewriting it. The original stays intact in the archive and the store.
* **Dashboard** reads only class C statistics and the price-free views. Its manifest switches to those files, and no token is needed.
* **Tests** use committed synthetic fixtures (`tests/fixtures/price_store/`), never real prices.

### 7.2 Order of operations (each step leaves every run green)

| When | Step | Risk |
|---|---|---|
| Done 11-16 | Phase 2 archive copy, sha256-verified | none |
| Now, before go-live (owner) | Create the deploy keys and secrets (7.3). No workflow reads them yet. | none |
| Before go-live (optional) | Merge the resolver + `price_store.sh`, **default off** (resolver returns public paths when `PRICE_STORE_DIR` is unset). Tests with fixtures. | about none, but it is a code change during the freeze, so the default is to wait until 11-22 |
| **Sat 11-21** | GO-LIVE. No migration work. | - |
| Sun 11-22 | Seed `af-prices-live` from public HEAD and verify with sha256. Turn on **dual-write**: writers keep writing public as today AND append to the store. A daily check compares row counts and appended-row hashes public vs store. | low: the store push runs after the public commit, so it cannot fail the public write |
| Mon 11-23 | Parity gate (section 5) on pinned commits. **Any difference stops here.** | none (read-only) |
| Mon 11-23 | Flip readers to the store (`PRICE_STORE_DIR` set in workflows). Dual-write continues. Watch 24 h: zero red runs, settlement and CLV counts equal. | medium, but rollback is one env line |
| Tue 11-24 | Writers stop writing classes A and O publicly (store only). Publish the price-free views and class C stats. Switch the dashboard manifest to them. Rerun the parity gate. | low |
| After 11-24, owner approval | Remove classes A and O and the price columns of B from public HEAD: `git rm` of A and O files, and the B originals replaced by their views as new files. Only after a third sha256 check that the store holds every byte. Delete stale branches. | low (data verified in two private places) |
| After go-live, owner decision (deferred 2026-11-16) | Option 2 (history purge) or keep the history. | high, planned window |

### 7.3 Keys and secrets the owner must create

Agents are blocked from creating keys or secrets; create them in your own terminal.

1. **`PRICE_STORE_KEY`** (write). Run `ssh-keygen -t ed25519 -N "" -f price_store_key`.
   * Add `price_store_key.pub` as a deploy key on NevixAA/wowza-exec with "Allow write access".
   * Then `gh secret set PRICE_STORE_KEY -R NevixAA/wowza-betting < price_store_key` and the same for `-R NevixAA/wowzaV9-Pro` and `-R NevixAA/wowza_v11` (v11 writes its shadow and movement files).
   * Alternative: reuse the existing `SIDE_ODDS_STORE_KEY` deploy key, which is already write-enabled on wowza-exec. A separate key keeps the corners/cards store and the price store revocable independently.
2. **`PRICE_STORE_RO_KEY`** (read-only deploy key on wowza-exec) for the parity-gate workflow and any reader-only workflow: `gh secret set PRICE_STORE_RO_KEY -R <repo>`.
3. **Optional:** a fine-grained PAT (read-only "Contents" on wowza-exec) as a Streamlit Cloud secret, **only if** the owner wants the dashboard to show prices of our own tips. Without it, the dashboard shows the price-free views.
4. **Option 3 only:** read tokens for every reader in section 4.4, plus Streamlit private-repo access.

Delete the local private key files after `gh secret set`.

## 8. Phase 2 result (copy, no deletes)

* **Where.** NevixAA/wowza-exec (PRIVATE), new orphan branch **`af-prices-archive`**.
  * Final (manifest) commit: **`7a40a395b7f9bdd971b301f5de0c61252031a948`**, tree `1b1e434482df`.
  * Pushed in 8 stages (one per source prefix) so no single push was oversized.
* **Layout.** `<repo>/<branch>/<original path>`, with `/` in a branch name written as `__`. Includes `MANIFEST.csv` (path, source repo / branch / commit / path, class, dataset, provider, bytes, rows, date range, sha256, git blob id), a README and `.gitattributes` (`* -text`, so no line-ending conversion).
* **Scope.** Every class A and class B file (commit `7a40a395`), then every class O file (commit `e48e95cc`), at the commits in the table at the top. Straight copy: no transformation, no filtering, no dedupe. Class F is not copied.

| Source | Class | Files | MB | Rows |
|---|---|---:|---:|---:|
| wowza-betting @ main | A | 230 | 415.5 | 2,784,627 |
| wowza-betting @ main | B | 34 | 106.7 | 194,730 |
| wowza-betting @ data-truth-100 | A | 158 | 254.8 | 1,840,228 |
| wowza-betting @ data-truth-100 | B | 73 | 119.4 | 229,223 |
| wowza-betting @ prep/bars-oddsapi-hist | A | 158 | 254.8 | 1,840,228 |
| wowza-betting @ prep/bars-oddsapi-hist | B | 83 | 120.1 | 232,672 |
| wowzaV9-Pro @ main | A | 2,838 | 327.0 | 2,480,578 |
| wowzaV9-Pro @ main | B | 1,417 | 415.9 | 8,738,021 |
| wowza_v11 @ main | B | 4 | 118.1 | 256,874 |
| **Total** | | **4,995** | **2,132.2** | **18,597,181** |

* **Unique content.** 4,601 distinct files (1,436.3 MB). Files identical on several branches are stored once by git.
* **Verification.**
  1. **At write time.** Every file's git blob id in wowza-exec equals its source blob id, and the byte length matches.
  2. **Independent re-read.** Every manifest row was read back by PATH from the exec commit and from the public repo at the recorded source commit, and both were hashed with sha256. Result: **4,995 / 4,995 match, 0 mismatched** (2,132,212,173 bytes).
  3. **On GitHub.** The remote branch head is `7a40a395...`; GitHub reports tree `1b1e434482df...` with 4,998 blobs (4,995 data files + manifest, README, `.gitattributes`).
* **Left out on purpose.** 6 empty (1-byte) CSVs on data-truth-100 / prep could not be parsed and hold no rows.
* **Logs.** Counts only; no price was printed into any log or document.


### 8.1 Class O added (owner decision 2026-11-16: "Yes, move them too")

* **Commit.** `e48e95ccbfd6833272c2a1037960721d52f9986c` on `af-prices-archive`, child of `7a40a395`, tree `b0e979eb2dcb`.
  * `MANIFEST.csv` was extended append-only: the 4,995 earlier rows are byte-identical and 559 class O rows were added.
  * The README notes class O.
* **What was checked for more OddsAPI data.** Every branch inventoried was searched for other OddsAPI-sourced per-row prices (section 4.1). The OddsAPI-priced own records were already in class B; the `bars_oddsapi_hist` / `oddsapi_history_*` files are class C.

| Source | Dataset | Files | MB | Rows | Date range |
|---|---|---:|---:|---:|---|
| wowza-betting @ main | book_odds_snapshots (+ `_archive/`) | 81 | 51.8 | 431,608 | 2026-08-19 .. 2026-11-16 |
| wowza-betting @ main | exchange_quotes | 2 | 2.9 | 14,291 | 2026-10-20 .. 2026-11-15 |
| wowza-betting @ main | standard_odds_history | 1 | 1.1 | 12,118 | 2026-08-09 .. 2026-11-16 |
| wowza-betting @ main | newformat_odds_dense | 1 | 1.8 | 17,866 | 2026-08-15 .. 2026-11-16 |
| wowza-betting @ main | odds_history_v9.json | 1 | 0.2 | 339 keys | - |
| wowza-betting @ main | sharp_history | 6 | 35.3 | 5,995 keys | - |
| wowza-betting @ main | worldcup_history.json | 1 | 1.7 | 204 keys | - |
| wowza-betting @ data-truth-100 | same 7 datasets | 82 | 86.1 | 428,671 | to 2026-11-02 |
| wowza-betting @ prep/bars-oddsapi-hist | same 7 datasets | 82 | 86.1 | 428,671 | to 2026-11-02 |
| wowzaV9-Pro @ main | book_odds_snapshots (imported from v9) | 302 | 8.3 | 431,608 | 2026-08-19 .. 2026-11-16 |
| **Total class O** | | **559** | **275.2** | **1,771,371** | |

* **Unique content.** 401 distinct files (110.8 MB).
* **Verification.**
  * Every class O blob id equals its source blob id.
  * The full manifest was re-read by path from `e48e95cc` and from the public sources at the recorded commits: **5,554 / 5,554 sha256 match, 0 mismatched** (2,407,363,119 bytes, covering A + B + O).
  * GitHub reports tree `b0e979eb...` with 5,557 blobs.
* **Archive totals now:** 5,554 data files, 20,368,552 rows, 2,407.4 MB.
