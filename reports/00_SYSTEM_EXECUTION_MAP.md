# 00 — System Execution Map (AU1, 2026-11-01)

Audit team AU1. Read-only. Everything below was read from git objects on `origin/main` or from committed
files at the audited SHA; nothing was inferred from documentation alone.

## Audited commits

| Repo | Remote | SHA audited | Commit time | Role |
|---|---|---|---|---|
| v9 (production) | NevixAA/wowza-betting | `fb0e9697ec8556d4dd3df57f31f8ed5415a5012d` | 2026-11-01 13:16 +02 | Selects, tips, ledgers, settles |
| Pro | NevixAA/wowzaV9-Pro | `e0ccc27b98f57c60ee28eedba81812e9b5c5c25b` | 2026-11-01 09:49 UTC | Research / shadow / Bet Builder; reads v9 over HTTP |
| v11 | NevixAA/wowza_v11 | `36ed1cd71ecbeb837a1704980df72fbd7ab089ae` | 2026-11-01 02:22 UTC | Market-first shadow; reads v9 raw files |
| Dashboard (slim) | NevixAA/wowza-dashboard | `4811bdf37be412f2d19a033e2f90b401ee387311` | 2026-11-01 13:16 +02 | Streamlit viewer; mirrors v9 data |

Note: the local `pro` and `v11` clones were behind origin (`c4b1bb17`, `4813ed72`); all reads used `origin/main`.
The committed `output/predictions.csv` on fb0e9697 was produced by a predict run on code SHA `64c04590`
(2026-11-01 10:59:39Z, an ancestor), so the board predates the last merge by about 17 minutes.

## How production actually runs (v9)

Only one workflow produces tips: **`predict.yml`** → `python pipeline.py --mode predict` → `python telegram_bot/notifier.py`.
No other workflow calls `--mode predict` or `predict_upcoming` (grep of all 29 v9 workflows).

```
predict.yml (cron 15/30-min Fri-Sun, 30/60-min Mon-Thu; concurrency group predict, timeout 25 min)
  env: STANDARD_OU25_SERVING_RECIPE=v9_repaired  LEAGUE_SNIPER_CAP=0.12  MARKSMAN_THRESHOLD=0.08
       VALUABLE_THRESHOLD=0.03  REQUIRE_FORM_DATA=0 (self-expired 2026-09-15)  ODDS_API_KEY (secret)
       NOT set: APIFOOTBALL_KEY, WOWZA_FULL_ENRICH, WOWZA_USE_BACKTEST_ARCHIVE, NF_FEATURE_PARITY (code default 1),
                WOWZA_SAFE_ENRICH_JOINS (default 0 = legacy joins), WOWZA_MODEL_RECIPE (default legacy_v9)
  pipeline.mode_predict
    data_loader.load_all_matches()                      history (+ backtest archive, default ON at predict)
    src/predict.predict_upcoming
      _std_serving.check_config()                       fail-closed recipe gate
      _fetch_odds_api()                                 OddsAPI /odds totals, regions=eu; first book per side
      btts_odds.fetch()                                 per-event BTTS (uk,eu)
      drift.snapshot()/purge_old()                      odds_history_v9.json
      feature_engineering.build_upcoming_features()
      predict_proba(standard legacy) -> overwritten by standard_serving.apply (v9_repaired) for Standard rows
      predict_proba(newformat legacy)                   NF rows (fail-closed if artifact missing, since R2-A)
      side models btts/over15/over35 (legacy)           ONE batch of std+NF rows
      HT models (legacy)                                std rows
      betting.evaluate_value (x2, drift info-only since 10-26)
    provenance.stamp -> output/predictions.csv
    shadow_experiments / _log_shadow / _log_h2h / drift.log_shadow       research only
    betting.generate_bets -> bets ; pipeline._generate_side_bets -> side_bets
    direction_guard.apply -> ledgers: append_tips (bets_ledger.csv), append_side_market_tips (side_bets_ledger.csv)
    edge_engine_shadow.run (market_edge_v2 SHADOW) ; stake_plan.write (stake_plan.csv / stake_audit.csv)
    append_ht_tips / append_ht_observations
  telegram_bot/notifier.py  (continue-on-error)
  research steps: src.research.league_weekly, src.research.quality_tiers --shadow (tolerant)
  commit: ~45 files staged one by one, pull --rebase -X theirs, 5 push retries
```

### Models actually serving (artifact SHA-256 verified against `models/registry/current_champions.json`: all 8 MATCH)

| Family | Artifact | model_version on the board | Last artifact commit | Recipe | Persisted imputer |
|---|---|---|---|---|---|
| Standard O/U 2.5 | `model_v9_standard_repaired.pkl` | standard_repaired_20261028T123040Z_1f5805e6 | 2026-10-28 12:37Z (owner install) | v9_repaired | yes |
| (rollback only) | `model_v9_standard.pkl` | standard_20261018T080423Z_64313878 | 2026-10-18 | legacy | no |
| New Format O/U 2.5 | `model_v9_newformat.pkl` | newformat_20261018T080423Z_9eb6b4b4 | 2026-10-25 rollback to 10-18 blob | legacy | no |
| BTTS | `model_v9_btts.pkl` | btts_20261005T101905Z_238e3419 | 2026-10-05 | legacy | no |
| Over 1.5 | `model_v9_over15.pkl` | over15_20261018T080423Z_da4743a2 | 2026-10-18 | legacy | no |
| Over 3.5 | `model_v9_over35.pkl` | over35_20261005T101905Z_f546ce72 | 2026-10-05 | legacy | no |
| HT O0.5 | `model_ht_over05.pkl` | ht_over05_20261018T080423Z_359b491a | 2026-10-18 | legacy | no |
| HT O1.5 | `model_ht_over15.pkl` | ht_over15_20261004T093828Z_6f370f63 | 2026-10-04 | legacy | no |

The committed board (158 fixtures) shows `serving_recipe` legacy_v9 = 88 (all New Format) and v9_repaired = 70
(all Standard); `output/standard_serving_health.json` PASS, 70/70 served by repaired, 0 fail-closed.

### Thresholds actually in force

Not from `config.py` defaults. They are the `predict.yml` env plus approved `models/best_params_standard.json`
entries plus the fixed side-market bars and approved `models/best_params_side_markets.json` entries. The effective
floors are computed by executing `betting._base_tier` under the production env (see report 07 and
`reports/au1_data/effective_thresholds.csv`).

### Writers of serving models

| Workflow | Trigger | Writes models? | Gate |
|---|---|---|---|
| `retrain.yml` | cron `0 3 * * *` + gate step | yes (`pipeline.py --mode train`) | PG-1.0 inside `_train_one` |
| `backtest.yml` | manual dispatch only | yes — stages `model_v9_standard.pkl`, `model_v9_newformat.pkl`, HT pkls after its own `--mode train` | same `_train_one` path |
| `preseason_retrain.yml` | `0 2 1 8 *` | yes | same |
| `train_standard_repaired.yml` | manual | installs repaired payload only from an approved artifact | owner approval |
| `player_props_train.yml` | Sun/Mon/Tue 04:00 | props pkls | PG-PROPS-1.0 |

`backtest.yml` is a second writer of the O/U and HT artifacts (manual). It is gated by the same promotion code,
but it is a path a reader of "only retrain writes models" would miss.

### Settlement and ledgers

`update_results.yml` (`30 7,9,...,23 * * *`) → `update_results.py --days 3` fills result/pnl/closing/CLV in
`bets_ledger.csv`, `side_bets_ledger.csv`, `ht_ledger.csv`; also settles shadow experiments.
`daily_summary.yml` runs `update_results.py --days 2` before the digest.

Ledger facts that matter for every evaluation (measured on fb0e9697):

* `output/bets_ledger.csv` holds **5,921 rows = 4,115 `source=backtest` + 1,806 `source=live`**. Any reader that does
  not filter `source` mixes walk-forward backtest rows into live performance.
* `model_version` is blank on **5,705 of 5,921** rows (column introduced ~2026-10-18). Model attribution for
  earlier rows is only reconstructable from git (done in report 07, `reports/au1_data/regime_ledger_rows.csv`).
* 122 live O/U rows carry `side=AVOID` (all `VOID`, 2026-04 to 2026-08).

### Workflow inventory

All 60 workflows (v9 29, Pro 30, v11 1; the slim dashboard repo has none) with triggers, python entry points,
env keys (secret values never read), timeouts and referenced paths: `reports/au1_data/workflow_map.csv`
(generated by `scripts/audit/au1_workflow_map.py`). Cron strings are requests, not guarantees (CLAUDE.md:
1,367-run measurement, high-frequency delivery ~13%).

Production-relevant v9 workflows: `predict`, `update_results`, `retrain`, `std_odds_capture`, `nf_odds_capture`,
`daily_summary`, `player_props`, `player_props_train`, `backtest_matrix` (monthly; since 2026-10-26 writes threshold
PROPOSALS only, `src.threshold_apply` with owner approval is the only writer of `best_params_*.json`).
Research/health only: `integrity_health`, `evidence_refresh`, `props_r2_challengers`, `exchange_odds_capture`,
`prop_odds_snapshot`, `af_*`, `warm_nf_shot_cache`, `auto_onboard_leagues`, `sharp_*`, `live_scanner`, `worldcup`.

### Pro, v11, dashboard

* **Pro** (30 workflows): collection (`pro_collect`, `pro_live_odds`, `pro_team_news`, `pro_team_stats`,
  `pro_injury_capture`, `pro_weather_fetch`, `pro_scout*`), shadow/forward tests (`pro_standard_repair_shadow`,
  `pro_valuable_shadow`, `pro_player_v3_forward`, `pro_price_trader`, `pro_r2_repaired_challengers`), research
  (`pro_research`, `pro_studies`, `pro_prediction_lab`, `deep_analytics`), and two that emit outward-facing
  output: `pro_bet_builder` (notifies, per CLAUDE.md) and `monthly_bankroll_policy` (`python -m src.bankroll.run`
  in Pro). v9's `config/bankroll_policy.json` changed in 5 direct v9 commits (10-13 .. 10-26); AU1 found no
  automated cross-repo write and did not trace whether the Pro output is hand-copied. Pro never writes into v9.
* **v11**: one workflow `v11_collect.yml` (`25 8-23 * * *`), reads v9 raw GitHub files, writes only its own output.
* **Dashboard**: `dashboard_data/remote_mirror.py` mirrors v9 `main` raw files every 10 min according to
  `deploy/dashboard_manifest.json` (v9 commit `2ce709f5`, 81 data files incl. `bets_ledger.csv`,
  `side_bets_ledger.csv`, `predictions.csv`, `stake_plan.csv`, `edge_engine_shadow.csv`, `v91_shadow.csv`,
  `backtest_by_league_standard.csv`). It is a viewer; its numbers inherit every ledger caveat above.

## Defaults that differ between environments (detail in report 01)

| Knob | train (`retrain.yml`) | serve (`predict.yml`) | effect |
|---|---|---|---|
| `WOWZA_FULL_ENRICH` | `1` (xG/inside-box for every AF league) | unset (NF leagues only) | xG features: populated in training, partly NaN at serve for Standard |
| `WOWZA_USE_BACKTEST_ARCHIVE` | `0` (archive held out) | unset → `1` (archive of 44,207 rows to 2026-04-23 loaded) | different history frame feeds serving features than training |
| `APIFOOTBALL_KEY` | set | not set | serve-time API-Football enrichments are no-ops |
| `best_params.json` (v7) | `DATA_DIR/best_params.json` = parent of repo | same path, absent on CI runners | both_losing guard and per-league min odds inert in CI, active on a local run |

## Reproduce

```
git -C C:\Users\nevo.APOLLO\dev\wowza\v9 worktree add ...\v9_audit_au1 -b audit/au1 origin/main
python scripts/audit/au1_workflow_map.py        # -> reports/au1_data/workflow_map.csv
python scripts/audit/au1_config_timeline.py     # -> config_timeline.csv, artifact_epochs.csv
```
