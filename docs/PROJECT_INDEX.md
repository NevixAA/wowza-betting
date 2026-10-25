# WOWZA estate — project index (2026-10-25)

One page that answers "what is what" across the three repos. Verified on 2026-10-25 against:

* v9 `958a126e`;
* Pro `9b32e77`;
* v11 local `4813ed7` (stale; last remote seen 2026-10-22).

Architecture detail: `docs/WOWZA_PLATFORM_ARCHITECTURE_2026_10.md`.

---

## 1. Which repo owns what

| Repo | Role | Owns | Never does |
|---|---|---|---|
| **v9** `NevixAA/wowza-betting` | **PRODUCTION** | Tips, Telegram, dashboard, staking plan, serving models, promotion gates (PG-1.0, PG-PROPS-1.0), research lock, system health, the shadow logs that need predict-time first sight (H-MODEL-01, market_edge_v2, weekly panel, v91_shadow) | Read Pro/v11 for any decision, except Pro league ratings, which can only shrink a stake (`src/stake_plan.py:96-119`) |
| **Pro** `NevixAA/wowzaV9-Pro` (folder `v10/`) | **RESEARCH + evidence store** | Forward-test registries (H-EDGE2, H-GATE), studies (props segments, Player V3, price trader, learned tiers), bankroll allocator, league scout/promotion, system contract, Bet Builder (notifies) | Write into v9 (reads it over HTTP, `src/data/v9_source.py`) |
| **v11** `wowza-v11` | **RESEARCH (market-first shadow)** | `v11_shadow_log.csv` (1,875 rows locally, last 2026-10-20), residual, momentum, microstructure studies; workflow `v11_collect.yml` `25 8-23 * * *` | Telegram, stakes, writes into v9 |
| `mixed/` root | Shared data + validation harness | `England_Leagues_4_Seasons_With_Summary.xlsx` (local only), `validation/`, `PlayerProps_Research/` | — |

## 2. What is production

* **Team O/U and side-market tips**, produced by `.github/workflows/predict.yml` → `pipeline.py --mode predict`.
  The chain is in the acceptance report, Q1.
* **Selection engine:** legacy raw edge (`config/track_registry.json` `selection_engine_version.production = "legacy_raw_edge"`).
* **Real-money tracks** (`config/bankroll_policy.json`, policy `balanced-2026-10-13`):

  | Track | State | Base fraction | Odds limits |
  |---|---|---|---|
  | `new_format\|BTTS` | ACTIVE | 0.0125 | max 2.6 |
  | `new_format\|OU25` | ACTIVE_LOW | 0.005 | max 2.5 |
  | `new_format\|OVER15` | ACTIVE_LOW | 0.005 | 1.45–1.9 |

* **Zero-stake tracks:**
  * every `standard|*` track is PAPER;
  * `*|OVER35` is BLOCKED;
  * player props are BLOCKED_FROM_REAL;
  * Cloudbet is PAPER_ONLY.
* **Production state counts** (`track_registry.json`): ACTIVE 12, ACTIVE_LOW 36, BLOCKED 19, PAPER 28, PAPER_LEAGUE 5. These are league×market×side entries.
* **Workflows (v9, 26 files).** Scheduled ones and their crons:

  | Workflow | Cron (UTC) |
  |---|---|
  | `predict` | `9-59/15 8-23 * * 5,6,0`, `9-59/30 0-7 * * 5,6,0`, `9-59/30 8-23 * * 1-4`, `9 0-7 * * 1-4` |
  | `update_results` | `30 7,9,…,23 * * *` |
  | `daily_summary` | `0 1,3,5,7,9 * * *` (5 slots) |
  | `retrain` | `0 3 * * *`, Sundays only after 2026-10-05 via a guard step |
  | `player_props_train` | `0 4 * * 0,1,2` |
  | `backtest_matrix` | `0 3 1 * *` |
  | `evidence_refresh` | `45 6 * * *` |
  | `std_odds_capture` | `5 0,6,12,18` + `23,52 0-3,9-23` |
  | `nf_odds_capture` | `35 1,7,13,19` + `20,50 0-3,9-23` |
  | `player_props` | Fri–Sun every 2 h, Mon–Thu every 6 h |
  | `live_scanner` | `4 */3` |
  | `sharp_tracker` | every 2 h 08–22 |
  | `exchange_odds_capture` | `35 10` |
  | `fantasy_refresh` | `30 6` |
  | `injury_refresh` | `0 4` |
  | `player_history_extend` | `40 3` |
  | `af_history_extend` | `0 6` |
  | `auto_onboard_leagues` | Mon `20 9` |
  | `weekly_summary` | Mon `0 9` |

## 3. What is research

The research modules are listed in the acceptance report, Q2. In short, everything that writes only to:

* `output/edge_engine_*`;
* `output/league_weekly_*`;
* `output/v91_shadow.csv`;
* `output/model_lineage/shadow_*`;
* `config/league_market_regimes.json`, `config/track_registry.json`;
* Pro `output/studies|edge_reassessment|bankroll|price_trader|player_v3`;
* v11 `output/`.

## 4. Which artifact is canonical

| Question | Canonical artifact |
|---|---|
| What was tipped | `output/bets_ledger.csv`, `output/side_bets_ledger.csv` (first tip per fixture) |
| What stake the policy assigns | `output/stake_plan.csv` (current), `output/stake_audit.csv` (first sight) |
| Which model serves | the serving `.pkl` (`config.py:35-36`); identity in `models/registry/current_champions.json`; history in `output/model_lineage/promotion_decisions.csv` |
| Is the system healthy | `output/system_health.json` |
| Is an artifact current | `output/artifact_manifest.json` |
| What thresholds are live | **not** `config.py` alone: `predict.yml` env + `models/best_params_*.json` (approved leagues) + side-market fixed bars. The resolved view is `config/track_registry.json` `gates` per entry. |
| Which tracks research may not touch | `config/research_lock_snapshot.json` (approved 2026-10-25T12:09:26Z) |
| Where research effort goes | `config/league_market_regimes.json` |
| What the system contract says | Pro `registry/WOWZA_SYSTEM_CONTRACT.json` (wins over CLAUDE.md) |

## 5. Deprecated components

See `docs/DEPRECATION_REGISTER.md`. The headline items:

* the legacy raw-edge tier engine is **still the production selection** (status PRODUCTION_REFERENCE);
* old `src/edge_engine.py` is not a decision engine (its de-vig helpers are still used by `clv_capture`);
* H-GATE is LEGACY_RESEARCH;
* synthetic O1.5 history is NOT_CERTIFIABLE;
* `champion_gate` and the promotion tolerances are REMOVED;
* `PROPS_MAX_LOGLOSS_RISE` is REMOVED;
* single-file and season-part `player_history` are REPLACED.

## 6. Experiments running now (2026-10-25)

| Experiment | Where | Registered / started | Current n | Decision rule |
|---|---|---|---|---|
| **H-MODEL-01 standard** (HGB `pro_hgb_standard_20261022T110238Z_e4191ad5` vs champion `standard_20261018T080423Z_64313878`) | v9 `output/model_lineage/shadow_experiments.csv` | 2026-10-22T11:02:38Z; first row 13:45:31Z | 151 rows / 62 settled. Eval at 12:50Z: ΔLL +0.0107 (challenger worse), no CI (<5 dates) | n ≥ 1000 per track (`models/registry/experiments.json`) |
| **H-MODEL-01 newformat** (`pro_hgb_newformat_20261022T110238Z_8e11e6cf` vs `newformat_20261018T080423Z_9eb6b4b4`) | same | same | 173 rows / 72 settled. ΔLL −0.0043 (challenger better), no CI | n ≥ 1000 per track |
| **PG-1.0 parked candidate** `newformat_20261025T082056Z_cfa68e5c` | v9 `shadow_h2h.csv` (2,389 rows since 2026-10-18) | parked SHADOW_MORE 2026-10-25T08:46:06Z | prospective n per PG-1.0 | prospective n ≥ 1000, ΔLL < 0; early fail at 300; expires 180 d |
| **H-EDGE2 family** (NF-OU25-01 n 400, NF-BTTS-01 n 400, NF-O15-01 n 150, STD-OU25-01 control n 400; all ≥ 8 gameweeks; BH q ≤ 0.05) | Pro `registry/edge2_hypotheses_v1.json`, evaluator `src/studies/edge2_forward.py` | registered 2026-10-25, forward start 2026-10-26T00:00Z | 0 counted (99/99/99/84 pre-registration rows not counted) | ELIGIBLE_FOR_REVIEW at n + gameweeks; never auto-CONFIRMED |
| **market_edge_v2 shadow** | v9 `output/edge_engine_shadow.csv` | 2026-10-25T09:25:36Z | 752 rows / 188 fixtures, 0 settled, 100% NO_BET | feeds H-EDGE2 |
| **VALUABLE / v9.1 shadow** | v9 `output/v91_shadow.csv` | 2026-10-04 | 790 rows, 598 settled. VALUABLE 182 rows / 138 settled | descriptive, plus the gate's outcome source |
| **H-GATE (legacy raw-edge candidates)** H-GATE-01..14 | Pro `registry/candidate_forward_tests.json` | 2026-10-20T11:43:55Z | 1 DISCOVERY, 3 ACCRUING, 10 REGISTERED; 6 of 1,200 forward bets (system_health) | `LEGACY_RAW_EDGE_FORWARD_TEST` |
| **Props segments** (ATT_DEF / ATT_MID / CONTROL × scorer / SOT1+ / assist) | Pro `src/studies/props_segments.py` | 2026-10-13, window from 2026-10-14 | 80 tips; largest cell CONTROL scorer 43; all COLLECTING | EDGE needs n ≥ 150 + Bonferroni ROI bound > 0 + median CLV > 0 |
| **Player V3 forward** | Pro `src/player_v3/forward.py` | 2026-10-14, window from 2026-10-15 | 16,760 props in window | n ≥ 150, paper only |
| **Fantasy V2 residual** | v9 `player_model/fantasy_v2.py`, `fantasy_refresh.yml:90` | mode `shadow` since `d0c4ac86` (2026-10-22) | ledger 858 rows GW4–7 (all settled); projection log 10,115 rows | owner decision (D6) |
| **Weekly league panel LWR-1** | v9 `output/league_weekly_research.csv` | true forward start 2026-10-25T13:12:19Z | 60 panel rows (5 leagues, 2026-W43), 0 settled; 752 candidates | NOT_ENOUGH_DATA until 30 priced settled rows / 8 gameweeks |
| **Price Trader v1/v2** | Pro `src/price_trader/` | paper from 2026-10-15 | — | research only |

## 7. Protected tracks

From `config/league_market_regimes.json` (REGIME-1, generated for 2026-10-25, evidence cutoff
2026-10-20), locked by `config/research_lock_snapshot.json`:

| Track | Production | Evidence |
|---|---|---|
| **Argentina Primera Division \| BTTS \| YES \| new_format** | ACTIVE, base 0.0125 | n 81, ROI +36.4%, 90% CI [+15.8%, +56.4%], +29.51 u |

That is the only PROTECTED entry (1 of 100). The lock also fingerprints the global production
facts:

* bankroll policy, deployed env and resolved thresholds;
* league eligibility;
* Cloudbet mode.

As a result, `standard|OU25` PAPER, `*|OVER35` BLOCKED and props BLOCKED_FROM_REAL are locked too.

Note: the 2026-10-08 auto-bet review found Argentina BTTS profit to be league-wide (no selection skill). PROTECTED
means "credible positive result on a staked track". It is not proof that the model adds anything.

## 8. Repair priorities

`REPAIR` entries (6 of 100). `repair_priority_high` is empty: no entry crosses the high-priority bar.

| Entry | Production | Why REPAIR | Priority score |
|---|---|---|---|
| League One \| OU25 \| UNDER \| standard | PAPER | market beats model, dLL CI lower 0.01456 > 0 on 114 fixtures | 0.351 |
| League One \| OU25 \| OVER \| standard | PAPER | same, plus ROI −74% (n 7) | 0.265 |
| USA MLS \| BTTS \| YES \| new_format | PAPER_LEAGUE | market beats model, dLL CI lower 0.00634 > 0 on 133 fixtures | 0.196 |
| League Two \| OU25 \| UNDER \| standard | PAPER | market beats model, dLL CI lower 0.00595 > 0 on 115 fixtures | 0.165 |
| League Two \| OU25 \| OVER \| standard | PAPER | same | 0.159 |
| League One \| BTTS \| YES \| standard | PAPER | market beats model, dLL CI lower 0.00081 > 0 on 101 fixtures | 0.110 |

Every REPAIR entry is already zero-stake. A repair here changes research effort, not money.

The other 93 entries are **LEARNING**:

* 43 have no evidence at all;
* 5 have adequate evidence (n ≥ 30 and ≥ 8 matchdays) but no credible result either way.

Those 5 are:

* Argentina OVER15 (staked, ROI +0.3%);
* MLS OU25 UNDER;
* Bundesliga 2 OU25 UNDER;
* La Liga 2 OU25 OVER;
* Liga MX OU25 UNDER.
