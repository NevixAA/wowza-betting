# WOWZA platform architecture (2026-10-25)

Owner brief "WOWZA platform architecture" 2026-10-25, sections 43–46. **Documentation only — this
file changes no code, config, workflow or data.**

Evidence basis: v9 `main` at `958a126e` (2026-10-25), Pro `main` at `9b32e77`, v11 local checkout
at `4813ed7` (stale, see §10). Every claim below cites a file:line, an artifact with its recorded
`generated_at`, or a commit. Where the design and the code disagree, the code is described and the
gap is listed in §10. Companion documents: `docs/WOWZA_ROADMAP_2026_10.md`, `docs/PROJECT_INDEX.md`,
`docs/DEPRECATION_REGISTER.md`, `docs/WOWZA_ARCHITECTURE_ACCEPTANCE_2026_10.md`.

---

## 1. System diagram

```
                    ┌──────────────────────── EXTERNAL SOURCES ────────────────────────┐
                    │ OddsAPI (board prices)  API-Football (results, book_quotes, props)│
                    │ football-data.co.uk (history)  exchanges (lay capture)  FPL API   │
                    └───────────────┬──────────────────────────────────┬───────────────┘
                                    │                                  │
 L1 DATA FOUNDATION                 ▼                                  ▼
   capture jobs ─► output/book_odds_snapshots.csv   output/book_quotes/<date>.csv
   (std/nf_odds_capture, predict)   output/newformat_odds_dense.csv  output/exchange_quotes/
   history     ─► data_loader (football-data) · af_history.parquet · player_history.parquet/ (51 month parts)
   contracts   ─► src/contracts (CONTRACTS-1, observe-only)   freshness ─► output/artifact_manifest.json
                                    │
 L2 PREDICTION MODELS               ▼
   models/model_v9_{standard,newformat}.pkl + side slots (btts/over15/over35/ht_*)   ← serving files
   models/registry/current_champions.json (identity/lineage pointer, NOT read by predict)
   challengers: H-MODEL-01 HGB (src/shadow_experiments.py)   PG-1.0 parked candidates (shadow_h2h.csv)
   player props: player_model/*  (PAPER only, PG-PROPS-1.0)
                                    │ p_over25, p_btts, p_over15, p_over35 per fixture
 L3 BETTING INTELLIGENCE            ▼
   PRODUCTION  legacy raw-edge engine: betting.evaluate_value/_base_tier (O/U 2.5)
               pipeline._generate_side_bets (BTTS/O1.5/O3.5) → direction_guard.apply
               → stake_plan.write → bankroll_policy.size_day   → bets.csv / side_bets.csv / stake_plan.csv
   SHADOW      market_edge_v2: edge_engine_shadow.run → market_edge.evaluate (+ edge_af_quotes)
               → output/edge_engine_shadow.csv (first sight, never read by a served path)
   RESEARCH    regimes (config/league_market_regimes.json) · weekly panel (src/research/league_weekly*)
               · v91_shadow (VALUABLE etc.) · top picks · Pro forward tests (H-EDGE2, H-GATE, props segments)
                                    │
 L4 DEPLOYMENT & GOVERNANCE         ▼
   GitHub Actions (26 v9 workflows) · Telegram (notifier, daily_pack by tip_day) · Streamlit dashboard
   PG-1.0 / PG-PROPS-1.0 promotion gates · research_lock (non-regression fingerprint)
   system_health.json · artifact_manifest.json · track_registry.json (read-model) · Pro contract
```

Two other repos sit beside v9 and only **read** its committed output over HTTP:
Pro (`NevixAA/wowzaV9-Pro`, `src/data/v9_source.py`) and v11 (`wowza-v11`, `scripts/v11_shadow.py`).
One flow runs the other way: v9's stake plan reads Pro's `output/studies/league_ratings.json` at
predict time (`src/stake_plan.py:96-119`). It can only scale a stake down
(`league_rating_sizing.min_scale 0.2`, `config/bankroll_policy.json`).

---

## 2. The four layers and the modules that implement them

The layers are **logical**. The code is not physically arranged by layer (§10, G3). Paths are v9 unless
prefixed `Pro:`.

### L1 DATA FOUNDATION — capture, history, identity, contracts, freshness

| Concern | Module (verified) | Notes |
|---|---|---|
| Board odds (OddsAPI totals, pre-match only) | `src/predict.py::_fetch_odds_api` (46) | `if dt <= now: continue` (96); per-book rows → `book_odds_snapshots.csv` (191–217) |
| BTTS odds | `src/btts_odds.py` | called per event from `_fetch_odds_api` (181–185) |
| Batch odds capture, NEAR loop, heartbeat | `src/clv_capture.py`, `src/capture_schedule.py`, `src/odds_heartbeat.py`, `src/clv_collection_health.py` | commit `1187386c` (2026-10-22): last-seen heartbeat, adaptive NEAR loop, certified-close guard |
| API-Football per-book quotes | `src/book_quotes.py` → `output/book_quotes/<date>.csv` | 22 day parts, 2026-10-04 … 2026-10-25 |
| Rolling odds logs | `src/rolling_store.py` | commit `964ffa5a` (2026-10-22): stops re-storing full history per commit |
| Drift (price change-log) | `src/drift.py` → `odds_history_v9.json` | change-log, not a panel (CLAUDE.md trap 2) |
| Team history | `src/data_loader.py`, `src/api_football.py`, `af_history.parquet` | |
| Player history | `player_model/pipeline.py::_save_history` → `player_history.parquet/` | 51 `month_YYYY-MM.parquet` parts since `a418b9a3` (2026-10-25) |
| Club-name identity | `src/team_names.py::resolve` | league-scoped, refuses ambiguity |
| Gameweek id | `src/gameweek.py` (rule GW-1) | research/display only (docstring line 6) |
| Tip day | `src/tip_day.py` | Asia/Jerusalem, rollover 10:00 (28–39); grouping only |
| Data contracts | `src/contracts/` (`schemas.py`, `schema.py`, `shadow.py`, `__main__.py`) | CONTRACTS-1, 14 schemas, observe-only (§5) |
| First-sight freeze | `src/research/first_sight.py::FirstSightStore` | append-only, earliest row wins, refuses post-kickoff rows |
| Freshness | `src/artifact_freshness.py` → `output/artifact_manifest.json` | recorded timestamps only, unknown age = stale |

### L2 PREDICTION MODELS — features, fitting, serving, challengers

| Concern | Module | Notes |
|---|---|---|
| Features | `src/feature_engineering.py`, `src/ht_features.py` | |
| Team models (LogReg + GB, Platt) | `src/model.py` | `load_models` pickle-loads the serving path (`model.py:296-299`) |
| Serving files | `config.MODEL_FILE_STANDARD` / `MODEL_FILE_NEWFORMAT` (`config.py:35-36`) | predict reads these, **not** the pointer |
| Champion pointer + lineage | `src/model_registry.py`, `models/registry/current_champions.json` | roles `CHAMPION/CHALLENGER/REJECTED/SHADOW/ARCHIVED` (`model_registry.py:80`) |
| Promotion gate (team) | `src/promotion_gate.py`, `config/promotion_gate.json` (PG-1.0) | only writer of the serving file + pointer (`_install_champion`, 563) |
| Challenger first sight | `src/shadow_experiments.py` (H-MODEL-01) | `score` (170), `append_first_sight` (267), `settle` (299), `evaluate` (368), `guard` (415) |
| Gate prospective log | `output/model_lineage/shadow_h2h.csv` | PG-1.0 period-2 rows (`promotion_gate.json:41`) |
| Player props | `player_model/` (`model.py`, `pipeline.py`, `props_gate.py` PG-PROPS-1.0) | PAPER only (invariant 2) |
| Fantasy V2 | `player_model/fantasy_v2.py` | residual mode `shadow` (`fantasy_refresh.yml:90`) |

### L3 BETTING INTELLIGENCE — pricing, selection, staking, research about selection

| Concern | Module | Status |
|---|---|---|
| O/U 2.5 edge + tiers | `src/betting.py::evaluate_value` (219), `_base_tier` (140), `_apply_drift_adjustment` (198), `generate_bets` (441) | **PRODUCTION (legacy raw edge)** |
| Side markets | `pipeline.py::_generate_side_bets` (78) | **PRODUCTION (legacy raw edge)** |
| One goal direction per fixture | `src/direction_guard.py::apply` (134), `record` (229) | PRODUCTION since `47716cb9` |
| Stake sizing | `src/stake_plan.py::write` (226) → `src/bankroll_policy.py::size_day` (90) | PRODUCTION (stake plan is an output; see §8) |
| Threshold mirror | `src/gate_resolver.py` | read-only mirror, not on the predict path (`gate_resolver.py:15-17`) |
| Market-first engine | `src/market_edge.py`, `src/edge_engine_shadow.py`, `src/edge_af_quotes.py`, `config/market_edge_v2.json` | **SHADOW** (`065209bc`, `0ae781c7`) |
| Old candidate engine | `src/edge_engine.py` | **DEPRECATED** as a decision engine; its de-vig helpers are still imported by `src/clv_capture.py:24` |
| Regime registry | `scripts/build_league_regimes.py` → `config/league_market_regimes.json` | RESEARCH (file `about`: "Production never reads this file") |
| Weekly league panel | `src/research/league_weekly.py`, `league_weekly_eval.py` | RESEARCH |
| Retrospective replay | `src/research/edge_replay.py`, `scripts/edge_engine_replay.py` | RESEARCH, labelled RETROSPECTIVE |
| v9.1 shadow (VALUABLE etc.) | `src/shadow_compare.py` → `output/v91_shadow.csv`; settlement `src/shadow_settlement.py` | RESEARCH |
| Top picks | `src/top_picks.py` → `output/top_picks_log.csv` | display product, frozen daily |
| Allocation registry | `src/allocation.py` via `scripts/allocation_registry.py` | RESEARCH (nothing in predict reads it) |
| Pro forward tests | Pro `src/studies/edge2_forward.py`, `candidate_forward.py`, `props_segments.py`, `src/player_v3/forward.py`, `src/price_trader/*` | RESEARCH, paper only |

### L4 DEPLOYMENT & GOVERNANCE — scheduling, delivery, gates, locks, health

| Concern | Module / workflow |
|---|---|
| Scheduling | 26 workflows in `.github/workflows/` (crons in `docs/PROJECT_INDEX.md` §2) |
| Telegram | `telegram_bot/notifier.py` (instant, gated by `instant_tip_allowed` 340–367), `daily_pack.send_daily_pack` (morning album via `daily_summary.yml`, 5 slots) |
| Dashboard | `app.py` + `pages/` (24 pages incl. 19 System Health, 20 Model Lab, 21 Market Intelligence, 22 Production, 18 League Research) — a viewer only |
| Promotion | PG-1.0 (`src/promotion_gate.py`), PG-PROPS-1.0 (`player_model/props_gate.py`) |
| Non-regression lock | `src/research_lock.py` + `config/research_lock_snapshot.json` |
| Health | `src/system_health.py` → `output/system_health.json` (daily, `evidence_refresh.yml`) |
| Track read-model | `scripts/build_track_registry.py` → `config/track_registry.json` |
| Contract (cross-repo) | Pro `registry/WOWZA_SYSTEM_CONTRACT.json` (training_contract 697–735) |

---

## 3. Layer boundaries — what is enforced and where

| Boundary rule | Enforced by | Strength |
|---|---|---|
| Research never changes a threshold, stake, eligibility, state or selection rule | `src/research_lock.py` fingerprints production-facing behaviour; `--check` exits 1 on a diff | **Detector, not a blocker.** Runs inside `league_weekly.run` (`league_weekly.py:796-798`, in `predict.yml`) and `system_health` (`evidence_refresh.yml`). It turns health RED; it does not stop a commit. |
| The shadow engine cannot change a served tip | `pipeline.py:854-859` calls `edge_engine_shadow.run` after `generate_bets`/`_generate_side_bets`/`direction_guard`, inside try/except; it writes only `edge_engine_shadow.csv` + `edge_engine_health.json` (`edge_engine_shadow.py:347,356`); `stake_plan.attach_shadow` adds trailing columns after `size_day` (`stake_plan.py:151-183`) | Structural (call order) + byte-identical non-regression runs in `065209bc`, `0ae781c7` |
| Only the gate changes the serving model | `promotion_gate._install_champion` (563) is the only writer of serving file + pointer; `submit_candidate` "NEVER returns PROMOTE" (816); PROMOTE only via `review_parked` (697) | Structural, for `pipeline.py --mode train`. **Not** for `backtest.yml`, which commits `.pkl` files without the registry (§10 G10) |
| Logged opinions are never revised | `FirstSightStore.append` (earliest row wins, post-kickoff refused, optional digest fail-closed) | Structural; daily digest recorded in `system_health.json` |
| Pro and v11 never write into v9 | by convention + read-only HTTP readers | Convention (Pro `src/data/v9_source.py`) |
| Standard and new-format never mix | separate `.pkl`, separate tracks everywhere | Design; runtime fallback to the standard model for unknown leagues (`src/predict.py:333-358`) |
| Dashboard renders, never computes | memory rule; `pages/19_🩺_System_Health.py` is "a viewer of system_health.json" (`155a44c8`) | Convention |

---

## 4. Canonical artifacts

"Freshness rule" is the `max_age_hours` recorded in `output/artifact_manifest.json`
(generated_at 2026-10-25T13:28:45Z, 15 artifacts, 0 stale) where the artifact is in the manifest.
Otherwise it is the rule stated in the writing module. The manifest uses `artifact` for the path and
`owner` for the writer.

| Artifact | Canonical for | Writer | Cadence | Freshness rule |
|---|---|---|---|---|
| `output/predictions.csv` | every fixture's model probabilities + board prices | `pipeline.mode_predict` (787–791) | each predict run (15/30/60 min) | provenance-stamped; not in manifest |
| `bets.csv`, `output/bets_ledger.csv` | served O/U 2.5 tips (ledger = first tip per fixture, 5,825 rows) | `pipeline.py:884-899` | each predict run | ledger append-only |
| `side_bets.csv`, `output/side_bets_ledger.csv` | served BTTS/O1.5/O3.5 tips (613 rows) | `pipeline.py:837-847` | each predict run | append-only |
| `output/stake_plan.csv` | today's stake per selection | `stake_plan.write` | each predict run | manifest max 12 h (owner `predict.yml`) |
| `output/stake_audit.csv` | first stake ever computed per (policy, fixture, track), 472 rows | `stake_plan.write` (249–264) | each predict run | append-only |
| `output/direction_conflicts.csv` | suppressed opposite-direction tips (2 rows) | `direction_guard.record` | each predict run | first sight |
| `models/registry/current_champions.json` | which version is champion per slot | `promotion_gate._install_champion` only | on gate decision | — |
| `output/model_lineage/promotion_decisions.csv` | every PG-1.0 decision (2 rows, 2026-10-25 08:46Z) | `promotion_gate` | per train run | append-only |
| `output/model_lineage/props_promotion_decisions.csv` | PG-PROPS-1.0 decisions (16 rows) | `props_gate` | weekly props train | append-only |
| `output/model_lineage/shadow_experiments.csv` | H-MODEL-01 challenger first sight (324 rows) | `shadow_experiments.append_first_sight` (`pipeline.py:803-804`) | each predict run | first sight; settled by `update_results.yml:114-115` |
| `output/model_lineage/shadow_h2h.csv` | PG-1.0 prospective rows (2,389 rows since 2026-10-18) | `pipeline._log_h2h` (810) | each predict run | first sight |
| `output/v91_shadow.csv` | outcome source for the gate + v9.1 shadow (790 rows) | `shadow_compare.append_shadow`; settled by `update_results.py` / `shadow_settlement.py` | predict / every 2 h | `shadow_settlement_health.json` max 30 h |
| `output/edge_engine_shadow.csv` | market_edge_v2 first-sight decisions (752 rows) | `edge_engine_shadow.run` | each predict run | first sight; `edge_engine_health.json` |
| `output/clv_collection_health.json` | certified-close coverage | odds capture | each capture | manifest max 30 h |
| `config/league_market_regimes.json` | research regime per league×market×side (100 entries) | `scripts/build_league_regimes.py` | on demand (generated_for 2026-10-25, evidence cutoff 2026-10-20) | "do not hand-edit" |
| `config/track_registry.json` | one read-model of each track's production + research state (100 entries) | `scripts/build_track_registry.py` (`evidence_refresh.yml:77`) | daily | `system_health` drift check (0 drift) |
| `config/research_lock_snapshot.json` | approved production fingerprint | `research_lock --approve` only | on owner approval (2026-10-25T12:09:26Z) | — |
| `output/system_health.json` | one health verdict for the estate | `src/system_health.py` (`evidence_refresh.yml`) | daily 06:45 UTC + on demand | its own `generated_at` (2026-10-25T13:28:49Z) |
| `output/artifact_manifest.json` | freshness of decision-facing artifacts | `src/artifact_freshness.py` | daily (`evidence_refresh.yml`) | itself |
| `output/league_weekly_research.csv` (+ `_candidates`, `_summary.json`) | weekly per-league research panel | `src/research/league_weekly.py` (`predict.yml:241`), settled in `update_results` | each predict run | first sight |
| `player_history.parquet/` | player match log (870,856 rows × 212 cols at migration) | `player_model/pipeline.py::_save_history` | daily extend + props runs | size guard 70/100 MB per part |
| Pro `output/edge_reassessment/edge2_forward_progress.json` | H-EDGE2 forward status | Pro `src/studies/edge2_forward.py` | daily 09:50 (`pro_valuable_shadow.yml`) | generated_at 2026-10-25T13:18:22Z |
| Pro `registry/WOWZA_SYSTEM_CONTRACT.json` | authoritative system contract | owner-approved edits | on change (`dcc51b4`) | validator `python -m src.validation.validate_system_contract` |

---

## 5. Data contracts (CONTRACTS-1, `f08b699e`)

`src/contracts/schemas.py` defines 14 schemas (`CONTRACTS_VERSION = "CONTRACTS-1"`, line 37):
`PREDICTION_ROW` (108), `MARKET_SNAPSHOT_ROW` (133), `SELECTION_DECISION` (187),
`RESEARCH_OBSERVATION` (213), `SHADOW_EXPERIMENT_ROW` (254), `TOP_PICKS_ROW` (285),
`PROPS_SHOWN_ROW` (302), `LEAGUE_WEEKLY_RESEARCH_ROW` (320), `STAKE_PLAN_ROW` (396),
`STAKE_AUDIT_ROW` (412), `SETTLEMENT_ROW` (420), `CLV_RECORD` (438),
`MODEL_EXPERIMENT_REGISTRATION` (458), plus `V91_SHADOW_ROW`. `COMMITTED_FILES` (491) maps them to
committed files.

**Mode: observe only.** `src/contracts/shadow.py::observe` validates a frame after its writer has
written it. It records the result in the runner-local, gitignored `output/contract_violations.json`.
It never filters or blocks a write. The committed signal is the `Data contracts` check in
`output/system_health.json`.

`python -m src.contracts`, run 2026-10-25 on this worktree (read-only):

```
  0 violation(s)  PREDICTION_ROW                 output/predictions.csv (179 rows)
  4 violation(s)  MARKET_SNAPSHOT_ROW            output/book_odds_snapshots.csv (37467 rows)
      - missing_column minutes_to_kickoff / overround / opposite_odds / source: 37467 row(s) each
  0 violation(s)  SELECTION_DECISION             output/edge_engine_shadow.csv (752 rows)
  0 violation(s)  V91_SHADOW_ROW                 output/v91_shadow.csv (790 rows)
  0 violation(s)  SHADOW_EXPERIMENT_ROW          output/model_lineage/shadow_experiments.csv (324 rows)
  0 violation(s)  TOP_PICKS_ROW                  output/top_picks_log.csv (100 rows)
  0 violation(s)  PROPS_SHOWN_ROW                output/props_shown_log.csv (0 rows, ABSENT)
  0 violation(s)  LEAGUE_WEEKLY_RESEARCH_ROW     output/league_weekly_research.csv (60 rows)
  0 violation(s)  STAKE_PLAN_ROW                 output/stake_plan.csv (93 rows)
  3 violation(s)  STAKE_AUDIT_ROW                output/stake_audit.csv (472 rows)
      - allowed side: 110 row(s) e.g. ['YES/OVER']
      - null odds: 110 row(s)
      - rule one_position_per_fixture: 2 row(s) (row 354, row 375)
  0 violation(s)  SETTLEMENT_ROW                 output/bets_ledger.csv (5825 rows)
  0 violation(s)  SETTLEMENT_ROW                 output/side_bets_ledger.csv (613 rows)
  0 violation(s)  CLV_RECORD                     output/clv_records.csv (6351 rows)
  0 violation(s)  MODEL_EXPERIMENT_REGISTRATION  models/registry/experiments.json (2 rows)
exit=0
```

What the violations are:

* **`book_odds_snapshots.csv`, 4 missing columns.** The writer predates the contract. The contract
  describes the target shape, not the current file.
* **`stake_audit.csv`, 110 rows with side `YES/OVER` and blank odds.** All 110 are `unknown|BTTS/OVER15/OVER35`
  with reason `NO_REAL_STAKE_UNSUPPORTED`, generated 2026-10-14 13:39 to 2026-10-22 10:07. These are
  the side-market rows from before the P0-1 fix (`b4ce9e22`). They are historical, zero-stake and
  append-only.
* **`stake_audit.csv`, rows 354 and 375.** This is a **real double position**. San Lorenzo v Newell's
  Old Boys (2026-10-24) was staked twice:
  * `new_format|OVER15` at 0.34 units, generated 2026-10-22 11:00;
  * `new_format|BTTS` at 0.79 units, generated 2026-10-22 13:45.

  `one_position_per_fixture` is applied only inside one `size_day` call
  (`bankroll_policy.py:178-186`). The cross-run `prior_directions` input (`stake_plan.py:196-223`)
  carries direction only, not "already staked". See §10 G6.

---

## 6. Ownership

| Decision | Owner | Mechanism (verified) |
|---|---|---|
| **Policy** — which tracks get money, base fractions, caps, odds blocks | Owner (Nevo) | `config/bankroll_policy.json` (`balanced-2026-10-13`). No workflow writes it; it changes by PR only. Fingerprinted by `research_lock`. Recent owner changes: `db08a0af` (equal OU25 tier stakes, VALUABLE staked on OU25), `7c3f5ab5` (NF OU25 price cap 3.00). |
| **Model** — which fitted artifact serves | PG-1.0 / PG-PROPS-1.0 automatically, owner override by exact version | `promotion_gate.py`; owner override env `PROMOTION_GATE_OWNER_OVERRIDE` (727–741), set in no workflow. Owner rollback example: `92d7f0ba`. |
| **Selection** — the rule that turns a probability into a tip | Owner, plus one automatic exception | Code (`betting.py`, `pipeline._generate_side_bets`) plus the deployed env in `predict.yml:188-192`. **Exception:** the monthly `backtest_matrix.yml` rewrites `models/best_params_*.json`, which changes thresholds for leagues the backtest itself marks `approved` (`betting.py:154-157`; `backtest.py:666`). There is no gate and no owner step (§10 G5). |
| **Deployment** — when and how it runs and is delivered | Owner | Workflows, Telegram gating (`TIPS_INSTANT_ALERTS`), dashboard. |
| **Research** — what is studied and what is registered | Agents propose, owner approves registrations | Frozen registries: v9 `models/registry/experiments.json`, Pro `registry/edge2_hypotheses_v1.json`, `registry/candidate_forward_tests.json` |

---

## 7. Research lifecycle — as actually implemented

The brief describes one lifecycle: DISCOVERY → REGISTERED → ACCRUING → ELIGIBLE_FOR_REVIEW →
OWNER_REVIEW → PROMOTED / REJECTED / CONTINUE. **No module implements that full chain.** Each
subsystem has its own vocabulary:

| Subsystem | States actually emitted | Where |
|---|---|---|
| Pro H-EDGE2 (market_edge_v2 forward family) | `REGISTERED`, `ACCRUING`, `ELIGIBLE_FOR_REVIEW`, `BLOCKED_CONFIG_CHANGED`. Never CONFIRMED (`confirmed_assignable_here: false`). | Pro `src/studies/edge2_forward.py:12,254` |
| Pro H-GATE (legacy raw-edge candidates) | `DISCOVERY`, `REGISTERED`, `ACCRUING`, `INSUFFICIENT`, `REJECTED`, `FORWARD_SUPPORTED` | Pro `src/studies/candidate_forward.py:105`, `edge_reassessment.py:21` |
| Pro props segments | `COLLECTING` → `EDGE` (n≥150, ROI bound > 0, median CLV > 0) | Pro `src/studies/props_segments.py:1-52` |
| Weekly league panel | `NOT_ENOUGH_DATA` below 30 priced settled rows or 8 gameweeks | `league_weekly_research_summary.json` |
| Regime registry | `PROTECTED` / `REPAIR` / `LEARNING` (rule REGIME-1; no rule fires on fewer than 8 matchdays) | `config/league_market_regimes.json` `rules` |
| Team models | gate verdicts `PROMOTE`, `KEEP_CHAMPION`, `SHADOW_MORE`, `PARK`, `ROLLBACK`, `BOOTSTRAP`; roles `CHAMPION/CHALLENGER/REJECTED/SHADOW/ARCHIVED` | `promotion_gate.py`, `model_registry.py:80` |
| H-MODEL-01 | `ACTIVE` in `models/registry/experiments.json`; `evaluate` is descriptive only ("decision rule is the preregistered one … not re-decided here", `shadow_experiments.py:374`) | |

So **OWNER_REVIEW** is a human step with no recorded state, **CONTINUE** exists nowhere, and
**PROMOTED** exists only for models (gate), never for a selection rule or a track.

---

## 8. Production lifecycle (one predict run, then settlement)

1. `predict.yml` (cron :09 every 15 min Fri–Sun daytime, 30 min otherwise, hourly Mon–Thu overnight)
   runs `python pipeline.py --mode predict` with `LEAGUE_SNIPER_CAP=0.12`, `MARKSMAN_THRESHOLD=0.08`,
   `VALUABLE_THRESHOLD=0.03` (lines 188–192).
2. `pipeline.mode_predict` (750): `load_all_matches` → load serving `.pkl`s → `predict_upcoming` (771).
3. `src/predict.py::predict_upcoming` (270):
   * `_fetch_odds_api` (46);
   * features (327);
   * league routing (333–358);
   * side and HT models;
   * `betting.evaluate_value` twice, before and after drift (437, 441–444).
4. Shadow and experiment logs:
   * `shadow_experiments.score` + `append_first_sight` (803–804);
   * `_log_shadow` → `v91_shadow.csv` (809);
   * `_log_h2h` → `shadow_h2h.csv` (810).
5. Tip selection:
   * `generate_bets` (819);
   * `_generate_side_bets` (822);
   * `direction_guard.apply` (834).
6. `edge_engine_shadow.run` on the **unguarded** frames (857). This step is shadow only.
7. `stake_plan.write` (869) → `bankroll_policy.size_day` → `stake_plan.csv` / `stake_audit.csv`.
8. `bets.csv`, `side_bets.csv` and the ledgers are written (884–899).
9. `telegram_bot/notifier.py`:
   * Instant alerts go out only on today's tip day, after the album or after 12:00 UTC (`instant_tip_allowed`).
   * The morning album comes from `daily_summary.yml` (01/03/05/07/09 UTC). It reads the ledgers, grouped by `tip_day`.
10. `python -m src.research.league_weekly` (`predict.yml:241`). This step is research only and runs after Telegram.
11. `update_results.yml` (every 2 h) settles the following:
    * ledgers;
    * `v91_shadow`;
    * H-MODEL-01 (`src.shadow_experiments settle`/`evaluate`/`guard`);
    * the weekly panel.
12. `evidence_refresh.yml` (06:45 UTC) writes:
    * `track_registry.json`;
    * `artifact_manifest.json`;
    * `system_health.json`, which includes `research_lock`.
13. Retrain: `retrain.yml` `0 3 * * *` with a guard that runs Sundays only after 2026-10-05. It runs
    `pipeline.py --mode train` (line 135), and each slot goes through `_train_one`:
    * `PG.review_parked` (558);
    * `PG.submit_candidate` (577).

**The stake plan does not reach Telegram.** No `telegram_bot` file reads `stake_plan.csv`. Its readers
are dashboard pages and `pages/free_tips.py`. The bankroll is a fixed `STAKE_PLAN_BANKROLL` of 100
units with `peak = bankroll` (`stake_plan.py:234`), so the drawdown throttle never engages.

---

## 9. ARCHITECTURE MAP (section 46)

Format: `LAYER › component — STATUS — implementation — canonical artifact`. STATUS is one of
PRODUCTION, SHADOW, RESEARCH, GOVERNANCE, LEGACY or DEPRECATED.

```
L1 DATA FOUNDATION
  › board odds (OddsAPI)            — PRODUCTION — src/predict.py::_fetch_odds_api          — output/book_odds_snapshots.csv
  › per-book quotes (API-Football)  — PRODUCTION capture / SHADOW consumer — src/book_quotes.py, src/edge_af_quotes.py — output/book_quotes/<date>.csv
  › NF dense O/U archive            — PRODUCTION — src/drift.py                             — output/newformat_odds_dense.csv
  › CLV capture + NEAR loop         — PRODUCTION — src/clv_capture.py, capture_schedule.py  — output/clv_collection_health.json
  › exchange lay capture            — RESEARCH   — exchange_odds_capture.yml                — output/exchange_quotes/2026-10.csv
  › team history                    — PRODUCTION — src/data_loader.py, af_history           — af_history.parquet
  › player history                  — PRODUCTION — player_model/pipeline.py::_save_history  — player_history.parquet/month_*.parquet (51)
  › identity / time keys            — PRODUCTION — team_names.resolve, tip_day, gameweek(research) — —
  › data contracts                  — GOVERNANCE (observe-only) — src/contracts            — system_health.json "Data contracts"
  › freshness                       — GOVERNANCE — src/artifact_freshness.py               — output/artifact_manifest.json
  › first-sight freeze              — GOVERNANCE — src/research/first_sight.py             — sidecar *.first_sight.json

L2 PREDICTION MODELS
  › O/U 2.5 standard / new-format   — PRODUCTION — src/model.py                             — models/model_v9_{standard,newformat}.pkl
  › side + HT models                — PRODUCTION — src/model.py, ht_features                — models/registry/<slot>/
  › champion pointer                — GOVERNANCE — src/model_registry.py                    — models/registry/current_champions.json
  › PG-1.0 gate                     — GOVERNANCE — src/promotion_gate.py                    — output/model_lineage/promotion_decisions.csv
  › H-MODEL-01 HGB challenger       — SHADOW     — src/shadow_experiments.py                — output/model_lineage/shadow_experiments.csv
  › player props                    — PAPER      — player_model/*, props_gate (PG-PROPS-1.0) — output/model_lineage/props_promotion_decisions.csv
  › Fantasy V2 residual             — SHADOW     — player_model/fantasy_v2.py               — output/fantasy_v2_summary.json

L3 BETTING INTELLIGENCE
  › legacy raw-edge selection       — PRODUCTION (LEGACY engine) — betting.evaluate_value, pipeline._generate_side_bets — bets.csv, side_bets.csv, *_ledger.csv
  › direction guard                 — PRODUCTION — src/direction_guard.py                   — output/direction_conflicts.csv
  › staking                         — PRODUCTION (output only) — stake_plan.write → bankroll_policy.size_day — output/stake_plan.csv, stake_audit.csv
  › market_edge_v2                  — SHADOW     — market_edge, edge_engine_shadow, edge_af_quotes — output/edge_engine_shadow.csv
  › old edge_engine.py              — DEPRECATED (de-vig helpers still used by clv_capture) — src/edge_engine.py — —
  › regime registry                 — RESEARCH   — scripts/build_league_regimes.py          — config/league_market_regimes.json
  › weekly league panel             — RESEARCH   — src/research/league_weekly*.py           — output/league_weekly_*
  › v9.1 / VALUABLE shadow          — RESEARCH   — src/shadow_compare.py                    — output/v91_shadow.csv
  › top picks                       — PRODUCTION display — src/top_picks.py                 — output/top_picks_log.csv
  › Pro forward tests               — RESEARCH   — Pro src/studies/*, src/player_v3, src/price_trader — Pro output/edge_reassessment/, output/studies/

L4 DEPLOYMENT & GOVERNANCE
  › scheduling                      — PRODUCTION — 26 v9 workflows                          — workflow_health.json
  › delivery                        — PRODUCTION — telegram_bot/notifier.py, daily_pack     — telegram_bot/notified.json
  › dashboard                       — PRODUCTION viewer — app.py, pages/*                   — —
  › research lock                   — GOVERNANCE (detector) — src/research_lock.py          — config/research_lock_snapshot.json
  › system health                   — GOVERNANCE — src/system_health.py                     — output/system_health.json
  › track registry                  — GOVERNANCE read-model (not read by production) — scripts/build_track_registry.py — config/track_registry.json
  › system contract                 — GOVERNANCE — Pro registry/WOWZA_SYSTEM_CONTRACT.json  — validator in Pro
```

---

## 10. Does the implementation match the diagram? Gaps

| # | Gap | Evidence | Consequence |
|---|---|---|---|
| G1 | **Production still selects with the legacy raw-edge engine.** The market-first engine is shadow only, with `UNVALIDATED_DEFAULTS`. | `config/track_registry.json` `selection_engine_version = {"production":"legacy_raw_edge","shadow":"market_edge_v2"}`; `edge_engine_health.json` (2026-10-25T13:12:14Z) `config_status UNVALIDATED_DEFAULTS`. All 752 shadow rows are NO_BET. | The architecture's L3 "market-first" box is not what serves. The tier is still "model disagrees with one vigged price". |
| G2 | **Nothing on the predict/stake path reads `track_registry.json`.** | Readers: `src/research/league_weekly.py:113,236`, `src/system_health.py:366-376`, dashboard. File `about`: "Production does NOT read this file yet". | It is a read-model checked against production (`tests/test_track_registry.py`), not the source of truth. |
| G3 | **The module layout is not physically refactored into `src/data`, `src/models`, `src/market`, `src/selection`, `src/research` and `src/deployment`.** | `src/` is flat (≈50 modules). Only `src/research/`, `src/contracts/` and `src/cloudbet/` are subpackages. | The layers are documentation, not import boundaries. Nothing stops a selection module importing research code. |
| G4 | **The research lifecycle is not one state machine** (§7). | Separate vocabularies; no OWNER_REVIEW, PROMOTED or CONTINUE states for selection rules. | Promotion of a selection rule is a manual owner decision with no recorded state. |
| G5 | **One automatic path changes selection without a gate.** `backtest_matrix.yml` (monthly, `0 3 1 * *`) rewrites `models/best_params_*.json`. A league becomes `approved` when its OOS ROI > 0 and n ≥ `min_oos_bets` (`backtest.py:654-666`), and `betting.py:154-157` then uses its thresholds. | Approved today: standard O/U 2.5 League Two (0.13/0.11) and Serie B (0.12/0.10); BTTS Bundesliga 2 and Championship. | `research_lock` only flags it after the commit. The next run is 2026-11-01. |
| G6 | **One position per fixture is enforced per run, not across runs.** | `stake_audit.csv` rows 354/375: San Lorenzo v Newell's 2026-10-24 staked OVER15 (0.34 u) and BTTS (0.79 u) in two runs on 2026-10-22. | A contract violation on a real-money track (paper sizing today). |
| G7 | **Thresholds are not one system.** The env thresholds do not reach side markets: those use the fixed bars 0.10/0.08, and VALUABLE is hard-coded at `edge >= 0.04` (`pipeline.py:171-175`). Side edges use `(1/odds)/1.08` (`pipeline.py:125-126`), while O/U 2.5 uses `1/odds` (`betting.py:246-249`). `EDGE_CEILING` (0.15–0.19) cannot fire under the 0.12 SNIPER cap. | `track_registry.json` records `valuable_source: "pipeline.py hard-coded 0.04 (does NOT follow VALUABLE_THRESHOLD)"`. | A reader of `config.py` or the env alone misreads side-market selection. |
| G8 | **The stake plan is not delivered, and the bankroll is notional.** | No `telegram_bot` reader of `stake_plan.csv`; `STAKE_PLAN_BANKROLL` defaults to 100 with `peak = bankroll`. | The drawdown throttle in the policy never engages. |
| G9 | **Two different prospective logs for "the challenger".** | H-MODEL-01 (HGB) → `shadow_experiments.csv`; PG-1.0 period 2 reads `shadow_h2h.csv` (`promotion_gate.json:41`). | The H-MODEL-01 n is not the n that PG-1.0 counts. |
| G10 | **Training paths outside the gate's bookkeeping.** | `backtest.yml` (manual) runs `--mode train` and commits `.pkl`s but not `models/registry` or `output/model_lineage`. `retrain.py` calls the gate (265–285), but no workflow runs it. `preseason_retrain.yml:64` runs `--mode collect`, which is not a valid choice (`pipeline.py:1161`), and the failure is hidden by `continue-on-error`. | Pointer and serving file can diverge if `backtest.yml` is run by hand. |
| G11 | **Stale self-description.** | `output/v91_readiness.json` (2026-10-25T13:28:44Z) says the promotion gate is "Not yet wired into pipeline.py", but `promotion_decisions.csv` holds 2 PG-1.0 decisions. | Misleads a reader of that artifact. |
| G12 | **The edge shadow has no per-row config hash.** | `edge_engine_shadow.csv` has `selection_engine_version` and `legacy_engine_version` columns only. The config sha lives in `edge_engine_health.json` and Pro's registration. | A config change mid-window can only be detected run-level (Pro marks `BLOCKED_CONFIG_CHANGED`). This is an owner decision (roadmap). |
| G13 | **Contracts observe; they do not enforce.** | `src/contracts/shadow.py` docstring. | Violations (§5) persist until a writer is changed. |
| G14 | **The v11 shadow cannot be verified from this machine.** | Local `wowza-v11` HEAD `4813ed7` (2026-10-20); last fetched remote `428c49c` (2026-10-22, bot). | The v11 status in this pack is as of 2026-10-22. |
| G15 | **Pro's research index omits the newest forward tests.** | Pro `src/monitoring/research_index.py:339-361` (23 studies) has no `edge2_forward` or `candidate_forward`. | The dashboard "Latest findings" does not show H-EDGE2. |
