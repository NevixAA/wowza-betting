# WOWZA roadmap — status as of 2026-10-25

Owner brief "WOWZA platform architecture" 2026-10-25. **Documentation only.**

**Source note.** The text of the brief's section 34 was not available to the author as a file. The
item list below is reconstructed from three sources:

* the 2026-10-22 repair brief as recorded in `docs/WOWZA_SYSTEM_REPAIR_2026_10_22.md` (P0-1 … P1-6);
* the 2026-10-22..25 work list the owner handed over;
* the commit messages that cite owner briefs.

If section 34 names an item that is not here, it is missing from this table. It is not done.

Status: **DONE** (commit given), **PARTIAL** (what is missing is stated), **NOT STARTED**.
v9 commits unless prefixed `Pro`.

---

## P0 — correctness of what is served and what is logged

| Item | Status | Evidence | Missing |
|---|---|---|---|
| P0-1 side-market staking contract (side tips reached the policy as `unknown\|*`) | **DONE** | `b4ce9e22` (2026-10-22). After it, `stake_audit.csv` carries real tracks; the 110 pre-fix rows remain as history. | — |
| P0-2 challenger first-sight logging (H-MODEL-01) | **PARTIAL** | Logger `50ea8012`, guard ordering `8e1c3d38`, owner decision on the comparator `cbe50d3f`. 324 rows logged since 2026-10-22 13:45Z. | `p_pro_hgb` coverage is AMBER: 58% of 209 recent rows (`system_health.json` 2026-10-25T13:28:49Z). |
| Canonical promotion gate PG-1.0 (one writer of the champion) | **DONE** | `03508097`; newformat rollback + park `92d7f0ba`; Pro contract `5c5584c`. | — |
| Props promotion via PG-PROPS-1.0 | **DONE** | `9ce0d8a7`, `673fff95`; 16 decisions logged 2026-10-25 (8 REGISTER + 8 HOLDOUT, all KEEP_CHAMPION). | The period-2 file `output/model_lineage/props_shadow_h2h.csv` is absent. That is expected: it is written only for a parked challenger (`props_gate.py:33-37`, `review_parked` 633), and none has qualified yet. |
| Research non-regression lock + regime registry + gameweek id | **DONE** | `b7430a35`; owner approval of the baseline `acc2ec4b` (2026-10-25T12:09:26Z). | — |
| One goal direction per fixture | **PARTIAL** | `47716cb9` (direction guard + size_day step 6b). | The guard covers direction only. The one-position-per-fixture rule is still per-run: San Lorenzo v Newell's was double-staked on 2026-10-22 (open decision D4). |
| Side-market results fall back to API-Football | **DONE** | `a4bef8af` | — |
| Prop prices scoped to the tip's own fixture | **DONE** | `95836478` | — |
| Daily pack: props results = props shown | **PARTIAL** | `8a24e151` | `output/props_shown_log.csv` is ABSENT in the committed tree (`python -m src.contracts`), so the contract has nothing to check yet. |

## P1 — evidence quality and governance

| Item | Status | Evidence | Missing |
|---|---|---|---|
| P1-1 CLV collection density (heartbeat, adaptive NEAR loop, certified close) | **DONE** | `1187386c`; `clv_collection_health.json` (2026-10-25T12:50:05Z): T-30 certified close 71.6% of 134 eligible since 2026-10-23 (all-time 22.1% of 598). | The all-time figure stays AMBER until pre-fix rows age out of the window. |
| P1-2 Pro canonical requirements | **DONE** | Pro `e2572d2`, `263e5fe`, `7c14072` | — |
| P1-3 artifact freshness manifest | **DONE** | `d21b7ed1`; manifest 15 artifacts, 0 stale (2026-10-25T13:28:45Z) | — |
| P1-4 / P1-6 forward-test dashboard, tier ≠ evidence rank | **DONE** | `ad4f1dc2`; Pro `da29320` | — |
| P1-5 standard OU25 stays PAPER | **DONE (deliberately unchanged)** | `config/bankroll_policy.json` `standard\|OU25` base 0 | — |
| system_health (one canonical health verdict) | **DONE** | `57f734c7`, `cb68ab27`; overall WARN, 13 GREEN / 3 AMBER / 0 RED | — |
| Data contracts | **PARTIAL** | `f08b699e` (CONTRACTS-1, 14 schemas, observe-only) | Not enforced. There are 7 open violations in 2 files (see the architecture doc §5). |
| Shared first-sight store | **PARTIAL** | `23577263`. 5 writers use `FirstSightStore`: shadow_experiments, edge_engine_shadow, league_weekly ×2, props_gate h2h. | `v91_shadow.csv`, `shadow_h2h.csv`, `stake_audit.csv`, `top_picks_log.csv` and `props_shown_log.csv` keep their own writers. They are only integrity-checked (`system_health.py:254-267`). |
| Track registry (one read-model of every track) | **PARTIAL** | `f67c3928`; 100 entries, 0 drift | Production does not read it (file `about`). |
| market_edge_v2 stage (a) engine in shadow | **DONE** | `065209bc` (live since 2026-10-25T09:25:36Z) | — |
| market_edge_v2 stage (b) API-Football quotes as second price source | **DONE** | `0ae781c7`. OU15 two-sided coverage on the 09:25 board: NF 7→53/99, std 0→12/84. | — |
| market_edge_v2 stage (c) retrospective replay + migration report | **DONE** | `03e3a5e8`; `docs/MARKET_FIRST_EDGE_ENGINE_MIGRATION_2026_10.md`, `docs/EDGE_ENGINE_AUDIT_2026_10.md` | — |
| Pro H-EDGE2 forward registry | **DONE (registered; accruing from 2026-10-26)** | Pro `5a8cee7`; `edge2_forward_progress.json` (2026-10-25T13:18:22Z): 4 hypotheses REGISTERED, n=0 | — |
| H-GATE tagged LEGACY_RAW_EDGE_FORWARD_TEST | **DONE** | Pro `5a8cee7`; `registry/candidate_forward_tests.json` (`family_class` at line 45 and each later hypothesis) | — |
| Pro contract `training_contract` | **DONE** | Pro `5c5584c`, `dcc51b4`; `registry/WOWZA_SYSTEM_CONTRACT.json:697-735` | `docs/WOWZA_SYSTEM_CONTRACT.md` has no matching prose section. |
| Weekly per-league research panel | **PARTIAL** | `9c4ae829`, `7add0591`, `acf2dc24` | 0 of 60 panel rows settled; every league NOT_ENOUGH_DATA (needs 30 priced settled rows / 8 gameweeks). |
| Tip day grouping for Telegram | **DONE** | `23b225cf` | — |
| player_history monthly parts | **DONE** | `a418b9a3` (5 season parts → 51 month parts; extend commit 62.4 → 25.0 MB) | The writer behaviour that still rewrites history is owner decision D1. |
| Dashboard restructure + redesign | **DONE (stages 1, 2a, 2b)** | Navigation `32032fa5`; new pages `155a44c8`, `ca6592fa`, `689176bc`, `9a4d5d9f`, `19b5d7e4`; design system `c32a9e6d`; stage 2b `749fd9e4`, `30d03670` | — |

## P2 — architecture completion

| Item | Status | Missing |
|---|---|---|
| Production selects with market_edge_v2 (cutover) | **NOT STARTED (by design)** | Needs H-EDGE2 to reach ELIGIBLE_FOR_REVIEW (n 150–400 settled per hypothesis plus 8 gameweeks), then owner review. The config is still `UNVALIDATED_DEFAULTS`. |
| Fit market_edge_v2 parameters | **PARTIAL** | `output/edge_engine_proposed_params.json` exists but is labelled PROPOSED / RETROSPECTIVE. It is not the production config, and adopting it would block the H-EDGE2 registration (`BLOCKED_CONFIG_CHANGED`). |
| Production reads `track_registry.json` as its source of truth | **NOT STARTED** | — |
| Physical module layout `src/data\|models\|market\|selection\|research\|deployment` | **NOT STARTED** | `src/` is flat apart from `research/`, `contracts/` and `cloudbet/`. |
| One research lifecycle state machine (DISCOVERY → … → PROMOTED/REJECTED/CONTINUE) | **NOT STARTED** | Each subsystem has its own states (architecture doc §7). |
| Contracts enforced (fail closed) at write points | **NOT STARTED** | — |
| Monthly optimizer behind a gate / owner approval | **NOT STARTED** | `backtest_matrix.yml` still auto-approves per-league thresholds (open decision D5). |
| Cross-run one-position-per-fixture | **NOT STARTED** | Open decision D4. |
| Exchange quotes validated (commission, spread) | **NOT STARTED** | Capture is live (`output/exchange_quotes/2026-10.csv`); validation is not. |
| Official round ids for GW-1 | **NOT STARTED** | `src/gameweek.py` round branch dormant: no artifact carries an API-Football round. |

## Future research (registered or proposed)

| Item | Status |
|---|---|
| H-EDGE2 forward family (NF OU25 / NF BTTS / NF O1.5 / STD OU25 control) | REGISTERED, forward from 2026-10-26 |
| H-MODEL-01 HGB vs serving champion, per track n ≥ 1000 | ACCRUING: 173 NF / 151 std rows, 72 / 62 settled |
| Pro props segments (attacking D / attacking M) | COLLECTING: 80 qualifying tips, all cells below n=150 |
| Pro Player V3 forward | ACCRUING (paper) since 2026-10-15 |
| Drift gate recalibration (Confirmed −10.8% vs Neutral +8.0%) | Parked as a May 2027 item; not started |
| NF OU25 odds > 3.0 | Done as a production cap (`7c3f5ab5`, `NF_MAX_OU_ODDS=3.00`); the evidence is in CLAUDE.md |

---

## Open owner decisions

| # | Decision | Evidence | Options |
|---|---|---|---|
| D1 | **`player_history` writers rewrite historical rows.** This is a feature-leak risk and drives repo growth. | `a418b9a3` body: `injury_refresh` rewrites `days_since_last_injury` on historical rows of every active player (~144 MB per commit). The collect refreshes old rows. "Writer behaviours that change stored feature values … left for an owner decision". A stored past feature value that changes after the match is the same leak shape that produced the false lineup "pass" in the 2026-10-19 O/U info-inputs study. | (a) freeze rows older than N days and compute injury recency at feature time; (b) keep as is and accept the growth and the leak risk. |
| D2 | **Per-row config hash in `edge_engine_shadow.csv`.** | Only `selection_engine_version` / `legacy_engine_version` columns exist. The config sha is run-level (`edge_engine_health.json`) and registration-level (Pro `edge2_hypotheses_v1.json`). | Add a `config_sha` column (the file is first-sight, so it applies to new rows only), or rely on Pro's `BLOCKED_CONFIG_CHANGED` run-level check. |
| D3 | **Over 1.5 capture cadence.** | After `0ae781c7`, two-sided OU15 on the 09:25 board is NF 53/99 and std 12/84. `edge_engine_health.json` shows OU15 two-sided share 0.074 NF / 0.0 std on the OddsAPI path. H-EDGE2-NF-O15-01 needs 150 settled two-sided rows. No committed cadence proposal was found in `docs/`. | Raise `book_quotes` capture frequency on match days, or accept a slower accrual for NF-O15. |
| D4 | **Contract violation: double stake on one fixture.** | `stake_audit.csv` rows 354/375 (San Lorenzo v Newell's, 2026-10-24, OVER15 0.34 u + BTTS 0.79 u). `one_position_per_fixture` is per `size_day` call (`bankroll_policy.py:178-186`). | Feed "already staked fixtures" from `stake_audit.csv` into `size_day`, as `prior_directions` does for direction. This is a staking change and needs approval. |
| D5 | **Monthly optimizer auto-approves thresholds.** | `backtest_matrix.yml` `0 3 1 * *` → `best_params_*.json`; `approved` is set by the backtest itself (`backtest.py:654-666`). The next run is 2026-11-01. | Route through owner approval or the research lock (fail the commit when `protected_tracks_modified`). |
| D6 | **Fantasy V2 residual mode.** | `972972d0` set `auto`, `d0c4ac86` reverted it to `shadow`. `fantasy_v2_summary.json`: `decision_residual GO`, `residual_applied false`. | Keep it in shadow, or apply. |
| D7 | **Stale/invalid workflow paths.** | `preseason_retrain.yml:64` `--mode collect` is an invalid choice. `backtest.yml` commits `.pkl` without the registry. `retrain.py` is unused by CI. | Fix or retire (these are workflow changes). |
