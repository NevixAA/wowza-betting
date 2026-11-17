# Go-live merge rehearsal, 2026-11-17 (go-live Sat 2026-11-21)

Owner approval 2026-11-17: "Yes, start now". This rehearsal did the go-live merge on a rehearsal branch, ran
every check the runbook names, ran the go-live-day steps as dry runs, and wrote down what is left.

- **Branches.** v9 `rehearsal/golive-2026-11-17` (this file); Pro `rehearsal/golive-2026-11-17`.
- **Nothing else was changed.** No commit went to `main`, `data-truth-100` or `prep/bars-oddsapi-hist`. No PR was
  merged, no repository variable was set, no workflow was enabled or disabled. The results sync of
  `data-truth-100` was done on a LOCAL branch only (it rides into this branch as part of the merge).
- **Calls.** No paid API was called. The predict and update_results runs had every socket connection blocked
  and no keys in the environment.
- **Commits marked REHEARSAL-ONLY must never be merged**: the U3a lock approval names the rehearsal agent, not
  the owner.

Source heads merged (all current at the end of the rehearsal, after catch-up merges):

| Branch | SHA |
|---|---|
| v9 `main` | `1ed64dabd` |
| v9 `data-truth-100` | `738e24e78` (PRs #46 and #52 included), plus a local results sync |
| v9 `prep/bars-oddsapi-hist` | `48eecf49f` (includes U3a `cf3b4ffc1`, `d7bf522ef`, PRs #49 and #51) |
| v9 `dashboard/clean-record` | `9f60b8b66` |
| v9 `research/top3-ranking` | `addc4dca1` (already contained in prep) |
| Pro `main` / `data-truth-100` | `55f1d4cb` / `380d0b8b` |

## Verdict

| # | Item | Status | One line |
|---|---|---|---|
| 1 | Merges (v9 + Pro) | **GREEN** | 39 v9 file conflicts and 16 Pro conflicts, all resolved and recorded in section 1. 4 silent merge breaks and 6 cross-branch rule clashes were found and fixed. |
| 2a | v9 full suite on the merged tree | **GREEN** | Final: 3,746 passed, 0 failed. The first run (3,716 passed, 5 failed, 5 errors) was traced and fixed (2.1). |
| 2b | `data_truth --gate` | **GREEN** | STOPLIGHT GREEN. Rule 3 is RED but ungated (operational). It was RED on rules 2 and 7 after the merge; both fixed. |
| 2c | `research_lock --check` | GREEN (rehearsal approval) | The stake proposal needs U3a by the owner |
| 2d | `build_track_registry --check` | **GREEN** | 100 entries in sync |
| 2e | `dashboard_manifest --check` + headless render | **GREEN** | 0 missing code or data. 28 of 30 pages render with 0 exceptions. Dashboard and Live hit only the expected "subprocess blocked" admin buttons. |
| 2f | actionlint | **GREEN** | actionlint 1.7.12: 0 findings on 42 v9 and 35 Pro workflows. shellcheck is not installed, so `run:` scripts were not shell-linted. |
| 2g | Team train/serve parity | **GREEN** | 7 served payloads; 6 match dates (11-08 to 11-16); 140 fixtures; 6,020 cells; 0 mismatched; 140/140 probabilities identical per payload |
| 2h | Props train/serve parity | **GREEN** | 5 served markets; 50 fixtures; 1,411 player rows; 128 features; 0 mismatches; every probability identical |
| 2i | Offline predict / update_results | GREEN (with 1 defect noted) | Both run end to end with the network blocked (136 and 86 calls blocked). `update_results --dry-run` still writes the HT ledgers (section 3). |
| 2j | Pro suite against the merged v9 | **GREEN as planned** | 581 passed, 1 failed. The failure is `test_fixture_key_v2` (mixed v1/fk2 warehouse), which P1 fixes, as the runbook expects. |
| 3 | Runbook dry runs | GREEN | U3b: would apply 63 (and the real apply ran clean). U3e freeze dry run: 40 cells. TOP-3 dry runs. U3d exec: 48 PAPER cells. Details in section 3. |
| 4 | CI on the rehearsal ref | **GREEN** | Tests and gates (tests + gates) success; Data Truth stoplight success |
| 5 | Stake decision (owner 2026-11-17, final) | **ADOPTED by the owner; U3a on go-live day** | Every TOP-3 tip 0.5% flat, placed manually; nothing else stakes; nothing auto-executes. 18 tests. Section 5. |

## 1. Merges: every conflict and its resolution

Rule: keep both sides' intended behaviour. On data handling, the repaired data of data-truth-100 wins. Main's
newer live, collector, dashboard and price-store changes stay. No fix was dropped.

Order (runbook 0c): `sync_results_from_main.py` on data-truth-100 (local), then merge data-truth-100,
prep/bars-oddsapi-hist, research/top3-ranking (already in prep: no-op) and dashboard/clean-record, then the
housekeeping. The heads moved during the rehearsal, so main, dt (twice) and prep were merged again at the end
(catch-up).

### 1.1 v9, merge data-truth-100 into main: 28 conflicts

| File | Cause | Resolution | Why |
|---|---|---|---|
| `.github/workflows/phase2_retrain.yml`, `props_clean_retrain.yml` | add/add: the same workflow on both sides; main adds PR #43's gated price-store read step | **main** | A strict superset of dt (only the `vars.PRICE_STORE_READ` step, default off) |
| `tests/test_af_restored_rows_wiring.py`, `tests/test_af_writeback_preserving.py`, `tests/test_dedupe_source_tiebreak.py`, `output/corrections/af_history_restored_provider_rows_2026-11_audit.json` | add/add: main PRs #35/#37/#38 and their dt ports #36/#39/#40 | **dt** | The tests pin dt's architecture (the dt loader never writes af_history back; source rank by `row_source`). Main's later "Exclude them" data (`3887bb387`) is byte-identical on dt. |
| `src/data_loader.py` (18 hunks) | main: HT hotfix plus PRs #35/#37/#38; dt: the full repair plus the ports | **dt** | Every main loader commit is either the superseded HT hotfix or ported to dt. No main-only caller uses a main-only loader API (checked). |
| `src/team_names.py` | main hotfix `48887b83` (5 aliases plus `_KNOWN_DIFFERENT`); dt: the full resolver | **dt**, byte-identical | See 1.5, silent break 1 |
| `src/cloudbet/bot.py`, `telegram_bot/notifier.py`, `dashboard_data/tips_view.py`, `dashboard_data/props_view.py`, `pages/5_🎯_Success_Rates.py`, `pages/11_👤_Player_Props.py`, `tests/test_p2_nf_under_paper_e2e.py` | main hotfixes `062cb2e8`/`936c80b9` against dt's superset (no default bankroll, league evidence, `positive_fraction`, Bar column, flagged rows) | **dt** | Supersets (runbook section 5) |
| `player_model/predict.py` | main hotfix `13cd81e1` plus the prediction ledger; dt: the same no-invented-price rule | **dt plus main's `market_odds` column guard** | Keeps main's KeyError guard |
| `scripts/capture_nf_odds_forward.py`, `scripts/capture_std_sidemarket_odds_forward.py` | main PR #34 (corners/cards quotes to the private store) next to dt's edits | **main's blocks kept** (dt's side was empty) | A main-only collector feature; zero extra calls |
| `scripts/backfill_af_history.py` | both added a column-preserving upsert; dt adds team ids; main casts `fixture_id` to Int64 | **dt** (ids, one upsert) | The stored dt file holds `fixture_id`/team ids as float64, so the Int64 cast would mix dtypes |
| `src/clv_collection_health.py` | main: acknowledged capture incidents; dt: no default window | **both** (dt's no-default call plus main's incidents) | A later fix also needed: section 1.6, clash 1 |
| `tests/test_clv_capture.py` | both added a test | **both** | |
| `dashboard_access.py`, `tests/test_dashboard_navigation.py` | main: Daily Progress, Clean Record, Bars, Fixes pages; dt: Specials page | **both pages** | |
| `config/track_registry.json` | generated | dt, then **rebuilt** | `build_track_registry.py` |
| `deploy/dashboard_manifest.json` | generated | **regenerated** (`--write`) at the end | |
| `output/af_history.parquet` | main: 10 daily extend commits (91,817 rows); dt: 93,494 rows plus ids | **dt** | 0 main keys missing from dt (league/date/home/away); both end on 2026-11-14 |
| `output/collect/fd_history_ht_fill.parquet`, `ht_fill_audit.json` | add/add: main hotfix 33,259 rows; dt 36,393 | **dt** | Runbook section 5 |

### 1.2 merge prep/bars-oddsapi-hist: 2 conflicts plus 1 silent collision

| File | Cause | Resolution |
|---|---|---|
| `src/integrity/data_truth.py` | **Rule-number collision.** dt calls real-data provenance "Data truth 8a/8b/8c" and injuries freshness "11". prep's 6-10 are no global bars, no retired tiers, bet-time prices (8), clean ledgers (9) and props features (10). Main's prediction ledger is 12. With the letter-suffix gate match, prep's gated "Data truth 8" would also have gated 8a. | prep numbering kept; dt's real-data lines **renumbered 13a/13b/13c** (`real_data_provenance.py`, gate list `Data truth 13b`, 2 dt tests, manifest note). New test: a lettered sub-rule never reuses another rule's number. |
| `deploy/dashboard_manifest.json` | generated | regenerated at the end |

### 1.3 merge dashboard/clean-record: 5 conflicts

| File | Resolution | Why |
|---|---|---|
| `pages/24_🧾_Clean_Record.py` | **main** | Per the brief: main's page (PR #42) carries the branch's split path |
| `pages/24_📬_Daily_Progress.py` | **main** | Main's version is newer and on the design system (`bfa216278`) |
| `dashboard_access.py`, `tests/test_dashboard_navigation.py` | both Bars pages registered at first; **owner O3 "Today's Bars page": page 26 kept, page 25 dropped** | section 10.3 |
| `deploy/dashboard_manifest.json` | regenerated | |

### 1.4 Housekeeping in the merge (runbook 0c)

- `git rm tests/test_resolver_hotfix_main.py tests/test_ht_fill_hotfix.py`: main-only hotfix tests, superseded (D16).
- `python -m src.clv_corrections build`: +1 quarantined row (HJK v AC Oulu 2026-11-08, CLV 195%), the same as on 11-15.
- `python scripts/dashboard_manifest.py --write`: 181 code and 115 data files.

### 1.5 Silent merge breaks (no conflict marker; found by checks)

1. **`src/team_names.py`, NameError.** Main's two filter lines auto-merged into dt's `_only()`, but the
   `_KNOWN_DIFFERENT` set they read went with the conflict block. Every token-tier lookup raised. The Live page
   trace caught it. Fixed: the file now equals dt exactly; dt removed the guessing tiers those pairs needed.
2. **Data truth numbering** (1.2).
3. **Pro `tests/test_bankroll_side_tracks.py`.** Main's fixture signature (`staked_pol`) landed on dt's test body.
   Fixed: dt's signature.
4. **Pro `src/scout`.** Only the conflicted scout files were taken at first. The 11-15 rule takes the whole set,
   so the 3 scout test files were added from `origin/rehearsal/go-live-2026-11-21` (unchanged since).

### 1.6 Cross-branch rule clashes (each side green alone; found by the merged-tree suite or gate)

1. **CLV health guard.** dt's `check()` returned INSUFFICIENT_SAMPLE as soon as one section was unmeasured, which
   hid a measured collection FAIL (main's 2026-11-14 capture-incident guard). Now an unmeasured section is a
   reason, never a 0% FAIL, and the other section is still judged.
2. **Main's prediction ledger (v9) read the raw history banks** for settlement. dt's fix16 rule requires the
   loader-served history. Now it uses `serve_history` and `apply_af_history_corrections`.
3. **Main's prediction ledger (Pro) read `team_match_stats` without the kickoff guard** (dt round-3 rule). Now it
   uses `result_guard.drop_unplayed`.
4. **Data truth 7 RED: 27 rows.** Main's props run wrote `VALUABLE` rows on 11-15 and 11-17 for matches on 11-21
   or later, and `player_ledger.csv` has no `generated_at`. The rule now uses `signal_date`, the write day, as it
   already uses the board's predict-run time. Test added.
5. **Data truth 2 RED: 2 counters** in main-only code (`dashboard_data/bars_stats_view.py` PR #42,
   `src/prediction_ledger.py` PR #44). They are allow-listed with reasons; both are tallies, not defaults.
6. **TOP-3 isolation test.** dt's `sharp_paper_capture.py` writes its own `paper_ledger.parquet`, so it matched by
   name only. Allowed, with the reason.

Other: `tests/test_fix16_readers.py` now classifies `scripts/collect_standard_team_cards.py` (reads only its own file).

### 1.7 Catch-up merges (the heads moved during the rehearsal)

| Merge | Conflicts | Resolution |
|---|---|---|
| main (8 commits incl. PR #50 price-store triggers) | 0 | - |
| data-truth-100 (PR #46, re-synced locally) | 2 add/add: `.github/workflows/price_store_sync.yml`, `tests/test_price_store_sync.py` | **main** (PR #50 is a superset of PR #46) |
| prep (PR #51) | 0 | - |
| data-truth-100 (PR #52) | 0 | - |

**v9 total: 39 file conflicts (28 + 2 + 5 + 2 + 2 on the catch-up).**

### 1.8 Pro, merge data-truth-100 into main: 16 conflicts

| Files | Resolution |
|---|---|
| 6 bot CSVs (`bet_builder_*` ×3, `clv_enriched`, `paper_1x2`, `studies/ou_challenger_forward`) | **main** (live rows; P1 re-keys them) |
| `src/scout/{hypotheses,model,parse,promotion,report}.py`, `tests/test_scout.py` | from the 11-15 rehearsal resolution (neither side changed them since) |
| `src/scout/collect.py` | 11-15 resolution plus **main's `19b3b5dc` side-odds store call**, re-applied |
| `src/pipelines/team_stats.py` | Main also changed the file's line endings, so the whole file conflicted. Re-merged LF-normalised: **main's `fetched_at` and corners/cards plus dt's kickoff guard and team ids** |
| `tests/test_lab_identity.py` | **dt** (the merged v9 serves dt's loader: no `SOURCE_PRIORITY*`) |
| `tests/test_bankroll_side_tracks.py` | **both fixtures**, dt's test signature (1.5, break 3) |

## 2. Validation of the merged tree

### 2.1 v9 suite

- **First run** (`dad78e95a`, before the fixes): 3,716 passed, 5 failed, 46 skipped, 6 xfailed, 5 errors (22 min 37 s).
  The 5 failures: 2 DT2 counters (1.6, clash 5), the CLV guard (clash 1), the reader classification (clash 2) and
  the TOP-3 isolation test (clash 6). All fixed.
- **The 5 errors** are `tests/test_price_store_sync.py` (main PR #41): `git push` into a local bare repo exits 128,
  only in the full run on this Windows worktree. The file passes alone (5/5) and with tests n-p (907 passed).
  It is order-dependent on Windows. The Linux CI run is the arbiter (section 4).
- **Final run** (`89d5e15d1`, the tip before this report): **3,746 passed, 0 failed, 46 skipped, 6 xfailed, 0 errors** (13 min 37 s). The run before it (`08919047f`) found 1 more test pinning the old cell stakes (`test_clean_rebuild_tiers`), fixed in `89d5e15d1`. The price-store errors did not recur in either final run.

### 2.2 Gates and checks on the final tip

`data_truth --gate`: STOPLIGHT GREEN (exit 0).

- GREEN: 1, 2 (0 new defaults; 195 allow-listed), 4, 5 (107,108 bank rows to 71,844; 0 duplicates), 6, 7, 8, 9, 10a-c, 11, 12, 13b.
- AMBER: 13a (8 serving coverage gaps; props PENDING) and 13c (7 registered exceptions).
- RED, ungated: 3. NF `home_corners_pg_roll` is 20% missing on the board.

Lock, registry and manifest: see the verdict table.

### 2.3 Parity

| Check | Command | Result |
|---|---|---|
| Team | `scripts/team_train_serve_parity.py <out> models/model_v9_{standard,newformat,btts,over15,over35}.pkl,models/model_ht_over0{5,15}.pkl 6` | 0 of 6,020 cells, 140/140 probabilities identical per payload |
| Props | `scripts/props_train_serve_parity.py --models <installed props_clean_*.pkl> --per-league 2` | 0 of 128 features, 1,411 rows, 5 markets identical |

The props artifacts used are the promoted ones; they match `models/registry/props/*/versions` (for example
goals `c8579f8a`).

### 2.4 Offline runs on real inputs (throwaway worktree, bars applied = the go-live state)

- **`pipeline.py --mode predict`.** The history loads (71,845 rows), the models load and coverage is written.
  Every fixture fetch is blocked (136 calls), and the run ends with "No predictions generated". There is no crash
  and no fake board. The scoring path on real fixtures is covered by the parity runs (2.3).
- **`update_results.py --dry-run`.** It runs every grader end to end with the corrected history (cup and awarded
  rows out, 90-minute scores, provider dates); 86 calls were blocked.
  **Defect: `update_ht_results` and `update_ht_observations` ignore `--dry-run`** and wrote `ht_ledger.csv`
  (2 rows graded) and `ht_observations.csv` (8 filled). The grades were correct, but the flag cannot be trusted
  for those two files (risk R5).
- **`python -m src.top3_league run`** (flag on, lead 120). It appended 2,504 candidates and ended with status
  `RANK_STATS_MISSING`. That is fail-closed until the U3e freeze. Start `2026-11-21T00:00Z`; stake 0.005 read
  from the policy.

### 2.5 Pro suite (`V9_LOCAL` = the merged v9)

581 passed, 1 failed: `test_fixture_key_v2::test_stored_warehouse_is_v2_and_old_new_keys_agree`, as planned
(P1). The first run had 7 failures, fixed by 1.5 (breaks 3 and 4) and 1.6 (clash 3).

## 3. Runbook steps in the rehearsal

| Step | Dry run | Result |
|---|---|---|
| 0c sync | `scripts/sync_results_from_main.py` on dt (local) | run 1: took main's `fd_history` and player month 2026-11; af_history +0 (dt already newer). Run 2 (catch-up): `fd_history` again |
| U3a | `research_lock --approve` (rehearsal name) | lock clean. **The track registry must be rebuilt after U3a** (it embeds the approval). This was missing from the runbook. |
| U3b | `threshold_apply --proposal .../2026-11_bade2c0b.json --dry-run`, then the real apply in a throwaway worktree | would apply 63 entries; the approval IS recorded (the 11-15 R1 is closed). The real apply writes `best_params_standard.json`, `best_params_side_markets.json`, `threshold_applied.json` and the lock snapshot; the lock stays clean. |
| Stakes (old 0c step) | `cell_staking_from_proposal.py --dry-run` | the proposal still derives 4 staked cells. **Under the 11-17 decision this step must NOT run**; the proposed script refuses to write (exit 3). |
| TOP-3 (U3c) | `scripts/top3_t3_dry_run.py --lead 120` | 11-03..09: 14 league-weeks, 10 TOP-3 picks (Championship 3, Norway 4, Japan 2, USA MLS 1), 25 league-rule picks; 9 weeks with 0. 11-10..16: 5 league-weeks, **0 TOP-3 picks**, 3 league-rule picks. |
| U3d (exec) | `exec_cells_from_v9.py` on `prep/exec-clean-rebuild`, `V9_LOCAL` = merged v9 with bars applied | 48 cells (43 UNPROVEN, 5 PROMISING), 2 held by the manual USA MLS block; `deployment_mode: PAPER_ONLY`, `real_betting_allowed: false` |
| U3e | `scripts/top3_paper_freeze.py --dry-run --out <tmp>` | DRY_RUN record: 40 cells, 35 with R2/R3, frame `58f10476`, sha `75d0f4c4...`. It passes now; the 11-15 blocker is fixed. |
| Gate / lock / registry / manifest | as in the runbook | all GREEN (section 2) |

**Cannot be rehearsed, and why:**

| Step | Why |
|---|---|
| 2.1 freeze / U1 unfreeze, U3/U3c/U3f variables | They change live GitHub state (workflow enable/disable, repository variables), which this task forbids |
| M1-M4 PR merges into `main` | live branches |
| P1 Pro fk2 re-migration | It rewrites the warehouse and needs the owner's OK on its counts; its runtime is unmeasured (it ran over 45 min on 11-06) |
| P2 first null-rule settlement | It runs on main; it grades about 1,971 rows |
| U3e real freeze | The script refuses outside 2026-11-20 UTC |
| R6 refit / R7 re-check | They are training; no training is allowed until the owner's gate. The parity part of G1 is rehearsed (2.3). |
| M5 dashboard sync, Telegram sends, exec `--write` | They write live repos or send messages |
| Pro CI on the rehearsal ref | Pro's `tests.yml` pins v9 to `data-truth-100` for any ref that is not main, and fk2 fails by design until P1: a guaranteed red run. Validated locally instead (2.5). |
| U6 / U7 milestone edits; live validation (runbook section 4) | They need the real go-live |

## 4. CI on the rehearsal ref

A push to `rehearsal/**` triggers no workflow in v9 or Pro (every push trigger is branch-filtered; checked).
Only the two read-only workflows (`contents: read`, no secrets used) were dispatched on the v9 ref:

| Run | Workflow | Ref / SHA | Jobs | Conclusion |
|---|---|---|---|---|
| [44143372990](https://github.com/NevixAA/wowza-betting/actions/runs/44143372990) | Tests and gates (`tests.yml`, workflow_dispatch) | `rehearsal/golive-2026-11-17` @ `be877ca27` | tests (full suite on Linux), gates (lock, registry, manifest) | **success** |
| [44143377532](https://github.com/NevixAA/wowza-betting/actions/runs/44143377532) | Data Truth stoplight (`data_truth_gate.yml`, workflow_dispatch) | same | ratchet success, upgrades-wait skipped | **success** |

The Linux full suite is green, so the Windows-only price-store order dependency (2.1) does not reproduce in CI.
No other workflow was dispatched. Every other workflow writes (commits, sends, calls a paid API) or trains,
and Pro CI would be red by design (section 3). Pro branch `rehearsal/golive-2026-11-17` (`7c136285`) was pushed;
it triggered no run. **0 red runs.**

## 5. Stakes: the owner's final decision of 2026-11-17 (adopted: section 10)

Owner: "From Nov 21, EVERY TOP-3 tip carries a real stake of 0.5% of bankroll, flat. I place the bets manually;
exec and Cloudbet stay off. Tips carry no paper label; every TOP-3 tip counts in the Clean Record from 21.11."

### 5.1 Stake settings as they stand on the source branches (facts, before the proposal)

- `config/bankroll_policy.json` `cell_staking.enabled = true`: only cell entries stake, and every other cell is PAPER.
  - Cells at ACTIVE_LOW 0.005: `new_format|BTTS|Argentina Primera Division` (max odds 2.6), `standard|OU25_OVER|La Liga 2`,
    `new_format|OU25_OVER|Finland Veikkausliiga`, `new_format|OVER35|Finland Veikkausliiga` (block override "Lift for Finland only").
- Track entries, used only if `cell_staking.enabled` were set false:
  - `new_format|BTTS` ACTIVE 0.0125;
  - `new_format|OU25_OVER` ACTIVE_LOW 0.005;
  - `new_format|OVER15` ACTIVE_LOW 0.005;
  - `player_props` BLOCKED_FROM_REAL; every other track PAPER or BLOCKED.
- TOP-3 (`src/top3_league.py`) never sizes a stake: `stake_effect` "NONE". Its Telegram line labels each pick by
  its cell's state (Staked / Paper), and the footer says "TOP-3 is a Paper track".
- `output/stake_plan.csv` (`src/stake_plan.py`, `size_day`) sizes every tip by the cell entries. It feeds Cloudbet
  and exec.
- Cloudbet: `src/cloudbet/client.py` `MODE = os.getenv("CLOUDBET_MODE", DRY)`. No v9 workflow sets
  `CLOUDBET_MODE`, and the policy note says PAPER_ONLY. exec's `cloudbet_shadow.yml` is DRY by construction.
- exec: `registry/execution_policy.json` `deployment_mode: PAPER_ONLY`, `real_betting_allowed: false`, real caps 0,
  on main and on `prep/exec-clean-rebuild`.
- v9 repository variables today: `DAILY_COLLECT_ENABLED`, `PHASE2_RETRAIN_ENABLED=true`, `PRICE_STORE_SYNC=1`,
  `SIDE_ODDS_ENABLED`, `STD_TEAM_CARDS_ENABLED`. No TOP-3 or prematch variable is set yet.

### 5.2 The proposed change (commits `1545a430d`, `08919047f`, `89d5e15d1`; U3a needed)

- **`config/bankroll_policy.json`.**
  - The 4 cells become PAPER 0.0; their old values are kept in `before_2026_11_17`.
  - A new block `top3_league.stake = {rule FLAT, fraction 0.005, execution MANUAL}` is added.
  - `policy_version` gets `+top3-flat-manual-2026-11-17`.
- **`src/top3_league.py`.**
  - `top3_stake()` reads the block with no default: absent, invalid, NaN, above the hard cap or not MANUAL means no stake.
  - Every TOP-3/TIER2 pick records `stake_effect` "REAL 0.5% ... placed MANUALLY" and `track_is_real` True.
  - Its Telegram line shows "💰 Stake 0.5% (manual)", and the footer drops "Paper track".
  - A league-rule line keeps its cell's label, so it is now Paper.
  - A paper league's pick (USA MLS) is never sent, so it is never staked.
  - The stake is never written to `stake_plan.csv`, so Cloudbet and exec cannot see it.
- **`src/research_lock.py`.** A new watched block `stakes` (the cell states, fractions and odds caps, plus the
  TOP-3 stake). **Finding:** until now the lock never saw `cell_staking`, so a cell could start staking without
  a re-approval.
- **`scripts/cell_staking_from_proposal.py`.** It refuses to write while `top3_league.stake` is set. Otherwise
  the runbook's "Stakes" step would restore the PROMISING cells to ACTIVE_LOW.
- **Registries.** Production fields refreshed in `league_market_regimes.json` (12 fields) and `track_registry.json`
  (PAPER 77, PAPER_LEAGUE 5, BLOCKED 18). Regimes and evidence are unchanged.
- **Tests.** The new `tests/test_top3_flat_stake.py` (18 tests) proves three things:
  1. every TOP-3 pick shows 0.5% (TOP and TIER2), and an invalid block stakes nothing;
  2. nothing else stakes: every cell is PAPER, `size_day` gives 0 on the 4 former cells (it gave 0.5% on each
     under the old policy), the league-rule line carries no 0.5%, and a paper-league pick is not staked;
  3. nothing executes: the TOP-3 stake never reaches `stake_plan`, Cloudbet or the client, Cloudbet keeps no real
     candidate, the client defaults to DRY, and no workflow sets `CLOUDBET_MODE`.

  Tests that pinned the 11-15 stakes now pin the 11-17 decision, or test the mechanics on the pre-decision policy.

### 5.3 Texts that had to change (all DONE on 2026-11-17 evening, section 10)

| Where | Today | Needed |
|---|---|---|
| `reports/dashboard/bars_statistics.json` `rule` (main PR #42; Bars page "Go-live stake plan") | "Real stakes on 2026-11-21 ONLY on the 3 PROMISING cells ..." plus per-cell `stake_state: ACTIVE_LOW` (8 rows) | Rewrite: every TOP-3 tip 0.5% flat, manual; every cell PAPER |
| `pages/26_📏_Bars.py` "Go-live stake plan (owner decision 2026-11-16)"; "Staked cells - evidence" | 3-4 staked cells | the TOP-3 rule |
| `pages/25_📐_Bars_and_Cells.py` metric "Staked cells ... Real money (everything else PAPER)" | counts cell entries | 0 cells; the TOP-3 rule |
| `pages/24_🧾_Clean_Record.py` "TOP-3 per league (paper track)"; caption "TOP-3 starts as a named paper track"; `dashboard_data/clean_record.py` (flat 1u) | paper | a real 0.5% track, counted from 21.11 |
| `pages/17_🏟️_Leagues.py` "A paper track" | paper | real 0.5% |
| **Telegram per-tip messages** (`telegram_bot/notifier.py` `_paper_track_note`) | every cell is PAPER, so a TOP-3 fixture's ordinary tip message shows "📄 PAPER - not staked", while its TOP-3 message shows "Stake 0.5%" | Owner choice (O1): label ordinary tips "not a TOP-3 tip", or stop sending them |
| **"Staked" figures** (`src/staked.py`; Overview, Dashboard, Bankroll pages) | from the stake plan, so 0 staked after the change | Owner choice (O2): count TOP-3 picks as staked in the dashboard |
| Placed price | the owner bets by hand at his own price | Owner choice (O2): record the taken odds, or settle at the tip price |

## 6. Remaining risks

1. **TOP-3 volume is small and uneven.** 10 picks in 14 league-weeks (11-03..09) and 0 in 5 (11-10..16). A weekend
   can have no stake at all. That is correct by the rule (never padded), but expect it.
2. **The stakes rest on UNPROVEN cells.** In the 11-03..09 dry run 33 of 35 picks were UNPROVEN.
   - Owner rule 2026-11-17: backtests prove nothing; real stakes need live tipped results.
   - The 11-17 decision stakes every TOP-3 tip, so this is the owner's explicit choice, recorded here.
   - The bars and the ranking statistics are backtest-derived (TP-2026-11-15-bade2c0b; the U3e freeze).
3. **`PHASE2_RETRAIN_ENABLED=true`** is still set (runbook R4 says delete it after the refit). A dispatch of
   `phase2_retrain.yml` can fit models.
4. **Price-store push triggers on `data-truth-100` and `prep/bars-oddsapi-hist`.** `PRICE_STORE_SYNC=1` is set, so
   every push to those branches runs a mirror. After go-live these branches are retired, and the triggers should
   move to main (PR #50 already covers main writers).
5. **`update_results --dry-run` writes the HT files** (2.4). Do not rely on it as a no-write check.
6. **The price-store test is order-dependent on Windows** (2.1).
7. **The track fallback.** If `cell_staking.enabled` were ever switched off, `new_format|BTTS` (1.25%), NF O2.5 Over
   and NF Over 1.5 would stake from the track entries. Recommend setting those tracks to PAPER as well (owner).
8. **Data truth 3 is RED** (NF corners roll 20% missing on the board). It is ungated; the models read the column.
9. **The heads keep moving** (4 PRs landed during this rehearsal). On go-live day the merge must be redone on the
   final heads; the resolutions in section 1 are the rules.

## 7. Owner decisions (answered 2026-11-17)

| # | Decision | Owner's answer | Where it is |
|---|---|---|---|
| O0 | Adopt the stake proposal (section 5.2); U3a by the owner | **"Yes, adopt"** | `1545a430d`, `08919047f`, `89d5e15d1`, `42063eb5a`; U3a on go-live day |
| O1 | Telegram labels | **"Show stake on TOP-3"**: TOP-3 picks show "0.5% stake"; every other prediction has NO stake label; the word "paper" never reaches a user | `42063eb5a` (section 10) |
| O2 | "Staked" figures; price of the manual bets | "Staked" counts TOP-3 tips. **"Use tip's odds"**: settle at the odds shown on the tip at send time, with no manual entry form | `42063eb5a` |
| O3 | Bars page | **"Today's Bars page"**: keep `pages/26_📏_Bars.py`; page 25 (Bars & Cells) is not merged | `42063eb5a` |
| O4 | Clean Record start | **"21.11 00:00 UTC"**: `owner_confirmed: true`. Rule kept: if any tip is written on 11-21 before the merge, use the merge instant | `42063eb5a`; command 17 |
| O5 + O6 | `PHASE2_RETRAIN_ENABLED`; R6 | **REFIT on Nov 19-20**: champion_recipe on the clean data incl. the verified restored rows, reading the private price store (`PRICE_STORE_READ=1`); Nov 20 confirm-diff and owner approval; delete `PHASE2_RETRAIN_ENABLED` only AFTER the refit | commands B1-B7 |
| O7 | Delete the two main-only hotfix tests in the merge | **yes** | command 6 |
| O8 | P1 fk2 re-migration | **yes**: dry run first; it must finish before the Pro writers are unfrozen | command 14 |
| O9 | Fallback tracks to PAPER | **"Yes, adopt"** | `42063eb5a` |
| O10 | Argentina NF BTTS league rule | **no stake** (not a TOP-3 tip) | `1545a430d` |

No owner decision is open. Agent-side follow-ups only: the prediction-ledger agent adopts the two reader fixes
(section 9).

## 8. Go-live command list (ordered; each step with its rollback)

Shell prelude: runbook "Shell prelude" (`$GL`, `$py`, git and gh on PATH). `R` = `origin/rehearsal/golive-2026-11-17`,
`RP` = the Pro branch of the same name.

**Nov 19-20: the approved refit (O5/O6)**

| # | Step | Command | Rollback |
|---|---|---|---|
| B1 | R6 gate (runbook 2.6b) | G1 parity (rehearsal: team 0 of 6,020 cells, props 0 of 128 features), G2 collectors, G3 this rehearsal; record `$GL\r6_gate.txt` | stop; no refit |
| B2 | Store read on | `gh variable set PRICE_STORE_READ --body 1 --repo NevixAA/wowza-betting` (locally: `bash scripts/price_store_read.sh wowza-betting data-truth-100` before the refit; it stops unless the store bytes equal the commit) | `gh variable set PRICE_STORE_READ --body 0 --repo NevixAA/wowza-betting` |
| B3 | Refit (runbook R6) | `$env:PHASE2_RETRAIN_ENABLED="true"; $py scripts/phase2_refit_serving.py --out $GL\refit` (champion_recipe, clean data with the verified restored rows); then `$py scripts/team_train_serve_parity.py $GL\parity <12 pkl> 6` = 0 differing cells | do not install |
| B4 | Nov 20 confirm-diff + owner approval | `$py scripts/promote_clean_switch.py assemble --refit $GL\refit --out $GL\asm`; review the diff with the owner; put the 7 sha256 into `final_refit`; `$py scripts/promote_clean_switch.py install --assembled $GL\asm --override clean-data-switch-2026-11-09` | revert the install commit |
| B5 | R7 bars re-check | runbook R7 (exit 0 = the approved bars stand; exit 1 = back to the owner) | - |
| B6 | U3e freeze (11-20 UTC) | add the refit frame to `clean_frames`; `$py scripts/top3_paper_freeze.py --dry-run --out $GL\freeze` (rehearsal: 40 cells), then `$py scripts/top3_paper_freeze.py`; commit both files by path | delete the 2 files before the first pick |
| B7 | **Only after the refit is installed** | `gh variable delete PHASE2_RETRAIN_ENABLED --repo NevixAA/wowza-betting` | `gh variable set PHASE2_RETRAIN_ENABLED --body true --repo NevixAA/wowza-betting` |

**Go-live day (v9)**

| # | Step | Command | Rollback |
|---|---|---|---|
| 1 | Snapshot | `$py scripts/golive/snapshot_state.py --out $GL\state_before.json`; premerge SHAs (runbook 1.6) | - |
| 2 | Freeze the writers | runbook 2.1 (`frozen.csv`) | `gh workflow enable` each id in `frozen.csv` |
| 3 | Results sync on dt | worktree on `origin/data-truth-100`: `$py scripts/sync_results_from_main.py`; commit the synced files by path; push to data-truth-100 | `git revert` that commit |
| 4 | Merge branch | `git checkout -b golive/m1 origin/main`; `git merge --no-ff --no-commit` `origin/data-truth-100`, then `origin/prep/bars-oddsapi-hist`, then `origin/dashboard/clean-record` (page 25 is NOT taken: `git rm "pages/25_📐_Bars_and_Cells.py"`); resolve per section 1. A file neither side changed since the rehearsal: `git checkout R -- <file>` | delete the branch |
| 5 | Rehearsal merge fixes | `git checkout R -- src/team_names.py src/integrity/data_truth.py src/integrity/real_data_provenance.py scripts/dashboard_manifest.py config/data_truth_allowlist.json src/clv_collection_health.py src/prediction_ledger_v9.py tests/test_no_global_bars.py tests/test_daily_injuries_lineups.py tests/test_real_data_pipeline_trace.py tests/test_fix16_readers.py tests/test_top3_paper.py tests/test_clv_capture.py` (first `git diff R origin/<side> -- <file>` must show only the rehearsal fix) | part of the merge PR |
| 6 | Housekeeping (O7) | `git rm tests/test_resolver_hotfix_main.py tests/test_ht_fill_hotfix.py`; `$py -m src.clv_corrections build` (keep content changes only) | part of the merge PR |
| 7 | Owner decisions (O0-O4, O9, O10) | `git cherry-pick 1545a430d 08919047f 89d5e15d1 42063eb5a` in that order. **Never `c609e92b8`, `84dc83faf`, `3d4e1349e`, `8358223da`** (REHEARSAL-ONLY approvals and registries). | drop the commits |
| 8 | U3a (owner), then the registry | `$py -m src.research_lock --approve --by "Nevo (owner)" --note "2026-11-21 go-live: bade2c0b bars; TOP-3 0.5% flat manual; every cell and track at 0"`; **then `$py scripts/build_track_registry.py`**; then `$py scripts/dashboard_manifest.py --write` | revert |
| 9 | Validate | full suite 0 failed; `data_truth --gate` GREEN; `research_lock --check`; `build_track_registry.py --check`; `build_bars_statistics.py --check`; `dashboard_manifest.py --check`; `$py scripts/ui_wording_scan.py` (0 occurrences); `git show --stat HEAD` (no stray `output/*`) | - |
| 10 | M1 PR | push `golive/m1`; `gh pr create --base main`; `gh pr checks --watch`; `gh pr merge --merge` | `git revert -m 1 <merge>` via PR (runbook 2.4) |
| 11 | U3b | `$py -m src.threshold_apply --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11_bade2c0b.json --dry-run` (63), then without `--dry-run`; commit `models/best_params_standard.json models/best_params_side_markets.json config/threshold_applied.json config/research_lock_snapshot.json`, then `$py scripts/build_track_registry.py` and commit `config/track_registry.json` | revert the apply commit |
| 12 | Do NOT run `cell_staking_from_proposal.py` | it refuses anyway (exit 3) | - |

**Go-live day (Pro, exec, v11, dashboard, switch-ons)**

| # | Step | Command | Rollback |
|---|---|---|---|
| 13 | M2 Pro | merge dt into main; the 6 CSVs → main; `git checkout RP -- src/scout tests/test_scout.py tests/test_data_truth_pro.py tests/test_data_truth_round3_pro.py tests/test_scout_data_truth.py src/pipelines/team_stats.py tests/test_bankroll_side_tracks.py tests/test_lab_identity.py src/prediction_ledger_pro.py tests/test_bankroll.py tests/v9_policy_before.py` (after the same diff check); suite with `V9_LOCAL` = new v9 main: only fk2 fails; PR, merge commit | revert v9 and Pro **together** |
| 14 | P1 fk2 (O8) | `python -m src.pipelines.migrate_fixture_key_v2 --season season_2026_27 --dry-run` first (counts to the owner), then the run, suite 0 failed, PR. **Must finish before command 18 unfreezes the Pro writers.** | revert the migration commit first |
| 15 | M3 exec | merge `prep/exec-clean-rebuild`; `$py scripts/exec_cells_from_v9.py` (48 cells), then `--write`; suite; commit `registry/execution_policy.json` (stays `PAPER_ONLY`, real betting off) | revert |
| 16 | M4 v11, M5 dashboard | runbook M4 / M5 | runbook |
| 17 | P0 + U6 (O4) | record the merge time. If no tip was written on 11-21 before it: keep `clean_record_start_utc = 2026-11-21T00:00:00Z`; else set it, `PERFORMANCE_CUTOFF_DATE` and `PROPS_CLEAN_MODELS_LIVE_DATE` to the merge instant (PR). Milestone/2 due = start + 2 months | revert the PR; PATCH the milestone back |
| 18 | P2-P4; U1 unfreeze (Pro writers only after P1) | runbook | `gh workflow disable` |
| 19 | Switch-ons | `gh variable set PREMATCH_CAPTURE_ENABLED --body true`; `gh variable set TOP3_FREEZE_LEAD_MIN --body 120`; `gh variable set TOP3_LEAGUE_ENABLED --body true`; after `pytest tests/test_top3_paper.py tests/test_top3_ranking.py tests/test_top3_flat_stake.py tests/test_ui_wording.py` on main: `gh variable set TOP3_PAPER_ENABLED --body true` (all `--repo NevixAA/wowza-betting`) | `gh variable delete <name> --repo NevixAA/wowza-betting` |
| 20 | (`PHASE2_RETRAIN_ENABLED`) | already deleted in B7, after the refit | - |
| 21 | U8 and retirements | Pro specials to main; retire `data_truth_collect.yml` (syncmain), `oddsapi_history_probe.yml` and the `price_store_sync.yml` push branches | revert |
| 22 | Live validation | runbook section 4. Plus: the first TOP-3 message shows "0.5% stake" on every pick and no other tip carries a stake label; `stake_plan.csv` has 0 stake on every row; the dashboard "Staked" counts the TOP-3 picks | abort criteria, runbook section 4 |

## 9. Other agents' work and the merge order

| Work | State | Lands before or after go-live |
|---|---|---|
| Price store (PRs #41, #43, #45, #50 main; #46, #52 dt; #49, #51 prep) | all merged; in this rehearsal | Anything new: port to main only after go-live (dt and prep retire). Before go-live, keep the same-text ports, so the merge sees add/add with identical text and no conflict. |
| Prediction ledger (PR #44 main; Pro `prediction_ledger_pro.py`) | merged; flag OFF | **Its owner should take over the two rehearsal fixes** (v9 `serve_history` / `apply_af_history_corrections` in `TeamResults`; Pro `result_guard.drop_unplayed` in `_scorelines`). After go-live they are required by the merged rules; before go-live they are harmless on main. Rule numbers: 12 is the ledger, 13 is real-data provenance (not "11 is the last"). |
| Open PRs at the end of the rehearsal | none in v9 or Pro | - |

Rule for anything landing between now and go-live: a PR to main that touches a file in section 1 adds a conflict
on the day. Prefer porting to dt in the same text, or wait until after the merge.

## 10. Owner decisions applied (2026-11-17 evening): changes and validation

Commits on the v9 rehearsal branch: `42063eb5a` (the decisions) and `3d4e1349e` / `8358223da`
(REHEARSAL-ONLY lock approval and registry). Pro: `1cf2ab78`.

### 10.1 Stakes (O0, O9, O10)

- **Every cell and every track at 0.** The fallback tracks were `new_format|BTTS` 1.25%, `new_format|OU25_OVER`
  0.5% and `new_format|OVER15` 0.5%; now PAPER 0, with the old values kept in `before_2026_11_17`.
- **The only real stake** is `top3_league.stake` (FLAT 0.005, MANUAL).
- **`policy_version`** ends `+top3-flat-manual-tracks-paper-2026-11-17`.
- **Tests on the pre-decision policy.** The mechanics tests (sizer, side routing, caps, Cloudbet bot) use it:
  v9 `tests/policy_helpers.py`, Pro `tests/v9_policy_before.py`. The committed state is pinned by
  `tests/test_top3_flat_stake.py` and `tests/test_cell_staking.py`.

### 10.2 Wording (O1, O2): TOP-3 "0.5% stake", nothing else, never "paper"

- **TOP-3 Telegram message.** Each pick ends "· 0.5% stake"; the footer reads "Every TOP-3 tip: 0.5% stake of
  bankroll, flat, placed manually." League-rule lines carry no label.
- **Ordinary Telegram tips.** No stake label at all (`_paper_track_note` returns ''). The HT tip header lost its
  "(PAPER)". The auto-onboard message no longer promises a stake.
- **Dashboard board.** The "Stake note" column is "0.5% stake" on a TOP-3 pick and empty on every other tip.
  TOP-3 picks are overlaid from their ledger onto the board (`Stake %`, `plan_staked`).
- **"Staked" figures.** `src/staked.load_records` reads every TOP-3 pick whose frozen `stake_effect` is REAL as a
  recorded decision. A TOP-3 decision before kickoff decides its position, so later 0-stake plan rows never
  withdraw it. Its P/L is at the ledger's tip odds (O2 "Use tip's odds": the odds frozen on the pick at send time).
  The definition text says so.
- **"Paper" never reaches a user.** `dashboard_words.py` relabels at the display layer:
  - every Streamlit text, table, metric, tab and widget call (option values stay raw; only what is shown is relabelled);
  - PIL text in the Telegram pictures;
  - every Telegram send and caption.

  Internal states (PAPER in the policy and ledgers) are unchanged. The owner-named pages were reworded at the
  source: Bars, Clean Record ("TOP-3 per league (0.5% stake, flat; P/L flat 1u)", "staked or not"), Leagues and Home.
- **Proof.** `scripts/ui_wording_scan.py` renders all 30 pages headless (every visible string, table cells
  included). Before: 159 occurrences on 23 pages. After: **0**. `tests/test_ui_wording.py` (6 tests) pins the
  mechanism. The scan covers each page's initial render (every tab); views behind other widget choices are covered
  by the relabel layer, not by the scan.
- **`bars_statistics.json`.** `scripts/build_bars_statistics.py` now reads this tree's policy. The stake plan is
  the 11-17 decision with 0 staked cells, and the per-cell rule reads "no stake of its own; a TOP-3 pick of this
  cell: 0.5% flat". `--check` is green.

### 10.3 Bars page (O3), Clean Record (O4), runbook

- **Page 25 removed:** `git rm`, navigation, the design-system list, the dashboard truth reproducer (its 3 figures).
  `cells_view` stays, because Bankroll and Clean Record use it.
- **`config/trustworthy_performance.json`:** `owner_confirmed: true`, with the owner's words "21.11 00:00 UTC" and
  the merge-instant rule.
- **`reports/GO_LIVE_RUNBOOK.md`:** U3a is now followed by `build_track_registry.py`, in the U3a row and in 0c.

### 10.4 Fixes asked for

- **`update_results --dry-run`.** `update_ht_results` and `update_ht_observations` now take `dry_run`, and main()
  passes it to them. The offline rerun logs "[DRY] would grade 2" and "[DRY] would fill 8", and the worktree stays
  clean. `tests/test_update_results_dry_run.py` checks every grader.
- **Prediction-ledger reader fixes.** Already in the tree (`69e7dc623` v9, `7c136285` Pro).

### 10.5 Validation of the new tip

| Check | Result |
|---|---|
| v9 full suite | FINAL3_RESULT |
| `data_truth --gate` | STOPLIGHT GREEN (rule 3 RED, ungated) |
| `research_lock --check` | clean (rehearsal approval) |
| `build_track_registry.py --check` | in sync: 100 entries (BLOCKED 18, PAPER 77, PAPER_LEAGUE 5) |
| `build_bars_statistics.py --check` | up to date |
| `dashboard_manifest.py --check` + render | 0 missing; 28 pages with 0 exceptions (Dashboard and Live: expected blocked admin buttons) |
| `ui_wording_scan.py` | 0 occurrences on 30 pages |
| actionlint 1.7.12 | 0 findings (v9 and Pro) |
| Offline `update_results --dry-run` | no file written; 86 calls blocked |
| Pro suite against the v9 tip | 581 passed; 1 failed (fk2, P1) |
| CI on the rehearsal ref | CI3_RESULT |
