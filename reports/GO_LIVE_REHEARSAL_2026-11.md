# Go-live rehearsal, 2026-11-15 (go-live 2026-11-21)

Owner 2026-11-15: "all the architecture should work perfect". This rehearsal did the go-live merges on
`rehearsal/go-live-2026-11-21` branches in v9, Pro, exec and v11. It ran every check, simulated one live day
offline, and fixed what broke on the source branches, each fix with tests.

- **No default branch, secret, repository variable or paid API was touched.**
- One GitHub Actions artifact (the props clean-retrain models of run 43035218594) was downloaded for the props
  parity check. The actionlint download was refused (section 5).

## Verdict: architecture GREEN after fixes; 4 RED items open (owner actions and one data fix)

| # | Step | Status | One line |
|---|---|---|---|
| 1 | Merges in 5 repos | GREEN | All merges done. Every conflict is recorded with its rule (section 1). Two silent breaks were found and fixed: the rule-number collision and the scout `lg`. |
| 2a | v9 full suite on the merged tree | AMBER (owner items) | 9 failed, 3,487 passed, 42 skipped, 6 xfailed (27 min). Of the 9: 2 wait on R1, 5 on R2 (lock), 1 on R3. The last one, `test_shadow_settlement[Kaiserslautern]`, depends on test order: the file passes alone (42/42), the same symptom runbook 5.1 noted. Expected after R1-R3: 1 order-dependent test, to be checked on the Linux CI run of the PR |
| 2b | `dashboard_manifest` (`--write` traces every page; `--static`) | GREEN | 176 code + 110 data files; 0 missing; every page rendered headless during the trace without an exception |
| 2c | `data_truth --gate` | GREEN after fix | It was RED on rule 7 (main's pre-merge board). Fixed: a board takes its predict-run time. 1, 2, 4, 5, 6, 7, 8 and 10a-c are GREEN; 3 (ungated, operational) is RED |
| 2d | `threshold_apply --dry-run` TP-2026-11-15-bade2c0b | **RED (owner)** | REFUSED: no owner approval in `config/threshold_approvals.json`. With a scratch approval it applies 63 entries after U3a |
| 2e | Cell staking | GREEN after fix | It builds exactly the 4 owner cells. Argentina at 0.5% now needs `--set-fraction` (added); the script carried 1.25% verbatim before |
| 2f | Exec cells dry run (bade2c0b, merged v9) | GREEN | 48 cells (5 PROMISING, 43 UNPROVEN), all PAPER; USA MLS held by the manual block; 17 entries skipped (no bars_opt cell) |
| 2g | TOP-3 T3 dry run, latest real board (2026-11-09..15) | GREEN | 6 league-weeks, 160 candidates: 1 TOP-3 pick (Norway) + 4 league-rule picks. All UNPROVEN, no retired tier |
| 2h | Stats freeze dry run | **RED (U3e input)** | REFUSED: the v5 run rebuilt `universe_standard` (f833a9…), which is not a listed clean frame. It also reads cells from `approved_proposals` = d999073a. Simulated locally with both fixed: 60 cells frozen, mechanics OK |
| 2i | Parity, team + props | GREEN | team 0 (7 served payloads, last 6 match dates 11-04..09: 221 fixtures, 9,503 cells, 0 mismatched, max 0.0 pp); props 0 (5 served markets, 50 fixtures, 1,411 player rows, 128 features: 0 mismatches, every probability identical) |
| 3 | One live day offline | GREEN after fixes | Every artifact is produced. 6 label and stake defects were found and fixed (section 3). Calibrated edge stays 0 (decision D-T3) |
| 4 | Pro / exec / v11 / dashboard on the merged v9 | GREEN | Pro 564 passed, 1 failed (fk2 warehouse, fixed by P1). exec 195 passed. v11 26 passed (after the restate step). Dashboard slim build: 182 files |
| 5 | Workflows | GREEN with notes | No lint error. Secrets present by name except 3 optional ones, each handled. Switch-on variables listed (section 5) |
| R1 | **bade2c0b approval not recorded** | **RED (owner)** | Blocks U3b. It also fails 2 Clean Record tests on the merged tree. It must name a `price_provenance` record, or Data truth 8 turns RED |
| R2 | **Research lock (U3a) must be in the go-live PR** | **RED (sequence)** | 4 lock-dependent tests fail until the lock is re-approved. Without it in the PR, the PR's `tests` job fails |
| R3 | **Provider score conflict** | **RED (data)** | Ireland 2026-11-07: football-data UC Dublin 2-1 Drogheda vs API-Football UCD 1-2 Drogheda United. `test_identity_bray` fails |
| R4 | **TOP-3 calibrated edge is always 0** | **RED (decision)** | Simulated freeze: 35 cells have a calibration slope, 0 have a cautious slope above 0. Every model pick shows "+0.0 pp" and ranks by the bookmaker margin |

## 1. Merges: conflicts and resolutions

| Repo | Merge | Conflicts | Resolution |
|---|---|---|---|
| v9 | `scripts/sync_results_from_main.py` on data-truth-100, **then** dt into main `ab48dec99` | 17 (before the sync: 12 + 3 binary player months) | 15 resolved to dt: team_names, data_loader, cloudbet/bot, notifier, player_model/predict, tips_view, props_view, pages 5 and 11, e2e test, track_registry, af_history, HT fill + audit, manifest. `dashboard_access.py` and `test_dashboard_navigation.py` keep both pages (Specials + Daily Progress). After the sync, `fd_history` and `player_history` merge cleanly |
| v9 | prep/bars-oddsapi-hist | 1 (`data_truth.py` GATED) **+ a silent collision**: "Data truth 7" was props features on dt and "no retired tiers" on prep | Props checks renumbered **Data truth 10** (gated). Now on the branch (`10f742029`), with a test that every rule number names one rule |
| v9 | research/top3-ranking | 0 | - |
| v9 | dashboard/clean-record | 2: manifest; Daily Progress page add/add (main vs the branch's port) | The branch's page (now on `ui.page_header`); manifest regenerated with `--write` |
| v9 | after the merges | - | `git rm tests/test_resolver_hotfix_main.py tests/test_ht_fill_hotfix.py` (main-only hotfix tests, superseded; the HT one after D16); `python -m src.clv_corrections build` (+1 quarantined row: HJK v AC Oulu 2026-11-08, close 1.0, CLV 195%) |
| Pro | dt `a451ee79` (+ `b12b4300`) into main | 13: 6 bot CSVs; 7 scout files (both sides changed the scout: main's Data Truth R1-R15, dt's round 3 + 1X2 prices) | CSVs to **main** (P1 re-keys them). The scout was merged by meaning: main's design (UNKNOWN_TEAM rows, truth.results, v9_league_ids), plus dt's result_guard (results observed before kickoff dropped) and dt's 1X2 league parameter. **Silent break found:** `parse.odds_rows` lost its `lg` assignment (10 tests). Fixed in the rehearsal merge. On go-live day: `git checkout origin/rehearsal/go-live-2026-11-21 -- src/scout tests/test_scout.py tests/test_data_truth_pro.py tests/test_data_truth_round3_pro.py tests/test_scout_data_truth.py` |
| exec | prep/exec-clean-rebuild `9bdacdb` into main | 3: execution_policy.json, test_ops6_rating_sizing_off.py, test_pipeline.py | All to the branch (runbook rule). main's daily-report commits merged cleanly |
| v11 | dt `51c8a9d` into main | 0 | Then `scripts/v11_restate_p_market.py --v9 <merged v9> --append` (3,368 rows added, 83,173 kept), which V11 Collect runs anyway |
| dashboard | no merge (generated) | - | M5 = `Sync from v9`. Slim build from the merged v9: 182 files, 28 pages |

## 2. Checks on the merged v9 (details)

- **Suite.**
  - First run, 15 failed of 3,482.
  - Each failure was triaged:
    - 4 were fixed on source branches (CLV pin, 2 new defaults, design system, dispatch path);
    - 2 were removed as main-only hotfix tests;
    - 2 wait on R1;
    - 4 wait on R2 (lock: `test_research_lock` ×2, `test_team_frame_health`, `test_threshold_proposals` ×2);
    - 1 waits on R3 (`test_identity_bray`).
  - Second run: 9 failed, 3,487 passed, 42 skipped, 6 xfailed (27 min). Of the 9: 2 wait on R1, 5 on R2 (lock), 1 on R3. The last one, `test_shadow_settlement[Kaiserslautern]`, depends on test order: the file passes alone (42/42), the same symptom runbook 5.1 noted. Expected after R1-R3: 1 order-dependent test, to be checked on the Linux CI run of the PR.
- **Gate:** STOPLIGHT GREEN on prep after the rule-7 fix.
  - Rule 3 is RED and ungated: `home_corners_pg_roll` 25% missing on new-format, and HT history on board fixtures.
  - The rule-7 fix first shadowed rule 1's `BOARD_FILES` list and crashed rule 1 (`bf0f655b7`). It was caught and fixed within minutes (`f027d8508`), with a guard test.
- **Track registry `--check`:** in sync, 100 entries.

## 3. One live day, offline

- **Board:** the committed `predictions.csv` of 2026-11-07 05:52Z, 112 fixtures (11-07..09). History was cut before 11-07, the clock was simulated, no network was used and Telegram text was rendered only.
- **Go-live state, simulated locally and never committed:** bade2c0b applied through a scratch approval after a local U3a, the 4 cells at 0.5%, a local freeze, and the Clean Record start set to the board day.

| Artifact | Result |
|---|---|
| Predictions | 112 rows. Leagues without a cell get NO_LEAGUE_BAR. 26 direction conflicts suppressed |
| O/U tips + ledger | 15 rows, QUALIFIED, no tier. **Defect: cell_status / qualifying_bar / calibrated edge blank.** Fixed (`45d56dc2b`) |
| Side tips + ledger | 22 rows, QUALIFIED with cell status and bar |
| Stake plan / audit | 60 rows, tier_multiplier 1.0. Staked: Argentina NF BTTS 0.5% (7) and Finland NF O3.5 0.5% (2); the rest PAPER/BLOCKED; Cloudbet PAPER_ONLY. **Defect: 4 Argentina stakes had no ledger row** (dedup vs older tips). Fixed (`b9512083c`): only ledgered tips are sized |
| Telegram text | No retired tier word. **Defects:** the PAPER line came from the track, not the cell (NF Over cells showed no PAPER line, a staked La Liga 2 Over would show PAPER), and side tips had no PAPER line. Fixed (`41616a13b`) |
| TOP-3 | 32 picks (12 TOP-3 model cells, 20 on the league-rule line, kept apart). **Defects:** the first run decided 44 past weeks from pre-go-live candidates; a USA MLS (paper league) pick was sent; the footer said "Paper" over the staked Argentina line. Fixed (`641220c53`): gate at `start_utc`, candidates first seen before the start excluded, paper leagues dropped, stake label per line |
| Paper track R2 | Status OK, 0 picks (the protocol starts 11-21) |
| Settlement | O/U 15 settled (5W 10L, -4.82u); side 22 (13W 9L, +1.25u); TOP-3 25 of 32 (7 fixtures with no v91 row) |
| Clean Record | Figures produced. **Defect: the split used the start DATE**, so a row written on the go-live day before the merge would count. Fixed (`72bbf6922`): the full `clean_record_start_utc` instant, with its test following the configured instant |
| Bars & Cells | Fails closed until the approval is recorded. **Defect: it read the old evidence file**, so staked Finland cells showed OFF. Fixed (`78c2a58c1`): the proposal's own `source.evidence` |
| Dashboard pages | Home, Clean Record, Bars & Cells, Dashboard and Success Rates rendered; no retired tier word |
| Data truth 7 | GREEN on every simulated row |

**Not fixed, noted for the owner:**
- A rescheduled fixture got the opposite side on its new date. The direction guard keys on the match date (Rosario Central v Boca, moved 11-08 to 11-07).
- `league_weekly` stamps old rows with the current model version.
- The side ledger stores odds with float noise.

## 4. Cross-repo against the merged v9 (V9_LOCAL = rehearsal tree)

- **Pro:** the first run had 22 failed.
  - 9 bankroll tests pinned v9's old per-track tier multipliers. Fixed on Pro dt (`b12b4300`); green against both v9 dt and the merged v9.
  - 12 were scout merge fallout. Fixed in the rehearsal merge (section 1).
  - Result: **564 passed, 1 failed.** The failure is `test_fixture_key_v2`, the mixed v1/fk2 warehouse, and P1 fixes it as planned.
- **exec:** 195 passed, 1 skipped.
- **v11:** 26 passed after the restate step.
- **Dashboard:** the slim build from the merged v9 succeeds (182 files).

## 5. Workflows

- **actionlint** was refused by the session's permission check (a binary download). Instead I ran a structural lint on every workflow in the 4 merged repos and the dashboard's sync: YAML, jobs, `needs`, `inputs`, expressions, and secrets/vars by name. Result: **0 errors**.
  - Owner: run `actionlint` once locally if you want the full check.
- **Secrets missing by name**, each handled at runtime:
  - v9 `DASHBOARD_DISPATCH_TOKEN`: notice, then the hourly sync runs.
  - Pro `WOWZA_BETTING_PR_TOKEN`: the bankroll PR is skipped and the recommendation is committed.
  - exec `CLOUDBET_API_KEY`: the shadow DRY runs succeed without it; last 3 runs success.
- **Variables:**
  - Set today: v9 `PHASE2_RETRAIN_ENABLED=true` (R4 says delete after the refit); Pro `PRO_SPECIALS_ENABLED=true`.
  - **To set at go-live:**
    - v9 `PREMATCH_CAPTURE_ENABLED=true` (U3);
    - `TOP3_FREEZE_LEAD_MIN=120` and `TOP3_LEAGUE_ENABLED=true` (U3c);
    - `TOP3_PAPER_ENABLED=true` (U3f).
  - **Owner decision:** Pro `PRO_1X2_CLEAN_ENABLED` (`pro_train_1x2_clean.yml`, monthly). Its retrain is gated by the R6 rule.
  - **Leave unset:** `BARS_MONTHLY_ENABLED` (monthly re-proposal; owner decision).
- **Disabled, keep disabled** (STOPLIGHT, U4/U5):
  - v9: Backtest, the one-off DNP recheck, Player Props Train, Pre-season Retrain, Retrain;
  - Pro: Pro Research, the Standard Repair shadow.
- **Retire after go-live (they target data-truth-100):**
  - `data_truth_collect.yml` (its `syncmain` step);
  - `oddsapi_history_probe.yml`;
  - Pro `pro_specials_collect.yml` (U8: re-point to main).
- **Push-triggered writers** fire on the merge push only for their path filters. Freeze them per runbook 2.1, then dispatch them by hand (P4).

## 6. Fixes made (all pushed to source branches, each with tests)

| Branch | Commit | Fix |
|---|---|---|
| v9 data-truth-100 | `0a909c6ed` | `sync_results_from_main.py`: a second sync took nothing (fd_history / player_history counted as "both changed"). 208 matches and 148 player rows of main would have been dropped at go-live |
| v9 data-truth-100 | `82eadf5a4` | CLV finding test pinned to its own 170 rows, not to the sidecar length |
| v9 prep/bars-oddsapi-hist | `10f742029` | dt merged in; props checks = Data truth 10; unique-number test |
| v9 prep/bars-oddsapi-hist | `45d56dc2b` `41616a13b` `641220c53` `b9512083c` `b5219b6e6` | Cell columns in bets; PAPER line per cell; TOP-3 go-live gate, paper leagues and stake labels; stake only ledgered tips; `--set-fraction` / `--record-approval` |
| v9 prep/bars-oddsapi-hist | `42b5cd412` | Runbook: R6 gate 2.6b + rehearsal corrections 0c |
| v9 prep/bars-oddsapi-hist | `bf0f655b7` `f027d8508` | Data truth 7 dates a board by its predict run; rename after the shadowing slip |
| v9 dashboard/clean-record | `78c2a58c1` `9f777a565` `72bbf6922` `9f60b8b66` | Evidence file from the proposal; no new defaults; pages on the design system; dispatch path for the Clean Record config; Clean Record split on the full start instant |
| Pro data-truth-100 | `b12b4300` | Bankroll tests use v9's per-cell-off policy and v9's own tier factor |
| Rehearsal branches only | v9 `fb110f476`; Pro `743e7b44`; v11 `83a08f2` | Go-live-day merge steps (housekeeping, scout resolution, restate). Reuse them on Nov 21 |

## 7. R6 gate (owner 2026-11-15) - now runbook 2.6b

| Gate | Status today | Evidence |
|---|---|---|
| G1 all collected data wired, parity 0 | **RED** | `feat/team-info-features` is **not** merged into data-truth-100 (7 commits ahead). Parity on the currently served models: team 0 (7 served payloads, last 6 match dates 11-04..09: 221 fixtures, 9,503 cells, 0 mismatched, max 0.0 pp); props 0 (5 served markets, 50 fixtures, 1,411 player rows, 128 features: 0 mismatches, every probability identical) |
| G2 live collection produces every input in time | **AMBER** | Last runs: injury, player history, NF odds = success today; af_history_extend last success 11-14; `data_truth_collect` last 11-09 (collection ended); Data truth 3 RED (new-format `home_corners_pg_roll` 25% missing on the board) - an input the models read, so G2 cannot be GREEN until it is filled or explained |
| G3 rehearsal GREEN end to end | **RED** | R1-R4 above |

**If the gate is not met by 2026-11-19, the owner's options for Nov 21:**
- **A. Go live on Nov 21 with the CURRENT clean models (no R6 refit).**
  - Merge, bars and Clean Record as planned. The clean champion recipe installed 2026-11-09 serves.
  - R7 is not needed: no refit means the approved bars stand on the same predictions.
  - The refit runs later through PG-1.0's two periods.
  - Risk: the models miss the newest data (≈ 2 weeks) and the team-info features.
- **B. Slip go-live to the first weekday after the gate is GREEN** (e.g. Tue 11-24 or Mon 11-30).
  - Same plan.
  - The Clean Record starts later, which moves milestone/2's 2-month window.
- **C. Go live on Nov 21 PAPER-ONLY:** merge, but the 4 cells stay PAPER until R6 + R7 pass. Then switch the stakes on.
  - The Clean Record counts from the switch-on, or from Nov 21 with the stakes column starting later.

My recommendation: **A if only G1 is late** (wiring finishes after go-live and enters through the normal promotion gate); **B if R1-R3 are not closed by Nov 19.**

## 8. Command sequences

Shell prelude: runbook section "Shell prelude". `gh`/`git` paths as in the runbook.

### Nov 19-20: R6 gate, refit and R7 re-check

1. Gate 2.6b G1-G3 (runbook 2.6b, exact commands there). Record `$GL\r6_gate.txt`. **Stop on RED** and choose A/B/C (section 7).
2. R6 (runbook R6):
   - `$env:PHASE2_RETRAIN_ENABLED="true"; python scripts/phase2_refit_serving.py --out $GL\refit`
   - `python scripts/team_train_serve_parity.py $GL\parity $GL\refit\<12 pkl comma-separated> 6` must show 0 differing cells.
   - `python scripts/promote_clean_switch.py assemble --refit $GL\refit --out $GL\asm`
   - Put the 7 sha256 in the `final_refit` artifacts.
   - `python scripts/promote_clean_switch.py install --assembled $GL\asm --override clean-data-switch-2026-11-09`
   - Then registry, gate, lock, manifest, full suite.
3. R7 on the same commit:
   - `python scripts/phase2_retrain.py --out $GL\p2 --stage all --tracks <track>` (Standard via `phase2_retrain.yml`).
   - `python scripts/bars_opt.py universe --track standard --predictions <std>/predictions.parquet --fd-raw output/collect/fd_raw --certified-fd --aliases <identity map> --out $GL\u`, and the same with `--track new_format` without `--fd-raw`.
   - `python scripts/bars_opt.py select --universe $GL\u --out $GL\s`, then `final`.
   - `python scripts/bars_opt_oddsapi.py recheck --approved-evidence output/threshold_proposals/bars_oddsapi_hist/evidence_2026-11_v5.json --universe $GL\u --study-cells $GL\s/cells.json --month 2026-11`
   - Exit 0 = the bars stand. Exit 1 = back to the owner.
4. U3e on Nov 20 (UTC):
   - In `config/top3_paper_tracks.json`, **replace d999073a with bade2c0b in `approved_proposals`** and add the final frame to `clean_frames`.
   - `python scripts/top3_paper_freeze.py --dry-run --out $GL\freeze`, then `python scripts/top3_paper_freeze.py`.
   - Commit both files (explicit paths) before Nov 21.
5. `gh variable delete PHASE2_RETRAIN_ENABLED --repo NevixAA/wowza-betting` (R4).

### Nov 21: go-live (a quiet window, after the owner's actions)

0. **Owner first:**
   - Record the bade2c0b approval: all 63 entries by id + hash, `price_provenance` named, Argentina exemption with its basis.
   - Decide D16 and the Ireland score (R3).
1. Snapshot + freeze: `python scripts/golive/snapshot_state.py --out $GL\state_before.json`. Disable the active writers (runbook 2.1). Make sure **no predict run happens between the freeze and P0**.
2. v9 M1 on a branch `golive/m1` from `origin/main`:
   - `git checkout origin/data-truth-100`, `python scripts/sync_results_from_main.py`, commit the synced files by path.
   - Merge in order: data-truth-100 (synced), prep/bars-oddsapi-hist, research/top3-ranking, dashboard/clean-record.
   - Resolve as in section 1. Shortcut for the identical files: `git checkout origin/rehearsal/go-live-2026-11-21 -- <file>` where neither side changed since 11-15.
   - `git rm tests/test_resolver_hotfix_main.py tests/test_ht_fill_hotfix.py`
   - `python -m src.clv_corrections build`
   - U3a: `python -m src.research_lock --approve --by "Nevo (owner)" --note "2026-11 go-live: bade2c0b bars serving, per-cell staking"`
   - `python scripts/dashboard_manifest.py --write`
   - Full suite: 0 failed. `python -m src.integrity.data_truth --gate`: GREEN.
   - Open the PR (merge commit). Checks green, then merge.
3. U3b: `python -m src.threshold_apply --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11_bade2c0b.json --dry-run` (63), then without `--dry-run`. Commit the params, applied ledger and lock snapshot.
4. Stakes:
   - `python scripts/cell_staking_from_proposal.py --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11_bade2c0b.json --set-fraction "new_format|BTTS|Argentina Primera Division=0.005" --fraction-note "owner 2026-11-15: 0.5% each on the staked cells" --record-approval`
   - Check exactly 4 cells at 0.005; Japan/Austria NF Unders PAPER. Commit `config/bankroll_policy.json`.
5. M2 Pro:
   - Merge dt into main. Bot CSVs to main. Scout files from `origin/rehearsal/go-live-2026-11-21` (Pro).
   - Suite with V9_LOCAL = new v9 main: only `test_fixture_key_v2` fails.
   - **P1** fk2 re-migration (long, no time limit) before the Pro writers are unfrozen.
6. M3 exec: merge prep/exec-clean-rebuild (3 conflicts to the branch). U3d: `python scripts/exec_cells_from_v9.py` then `--write` (expect 48 PAPER cells). Suite, commit.
7. M4 v11: merge dt (0 conflicts). Run `scripts/v11_restate_p_market.py --append` (or let V11 Collect run it). Suite 26 passed.
8. M5: `gh workflow run sync_from_v9.yml -R NevixAA/wowza-dashboard`.
9. P0: record the merge time. U6: `config/trustworthy_performance.json` `clean_record_start_utc` = P0 instant (the split now uses the full instant), plus `PERFORMANCE_CUTOFF_DATE` / `PROPS_CLEAN_MODELS_LIVE_DATE`, and the milestone/2 due date.
10. P2-P4, then unfreeze (U1, Pro writers after P1).
11. Switch-ons:
    - `gh variable set PREMATCH_CAPTURE_ENABLED --body true`
    - `gh variable set TOP3_FREEZE_LEAD_MIN --body 120`
    - `gh variable set TOP3_LEAGUE_ENABLED --body true`
    - then, after `pytest tests/test_top3_paper.py tests/test_top3_ranking.py` on main, `gh variable set TOP3_PAPER_ENABLED --body true`
    - all `--repo NevixAA/wowza-betting`
12. U8: Pro `pro_specials_collect.yml` re-pointed to main. Retire the data-truth-100-targeted workflows.
13. Live validation: runbook section 4 (+1 h / +6 h / +24 h).

## 9. Open owner decisions

1. **R1:** record the bade2c0b approval (63 entries, `price_provenance`, Argentina exemption). Agents may not write owner approvals.
2. **R3:** which score is right for UCD v Drogheda 2026-11-07 (football-data 2-1, API-Football 1-2)? A provider check; the wrong one goes to a correction sidecar.
3. **D-T3 (R4):** with 0 of 35 cautious slopes above 0, the T3 key ranks by the bookmaker margin. Options:
   - keep T3 as registered (honest: "no calibrated edge yet");
   - rank by the point slope as research only;
   - show no TOP-3 ranking until slopes exist.
4. **R6 gate:** option A / B / C if it is not GREEN by Nov 19 (section 7).
5. **D16:** dt's HT-fill placement of the ambiguous alias, so that `test_ht_fill_hotfix.py` can go.
6. **TOP-3 first weekend:** with candidates first seen before the start now excluded, the first weekend's pool is small. Is that accepted, or do you want week 1 voided?
7. **Argentina cell:** `max_odds` 2.6 is carried from the old policy, while the v5 cell has no odds range. Keep 2.6?
8. **`PRO_1X2_CLEAN_ENABLED` and `BARS_MONTHLY_ENABLED`:** switch on after go-live, or not.
9. **Rescheduled fixtures:** key the direction guard on fixture id / kickoff rather than match date (found in the simulation).
