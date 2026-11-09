# Go-live runbook: data-truth-100 to main, all five repos

Status: **PLAN, for the owner to approve.** Written 2026-11-06. Nothing in this document has been
executed against GitHub. No branch was merged to main, no repository variable was set and no
workflow was enabled or disabled. The only real-repository action was the dry run in section 5,
which used throwaway clones that were never pushed and have been deleted.

Repos (GitHub `NevixAA/...`): `wowza-betting` (v9), `wowzaV9-Pro` (Pro), `wowza_v11` (v11),
`wowza-dashboard` (dashboard), `wowza-exec` (exec).

Tools added on this branch (`phase2/golive`). Both are read-only towards GitHub.

| Script | What it does |
|---|---|
| `scripts/golive/snapshot_state.py --out <file>` | Records each repo's `main` and `data-truth-100` SHAs, every workflow's id/path/state, the repository variables, and any queued or in-progress runs. Exits 1 if any run is queued or in progress. Its JSON is the record that every rollback and every unfreeze reads. |
| `scripts/golive/dryrun_merge.py --root <dir>` | Section 5: clones each repo into `<dir>`, merges `origin/data-truth-100` into `origin/main` without committing, and lists the conflicts and the commit counts. The clones cannot push (`remote set-url --push origin DISABLED`). `--cleanup` deletes them. |

Shell prelude for every PowerShell block below:

```powershell
$env:PATH="C:\Program Files\Git\cmd;$env:LOCALAPPDATA\Programs\gh\bin;$env:PATH"
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONDONTWRITEBYTECODE="1"
$py="$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$GL="C:\Users\nevo\dev\golive"          # working folder for the go-live day (state files, merge clones)
New-Item -ItemType Directory -Force $GL | Out-Null
```

---

## 0. Findings that change the plan

These come from the checks and the dry run made today. Each one is either a precondition that is
not met yet or a step the plan has to include.

1. **Only v9 has CI on `data-truth-100`.** In v9, "Tests and gates" (`tests`, `gates`) and "Data
   Truth stoplight" (`ratchet`) are green on `40ca0f05`. Pro, v11, exec and the dashboard have **no
   test workflow at all**, on any branch. For them, "green in CI" can only mean the local suite run
   in 1.1.
2. **On `a18bf544`, exec `data-truth-100` failed 2 of its own tests, before any merge.** It was fixed
   on the branch while this runbook was being written.
   - The failures were:
     - `test_cloudbet_feed.py::test_resolver_is_v9s_canonical_copy`: the vendored `src/team_names.py` was stale.
     - `test_data_truth_round3_exec.py::test_unrated_league_still_stakes_nothing`: the unrated-league gate did not run while `league_rating_sizing=false`.
   - They were identical on exec dt alone and on the merged result.
   - exec dt is now at `2cce54f`, after `ef5bfda` (vendor v9 `team_names.py`) and `3d1c3d2` ("unrated league = no stake whatever the sizing flag"). There, the merged result gives **149 passed, 0 failed**.
   - Without CI, this kind of break is invisible. That is why 1.1 runs the exec suite by hand.
3. **The dashboard's `data-truth-100` should not be merged.** The dashboard repo is generated. Its
   hourly `sync_from_v9.yml` rebuilds it from **v9 main** with v9's own generator and mirrors it
   with `--delete`. I built the slim tree with the merged v9 generator and compared it with the
   merged dashboard: **158 files differ** (12,570 insertions, 12,924 deletions). The dashboard
   `data-truth-100` is already stale against v9. The correct go-live action is to merge v9, then
   dispatch `Sync from v9`.
4. **The collection is not complete.** These are the dry-run plans on the committed data at v9 dt
   `40ca0f05` (no API call was made):
   - lineups: up to 43,370 calls left;
   - player re-fetch: 32,724 calls left, 0 done;
   - inside-box: 352 calls left (7,297 is the upper bound).

   Injuries, coaches and transfers are done (0 planned). The job's cap is 45,000 calls per day, so
   the remaining ~76,400 calls need at least 2 more run-days. That only just fits the 2026-11-08 end
   date.
5. **Data Truth milestone: 21 open issues, 0 closed.** By `docs/STOPLIGHT.md`, the stoplight is RED
   while it is open. The gate command itself (`data_truth --gate`) is GREEN on v9 dt today.
6. **None of the merged repos is free of conflicts except v11.** All conflicts resolve
   mechanically: section 5 gives the rule for each file. In v9 and Pro, the only files changed on
   both sides since the merge base are the conflicted ones. **No data file was text-merged
   silently.**
7. **On the merged tree, v9 fails 6 tests and Pro fails 1** (section 5). None of them is a code
   conflict:
   - two v9 test files exist on main only, and they pin interim hotfix behaviour;
   - two v9 tests pin counts of files that bots keep appending to;
   - one v9 test depends on test order;
   - Pro's failure is the stale fk2 warehouse.

   The v9 ones must be fixed before M1, or the PR's own `tests` check fails.
8. **Several workflows fire on a push to main, when their paths change.** The merge changes those
   paths. v9 has eight such workflows on dt:
   - two read-only checks: "Tests and gates" and "Data Truth stoplight";
   - "Dashboard sync dispatch";
   - five writers: Evidence refresh, Model integrity health, Exchange Odds Capture, Auto-onboard leagues, and Player Props Train (disabled).

   Pro has twelve writers. If the writers are frozen (2.1), they do not run on the merge push. P4
   dispatches them once, by hand, and their logs are the first evidence from the new code.

---

## 1. Preconditions

Every line must give the expected output before step 2 starts. Record the outputs in
`$GL\preconditions.txt`.

### 1.1 Every repo's data-truth-100 is green

```powershell
# v9: the CI checks on the dt head
gh api repos/NevixAA/wowza-betting/commits/data-truth-100/check-runs --jq '.check_runs[] | [.name, .conclusion] | @tsv'
```
Expected: `tests success`, `gates success`, `ratchet success`, `upgrades-wait skipped`. Today, on
`40ca0f05`, it shows exactly that.

```powershell
# Pro, v11, exec: no CI exists, so run each suite locally on a fresh dt checkout, with V9_LOCAL = v9 dt
$py C:\Users\nevo\dev\wtp2-golive\scripts\golive\dryrun_merge.py --root $GL\pre --repos v9,pro,v11,exec
# then in each clone: git checkout -q origin/data-truth-100, and
$env:V9_LOCAL="$GL\pre\wowza-betting"
foreach ($r in 'wowzaV9-Pro','wowza_v11','wowza-exec') { Set-Location "$GL\pre\$r"; & $py -m pytest -q -p no:cacheprovider tests 2>&1 | Select-Object -Last 2 }
```
Expected: `0 failed` in every repo. exec gave `2 failed, 113 passed, 1 xfailed` on `a18bf544`, and
the merged result is clean on `2cce54f` (finding 2). The v11 and Pro results are in section 5.

### 1.2 data_truth --gate is GREEN

```powershell
Set-Location C:\Users\nevo\dev\wowza-betting; git fetch -q origin; git checkout -q origin/data-truth-100
& $py -m src.integrity.data_truth --gate; "exit=$LASTEXITCODE"; git status --porcelain
```
Expected: five `GREEN Data truth 1..5` lines, then `STOPLIGHT GREEN`, `exit=0`, and an empty
`git status`.

Today on `40ca0f05`:
- DT1 0 unknown leagues;
- DT2 0 new / 0 known defaults over 324 files;
- DT3 features present on 10 served tracks;
- DT4 clean;
- DT5 105,999 bank rows reduced to 70,743, with 0 duplicate pairs.

**Re-run this after the final collection commit.** Data truth 3 is the check that can turn RED for
Standard tracks while the inside-box fill is incomplete (see decision D3).

### 1.3 The collection is complete

```powershell
Set-Location C:\Users\nevo\dev\wowza-betting; git fetch -q origin; git checkout -q origin/data-truth-100
foreach ($s in 'collect_lineups','collect_player_refetch','collect_injuries','collect_coaches','collect_transfers') { & $py scripts\$s.py --dry-run 2>&1 | Select-Object -Last 1 }
& $py scripts\collect_af_history_gap.py --insidebox --dry-run 2>&1 | Select-Object -Last 1
gh run list -R NevixAA/wowza-betting --workflow data_truth_collect.yml --limit 3
git status --porcelain     # must be empty: the dry runs make no call and write nothing
```

| Audit value (JSON key) | Expected at completion | Today (dt `40ca0f05`) |
|---|---|---|
| lineups `planned_fixture_calls_upper_bound` | 0 | 43,370 (75,747 fixtures journaled in `output/collect/lineups_fixtures.parquet`) |
| player re-fetch `planned_calls` | 0 | 32,724 (`journaled_fixtures` 0) |
| player re-fetch `output/collect/player_history_refetch_audit.json` → `run.budget_stop` | absent, with `stored_rows` > 0 | "job cap reached: 45,000 >= 45,000 calls today", `stored_rows` 0 |
| insidebox `planned_stats_calls_excluding_no_statistics_seasons` | 0 | 352 |
| injuries / coaches / transfers `planned_calls(_total)` | 0 | 0 / 0 / 0 |
| latest "Data Truth Collect" run | `completed success` | success (2026-11-06 07:06) |

After the re-fetch, also run the props harness dry run. It gives the real verified share for
decision D6:

```powershell
& $py -m player_model.props_harness --dry-run
```

### 1.4 The retrain decisions are made

```powershell
& $py -c "import json; c=json.load(open('config/phase2_retrain.json')); print(c['holdout'])"
& $py -c "import json; p=json.load(open('config/experiments/props_clean_retrain/plan.json')); print(p['owner_decisions'])"
```
Expected:
- team models: `holdout.approved_by` and `approved_at` are set, and decisions 1-7 of
  `P0_NEW_FORMAT_CLEAN_RETRAIN_PLAN.md` section 8 are recorded;
- props: none of the 5 `owner_decisions` is `null`.

Today both are unset. The retrain runs **after** go-live, so this is a precondition only for
section 2.7, not for the merge. Props tips stay research-only until it is done (D7).

### 1.5 The owner's approvals are recorded

One GitHub issue, `Go-live approval`, in `NevixAA/wowza-betting`, with one comment from the owner
per item of section 6 that is marked **needed before merge**. Check:

```powershell
gh issue list -R NevixAA/wowza-betting --search "Go-live approval in:title" --state all
gh issue view <n> -R NevixAA/wowza-betting --comments
```
Expected: every item in section 6 marked "needed before merge" has an owner answer.

### 1.6 State snapshot (the rollback record)

```powershell
& $py C:\Users\nevo\dev\wtp2-golive\scripts\golive\snapshot_state.py --out $GL\state_before.json
```
Expected: one line per repo, `running 0` everywhere, and exit 0. If a run is queued or in
progress, wait for it and run the snapshot again. `state_before.json.repos.<repo>.main` is the
**pre-merge SHA** for that repo. Also write the SHAs to text files:

```powershell
foreach ($r in 'wowza-betting','wowzaV9-Pro','wowza_v11','wowza-dashboard','wowza-exec') {
  gh api repos/NevixAA/$r/branches/main --jq .commit.sha | Set-Content -Encoding ascii "$GL\premerge_$r.sha" }
```

---

## 2. Order of operations

Total window: about 2 h of the owner's time. The busiest automated schedule is Predict
(every 15 min on Friday to Sunday), so pick a **Monday to Thursday morning, 08:00-10:00 UTC**:
- after the 07:00 Daily Summary;
- between Update Results runs (07:30, 09:30);
- with Predict every 30 min.

Collection must already have ended (2026-11-08 at the earliest), so the first possible day is
**Monday 2026-11-09**.

### 2.1 Freeze: pause every workflow that commits to main

All of the workflows below push to `main` (`git push` in the workflow). Disable the **active** ones,
take the list from `state_before.json`, and re-enable exactly that list in 2.6. A workflow that was
already disabled (STOPLIGHT table, the one-off DNP recheck, Pro's shadow and research) stays out of
the list.

```powershell
$S = Get-Content $GL\state_before.json | ConvertFrom-Json
$freeze = @()
foreach ($r in 'wowza-betting','wowzaV9-Pro','wowza_v11','wowza-dashboard','wowza-exec') {
  foreach ($w in $S.repos.$r.workflows) {
    $keep = $w.path -match 'tests\.yml|data_truth_gate\.yml|dashboard_sync_dispatch\.yml'   # read-only checks stay on
    if ($w.state -eq 'active' -and -not $keep) { $freeze += [pscustomobject]@{repo=$r; id=$w.id; name=$w.name} } } }
$freeze | Export-Csv -NoTypeInformation -Encoding ascii $GL\frozen.csv
foreach ($f in $freeze) { gh workflow disable $f.id -R "NevixAA/$($f.repo)" }
# verify: every frozen id is now disabled_manually, and no run is queued or running
& $py C:\Users\nevo\dev\wtp2-golive\scripts\golive\snapshot_state.py --out $GL\state_frozen.json
```
Expected: the `active` count drops to the read-only ones: in v9, "Tests and gates" and "Data Truth
stoplight", plus "Dashboard sync dispatch" once it exists on main. Every other repo shows 0 active,
and `running 0`.

**Rollback of 2.1:** `foreach ($f in Import-Csv $GL\frozen.csv) { gh workflow enable $f.id -R "NevixAA/$($f.repo)" }`.
Nothing else changes. A missed schedule is skipped, not queued, so the next tick runs normally.

**What the freeze costs:** odds captures (NF, Standard side-market, exchange, prop snapshot), Predict
and Pro Live Odds pause for the window. The closing lines for matches that kick off during the
window are not captured, and that cannot be backfilled. This is why the window avoids weekends and
match-heavy evenings.

### 2.2 Merge mechanics: PR with a merge commit (recommended)

**Recommendation: one PR per repo, from a `golive/merge` branch that already contains the resolved
merge, merged with "Create a merge commit".** Do not squash and do not rebase. Reasons:
- a PR to main runs v9's "Tests and gates" and "Data Truth stoplight" on the **exact merged tree**
  before main changes;
- a merge commit keeps both histories, including the bot data commits. `git revert -m 1 <merge>`
  is then a clean, single rollback;
- squash would rewrite 234 v9 dt commits into one. It would lose the provenance that the reports
  cite by SHA.

No branch is protected (`protected=false` on all five), so a direct `git push origin HEAD:main`
would also work. It is not recommended, because main would change before the full suite ran on
the merged tree.

Per repo (shown for v9; replace the repo name):

```powershell
$r='wowza-betting'; $d="$GL\m_$r"
git clone -q --reference C:\Users\nevo\dev\$r https://github.com/NevixAA/$r.git $d
git -C $d checkout -q -b golive/merge origin/main
git -C $d rev-parse HEAD                          # must equal (Get-Content $GL\premerge_$r.sha)
git -C $d -c core.autocrlf=false merge --no-ff --no-commit origin/data-truth-100
git -C $d diff --name-only --diff-filter=U        # must equal the section 5 list for this repo
# resolve exactly as section 5 says, for example:
#   git -C $d checkout --theirs -- <file> ; git -C $d add -- <file>
git -C $d ls-files -u                             # must be empty
git -C $d -c user.name="Nevo Hershkovitz" -c user.email="89246276+NevixAA@users.noreply.github.com" commit -F C:\Users\nevo\dev\golive_merge_msg_$r.txt
git -C $d push -u origin golive/merge
gh pr create -R NevixAA/$r --base main --head golive/merge --title "Go-live: merge data-truth-100" --body-file C:\Users\nevo\dev\golive_pr_$r.md
gh pr checks --watch -R NevixAA/$r golive/merge    # v9: tests + gates + ratchet must pass
gh pr merge -R NevixAA/$r golive/merge --merge     # merge commit; NOT --squash / --rebase
git -C $d fetch -q origin; git -C $d rev-parse origin/main | Set-Content -Encoding ascii "$GL\merge_$r.sha"
```
The commit message file ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
when an agent prepares it. Write it with `Set-Content -Encoding ascii`.

If main moved after the clone (it should not, because of the freeze), `gh pr merge` refuses or shows
the PR as conflicting. **Stop and find out which workflow pushed.** Do not re-resolve under time
pressure.

### 2.3 Merge order

| Step | Repo | Action | Why this position |
|---|---|---|---|
| M1 | **v9** `wowza-betting` | PR as in 2.2 (11 conflicts, section 5) | Every consumer reads v9 `main` (Pro, v11, exec raw URLs and CI checkouts, and the dashboard sync). Pro dt needs v9's registry LR-1.1+ (`v9_config.leagues_of_type` raises on LR-1.0). |
| M2 | **Pro** `wowzaV9-Pro` | PR as in 2.2 (6 conflicts), **straight after M1** | Pro's CI checks out v9's default branch. Between M1 and M2, old Pro code would run against new v9 if Pro were not frozen. |
| M3 | **exec** `wowza-exec` | PR as in 2.2 (3 conflicts on dt `2cce54f`; all resolve to dt) | Reads v9 main (registry, resolver copy). PAPER_ONLY, so no money risk. |
| M4 | **v11** `wowza_v11` | PR as in 2.2 (0 conflicts on `27acd24a`; dt has moved to `08d8f184` since, so re-run the dry run) | Reads v9 main raw files. |
| M5 | **dashboard** | **No merge.** `gh workflow run sync_from_v9.yml -R NevixAA/wowza-dashboard` (temporarily enable it if frozen: `gh workflow enable <id>`, run it, and leave it enabled, since it is in the unfreeze list anyway) | Finding 3: the generator from v9 main is the source of truth. Leave the dashboard `data-truth-100` branch unmerged, and delete it later. |

**Post-merge checks for each repo** (run each before the next merge):

| Repo | Check | Expected |
|---|---|---|
| v9 | `gh run list -R NevixAA/wowza-betting --branch main --limit 10` | "Tests and gates" and "Data Truth stoplight" on the merge SHA: `success` |
| v9 | In a fresh checkout of `origin/main`: `python -m src.integrity.data_truth --gate` | `STOPLIGHT GREEN`, exit 0 |
| v9 | `python scripts/dashboard_truth.py --root <checkout> --json $GL\dashtruth_main.json` | exit 0 (0 MISMATCH). It was 58/58 MATCH on dt, and 27 MISMATCH on main before the merge (DASHBOARD_TRUTH_VALIDATION). |
| v9 | `python scripts/build_track_registry.py --check`; `python scripts/dashboard_manifest.py --check`; `python -m src.research_lock --check` | all OK (the `gates` job runs these too) |
| v9 | the push-triggered runs that fire on merge (paths changed): Evidence refresh, Model integrity health (R2), Exchange Odds Capture, Auto-onboard leagues, Dashboard sync dispatch (new). Player Props Train also has a push trigger, but its workflow is disabled, so it does not run. | each `success`, or `skipped`. Read each log: they are the first runs of the new code on main. They are **frozen** in 2.1, so they fire only for the merge push if they were left on. Decide: leave them frozen (default), then dispatch each one by hand in 2.5. |
| Pro | `gh run list -R NevixAA/wowzaV9-Pro --branch main --limit 15` | Pro has no test workflow. Run the suite on a fresh `origin/main` with V9_LOCAL = fresh v9 `origin/main`. Expect exactly **1 failure**, `test_fixture_key_v2.py::test_stored_warehouse_is_v2_and_old_new_keys_agree` (section 5: the warehouse holds a mix of v1 and fk2 keys), until P1 has run. After P1, expect `0 failed`. |
| Pro | the paths in the Pro push triggers changed by dt (Monthly Bankroll Policy, Learned tiers, Player V3 Forward, Prediction Lab, Price Trader, Props learner, Research Index, Studies, VALUABLE shadow, Weather fetch, Weekly League Promotion, fd raw fetch) | frozen; dispatch by hand in 2.5 |
| exec | suite on `origin/main`, V9_LOCAL = v9 `origin/main` | `0 failed` |
| v11 | suite on `origin/main` | `0 failed` **after the first V11 Collect run**. See section 5: `test_restated_sidecar_covers_every_committed_snapshot` fails until `scripts/v11_restate_p_market.py --append` adds the snapshots main wrote since the branch point. The V11 Collect workflow runs that step on every run. |
| dashboard | `gh run list -R NevixAA/wowza-dashboard --limit 3`, then `git diff` between the new dashboard main and `build_dashboard_repo.py` output from v9 main | sync `success`, and no code difference apart from build stamps |

### 2.4 Rollback for each merge step

Before every merge, `$GL\premerge_<repo>.sha` holds the pre-merge SHA (1.6). After the merge,
`$GL\merge_<repo>.sha` holds the merge commit.

**Default rollback (any time): revert the merge commit.** History is kept, and so is anything that
bots pushed afterwards.

```powershell
$r='wowza-betting'; $d="$GL\rb_$r"
git clone -q --reference C:\Users\nevo\dev\$r https://github.com/NevixAA/$r.git $d
git -C $d checkout -q -b golive/rollback origin/main
git -C $d revert --no-edit -m 1 (Get-Content $GL\merge_$r.sha)
git -C $d diff --stat (Get-Content $GL\premerge_$r.sha) HEAD -- . ':(exclude)output'   # code must equal pre-merge: expect no output
git -C $d push origin golive/rollback; gh pr create -R NevixAA/$r --base main --head golive/rollback --title "Go-live rollback" --body "Revert of the go-live merge"
gh pr merge -R NevixAA/$r golive/rollback --merge
```
Re-merging later needs a revert of this revert first (`git revert <revert-sha>`), not a new merge
of `data-truth-100`. Otherwise git considers dt's commits already merged and brings nothing back.

**Hard reset (only inside the freeze, before any non-merge commit lands on main):**

```powershell
$pre = Get-Content $GL\premerge_$r.sha; $cur = Get-Content $GL\merge_$r.sha
git -C $d push --force-with-lease=main:$cur origin "${pre}:main"
```
`--force-with-lease` refuses if main is no longer exactly the merge commit, which happens when a bot
or a person has pushed since. In that case use the revert.

**Order of rollbacks:**
- the reverse of the merges: dashboard (re-dispatch the sync after v9 is rolled back), v11, exec, Pro, v9;
- Pro must not stay merged while v9 is rolled back: Pro dt raises on registry LR-1.0;
- v9 must not stay merged while Pro is rolled back, unless old Pro on new v9 has been tested. It has not been. **Roll back v9 and Pro together.**

#### Data written between the merge and a rollback

After the unfreeze (2.6), workflows commit on top of the merge. A **revert** keeps those commits. A
**hard reset** is not possible once they exist. What that means, by kind of file:

| Kind of file | Written after the merge by | After a revert |
|---|---|---|
| Append-only captures (`newformat_odds_history.csv`, `standard_sidemarket_odds_history.csv`, `book_odds_snapshots*`, `book_quotes/`, `prop_quotes/`, `player_prop_odds_history*`, `output/prematch/**`) | odds captures, prematch capture | **Kept, and correct.** They are real observations, independent of the code version. The `merge=union` attributes keep both sides on later rebases. Keep them. |
| Ledgers edited in place (`bets_ledger.csv`, `side_bets_ledger.csv`, `player_ledger.csv`, `ht_ledger.csv`) | Predict, Update Results, Player Props | Rows logged or settled by **new** code stay. The revert restores the **pre-merge versions of every file that the merge changed**. A ledger changed later by a bot conflicts with the revert, and git refuses until each one is resolved: the revert stops with conflicts. Rule for that case: keep the bot's rows (they are real tips and results), and record the rows settled under the null rule in a corrections note. Do not delete them. |
| `output/corrections/player_ledger_null_rule_settled.csv` | the first Update Results after the merge (about 1,971 rows) | Its content stays as evidence even if the code is rolled back. The settlement it records stays in `player_ledger.csv`, unless the owner decides otherwise. |
| Generated status files (`system_health.json`, `data_truth_health.json`, `stake_plan.csv`, `track_registry.json`, `workflow_health.json`) | Evidence refresh, Predict | They are regenerated by the next run of the rolled-back code. On a conflict, take the revert's version. |
| Pro warehouse (`data/season_2026_27/**`) re-keyed by the fk2 migration | 2.5 step P1 (manual) | Revert the migration commit **first**: it is its own commit. Rows collected after the migration carry fk2 keys, and old Pro reads v1. Revert the migration commit, then the merge, and do not run Pro collection in between. |
| Pro outputs (`paper_1x2.csv`, `bet_builder_*.csv`, `clv_enriched.csv`, studies) | Pro daily jobs | Kept. On a conflict, take the bot's version (live rows). |
| GitHub state (variables, workflow enable/disable) | 2.5, 2.6 | **Not in git.** Undo with the inverse `gh` command listed with each step. |

So a rollback **after** the unfreeze is a merge-conflict resolution under the table above, not a
single command. That is the main reason for the 24 h abort criteria in section 4. Roll back early or
roll forward with a fix.

### 2.5 Post-merge actions, in order (still frozen unless stated)

| Step | Action | Command | Needs | Rollback |
|---|---|---|---|---|
| P0 | Record the go-live time | `git -C $GL\m_wowza-betting log -1 --format=%cI (Get-Content $GL\merge_wowza-betting.sha)` | - | - |
| P1 | **Pro fk2 vocabulary 1.1 + re-migration** (Pro, on a branch, then PR) | 1. In Pro: set `VOCAB_VERSION="fk2-clubs/1.1"`, `VOCAB_PATH=.../fixture_key_v2_clubs_1.1.json` in `src/data/fixture_key_v2.py`. 2. `$env:V9_LOCAL=<fresh v9 origin/main>`; `python -m src.pipelines.fixture_key_v2_vocab`. 3. Pin the new sha256 in `fixture_key_v2.py`. 4. `python -m src.pipelines.migrate_fixture_key_v2 --season season_2026_27 --dry-run` and read `output/fixture_key_v2/migration_report.json`: `resolver.v9_commit` must be the merged v9 SHA (today it is `0704142d93f2`), with the counts of `unresolved_clubs.csv`, `unmapped_v1_keys.csv` and `grain_conflicts.csv`. 5. **Owner OK on those counts.** 6. Run without `--dry-run`, run the Pro suite, commit (one commit: `git add data/season_2026_27 output registry/identity src/data/fixture_key_v2.py` by explicit path), then PR and merge. | D1, D2 | `git revert -m 1` of the migration PR merge commit. Pro collection stays frozen until it is done. |
| P2 | First settlement run (null rule) | `gh workflow enable <Update Results id> -R NevixAA/wowza-betting`; `gh workflow run update_results.yml -R NevixAA/wowza-betting`; wait for it, then pull and read `output/corrections/player_ledger_null_rule_settled.csv` | - | revert its commit (the table in 2.4); `gh workflow disable` |
| P3 | Verify P2 | `python -c "import pandas as pd; d=pd.read_csv('output/corrections/player_ledger_null_rule_settled.csv',dtype=str); print(len(d)); print(d.iloc[:,:8].tail())"` | - | - |
| P4 | Dispatch each frozen push-triggered workflow once, by hand, and read its log: v9 Evidence refresh, Model integrity health, then Pro Studies, Research Index | `gh workflow enable <id>`; `gh workflow run <file> -R ...` | - | `gh workflow disable <id>` |
| P5 | Dashboard sync (M5) | `gh workflow run sync_from_v9.yml -R NevixAA/wowza-dashboard` | - | after a v9 rollback, run the same command again |

Expected for P3: about 1,971 rows. That is the owner's program figure. The row count measured on
merged main is the one to record, together with columns `before` = pending (blank) and the after
value. Before trusting it, sample 10 rows against API-Football with the owner's null rule:
- null means 0 only when the player played (minutes > 0) and the provider answered the fixture;
- otherwise the row stays pending/missing.

### 2.6 Unfreeze and switch-ons

| Step | Action | Command | Needs | Inverse |
|---|---|---|---|---|
| U1 | Re-enable exactly the frozen list. v9, v11, exec and the dashboard go first. **Pro's writers wait until P1 has finished**, because its runtime is unmeasured (section 5). | `foreach ($f in Import-Csv $GL\frozen.csv \| ? repo -ne 'wowzaV9-Pro') { gh workflow enable $f.id -R "NevixAA/$($f.repo)" }`; after P1, the same command with `-eq` | M1-M5 checks passed | `gh workflow disable` each id |
| U2 | State check | `snapshot_state.py --out $GL\state_after.json`. Compare the `state` per workflow id with `state_before.json`. The only expected differences are new workflows from dt: "Pre-kickoff capture", "Phase 2 retrain", "Props clean retrain" (already registered from the branch: "Tests and gates", "Data Truth Collect", "Dashboard sync dispatch"). | - | - |
| U3 | **Pre-kickoff capture on** | `gh variable set PREMATCH_CAPTURE_ENABLED --body true --repo NevixAA/wowza-betting` | collection ended (owner approval 2026-11-08 already given) | `gh variable delete PREMATCH_CAPTURE_ENABLED --repo NevixAA/wowza-betting` |
| U4 | **Re-enable the workflows the program disabled** (STOPLIGHT table) | v9: `retrain.yml`, `player_props_train.yml`, `backtest_matrix.yml`, `preseason_retrain.yml`; Pro: `pro_research.yml`, `pro_standard_repair_shadow.yml`. Command: `gh workflow enable <file> -R ...` | **D4, not automatic.** The STOPLIGHT rule allows it only when the milestone closes **and** after the clean retrain (no training on unclean data). `retrain.yml` and `player_props_train.yml` would retrain **production** models on their own schedule, outside the pre-registered Phase 2 harness. Recommendation: keep them disabled until the Phase 2 retrain verdicts are approved. | `gh workflow disable <file>` |
| U5 | Leave disabled | `oneoff_dnp_recheck.yml` (it rewrites results in place, which breaks the settlement rule), `Backtest` (last main run failed 2026-07-13; D9) | - | - |
| U6 | **Clean Record start** | A PR on v9: `config/trustworthy_performance.json` → `"clean_record_start_utc": "<P0 time, UTC>"`; `config.py` `PERFORMANCE_CUTOFF_DATE` (issue #22) → the same date; then `gh api -X PATCH repos/NevixAA/wowza-betting/milestones/2 -f due_on=<start + 2 months>T00:00:00Z` | D5 | revert the PR; PATCH `due_on` back to `2027-01-10T00:00:00Z` |
| U7 | Close the Data Truth 100% milestone | close each fixed issue with a reference to the merge SHA, then `gh api -X PATCH repos/NevixAA/wowza-betting/milestones/1 -f state=closed` | D8 (an issue not fixed must stay open, which keeps the stoplight RED) | `-f state=open` |

### 2.7 Retrains (after go-live, gated, owner-promoted)

| Step | Action | Command | Needs |
|---|---|---|---|
| R1 | Team models, clean retrain | Set `holdout.approved_by/approved_at` in `config/phase2_retrain.json` (PR). Then `gh variable set PHASE2_RETRAIN_ENABLED --body true --repo NevixAA/wowza-betting`, then `gh workflow run phase2_retrain.yml -f stage=walk_forward`, and later `-f stage=holdout` once. | D6, P0 plan section 8 |
| R2 | Props clean retrain | Fill `owner_decisions` in `config/experiments/props_clean_retrain/plan.json` (PR). Then `gh workflow run props_clean_retrain.yml`. | D6, D7, player re-fetch complete |
| R3 | Promotion | Read each run's artifact (verdict, calibration, per league). The owner decides promotion through the promotion gate (team) or the props gate (props). Nothing promotes automatically. | owner |
| R4 | Turn the gate off again | `gh variable delete PHASE2_RETRAIN_ENABLED --repo NevixAA/wowza-betting` | - |
| R6 | **FINAL pre-go-live team refit, 2026-11-19/20** (owner 2026-11-09: "Approve 2nd + final refit"; `config/promotion_overrides.json` override `clean-data-switch-2026-11-09`, stage `final_refit`, install window 2026-11-19..2026-11-20) | AFTER the last data-truth collection run and the last results sync from main have landed on data-truth-100: (1) `PHASE2_RETRAIN_ENABLED=true python scripts/phase2_refit_serving.py --out <dir>` (same recipe, champion_recipe per track × market, all clean data through the holdout end); (2) **GATE:** `python scripts/team_train_serve_parity.py <out> <all 12 refit pkl, comma-separated> 6` must report **0 differing cells and every probability identical**: if not, stop (fix the live builder, never the model); (3) `python scripts/promote_clean_switch.py assemble --refit <dir> --out <asm>`; (4) write the 7 sha256s from `<asm>/assembly.json` into the `final_refit` stage's `artifacts` (the owner approval already covers this one refit; nothing beyond it); (5) `python scripts/promote_clean_switch.py install --assembled <asm> --override clean-data-switch-2026-11-09` (the gate refuses outside the window); (6) track registry, gate, lock, manifest, full suite; merge. Every later team promotion goes through PG-1.0's two periods (the 1,000-row shadow rule). | Last collection + results sync landed; owner approval of 2026-11-09 |
| R5 | Restart the weekly per-league research clock (owner 2026-11-08: "recounting after the retrain fixed and data fixed") | After R3 promotes the retrained models and go-live has merged the repaired data, the per-league weekly panel (`src/research/league_weekly*.py`, TOP-1/3/5 per league × market × gameweek) counts ONLY rows whose `prediction_engine_version` is a promoted clean-retrain model and whose `first_seen_at` is on or after the restart date. Rows from the pre-repair engines (dated 2026-10-05..10-28; 820 rows over W43-W45, TOP-3: 272 priced, ROI −7.8%, CI95 [−21.5%, +2.5%]) stay in the file untouched and are never compared. TOP-3 per league becomes a named PAPER track in the Clean Record; its verdict needs the panel's own bar (≥ 30 priced settled rows and ≥ 8 gameweeks per league × market). | R3, go-live merge, owner OK on the restart date |

Both retrain workflows are `workflow_dispatch` only, `contents: read`, and commit nothing. Their
rollback is "do not promote".

**Props tips until R2/R3:** the props models in production were trained on placeholder profiles
(age 25 / 180 cm / 75 kg on all 876,130 rows), while v9 dt serves real profiles. After the merge,
**the production props model sees inputs it never saw in training.** Props are already PAPER /
BLOCKED_FROM_REAL in the bankroll policy, so no money is at risk. Still, the props tips on the board
and in Telegram are not trustworthy until R3. Decision D7: label them research-only, or hide them,
until then.

### 2.8 README update (after the 7-day check)

One PR per repo that rewrites the README sections changed by the program: data sources, the Staked
definition, settlement rules, the stoplight, and the Clean Record. Use the measured numbers of the
7-day check, and name no figure without its source. The dashboard README is generated: change v9's
generator, not the dashboard.

---

## 3. Rollback quick reference

| Step | Undo |
|---|---|
| 2.1 freeze | `gh workflow enable` each id in `frozen.csv` |
| M1-M4 merge | `git revert -m 1 <merge sha>` via PR (default); `push --force-with-lease=main:<merge> origin <premerge>:main` only if main is still exactly the merge commit |
| M5 dashboard sync | run `sync_from_v9.yml` again after v9 is rolled back |
| P1 fk2 re-migration | revert its merge commit **before** the Pro go-live revert |
| P2 first settlement | revert its commit and keep the corrections CSV as evidence (2.4 table) |
| U3 prematch | `gh variable delete PREMATCH_CAPTURE_ENABLED --repo NevixAA/wowza-betting` (captured files stay: real observations) |
| U4 re-enabled training workflows | `gh workflow disable <file>`, then **revert any model commit** that a scheduled retrain pushed |
| U6 Clean Record | revert the PR; PATCH the milestone due date back |
| U7 milestone close | PATCH `state=open` |
| R1/R2 retrain | `gh variable delete PHASE2_RETRAIN_ENABLED`; nothing to revert (no commits) |

---

## 4. Live validation

### First 24 h (check at +1 h, +6 h, +24 h)

| What | How | Pass |
|---|---|---|
| Workflows healthy | `gh run list -R NevixAA/<repo> --branch main --limit 40 --json name,conclusion,createdAt` for all five | no `failure` from a scheduled job; a failure is read the same hour |
| Board renders | open the live dashboard (Home, Tips, Overview, System Health); `python scripts/dashboard_truth.py --root <fresh main> --json` | pages load with no exception; reproducer exit 0 (0 MISMATCH); "Staked" counts follow `src/staked.py` (on main before the merge: 107 policy-staked bets against 815 tier rows shown) |
| Stake plan obeys the policy | on fresh main: `python -c "import pandas as pd, json; s=pd.read_csv('output/stake_plan.csv'); p=json.load(open('config/bankroll_policy.json')); print(s.groupby(['track_key','state'])['stake'].agg(['count','sum']))"` | every row with `stake > 0` is on an ACTIVE / ACTIVE_LOW track (NF BTTS, NF O/U 2.5 Over, NF Over 1.5 by the current policy); 0 stake on PAPER / BLOCKED tracks (NF O/U Under, all Standard, props); no fixture with opposite directions (owner direction rule) |
| Predict runs | Predict runs and its commit; `output/system_health.json` overall | not FAIL; Data truth 1-5 GREEN |
| Settlement corrections | `output/corrections/player_ledger_null_rule_settled.csv` row count (P3); new Update Results commits | about 1,971 rows on the first run, then only new rows; 10-row sample matches the provider |
| Data truth stoplight | "Data Truth stoplight" run on each code push to main; `python -m src.integrity.data_truth --gate` on fresh main | GREEN |
| Prematch capture | `output/prematch/slots/<YYYY-MM>/` new parquet per run; `output/prematch/af_budget.json` | calls ≤ 1,000/day; `minutes_before_kickoff > 0` on captured rows |
| API-Football budget | `output/af_usage_log`, API-Football Usage Monitor run | remaining quota never below 25,000 |
| Pro reads new v9 | Pro Collect `collect_health.json` | no `SchemaIncompatible` / `SourceUnavailable` for v9 sources |

### First 7 days (daily)

- The dashboard truth reproducer on main gives 0 MISMATCH every day. Keep the JSON per day in
  `$GL\dashtruth_<date>.json`.
- `system_health.json`: no RED. AMBER items are only the ones known at go-live: data contracts and
  the edge-shadow age.
- Data truth 3 on all tracks stays GREEN. If it shows RED on Standard, see D3.
- Settlement: every finished fixture of the window is settled within one Update Results day. 0 rows
  are rewritten in place outside a corrections file. `git log -p` on the ledgers shows only appends
  and result fills.
- The stake plan obeys the policy every day, with the same check as at 24 h.
- Prematch capture coverage: the share of registry fixtures with an `injuries_120` and a
  `lineups_45` slot. A miss is fine. A missing month directory is not.
- Pro after P1: there are no new v1 keys in `data/season_2026_27` partitions written after the
  migration (`fixture_key_version == 'fk2'`).
- At day 7, the Clean Record has started (`clean_record_start_utc` set). Weekly checks under
  milestone 2 are informational only.

### Abort criteria (roll back, section 2.4)

Roll back v9 and Pro together, plus exec and v11, if any of these happens in the first 24 h, or
in the first 7 days without a fix within the same day:
1. Any real-money stake (`stake > 0` in the stake plan, or in Cloudbet/exec beyond PAPER_ONLY) on a
   PAPER or BLOCKED track, or two opposite positions on one fixture.
2. The dashboard truth reproducer reports a MISMATCH on a money figure (Staked, P/L, ROI) that is
   not explained by a newly settled row.
3. A ledger row rewritten in place without a corrections entry, or a settled row lost (the row count
   of a ledger goes down).
4. `data_truth --gate` RED on main from a code cause: naming, defaults, league relations or the
   ratchet. A RED on rule 3 (missing data) from a provider outage is an operational incident, not
   a rollback.
5. Predict or Update Results fails 2 runs in a row on new code.
6. API-Football remaining quota drops below 25,000 because of the new jobs.

Prefer rolling forward (a fix PR) to a rollback after the first bot commit lands, unless criterion 1
applies. Criterion 1 always means an immediate rollback or switching the stake path off.

---

## 5. Dry run of the merge mechanics (2026-11-06)

`scripts/golive/dryrun_merge.py --root C:\Users\nevo\dev\golive-dry`. It used fresh clones with
refs fetched from GitHub, `git merge --no-ff --no-commit origin/data-truth-100` on `origin/main`,
the resolution below, and a local commit. **Nothing was pushed** (push URL disabled), and the
clones were deleted afterwards. Main moves with every bot commit, so these counts are a snapshot of
the SHAs below.

| Repo | origin/main | origin/data-truth-100 | merge base | main-only commits | dt-only commits | conflicts |
|---|---|---|---|---|---|---|
| v9 | `e9b35b8d` | `40ca0f05` | `a62082b6` | 241 (236 bot, 5 Nevo hotfixes) | 234 | 11 |
| Pro | `904a8b2f` | `ba484041` | `d8816d20` | 170 (169 bot, 1 wowza-bot) | 23 | 6 |
| v11 | `5dbf91a1` | `27acd24a` | `ba1bbae1` | 12 (bot) | 9 | 0 |
| dashboard | `b0dd507f` | `ff53514e` | `4811bdf3` | 4 (3 sync bot, 1 Nevo) | 2 | 5 |
| exec | `8377d5ce` | `a18bf544` | `56670768` | 1 (Nevo hotfix) | 10 | 2 |

In v9 and Pro, the set of files changed on **both** sides since the merge base is exactly the
conflicted set, so no data file was text-merged silently. v11 has no file changed on both sides.

### v9: 11 conflicts, all resolved to the data-truth-100 side

| File | Type | Main side | Resolution and reason |
|---|---|---|---|
| `src/team_names.py` | both modified | hotfix `48887b83` (5 aliases + `_KNOWN_DIFFERENT`, a partial port of dt) | **dt.** dt has the full resolver: guessing tiers removed, plus the fixture-proven alias store. The hotfix message itself says it is a subset of dt. |
| `src/data_loader.py` | both modified (6 hunks) | hotfix `cc3f7ecf` (HT fill on main) | **dt.** dt carries the HT fill wiring plus fix9/bank (provider values served, write-back values excluded). |
| `src/cloudbet/bot.py` | both modified (5 hunks) | hotfix `062cb2e8` (no stake on a paper track) | **dt.** A superset: same paper-track drop, plus no default bankroll and no stake without per-league evidence. |
| `telegram_bot/notifier.py` | both modified | `062cb2e8` | **dt** (`positive_fraction`, the same rule) |
| `dashboard_data/tips_view.py` | both modified | `062cb2e8` | **dt** (adds the "Bar" column) |
| `player_model/predict.py` | both modified | hotfix `13cd81e1` (no invented props price) | **dt** (the same rule with its data-truth note) |
| `tests/test_p2_nf_under_paper_e2e.py` | both added | `062cb2e8` (`raising=False` because main lacked the gate) | **dt** (the gate exists after the merge) |
| `config/track_registry.json` | both modified | bot evidence refresh (approved_at 2026-11-02) | **dt** (approved_at 2026-11-04, direction rule). `build_track_registry.py --check` runs in the `gates` job. |
| `output/af_history.parquet` | binary | daily af_history extend: main 91,712 rows | **dt**: 93,389 rows. By (league, date, home, away), **0 main rows are missing from dt**, and dt adds 1,677 rows plus `fixture_id`, `home_team_id` and `away_team_id`. The daily extend adds newer matches after the merge. |
| `output/collect/fd_history_ht_fill.parquet` | both added | hotfix: 33,259 rows | **dt**: 33,992 rows, plus `provider_ft_*` / `ft_check` (owner: store FT in the HT fill) |
| `output/collect/ht_fill_audit.json` | both added | hotfix | **dt** (it goes with the parquet above) |

Commands: `git checkout --theirs -- <the 11 paths>; git add -- <the 11 paths>`.

### Pro: 6 conflicts, all resolved to the main side, then the fk2 re-migration (P1)

`output/bet_builder_candidates.csv`, `output/bet_builder_daily.csv`, `output/bet_builder_settled.csv`,
`output/clv_enriched.csv`, `output/paper_1x2.csv`, `output/studies/ou_challenger_forward.csv`.
Both sides modified each file:
- main: bot runs appended live rows (2026-11-03 to 2026-11-06);
- dt: the one fk2 migration commit `69eac7d2` re-keyed the old rows.

Resolution: **main** (`git checkout --ours`). It keeps every live row. The re-run of the migration
in P1 re-keys them all again, because the migration is re-runnable and reads the v1 key from
`old_fixture_key` where it exists.

The same applies to the warehouse without a conflict:
- main's 409 new `data/season_2026_27` partition files carry v1 keys;
- dt's migrated partitions carry fk2.

After the merge the warehouse is **mixed v1/fk2 until P1 runs**. P1 must run before Pro collection
is unfrozen.

I started `python -m src.pipelines.migrate_fixture_key_v2 --season season_2026_27 --dry-run` on the
merged Pro clone, with V9_LOCAL = the merged v9 clone. **After about 45 minutes on the laptop it had
printed nothing, so I stopped it.** Its counts and its runtime are therefore **not measured**.
Plan P1 as a long step with no time limit: run it on a quiet machine, or as a dispatch-only Pro
workflow. Pro collection stays frozen until it has finished.

### exec: 2 conflicts on `a18bf544`, 3 on `2cce54f`, all resolved to dt

On `a18bf544` there were two conflicts:
- `registry/execution_policy.json`: only the `_league_rating_sizing` note text differs; the flag is `false` on both sides.
- `tests/test_pipeline.py`: dt's "unrated league stakes nothing" replaces main's "stakes the minimum".

dt follows the owner rule (no invented rating). A second run on dt `2cce54f` (main `8377d5ce`, 1 main-only and
13 dt-only commits) adds `tests/test_ops6_rating_sizing_off.py` (both added: main hotfix `8377d5ce`
vs dt's longer version). Resolving all 3 to dt gives **149 passed**.

### dashboard: 5 conflicts, but do not merge (finding 3)

`README.md`, `deploy/BUILD_INFO.json`, `src/data_loader.py` (both added), `src/feature_engineering.py`
and `src/team_names.py`. All of them are generated copies.

The merged tree also differs from `build_dashboard_repo.py` output of the merged v9 in 158 files.
The sync rebuilds the dashboard from v9 main, so step M5 replaces the merge.

### v11: 0 conflicts

v11 `data-truth-100` moved to `08d8f184` during this work (`align/v9-dt`, vendoring v9's
`team_names.py`). A second run on `08d8f184` (12 main-only, 11 dt-only) also gave **0 conflicts**.
After the restate `--append`, the suite gives **26 passed**.

### Test suites on the merged result

| Repo | Command | Result |
|---|---|---|
| v9 | `python -m pytest -q -p no:cacheprovider tests` | **6 failed, 2,798 passed, 36 skipped, 7 xfailed** (11 min 18 s). The tree was clean afterwards. The dt head alone is green in CI. Section 5.1 breaks the 6 failures down. |
| Pro | same, with `V9_LOCAL` = the merged v9 clone | **1 failed, 472 passed** (114 s). The failure is `test_fixture_key_v2.py::test_stored_warehouse_is_v2_and_old_new_keys_agree`, with `ArrowInvalid: No match for FieldRef.Name(old_fixture_key)`: main's new v1 partitions do not have the fk2 columns. Pro dt alone: `tests/test_fixture_key_v2.py` gives 19 passed. **The failure comes from the merge, and P1 (the re-migration) is what fixes it.** That makes P1 a go-live step, not an option. |
| exec | same, with `V9_LOCAL` = the merged v9 clone | On dt `a18bf544`: **2 failed, 113 passed, 1 xfailed.** Both failures are identical on exec dt alone (finding 2), so the merge did not cause them. On dt `2cce54f`: **149 passed**. |
| v11 | same | **1 failed, 25 passed.** The failure is `test_restated_sidecar_covers_every_committed_snapshot`: main's V11 Collect wrote 1,815 snapshots with no restated row. After `python scripts/v11_restate_p_market.py --v9 <merged v9> --append` (the step V11 Collect runs every time), it gives 1,815 rows added and 83,173 kept verbatim, and the suite is **26 passed**. v11 dt alone: 26 passed. |
| dashboard | no test suite | compared with the generator instead (158 files differ, see above) |

### 5.1 The 6 v9 failures on the merged tree, and what to do before M1

| Test | Cause | Fix in the go-live merge commit (or on dt before it) |
|---|---|---|
| `test_resolver_hotfix_main.py::test_main_tiers_unchanged_for_real_matches[Stockport ...]` and `[Cottbus ...]` | **This test file exists on main only.** It came with hotfix `48887b83` and pins main's interim resolver tiers. The hotfix commit itself names Stockport and Cottbus as matches that dt resolves through its identity merge, not through the tiers. | `git rm tests/test_resolver_hotfix_main.py` in the merge commit. dt's resolver tests supersede it. |
| `test_ht_fill_hotfix.py::test_ambiguous_alias_stays_unplaced` | **This test file exists on main only** (hotfix `cc3f7ecf`). It pins main's interim HT-fill placement. dt's loader placed the alias (`ht_home_goals` = 1 where the test expects missing). | `git rm tests/test_ht_fill_hotfix.py` in the merge commit, **after** the owner confirms that dt's placement of that alias is proven and that dt's HT tests cover the rule (D16). |
| `test_fix9_defaults.py::test_loader_corrections_remove_them_from_the_real_history` | It pins `len(fd_history) == 65,940` ("the original is never rewritten"). Main's Update Results appended matches, so after the merge it has 66,213 rows. That growth is legitimate. | A fix on dt before M1: assert that the original rows are unchanged, not the exact count. |
| `test_round2_sharp_clv.py::test_committed_clv_sidecar_matches_the_finding` | It pins the mean stored CLV at 7.94 on dt's ledger snapshot. Main's ledger has more rows (1,347 with a value), so the mean is 7.75. | A fix on dt: pin the finding to the rows the sidecar lists, not to the live ledger. |
| `test_shadow_settlement.py::test_unsettled_over_row_becomes_loss` | It fails only in the full run. Run alone, the file gives **42 passed**. It depends on test order (state leaks from an earlier test). | Find the leaking test on dt. CI on the PR shows whether it reproduces on Linux. |

Expected v9 PR result once those fixes are in: `0 failed`. Until then, **the v9 PR's `tests` job
fails, so M1 cannot pass its own gate.** These fixes are a precondition (1.1). The two pinned-count
tests also show a general rule: a test must not pin a count of a file that bots keep appending to.

---

## 6. Open owner decisions

Before merge (blocking):

- [ ] **D1 Go-live date and window.** Proposed: Monday 2026-11-09, 08:00-10:00 UTC, the first weekday after collection ends. The freeze loses the closing-line captures for that window.
- [ ] **D2 Pro fk2 re-migration (P1).**
  - Approve a new vocabulary version `fk2-clubs/1.1` (rebuilt from merged v9 main).
  - Approve the warehouse rewrite of `data/season_2026_27` once the dry-run counts are shown (unresolved clubs, unmapped v1 keys, grain conflicts).
  - Yes / no, with the counts in hand.
- [ ] **D3 Data truth 3 and the inside-box fill.** If the final gate (1.2) shows RED on Standard tracks, either wait for the inside-box fill (352 calls left today, up to 7,297) or go live with the Standard tracks marked RED. They are all PAPER, so no stake depends on them.
- [ ] **D8 Data Truth 100% milestone.** It has 21 open issues. Close the ones that the merge fixes (list them per issue, with the merge SHA). Decide whether go-live waits for every issue, or whether the stoplight stays RED for the ones left open (no upgrades while RED).
- [ ] **D10 CI for Pro, v11 and exec.** None of them has a test workflow, which is how the exec break (finding 2) went unseen. Decide whether to add a `tests.yml` like v9's before go-live, or to rely on the manual runs in 1.1 and 2.3.
- [ ] **D16 v9 merged-tree test failures (5.1).** Approve:
  - deleting the two main-only hotfix test files in the merge commit, after confirming the HT alias placement;
  - the dt fixes for the two pinned-count tests and the order-dependent test, before M1.
- [ ] **D11 Merge method.** PR + merge commit (recommended) or direct push.
- [ ] **D12 Push-triggered workflows on the merge.** Keep them frozen and dispatch by hand in P4 (recommended), or let them fire on the merge push.

After merge:

- [ ] **D4 Re-enabling the disabled workflows (STOPLIGHT table).** Recommendation:
  - keep `retrain.yml`, `player_props_train.yml`, `preseason_retrain.yml` and `backtest_matrix.yml` disabled until the Phase 2 retrain verdicts are approved;
  - re-enable `pro_research.yml` and `pro_standard_repair_shadow.yml` with the owner's yes;
  - keep `oneoff_dnp_recheck.yml` off for good.
- [ ] **D5 Clean Record start.** It is the UTC time of the v9 merge commit (P0). Set the milestone due date to start + 2 months, and set `PERFORMANCE_CUTOFF_DATE` (issue #22) to the same day. Confirm `owner_confirmed` and the minimums in `config/trustworthy_performance.json` (30 settled, 8 matchdays, CLV 150, all copied from existing rules).
- [ ] **D6 Retrain decisions.**
  - Team models: decisions 1-7 of `P0_NEW_FORMAT_CLEAN_RETRAIN_PLAN.md` section 8 (holdout 2026-07-01..2026-11-08 proposed, thresholds PG-1.0, leagues ≥150, the NF HT block, context features, prices, walk-forward sizes).
  - Props: the 5 `owner_decisions` in `P0_PROPS_CLEAN_RETRAIN_PLAN.md` section 6 (holdout 2026-09-01..2026-10-31 proposed, the minimum verified share, markets, the feature rule, ...).
  - Then `PHASE2_RETRAIN_ENABLED=true`.
- [ ] **D7 Props tips until the props retrain is promoted.** Mark them research-only on the board and in Telegram (recommended), or hide them. Background: the train/serve gap, where the models were trained on placeholder profiles and are served real ones.
- [ ] **D9 "Backtest" workflow.** It is active, and its last main run failed on 2026-07-13. Disable it or fix it. The no-backtest-on-unclean-data rule applies until the retrain.
- [ ] **D13 First null-rule settlement.** Accept the about 1,971 re-graded rows after the 10-row sample check (P3), or hold Update Results until the check is done.
- [ ] **D14 README update** for every repo after the 7-day check (2.8).
- [ ] **D15 Delete the stale branches** after 7 clean days: `data-truth-100` in all repos, and the dashboard dt (never merged).
