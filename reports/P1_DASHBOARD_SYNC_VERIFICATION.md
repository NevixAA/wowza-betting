# P1 item 12 - Dashboard sync verification (2026-11-22)

## HEAD SHAs recorded (main, at verification time)
| Repo | main HEAD | latest CI on main (24h) |
|---|---|---|
| NevixAA/wowza-betting (v9) | `61e869877be7bb3c2e152660fc1907d0335615a0` (19:26 UTC, auto predict) | Tests and gates success 19:20. 2 Player Props failures (19:01, 19:07), recovered 20:01. No red run on main at verification time |
| NevixAA/wowzaV9-Pro | `1a413050cecd5814794113c91b814a634543ae5b` at start; `4e96caab014f1140de590ad193753ffd8beea06c` after PR #18 | all green after the Specials (15:42) and Player V3 (17:34) failures, which were fixed by other changes (see the Pro report) |
| NevixAA/wowza_v11 | `c0d2961ea9a709ba64ef9350f5b4ecdd673c2218` at start; `85c956f09caa440eaed63746544aa83a6ba30d92` after PR #4 | V11 Collect 4/4 success; post-merge Tests success |
| NevixAA/wowza-dashboard | `b233402013ef03c6a13abcfc4ad0befddd541e48` (19:47:57 UTC, "Sync from v9 61e8698") | Sync from v9 6/6 success |
| NevixAA/wowza-exec (private, read via gh) | `82f6d7d9fcbd4300cdfc76565aa8b07adb997fbc` (main); prediction-ledger-v11 `ef5b5d1410cf` | 5 success, 3 skipped (Sharp paper capture is gated off) |

## Sync source
- `.github/workflows/sync_from_v9.yml` checks out **wowza-betting `main`** (fetch-depth 1), runs v9's
  `scripts/build_dashboard_repo.py`, and mirrors the result with `rsync --delete` (keeping .git/.github). It runs hourly at
  :17, plus on workflow_dispatch and on `repository_dispatch v9-main-updated`.
- `DASHBOARD_DISPATCH_TOKEN` is **not set**. The v9 "Dashboard sync dispatch" run logs the notice
  "no dispatch sent; wowza-dashboard syncs on its hourly schedule", so the code lag after a v9 merge can be up to about 1h.
- `deploy/BUILD_INFO.json` on dashboard main: v9_head_at_build `61e869877be7...` = v9 main HEAD; v9_commit_of_manifest
  `0b8f182918aa...` = the v9 manifest's own `v9_commit`; code_files 189; data_files_mirrored 125; schema_version 1;
  dashboard_build_id `7b5808fb52c09a07`; built 2026-11-22T19:47:57Z.
- Note: the sync commits only when code changes. When only the build stamps differ, BUILD_INFO keeps the older
  v9_head_at_build, so a lag in that field alone does not mean the code is stale.

## Byte-level verification (independent rebuild)
v9 main `61e86987` was rebuilt locally with `scripts/build_dashboard_repo.py` and compared with dashboard main `b2334020`
(LF-normalised SHA-256):
- File sets: 195 built vs 195 tracked. 0 only-in-build, 0 only-in-dashboard.
- Content: **0 differences** outside the two stamped files (BUILD_INFO.json, README.md).
- BUILD_INFO head, manifest commit, counts and schema version are equal.

## Manifest completeness
- 189 code files: all tracked in v9 main.
- 125 data files: 116 tracked. The 9 not in v9 main are all flagged `optional: true` in the manifest (8
  `models/registry/*/challenger.json` with no current challenger, plus `output/af_ht_history.parquet`). The runtime mirror
  records them as missing (404), never as fake data.
- `scripts/dashboard_manifest.py --check` on v9 main 61e86987: missing code [], missing data [], exit 0. The trace
  recorded 3 page exceptions on `pages/1_📊_Dashboard.py`: "column config cannot be serialized into JSON: Out of range
  float values" (an inf/NaN reaches a column config). This defect predates this work and is not fixed here; it is
  reported for the dashboard owner. Two admin-button subprocesses were blocked by design.

## Shared file hashes (git blob, main)
| File | v9 | Pro | v11 | dashboard |
|---|---|---|---|---|
| src/prediction_ledger.py | 66260847 | 66260847 | 66260847 | 66260847 |
| config/prediction_ledger_markets.json | 2a3da3c7 | 2a3da3c7 | 2a3da3c7 | (not shipped) |
| src/team_names.py | 1077c8be | (none) | 081e26e3 -> **1077c8be after PR #4** | 1077c8be |
| config/leagues.json (league registry) | d083a3af | read from v9 | read from v9 (src/league_registry.py) | d083a3af |
| src/league_registry.py | 7c14b0ba | - | own module (reads v9's JSON) | 7c14b0ba |
| side_odds parity files | scripts/side_odds_store.sh, src/side_odds_quotes.py | .github/scripts/side_odds_store.sh, src/scout/side_odds.py | - | not shipped | 

The side-odds files are different implementations per repo (v9 quote writer, Pro scout reader), not byte copies. Their
parity is covered by each repo's own tests (v9 `tests/test_side_odds_quotes.py`, Pro `tests/test_scout_side_odds.py`), and
both pass on main.

## Compatibility
- **League registry:** the dashboard ships v9's `config/leagues.json` and `src/league_registry.py` byte-identical. v11 and
  Pro read v9's file at run time. Ledger `league_slug` values in v9 and Pro all come from that registry (tested).
- **Model versions:** the dashboard runs v9's code at the synced SHA and reads v9 `main` data at run time (remote mirror,
  every 10 min at most), so model files and registry JSON come from v9 main. `remote_mirror.compatibility()` compares
  BUILD_INFO with the data-side manifest's schema_version and raises SHA skew (covered by `tests/test_dashboard_compat.py`).
- **Ledger schemas:** the 31-column ledger schema is identical in v9, Pro and v11 (P1_V11_RECONCILIATION.md).
- **Fixture identities:** 0 conflicts across v9, Pro and v11 on oa: fixture_refs (P1_V11_RECONCILIATION.md).
- **Data timestamps:** dashboard data is v9 main at fetch time, so there is no stored copy to go stale. v9 main last
  auto-commit was 19:26 UTC.

## Deployed Streamlit version
**UNVERIFIED.** The app at streamlit.app is login-protected. There is no direct evidence of which commit Streamlit Cloud is
serving. The repo it deploys from is verified as above. The owner can confirm on the System Health page, which shows
`dashboard_build_id` / `source_v9_sha`; it should read `7b5808fb52c09a07` / `61e86987...` or a later build.

## Root cause / defects
None found in the sync. The repo is an exact rebuild of v9 main. The only cross-repo drift found (v11's stale
team_names copy) was fixed in v11 PR #4.

## Files changed
v9: `tests/test_dashboard_sync_integrity.py` (new, 6 tests) and these three reports.

## Tests
`tests/test_dashboard_sync_integrity.py`: 6 passed locally with DASHBOARD_LOCAL = dashboard main checkout. The live-tree
comparison skips in CI, where no dashboard checkout exists. v9 gates run locally on 61e86987 + this change:
`data_truth --gate` STOPLIGHT GREEN; `research_lock --check` protected_tracks_modified=False; `build_track_registry
--check` in sync (100 entries); `dashboard_manifest --check` exit 0. The tree was clean afterwards apart from the new files.
No `--write-after-adds` was needed because no data file was added.

## Blockers
None. DASHBOARD_DISPATCH_TOKEN stays unset by design; adding it would bring the lag from about 1h to minutes and needs a
PAT from the owner.

## Production impact
None (a test and documentation only).

## Owner approval
Not required. Optional: set DASHBOARD_DISPATCH_TOKEN, and confirm the served build id on the System Health page.
