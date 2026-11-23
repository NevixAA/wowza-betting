# P0 item 3 — join integrity and lineage of the active models

Owner P0 deep-repair list, 2026-11-22, item 3. Repo NevixAA/wowza-betting, branch `p0-health-join-lineage`
(based on origin/main `2a86ab87a824d37df2ac1cd032438af91135552b`, origin/main merged in before the PR).
HEAD SHAs at start (Stage A): wowza-betting `2a86ab87a824`, wowzaV9-Pro `eacaeec28bc3`, wowza_v11 `c0d2961ea9a7`,
wowza-dashboard `e2e96f0a8f5c`, wowza-exec `82f6d7d9fcbd`.

## Finding (reproduced on main before any change)

`output/join_integrity_health.json` (generated 2026-11-22T16:26:08Z): `status: FAIL`.

| join report | mode | status | left rows | output rows | right duplicate key groups | legacy expansion | legacy misaligned |
|---|---|---|---|---|---|---|---|
| af_history_shots | safe | PASS | 80,532 | 80,532 | 0 | - | - |
| fd_history_ht_fill | safe | PASS | 80,532 | 80,532 | 1,920 (collapsed, 0 ambiguous) | - | - |
| af_stats_fill | safe | PASS | 80,532 | 80,532 | 1 | - | - |
| af_fullstats_fill | safe | PASS | 80,532 | 80,532 | 1 | - | - |
| af_restored_rows | safe | PASS | 80,532 | 80,532 | 1 | - | - |
| std_sidemarket_odds | safe | PASS | 80,532 | 80,532 | 0 | - | - |
| ou25_captures | safe | PASS | 80,532 | 80,532 | 0 | - | - |
| **fd_history_ht_fill.legacy** | legacy_audit | **FAIL** | 80,532 | 82,452 | 1,920 | **1,920** | **64,950** |
| af_history_shots.legacy / std_sidemarket_odds.legacy / ou25_captures.legacy | legacy_audit | PASS | 80,532 | 80,532 | 0 | 0 | 0 |

All 7 serving (`safe`) joins pass. The only FAIL is the legacy replay. `src/integrity/model_health.py` copied the
artifact's aggregate `status` into every team track (`join_integrity: FAIL` on all 12 tracks).
Before-state copies: `C:\Users\nevo\dev\p0-evidence\before\` (join_integrity_health.json, model_integrity_health.json,
reproduction.txt).

## Root cause

`src/integrity/joins.write_join_health` takes the worst status of EVERY report. `audit_legacy_merge` (called in
`src/data_loader._enrich_ht_scores` as `_IJ.audit_legacy_merge(df, ht, _k3, name=f"{name}.legacy")`) is a
counterfactual measurement: it replays the league-free `(home, away, date)` merge + positional copy-back that was
removed on 2026-11-01 and reports what it would do. Its return value is discarded; the frame is filled by the
next statement, `safe_left_enrich(... keys=["league"] + _k3 ...)`. The HT sidecar holds 1,920 duplicated
`(league, home, away, date)` groups whose values agree; the safe join collapses them (0 ambiguous cells, row count
unchanged), the legacy replay would expand by 1,920 rows and shift 64,950. That is historical evidence about the
removed code, not a defect in the active pipeline. model_health could not tell the two apart.

## Lineage of every ACTIVE model

Evidence: `reports/lineage/training_lineage_evidence.json` (scripts/verify_training_lineage.py, 2026-11-22T17:32:47Z).
Each part checked: (a) training code at the payload's own `code_sha` (`git show`), (b) the training rows, (c) the
link payload sha256 = install record = registry, frame hash = refit's recorded hash.

| payload (registry sha) | part | code | frame | rows (incl. holdout tail) | to train end | payload training_rows | dup fixture keys | club twice / day | other-track / unregistered leagues | leagues | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model_v9_standard.pkl (67ee3666) | standard/over25 | 047442d95 | ebe47c041a7f595f | 49,597 | 49,407 | 49,407 | 0 | 0 | 0/0 | 17 | PASS |
| model_v9_newformat.pkl (303c9a05) | new_format/over25 | 694995f90 | 9ed237ebc04aee09 | 26,965 | 26,846 | 26,846 | 0 | 0 | 0/0 | 15 | PASS |
| model_v9_btts.pkl (44c67049) | standard/btts | 047442d95 | ebe47c041a7f595f | 49,597 | 49,407 | 49,407 | 0 | 0 | 0/0 | 17 | PASS |
| | new_format/btts | 694995f90 | 9ed237ebc04aee09 | 26,965 | 26,846 | 26,846 | 0 | 0 | 0/0 | 15 | PASS |
| model_v9_over15.pkl (ca01b7d3) | standard/over15 | 047442d95 | ebe47c041a7f595f | 49,597 | 49,407 | 49,407 | 0 | 0 | 0/0 | 17 | PASS |
| | new_format/over15 | 694995f90 | 9ed237ebc04aee09 | 26,965 | 26,846 | 26,846 | 0 | 0 | 0/0 | 15 | PASS |
| model_v9_over35.pkl (e8b12d74) | standard/over35 | 047442d95 | ebe47c041a7f595f | 49,597 | 49,407 | 49,407 | 0 | 0 | 0/0 | 17 | PASS |
| | new_format/over35 | 694995f90 | 9ed237ebc04aee09 | 26,965 | 26,846 | 26,846 | 0 | 0 | 0/0 | 15 | PASS |
| model_ht_over05.pkl (5a76d680) | standard/ht_over05 | 047442d95 | ebe47c041a7f595f | 49,446 | 49,256 | 49,256 | 0 | 0 | 0/0 | 17 | PASS |
| model_ht_over15.pkl (e9a12ce1) | standard/ht_over15 | 047442d95 | ebe47c041a7f595f | 49,446 | 49,256 | 49,256 | 0 | 0 | 0/0 | 17 | PASS |
| model_v9_ht_over05_nf.pkl (ae89ea10) | new_format/ht_over05 | 694995f90 | 9ed237ebc04aee09 | 26,786 | 26,759 | 26,759 | 0 | 0 | 0/0 | 15 | PASS |
| model_v9_ht_over15_nf.pkl (ead4bb74) | new_format/ht_over15 | 694995f90 | 9ed237ebc04aee09 | 26,786 | 26,759 | 26,759 | 0 | 0 | 0/0 | 15 | PASS |

Sources of the rows: Standard parts = `protocol.eligible_rows` on the saved Standard training frame
(`golive/std_retrain_1611/feat_new.parquet`, recomputed hash ebe47c041a7f595f = install record PR #62 / assembly.json);
New-Format parts = training keys saved by the 2026-11-18 refit (`golive/rows_refit_1118b/keys.parquet`, hash
9ed237ebc04aee09 = refit_1118b manifest). Row counts to the training end equal every payload's `training_rows` exactly.
Leagues are checked against canonical registry names only (aliases not accepted).

Code at both training commits (src/data_loader.py, src/integrity/joins.py, src/feature_engineering.py,
src/retrain_harness/protocol.py), comments and docstrings excluded:
- 0 positional zip copy-backs, 0 reads of the removed `WOWZA_SAFE_ENRICH_JOINS` switch, 0 `require_identity=False`,
  every `audit_legacy_merge` call is a bare statement (its result is never a frame);
- 10 `safe_left_enrich` calls in the loader; joins.py carries all four guards (many-to-one validate, row-count
  JoinIntegrityError, 1:1 write-back check, refusal of keys without league / fixture_id);
- joins.py is byte-identical at 694995f90, 047442d95 and origin/main (unchanged since 1c8d6fa79). At 047442d95 the
  four files are identical to origin/main; the only training-code difference to main is a serving-file write guard
  in src/model.py. Other merges in the training path (feature_engineering `_src_idx`, PR #59 fd_raw provenance
  `drop_duplicates(k)` lookups, src/fd_prematch one-to-one join) are keyed on unique ids / de-duplicated keys.

Point in time (Standard frame, 76,792 home / 76,807 away rows): `home/away_scored_last5` equals the (league, club)
mean of the previous <= 5 matches on 100.0% of rows and the window including the match itself on 26.1% / 26.9%
(the share where both windows happen to agree). Rolling form is `shift(1)` per (league, club); the holdout tail is
after the training end on every part (rows after train end 27-190, never fitted).

Props champions (9 payloads, run 43035218594, code 6fd794964): code audit of player_model/props_harness.py and
player_model/feature_engineering.py PASS; joins keyed on fixture_id / player_id / league; the team enrichment joins
(src.data_loader, safe_left_enrich) are not imported. Row-level props frame NOT re-run here (stated in the evidence).

## Fix

| file | change |
|---|---|
| src/integrity/training_lineage.py (new) | `join_scopes` (active serving vs legacy replay; a safe join that changed the row count is an active FAIL), `code_audit`, `frame_audit`, `pit_audit`, `evaluate` (active payload PASS only when the committed evidence names its CURRENT sha256 and recorded PASS; otherwise UNVERIFIED) |
| scripts/verify_training_lineage.py (new) | writes the evidence from the training commits + saved frames; read-only |
| reports/lineage/training_lineage_evidence.json (new) | the evidence above |
| src/integrity/joins.py | `write_join_health` adds `active_serving_join_status`, `legacy_replay_status`, per-scope lists, `status_scope`; `status` keeps its old meaning (worst of all, legacy included) |
| src/integrity/model_health.py | track `join_integrity` = active serving joins; new `training_lineage` per track; top-level `active_serving_join_status`, `active_training_lineage_status`, `legacy_replay_status`, `overall_model_integrity_status`, `failing_tracks`; UNKNOWN / UNVERIFIED can no longer rank as PASS |

## After (local run on this branch, same committed inputs)

| field | value |
|---|---|
| active_serving_join_status | PASS |
| active_training_lineage_status | PASS |
| legacy_replay_status | FAIL (kept as historical evidence) |
| overall_model_integrity_status | **FAIL** |

The overall stays FAIL for a REAL active reason, now visible instead of masked: train/serve feature parity
(output/feature_parity_health.json, 2026-11-22T16:27:57Z) FAILs `home_has_ht_history` / `away_has_ht_history` on 9 of
10 parity-checked tracks, and Data truth 3 is RED because the Standard HT payloads were trained with
`home_has_ht_history` = 1.0 on every row. Standard O/U 2.5 is WARN (parity WARN, weight alignment UNKNOWN); NF HT
tracks WARN (parity not measured). Local copy: `C:\Users\nevo\dev\p0-evidence\after\model_integrity_health.local.json`.

## Tests

- tests/test_training_lineage.py: `test_active_safe_joins`, `test_legacy_join_replay_isolation`,
  `test_code_audit_flags_contaminated_lineage`, `test_training_lineage` (row counts, duplicate keys, club twice per
  day, other-track and unregistered leagues), `test_point_in_time_audit`,
  `test_lineage_never_pass_without_matching_evidence`, `test_committed_lineage_evidence_is_consistent`.
- Full-suite result: see the PR and reports/P0_HEALTH_STATUS_CORRECTION.md.

## Remaining blockers

1. Code commit 047442d95 (training code of the six 2026-11-19 Standard payloads) exists only in the local clone
   (branch `local/std-retrain-1611`, not pushed). Its four training-code files equal origin/main byte for byte, so the
   audit is reproducible from main; pushing that branch (or a tag) would make the commit itself public. Owner decision.
2. The `home_has_ht_history` constant / parity FAIL (active, real) - handled by the HT retrain work stream; model
   integrity stays FAIL until it lands.
3. The evidence is produced locally (the saved frames are not in the repo). A future install changes the payload
   sha256 and the lineage turns UNVERIFIED until the script is rerun on that install's frame.

## Production impact

None on tips, stakes, bankroll, promotions or records. Health artifacts only: model_integrity_health.json gains the
four scope fields and stops showing the legacy replay as an active join failure.

## Owner approval required?

No for this change (health reporting only, research lock unchanged). Yes for blocker 1 (push the local training branch).
