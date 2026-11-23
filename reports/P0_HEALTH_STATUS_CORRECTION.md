# P0 item 2 — health status correction: four independent indicators

Owner P0 deep-repair list, 2026-11-22, item 2. Repo NevixAA/wowza-betting, branch `p0-health-join-lineage`
(based on origin/main `2a86ab87a824d37df2ac1cd032438af91135552b`; origin/main merged in before the PR).
HEAD SHAs at start (Stage A): wowza-betting `2a86ab87a824`, wowzaV9-Pro `eacaeec28bc3`, wowza_v11 `c0d2961ea9a7`,
wowza-dashboard `e2e96f0a8f5c`, wowza-exec `82f6d7d9fcbd`.

## Finding (reproduced on main before any change)

| artifact | generated_at | verdict |
|---|---|---|
| output/system_health.json | 2026-11-22T14:56:17Z | overall **FAIL**: 28 GREEN / 4 AMBER / 1 RED / 1 NO_DATA |
| output/data_truth_health.json | 2026-11-22T14:58:31Z | `stoplight` **GREEN**, `gate_reasons` [] — while its own check "Data truth 3: no missing data" is **RED** |
| dashboard `truth_view.stoplight()` | (reads the newer copy = data_truth_health.json) | **GREEN**, basis "gate verdict as recorded" |

The RED check: `team:model_ht_over05.pkl` / `model_ht_over15.pkl` — served feature `home_has_ht_history` holds one
value (1.0) on 100% of training rows. Rule 3 is not gated (operational data must not block code), so the gate is
GREEN, and the dashboard displayed that gate verdict as the data truth stoplight. Before-state:
`C:\Users\nevo\dev\p0-evidence\before\reproduction.txt` and the four JSON copies beside it.

## Root cause

`dashboard_data/truth_view.stoplight()` returned `gate` whenever the artifact recorded one
(`if gate in ("GREEN", "RED"): status = gate`) and fell back to the worst check only when no gate existed. One word
answered two different questions: "may this commit merge" (gate) and "is the current data clean" (worst check).
Nothing reported model integrity or full-system certification as separate verdicts.

## Fix: four indicators, never one GREEN for all four

`dashboard_data/health_status.py` (new) reads committed artifacts only; a missing artifact is NO_DATA, never a pass.

| indicator | rule | values |
|---|---|---|
| CI / change-control gate | data truth `--gate` verdict (gated rules + ratchet) AND research lock (`protected_tracks_modified`) | GREEN / RED / NO_DATA |
| Operational data quality | WORST current data truth check, gated or not (newest recorded copy) | GREEN / AMBER / RED / NO_DATA |
| Production model integrity | model_integrity_health.json `overall_model_integrity_status` (parity, chronology, holdout, batch invariance, active serving joins, training lineage); legacy join replay shown, not counted | PASS / WARN / FAIL / NO_DATA |
| Full-system certification | every required component (system health, data truth operational, data truth gate, model integrity, active serving joins, train/serve parity, training lineage) has evidence <= 36 h old and passes | CERTIFIED / NOT CERTIFIED |

Every failing item carries component, status, detail, source artifact and generated_at (operational items also
`gated`).

| file | change |
|---|---|
| dashboard_data/health_status.py (new) | the four indicators + `render()` |
| dashboard_data/truth_view.py | `stoplight()["status"]` = worst check always; gate verdict returned beside it as `gate`; `panel()` adds `indicators`; `render()` shows the four indicators (Home compact, System Health with the full failing table); Home metric renamed "Operational data quality (worst check)" |
| src/system_health.py | system_health.json records `data_truth_gate` (gate on the same checks) and `indicators`; CLI prints each indicator |
| src/integrity/data_truth.py | data_truth_health.json adds `stoplight_scope` (gate only), `operational_status`, `operational_failing`; `stoplight` keeps its meaning (CI gate) |
| tests/test_dashboard_truth.py | two assertions encoded the bug (gate GREEN shown with an AMBER check); now expect the worst check + `gate` |
| deploy/dashboard_manifest.json | regenerated (new dashboard module) |

Pages: Home and System Health both render through `truth_view.render`; no other caller of `stoplight()` exists.

## After (current committed artifacts, this branch)

| indicator | status | failing items |
|---|---|---|
| CI / change-control gate | GREEN | 0 |
| Operational data quality | RED | Data truth 3 RED (ungated); 13a AMBER; 13c AMBER |
| Production model integrity | FAIL | parity FAIL `home/away_has_ht_history` on 9 tracks; NF HT parity unmeasured; weight alignment UNKNOWN |
| Full-system certification | NOT CERTIFIED | system health FAIL; data truth operational RED; model integrity FAIL; feature parity FAIL; (until the integrity job reruns) training lineage not yet recorded |

Post-merge artifact regeneration: see "Post-merge confirmation" below.

## Tests

- tests/test_health_indicators.py: `test_data_truth_gate_vs_operational_health` (CI gate GREEN via the real
  `data_truth.gate`, operational RED, model integrity FAIL, certification NOT CERTIFIED; every failing item has
  component/source/generated_at), `test_dashboard_status_consistency` (dashboard stoplight = operational indicator;
  panel indicators = module indicators; four distinct statuses; system_health override gives the same verdicts),
  `test_gate_red_and_research_lock_turn_ci_red`, `test_missing_or_stale_evidence_is_never_certified`
  (CERTIFIED reachable only from passing current evidence), `test_no_artifacts_is_no_data`.
- Gates and full suite: see PR.

## Remaining blockers

- The RED itself (Standard HT payloads trained with `home_has_ht_history` constant) is real and is the HT retrain
  work stream's to fix; this change only stops hiding it.
- Certification will stay NOT CERTIFIED until system health, data truth, parity and model integrity all pass.

## Production impact

None on tips, stakes, bankroll policy, promotions or historical records. Dashboard and health artifacts only.

## Owner approval required?

No (reporting change; research lock unchanged, protected_tracks_modified=False).
