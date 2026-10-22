# Wowza system repair — 2026-10-22

Repair brief from the owner, 2026-10-22. Every number below was read from committed artifacts on
`origin/main` or from GitHub Actions runs on the same day, not from local runs. Nothing here
changes a served probability, a model, a threshold or a Telegram message, except where a section
says so and names the owner decision.

## 1. Summary

| item | root cause | status |
|---|---|---|
| P0-1 side-market staking | stake plan lost `model_type` and `odds` for side markets, so every BTTS/O1.5/O3.5 tip landed on `unknown\|*` = UNSUPPORTED | **fixed, verified on CI** |
| P0-2 challenger logging | challenger columns hard-coded to NA; challenger slot only filled on a SHADOW_MORE retrain (never happened); HGB challenger fitted in memory and thrown away | **fixed, verified on CI** (203/203 rows) |
| P1-1 CLV collection density | archives are change-logs, so an unchanged near-kickoff price wrote nothing; NEAR runs lived 40–50 min | **fixed and deployed; effect not yet measurable** (needs a match day) |
| P1-2 bankroll dependencies | monthly job imported scipy which Pro did not install; Pro had no canonical requirements file | **fixed, verified on CI** |
| P1-3 artifact freshness | no manifest; dashboard showed months-old artifacts as current | **fixed, verified on CI** |
| P1-4 forward-test dashboard | three windows (discovery / registered / true forward) shown mixed | **fixed** |
| P1-5 standard OU25 | stays PAPER | **unchanged, deliberately** |
| P1-6 tier vs evidence rank | dashboard implied SNIPER > MARKSMAN > VALUABLE = more confidence; edge does not rank O/U outcomes | **fixed** — plus a bug found today: the evidence file was never committed (Pro `da29320`) |
| Fantasy | V2 residual mode had been set to `auto` | **back to `shadow`**; chronological scoreboard added |
| V11 | research only | **unchanged** |
| Repo growth (approved separately) | odds CSVs re-stored whole on every commit | **fixed, verified on CI** |

## 2. Root causes and files changed

### P0-1 — side-market tips never reached the bankroll policy
`stake_plan` built its track key from a frame that had dropped `model_type` and `odds` for side
markets, so the policy saw `unknown|BTTS` and returned UNSUPPORTED. The policy's own ACTIVE
`new_format|BTTS` track never received a tip.
Commit `b4ce9e22`. Equal O/U tier stakes plus staked VALUABLE (owner decision): `db08a0af`
(`config/bankroll_policy.json` `track_tier_multipliers`, `src/bankroll_policy.py`).

| stake plan | before (predict 10:07, `c51affe5`) | after (predict 13:45, `cf338dc2`) |
|---|---|---|
| `unknown\|BTTS` UNSUPPORTED | 24 tips, 0 staked | 0 |
| `unknown\|OVER15` / `unknown\|OVER35` | 7 + 1, 0 staked | 0 |
| `new_format\|BTTS` ACTIVE | 0 | 16 tips, 5 staked |
| `new_format\|OVER15` ACTIVE_LOW | 0 | 7 tips, 0 staked (below floor) |
| `new_format\|OVER35` | 0 | 1 tip, BLOCKED by policy |
| `standard\|BTTS` PAPER | 0 | 7 tips, paper |
| `new_format\|OU25` ACTIVE_LOW | 58 tips, 38 staked | 61 tips, 40 staked |
| `standard\|OU25` PAPER | 19, paper | 18, paper |

All of this is **paper money**. Cloudbet stays PAPER_ONLY.

### P0-2 — the champion/challenger experiment logged nothing
Three causes (`pipeline.py` `_log_shadow`, `_log_h2h`; `src/model_validation.py` `_hgb`):
challenger columns written as NA by construction; the challenger slot fills only on a SHADOW_MORE
verdict, which no retrain has produced; the HGB Over 2.5 model behind H-MODEL-01 was never saved.

Fix (commits `50ea8012`, `8e1c3d38`):
- `scripts/register_shadow_experiment.py` trains the HGB challenger once per track (standard,
  new-format — never pooled), cutoff = H-MODEL-01's development period end (2026-09-28), frozen
  and registered by sha256 under `models/registry/experiments/`.
- `src/shadow_experiments.py` scores champion-raw and challenger on the same rows at first sight
  (`output/model_lineage/shadow_experiments.csv`, append-only, earliest row wins), settles by
  adding result columns only, evaluates paired log loss / Brier with a matchday bootstrap, and
  guards (exit 1 if an active experiment logs nothing over 72 h after a 48 h grace period).
- Predict runtime cost: 0.34 s.

| | before | after (first CI run, predict `cf338dc2`) |
|---|---|---|
| `shadow_h2h` rows with `p_challenger` | 0 of 1,388 | 203 of 203 new rows |
| `shadow_experiments.csv` | did not exist | 203 rows (88 standard, 115 new-format), all fields filled |
| `v91_shadow.p_pro_hgb` | 0 of 669 | 0 of 669 — fills from the next *new* fixture; existing rows are first-sight and never rewritten |

### P1-1 — certified closes were 7.6%
Measured on kickoffs 2026-10-04 to 10-22: a capture *ran* inside T-30m for ~30% of
fixture-markets, but only 2.8% had a *recorded* observation there, because the archives store
price changes only. Fix (commit `1187386c`):
- `src/odds_heartbeat.py`: the latest strictly-pre-kickoff observation per fixture × market,
  overwritten in place; change-logs untouched.
- `src/capture_schedule.py`: per-fixture due intervals and a keep-alive while a kickoff is within
  90 min. Replayed on measured run starts: capture inside T-30m ~46% → ~76% (new-format),
  ~34% → ~82% (standard), with ~25% fewer odds calls. **This is a replay estimate, not a result.**
- `src/shadow_settlement.py`: the close search no longer stops at a stale first archive; settle
  240 min after kickoff.
- `output/clv_collection_health.json` + `scripts/check_clv_collection_health.py` (last step of
  `update_results.yml`), floors in `config/clv_collection_expectation.json`.

| certified close (≤ 30 min, strictly pre-kickoff) | value on 2026-10-22 13:09 UTC |
|---|---|
| settled shadow rows | 464 |
| with certified T-30 close | 36 (7.8%) |
| with T-10 close | 8 (1.7%) |
| median minutes of last pre-kickoff price | 321.5 |

This is still the *before* number: the new loop has run once for standard (11:26, green, no
fixture inside its window on a Thursday) and is running for new-format now. The first honest
after-number comes from this weekend's fixtures.

### P1-2 — bankroll dependencies
`src/bankroll/allocator.py` imported scipy; Pro never installed it. Fix: pandas Spearman
(`28ed8d3`) and a canonical Pro `requirements.txt` used by 23 workflows, guarded by
`tests/test_requirements_cover_imports.py`. **Self-inflicted incident:** Pro's `.gitignore` is an
allow-list and silently ignored the new root file, so 8 workflows failed on `253ea53` for a few
minutes. Fixed in `263e5fe`; all 8 re-ran green on `72ae3ab`.

### P1-3 — artifact freshness
`src/artifact_freshness.py` (manifest with recorded `generated_at`, never mtime; unknown age =
stale), `dashboard_data/freshness.py` (STALE banner), `src/system_health.py` (nine checks, each
with a written rule), `.github/workflows/evidence_refresh.yml` (daily 06:45 UTC). Commits
`d21b7ed1`, `91bfe1fe`. Manifest at 11:00 UTC: 1 stale decision-facing artifact
(`clv_collection_health.json`, which did not exist yet — it does now).

### P1-4 / P1-6 — dashboard
Forward tests show three windows apart with progress bars; O/U stakes read "equal stake · tier ≠
confidence" (`ad4f1dc2`). **Self-inflicted incident:** a heredoc turned `\n` into real newlines
and broke `pages/7_🔬_Pro.py` (`ec4382b2`); fixed in `33cd9eea` and guarded by
`tests/test_pages_compile.py`.
Found today: Pro's monthly job computed `tier_rules.csv` but never committed it (not allow-listed,
not staged), so the panel read an empty table. Fixed in Pro `da29320`.

### Fantasy
`FANTASY_V2_RESIDUAL_MODE` back to `shadow` (`d0c4ac86`) per the brief. Chronological scoreboard
(`b9793fae`): the linear residual beats FPL's baseline on points MAE in GW5–7, **not** on pick
or captain hit rate. Not promoted.

### Repo growth (separate approval)
`964ffa5a`, `51b532d8`: live odds files keep recent and upcoming matches; played matches move to
`<stem>_archive/<date>.csv`. Verified on predict, prop_odds_snapshot and player_props runs. Live
files now 6.8 MB and 3.0 MB; projected growth cut ~90%. Old history is still in `.git` (rewrite
needs approval).

## 3. Tests

| suite | result (2026-10-22, after all commits) |
|---|---|
| v9 `pytest tests/` | **588 passed, 24 skipped** |
| Pro `pytest tests/` | **179 passed** |

New v9 tests: `test_shadow_experiments` (14, incl. end-to-end prediction → settlement → paired
log loss checked by hand), `test_artifact_freshness`, `test_system_health`, `test_pages_compile`,
`test_track_tier_multipliers`, `test_nf_max_odds_cap`, `test_af_seasons_complete`,
`test_team_aliases_af`, `test_fantasy_settle_dropoff`, plus the CLV heartbeat/schedule tests.

## 4. Workflow status

Since 11:30 UTC: **v9 7 success / 0 failed** (2 running), **Pro 14 success / 0 failed** (1
running), v11 no runs in the window. The 9 failures in the 11:01 workflow record were the
requirements incident above and the earlier monthly scipy failure; every one has re-run green
(learned_tiers, weather_fetch, weekly_league_promotion, props_learner and monthly bankroll
confirmed individually).

## 5. Deliberately not changed

- Served probabilities, thresholds, tier rules, Telegram semantics.
- Standard OU25 stays PAPER (P1-5).
- No league switched off: learned tiers never remove a league; the league-off test needs ≥ 600
  tips (`registry/learned_tiers/amendment_v1_1.json`).
- Cloudbet PAPER_ONLY; props paper-only; V11 research only.
- `v91_shadow` historical rows are not back-filled with challenger probabilities — that would be
  hindsight.

## 6. Blockers and owner decisions

1. **`merge=union` for `output/player_prop_odds_history.csv` and `output/prop_quotes/*.csv`.**
   Two jobs writing the same file overlapped and 511 prop-odds rows were lost today. Recommended.
2. **Git history rewrite** to reclaim the old copies. Destructive; needs approval.
3. **H-MODEL-01 comparator:** forward comparison is against the *serving* champion, not the
   never-built "corrected stack". Confirm. The 10-02 → 10-22 window cannot be recovered.
4. **H-MODEL-01 `n_min` = 1,000 per track** (the historical n=11,836 may have been pooled).
5. `p_market` in the experiment ledger is the vigged 1/odds, not de-vigged.
6. API-Football pricing path for Scout leagues.

## 7. Milestones to watch

| when | what | how to read it |
|---|---|---|
| this weekend (10-24/25) | first certified-close rate under the new NEAR loop | `output/clv_collection_health.json`; target ≥ 50% |
| 2026-10-24 11:02 UTC | experiment guard leaves its grace period | `update_results.yml` goes red if challenger logging stops |
| first new fixture | `v91_shadow.p_pro_hgb` non-empty | system-health "Challenger prediction coverage" |
| next monthly bankroll run | `tier_rules.csv` committed | Pro page, "tier vs evidence" panel |
| ~1,000 settled per track | H-MODEL-01 decidable | `shadow_experiments.evaluate()` |
| ≥ 80–100 forward bets each | H-GATE-01..14 decidable | Pro "Forward tests" tab (now 1 forward bet in total) |
| ≥ 600 tips per league | league-off test | learned tiers registration |
