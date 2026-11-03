# P1 - Stake sizing policy validation (pooled league rating must stay OFF)

Phase 2, Part 9. Independent QA audit, 2026-11-03. Every figure below was re-derived for this report from the commits listed in section 6. Nothing was retrained, backtested, sent to Telegram, or pushed to `main` / `data-truth-100`, and no threshold, stake or track state was changed.

## Verdict

**In v9 the rule is honoured on both branches. In wowza-exec production it is not.**

- **v9 (wowza-betting):** on `main` and `data-truth-100`, the bankroll-policy loader and `src/stake_plan` do not scale any stake by the pooled rating. The committed stake ledger confirms it: all 13 stakes planned after the switch-off are unscaled. Before the switch-off, 116 of 168 were scaled down.
- **Pro's simulator** never applies the rating, whatever the flag says.
- **wowza-exec `main` (production) still has `league_rating_sizing: true`.** Its paper simulation runs every 3 hours and still scales stakes by the pooled rating. A rating-1 league is staked 1.00 USDT instead of 10.00. The fix exists only on `data-truth-100` and is not merged. Real-money caps are 0 (PAPER_ONLY), so no money is affected.
- **Dashboard and Telegram labels were wrong.** They said the rating is "the number the stake plan sizes by". The labels are corrected on branch `phase2/sizing`, and a test now guards them.

| Path | `main` | `data-truth-100` | Evidence |
|---|---|---|---|
| v9 policy file `config/bankroll_policy.json` | **OFF** (`enabled: false`) | **OFF** | flipped in v9 commit `820bb6cd` (2026-11-02 08:33:56 UTC); research-lock snapshot also `false` |
| v9 loader + sizer `src/bankroll_policy.size_day` | honours flag | honours flag | scales only if flag **and** a ratings dict are both present (dt-100 `bankroll_policy.py:163-164`) |
| v9 `src/stake_plan.write` | honours flag | honours flag | ratings loaded only when flag on (`stake_plan.py:247` main / `:281` dt-100); new test proves it never even loads them |
| v9 committed ledger `output/stake_audit.csv` | 13 / 13 post-flip OK rows unscaled | no post-flip rows yet | recomputed requested = base x tier x league for every row |
| Dashboard displays (v9 pages + wowza-dashboard copy) | no sizing effect; **caption wrong** | no sizing effect; **caption wrong** | fixed in v9 on `phase2/sizing`; wowza-dashboard copy still wrong (open issue) |
| Telegram tips image `telegram_bot/tips_table.py` | display only; **implied sizing** | same | fixed on `phase2/sizing` (footer + docstring) |
| wowza-exec `registry/execution_policy.json` | **ON - VIOLATION** | OFF | main commit `a149118b` (2026-10-18); dt-100 fix `f6064e54` (2026-11-02) not merged |
| Pro simulator `src/bankroll/simulate.py day_plans` | never scales | never scales | calls v9 `size_day` without `league_ratings` |

## 1. What the rule is

Owner rule (2026-11-02): pooled league-rating stake sizing stays OFF. Pro's `output/studies/league_ratings.json` pools every market, tier and track into one rating per league, so it cannot justify sizing one league x market track. The rule is recorded beside the flag in `config/bankroll_policy.json` (`league_rating_sizing.note`), and the previous state is kept under `previous_state` rather than overwritten.

## 2. Path-by-path findings

### 2.1 v9 bankroll-policy loader and sizer (`src/bankroll_policy.py`)

`size_day` sets `use_rating = bool(lrs.get("enabled")) and league_ratings is not None`. Both gates must hold before any stake is multiplied by `rating_scale`. With the flag off, `rs = 1.0` and `requested_fraction = base x tier x league_multiplier`, and the `league_rating` / `rating_scale` columns are not even emitted. The logic is the same on `main` (lines 149-184) and `data-truth-100` (lines 163-207). The only difference is what an unrated league gets when the flag is ON: `main` gives `min_scale` (an invented rating), while dt-100 gives no stake (`NO_LEAGUE_RATING`). That difference is irrelevant while the flag is off.

### 2.2 v9 stake plan (`src/stake_plan.py`)

`ratings = load_league_ratings() if (policy.get("league_rating_sizing") or {}).get("enabled") else None`. With the flag off, Pro's ratings are never fetched and `None` is passed, which closes the second gate as well.

**Real-data check** (script recomputation, column-projected CSV reads via `git show`):

| Ledger (commit) | Rows | Status `OK` | requested = base x tier x league | requested < that product (scaled) |
|---|---|---|---|---|
| `output/stake_audit.csv` on main `f6c329a5`, rows generated **before** 2026-11-02 08:33:56 UTC | 726 | 168 | 52 | **116** |
| same file, rows generated **after** the flip | 91 | 13 | **13** | **0** |
| `output/stake_plan.csv` on main `f6c329a5` (generated 2026-11-03 06:44 UTC) | 71 | 10 | **10** | **0** |
| `output/stake_plan.csv` on dt-100 `e07a4591` (generated 2026-11-02 05:59 UTC, before the flip) | 70 | 20 | 5 | 15 (62 of 70 rows carry a rating) |

The pre-flip rows show the check can tell scaled from unscaled stakes: 116 of 168 were scaled while the rule was on. After the flip, none were.

### 2.3 Dashboard displays

The v9 pages `pages/1_*_Dashboard.py` (Rating column help) and `pages/17_*_Leagues.py` (caption), plus `dashboard_data/leagues.py`, show Pro's pooled rating and describe it as "the number the stake plan sizes by" / "the stake plan's sizing". The rating changes no stake, but the label states something false. **Fixed on `phase2/sizing`:** the labels now read "display only - not used to size stakes since 2026-11-02". The `wowza-dashboard` repo carries the same wording on `main` (`aa4e6a26`) and `data-truth-100` (`ff53514e`); it is not fixed here (open issue P2-1). Its copies of `src/stake_plan.py` / `src/bankroll_policy.py` have no callers in that repo (`git grep` for `size_day(` / `stake_plan.write` outside the modules themselves: none), so they size nothing.

### 2.4 Telegram presentation (`telegram_bot/tips_table.py`)

The tips image shows a "Rating" column, colour-coded green at 8 or above and red at 3 or below. Its docstring said "the same source the stake plan sizes by". The rating never reaches a stake: Telegram output contains no `rating_scale`. **Fixed on `phase2/sizing`:** the docstring and the image footer now say "not used to size stakes". The new footer text has not been checked visually in a rendered image: it is about 20 characters longer than before (open issue P3-2). No message was sent.

### 2.5 wowza-exec execution policy

| | `main` `56670768` (production; `cloudbet_shadow.yml` cron `20 */3 * * *`) | `data-truth-100` `a18bf544` |
|---|---|---|
| `global.league_rating_sizing` | **`true`** (commit `a149118b`, 2026-10-18) | `false` (commit `f6064e54`, 2026-11-02) |
| Note beside the flag | "Missing / stale ratings = rating 1" (an invented default) | owner rule text, unrated = no stake |
| Effect in `src/bot.py` | `load_ratings()`, then `stake_for(..., rating=)` then `scaled_stake`: linear from `min_stake` (rating 1) to the per-bet max (rating 10) | ratings not loaded; `league_rating` is None; stake = flat simulation cap |

Test evidence: dt-100's own `tests/test_ops6_rating_sizing_off.py` was copied unchanged onto a detached `main` checkout and run there. Three tests fail:
- `test_committed_policy_has_rating_sizing_off`
- `test_policy_note_matches_the_flag`
- `test_paper_simulation_does_not_scale_by_the_pooled_rating`: a rating-1 league got **1.0** instead of **10.0** (= 2000 USDT simulation bankroll x 0.005 per-bet cap).

On `data-truth-100`, `test_ops6_rating_sizing_off.py`, `test_league_rating.py` and `test_pipeline.py` give 29 passed and 1 xfail (strict).

The strict xfail records a real behaviour difference. With the flag off, the exec "unrated league = no stake" gate (`bot.py:243`) does not run, so unrated leagues are staked at the flat cap. v9 behaves the same way: `NO_LEAGUE_RATING` applies only while rating sizing is on. This needs an owner decision; see P2-2.

### 2.6 Pro simulator (`wowzaV9-Pro src/bankroll/simulate.py`)

`day_plans` calls v9's own `size_day` (loaded read-only via `src/bankroll/data.py v9_reader`) with `sel, policy, bankroll, peak, dd_floor` only, and never `league_ratings`. The rating gate therefore cannot open in Pro's simulator, on either branch (`main` `5b29395b`, dt-100 `78580b42`). A side note follows from this. While v9 rating sizing was ON (2026-10-18 to 2026-11-02), Pro's monthly replays sized differently from production: the simulator was unscaled and production was scaled. Any policy-comparison numbers from that window describe a policy that was not the one deployed.

Pro's committed copies of the policy also differ between branches:
- On `main` (commit `5220c892`, 2026-11-02), `output/bankroll/current_policy_recommendation.json` and `v9_bankroll_policy.json` say `enabled: false`.
- On `data-truth-100` (commit `847265c8`, 2026-10-26), the same files still say `enabled: true`. These are stale outputs generated before the flip. They are not read by any sizing path. The dashboard displays the recommendation from Pro `main`. See P3-1.

## 3. Tests added (failing-test-first)

File: `tests/test_p2_sizing_policy_off.py` (wowza-betting, branch `phase2/sizing`).

| Test | Fails when | Before fix (dt-100 base) | After fix | On v9 `main` |
|---|---|---|---|---|
| `test_committed_policy_keeps_pooled_rating_sizing_off` | the policy flag is turned back on, or its reason note is dropped | pass | pass | pass |
| `test_stake_plan_never_reads_pooled_ratings_while_off` | `stake_plan.write` loads ratings or any OK stake differs from base x tier x league (every league rated 1 in the stub, so any scaling would cut it to 20%) | pass | pass | pass |
| `test_power_stake_plan_check_catches_scaling_when_flag_is_on` | power check: the same assertion must detect scaling when the flag is forced on in-memory | pass | pass | pass |
| `test_size_day_needs_flag_and_explicit_ratings` | `size_day` scales with only one of the two gates (the Pro simulator contract) | pass | pass | pass |
| `test_no_surface_claims_the_rating_sizes_stakes_while_off` x4 surfaces | a dashboard/Telegram surface claims the rating sizes stakes while the flag is off | **4 fail** | pass | **4 fail** |
| `test_telegram_footer_says_rating_is_display_only` | the Telegram footer drops "not used to size stakes" | **fail** | pass | **fail** |

Focused regression runs after the fix, all on `phase2/sizing`:
- the new file plus `test_league_rating_sizing.py`, `test_stake_plan.py`, `test_leagues_page.py` and `test_research_lock.py`: 39 passed;
- `test_tier_labels.py`, `test_telegram_harness.py` and `test_home_tips_values.py`: 93 passed.

The full suite was not run (laptop rule). No existing test or baseline was changed.

## 4. Open issues

| ID | Priority | Issue | Owner action |
|---|---|---|---|
| P1-1 | **P1** | wowza-exec `main` still scales paper-simulation stakes by the pooled rating, every 3 h, and records "missing = rating 1" | Approve merging wowza-exec `data-truth-100` (or cherry-picking `f6064e54` + `c64215ad`) into `main`. Real caps are 0, so this is paper only. |
| P2-1 | P2 | wowza-dashboard repo (both branches) still labels the rating as "the stake plan's sizing" | Approve the same three caption edits there (the `phase2/sizing` diff applies verbatim). |
| P2-2 | P2 | With pooled sizing off, "unrated league = no stake" is enforced nowhere (v9 `size_day` and exec `bot.py:243` both tie it to the flag; exec has a strict xfail) | Decide whether that gate should stand on its own; if yes it becomes a separate, tested change (not a re-enable). |
| P2-3 | P2 | v9 CI runs no sizing or policy test (the only pytest steps are onboarding, freshness, exchange, integrity and props) | Add `tests/test_p2_sizing_policy_off.py`, `test_league_rating_sizing.py` and `test_research_lock.py` to the Phase 2 CI regression job (Part 13). |
| P3-1 | P3 | Pro `data-truth-100` carries stale `output/bankroll/*.json` with `enabled: true` (Oct 26) | On merge, keep `main`'s Nov 2 versions; regenerate rather than hand-edit. |
| P3-2 | P3 | The new Telegram footer is longer and has not been checked in a rendered image | Render one test image (no send) before merging. |
| P3-3 | P3 | Pro monthly replays from 2026-10-18 to 2026-11-02 sized differently from production (unscaled vs scaled) | Label those comparisons when reporting them (Part 11, TRUSTWORTHY_PERFORMANCE). |

## 5. Proposal only: a replacement for the pooled rating (NOT enabled)

The proposal is to let evidence shrink a stake only for a specific **league x market x side x model x model-version** cell. It never increases a stake.

1. **Unit of evidence.** One settled, staked or shadow-recorded bet in the cell, keyed by `track_key(model_type, market, side)` + `league` + `model_version`. A bet whose model version is not the current one does not count. A retrain resets the cell (no transfer without a stated, tested mapping).
2. **Out-of-sample only.** Only bets whose prediction was made before kickoff by the deployed model version count, so no backfilled or replayed predictions. Freeze the inputs and the result per bet (fixture id, odds taken, closing odds, result).
3. **Metric.** Use closing-line value, log(odds_taken / closing_odds), as the primary measure. It is less noisy than ROI and needs about 10x fewer bets. Realised ROI per unit staked is secondary.
4. **Shrinkage.** Hierarchical partial pooling. Each cell's CLV is shrunk toward its market x model-version mean, which is in turn shrunk toward zero. In closed form: `theta_cell = w * mean_cell + (1 - w) * mean_parent`, with `w = n / (n + k)` and `k = sigma^2_within / tau^2_between`, both estimated from the data. These are not typed constants: if tau^2 cannot be estimated (too few cells), w = 0 and nothing scales.
5. **Mapping to a stake (down only).** `scale = clip(P(theta_cell > 0), floor, 1.0)`, where the posterior probability comes from step 4. The floor stays a policy-file value approved by the owner. A cell with fewer than n_min out-of-sample bets has no rating, and the existing "no evidence" rule applies.
6. **Validation before any switch-on.** Walk-forward by month: does month-N scaling, fitted only on bets before month N, improve month-N log-growth versus unscaled, on an untouched holdout? A shuffled-label negative control must show no improvement. The result must be signed off by the owner. The test in section 3 stays in place and is updated only by an explicit owner commit that flips the flag.

## 6. What was inspected

| Repo | `origin/main` | `origin/data-truth-100` |
|---|---|---|
| wowza-betting (v9) | `f6c329a547118e50e5cbcf0cba9d7538c9762d27` | `e07a45915ba53608dbcfe71bccb3191a3edcbed5` |
| wowzaV9-Pro | `5b29395b55dce941ae1344106e5168a6cbe00cdc` | `78580b4286f3748f049b1a77a822b4b47a2b7a0a` |
| wowza-dashboard | `aa4e6a26f2e9f49f81cd1267c7b86741dfa5f626` | `ff53514e70eb91a5787cf42e4fe3bbda078c678c` |
| wowza-exec | `56670768db0fa9e4fbfa02ab031542a83a889687` | `a18bf544624915cd191fcdcd85f0b18769139ad2` |
| wowza_v11 | `e31e1824b65c984fdbbf4907437cae6f024c1cc5` | `27acd24ae2d39ee626a2a13c5afac1479c17c139` |

wowza_v11 contains no rating-sizing code on either branch (`git grep` for `league_rating_sizing|rating_scale|load_league_ratings|league_ratings`: no hits).

**UNIDENTIFIABLE:** the actual stakes from exec `main` shadow runs since 2026-11-02. The exec ledgers (`config.BET_LEDGER`, candidate log) are not committed (no `.csv` / `.jsonl` in `main`'s tree), so the size of the post-flip scaling in real runs cannot be counted from committed files. Section 2.5 shows the mechanism, not a run count.
