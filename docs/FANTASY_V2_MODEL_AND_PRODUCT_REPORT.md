# Fantasy V2 — model and product report

*2026-10-20 · v9 `player_model/fantasy_v2.py` · calculation `fantasy_v2-1.0.0`*

## 1. Decision (read this first)

| Question | Decision | Why |
|---|---|---|
| Replace the old standalone Wowza xPts as the headline projection? | **GO** | It lost to FPL `ep_next` in every settled gameweek (MAE 2.04 vs 1.36, n=724). V2 makes FPL `ep_next` the mean forecast and puts Wowza's work in a residual and in the parts FPL does not publish. |
| Add the Wowza residual to the headline mean (`final = ep_next + residual`)? | **NO-GO for now** | Out of sample it beats FPL in both completed test gameweeks (pooled MAE −0.095, 90% CI −0.157 to −0.032, n=839). But that is **2 completed gameweeks**, and the rule fixed for this decision needs 3. It also gets no clear win on the players the old projection covered (MAE −0.064, CI −0.171 to +0.039). It runs in **shadow**: it is computed, published and scored every day, but not added to the headline. |

**What has to happen next.** GW7 becomes a completed gameweek once FPL sets `data_checked` and the GW8 pre-deadline snapshot has been taken. After that the gate re-scores itself every day. If it reports **GO**, the residual still stays in shadow until the owner changes `FANTASY_V2_RESIDUAL_MODE: shadow` to `auto` in `fantasy_refresh.yml`. A gate result alone never changes the board.

**Selection caveat, stated plainly.** I chose LINEAR as the production arm, and added the "projection ≥ 0" floor, **after** I had seen GW5–7. Both are defensible (LINEAR was the simplest arm, and expected points cannot be negative). Even so, the GW5–6 numbers below are not a clean prospective test. GW7-final, GW8 and later are.

## 2. Diagnosis of the existing product

* `player_model/fantasy.py` adds up a **full-match** projection (2 appearance points, attack, clean sheet, defensive contribution, bonus) and multiplies the total by `p_start`. That is the "full match × start probability" error. A player who comes off the bench is still credited with clean-sheet and appearance points he cannot earn.
* Wowza's own projection was never anchored to FPL's free `ep_next`. The settlement (`fantasy_settle.py`) measured the gap but nothing used it.
* No point-in-time history existed. `fpl_bootstrap.json` is overwritten every run, and a CI checkout is depth-1, so the pipeline could never see what was known before a past deadline.
* Coefficients such as the FDR step and home edge are conventions, not fits. That is already stated in the code.

## 3. Benchmark, recomputed

**Published ledger** (`output/fantasy_ledger.csv`, scored exactly as Pro's `fantasy_benchmark.py`):

| arm | n | MAE | RMSE | Spearman | bias | GWs |
|---|---|---|---|---|---|---|
| FPL ep_next | 724 | **1.357** | **2.403** | **0.731** | −0.093 | 4, 5, 6 |
| Old Wowza (proj × P(start)) | 724 | 2.038 | 2.843 | 0.576 | +1.012 | 4, 5, 6 |

These match Pro's figures exactly, and FPL won every gameweek. A regression test pins them (`test_published_ledger_benchmark_reproduces`).

**New point-in-time panel.** I rebuilt an archive of pre-deadline FPL snapshots for GW4–8 from the 40 committed versions of `fpl_bootstrap.json` (`scripts/fantasy_backfill_snapshots.py`). Each gameweek's actual points are the cumulative change to the next pre-deadline snapshot. This independent settlement **agrees with the ledger to the point for every player both can see** (`test_archive_actuals_match_the_settled_ledger`). It also covers every FPL player, not just the ~280 the old model projected.

## 4. Where the old Wowza +1 bias came from

The breakdown covers 901 player-gameweeks of the old projection across GW4–7 (GW7 provisional). Wowza bias is **+1.04**; FPL bias is −0.15.

| actual minutes | n | Wowza bias | share of total bias |
|---|---|---|---|
| 0 | 448 | +1.41 | **68%** |
| 1–59 | 144 | +2.03 | **31%** |
| 60+ | 309 | +0.04 | 1% |

* **It is a minutes problem, not a points problem.** For players who actually played 60+ minutes the projection is almost unbiased (+0.04).
* Mean `p_start` was 0.59, but only 50% of players played at all and 34% played 60+.
* The conditional projection is a full-match number (4.6 for 60+ players, against 4.1 actually scored). Applied to substitutes, it credits clean sheets and appearance points they cannot earn.
* By position, defenders (+1.45) and goalkeepers (+1.40) carry 72% of the bias. That is clean-sheet points paid to players who did not play 60.
* 280 of the 901 rows were players with **no Premier League minutes yet** this season.
* Smaller effects:
  * By price, cheap players (≤ £5.0m) are 70% of the bias.
  * By FDR, bias rises from +0.60 at FDR 2 to +1.54 at FDR 5.
  * Home and away are about equal (+0.96 / +1.13).
  * By club, Sunderland (+2.35) is the highest and Brighton (−0.38) the lowest.

Section 7 shows that the minutes mixture fixes this specific failure.

## 5. Architecture

```
fpl_bootstrap.json ─┐                          ┌─> fantasy_v2_projections.csv  (board)
fpl_fixtures.json  ─┼─> fantasy_snapshots ──> fantasy_panel ──> fantasy_residual ──> fantasy_v2 ─┼─> fantasy_v2_benchmark.json
fantasy_projection_log.csv ┘ (archive)   (pre-deadline features   (arms, minutes        ├─> fantasy_v2_summary.json (command centre, captaincy)
                                          + cumulative targets)    mixture, dists)      ├─> fantasy_v2_transfers.csv
                                                                                        ├─> fantasy_v2_fixture_matrix.csv
                                                                                        └─> fantasy_v2_oos.csv (audit)
```

* **Information set (pre-deadline only).**
  * The archived snapshot: `ep_next`, price, status, chance of playing, ownership, transfers, penalty and set-piece order, and cumulative minutes, starts, xG, xA, BPS and bonus.
  * Last-gameweek deltas against the previous snapshot.
  * Team strength from fixtures that finished before the deadline (player-xG sums, goalkeeper xGC per 90, goals; shrunk with a 4-game prior).
  * The gameweek's own fixture list (opponent, home or away, FDR, blank or double).
  * Wowza's own projection-log row from before the deadline.
* **Known leakage risk, small.** The fixtures cache is the latest file, so a fixture re-rated or postponed after a deadline shows its final value.
* **Targets.** Next snapshot minus this one. A **provisional** gameweek (partly played) keeps only teams whose fixtures had all finished. An unplayed match is unknown, never zero.
* **Arms.** Walk-forward: test gameweek *g* is predicted by models fitted on completed gameweeks before *g*.
  1. **FPL**: `ep_next`, the mandatory baseline.
  2. **WOWZA**: the old standalone projection × P(start).
  3. **LINEAR**: OLS residual on 9 features (`ep_next`, chance of playing, start rate, last-GW minutes, attacking xPts/90 × start rate, CS expectation × start rate, number of fixtures, gap to the old Wowza number, has-Wowza flag). This is the production arm.
  4. **RIDGE**: all 38 features standardised, alpha chosen by leave-one-gameweek-out.
  5. **HGB**: depth 3, 150 iterations, min 40 per leaf, L2 = 5.
  6. **Rank learning: not built.** Learning-to-rank needs many query groups (gameweeks), and there are three. FPL's within-gameweek Spearman is already 0.54 to 0.73.
* **Minutes mixture.** A multinomial logistic model gives P(0), P(1–59) and P(60+) from chance of playing, status, news, start rate, minutes per team game, last-GW minutes and starts, `ep_next`, price, number of fixtures, ownership and position. A blank gameweek is forced to P(0) = 1. Points conditional on each band:
  * 0 minutes: 0 points.
  * 1–59 minutes: the empirical pmf for the position.
  * 60+ minutes: 2 + negative binomial. Its mean is **solved so the mixture mean equals the headline xPts**. Its dispersion per position is chosen by the likelihood of the full mixture on training gameweeks.
* **Per-player outputs.**
  * The three numbers: FPL `ep_next`, Wowza residual, and final xPts (plus the shadow candidate).
  * The distribution: median, P25, P75, P90 (ceiling), P(blank), P(6+), P(10+), P(15+), and an uncertainty score and band.
  * Minutes: P(0), P(1–59), P(60+), P(start), expected minutes.
  * Event probabilities: P(goal) and P(assist) from xG/90 and xA/90 × expected minutes × fixture; clean-sheet probability; expected bonus; team goal expectancy.
  * Planning: 3-GW and 5-GW xPts, value per £m, a decomposition by group, confidence and risk.
* **3-GW and 5-GW projections.**
  * A per-fixture rate (if he plays) is backed out of the next-gameweek mean.
  * Each future fixture applies a position-weighted multiplier: an attack share scales with team goal expectancy, a defence share with clean-sheet probability. The shares (FWD .55/0, MID .45/.08, DEF .20/.40, GKP .05/.45) are a **stated convention, not a fit**.
  * Availability relaxes from the next-gameweek play probability toward the long-run rate with a half-life of one gameweek.
* **Transfers.**
  * For each of the 120 most-owned players (the realistic "current player"), the best same-position replacements priced within OUT price + £0.5m.
  * Ranked by **delta** over 3 and 5 gameweeks, with the −4 hit shown.
  * The dashboard planner recomputes this for any OUT player and any bank from the small projections table.
* **Captaincy.**
  * BALANCED: the best EV with P(start) ≥ 0.75.
  * SAFE: the lowest P(≤2 points) among nailed starters (P(start) ≥ 0.85) within 70% of the best EV.
  * HIGH-UPSIDE: the highest P(15+) within 80% of the best EV.
  * The three are distinct players. If the BALANCED pick would also have won a category, that is recorded.
  * Each pick carries its reasoning: P(6+/10+/15+), P(start), minutes, ceiling, opponent weakness, penalty and set-piece flags. A shadow view ranks on FPL + residual.

## 6. Validation results (walk-forward, out of sample)

**ACTIVE population**: every player with PL minutes or `ep_next` ≥ 0.5. GW5 was trained on GW4; GW6 on GW4–5; GW7 (provisional) on GW4–6.

| arm | n | MAE | RMSE | Spearman | bias | GWs |
|---|---|---|---|---|---|---|
| FPL ep_next | 1275 | 1.848 | 2.951 | 0.532 | +0.007 | 5, 6, 7* |
| **LINEAR (V2)** | 1275 | **1.708** | **2.593** | **0.580** | +0.044 | 5, 6, 7* |
| RIDGE | 1275 | 1.786 | 2.745 | 0.526 | +0.005 | 5, 6, 7* |
| HGB | 1275 | 1.817 | 2.756 | 0.527 | +0.072 | 5, 6, 7* |
| Old Wowza (where published) | 408 | 2.346 | 3.111 | 0.432 | +1.074 | 5, 6, 7* |

\* GW7 is provisional: every fixture has been played, but FPL has not yet set `data_checked`. **Completed gameweeks only (5–6, n=839):**

| arm | MAE | RMSE | Spearman | bias | MAE − FPL, 90% CI | RMSE − FPL, 90% CI |
|---|---|---|---|---|---|---|
| FPL | 1.799 | 2.902 | 0.565 | −0.040 | — | — |
| **LINEAR** | **1.704** | **2.622** | **0.593** | +0.016 | **−0.095 [−0.157, −0.032]** | **−0.281 [−0.380, −0.189]** |
| RIDGE | 1.788 | 2.774 | 0.536 | −0.041 | −0.011 [−0.056, +0.035] | −0.128 [−0.182, −0.074] |
| HGB | 1.820 | 2.804 | 0.541 | +0.025 | +0.021 [−0.036, +0.080] | −0.098 [−0.170, −0.028] |

The CIs are a player-cluster bootstrap with 2,000 resamples.

**Per gameweek (ACTIVE, MAE):**

| GW | FPL | LINEAR | RIDGE | HGB |
|---|---|---|---|---|
| 5 | 1.815 | 1.756 | 1.835 | 1.887 |
| 6 | 1.783 | 1.654 | 1.743 | 1.756 |
| 7 (provisional) | 1.944 | 1.716 | 1.783 | 1.810 |

LINEAR beats FPL on MAE and RMSE in all three gameweeks.

**LEDGER population** (the players the old projection covered, the published head-to-head):

* Pooled with GW7 (n=408): LINEAR MAE 1.624 vs FPL 1.679, RMSE 2.621 vs 2.770, Spearman 0.632 vs 0.599. LINEAR is lower on MAE in all three gameweeks (GW7 by only 0.02).
* Completed gameweeks only (n=320): MAE −0.064 (CI −0.171 to +0.039) and RMSE −0.110 (CI −0.243 to +0.021). Neither CI excludes zero, so this is not yet a demonstrated win.

**Decisions (top-k).** Top-10 overlap and "mean actual points of the top 15 by projection" are mixed. FPL was better in GW5 (5.13 vs 4.33 points), LINEAR in GW6 (3.6 vs 3.4) and GW7 (4.40 vs 3.07). Three gameweeks cannot show a ranking improvement, so I make **no decision-quality claim**.

**What the residual is actually correcting: FPL's `ep_next` is over-dispersed.**

| ep_next band | n | mean ep_next | mean actual | mean V2 |
|---|---|---|---|---|
| ≤ 0.5 | 436 | 0.20 | 0.76 | 0.70 |
| 0.5–2 | 374 | 1.36 | 2.15 | 2.07 |
| 2–4 | 241 | 3.10 | 3.01 | 3.07 |
| 4–6 | 139 | 5.12 | 3.75 | 3.83 |
| > 6 | 84 | **7.73** | **3.70** | 4.77 |

`ep_next` behaves like recent form. Players above 6 averaged half of it. The overall bias is ≈ 0 because the over-estimates at the top cancel the under-estimates at the bottom. A near-zero bias therefore hid a slope error. The linear residual's coefficient on `ep_next` is −0.89 (fitted on every settled gameweek), so the slope is ≈ 0.11 before the other features. A recalibration of `ep_next` alone (a + b·ep) captures part of the RMSE gain and little of the MAE gain (GW5–7 MAE 1.83/1.76/1.90 vs FPL 1.82/1.78/1.94, measured before the zero floor). The rest comes from minutes evidence.

**Distribution and minutes calibration** (walk-forward, FPL mean, n=1275):

* **Minutes mixture vs the heuristic** (chance × usage rate):
  * Log-loss 0.689 vs 0.897.
  * Brier for P(plays) 0.126 vs 0.168.
  * Brier for P(60+) 0.119 vs 0.140.
* **Against the old Wowza `p_start`, on its own players (n=408):** Brier for P(plays) 0.126 vs 0.156. This is the bias in section 4, fixed.
* **Goals, assists, clean sheets** (Brier, model vs climatology):
  * P(goal) 0.0516 vs 0.0567.
  * P(assist) 0.0567 vs 0.0581.
  * Defender and keeper clean sheet 0.122 vs 0.126.
  * These are small but real gains.
* **Expected bonus does not beat a constant** (RMSE 0.590 vs 0.576). It is shown, but not trusted.
* **Intervals.** Actual points were inside P25–P75 (inclusive) 68% of the time. They were above P90 13% of the time with the FPL mean and 11% with the V2 mean (target ≤ 10%).
* **Upper tail.** It is too fat around the FPL mean: predicted P(15+) is 1.9% against 0.5% observed, and P(10+) is 4.2% against 2.9%. With the V2 mean: P(15+) 1.1% and P(10+) 3.4%. This is the over-dispersion of `ep_next` flowing into the tails. The dashboard says to read upside probabilities as a ranking.

## 7. Feature importance

* **LINEAR, permutation on GW7** (RMSE increase when the group is shuffled): minutes 0.142, attacking role 0.065, calibration 0.031, availability 0.014, fixture 0.010, old Wowza projection 0.004.
* **RIDGE and HGB:** the "value/recency" group dominates (form, points last gameweek, ownership, price), then minutes, then attack. Fixture context is the weakest group in every model.
* **The old Wowza projection adds nothing beyond the other evidence.** Its gap feature has a coefficient of about −0.07, and its permutation importance is about 0.

## 8. Examples (walk-forward, LINEAR vs FPL)

**Where it correctly deviates** (the big `ep_next` numbers built on one haul):

| GW | player | ep_next | V2 | actual |
|---|---|---|---|---|
| 6 | Pascal Groß | 15.5 | 7.5 | 6 |
| 6 | Kevin Schade | 12.0 | 6.4 | 1 |
| 7* | Brian Brobbey | 8.5 | 3.0 | 0 (did not play) |
| 7* | Antoine Semenyo | 10.0 | 4.8 | 1 (13 min) |
| 7* | Matheus Cunha | 10.0 | 4.8 | 1 (6 min) |

**Where it hurts** (it shrinks a player who then hauls):

| GW | player | ep_next | V2 | actual |
|---|---|---|---|---|
| 7* | Johan Manzambi | 9.0 | 4.5 | 11 |
| 7* | Luka Vušković | 8.0 | 4.4 | 11 |
| 6 | Maxim De Cuyper | 8.5 | 5.4 | 15 |
| 5 | Aaron Wan-Bissaka | 1.7 | 4.7 | 1 (45 min) |

The residual is a regression-to-the-mean machine with minutes evidence. It is right on average and wrong exactly on the repeat hauls. That is why it stays in shadow until more gameweeks confirm it.

## 9. Dashboard (`pages/13_⚽_Fantasy.py`, a viewer)

The page reads only `fantasy_v2_*.csv/json` and `fpl_cache_meta.json`, with column projection. It does not import the model stack, does not read the parquet, and never uses mtime. The freshness age is computed at render time from the recorded fetch time.

1. **Gameweek command center**: the gameweek, deadline and countdown, FPL data age with its state, projection health (players, minutes flags, archived, final and provisional gameweeks), and which mean is live and whether the residual is in shadow.
2. **Model benchmark (always visible)**: every arm against FPL ep_next (MAE, RMSE, Spearman, bias, n, GWs, top-10 overlap), with population and completed-only toggles, a per-gameweek MAE chart, and the published-ledger line. FPL is always the first row.
3. **Top-recommendation cards**: player and team, position, price, final xPts, FPL baseline, Wowza adjustment (shadow or applied), P(start), minutes, ceiling, P(10+), next fixture, a coloured next-5 FDR strip with numbers, confidence, and risk with news.
4. **Projection decomposition**: a waterfall from FPL ep_next through calibration, availability, minutes, attacking role, fixture and the old-Wowza signal to the candidate, plus the minutes mixture, event probabilities and distribution.
5. **Player explorer**: filters by position, team, price, minimum P(start) and name, with six sort keys.
6. **Transfer planner**: OUT and IN side by side, a bank input, next/3/5-gameweek horizon, gain and the gain after a −4 hit, plus a precomputed table of the best moves.
7. **Captaincy lab**: SAFE, BALANCED and HIGH-UPSIDE cards with P(6+/10+/15+) and reasoning, and an EV vs P(15+) scatter (dot size = P(start)). Headline and shadow views.
8. **Fixture matrix**: a 20-team × 5-gameweek heatmap that can be coloured by attack ease, defence ease or official FDR. Each cell is labelled with its opponent.
9. **Best legal squad**: the existing MILP optimiser on any horizon, on demand.
10. **Method & calibration**: the decision and its reasons, minutes and interval calibration, P(6+) reliability, the over-dispersion table, importance, and the old-bias breakdown.

## 10. Files, workflow, tests

* **New modules**
  * `player_model/fantasy_snapshots.py`: the archive.
  * `player_model/fantasy_panel.py`: point-in-time features and targets.
  * `player_model/fantasy_residual.py`: the arms, metrics, gate, bias diagnosis, minutes mixture, distribution and importance.
  * `player_model/fantasy_v2.py`: orchestration, projections, captaincy, transfers, fixture matrix and the artifacts.
  * `scripts/fantasy_backfill_snapshots.py`: a one-off rebuild from git history.
* **Changed**
  * `pages/13_⚽_Fantasy.py`: rewritten.
  * `dashboard_data/fantasy.py`: V2 readers added; existing readers unchanged.
  * `.github/workflows/fantasy_refresh.yml`: one new non-fatal step, `python -m player_model.fantasy_v2`, with `FANTASY_V2_RESIDUAL_MODE: shadow`. The seven new artifacts are added to the existing **per-file** `git add` loop. The commit, `pull --rebase -X theirs` and retry pattern is unchanged.
* **Unchanged**: the v1 Fantasy pipeline (`fantasy.py`, the log, settle, health), which keeps producing the ledger the benchmark uses, and everything outside Fantasy (betting, thresholds, staking, models, other workflows).
* **Artifacts** (committed by the workflow): `output/fantasy_fpl_snapshots.csv` (~670 rows per gameweek) and `output/fantasy_v2_{projections,benchmark,oos,summary,transfers,fixture_matrix}`.
* **Tests**: `tests/test_fantasy_v2.py`, 20 tests.
  * Snapshot timing and replacement rules; cumulative targets; no use of post-deadline fixtures; a provisional gameweek drops unplayed teams; the Wowza projection is taken as of the deadline.
  * The walk-forward never trains on the test gameweek or later; linear contributions add up exactly; the gate needs 3 completed gameweeks and finds nothing when FPL is already right.
  * The pmf is valid and keeps the headline mean; a blank gameweek means 0 minutes; captaincy picks are distinct; transfers are same-position and affordable; the default mode is shadow.
  * Real-data regressions: the published ledger reproduces; the two independent settlements agree; the committed benchmark shows FPL; the page renders under `AppTest`.

## 11. Limitations

* Three test gameweeks, one of them provisional, is a small sample. The gate exists precisely so that this report cannot promote the residual on its own say-so.
* The board built on 2026-10-19 used a snapshot taken mid-GW7, so FPL's GW8 `ep_next` there contains partial-gameweek form. The daily run before the GW8 deadline replaces it.
* Fixture multipliers for future gameweeks, and the availability half-life, are conventions.
* Expected bonus is not better than a constant.
* Upside probabilities are too generous at the top end.
