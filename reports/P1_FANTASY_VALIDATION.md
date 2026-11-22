# P1 Fantasy validation (owner deep-repair brief 2026-11-22, item 11)

Repo `NevixAA/wowza-betting`. Branch `p1/fantasy-validation` from `origin/main` 61e869877
(2026-11-22 19:26 UTC). Nothing was retrained, promoted or staked. The residual stays shadow-only
(`FANTASY_V2_RESIDUAL_MODE: shadow` in `fantasy_refresh.yml`, unchanged and pinned by a test).

## Verdict

- **The data under the Fantasy benchmark is real and checks out.** All 2,631 FINAL player-gameweeks
  (GW5-GW10) have the same actual points and minutes as FPL's official `event/{gw}/live` feed:
  2,631 of 2,631 exact, 0 mismatches. Every archived snapshot was fetched before its deadline
  (0 of 6,002 rows at or after it). `ep_next` in the OOS frame equals the archived FPL value on
  2,959 of 2,959 rows.
- **The walk-forward is chronological and reproducible.** 0 chronology violations. A local re-run of
  `python -m player_model.fantasy_v2` on the committed files rebuilt `fantasy_v2_oos.csv`
  byte-identical, and the benchmark JSON matched value for value (the only differences were NaN
  fields and line endings).
- **The old pooled table compared different players.** On ACTIVE, FPL and HGB were scored on 2,631
  rows and WOWZA/LINEAR/RIDGE on 968. The new paired report scores every arm on the same rows.
- **On identical rows, V2 LINEAR is slightly better than FPL on error, not on captaincy.** MAE
  -0.036 (90% CI -0.085 to -0.008), RMSE -0.141, lower MAE in 5 of 6 gameweeks. Captaincy regret
  is worse: 73 points vs 66 for FPL. The evidence is still modest: 6 gameweeks, 968 rows.
  **Recommendation: keep shadow.**
- **Wowza V1 is clearly worse than FPL.** MAE +0.613 (CI +0.526 to +0.739), bias +1.12 points, worse
  or equal on MAE and captaincy in every gameweek.

## 1. Method

- **Source:** `output/fantasy_v2_oos.csv` on main. It holds the walk-forward forecasts of
  `fantasy_residual.walk_forward`. Gameweek g is fitted only on FINAL gameweeks below g.
- **New module** `player_model/fantasy_paired_eval.py`. It writes
  `output/fantasy_v2_paired_walkforward.json` and makes no decision.
  - **Primary population (ALL_ARMS_PAIRED):** FINAL gameweeks only. A row counts only if FPL, WOWZA,
    LINEAR, RIDGE and HGB all have a forecast and the actual is known.
  - Rows whose `train_gws` is empty or includes their own gameweek or a later one are excluded and
    counted. Duplicate (gw, element) rows are refused.
  - **Secondary population (FPL_HGB_ALL_FINAL):** every FINAL row. These two arms cover all of them.
- **Metrics per arm, pooled and per gameweek:** MAE, RMSE, bias (forecast - actual), Spearman,
  top-10 overlap with the actual top 10, mean actual points of the forecast top 10, and captaincy
  regret. Regret = best actual score in the gameweek minus the actual score of the arm's top pick,
  in raw FPL points (a captain doubles both). Ties break by FPL element id.
- **Paired difference vs FPL:** a gameweek-cluster bootstrap 90% interval (2,000 resamples, fixed
  seed). With 6 gameweeks it is wide by construction.
- **Scheduling:** `fantasy_refresh.yml` runs the module daily after `fantasy_v2` and commits the
  JSON. The step is non-fatal and reads files only.
- **Live-score check:** 6 calls to FPL's free `event/{gw}/live` (GW5-GW10), run once, locally.
  The responses were not committed.

## 2. Numbers: identical population (968 player-gameweeks, GW5-GW10)

| Arm | MAE | RMSE | Bias | Spearman | Top-10 overlap (mean/10) | Captain regret (sum, pts) |
|---|---|---|---|---|---|---|
| FPL baseline (ep_next) | 1.7872 | 2.7566 | -0.1376 | 0.5729 | 2.50 | 66 |
| Wowza V1 | 2.4002 | 3.0696 | +1.1199 | 0.4186 | 1.50 | 79 |
| V2 LINEAR (shadow) | **1.7512** | **2.6158** | +0.0087 | **0.5994** | **2.67** | 73 |
| V2 RIDGE (shadow) | 1.7653 | 2.6242 | +0.0106 | 0.5898 | 2.50 | 69 |
| V2 HGB (shadow) | 1.7987 | 2.6460 | +0.0517 | 0.5717 | 2.00 | **63** |

**Difference vs FPL** (arm minus FPL; negative = better). The CIs are 90% gameweek-cluster
intervals.

| Arm | MAE diff | MAE 90% CI | RMSE diff | RMSE 90% CI | |bias| diff | GWs MAE < FPL | GWs regret < FPL |
|---|---|---|---|---|---|---|---|
| Wowza V1 | +0.613 | +0.526 to +0.739 | +0.313 | +0.213 to +0.387 | +0.982 | 0 / 6 | 0 / 6 |
| LINEAR | -0.036 | -0.085 to -0.008 | -0.141 | -0.251 to -0.059 | -0.129 | 5 / 6 | 1 / 6 |
| RIDGE | -0.022 | -0.053 to +0.002 | -0.132 | -0.214 to -0.070 | -0.127 | 3 / 6 | 2 / 6 |
| HGB | +0.012 | -0.042 to +0.055 | -0.111 | -0.199 to -0.043 | -0.086 | 4 / 6 | 3 / 6 |

**Per gameweek.** Each cell is MAE / captain regret.

| GW | n | FPL | Wowza V1 | LINEAR | RIDGE | HGB |
|---|---|---|---|---|---|---|
| 5 | 217 | 2.011 / 12 | 2.464 / 15 | 2.023 / 15 | 2.033 / 11 | 2.110 / 11 |
| 6 | 103 | 1.242 / 11 | 2.125 / 11 | 1.184 / 11 | 1.250 / 11 | 1.323 / 11 |
| 7 | 88 | 1.374 / 13 | 2.316 / 13 | 1.331 / 13 | 1.375 / 13 | 1.361 / 9 |
| 8 | 93 | 1.823 / 10 | 2.278 / 13 | 1.605 / 7 | 1.685 / 7 | 1.660 / 13 |
| 9 | 237 | 1.790 / 13 | 2.388 / 13 | 1.775 / 13 | 1.749 / 13 | 1.780 / 9 |
| 10 | 230 | 1.962 / 7 | 2.558 / 14 | 1.945 / 14 | 1.943 / 14 | 1.962 / 10 |

**Secondary population** (FPL vs HGB, all 2,631 FINAL rows):

| | MAE | RMSE | Bias | Spearman | Regret |
|---|---|---|---|---|---|
| FPL | 1.7671 | 2.8122 | -0.0515 | 0.5443 | 64 |
| HGB | 1.7166 | 2.5999 | +0.0497 | 0.5606 | 73 |

HGB vs FPL: MAE -0.051 (CI -0.126 to +0.020), RMSE -0.212 (CI -0.332 to -0.107). HGB had the lower
MAE in 3 of 6 gameweeks and the lower regret in 1 of 6.

**Cross-check (second way).** The primary pooled MAE, RMSE, bias and Spearman match the existing
benchmark's LEDGER `pooled_final_only` to 4 decimals for all five arms (e.g. FPL 1.7872 and LINEAR
1.7512). That population is the same 968 rows.

**Break-even.** LINEAR's MAE edge is 0.036 points per player-gameweek. Its CI excludes 0, but only 6
clusters stand behind it. The captaincy decision has the opposite sign: LINEAR lost 7 regret points
to FPL over 6 gameweeks: GW5 +3, GW8 -3 and GW10 +7. The whole net loss is one pick: Haaland in GW10,
regret 14 vs 7.
Promotion would need the MAE CI to stay below 0 **and** captaincy regret no worse than FPL over more
gameweeks. That rule has not been pre-registered; it is an owner decision.

## 3. Data audit

| Item | Source | Finding |
|---|---|---|
| Player IDs | FPL `element` | 674 players in the current bootstrap. 0 duplicate elements per archived gameweek (GW4-GW12). |
| Team IDs / club crosswalk | FPL `team_id`; `fpl_af_crosswalk.csv` | 606 FPL-to-API-Football links (534 name+club+age, 72 name+club). 0 duplicate ids on either side. 64 refusals, all "no proven candidate" (left unlinked, never guessed). The 190 rows whose club text differs are 6 spelling aliases only (Man City / Manchester City, Spurs / Tottenham, Nott'm Forest, Man Utd, Ipswich, Coventry). Crosswalk built 2026-11-02. |
| Transfers / current club | FPL bootstrap, refreshed daily | The team comes from the live FPL squad, not from history. |
| Fixture mapping | FPL `fixtures` | 380 fixtures, 0 without an event. |
| Double / blank gameweeks | FPL `fixtures` | **None so far this season.** No team had 2 fixtures in one event, and no event was missing a team. The DGW path (actual = cumulative difference, `n_fix`) is covered by `test_blank_gameweek_means_zero_minutes`, but no real DGW exists to validate it on. |
| Injuries / suspensions | FPL `status`, `chance_of_playing_next_round`, `news` | Status is never null (6,002 snapshot rows: a 4,227, u 949, i 588, d 203, s 34, n 1). `chance_next` is null on 3,019 rows, **all status 'a'**. That is FPL's convention for fully available players, now written into the defaults register. |
| Start probabilities / expected minutes | NaN-native minutes model fitted on FINAL gameweeks below g | Missing inputs give NaN, not a number. `p_start = P60 + 0.227 x P(1-59)`; 0.227 is measured (CI 0.213-0.241). The register still said "0.35, unfitted"; **corrected**. |
| Set-piece roles | FPL `penalties_order`, `corners_and_indirect_freekicks_order`, `direct_freekicks_order` | Taken from FPL only. A player FPL does not list is now shown as "not listed by FPL", never as a guessed role. `fantasy_set_pieces.csv` (58 rows) counts real penalty and set-piece events from API-Football history. A missing source column gives NaN. |
| xG / xA | FPL `expected_goals`, `expected_assists` | 0 nulls in 6,002 snapshot rows. |
| FPL prices | FPL `now_cost` | 0 nulls. |
| Historical FPL scoring | Snapshot cumulative differences | **Verified** against FPL `event/{gw}/live`: 2,631 of 2,631 points and minutes exact (GW5-GW10). |
| Captaincy | Highest forecast in the pool | Now scored as regret per arm (section 2). |
| Bench optimisation | `fantasy_optimizer.py` (squad solver) | The solver uses only projection columns. It was not re-validated against settled results; there is no settled-squad history. |

## 4. Defaults and certification

- **Missing-value defaults on the Fantasy path: gone.** The register (`FANTASY-DEFAULTS-1.0`, 43
  entries) has 15 MISSING_DATA_FALLBACK entries. 14 are FIXED. The one still open is
  `props_batch_median` (legacy props payloads only). The rule-2 scanner finds 0 new sites in the
  changed files. Two inline `.get(k, 0)` and one `or 0` added during this work were removed before
  commit.
- **I could not identify the "3 registered defaults on the old baseline" by name.** The register
  today has no open Fantasy-specific missing-input fill. What remains are 22 DOMAIN_PRIOR
  conventions, already labelled on the page.
- **Stale register text corrected (no behaviour change):**
  - `v2_p_start_from_mix`: 0.35 unfitted -> 0.227 measured, class TRAIN_FITTED_PRIOR.
  - `v2_chance_fill`: the note said the residual's prepare still imputes. It does not. The 'a' to
    100 rule is FPL's convention. Line reference updated.
- **Wowza V1 runs the props models on an uncertified frame. Flagged, not certified.**
  `build_fantasy_projections` scores `goals`, `assists` and `sot2` on the latest `player_history`
  row per player. The parity of that frame was not certified (P0_PROPS_PRODUCTION_CERTIFICATION,
  finding 3). Certifying it needs `player_history.parquet` and a serving-vs-training parity run, out
  of scope here. The dashboard's `V1_NOTE`, shown on every V1 tool, and the paired tab now say so.
  V1 reaches V2 only as the WOWZA arm and the shadow `wowza_gap` feature.

## 5. Changes on this branch

- `player_model/fantasy_paired_eval.py` (new): the identical-population chronological comparison.
- `output/fantasy_v2_paired_walkforward.json` (new): generated from the committed OOS frame.
- `.github/workflows/fantasy_refresh.yml`:
  - Runs the paired evaluation daily and commits its JSON.
  - Fixes a literal `\n` in the commit loop. It passed a bogus file `n` to `git add`; that one was
    skipped, and `fantasy_set_pieces.csv` was still staged.
- `pages/13_⚽_Fantasy.py`:
  - Model lab: new **Paired walk-forward** tab (table, per-gameweek chart and table, exclusions,
    chronology count).
  - Players: new **Player data & missingness** section. It has a per-field source and missing count,
    and a per-player table with FPL set-piece order, FPL chance and news, and the named missing
    inputs. Blanks stay blank.
  - The Benchmark caption now warns that ACTIVE arms cover different players.
- `dashboard_data/fantasy.py`: readers `v2_paired_walkforward`, `v2_paired_table`,
  `v2_paired_by_gw`.
- `dashboard_data/fantasy_view.py`: `missingness_summary`, `player_data_table`, `set_piece_role`;
  `V1_NOTE` flags the uncertified props frame.
- `player_model/fantasy_defaults.py`: the two register corrections above.
- `tests/test_fantasy_paired_walkforward.py` (new). It checks:
  - identical n per arm;
  - chronology exclusion;
  - that a future gameweek cannot move past forecasts;
  - duplicate refusal and provisional exclusion;
  - regret and top-k on known values;
  - the shadow pin;
  - the committed OOS and report;
  - the dashboard helpers and the V1 flag.

## 6. Blockers and owner questions

1. **Promotion: not requested, not done.** Should a pre-registered rule be set before more
   gameweeks accrue? Suggested: MAE CI upper bound < 0 **and** captain regret <= FPL over >= 10
   FINAL gameweeks.
2. **Wowza V1 frame certification** needs a parity run on `player_history.parquet`. Should V1 keep
   feeding the shadow residual (LINEAR/RIDGE depend on it, which is why they cover only 968 of 2,631
   rows), or should a V1-free residual be the shadow candidate? HGB covers every row.
3. **No DGW or BGW has occurred yet.** That path will be validated against `event/live` on the first
   one.
4. **Bench and squad optimisation** has no settled history to score.

Owner approval needed: only for items 1-2. Nothing on this branch changes a served number.
