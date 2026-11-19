# P2 - Props and Fantasy: remaining invented values (Phase 2, key `propsdef`)

Audit run 2026-11-05 on branch `phase2/propsdef` (from `origin/data-truth-100` f66ebbe5). Production is
`origin/main` f9127e10. Every number below was recomputed today from committed files. Nothing was
retrained, backtested, re-thresholded, sent or staked. Props are `BLOCKED_FROM_REAL` in
`config/bankroll_policy.json` (paper only), so no props row has ever been a real stake.

## Verdict

- **The regex scanner is clean on the props/Fantasy path.** `src.integrity.data_truth` rule 2 finds
  23 sites in the 49 props/Fantasy/Telegram/dashboard files on data-truth-100. All 23 are allow-listed
  counters or display code, with reasons. On `main` the same patterns find **282** sites in 43 files.
- **Reading the code found what the regex cannot see.** There are four classes:
  1. **Tip probabilities adjusted by typed formulas and written as `model_prob`.** In the committed
     board, **184 of 2,332** tips that can be joined to the model's logged raw output (7.9%) carry a
     `model_prob` **lowered** by the quality-mismatch discount (ratio 0.338 to 0.997). Of those, 5 are not
     AVOID (4 VALUABLE, 1 PAPER). **Fixed beside the original:** each tip now also carries
     `model_prob_raw` and `prob_adjustment`. The arithmetic is unchanged.
  2. **Fair probability from an assumed overround** (`config.OVERROUND_BY_ODDS` 1.06 / 1.10 / 1.15).
     This affects every priced tip: 554 on the committed board and 5,539 priced ledger rows, including all
     1,177 tiered rows (P/L as stored: -231.34u). The tier gates read it. **It cannot be replaced by real
     data today:** none of the 42,395 quote keys in `output/prop_quotes/` has both sides. **Labelled, not
     removed:** a priced tip now carries `fair_basis = "assumed_overround"`. Replacing it would change
     tiers, so it is listed as an owner question.
  3. **Fantasy figures built from typed coefficients.** These include the V1 bonus formula, the opponent xG
     from FDR (0.5 + 0.42 x FDR), the FDR step and home edge, the V2 3- and 5-gameweek horizon shares, the
     availability half-life, and the clip bounds. All are listed in section 4. They are conventions, not
     missing-value fills, so they were **registered, not removed**. Removing them would delete the
     product figures. That is an owner question.
  4. **Model-side.** These are the props batch-median imputer and the 7 profile features that the frozen
     props payloads learned from placeholders. They are left for the Phase 2 retrain. See
     `reports/P9_MODEL_IMPUTATION.md`.
- **Fixed on this branch (no retrain needed):** (a) `model_prob_raw` / `prob_adjustment` / `fair_basis`
  beside the tip; (b) the dashboard Fantasy availability split no longer reads an unknown start
  probability as 0 = "unavailable"; (c) a dead legacy Fantasy fixture path with a typed x1.10 / x0.92
  home factor was removed; (d) the Fantasy defaults register was brought up to date with the code.
  It had 2 stale "open" entries and was missing 11. 6 new tests were added; each fails on f66ebbe5.

## 1. Method

- **Regex:** `src.integrity.data_truth.scan_file` (all 7 rule-2 patterns) over `player_model/*.py`,
  `telegram_bot/*.py`, `dashboard_data/props_view.py`, `dashboard_data/fantasy*.py`, and
  `pages/11_*` / `pages/13_*`. I ran it on both branches. `main` was extracted with `git archive origin/main`.
- **Reading:** I read `predict.py` in full. I read the value-producing parts of `feature_engineering.py`,
  `pipeline.py`, `fantasy.py`, `fantasy_v2.py`, `fantasy_residual.py`, `fantasy_panel.py`,
  `fantasy_defaults.py`, `model.py` and `props_contract.py`, the Telegram props and Fantasy senders, and
  the dashboard props and Fantasy views. I also grepped for `fillna(`, `clip(`, `median()`, `min/max` with
  constants, `else 0`, `or 1` and literal multipliers.
- **Reach measured on committed data:** `output/player_tips.csv` (2,905 rows, match dates 2026-11-02 to
  2026-11-09), `output/player_ledger.csv` (33,677 rows, 2026-06-14 to 2026-11-02),
  `output/props_r2_shadow.csv` + `output/model_lineage/props_shadow_h2h.csv` (52,809 logged raw model
  probabilities, 2026-11-01T11:46Z to 2026-11-02T01:48Z), `output/prop_quotes/*.csv` (13 files,
  281,657 quotes), `output/fantasy_tips.csv` (283), `output/fantasy_v2_projections.csv` (455, GW10).
  The committed `output/` on data-truth-100 is a snapshot of production from about 2026-11-02 06:00 UTC
  (see PHASE2_CURRENT_BASELINE).

## 2. Findings: props path (data-truth-100)

"Reach": T = tips/tier, S = stake, D = display (dashboard/Telegram), TR = training.

| # | file:line (dt100 HEAD) | What is invented | Reach | Rows / tips touched (committed) | Status |
|---|---|---|---|---|---|
| P1 | `player_model/predict.py` `hand_adjusted_prob` (was inline at 941-961) | `model_prob` = model output x team-strength multiplier (typed piecewise: 0.60 / 0.40 bands, x1.30 max, x0.40 floor, assists x0.70) x quality discount (career_q / max(opp_q, 0.1) when < 0.60, goals/SOT markets) | T (EV, rel_edge, tier, PAPER rank), D (Telegram %, dashboard "Fair odds" = 1/model_prob), CLV log | Board: **184 / 2,332** joinable tips (7.9%), all moved down (ratio 0.338 to 0.997); 179 AVOID, 4 VALUABLE, 1 PAPER; markets sot 146, goals 29, sot2 8, assists 1. The team-strength multiplier fires only where `worldcup_history.json` has a full 1X2 price, so on this board it fired on 1 tip at most (the assists row). Ledger: **UNIDENTIFIABLE**, because ledger rows are first-sight from earlier runs and 66 of the joined rows moved *up* by up to x1.986, which no hand factor can produce | **FIXED beside**: `model_prob_raw`, `prob_adjustment` on every new tip; `model_prob` unchanged. Owner question Q1 |
| P2 | `predict.py:_devig_fair_prob` + `config.py:120` | Fair probability = (1/odds) / an overround **assumed** by odds band (1.06 / 1.10 / 1.15), not a de-vigged two-sided price | T (rel_edge gates SNIPER/MARKSMAN/VALUABLE), D | Board: 554 priced tips (504 AVOID, 45 VALUABLE, 5 PAPER). Ledger: 5,539 priced rows; 1,177 tiered (VALUABLE 996, MARKSMAN 165, SNIPER 16), stored P/L -231.34u. Two-sided quotes available: **0 of 42,395** quote keys (sides logged: yes 212,093, over 69,398, no 150, under 16) | **Labelled**: `fair_basis = "assumed_overround"` on priced tips. Owner question Q2 |
| P3 | `predict.py:_compute_market_caps` | Rows with unknown minutes are excluded from "starts" (`fillna(0) >= 60`); with no minutes column it falls back to all appearances | T (the cap gate) | 0 measurable: minutes are present in history | Not a value written anywhere. A missing value excludes the row. Kept |
| P4 | `predict.py:_kelly_stake` / `kelly_stake = 0.0` for AVOID | 0.0 means "no stake" for a non-bet | S (paper) | n/a | Real rule, not data. Kept |
| P5 | `predict.py` GES (`compute_ges`, `feature_engineering.py:1716-1737`) | Normalisers 0.50 / 3.5 / 1.5 / 85 / 0.35 and weights 0.40 / 0.20 / 0.15 / 0.15 / 0.10 | T (goals/SOT gate) | Every goals/SOT tip | A typed **score formula** with measured inputs (NaN if any is missing). Not a fill. Reported for Phase 2 (Q3) |
| P6 | `feature_engineering.py:971-1023` composite features | Typed weights and normaliser constants (e.g. `passes_pg / 50`, `x 0.4`, `.clip(upper=1.0)`) | TR + serving features | All rows | Feature definitions, shared by training and serving (parity), not fills. Model-side, Phase 2 |
| P7 | `player_model/model.py:43` `_prep` | NaN filled with the **scoring-batch median** (legacy props payloads) | T | UNIDENTIFIABLE on committed data (the props feature frame is not committed). Last record: one injected NaN moved 16 of 120 rows (`player_props_integrity_health.json`) | **Model-side, left for the retrain**. See P9_MODEL_IMPUTATION section 1 |
| P8 | `props_contract.py:156-182` | R2 challenger imputer: training medians, persisted | Shadow only (`props_r2_shadow.csv`) | 0 tips (shadow) | Model-side. P9 |
| P9 | `models/model_player_*.pkl` `feature_cols` | All 9 serving payloads still list the 7 profile features (`age`, `age_peak_delta`, `height_cm`, `height_aerial_interaction`, `opp_def_mean_height`, `height_diff_vs_opp_def`, `height_advantage_score`). They were trained on the old placeholders (age 25, 180 cm, 75 kg on **876,130 of 876,130** rows of the committed `player_history.parquet`). dt100 serving now builds them from real profiles (46,190 of 51,723 collected profiles carry a height) or NaN. **Training and serving now differ for these 7 columns.** | T | UNIDENTIFIABLE (no committed feature frame). A column that was constant in training should carry ~0 weight. `height_aerial_interaction` was not constant: it was 0.5 x `aerial_won_rate` | Model-side. `config.PROFILE_FEATURES_WITHHELD` already drops them from the next training. Retrain item |
| P10 | `pipeline.py:770,788` | Comment and log still said assists tips are tiered "via calibration base rates" (the synthetic price was removed 2026-11-01) | D (CI log) | 0 | **FIXED** (text only) |

**Checked and clean (no invented value found):** `data_fetcher.py` (stat parsing keeps None; ids are
never minted from hashes), `api_football.py`, `odds_fetcher.py` (no outcome price means None, never 0;
"No" sides are never read as the tip's price), `prop_odds_af.py`, `league_quality.py` (NaN without real
cross-league matches), `ledger.py`, `clv_tracker.py`, `enrich_no_odds_markets` (no synthetic assists
price), `capped_prob` (no cap written as a probability), `_confidence_score` (measured components only;
the gate is OFF), the referee feature (measured z-score; the typed 3.5 is gone), `_build_match_contexts`
(no context without all three 1X2 prices), the Telegram props alert and daily picture (they show
`model_prob`, `market_odds` and `ev` as stored; non-bet tiers are dropped before any count), and
`dashboard_data/props_view.py` and the Player Props page (edge = model_prob - 1/odds, computed from
stored values).

## 3. main (live) vs data-truth-100 (goes live)

The same scanner and a named-signature count were run on both trees. `main` = f9127e10; the committed
board on main was generated 2026-11-05 13:41Z.

| Invented value | main (live) | data-truth-100 | Go-live |
|---|---|---|---|
| Rule-2 regex sites, props/Fantasy path | **282** (numeric_get_default 135, or_zero 99, fillna_literal 47, fillna_zero_price 1) | 23, all allow-listed | removes 259 |
| Probability above the cap written AS the cap (`min(p_model, cap, 0.95)`, main `predict.py:768`) | yes. On main's live board, 18 rows sit on a repeated per-market maximum: goals 10 at 0.685, sot2 5 at 0.756, assists 3 at 0.467. 5 of them are VALUABLE. That run's caps are not committed, so "= cap" is consistent but **UNIDENTIFIABLE** strictly. Historic: 376 ledger rows listed in `output/corrections/props_sentinel_prob.csv` (dt100) | `capped_prob`: no tip | removes |
| Win probability default 0.5 (main `predict.py:601`) | yes | NaN, multiplier skipped | removes |
| De-vig of no price returns 0.5 (main `predict.py:104`) | yes | NaN | removes |
| Profile placeholders written (`height_cm = 180`, main `feature_engineering.py:257`), rating prior 7.0 (main `:102`, `:1067`), opponent quality priors 6.8 / 7.0 / 0.50 with clip 0.2-4.0 (main `:1069`), per-rate `fillna(0.0)` (main `:54`-`:215`, 55 sites) | yes | NaN wherever an input is missing | removes |
| Fantasy V1 minutes NaN -> 60 (main `fantasy.py:232`), p_start NaN -> 0.6 (main `:376`) | yes; main's live `fantasy_tips.csv` (138 rows) has 2 rows with p_start exactly 0.6 (whether they are fills is UNIDENTIFIABLE) | player skipped / p_start NaN | removes |
| Fantasy V2 minutes heuristic `max(start_rate.fillna(0), 0.35)`, `x 0.92` (main `fantasy_residual.py:667-668`) | yes | player's own measured shares, NaN when missing | removes |
| Synthetic assists price | already removed on main too | removed | nothing |
| **Hand adjustments written as `model_prob` (P1)** | yes | yes, now with `model_prob_raw` / `prob_adjustment` beside it (this branch) | **remains**, labelled |
| **Assumed overround fair probability (P2)** | yes | yes, now with `fair_basis` (this branch) | **remains**, labelled |
| Dashboard availability split, unknown p_start -> 0.0 (`dashboard_data/fantasy.py:166`) | yes | yes until this branch | removed by this branch |
| Dead V1 fixture path, home x1.10 / x0.92 | yes (dead) | yes (dead) until this branch | removed by this branch |
| Batch-median imputer (P7), profile-feature train/serve gap (P9) | batch median yes; P9 does not apply (main serves placeholders, the same as training) | both | **remains** until the Phase 2 retrain. **Note: go-live creates P9.** dt100 serving feeds real heights and ages into payloads trained on constants |
| Fantasy typed coefficients (section 4) | yes, plus the fills above | yes | **remains**. Owner question Q4 |

## 4. Fantasy: real data vs typed constants

Owner rule: a Fantasy figure without real inputs is missing, not estimated. On data-truth-100 every
**missing-input** fill is gone: unknown minutes, start rate, position, FDR, status, chance or team
produce NaN, a skipped player or an "unknown" label. The register is
`player_model/fantasy_defaults.py` (`FANTASY-DEFAULTS-1.0`, now 40 entries). What remains is
**typed coefficients applied to real inputs.**

### V2 (product: `fantasy_v2_projections.csv`, Fantasy page, 455 rows, GW10)

| Output | Built from | Typed constant in it? |
|---|---|---|
| `final_xpts` (headline), `fpl_ep_next` | FPL's own `ep_next` (provider value). The residual is **not applied**: `residual_applied` = True on 0 of 455 rows; `final_xpts == fpl_ep_next` on 455 of 455 | **REAL** (provider) |
| `wowza_residual`, `v2_candidate_xpts`, `adj_*` | Ridge residual fitted on FINAL gameweeks (leave-one-GW-out alpha) | REAL (fitted), shadow only |
| `p_0`, `p_1_59`, `p_60`, `p_play`, `exp_minutes` | NaN-native minutes classifier fitted on the panel; conditional minutes from training GWs (NaN if a bucket never occurred) | REAL (fitted). Hyperparameters are unfitted (`v2_minutes_model_C`) |
| `p_start` | `p_60 + 0.227 x p_1_59`. 0.227 was measured (CI 0.213-0.241, re-tested in `test_fantasy_fpl_rules.py`) | REAL (measured) |
| `goal_prob`, `assist_prob` | FPL xG90 / xA90 x expected minutes x attack fixture factor | Bound: fixture factor clipped 0.4-2.5 (`v2_attack_fixture_clip`) |
| `cs_prob`, `team_cs_prob`, `team_goal_exp` | Team lambdas measured as-of the deadline (home advantage measured) | Bound: lambdas clipped 0.2-4.5 (`v2_lambda_clip`). FWD `cs_prob` NaN by rule (52 rows) |
| `exp_bonus` | Player's own bonus per start x p_60 | REAL |
| `median`, `p25`, `p75`, `p90`, `ceiling`, `sd`, `p_blank`, `p6`, `p10`, `p15` | Mixture of the fitted minutes buckets and a fitted negative-binomial dispersion | Bound: the 60+ component mean is clipped 1.5-16 (`v2_m60_mean_clip`) |
| `xpts_3gw`, `xpts_5gw`, `value_5gw_per_m` ("Best value" cards) | rate x availability path x fixture multiplier | **TYPED**: position attack/defence shares (`v2_attack_defence_share`), long-run play = start_rate + 0.25 (`v2_long_play_rate`), half-life 1 GW (`v2_availability_half_life`), multiplier clip 0.55-1.6 (`v2_fixture_mult_clip`). Missing inputs give NaN |
| `uncertainty_score`, `uncertainty`, `confidence`, `risk`, captain pools | Distribution sd / mean and p_start | **TYPED thresholds** (0.5 floor; 45 / 60 cut points; p_start 0.85 / 0.60 / 0.75; 0.7 / 0.8 x best). These are labels, not figures |

### V1 (`fantasy_tips.csv`, 283 rows; feeds the V2 shadow residual through `fantasy_projection_log.csv`; `telegram_bot.notifier.send_fantasy_tips` has no workflow caller)

| Output | Typed constant in it? |
|---|---|
| `p_goal`, `p_assist`, `p_sot2` | Props models (the model-side issues P7 / P9 apply) |
| `attack_pts` | FPL goal/assist points (real rules) + `p_sot2 x 0.45` (`sot2_weight_0_45`, an lstsq fit that is not reproducible in CI) |
| `dc_pts` | FPL DC thresholds (real) on FPL's own DC count per 90; Poisson |
| `cs_prob`, `cs_pts` | **TYPED**: opponent xG = 0.5 + 0.42 x FDR (`fdr_heuristic_xg`); 0.6 conceded + 0.4 FDR (`cs_lambda_blend`); conceded rate outside 0.3-3.5 ignored (`v1_cs_plausibility_band`) |
| `bonus_pts` | **TYPED** weights 1.5 / 0.9 / 0.3 / 0.8 / 0.25 x 0.66, bounded 0-3 (`v1_bonus_formula`, `bonus_scale_0_66`) |
| `fantasy_pts`, `xpts_rot`, `xpts_uncond`, `value`, `overall_rank`, `captain_pick` | Sum of the above, so they inherit the typed parts. 283 of 283 rows carry a value |
| `p_start`, `start_confidence` | 0.6 recent + 0.4 season start rate (`p_start_blend_0_6_0_4`, typed); FPL chance / injured applied |
| `fixture_adj_pts`, `total_xpts_next`, `avg_fdr` | **TYPED**: x(1 + 0.10 x (3 - FDR)) x 1.05 home / 0.95 away, clip 0.6-1.4 (`v1_fdr_step_home_edge`, env-overridable) |

**Summary.** The figures that rest **only** on real or fitted inputs are: V2 `final_xpts`,
`fpl_ep_next`, the minutes buckets, `p_start`, `exp_bonus` and the residual (shadow). Bounded by a
typed clip: V2 `goal_prob`, `assist_prob`, `cs_prob` and the distribution quantiles. **Built on typed
coefficients:** V2 `xpts_3gw`, `xpts_5gw` and `value_5gw_per_m`, and every V1 points figure. On this
branch no Fantasy figure is built from a typed stand-in for a **missing** input.

## 5. Changes on `phase2/propsdef`

| File | Change |
|---|---|
| `player_model/predict.py` | New `hand_adjusted_prob()`, which returns (p, factor, quality_measured) with the same arithmetic in the same order. Every tip carries `model_prob_raw` (the model's own probability, 4 dp) and `prob_adjustment` (1.0 = none) beside `model_prob`. `enrich_with_odds` sets `fair_basis = "assumed_overround"` on a priced tip. **No tip, tier, EV or probability value changes.** |
| `dashboard_data/fantasy.py` | `availability_split`: an unknown p_start is counted as `unknown` / `unknown_in_top20`, not as unavailable. Committed data has 0 unknown p_start rows today, so the displayed counts do not change. |
| `player_model/fantasy.py` | Removed the dead `build_fantasy_projections_fixtures` and `_team_defense_map` (no caller; typed x1.10 / x0.92 home factor, clip 0.6-1.5). They were the only caller of `upcoming_by_team`. That function writes no value (it returns the API-Football fixture list, which is empty off-season) and stays in place, now with no caller. |
| `player_model/fantasy_defaults.py` | `referee_cards_3_5` and `season_hardcoded` are marked FIXED (the code had already fixed them). Line references were corrected. Added `v1_cs_plausibility_band`, `v1_bonus_formula`, `v1_fdr_step_home_edge`, `v1_home_factor_legacy` (DEAD, removed), `v2_fixture_mult_clip`, `v2_availability_half_life`, `v2_m60_mean_clip`, `v2_attack_fixture_clip` and `v2_lambda_clip`. Descriptive only. |
| `player_model/pipeline.py` | Comment and log text: no "calibration base rate" assists pricing. |
| `tests/test_props_defaults_propsdef.py` | 6 tests. All 6 fail on f66ebbe5 and pass here. The arithmetic test compares `hand_adjusted_prob` bit-for-bit with the old inline code. |

`player_tips.csv` gains three trailing columns. `player_ledger.csv` (read by Pro) is **unchanged**, so
no cross-repo contract moves.

## 6. Open questions for the owner

- **Q1 (P1):** should a tip's tier and EV use the model's own probability (`model_prob_raw`) or keep
  using the hand-adjusted one? The quality discount lowered 184 tips on this board, 5 of them tiered.
  Changing it alters tiers, so I made no change.
- **Q2 (P2):** the fair probability rests on an assumed overround, and we hold no two-sided prop
  quotes. Options: (a) keep it, now labelled; (b) collect the Under / No side where books offer it
  (OddsAPI sends it for some markets); (c) no fair probability means no tier. Option (c) would make
  every priced props tip AVOID today.
- **Q3:** the GES score (P5) and the V1/V2 Fantasy coefficients (section 4) are typed formulas on real
  inputs. Should they be fitted in the Phase 2 retrain, or dropped, or kept as stated conventions?
- **Q4:** should V2 `xpts_3gw` / `xpts_5gw` / `value_5gw_per_m` (shown as "Best value" on the Fantasy
  page) be marked as convention-based, or hidden until the shares and half-life are fitted?

## 7. Owner decisions 2026-11-05 (branch `phase2/propsdecide`)

Implemented on `phase2/propsdecide` from `origin/data-truth-100` 976dc9e9. Nothing was retrained,
backtested or re-thresholded; no API was called. Props stay PAPER-only.

### Q1: the model's own probability decides

`run_player_predictions` now caps, floors and writes the model's own probability as `model_prob`
(`model_prob_raw` repeats it at 4 dp). Tier, EV, edge and Kelly read it. `hand_adjusted_prob` is still
computed and logged beside the tip as `model_prob_hand_adjusted` and `prob_adjustment` (information
only). `player_ledger.csv` keeps its fixed column list.

Measured on the committed board (`output/player_tips.csv`, 2,905 rows, written 2026-11-02 04:52Z by
**main's** code), joined to the logged raw model output (`props_r2_shadow.csv` p_served +
`props_shadow_h2h.csv` p_champion_raw, latest per market/fixture/player; all 2,905 rows join):

* 271 rows carry a `model_prob` below the raw output beyond 3-dp rounding (0 above). **8 of them are
  main's `min(p, cap)` sentinel, not the hand adjustment**: the run's cap values goals 0.685 (6 rows),
  sot2 0.761 (1), assists 0.467 (1), each with a raw output above it. They are the 4 VALUABLE + 1 PAPER
  section 2 counted as hand-adjusted. data-truth-100 drops them (`capped_prob`) under either rule, so
  they are left out of the comparison below. The other **263** are the hand adjustment (all AVOID).
* Tier recomputed with the branch's own functions (`enrich_with_odds`, `tag_paper_feed`, `capped_prob`,
  `MIN_SIGNAL_PROB`); the board's own tiers reproduce on all 554 priced rows first. Caps: the run's
  goals / sot2 / assists values above, the others from the committed `player_history.parquet`.

| | AVOID | VALUABLE | PAPER | not tipped (raw above cap) |
|---|---|---|---|---|
| before (hand-adjusted decides) | 2,851 | 41 | 5 | 0 |
| after (model's own probability) | 2,844 | 46 | 5 | 2 |

**9 tips change tier**: 5 AVOID to VALUABLE (L. Fernandez sot and sot2, D. Sanches Fernandes goals,
J. Taylor goals, J. Moreno goals), 1 AVOID to PAPER (J. Moylan sot) and 1 PAPER to AVOID (N. Krstovic
goals, displaced from the top-5 EV), 2 AVOID no longer tipped because the raw output is above the goals
cap (L. Fernandez goals 0.752, V. Osimhen goals 0.701). Cross-checked a second way (an independent
script with `_classify_tier` directly): same counts. Not measurable from committed files: player-markets
whose hand-adjusted probability fell under `MIN_SIGNAL_PROB` (never on the board) and may now appear.

### Q2: both sides collected, real two-sided price first

* OddsAPI and API-Football quotes already logged every side. New: `prop_odds_af.side_word` names
  Yes / No as well as Over / Under; an API-Football "No" side is kept in the quote log and never read as a
  tip price (it was read as the Yes side if the provider sent one). The pre-kickoff capture keeps every
  value row and now names Yes / No in `side_word` (`PREMATCH_CAPTURE_CONTRACT.md`).
* `odds_fetcher` builds real pairs per fixture (same book, same answer, same line, one price per side),
  `match_fair_to_tips` resolves each tip to the same quoted player as its price and takes the median over
  books of the power de-vig (`src.market_edge.power_devig`, as the side markets). `enrich_with_odds`
  uses it (`fair_basis = "two_sided_devig"`); only without one does the assumed overround apply
  (`fair_basis = "assumed_overround"`).
* Labels: "two-sided price" / "margin assumed" on the Telegram instant alert and briefing, the daily props
  picture ("Fair price" column) and the dashboard Player Props page (board, cards, raw table).
* Committed quotes (`output/prop_quotes/`, 13 day files, 281,657 quotes): **139 real same-answer pairs**,
  all Pinnacle anytime-scorer Yes/No via OddsAPI, in 38 fixtures (match dates 2026-10-14 to 2026-11-01).
  Section 2's "0 of 42,395" did not hold on this count. The 16 API-Football "under" goals rows are the
  old substring bug ("Noah Sa s"), not Under prices. Committed board: 0 of 554 priced tips have a pair
  (its fixtures are 2026-11-02 to 11-09; the last pair is 11-01), so no board tip changes from Q2 today.
  The quote log stores a price only the first time it is seen, so pairs whose one side did not move
  are under-counted there; the live run pairs from the full answer.

### Q4: Fantasy convention figures labelled

The Fantasy page shows a note listing the fitted / real figures and the conventions. `xpts_3gw`,
`xpts_5gw`, `value_5gw_per_m` (Best value cards, explorer columns, 3/5-GW transfer gains, squad solve)
and every V1 points figure (V1 tools, chip advisor, lineup, mini-league) carry "convention-based, not
fitted on data". No value changes.

Tests: `tests/test_props_owner_decisions_20261105.py`.
