# P4 - Bars optimization study (PRE-REGISTRATION, then results)

Owner, 2026-11-09: *"u have 2 weeks to find the optimize bars for every league for every market."*
Goal: per-league tier bars and a TOP-3-per-league-per-gameweek selection that are profitable **out of sample**,
for go-live on 2026-11-21. Everything here is a PROPOSAL: nothing writes `models/best_params_*.json` or
`config/threshold_approvals.json`.

**Status of this section:** written and committed BEFORE any search was run (pre-registration). Results are
appended below it in later commits; this section is not edited after the search starts.

## 1. Data

* Predictions: the clean Phase 2 retrain, candidate `champion_recipe`, deployable rows, both stages
  (`walk_forward` + `holdout`).
  * Standard: GitHub run 42512834177, `run_20261108T190758Z/predictions.parquet`.
  * New-Format: local run `run_20261108T214340Z/predictions.parquet` (same code 804799a1, committed data and
    approved holdout; run locally because the GitHub job died twice in its comparison step).
* Bet-time prices, from `src/league_bars.py` unchanged:
  * Standard O/U 2.5: football-data pre-match same-book pair (Bet365, else Pinnacle).
  * Every other priced cell: the earliest provable pre-kickoff same-snapshot two-sided pair.
  * Fair probability = power de-vig. Clubs are matched only through `src.team_names.resolve_identity`.
  * A row with no pair is no bet. The closing price is never an input; it is used only to measure CLV
    afterwards.
* Markets (each betting **one side**, so every cell is league × market × side):
  `over25_over`, `over25_under`, `btts_yes`, `over15_over`, `over35_over`.
  * HT Over 0.5 and HT Over 1.5 have no price at any time, so they are NO_PRICE_DATA and not searched.

## 2. Time split (fixed)

| Window | Dates | Use |
|---|---|---|
| SELECT | every priced row dated **before 2026-09-01** | all choices: configuration, bars, blend and shrink parameters, ON/OFF |
| FINAL CHECK | **2026-09-01 .. 2026-11-08** | scored **once**, after every choice is frozen and committed |

* The FINAL window is never read by the selection code. A test enforces it.
* Inside SELECT, configurations are compared by **nested expanding time-series validation**:
  * The priced SELECT matchdays of each track × market cell are cut into 5 equal consecutive blocks.
  * For k = 2..5: fit everything on blocks < k, then bet block k.
  * The inner score is the pooled out-of-sample ROI over blocks 2..5.
* Cells priced only from 2026-04-01 (Standard Over 1.5 / Over 3.5) have a SELECT window of Apr–Aug 2026
  only. That is weaker evidence, and it is labelled.
* **The Clean Record from 2026-11-21 is the only truly fresh test.** The FINAL window shares its season with
  the data the models and the earlier bars saw, so it is a check, not proof of the future.

## 3. Search space (complete list; every configuration is counted)

Per track × market × side, one configuration = a × b × d:

| Dimension | Options | Count |
|---|---|---|
| a. Probability source | **RAW**: `champion_recipe` p. **BLEND**: logistic on [logit p_model, logit fair_entry], fitted on earlier rows only (entry prices, never closing). **SHRINK**: p = fair + w·(p_model − fair), w per league learned on earlier rows (log-loss optimal, empirical-Bayes pooled toward the cell's w). | 3 |
| b. Edge measure | **EDGE** = p − fair; **EV** = p·odds − 1 | 2 |
| d. Odds range | max odds ∈ {none, 2.50, 3.50} × min odds ∈ {none, 1.40} | 6 |

* 36 configurations per cell family; 5 market-sides × 2 tracks → **360 configurations**.
* Within a configuration, **c. bars per league** (VALUABLE ≤ MARKSMAN ≤ SNIPER) come from
  `src/league_bars.choose_bars`, unchanged:
  * the same coarse grid;
  * empirical-Bayes shrinkage toward the pooled league curve;
  * no global bar, no typed constant.
* **f. Market inclusion:** a league × market-side is **OFF** when its shrunk SELECT ROI curve has no bar with
  ROI > 0 (no VALUABLE bar exists) under the chosen configuration.
* **e. Sharp-money agreement:** **excluded, not possible.**
  * `output/sharp_ledger.csv` starts on 2026-06-14, so it covers under 3 months of the SELECT window, mostly
    the off-season.
  * Most of its signals are 1X2.
  * There are no signals before 2026-06.
  * Using it would mean a window too short to select on.
* **Not searched**, because it is not reproducible on historical predictions: the engine's `ev_lower_bound`
  (its uncertainty band exists only for live first-sight rows). Option b uses EDGE vs EV instead.

## 4. Choosing and multiple testing

* For each cell family (track × market-side), the configuration with the **highest inner out-of-sample ROI**
  is chosen. It must have at least 100 inner out-of-sample bets; with fewer, the RAW / EDGE / no-cap
  configuration is kept.
* The selection's own multiple testing is reported with **White's reality check** over that family's 36
  configurations: a block bootstrap over match dates of the inner out-of-sample P/L, testing whether the best
  configuration's ROI > 0 is more than the maximum of 36 noisy ones.
* The FINAL check tests **one frozen configuration per cell**. The search multiplicity is handled by the
  data split. The per-league tests on FINAL are corrected with **Holm across every league × market-side ×
  track** tested.

## 5. Verdicts (per league × market-side × track, on FINAL)

| Verdict | Rule |
|---|---|
| **PROVEN** | ≥ 150 FINAL bets AND ROI 95% CI above 0 after Holm (one-sided Holm-adjusted p < 0.025). The owner's strict rule. |
| **PROMISING** | Not PROVEN, FINAL ROI > 0, AND CLV > 0 where a closing price is measurable. Clearly labelled: direction only, no proof. |
| **UNPROVEN** | Bars exist, but neither of the above. |
| **OFF** | No VALUABLE bar on SELECT (nothing held). |
| **NO_PRICE_DATA** | No real two-sided bet-time price at any time. |

* CLV = odds taken / closing odds − 1. The closing price is:
  * football-data's same-book closing columns for Standard O/U 2.5;
  * the latest pre-kickoff same-snapshot pair for snapshot cells.
* CIs: block bootstrap over match dates, 2,000 resamples.

## 6. TOP-3 replay (FINAL window)

* Per league × ISO week (the GW-1 rule approximated by the match date's ISO week), every non-OFF cell's
  candidates at or above its VALUABLE bar are ranked by EV under the chosen configuration.
* The direction rule: on one fixture, Under 2.5 never stands beside Over 2.5, BTTS Yes, Over 1.5 or
  Over 3.5. The higher-ranked tip wins.
* At most 3 per league-week. **Never padded.**
* **Baseline on the same FINAL window:** RAW / EDGE / no caps, bars chosen on SELECT, every cell ON (the
  method behind the −13.3% replay).
* Reported: picks, ROI, 95% CI, CLV, by market and league, optimized vs baseline.

## 7. Rare research props (goals2, goals3, sot3, sot4)

* Owner: "2-3 max tips in a GW, only if the formulas find awesome opportunities".
* An exceptional bar needs a reliable bet-time price over enough history. Checked before the search:
  * `output/prop_quotes` starts 2026-10-14;
  * goals2 and goals3 have no line at all (the goals market is anytime only);
  * sot3 and sot4 (over 2.5 / 3.5) have one-sided prices only.
* So no exceptional bar can be set honestly yet. Section 9 reports what's missing; no number is invented.

## 8. What would count as success

* A configuration whose TOP-3 FINAL replay has ROI > 0 with a CI excluding zero would be a real result.
* ROI > 0 with a CI including zero is direction only.
* If neither happens, the honest conclusion is that no bar setting makes the current models profitable,
  and the report says so plainly.
