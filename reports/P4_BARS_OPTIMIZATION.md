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

## 10. Amendment 1 (2026-11-09, BEFORE any search was run) - owner direction: monthly, live test this season

Owner: *"optimized as u can, and from month to month. This season it's the live test with research live and
Pro tools. Next season is the big deal. Right now we need to optimize as best as we can."*
Added BEFORE any configuration was evaluated; sections 1-9 are unchanged except as stated here.

**10.1 Search space, maximum effort.**
* A 4th probability source is added: **LEARNED**.
  * It is ported from Pro `src/studies/learned_tiers.py` (owner 2026-10-22, "it must be per league").
  * It fits an L2-regularised logistic regression of WIN on logit(fair), logit(p_model), the raw edge,
    log odds, league one-hot, and league × raw edge.
  * It is fitted only on earlier rows; the L2 penalty is the per-league shrinkage.
* Configurations: 4 × 2 × 6 = **48 per cell family, 480 in total**.
* The multiple-testing rules of section 4 are unchanged and apply to all 480.

**10.2 Statistics reused from Pro** (`src/studies/evidence.py`, ported as pure functions, credited in the code):
* White's reality check: max over configurations, centred block bootstrap over match dates.
* The SE used for intervals: the LARGEST of independent, matchday-block and ISO-week-block SEs.
* The anytime-valid normal-mixture confidence sequence on monthly live ROI and CLV, which may be read after
  every bet.

**10.3 Monthly rolling re-optimization is part of the design.**
* At the start of each month M, configurations and bars per league × market-side are re-selected:
  * on every priced row dated before M;
  * with the same nested protocol;
  * then frozen and served for month M.
* Month M is scored as fresh out-of-sample evidence.
* Two results are reported:
  * **(a) Walk-forward simulation over SELECT.** For every month from the first month with at least 6 months
    of earlier priced data in that cell family, up to August 2026: the monthly re-fit is chosen only on
    earlier data, against STATIC choices (made once, on data before the first simulated month). Both are
    scored on the same months, out of sample.
  * **(b) The monthly cycle on the FINAL window, as a design check.**
    * September is scored with the SELECT choice.
    * October is scored with a re-selection on data before October.
    * November 1-8 is scored with a re-selection on data before November.
    * Each month is scored out of sample, but (b) reads FINAL months for the later re-selections. So the
      **main verdict stays the STATIC frozen-on-SELECT result of section 5**, and (b) is reported separately,
      labelled MONTHLY_ROLLING.
* A scheduled monthly job writes PROPOSALS through the existing `src/threshold_proposals` monthly path
  (`output/threshold_proposals/<YYYY-MM>`). The owner approves and `src/threshold_apply` applies.
  * It is off unless a repository variable is set.
  * Its data: the retrain predictions plus the live first-sight rows after them.

**10.4 Framing.**
* From 2026-11-21 this season is the **LIVE TEST**: research tracks run live beside the Pro tools.
* Each month's live results per league × market-side × month are the evidence trail that can justify real
  money next season.
* Nothing in this study is called profitable unless its out-of-sample interval says so.

## 11. Amendment 2 (2026-11-09, BEFORE any search was run) - the owner's biggest rule

Owner: *"100% real data, no fakes, no bugs, no defaults, no global vars."* This overrides any optimization gain.

* **Prices.** Only real, provable two-sided pairs are used.
  * There are no assumed margins and no synthetic prices.
  * The closing price is used only to measure CLV, and only where a real closing pair exists. Otherwise CLV
    is missing and counted as missing.
* **No default or fallback values.**
  * A row missing a price, a result or a probability is excluded and counted.
  * Section 4's fallback ("keep RAW/EDGE/no-cap when no configuration has 100 inner bets") is **withdrawn**.
    Such a cell family is **OFF** with the reason `INSUFFICIENT_SELECTION_EVIDENCE`.
* **No global bar and no pooled stand-in served to a league.** Shrinkage toward the pooled curve is allowed
  only as a learned weight inside a league's own estimate, and that weight is reported per league.
  * A league whose bars would come only from the pooled curve (`src/league_bars` basis `SHRUNK_TO_POOL`,
    fewer than MIN_BETS_FIT = 40 own selection bets at every bar) gets **no bars**.
  * Such a league is **OFF** with the reason `OWN_DATA_TOO_THIN`, its own counts reported. It would otherwise
    serve a pooled stand-in.
  * This tightens the 2026-11-08 instruction "every league gets its own bars". The owner is told which leagues
    it affects.
* **Every constant that decides a tip is declared here and reported:**

| Constant | Value |
|---|---|
| Bar grid (`scripts/league_thresholds.GRID`) | 0.02..0.30 in steps of 0.01 |
| MIN_BETS_FIT (own selection bets for a bar) | 40 |
| Odds caps | max {none, 2.50, 3.50} × min {none, 1.40} |
| Inner validation blocks | 5 (4 scored) |
| Minimum inner bets to choose a configuration | 100 |
| LEARNED L2 strength | C = 0.5 (Pro's frozen value) |
| SELECT / FINAL dates | as in section 2 |
| PROVEN rule | ≥ 150 bets, Holm one-sided p < 0.025 |
| Bootstrap resamples | 2,000 |
| Monthly simulation start | at least 6 months of earlier priced data |

* `python -m src.integrity.data_truth --gate` runs before every commit of this study.
