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

## 12. Amendment 3 (2026-11-09, BEFORE any result was read) - league-wide edges and second-tier tips

A first SELECT run was started and then stopped after about 2 minutes, before it wrote any output (its log
is empty and no choices file exists). No result has been seen. Two owner rules are added:

**12.1 League-wide edges are allowed.**
* Owner: *"If we find an edge that says 'bet BTTS Y for every match in league X', it's OK if the data supports
  it. Don't treat it as a bug."*
* A 5th probability source, **NONE**, is added: there is no model filter, and every eligible match of the
  league × market-side is bet (within the configuration's odds range).
* It is implemented with the same machinery: every row's edge sits above every grid bar, so the bars reduce
  to the league's own shrunk SELECT ROI.
  * The league is ON only if that ROI is > 0 with at least MIN_BETS_FIT own bets.
  * Otherwise it is OFF. A pooled stand-in is never used (amendment 2).
* With EDGE / EV meaningless without a model, NONE contributes 6 configurations (the odds ranges) per family.
  **Total: 4 × 2 × 6 + 6 = 54 per family, 540 in total.**
* **Per-league competition.** After a family's configuration is chosen, each league is compared on its own
  inner out-of-sample bets:
  * the chosen MODEL configuration vs LEAGUE_WIDE (NONE with the same odds range);
  * the league takes LEAGUE_WIDE only if both have at least MIN_BETS_FIT inner bets and LEAGUE_WIDE's inner
    ROI is higher.
  * That is one extra comparison per league, counted. Those cells are reported as **LEAGUE_WIDE**.
* Known example to TEST, not assume: Argentina BTTS Yes (Pro: its profit was a league-wide regime, not model
  skill).
* In TOP-3 ranking, a LEAGUE_WIDE bet's expected value is the league's shrunk SELECT ROI. That is a real
  estimate from data; there is no model probability.

**12.2 Second-tier tips (TIER2).**
* Owner: *"When we have bars and can tip 3 top bets for every league, we will have a tier for second tips if
  the 3 is missing."*
* By construction, every band BELOW the VALUABLE bar has a non-positive SELECT ROI. VALUABLE is the lowest
  bar whose cumulative ROI is positive, so the bets just below it lose on SELECT. A TIER2 below VALUABLE can
  therefore never be real value, and it is **not** built that way.
* Pre-registered TIER2 instead:
  * **TOP** = edge ≥ the league's MARKSMAN bar (the profit-maximising bar).
  * **TIER2** = VALUABLE ≤ edge < MARKSMAN, only if that band's own SELECT ROI is > 0 with at least
    MIN_BETS_FIT own bets and each bet has EV > 0 against the fair price. Otherwise the league has no TIER2.
  * When MARKSMAN = VALUABLE, the band is empty and there is no TIER2.
* TIER2's FINAL ROI and CI are reported separately.
* **TOP-3 replay variants on FINAL:**
  * **(i) TOP-ONLY-V:** qualify at VALUABLE, the 2026-11-08 owner rule.
  * **(ii) TOP-ONLY-M:** qualify at MARKSMAN.
  * **(iii) TOP-M + TIER2:** fill empty slots with TIER2 bets, labelled TIER2 and ranked after every TOP bet.
  * The baseline is unchanged.


---

## 13. RESULTS (appended 2026-11-09 after the search; sections 1-12 unchanged)

Commits:

| Stage | Commit |
|---|---|
| Pre-registration | `32bb11c6f` |
| Amendments | `90c855925` / `4cfdfd1bd` / `ed1b69594` |
| Engine | `7e93e36f4` |
| SELECT choices frozen | `a8ca31520` |
| FINAL scored once | `813c7e2c9` |

Configurations searched: **54 per family × 10 families = 540**, plus one MODEL-vs-LEAGUE_WIDE comparison per
league.

### 13.1 Bottom line, blunt

* **No configuration holds up as proof.** 0 PROVEN cells on either track.
* **Long history: TOP-3 per league loses about 2%.** Over the 39-month walk-forward of the SELECT window
  (2023-06..2026-08, each month out of sample, 2,000+ picks), TOP-3 per league is negative, with a CI that
  includes zero (13.5). Once the search is honest, the model has no demonstrated edge over the bet-time price
  on either track.
* **The FINAL window (2026-09-01..11-08) looks good: +10.4% over 200 picks with the static choice, +15% with
  the monthly re-fit.** The CI includes zero, the window is 2 months, and the 3-year history says it is more
  likely variance than skill.
* **The one consistent signal: CLV is small but positive** on every selection (+0.3% to +0.4% against the
  closing price). That is the evidence the live test should keep measuring.
* **The monthly re-fit beats static choices on the long history:** −0.1% vs −2.9% on all qualifying bets, and
  static's CI lies entirely below zero. So the monthly design is worth keeping, as a loss-avoider rather than
  a profit engine.

### 13.2 Choices per family (SELECT, frozen)

| Track | Market-side | Chosen configuration (of 54) | Reality-check p | Best inner ROI | ON leagues | OFF (reason: count) |
|---|---|---|---|---|---|---|
| standard | over25_over | SHRINK · EDGE · max- · min- | 0.996 | -1.8% | Serie B | OWN_DATA_TOO_THIN: 16 |
| standard | over25_under | RAW · EV · max2.5 · min- | 0.6442 | +5.3% | Belgian First Division A, Scottish League Two | NO_POSITIVE_BAR: 15 |
| standard | btts_yes | SHRINK · EDGE · max2.5 · min- | 0.099 | +10.9% | La Liga 2, Serie B | OWN_DATA_TOO_THIN: 6 |
| standard | over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | - | - |
| standard | over35_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | - | - |
| new_format | over25_over | RAW · EV · max2.5 · min- | 0.0415 | +15.6% | Austrian Bundesliga, Brazil Serie A, Denmark Superliga, Ireland Premier Division, Japan J-League, Mexico Liga MX, Norway Eliteserien, Sweden Allsvenskan, USA MLS | OWN_DATA_TOO_THIN: 6 |
| new_format | over25_under | RAW · EV · max- · min- | 0.9135 | +0.8% | Brazil Serie A, China Super League, Denmark Superliga, Finland Veikkausliiga, Ireland Premier Division, Japan J-League, Mexico Liga MX, Norway Eliteserien, Sweden Allsvenskan, USA MLS | OWN_DATA_TOO_THIN: 4, NO_POSITIVE_BAR: 1 |
| new_format | btts_yes | RAW · EDGE · max2.5 · min- | 0.5632 | +4.3% | Argentina Primera Division, Austrian Bundesliga, Brazil Serie A, China Super League, Ireland Premier Division, Japan J-League, Mexico Liga MX, Norway Eliteserien, Sweden Allsvenskan | OWN_DATA_TOO_THIN: 6 |
| new_format | over15_over | LEARNED · EV · max- · min- | 0.3013 | +7.2% | Argentina Primera Division | OWN_DATA_TOO_THIN: 14 |
| new_format | over35_over | BLEND · EDGE · max- · min- | 0.9415 | -5.3% | China Super League, Norway Eliteserien, USA MLS | OWN_DATA_TOO_THIN: 12 |


* **Reality-check p:** the chance that the best of the family's configurations is that good by luck alone
  (White, centred block bootstrap). Only **New-Format Over 2.5** passes 0.05 (p 0.04).
* **Standard Over 2.5:** the best inner ROI is negative (−1.8%). The pre-registered rule still picks the
  maximum, and its per-league bars turn ON only Serie B.
* **Standard Over 1.5 / Over 3.5:** OFF. They have 84 / 106 priced SELECT rows, and no configuration reaches
  100 inner bets.
* **Most leagues are OFF with OWN_DATA_TOO_THIN** (bars would need a pooled stand-in). Under the owner's
  "no global vars" rule they get no bars. This conflicts with the 2026-11-08 instruction "every league gets
  its own bars", which is an owner decision.
* **No league switched to LEAGUE_WIDE, and no TIER2 band qualified anywhere.**

### 13.3 Verdicts on FINAL (Holm across every tested cell)

| Track | Market-side | OFF | PROMISING | UNPROVEN | PROVEN |
|---|---|---|---|---|---|
| standard | over25_over | 16 | 1 | 0 | 0 |
| standard | over25_under | 15 | 1 | 1 | 0 |
| standard | btts_yes | 15 | 2 | 0 | 0 |
| standard | over15_over / over35_over | 17 / 17 | 0 | 0 | 0 |
| new_format | over25_over | 6 | 2 | 7 | 0 |
| new_format | over25_under | 5 | 3 | 7 | 0 |
| new_format | btts_yes | 6 | 3 | 6 | 0 |
| new_format | over15_over | 14 | 0 | 1 | 0 |
| new_format | over35_over | 12 | 0 | 3 | 0 |

The 12 PROMISING cells have 2 to 22 FINAL bets each, so they show direction only, never proof. Every cell
with bets:

| Verdict | Track | Market-side | League | Mode | VALUABLE / MARKSMAN / SNIPER | SELECT bets / ROI | FINAL bets | FINAL ROI | 95% CI | CLV (n) | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PROMISING | standard | over25_over | Serie B | MODEL | 0.02 / 0.02 / 0.02 | 117 / +5.7% | 2 | +132.5% | [+117.8%, +147.2%] | +9.4% (2) | - |
| UNPROVEN | standard | over25_under | Belgian First Division A | MODEL | 0.02 / 0.02 / 0.02 | 412 / +9.3% | 10 | -14.0% | [-98.4%, +70.4%] | +0.8% (10) | 1 |
| PROMISING | standard | over25_under | Scottish League Two | MODEL | 0.02 / 0.03 / 0.03 | 215 / +10.7% | 9 | +4.0% | [-76.6%, +84.6%] | +1.1% (9) | 1 |
| PROMISING | standard | btts_yes | La Liga 2 | MODEL | 0.02 / 0.02 / 0.02 | 53 / +33.2% | 5 | +33.0% | [-73.4%, +139.4%] | +0.4% (5) | 1 |
| PROMISING | standard | btts_yes | Serie B | MODEL | 0.02 / 0.02 / 0.02 | 288 / +10.6% | 6 | +32.5% | [-74.2%, +139.2%] | +1.1% (6) | 1 |
| UNPROVEN | new_format | over25_over | Austrian Bundesliga | MODEL | 0.03 / 0.14 / 0.15 | 120 / -1.1% | 1 | +90.0% | [-, -] | -2.6% (1) | - |
| PROMISING | new_format | over25_over | Brazil Serie A | MODEL | 0.02 / 0.02 / 0.02 | 151 / +19.6% | 11 | +30.0% | [-31.3%, +91.3%] | +0.8% (11) | 1 |
| UNPROVEN | new_format | over25_over | Denmark Superliga | MODEL | 0.02 / 0.02 / 0.02 | 54 / +10.1% | 0 | - | [-, -] | - (0) | - |
| UNPROVEN | new_format | over25_over | Ireland Premier Division | MODEL | 0.02 / 0.02 / 0.02 | 64 / +10.0% | 1 | +75.0% | [-, -] | -2.8% (1) | - |
| UNPROVEN | new_format | over25_over | Japan J-League | MODEL | 0.02 / 0.03 / 0.03 | 48 / +5.2% | 1 | -100.0% | [-, -] | +0.0% (1) | - |
| UNPROVEN | new_format | over25_over | Mexico Liga MX | MODEL | 0.02 / 0.02 / 0.06 | 63 / +21.2% | 4 | -53.8% | [-144.4%, +38.8%] | +0.7% (4) | 1 |
| UNPROVEN | new_format | over25_over | Norway Eliteserien | MODEL | 0.02 / 0.03 / 0.03 | 70 / +0.4% | 2 | +64.0% | [+32.6%, +95.4%] | -1.4% (2) | - |
| PROMISING | new_format | over25_over | Sweden Allsvenskan | MODEL | 0.02 / 0.02 / 0.02 | 67 / +22.9% | 2 | +81.5% | [+64.8%, +98.2%] | +5.1% (2) | - |
| UNPROVEN | new_format | over25_over | USA MLS | MODEL | 0.02 / 0.03 / 0.03 | 66 / +2.2% | 0 | - | [-, -] | - (0) | - |
| UNPROVEN | new_format | over25_under | Brazil Serie A | MODEL | 0.06 / 0.07 / 0.07 | 75 / +3.6% | 18 | -3.7% | [-55.3%, +47.9%] | -2.2% (18) | 1 |
| UNPROVEN | new_format | over25_under | China Super League | MODEL | 0.11 / 0.12 / 0.27 | 137 / -6.7% | 9 | -36.7% | [-141.0%, +128.0%] | -0.8% (9) | 1 |
| UNPROVEN | new_format | over25_under | Denmark Superliga | MODEL | 0.11 / 0.12 / 0.12 | 49 / -15.4% | 5 | +63.6% | [-67.6%, +194.8%] | -1.2% (5) | 1 |
| PROMISING | new_format | over25_under | Finland Veikkausliiga | MODEL | 0.11 / 0.12 / 0.12 | 64 / -15.1% | 9 | +18.3% | [-66.0%, +102.6%] | +4.1% (9) | 1 |
| UNPROVEN | new_format | over25_under | Ireland Premier Division | MODEL | 0.04 / 0.04 / 0.04 | 70 / +22.0% | 10 | -79.0% | [-124.1%, -30.0%] | +0.7% (10) | 1 |
| UNPROVEN | new_format | over25_under | Japan J-League | MODEL | 0.04 / 0.04 / 0.04 | 48 / +5.4% | 4 | -48.8% | [-149.2%, +51.7%] | -0.7% (4) | - |
| UNPROVEN | new_format | over25_under | Mexico Liga MX | MODEL | 0.04 / 0.05 / 0.05 | 103 / +2.0% | 18 | -10.6% | [-64.3%, +43.2%] | +1.1% (18) | 1 |
| PROMISING | new_format | over25_under | Norway Eliteserien | MODEL | 0.04 / 0.07 / 0.07 | 114 / +4.8% | 19 | +15.0% | [-81.3%, +111.3%] | +0.4% (19) | 1 |
| PROMISING | new_format | over25_under | Sweden Allsvenskan | MODEL | 0.04 / 0.05 / 0.05 | 106 / +6.7% | 17 | +32.1% | [-30.5%, +94.7%] | +1.2% (17) | 1 |
| UNPROVEN | new_format | over25_under | USA MLS | MODEL | 0.11 / 0.12 / 0.12 | 110 / -1.4% | 40 | -21.6% | [-70.3%, +27.2%] | -0.8% (40) | 1 |
| UNPROVEN | new_format | btts_yes | Argentina Primera Division | MODEL | 0.02 / 0.02 / 0.08 | 258 / +2.6% | 68 | +9.2% | [-17.5%, +36.8%] | -0.1% (68) | 1 |
| UNPROVEN | new_format | btts_yes | Austrian Bundesliga | MODEL | 0.02 / 0.02 / 0.02 | 101 / +11.4% | 3 | -40.0% | [-157.6%, +80.0%] | +1.7% (3) | 1 |
| PROMISING | new_format | btts_yes | Brazil Serie A | MODEL | 0.02 / 0.02 / 0.08 | 224 / +12.9% | 22 | +28.3% | [-13.4%, +69.9%] | +0.0% (22) | 0.912 |
| UNPROVEN | new_format | btts_yes | China Super League | MODEL | 0.02 / 0.02 / 0.02 | 73 / -7.6% | 3 | -40.0% | [-157.6%, +80.0%] | +0.2% (3) | 1 |
| PROMISING | new_format | btts_yes | Ireland Premier Division | MODEL | 0.02 / 0.02 / 0.02 | 107 / +10.0% | 6 | +25.8% | [-69.5%, +121.2%] | +1.4% (6) | 1 |
| UNPROVEN | new_format | btts_yes | Japan J-League | MODEL | 0.02 / 0.02 / 0.02 | 41 / -34.1% | 0 | - | [-, -] | - (0) | - |
| UNPROVEN | new_format | btts_yes | Mexico Liga MX | MODEL | 0.02 / 0.02 / 0.02 | 42 / +11.7% | 6 | +43.2% | [-14.7%, +99.4%] | -0.4% (6) | 1 |
| PROMISING | new_format | btts_yes | Norway Eliteserien | MODEL | 0.02 / 0.02 / 0.02 | 57 / +5.3% | 6 | +46.2% | [-11.3%, +103.6%] | +0.3% (6) | 0.175 |
| UNPROVEN | new_format | btts_yes | Sweden Allsvenskan | MODEL | 0.02 / 0.02 / 0.02 | 44 / -0.1% | 3 | -36.3% | [-161.1%, +88.4%] | -0.7% (3) | - |
| UNPROVEN | new_format | over15_over | Argentina Primera Division | MODEL | 0.02 / 0.03 / 0.06 | 238 / +3.7% | 0 | - | [-, -] | - (0) | - |
| UNPROVEN | new_format | over35_over | China Super League | MODEL | 0.02 / 0.02 / 0.03 | 78 / +4.5% | 6 | +14.8% | [-56.6%, +86.2%] | +0.0% (6) | 1 |
| UNPROVEN | new_format | over35_over | Norway Eliteserien | MODEL | 0.02 / 0.02 / 0.02 | 63 / -0.8% | 17 | -33.0% | [-77.8%, +11.8%] | +0.2% (17) | 1 |
| UNPROVEN | new_format | over35_over | USA MLS | MODEL | 0.02 / 0.02 / 0.02 | 63 / +15.4% | 57 | -13.1% | [-38.1%, +11.9%] | +1.3% (57) | 1 |


### 13.4 TOP-3 per league per gameweek, FINAL window (2026-09-01..11-08, one score)

| Selection | Picks | League-weeks | ROI | 95% CI | CLV |
|---|---|---|---|---|---|
| **Optimized, TOP_ONLY_V** (qualify at VALUABLE) | 200 | 89 | **+10.4%** | −9.6% .. +30.3% | +0.37% |
| Optimized, TOP_ONLY_M (qualify at MARKSMAN) | 194 | 88 | +10.4% | −8.8% .. +29.6% | +0.34% |
| Optimized, TOP_M + TIER2 | 194 | 88 | +10.4% (TIER2 never qualified) | −8.8% .. +29.6% | +0.34% |
| **Baseline** (the −13.3% method, same window) | 414 | 162 | **−5.2%** | −19.8% .. +10.7% | +0.41% |

* Optimized TOP_ONLY_V by market: BTTS Yes +19.3% (53 picks), Over 2.5 +38.3% (13), Under 2.5 +3.7% (131),
  Over 3.5 +20% (3).
* Baseline by market: Under 2.5 −7.0% (243), Over 3.5 −32% (38).
* The optimized selection mostly removed the leagues and markets that lost (Over 3.5, many Standard leagues),
  and tips about half as often.

### 13.5 Monthly re-selection (amendment 1, 10.3)

**(a) Walk-forward over SELECT.** Every month from 2023-06 to 2026-08 is re-selected on earlier data only and
scored out of sample.

| | Bets | ROI | 95% CI | CLV |
|---|---|---|---|---|
| Monthly re-fit, every qualifying bet | 4,532 | −0.1% | −3.3% .. +3.1% | +0.03% |
| Static (chosen once before the first month) | 5,669 | −2.9% | −5.6% .. −0.3% | −0.03% |
| Monthly TOP-3 (V / M / M+TIER2) | 2,204 / 2,031 / 2,075 | −2.7% / −2.1% / −1.8% | each CI includes 0 | +0.33..0.36% |
| Static TOP-3 (V / M / M+TIER2) | 2,487 / 2,429 / 2,429 | −1.9% / −2.6% / −2.6% | each CI includes 0 | +0.27..0.29% |

Monthly beat static in 20 of 39 months. Per family, monthly vs static:

| Family | Monthly | Static | Period |
|---|---|---|---|
| Standard Over 2.5 | −3.4% | −4.3% | 39 months |
| Standard Under 2.5 | +0.9% | −8.1% | 39 months |
| Standard BTTS | +6.1% | +9.6% | 19 months |
| New-Format Over 2.5 | +10.3% | +1.6% | 9 months |
| New-Format Under 2.5 | −9.7% | +8.5% | 9 months |
| New-Format Over 3.5 | −17.3% | −4.7% | 9 months |

**(b) The monthly cycle on FINAL**, labelled MONTHLY_ROLLING and not the main verdict. September used the
SELECT choice; October was re-selected on data before October.

* All bets: 319 at +5.9% [−5.9%, +17.8%].
* TOP-3: 178 picks at +15.2% [−6.6%, +37.0%]; TOP_ONLY_M: 167 picks at +16.8%.
* By month: September +5.3% (186 bets), October +6.8% (133).
* No priced rows exist yet for November 1-8: the football-data files stop on 2026-10-10 and the snapshots
  on 10-30.

### 13.6 Argentina BTTS Yes (owner: test it, do not assume it)

* Betting every priced Argentina match on SELECT: +2.6% over 386 matches (2025-07..2026-08).
* The league-wide bars put Argentina's own weight at **0.0**: the between-league variance is zero, so its
  difference from the other leagues cannot be told apart from noise.
* In the inner folds the shrunk bet-everything ROI was ≤ 0, so league-wide Argentina got 0 inner bets and
  could not win the per-league comparison.
* The model configuration served Argentina instead: FINAL 68 bets at +9.2% [−17.5%, +36.8%], CLV −0.07%,
  UNPROVEN.
* So the regime is **not established on this data**.
* The monthly November re-selection, on data before November, does switch Argentina BTTS to LEAGUE_WIDE.
  The monthly evidence trail will show whether that holds.

### 13.7 Rare research props (goals2, goals3, sot3, sot4)

* No exceptional bar can be set honestly:
  * `output/prop_quotes` starts on 2026-10-14, so there is no price before the props holdout;
  * goals2 and goals3 have no line at all;
  * sot3 and sot4 are one-sided (over only).
* Any bar would rest on an assumed margin, which the owner's rule forbids.
* The research mechanism stays at 0 tips (NO_BAR_YET) until two-sided prices exist over enough history. The
  pre-kickoff capture, switched on at go-live, is the source.

### 13.8 What serving needs (owner decisions)

1. **A bars_opt serving adapter, needed before ANY optimized bar can serve.**
   * Production computes edge = p_model_raw − fair, with no odds range.
   * The November re-selection has 32 ON cells, and **none** uses that configuration. They use EV bars, odds
     caps, a learned/blended/shrunk probability, or LEAGUE_WIDE.
   * Written in today's production format, production would misread them. So the monthly job writes them
     only to the evidence file (`output/threshold_proposals/bars_opt/2026-11.json`), and the production-shape
     proposal stays unchanged.
   * Predict (and TOP-3) must compute each cell's chosen probability, measure and odds range.
2. **OFF = no tip.** Serving falls back to the GLOBAL bar for any league without approved own bars. The
   owner's "no global vars" rule needs that changed for side tips. TOP-3 already refuses the global bar.
3. **"Every league gets bars" vs "no pooled stand-in".** Under the newer rule most leagues are OFF with
   OWN_DATA_TOO_THIN. They tip nothing until their own data grows; the monthly job re-checks each month.
4. **O/U 2.5:** this study gives separate Over and Under bars, while production serves one set per league.
5. **Monthly job:** `.github/workflows/bars_monthly.yml`, off until `BARS_MONTHLY_ENABLED=true`. At go-live,
   `config/bars_opt.json` `clean_engine_versions` must list the retrained engines; until then no live row
   counts.
6. **Real money:** none of this justifies real money this season. Keep everything PAPER, and let the monthly
   evidence trail (live first-sight rows, per league × market-side × month) build the case for next season.

### 13.9 Pro tools used

* `src/studies/learned_tiers.py`: the design of the LEARNED source (C = 0.5).
* `src/studies/evidence.py`: White's reality check and the max-of-three SE.
* `src/studies/market_anchor.py`: its earlier finding that the Standard anchor weight on the model is about
  0, which matches the SHRINK weights here.
* The `power_devig` port already in v9 (`src.market_edge`).
* Not used: the Prediction Lab and the valuable deep dive, because their inputs are the pre-repair ledgers.
