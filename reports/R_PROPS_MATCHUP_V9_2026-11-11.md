# Props matchup research (v9): inventory, absence effect, player bars

Date: 2026-11-11. Branch `research/props-matchup`. **Research only. Nothing changes in serving.** The owner decides any use.
Code: `scripts/research_props_matchup.py`. Tests: `tests/test_research_props_matchup.py`.
Pre-registration: `config/experiments/props_matchup/protocol.json`, sha256
`bae12cec60d191dab9ebb87904fd0881329aa16925fc79668607102b202266be`. It was committed (cdfde34da) before any absence effect was computed.
Result tables (counts and estimates only, no prices): `reports/props_matchup_data/`.

## In plain language

* **Opponent absences.** We cannot show that a missing opposing goalkeeper or defender beats the served model.
  * The clean data (absence lists stamped before kickoff) covers 125 fixtures in 5 leagues.
  * Only 2 of the 20 per-league cells are big enough to read, and both point the wrong way.
  * The bigger post-match data (context only, because it contains hindsight) shows pooled effects of +0.3 to +1.0 percentage points. Every interval crosses zero.
  * 0 of 158 per-league tests survive Holm.
* **Player bars.** No player × league × market cell is close to a bar. The largest cell has **4** settled, really priced tips; the minimum is 150. One player plays about once a week, so a player cell alone cannot reach 150 within a season. Player bars can only come from shrinking toward the league × market cell, and no league × market cell has 150 priced bets yet. The first one is projected for 2026-11-23 (Brazil Serie A SOT).
* **What blocks the clean test.**
  * The v9 pre-kickoff capture has never run: `prematch_capture.yml` is only on `data-truth-100`, and GitHub returns 404 for it on the default branch.
  * Pro's injury capture has written **0 pre-kickoff rows on every run since 2026-11-06 19:32Z**, although its runs succeed (for example, run 42889493488: `pre_kickoff_rows: 0`).

## Inputs (exact versions)

| Input | Version | Clock |
|---|---|---|
| Served-model probabilities | GitHub run 42855107089 artifact `props_harness_holdout_scored.csv.gz`, `p_hgb_nan_native`, holdout 2026-09-01..10-31, fitted through 2026-08-31. The sot pickle sha256 `36b0db6e…` equals the registry's `clean_retrain.artifact_sha256` of the served champion | out-of-sample |
| Labels | the same file (`y`, verified counts, all_prior_verified rule) | post-match outcome |
| STRICT absence lists | Pro `main` 40633097, `output/studies/injury_snapshots.parquet`: rows with `pre_kickoff`, and `captured_at` earlier than the final kickoff in `af_ht_scores`. The latest pre-kickoff list per fixture is used | **before kickoff, proven** |
| RETRO_INJ lists | v9 `output/collect/injuries.parquet` (fetched 2026-11-05..09) | after the match: context only |
| RETRO_XI | v9 `output/collect/lineups/` (fetched after the match). A regular is absent when he is not in the XI | after the match: context only |
| Regulars and positions | v9 lineups (earlier fixtures only), `player_history` positions | strictly earlier fixtures |
| Ledger, prices | v9 `origin/main` 62f4940cf: `player_ledger.csv`, `player_prop_odds_history.csv` + `_archive/` (682,909 price rows) | snapshot_ts |

Pro's own forward check (run 42889493488) now measures 15.1% of a fixture's final injury list as added after kickoff (125 fixtures). The 4 Nov spec said 18.8% (82 fixtures). In both cases the post-match lists contain hindsight.

## 1. Inventory (settled ledger player-fixtures, SOT 1+ / SOT 2+ / anytime scorer, match date ≥ 2026-10-14)

Column definitions:
* **settled**: WIN or LOSS.
* **ledger_priced**: the ledger row carries a price. The other rows are "unpriced — result only, no stake".
* **(a) a_price_real**: a stored snapshot, taken before the final kickoff, for the same date, player and market carries exactly the ledger's price.
* **(b) b_absence_strict**: the opponent's side is proven, and its injury list was observed before kickoff.
* **(c) c_both_strict**: both (a) and (b).

| league | settled | ledger_priced | (a) price real | of which API-Football | (b) absence strict | absence retro (context) | (c) both |
|---|---|---|---|---|---|---|---|
| Argentina Primera Division | 354 | 169 | 30 | 30 | 0 | 0 | 0 |
| Austrian Bundesliga | 308 | 52 | 17 | 17 | 0 | 0 | 0 |
| Brazil Serie A | 654 | 285 | 180 | 180 | 244 | 342 | **80** |
| Bundesliga | 585 | 103 | 6 | 0 | 0 | 433 | 0 |
| Bundesliga 2 | 658 | 50 | 25 | 25 | 0 | 0 | 0 |
| Champions League | 494 | 61 | 10 | 0 | 0 | 0 | 0 |
| Championship | 849 | 319 | 59 | 59 | 444 | 554 | **39** |
| China Super League | 87 | 5 | 5 | 5 | 0 | 0 | 0 |
| Conference League | 238 | 16 | 2 | 2 | 0 | 0 | 0 |
| Denmark Superliga | 307 | 69 | 3 | 3 | 43 | 73 | 2 |
| Europa League | 298 | 100 | 25 | 25 | 0 | 0 | 0 |
| Finland Veikkausliiga | 144 | 2 | 0 | 0 | 0 | 0 | 0 |
| Ireland Premier Division | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| Japan J-League | 141 | 3 | 0 | 0 | 0 | 0 | 0 |
| La Liga | 554 | 56 | 24 | 0 | 0 | 405 | 0 |
| La Liga 2 | 562 | 197 | 104 | 104 | 0 | 0 | 0 |
| League One | 278 | 19 | 2 | 2 | 0 | 0 | 0 |
| League Two | 195 | 19 | 1 | 1 | 0 | 0 | 0 |
| Ligue 1 | 441 | 146 | 11 | 0 | 0 | 312 | 0 |
| Ligue 2 | 444 | 33 | 6 | 6 | 0 | 0 | 0 |
| Mexico Liga MX | 423 | 125 | 8 | 8 | 0 | 0 | 0 |
| Norway Eliteserien | 471 | 45 | 3 | 3 | 165 | 215 | 3 |
| Premier League | 695 | 209 | 33 | 0 | 0 | 518 | 0 |
| Romanian Superliga | 184 | 1 | 0 | 0 | 0 | 0 | 0 |
| Serie A | 767 | 307 | 80 | 0 | 0 | 598 | 0 |
| Serie B | 714 | 121 | 31 | 31 | 0 | 0 | 0 |
| Sweden Allsvenskan | 557 | 33 | 5 | 5 | 168 | 196 | 5 |
| USA MLS | 648 | 109 | 20 | 0 | 0 | 510 | 0 |
| **All** | **12,078** | **2,654** | **690** | **506** | **1,064** | **4,156** | **129** |

What is too thin:
* **(c) exists in only 5 leagues and totals 129 settled bets.** By market: sot 76, goals 43, sot2 10. No league × market cell reaches the 150 minimum.
* **(b)** is limited to Pro's 5 leagues. Lineups with player ids observed before kickoff: **0 everywhere**.
* **(a)**: 1,964 priced, settled rows have **no stored snapshot** for that date, player and market. Their price cannot be proven to be the bet-time price, so they are not counted. The largest gaps are Championship 260, Serie A 227, Premier League 176, Argentina 139, Ligue 1 135 and Liga MX 117.
* The provider has **no injury lists at all** for several leagues: Argentina, La Liga 2, Serie B, Bundesliga 2, Ligue 2, Liga MX and others (`injuries_seasons.coverage_injuries` false; retro column = 0). Even with the capture switched on, these leagues could only get a lineup clock.
* Rare markets (goals2/goals3/sot3/sot4): 24 settled rows in total. They are excluded here (research-only by owner rule).

## 2. Opponent GK / regular-defender absence vs the served model

**Definitions.** These were registered before any result was computed:
* **Window.** The defending team's last 10 fixtures with a lineup, dated strictly before the match. With fewer than 5 such fixtures, the team's regulars are unknown and the exposure stays missing.
* **Regular players.**
  * Regular GK: the G with the most starts, if he started ≥ 60% of the window.
  * Regular defenders: every D who started ≥ 60% of the window.
  * API-Football gives G/D/M/F only, so centre-backs and full-backs cannot be told apart. No role was invented.
* **Absent** means "Missing Fixture" on the latest list observed before kickoff. "Questionable" is not counted.
* **Residual** = y − p_served, for outfield players.
* **Effect** = mean residual of exposed rows − mean residual of unexposed rows, within a league.
  * CI: 95% fixture-cluster bootstrap (4,000 draws).
  * Test: one-sided.
  * Multiple testing: Holm over **all 158 per-league tests searched** (all three clocks in one family).
  * A cell with fewer than 10 exposed or 10 unexposed fixtures is printed but not read. It still counts in the family, entering with p = 1.

### STRICT clock (the only decision-grade clock). Overlap with the holdout: 2026-10-19..31

| market | exposure | league | exposed fixt. | unexposed fixt. | exposed rows | effect | 95% CI | Holm p |
|---|---|---|---|---|---|---|---|---|
| sot | gk_out | Brazil Serie A | 2 | 21 | 17 | −0.006 | too thin | – |
| sot | gk_out | Championship | 1 | 25 | 13 | +0.283 | too thin | – |
| sot | gk_out | Denmark Superliga | 2 | 8 | 26 | −0.018 | too thin | – |
| sot | gk_out | Norway / Sweden | 0 | 11 / 17 | 0 | – | – | – |
| sot | def_out | Brazil Serie A | 12 | 19 | 168 | −0.037 | [−0.116, +0.037] | 1.00 |
| sot | def_out | Championship | 8 | 25 | 108 | +0.011 | too thin | – |
| sot | def_out | Denmark Superliga | 3 | 8 | 39 | +0.005 | too thin | – |
| sot | def_out | Norway Eliteserien | 4 | 11 | 53 | −0.012 | too thin | – |
| sot | def_out | Sweden Allsvenskan | 1 | 17 | 12 | +0.014 | too thin | – |
| goals | gk_out | Brazil Serie A | 2 | 21 | 17 | −0.066 | too thin | – |
| goals | gk_out | Championship | 1 | 25 | 13 | −0.022 | too thin | – |
| goals | gk_out | Denmark Superliga | 2 | 8 | 26 | +0.024 | too thin | – |
| goals | def_out | Brazil Serie A | 12 | 19 | 168 | −0.039 | [−0.084, +0.004] | 1.00 |
| goals | def_out | Championship | 8 | 25 | 108 | +0.013 | too thin | – |
| goals | def_out | Denmark Superliga | 3 | 8 | 39 | +0.005 | too thin | – |
| goals | def_out | Norway Eliteserien | 4 | 11 | 53 | −0.023 | too thin | – |
| goals | def_out | Sweden Allsvenskan | 1 | 17 | 12 | +0.009 | too thin | – |

The Championship SOT GK row (+0.283) is **one fixture**. A first version of the code printed a narrow CI for it, because a bootstrap over a single fixture has no spread. That was caught during development, and the read minimum now blocks it (test `test_one_exposed_fixture_is_too_thin_for_a_ci`).

### Pooled estimates (context only, never a decision)

| clock | market | exposure | exposed fixt. | exposed rows | effect | 95% CI |
|---|---|---|---|---|---|---|
| STRICT | sot | gk_out | 5 | 56 | +0.057 | too thin |
| STRICT | sot | def_out | 28 | 380 | −0.013 | [−0.062, +0.036] |
| STRICT | goals | gk_out | 5 | 56 | −0.014 | too thin |
| STRICT | goals | def_out | 28 | 380 | −0.013 | [−0.045, +0.021] |
| RETRO_INJ | sot | gk_out | 61 | 792 | −0.002 | [−0.031, +0.027] |
| RETRO_INJ | sot | def_out | 364 | 5,601 | +0.001 | [−0.012, +0.013] |
| RETRO_INJ | goals | gk_out | 61 | 792 | +0.010 | [−0.008, +0.027] |
| RETRO_INJ | goals | def_out | 364 | 5,601 | +0.005 | [−0.004, +0.014] |
| RETRO_XI | sot | gk_out | 555 | 8,037 | +0.005 | [−0.006, +0.015] |
| RETRO_XI | sot | def_out | 1,574 | 29,804 | +0.003 | [−0.004, +0.011] |
| RETRO_XI | goals | gk_out | 555 | 8,037 | +0.003 | [−0.004, +0.010] |
| RETRO_XI | goals | def_out | 1,574 | 29,804 | +0.003 | [−0.002, +0.008] |

Per-league readable cells by clock:

| clock | readable cells | positive | raw CI above 0 |
|---|---|---|---|
| STRICT | 2 | 0 | 0 |
| RETRO_INJ | 26 | 14 | 0 |
| RETRO_XI | 98 | 54 | 5 |

The best cells are RETRO_XI Liga MX sot def_out (+0.054, raw p 0.001) and RETRO_XI Premier League sot gk_out (+0.063, raw p 0.003). After Holm their p-values are 0.20 and 0.43. 5 raw hits out of 98 is what chance alone produces. Full table: `reports/props_matchup_data/absence_effects.csv`.

**Reading.** The served model already prices most of whatever an absence carries, or the residual effect is below about 1 percentage point. That is far under what any testable sample can detect (next table).

### Power (approximate, from the pooled standard errors; `power.csv`)

The 150-bet PROVEN minimum is a floor, not a guarantee of power:
* At 150 exposed rows, the minimum detectable effect (one-sided 5%, 80% power) is **≈ 5–13 percentage points**.
* Detecting **+1 pp** needs about **4,000–27,000** exposed rows per cell.
* Detecting **+2 pp** needs about **1,000–6,800**.

On the model-residual question alone, a per-league cell will not be PROVEN this season unless the true effect is several points. The pooled point estimates are 0.3–1.0 points. The open question with real upside is H-PM-5: whether the **price** lags the news, tested at a takeable price.

## 3. Bars per player × league × market

* **Data today.** Settled tips with a real price, SOT/SOT2/scorer, since 2026-10-14: 586 player cells.
  * Largest cell: **4**.
  * Cells with ≥ 30: **0**. Cells with ≥ 150: **0**.
  * The most active player adds 1 priced tip a week, so a single player cell needs ≥ 150 weeks. **No player bar can be fitted, and none was fitted.**
* **League × market parents** (the level a player cell shrinks toward). Trailing 28-day rate of really priced, settled tips:

| league | market | now | per week | reaches 150 |
|---|---|---|---|---|
| Brazil Serie A | sot | 104 | 26.0 | 2026-11-23 |
| La Liga 2 | sot | 61 | 15.3 | 2026-12-21 |
| Brazil Serie A | goals | 58 | 14.5 | 2026-12-25 |
| Serie A | sot | 47 | 11.8 | 2027-01-11 |
| Championship | sot | 39 | 9.8 | 2027-01-29 |
| La Liga 2 | goals | 37 | 9.3 | 2027-02-04 |
| Serie A | goals | 29 | 7.3 | 2027-03-07 |
| Bundesliga 2 | goals | 25 | 6.3 | 2027-03-31 |
| Premier League | goals | 23 | 5.8 | 2027-04-14 |
| Serie B / Argentina | sot | 21 | 5.3 | 2027-05-02 |

All rows: `accrual_league_market.csv`.

Caveat: the ledger has accrued only about 20 rows a day since 2026-11-05 (≈ 800 a day before). If that stays, these dates slip. They assume the trailing 28-day rate.

* **Fix that changes the timeline.** Log **every** served-model candidate with a pre-kickoff price, not only tips. `props_r2_shadow.csv` already logs `p_served` for every player since 2026-11-01. The priced-candidate volume is 100–1,300 a week per league × market (`forward_capacity.csv`).
  * Projected date for 150 exposed candidates per league (planning only; the exposure share comes from the hindsight lists, so it is optimistic):
    * **def_out**, in Pro's leagues at today's coverage: about 2026-11-15 to 2026-12-09.
    * **gk_out**: 2026-11-25 (Brazil, goals) to 2027-07 (Sweden).
  * If the v9 prematch capture is switched on, top-5 leagues and MLS also get a clock. Their def_out cells are projected for late November to mid-December.
  * Per the power section, 150 is a gate, not enough power.

## 4. What is testable now and what has to wait

| Question | Now | When |
|---|---|---|
| Inventory | done (above) | refresh monthly |
| Absence vs model, STRICT, per league | **not testable**: 2 readable cells, both negative | Readable (≥ 10 exposed fixtures) def_out cells only accrue once a pre-kickoff clock runs again: Pro's capture has added 0 rows since 2026-11-06, and the v9 capture has never run. With all candidates logged, 150 exposed rows arrive Nov–Dec 2026 for def_out and 2027 for gk_out. Power for a +1 pp effect needs thousands of rows, so expect the season to end without a PROVEN residual cell unless the effect is large |
| Absence × price (H-PM-5, the bookmaker test) | **not testable**: there are no prices with fixture id, player id and kickoff before and after team news | starts the day `prematch_capture` (odds_75 / odds_15 / lineups_45 / injuries_120) is on the default branch and enabled |
| Player × league × market bars | **not fittable**: the largest cell is 4 | First parent cell (Brazil sot) reaches 150 on 2026-11-23. Others follow from Dec 2026 to 2027. Player cells only through shrinkage |

## 5. Pre-registered forward protocol (PROPS-MATCHUP-1)

The full text and hash are in `config/experiments/props_matchup/protocol.json` (sha256 `bae12cec…66be`). Any change is a new hypothesis.

* **Window.** Confirmation starts 2026-11-12. Development period: 2026-10-19..2026-11-11, which is never confirmation data.
* **Hypotheses H-PM-1..4.** Per league, a STRICT-clock GK or ≥1-regular-defender absence leaves a positive residual against the served model. Markets: SOT 1+ and anytime scorer.
* **Hypothesis H-PM-5.** Paper flat-stake back bets on exposed candidates that clear the cell's bar, at a **takeable pre-kickoff price**. Never the best-of-books close.
* **Population.** Every priced candidate, not only tips. The served model version is logged on each row.
* **PROVEN.** ≥ 150 settled bets per league × market × exposure cell, and a Holm-adjusted 95% CI > 0 over every registered cell plus any cell added later.
* **PROMISING.** ≥ 50 bets and a raw CI > 0. Paper only.
* **Kill.** After 150 bets, a cell whose upper bound is < 0 is closed.
* **Player bars.** Shrinkage market → league → role → player (Pro `nested_shrinkage`).
  * Nothing is fitted until the league × market parent has ≥ 150 out-of-sample priced bets.
  * A player cell under 30 bets takes its role's value.
  * Bar = max(0, −mean + 1.645·sd).
  * BY q < 0.05 is used as a screen; Holm applies to any PROVEN claim.
* **Stake.** PAPER only. Monthly proposal to the owner.

## 6. Owner decisions / actions found

1. **v9 prematch capture.** It was approved on 2026-11-04 to be switched on after 2026-11-08, but `prematch_capture.yml` is not on the default branch, so it has never run. Without it there is no lineup clock and no before/after-news price.
2. **Pro injury capture: 0 pre-kickoff rows since 2026-11-06 19:32Z**, although every run since reports success. This is the only STRICT source today; worth a look in Pro.
3. **Price storage.** 1,964 priced, settled props tips since 2026-10-14 have no stored snapshot, so their price cannot be proven. Storing every price the ledger uses (with source and timestamp) would make them count.
4. **Log all candidates** (served p plus pre-kickoff price) for the forward test. Tips alone are selection-biased and too few.

## Reproduce

```
python scripts/research_props_matchup.py \
  --scored <run 42855107089 artifact>/output/props_harness/props_harness_holdout_scored.csv.gz \
  --pro-injuries <wowzaV9-Pro main>/output/studies/injury_snapshots.parquet \
  --ledger <v9 main>/output/player_ledger.csv --prices <v9 main>/output/player_prop_odds_history.csv \
  --price-archive <v9 main>/output/player_prop_odds_history_archive --out <dir> --today 2026-11-11
python -m pytest tests/test_research_props_matchup.py -q
```
It runs single-threaded and read-only on every input, takes about 30 seconds, and writes no prices.
