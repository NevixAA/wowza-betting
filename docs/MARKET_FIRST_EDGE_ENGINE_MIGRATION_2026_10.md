# Market-first edge engine migration, stage (c): replay, preregistration, cutover criteria (2026-10-25)

Owner brief 2026-10-25, stage (c). **Nothing served changes.** Legacy stays the production selection
path; `market_edge_v2` stays SHADOW with its committed `UNVALIDATED_DEFAULTS` config. Every historical
number below is **RETROSPECTIVE**: it can describe what the engines would have done on fixtures already
played and can never be CONFIRMED. The forward test that can decide is preregistered in Pro
(`registry/edge2_hypotheses_v1.json`, forward start 2026-10-26T00:00Z).

Artifacts (all labelled RETROSPECTIVE):

| File | What |
|---|---|
| `src/research/edge_replay.py` | replay library: first sight from git, same-book LOCF pairs, legacy reproduction, metrics |
| `scripts/edge_engine_replay.py` | runner (≈2 min cold, reads 723 committed `predictions.csv` runs from git) |
| `output/edge_engine_backtest.csv` | 3,945 rows: one per fixture × family at first sight, both engines, outcome, certified close |
| `output/edge_engine_comparison.json` | per track × market comparison, disagreement test, current board, shadow counts |
| `output/edge_engine_proposed_params.json` | **PROPOSED** discovery-fitted weights (NOT production config) |
| `tests/test_edge_replay.py` | 22 tests |

## 1. Old production path (legacy, unchanged)

* **O/U 2.5** (`src/betting.py::evaluate_value`): `edge = p_model − 1/odds` on each side at the board
  price (first book quoting each side, `src/predict.py`); `_base_tier` by edge size; min-odds, new-format
  price cap 3.00, blind-fixture guard, drift up/downgrade; `signal_tier` on the larger-edge side.
* **BTTS-YES / OVER 1.5 / OVER 3.5** (`pipeline._generate_side_bets`): `edge = p_model − (1/odds)/1.08`,
  bars 0.10 / 0.08 / 0.04 unless the league is `approved` in `models/best_params_side_markets.json`.
  One side only.
* **Thresholds in force** — never read from `config.py` alone. `git log -G` on
  `.github/workflows/predict.yml` since 2026-08-10 shows exactly one change: `b82cec9f`
  (2026-08-21T16:06:04Z) set `VALUABLE 0.04→0.03`, `MARKSMAN 0.14→0.08`, `LEAGUE_SNIPER_CAP 1.0→0.12`.
  `best_params_standard.json` / `best_params_side_markets.json` are read from git as of each entry time
  (versions 2026-08-15, 09-01, 10-01, 10-06).
* The tier is the **size of the model's disagreement with one vigged price**: no market anchor, no
  uncertainty.

## 2. New production path (shadow only)

`pipeline.py` → `src/edge_engine_shadow.run` (wrapped, cannot raise) after legacy selection →
`src/market_edge.evaluate` per fixture × family → `output/edge_engine_shadow.csv` (append-only, first
sight per `(selection_engine_version, fixture_id, market)`) + `output/edge_engine_health.json`; trailing
shadow columns on `stake_plan.csv`. No tip, tier, stake, ledger row or Telegram message reads it.
Live since 2026-10-25T09:25:36Z.

## 3. Formulas (market_edge_v2, `config/market_edge_v2.json`, sha256-LF `a85ae726…44444`)

```
p_market_fair      = median over valid same-book two-sided pairs of devig(odds_a, odds_b)
raw                = logit(p_model) − logit(p_market_fair)
p_decision         = sigmoid(logit(p_market_fair) + w[track|family] · clip(raw, ±0.35))     w = 0.10 everywhere
p_lower_bound      = p_decision − Σ penalties
                     books 1.2816·max(sd,0.02)/√n · devig |p_pow−p_prop| · residual |p_dec−p_mkt|·300/(300+n_eff)
                     · league 0.02·200/(200+n_hist) · disagreement 0.25·max(0,|p_model−p_mkt|−0.08)
                     · long odds 0.02·max(0,odds−2.5) · stale 0.002/h past 60 min · retrain 0.01 (<7 d)
                     · weak track CLV 0.01
ev_point           = p_decision·odds − 1        ev_lower_bound = p_lower_bound·odds − 1
odds               = the board price legacy decides at (price_source=row); median valid book if absent
```

With `n_eff = 0` (today) the residual penalty removes the whole residual, so `p_lower_bound ≤
p_market_fair` and **EV_LB > 0 is only possible when the board price beats the de-vigged consensus by
more than the penalties** — i.e. a price edge, never a model edge.

## 4. De-vig method

Per book, both proportional `(1/a)/(1/a+1/b)` and power `(1/a)^k, (1/a)^k+(1/b)^k=1` (bracket-checked
bisection; failure → `DEVIG_FAILED`); median across books. Config uses **power**; both are recorded.
Retrospective market log loss (two-sided first-sight rows): power vs proportional — NF OU25 0.67136 vs
0.67093; STD OU25 0.68329 vs 0.68354; NF BTTS 0.68015 vs 0.67918; STD BTTS 0.67834 vs 0.67927; NF OU15
0.60026 vs 0.60523. The differences are ≤0.001 except NF OU15 (n 69): no method is clearly better.
Discovery-window choice (section 8): power everywhere except NF BTTS (proportional).

## 5. Market baseline logic

Consensus hierarchy: validated exchange (skipped — `exchange_validated=false`) > median of ≥2 same-book
two-sided books > single two-sided book > one-sided only (`OBSERVE / ONE_SIDED_PRICE`, no fair
probability) > nothing (`NO_BET / NO_PRICE`). Per-book validation: odds > 1 both sides, quote time ≤ now
and < kickoff, overround in [0, 0.15], age ≤ 720 min (180 inside 6 h). Book archives are CHANGE-LOGS,
carried forward per book and side; a pair's timestamp is the later side's last change. A pair is never
formed across books (tested). The BTTS board pair (median-yes / median-no from different books) is used
only as an execution price, never as a market.

## 6. Uncertainty method

Additive lower bound (section 3); every penalty is written to `penalty_breakdown` on each shadow row.
Caps can only lower a state: odds > 3.0 → ≤ OBSERVE; `|p_model − p_market| > 0.20` → ≤ OBSERVE;
OU35 → NO_BET always. Extreme disagreement can never raise a state.

## 7. Decision-state definitions

| State | Rule |
|---|---|
| NO_BET | fail-closed gate (unsupported market, unknown track, missing model/kickoff, post-kickoff, synthetic price, no/stale/invalid price, de-vig failure), `ev_point ≤ 0`, or blocked family |
| OBSERVE | `ev_point > 0 ≥ ev_lower_bound`; or EV_LB > 0 without ≥2 books or without track CLV evidence; or a cap; or one-sided price only |
| QUALIFIED | EV_LB > 0, ≥2 books, track CLV record n ≥ 50 with clipped mean > 0 **and** median > 0 |
| STRONG | QUALIFIED + PROSPECTIVE CLV evidence for `market_edge_v2` on that track (n ≥ 150, CI lower > 0). No evidence file exists → unreachable |

`display_tier` (STRONG→SNIPER, QUALIFIED→MARKSMAN, OBSERVE→VALUABLE) is presentation only.

## 8. Legacy vs new — historical comparison (RETROSPECTIVE)

**Design.** First sight = the earliest committed `predictions.csv` run strictly before kickoff that
carries the family's model probability and a real board price (OU25: both sides); 723 runs,
2026-08-15 → 10-25, 115,379 board rows. Both engines judged at that same instant. NEW gets only
same-book pairs from the OddsAPI change-log (345,693 rows, 2026-08-19 →) carried forward to the
instant; nothing written later reaches it. Point-in-time engine context: track CLV from ledger rows with
match date before the entry date; league history before the entry date; no prospective evidence.
Settlement: `af_history` + `fd_history`, league-scoped `team_names.resolve_result`, unique ±1 day match
(2,690 of 3,945 rows settled; the rest are unplayed or unmatched — never guessed). No synthetic price
enters: the serving path has no fill constant (the 1.85/1.40/2.60 fills live only in
`src/backtest.py`, whose outputs are not read); flagged rows and synthetic quotes are dropped (tested).
Certified CLV only (book write ≤30 min before kickoff, ≥3 two-sided books at kickoff, power median,
CLV% = odds·p_close_fair − 1); no close → NaN.

**Legacy reproduction.** OU25 legacy = the tier production RECORDED at first sight. Reproducing
`tier_over`/`tier_under` from the threshold regime of the time matches **100.0%** of first-sight rows
(NF and STD), and 100% of the committed 183-fixture board (test). Against the ledgers at the same price:
OU25 343/622 equal (55.1%) — **all 279 differences are the ledger being HIGHER (in-place upgrades after
first sight), 0 lower**; BTTS 183/200 (91.5%, 17 higher, 0 lower); OU15 89/95 (93.7%, 6 higher, 0
lower). The ledger is not a first-sight record; the board is.

**Engines.** `LEGACY` = legacy tier ≠ AVOID. `NEW_QUALIFIED` = state QUALIFIED/STRONG (the engine as
configured). `NEW_EVLB` = the engine's value path ignoring the evidence gate (EV_LB > 0, ≥2 books, no
cap). `NEW_OBSERVE_POOL` = any non-NO_BET two-sided row — **not** a bet recommendation, shown only for
completeness. Flat 1u at the entry price; ROI CI = matchday block bootstrap (2,000).

Two-sided universe (rows where NEW could form a market at entry):

| cell | status | settled first sight | two-sided | engine | N | bets/day | hit % | ROI % [95% CI] | max DD u | longest losing | CLV cov % | CLV mean / median % | CLV >0 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NF OU25 | OK | 846 | 681 | LEGACY | 277 | 4.10 | 40.8 | −2.8 [−20.1, 12.4] | 29.6 | 15 | 15.5 | −9.28 / −6.92 | 18.6 |
| | | | | NEW_QUALIFIED = NEW_EVLB | 1 | 0.01 | 100 | +72.0 [n/a] | 0 | 0 | 0 | – | – |
| | | | | NEW_OBSERVE_POOL | 13 | 0.19 | 38.5 | −27.0 [−78.8, 20.9] | 4.6 | 3 | 7.7 | −8.79 (n 1) | 0 |
| STD OU25 | OK | 524 | 424 | LEGACY | 98 | 1.58 | 36.7 | −20.2 [−37.7, 4.1] | 21.3 | 8 | 19.4 | −9.01 / −10.21 | 10.5 |
| | | | | NEW_QUALIFIED / NEW_EVLB | 0 | 0 | – | – | – | – | – | – | – |
| | | | | NEW_OBSERVE_POOL | 4 | 0.06 | 0 | −100 | 4 | 4 | 0 | – | – |
| NF BTTS | OK | 741 | 741 | LEGACY | 133 | 2.19 | 60.2 | +26.3 [8.2, 44.7] | 7.0 | 7 | 24.8 | −9.03 / −9.25 | 9.1 |
| | | | | all NEW variants | 0 | 0 | – | – | – | – | – | – | – |
| STD BTTS | OK | 468 | 468 | LEGACY | 50 | 0.91 | 52.0 | −0.8 [−26.7, 33.0] | 10.3 | 5 | 24.0 | −1.74 / −5.20 | 25.0 |
| | | | | all NEW variants | 0 | 0 | – | – | – | – | – | – | – |
| NF OU15 | OK | 92 | 69 | LEGACY | 62 | 0.92 | 71.0 | +9.7 [−8.0, 27.7] | 4.4 | 3 | 12.9 | −4.64 / −5.27 | 0 |
| | | | | NEW_QUALIFIED / NEW_EVLB | 0 | 0 | – | – | – | – | – | – | – |
| | | | | NEW_OBSERVE_POOL | 6 | 0.09 | 66.7 | +7.7 [−47, 62] | 1 | 1 | 16.7 | −0.03 (n 1) | 0 |
| STD OU15 | **INSUFFICIENT** (18 two-sided < 30) | 19 | 18 | LEGACY | 10 | 0.16 | 50 | −29.9 [−81.9, 13.2] | 4 | 4 | 0 | – | – |

Legacy on the full first-sight universe (incl. rows with no two-sided price): NF OU25 n 327 −3.0%
[−18.3, 10.2]; STD OU25 108 −21.6% [−38.2, −0.6]; NF BTTS 133 +26.3% [8.2, 44.7]; STD BTTS 50 −0.8%;
NF OU15 84 +7.1% [−7.2, 22.3]; STD OU15 11 −36.3%.

Agreement (two-sided universe, LEGACY vs NEW_EVLB): NF OU25 legacy-only 277 / new-only 1 / both 0 /
disagree-side 0 / neither 403; STD OU25 98/0/0/0/326; NF BTTS 133/0/0/0/608; STD BTTS 50/0/0/0/418;
NF OU15 62/0/0/0/7; STD OU15 10/0/0/0/8. **The engines never selected the same fixture.**

Probability quality, side A (OVER/YES), two-sided rows — Brier / log loss / calibration slope / ECE:

| cell | n | model | market (power) | market (proportional) | decision (w = 0.10) | model − market Brier gain [CI] |
|---|---|---|---|---|---|---|
| NF OU25 | 681 | .24404 / .68102 / 0.93 / .052 | .23955 / .67136 / 0.78 / .076 | .23929 / .67093 / 0.86 / .059 | .23909 / .67047 / 0.82 / .068 | −0.0045 [−0.0112, +0.0018] |
| STD OU25 | 424 | .25152 / .69622 / **−0.25** / .033 | .24516 / .68329 / 1.13 / .029 | .24527 / .68354 / 1.28 / .036 | .24529 / .68358 / 1.19 / .035 | **−0.0064 [−0.0115, −0.0005]** |
| NF BTTS | 741 | .24342 / .67990 / 0.66 / .057 | .24365 / .68015 / 0.34 / .048 | .24317 / .67918 / 0.39 / .037 | .24312 / .67907 / 0.37 / .045 | +0.0002 [−0.0053, +0.0061] |
| STD BTTS | 468 | .24402 / .68117 / 0.95 / .090 | .24264 / .67834 / 0.60 / .068 | .24309 / .67927 / 0.68 / .073 | .24259 / .67825 / 0.66 / .076 | −0.0014 [−0.0060, +0.0032] |
| NF OU15 | 69 | .19644 / .58022 / 1.69 / .035 | .20547 / .60026 / 0.77 / .094 | .20765 / .60523 / 0.88 / .099 | .20433 / .59776 / 0.80 / .094 | +0.0090 [−0.0081, +0.0280] |
| STD OU15 | 18 | INSUFFICIENT | | | | |

Certified-close coverage is thin: 15.0% (NF OU25), 23.3% (STD OU25), 22.0% (NF BTTS), 30.8% (STD BTTS),
11.6% (NF OU15) of two-sided rows. That is why CLV cannot be the primary forward metric yet.

## 9. Track-by-track results

* **NF OU25.** The model does not beat the de-vigged market (Brier gain −0.0045, CI spans 0). Legacy
  ROI −2.8% (CI spans 0), certified CLV −9.3% mean (n 43). NEW at the committed config essentially never
  bets (1 QUALIFIED in 681). The only retrospective sign of residual information on the whole estate is
  here: on the discovery/holdout split a small residual (w 0.05 fitted; the current 0.10 too) lowers
  holdout log loss below the market by ~0.001–0.002 (section 8 / proposed params). The isotonic
  disagreement fit is increasing (holdout Spearman 0.16, p 0.003) but no bin's outcome advantage clears
  zero after BH.
* **STD OU25.** The market beats the model with a CI excluding zero, and the model's calibration slope
  is negative (−0.25): at first sight the standard O/U model has no usable discrimination against the
  outcome on this window. Legacy −20.2% (two-sided) / −21.6% full universe, CI [−38.2, −0.6]; CLV −9.0%.
  Discovery-fitted residual weight = 0.00. Under NEW nothing is selected — correctly.
* **NF BTTS.** Legacy BTTS-YES shows +26.3% [8.2, 44.7] on 133 bets, **but** certified CLV is −9.0%
  (only 9.1% positive, n 33) and the model adds nothing to the market (Brier gain +0.0002). A positive
  ROI with negative CLV and no probability skill is the signature of a league-wide outcome regime, not of
  selection (consistent with the 2026-10 upgrade study). It is not corrected for the six legacy cells
  examined and is RETROSPECTIVE. NEW selects nothing.
* **STD BTTS.** No signal either way; NEW selects nothing.
* **NF OU15.** n 69 two-sided. The model's Brier is better than the market's (+0.009) but the CI spans
  zero and calibration slope 1.69 says it is underconfident; discovery w 0.45, holdout −0.020 LL
  [−0.044, +0.002], BH q 0.052. Too small to call.
* **STD OU15.** INSUFFICIENT (18 two-sided rows). OddsAPI EU totals almost never quote Under 1.5 for
  these leagues; API-Football `book_quotes/` (7–8 books, from 2026-10-04) is not wired.

**Non-linear disagreement test** (task 2). `d = |p_model − p_market_fair|`; bin edges = tertiles of the
FIRST half (chronological), evaluated on the second half; matchday-block CIs; one-sided bootstrap
"moderate > extreme"; BH across all 5 cells × 5 metrics (17 testable):

| cell | holdout n (small / moderate / extreme) | outcome advantage toward model: small / moderate / extreme | moderate − extreme: adv, ΔLL, ROI (p, BH q) | isotonic (fit on 1st half), holdout Spearman |
|---|---|---|---|---|
| NF OU25 | 54 / 77 / 210 | +0.021 / −0.000 / +0.073 [−0.004, 0.141] | −0.073 (p .87, q .91); +0.015 (p .25, q .36); −0.112 (p .77, q .87) | increasing, 0.16 (p 0.003) |
| STD OU25 | 71 / 55 / 86 | −0.007 / +0.031 / −0.007 | +0.038 (p .35, q .42); +0.035 (p .06, q .36); +0.098 (p .28, q .36) | increasing, 0.19 (p 0.007) |
| NF BTTS | 163 / 123 / 85 | +0.015 / +0.065 / +0.019 | +0.046 (p .25, q .36); +0.030 (p .16, q .36); +0.138 (p .20, q .36) | increasing, 0.22 (p <0.001) |
| STD BTTS | 70 / 68 / 96 | +0.025 / **+0.161 [0.041, 0.248]** / +0.056 | +0.105 (p .10, q .36); +0.013 (p .23, q .36); +0.216 (p .09, q .36) | increasing, 0.05 (p 0.44) |
| NF OU15 | 19 / 12 / 4 | too few in the extreme bin | not testable | increasing, 0.15 (p 0.38) |
| STD OU15 | INSUFFICIENT | | | |

Answer: in 3 of 4 testable cells the moderate bin has the higher point estimate (STD OU25, NF BTTS,
STD BTTS), and in the extreme bin the model's log loss is worse than the market's in 3 of 4 cells
(log-loss gain of model over market −0.016 NF OU25 / −0.033 STD OU25 / −0.023 NF BTTS; STD BTTS
+0.005), **but no moderate-vs-extreme difference survives BH (min q 0.36)**. Disagreement size carries some rank information (isotonic Spearman 0.16–0.22 in three cells) —
which is NOT the same as value: the outcome advantage is measured against the de-vigged market, before
the vig is paid, and certified CLV in the extreme bins is n ≤ 29. The engine's design choice (clip the
residual, penalise disagreement beyond 0.08, cap at 0.20) is not contradicted by this data; it is not
proven by it either.

**Discovery-only parameter fit** (task 3; `output/edge_engine_proposed_params.json`, status PROPOSED,
NOT applied — `config/market_edge_v2.json` unchanged, test pins it). Chronological 60/40 split of
settled two-sided first-sight rows; de-vig by discovery log loss; w on a 0–1 grid by discovery log loss:

| cell | discovery (n) | holdout (n) | de-vig | w fitted | holdout LL: market / fitted / current 0.10 | fitted − market [CI], p, BH q |
|---|---|---|---|---|---|---|
| NF OU25 | 08-21→09-27 (408) | 09-27→10-22 (273) | power | 0.05 | .67994 / .67880 / .67770 | −0.0011 [−0.0019, −0.0004], p .004, q .012 |
| STD OU25 | 08-19→09-12 (254) | 09-12→10-16 (170) | power | 0.00 | .68588 / = market / .68615 | no residual (w 0) |
| NF BTTS | 08-22→09-20 (444) | 09-20→10-22 (297) | proportional | 0.75 | .67401 / .67458 / .67372 | +0.0006 [−0.0074, +0.0075], p .57, q .57 |
| STD BTTS | 08-22→09-12 (280) | 09-12→10-16 (188) | power | 0.00 | .68953 / = market / .68806 | no residual (w 0) |
| NF OU15 | 08-19→09-26 (41) | 09-27→10-21 (28) | power | 0.45 | .60523 / .58517 / .60039 | −0.020 [−0.044, +0.002], p .035, q .052 |
| STD OU15 | INSUFFICIENT (18) | | | | | |

The NF OU25 improvement is real on this holdout but tiny (0.0011 nats ≈ 0.17% of the log loss); it is
one split of one 9-week season start, so it is a hypothesis (H-EDGE2-NF-OU25-01), not a parameter.

## 10. Prospective shadow design

* Evidence: v9 `output/edge_engine_shadow.csv`, first sight per `(market_edge_v2, fixture_id, market)`,
  pre-kickoff only, append-only (`src/research/first_sight.FirstSightStore`).
* Preregistration: Pro `registry/edge2_hypotheses_v1.json` (frozen, content hash pinned in
  `src/studies/edge2_forward.py`), registered 2026-10-25, **forward_start 2026-10-26T00:00:00Z** (rows
  logged 10-25 are PRE_REGISTRATION_NOT_COUNTED). Engine frozen by sha256-LF of
  `config/market_edge_v2.json` (`a85ae726…44444`) and `src/market_edge.py` (`bd415829…a2bb6`); any change
  → BLOCKED_CONFIG_CHANGED.
* Family H-EDGE2 v1: **NF-OU25-01, NF-BTTS-01, NF-O15-01, STD-OU25-01** (STD BTTS, STD OU15 and OU35
  excluded with reasons in the registry). Primary metric: paired log-loss difference
  `LL(p_decision) − LL(p_market_fair)` on settled two-sided first-sight rows, one-sided, matchday block
  bootstrap 10,000, BH q ≤ 0.05 across eligible members. Secondary: Brier, calibration, certified CLV and
  ROI of QUALIFIED/STRONG selections, state counts. Minimums: 400 settled two-sided rows and 8 ISO weeks
  (NF-O15: 150 / 8). No effect size is published before eligibility.
* Evaluator: Pro `src/studies/edge2_forward.py` → `output/edge_reassessment/edge2_forward_progress.json`,
  daily in `pro_valuable_shadow.yml` (09:50 UTC). Statuses REGISTERED / ACCRUING / ELIGIBLE_FOR_REVIEW /
  BLOCKED_CONFIG_CHANGED; CONFIRMED never assigned.
* The 14 H-GATE entries in Pro `registry/candidate_forward_tests.json` are now tagged
  `family_class: LEGACY_RAW_EDGE_FORWARD_TEST` (additive; evaluator output byte-identical before/after,
  sha `d8b992e7…`). They test thresholds on the legacy raw edge and **cannot validate market_edge_v2**.

## 11. Current sample counts (`output/edge_engine_shadow.csv` at v9 `b8900ed0`)

732 rows, one run (2026-10-25T09:25:36Z), all pre-kickoff, all `NO_BET`:

| track × market | rows | two-sided | states |
|---|---|---|---|
| new_format OU25 | 99 | 99 | NO_BET 99 |
| new_format BTTS | 99 | 99 | NO_BET 99 |
| new_format OU15 | 99 | 7 | NO_BET 99 |
| new_format OU35 | 99 | 33 | NO_BET 99 (blocked) |
| standard OU25 | 84 | 84 | NO_BET 84 |
| standard BTTS | 84 | 84 | NO_BET 84 |
| standard OU15 | 84 | 0 | NO_BET 84 |
| standard OU35 | 84 | 22 | NO_BET 84 (blocked) |

Reasons: NO_POSITIVE_EV 373, NO_PRICE 303, MARKET_BLOCKED 55, STALE_PRICE 1. Settled forward rows: 0.

## 12. What is still PAPER

Everything market_edge_v2 emits: the shadow log, the stake-plan shadow columns, any OBSERVE/QUALIFIED
row. The proposed parameters. All H-EDGE2 hypotheses (REGISTERED). This stage changes no legacy
staking, tier or notification.

## 13. What is blocked

* **STRONG**: unreachable — no prospective evidence file (`output/edge_engine_evidence.json`).
* **QUALIFIED**: gated by the retrospective track CLV record (n ≥ 50, mean > 0 and median > 0); at
  registration no track passes (e.g. new_format|OU25 mean +4.69 but median 0.00).
* **OU35**: `blocked_families`. **STD OU15**: no two-sided price source wired.
* **CLV as a primary metric**: certified-close coverage 11–31%.
* **Exchange top of hierarchy**: `exchange_validated=false`.
* **Applying fitted parameters**: owner review + PR, and it would require a NEW registration version.

## 14. Unresolved risks

1. **Change-log staleness.** The archive stores price changes only, so an unchanged price ages; the
   board re-evaluation without the run's live quotes marked 24 of 98 legacy selections STALE_PRICE. The
   live shadow sees the run's own quotes (1 STALE on the board). A per-fixture fetch timestamp would fix
   the measurement.
2. **Shadow rows carry no config hash.** The forward evaluator can only check the CURRENT v9 config/code
   hash; a change-and-revert between runs would be invisible. Recommend stamping the hash per row in a
   later, separately approved change (the shadow store is first-sight-frozen, so a new column needs care).
3. First sight before 2026-08-19 has no book archive (NO_PRICE 165 NF / 100 STD OU25 rows).
4. Execution price = board price (first book quoting OU25; median-yes/no for BTTS), not the best price.
5. 9-week retrospective window, a threshold-regime change on 08-21 and daily model retrains inside it.
6. 1,255 of 3,945 replay rows unsettled (future fixtures or no unique league-scoped name match).
7. Legacy reproduction uses the current `config.py` league dicts; it still reproduces 100% of recorded
   base tiers, so no historical edit to those dicts mattered on this window.
8. Pro `pro_valuable_shadow.yml` had a literal `\n` inside its commit `for` list (harmless by accident —
   `[ -e n ]` was false); fixed while adding the new file.

## 15. Production cutover criteria

All of the following, in order, with explicit owner approval at the end (v9 change control):

1. H-EDGE2 hypothesis ELIGIBLE_FOR_REVIEW with `PROBABILITY_LAYER_SUPPORTED` (primary CI upper < 0, BH
   q ≤ 0.05, calibration slope of p_decision in [0.8, 1.2]) on the frozen engine hash.
2. For any selection use: `SELECTION_SUPPORTED` — ≥150 settled QUALIFIED/STRONG selections, certified-
   CLV coverage ≥ 50%, CLV mean CI lower > 0, ROI CI lower > −5%.
3. No REJECT/FUTILITY on the same track × market; standard and new-format decided separately.
4. Prospective evidence file written by the evaluator (not by hand) before STRONG can exist.
5. A cutover plan that keeps legacy as a parallel log for ≥4 weeks and is reversible by one config line.

## Evidence-only answers

**(a) Why the market-first engine was present but not driving production.**
`src/edge_engine.py::classify` has never been called anywhere — the only import is
`src/clv_capture.py:24` (`power_devig`, `proportional_devig`). As written it could not have driven
anything: its lineup gate returns OBSERVE first with `lineup_confirmed=False` by default (audit D3) and
with `n_eff=0` its weight and edge are zero so its floors never pass (D9). The rebuilt `market_edge_v2`
was added on 2026-10-25 deliberately as SHADOW (called only through `src/edge_engine_shadow.run`, which
writes a log), with `UNVALIDATED_DEFAULTS`, because v9 is change-controlled and a new selection
methodology needs prospective evidence and owner approval. Production therefore still tiers on
`p_model − 1/odds` (`src/betting.py`) and `p − (1/odds)/1.08` (`pipeline._generate_side_bets`).

**(b) Current live selections that change under the new engine.** Committed first-sight shadow
(2026-10-25T09:25Z): **101** legacy selections (OU25 71, BTTS 21, OU15 6, OU35 3 — SNIPER 15, MARKSMAN
36, VALUABLE 50). Under market_edge_v2: **101 / 101 become NO_BET** (98 NO_POSITIVE_EV, 3
MARKET_BLOCKED), **0 keep a tier/state above NO_BET, 0 become OBSERVE, 0 switch to a bet on the other
side.** The engine's preferred (least-bad) side differs from legacy's on 72 of the 101 (OU25 59/71, BTTS
11/21, OU35 2/3), but every such side is still NO_BET. No row on the board — legacy-selected or not —
has `ev_point > 0`. Re-evaluating the committed board at its own run time from the change-log alone
gives the same answer: 98 legacy selections → 96 NO_BET (72 NO_POSITIVE_EV, 24 STALE_PRICE), 2 OBSERVE,
0 QUALIFIED/STRONG.

**(c) Markets where the new engine adds value over the de-vigged market after uncertainty, CLV and
chronological OOS validation.** **None.** At the committed configuration the engine selected 1 bet in
2,401 settled two-sided first-sight rows, so there is no selection record to validate, and in no market
does any selection show positive certified CLV. On probability alone, the only out-of-sample improvement
over the de-vigged market is new-format O/U 2.5: a residual weight of 0.05 (and the current 0.10) lowers
holdout log loss by ~0.001–0.002 (CI [−0.0019, −0.0004], BH q 0.012) — RETROSPECTIVE, one split, no CLV
support, far too small to price an edge. It is registered as H-EDGE2-NF-OU25-01 and will be decided
forward, from 2026-10-26.
