# Clean model backtest: props section (P0 "real data only", 2026-11-16)

Branch `fix/props-data-audit`. Candidates only: **nothing promoted, nothing merged**. Props remain paper.

## Protocol

Props clean retrain harness (`player_model.props_harness`, workflow `props_clean_retrain.yml`, owner
plan `config/experiments/props_clean_retrain/plan.json`): verified rows only (`all_prior_verified`),
fixed chronological holdout **2026-09-01..2026-10-31** (55,907 player-matches per market), training
2022-08-18..2026-08-31 (fit / calibration cut by matchday), walk-forward folds as before. Verified
share: training 0.998, holdout 0.988.

| label | what | run | code | features |
|---|---|---|---|---|
| A | served champion, exactly as validated: its own holdout predictions (sot / sot2 / goals / assists hgb_nan_native, cards recipe_retrained) | 43035218594 | 6fd79496 | 128 (old schema) |
| B | clean schema: corrected started (lineups), injury lists, unpublished counts missing, league-relative shots, typed-constant composites withheld; hgb_nan_native for every market (cards included, plan `candidate_by_market`) | 43745808793 | d051ee5c7 | 125 |
| C-injury | B without the 3 injury features | 43745811817 | d051ee5c7 | 122 |
| C-profile | B without the 7 profile features | 43745820140 | d051ee5c7 | 121 |
| C-started | B without starter_rate / season_start_rate / prev_season_start_rate | 43761230542 | 36f1609e3 | 122 |
| C-shots_lgrel | B without the 17 league-relative shots features | 43761252256 | 36f1609e3 | 108 |

Runs 43745814551 (started) and 43745817424 / 43746668098 (shots_lgrel) died of runner OOM; the
memory fix `36f1609e3` (17.3 -> 7.8 GB peak, identical frame hash `4f6c603d00c14567`) let the
re-dispatched runs finish. A and B are compared on the SAME holdout rows (paired on market,
fixture_id, player_id, same label). Interval: ISO-week block bootstrap of the per-row log-loss
difference (2,000 resamples). Script `compare.py` (scratchpad), output `cmp_final.json`.

## Headline per market (holdout, 55,907 rows each)

| market | events | log loss A | log loss B | B − A (95% CI, week blocks) | Brier A / B | AUC A / B | ECE10 A / B | calib slope A / B | base rate (player) log loss | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| sot | 13,472 | 0.48117 | 0.48116 | −0.00002 (−0.00031, +0.00029) | 0.15701 / 0.15700 | 0.7488 / 0.7489 | 0.029 / 0.030 | 0.989 / 1.000 | 0.49461 | **equivalent** |
| sot2 | 3,311 | 0.19428 | 0.19426 | −0.00002 (−0.00048, +0.00042) | 0.05173 / 0.05161 | 0.8164 / 0.8161 | 0.020 / 0.019 | 0.936 / 0.964 | 0.19841 | **equivalent** |
| goals | 4,980 | 0.26856 | 0.26860 | +0.00004 (−0.00014, +0.00024) | 0.07514 / 0.07512 | 0.7571 / 0.7572 | 0.018 / 0.018 | 0.983 / 0.992 | 0.27496 | **equivalent** |
| assists | 3,696 | 0.22973 | 0.22954 | −0.00019 (−0.00034, −0.00006) | 0.05999 / 0.05993 | 0.7015 / 0.7023 | 0.013 / 0.013 | 0.993 / 1.004 | 0.23481 | **B better (small, significant)** |
| cards | 7,230 | 0.37267 | 0.37235 | −0.00032 (−0.00056, −0.00005) | 0.10985 / 0.10978 | 0.6387 / 0.6397 | 0.013 / 0.012 | 1.085 / 1.078 | 0.37658 | **B better (small, significant)**; B is hgb_nan_native, A the median-filled recipe |

**The clean schema is not worse on any market.** It is statistically indistinguishable on sot / sot2
/ goals and marginally better on assists and cards. Both A and B beat the player base rate on every
market. The gains are small: the value of B is that its inputs are real (no drifted provider flag,
no null read as 0, no stale injury source, no shots-definition shift, no typed constants), not
holdout accuracy.

## Ablations (log loss on the same rows; B in brackets)

| market | − injury | − profile | − started | − shots_lgrel |
|---|---|---|---|---|
| sot | 0.48117 (0.48116) | 0.48116 | 0.48114 | **0.48258** (+0.0014) |
| sot2 | 0.19423 (0.19426) | 0.19431 | 0.19415 | **0.19483** (+0.0006) |
| goals | 0.26865 (0.26860) | 0.26869 | 0.26862 | **0.26914** (+0.0005) |
| assists | 0.22952 (0.22954) | 0.22944 | 0.22946 | 0.22957 |
| cards | 0.37240 (0.37235) | 0.37245 | 0.37239 | 0.37237 |

* **shots_lgrel carries real signal** for the shot and goal markets: removing it costs 0.0005-0.0014
  log loss (AUC sot 0.7489 -> 0.7465). The league-relative counterparts keep that signal through the
  provider's definition change.
* **injury, profile and started add nothing measurable** (|Δ| ≤ 0.0001; dropping started is even
  0.00002-0.0001 better). They are kept for now (real data, no harm); dropping them is a candidate
  simplification for the owner, not made here.

## By league (holdout log loss A / B, the 12 largest leagues)

| league | n | sot | sot2 | goals | assists | cards |
|---|---|---|---|---|---|---|
| USA MLS | 4,446 | 0.4549 / 0.4548 | 0.2118 / 0.2117 | 0.2668 / 0.2668 | 0.2492 / 0.2486 | 0.3732 / 0.3734 |
| League Two | 3,468 | 0.4668 / 0.4665 | 0.1673 / 0.1662 | 0.2608 / 0.2607 | 0.2181 / 0.2178 | 0.3897 / 0.3895 |
| Championship | 3,445 | 0.4776 / 0.4767 | 0.1803 / 0.1801 | 0.2703 / 0.2705 | 0.2299 / 0.2301 | 0.3687 / 0.3680 |
| League One | 3,287 | 0.4809 / 0.4804 | 0.1922 / 0.1930 | 0.2591 / 0.2590 | 0.2128 / 0.2127 | 0.3702 / 0.3695 |
| Argentina Primera | 3,071 | 0.4725 / 0.4735 | 0.1923 / 0.1932 | 0.2541 / 0.2544 | 0.2169 / 0.2170 | 0.4165 / 0.4156 |
| La Liga 2 | 2,640 | 0.4877 / 0.4881 | 0.1786 / 0.1780 | 0.2621 / 0.2624 | 0.2128 / 0.2128 | 0.4580 / 0.4585 |
| Japan J-League | 2,509 | 0.4508 / 0.4511 | 0.1668 / 0.1677 | 0.2604 / 0.2606 | 0.2347 / 0.2342 | 0.3111 / 0.3111 |
| Brazil Serie A | 2,381 | 0.5129 / 0.5141 | 0.2242 / 0.2255 | 0.2616 / 0.2610 | 0.2046 / 0.2043 | 0.4090 / 0.4086 |
| La Liga | 2,170 | 0.4784 / 0.4783 | 0.1932 / 0.1913 | 0.2656 / 0.2657 | 0.2209 / 0.2206 | 0.3997 / 0.3993 |
| Mexico Liga MX | 2,132 | 0.4789 / 0.4802 | 0.2044 / 0.2060 | 0.2628 / 0.2624 | 0.2277 / 0.2272 | 0.3573 / 0.3569 |
| Serie A | 2,112 | 0.5091 / 0.5101 | 0.1989 / 0.1976 | 0.2943 / 0.2946 | 0.2599 / 0.2598 | 0.3191 / 0.3188 |
| Serie B | 2,086 | 0.4666 / 0.4671 | 0.1677 / 0.1694 | 0.2514 / 0.2517 | 0.2247 / 0.2244 | 0.3712 / 0.3708 |

Per league the differences are within ±0.002 with no consistent direction: B is slightly worse on
sot / sot2 in Argentina, Brazil Serie A, Liga MX and Serie B (≤ 0.0017), slightly better in the
English leagues and La Liga. Single-league samples (2-4k rows) cannot separate these from noise.

## By odds band (first price seen, rows with a real price; log loss A / B)

| market | ≤1.5 | 1.5-2 | 2-3 | 3-5 | >5 |
|---|---|---|---|---|---|
| sot | 0.6567 / 0.6561 (827) | 0.6392 / 0.6369 (1,074) | 0.5202 / 0.5208 (940) | 0.3978 / 0.3984 (1,262) | 0.2951 / 0.2948 (399) |
| sot2 | - | 0.6482 / 0.6313 (101) | 0.4880 / 0.4845 (435) | 0.3669 / 0.3653 (850) | 0.1325 / 0.1321 (2,200) |
| goals | - | 0.7208 / 0.7184 (129) | 0.5588 / 0.5585 (1,389) | 0.4051 / 0.4058 (3,461) | 0.1791 / 0.1789 (7,619) |
| assists | - | - | - | 0.4070 / 0.4050 (275) | 0.2093 / 0.2091 (832) |
| cards | - | - | 0.6491 / 0.6415 (162) | 0.4031 / 0.4026 (1,524) | 0.2699 / 0.2702 (932) |

No band where B is materially worse; B is better in the short-price bands where samples are small.

## Against the bookmaker (real prices only)

* **Two-sided de-vigged closing prices exist only for goals (anytime scorer), 54 rows.** Log loss:
  A 0.4981, B 0.4899, closing fair price 0.5511. Both models beat the close on these 54 rows; the
  sample is far too small for a conclusion.
* CLV of the rows the model would have backed (first price × closing fair probability − 1; the fair
  close uses the labelled ASSUMED overround except goals two-sided):

| market | A selected / mean CLV | B selected / mean CLV | basis |
|---|---|---|---|
| sot | 145 / −1.9% | 132 / −1.7% | assumed overround |
| sot2 | 162 / −1.9% | 138 / −1.5% | assumed overround |
| goals | 519 / +6.2% | 476 / +6.3% | assumed overround |
| goals | 27 / −7.4% | 26 / −7.2% | two-sided de-vig |
| assists | 37 / −10.1% | 39 / −9.9% | assumed overround |
| cards | 64 / −5.5% | 94 / −7.4% | assumed overround |

CLV is negative everywhere except goals on the assumed margin, and on the only real two-sided
sample (goals, 26-27 rows) it is negative for both. **No props edge is demonstrated against the
market; props stay paper-only** (no stake, no promotion). The clean schema does not change that.

## Hidden markets (goals2, goals3, sot3, sot4)

Their served legacy payloads are refused (fabricated set-piece features). In run B the clean
candidates do NOT beat the player base rate on the holdout: goals2 (hgb log loss 0.04603 vs base
0.04493; CI of the difference +0.0006..+0.0016), sot4 (0.02165 vs 0.01947; +0.0015..+0.0029), goals3
and sot3 (CI straddles 0). They stay hidden and without a model (fail-closed).

## Limitations

* The holdout ends 2026-10-31. November, the month with the largest provider drift (started flag
  0.97, shots +45%), is not in it, so the size of B's advantage on drifted inputs is under-measured.
* A's predictions are the champion's own validation predictions (its old inputs). Served today, the
  champion reads the corrected inputs and the withheld raw shots features at their new scale.
* The ablations ran on two commits (d051ee5c7, 36f1609e3); the memory change between them leaves
  the frame identical (hash 4f6c603d00c14567), so they are comparable.
* Bookmaker comparison: 54 two-sided rows (goals only); every other market uses an assumed margin.

## Decision record

No promotion (owner rule: retrain/promotion only when the architecture is wired and the go-live
rehearsal is green). B is a candidate the owner may promote later for sot / sot2 / goals / assists /
cards; on this evidence it is safe (not worse) and the inputs are real. Props remain paper.
