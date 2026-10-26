# Quality layers (Best / Second / Third) — study, 2026-10-26

Owner request (2026-10-26): replace the gap-size tiers (SNIPER / MARKSMAN / VALUABLE) with three layers
ranked by actual quality. **This stage is research and shadow only. No tier, stake, threshold, Telegram
message or ledger changed.**

## Bottom line in plain English

**No market earns three real layers yet. Every market should stay as one layer ("Tip") for now.**

We built a quality score per tip: the chance it wins (learned only from tips that had already been
settled) times the odds, minus 1 — the expected profit. We then split each day's tips into Best,
Second and Third by that score. We picked the layering rule on the first 60% of the season and
then tested it, without changes, on the last 40% (5 Oct – 25 Oct). In the test period, Best did
**not** beat Third in any market:

| market (track) | test-period tips | Best | Second | Third | real 3 layers? |
|---|---|---|---|---|---|
| O/U 2.5 (new-format) | 200 | −11.0% | **+26.1%** | +3.3% | **No** — the middle layer won |
| BTTS (new-format) | 94 | −3.8% | **−44.6%** | **+19.4%** | **No** — Third was best |
| O/U 2.5 (standard) | 130 | −20.3% | +8.3% | (empty) | **No** — Best was worst |
| BTTS (standard) | 48 | **−43.7%** | +2.6% | **+24.9%** | **No** — the order is reversed |
| Over 1.5 (new-format) | 28 | too few | | | No — not enough tips |
| Over 1.5 / Over 3.5 (standard), Over 3.5 (new-format) | ≤ 4 | too few | | | No — not enough tips |

A worked example. In new-format BTTS during the test weeks, if you had backed only the 15 tips the
score called "Best", you would have lost 0.6 units (−3.8%); the 46 tips it called "Third" made
+8.9 units (+19.4%). The score put the tips in the wrong order.

**Why.** The learned chance of winning is no better than the bookmaker's own (de-vigged) price. In
the test period its Brier score was worse than the price in three of the four main markets (new-format
BTTS 0.280 vs the price 0.251; new-format O/U 2.5 0.247 vs 0.245; standard BTTS 0.253 vs 0.249) and only
level in standard O/U 2.5 (0.242 vs 0.243). A ranking can only be as good as the probability behind it,
and with ~800 settled tips per track the model mostly learns this season's noise (for example, which
leagues happened to run hot in August). This is the same finding as before, from a new angle:
**the price already knows what the model knows.**

**The current gap tiers are not better.** Measured on the same test rows, at the tier each tip had
when it was **first** sent:

| market (track) | SNIPER | MARKSMAN | VALUABLE | ordered? |
|---|---|---|---|---|
| O/U 2.5 (new-format) | −5.7% (38) | −23.8% (42) | **+20.2%** (120) | No (reversed) |
| BTTS (new-format) | **+47.6%** (6) | −63.4% (11) | −2.9% (77) | No (MARKSMAN worst) |
| O/U 2.5 (standard) | −40.0% (9) | −14.8% (11) | +0.8% (110) | No (reversed) |
| BTTS (standard) | — (0) | +75.5% (1) | −7.1% (47) | — |

**The owner's BTTS ladder is partly a bookkeeping effect.** Over the whole season, new-format BTTS in
the *ledger* reads SNIPER +24.7% / MARKSMAN +16.0% / VALUABLE +7.7% (our reproduction of the owner's
+25.7 / +17.5 / +7.7). But the ledger **upgrades a tip's tier in place** when its gap later widens, so a
tip that was sent as VALUABLE and later grew is filed as SNIPER. At the tier each tip had when first
sent, the same season reads SNIPER **+52.2%** (48 tips, 95% CI +21% to +83%) / MARKSMAN **+3.1%** (33) /
VALUABLE **+6.7%** (164). So in new-format BTTS the **top bucket is genuinely strong**, but MARKSMAN is
not better than VALUABLE — the ladder is a "SNIPER vs the rest" split, not three steps. It is the one
lead worth watching, and it is what the forward test below will check without hindsight.

**What happens next.** Nothing changes for users. Every new tip now gets its quality score and layer
written, at first sight, to `output/quality_tier_shadow.csv`. A forward test is preregistered in Pro
(`registry/quality_tiers/registration_v1.json`, starts **2026-10-27 00:00 UTC**). Layers can only be
switched on for a market after that forward test passes **and** the owner approves.

---

## 1. Method

### Population (first sight)
* The tips actually sent: both v9 ledgers (`bets_ledger.csv`, `side_bets_ledger.csv`), `source=live`, paper
  leagues dropped, one per fixture × market — exactly `src/top_picks.load_tips()`. Season from
  2026-08-15; 1,630 tips.
* **Every feature is taken at first sight**, from the committed `predictions.csv` snapshot of the run that
  created the tip (Pro `experiments/valuable_2026_10/prematch_snapshots.parquet`, refreshed to
  2026-10-26 09:50 UTC — 118,181 snapshot rows). Matching = same league/date/home/away, nearest snapshot
  within −3 h…+15 min of the ledger's `generated_at`. 1,555 matched (median lag 0.5 min; 75 dropped).
* O/U 2.5 ledger odds are never upgraded and equalled the snapshot price on 100% of rows. Side-market odds
  and every tier/edge **are** upgraded in the ledger, so those come from the snapshot. First-sight O/U tier
  = the snapshot's own `tier_over/tier_under`; side tiers are recomputed with production's rule
  (`pipeline._generate_side_bets`, 0.10/0.08/0.04 on `p − (1/odds)/1.08`, today's approved per-league gates
  via `gate_resolver`). First-sight tier = ledger tier on 72% of tips (the other 28% were upgraded later).
* Results (WIN/LOSS) from the ledgers; flat 1u P&L recomputed at the **first-sight** price.

### The score (frozen in `src/research/quality_tiers.py` before any holdout result)
* `p_win`: L2 logistic regression (C = 0.5, the same fixed value as Pro's learned tiers; standardised inputs),
  **one model per track** (standard / new-format never pooled), on:
  `logit(fair market p)` (two-sided de-vig where the opposite price exists — O/U 2.5 always, BTTS when the
  No price exists — else 1/odds/1.08), the gap `logit(model p) − logit(fair p)`, an `odds ≥ 3.00` flag, a
  **league rating as known at the time** (the league's shrunk past ROI on that track × market,
  Σpnl/(n+50), from tips played strictly before), market dummies, an Under dummy, and market × (fair, gap)
  interactions (so the gap is allowed to rank BTTS without ranking O/U 2.5).
* **Walk-forward**: tips first seen on day D are scored by a model fitted only on tips whose match was played
  strictly before D. A track needs ≥ 150 settled tips first (first scored day 2026-08-31).
* `score = p_win × odds − 1` (expected profit). Compared with two simpler rankings: `p_win` alone and the
  gap alone.

### Layers — two definitions, chosen on discovery only
* (a) **TERCILE** — per tip day, within track × market, rank by score: top third Best, middle Second, bottom
  Third (rank percentile (r − 0.5)/n; ties broken by the row key, so the result never depends on row order).
* (b) **FIXED** — two cut-offs per track × market = the 1/3 and 2/3 quantiles of the discovery-window scores
  (`docs/quality_tiers_2026_10/frozen_cuts_v1.json`), applied unchanged afterwards.
* Rule: per cell, the definition with the larger discovery Best − Third ROI difference.
* **Discovery** = tip days before 2026-10-05 (60% of the scored, settled tip days); **holdout** = 2026-10-05 to
  2026-10-25. Holdout was not looked at until every rule above was in code.

### Verdict rule (frozen)
REAL_3_LAYERS needs, on the holdout: ROI Best > Second > Third **and** a matchday block-bootstrap (2,000
resamples, days resampled jointly for Best and Third) one-sided p of (Best − Third ≤ 0), Benjamini–Hochberg
adjusted across the evaluated cells, below 0.10. Otherwise ONE_LAYER, with the reason: ORDER_SUGGESTIVE
(ordered, not significant), ORDER_FAILS, or INSUFFICIENT (holdout n < 45).

CLV: close = the last committed snapshot price strictly before kickoff. "Certified" = within 30 minutes of
kickoff (v9's certified-close rule; coverage is low, ~15% of tips); "≤ 6 h" = Pro's all-fixture rule.
CLV% = odds / close − 1.

## 2. Results by cell (never pooled)

ROI with 95% matchday block-bootstrap CI; CLV mean % (n).

### New-format O/U 2.5 — ONE_LAYER (ORDER_FAILS); chosen TERCILE
| window | layer | n | days | hit | ROI | 95% CI | CLV cert | CLV ≤6h |
|---|---|---|---|---|---|---|---|---|
| discovery | Best | 51 | 21 | 43% | +5.6% | [−35, +47] | −2.5 (9) | −0.95 (49) |
| discovery | Second | 53 | 21 | 47% | +2.8% | [−35, +41] | +5.5 (6) | +1.12 (52) |
| discovery | Third | 51 | 21 | 45% | +4.9% | [−26, +34] | +2.0 (10) | +0.39 (49) |
| **holdout** | Best | 67 | 15 | 37% | −11.0% | [−42, +20] | −1.4 (8) | −1.12 (65) |
| **holdout** | Second | 66 | 15 | 58% | +26.1% | [−2, +57] | −1.8 (12) | −1.14 (63) |
| **holdout** | Third | 67 | 16 | 48% | +3.3% | [−22, +24] | −3.3 (10) | +0.30 (64) |

Monotonicity (holdout): Spearman(layer, pnl) +0.02 (p 0.77); Best − Third −14.3 pp, CI [−55, +27], one-sided
p 0.75, BH q 0.98. Gap tiers on the same rows: SNIPER − VALUABLE −25.9 pp (p 0.96). Rankings on holdout,
Spearman with pnl: score −0.05, p_win only −0.11, gap only −0.03 — nothing ranks O/U 2.5 tips.

### New-format BTTS — ONE_LAYER (ORDER_FAILS); chosen FIXED
| window | layer | n | days | hit | ROI | 95% CI | CLV cert | CLV ≤6h |
|---|---|---|---|---|---|---|---|---|
| discovery | Best | 33 | 18 | 67% | +37.6% | [+10, +70] | +1.7 (4) | −1.40 (30) |
| discovery | Second | 33 | 11 | 73% | +43.3% | [+6, +76] | +2.5 (8) | +1.10 (30) |
| discovery | Third | 33 | 15 | 58% | +10.8% | [−16, +38] | −0.8 (4) | +0.10 (32) |
| **holdout** | Best | 15 | 9 | 47% | −3.8% | [−57, +52] | −1.3 (2) | +0.67 (15) |
| **holdout** | Second | 33 | 12 | 30% | −44.6% | [−83, −14] | −2.4 (5) | +0.30 (31) |
| **holdout** | Third | 46 | 18 | 61% | +19.4% | [−9, +55] | +3.0 (9) | +0.25 (46) |

Holdout: Spearman −0.20 (p 0.06, the wrong direction); Best − Third −23.2 pp, CI [−99, +49], p 0.77, q 0.98.
Gap tiers on the same rows: SNIPER +47.6% (6) / MARKSMAN −63.4% (11) / VALUABLE −2.9% (77); SNIPER − VALUABLE
+50.5 pp, CI [−45, +145], p 0.14, q 0.41 — not ordered (MARKSMAN worst), not significant.
Rankings, holdout Spearman: score −0.23 (p 0.03), p_win only −0.27 (p 0.01), **gap only +0.06**. In discovery
the gap alone ranked BTTS strongly (+0.45, p < 0.001); the learned score did not keep that, and on the
holdout neither did the gap. The fixed cut-offs also drifted: only 15 holdout tips cleared the Best bar.

### Standard O/U 2.5 — ONE_LAYER (ORDER_FAILS); chosen FIXED
| window | layer | n | days | hit | ROI | 95% CI |
|---|---|---|---|---|---|---|
| discovery | Best | 37 | 14 | 32% | −34.8% | [−61, −6] |
| discovery | Second | 35 | 11 | 51% | +5.5% | [−23, +49] |
| discovery | Third | 37 | 9 | 54% | +15.2% | [−6, +50] |
| **holdout** | Best | 53 | 12 | 38% | −20.3% | [−45, +2] |
| **holdout** | Second | 77 | 15 | 49% | +8.3% | [−16, +39] |
| **holdout** | Third | 0 | 0 | — | — | — |

Reversed in both windows (the score's top layer was the worst both times). The fixed cut-offs were learned
when the model scored standard tips very negatively; by October every score sat above the lower cut, so the
holdout Third layer is empty — fixed cut-offs drift as the model refits. Gap tiers on the same rows:
SNIPER −40.0% (9) / MARKSMAN −14.8% (11) / VALUABLE +0.8% (110), also reversed.

### Standard BTTS — ONE_LAYER (ORDER_FAILS); chosen TERCILE
| window | layer | n | days | hit | ROI | 95% CI |
|---|---|---|---|---|---|---|
| discovery | Best | 20 | 11 | 60% | +5.6% | [−26, +32] |
| discovery | Second | 21 | 12 | 62% | +11.8% | [−21, +48] |
| discovery | Third | 20 | 11 | 60% | +13.9% | [−25, +42] |
| **holdout** | Best | 16 | 9 | 31% | −43.7% | [−83, −9] |
| **holdout** | Second | 16 | 8 | 56% | +2.6% | [−47, +65] |
| **holdout** | Third | 16 | 9 | 69% | +24.9% | [−15, +65] |

Holdout Spearman −0.30 (p 0.04): the score ranks standard BTTS **backwards**. Gap tiers: 47 of 48 holdout tips
are VALUABLE, so the gap ladder has nothing to say here either.

### New-format Over 1.5 — ONE_LAYER (INSUFFICIENT)
28 holdout tips (Best 1 / Second 15 / Third 12). Third +28.7%, Second +11.7%. Gap tiers: SNIPER +26.0% (11),
MARKSMAN −14.0% (9), VALUABLE +50.7% (8). Too few to say anything.

### Standard Over 1.5, both Over 3.5 cells — ONE_LAYER (INSUFFICIENT)
≤ 4 holdout tips each.

### Season-long tiers, ledger vs first sight (context for the owner's numbers)
| cell | tier | ledger (upgraded) ROI (n) | first-sight ROI (n) | first-sight 95% CI |
|---|---|---|---|---|
| NF BTTS | SNIPER | +24.7% (82) | **+52.2% (48)** | [+21, +83] |
| NF BTTS | MARKSMAN | +16.0% (51) | +3.1% (33) | [−34, +41] |
| NF BTTS | VALUABLE | +7.7% (112) | +6.7% (164) | [−7, +23] |
| NF O/U 2.5 | SNIPER | +2.2% (83) | +3.2% (50) | [−27, +32] |
| NF O/U 2.5 | MARKSMAN | −1.3% (198) | −23.4% (70) | [−47, +3] |
| NF O/U 2.5 | VALUABLE | +8.4% (206) | +8.5% (367) | [−7, +23] |
| Std O/U 2.5 | SNIPER | −10.2% (42) | −25.4% (28) | [−59, +17] |
| Std O/U 2.5 | MARKSMAN | −17.9% (107) | −6.6% (33) | [−41, +30] |
| Std O/U 2.5 | VALUABLE | −10.1% (234) | −11.7% (322) | [−22, +0] |
| Std BTTS | SNIPER / MARKSMAN / VALUABLE | −18.6% (16) / −13.3% (17) / +3.4% (129) | −3.8% (4) / −13.1% (9) / +0.3% (149) | |

(Settled tips in the matched record; the owner's figures include a few tips this record could not match.)

## 3. Comparison with the current gap tiers (holdout, same rows)

| cell | learned layers ordered? | Best − Third (p, q) | gap tiers ordered? | SNIPER − VALUABLE (p, q) |
|---|---|---|---|---|
| NF O/U 2.5 | No | −14.3 pp (0.75, 0.98) | No | −25.9 pp (0.96, 0.96) |
| NF BTTS | No | −23.2 pp (0.77, 0.98) | No | +50.5 pp (0.14, 0.41) |
| Std O/U 2.5 | No (Third empty) | — | No | −40.8 pp (0.81, 0.96) |
| Std BTTS | No (reversed) | −68.6 pp (0.98, 0.98) | — (no SNIPER) | — |

Neither scheme orders tips on this season's holdout. The learned score is not an improvement on the gap
tiers; on BTTS the gap is, if anything, the better of the two signals (the SNIPER bucket), but it is not
significant on the holdout alone.

## 4. Shadow and forward test

**Shadow (live from the next predict run, research only).** `python -m src.research.quality_tiers --shadow`
runs as its own tolerant step in `predict.yml`, after the Telegram step and before the commit (a failure is a
warning; it can never stop tips or the commit). For every tip on the current pre-match board that is not yet
logged it appends one row to `output/quality_tier_shadow.csv`: first-sight odds / opposite odds / model p /
fair p / gap / first-sight gap tier, the model version, `p_win`, `score`, the FIXED layer and its cut-offs,
`quality_version = qt_v1`, `fitted_through` (last match day the fit saw) and `train_n`. Written through
`src/research/first_sight.FirstSightStore` with a digest sidecar: earliest row wins, a logged row is never
revised (a later in-place ledger upgrade does not move it), rows observed at/after kickoff are refused, and a
tampered file blocks further appends. The p_win model is refitted each run from the frozen seed
(`docs/quality_tiers_2026_10/seed_first_sight_tips.csv`, the season's first-sight tips) plus logged shadow rows
whose results the ledgers now hold, using only matches played before today. The TERCILE layer is not written
at first sight (it needs the whole day's tips); the evaluator recomputes it from the logged first-sight scores.

**Forward test** — preregistered in Pro, `registry/quality_tiers/registration_v1.json` (+ `LOCK_v1.json`
SHA-256): rows with `ledger_generated_at ≥ 2026-10-27T00:00:00Z` and `lag_min ≤ 60`; layers by each cell's
frozen definition; per cell ≥ 150 settled tips and ≥ 20 matchdays (and ≥ 30 per layer) before any verdict;
primary metric Best − Third flat ROI difference with a matchday block-bootstrap, plus Best > Second > Third
monotonicity; BH-FDR across cells; the same test run on the first-sight gap tiers on the same rows. Statuses:
ACCRUING / INSUFFICIENT / FORWARD_SUPPORTED / REJECTED — **never CONFIRMED** from this registration.

## 5. Switch-on criteria (all required, per track × market)

1. The forward test reaches its minimums and the cell is FORWARD_SUPPORTED (ordered, BH q < 0.10).
2. The learned layers beat the gap tiers on the same forward rows (larger Best − Third difference).
3. Certified-close CLV of the Best layer is not below the Third layer's.
4. Owner approval, recorded. A cell that fails keeps one layer ("Tip") — said plainly, not dressed up.

Any change to features, C, cut-offs, the split or the verdict rule is a new version (`qt_v2`) with a new
forward window; the v1 numbers are never re-run with tweaks.

## 6. How stakes could follow layers later (only where monotonic)

Only in a cell that passed section 5, and only as a re-weighting inside the cell's existing budget: e.g. Best
1.0×, Second 0.75×, Third 0.5× of the track's current base stake, never raising total exposure, never turning
a PAPER track into a real-money one, and subject to the bankroll policy's caps and the research lock. A cell
with one layer keeps flat staking. Today that is **no cell**.

## Files

* `src/research/quality_tiers.py` — score, walk-forward, layers, evidence, shadow logger.
* `docs/quality_tiers_2026_10/study_results.json` — every table above, plus rankings and Brier per cell.
* `docs/quality_tiers_2026_10/frozen_cuts_v1.json` — the FIXED cut-offs and each cell's chosen definition.
* `docs/quality_tiers_2026_10/seed_first_sight_tips.csv` — the first-sight record (training seed for the shadow).
* `tests/test_quality_tiers.py` — no look-ahead, deterministic layers, immutable shadow, no production effect.

Reproduce: `python -m src.research.quality_tiers --study --snapshots <prematch_snapshots.parquet>` (the
snapshot parquet is Pro's `experiments/valuable_2026_10/prematch_snapshots.parquet` refreshed with
`src/studies/valuable/snapshots.py`).
