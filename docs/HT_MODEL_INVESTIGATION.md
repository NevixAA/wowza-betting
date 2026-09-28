# The half-time models: what was asked, what was found, what changed

Date: 2026-09-28. Written for: Nevo.

You asked whether we track HT over/under 0.5 and 1.5, then said to upgrade the models. This is
what came out of that. **The headline is not the one either of us expected.**

---

## Bottom line

1. The HT tips **were never graded once**, since the ledger was created. Fixed — 38 now settled.
2. That grading exposed a bigger problem. **74% of the HT training labels were fabricated.**
3. With honest labels the HT models have **no predictive power at all** (AUC 0.512 vs 0.50 for a
   coin flip), and they **stop sending tips by themselves** at the next retrain.
4. So the "upgrade" is a deletion, not an addition: the thing that looked like a working model
   was measuring which leagues had missing data.

Nothing about the main O/U 2.5 track changed. Verified explicitly.

---

## What the question surfaced

Yes — all four lines are tipped and logged: `ht_over05`, `ht_under05`, `ht_over15`,
`ht_under15`, in `output/ht_ledger.csv`.

But **none had a result, P/L, entry odds or CLV.** Three faults, all in the grader:

- It graded against `af_ht_history.parquet`, which holds only **new-format** leagues (MLS,
  Argentina, Brazil, Japan). HT tips are **standard-only** (League One, League Two, La Liga 2).
  The league overlap is **empty** — not one row could ever match. `fd_history.parquet` had the
  right scores all along: 21,840 rows covering exactly those leagues.
- Club names differ between sources (invariant 11): the ledger says `Plymouth Argyle`,
  football-data says `Plymouth`. Swapping the source alone lifted matching 0 → 2 of 47;
  league-scoped `team_names.resolve` gives **38 of 47**.
- The odds join had the same fault: 4,478 HT prices are captured, but only 4 of 38 matched.
  Now 9.

Fixed in `f005cecb`.

---

## The real finding

Grading 38 tips showed the model claiming **0.782** and delivering **0.632** — 15pp
overconfident. To find out whether that was a bad model or a bad selection rule, I built the HT
backtest, **which did not exist anywhere**.

The first run looked encouraging: AUC 0.663 out-of-sample over 25,457 fixtures, Brier skill
+5.8%. A good ranker with broken tail calibration — a fixable problem.

Then the robustness check asked whether the test window was normal, and it was not: half-time
scoring ran at 68.3% in the priced window against 44.1% across the whole backtest. **A 24-point
gap that should not exist.**

44% is the impossible number. Roughly 70% of football matches have a goal by half time. Broken
down by league it was obvious:

| league | HT-scoring rate in training | reality |
|---|---|---|
| Brazil Serie A | **8.6%** | 68.2% |
| USA MLS | **11.4%** | 72.2% |
| Japan J-League | **11.7%** | 62.9% |
| League One | 57.7% | 69.6% |

Both source files were fine — `af_ht_history` 68.8%, `fd_history` 70.2%. The zeros were being
manufactured in `src/data_loader.py`:

```python
out["ht_over05"] = (out["ht_total_goals"] >= 1).astype(float)
```

`NaN >= 1` is `False`, and `.astype(float)` makes that **0.0**. Every fixture with no half-time
score was labelled *"no half-time goal"*. The comment directly above it already said "only
populated when HTHG/HTAG available" — the code just didn't do it. And `dropna()` could never
remove them, because the column was never null.

**62,470 of 84,511 rows — 73.9% — were invented negatives.**

`src/feature_engineering.py:387` has always masked this correctly. These two sites didn't.

---

## What that did to the models

| | corrupt labels | honest labels |
|---|---|---|
| AUC | 0.663 | **0.512** |
| Brier skill vs base rate | +5.8% | **+0.0%** |
| calibration in the tip band | **+22.9pp** overconfident | flat (−0.3pp) |
| tips that would fire, 11,198 fixtures | 1,117 | **0** |
| vs the market (Brier) | 0.2219 vs 0.2148 | 0.2010 vs 0.1998 |

The apparent skill was the model learning **which leagues had missing data**, not which matches
score early. MLS, Brazil and Japan were labelled ~90% "no HT goal" while English leagues were
~58%, so any league-correlated feature separated *missingness* — and scored a healthy AUC for it.
That also explains the +15pp measured on the live tips: the model was confidently predicting
"no half-time goal" for profiles that in reality score in the first half about 70% of the time.

**On honest labels neither model reaches its own tip threshold even once in 11,198 out-of-sample
fixtures.** HT tips stop by themselves at the next retrain. That is the correct outcome, and it
is self-correcting — no threshold change needed.

---

## The result I nearly reported, and why I didn't

The corrupt run also showed a 50/50 blend of model and de-vigged market beating the market by
+0.0086 Brier, with +8.8% ROI over 226 bets at ≥8% edge. That would have been a big claim.

`scripts/ht_robustness.py` attacked it four ways and it failed three:

- **split-half:** +12.5% in the first half, **−8.8%** in the second. Sign flip.
- **bootstrap:** 95% CI [−11.9%, +16.1%] at ≥5% edge — spans zero.
- **placebo:** shuffling the model's probabilities was **indistinguishable** at ≥5% (p=0.095).
- every selected bet was UNDER — a directional bet on the base rate wearing a model's clothes.

On honest labels it is −0.0001 anyway. Recorded so nobody revives it.

---

## A bug I made, kept in the code as a warning

My first walk-forward isotonic helper did a defensive `sort_values().reset_index()` inside the
function. pandas' sort is not stable, so with thousands of same-date fixtures it permuted rows,
and the returned Series aligned onto the **wrong fixtures**.

It announced itself: **AUC fell from 0.667 to 0.475.** Isotonic regression is monotone and
cannot change ranking, so any AUC movement at all is proof of misalignment. That is the same
class of trap as `merge_asof` in v11's momentum work — where the symptom was plausible and it
went unnoticed for weeks. There is now a guard test asserting AUC is preserved.

---

## Was the HT market ever beatable?

Separately worth knowing, measured on 551 fixtures with a captured closing pair:

- bookmaker margin **6.89%** on the 0.5 line, **7.24%** on 1.5 — roughly twice the O/U 2.5
  market Wowza actually targets
- to break even you must beat the de-vigged price by **+4.78pp** (0.5) or **+2.62pp** (1.5)
- the market itself has almost no fixture-level discrimination (+0.2% / +2.1% Brier skill)

That last point is not an opening. Half-time goals are close to a pure base-rate process — there
is little to discriminate, which is why model and market both look unskilled. The bookmaker
handles that by charging 7%. We cannot.

---

## What changed, and what didn't

**Changed**
- `update_results.py` — HT grading now works (right source, league-scoped name resolution, odds
  join fixed, fair-odds settlements flagged `settled_at_fair_odds_no_market_price`)
- `src/data_loader.py` — two lines, HT targets masked to NaN where no half-time score exists
- new read-only scripts: `ht_backtest.py`, `ht_recalibrate.py`, `ht_robustness.py`

**Not changed**
- no threshold, stake, tier, selection rule or notification
- no model retrained by hand — the daily 03:00 retrain picks the fix up on its own
- the main O/U track: `over25`, `btts`, `over15`, `over35` all still 84,511 rows at
  0.5058 / 0.5278 / 0.7463 / 0.2824. Checked explicitly, because invariant 1 requires the tracks
  stay isolated.

**Worth your decision**
- HT tips will go quiet after tonight's retrain. If you would rather they stop immediately and
  visibly, that is a notifier change and yours to make.
- The three research scripts are in `v9/scripts/`. Strictly this research belongs in Pro; they
  are read-only and touch no production path, but moving them there is reasonable.
