# Pro vs v9 cross-check — 2026-10-07

The gate brief asked for this and it had never been done: **if Pro disagrees with v9's backtest
optimiser, report the disagreement rather than quietly choosing the result you prefer.** Until
2026-10-06 there was nothing meaningful to compare — v9's artifacts had been frozen for five
weeks. Now both sides are current.

Repos at `wowza-betting 24da45df`, `wowzaV9-Pro 8b4770f`, `wowza_v11 1636111`. All three are
running; a claim in an earlier session that v11 was stale was wrong and came from reading an
unpulled local checkout.

---

## 1. Headline: they agree on what is bad and disagree on what is promising

Comparing `v10/output/threshold_verdicts.csv` (Pro, regenerated 2026-10-06) against
`v9/output/league_market_gate_registry.json` (v9, regenerated 2026-10-06) on the 19 cells both
cover:

```
agree 11   disagree 8
```

**Every single disagreement runs the same way: Pro is more conservative.** There is no cell where
Pro is positive and v9 is negative.

| cell | Pro | v9 |
|---|---|---|
| Austrian Bundesliga · OU25 | INSUFFICIENT_DATA | CANDIDATE |
| Denmark Superliga · OU25 | INSUFFICIENT_DATA | CANDIDATE |
| Finland Veikkausliiga · OU25 | UNCHANGED | CANDIDATE |
| Ireland Premier Division · OU25 | INSUFFICIENT_DATA | CANDIDATE |
| Japan J-League · OU25 | UNCHANGED | CANDIDATE |
| Mexico Liga MX · OU25 | UNCHANGED | CANDIDATE |
| Sweden Allsvenskan · OU25 | INSUFFICIENT_DATA | CANDIDATE |
| League One · OU25 | UNCHANGED | CANDIDATE |

And they agree completely on the negatives — **USA MLS, Ligue 2, Norway, China, Bundesliga 2,
League Two, Championship, Brazil, La Liga 2 OU25 and Argentina OU25** are flagged by both.

**Pro's own bottom line is blunter than v9's: `holds_oos` is True for 0 of 23 cells.** Its
verdicts are 12 `UNCHANGED` (keep the existing threshold) and 11 `INSUFFICIENT_DATA`. Not one
league earns a new threshold on Pro's test.

That is the same conclusion `scripts/league_thresholds.py` reaches independently in v9 — 0 of 19
clear all three windows — so the two systems agree exactly where it matters most: **no per-league
threshold is currently justified.**

---

## 2. Pro's calibration result, which v9 does not measure

`v10/output/calibration_study.csv`, regenerated 2026-10-06:

| track | n | claimed | realised | gap | ECE | Platt helps? |
|---|---|---|---|---|---|---|
| new_format · OU25 | 764 | 48.1% | **55.6%** | **−7.6pp** | 0.088 | no |
| standard · OU25 | 548 | 50.6% | 52.7% | −2.2pp | 0.059 | yes (small) |

The new-format model is **UNDER**-confident by 7.6 points — it claims 48% and realises 56%. That
is the opposite direction to the headline defect recorded elsewhere in this estate (+13.64pp
over-confidence measured on *staked selections*), and both can be true: the model is
under-confident across all scored fixtures and over-confident on the subset it chooses to bet.
Which is itself a statement about the tier ladder rather than the model.

Platt scaling improves standard (brier delta −0.005, CI excluding zero) and does nothing for
new-format. Isotonic helps neither.

---

## 3. Where Pro and v9 answer different questions, and both are right

Pro's accuracy study says the model has **no lift over a base-rate predictor**:

```
OU25 standard     -0.06pp        BTTS            +0.74pp
OU25 new_format   -1.37pp        Over 1.5        +0.00pp
```

and on actual picks, `BTTS / new_format` is **−0.81pp** — worse than always predicting the base
rate. That is the same track as Argentina BTTS, the estate's strongest live cell (+28.42u on 73
staked bets, CI [+0.168, +0.626]).

These are not contradictory. Pro measures **prediction** — does the model guess the outcome
better than the base rate. v9's ledger measures **betting** — did the selections beat their
prices. A model with zero prediction lift can still profit by being selective about *price*
rather than knowledgeable about *outcome*.

But note which way the evidence leans when all three are put together:

| source | what it says about Argentina BTTS |
|---|---|
| v9 ROI | **+28.42u**, CI excludes zero |
| v9 CLV | **−1.91%**, positive on only 44% of bets |
| Pro accuracy | **−0.81pp** lift on BTTS / new_format |

**Two of the three say there is no edge.** The one that says there is, is the one most exposed to
luck. This is exactly what preregistered hypothesis `H-BTTS-03` was written to settle, and the
cross-check strengthens rather than weakens the case for waiting.

---

## 4. Coverage gap

**Pro's threshold verdicts cover OU25 only** — no BTTS, Over 1.5 or Over 3.5. So the cell the
owner considers the clearest winner has **no independent Pro verdict at all**, and the Pro number
quoted above is an accuracy study, not a threshold one.

That is the single most useful thing to close: Argentina BTTS is carrying the staked book
(+33.95u of new-format's +21.12u comes from BTTS), and only one of the two research systems
examines it.

---

## 5. A defect found while doing this

`v10/.github/workflows/pro_research.yml` runs `python -m src.validation.pick_accuracy --write`
daily, but **`output/pick_accuracy.json` is not in its `git add` list.** So the study is
recomputed every day and the result discarded; the committed file is from 2026-09-22.
`accuracy_vs_edge.json` is worse — the workflow never invokes it at all, so that file is a
one-off from the same day.

Both numbers quoted in §3 are therefore **from before the September data fixes**, i.e. computed
on a model that has since changed. They are reported here as the current artifact because that is
what the repo holds, and flagged as stale because they are.

This is the same failure shape that froze v9's gates for five weeks: a job that runs green and
persists nothing. v9 now has a test that fails CI when a workflow runs the full enrichment
without a cache; Pro has no equivalent guard on its staging lists.

---

## 6. What this changes

Nothing in production. Concretely:

* **The conservative reading wins.** Where the two disagree, Pro says "not yet" and v9 says
  "promising". Nothing should be promoted on a v9 CANDIDATE that Pro calls INSUFFICIENT_DATA.
* **The negatives are now confirmed twice.** USA MLS, Norway, Ligue 2, China, Bundesliga 2 and
  League Two are flagged by two independent methods. That is a stronger basis for the MLS
  decision already taken, and for considering Norway next.
* **Argentina BTTS remains the open question**, and the cross-check makes it sharper rather than
  safer: ROI says yes, CLV and Pro's accuracy study both say no.

### Worth doing next

1. Add BTTS / Over 1.5 / Over 3.5 to Pro's threshold verdicts so the strongest live cell gets an
   independent read.
2. Stage `pick_accuracy.json` and invoke `accuracy_vs_edge` in `pro_research.yml`, so the
   accuracy verdicts stop being a September snapshot.
3. Re-run this cross-check after Sunday's retrain, which is the first test of whether the
   Championship / League Two approval flip holds.
