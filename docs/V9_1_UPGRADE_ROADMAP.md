# v9.1 upgrade roadmap

Date: 2026-10-01. Status: **v9 remains champion. `safe_to_replace_v9: false`.**
Machine-readable: `output/v91_readiness.json`.

v9.1 is a **shadow challenger path**, not a release. Nothing below replaces production.

---

## 1. Current production weaknesses

**The meta-stack is validated against itself.** `src/model.py:240-245` fits the stacker on
`y_test` and scores it on the same block. Every meta metric ever reported is in-sample, and the
meta has seen the holdout labels — which burns that block for the promotion gate that scores
incumbent against candidate on exactly it. The meta is live (`src/model.py:349`).

Measured cost: **0.0006–0.0014 log loss** of inflation. That is small in absolute terms and
**larger than the deltas the promotion gate decides on (0.00004–0.003)**. The gate has been
resolving a signal finer than its own measurement bias.

**The promotion gate promoted worse models.** It shipped anything whose log loss had not risen
beyond a tolerance. **17 of 56 logged decisions promoted a model that got worse**, including
`0.68778 → 0.68782` and a **+0.019** regression on `ht_over05`. Daily retraining under that rule
is a random walk on the live model.

**Calibration is the core defect, and it is visible everywhere.** Every honest candidate shows a
calibration slope of 0.74–0.79 (mean-blend and best-single: 0.33–0.50). Only the leaky meta looks
calibrated at 1.006 — because it was fitted on the block it is scored on. Live, the model claims
53.07% and realises 40.44% across 1,328 settled bets: **+12.6pp overconfident, z = 9.38**.

**The tier ladder is inverted.** SNIPER — full stake — is the worst-calibrated tier (+19.8pp)
and beats its break-even by −0.3pp, i.e. not at all. No tier clears its own bar.

**The drift rule is unvalidated and probably backwards.** `grep drift src/backtest.py` returns 0.
Live, `Confirmed` (which the rule upgrades on) runs −10.8% while `Neutral` (ignored) runs +8.0%.
It reads `odds_history_v9.json` — **260 keys against 408,405 rows of captured curve**.

**Lineage.** 80,868 available fixtures, 57,699 used, **23,169 stranded (28.65%)**, ~2.9k dupes.

---

## 2. Strong evidence from Pro

Chronological, 4 folds, n=26,185 — reproduced on current data:

| target | AUC | LogLoss gain | accuracy lift |
|---|---|---|---|
| BTTS | 0.5688 | +0.0099 | +1.30pp |
| Over 1.5 | 0.5950 | +0.0113 | +0.07pp |
| **Over 2.5** | 0.5923 | **+0.0140** | **+4.84pp** |
| Over 3.5 | 0.6003 | +0.0140 | +0.23pp |

**The football signal is real and modest.** HGB is the strongest single learner. Over 2.5 shows
the clearest lift. **Retraining beats freezing** (canonical 0.24697 vs frozen 0.25525 over 41
walk-forward months) — the estate's one strong positive, and the reason never to delete history.

**Weak:** per-league threshold tuning. 23 cells, `holds_oos` False on all — but 13 are
`INSUFFICIENT_DATA`, so only 10 were genuinely tested. A warning, not a closed question.

---

## 3. Weak / negative evidence from V11

**Market + Wowza = market.** Brier 0.2405 → 0.2404, delta **+0.00006**, and Wowza helps in
**10 of 20 segments** — a coin flip. Zero BET-eligible selections, because `v11_ev_lb` is
negative on all 1,199 rows (max −0.038). That is the engine working: blending a no-information
model into an efficient price returns the price, and the price minus vig is negative EV by
construction.

**Useful infrastructure:** de-vig, consensus, movement capture, the NO_BET default, the CLV gate.
**Failed hypothesis:** that Wowza adds incremental probability information over the market.
**Void:** all momentum results from commits `929f783` / `4ed8aa9` (merge_asof index reset).

---

## 4. Safe immediate engineering fixes — CLASS A

Correctness only; no selection behaviour changes.

- `src/model_validation.py` — leak-free FIT|CAL|META|HOLDOUT harness, dataset ids, model SHAs **DONE**
- `src/promotion_gate.py` — CHAMPION/CHALLENGER/RESEARCH/REJECTED, nine checks **DONE, not yet wired**
- `src/market_movement.py` — per-market open→close curve, NO_CURVE ≠ NEUTRAL **DONE**
- HT grading fix, price-at-tip-time, forward observation log **DONE**
- per-book quote retention, T-6h/3h/1h/30m/10m capture ladder **TODO**
- canonical fixture identity, dataset versioning, stranded-data monitor **TODO**

---

## 5. Shadow challengers — CLASS B

Must run alongside v9 and never replace it without clearing the gate.

| challenger | status | evidence |
|---|---|---|
| corrected meta stack | SHADOW | leak measured at ~0.001 |
| **HGB Over 2.5** | **CHALLENGER** | −0.00242, CI [−0.0044, −0.0004] |
| HGB BTTS | CHALLENGER | −0.00131, CI spans zero, ECE +0.011 breaches limit |
| HGB Over 3.5 | CHALLENGER | −0.00180, CI spans zero |
| HGB Over 1.5 | **REJECTED** | +0.00171 — worse |
| goal-distribution model | RESEARCH | not yet built |
| canonical training source | SHADOW | clean lineage, do not repoint production |

**HGB Over 2.5 is the strongest candidate and is still not close.** Its CI upper bound
(−0.00038) does not clear the 0.001 material floor, its holdout is historical rather than
prospective, and the incumbent it was measured against is the *corrected* stack, not the leaky
one actually serving. All three must be fixed before promotion is even discussable.

---

## 6. Research only — CLASS C

V11 residual betting · movement-driven tier changes · builder bets · 1X2 betting · player-prop
betting · league-specific anomalies · high-residual microstructure cells. **BTTS weighting.**

---

## 7. Required data collection

1. **Closing-line ladder** — T-6h/3h/1h/30m/10m. A "last snapshot" hours before kickoff is not
   a close, and CLV computed against it is not CLV.
2. **Per-bookmaker quotes retained** — currently collapsed into consensus and discarded. Without
   them there is no lead/lag research, no dispersion, no line shopping, no best executable price.
3. **Market anchor hierarchy** — exchange fair → cross-book median de-vig → single-book de-vig.
   A one-sided price must be flagged `INSUFFICIENT_MARKET_DATA`, never silently converted to EV.
4. **Settlement provider alignment** — for every prop market, record our stat provider against
   the book's. Unaligned ⇒ `deployment_mode = BLOCKED`.

---

## 8. Model promotion gates

`src/promotion_gate.py`, every check required:

same dataset id · n ≥ 1,000 · log loss not worse (no tolerance band) · Brier not worse ·
block-bootstrap CI excluding zero · **material improvement ≥ 0.001** · ECE not worse by >0.01 ·
prediction canary · no league regression >0.02 on n ≥ 150 · reproduced on a second independent
period.

The material floor is set **above the measured 0.001 leak bias**: an improvement smaller than
our own measurement error is not one we can see.

---

## 9. Betting promotion gates

Real prices only · sufficient closing coverage · mean CLV > 0 · block-bootstrap CLV lower bound
> 0 · P/L positive after realistic costs · not dependent on one league or cell · survives
Benjamini-Hochberg FDR at q=0.10.

**ROI alone is never sufficient.** BTTS is the worked example: +30.6% ROI, and 85% of it is one
league.

---

## 10. October tasks

- wire `promotion_gate` into `pipeline.py` behind a flag; log verdicts without acting
- build the goal-distribution challenger (§7) and score it on the same holdout
- closing-line ladder + per-book retention
- market anchor hierarchy with `INSUFFICIENT_MARKET_DATA`
- begin prospective shadow logging: `p_v9`, `p_v91`, `p_pro_hgb`, `p_market` per fixture
- settlement-provider alignment audit

## 11. November confirmation tasks

- first prospective period closes; score HGB O2.5 against the *leaky production* model, which is
  the only honest comparison once a clean forward window exists
- half-time decision rule fires at 1,000 priced fixtures (`registry/ht_decision_rule.yaml`)
- BTTS preregistered test continues accruing (H-BTTS-01/02)

## 12. December decision checkpoint

Re-run `scripts/v91_readiness.py`. Promote only components reading CHAMPION. Expect: none.
A December with everything still CHALLENGER is a **success**, not a failure — it means the gate
is doing its job.

## 13. 2027/28 migration path

Only if the evidence arrives: canonical training source → corrected stack → best per-market
challenger → segmented movement rules → recalibrated tier ladder. Each one promoted alone, each
with its own forward window. Never as a bundle.

---

## The principle

The question is not *how do we make the backtest look better*. It is **what information exists
before kickoff that improves future probability forecasts or beats the closing market after
proper controls.**

The goal of v9.1 is not more bets. It is better probabilities → better calibration → cleaner
value estimation → fewer false edges → better selective betting.
