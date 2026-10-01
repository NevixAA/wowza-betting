# v9.1 — final deliverable

Date: 2026-10-01. **v9 remains champion. `safe_to_replace_v9: false`.**
Nothing in production changed: `src/model.py`, `src/betting.py`, `src/predict.py` untouched.

---

## A. Audit — what is broken or statistically weak in v9

**1. The meta-stack is validated against itself.** `src/model.py:240-245` fits the stacker on
`y_test` and scores it on the same block. Every meta metric ever reported is in-sample, and the
meta has seen the holdout labels — which burns that block for the promotion gate that scores
incumbent against candidate on exactly it. The meta is live (`src/model.py:349`).
**Measured cost: 0.0006–0.0014 log loss — larger than the 0.00004–0.003 deltas the gate decides
on.** The gate was resolving a signal finer than its own bias.

**2. The gate promoted worse models.** 17 of 56 logged decisions, including `0.68778 → 0.68782`
and a **+0.019** regression on `ht_over05`.

**3. Overconfidence is the core defect.** Live: claimed 53.07%, realised 40.44% on 1,328 settled
bets — **+12.6pp, z=9.38**. Every honest candidate shows calibration slope 0.74–0.79; mean-blend
and best-single 0.33–0.50. Only the leaky meta looks calibrated (1.006), because it was fitted on
the block it is scored on.

**4. The tier ladder is inverted.** SNIPER — full stake — is worst calibrated (+19.8pp) and
clears its break-even by −0.3pp. No tier clears its own bar.

**5. Drift is unvalidated and probably backwards.** `grep drift src/backtest.py` = 0. Live,
`Confirmed` (upgraded on) runs −10.8%, `Neutral` (ignored) runs +8.0%. It reads
`odds_history_v9.json` — **260 keys against 408,405 rows of captured curve**.

**6. Lineage.** 80,868 available fixtures, 57,699 used, **23,169 stranded (28.65%)**, ~2.9k dupes.

**7. No prospective record existed.** Every comparison was retrospective, so the second-period
check could never pass and every challenger was permanently stuck.

---

## B. What Pro has genuinely discovered

**Strong, replicated on current data** (4 chronological folds, n=26,185):

| target | AUC | LogLoss gain | lift |
|---|---|---|---|
| BTTS | 0.5688 | +0.0099 | +1.30pp |
| Over 1.5 | 0.5950 | +0.0113 | +0.07pp |
| **Over 2.5** | 0.5923 | **+0.0140** | **+4.84pp** |
| Over 3.5 | 0.6003 | +0.0140 | +0.23pp |

The football signal is real and modest. **HGB is the strongest single learner.**
**Retraining beats freezing** (0.24697 vs 0.25525 over 41 months) — the one strong positive, and
the reason never to delete history.

**Weak:** per-league threshold tuning — 23 cells, `holds_oos` False on all, but **13 are
`INSUFFICIENT_DATA`**, so only 10 were genuinely tested. A warning, not a closed question.

**Negative:** replacing the training source with the canonical set improves OOS log loss only
marginally. Clean the lineage; do not repoint production because the row count is larger.

---

## C. What V11 has genuinely discovered

**Useful infrastructure:** de-vig, consensus, movement capture, NO_BET default, the CLV gate.

**Failed hypothesis:** that Wowza adds incremental probability information over the market.
Brier 0.2405 → 0.2404, **delta +0.00006**, and Wowza helps in **10 of 20 segments — a coin
flip**. Zero BET-eligible selections because `v11_ev_lb` is negative on all 1,199 rows
(max −0.038). That is the engine working, not failing.

**Void:** all momentum results from commits `929f783` / `4ed8aa9` (merge_asof index reset).

---

## D. v9.1 architecture

```
DATA     canonical lineage (Pro)          [SHADOW — do not repoint production]
           |
FEATURES build_features + ht_features
           |
MODEL    FIT | CAL | META | HOLDOUT       src/model_validation.py   [harness, SAFE_NOW]
           ├─ v9 ensemble                 CHAMPION
           ├─ corrected meta stack        SHADOW
           ├─ HGB per market              CHALLENGER (O2.5/BTTS/O3.5), REJECTED (O1.5)
           └─ goal distribution           CHALLENGER — coherent, 1X2 free
           |
MARKET   exchange → cross-book median → single book → INSUFFICIENT_MARKET_DATA
           src/market_anchor.py           [SAFE_NOW]
           src/market_movement.py         NO_CURVE ≠ NEUTRAL
           |
DECISION edge = p_model − p_market_anchor (never 1/odds from a one-sided price)
           tiers UNCHANGED                 [production untouched]
           |
GATE     src/promotion_gate.py            CHAMPION/CHALLENGER/RESEARCH/REJECTED
           |
SHADOW   src/shadow_compare.py            per-fixture prospective log  [the unblocker]
           |
CONTROL  BH-FDR q=0.10 + White's Reality Check + preregistration registry
```

---

## E. Code changes, repo by repo

**v9 (`NevixAA/wowza-betting`) — all new files; nothing existing in the predictive path edited.**

| file | purpose |
|---|---|
| `src/model_validation.py` | leak-free FIT/CAL/META/HOLDOUT, dataset ids, model SHAs, ECE, calibration slope |
| `src/promotion_gate.py` | 10-check gate, block bootstrap, canary, material floor |
| `src/goal_distribution.py` | Dixon-Coles coherent score distribution, RPS, coherence checks |
| `src/market_anchor.py` | anchor hierarchy + `INSUFFICIENT_MARKET_DATA` |
| `src/shadow_compare.py` | prospective per-fixture log, BH-FDR, White's Reality Check |
| `src/market_movement.py` | per-market open→close curve *(earlier this session)* |
| `scripts/movement_segments.py` | §9 segmentation table |
| `scripts/v91_readiness.py` | generates `output/v91_readiness.json` from evidence |
| `registry/preregistered_hypotheses.json` | 6 frozen hypotheses |
| `docs/BTTS_EDGE_VALIDATION.md`, `docs/V9_1_UPGRADE_ROADMAP.md`, this file | |
| `tests/` ×5 | 51 tests |

**Pro (`wowzaV9-Pro`)** — read only; its research was reproduced, not modified.
**V11 (`wowza_v11`)** — untouched; conclusion is "keep collecting", which needs no change.

---

## F. Tests — 51 passing

Leakage (a model fitted on its own scoring block must score better on pure noise) ·
chronological order · determinism over tied dates · dataset identity · ECE and calibration
slope · gate: worse model never promotes, tiny regression stays challenger, noise cannot promote
(12 seeds), small sample refused, collapse rejected, league regression blocks, block bootstrap
wider than iid on correlated data, replay of the 17 historical bad promotions · distribution
coherence · RPS ordering · shadow dedup and prospective-window semantics · BH rejects a lone
p=0.047 among 19 · White's punishes the winner of a search · anchor refuses one-sided prices.

---

## G. Shadow outputs — how v9 and v9.1 get compared

`output/v91_shadow.csv`, one row per fixture per run, written **before kickoff**:

```
fixture_id, logged_at, match_date, league, model_type,
p_v9, p_v91, p_pro_hgb, p_goal_distribution, p_market,
v9_side, v91_side, v9_tier, v91_tier, v9_edge, v91_edge,
entry_odds, closing_odds, clv, result,
v9_model_sha, v91_model_sha, dataset_id
```

Append-only, deduped on first sight. `prospective_window(since)` filters on **`logged_at`, not
`match_date`** — a fixture played after registration but predicted before it is still
retrospective.

---

## H. Production recommendations

| proposal | class | why |
|---|---|---|
| validation harness | **SAFE_NOW** | measurement only, 51 tests |
| promotion gate (wired, log-only first) | **SAFE_NOW** | strictly stricter than today |
| market anchor + INSUFFICIENT_MARKET_DATA | **SAFE_NOW** | refuses fabricated anchors |
| NO_CURVE ≠ NEUTRAL | **SAFE_NOW** | already shipped |
| shadow log | **SAFE_NOW** | records, never acts |
| corrected meta stack | SHADOW | leak measured, not wired |
| HGB Over 2.5 | SHADOW | strongest candidate, still not close |
| HGB BTTS / Over 3.5 | SHADOW | CI spans zero |
| goal distribution | SHADOW | coherent, but worse on O1.5 and BTTS |
| canonical training source | SHADOW | clean lineage, don't repoint |
| **HGB Over 1.5** | **REJECT** | +0.00171 — worse |
| BTTS weighting | RESEARCH | 85% Argentina, 5 weeks |
| movement-driven tiers | RESEARCH | 19 cells, 1 clears, 1.0 expected by chance |
| V11 residual betting | RESEARCH | +0.00006, 10 of 20 segments |
| player-prop betting | RESEARCH | permanently paper |

**Nothing is SAFE_NOW because it raised historical ROI.** Every SAFE_NOW item is a correctness
or measurement fix that cannot change a bet.

---

## Not done — stated plainly

- **§10 closing-line ladder** (T-6h/3h/1h/30m/10m) and **per-bookmaker quote retention**.
  Capture was widened 12→28 fixtures per league earlier this session, but the time ladder and
  per-book persistence are not built. **This is the highest-value remaining item**: without it
  CLV is measured against a "last snapshot" that may be hours before kickoff, and the anchor
  hierarchy cannot reach its cross-book tier.
- **§15 settlement-provider alignment** for prop markets. Not built. Until it is, no prop
  EV/ROI/CLV conclusion should be trusted.
- **§17 Fantasy challenger.** Not built. Current state unchanged: Wowza MAE 2.16 vs FPL
  `ep_next` 1.54, 0 of 2 gameweeks won.
- **Gate not yet wired into `pipeline.py`.** It exists and is tested; the retrain still uses the
  old tolerance. Wiring it log-only is the first October task.

## The one correction worth repeating

I earlier called HGB Over 2.5 "one check from CHAMPION". That was wrong. Its holdout is
historical (2025-10-19 → 2026-09-28), not prospective; its CI upper bound (−0.00038) does not
clear the 0.001 material floor; and it was measured against the *corrected* stack while
production serves the leaky one — against which no fair comparison is possible on that block,
because the live model has already seen those labels.

**It is a promising challenger. Nothing more.**
