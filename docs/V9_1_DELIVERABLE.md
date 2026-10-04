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
| `src/book_quotes.py` | §10 per-book quotes, kickoff ladder, honest close |
| `scripts/fantasy_challenger.py` | §17 blend sweep against FPL `ep_next` |
| `registry/settlement_alignment.json` | §15 our label vs the book's settlement, per market |
| `tests/` ×6 | 63 tests |

**Pro (`wowzaV9-Pro`)** — read only; its research was reproduced, not modified.
**V11 (`wowza_v11`)** — untouched; conclusion is "keep collecting", which needs no change.

---

## F. Tests — 63 passing

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

## §10, §15, §17 — closed since the first draft

**§10 — per-bookmaker quotes on a kickoff ladder.** `src/book_quotes.py`. The side-market
capture asked API-Football for **one** bookmaker and stored one price, discarding every other
book at the request. Dropping that filter costs **the same number of API calls** — the response
simply carries all of them. A live probe returned **9 books including Pinnacle and Betfair**.
Every quote is tagged T-6h / T-3h / T-1h / T-30m / T-10m / FAR / POST, and `closing_quote()`
returns the last quote inside T-30m **or nothing** — it never substitutes an earlier price, so
coverage can be measured instead of assumed. POST is its own band and is never eligible as a
close: an in-play price treated as a closing line manufactures enormous fake CLV.

Two bugs found while building it, both the kind that produce a confident wrong number:

* **The first classifier matched any bet whose NAME contained "over/under"** and swallowed seven
  unrelated bets — 26 second-half, 57/58 corners, 197/198 time windows. On a live fixture that
  produced "O/U 2.5" quotes from 1.25 to 3.50 and a cross-book anchor of **p=0.254 where the real
  market was 0.63**. Now keyed on numeric bet ids (`{5, 6, 8, 1}`), which a rename cannot break.
  After the fix: over 1.50–1.57, p=0.6185, Betfair 0.6311.
* **The dedup key contained `line`, which is NaN for BTTS and 1X2 — and NaN never equals
  itself**, so those rows could never match their own earlier row and were rewritten every run.
  Caught by the test asserting a re-append writes nothing: it let exactly the 2 line-less rows
  through out of 8.

OFF by default (`CAPTURE_BOOK_QUOTES`); production capture is byte-identical until enabled.

**§15 — settlement-provider alignment.** `registry/settlement_alignment.json`. For each market,
is our outcome label defined the way the bookmaker *settles*? **3 ALIGNED** (ou, btts, ht_ou —
90-minute goals is the least ambiguous definition in football) and **6 UNVERIFIED, therefore
BLOCKED** (goals, assists, sot, sot2, sot3, cards). `sot` is the one that matters: whether a
blocked shot counts as on target is the largest source of provider disagreement in player props,
and sot is our highest-volume prop market — **1,694 of 2,570 CLV records**. Every BLOCKED market
is already PAPER under invariant 2, so this blocks nothing that was going to be bet; its effect
is that **prop EV, ROI and CLV figures must not be quoted as measured against the paying event.**

**§17 — Fantasy challenger. REJECTED.** `scripts/fantasy_challenger.py`, 575 settled
player-gameweeks across GW4–5:

| projection | MAE | RMSE | rank rho | bias |
|---|---|---|---|---|
| Wowza × p_start | 2.164 | 2.966 | 0.543 | +0.993 |
| **FPL `ep_next`** | **1.535** | **2.588** | **0.705** | −0.118 |
| Wowza conditional (ref) | 3.430 | 3.867 | 0.195 | +2.439 |

The brief allowed a third answer — a blend better than either source alone. **The optimal blend
weight on Wowza is 0.0.** Any Wowza makes it worse. Do not replace `ep_next` and do not blend.

The third row is the diagnostic. Start-weighting lifts rank correlation **0.195 → 0.543**, so
almost everything Wowza contributes here is *who plays*, not *how many points he scores when he
does* — and "who plays" is exactly what `ep_next` already prices. Two gameweeks is a reading,
not a conclusion, but it points at the points head, not the availability head.

---

## Still not done — stated plainly

- **Ladder scheduling.** `book_quotes.py` tags whatever rung a quote lands on; nothing yet
  *aims* a capture at T-30m or T-10m. Per the estate's own measurement, cron cannot hit a clock
  — so this must be an in-run adaptive loop sampling against real kickoff time, not a new cron.
- **Gate not yet wired into `pipeline.py`.** It exists and is tested; the retrain still uses the
  old tolerance. Wiring it log-only is the first October task.
- **No prospective period exists yet.** `v91_shadow.csv` starts accumulating now. Until a real
  forward window exists, every challenger stays CHALLENGER by construction — which is the
  harness working, not a shortage of candidates.

## The one correction worth repeating

I earlier called HGB Over 2.5 "one check from CHAMPION". That was wrong. Its holdout is
historical (2025-10-19 → 2026-09-28), not prospective; its CI upper bound (−0.00038) does not
clear the 0.001 material floor; and it was measured against the *corrected* stack while
production serves the leaky one — against which no fair comparison is possible on that block,
because the live model has already seen those labels.

**It is a promising challenger. Nothing more.**
