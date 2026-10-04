# Champion/challenger build — final report

Date: **2026-10-04**. Repos at `wowza-betting 6cf72bd3`, `wowzaV9-Pro ef8c7b9`, `wowza_v11 6d3d699`.

**One production behaviour changed: a candidate that scores worse than the incumbent on the
incumbent's own holdout no longer replaces it.** No threshold, stake, tip rule, league routing or
notification was touched.

---

## A. What I inspected

All three repo HEADs; `pipeline.py` promotion path and `retrain.py`; `src/model.py` meta stack;
`output/retrain_log.json` (13 days, 91 decisions); `bets_ledger.csv` + `side_bets_ledger.csv`
(989 settled bets since the 2026-08-10 cutoff); `registry/preregistered_hypotheses.json`;
`src/book_quotes.py`, `market_anchor.py`, `shadow_compare.py`, `model_validation.py`;
`std_odds_capture.yml`, `nf_odds_capture.yml`, `predict.yml`, `retrain.yml`; v11's
`v11_residual.csv` / `v11_scoreboard.csv` / `v11_shadow_log.csv`; Pro's `data/season_2026_27/`
stores and `pure_prediction/research_manifest.json`.

## B. What I independently reproduced

Nothing below is quoted from the brief. Every figure was recomputed from current files.

**P0, exactly as cited.**

```
2026-10-04  newformat   0.68110 -> 0.68142  (+0.00032)  WORSE  promoted=True
2026-10-04  ht_over05   0.60543 -> 0.60569  (+0.00026)  WORSE  promoted=True
```

**The replay**, over all 91 recorded decisions:

| | |
|---|---|
| decisions | 91 |
| live gate promoted | 88 |
| candidate was measurably worse | 32 |
| …and promoted anyway | **29** |
| strict gate passes replayable checks | 26 ← upper bound |
| strict would block a live promotion | 62 |

The upper bound is honest: `retrain_log.json` stores aggregates only, so **three** of the strict
gate's ten checks are replayable (`log_loss_not_worse`, `material_improvement`, canary). The other
seven need per-row probabilities and can only ever *reject*, so the true strict count is ≤ 26.

**Betting evidence**, staked tiers (SNIPER+MARKSMAN), since 2026-08-10, CIs block-bootstrapped by
matchday:

| cell | n | P/L | ROI/bet | status |
|---|---|---|---|---|
| ou25 standard | 103 | −12.00u | −0.117 | NEUTRAL |
| ou25 new-format | 237 | −16.40u | −0.069 | NEUTRAL |
| **ou25 new-format, odds > 3.00** | 34 | **−21.25u** | **−0.625** | **NEGATIVE_SIGNAL** |
| **btts new-format 2.00–2.50** | 75 | **+25.98u** | **+0.346** | **POSITIVE_SIGNAL** |
| BTTS Argentina | 68 | +26.46u | +0.389 | POSITIVE_SIGNAL |
| BTTS ex-Argentina | 37 | +5.45u | +0.147 | NEUTRAL |
| **Argentina, all markets** | 166 | — | +0.180 | **VALIDATABLE** |

**Model ranking**, all candidates rescored on exactly the market-priced holdout rows:

| | standard (n=5,018) | new-format (n=2,966) |
|---|---|---|
| **market de-vigged** | **0.67702** | **0.67899** |
| pro_hgb | 0.68752 | 0.68714 |
| v9_meta_leaky | 0.68872 | 0.68526 |
| mean_blend | 0.68975 | 0.68928 |
| meta_corrected | 0.69019 | 0.69024 |

**V11**: Brier 0.2405 → 0.2404, delta −0.00010 on n=1,048. Wowza helps in **10 of 19** segments —
one-sided binomial p = **0.500**, a coin flip. 0 BET selections of 1,199; `v11_ev_lb` never
positive (max −0.0384). Artifacts dated 2026-09-22, **12 days stale**.

---

## C. Confirmed defects

### C1 — a worse candidate replaced a serving champion

* **Root cause** `pipeline.py` promoted on a pure tolerance (`TRAIN_MAX_LOGLOSS_RISE=0.030`),
  which asks only "did it collapse?" and never "is it better?"
* **Impact** 29 of 88 promotions shipped a model measurably worse than the one it replaced, worst
  `ht_over05` at **+0.019**.
* **Period / scope** every track, all 13 logged days.
* **Fix** never promote a candidate whose log loss is higher on `same_holdout`.
* **Why only that** the replay separates two rules the strict gate bundles:

  | rule | promotes | per track / 13 days |
  |---|---|---|
  | live tolerance | 88/91 | 11–13 |
  | **no-harm** | 59/91 | 6–10 |
  | material floor ≥0.001 | 28/91 | 2–6 |

  The material floor does the heavy blocking, and the evidence runs *against* it: 1,568 simulated
  comparisons found no promotion rule beat having no gate, every rejection costing on average, and
  retraining separately beats freezing. `over15` would have updated twice in thirteen days. So the
  floor stays log-only. The no-harm rule needs no such evidence — on a tie or a loss the incumbent
  is a known quantity and the challenger is not, and tomorrow's retrain tries again.
* **Exemption** only on `same_holdout`; under the stored-metrics fallback the numbers come from
  different test sets.
* **Tests** 16, including all 8 historically bad promotions asserted to have promoted under the
  old rule and to be blocked now.
* **Production changed** **YES** (promotion only). Revert: `TRAIN_BLOCK_WORSE_CANDIDATE=0`.

### C2 — the prospective shadow log was never running

* **Root cause** `src/shadow_compare.py` existed and was tested but was **called from nowhere**.
  `output/v91_shadow.csv` had never been written.
* **Impact** no prospective evidence existed at all, so the gate's `second_period` check could
  never pass and every challenger was pinned at CHALLENGER by construction. **This also corrects a
  statement I made earlier today** — I said the shadow log had started; it had not.
* **Fix** `pipeline._log_shadow()` runs inside predict, after `predictions.csv` and before tips.
  157 of 157 fixtures logged; re-run adds 0 rows. `predict.yml` now stages the file.
* **Not recoverable** no amount of history can create a prospective record retrospectively. The
  clock starts now.
* **Production changed** no — a recorder that swallows its own exceptions.

### C3 — `prospective_window()` would have crashed on first use

* **Root cause** `logged_at` parsed tz-aware, compared against a naive `Timestamp`; pandas raises
  `TypeError` rather than coercing.
* **Impact** the forward-period check would have failed months from now with no data lost but the
  entire promotion path blocked. Found by a test, not in production.
* **Fix** both sides normalised to UTC.

### C4 — change-only quote storage cannot prove a close

* **Root cause** mine, from this morning. A book quoting 1.90 at T-3h and still 1.90 at T-10m wrote
  one row at T-3h, so "the price never moved" and "we never looked again" were indistinguishable.
* **Impact** a CLV computed against that row is measured against a three-hour-old price while
  appearing to be a close.
* **Fix** `obs_reason = change | heartbeat`; an unchanged price on T-1h/T-30m/T-10m is kept once
  per rung. Far rungs stay change-only — a heartbeat on every band would push a monthly part past
  the 100 MB limit the partitioning exists to avoid.
* **Tests** the brief's exact four-rung scenario, plus "resampling one rung adds one row, not five".

### C5 — the meta-stack leak is still live

* **Root cause** `src/model.py:240-245` — `meta_clf.fit(meta_X, y_test)` then reported on the same
  `meta_X`. The in-code comment claiming no leakage is itself the error: true of the base models,
  false of the meta.
* **Impact, measured** standard **+0.001468**, new-format **+0.004981**. This **supersedes my own
  earlier estimate of 0.0006–0.0014**, which was taken on a pooled frame. Both exceed the 0.001
  material floor, so the leak is large enough to corrupt gate decisions.
* **It is an evaluation-integrity number and does not explain betting losses.**
* **Production changed** no — `meta_corrected` stays in shadow.

### C6 — two of my own methodology bugs, caught while running the comparison

* The track filter tested `if "model_type" in df.columns` and fell through to the whole frame when
  absent — which it is. Both "tracks" returned byte-identical numbers on 84,550 pooled rows: a
  silent invariant-1 violation that *looked* like two results.
* The market was scored on 5,018 rows while the models were scored on 5,882 — comparing test sets,
  the exact error `model_validation` exists to prevent. `evaluate_candidates` now takes a
  `score_mask`; training is unaffected, only scoring is narrowed.

---

## D. NOT CONFIRMED

**D1 — "standard positive, new-format strongly negative" is not statistically supported.**
The point estimates differ (ou25 at 30d: standard +1.41u, new-format −16.12u) but **both CIs span
zero at every window**. The divergence is real as an observation and not established as a fact.

**D2 — the direction of the pooled headline depends entirely on how you slice it.**
Pooling markets and tiers gives standard **−42.51u NEGATIVE** and new-format **+8.89u NEUTRAL** —
the opposite of the brief's premise. Split by market and restricted to staked tiers it reproduces
the premise exactly. Both are arithmetically correct. Pooling is not a presentation choice.

**D3 — "USA MLS is one of the largest negative contributors" does not survive the tier split.**
At staked tiers over the full window only **1 of 19** leagues has a CI excluding zero (Argentina,
positive). USA MLS, League One and League Two lose significance — they were VALUABLE-tier driven.
This also corrects a figure I gave earlier today.

**D4 — HGB is not unambiguously the best model.** It wins against the honest baseline on both
tracks, but on new-format its **ECE is 0.0527** against the leaky stack's 0.0169 — better log loss,
worse calibration. Overconfidence is this estate's measured defect, so that is not a detail.

**D5 — no model beats the market.** The de-vigged close beats every candidate on both tracks, by
0.0105 (standard) and 0.0063 (new-format) log loss. HGB is a **forecasting** improvement. Nothing
here supports calling it a betting edge.

---

## E. Promotion / readiness matrix

| Component | Status | Evidence missing |
|---|---|---|
| v9 production | **CHAMPION** | — |
| no-harm promotion rule | **LIVE** | — |
| material-improvement floor | **SHADOW** | forward evidence that blocking pays |
| corrected meta stack | SHADOW | prospective replication |
| HGB Over 2.5 standard | CHALLENGER | second independent period; beats v9, loses to market |
| HGB Over 2.5 new-format | CHALLENGER | calibration (ECE 0.053) before anything else |
| goal distribution | CHALLENGER | worse on O1.5/BTTS |
| per-book capture + heartbeat | SAFE_NOW | first CI run must be verified |
| three CLV types | SAFE_NOW | no consumer yet |
| Argentina BTTS | RESEARCH | 4 of 150 preregistered forward bets |
| ex-Argentina BTTS | RESEARCH | no edge demonstrated (CI spans zero) |
| V11 residual | RESEARCH | helps in 10 of 19 segments, p=0.50 |
| new-format odds > 3.00 | NEGATIVE_SIGNAL | a live selection change — needs its own evidence |
| player props | BLOCKED/PAPER | settlement alignment unverified on 6 of 9 markets |

---

## F. Architectural finding — v9 and Pro now overlap

Pro already holds `data/season_2026_27/book_odds_snapshots/` with **the same shape** as the
`book_quotes` store I added to v9 today — `bookmaker, market, side, odds, kickoff_utc,
minutes_to_kickoff, is_post_kickoff` — plus richer provenance (`run_id`, `source_sha`,
`pro_git_sha`, `quality_flags`) and its own independent multi-book source.

Per §10, market and CLV evidence belongs in **Pro**, not v9. v9's copy is still justified today —
v9 needs a closing line for its own tips and Pro reads v9 over HTTP — but **this is a duplication
that should be resolved deliberately rather than left to drift**, and the two will diverge if it
is not. I am flagging it rather than unilaterally picking one.

---

## G. Not done

Stated plainly rather than glossed:

* **§11 canonical data** — not reproduced. The "23,169 stranded fixtures" figure was not
  recomputed; Pro's canonical fixture store was only surveyed, not audited.
* **§12 goal distribution / joint Dixon-Coles** — existing challenger unchanged; no bivariate
  model fitted.
* **§13 Fantasy** — the blend sweep stands (optimal weight on Wowza = 0.0); captain-rank and
  positional metrics not added.
* **§14 Bet Builder naming** — synthetic-ROI relabelling in Pro not done.
* **§16 dashboard** — deliberately untouched, per the brief's own instruction not to spend the
  session on visuals before the evidence architecture is right.
* **V11 artifacts are 12 days stale** — reproduced as-is, not regenerated.
* **Stage B of the strict gate** — not flipped, and on current evidence should not be.

---

## H. Verification that local tests cannot give

Three signals, all from real CI runs:

1. `output/v91_shadow.csv` grows in a predict commit, and row count exceeds 157.
2. `output/book_quotes/2026-10.csv` appears in commits from **both** capture workflows, with
   `obs_reason` carrying both `change` and `heartbeat`.
3. Tomorrow's retrain entry carries a `v91_gate` block, and the no-harm rule appears as a
   `NOT PROMOTED` line the first time a candidate is worse.
