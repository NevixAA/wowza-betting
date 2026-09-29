# The HT model upgrade — what was built, and what it did and didn't buy

Date: 2026-09-29. Written for: Nevo. Follows `HT_MODEL_INVESTIGATION.md`.

You asked for the half-time model to be as deep as the other models. It now is. The honest
result: **it is a genuinely better model, and it still does not beat the bookmaker.**

---

## What was wrong

The HT models trained on the **standard feature set** — a set designed for full-time Over/Under
2.5. It carries corners, fouls, shot ratios and the full-time O/U implied probabilities, plus
exactly four genuinely half-time columns:

```
home_ht_over05_rate   away_ht_over05_rate
home_ht_over15_rate   away_ht_over15_rate
```

Each is a **5-match rolling binary rate**, so it can take only eleven distinct values. A
5-match rolling average of a near-coin-flip is mostly noise. Measured alone it reaches **AUC
0.522** — barely above the 0.500 of a constant.

So the half-time models were essentially full-time models wearing a half-time target.

---

## What was built

`src/ht_features.py` — 23 dedicated features, the core of which is a multiplicative Poisson rate
model on **first-half goals only**:

```
lambda_home = league_HT_home_mean  x  home_attack  x  away_defence
lambda_away = league_HT_away_mean  x  away_attack  x  home_defence
P(at least one HT goal) = 1 - exp(-(lambda_home + lambda_away))
```

Half-time goals are a low-count process (~1.0 per match by the break), so a rate model is the
right shape and a rolling average is not. Three things make it work:

- **Empirical-Bayes shrinkage** toward the league mean (`PRIOR_MATCHES = 30`). With ~0.5 HT
  goals per team per match, a team needs most of a season before its own rate says anything.
  Shrinkage is what stops a 3-match hot streak becoming a 0.9 attack multiplier.
- **Recency weighting** (40-match half-life) — old form still counts, just less.
- **Strictly as-of accumulation.** The builder walks the frame in date order and updates each
  accumulator *after* emitting the row, so a fixture can never contribute to its own features.
  Verified: a team's first half-time match shows `ht_home_n = 0.000000`.

Plus league baselines, home/away splits, H2H half-time history, and each team's half-time share
of its own full-time goals.

---

## What it bought

Walk-forward, 14,041 out-of-sample fixtures, retraining each window:

| feature set | HT O/U 0.5 AUC | Brier skill | HT O/U 1.5 AUC | Brier skill |
|---|---|---|---|---|
| standard (what it used) | 0.5400 | +0.26% | 0.5373 | +0.22% |
| **HT-specific (new)** | **0.5498** | **+0.53%** | **0.5414** | **+0.33%** |
| standard + HT | 0.5451 | +0.35% | 0.5410 | +0.20% |

**The dedicated features roughly double the skill on both lines**, and calibration is now
honest: claimed 0.6975 against a realised 0.7000 — a gap of 0.25 percentage points, against the
**+22.9pp** the old setup carried in the band it tipped from.

For scale, the best *trivial* baseline — predicting each league's mean and nothing else — gets
AUC 0.5295. So the model is now meaningfully above the league baseline, where before it was
roughly level with it.

---

## What it did not buy, which is the part that matters

On the 332 fixtures where we hold a captured closing half-time price:

| | Brier | skill | AUC |
|---|---|---|---|
| **market (de-vigged)** | **0.1972** | **+0.22%** | **0.5680** |
| model, old features | 0.1978 | −0.11% | 0.5293 |
| model, new features | 0.1994 | −0.93% | 0.5327 |

**The bookmaker's price beats the upgraded model, and by more than it beat the old one.** The
market ranks half-time fixtures better than we do (AUC 0.568 vs 0.533 on the 0.5 line; 0.613 vs
0.552 on 1.5).

Blending the model into the price **adds nothing** — −0.00018 at a 25% weight, −0.00061 at 50%.
Edge selection loses at every threshold with a usable sample:

```
HT O/U 0.5   edge>=2%  221 bets   -8.15%
             edge>=3%  185 bets   -9.06%
             edge>=5%  119 bets  -12.51%
             edge>=8%   16 bets   +1.50%   <- 16 bets, noise
```

The vig is **6.94%** on the 0.5 line and **7.19%** on 1.5, so break-even needs +4.77pp / +2.54pp
over the de-vigged price. The model disagrees with the market by 4.42pp (sd) — it clears the bar
often enough by magnitude, but in the wrong direction more often than the right one.

**Verdict: a better model, still no bet.** Half-time over/under stays unbet. That is the same
shape as the player-props result — an accurate model against an efficient market.

---

## What changed in the code

- **`src/ht_features.py`** (new) — the feature module, plus `attach_ht_features()` for the
  predict path.
- **`pipeline.py`** — HT training now passes the dedicated feature set (65 columns, 23
  half-time specific). Falls back to the standard set if the build fails, so a retrain never
  dies on this.
- **`src/predict.py`** — builds the same features for upcoming fixtures.
- **`scripts/ht_backtest.py`** — now measures the shipped feature set, not the old one.

### The trap this had to avoid

Training and prediction use **different builders** (`build_features` vs
`build_upcoming_features`). A feature present in one and missing in the other is the most
dangerous shape in this codebase: `model._prep` imputes an all-NaN column to 0.0, the scaler
maps that to roughly z = −37, and the logistic collapses. `pipeline.py`'s `_NF_DROP` block
records exactly that happening — it crushed new-format P(over) from ~0.51 to ~0.36.

So the predict path was verified end-to-end rather than assumed:

```
upcoming fixtures built: 60
attach_ht_features returned 60 rows (expected 60)
feature coverage: min 100%  mean 100%
SCORED p_ht_over05: mean 0.6990  range 0.6454-0.7302  sd 0.0184
VERDICT: HEALTHY — mean near the ~0.70 base rate with real spread
```

`predict.py` also logs the coverage every run and warns if it drops below 50%, so a future
break announces itself instead of silently degrading. The build costs ~3 seconds, which is
immaterial against the 15-minute predict cadence.

---

## Consequences you should know about

**HT tips stay off.** The upgraded model's output spans roughly 0.645–0.730 — it never reaches
the 0.75 threshold the notifier tips from. That is correct: a model with +0.5% skill should not
be claiming 75% confidence. No threshold was changed; the tips stop because the model is now
honest.

**The models themselves were not committed.** `England_Leagues_4_Seasons_With_Summary.xlsx` is
local-only, so a local retrain sees more history than CI and produces different artifacts. The
code is committed; the nightly 03:00 retrain builds the models in CI where they belong. Local
training confirmed the promotion gate will accept them — log loss **0.6037** against the
incumbent's 0.61047.

**One evening of exposure.** Until that retrain runs, the live HT models are still the ones
trained on the corrupt labels fixed yesterday, and those *are* confident enough to tip. HT is
paper-only, so the cost is a few misleading paper tips, but it is worth knowing. If you want
them silenced immediately that is a notifier change and yours to make.

---

## How to re-run any of this

```bash
python scripts/ht_backtest.py --line 05      # walk-forward, both parts
python scripts/ht_recalibrate.py --line 05   # calibration check
python scripts/ht_robustness.py              # attacks any edge claim
```

All three are read-only and run with `APIFOOTBALL_KEY` unset, so they cannot spend quota or
re-bank `af_history.parquet`.
