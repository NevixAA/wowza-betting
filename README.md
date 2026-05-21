# Wowza v9 — Football Over/Under 2.5 Prediction System

**Locked:** 2026-05-04  
**Status:** Production-ready. Live tracking since April 2026.

---

## What It Does

Predicts whether football matches will have Over or Under 2.5 total goals, using an ensemble ML model trained on 6+ seasons of historical data. Generates daily betting tips with edge scores, stake sizing, and drift signals.

---

## Performance (Backtest — walk-forward, leak-free)

| Metric | Value |
|---|---|
| ROI | **8.55%** |
| Sharpe Ratio | **1.629** |
| Max Drawdown | **-178 units** |
| Total Bets | **4,691** |
| Win Rate | ~55% |

### By League

| League | Bets | ROI % |
|---|---|---|
| League One (ENG) | 685 | **17.24%** |
| Bundesliga 2 (GER) | 478 | **13.52%** |
| La Liga 2 (ESP) | 971 | **10.03%** |
| Ligue 2 (FRA) | 760 | **9.19%** |
| League Two (ENG) | 896 | **7.48%** |

> Austria, Romania, Sweden: enabled but insufficient history for full walk-forward backtest.  
> Denmark 1st Div: borderline (-0.37%) — candidate for removal next season.

---

## Daily Workflow

```
# Morning — generate tips for upcoming fixtures
python pipeline.py --mode predict

# After matches — fill in results automatically
python update_results.py

# Check live P&L
cat output/bets_ledger.csv
```

---

## End-of-Season Workflow (every July)

```
# Download latest data + retrain + backtest + compare metrics
python retrain.py

# Optional: check only what changed in backtest
python retrain.py --no-download

# Then generate new season's first predictions
python pipeline.py --mode predict
```

---

## Architecture

```
pipeline.py              ← entry point (train / predict / backtest / all)
retrain.py               ← end-of-season data refresh + retrain
update_results.py        ← fill WIN/LOSS/PnL in bets_ledger.csv
config.py                ← all thresholds, paths, API keys

src/
  data_loader.py         ← loads XLSX + CSV overrides → unified DataFrame
  feature_engineering.py ← rolling form, shot features, attack/defense strength
  model.py               ← LogisticRegression + GradientBoosting + isotonic calibration
  predict.py             ← fetches upcoming fixtures from OddsAPI, applies model
  backtest.py            ← strict walk-forward backtest (no leakage)
  betting.py             ← SNIPER / VALUE / AVOID tier logic
  ledger.py              ← persistent bets_ledger.csv append + dedup

output/
  bets_ledger.csv        ← every tip ever generated + results
  predictions.csv        ← latest full prediction run
  bets.csv               ← latest tips only (VALUE + SNIPER)
  backtest_results.csv   ← row-level backtest output
  backtest_by_league.csv ← league-level summary
  backtest_metrics_history.json ← season-over-season comparison

models/
  model_v9.pkl           ← trained ensemble (LR + GBM + calibration)
  feature_importances_v9.csv
```

---

## Enabled Leagues (10)

| League | Country | Data Source |
|---|---|---|
| League One | England | football-data.co.uk (standard) |
| League Two | England | football-data.co.uk (standard) |
| Bundesliga 2 | Germany | football-data.co.uk (standard) |
| Ligue 2 | France | football-data.co.uk (standard) |
| La Liga 2 | Spain | football-data.co.uk (standard) |
| Denmark Superliga | Denmark | football-data.co.uk (new format) |
| Danish 1st Div | Denmark | Excel workbook (contamination warning) |
| Austrian Bundesliga | Austria | football-data.co.uk (new format) |
| Swedish Allsvenskan | Sweden | football-data.co.uk (new format) |
| Romanian Superliga | Romania | football-data.co.uk (new format) |

---

## Model

**Type:** Ensemble — LogisticRegression + GradientBoosting, isotonic calibration  
**Target:** `over25` (1 if total goals > 2.5, else 0)  
**Rolling window:** N=5 matches (uniform mean, shift(1) to prevent leakage)

### Features

| Feature | Description |
|---|---|
| `home/away_form_last5` | Rolling win rate (last 5 games) |
| `home/away_scored_last5` | Rolling goals scored |
| `home/away_conceded_last5` | Rolling goals conceded |
| `home/away_shots_last5` | Rolling shots |
| `home/away_sot_last5` | Rolling shots on target |
| `home/away_sot_ratio_last5` | Rolling SOT / shots |
| `combined_sot_ratio` | Average of home + away SOT ratios |
| `home/away_attack_str` | Attack strength index |
| `home/away_defense_str` | Defense strength index |
| `rest_days_home/away` | Days since last match |
| `implied_prob_over/under` | From market odds (1/odds) |
| `sp_goals_home/away` | Sofascore set-piece goals |
| `ref_foul_avg` | Referee avg fouls per game |

---

## Betting Tiers

| Tier | Edge Threshold | Action |
|---|---|---|
| SNIPER | ≥ 10% | Bet full stake |
| VALUE | 4–10% | Bet half stake / monitor |
| AVOID | < 4% | Skip |

**Stake sizing:** Flat 1 unit default. Kelly staking (25% fraction) available via `USE_KELLY=1`.  
**Min odds filter:** 1.75 for both OVER and UNDER.

---

## Drift Signals

Every predict run snapshots the current odds and compares to the first recorded odds for that fixture:

| Signal | Meaning |
|---|---|
| Confirmed | Odds moved in our direction (≥ 0.03 improvement) |
| Conflicted | Odds moved against us (≥ 0.03 adverse) |
| Neutral | Small or no movement |
| New | First time we've seen this fixture |

Drift does not change the SNIPER/VALUE tier — it's advisory context.

---

## Results Tracking

`update_results.py` fetches completed scores automatically:

- **OddsAPI** → major leagues (League One, League Two, Bundesliga 2, La Liga 2, Ligue 2, etc.)
- **football-data.co.uk** → Austria (AUT), Sweden (SWE), Denmark (DNK)

**CLV (Closing Line Value)** is calculated for each settled bet:  
`CLV = (entry_odds - closing_odds) / closing_odds × 100`  
Positive CLV = we got better odds than the closing line = sharp model signal.

---

## Data Sources

| Source | Used For |
|---|---|
| football-data.co.uk | Historical match data (goals, shots, odds) |
| OddsAPI (the-odds-api.com) | Live upcoming odds + completed scores |
| Sofascore (cached) | Set-piece goals per team |
| API-Football (RapidAPI) | Referee stats |

---

## Key Design Decisions

**No data leakage:** The original `combined_sot_ratio` feature used same-match shot data (future information at prediction time). Fixed to use rolling team averages from prior games only (`shift(1)`). This dropped inflated ROI from 27%+ to the true 8.55%.

**League selection:** Leagues are only enabled after walk-forward backtest confirms positive ROI. Belgian, Scottish, Turkish, Dutch, Polish leagues all showed negative ROI after the leakage fix and were removed.

**No time-decay / no Poisson:** Both were tested and reverted — they hurt ROI (8.55% → 4.39%) and worsened Sharpe and drawdown. Uniform rolling mean is better for this dataset size and signal type.

---

## APIs & Keys

Set via environment variables (or hardcoded in config.py for dev):

```
ODDS_API_KEY    → the-odds-api.com key
API_KEY         → RapidAPI key (API-Football, for referee stats)
```

---

## Live Track Record

Started: April 2026  
As of 2026-05-04: **17 settled bets — 5W / 12L — PnL: -5.29u**  
*(Small sample, rough first week. Need ~200 bets for statistical significance.)*
