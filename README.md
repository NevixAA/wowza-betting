# Wowza v9.2 — Football Betting Intelligence System

**Version:** v9.2  
**Status:** Production — fully automated via GitHub Actions  
**Live since:** April 2026

---

## What It Does

A complete football betting intelligence platform with multiple prediction layers:

| Module | What it does |
|---|---|
| **O/U 2.5 Model** | Predicts Over/Under 2.5 goals — SNIPER/MARKSMAN/VALUABLE tiers |
| **Side Markets** | BTTS, O/U 1.5, O/U 3.5 predictions with per-league walk-forward backtest |
| **HT Model** | Half-time O/U 0.5 and 1.5 predictions |
| **Sharp Money Tracker** | Odds drift detection — STEAM, STRONG, SHARP signals across 20 leagues |
| **World Cup Tracker** | WC 2026 drift + ML model value on O/U 1.5/2.5/3.5 and 1X2 |
| **Live Scanner** | In-play Poisson signals during match hours |
| **Player Props** | 9-market ML ensemble — SOT×4, Goals×3, Cards, Assists |

All runs automatically in the cloud — no server or PC required.

---

## Architecture

```
pipeline.py                ← team model: train / predict / backtest
update_results.py          ← fill WIN/LOSS/PnL via football-data.co.uk
config.py                  ← all thresholds, paths, API keys

src/
  data_loader.py           ← Excel + CSV + CI download from FD
  feature_engineering.py  ← rolling form, HT features, home advantage, league isolation
  model.py                 ← LogisticRegression + GradientBoosting + Platt calibration
  predict.py               ← OddsAPI fixtures → model → SNIPER/MARKSMAN/VALUABLE
  backtest.py              ← walk-forward backtest (no leakage)
  betting.py               ← 3-tier signal logic (per-league thresholds)
  sharp_tracker.py         ← volume-weighted drift, steam detection, consensus scoring
  live_scanner.yml         ← Poisson in-play signals + HT sub-formula

side_markets/
  config.py                ← BTTS / O1.5 / O3.5 thresholds + league list
  predict.py               ← side market predictions → side_bets.csv

worldcup/
  tracker.py               ← WC drift + ML model value on WC fixtures

player_model/
  config.py                ← prop league/tier thresholds, PLAYER_FEATURE_COLS (132 features)
  api_football.py          ← API-Football client (match stats, lineups, referees, injuries)
  feature_engineering.py  ← 132-feature rolling player features (season stats, profile,
                             injury, referee, opponent matchup composites)
  model.py                 ← per-market Platt-calibrated ensemble; graceful feature fill
  pipeline.py              ← collect → enrich-season → enrich-profiles → enrich-sidelined
                             → enrich-sidelined-live → train → predict
  predict.py               ← de-vig EV, relative edge, lazy market factors

agent/
  player-props-architecture.md  ← full player props ML architecture (v2)
  formulas.md              ← formula library (Poisson, Dixon-Coles, Kelly, EV)

output/
  bets_ledger.csv          ← every tip + results
  predictions.csv          ← latest prediction run (team model)
  bets.csv                 ← latest SNIPER/MARKSMAN/VALUABLE tips (O/U 2.5)
  side_bets.csv            ← BTTS / O1.5 / O3.5 tips
  sharp_tips.csv           ← sharp money signals
  worldcup_tips.csv        ← WC drift signals
  worldcup_model_tips.csv  ← WC ML fair price vs market
  player_tips.csv          ← player prop signals
  live_tips.csv            ← in-play signals

pages/                     ← Streamlit dashboard pages
  1_📊_Dashboard.py
  2_📈_Performance.py
  3_📋_Ledger.py
  4_ℹ️_Model_Info.py
  5_🌍_World_Cup.py
  6_⚡_Live.py
  7_💰_Sharp_Money.py
  8_⏱_HalfTime.py
  9_📜_Live_History.py
  10_🤖_Agent_Analysis.py
  11_👤_Player_Props.py

.github/workflows/
  predict.yml              ← every 5min 08-23 UTC (offset :01): train + predict + Telegram
  player_props.yml         ← every 5min 08-23 UTC (offset :02): injury refresh + props
  live_scanner.yml         ← every 5min 08-23 UTC (offset :03): live signals
  update_results.yml       ← every 2h: fill results from football-data.co.uk
  sharp_tracker.yml        ← every 2h: drift signals across 20 leagues
  worldcup.yml             ← every 1h 08-23 UTC: WC tracker (match hours only)
  retrain.yml              ← every Sunday: full model retrain
  backtest.yml             ← manual trigger: walk-forward validation
  weekly_summary.yml       ← every Monday 09:00: Telegram performance summary
  daily_summary.yml        ← 07:00 UTC daily: digest + notified.json commit
```

---

## Backtest Performance (Standard Model — post-COVID, walk-forward)

> COVID seasons (2019/20, 2020/21) excluded from training

### O/U 2.5

| Tier | Bets | Win Rate | ROI |
|---|---|---|---|
| 🎯 SNIPER | 1,929 | 43.6% | **+59.0%** |
| 🔫 MARKSMAN | ~2,400 | 41.2% | **+18.4%** |
| ~~💎 VALUABLE~~ | — | — | -8.2% (disabled) |

### Side Markets (BTTS / O1.5 / O3.5)

| Market | Bets | ROI |
|---|---|---|
| BTTS Yes | ~800 | **+22–31%** |
| Over 1.5 | ~600 | **+18–27%** |
| Over 3.5 | ~400 | **+14–21%** |

*Side market backtests are per-league optimised — ROI range across enabled leagues.*

---

## Signal Tiers

| Tier | Edge | Stake | Note |
|---|---|---|---|
| 🎯 **SNIPER** | Per-league threshold (14–25%) | Full stake | Highest confidence |
| 🔫 **MARKSMAN** | 8% to threshold | 3/4 stake | Medium-high confidence |
| 💎 **VALUABLE** | 4–8% | Half stake | Side markets only (O/U 2.5 VALUABLE disabled: -8.2% ROI) |

---

## Automation (GitHub Actions)

Everything runs in the cloud — PC can be completely off.

| Workflow | Schedule | Does |
|---|---|---|
| **Predict** | Every 5min 08-23 UTC (at :01) | O/U 2.5 + side markets + Telegram |
| **Player Props** | Every 5min 08-23 UTC (at :02) | Injury refresh → 9-model props + Telegram |
| **Live Scanner** | Every 5min 08-23 UTC (at :03) | In-play Poisson signals |
| **Update Results** | Every 2h | Fill WIN/LOSS from football-data.co.uk |
| **Sharp Tracker** | Every 2h | Drift signals across 20 leagues |
| **World Cup** | Every 1h 08-23 UTC | WC drift + ML value (match hours only) |
| **Retrain** | Sunday 04:00 | Full model retrain with latest data |
| **Daily Summary** | 07:00 UTC | Telegram digest + commit notified.json |
| **Weekly Summary** | Monday 09:00 | Telegram performance summary |

*Workflows are staggered (:01/:02/:03) to avoid simultaneous GitHub Actions runner contention.*

---

## Model Details

### Team O/U 2.5 Model
- Ensemble: LogisticRegression + GradientBoosting + Platt calibration
- Walk-forward validation, no random splits, COVID seasons excluded
- Time-decay: recent seasons weighted 2–4× higher
- Per-league SNIPER thresholds (14–25%) via walk-forward optimisation

### Player Props Model (v9.2)
- **9 markets:** Goals, Goals×2, Goals×3, SOT, SOT×2, SOT×3, SOT×4, Yellow Cards, Assists
- **AUC range:** 0.714 – 0.844 (post-retrain June 2026)
- **132 feature columns** per player per match:
  - Rolling form (10-game window): goals, shots, SOT, cards, key passes, ratings
  - Season stats: full-season aggregates from API-Football enrichment
  - Profile: age, height, peak-age delta, height × aerial interaction
  - Injury: chronic risk, days since last injury, return-from-injury flag
  - Referee: yellows per game, strictness z-score
  - Opponent matchup: 8 composite scores (carrier vs press, box threat vs leaky defense, etc.)
  - Venue splits, position flags, career priors
- **Enrichment pipeline:**
  - `enrich-season` — full season API stats for all 4,054 players
  - `enrich-profiles` — age/height from player profile API
  - `enrich-sidelined` — bulk injury history
  - `enrich-sidelined-live` — fast pre-predict refresh for players active in last 60 days only

**Key fixes (v9.2 — June 2026):**
- `build_upcoming_features()` now emits all 132 training features at predict time (was causing CI crash with KeyError on 28 missing columns)
- `model._prep()` fills any still-missing columns with 0 for graceful degradation
- Merge-on-write pattern for parallel enrichment (no column clobbering)

---

## Enabled Leagues

### Standard Model (full stats + odds history)
| League | Country | SNIPER Threshold |
|---|---|---|
| League One | England | 25% |
| League Two | England | 14% |
| Bundesliga 2 | Germany | 20% |
| La Liga 2 | Spain | 20% |
| Ligue 2 | France | 25% |
| Championship | England | 15% |
| Serie B | Italy | 15% |

### New-Format Model (goals only)
Ireland, Finland, Denmark, Austria, Sweden, Norway, Brazil, Japan, Mexico, China, USA MLS, Argentina

### Player Props Leagues
Championship, League One, Bundesliga 2, Ireland Premier, Finland Veikkausliiga, WC 2026 (until Jul 19)

---

## Data Sources

| Source | Used For | Cost |
|---|---|---|
| football-data.co.uk | Historical match data | Free |
| OddsAPI (the-odds-api.com) | Live odds + WC odds | ~$30/month (20K credits) |
| API-Football (api-sports.io) | Player stats, lineups, injuries, referees | 5K/day plan |
| Sofascore (cached) | Set-piece goals per team | Free (scraping) |
| FBref (scraping) | Player season stats for training | Free (scraping) |

---

## Setup

### GitHub Secrets required
```
ODDS_API_KEY      → the-odds-api.com
TELEGRAM_TOKEN    → @TheWowzaBot token
TELEGRAM_CHAT_ID  → group chat ID
APIFOOTBALL_KEY   → api-sports.io key (direct endpoint, not RapidAPI)
```

### Local run
```bash
python pipeline.py --mode predict                          # O/U 2.5 tips
python -m player_model.pipeline --mode predict             # player props
python -m player_model.pipeline --mode enrich-sidelined-live  # injury refresh
```

---

## Dashboard

Public: `https://nevixaa-wowza-betting-app-kqbjnm.streamlit.app/`  
Local: `http://localhost:8501`
