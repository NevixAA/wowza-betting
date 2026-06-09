# Wowza v9.1 — Football Betting Intelligence System

**Version:** v9.1  
**Status:** Production — fully automated via GitHub Actions  
**Live since:** April 2026

---

## What It Does

A complete football betting intelligence platform with multiple prediction layers:

| Module | What it does |
|---|---|
| **O/U 2.5 Model** | Predicts Over/Under 2.5 goals using ML ensemble — SNIPER/MARKSMAN/VALUABLE tiers |
| **HT Model** | Half-time O/U 0.5 and 1.5 predictions |
| **Sharp Money Tracker** | Odds drift detection — STEAM, STRONG, SHARP signals across 20 leagues |
| **World Cup Tracker** | WC 2026 drift + ML model value on O/U 1.5/2.5/3.5 and 1X2 |
| **Live Scanner** | In-play Poisson signals during match hours |
| **Player Props** | SNIPER/MARKSMAN/VALUABLE props — SOT, Goals, Cards, Assists *(Phase 1)* |

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
  live_scanner.py          ← Poisson in-play signals + HT sub-formula

worldcup/
  tracker.py               ← WC drift + ML model value on WC fixtures

player_model/              ← Player Props module (Phase 1)
  config.py                ← prop league/tier thresholds
  api_football.py          ← API-Football client (match stats, lineups, referees)
  feature_engineering.py  ← rolling player features, referee factor, set-piece, GES
  model.py                 ← per-market Platt-calibrated ensemble
  pipeline.py              ← collect → train → predict
  predict.py               ← de-vig EV, relative edge, lazy market factors

agent/
  player-props-architecture.md  ← full player props ML architecture (v2)
  formulas.md              ← formula library (Poisson, Dixon-Coles, Kelly, EV)

output/
  bets_ledger.csv          ← every tip + results
  predictions.csv          ← latest prediction run (team model)
  bets.csv                 ← latest SNIPER/MARKSMAN/VALUABLE tips
  sharp_tips.csv           ← sharp money signals
  worldcup_tips.csv        ← WC drift signals
  worldcup_model_tips.csv  ← WC ML fair price vs market
  player_tips.csv          ← player prop signals
  live_tips.csv            ← in-play signals
  backtest_results_standard.csv
  backtest_by_league_standard.csv

pages/                     ← Streamlit dashboard pages
  1_📊_Dashboard.py
  2_📈_Performance.py
  3_📋_Ledger.py
  4_ℹ️_Model_Info.py       ← all tools documented in 8 tabs
  5_🌍_World_Cup.py
  6_⚡_Live.py
  7_💰_Sharp_Money.py
  8_⏱_HalfTime.py
  9_📜_Live_History.py
  10_🤖_Agent_Analysis.py  ← formula-based match validator
  11_👤_Player_Props.py

.github/workflows/         ← GitHub Actions (fully automated)
  predict.yml              ← 5×/day: train + predict + Telegram
  update_results.yml       ← every 2h: fill results from FD
  sharp_tracker.yml        ← every 2h: sharp money signals
  worldcup.yml             ← every 1h: WC tracker
  live_scanner.yml         ← every 30min (11-23 UTC): live signals
  retrain.yml              ← every Sunday: full model retrain
  backtest.yml             ← manual: walk-forward backtest
  player_props.yml         ← 30min after predict: player prop signals
  weekly_summary.yml       ← every Monday 09:00: Telegram summary
```

---

## Backtest Performance (Standard Model — post-COVID, walk-forward)

> COVID seasons (2019/20, 2020/21) excluded from training — empty stadiums created anomalous patterns

| League | Bets | ROI % |
|---|---|---|
| League One (ENG) | 654 | **+18.3%** |
| League Two (ENG) | 902 | **+14.9%** |
| Ligue 2 (FRA) | 656 | **+13.7%** |
| La Liga 2 (ESP) | 987 | **+13.3%** |
| Bundesliga 2 (GER) | 594 | **+12.8%** |

---

## Signal Tiers (3-tier system)

| Tier | Edge | Stake | Note |
|---|---|---|---|
| 🎯 **SNIPER** | Per-league threshold (14–25%) | Full stake | Highest confidence |
| 🔫 **MARKSMAN** | 8% to threshold | 3/4 stake | Medium-high confidence |
| 💎 **VALUABLE** | 4–8% | Half stake | Monitor / small bet |

### Per-League SNIPER Thresholds

| League | Threshold | Rationale |
|---|---|---|
| League Two | 14% | Most data, reliable at lower bar |
| Bundesliga 2 | 20% | Needs higher bar |
| La Liga 2 | 20% | High threshold, big ROI |
| League One | 25% | Very high bar needed |
| Ligue 2 | 25% | High bar, still good ROI |

---

## Automation (GitHub Actions)

Everything runs in the cloud — PC can be completely off.

| Workflow | Schedule | Does |
|---|---|---|
| **Predict** | 06/10/14/18/22 UTC | Train + predict + Telegram alerts |
| **Update Results** | Every 2h | Fill WIN/LOSS from football-data.co.uk |
| **Sharp Tracker** | Every 2h | Drift signals across 20 leagues |
| **World Cup** | Every 1h | WC drift + ML value |
| **Live Scanner** | Every 30min (11-23 UTC) | In-play Poisson signals |
| **Retrain** | Sunday 04:00 | Full model retrain with latest data |
| **Backtest** | Manual trigger | Walk-forward validation |
| **Player Props** | 30min after predict | Player prop signals |
| **Weekly Summary** | Monday 09:00 | Telegram performance summary |

---

## Model Details

**Team O/U 2.5 Model:**
- Ensemble: LogisticRegression + GradientBoosting
- Calibration: Platt scaling (sigmoid) — replaces IsotonicRegression for better small-sample calibration
- Validation: Strict walk-forward, no random splits
- COVID excluded: 2019/20 + 2020/21 seasons removed from training
- Time-decay: recent seasons weighted 2–4× higher

**Key fixes applied (Senior ML audit 2026-06-08):**
- Removed `implied_prob` circular dependency feature
- Fixed corners/fouls to use rolling historical averages (not match-day actuals)
- Fixed `_ht_rate()` league isolation bug
- Added per-team rolling home advantage (replaced constant 1.0)
- WC ML model now clearly labeled as informational only (trained on domestic leagues)

---

## Enabled Leagues

### Standard Model (full stats + odds history → backtestable)
| League | Country | SNIPER Threshold |
|---|---|---|
| League One | England | 25% |
| League Two | England | 14% |
| Bundesliga 2 | Germany | 20% |
| La Liga 2 | Spain | 20% |
| Ligue 2 | France | 25% |
| Championship | England | 15% |
| Serie B | Italy | 15% |

*Belgian, Dutch, Scottish, Turkish leagues: in training data but not predicted (negative backtest ROI)*

### New-Format Model (goals only — no backtest available)
Ireland, Finland, Denmark, Austria, Sweden, Norway, Brazil, Japan, Mexico, China, USA MLS, Argentina

---

## Player Props Module (Phase 1)

**Markets:** SOT, Goals, Yellow Cards, Assists  
**Leagues:** Championship, League One, Bundesliga 2, Ireland Premier, Finland Veikkausliiga  
**Data:** FBref season stats + API-Football match-by-match (requires `APIFOOTBALL_KEY`)

**Architecture v2 highlights:**
- Referee factor for cards (z-score vs league avg)
- Set-piece sub-formula for defender SOT edge
- Proper de-vigging (assumed overround by odds tier)
- Relative edge floors (30% at 6.0 odds, 20% at 4.0)
- 5-component confidence scoring
- GES (Goal Edge Score) for goals/SOT signal gating

---

## Data Sources

| Source | Used For | Cost |
|---|---|---|
| football-data.co.uk | Historical match data (free) | Free |
| OddsAPI (the-odds-api.com) | Live odds + WC odds | $30/month (20K credits) |
| API-Football (api-sports.io) | Player match stats, lineups, referees | Free tier (100/day) → Starter ($15/mo) for 2026 season |
| Sofascore (cached) | Set-piece goals per team | Free (scraping) |
| FBref (scraping) | Player season stats for training | Free (scraping) |

---

## Setup

### GitHub Secrets required
```
ODDS_API_KEY      → the-odds-api.com (20K plan)
TELEGRAM_TOKEN    → @TheWowzaBot token
TELEGRAM_CHAT_ID  → group chat ID (-5181665372)
APIFOOTBALL_KEY   → api-sports.io key (for player props)
```

### Local `.env`
```
ODDS_API_KEY=your_key
APIFOOTBALL_KEY=your_key
```

### Local run
```bash
python pipeline.py --mode predict          # generate tips
python update_results.py                   # fill results
python -m player_model.pipeline --mode all # player props
```

---

## Roadmap

See `ROADMAP.md` for planned features:
1. **More markets** — 1X2, Asian Handicap, BTTS (same leagues, zero extra cost)
2. **More leagues** — Scottish League 1/2 (7.5% bookmaker margin, highest found)
3. **Cross-market confirmation** — SNIPER only when O/U + Sharp Money + Formula agree
4. **API-Football data** — first-half shots for HT live scanner PRESSURE_COOKER signal

---

## Dashboard

Public: `https://nevixaa-wowza-betting-app-kqbjnm.streamlit.app/`  
Local: `http://localhost:8501`
