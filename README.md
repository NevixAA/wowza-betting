# Wowza v9 — Football Betting Intelligence System

**Status:** Production — fully automated via GitHub Actions
**Live since:** April 2026
**Last doc update:** 2026-07-07

---

## What It Does

A football betting + prediction platform with several independent signal families, all
running automatically in the cloud (no server/PC needed).

| Module | What it does | Money? |
|---|---|---|
| **Standard O/U 2.5** | Over/Under 2.5 goals for our 7 second-division leagues — SNIPER/MARKSMAN/VALUABLE tiers, per-league thresholds | **Real-money candidate** |
| **Side Markets** | BTTS, O/U 1.5, O/U 3.5 — per-league walk-forward thresholds | Real-money candidate |
| **HT Model** | Half-time O/U 0.5 and 1.5 | Paper |
| **New-Format Model** | O/U for goals-only leagues (Brazil, Norway, MLS, …) — separate model, never mixed with standard | Paper |
| **Sharp Money Tracker** | Odds-drift detection — STEAM / STRONG / SHARP across enabled leagues | Info |
| **World Cup Tracker** | WC 2026 drift + ML value on O/U 1.5/2.5/3.5 and 1X2 | Paper |
| **Live Scanner** | In-play Poisson signals during match hours (UNDER_HOLD, SLEEPING_GAME, STRONG_STUCK, COMEBACK, HT_*) | Info/alert |
| **Player Props** | 7-market ML ensemble (Goals, Goals 2+, Assists, SOT 1-3+, Cards) | **Paper only — no betting edge** |
| **Fantasy (FPL)** | Repackages the accurate props model into FPL point projections (PL to start) | Prediction product |
| **Health Monitor** | Alerts if predict fetches 0 fixtures / all leagues error (outage detection) | Reliability |

---

## Signal Tiers (betting families)

| Tier | Edge | Stake | Note |
|---|---|---|---|
| 🎯 **SNIPER** | Per-league threshold (14–25%) | Full | Highest confidence |
| 🔫 **MARKSMAN** | ~14% to threshold | 3/4 | Medium-high |
| 💎 **VALUABLE** | 4–8% | Half | Side markets only (O/U 2.5 VALUABLE disabled — net-negative) |

---

## Architecture

```
pipeline.py                ← team models: train / predict / backtest
retrain.py                 ← full download + retrain (standard + new-format)
update_results.py          ← fill WIN/LOSS/PnL from football-data.co.uk
config.py                  ← paths, leagues, thresholds, MAX odds bounds

src/
  data_loader.py           ← Excel workbook + CSV + CI web download (last 4 seasons)
  feature_engineering.py   ← rolling form, HT, home advantage, per-format isolation
  model.py                 ← LogReg + GradientBoosting + Platt calibration
  predict.py               ← OddsAPI fixtures → model → tiers  (PRE-MATCH ONLY)
  betting.py               ← 3-tier signal logic (per-league thresholds)
  backtest.py              ← walk-forward backtest (no leakage)
  sharp_tracker.py         ← volume-weighted drift, steam, consensus
  live_scanner.py          ← in-play Poisson signals + HT sub-formula
  health_check.py          ← records fetch health → outage alert (via notifier)

player_model/
  config.py                ← 7 markets, prop league/tier thresholds, 139 feature cols
  api_football.py          ← API-Football client (stats, lineups, referees, injuries, live)
  feature_engineering.py   ← rolling player features (form / season / career)
  model.py                 ← per-market Platt-calibrated ensemble
  pipeline.py              ← collect → train → predict
  predict.py               ← de-vig EV, relative edge, lazy-market factors
  fantasy.py               ← FPL point projections + squad builder (Fantasy family)

telegram_bot/notifier.py   ← all Telegram sends + notify_predict_health()
pages/                     ← Streamlit dashboard
```

---

## Models

Every model is a **LogReg + GradientBoosting ensemble with Platt calibration**, trained
per-market with a chronological (walk-forward) split — no random leakage.

| Model file | Scope |
|---|---|
| `model_v9_standard.pkl` | O/U 2.5, standard-format (our 7 second divs, trained on the full standard pool) |
| `model_v9_newformat.pkl` | O/U 2.5, goals-only leagues — **never mixed with standard** |
| `model_v9_btts / over15 / over35.pkl` | Side markets |
| `model_ht_over05 / over15.pkl` | Half-time O/U |
| `model_player_{goals,goals2,assists,sot,sot2,sot3,cards}.pkl` | 7 player-prop markets |

### Player Props (7 markets — retrained 2026-07 on 4 seasons / 230k rows)

| Market | AUC |
|---|---|
| goals (1+) | 0.745 |
| goals2 (2+) | 0.828 |
| assists (1+) | 0.680 |
| sot (1+) | 0.734 |
| sot2 (2+) | 0.798 |
| sot3 (3+) | 0.844 |
| cards | 0.625 |

- **139 feature columns** per player-match: rolling form (5-game), season aggregates,
  career priors, opponent matchup composites, referee, injury, venue/position flags.
- **The model is accurate but has NO betting edge** on top-5 props (rigorously confirmed
  OOS, per league × market × role — ROI −41% to −57%, zero repeatable +EV cells).
  Top-5 prop markets are efficiently priced → **props stay paper.** Their real home is
  the **Fantasy** family (no bookmaker / no vig → accuracy converts to value).

---

## Automation (GitHub Actions)

| Workflow | Schedule (UTC) | Does |
|---|---|---|
| **predict** | every 5 min, 08–23 (at :01) | O/U 2.5 + side markets → Telegram + health record |
| **player_props** | hourly (:02) + nightly 23:30 + Sun 05:00 | props predict / WC collect / Sunday club retrain |
| **live_scanner** | every 5 min, 08–23 (at :03) | in-play Poisson signals |
| **worldcup** | hourly, 08–23 | WC drift + ML value |
| **sharp_tracker** | every 2 h (08–22) | drift signals |
| **update_results** | ~every 2 h | fill WIN/LOSS from football-data.co.uk |
| **injury_refresh** | daily 04:00 | refresh injury data |
| **daily_summary** | 07:00 | Telegram digest + props digest |
| **weekly_summary** | Mon 09:00 | performance summary |
| **retrain** | **Sun 03:00** | full team + player retrain, commit + push |
| **backtest** | monthly (1st, 03:00) | heavy walk-forward + per-league ROI |
| **preseason_retrain** | Aug 1 | season-start retrain |

Predict runs are **pre-match only** — matches that have already kicked off are skipped
(they belong to the live scanner, not the pre-match model).

---

## Reliability / monitoring

- **Pre-match filter** — `predict.py` skips already-started matches (an in-play match's
  live odds would otherwise produce a false SNIPER).
- **Outage health alert** — `health_check.py` records each fetch; `notify_predict_health()`
  pings Telegram if every league errors (0 fixtures) or the system goes stale. Guards
  against silent multi-day outages.
- **Team-name matching** — identity-token match (won't confuse Manchester City with
  Manchester United, etc.) when joining fixtures.
- **Odds sanity** — main O/U line odds are bounded (`MAX_OU_ODDS`) to reject stale/fringe prices.

---

## Leagues

### Standard O/U — BET (our 7 second divisions)
Championship · League One · League Two · Bundesliga 2 · La Liga 2 · Serie B · Ligue 2

*Additional full-stats leagues (Dutch, Portuguese, Greek, Turkish, Belgian, Scottish,
National League) live in `STANDARD_FORMAT_LEAGUES` as **training-only data** — they
improve the model but are **not bet** (not in `ENABLED_LEAGUES`).*

### New-Format O/U (goals only)
Denmark · Austria · Sweden · Norway · Finland · Ireland · Argentina · Brazil · Japan ·
Mexico · China · USA MLS · Romania (+ Saudi, K-League as training-only)

### Player Props / Fantasy
Top-5 + Champions League + Europa League + World Cup. (Player-prop **odds** exist only
for these on OddsAPI — smaller leagues return none.)

---

## Data Sources

| Source | Used for | Notes |
|---|---|---|
| football-data.co.uk | Historical match data | CI download = **last 4 seasons** |
| `England_Leagues_4_Seasons_With_Summary.xlsx` | Deep English history | **Local-only, not in git** — CI never uses it |
| OddsAPI (the-odds-api.com) | Live + WC odds, player props | credit-metered |
| API-Football (api-sports.io) | Player stats, lineups, injuries, referees, live scores | |
| Sofascore / FBref | set-piece + season stats | scraping |

> ⚠️ **Data note:** because the deep Excel is local-only, the automated CI retrain trains
> on the **last 4 seasons**. A local run *with* the Excel sees more history and different
> bet counts/ROI — so treat per-league ROI as re-validated each retrain
> (`output/backtest_by_league_standard.csv`), not as a fixed headline number.

---

## Setup

**GitHub secrets:** `ODDS_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `APIFOOTBALL_KEY`.
Local secrets live in gitignored `.env` / `.api_keys` (never committed).

```bash
python pipeline.py --mode predict                 # O/U 2.5 + side markets
python -m player_model.pipeline --mode predict     # player props
python retrain.py                                  # full team retrain (needs data)
streamlit run app.py                               # dashboard
```

**Dashboard pages:** Dashboard · Model Info · World Cup · Live · Sharp Money · HalfTime ·
Live History · Agent Analysis · Player Props · Portfolio · Fantasy.

---

## Development process

**Big / risky changes are staged in a `v10` copy first** — build → validate → test →
decide → then promote to live `v9`. Small validated bug fixes may go straight to `v9`.
(See `v10\V10_STAGING_README.md`.)
