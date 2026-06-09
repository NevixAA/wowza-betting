# Player Props Betting System — ML Architecture v2
**Status:** Design Phase — Formulas Updated by Player Prop Analyst + Master Player Prop Analytics Agents
**Version:** 2.0 — Modeling fixes applied, new markets added
**Last updated:** 2026-06-09

---

## 1. System Overview

Parallel module alongside the existing O/U 2.5 system.
Core thesis: bookmakers price player props using population averages by position,
not situational decomposition. The system exploits systematic mispricings.

**Shared with existing system:** TeamAttackFactor, OpponentDefWeakness, Poisson base, odds pipeline
**New data added:** Player-level stats, referee profiles, set-piece delivery, keeper save rates

### Lazy Market Hypothesis
High-odds (4.0+) markets are lazy because:
- Low volume → less bookmaker risk management
- Generic pricing → position averages, situational factors ignored
- Asymmetric information → most bettors have no player situational model

---

## 2. Data Pipeline

### Data Sources
| Source | Data | Cost | Refresh |
|---|---|---|---|
| API-Football v3 | Player stats, lineups, xG, shots, cards, minutes | ~$15/mo | 6h post-match |
| football-data.co.uk | Referee data, corners, fouls | Free | Weekly CSV |
| Sofascore (scraping) | Set-piece delivery, touches in box | Free | 12h post-match |
| Existing odds pipeline | Player prop odds (extend) | Existing | 2h pre-match |
| FBref (scraping) | Progressive carries, PPDA, pressures | Free | Weekly |

### Database Schema

**player_match_stats**
```
player_id, match_id, team_id, opposition_id, date, competition_id,
position, minutes_played, started, shots_total, shots_on_target,
goals, assists, key_passes, xg, xa, yellow_card, red_card,
fouls_committed, fouls_drawn, tackles, interceptions, saves,
corners_taken, dribbles_attempted, dribbles_completed,
touches_opp_box, set_piece_shots, aerial_won, aerial_total
```

**player_rolling_stats** (pre-computed, refreshed post-match)
```
player_id, as_of_date, competition_id,
shots_per90_l5, shots_per90_l10, sot_per90_l5, sot_per90_l10,
sot_rate_l10, xg_per90_l10, goals_per90_l10, assists_per90_l10,
fouls_per90_l10, tackles_per90_l10, yellow_rate_l10, saves_per90_l10,
corners_per90_l10, minutes_per_game_l5, set_piece_shot_rate,
touches_box_per90, key_passes_per90_l10, xa_per90_l10
```

**referee_profiles**
```
referee_id, referee_name, competition_id, matches_officiated,
yellows_per_game, reds_per_game, fouls_per_game,
home_bias_factor, strictness_score (z-score vs league mean),
card_per_foul_rate
```

**keeper_profiles**
```
keeper_id, team_id, shots_faced_per90, save_rate_l10,
saves_per90_l10, set_piece_concede_rate, high_ball_claim_rate
```

**team_set_piece_profile**
```
team_id, competition_id, corners_per90, set_piece_goals_rate,
corner_taker_id, corner_taker_delivery_acc, freekick_taker_id,
set_piece_shot_freq
```

### ⚠️ Settlement Rules (Critical — fake edge risk)

- **Saves:** Opta counts "parried-and-recovered" saves; StatsBomb may not. Confirm per-book data feed before modeling saves
- **Assists:** Some books require final pass only; others include pre-assist (second assist). Verify per bookmaker
- **Cards after final whistle:** Some books settle "1+ card" including stoppage-time bookings after final whistle. Ensure λ_C covers 90+ minutes if applicable
- **Direct corner/FK goals:** Most books do NOT award an assist for direct corner goals. Filter these from xA calculation

---

## 3. Lineup State Mixture Model (⭐ CRITICAL FIX v2)

**Old approach (wrong):** `xStat = (Minutes/90) × StatPer90`  ← scalar multiplier

**Correct approach — discrete mixture over lineup states:**
```
E[Stat] = P(Start) × E[Stat | Start, E[min|Start]]
        + P(SubOn) × E[Stat | SubOn, E[min|SubOn]]
        + P(DNP)   × 0

P(Start) = logistic(last5_starts, injury_flag, rotation_pattern, cup_vs_league)
E[min | Start] = historical_avg_l5 × rest_day_factor
E[min | SubOn] = truncated_normal(mean=30, sd=15)  -- fit from historical sub times
```

**Always gate on `P(Start) + P(SubOn)` before computing EV.**
A 6.0 odds signal on a 60% starter has different true EV than on a confirmed starter.

---

## 4. Feature Engineering & Formulas (v2 — Corrected)

### 4.1 Shots on Target (SOT)

**⚠️ GLM with log-offset replaces simple multiplier:**
```
log(E[SOT]) = β₀ + β₁·log(xG_role) + β₂·OpponentPSxG + log(minutes/90)
```
The `log(minutes/90)` is an exposure OFFSET, not a coefficient. Fit via Poisson GLM.

**Set-piece SOT sub-formula (KEY EDGE — defender/CB case):**
```
xSOT_setpiece = defender_in_box_rate
              × corner_taker_delivery_acc
              × keeper_set_piece_concede_rate
              × set_piece_shot_rate
              × (corners_per90 × projected_minutes/90)

xSOT_openplay = (min/90) × shots_per90_l10 × sot_rate_l10
              × opp_sot_conceded_per90 / league_avg_sot_conceded

xSOT_total = xSOT_openplay + xSOT_setpiece
```

**Concrete defender example:**
```
xSOT_setpiece = 0.40 × 0.65 × 0.25 × 1.30 × 6.5 = 0.219
xSOT_openplay = 0.050
xSOT_total    = 0.269

P(SOT >= 1) = 1 - e^(-0.269) = 23.6%
Market @ 6.0 → fair_implied = 16.7% / 1.10 = 15.2% (de-vigged)
EV = (0.236 × 6.0) - 1 = +41.6%  ← SNIPER
```

**Missing feature added (v2):**
```
f_opp_low_block_rate  -- deep-defending teams suppress shots for dribblers
f_opp_block_rate      -- opponent block rate reduces SOT conversion
```

### 4.2 Goals (Anytime Scorer)

**⚠️ GoalsOverXgFactor fix — use shrunk finishing residual:**
```
-- OLD (wrong — double-counts shot selection):
xG_player = (min/90) × PlayerxG90 × TeamAttackFactor × OppDefWeakness × GoalsOverXgFactor

-- NEW (correct):
FinishResidual = (Goals - xG) / sqrt(xG)                    # z-score stabilized
ShrunkenFinish = FinishResidual × (N_shots / (N_shots + 150))  # Bayesian shrinkage
xG_player = (min/90) × PlayerxG90 × TeamAttackFactor × OppDefWeakness
           × exp(ShrunkenFinish × σ_finishing)               # σ_finishing ≈ 0.15
```

**Goal Edge Score (GES) — integrate as multiplier:**
```
GES = (0.40 × xG_form_score)     -- normalized 0→1
    + (0.20 × shot_volume_score)
    + (0.15 × penalty_duty_flag)
    + (0.15 × opp_weakness_score)
    + (0.10 × minutes_security_score)

GES_multiplier = 0.70 + (GES × 0.60)   -- GES=0 → ×0.70, GES=1.0 → ×1.30
xG_player_final = xG_player × GES_multiplier

-- GES thresholds for signal gating:
GES >= 0.70 → required for SNIPER goals bets
GES >= 0.50 → required for MARKSMAN goals bets
GES  < 0.35 → suppress goals signal even if EV passes
```

**P(Goal) = 1 - e^(-xG_player_final)**

### 4.3 Yellow Cards — Zero-Inflated Poisson (ZIP)

**⚠️ Structural zeros exist — plain Poisson is wrong:**
```
-- Structural zero gate (player who never fouls/tackles → always 0):
P(Y=0)   = π + (1-π) × e^(-λ_C)
P(Y=k>0) = (1-π) × e^(-λ_C) × λ_C^k / k!

π = logistic(β·no_tackle_flag + β·referee_lenient + β·low_intensity_match)

λ_C = Yellows90 × MinutesFactor × RefFactor × IntensityFactor × OppDribbleFactor
```

**CRS (Card Risk Score) — use for ranking/ranking, not as λ:**
```
CRS = 0.30(Fouls90) + 0.20(Tackles90) + 0.15(Yellows90)
    + 0.15(RefereeStrictness) + 0.10(PositionFactor) + 0.10(MatchIntensity)

-- CRS maps to tier gate: CRS >= 0.65 required for card SNIPER signal
```

**Referee factor (KEY EDGE):**
```
RefFactor = Ref_yellows_per_game / League_avg_yellows_per_game
-- Player at 4.0 (implied 22%) under avg ref  → P(card) ≈ 22%  [no edge]
-- Same player under +2σ ref (+46% more yellows) → P(card) ≈ 31%  [EV = +40%]
```

**Position factors:**
| Position | Factor |
|---|---|
| Defensive Midfielder | 1.20 |
| Center Back | 1.10 |
| Full Back | 1.05 |
| Central Midfielder | 1.00 |
| Winger | 0.75 |
| Striker | 0.65 |

### 4.4 Goalkeeper Saves — Negative Binomial (NB2)

**⚠️ Saves are overdispersed — Poisson underestimates P(saves >= 5) by 15-25%:**
```
E[Saves] = μ = (min/90) × opp_sot_p90 × SaveRate_shrunk
Var[Saves] = μ + μ²/r          -- estimate r ≈ 4–8 via MLE

P(Saves >= k) = 1 - NB_CDF(k-1; μ, r)
```

### 4.5 Assists — Decomposed Open Play + Set Piece

**⚠️ Not independent — separate components:**
```
xA_openplay = (min/90) × key_passes_per90 × team_conversion_rate × opp_press_factor
xA_setpiece = corner_taker_flag × team_set_piece_goal_rate × (min/90)
xA_total    = xA_openplay + xA_setpiece
P(Assist)   = 1 - e^(-xA_total)
```

**Set-piece assists — Zero-Inflated Negative Binomial:**
```
-- Structural zero: non-takers get π_SP ≈ 0.90+ (nearly always zero)
P(xSPA > 0) = (1 - π_SP) × [1 - NB_CDF(0; μ_SP, r_SP)]
π_SP = logistic(β·NonCornerTakerFlag + β·NoDirectDeliveryHistory)
```

**KEY EDGE — confirmed corner taker:**
Corner takers have dramatically elevated assist P. Market prices them as generic assisters.
`corner_taker_flag` is binary and the single most exploitable factor in the assists market.

### 4.6 Shots Total
```
xShots = (min/90) × ShotRate × OpponentWeakness × GameStateFactor × f_opp_low_block_rate
P(Shots >= k) = 1 - Poisson_CDF(k-1, xShots)
```

### 4.7 Corners Taken
```
xCorners = corner_taker_confirmed × team_corners_per90 × (min/90)
```
Nearly deterministic for confirmed takers. Edge from lineup confirmation only.

### 4.8 Fouls / Tackles
```
xFouls   = Fouls90   × MinutesFactor × OppDribbleFactor × RefFactor
xTackles = Tackles90 × MinutesFactor × OppPossessionFactor
```

---

## 5. NEW Markets (v2)

### 5.1 Booking Points (10pts yellow, 25pts red) — ⭐ Most mispriced market

```
xBP = (10 × P_yellow) + (25 × P_red)

P_yellow = 1 - e^(-λ_C)
P_red    = P_yellow × red_given_yellow_rate   -- historical player rate, default 0.08

-- Lines:
P(BP >= 10) = P_yellow + P_red × (1 - P_yellow)
P(BP >= 25) = P_red
P(BP >= 35) = P_yellow × P_red   -- correlation: 2nd yellow leads to red
```

### 5.2 First Goalscorer

```
P(FGS) ≈ xG_player / λ_match_total   [first-order approximation]

Refined:
P(FGS) = xG_player × (1 - e^(-λ_match_total))^(-1) × positional_first_shot_factor

positional_first_shot_factor:
  ST=1.15, CAM=0.95, WM=0.85   [forwards shoot earlier in match]
```

### 5.3 2+ Shots on Target

```
P(SOT >= 2) = 1 - Poisson_CDF(1, xSOT_total)
            = 1 - [e^(-λ) + λ×e^(-λ)]

⭐ NOTE: bookmakers price SOT >= 2 ~3× more lazily than SOT >= 1 — highest exploit rate
```

### 5.4 Carded AND Scores (Correlation Play)

```
-- NOT independent events — use correlation adjustment:
P(card AND goal) ≈ P(card) × P(goal) × correlation_factor

correlation_factor by position:
  ST/CF:     1.25   [fouled in box → goal chance + card risk are correlated]
  WM:        1.10
  CM/CDM:    0.85   [card reduces playing time → fewer goal opportunities]
  CB:        0.90
```

---

## 6. EV & Edge Detection (v2 — Corrected)

### 6.1 De-Vigging (Fixed)

**⚠️ Old approach: `edge = P_model - 1/Odds` — ignores bookmaker margin**

**New approach — de-vig with assumed overround:**
```
-- When both sides available (two-way market):
raw_yes  = 1 / Odds_yes
raw_no   = 1 / Odds_no
overround = raw_yes + raw_no

fair_yes = raw_yes / overround    ← use this for edge calculation
edge     = P_model - fair_yes
EV       = (P_model × Odds_yes) - 1   ← unchanged (uses decimal odds)

-- When only one side available (most player prop markets):
assumed_overround:
  Odds < 3.0:  1.06  (competitive market)
  3.0-5.0:     1.10
  Odds >= 5.0: 1.15  (thin market, higher margin)

fair_prob = (1/Odds) / assumed_overround
edge      = P_model - fair_prob
```

**Why it matters:** At 6.0 odds with 15% margin: raw implied = 16.7%, fair = 14.5%.
Model at 20% has +5.5pp edge vs fair (not +3.3pp vs raw) → more signals qualify.

### 6.2 Relative Edge (Fixed for High-Odds Bets)

**⚠️ Flat absolute edge unfairly penalizes high-odds bets**

```
relative_edge = (P_model - fair_prob) / fair_prob

-- HOO minimum relative edge by odds tier (replaces flat 1.15× criterion):
Odds >= 6.0:  relative_edge >= 0.30  (30% relative)
Odds >= 4.0:  relative_edge >= 0.20  (20% relative)
Odds >= 3.0:  relative_edge >= 0.12  (12% relative)
```

### 6.3 Kelly-Weighted EV for High-Odds

```
relative_edge_weight = 1 + log(Odds / 3.0)   -- 1.0 at 3.0, 1.49 at 6.0, 1.92 at 10.0
kelly_EV = EV × relative_edge_weight
```

---

## 7. Signal Tiers (v2 — Stricter)

| Tier | EV | Odds | Confidence | Lazy factors |
|---|---|---|---|---|
| SNIPER | > 0.40 | >= 5.0 | >= 0.72 | >= 2 |
| MARKSMAN | > 0.25 | >= 4.0 | >= 0.62 | >= 1 |
| VALUABLE | > 0.15 | >= 3.0 | >= 0.50 | any |

**For goals/SOT signals, also require:**
- SNIPER: GES >= 0.70
- MARKSMAN: GES >= 0.50

**Kelly staking (20% fractional — more conservative than team model):**
```
Kelly_fraction = EV / (Odds - 1) × 0.20
Caps: SNIPER max 3%, MARKSMAN max 2%, VALUABLE max 1%
Never combine same-match player props in a parlay
```

---

## 8. Confidence Scoring (v2 — 5-component)

**Replaces single float with composite:**
```
confidence = (
    0.30 × data_volume_score      -- n_games: 0→0, 5→0.40, 10→0.70, 20→1.0
  + 0.25 × recency_weight         -- exp decay: w_i = 0.90^(i-1), normalized
  + 0.20 × formula_xgb_agreement  -- 1 - abs(P_formula - P_xgboost)
  + 0.15 × lazy_factor_count      -- min(n_lazy_factors / 2, 1.0)
  + 0.10 × minutes_certainty      -- P(starts) confirmed or estimated
)
```

**Minimum guard:** If player has < 5 games this season → NO signal regardless of EV.

---

## 9. Model Architecture

### Three-Layer System
```
Layer 1: FORMULA ENGINE (Poisson/NegBin/ZIP as appropriate per market)
Layer 2: CALIBRATION (XGBoost — optional, requires >=500 player-match obs)
Layer 3: EDGE FILTER (relative edge + confidence + lazy factor gates)
```

### Distribution by Market
| Market | Distribution | Why |
|---|---|---|
| SOT | Poisson (GLM + offset) | Adequate for moderate counts |
| Goals | Poisson (shot-level) | Standard, works well |
| Cards | Zero-Inflated Poisson | Structural zeros (non-tackling players) |
| Saves | Negative Binomial (NB2) | Overdispersed, r≈4-8 |
| Assists | Poisson (open play) + ZIP (set piece) | Structural zeros for non-takers |
| Booking Pts | Composite Poisson | Derived from card model |
| Corners taken | Deterministic | Binary taker flag |
| Fouls/Tackles | Poisson | Adequate |

---

## 10. Telegram Alert Format (v2)

```
[TIER] | PLAYER PROP
[Home] vs [Away] | [League] | [KO Time]
PLAYER: [Name] ([Position], [Team])
MARKET: [Market description]
ODDS: [X.XX] @ [Bookmaker]
━━━━━━━━━━━━━━━━
MODEL P: [X.X%] | FAIR IMPLIED: [X.X%] | RAW IMPLIED: [X.X%]
EDGE: +[X.X]pp absolute | +[X.X%] relative | EV: +[X.X%]
GES: [0.XX] (goals/SOT only)
━━━━━━━━━━━━━━━━
WHY THE EDGE:
• [LAZY_FACTOR_1]: [explanation]
• [LAZY_FACTOR_2]: [explanation]
━━━━━━━━━━━━━━━━
CONFIDENCE: [X.XX] | DATA: [n] games | MINUTES: [n] (P_start=[X%])
  Volume=[X] Recency=[X] Agreement=[X] Lazy=[X] Minutes=[X]
KELLY: [X.X%] of bankroll
```

---

## 11. Lazy Market Factor Taxonomy

| Code | Description | Example |
|---|---|---|
| SET_PIECE | Set-piece involvement ignored | CB with high box presence, team 7+ corners/game |
| REFEREE | Strict referee not in price | Player at 4.0, ref books 46% above avg |
| KEEPER_WEAK | Opponent keeper weak | GK saves over 2.5, keeper has 55% save rate |
| MINUTES | Expected minutes understated | Rotation player 65 min, priced as if 45 |
| OPPONENT_SHAPE | Deep block → more set pieces | Parking bus → 8+ corners → aerial threats underpriced |
| GAME_STATE | Match competitive late | Relegation battle → more fouls/cards |
| CORNER_TAKER | Lineup-confirmed taker | Same player, 1.5x assists from set pieces |
| POSITIONAL_MISLABEL | Wrong archetype | Inverted winger priced as LB for shots |

---

## 12. Data Costs

**MVP:** ~$15/month (API-Football v3) — covers all markets
**Everything else:** Free (football-data.co.uk, Sofascore scraping, FBref)

---

## 13. Implementation Roadmap

| Phase | Weeks | Build |
|---|---|---|
| Phase 1 | 1-2 | Database schema + API-Football backfill (2 seasons) |
| Phase 2 | 3-4 | Formula engine: SOT set-piece sub-formula first, then cards referee model |
| Phase 3 | 5-6 | Edge detection + de-vig + relative edge + HOO pipeline |
| Phase 4 | 7-9 | XGBoost calibration (SOT, Goals, Cards) |
| Phase 5 | 10 | Telegram integration (separate player props channel) |
| Phase 6 | 11-16 | Live validation: VALUABLE only → MARKSMAN → SNIPER |

**Critical path: 6 weeks to first signal (Phase 1-3)**
**Build first:** SOT set-piece formula (defenders) + cards referee factor

---

## 14. Formulas Still Needed (Next Research Agent)

- [ ] Dixon-Coles correction for player-level count events
- [ ] Bivariate Poisson for correlated events (goals + assists same match)
- [ ] Bayesian shrinkage for new players / small samples
- [ ] Pressing intensity model (PPDA → foul probability)
- [ ] Aerial duel model (header goals, set-piece clearances)
- [ ] Goalkeeper distribution model (long ball → second ball shots)
- [ ] Injury/fatigue decay model (minutes in last 7 days → performance decay)
- [ ] Home/away splits for player stats (some players significantly better at home)
- [ ] Opposition shape adjustment (low block vs high line → shot quality change)
- [ ] Opponent-specific defender matchup ratings
- [ ] Monte Carlo simulation framework (10,000+ iterations for correlated props)
- [ ] CLV tracking module (compare signals against sharp closing line)
