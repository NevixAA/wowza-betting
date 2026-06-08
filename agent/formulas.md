# Betting Edge Formula Library

## Core Workflow
1. Estimate expected event rate (λ)
2. Convert λ into probabilities
3. Convert probabilities into fair odds
4. Compare with bookmaker odds
5. Calculate EV and edge

---

## 1. Over / Under 2.5 Goals

### Team Goal Expectancy Model

**Home Expected Goals:**
```
λ_H = LHG × HAS × ADS × HA × TF × TA
```
- LHG = League Home Goals Average
- HAS = Home Attack Strength
- ADS = Away Defensive Weakness
- HA  = Home Advantage
- TF  = Tempo Factor
- TA  = Tactical Adjustment

**Away Expected Goals:**
```
λ_A = LAG × AAS × HDS × AF × TA
```
- LAG = League Away Goals Average
- AAS = Away Attack Strength
- HDS = Home Defensive Weakness
- AF  = Away Factor

**Total Expected Goals:**
```
λ_T = λ_H + λ_A
```

### Over 2.5 Probability (Poisson)
```
P(Over 2.5) = 1 - [P(0) + P(1) + P(2)]

P(k) = (e^(-λ_T) × λ_T^k) / k!
```

### Fair Odds
```
Fair Odds = 1 / P(Over 2.5)
```

### Value Edge
```
Edge            = Model Probability - Market Probability
Market Probability = 1 / Bookmaker Odds
```

### [Custom Model] Goal Pressure Score (GPS)
```
GPS = 0.25(SOT) + 0.25(BigChances) + 0.25(xG) + 0.15(BoxEntries) + 0.10(SetPieceThreat)
```
Higher combined GPS → higher Over likelihood.

### [Custom Model] Pace Index (PI)
```
PI = 0.4(PossessionTransitions) + 0.3(Shots) + 0.3(PPDA^-1)
```
Fast games outperform market on Overs.

---

## 2. First Half Betting

### First-Half Goal Share (league-specific)
```
FH% = FHGoals / TotalGoals
```
Typical values: EPL ≈ 44%, Bundesliga ≈ 45%, Eredivisie ≈ 46%

### First Half Expected Goals
```
λ_FH = λ_T × FH%
```
Example: 2.8 × 0.45 = 1.26

### HT Over 0.5
```
P(HT > 0.5) = 1 - e^(-λ_FH)
```

### HT Over 1.5
```
P(HT > 1.5) = 1 - [P(0) + P(1)]   using Poisson(λ_FH)
```

### HT Result Model
```
λ_H,FH = λ_H × FH%
λ_A,FH = λ_A × FH%
```
Then apply Poisson matrix for HT scoreline probabilities.

### HT Win Probability
```
P(Home HT Win) = Σ P(i,j)   for all i > j
```

### [Custom Model] HT Dominance Index (HTDI)
```
HTDI = 0.35(EarlyGoals) + 0.25(FHxG) + 0.20(FHShots) + 0.10(Pressing) + 0.10(HomeAdvantage)
```

---

## 3. Player Props

### Expected Shots On Target
```
xSOT = (Minutes / 90) × ShotsPer90 × SOTRate × OpponentAdjustment
```
Example: 4.2 shots/90, 38% on target, 85 min → 4.2 × 0.38 × (85/90) = 1.51

**Over 1.5 SOT:**
```
P(SOT ≥ 2) = 1 - [P(0) + P(1)]   using Poisson(xSOT)
```

### Expected Shots
```
xShots = (Minutes / 90) × ShotRate × OpponentWeakness × GameStateFactor
```

### Anytime Goalscorer
```
xG_player = (Minutes / 90) × PlayerxG90 × TeamAttackFactor × OpponentDefWeakness
P(Goal)    = 1 - e^(-xG_player)
```

### Assists
```
xA_player  = (Minutes / 90) × xA90 × FinishingFactor
P(Assist)  = 1 - e^(-xA_player)
```

### [Custom Model] Card Risk Score (CRS)
```
CRS = 0.30(Fouls90) + 0.20(Tackles90) + 0.15(Yellows90) + 0.15(RefereeStrictness) + 0.10(PositionFactor) + 0.10(MatchIntensity)
```

### Expected Cards
```
λ_C     = Yellows90 × MinutesFactor × RefFactor × IntensityFactor
P(Card) = 1 - e^(-λ_C)
```

### Expected Fouls
```
xFouls = Fouls90 × MinutesFactor × OpponentDribbleFactor
```

### Expected Tackles
```
xTackles = Tackles90 × OpponentPossessionFactor × MinutesFactor
```

---

## 4. Core Edge Formulas

### Betting Value Signal (BVS)
```
BVS = P_model - P_market
```

### Expected Value (EV)
```
EV = (P_model × Odds) - 1
```

### Kelly Criterion
```
Kelly = (b×p - q) / b

b = decimal odds − 1
p = model probability
q = 1 − p
```

**Staking tiers:**
- Full Kelly = aggressive
- Half Kelly = professional (recommended)
- Quarter Kelly = conservative

---

## 5. BTTS (Both Teams To Score)

```
P(BTTS) = (1 - e^(-λ_H)) × (1 - e^(-λ_A))
```
- λ_H = Home expected goals
- λ_A = Away expected goals

**With Dixon-Coles correction** (more accurate for low-scoring games):
Sum all scoreline cells (i,j) from the corrected matrix where i ≥ 1 AND j ≥ 1.

```
Fair Odds (BTTS Yes) = 1 / P(BTTS)
Fair Odds (BTTS No)  = 1 / (1 - P(BTTS))
EV                   = (P(BTTS) × Bookmaker Odds) - 1
```

---

## 6. Correct Score (Dixon-Coles)

Pure Poisson underestimates low-scoring draws. Dixon-Coles applies a correction factor τ to the four low-score cells.

**Corrected scoreline probability:**
```
P(Home=x, Away=y) = τ(x,y) × Poisson(x, λ_H) × Poisson(y, λ_A)
```

**Correction factor τ(x, y):**
```
τ(0,0) = 1 - λ_H × λ_A × ρ
τ(1,0) = 1 + λ_A × ρ
τ(0,1) = 1 + λ_H × ρ
τ(1,1) = 1 - ρ
τ(x,y) = 1   for all other scorelines
```
- ρ (rho) = low-score correction parameter, estimated from historical data (typical range: −0.03 to −0.15)
- All other cells (x ≥ 2 or y ≥ 2) use standard Poisson, τ = 1

**Workflow:**
1. Compute λ_H and λ_A from Team Goal Expectancy Model
2. Apply τ correction to cells (0,0), (1,0), (0,1), (1,1)
3. Compute standard Poisson for all other cells
4. Normalize full matrix so probabilities sum to 1.0
5. Fair Odds for any scoreline = 1 / P(x, y)

---

## 7. Asian Handicap

Build the full Poisson scoreline matrix then aggregate by goal differential.

**Scoreline matrix cell:**
```
P(Home=i, Away=j) = Poisson(i, λ_H) × Poisson(j, λ_A)
```

**Aggregate by handicap line:**
```
AH -0.5  (Home)   → P(i - j ≥ 1)    = sum all cells where i > j
AH -1.0  (Home)   → P(i - j ≥ 2)    = sum all cells where i - j ≥ 2 (push if diff = 1)
AH -1.5  (Home)   → P(i - j ≥ 2)    = sum all cells where i - j ≥ 2
AH +0.5  (Away)   → P(j - i ≥ 0)    = sum all cells where j ≥ i
AH +1.0  (Away)   → P(j - i ≥ 0)    = sum all cells where j ≥ i (push if diff = -1)
```

**Quarter lines (e.g. AH -0.25, AH -0.75):**
Quarter lines split the stake across two adjacent half-lines:
```
P(AH -0.25) = 0.5 × P(AH -0.0) + 0.5 × P(AH -0.5)
P(AH -0.75) = 0.5 × P(AH -0.5) + 0.5 × P(AH -1.0)
```

**Push (half-stake refund on integer lines):**
On AH -1.0, if goal diff = exactly 1 → push (stake returned, no profit/loss).

```
EV (AH -0.5) = (P(Home wins) × Odds) - 1
```

**Alternative — Skellam Distribution:**
Models goal difference Z = Home − Away directly without building the full matrix.
```
Z ~ Skellam(λ_H, λ_A)
P(Z = k) = e^(-(λ_H + λ_A)) × (λ_H/λ_A)^(k/2) × I_|k|(2√(λ_H × λ_A))
```
Where I_k is the modified Bessel function of the first kind. Use when speed matters over granularity.

---

## 8. Advanced Player Props

### Minutes Played Model
```
xMinutes = BaseMinutes × FitnessIndex × ManagerRotationFactor × FixtureImportance
```
- BaseMinutes = rolling 5-game average minutes
- FitnessIndex = 1.0 if fully fit, scaled down for fatigue/minor knocks
- ManagerRotationFactor = derived from historical squad rotation patterns
- FixtureImportance = 1.0 for key matches, < 1.0 for cup rotation risk

### Key Passes
```
xKeyPasses = (Minutes / 90) × KeyPasses90 × OpponentMidfieldPressure × TeamPossessionStyle
```

### Pass Completions
```
xPassComp = (Minutes / 90) × Passes90 × CompletionRate × OpponentPressingIntensity
P(Over line) = 1 - Poisson CDF at line using xPassComp as λ
```

### Dribbles Completed
```
xDribbles = (Minutes / 90) × DribblesAttempted90 × DribbleSuccessRate × OpponentTacklingFactor
```

### Offsides
```
xOffsides = (Minutes / 90) × Offsides90 × OpponentDefLineFactor
```
- OpponentDefLineFactor > 1.0 for high defensive lines (more offside traps)

### [Custom Model] Player Performance Score (PPS)
Combines all prop signals into a single confidence index:
```
PPS = 0.30(xG) + 0.20(xSOT) + 0.20(xKeyPasses) + 0.15(xDribbles) + 0.15(xMinutes/90)
```
Higher PPS → more likely player hits multiple prop lines in same game.

---

## 9. Guaranteed Bets & Arbitrage Detection

### Arbitrage Detection (Cross-Bookmaker)
```
ArbitrageSum = (1/Odds_H) + (1/Odds_D) + (1/Odds_A)

If ArbitrageSum < 1.0 → arbitrage exists
Guaranteed Profit% = (1 - ArbitrageSum) × 100
```

**Optimal stake allocation:**
```
Stake_i = (TotalStake / ArbitrageSum) × (1 / Odds_i)
```
This guarantees the same return regardless of outcome.

**Two-way markets (BTTS, Over/Under, AH):**
```
ArbitrageSum = (1/Odds_Yes) + (1/Odds_No)
If Sum < 1.0 → arbitrage
```

### Model-Driven High-Confidence Value Signal
When multiple independent methods agree, confidence escalates:

**Consensus Probability:**
```
P_consensus = (w₁×P₁ + w₂×P₂ + w₃×P₃) / (w₁ + w₂ + w₃)
```
Where w_i = confidence weight of each model (Poisson, Dixon-Coles, Elo, etc.)

**High-Confidence threshold:**
```
Signal is HIGH CONFIDENCE when:
  - P_consensus vs P_market edge > 7%
  - All individual models agree on same outcome
  - EV > 0 on ≥ 3 independent models

Signal is MAXIMUM CONFIDENCE ("closest to guaranteed") when above + ArbitrageSum < 1.0
```

**Multi-Model EV:**
```
EV_consensus = (P_consensus × Bookmaker_Odds) - 1
```

### Value Bet Ranking (combined model)
```
Tier S ("guaranteed-like"): Arbitrage confirmed + model edge > 7%
Tier A: All models agree + EV > 10%
Tier B: Majority models agree + EV 5–10%
Tier C: Single model edge 3–7%
```

> Note: "Guaranteed" in betting means arbitrage-locked profit across bookmakers.
> High model consensus is the closest equivalent when true arbitrage is unavailable.
