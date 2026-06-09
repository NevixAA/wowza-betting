---
name: master-player-prop-analysis
displayName: Master Player Prop Analysis Prompt
description: "Prompt template for executing strict quantitative player prop analysis with calculated fair odds and positive EV detection."
---

# Master Player Prop Analysis Prompt

Use this prompt with the `master-player-prop-analytics` agent.
The goal is to apply a fixed quantitative framework that forces probability calculations, fair-odds derivation, and edge identification.
Do not use narrative reasoning. Use only mathematical models.

## Instructions

Provide all available player and market data. Minimum required: player, opponent, market type, player stats, and bookmaker odds.

- Player name and position
- Club and opponent
- Match date and competition
- Player statistics (last 5 matches, last 10 matches, season average, per-90 rates)
- Market type and half-line (e.g., "Shots On Target Over 1.5")
- Bookmaker odds for both sides (Over/Under or Yes/No)
- Team news (injuries, rotation risk, tactical changes, confirmed lineup)
- Lineup confirmation state (pre-match or confirmed)
- Recent referee history (if yellow cards market)
- Any opponent-specific context

## Prompt

Perform a strict quantitative analysis of the following player prop using the Master Player Prop Analytics framework.
Calculate fair odds, identify edge, and assign a clear BET / SMALL BET / NO BET verdict.

**MATCH**
- Team A vs Team B
- Date: `{{match_date}}`
- Competition: `{{competition}}`

**PLAYER**
- Name: `{{player_name}}`
- Position: `{{position}}`
- Club: `{{club}}`

**MARKET**
- Type: `{{market_type}}`
- Half-Line: `{{half_line}}`
- Over Odds: `{{over_odds}}`
- Under Odds: `{{under_odds}}`
(Or Yes/No odds if applicable)

**PLAYER STATISTICS**
- Last 5 matches: `{{last_5_stats}}`
- Last 10 matches: `{{last_10_stats}}`
- Season average: `{{season_avg}}`
- Per-90 rate: `{{per_90}}`

**RELEVANT METRICS** (by market)

*If Shots On Target market:*
- Shots/90: `{{shots_per_90}}`
- SOT/90: `{{sot_per_90}}`
- Shot accuracy %: `{{shot_accuracy}}`
- Touches in box/90: `{{touches_in_box}}`

*If Goals market:*
- xG/90: `{{xg_per_90}}`
- Goals/90: `{{goals_per_90}}`
- xG per shot: `{{xg_per_shot}}`
- Big chances (last 5/10): `{{big_chances}}`

*If Yellow Cards market:*
- Yellow cards/90: `{{cards_per_90}}`
- Fouls/90: `{{fouls_per_90}}`
- Tackles/90: `{{tackles_per_90}}`

*If Assists market:*
- xA/90: `{{xa_per_90}}`
- Assists/90: `{{assists_per_90}}`
- Key passes/90: `{{key_passes_per_90}}`

**CONTEXT**
- Opponent team: `{{opponent}}`
- Opponent defensive metric (SOT allowed/xGA/fouls drawn, depending on market): `{{opponent_metric}}`
- Expected minutes: `{{expected_minutes}}`
- Rotation risk: `{{rotation_risk}}`
- Lineup confirmed: `{{lineup_confirmed}}`
- Tactical notes: `{{tactical_notes}}`
- Referee history (if cards): `{{referee_history}}`

---

**REQUIRED OUTPUT (use exact structure below):**

### Match
`{{team_a}}` vs `{{team_b}}` | `{{date}}` | `{{competition}}`

### Player
`{{player_name}}` | `{{position}}` | `{{club}}`

### Market
`{{market_type}}` Over `{{half_line}}`

### Model Inputs
- Expected Minutes: XX
- Shots/90 (or goals/90, cards/90, xA/90): XX
- SOT/90 (or relevant metric): XX
- Opponent Adjustment: X.XX
- Tactical Notes: [brief]
- League Average for Opponent: XX

### Probability Calculation
**Formula:**
```
[Show exact model formula with values inserted]
```

**Calculation Steps:**
[Walk through each calculation step]

**Over Probability:** XX%
**Under Probability:** XX%

### Fair Odds
Over: X.XX
Under: X.XX

### Market Odds & De-Vigging
Over Odds: `{{over_odds}}`
Under Odds: `{{under_odds}}`

No-Vig Implied Probability (Over): XX%
No-Vig Implied Probability (Under): XX%

### Edge Analysis
**Edge (Over):** +X.X% (or −X.X%)
**EV (Over):** +X.X% (or −X.X%)

**Edge (Under):** +X.X% (or −X.X%)
**EV (Under):** +X.X% (or −X.X%)

### Confidence Score
**XX / 100**

- Data Quality: X/10
- Minutes Certainty: X/10
- Role Stability: X/10
- Market Agreement: X/10

### Verdict
**BET** | **SMALL BET** | **NO BET**

**Reasoning:** [One-paragraph explanation of verdict based on edge, EV, confidence, and comparison to market efficiency]

---

**ADDITIONAL ANALYSIS** (if sufficient data exists):

- **Opponent-Specific Defender Matchup:** [If applicable, defender matchup rating for SOT/goals]
- **Team Projected Goals:** [If available, compare team xG from betting market vs model λ]
- **Lineup Probability Model:** [Start/Sub/DNP probabilities with reasoning]
- **Monte Carlo Simulation:** [If appropriate, 10,000+ iteration result showing 95% CI on fair odds]
- **Closing-Line Value:** [Historical CLV tracking if prior bets exist; validate edge]

---

**BEHAVIORAL REQUIREMENTS:**

- Show all formulas explicitly.
- Insert all numbers into formulas before calculating.
- Do not use narrative reasoning; use only mathematical models.
- State all assumptions.
- Apply league-average adjustments where specified.
- De-vig odds before comparing to model probability.
- Only recommend bets with edge ≥ 2% and EV ≥ 0.05.
- Flag low-confidence bets (< 65/100) with clear explanation.
- Reference data source for all statistics (FBref, Opta, StatsBomb, etc.).
