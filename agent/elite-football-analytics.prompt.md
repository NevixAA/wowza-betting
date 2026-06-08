---
name: elite-football-analytics-prompt
displayName: Elite Football Analytics — Sniper Value Signal Prompt
description: "Prompt template for hunting sniper value signals and high-confidence betting edges using the Elite Football Analytics agent."
---

# Elite Football Analytics — Sniper Value Signal Prompt

Use this prompt with the `elite-football-analytics` agent.
The goal is NOT general match analysis. The goal is to **find specific, high-confidence value signals** — sniper bets where the model has a clear edge over the market.

## Instructions

Fill in as many fields as possible. More data = sharper signals. Required minimum: teams, competition, and at least one set of market odds.

- Home team
- Away team
- Competition / league
- Match date and kickoff time
- Market odds across all available lines (1X2, totals, AH, BTTS, correct score, player props)
- Team news (injuries, suspensions, rotation risk, fatigue)
- Recent form and season stats (football-data.co.uk, API-Football, Sofascore)
- Any known context: referee, weather, motivation, line movement

## Prompt

Hunt for sniper value signals in the following match using the Elite Football Analytics & Betting Intelligence framework.
Run the full model pipeline, then surface only the strongest edges.

**Match:**
- Home team: `{{home_team}}`
- Away team: `{{away_team}}`
- Competition: `{{competition}}`
- Date / kickoff: `{{match_date}}`

**Market odds:**
- 1X2 — Home: `{{home_odds}}` | Draw: `{{draw_odds}}` | Away: `{{away_odds}}`
- Over 2.5 goals: `{{over25_odds}}` | Under 2.5: `{{under25_odds}}`
- BTTS Yes: `{{btts_yes_odds}}` | BTTS No: `{{btts_no_odds}}`
- Asian Handicap line: `{{ah_line}}` — Home: `{{ah_home_odds}}` | Away: `{{ah_away_odds}}`
- Correct score (top lines): `{{correct_score_odds}}`
- HT Over 0.5: `{{ht_over05_odds}}` | HT Result: `{{ht_result_odds}}`
- Player props: `{{player_props}}`

**Team news:** `{{team_news}}`
**Additional context:** `{{additional_context}}`

---

**Required output — lead with the signals, support with the model:**

### 1. STRONGEST SIGNALS (top of response)
List only Tier S, A, and B bets. For each:
- Market + selection
- Model probability vs implied probability
- Edge % and EV
- Tier (S / A / B)
- Confidence note (what drives this edge)

### 2. Expected Goals
- λ_H, λ_A, λ_T with full derivation

### 3. Probability Model
- Over/Under, BTTS, Correct Score, AH, 1X2 — all markets with fair odds

### 4. Arbitrage Check
- Scan all provided odds for ArbitrageSum < 1.0
- If found, flag as Tier S and show optimal stake split

### 5. Cross-Market Consistency
- Flag any outcome confirmed by ≥ 3 independent markets or models

### 6. Player Prop Signals
- xG, xSOT, xKeyPasses, CRS for named players where data is available
- Flag any prop that aligns with team-level model (correlated value cluster)

### 7. Model Assumptions & Risk
- Key assumptions, data gaps, sample size warnings
- Confidence intervals on λ estimates

### 8. Responsible Betting Note

Use rigorous calculations throughout. Show all EV and edge figures. Do not include narrative predictions — only model-driven signals.
