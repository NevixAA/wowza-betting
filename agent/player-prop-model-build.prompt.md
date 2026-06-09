---
name: player-prop-model-build
displayName: Player Prop Model Build Prompt
description: "Prompt template for designing and validating a player property model using the Player Prop Analyst agent."
---

# Player Prop Model Build Prompt

Use this prompt with the `player-prop-analyst` agent.
The goal is to design, code, and validate a production player prop model for a specific market (shots on target, yellow cards, goals, or set-piece assists).

## Instructions

Provide as much detail as possible about your data, rules, and requirements. Required minimum: target market, settlement rules, and available features.

- Target prop market (SOT, yellow cards, goals, set-piece assists)
- Bookmaker settlement rules and voidance conditions
- Data provider (Opta, StatsBomb, mixed)
- Available features and data schema
- Historical data sample (if available)
- Lineup certainty state (pre- or post-lineup)
- Preferred model family or constraints
- Backtest sample size and time range

## Prompt

Design and validate a production player prop model for the following market:

**Market:**
- Target prop: `{{target_prop}}`
- Half-line(s): `{{half_lines}}`
- Settlement rules: `{{settlement_rules}}`
- Voidance rules: `{{voidance_rules}}`

**Data:**
- Provider: `{{data_provider}}`
- Available events/stats: `{{available_features}}`
- Lineup state: `{{lineup_state}}`
- Historical sample: `{{sample_description}}`

**Requirements:**
- Model family preference: `{{model_preference}}`
- Backtest period: `{{backtest_period}}`
- Minimum sample constraints: `{{sample_constraints}}`

---

**Deliverables:**

1. **Data Foundation & Definitions**
   - Confirm settlement rule match between bookmaker and data provider
   - State exact label definitions (e.g., Opta SOT definition)
   - Identify any definition gaps or assumptions

2. **Feature Engineering**
   - List the primary feature block for this market
   - Rank features by signal strength and availability
   - Flag any missing features that would materially improve edge

3. **Model Design**
   - Specify the recommended count model family
   - Detail the state model (start/sub/dnp) structure
   - Show the exposure/offset term for minutes
   - Provide the exact probability derivation for each half-line

4. **Calibration & De-Vigging**
   - Sketch the calibration approach (e.g., isotonic regression, Platt scaling)
   - Show the no-vig de-vigging formula for your market
   - Explain how to compare against market probabilities

5. **Backtest Plan**
   - Time-based validation scheme (rolling-origin or expanding-window)
   - Brier score, log-loss, and calibration-curve targets
   - CLV measurement method
   - Segmentation for results (by market, line, competition, lineup state)

6. **Python Implementation Sketch**
   - Pseudocode or template for the count model fit
   - Code for state mixture and probability derivation
   - Edge and EV calculation
   - Time-based backtest skeleton

7. **Risk & Validation**
   - Key failure modes for this market
   - Data quality checks and outlier flagging
   - Sample-size sufficiency given expected event frequency
   - Provider-rule mismatch risks

8. **Next Steps**
   - Implementation priority (rank among other props)
   - Data collection and feature engineering tasks
   - Model fit and validation milestones

Use rigorous probability and calibration logic. Show all formulas and assumptions. Do not provide narrative analysis — only model design and validation structure.
