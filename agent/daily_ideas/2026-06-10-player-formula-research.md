# Player Prop Formula Research — 2026-06-10

---

## Executive Summary

- **Dataset**: 30,083 player-match rows, 5,291 players, 38 columns. Four binary targets: target_sot (22.4%), target_cards (12.1%), target_goals (8.5%), target_assists (6.2%).
- **What we can implement now (zero new data required)**: 15 rolling features and 5 predict-time scoring functions built entirely from columns already present in player_history.parquet. Priority 1 features (shot_accuracy_rate, kp_per90, sot_quality_score, opp_adjusted_shot_threat, creative_playmaker_score) use confirmed correlations of r=0.903, r=0.742, and r=0.623 for SOT — the strongest pre-match signals in the dataset.
- **Critical data-leak fix required before any retraining**: yellow_cards has r=0.992 with target_cards because it records the same match. Drop it from all feature inputs immediately. cards_pg (rolling history) is the safe substitute at r=0.060.
- **What new data unlocks**: /players/statistics (xG/xA), /fixtures/events (set-piece delivery, fouls), /fixtures/statistics (inside-box shot share), and /standings (high-stakes flag) together cost 3,150 calls/day out of 7,200 available and would enable the remaining 5 composite scoring functions at materially higher correlation.

---

## Part 1: Features from Current Data (implement NOW)

All features follow the `shift(1).rolling(n, min_periods=1)` pattern — no lookahead leakage.
Add each name to `config.PLAYER_FEATURE_COLS` and retrain after implementing.

---

### 1. shot_accuracy_rate
**Market**: sot | **Priority**: 1

shot_accuracy_rate is the rolling ratio of shots_on_target to shots_total per match, averaged over the previous N matches. It is the single strongest non-leaked predictor of target_sot (r=0.903 in per-90 analysis). Using the rolling mean of the per-match ratio avoids denominator collapse from aggregating totals directly.

**build_features():**
```python
df["shot_accuracy_rate"] = grp.apply(
    lambda g: (g["shots_on_target"] / g["shots_total"].replace(0, np.nan))
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
shot_accuracy_rate = float(
    (phist["shots_on_target"] / phist["shots_total"].replace(0, np.nan))
    .fillna(0.0)
    .mean()
)
```

---

### 2. kp_per90
**Market**: assists | **Priority**: 1

Rolling key passes per 90 minutes. Normalises creative output by time played, removing the bias from players with variable minutes. kp_per90 is the top rolling predictor of target_assists (r=0.132 raw, r=0.311 when combined with rating as playmaker_score).

**build_features():**
```python
df["kp_per90"] = grp.apply(
    lambda g: (g["key_passes"] / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
kp_per90 = float(
    (phist["key_passes"] / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 3. starter_rate
**Market**: goals (multiplies all rate features) | **Priority**: 2

Rolling fraction of previous N matches in which the player started. Already computed in build_features() but not yet in PLAYER_FEATURE_COLS. High starter_rate means reliable full-game exposure, which directly multiplies all per-game rate features. Ready to activate — add to PLAYER_FEATURE_COLS and retrain.

**build_features():**
```python
df["starter_rate"] = grp["started"].transform(
    lambda x: x.shift(1).rolling(n, min_periods=1).mean()
)
```

**build_upcoming_features() / predict:**
```python
starter_rate = float(phist["started"].mean()) if "started" in phist.columns else 0.8
```

---

### 4. goal_involvement_rate
**Market**: goals | **Priority**: 2

Rolling (goals + assists) per 90 minutes. Captures players who alternate between scoring and assisting — wide forwards and second strikers who are undervalued by goals_pg or assists_pg alone.

**build_features():**
```python
df["goal_involvement_rate"] = grp.apply(
    lambda g: ((g["goals"] + g["assists"]) / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
goal_involvement_rate = float(
    ((phist["goals"] + phist["assists"]) / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 5. box_actions_per90
**Market**: sot | **Priority**: 2

Rolling (shots_total + duels_won) per 90 minutes. Proxies inside-box presence and physical contest wins. Directly addresses the missing inside-box shot share metric until /fixtures/statistics data is available.

**build_features():**
```python
df["box_actions_per90"] = grp.apply(
    lambda g: ((g["shots_total"] + g["duels_won"]) / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
box_actions_per90 = float(
    ((phist["shots_total"] + phist["duels_won"]) / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 6. shooting_efficiency_index
**Market**: goals | **Priority**: 2

Rolling goals per shot-on-target (goals_pg / sot_pg). Measures finishing quality — how often a player converts on-target shots. Clamped at 1.0 to handle small-sample noise. When multiplied by sot_quality_score at prediction time, it produces an end-to-end attack quality chain.

**build_features():**
```python
df["shooting_efficiency_index"] = (
    df["goals_pg"] / df["sot_pg"].replace(0, np.nan)
).fillna(0.0).clip(upper=1.0)
```

**build_upcoming_features() / predict:**
```python
shooting_efficiency_index = float(min(goals_pg / sot_pg if sot_pg > 0 else 0.0, 1.0))
```

---

### 7. fouls_drawn_per90
**Market**: assists | **Priority**: 3

Rolling fouls drawn per 90 minutes. Players who draw fouls in dangerous areas (advanced midfielders, pacey wingers) earn penalty-box touches and set-piece restart opportunities that create indirect assist chances.

**build_features():**
```python
df["fouls_drawn_per90"] = grp.apply(
    lambda g: (g["fouls_drawn"] / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
fouls_drawn_per90 = float(
    (phist["fouls_drawn"] / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 8. aerial_won_rate
**Market**: goals | **Priority**: 3

Rolling aerial duels won divided by total duels contested. Identifies aerial specialists (target strikers, set-piece header threats, defensive set-piece takers) whose goal contributions materialise from headers on corners and free kicks. Explains the pos_defender positive correlation with target_sot flagged in the analysis.

**build_features():**
```python
df["aerial_won_rate"] = grp.apply(
    lambda g: (g["duels_won"] / g["duels_total"].replace(0, np.nan))
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
aerial_won_rate = float(
    (phist["duels_won"] / phist["duels_total"].replace(0, np.nan))
    .fillna(0.0)
    .mean()
)
```

---

### 9. foul_committer_ratio
**Market**: cards | **Priority**: 3

Rolling fouls_committed / (fouls_committed + fouls_drawn + 0.01). A ratio close to 1.0 indicates a player who commits far more fouls than they draw — typical of defensive midfielders and full-backs in tracking roles. Better card-risk proxy than cards_pg alone when combined with non-forward position. Epsilon 0.01 prevents division by zero.

**build_features():**
```python
df["foul_committer_ratio"] = grp.apply(
    lambda g: (
        g["fouls_committed"] / (g["fouls_committed"] + g["fouls_drawn"] + 0.01)
    )
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
foul_committer_ratio = float(
    (phist["fouls_committed"] / (phist["fouls_committed"] + phist["fouls_drawn"] + 0.01))
    .mean()
)
```

---

### 10. fouls_per90
**Market**: cards | **Priority**: 4

Rolling fouls committed per 90 minutes. cards_pg measures actual bookings (rare, noisy, r=0.060). fouls_per90 measures the underlying behaviour that causes cards, providing a leading indicator less dependent on referee decisions.

**build_features():**
```python
df["fouls_per90"] = grp.apply(
    lambda g: (g["fouls_committed"] / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
fouls_per90 = float(
    (phist["fouls_committed"] / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 11. card_exposure_index
**Market**: cards | **Priority**: 4

cards_pg * (minutes_pg / 90.0) * (1 - pos_forward). Defensive and midfield players commit more bookable fouls in tracking roles; minutes_pg accounts for exposure time. Existing card_risk (cards_pg * (1 - pos_forward)) reached r=0.070; adding minutes_pg should push this further. Gate on n_prev_games >= 4 for stability.

**build_features():**
```python
df["card_exposure_index"] = (
    df["cards_pg"] * (df["minutes_pg"] / 90.0) * (1 - df["pos_forward"])
)
```

**build_upcoming_features() / predict:**
```python
card_exposure_index = round(cards_pg * (minutes_pg / 90.0) * (1 - pos_forward), 4)
```

---

### 12. duel_intensity_per90
**Market**: cards | **Priority**: 5

Rolling total duels contested per 90 minutes. High duel volume indicates a combative style common in defensive midfield and centre-back roles. Combined with foul_committer_ratio and non-forward position, this provides a three-dimensional card-risk signal: who fouls, who fouls often, and who contests many duels overall.

**build_features():**
```python
df["duel_intensity_per90"] = grp.apply(
    lambda g: (g["duels_total"] / g["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .shift(1)
    .rolling(n, min_periods=1)
    .mean()
).reset_index(level=0, drop=True)
```

**build_upcoming_features() / predict:**
```python
duel_intensity_per90 = float(
    (phist["duels_total"] / phist["minutes"].replace(0, np.nan) * 90)
    .fillna(0.0)
    .mean()
)
```

---

### 13–15. Composite features (computed from features above, no new source columns)

#### sot_quality_score
**Market**: sot | **Priority**: 1

shot_accuracy_rate * sot_pg. Quality (accuracy) combined with volume. The analysis found shot_accuracy r=0.903 for target_sot; multiplying by volume distinguishes high-accuracy/low-volume players from high-volume/moderate-accuracy players.

```python
# build_features():
df["sot_quality_score"] = df["shot_accuracy_rate"] * df["sot_pg"]

# predict:
sot_quality_score = round(shot_accuracy_rate * sot_pg, 4)
```

#### opp_adjusted_shot_threat
**Market**: sot | **Priority**: 1

shots_pg * opp_sot_conceded_pg. The interaction term confirmed at r=0.237 for target_sot and r=0.183 for target_goals, outperforming either component alone.

```python
# build_features():
df["opp_adjusted_shot_threat"] = df["shots_pg"] * df["opp_sot_conceded_pg"]

# predict:
opp_adjusted_shot_threat = round(shots_pg * opp_sot_conceded_pg, 4)
```

#### creative_playmaker_score
**Market**: assists | **Priority**: 1

kp_per90 * (pos_midfielder + 0.5 * pos_forward). Weights midfielders fully, forwards at half weight. mid_creativity version (kp_per90 * pos_midfielder only) reached r=0.106 for target_assists; broadening to include forwards improves recall on wide forwards and attacking midfielders.

```python
# build_features():
df["creative_playmaker_score"] = df["kp_per90"] * (df["pos_midfielder"] + 0.5 * df["pos_forward"])

# predict:
creative_playmaker_score = round(kp_per90 * (pos_midfielder + 0.5 * pos_forward), 4)
```

---

## Part 2: Composite Scoring Functions (predict-time)

These are computed in predict.py at prediction time. They are not training features — they layer on top of `_detect_lazy_factors()` and `compute_ges()`.

---

### CLINICAL_EFFICIENCY_SCORE
**Market**: goals | **Priority**: 2 | **Needs new data**: No

Combines a forward's shot accuracy with opponent defensive weakness, gated on pos_forward to exclude defenders who occasionally shoot. Replaces xg_form in compute_ges() at the same 0.40 weight.

**predict.py — replace compute_ges():**
```python
def compute_ges(row: dict, opp_weakness: float = 1.0, penalty_duty: bool = False) -> float:
    # Legacy xg_form component (kept for non-forwards)
    xg_form   = min(row.get("goals_pg", 0) / 0.50, 1.0)
    shot_vol  = min(row.get("shots_pg", 0) / 3.5, 1.0)
    pen       = 1.0 if penalty_duty else 0.0
    opp       = min(opp_weakness / 1.5, 1.0)
    min_sec   = min(row.get("minutes_pg", 0) / 85.0, 1.0)

    # New clinical efficiency component (replaces xg_form for forwards)
    sot_rate  = row.get("sot_rate", 0.0)       # = sot_pg / shots_pg
    is_fwd    = float(row.get("pos_forward", 0))
    clinical  = min(sot_rate * opp_weakness * is_fwd / (0.35 * 1.5), 1.0)
    eff_form  = clinical if is_fwd else xg_form

    return round(
        0.40 * eff_form
        + 0.20 * shot_vol
        + 0.15 * pen
        + 0.15 * opp
        + 0.10 * min_sec,
        3,
    )
# Example: forward with sot_rate=0.40, opp_weakness=1.8:
#   clinical = min(0.40 * 1.8 / 0.525, 1.0) = min(1.37, 1.0) = 1.0 -> full GES boost
```

---

### CREATIVE_OUTPUT_SCORE
**Market**: assists | **Priority**: 3 | **Needs new data**: No

kp_per90 * team_attack_strength * opp_defensive_weakness, with position weighting. Fires CREATIVE_MISMATCH lazy factor for elite no.10 profiles against weak defences.

**predict.py — add to _detect_lazy_factors():**
```python
if market == "assists":
    kp_90      = feat_row.get("kp_per90",
                     feat_row.get("key_passes_pg", 0.0) / max(feat_row.get("minutes_pg", 90) / 90, 0.1))
    team_atk   = feat_row.get("team_attack_str",
                     feat_row.get("team_goals_pg_roll", 1.3))   # fallback to existing field
    opp_def_wk = feat_row.get("opp_goals_conceded_pg", 1.3)
    pos_mid    = float(feat_row.get("pos_midfielder", 0))
    pos_fwd    = float(feat_row.get("pos_forward", 0))
    pos_weight = pos_mid + 0.5 * pos_fwd                        # mirrors creative_playmaker_score
    creative_score = kp_90 * team_atk * opp_def_wk * pos_weight
    row["creative_output_score"] = round(creative_score, 4)
    # Fire lazy factor: elite playmaker (kp_90 > 1.5) vs leaky defence (opp > 1.5)
    if kp_90 > 1.5 and opp_def_wk > 1.5 and pos_weight >= 0.5:
        factors.append("CREATIVE_MISMATCH")
# Tier wiring: CREATIVE_MISMATCH increments lazy_count, satisfying
# the lazy_count >= 1 requirement for MARKSMAN tier in _classify_tier().
```

---

### SET_PIECE_THREAT_SCORE
**Market**: sot | **Priority**: 1 | **Needs new data**: Yes (aerial_won_rate + /fixtures/events + /fixtures/statistics)

aerial_won_rate * team_corners_per90_opp * opp_sp_concession_rate. Targets defenders and wide midfielders whose entire SOT/goal threat derives from dead-ball situations — the group currently invisible in shot-count features.

**predict.py — add/replace SET_PIECE block in _detect_lazy_factors():**
```python
if market == "sot":
    aerial_rate = feat_row.get("aerial_won_rate", 0.0)
    corners_opp = feat_row.get("team_corners_per90_opp",
                      feat_row.get("team_corners_per90", 0.0))
    sp_concede  = feat_row.get("opp_sp_concession_rate", 0.30)   # league avg fallback
    pos_def     = feat_row.get("pos_defender", 0)
    pos_mid     = feat_row.get("pos_midfielder", 0)
    sp_score    = aerial_rate * corners_opp * sp_concede
    # Threshold: aerial rate >45%, team earning >5.5 corners, opp concedes >25% from SP
    if (pos_def or pos_mid) and sp_score > (0.45 * 5.5 * 0.25):
        factors.append("SET_PIECE")
    row["set_piece_threat_score"] = round(sp_score, 4)
```

**feature_engineering.py — build_features() composite (computed only after /fixtures/events data lands):**
```python
df["team_corners_per90_opp"] = df.get("team_corners_per90_opp",
                                      df.get("team_corners_per90", 5.0))
df["opp_sp_concession_rate"] = df.get("opp_sp_concession_rate", 0.30)  # league avg fallback
df["set_piece_threat_score"] = (
    df["aerial_won_rate"]
    * df["team_corners_per90_opp"].clip(upper=12)
    * df["opp_sp_concession_rate"].clip(upper=1.0)
)
```

---

### BOX_PRESENCE_SCORE
**Market**: sot | **Priority**: 2 | **Needs new data**: Yes (/fixtures/statistics for inside-box shots)

box_actions_per90 * team_corners_context * minutes_certainty. Combines open-play box threat with set-piece corner volume and playing-time confidence. Fires KEEPER_WEAK lazy factor.

**predict.py — add to _detect_lazy_factors() sot block:**
```python
    # BOX_PRESENCE sub-block (add after SET_PIECE sub-block above):
    box_per90   = feat_row.get("box_actions_per90",
                      feat_row.get("shots_pg", 0.0) * 0.6)    # degraded fallback: ~60% shots inside box
    corners_ctx = feat_row.get("team_corners_per90",
                      feat_row.get("team_corners_per90_opp", 5.0))
    starter_r   = feat_row.get("starter_rate", 0.8)
    minutes_est = float(feat_row.get("minutes_est", 75))
    min_cert    = min(starter_r * (minutes_est / 90.0), 1.0)
    box_score   = box_per90 * (corners_ctx / 6.0) * min_cert   # normalise corners to ~1.0 at league avg
    row["box_presence_score"] = round(box_score, 4)
    opp_sot_c   = feat_row.get("opp_sot_conceded_pg", 4.5)
    if box_score > 1.0 and opp_sot_c > 5.5:
        if "KEEPER_WEAK" not in factors:
            factors.append("KEEPER_WEAK")
# Optional GES extension — replace 0.15*pen with 0.10*pen + 0.05*box_comp:
#   box_comp = min(row.get("box_presence_score", 0) / 2.0, 1.0)
```

---

### CARD_RISK_SCORE
**Market**: cards | **Priority**: 4 | **Needs new data**: Partial (fouls_committed from /fixtures/events; standings for high_stakes_match)

foul_committer_ratio * referee_strictness * high_stakes_match. Requires n_prev_games >= 4 for stability. Until foul data is available, the degraded form uses cards_pg / 0.15 as a foul-ratio proxy.

**predict.py — replace cards block in _detect_lazy_factors():**
```python
if market == "cards":
    ref_strict  = feat_row.get("referee_strictness", 0.0)
    foul_ratio  = feat_row.get("foul_committer_ratio",
                      feat_row.get("cards_pg", 0.0) / 0.15)    # degraded fallback
    high_stakes = float(feat_row.get("high_stakes_match", 0.0))
    is_non_fwd  = 1.0 - float(feat_row.get("pos_forward", 0))
    n_prev      = int(feat_row.get("n_games", 0))
    if n_prev >= 4:
        card_score = foul_ratio * ref_strict * max(high_stakes, 0.5) * is_non_fwd
        row["card_risk_score"] = round(card_score, 4)
        if ref_strict > 0.5 and card_score > 0.20:
            factors.append("REFEREE")
            factors.append("HIGH_PRESSURE")
# Tier wiring: CARD market uses standard _classify_tier() without GES gate.
# card_risk_score is exposed in output row for downstream filtering.
```

---

## Part 3: API Collection Plan (7,200 unused calls/day available)

| Endpoint | Calls/day | Formulas enabled | Cache TTL | Priority |
|---|---|---|---|---|
| /players/statistics | 400 | xG_per90, xA_per90, aerial_won_rate (season level) — top feature for goals/SOT once available | 3 days | 1 |
| /fixtures/events | 1,200 | opp_sp_concession_rate (SET_PIECE_THREAT), fouls_committed per player (foul_committer_ratio, CARD_RISK), corner delivery counts, free-kick disambiguation for creative_output | 7 days completed / 0 live | 1 |
| /fixtures/statistics | 1,500 | box_actions_per90 (BOX_PRESENCE_SCORE), inside-box shot share, team_corners_per90_opp (SET_PIECE_THREAT) | 7 days | 2 |
| /fixtures/lineups | 800 | confirmed_starter flag (eliminates starter_rate uncertainty), formation_role encoding (lone vs twin striker), penalty_taker_flag cross-referenced with /fixtures/events | 6 hours | 3 |
| /standings | 50 | high_stakes_match flag (top-half vs bottom-half for CARD_RISK_SCORE), team_attack_strength league-adjusted index (removes fixture-schedule collinearity in team_goals_pg_roll) | 24 hours | 4 |
| **Total allocated** | **3,950** | | | |
| **Remaining available** | **3,250** | Reserved for existing team model endpoints | | |

**Backfill note**: /fixtures/events and /fixtures/statistics require historical backfill of ~6 months across 5 leagues before rolling features stabilise. Run the backfill collector nightly at off-peak hours; do not block same-day prediction runs. Budget 1,200 calls/day during the 4-month backfill window, dropping to ~550 calls/day ongoing.

---

## Part 4: Implementation Order

1. **Purge yellow_cards from all training pipelines.** Verify no column named yellow_cards, yellow_card, or similar appears in any feature input. r=0.992 data leak — this corrupts any model that has ever trained with it. Retrain immediately after purge.

2. **Add priority-1 features to PLAYER_FEATURE_COLS and retrain** (no new data required, highest confirmed correlations):
   - shot_accuracy_rate (r=0.903 for target_sot)
   - kp_per90 (r=0.132 for target_assists; r=0.311 as playmaker_score composite)
   - sot_quality_score (shot_accuracy_rate * sot_pg)
   - opp_adjusted_shot_threat (shots_pg * opp_sot_conceded_pg, r=0.237 target_sot)
   - creative_playmaker_score (kp_per90 * position weights)

3. **Deploy CLINICAL_EFFICIENCY_SCORE in compute_ges()** (no new data, replaces weaker xg_form for forwards). Wire immediately — any forward with sot_rate=0.40 facing a leaky defence saturates the GES forward component to 1.0.

4. **Deploy CREATIVE_OUTPUT_SCORE and CREATIVE_MISMATCH lazy factor** in _detect_lazy_factors() for market='assists'. Uses kp_per90 already available in feat_row after step 2.

5. **Add priority-2 features** (no new data, adds normalised-by-minutes versions and activates existing columns):
   - starter_rate (already computed, add to PLAYER_FEATURE_COLS)
   - goal_involvement_rate
   - box_actions_per90 (degraded form using shots_total + duels_won)
   - shooting_efficiency_index
   - Retrain.

6. **Wire /players/statistics API collector** (400 calls/day). Populates xG_per90 and xA_per90 for ~2,000 active players. Expected to become the top goals/SOT feature once available, replacing or augmenting shots_pg.

7. **Wire /fixtures/events API collector** (1,200 calls/day). Enables foul_committer_ratio (CARD_RISK_SCORE), opp_sp_concession_rate (SET_PIECE_THREAT_SCORE), and corner delivery counts. Begin historical backfill.

8. **Wire /fixtures/statistics API collector** (1,500 calls/day). Enables inside-box shot share and precise team_corners_per90_opp. Retire box_actions_per90 degraded approximation; retrain with actual box data.

9. **Add priority-3 to 5 card-market features** (fouls_per90, foul_committer_ratio from /fixtures/events, duel_intensity_per90). Cards market is near-random until this data lands; do not raise bet volume on cards before step 7 completes.

10. **Wire /fixtures/lineups collector** (800 calls/day). Retire starter_rate rolling proxy with confirmed_starter flag. Enables formation_role encoding and penalty_taker_flag auto-detection.

---

## Part 5: Expected Impact by Market

### SOT (shots on target) — Highest exploitability

**Highest-impact feature**: shot_accuracy_rate (r=0.903 for target_sot). No other non-leaked feature comes close. Activating it alone, combined into sot_quality_score with sot_pg, is likely the single highest-ROI code change in this entire plan.

The SOT market is the most exploitable because: (a) pre-match signal is unusually strong compared to other markets, (b) bookmaker SOT lines are priced at short odds (1.5-2.5) where the overround is only 6% and the relative edge floor is 12% — a realistic threshold given the correlation magnitudes. CLINICAL_EFFICIENCY_SCORE for forwards and SET_PIECE_THREAT_SCORE for defenders/midfielders together cover the two distinct SOT archetypes (open-play shooters vs dead-ball aerial threats) that current shot-count features conflate.

**Next unlock**: /fixtures/statistics for inside-box shot share — separates clinical penalty-box finishers from long-range speculators without requiring xG.

### Goals (goals anytime) — Second in exploitability

**Highest-impact feature**: sot_vs_opp composite (sot_per90 * opp_goals_conceded_pg, r=0.417 for target_goals). Already partially implemented via opp_adjusted_shot_threat; activating the per-90 normalisation (sot_per90 rather than sot_pg) is the key upgrade. The Striker Shot Volume Gate (forwards at or above median shots_pg scoring goals at 0.213 vs 0.123, a 73% uplift) is a powerful hard filter to drop low-shot forwards from goals-anytime consideration entirely.

Goals-anytime lines sit at longer odds (3.0-6.0) where overround is 10-15% and the relative edge floor rises to 20-30%. SNIPER tier requires GES >= 0.70 — achievable for a high-accuracy forward vs a leaky defence after the CLINICAL_EFFICIENCY_SCORE upgrade to compute_ges().

**Next unlock**: /players/statistics for xG_per90 — shot location weighting rather than shot counts is expected to be the top goals feature once available.

### Assists — Moderate exploitability

**Highest-impact feature**: playmaker_score (kp_per90 * rating, r=0.311 for target_assists). kp_per90 is already available from Part 1 step 2. The creative_playmaker_score and CREATIVE_OUTPUT_SCORE with CREATIVE_MISMATCH lazy factor provide a structured path to MARKSMAN-tier signals for elite no.10 profiles.

The assists market is thin at most bookmakers outside the Premier League and Champions League. Focus CREATIVE_MISMATCH on PL/UCL fixtures where assist markets have depth and the overround is lower. creative_vs_weak (kp_per90 * opp_goals_conceded_pg, r=0.240) should be added as a secondary ranking score alongside playmaker_score.

**Next unlock**: /fixtures/events for free-kick disambiguation — separates set-piece assists (cross-dependent) from open-play key passes to avoid mixing two structurally different assist types in a single feature.

### Cards — Currently unexploitable

**Highest-impact feature**: cards_pg (r=0.060). This is the best non-leaked predictor available and it is near zero. The entire cards market is near-random with current data.

Do not increase bet volume or loosen tier thresholds for the cards market until /fixtures/events data provides fouls_committed per player per match. At that point, foul_committer_ratio combined with referee_strictness and high_stakes_match (from /standings) should push predictive correlation materially above the current 0.060 baseline. Until then, CARD_RISK_SCORE in its degraded form (cards_pg / 0.15 proxy) provides marginal improvement over the current card_risk feature and should be treated as a conservative filter only, not a positive selection signal.

**Critical prerequisite for all markets**: confirm yellow_cards is absent from every feature input before any retraining. This single action is higher priority than any new feature.
