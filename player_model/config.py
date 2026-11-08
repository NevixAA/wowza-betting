"""Player Model Configuration — v2 (3-tier SNIPER/MARKSMAN/VALUABLE system)."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_FILE = BASE_DIR / "player_stats_cache.json"

# ── API-Football (same RapidAPI key as main system) ───────────────────────────
API_HOST   = os.getenv("API_HOST",   "api-football-v1.p.rapidapi.com")
API_KEY    = os.getenv("API_KEY",    "")
# DEPRECATED (R2 finding 6, 2026-11-01): was os.getenv("API_SEASON", "2025"). Nothing reads it; props
# seasons come from player_model/data_fetcher (_current_season + api_current_season) and the root
# src/season_resolver. The sentinel makes any forgotten consumer fail visibly, not quietly.
API_SEASON = "SEASON_UNKNOWN"

# ── Markets ───────────────────────────────────────────────────────────────────
MARKETS    = ["goals", "goals2", "assists", "sot", "sot2", "sot3", "sot4", "cards"]

# sot4 (4+ shots on target) ADDED 2026-08-27, on the same footing as sot3 and goals2: RESEARCH
# DATA, never a tip source. The base rate makes that unavoidable rather than a policy choice --
# measured over 302,456 player-matches in player_history.parquet:
#
#     1+ SOT  22.48%   (starters 25.51%)
#     2+ SOT   5.24%   (starters  6.57%)
#     3+ SOT   1.21%   (starters  1.61%)
#     4+ SOT   0.27%   (starters  0.38%)   <- sot4
#
# MIN_SIGNAL_PROB is 0.15, so emitting a sot4 tip would need a ~40x lift over the base rate on
# a calibrated model. It will not clear the floor, and it is in VALUABLE_ONLY_MARKETS so it
# could not be staked even if it did. What it IS for: bookmakers price 4+ (8 quotes on the probe
# fixture) and those prices arrive on the SAME `player_shots_on_target` response as sot/sot2/sot3
# at no extra API cost, and a prop price is pre-match only so an uncaptured one is gone forever.
# Capturing the ladder complete costs nothing and cannot be done retroactively.

# goals2 (2+ goals / brace) is MODELLED but no longer requested from OddsAPI (2026-08-18).
# The two sides of it fail for different reasons and deserve different treatment:
#   * ODDS: OddsAPI rejects `player_to_score_2_or_more` as an invalid market on every soccer
#     event — 0 prices in the entire history — and the fetcher's self-heal then dropped it
#     and re-requested, costing an extra API call per event. Removed from odds_fetcher.
#   * MODEL: it stays in MARKETS because the FANTASY family consumes it as "2+ goals
#     (brace)" (fantasy_features.market_leaderboards) and needs no bookmaker price at all.
#     Invariant 2: prop accuracy is monetised through no-vig Fantasy, not through betting.
# It could never emit anything before today regardless, because its cap was computed at
# 0.0878 — below MIN_SIGNAL_PROB. That was a cap bug, now fixed (see _compute_market_caps).

# Below this, a prediction is not emitted at all. Named rather than inlined because it
# interacts with the per-market caps in predict._compute_market_caps: when a market's cap
# falls BELOW this floor the market silently cannot produce anything, which is exactly how
# sot3 (cap 0.1288) and goals2 (cap 0.0878) became dead code unnoticed. predict.py now logs
# that condition loudly at startup.
#
# sot3 is deliberately LEFT in MARKETS despite being unable to clear the floor today: it is
# in VALUABLE_ONLY_MARKETS (collection, never staked), bookmakers do price it (311 prices in
# 5 days), and the odds cost nothing extra because they arrive on the same
# `player_shots_on_target` response as sot and sot2. It stays as research data. Lowering the
# floor to chase it would push tips into exactly the longshot band where prop ROI was measured
# at -41% to -57%.
MIN_SIGNAL_PROB = 0.15

# ── Owner serving rules for props tips (2026-11-08) ──────────────────────────
# "No GK attacking tips": a goalkeeper (position 'G') is never tipped on shots on target, goals or
# assists. Cards stay allowed. Evidence: on the props clean-retrain holdout (2026-09-01..10-31) the
# 2,992 goalkeeper rows had a real SOT and goal rate of 0%, while the model gave 6.5% / 3.2%.
GK_BLOCKED_MARKETS = frozenset({"sot", "sot2", "sot3", "sot4", "goals", "goals2", "goals3", "assists"})
#: "Retrain them too": these markets still serve models trained on the placeholder history
#: (age 25 / 180 cm / 75 kg). Until the props clean retrain covers them they make NO tip and are
#: not on the board. The model files stay where they are.
HIDDEN_PENDING_CLEAN_RETRAIN = frozenset({"goals2", "goals3", "sot3", "sot4"})
#: reason codes, counted and printed by predict.generate_tips (a skipped tip is never silent)
SKIP_GK_ATTACKING = "GK_NO_ATTACKING_TIP"
SKIP_HIDDEN_MARKET = "MARKET_HIDDEN_PENDING_CLEAN_RETRAIN"
#: a tip whose player position is unknown is NOT guessed: it is kept and labelled
POSITION_UNKNOWN = "UNKNOWN"

# World Cup cards: features are imputed → predictions inflate to ~0.40 for everyone.
# Only emit WC card tips for players whose REAL booking rate (max of cards_pg /
# season_cards_pg) is at least this — for them, ~0.40 is roughly their true rate.
# (Club-league cards are unaffected; the model is calibrated there.)
WC_CARD_MIN_RATE = 0.35

# Starter proxy: when a confirmed lineup isn't available yet, only tip players who
# average at least this many minutes/game (filters fringe/rotation players who may
# not start). Confirmed-lineup filtering still takes precedence when available.
MIN_STARTER_MINUTES = 60.0

# A player outside every live squad list is only tipped if he appeared within this many days.
# Covers clubs we hold no squad list for; guards against players who moved to a league we do
# not collect and stay attached to their last collected club (Benzema at Real Madrid, 2026-10-08).
STALE_DAYS = 120

# World Cup non-card props use imputed national-team features (live ROI ~ -33%) →
# flag low-confidence and discount their confidence so they rank below club signals.
WC_PROP_CONF_PENALTY = 0.5
MODEL_FILES = {
    "goals":   MODELS_DIR / "model_player_goals.pkl",
    "goals2":  MODELS_DIR / "model_player_goals2.pkl",
    "assists": MODELS_DIR / "model_player_assists.pkl",
    "sot":     MODELS_DIR / "model_player_sot.pkl",
    "sot2":    MODELS_DIR / "model_player_sot2.pkl",
    "sot3":    MODELS_DIR / "model_player_sot3.pkl",
    "sot4":    MODELS_DIR / "model_player_sot4.pkl",
    "cards":   MODELS_DIR / "model_player_cards.pkl",
}
# Market → target column in training data
MARKET_TARGETS = {
    "goals":   "target_goals",
    "goals2":  "target_goals2",
    "assists": "target_assists",
    "sot":     "target_sot",
    "sot2":    "target_sot2",
    "sot3":    "target_sot3",
    "sot4":    "target_sot4",
    "cards":   "target_cards",
}
# Human-readable market labels for Telegram
MARKET_LABELS = {
    "goals":   "Anytime Scorer",
    "goals2":  "Score 2+",
    "assists": "Assist",
    "sot":     "SOT 1+",
    "sot2":    "SOT 2+",
    "sot3":    "SOT 3+",
    "sot4":    "SOT 4+",
    "cards":   "Carded",
}

# ── Signal tiers ──────────────────────────────────────────────────────────────
SNIPER_EV   = 0.40
MARKSMAN_EV = 0.25
VALUABLE_EV = 0.15
RELATIVE_EDGE_FLOORS  = {6.0: 0.30, 4.0: 0.20, 3.0: 0.12}
OVERROUND_BY_ODDS     = {3.0: 1.06, 5.0: 1.10, 99.0: 1.15}
GES_SNIPER_MIN        = 0.70
GES_MARKSMAN_MIN      = 0.50
GES_SUPPRESS          = 0.35
KELLY_FRACTION        = 0.20
MAX_STAKE_SNIPER      = 0.03
MAX_STAKE_MARKSMAN    = 0.02
MAX_STAKE_VALUABLE    = 0.01
CONFIDENCE_FLOORS     = {"SNIPER": 0.72, "MARKSMAN": 0.62, "VALUABLE": 0.50}
# OFF (owner decision 2026-11-01, Data Truth finding 52). These floors were tuned on a score that
# was 45% constant (+0.40 on every row) and they filtered nothing: tiers are identical with and
# without them. The honest score carries no ranking signal yet (n_games is the 5-game window on
# every row; lazy factors are 7% of rows). Turn ON only after floors are fitted on settled props.
PROPS_CONFIDENCE_GATE = False

# Relative edge gates — replace hard market_odds floors (5.0 / 4.0) with model-derived edge.
# rel_edge = (model_prob - fair_prob) / fair_prob  — odds-level-agnostic measure.
REL_EDGE_SNIPER   = 0.20   # ≥20% relative edge required for SNIPER
REL_EDGE_MARKSMAN = 0.12   # ≥12% relative edge required for MARKSMAN

# ── Role × market gate (real-odds backtest, Jun 2026) ─────────────────────────
# Combos that were clear money-losers vs REAL odds get capped at VALUABLE
# (data-only — tracked in the ledger, never staked as SNIPER/MARKSMAN, not sent
# as tips). WATCH tier removed; non-qualifying signals are AVOID.
#   - sot2/sot3 (multiple SOT) & goals2: longshot markets, negative across all roles
#   - defenders for anytime-goalscorer: defenders ~never score (≈0-3% hit rate)
VALUABLE_ONLY_MARKETS     = {"sot2", "sot3", "sot4", "goals2", "cards"}
VALUABLE_ONLY_ROLE_MARKET = {("D", "goals")}
# PAPER feed: send the top-N strongest props picks per run as tracking-only signals
# (no proven edge — validation 2026-07-09 — never real money). 0 disables.
PROPS_PAPER_FEED_N        = 5
# cards: the model outputs a near-constant ~0.378 for every player (referee features are
# leaked in training + mis-scaled at predict; card_clash_index reads an unpopulated key), so
# card "edges" are degenerate. Capped at VALUABLE (tracked, never a real SNIPER/MARKSMAN tip)
# until the referee-feature train/serve skew is fixed + the model actually discriminates.

# ── Features ──────────────────────────────────────────────────────────────────
ROLLING_N = 5
PLAYER_FEATURE_COLS = [
    # Base rolling stats (last-N)
    "goals_pg", "assists_pg", "shots_pg", "sot_pg",
    "cards_pg", "minutes_pg", "key_passes_pg", "sot_rate",
    "starter_rate", "n_prev_games", "rating_pg",
    # Career-to-date stable priors (expanding average over full history)
    "career_goals_pg", "career_sot_pg", "career_shots_pg", "career_assists_pg",
    # Phase 1: accuracy and per-90 rolling features (all computed from parquet)
    "shot_accuracy_rate",        # rolling SOT/shots — r=0.903 for target_sot
    "kp_per90",                  # rolling key_passes/90 — top assists predictor
    "goal_involvement_rate",     # rolling (goals+assists)/90
    "shooting_efficiency_index", # goals_pg / sot_pg, capped 1.0
    "box_actions_per90",         # rolling (shots+duels_won)/90
    "aerial_won_rate",           # rolling duels_won/duels_total
    "duel_intensity_per90",      # rolling duels_total/90
    "fouls_drawn_per90",         # rolling fouls_drawn/90
    "fouls_per90",               # rolling fouls_committed/90
    "foul_committer_ratio",      # rolling committed/(committed+drawn+eps)
    "card_exposure_index",       # cards_pg * (minutes_pg/90) * (1-pos_forward)
    # Composite scoring features
    "sot_quality_score",         # shot_accuracy_rate * sot_pg — r=0.903 combined
    "opp_adjusted_shot_threat",  # shots_pg * opp_sot_conceded_pg — r=0.237
    "creative_playmaker_score",  # kp_per90 * position weight — r=0.311
    "team_corners_pg",           # rolling corners earned per match
    # Match context
    "is_home",
    "opp_goals_conceded_pg", "opp_sot_conceded_pg",
    "team_goals_pg_roll",
    # Opponent defender matchup features (Phase 2)
    "opp_def_aerial_win_rate",    # opposing CBs' aerial duel win rate (low = weak in air)
    "opp_def_fouls_pg",           # fouls committed per game by opposing defense
    "opp_def_cards_pg",           # cards per game by opposing defense
    "aerial_matchup_score",       # player aerial_won_rate × (1 - opp_def_aerial_win_rate)
    "foul_draw_matchup_score",    # player fouls_drawn_per90 × opp_def_fouls_pg
    "opp_def_aggression",         # opp_def_fouls_pg × (1 + opp_def_cards_pg)
    # Position encoding
    "pos_forward", "pos_midfielder", "pos_defender",
    # Playing time / match context
    "rest_days",
    # Venue-split rates — home vs away performances differ significantly
    "goals_home_pg", "goals_away_pg",
    "shots_home_pg", "shots_away_pg",
    "sot_home_pg",   "sot_away_pg",
    # Extended raw rolling stats (tackles, dribbles, discipline, passing)
    "tackles_pg", "interceptions_pg", "defensive_actions_per90",
    "dribbles_pg", "dribble_success_rate", "dribbled_past_pg",
    "red_cards_pg", "passes_pg", "offsides_pg",
    "penalties_won_pg", "penalty_conversion_rate",
    # Opponent GK matchup
    "opp_gk_save_rate", "opp_gk_saves_pg",
    # Attacker vs GK composite features (agent-designed)
    "att_vs_gk_threat", "clinical_vs_gk",
    "volume_shot_penetration", "finishing_threat_index",
    "box_dominance_vs_gk", "aggression_adjusted_threat",
    # Positional composite features
    "dribble_creativity_score", "defensive_solidity",
    "penalty_threat_score", "card_risk_index",
    "midfield_engine_score", "offside_aggressiveness",
    # Agent-designed defender/midfielder composites
    "defensive_vulnerability_index", "progressive_carrier_score",
    "disciplinary_pressure_index", "foul_magnet_score",
    # Season-level priors (/players/statistics — more stable than last-5 rolling)
    "season_goals_pg", "season_assists_pg", "season_shots_pg",
    "season_sot_pg", "season_cards_pg", "season_minutes_pg",
    "season_appearances",
    # Physical profile (from /players profile endpoint)
    "age",
    "age_peak_delta",         # abs(age - 27) — distance from peak performance age
    "height_cm",
    "height_aerial_interaction",  # height × aerial_won_rate composite
    # Extended season stats (from /players/statistics, zero extra API calls)
    "season_start_rate",      # lineups / appearances — stable role signal
    "season_pass_accuracy",   # passing quality (0-1)
    "season_dribble_pg",      # dribble success per game (season-level)
    "season_fouls_pg",        # fouls committed per game (season-level prior for cards)
    "season_fouls_drawn_pg",  # fouls drawn per game (season-level)
    # Injury context
    "days_since_last_injury", # days since last sidelined event (365 = healthy)
    "return_from_injury_flag",# binary: first 3 games back from injury
    "chronic_injury_risk",    # number of sidelined events in past 12 months
    # Referee strictness
    "referee_yellows_pg",     # referee's historical yellow cards per game
    "referee_strictness",     # normalized 0-1 referee strictness score
    # Position-split opponent concede stats (Phase 4)
    "opp_goals_conceded_vs_fwd_pg",  # goals scored by forwards against this team
    "opp_sot_conceded_vs_fwd_pg",    # SOT by forwards against this team
    "opp_goals_conceded_vs_mid_pg",  # goals scored by midfielders against this team
    "opp_sot_conceded_vs_mid_pg",    # SOT by midfielders against this team
    "forward_matchup_score",         # sot_pg × opp_goals_conceded_vs_fwd × pos_forward
    "mid_threat_vs_defense",         # kp_per90 × opp_goals_conceded_vs_mid × pos_midfielder
    # Additional player-vs-player matchup formulas (Phase 3)
    "box_threat_vs_leaky_defense",    # box_actions_per90 × opp_goals_conceded_pg
    "efficiency_vs_leaky_keeper",     # shooting_efficiency_index × opp_sot_conceded_pg
    "kp_vs_aggressive_defense",       # kp_per90 × opp_def_cards_pg
    "team_momentum_forward_matchup",  # team_goals_pg_roll × opp_goals_conceded_pg × pos_forward
    "creative_pressure_matchup",      # creative_playmaker_score × opp_def_fouls_pg
    "dribbler_vs_defensive_line",     # dribble_creativity_score × (1 - opp_def_aerial_win_rate)
    "carrier_vs_press",               # progressive_carrier_score × opp_def_aggression
    # Card-market features (Phase 5)
    "opp_mid_fouls_pg",              # opponent midfielders fouls per game (card risk environment)
    "opp_mid_cards_pg",              # opponent midfielders cards per game
    "dribble_contact_rate",          # dribbles_pg × (1 - dribble_success_rate)
    "tackle_dribble_clash",          # dribble_contact_rate × opp_mid_fouls_pg
    "card_clash_index",              # (fouls_per90 + dribble_contact_rate) × opp_mid_cards_pg × referee_strictness
    "opp_mid_discipline",            # opp_mid_fouls_pg×0.6 + opp_mid_cards_pg×2.0
    # ── Goal-type features (from /fixtures/events goal detail — header / free kick) ──
    "headed_goals_pg",           # rolling header goals per game
    "fk_goals_pg",               # rolling direct free kick goals per game
    "headed_goals_share",        # headed_goals_pg / goals_pg
    # (Set-piece features: the ONLY real ones are the Understat us_sp_* columns; every typed
    #  set-piece proxy was purged 2026-11-02, see SETPIECE_PROXIES_REMOVED below.)
    # ── Height matchup vs opponent defenders ──────────────────────────────────
    "opp_def_mean_height",       # mean height of opponent starting defenders (cm)
    "height_diff_vs_opp_def",    # player height minus opp mean CB height
    "height_advantage_score",    # normalized: +1 = 10cm taller, clipped [-2,+2]
    # ── Opposition quality adjustment (fixes weak-league inflation) ───────────
    "opp_strength_index",      # rolling: opp goals conceded relative to league avg
    "quality_adj_goals_pg",    # goals_pg * (1 / opp_strength_index), league-scaled
    "quality_adj_sot_pg",      # sot_pg * (1 / opp_strength_index), league-scaled
    # ── League quality and player vs player strength ─────────────────────
    "league_tier",               # tier 1-4 of league this match is played in
    "league_quality",            # 0.30/0.50/0.75/1.0 quality score
    "player_career_avg_quality", # career rolling average league quality
    # Opponent defensive player individual quality
    "opp_def_player_rating_pg",  # rolling avg rating of opponent CBs
    "opp_top_def_rating",        # rolling avg best CB rating in opponent defence
    "opp_def_player_quality",    # composite: aerial × rating × top_def normalized
    # Quality mismatch — key Panama fix
    "context_quality_discount",  # player career quality / opp def quality
    "quality_mismatch_goals",    # goals_pg * context_quality_discount
    "quality_mismatch_sot",      # sot_pg * context_quality_discount
    # ── Previous season + longer-window shot accuracy (owner-approved 2026-11-02, fix6/props6) ──
    # Real values over the player's / team's PREVIOUS season (own names; never fill a season_*
    # value), NaN below MIN_PREV_SEASON_MATCHES; shot_accuracy_l10 = sum SOT / sum shots over the
    # last SHOT_ACC_WINDOW matches, NaN below MIN_SHOTS_FOR_ACCURACY shots. Absent from the stored
    # player_history until it is rebuilt; the props imputer drops an all-missing column at fit.
    "prev_season_goals_pg", "prev_season_assists_pg", "prev_season_shots_pg", "prev_season_sot_pg",
    "prev_season_cards_pg", "prev_season_minutes_pg", "prev_season_start_rate",
    "prev_season_appearances", "prev_season_shot_accuracy",
    "prev_season_team_goals_pg", "prev_season_team_conceded_pg",
    "prev_season_opp_goals_pg", "prev_season_opp_conceded_pg",
    "shot_accuracy_l10",
]

# Pruned 2026-06-30 (local): set-piece / goal-type features that can't be reliably populated —
# they need header/set-piece goal data that no accessible source provides for our prop leagues
# (API-Football goal events = "Normal Goal" only; Opta/StatsBomb-commercial = enterprise;
# StatsBomb-open = historical seasons only; FBref = no clean headed-goals). All were 0-0.5%
# nonzero (dead weight/noise). Removed so models train on RELIABLE features only.
_DEAD_GOAL_TYPE_FEATURES = {"headed_goals_pg", "fk_goals_pg", "headed_goals_share"}
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in _DEAD_GOAL_TYPE_FEATURES]

# PURGED 2026-11-02 (owner rule "no fake data, no proxies"; fix6/props6). These were named as
# set-piece data but built from aerial-duel / corner / height numbers with typed rates (x 0.30 "sp
# rate", x 5.0 "SOT proxy") and clip bounds, or from `sp_goal` (= ANY header or free-kick goal).
# They are no longer built for training or serving (feature_engineering.REMOVED_SETPIECE_FEATURES)
# and can never re-enter this list. The legacy serving payloads that list five of them read them as
# absent -> NaN, which is what they already read (all-NaN columns since 2026-11-01).
SETPIECE_PROXIES_REMOVED = frozenset({
    "sp_taker_score", "sp_receiver_score", "set_piece_threat_score", "set_piece_corner_matchup",
    "defender_sot_role_index", "defender_sp_edge", "defender_sot_edge",
    "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge", "aerial_height_sp_composite",
    "sp_goal_probability_composite", "opp_sp_goals_conceded_pg", "sp_goals_pg", "sp_assist_pg",
    "career_sp_goals_rate", "career_sp_assist_rate", "sp_goals_share",
})
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in SETPIECE_PROXIES_REMOVED]

# WITHHELD FROM TRAINING 2026-11-02 (fix6/props6): hand formulas whose VALUE is shaped by typed
# weights / normalisers / smoothing constants, not by a measurement (e.g. penalty_threat_score =
# penalties_won_pg * 3.0 + fouls_drawn_per90 * 0.08, capped at 1.0; midfield_engine_score =
# passes/50 * 0.4 + kp/3 * 0.4 + tackles/5 * 0.2). Their inputs are real and stay features under
# their own names. The serving payloads persist their own feature_cols, so serving keeps computing
# these (feature_engineering._add_composites, unchanged) until the next owner-approved retrain,
# which then trains without them. A feature whose only constant is a unit conversion (/90) or a
# position mask, or a product / ratio of measured rates, is NOT here.
TYPED_FORMULA_FEATURES_WITHHELD = frozenset({
    "creative_playmaker_score",       # kp_per90 * (pm + 0.5 * pf)
    "defensive_solidity",             # def_actions * (1 - dp / (dp + 1.0))
    "penalty_threat_score",           # pens_won * 3.0 + fouls_drawn * 0.08, clip 1.0
    "card_risk_index",                # cards * 2.0 + reds * 3.0 + ratio * 0.5, clip 1.0
    "midfield_engine_score",          # /50, /3, /5 normalisers, 0.4/0.4/0.2 weights, 0.3 * pd
    "aggression_adjusted_threat",     # / (1 + offsides * 0.3)
    "defensive_vulnerability_index",  # 1.5 / 0.5 / 0.5 weights, clip 2.0, 1.0/0.7/0.3 positions
    "progressive_carrier_score",      # /3, /60 normalisers, 0.45/0.35/0.20 weights, clips
    "disciplinary_pressure_index",    # 0.20/0.25/1.5/0.10 weights, clip 2.0
    "foul_magnet_score",              # 0.30/0.40/0.30 weights over two typed scores, clip 1.0
    "creative_pressure_matchup",      # creative_playmaker_score (typed 0.5) * opp_def_fouls_pg
    "carrier_vs_press",               # progressive_carrier_score (typed) * opp_def_aggression
    "opp_mid_discipline",             # opp_mid_fouls * 0.6 + opp_mid_cards * 2.0
})
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in TYPED_FORMULA_FEATURES_WITHHELD]

# 2026-07-08 (full A-Z audit): the referee/quality/height/rest features had BUGS (leak,
# train≠serve scale, call-order, wrong-opponent, unit/inversion). Those computations are now
# FIXED in feature_engineering.py / league_quality.py — so the features stay in the model.
# Only season_pass_accuracy is DROPPED: it's a genuine DATA GAP (no per-match pass-accuracy in
# the API feed), so there is nothing to compute as-of-date — dropping is the only honest fix.
_DATA_GAP_FEATURES = {"season_pass_accuracy"}
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in _DATA_GAP_FEATURES]

# WITHHELD UNTIL REAL DATA EXISTS (props_truth, 2026-11-02; round-2 finding V3). Measured on the
# committed player_history at data-truth-100 40d7dda9: age = 25.0, height_cm = 180, weight_kg = 75
# and age_peak_delta = 0 on 876,130 of 876,130 rows - the old build_features placeholders, never a
# collected value (enrich-profiles never populated them). Every feature below is that placeholder or
# arithmetic on it, so it carries no information about the player. They leave the props TRAINING
# feature set (config.PLAYER_FEATURE_COLS); the serving payloads keep their own persisted
# feature_cols and are untouched (no retrain - the drop takes effect at the next owner-approved
# props retrain). Re-admit a feature only when player_history carries a REAL value for it
# (collected profiles, output/collect/player_profiles.parquet) on a measured share of rows.
PROFILE_FEATURES_WITHHELD = {
    "age", "age_peak_delta", "height_cm", "height_aerial_interaction",
    "opp_def_mean_height", "height_diff_vs_opp_def", "height_advantage_score",
}
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in PROFILE_FEATURES_WITHHELD]

# A player-match row cannot have more minutes than a match with extra time (120) plus stoppage.
# Stored rows above this are provider errors (League One / League Two 2024-03-13/14: 142-174 min);
# the raw value is kept and the minutes-derived features treat it as missing. 121-123 min appear on
# extra-time matches (World Cup 2026-07-03, Argentina 2025-12-02) and are plausible.
MAX_MATCH_MINUTES = 130

# CURRENT-CLUB RULE at tip time (invariant 12): a player whose latest CLUB appearance in
# player_history is for a different club than the one we would tip him for, within this many days
# of the fixture, is not tipped (conflicting evidence, logged). Ghost tips measured 2026-11-02: 337
# graded player_ledger rows where the same player_id played another fixture that day (Anthony Gordon
# for Newcastle on 2026-08-23 while at Barcelona; Casemiro for Manchester United while at Inter Miami).
CURRENT_CLUB_CONFLICT_DAYS = 21

# Understat shot-level features (xG, headed/SP/FK goals, goals-xG) were scraped + integrated
# (3 seasons, 133k shots, 84% top-5 coverage) and tested 2026-06-30: NO measurable AUC gain
# (deltas +/-0.002, noise) even on top-5-only rows. Reason: redundant with existing goals_pg/
# sot_pg/shots/form. NOT added to the model. Scraper + parquet columns retained for future use.
# _UNDERSTAT_FEATURES = ["u_xg_pg","u_xg_recent","u_goals_minus_xg","u_headed_goals_pg","u_sp_goals_pg","u_fk_goals_pg"]

# Phase 2 (future — needs /fixtures/events API data):
#   "referee_strictness"       (/fixtures/events fouls + /standings high-stakes flag)
# Phase 3 (future — needs /players/statistics API data):
#   "xg_per90", "xa_per90"    (shot location weighting — top goals feature once available)
MIN_APPEARANCES  = 3   # season-stats fallback path
MIN_GAMES_SIGNAL = 5   # minimum match history to generate a signal
MIN_GAMES_SNIPER = 8

# ── Minimum samples for MEASURED context features (Data Truth, owner rule 2026-11-01) ──────────
# Opponent / keeper / penalty / referee / league context is computed from the matches we hold.
# Below these sample sizes the value is NaN (explicitly missing), never a typed prior.
MIN_OPP_MATCHES    = 3    # an opponent team's prior matches before its concede/discipline rates count
MIN_GK_MATCHES     = 3    # a keeper's prior matches before his save rate / saves count
MIN_PEN_ATTEMPTS   = 3    # a player's prior penalty attempts before his conversion rate counts
MIN_REF_MATCHES    = 3    # a referee's prior matches before his yellows/match counts
MIN_LEAGUE_MATCHES = 20   # a league's prior fixtures before its own average is used as a normaliser
MIN_PREV_SEASON_MATCHES = 5   # a player's / team's previous-season matches before a prev_season_* mean counts
SHOT_ACC_WINDOW = 10          # matches in shot_accuracy_l10 (the 5-match shot_accuracy_rate stays as is)
MIN_SHOTS_FOR_ACCURACY = 10   # shots in the window (or previous season) below which an accuracy is NaN

# ── Leagues with player prop markets ─────────────────────────────────────────
# DERIVED FROM THE REGISTRY (Data Truth, LR-1.1): config/leagues.json `props: true` entries,
# {canonical name: API-Football id}. The hand-typed table that lived here (29 leagues, with the
# probe notes) was copied into the registry verbatim and is no longer typed in code.
def _v9_src(module: str):
    """v9's src.<module>, even when the importer has its OWN `src` package (wowzaV9-Pro imports
    player_model from _v9/, and there `src` is Pro's - a plain import crashed Pro Live Odds on
    2026-11-01). A foreign `src` and sys.path are restored afterwards."""
    import importlib
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cur = sys.modules.get("src")
    foreign = cur is not None and (root / "src").resolve() not in {
        Path(p).resolve() for p in getattr(cur, "__path__", [])}
    if not foreign:
        if str(root) not in sys.path:
            sys.path.append(str(root))
        return importlib.import_module(f"src.{module}")
    if module in _SRC_FOREIGN:
        return _SRC_FOREIGN[module]
    saved_path = list(sys.path)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    sys.path.insert(0, str(root))
    try:
        _SRC_FOREIGN[module] = importlib.import_module(f"src.{module}")
        return _SRC_FOREIGN[module]
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)


_SRC_FOREIGN: dict = {}     # v9 modules privately loaded under a foreign `src`


def league_registry():
    """v9's src.league_registry (foreign-`src` safe)."""
    return _v9_src("league_registry")


PROP_LEAGUES = league_registry().prop_leagues()


def prop_season(league: str, on_date=None) -> str:
    """API-Football season for `league` on `on_date` (default today), from the canonical
    registry's season rules (src.season_resolver). An unknown league gives 'SEASON_UNKNOWN',
    which API-Football answers with nothing - a visible miss, never a guessed year.
    (The month>=7 `_prop_season_now` rule that used to sit here was deleted: it gave every league
    the European calendar, Data Truth audit findings 38, 41.)"""
    return _v9_src("season_resolver").resolve_api_season(league, on_date)


# One entry per league in PROP_LEAGUES, always. Never edit this by hand — add the league to
# PROP_LEAGUES and its season appears here. League-aware since 2026-11-01.
PROP_SEASONS = {league: prop_season(league) for league in PROP_LEAGUES}
# {league: (fbref id, fbref slug)} from the registry's provider_names.fbref (LR-1.1), no longer typed.
FBREF_LEAGUES = {e["name"]: (e["provider_names"]["fbref"]["id"], e["provider_names"]["fbref"]["slug"])
                 for e in league_registry().leagues()
                 if (e.get("provider_names") or {}).get("fbref")}
CACHE_DAYS    = 7
REQUEST_DELAY = 4.0
