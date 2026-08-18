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
API_SEASON = os.getenv("API_SEASON", "2025")

# ── Markets ───────────────────────────────────────────────────────────────────
MARKETS    = ["goals", "goals2", "assists", "sot", "sot2", "sot3", "cards"]

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

# World Cup cards: features are imputed → predictions inflate to ~0.40 for everyone.
# Only emit WC card tips for players whose REAL booking rate (max of cards_pg /
# season_cards_pg) is at least this — for them, ~0.40 is roughly their true rate.
# (Club-league cards are unaffected; the model is calibrated there.)
WC_CARD_MIN_RATE = 0.35

# Starter proxy: when a confirmed lineup isn't available yet, only tip players who
# average at least this many minutes/game (filters fringe/rotation players who may
# not start). Confirmed-lineup filtering still takes precedence when available.
MIN_STARTER_MINUTES = 60.0

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
VALUABLE_ONLY_MARKETS     = {"sot2", "sot3", "goals2", "cards"}
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
    "set_piece_threat_score",    # aerial_won_rate × (corners/6) × 0.30 sp rate × position
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
    "set_piece_corner_matchup",       # set_piece_threat_score × (team_corners_pg/6)
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
    # ── Set piece event features (from /fixtures/events — header/FK goals) ──────
    "sp_goals_pg",               # rolling SP goals per game (headers + FKs)
    "headed_goals_pg",           # rolling header goals per game
    "fk_goals_pg",               # rolling direct free kick goals per game
    "sp_assist_pg",              # rolling SP assists per game (taker signal)
    "career_sp_goals_rate",      # career SP goal rate (expanding window)
    "career_sp_assist_rate",     # career SP assist rate (taker signal, expanding)
    "sp_goals_share",            # sp_goals_pg / goals_pg capped [0,1]
    "headed_goals_share",        # headed_goals_pg / goals_pg capped [0,1]
    # ── Set piece role ────────────────────────────────────────────────────────
    "sp_taker_score",            # 0-1 probability this player DELIVERS set pieces
    "sp_receiver_score",         # aerial threat when NOT the taker (defender/CM)
    # ── Height matchup vs opponent defenders ──────────────────────────────────
    "opp_def_mean_height",       # mean height of opponent starting defenders (cm)
    "height_diff_vs_opp_def",    # player height minus opp mean CB height
    "height_advantage_score",    # normalized: +1 = 10cm taller, clipped [-2,+2]
    # ── Opponent set piece vulnerability ──────────────────────────────────────
    "opp_sp_goals_conceded_pg",  # rolling SP goals conceded per game by opponent
    # ── Set piece composites ──────────────────────────────────────────────────
    "defender_sp_edge",              # aerial_won * height_adj * opp_sp_weakness * (1-taker)
    "defender_sot_edge",             # SP aerial SOT threat — core defender-market edge (goals×5 proxy)
    "defender_sot_role_index",       # how central this defender is to team's aerial SP attack
    "sp_threat_vs_weak_sp_defense",  # sp_goals_pg * opp_sp_goals_conceded_pg / 0.12
    "aerial_height_sp_composite",    # aerial_won * height_adj * opp_sp_pg * pos_weight
    "sp_goal_probability_composite", # career + rolling + aerial + opp weighted sum
    "sp_taker_assist_edge",          # taker prob * sp_assist rate * corner volume
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
]

# Pruned 2026-06-30 (local): set-piece / goal-type features that can't be reliably populated —
# they need header/set-piece goal data that no accessible source provides for our prop leagues
# (API-Football goal events = "Normal Goal" only; Opta/StatsBomb-commercial = enterprise;
# StatsBomb-open = historical seasons only; FBref = no clean headed-goals). All were 0-0.5%
# nonzero (dead weight/noise). Removed so models train on RELIABLE features only.
_DEAD_GOAL_TYPE_FEATURES = {
    "sp_goals_pg", "headed_goals_pg", "fk_goals_pg", "sp_assist_pg",
    "career_sp_goals_rate", "career_sp_assist_rate", "sp_goals_share", "headed_goals_share",
    "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge", "opp_sp_goals_conceded_pg",
    "sp_goal_probability_composite", "defender_sp_edge", "defender_sot_edge",
    "aerial_height_sp_composite",
}
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in _DEAD_GOAL_TYPE_FEATURES]

# 2026-07-08 (full A-Z audit): the referee/quality/height/rest features had BUGS (leak,
# train≠serve scale, call-order, wrong-opponent, unit/inversion). Those computations are now
# FIXED in feature_engineering.py / league_quality.py — so the features stay in the model.
# Only season_pass_accuracy is DROPPED: it's a genuine DATA GAP (no per-match pass-accuracy in
# the API feed), so there is nothing to compute as-of-date — dropping is the only honest fix.
_DATA_GAP_FEATURES = {"season_pass_accuracy"}
PLAYER_FEATURE_COLS = [f for f in PLAYER_FEATURE_COLS if f not in _DATA_GAP_FEATURES]

# Understat shot-level features (xG, headed/SP/FK goals, goals-xG) were scraped + integrated
# (3 seasons, 133k shots, 84% top-5 coverage) and tested 2026-06-30: NO measurable AUC gain
# (deltas +/-0.002, noise) even on top-5-only rows. Reason: redundant with existing goals_pg/
# sot_pg/shots/form. NOT added to the model. Scraper + parquet columns retained for future use.
# _UNDERSTAT_FEATURES = ["u_xg_pg","u_xg_recent","u_goals_minus_xg","u_headed_goals_pg","u_sp_goals_pg","u_fk_goals_pg"]

# Phase 2 (future — needs /fixtures/events API data):
#   "set_piece_threat_score"   (aerial_won_rate * team_corners_per90_opp * opp_sp_concession_rate)
#   "referee_strictness"       (/fixtures/events fouls + /standings high-stakes flag)
# Phase 3 (future — needs /players/statistics API data):
#   "xg_per90", "xa_per90"    (shot location weighting — top goals feature once available)
MIN_APPEARANCES  = 3   # season-stats fallback path
MIN_GAMES_SIGNAL = 5   # minimum match history to generate a signal
MIN_GAMES_SNIPER = 8

# ── Leagues with player prop markets ─────────────────────────────────────────
# IMPORTANT: Only popular leagues have SOT/Goals/Cards prop markets at bookmakers.
# Lower leagues (League Two, Ligue 2, La Liga 2) rarely have these markets.
PROP_LEAGUES = {
    # ── Top tier — biggest player prop markets ────────────────────────────────
    "Premier League":  39,    # ⭐⭐⭐⭐⭐ Most liquid props globally
    "Bundesliga":      78,    # ⭐⭐⭐⭐  Very active German market
    "La Liga":         140,   # ⭐⭐⭐⭐  Active Spanish market
    "Serie A":         135,   # ⭐⭐⭐   Good Italian coverage
    "Ligue 1":         61,    # ⭐⭐⭐   French market
    # ── Our standard model leagues — moderate prop coverage ───────────────────
    "Championship":    40,
    "League One":      41,
    "Bundesliga 2":    79,
    # ── European club competitions ────────────────────────────────────────────
    "Champions League":    2,    # ⭐⭐⭐⭐⭐ Biggest club prop market
    "Europa League":       3,    # ⭐⭐⭐⭐  Active prop market
    "Conference League":   848,  # ⭐⭐⭐   Growing prop market
    # ── World Cup 2026 — biggest event globally ───────────────────────────────
    "World Cup":       1,
}
# Season year for each league (API-Football uses start year of season)
PROP_SEASONS = {
    # 2026/27 season (API-Football season = start year 2026). ROLLED 2025->2026 on 2026-08-05:
    # on "2025" the prop capture looked in the finished 2025/26 season and found no upcoming
    # club fixtures -> zero club-league prop-odds/CLV collection once club props start (~mid-Aug).
    "Premier League":  "2026",
    "Bundesliga":      "2026",
    "La Liga":         "2026",
    "Serie A":         "2026",
    "Ligue 1":         "2026",
    "Championship":    "2026",
    "League One":      "2026",
    "Bundesliga 2":    "2026",
    "Champions League":    "2026",
    "Europa League":       "2026",
    "Conference League":   "2026",
    "World Cup":       "2026",
}
FBREF_LEAGUES = {
    "Championship": (10, "Championship"), "League One": (15, "League-One"),
    "League Two": (16, "League-Two"),     "Bundesliga 2": (33, "2-Bundesliga"),
    "Ligue 2": (60, "Ligue-2"),           "La Liga 2": (17, "Segunda-Division"),
    "Serie B": (18, "Serie-B"),
}

CACHE_DAYS    = 7
REQUEST_DELAY = 4.0
