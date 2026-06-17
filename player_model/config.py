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
MARKETS    = ["goals", "assists", "sot", "cards"]
MODEL_FILES = {
    "goals":   MODELS_DIR / "model_player_goals.pkl",
    "assists": MODELS_DIR / "model_player_assists.pkl",
    "sot":     MODELS_DIR / "model_player_sot.pkl",
    "cards":   MODELS_DIR / "model_player_cards.pkl",
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
]
# Phase 2 (future — needs /fixtures/events API data):
#   "set_piece_threat_score"   (aerial_won_rate * team_corners_per90_opp * opp_sp_concession_rate)
#   "referee_strictness"       (/fixtures/events fouls + /standings high-stakes flag)
# Phase 3 (future — needs /players/statistics API data):
#   "xg_per90", "xa_per90"    (shot location weighting — top goals feature once available)
MIN_APPEARANCES  = 3   # season-stats fallback path
MIN_GAMES_SIGNAL = 3   # minimum match history to generate a signal
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
    "Premier League":  "2025",
    "Bundesliga":      "2025",
    "La Liga":         "2025",
    "Serie A":         "2025",
    "Ligue 1":         "2025",
    "Championship":    "2025",
    "League One":      "2025",
    "Bundesliga 2":    "2025",
    "Champions League":    "2025",
    "Europa League":       "2025",
    "Conference League":   "2025",
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
