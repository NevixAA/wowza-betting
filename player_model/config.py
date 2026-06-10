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

# ── Features ──────────────────────────────────────────────────────────────────
ROLLING_N = 5
PLAYER_FEATURE_COLS = [
    "goals_pg", "assists_pg", "shots_pg", "sot_pg",
    "cards_pg", "minutes_pg", "key_passes_pg", "sot_rate",
    "set_piece_shot_rate", "touches_box_per90", "aerial_won_rate",
    "is_home",
    "opp_goals_conceded_pg", "opp_shots_conceded_pg", "opp_sot_conceded_pg",
    "team_attack_str", "team_corners_per90", "team_goals_scored_pg",
    "referee_strictness", "ref_cards_per_game",
    "pos_forward", "pos_midfielder", "pos_defender",
    "rest_days",
]
# starter_rate is computed but not yet in PLAYER_FEATURE_COLS — add and retrain to activate:
#   "starter_rate",
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
    # ── Active summer leagues ─────────────────────────────────────────────────
    "Ireland Premier": 357,
    "Finland Veikk":   244,
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
    "Ireland Premier": "2026",
    "Finland Veikk":   "2026",
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
