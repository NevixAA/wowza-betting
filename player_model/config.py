"""Player Model Configuration."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[1]   # v9/
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_FILE = BASE_DIR / "player_stats_cache.json"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST   = os.getenv("API_HOST",   "api-football-v1.p.rapidapi.com")
API_KEY    = os.getenv("API_KEY",    "")
API_SEASON = os.getenv("API_SEASON", "2025")

# ── Markets ───────────────────────────────────────────────────────────────────
MARKETS = ["goals", "assists", "sot", "cards"]

MODEL_FILES = {
    "goals":   MODELS_DIR / "model_player_goals.pkl",
    "assists": MODELS_DIR / "model_player_assists.pkl",
    "sot":     MODELS_DIR / "model_player_sot.pkl",
    "cards":   MODELS_DIR / "model_player_cards.pkl",
}

# ── Features ──────────────────────────────────────────────────────────────────
ROLLING_N = 5   # last N appearances per player

PLAYER_FEATURE_COLS = [
    # Player rolling stats (last N games)
    "goals_pg",           # goals per game rolling
    "assists_pg",
    "shots_pg",
    "sot_pg",             # shots on target per game
    "cards_pg",           # yellow cards per game
    "minutes_pg",         # avg minutes played
    "key_passes_pg",
    # Match context
    "is_home",
    "opp_goals_conceded_pg",   # opponent defensive weakness
    "opp_shots_conceded_pg",
    "team_attack_str",         # team-level attack strength
    "team_goals_scored_pg",
    # Player position encoding
    "pos_forward",
    "pos_midfielder",
    "pos_defender",
    # Fatigue
    "rest_days",
]

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_APPEARANCES = 3    # minimum games to include a player
SNIPER_EDGE     = 0.08 # 8% edge → SNIPER player prop
VALUE_EDGE      = 0.04 # 4% edge → VALUE player prop

# ── Leagues to collect training data for (API-Football IDs) ──────────────────
TRAINING_LEAGUES = {
    "Premier League":  39,
    "Championship":    40,
    "League One":      41,
    "League Two":      42,
    "La Liga":          140,
    "La Liga 2":        141,
    "Bundesliga":       78,
    "Bundesliga 2":     79,
    "Ligue 1":          61,
    "Ligue 2":          62,
}

TRAINING_SEASONS = ["2023", "2024", "2025"]
