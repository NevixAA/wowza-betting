"""
v9 Configuration
================
Two separate model tracks:
  STANDARD  — leagues with full stats (shots, corners, O/U odds history)
  NEW_FORMAT — leagues with goals only (no shot/corner/odds history)

Each track has its own model file and backtest. They never mix.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent   # mixed/ — where historical XLSX + shared caches live

# ── Paths ────────────────────────────────────────────────────────────────────
SUMMARY_XLSX      = DATA_DIR / "England_Leagues_4_Seasons_With_Summary.xlsx"
SOFASCORE_CACHE   = DATA_DIR / "sofascore_cache.json"
ODDS_HISTORY_JSON = BASE_DIR / "odds_history_v9.json"
V7_BEST_PARAMS    = DATA_DIR / "best_params.json"

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
for d in (MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Two model files — NEVER mix standard and new-format data ─────────────────
MODEL_FILE_STANDARD  = MODELS_DIR / "model_v9_standard.pkl"
MODEL_FILE_NEWFORMAT = MODELS_DIR / "model_v9_newformat.pkl"
MODEL_FILE           = MODEL_FILE_STANDARD   # backward-compat alias

# ── League format classification ──────────────────────────────────────────────
# Standard: historical CSVs contain shots, corners, O/U odds → richer features
STANDARD_FORMAT_LEAGUES = {
    "League One", "League Two",
    "Bundesliga 2", "La Liga 2", "Ligue 2",
    "Championship",
    "Serie B",
    "Greek Super League",
    # Below have data but no OddsAPI live key → training-only contribution
    "National League",
    "Portuguese Primeira Liga",
    "Scottish Championship", "Scottish League One", "Scottish League Two",
}

# New-format: goals + 1X2 odds only — no shots/corners/O/U odds in history
NEW_FORMAT_LEAGUES = {
    # Existing
    "Denmark Superliga",
    "Austrian Bundesliga",
    "Sweden Allsvenskan",
    "Romanian Superliga",
    # New
    "Norway Eliteserien",
    "Finland Veikkausliiga",
    "Ireland Premier Division",
    "Argentina Primera Division",
    "Brazil Serie A",
    "Japan J-League",
    "Mexico Liga MX",
    "China Super League",
    "USA MLS",
}

# ── Live-prediction leagues (have OddsAPI coverage) ───────────────────────────
# Only leagues listed here generate live betting tips via pipeline.py --mode predict
ENABLED_LEAGUES = {
    # Standard format — live predictions enabled
    "League One", "League Two",
    "Bundesliga 2", "La Liga 2", "Ligue 2",
    "Championship",
    "Serie B",
    # "Greek Super League",  # removed: backtest ROI -7.26% (model finds no edge)
    # New format — live predictions enabled
    "Denmark Superliga",
    "Austrian Bundesliga",
    "Sweden Allsvenskan",
    "Norway Eliteserien",
    "Finland Veikkausliiga",
    "Ireland Premier Division",
    "Argentina Primera Division",
    "Brazil Serie A",
    "Japan J-League",
    "Mexico Liga MX",
    "China Super League",
    "USA MLS",
    # New format — training only (no OddsAPI key), kept for model quality
    "Romanian Superliga",
}

# ── football-data.co.uk league codes ─────────────────────────────────────────
FOOTBALL_DATA_LEAGUES = {
    # Standard format (mmz4281 URL)
    "Championship":            "E1",
    "League One":              "E2",
    "League Two":              "E3",
    "National League":         "EC",
    "Bundesliga 2":            "D2",
    "Ligue 2":                 "F2",
    "La Liga 2":               "SP2",
    "Serie B":                 "I2",
    "Greek Super League":      "G1",
    "Portuguese Primeira Liga": "P1",
    "Scottish Championship":   "SC1",
    "Scottish League One":     "SC2",
    "Scottish League Two":     "SC3",
    "Scottish Premiership":    "SC0",
    "Belgian First Division A": "B1",
    "Dutch Eredivisie":        "N1",
    "Turkish Super Lig":       "T1",
    # New format (football-data.co.uk/new/ URL)
    "Denmark Superliga":       "DNK",
    "Norway Eliteserien":      "NOR",
    "Austria Bundesliga":      "AUT",   # note: file code AUT
    "Austrian Bundesliga":     "AUT",
    "Romanian Superliga":      "ROM",
    "Sweden Allsvenskan":      "SWE",
    "Finland Veikkausliiga":   "FIN",
    "Ireland Premier Division": "IRL",
    "Argentina Primera Division": "ARG",
    "Brazil Serie A":          "BRA",
    "Japan J-League":          "JPN",
    "Mexico Liga MX":          "MEX",
    "China Super League":      "CHN",
    "USA MLS":                 "USA",
}

# ── Model ─────────────────────────────────────────────────────────────────────
TARGET_COL  = "over25"
ROLLING_N   = 5

# ── Signal tiers ──────────────────────────────────────────────────────────────
VALUE_THRESHOLD  = float(os.getenv("VALUE_THRESHOLD",  "0.04"))
SNIPER_THRESHOLD = float(os.getenv("SNIPER_THRESHOLD", "0.10"))
DRIFT_UPGRADE_EDGE = float(os.getenv("DRIFT_UPGRADE_EDGE", "0.07"))

KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
FLAT_STAKE     = float(os.getenv("FLAT_STAKE",     "1.0"))
USE_KELLY      = os.getenv("USE_KELLY", "0") == "1"

MIN_OVER_ODDS  = float(os.getenv("MIN_OVER_ODDS",  "1.75"))
MIN_UNDER_ODDS = float(os.getenv("MIN_UNDER_ODDS", "1.75"))

# ── Drift settings ────────────────────────────────────────────────────────────
DRIFT_CONFIRM_THRESHOLD  = 0.03
DRIFT_CONFLICT_THRESHOLD = 0.03

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_WALK_SIZE = 60
BACKTEST_MIN_TRAIN = 400

# ── APIs ──────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "61b2774a8935e8df42e3c66309606663")
API_KEY      = os.getenv("API_KEY",      "d96374d5cdmshe35b5a98fe9c502p175d62jsn647d9735d7d0")
API_HOST     = "api-football-v1.p.rapidapi.com"
API_SEASON   = os.getenv("API_SEASON", "2025")

API_FOOTBALL_IDS = {
    "Championship": 40,
    "League One":   41,
    "League Two":   42,
}

ODDS_API_SPORT_KEYS = {
    # Standard format
    "Championship":            "soccer_efl_champ",
    "League One":              "soccer_england_league1",
    "League Two":              "soccer_england_league2",
    "Bundesliga 2":            "soccer_germany_bundesliga2",
    "Ligue 2":                 "soccer_france_ligue_2",
    "La Liga 2":               "soccer_spain_segunda_division",
    "Serie B":                 "soccer_italy_serie_b",
    "Greek Super League":      "soccer_greece_super_league",
    # New format
    "Denmark Superliga":       "soccer_denmark_superliga",
    "Austrian Bundesliga":     "soccer_austria_bundesliga",
    "Sweden Allsvenskan":      "soccer_sweden_allsvenskan",
    "Norway Eliteserien":      "soccer_norway_eliteserien",
    "Finland Veikkausliiga":   "soccer_finland_veikkausliiga",
    "Ireland Premier Division": "soccer_league_of_ireland",
    "Argentina Primera Division": "soccer_argentina_primera_division",
    "Brazil Serie A":          "soccer_brazil_campeonato",
    "Japan J-League":          "soccer_japan_j_league",
    "Mexico Liga MX":          "soccer_mexico_ligamx",
    "China Super League":      "soccer_china_superleague",
    "USA MLS":                 "soccer_usa_mls",
    # Kept for results fetching (removed from enabled, negative ROI)
    "Belgian First Division A": "soccer_belgium_first_div",
    "Dutch Eredivisie":         "soccer_netherlands_eredivisie",
    "Scottish Premiership":     "soccer_spl",
    "Turkish Super Lig":        "soccer_turkey_super_league",
    "Poland Ekstraklasa":       "soccer_poland_ekstraklasa",
}

SOFASCORE_TOURNAMENT_IDS = {
    "Championship": 18, "League One": 24, "League Two": 25,
    "National League": 173, "Bundesliga 2": 44, "Ligue 2": 182,
    "La Liga 2": 54, "Serie B": 53, "Denmark Superliga": 39,
    "Norway Eliteserien": 317,
}
