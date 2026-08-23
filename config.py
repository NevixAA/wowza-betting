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

from dotenv import load_dotenv
_BASE = Path(__file__).resolve().parent
load_dotenv(_BASE / ".env",      override=False)
load_dotenv(_BASE / ".api_keys", override=False)

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

# ── HT models (standard-format leagues only — they have HTHG/HTAG data) ──────
HT_MODEL_FILE_05 = MODELS_DIR / "model_ht_over05.pkl"
HT_MODEL_FILE_15 = MODELS_DIR / "model_ht_over15.pkl"

# ── Side-market models (BTTS / O1.5 / O3.5) — same feature set as main model ─
MODEL_FILE_BTTS   = MODELS_DIR / "model_v9_btts.pkl"
MODEL_FILE_OVER15 = MODELS_DIR / "model_v9_over15.pkl"
MODEL_FILE_OVER35 = MODELS_DIR / "model_v9_over35.pkl"

# target column → model file
SIDE_MARKETS = {
    "btts":   MODEL_FILE_BTTS,
    "over15": MODEL_FILE_OVER15,
    "over35": MODEL_FILE_OVER35,
}
# Human-readable labels for Telegram / dashboard
SIDE_MARKET_LABELS = {
    # "BTTS" alone does not say whether the tip is that both teams DO score or that they do not.
    # The side-market generator is one-sided by construction (it reads p_{market}/odds_{market}),
    # so every BTTS signal is YES. "Over 1.5"/"Over 3.5" already state their side in the name.
    "btts":   "BTTS — YES",
    "over15": "Over 1.5",
    "over35": "Over 3.5",
}

# ── League format classification ──────────────────────────────────────────────
# Standard: historical CSVs contain shots, corners, O/U odds → richer features
STANDARD_FORMAT_LEAGUES = {
    "League One", "League Two",
    "Bundesliga 2", "La Liga 2", "Ligue 2",
    "Championship",
    "Serie B",
    "Greek Super League",
    # Training-only (no OddsAPI live key)
    "National League",
    "Portuguese Primeira Liga",
    "Scottish Championship", "Scottish League One", "Scottish League Two",
    # Below have API keys but backtest shows negative ROI — training only, not predicted
    "Belgian First Division A",  # -21.53%
    "Dutch Eredivisie",          # -3.61%
    "Scottish Premiership",      # -15.39%
    "Turkish Super Lig",         # -8.03%
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
    # API-Football-only (not on football-data.co.uk) — training only until validated
    "Saudi Pro League",
    "K-League 1",
}

def model_type_for_league(league) -> str:
    """Canonical model tag for a league: 'standard' | 'new_format' | 'unknown'.

    Single source of truth so the ledger, digest, and dashboard never disagree on which
    model a bet belongs to. Reporting was previously treating an untagged (blank) row as
    'standard', which let new-format bets leak into the standard column.
    """
    lg = str(league).strip()
    if lg in STANDARD_FORMAT_LEAGUES:
        return "standard"
    if lg in NEW_FORMAT_LEAGUES:
        return "new_format"
    return "unknown"


# Leagues loaded entirely from API-Football (no football-data.co.uk coverage)
API_FOOTBALL_ONLY_LEAGUES = {
    "Saudi Pro League",
    "K-League 1",
}

# Historical seasons to backfill for API-Football-only leagues
API_FOOTBALL_EXTRA_SEASONS: dict = {
    "Saudi Pro League": ["2019", "2020", "2021", "2022", "2023", "2024"],
    "K-League 1":       ["2020", "2021", "2022", "2023", "2024", "2025"],
}

# ── Live-prediction leagues (have OddsAPI coverage) ───────────────────────────
# Only leagues listed here generate live betting tips via pipeline.py --mode predict
ENABLED_LEAGUES = {
    # Standard format — live predictions enabled
    "League One", "League Two",
    "Bundesliga 2", "La Liga 2", "Ligue 2",
    "Championship",
    "Serie B",
    # "Greek Super League",   # -7.26% ROI
    # "Belgian First Division A", # -21.53% ROI
    # "Dutch Eredivisie",         # -3.61% ROI
    # "Scottish Premiership",     # -15.39% ROI
    # "Turkish Super Lig",        # -8.03% ROI
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
    # API-Football-only — training only until 1+ seasons validated
    # "Saudi Pro League",  # enable after ROI validation
    # "K-League 1",        # enable after ROI validation
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

# ── Training data filters ─────────────────────────────────────────────────────
# Exclude COVID seasons — empty stadiums created anomalous patterns not present today
EXCLUDE_COVID_SEASONS = True
COVID_SEASONS = {"2019/20", "2020/21"}   # seasons to drop from training

# Time-decay weights — recent seasons are more representative of current market
TRAINING_DECAY_WEIGHTS = {
    "2024/25": 4.0,
    "2023/24": 3.0,
    "2022/23": 2.0,
    "2021/22": 1.5,
    "2020/21": 0.0,   # COVID — excluded
    "2019/20": 0.0,   # COVID — excluded
    "2018/19": 1.0,
    "2017/18": 1.0,
    "2016/17": 0.8,
    "2015/16": 0.6,
}
DEFAULT_DECAY_WEIGHT = 1.0

# ── Three-tier signal system ──────────────────────────────────────────────────
#
#  SNIPER    — meets per-league threshold (12–25%)  → full stake (real tip)
#  MARKSMAN  — edge 14% to league threshold         → 3/4 stake (real tip)
#  VALUABLE  — edge 4–14%                           → info only, not a tip
#
#  EDGE_CEILING: above 19% the model is overconfident (backtest shows -20% ROI
#  at 16-20% range). Bets above the ceiling are downgraded to MARKSMAN.
#
#  MARKSMAN raised 8%→14% on 2026-06-17: sweep showed 8–14% zone is -1.6% ROI;
#  14%+ zone is +6.5% ROI. Bundesliga 2 excluded from MARKSMAN entirely (see
#  LEAGUE_MARKSMAN_THRESHOLDS). Combined effect: S+MM ROI 22% → ~45%+.
#
VALUABLE_THRESHOLD      = float(os.getenv("VALUABLE_THRESHOLD",      "0.04"))   # min edge to show
MARKSMAN_THRESHOLD      = float(os.getenv("MARKSMAN_THRESHOLD",      "0.14"))   # raised from 0.08 → 0.14
SNIPER_THRESHOLD        = float(os.getenv("SNIPER_THRESHOLD",        "0.12"))   # global SNIPER floor (raised from 0.10)
SNIPER_THRESHOLD_OVER   = float(os.getenv("SNIPER_THRESHOLD_OVER",   "0.12"))   # OVER needs a bit more edge
SNIPER_THRESHOLD_UNDER  = float(os.getenv("SNIPER_THRESHOLD_UNDER",  "0.12"))   # raised from 0.10 to match OVER
EDGE_CEILING            = float(os.getenv("EDGE_CEILING",            "0.19"))   # above this = model overconfident
DRIFT_UPGRADE_EDGE      = float(os.getenv("DRIFT_UPGRADE_EDGE",      "0.10"))

# Per-league edge ceilings — override global EDGE_CEILING for new-format leagues
# where thinner data / noisier bookmaker prices cause the model to overfit at high edges.
LEAGUE_EDGE_CEILING: dict = {
    "Brazil Serie A":            0.17,
    "Japan J-League":            0.17,
    "China Super League":        0.15,
    "Mexico Liga MX":            0.18,
    "Ireland Premier Division":  0.16,
    "Finland Veikkausliiga":     0.16,
    "Sweden Allsvenskan":        0.17,
    "Norway Eliteserien":        0.17,
    "Denmark Superliga":         0.17,
    "Austrian Bundesliga":       0.17,
    "Romanian Superliga":        0.15,
    "Argentina Primera Division": 0.18,
    "USA MLS":                   0.17,
}

# Backward-compat alias
VALUE_THRESHOLD = VALUABLE_THRESHOLD

# ── Per-league SNIPER thresholds (from backtest optimisation) ─────────────────
# These override SNIPER_THRESHOLD for specific leagues. All capped at EDGE_CEILING.
#
# LEAGUE_SNIPER_CAP (env, default 1.0 = no cap) clamps every value below. Set it to widen the
# SNIPER tier without editing these numbers, which are backtest-optimised and are kept here as the
# documented reference — the ROI beside each one is why it is what it is. A cap is reversible by
# deleting one env line; overwriting them would lose the calibration.
_SNIPER_CAP = float(os.getenv("LEAGUE_SNIPER_CAP", "1.0"))
_LEAGUE_SNIPER_CALIBRATED: dict = {
    # Standard format — live prediction leagues (thresholds from backtest optimisation)
    "League Two":     0.14,   # most data, reliable at 14%  → ROI +22.6%
    "Bundesliga 2":   0.20,   # needs higher bar             → ROI +22.4%
    "La Liga 2":      0.20,   # high threshold, big ROI      → ROI +53.5%
    "League One":     0.25,   # very high bar needed         → ROI +21.4%
    "Ligue 2":        0.25,   # high bar, still good ROI     → ROI +45.2%
    "Championship":   0.15,   # limited edge, conservative
    "Serie B":        0.15,   # limited edge, conservative
    "Greek Super League": 0.25,  # historically weak — high bar
}
LEAGUE_SNIPER_THRESHOLDS: dict = {k: min(v, _SNIPER_CAP)
                                  for k, v in _LEAGUE_SNIPER_CALIBRATED.items()}

# ── Per-league MARKSMAN thresholds (override global MARKSMAN_THRESHOLD) ────────
# Set equal to the league's SNIPER threshold to disable MARKSMAN for that league.
# Bundesliga 2: MARKSMAN 8-20% backtest was -10.8% ROI — no MARKSMAN bets here.
_LEAGUE_MARKSMAN_CALIBRATED: dict = {
    "Bundesliga 2": 0.20,   # match SNIPER — no MARKSMAN; 8-20% = -10.8% ROI
    "League Two":   0.14,   # match SNIPER — 8-14% range had low +4.3% ROI
}
# Capped by the same env knob. Without this a lowered SNIPER cap would leave a per-league MARKSMAN
# floor ABOVE its own SNIPER threshold, so MARKSMAN could never fire in that league — the tier
# would vanish silently rather than widen.
LEAGUE_MARKSMAN_THRESHOLDS: dict = {k: min(v, _SNIPER_CAP)
                                    for k, v in _LEAGUE_MARKSMAN_CALIBRATED.items()}

# Leagues where significant portion of matches are played on artificial turf.
# Artificial pitches are associated with lower goal counts (faster ball, tired legs).
ARTIFICIAL_PITCH_LEAGUES: set = {
    "Finland Veikkausliiga",
    "Sweden Allsvenskan",
    "Norway Eliteserien",
}

# ── Per-league display timezones (IANA) ───────────────────────────────────────
# Used ONLY for DISPLAY in the Telegram bot: OddsAPI commence_time is UTC, so a
# match's local calendar day can differ from its UTC day (e.g. Japan/Argentina).
# The stored `date` stays UTC (join key); the notifier converts kickoff_utc to the
# league's local tz for what the user reads. Unknown leagues fall back to UTC.
LEAGUE_TIMEZONES: dict = {
    # Standard format
    "League One":                "Europe/London",
    "League Two":                "Europe/London",
    "Championship":              "Europe/London",
    "National League":           "Europe/London",
    "Bundesliga 2":              "Europe/Berlin",
    "La Liga 2":                 "Europe/Madrid",
    "Ligue 2":                   "Europe/Paris",
    "Serie B":                   "Europe/Rome",
    "Greek Super League":        "Europe/Athens",
    "Portuguese Primeira Liga":  "Europe/Lisbon",
    "Scottish Championship":     "Europe/London",
    "Scottish League One":       "Europe/London",
    "Scottish League Two":       "Europe/London",
    "Scottish Premiership":      "Europe/London",
    "Belgian First Division A":  "Europe/Brussels",
    "Dutch Eredivisie":          "Europe/Amsterdam",
    "Turkish Super Lig":         "Europe/Istanbul",
    # New format
    "Denmark Superliga":         "Europe/Copenhagen",
    "Austrian Bundesliga":       "Europe/Vienna",
    "Austria Bundesliga":        "Europe/Vienna",
    "Romanian Superliga":        "Europe/Bucharest",
    "Sweden Allsvenskan":        "Europe/Stockholm",
    "Norway Eliteserien":        "Europe/Oslo",
    "Finland Veikkausliiga":     "Europe/Helsinki",
    "Ireland Premier Division":  "Europe/Dublin",
    "Argentina Primera Division": "America/Argentina/Buenos_Aires",
    "Brazil Serie A":            "America/Sao_Paulo",
    "Japan J-League":            "Asia/Tokyo",
    "Mexico Liga MX":            "America/Mexico_City",
    "China Super League":        "Asia/Shanghai",
    "USA MLS":                   "America/New_York",   # MLS spans zones; ET as display proxy
    "Saudi Pro League":          "Asia/Riyadh",
    "K-League 1":                "Asia/Seoul",
    "Poland Ekstraklasa":        "Europe/Warsaw",
}

KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
FLAT_STAKE     = float(os.getenv("FLAT_STAKE",     "1.0"))
USE_KELLY      = os.getenv("USE_KELLY", "0") == "1"

MIN_OVER_ODDS  = float(os.getenv("MIN_OVER_ODDS",  "1.75"))
MIN_UNDER_ODDS = float(os.getenv("MIN_UNDER_ODDS", "1.75"))

# ── Drift settings ────────────────────────────────────────────────────────────
DRIFT_CONFIRM_THRESHOLD  = 0.03
DRIFT_CONFLICT_THRESHOLD = 0.03

# ── Performance counting cutoff ───────────────────────────────────────────────
# Tips generated BEFORE this date are excluded from win/loss/ROI/PnL COUNTS shown
# on the dashboard and in Telegram digests — the new-format model had a calibration
# bug (fixed 2026-08-09) that produced garbage UNDER tips, so pre-fix results are not
# representative. The rows are NOT deleted: they stay in the ledgers/odds-history so
# CLV analysis keeps every data point. Only performance AGGREGATES honour this cutoff.
# Set to 2026-08-10 (2026-08-15): the fix landed ON 2026-08-09, so a plain date cutoff of
# "2026-08-09" still ADMITTED that day's pre-fix tips — the 10:55 batch of 64 one-sided
# new-format UNDERs. They settled days later and surfaced in the digest as 8 phantom "bets"
# for 2026-08-14 (7L/1W). Bumping to 2026-08-10 drops the whole pre-fix day, which is what
# the previous note here recommended and what wowza-v11/config.py already uses — keep the
# two in sync. Rows are NOT deleted: they stay in the ledgers so CLV keeps every data point.
PERFORMANCE_CUTOFF_DATE = "2026-08-10"

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_WALK_SIZE = 60
BACKTEST_MIN_TRAIN = 400

# ── APIs — loaded from .env / .api_keys / GitHub Actions secrets ─────────────

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
API_KEY      = os.getenv("API_KEY",      "").strip()
API_HOST     = "api-football-v1.p.rapidapi.com"
API_SEASON   = os.getenv("API_SEASON", "2025")

API_FOOTBALL_IDS = {
    # Standard format leagues
    "Championship":             40,
    "League One":               41,
    "League Two":               42,
    "Bundesliga 2":             79,
    "Ligue 2":                  65,
    "La Liga 2":               141,
    "Serie B":                 136,
    "Greek Super League":      197,
    # New format — European winter/spring
    "Denmark Superliga":       119,
    "Austrian Bundesliga":     218,
    "Romanian Superliga":      283,
    # New format — summer / Americas / Asia
    "Sweden Allsvenskan":      113,
    "Norway Eliteserien":      103,
    "Finland Veikkausliiga":   244,
    "Ireland Premier Division": 357,
    "Argentina Primera Division": 128,
    "Brazil Serie A":           71,
    "Japan J-League":           98,
    "Mexico Liga MX":          262,
    "China Super League":      169,
    "USA MLS":                 253,
    # API-Football-only leagues (not on FD)
    "Saudi Pro League":        307,
    "K-League 1":              292,
    # International
    "World Cup 2026":            1,
}
# IDs may need verification at dashboard.api-football.com/docs

# Per-league season year (API-Football start year convention)
API_FOOTBALL_SEASONS: dict = {
    # Standard format — 2026/27 European season (API-Football season = start year 2026).
    # ROLLED 2025->2026 on 2026-08-05: the 2025/26 season ended (~May 2026); on "2025" the
    # capture looked in the finished season and found NO upcoming fixtures -> zero standard
    # odds/1X2/HT/CLV collection for the whole 2026/27 season. New-format was already 2026.
    "Championship":             "2026",
    "League One":               "2026",
    "League Two":               "2026",
    "Bundesliga 2":             "2026",
    "Ligue 2":                  "2026",
    "La Liga 2":                "2026",
    "Serie B":                  "2026",
    "Greek Super League":       "2026",
    "Danish Superliga":         "2026",
    "Austrian Bundesliga":      "2026",
    "Romanian Superliga":       "2026",
    # Summer leagues (calendar year = season year)
    "Sweden Allsvenskan":       "2026",
    "Norway Eliteserien":       "2026",
    "Finland Veikkausliiga":    "2026",
    "Ireland Premier Division": "2026",
    "Brazil Serie A":           "2026",
    "Japan J-League":           "2026",
    "Mexico Liga MX":           "2026",
    "China Super League":       "2026",
    "USA MLS":                  "2026",
    # South America (2026 season under way by Aug)
    "Argentina Primera Division": "2026",
    # API-Football-only leagues (current season)
    "Saudi Pro League":          "2026",   # 2026/27 season starts Aug 2026
    "K-League 1":                "2026",   # calendar year 2026
    # International
    "World Cup 2026":            "2026",
}

ODDS_API_SPORT_KEYS = {
    # World Cup
    "World Cup":               "soccer_fifa_world_cup",
    "World Cup 2026":          "soccer_fifa_world_cup",
    # Standard format
    "Championship":            "soccer_efl_champ",
    "League One":              "soccer_england_league1",
    "League Two":              "soccer_england_league2",
    "Bundesliga 2":            "soccer_germany_bundesliga2",
    "Ligue 2":                 "soccer_france_ligue_two",
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
    # API-Football-only (for future live odds when enabled)
    "Saudi Pro League":         "soccer_saudi_arabia_pro_league",
    "K-League 1":               "soccer_korea_kleague1",
}

SOFASCORE_TOURNAMENT_IDS = {
    "Championship": 18, "League One": 24, "League Two": 25,
    "National League": 173, "Bundesliga 2": 44, "Ligue 2": 182,
    "La Liga 2": 54, "Serie B": 53, "Denmark Superliga": 39,
    "Norway Eliteserien": 317,
}
