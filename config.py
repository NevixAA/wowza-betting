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

# ── CANONICAL LEAGUE REGISTRY (R2 finding 7, 2026-11-01) ─────────────────────
# Every per-league table below (format sets, ENABLED/PAPER, API-Football ids and seasons, OddsAPI
# sport keys, football-data codes, timezones, Sofascore ids, PROVIDER_UNSUPPORTED) is DERIVED from
# config/leagues.json by src/league_registry.py. Edit the registry, not this file. Registered
# aliases ("Austria Bundesliga", "Danish Superliga", "World Cup") resolve on lookup but are never
# separate keys. An invalid registry stops config from loading (RegistryError) rather than serve
# half a league table. tests/test_league_registry.py proves the derived tables equal the
# pre-registry values (snapshot of origin/main 4e9c93357).
import sys as _sys
if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))
from src import league_registry as _league_registry      # noqa: E402
from src import season_resolver as _season_resolver      # noqa: E402
_LR = _league_registry.derive()
LEAGUE_REGISTRY_VERSION = _LR.version

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
for d in (MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Two model files — NEVER mix standard and new-format data ─────────────────
MODEL_FILE_STANDARD  = MODELS_DIR / "model_v9_standard.pkl"
MODEL_FILE_NEWFORMAT = MODELS_DIR / "model_v9_newformat.pkl"
MODEL_FILE           = MODEL_FILE_STANDARD   # backward-compat alias
# Standard O/U 2.5 on the REPAIRED recipe (owner 2026-10-28). A SEPARATE serving file, written only
# by src/promotion_gate.owner_install (model type "standard_repaired"); read by predict only when
# STANDARD_OU25_SERVING_RECIPE (below) is "v9_repaired". MODEL_FILE_STANDARD stays the rollback.
MODEL_FILE_STANDARD_REPAIRED = MODELS_DIR / "model_v9_standard_repaired.pkl"

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
# Side-market fair price (owner decision 2026-11-01, R2-C finding 4). "devig_two_sided": the median
# over books of the power-de-vigged same-book pair (src/integrity/side_edge_devig); a fixture with no
# two-sided pair gets NO tip. "legacy_1_08": the old (1/odds)/1.08, which inflated edges in the tails
# (+2.67pp on sharp books) and produced 92 of 617 logged tips on its own.
# ROLLBACK: set SIDE_EDGE_FAIR_PRICE: "legacy_1_08" in predict.yml (or here).
SIDE_EDGE_FAIR_PRICES = ("devig_two_sided", "legacy_1_08")
SIDE_EDGE_FAIR_PRICE = os.getenv("SIDE_EDGE_FAIR_PRICE", "devig_two_sided").strip() or "devig_two_sided"
if SIDE_EDGE_FAIR_PRICE not in SIDE_EDGE_FAIR_PRICES:     # a typo falls to the STRICTER rule
    import logging as _logging
    _logging.getLogger(__name__).error(
        f"SIDE_EDGE_FAIR_PRICE={SIDE_EDGE_FAIR_PRICE!r} not in {SIDE_EDGE_FAIR_PRICES}; using devig_two_sided")
    SIDE_EDGE_FAIR_PRICE = "devig_two_sided"
# Human-readable labels for Telegram / dashboard
SIDE_MARKET_LABELS = {
    # "BTTS" alone does not say whether the tip is that both teams DO score or that they do not.
    # The side-market generator is one-sided by construction (it reads p_{market}/odds_{market}),
    # so every BTTS signal is YES. "Over 1.5"/"Over 3.5" already state their side in the name.
    "btts":   "BTTS — YES",
    "over15": "Over 1.5",
    "over35": "Over 3.5",
}

# ── Provider coverage: enabled leagues the odds provider does NOT carry ───────
# An enabled league with no sport key is one of two completely different things, and they need
# opposite responses:
#
#   CONFIG_BROKEN         we mapped it wrong, or forgot to map it. A BUG. Fix the mapping.
#   PROVIDER_UNSUPPORTED  the provider does not sell this competition. Not a bug. Nothing to fix.
#
# Both look identical from outside — the league simply returns zero fixtures forever — which is how
# Ligue 2 stayed dark for three months on an invalid key. Declaring the second case explicitly is
# what lets an audit FAIL the first and merely note the second, instead of either crying wolf or
# staying silent.
#
# Each entry must record the EVIDENCE, so nobody re-litigates it from memory.
# (notes formerly inside the PROVIDER_UNSUPPORTED literal; values now live in config/leagues.json)
# Verified 2026-08-22 against the FULL OddsAPI catalogue (`/sports?all=true`, 67 soccer
# competitions enumerated in the commit message). No Romanian competition exists at any tier,
# under any of: romania / roman / liga_i / liga1 / superliga / super_liga / fcsb / cfr. The only
# "superliga" hit is Denmark. So this is provider coverage, not a naming mismatch, and a key
# cannot be invented for it.
#
# Kept in ENABLED_LEAGUES deliberately rather than removed: the intent to bet it is real, and
# deleting the row would erase the record of why it produces nothing. It has 0 ledger rows and
# has never appeared on a board.
PROVIDER_UNSUPPORTED: dict = _LR.provider_unsupported


def provider_supported(league: str) -> bool:
    """False when the odds provider does not carry this competition at all."""
    return league not in PROVIDER_UNSUPPORTED


def unsupported_reason(league: str) -> str:
    """Machine-readable reason, or "" when the league IS supported."""
    return str(PROVIDER_UNSUPPORTED.get(league, {}).get("reason", ""))


# ── League format classification ──────────────────────────────────────────────
# Standard: historical CSVs contain shots, corners, O/U odds → richer features
# (notes formerly inside the STANDARD_FORMAT_LEAGUES literal; values now live in config/leagues.json)
# Below have API keys but backtest shows negative ROI — training only, not predicted
#   Belgian First Division A: -21.53%
#   Dutch Eredivisie: -3.61%
#   Scottish Premiership: -15.39%
#   Turkish Super Lig: -8.03%
STANDARD_FORMAT_LEAGUES = set(_LR.standard)

# New-format: goals + 1X2 odds only — no shots/corners/O/U odds in history
# (notes formerly inside the NEW_FORMAT_LEAGUES literal; values now live in config/leagues.json)
NEW_FORMAT_LEAGUES = set(_LR.new_format)

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
API_FOOTBALL_ONLY_LEAGUES = set(_LR.api_football_only)

# Historical seasons to backfill for API-Football-only leagues
API_FOOTBALL_EXTRA_SEASONS: dict = _LR.api_football_extra_seasons

# ── Live-prediction leagues (have OddsAPI coverage) ───────────────────────────
# Only leagues listed here generate live betting tips via pipeline.py --mode predict
# (notes formerly inside the ENABLED_LEAGUES literal; values now live in config/leagues.json)
# Not enabled on negative backtest ROI (training only): Greek Super League -7.26%, Belgian First
# Division A -21.53%, Dutch Eredivisie -3.61%, Scottish Premiership -15.39%, Turkish Super Lig -8.03%.
#   USA MLS: PAPER LEAGUE — see PAPER_LEAGUES below. Predicted and collected, never sent, never counted, never bet.
# Saudi Pro League / K-League 1: API-Football-only, training only - enable after ROI validation.
ENABLED_LEAGUES = set(_LR.enabled)

# ── Paper leagues: collect EVERYTHING, send nothing, count nothing, bet nothing ─
#
# A paper league stays in ENABLED_LEAGUES so the whole collection machinery keeps running for
# it — predictions, the prospective shadow log, odds movement and drift, per-book quotes, sharp
# tracking, the live scanner, settlement and CLV. Its tips are generated and written to the
# ledgers exactly like any other league's. What it does NOT do:
#
#   * reach Telegram  — stripped from bets.csv / side_bets.csv (the send sources), and filtered
#                       out of HT, sharp-movement, live and sharp alerts
#   * count in a KPI  — excluded from the daily digest, weekly summary and dashboard
#   * reach the bot   — the Cloudbet candidate builder drops it
#
# WHY NOT SIMPLY DROP IT FROM ENABLED_LEAGUES. That was tried on 2026-10-06 and it stopped far
# more than tips: sharp_tracker and live_scanner both iterate ENABLED_LEAGUES, so MLS movement
# and in-play data stopped being collected too, and with no tips there was no CLV or shadow-log
# record either. Owner instruction 2026-10-07: keep collecting all of it so the league can be
# upgraded later on evidence.
#
# USA MLS — owner decision 2026-10-06/07. Evidence, stated on the STAKED basis (an earlier
# version of this comment quoted all-tier figures, which overstated it):
#   staked since the 2026-08-10 cutoff   72 bets over 20 matchdays, -12.95u,
#                                        hit 31.9% vs 39.2% break-even,
#                                        CI [-0.430, +0.044]  — includes zero, so NOT proven bad
#   diagnosis (scripts/allocation_registry.py --diagnose):
#     ONE_SIDED                 100% of staked O/U bets are UNDER
#     LONGSHOT_DRAG             the 2.80-3.50 band is 85% of the loss on 18 of 69 bets
#     MARKET_PRICES_IT_BETTER   hit rate 5pp below the break-even its prices imply
#   Pro and v9 independently flag it negative (docs/PRO_V9_CROSSCHECK_2026_10_07.md).
# The longshot drag is fixable without a model change — which is why the data is kept.
#
# UPGRADING A PAPER LEAGUE LATER: remove it from this set. Note its paper-period rows are
# already in the ledgers, so they will start counting in KPIs from that moment; decide then
# whether to count from the upgrade date instead.
PAPER_LEAGUES: set = set(_LR.paper)


def is_paper_league(league) -> bool:
    """True for a league that is collected and settled but never sent, counted or bet."""
    return str(league).strip() in PAPER_LEAGUES


def drop_paper_leagues(df, col: str = "league"):
    """Remove paper-league rows from a frame. The ONE filter every send path and KPI uses.

    Kept as a single function so a paper league cannot leak through a path that reimplemented
    the rule slightly differently — the same failure shape as reading thresholds from config.py
    while production read them from somewhere else.
    """
    if df is None or not hasattr(df, "columns") or col not in df.columns or len(df) == 0:
        return df
    return df[~df[col].astype(str).str.strip().isin(PAPER_LEAGUES)]

# ── football-data.co.uk league codes ─────────────────────────────────────────
# (notes formerly inside the FOOTBALL_DATA_LEAGUES literal; values now live in config/leagues.json)
# "Austria Bundesliga" is now a registered ALIAS of "Austrian Bundesliga" (file code AUT), not a second key.
FOOTBALL_DATA_LEAGUES = _LR.football_data

# ── Model ─────────────────────────────────────────────────────────────────────
TARGET_COL  = "over25"
ROLLING_N   = 5

# ── Model recipe switch (Standard forensic repair, 2026-10-28) ────────────────
# "legacy_v9"   = the recipe production trains and serves today, defects included. DEFAULT.
# "v9_repaired" = persisted train-only imputer, one canonical train/serve feature builder
#                 (src/features_repaired.py), league-scoped team stats, four-block chronological
#                 stacking, weights attached to rows, price features selectable.
# RESEARCH ONLY: pipeline.mode_train / predict never read this (they call the legacy defaults);
# it is honoured by the research entry point `python -m src.standard_repair`. Moving production
# to the repaired recipe is an owner-approved serving change, and src/model_registry.payload_bytes
# would first have to persist payload["imputer"].
MODEL_RECIPES = ("legacy_v9", "v9_repaired")
MODEL_RECIPE = os.getenv("WOWZA_MODEL_RECIPE", "legacy_v9").strip() or "legacy_v9"
if MODEL_RECIPE not in MODEL_RECIPES:
    MODEL_RECIPE = "legacy_v9"
# Price-derived features in the repaired recipe: none | asof (entry price only) | closing (legacy).
MARKET_FEATURE_MODES = ("none", "asof", "closing")
MARKET_FEATURE_MODE = os.getenv("WOWZA_MARKET_FEATURE_MODE", "none").strip() or "none"
if MARKET_FEATURE_MODE not in MARKET_FEATURE_MODES:
    MARKET_FEATURE_MODE = "none"

# ── Standard O/U 2.5 SERVING recipe (owner 2026-10-28: "switch Standard now") ──
# The ONE production switch. Scope: the over25 probability of STANDARD-format fixtures only.
#   "legacy_v9"   (DEFAULT, and the rollback) models/model_v9_standard.pkl on legacy features.
#   "v9_repaired" models/model_v9_standard_repaired.pkl on repaired as-of features
#                 (src/standard_serving.py: frozen identity alias map, train-only imputer).
# New Format, BTTS / O1.5 / O3.5 side models and HT models are NOT affected by either value.
# Deployed value = predict.yml env (CLAUDE.md trap 1).
# INVALID VALUES ARE A CONFIGURATION ERROR (R2 finding 17, 2026-11-01). They used to fall back to
# legacy_v9 silently, so a typo in predict.yml quietly reverted Standard to the legacy model. The
# raw value is kept in STANDARD_OU25_SERVING_RECIPE_RAW and src.standard_serving.check_config()
# raises ServingConfigError on it before predict fetches odds. Importing config never raises (the
# dashboard, notifier and results jobs must keep working); only predict refuses to start.
# ROLLBACK: set STANDARD_OU25_SERVING_RECIPE: "legacy_v9" in predict.yml (or delete the line).
# Owner doc: docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md
STANDARD_OU25_SERVING_RECIPES = ("legacy_v9", "v9_repaired")
STANDARD_OU25_SERVING_RECIPE_EXPLICIT = bool(os.getenv("STANDARD_OU25_SERVING_RECIPE", "").strip())
STANDARD_OU25_SERVING_RECIPE_RAW = os.getenv("STANDARD_OU25_SERVING_RECIPE", "legacy_v9").strip() or "legacy_v9"
STANDARD_OU25_SERVING_RECIPE = STANDARD_OU25_SERVING_RECIPE_RAW
if STANDARD_OU25_SERVING_RECIPE not in STANDARD_OU25_SERVING_RECIPES:
    import logging as _logging
    _logging.getLogger(__name__).error(
        f"STANDARD_OU25_SERVING_RECIPE={STANDARD_OU25_SERVING_RECIPE_RAW!r} is INVALID "
        f"(expected {STANDARD_OU25_SERVING_RECIPES}); predict will refuse to start")
    STANDARD_OU25_SERVING_RECIPE = "INVALID"

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

# ── DRIFT TIER ADJUSTMENT: REMOVED (owner decision 2026-10-26, "yes") ─────────────────────────
# The rule in src/betting._apply_drift_adjustment (SNIPER+Conflicted -> MARKSMAN, MARKSMAN+Confirmed
# -> SNIPER when edge >= DRIFT_UPGRADE_EDGE, VALUABLE+Confirmed -> MARKSMAN) was never validated and
# points the wrong way on settled live O/U tips 2026-04-26..10-25 (matchday block bootstrap):
# Confirmed (upgraded) n=135 ROI -9.1% [-28.9, +13.5]; Neutral n=265 +13.6% [+1.9, +24.6];
# Conflicted (downgraded) n=265 -12.6%; New n=604 -11.1%; Confirmed-Neutral -22.7 pts [-44.5, +3.5].
# Production behaviour needs evidence to be ON, so tiers are now computed exactly as without drift.
# drift_signal is still computed and written as INFORMATION, and the tier the old rule WOULD have
# produced is recorded as drift_shadow_tier (predictions.csv + first-sight output/drift_shadow.csv).
#
# A PLAIN CONSTANT ON PURPOSE, NOT os.getenv: no workflow can switch it back on silently, and a
# DRIFT_TIER_ADJUSTMENT environment variable is IGNORED with a warning (src/betting.py). Returning a
# drift rule requires a frozen forward test in Pro + owner approval, then a reviewed edit here.
# See docs/DEPRECATION_REGISTER.md.
DRIFT_TIER_ADJUSTMENT   = False
DRIFT_UPGRADE_EDGE      = 0.10    # the removed rule's MARKSMAN->SNIPER edge bar; used only by the shadow

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
# (notes formerly inside the LEAGUE_TIMEZONES literal; values now live in config/leagues.json)
#   USA MLS: MLS spans zones; ET as display proxy
LEAGUE_TIMEZONES: dict = _LR.timezones

KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
FLAT_STAKE     = float(os.getenv("FLAT_STAKE",     "1.0"))
USE_KELLY      = os.getenv("USE_KELLY", "0") == "1"

MIN_OVER_ODDS  = float(os.getenv("MIN_OVER_ODDS",  "1.75"))
MIN_UNDER_ODDS = float(os.getenv("MIN_UNDER_ODDS", "1.75"))

# NEW-FORMAT OU25 PRICE CAP (owner decision 2026-10-22: "yes, but keep collect"). New-format O/U 2.5
# tips priced above this are not sent. Evidence: odds > 3.00 were 6% of settled bets and 35% of the
# net loss (ROI -53.9%, CI [-80.7, -20.6]); on the all-fixture record -51.1% [-79.4, -21.1] n47
# (Pro docs/WOWZA_EDGE_REASSESSMENT_2026_10.md). NEW-FORMAT TRACK ONLY - standard is untouched
# (invariant 1). The fixture is still scored, written to predictions.csv and both shadow logs, so
# collection continues; it just never becomes a tip. Set to 0 to disable.
NF_MAX_OU_ODDS = float(os.getenv("NF_MAX_OU_ODDS", "3.00"))

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
# DEPRECATED (R2 finding 6, 2026-11-01). Was os.getenv("API_SEASON", "2025"): one generic season
# for every league, defaulting to a FINISHED season, and the silent fallback of four api_football_ou
# call sites plus feature_engineering / update_results. No code reads it any more; every call site
# asks src.season_resolver.resolve_api_season(league, kickoff_date) and skips (recording the
# degradation) on SEASON_UNKNOWN. The name is kept, set to the sentinel, so a forgotten consumer
# sends an obviously invalid season instead of a plausible wrong one.
API_SEASON   = _season_resolver.SEASON_UNKNOWN

# (notes formerly inside the API_FOOTBALL_IDS literal; values now live in config/leagues.json)
# WAS 65 — the defunct Coupe de la Ligue, not Ligue 2. Fixed 2026-10-08: with 65 every
# API-Football consumer (per-book odds capture, side-market capture, xG/shot enrichment,
# results fallback, live scanner) asked for a competition with no fixtures, so Ligue 2 had
# zero per-book quotes and no side-market odds since 2026-05-09 while 9 fixtures sat on the
# board. player_model/config.py had the same bug and was fixed to 62 earlier; this copy was
# missed. Verified on /leagues: Ligue 1 = 61, Ligue 2 = 62, Ligue 3 = 63.
# New format — European winter/spring
# New format — summer / Americas / Asia
# API-Football-only leagues (not on FD)
# UEFA club competitions (added 2026-09-05). COLLECTION ONLY — deliberately NOT added to
# ENABLED_LEAGUES or ODDS_API_SPORT_KEYS, and that ordering is the whole point.
#
# There is currently ZERO history for these competitions: af_history holds 15 leagues and
# none of them is UEFA, and football-data.co.uk does not cover them at all. A fixture with no
# rolling-form history is refused by invariant 8 (`no_form_data` -> AVOID), and rightly so —
# median-imputed features produce a confident-looking wrong edge, which is how York City once
# showed the second-strongest EFL edge on the board built on nothing.
#
# So wiring these into predict TODAY would spend OddsAPI credits — the pool that was exhausted
# on 09-04 and took every real tip down with it — to generate AVOID rows. Adding them here
# instead collects fixtures, results and team stats through API-Football, which has roughly
# 60,000/day of unused headroom, and BUILDS the history that any future model needs.
#
# Pricing and prediction come later, once there is a season of history to train on and the
# OddsAPI plan has proven stable across a full billing cycle.
# TOP DIVISIONS — COLLECTION ONLY, and they are the missing input for the tournaments above.
#
# A European tie is decided by the two clubs' CURRENT DOMESTIC FORM, not by their history in
# that competition; the tournament's own past carries almost no weight when the sides come
# from different leagues and meet twice a decade. That framing is right, and it is why the
# blocker here was never "no UEFA history" — it is that we do not collect the leagues those
# clubs actually play in. Measured against the real 2026 fixture lists:
#
# Champions League   81 clubs,  12 with domestic form (15%)
# Europa League      76 clubs,  17 (22%)
# Conference League 165 clubs,  26 (16%)
#
# and a fixture needs BOTH sides, so joint coverage was ~2%. Our 23 leagues are second
# divisions — Championship, Bundesliga 2, Ligue 2, La Liga 2, Serie B — while UEFA clubs are
# top-flight. Arsenal, Atletico, Milan and Ajax simply had no form with us.
#
# Every id below was READ FROM THE API (/leagues, season 2026, matched on country + name),
# not typed from memory. Greece is absent only because "Greek Super League" is already above.
#
# Cost is one-time and affordable: ~6,500 finished fixtures across these leagues at roughly
# one /fixtures/statistics call each, against ~60,000/day of unused API-Football headroom.
# None of them is added to ENABLED_LEAGUES or ODDS_API_SPORT_KEYS, so this spends no OddsAPI
# credit and produces no tips — it builds the form table the tournaments need.
# THE LAST TWO football-data.co.uk-ONLY TRAINING LEAGUES (added 2026-09-05).
#
# I claimed five training leagues had "no API-Football id" and were therefore a hard blocker
# on ever dropping football-data.co.uk. That was wrong, and it was wrong in a lazy way: I
# compared football-data's SPELLINGS against our config keys instead of asking the API.
# Three of the five were already here under different names — Belgian First Division A is
# `Belgian Pro League` (144), Dutch Eredivisie is `Eredivisie` (88), Portuguese Primeira Liga
# is `Primeira Liga` (94). Only these two were genuinely absent.
#
# Read from /leagues?country=... on the live API, not guessed. With them, every one of the 27
# leagues football-data supplies is coverable by API-Football, so the free source is no longer
# structurally required for CURRENT-season results or stats — only for HISTORICAL closing
# odds, which no API-Football plan sells.
# Added 2026-10-14 (owner OK). Read from /leagues?country=Scotland on the live API and checked
# against /teams for 2026: 183 = Hamilton, Alloa, Cove Rangers, Queen of the South...;
# 184 = Clyde, Stranraer, Elgin, Spartans... Both are standard-format TRAINING leagues (not in
# ENABLED_LEAGUES, no OddsAPI key), so this adds results/stats/forward side-market capture and
# produces no tips. Closes the last two "model league with no API-Football id" registry gaps.
API_FOOTBALL_IDS = _LR.api_football_ids
# IDs may need verification at dashboard.api-football.com/docs

# Per-league API-Football season, RESOLVED (R2 finding 6) - no longer hand-rolled each summer.
# src/season_resolver applies each league's season_rule from config/leagues.json to TODAY (UTC) at
# import: start_year leagues roll on 1 July, calendar_year leagues on 1 January, Japan J1 keys by
# END year from 2026-07-01. On 2026-11-01 it reproduces the old hand table exactly (tested). Leagues
# whose season cannot be resolved are ABSENT, never defaulted. Code that has a fixture date should
# call resolve_api_season(league, kickoff_date) directly instead of reading this table.
def _derived_api_football_seasons() -> dict:
    out = {}
    for _e in _league_registry.leagues():
        if _e.get("api_football_id") in (None, ""):
            continue
        _s = _season_resolver.resolve_api_season(_e["name"])
        if _season_resolver.is_known(_s):
            out[_e["name"]] = _s
    return _league_registry.LeagueMap(out, _LR.aliases)


# (notes formerly inside the API_FOOTBALL_SEASONS literal; values now live in config/leagues.json)
# Standard format — 2026/27 European season (API-Football season = start year 2026).
# ROLLED 2025->2026 on 2026-08-05: the 2025/26 season ended (~May 2026); on "2025" the
# capture looked in the finished season and found NO upcoming fixtures -> zero standard
# odds/1X2/HT/CLV collection for the whole 2026/27 season. New-format was already 2026.
# THE LEAGUE IS NAMED "Denmark Superliga" everywhere else (ENABLED_LEAGUES, ids, sport keys), so
# the "Danish Superliga" key above never matched and every lookup fell back to API_SEASON
# ("2025", finished 2026-05-21): Denmark results and odds could not be fetched from API-Football
# (found 2026-10-22 settling the v91 shadow; owner-approved fix). API current = 2026.
# Scottish lower tiers (2026-10-14). Without an entry a lookup falls back to API_SEASON
# ("2025" = the finished season) and finds nothing current. Scottish Championship had an id
# since 2026-09-05 but no entry here, so it carried that gap too.
# National League (id 43) had the same gap since 2026-09-05; API says current = 2026 (Aug 8 -> Apr 24).
# J1 MOVED TO AN AUTUMN-SPRING CALENDAR: API-Football season 2026 is the spring special
# tournament (2026-02-06 -> 06-06); the live league is season 2027 (2026-08-07 -> 2027-06-06),
# which the API marks current (verified 2026-10-22; owner-approved fix).
#   Saudi Pro League: 2026/27 season starts Aug 2026
#   K-League 1: calendar year 2026
API_FOOTBALL_SEASONS: dict = _derived_api_football_seasons()

# (notes formerly inside the ODDS_API_SPORT_KEYS literal; values now live in config/leagues.json)
ODDS_API_SPORT_KEYS = _LR.odds_api_sport_keys

SOFASCORE_TOURNAMENT_IDS = _LR.sofascore_ids


# ── AUTO-ONBOARDED LEAGUES (owner standing approval 2026-10-20) ─────────────────────────────────
# "If a new league is ready to become a tips generator, go." A league that passes EVERY League Scout
# gate in Pro (data quality, market-residual evidence, existing-league regression harness) is written
# into config/auto_leagues.json by .github/workflows/auto_onboard_leagues.yml (scripts/auto_onboard.py).
# It is merged here into every registry a new-format, API-Football-only league needs. ADDITIVE ONLY: a
# name already configured above is never overwritten, and any error leaves the config exactly as
# written above. Stakes stay protected by the league rating, which fails closed to the minimum scale
# for a league with no rating yet.
AUTO_LEAGUES_FILE = BASE_DIR / "config" / "auto_leagues.json"


def _merge_auto_leagues() -> list[str]:
    import json as _json
    added = []
    try:
        data = _json.loads(AUTO_LEAGUES_FILE.read_text(encoding="utf-8")) if AUTO_LEAGUES_FILE.exists() else {}
    except Exception:                                                 # noqa: BLE001
        return added
    for e in data.get("leagues", []):
        name = str(e.get("name", "")).strip()
        try:
            lid, season, sport_key = int(e["api_football_id"]), str(e["current_season"]), str(e["odds_api_sport_key"])
        except Exception:                                             # noqa: BLE001 - incomplete entry: skip
            continue
        if not name or name in STANDARD_FORMAT_LEAGUES or name in NEW_FORMAT_LEAGUES or name in API_FOOTBALL_IDS:
            continue
        NEW_FORMAT_LEAGUES.add(name)
        API_FOOTBALL_ONLY_LEAGUES.add(name)
        API_FOOTBALL_EXTRA_SEASONS[name] = [str(s) for s in e.get("history_seasons", [])]
        ENABLED_LEAGUES.add(name)
        API_FOOTBALL_IDS[name] = lid
        API_FOOTBALL_SEASONS[name] = season
        ODDS_API_SPORT_KEYS[name] = sport_key
        LEAGUE_TIMEZONES[name] = str(e.get("timezone") or "UTC")
        added.append(name)
    return added


AUTO_LEAGUES = _merge_auto_leagues()
