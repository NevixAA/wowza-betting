"""
Live Game Scanner — v9
======================
Scans in-progress matches from our enabled leagues and finds live value bets
using Poisson statistics + pre-match model predictions.

No live odds API needed — we calculate the FAIR live price and alert the user
to check their bookmaker's live screen.

Full-time signal types:
  UNDER_HOLD      — 0-0 or low score at 60+ min, model said UNDER, still good value
  SLEEPING_GAME   — two low-scoring teams, 0-0 at 75+ min, UNDER almost locked
  UNDER_RECOVERY  — score is 1-1 or 2-0, time left, Poisson says UNDER still value
  STRONG_STUCK    — high attack-strength team not scoring, pushing for goals → OVER value
  COMEBACK        — team losing at 60+ min has strong attack → OVER / draw value

Half-time signal types (first half only, zero extra API calls):
  HT_UNDER_0.5    — 0-0 at 35+ min, barely any time left, lock UNDER 0.5 HT
  HT_UNDER_1.5    — low score at 25+ min, UNDER 1.5 HT fair price attractive
  HT_OVER_0.5     — 0-0 at 20-38 min, strong attack teams, OVER 0.5 HT still live
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

LIVE_TIPS_FILE    = config.OUTPUT_DIR / "live_tips.csv"
LIVE_GAMES_FILE   = config.OUTPUT_DIR / "live_games.csv"
LIVE_HISTORY_FILE = config.OUTPUT_DIR / "live_signals_history.csv"
LIVE_NOTIFIED     = Path(__file__).resolve().parents[1] / "telegram_bot" / "live_notified.json"
INPLAY_LOG_FILE   = config.OUTPUT_DIR / "inplay_snapshots.csv"   # v2: Phase-1 collection

# v2 live-lambda blend constants. SOT_TO_GOAL is EMPIRICAL (our parquet: goals/SOT = 0.311,
# corr(SOT,goals)=0.58). K + the game-state multipliers are HEURISTIC PLACEHOLDERS — to be
# FITTED from the in-play snapshots we start collecting (Phase 2). Do not treat as calibrated.
SOT_TO_GOAL   = 0.311
BLEND_K_MINS  = 30.0

MIN_FAIR_UNDER    = 1.28   # only alert if fair UNDER odds >= this (meaningful value)
MIN_FAIR_OVER     = 2.00   # only alert OVER if fair odds >= this
MIN_ELAPSED       = 45     # don't alert before half-time
MIN_LIVE_EDGE     = 0.12   # 12% edge threshold for live alerts (higher bar than pre-match)
ATTACK_STR_HIGH   = 1.25   # threshold for "strong attack" signal

# Half-time signal thresholds
HT_MIN_ELAPSED    = 20     # min first-half minutes for HT signals
HT_LOCK_ELAPSED   = 35     # min minutes for HT UNDER 0.5 lock signal
HT_MIN_FAIR_UNDER = 1.18   # fair HT UNDER price must be >= this
HT_MIN_FAIR_OVER  = 1.55   # fair HT OVER price must be >= this

# Shot gate — SLEEPING_GAME suppressed when combined shots >= this (busy 0-0 is not sleeping)
SHOT_GATE_THRESHOLD       = 10
# Goal recency — suppress UNDER signals if goal scored in last N minutes
GOAL_RECENCY_SUPPRESS_MINS = 3

IDLE_RECHECK_SECS   = 1800  # re-check a league with no live games every 30 min
ACTIVE_RECHECK_SECS = 120   # re-check a league with live games every 2 min

# Per-league cache — persists between calls when imported inline by master_loop
_league_last_active:  dict = {}  # league -> datetime of last live game found
_league_last_checked: dict = {}  # league -> datetime of last API call

# API-Football live scan throttle (single call replaces per-league OddsAPI calls)
_last_apifootball_scan: Optional[datetime] = None


# ── Poisson helpers ───────────────────────────────────────────────────────────

def _poisson_prob(k: int, lam: float) -> float:
    """P(X = k) for Poisson with rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for Poisson with rate lam."""
    return sum(_poisson_prob(i, lam) for i in range(k + 1))


def _lambda_from_p_over(p_over: float) -> float:
    """
    Numerically invert Poisson CDF to find lambda (expected total goals)
    given pre-match P(over 2.5).
    P(X <= 2) = 1 - p_over → solve for lambda.
    """
    p_under = max(0.05, min(0.95, 1 - p_over))
    # Binary search for lambda
    lo, hi = 0.01, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _poisson_cdf(2, mid) > p_under:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def _live_lambda_remaining(prior_lambda: float, elapsed: float, live_sot: float,
                           home_g: int, away_g: int) -> float:
    """v2 — LIVE-adjusted expected goals for the REMAINING time. Blends our pre-match prior
    with in-play evidence (accumulated shots-on-target), time-weighted, plus a game-state
    multiplier. This is the upgrade over the naive `prior * remaining/90` (which ignores the
    live match). Grounded in SOT_TO_GOAL=0.311 (our data); K + multipliers are heuristic until
    fitted on collected in-play snapshots.

      r_prior = prior/90                         pre-match scoring rate per minute
      r_live  = (0.311 * SOT_so_far) / elapsed   in-play rate implied by shots on target
      w       = elapsed / (elapsed + K)          trust live more as the game unfolds
      r_blend = w*r_live + (1-w)*r_prior
      lambda_remaining = r_blend * remaining * game_state_multiplier
    """
    elapsed = max(1.0, min(float(elapsed), 90.0))
    remaining = max(0.0, 90.0 - elapsed)
    r_prior = max(prior_lambda, 0.05) / 90.0
    r_live = (SOT_TO_GOAL * float(live_sot or 0)) / elapsed
    w = elapsed / (elapsed + BLEND_K_MINS)
    r_blend = w * r_live + (1.0 - w) * r_prior
    margin = abs(int(home_g) - int(away_g))
    if remaining <= 30 and margin >= 2:
        gs = 0.85          # comfortable lead late -> game slows
    elif remaining <= 30 and margin == 1:
        gs = 1.12          # one-goal game late -> chasing side pushes
    elif margin == 0 and remaining <= 20:
        gs = 0.92          # cagey late level game
    else:
        gs = 1.0
    return max(0.05, r_blend * remaining * gs)


def _live_probs(goals_scored: int, elapsed_mins: float, lam_total: float,
                lam_remaining_override: float | None = None) -> dict:
    """
    Calculate live probabilities given current state.
    Returns dict with p_under, p_over, fair_under_odds, fair_over_odds.
    """
    remaining_frac = max(0.0, (90.0 - elapsed_mins) / 90.0)
    # v2: use the live-adjusted remaining lambda when supplied; else the naive time-decay.
    lam_remaining  = (lam_remaining_override if lam_remaining_override is not None
                      else lam_total * remaining_frac)

    goals_for_over = max(0, 3 - goals_scored)  # goals still needed to go OVER 2.5

    if goals_scored >= 3:
        # Already over 2.5 — no UNDER possible
        return {"p_under": 0.0, "p_over": 1.0,
                "fair_under_odds": 99.0, "fair_over_odds": 1.01,
                "lam_remaining": round(lam_remaining, 3)}

    # P(remaining goals < goals_for_over) = P(X <= goals_for_over - 1)
    p_under = _poisson_cdf(goals_for_over - 1, lam_remaining)
    p_over  = 1.0 - p_under

    fair_under = round(1.0 / max(p_under, 0.01), 2)
    fair_over  = round(1.0 / max(p_over,  0.01), 2)

    return {
        "p_under":        round(p_under, 4),
        "p_over":         round(p_over,  4),
        "fair_under_odds": fair_under,
        "fair_over_odds":  fair_over,
        "lam_remaining":   round(lam_remaining, 3),
    }


def _ht_probs(ht_goals: int, elapsed_h1: float, lam_total: float) -> dict:
    """
    Calculate fair HT O/U prices for lines 0.5, 1.5, 2.5.
    Uses first-half lambda = lam_total / 2.
    Only valid when elapsed_h1 < 45 (first half).
    """
    lam_ht         = lam_total / 2
    remaining_frac = max(0.0, (45.0 - elapsed_h1) / 45.0)
    lam_remaining  = lam_ht * remaining_frac
    result         = {"lam_ht_remaining": round(lam_remaining, 3)}

    for line in [0.5, 1.5, 2.5]:
        threshold = int(line)  # 0, 1, 2
        if ht_goals > threshold:
            p_under, p_over = 0.0, 1.0
        else:
            goals_needed = threshold + 1 - ht_goals
            p_under = _poisson_cdf(goals_needed - 1, lam_remaining)
            p_over  = 1.0 - p_under
        line_key = str(line).replace(".", "")
        result[f"ht_p_under_{line_key}"] = round(p_under, 4)
        result[f"ht_p_over_{line_key}"]  = round(p_over,  4)
        result[f"ht_fair_under_{line_key}"] = round(1.0 / max(p_under, 0.01), 2)
        result[f"ht_fair_over_{line_key}"]  = round(1.0 / max(p_over,  0.01), 2)

    return result


# ── Live scores fetch ─────────────────────────────────────────────────────────

def _leagues_with_games_today() -> set[str]:
    """Return leagues that have at least one prediction for today (saves API credits)."""
    try:
        preds = pd.read_csv(config.OUTPUT_DIR / "predictions.csv")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return set(preds[preds["date"].astype(str).str[:10] == today]["league"].unique())
    except Exception:
        return set(config.ENABLED_LEAGUES)  # fallback: scan all


def _fetch_live_scores_oddsapi() -> list[dict]:
    """Fetch in-progress scores — only for leagues with games today that are due a check."""
    live = []
    seen = set()
    now = datetime.now(timezone.utc)

    active_leagues = _leagues_with_games_today()

    # Always include World Cup when it's active (June 11 – July 19, 2026)
    wc_leagues = {"World Cup 2026": "soccer_fifa_world_cup"}
    scan_map = {**{l: k for l, k in config.ODDS_API_SPORT_KEYS.items()
                   if l in config.ENABLED_LEAGUES}, **wc_leagues}

    for league, sport_key in scan_map.items():
        if league not in active_leagues and league not in wc_leagues:
            continue  # no games today in this league

        last_checked = _league_last_checked.get(league)
        last_active  = _league_last_active.get(league)

        if last_checked is not None:
            secs_since_check = (now - last_checked).total_seconds()
            recently_active  = last_active and (now - last_active).total_seconds() < ACTIVE_RECHECK_SECS
            wait = ACTIVE_RECHECK_SECS if recently_active else IDLE_RECHECK_SECS
            if secs_since_check < wait:
                continue  # too soon to re-check this league

        _league_last_checked[league] = now
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores",
                params={"apiKey": config.ODDS_API_KEY, "daysFrom": 1},
                timeout=15,
            )
            if r.status_code != 200:
                continue

            for ev in r.json():
                if ev.get("completed"):
                    continue  # skip finished games

                commence = pd.to_datetime(ev.get("commence_time", ""), errors="coerce", utc=True)
                if pd.isna(commence):
                    continue

                now_utc = datetime.now(timezone.utc)
                wall_mins = (now_utc - commence).total_seconds() / 60.0

                if wall_mins < 1:
                    continue  # not started yet

                # Approximate game clock: subtract ~15min halftime break if in 2nd half
                # Wall clock > 60min likely means halftime has passed
                HT_BREAK = 15  # typical halftime break duration in minutes
                elapsed = wall_mins - HT_BREAK if wall_mins > 60 else wall_mins
                elapsed = max(0.0, min(elapsed, 95.0))

                scores = ev.get("scores") or []
                home_goals = away_goals = 0
                for s in scores:
                    try:
                        goals = int(s.get("score", 0))
                    except (ValueError, TypeError):
                        goals = 0
                    if s["name"] == ev["home_team"]:
                        home_goals = goals
                    else:
                        away_goals = goals

                key = f"{ev['home_team']}|{ev['away_team']}"
                if key in seen:
                    continue
                seen.add(key)

                _league_last_active[league] = now  # mark league as active right now

                live.append({
                    "league":      league,
                    "home_team":   ev["home_team"],
                    "away_team":   ev["away_team"],
                    "commence":    commence,
                    "elapsed_mins": min(round(elapsed), 95),
                    "home_goals":  home_goals,
                    "away_goals":  away_goals,
                    "total_goals": home_goals + away_goals,
                })

        except Exception as e:
            log.debug(f"Scores fetch {league}: {e}")

    return live


def _fetch_live_scores_apifootball() -> list[dict]:
    """
    Fetch live fixtures using API-Football /fixtures?live=all.
    One call returns ALL live fixtures — no per-league throttling needed.
    """
    global _last_apifootball_scan

    now = datetime.now(timezone.utc)
    if _last_apifootball_scan is not None:
        secs_since = (now - _last_apifootball_scan).total_seconds()
        # Reuse result within scan interval (api_football.py has its own 90s cache)
        if secs_since < ACTIVE_RECHECK_SECS:
            pass  # fall through to API call; caching handled in api_football.py

    _last_apifootball_scan = now

    try:
        from src.api_football import get_live_fixtures
    except ImportError:
        return []

    league_ids = list(config.API_FOOTBALL_IDS.values())
    id_to_name = {v: k for k, v in config.API_FOOTBALL_IDS.items()}

    active_leagues = _leagues_with_games_today()
    raw = get_live_fixtures(league_ids=league_ids)

    live = []
    seen: set[str] = set()

    for fix in raw:
        league_id   = fix.get("league_id")
        league_name = id_to_name.get(league_id, "")
        if not league_name:
            continue
        if league_name not in active_leagues:
            continue

        home   = fix["home_team"]
        away   = fix["away_team"]
        key    = f"{home}|{away}"
        if key in seen:
            continue
        seen.add(key)

        status  = fix.get("status", "1H")
        elapsed = fix.get("elapsed_mins") or 0

        # During halftime, treat elapsed as 45 so HT signals evaluate correctly
        if status == "HT":
            elapsed = 45

        _league_last_active[league_name] = now

        live.append({
            "fixture_id":   fix.get("fixture_id"),
            "league":       league_name,
            "home_team":    home,
            "away_team":    away,
            "elapsed_mins": elapsed,
            "home_goals":   fix.get("home_goals", 0),
            "away_goals":   fix.get("away_goals", 0),
            "total_goals":  fix.get("home_goals", 0) + fix.get("away_goals", 0),
            "status":       status,
        })

    return live


def _fetch_live_scores() -> list[dict]:
    """Dispatch to API-Football (primary) or OddsAPI (fallback)."""
    try:
        from src.api_football import _APIFOOTBALL_KEY
        if _APIFOOTBALL_KEY:
            return _fetch_live_scores_apifootball()
    except ImportError:
        pass
    return _fetch_live_scores_oddsapi()


# ── Pre-match prediction lookup ───────────────────────────────────────────────

def _load_predictions() -> pd.DataFrame:
    f = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _match_prediction(pred_df: pd.DataFrame, home: str, away: str) -> dict | None:
    """Find pre-match prediction row for this fixture (fuzzy name match)."""
    if pred_df.empty:
        return None

    # Exact match first
    mask = (pred_df["home_team"] == home) & (pred_df["away_team"] == away)
    if mask.any():
        return pred_df[mask].iloc[0].to_dict()

    # Fuzzy: first 8 chars
    h8, a8 = home[:8].lower(), away[:8].lower()
    mask2 = (pred_df["home_team"].str[:8].str.lower() == h8) & \
            (pred_df["away_team"].str[:8].str.lower() == a8)
    if mask2.any():
        return pred_df[mask2].iloc[0].to_dict()

    return None


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_signals(live_games: list[dict], pred_df: pd.DataFrame,
                    sot_by_fixture: dict | None = None) -> list[dict]:
    tips = []

    for game in live_games:
        elapsed   = game["elapsed_mins"]
        total_g   = game["total_goals"]
        home_g    = game["home_goals"]
        away_g    = game["away_goals"]
        home      = game["home_team"]
        away      = game["away_team"]
        league    = game["league"]

        pred = _match_prediction(pred_df, home, away)
        if pred is None:
            # WC teams are not in predictions.csv — use neutral defaults so Poisson signals still fire
            if "World Cup" not in league and "FIFA" not in league:
                continue
            pred = {}

        p_over     = float(pred.get("p_over25", 0.45))
        lam        = _lambda_from_p_over(p_over)
        # v2: live-adjust the remaining lambda from in-play SOT (falls back to naive if no SOT)
        _sot_live  = (sot_by_fixture or {}).get(game.get("fixture_id"))
        _lam_rem   = (_live_lambda_remaining(lam, elapsed, _sot_live, home_g, away_g)
                      if _sot_live is not None else None)
        probs      = _live_probs(total_g, elapsed, lam, lam_remaining_override=_lam_rem)
        pre_signal = str(pred.get("signal_tier", "AVOID"))

        home_atk   = float(pred.get("home_attack_str", 1.0) or 1.0)
        away_atk   = float(pred.get("away_attack_str", 1.0) or 1.0)
        league_avg = float(pred.get("league_avg_goals", 2.5) or 2.5)

        # ── Pre-compute shot gate and goal recency ────────────────────────────
        fixture_id   = game.get("fixture_id")
        shot_gate_ok = True   # True means SLEEPING_GAME is allowed
        recent_goal  = False  # True means suppress UNDER signals

        if fixture_id and elapsed >= 60 and total_g == 0:
            try:
                from src.api_football import get_fixture_statistics
                stats = get_fixture_statistics(fixture_id, cache_hours=0.025)
                if stats:
                    total_shots = (
                        (stats.get("home") or {}).get("shots", 0) +
                        (stats.get("away") or {}).get("shots", 0)
                    )
                    if total_shots >= SHOT_GATE_THRESHOLD:
                        shot_gate_ok = False  # high-shot 0-0 is not sleeping
            except Exception:
                pass

        if fixture_id and elapsed >= MIN_ELAPSED and total_g <= 2:
            try:
                from src.api_football import get_fixture_events, last_goal_elapsed
                events     = get_fixture_events(fixture_id)
                last_g_min = last_goal_elapsed(events)
                if last_g_min is not None and (elapsed - last_g_min) <= GOAL_RECENCY_SUPPRESS_MINS:
                    recent_goal = True
            except Exception:
                pass

        base = {
            "league":           league,
            "match":            f"{home} vs {away}",
            "score":            f"{home_g}-{away_g}",
            "elapsed_mins":     elapsed,
            "pre_signal":       pre_signal,
            "pre_p_over":       round(p_over, 3),
            "lambda":           lam,
            "live_p_under":     probs["p_under"],
            "live_p_over":      probs["p_over"],
            "fair_under_odds":  probs["fair_under_odds"],
            "fair_over_odds":   probs["fair_over_odds"],
            "lam_remaining":    probs["lam_remaining"],
            "updated_at":       datetime.now().strftime("%H:%M"),
        }

        # ── Signal 1: UNDER HOLD ─────────────────────────────────────────────
        # Model said UNDER, game still low-scoring, time running out
        if (elapsed >= MIN_ELAPSED
                and total_g <= 1
                and not recent_goal
                and p_over < 0.45
                and probs["fair_under_odds"] >= MIN_FAIR_UNDER
                and probs["p_under"] > (1 - p_over) * 1.1):  # live prob improved vs pre-match
            tips.append({**base,
                "signal_type": "UNDER_HOLD",
                "bet":         "UNDER 2.5",
                "reason":      f"Model said UNDER ({(1-p_over)*100:.0f}% pre-match). "
                               f"Score {home_g}-{away_g} at {elapsed}min. "
                               f"Live P(UNDER)={probs['p_under']*100:.0f}%. "
                               f"Fair price: {probs['fair_under_odds']}",
            })

        # ── Signal 2: SLEEPING GAME ──────────────────────────────────────────
        # Both teams have low attack, 0-0 late → UNDER almost certain
        elif (elapsed >= 70
                and total_g == 0
                and shot_gate_ok
                and home_atk < 1.1 and away_atk < 1.1
                and probs["fair_under_odds"] >= MIN_FAIR_UNDER):
            tips.append({**base,
                "signal_type": "SLEEPING_GAME",
                "bet":         "UNDER 2.5",
                "reason":      f"Both teams low attack (H:{home_atk:.2f} A:{away_atk:.2f}). "
                               f"0-0 at {elapsed}min. "
                               f"Live P(UNDER)={probs['p_under']*100:.0f}%. "
                               f"Fair price: {probs['fair_under_odds']}",
            })

        # ── Signal 3: UNDER RECOVERY ─────────────────────────────────────────
        # Score is 2-0 or 1-1 (needs 1 more for over), still 30+ mins left
        elif (total_g == 2
                and elapsed <= 65
                and probs["fair_under_odds"] >= MIN_FAIR_UNDER + 0.15
                and p_over < 0.50):
            tips.append({**base,
                "signal_type": "UNDER_RECOVERY",
                "bet":         "UNDER 2.5",
                "reason":      f"Score {home_g}-{away_g} at {elapsed}min (1 goal from over). "
                               f"Live P(UNDER)={probs['p_under']*100:.0f}%. "
                               f"Fair price: {probs['fair_under_odds']}",
            })

        # ── Signal 4: STRONG TEAM STUCK ──────────────────────────────────────
        # High-attack team not scoring, time running out → pushing harder → OVER
        elif (elapsed >= 55
                and elapsed <= 80
                and total_g <= 1
                and (home_atk >= ATTACK_STR_HIGH or away_atk >= ATTACK_STR_HIGH)
                and probs["fair_over_odds"] >= MIN_FAIR_OVER):
            stronger = home if home_atk >= away_atk else away
            atk_val  = max(home_atk, away_atk)
            tips.append({**base,
                "signal_type": "STRONG_STUCK",
                "bet":         "OVER 2.5",
                "reason":      f"{stronger} (attack str {atk_val:.2f}) not scoring at {elapsed}min. "
                               f"Score {home_g}-{away_g}. Expect pressure. "
                               f"Fair OVER price: {probs['fair_over_odds']}",
            })

        # ── Signal 5: COMEBACK ───────────────────────────────────────────────
        # Team losing at 60+ min with strong attack → push for goals → OVER value
        elif (elapsed >= 60
                and elapsed <= 82
                and total_g <= 2):
            if home_g < away_g and home_atk >= ATTACK_STR_HIGH:
                tips.append({**base,
                    "signal_type": "COMEBACK",
                    "bet":         "OVER 2.5",
                    "reason":      f"{home} losing {home_g}-{away_g} at {elapsed}min "
                                   f"with attack str {home_atk:.2f}. Expect push. "
                                   f"Fair OVER price: {probs['fair_over_odds']}",
                })
            elif away_g < home_g and away_atk >= ATTACK_STR_HIGH:
                tips.append({**base,
                    "signal_type": "COMEBACK",
                    "bet":         "OVER 2.5",
                    "reason":      f"{away} losing {away_g}-{home_g} at {elapsed}min "
                                   f"with attack str {away_atk:.2f}. Expect push. "
                                   f"Fair OVER price: {probs['fair_over_odds']}",
                })

        # ── HT Signals (first half only) ─────────────────────────────────────
        if HT_MIN_ELAPSED <= elapsed < 45:
            ht = _ht_probs(total_g, elapsed, lam)

            # HT Signal 1: UNDER 0.5 lock — 0-0 late in first half
            if (elapsed >= HT_LOCK_ELAPSED
                    and total_g == 0
                    and ht["ht_fair_under_05"] >= HT_MIN_FAIR_UNDER):
                tips.append({**base,
                    "signal_type": "HT_UNDER_0.5",
                    "bet":         "HT UNDER 0.5",
                    "reason":      f"0-0 at {elapsed}min first half. "
                                   f"P(HT UNDER 0.5)={ht['ht_p_under_05']*100:.0f}%. "
                                   f"Fair price: {ht['ht_fair_under_05']}",
                    "fair_under_odds": ht["ht_fair_under_05"],
                    "fair_over_odds":  ht["ht_fair_over_05"],
                    "live_p_under":    ht["ht_p_under_05"],
                    "live_p_over":     ht["ht_p_over_05"],
                })

            # HT Signal 2: UNDER 1.5 value — low score mid first half
            elif (elapsed >= HT_MIN_ELAPSED
                    and total_g <= 1
                    and ht["ht_fair_under_15"] >= HT_MIN_FAIR_UNDER
                    and ht["ht_p_under_15"] > 0.60):
                tips.append({**base,
                    "signal_type": "HT_UNDER_1.5",
                    "bet":         "HT UNDER 1.5",
                    "reason":      f"Score {home_g}-{away_g} at {elapsed}min first half. "
                                   f"P(HT UNDER 1.5)={ht['ht_p_under_15']*100:.0f}%. "
                                   f"Fair price: {ht['ht_fair_under_15']}",
                    "fair_under_odds": ht["ht_fair_under_15"],
                    "fair_over_odds":  ht["ht_fair_over_15"],
                    "live_p_under":    ht["ht_p_under_15"],
                    "live_p_over":     ht["ht_p_over_15"],
                })

            # HT Signal 3: OVER 0.5 — strong attack teams, 0-0, time still left
            elif (elapsed <= 38
                    and total_g == 0
                    and (home_atk >= ATTACK_STR_HIGH or away_atk >= ATTACK_STR_HIGH)
                    and ht["ht_fair_over_05"] >= HT_MIN_FAIR_OVER):
                tips.append({**base,
                    "signal_type": "HT_OVER_0.5",
                    "bet":         "HT OVER 0.5",
                    "reason":      f"0-0 at {elapsed}min, strong attack teams "
                                   f"(H:{home_atk:.2f} A:{away_atk:.2f}). "
                                   f"P(HT OVER 0.5)={ht['ht_p_over_05']*100:.0f}%. "
                                   f"Fair price: {ht['ht_fair_over_05']}",
                    "fair_under_odds": ht["ht_fair_under_05"],
                    "fair_over_odds":  ht["ht_fair_over_05"],
                    "live_p_under":    ht["ht_p_under_05"],
                    "live_p_over":     ht["ht_p_over_05"],
                })

    return tips


# ── Notified tracking ─────────────────────────────────────────────────────────

def _load_live_notified() -> set:
    if LIVE_NOTIFIED.exists():
        try:
            text = LIVE_NOTIFIED.read_text(encoding="utf-8-sig").strip()
            return set(json.loads(text).get("keys", []))
        except Exception:
            return set()
    return set()


def _save_live_notified(keys: set) -> None:
    LIVE_NOTIFIED.write_text(
        json.dumps({"keys": list(keys)}, indent=2), encoding="utf-8"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> list[dict]:
    log.info("Live Scanner — scanning in-progress matches...")

    live_games = _fetch_live_scores()
    if not live_games:
        log.info("  No in-progress games found right now.")
        _save_empty()
        _save_empty_games()
        return []

    log.info(f"  {len(live_games)} in-progress game(s) found")

    pred_df = _load_predictions()

    # Save all live games for dashboard visibility (with or without signal)
    game_rows = []
    for g in live_games:
        pred = _match_prediction(pred_df, g["home_team"], g["away_team"]) if not pred_df.empty else None
        p_over = float(pred["p_over25"]) if pred else None
        lam    = _lambda_from_p_over(p_over) if p_over else None
        probs  = _live_probs(g["total_goals"], g["elapsed_mins"], lam) if lam else {}
        game_rows.append({
            "league":          g["league"],
            "match":           f"{g['home_team']} vs {g['away_team']}",
            "score":           f"{g['home_goals']}-{g['away_goals']}",
            "elapsed_mins":    g["elapsed_mins"],
            "has_prediction":  pred is not None,
            "pre_signal":      pred.get("signal_tier", "-") if pred else "-",
            "pre_p_over":      round(p_over * 100, 1) if p_over else None,
            "live_p_under":    round(probs.get("p_under", 0) * 100, 1) if probs else None,
            "fair_under_odds": probs.get("fair_under_odds") if probs else None,
            "fair_over_odds":  probs.get("fair_over_odds")  if probs else None,
            "updated_at":      datetime.now().strftime("%H:%M"),
        })
    pd.DataFrame(game_rows).to_csv(LIVE_GAMES_FILE, index=False)

    if pred_df.empty:
        log.warning("  No predictions.csv found — run predict first.")
        _save_empty()
        return []

    # v2: fetch live SOT per game -> feeds the live-lambda AND the in-play collection log
    sot_by_fixture: dict = {}
    for _g in live_games:
        _fid = _g.get("fixture_id")
        if not _fid:
            continue
        try:
            from src.api_football import get_fixture_statistics
            _st = get_fixture_statistics(_fid, cache_hours=0.025)
            if _st:
                sot_by_fixture[_fid] = ((_st.get("home") or {}).get("shots_on_target", 0)
                                        + (_st.get("away") or {}).get("shots_on_target", 0))
        except Exception:
            pass
    _log_inplay_snapshot(live_games, sot_by_fixture)   # Phase-1 in-play dataset

    tips = _detect_signals(live_games, pred_df, sot_by_fixture)

    if tips:
        df = pd.DataFrame(tips)
        df = df.sort_values("elapsed_mins", ascending=False)
        df["date"] = datetime.now().strftime("%Y-%m-%d")
        df.to_csv(LIVE_TIPS_FILE, index=False)

        # Append to history file for tracking signal usefulness over time
        _append_to_history(df)

        log.info(f"\n{'='*60}")
        log.info(f"  LIVE VALUE SIGNALS: {len(tips)}")
        log.info(f"{'='*60}")
        for t in tips:
            log.info(f"  [{t['signal_type']}] {t['match']} | {t['score']} @{t['elapsed_mins']}min")
            log.info(f"    Bet: {t['bet']} | Fair odds: {t['fair_under_odds'] if 'UNDER' in t['bet'] else t['fair_over_odds']}")
            log.info(f"    {t['reason']}")

        # Send Telegram alerts for new signals
        try:
            from telegram_bot.notifier import notify_live_signals
            notify_live_signals()
        except Exception as e:
            log.error(f"Telegram live alert error: {e}")
    else:
        log.info("  No live value signals detected.")
        _save_empty()

    return tips


def _log_inplay_snapshot(live_games: list[dict], sot_by_fixture: dict | None = None) -> None:
    """v2 Phase-1 COLLECTION: append one row per in-progress game each scan (score, elapsed,
    SOT) -> builds the in-play dataset we currently lack, so the live-lambda model can be
    FITTED later. The final result is joined post-match by fixture_id. This is the data
    architecture step that makes the whole v2 upgrade possible."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for g in live_games:
        fid = g.get("fixture_id")
        rows.append({
            "snapshot_ts": ts, "fixture_id": fid, "league": g["league"],
            "match": f"{g['home_team']} vs {g['away_team']}",
            "elapsed": g["elapsed_mins"], "home_g": g["home_goals"],
            "away_g": g["away_goals"], "total_g": g["total_goals"],
            "sot": (sot_by_fixture or {}).get(fid),
        })
    if not rows:
        return
    new = pd.DataFrame(rows)
    if INPLAY_LOG_FILE.exists():
        try:
            new = pd.concat([pd.read_csv(INPLAY_LOG_FILE), new], ignore_index=True)
        except Exception:
            pass
    new.to_csv(INPLAY_LOG_FILE, index=False)


def _append_to_history(df: pd.DataFrame) -> None:
    """Append live signals to the history file for long-term tracking."""
    try:
        cols = ["date", "league", "match", "score", "elapsed_mins",
                "signal_type", "bet", "fair_under_odds", "fair_over_odds",
                "live_p_under", "live_p_over", "pre_p_over", "updated_at"]
        row = df[[c for c in cols if c in df.columns]].copy()
        if LIVE_HISTORY_FILE.exists():
            existing = pd.read_csv(LIVE_HISTORY_FILE)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row
        combined.to_csv(LIVE_HISTORY_FILE, index=False)
    except Exception as e:
        log.debug(f"History append error: {e}")


def _save_empty_games():
    pd.DataFrame(columns=[
        "league", "match", "score", "elapsed_mins", "has_prediction",
        "pre_signal", "pre_p_over", "live_p_under", "fair_under_odds",
        "fair_over_odds", "updated_at",
    ]).to_csv(LIVE_GAMES_FILE, index=False)


def _save_empty():
    pd.DataFrame(columns=[
        "league", "match", "score", "elapsed_mins", "pre_signal",
        "pre_p_over", "lambda", "live_p_under", "live_p_over",
        "fair_under_odds", "fair_over_odds", "lam_remaining",
        "signal_type", "bet", "reason", "updated_at",
    ]).to_csv(LIVE_TIPS_FILE, index=False)


def grade_live_signals(parquet_path: str | None = None) -> dict:
    """v2 VALIDATION: grade past O/U 2.5 live signals in live_signals_history.csv against the
    ACTUAL final total goals (from player_history.parquet, summed per fixture). Answers the
    question the old scanner never did: do these signals actually hit? Returns a per-signal-type
    hit-rate report. (HT signals need HT scores -> skipped here.) Read the hit rate, not vibes."""
    import re
    hist = LIVE_HISTORY_FILE
    if not hist.exists():
        print("[grade] no live_signals_history.csv yet"); return {}
    sig = pd.read_csv(hist)
    if sig.empty:
        return {}
    pqp = Path(parquet_path) if parquet_path else (config.OUTPUT_DIR.parent / "player_history.parquet")
    if not pqp.exists():
        print(f"[grade] parquet not found ({pqp})"); return {}
    pq = pd.read_parquet(pqp, columns=["home_team", "away_team", "date", "goals", "fixture_id"])
    _norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    tot = pq.groupby("fixture_id").agg(h=("home_team", "first"), a=("away_team", "first"),
                                       d=("date", "first"), g=("goals", "sum")).reset_index()
    finals = {(_norm(r.h), _norm(r.a), str(r.d)[:10]): r.g for r in tot.itertuples(index=False)}

    rows = []
    for s in sig.itertuples(index=False):
        m = str(getattr(s, "match", ""))
        if " vs " not in m or "UNDER 2.5" not in str(getattr(s, "bet", "")) and "OVER 2.5" not in str(getattr(s, "bet", "")):
            continue
        h, a = m.split(" vs ", 1)
        fg = finals.get((_norm(h), _norm(a), str(getattr(s, "date", ""))[:10]))
        if fg is None:
            continue
        under = fg < 2.5
        won = under if "UNDER" in str(s.bet) else (not under)
        rows.append({"signal_type": getattr(s, "signal_type", "?"), "won": bool(won)})
    if not rows:
        print("[grade] no gradable O/U 2.5 signals matched to results yet"); return {}
    gdf = pd.DataFrame(rows)
    print(f"[grade] {len(gdf)} live O/U 2.5 signals graded")
    rep = {}
    for st, sub in gdf.groupby("signal_type"):
        rep[st] = {"n": len(sub), "hit_%": round(sub["won"].mean() * 100, 1)}
        print(f"  {st:16} n={len(sub):3d}  hit={sub['won'].mean()*100:.1f}%")
    print(f"  {'ALL':16} n={len(gdf):3d}  hit={gdf['won'].mean()*100:.1f}%")
    return rep


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--grade":
        grade_live_signals(); sys.exit(0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
