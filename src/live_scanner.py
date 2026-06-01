"""
Live Game Scanner — v9
======================
Scans in-progress matches from our enabled leagues and finds live value bets
using Poisson statistics + pre-match model predictions.

No live odds API needed — we calculate the FAIR live price and alert the user
to check their bookmaker's live screen.

Signal types:
  UNDER_HOLD      — 0-0 or low score at 60+ min, model said UNDER, still good value
  SLEEPING_GAME   — two low-scoring teams, 0-0 at 75+ min, UNDER almost locked
  UNDER_RECOVERY  — score is 1-1 or 2-0, time left, Poisson says UNDER still value
  STRONG_STUCK    — high attack-strength team not scoring, pushing for goals → OVER value
  COMEBACK        — team losing at 60+ min has strong attack → OVER / draw value
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

LIVE_TIPS_FILE    = config.OUTPUT_DIR / "live_tips.csv"
LIVE_GAMES_FILE   = config.OUTPUT_DIR / "live_games.csv"
LIVE_NOTIFIED     = Path(__file__).resolve().parents[1] / "telegram_bot" / "live_notified.json"

MIN_FAIR_UNDER    = 1.28   # only alert if fair UNDER odds >= this (meaningful value)
MIN_FAIR_OVER     = 2.00   # only alert OVER if fair odds >= this
MIN_ELAPSED       = 45     # don't alert before half-time
MIN_LIVE_EDGE     = 0.12   # 12% edge threshold for live alerts (higher bar than pre-match)
ATTACK_STR_HIGH   = 1.25   # threshold for "strong attack" signal

IDLE_RECHECK_SECS   = 1800  # re-check a league with no live games every 30 min
ACTIVE_RECHECK_SECS = 120   # re-check a league with live games every 2 min

# Per-league cache — persists between calls when imported inline by master_loop
_league_last_active:  dict = {}  # league -> datetime of last live game found
_league_last_checked: dict = {}  # league -> datetime of last API call


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


def _live_probs(goals_scored: int, elapsed_mins: float, lam_total: float) -> dict:
    """
    Calculate live probabilities given current state.
    Returns dict with p_under, p_over, fair_under_odds, fair_over_odds.
    """
    remaining_frac = max(0.0, (90.0 - elapsed_mins) / 90.0)
    lam_remaining  = lam_total * remaining_frac

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


# ── Live scores fetch ─────────────────────────────────────────────────────────

def _leagues_with_games_today() -> set[str]:
    """Return leagues that have at least one prediction for today (saves API credits)."""
    try:
        preds = pd.read_csv(config.OUTPUT_DIR / "predictions.csv")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return set(preds[preds["date"].astype(str).str[:10] == today]["league"].unique())
    except Exception:
        return set(config.ENABLED_LEAGUES)  # fallback: scan all


def _fetch_live_scores() -> list[dict]:
    """Fetch in-progress scores — only for leagues with games today that are due a check."""
    live = []
    seen = set()
    now = datetime.now(timezone.utc)

    active_leagues = _leagues_with_games_today()

    for league, sport_key in config.ODDS_API_SPORT_KEYS.items():
        if league not in config.ENABLED_LEAGUES:
            continue
        if league not in active_leagues:
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
                elapsed = (now_utc - commence).total_seconds() / 60.0

                if elapsed < 1:
                    continue  # not started yet

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

def _detect_signals(live_games: list[dict], pred_df: pd.DataFrame) -> list[dict]:
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
            continue

        p_over     = float(pred.get("p_over25", 0.45))
        lam        = _lambda_from_p_over(p_over)
        probs      = _live_probs(total_g, elapsed, lam)
        pre_signal = str(pred.get("signal_tier", "AVOID"))

        home_atk   = float(pred.get("home_attack_str", 1.0) or 1.0)
        away_atk   = float(pred.get("away_attack_str", 1.0) or 1.0)
        league_avg = float(pred.get("league_avg_goals", 2.5) or 2.5)

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

    tips = _detect_signals(live_games, pred_df)

    if tips:
        df = pd.DataFrame(tips)
        df = df.sort_values("elapsed_mins", ascending=False)
        df.to_csv(LIVE_TIPS_FILE, index=False)
        log.info(f"\n{'='*60}")
        log.info(f"  LIVE VALUE SIGNALS: {len(tips)}")
        log.info(f"{'='*60}")
        for t in tips:
            log.info(f"  [{t['signal_type']}] {t['match']} | {t['score']} @{t['elapsed_mins']}min")
            log.info(f"    Bet: {t['bet']} | Fair odds: {t['fair_under_odds'] if 'UNDER' in t['bet'] else t['fair_over_odds']}")
            log.info(f"    {t['reason']}")
    else:
        log.info("  No live value signals detected.")
        _save_empty()

    return tips


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
