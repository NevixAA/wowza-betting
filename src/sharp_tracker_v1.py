"""
Sharp Money Tracker
===================
Tracks odds drift across all enabled leagues.
Sharp money = odds shortening significantly = someone knows something.

Runs every 2h. Stores opening odds + snapshots, flags significant moves.

Signals:
  STRONG  — >10% drift  (high confidence sharp money)
  SHARP   — 5–10% drift (notable move)
  FADING  — odds lengthening (public/square money, avoid)

Output:
  output/sharp_history.json  — opening + snapshots per match/market
  output/sharp_tips.csv      — flagged drift signals
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

SHARP_HISTORY_FILE = config.OUTPUT_DIR / "sharp_history.json"
SHARP_TIPS_FILE    = config.OUTPUT_DIR / "sharp_tips.csv"

DRIFT_SHARP_PCT  = 0.05   # 5% move = sharp signal
DRIFT_STRONG_PCT = 0.10   # 10% move = strong signal
DAYS_AHEAD       = 14     # track games up to 14 days out


# ── OddsAPI fetch ─────────────────────────────────────────────────────────────

def _fetch_odds_for_league(sport_key: str, league: str) -> list[dict]:
    """Fetch totals + h2h odds for one league."""
    results = []
    cutoff  = datetime.utcnow() + timedelta(days=DAYS_AHEAD)

    for market in ("totals", "h2h"):
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey":     config.ODDS_API_KEY,
                    "regions":    "eu",
                    "markets":    market,
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
        except Exception as e:
            log.debug(f"  {league} ({market}): {e}")
            continue

        if r.status_code != 200:
            log.debug(f"  {league} ({market}): HTTP {r.status_code}")
            continue

        for event in r.json():
            dt = pd.to_datetime(event.get("commence_time", ""), errors="coerce", utc=True)
            if pd.isna(dt):
                continue
            dt_naive = dt.tz_localize(None)
            if dt_naive > cutoff or dt_naive < datetime.utcnow():
                continue  # skip past or too-far-future games

            entry = {
                "id":         event["id"],
                "league":     league,
                "home":       event["home_team"],
                "away":       event["away_team"],
                "date":       str(dt_naive)[:16],
                "market":     market,
                "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            }

            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if market == "totals" and mkt["key"] == "totals":
                        for o in mkt["outcomes"]:
                            if o.get("point") == 2.5:
                                if o["name"] == "Over":
                                    entry["odds_over"]  = o["price"]
                                elif o["name"] == "Under":
                                    entry["odds_under"] = o["price"]
                    elif market == "h2h" and mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == event["home_team"]:
                                entry["odds_home"] = o["price"]
                            elif o["name"] == event["away_team"]:
                                entry["odds_away"] = o["price"]
                            elif o["name"] == "Draw":
                                entry["odds_draw"] = o["price"]
                if any(k in entry for k in ("odds_over", "odds_home")):
                    break

            results.append(entry)

    return results


# ── History management ────────────────────────────────────────────────────────

def _load_history() -> dict:
    if SHARP_HISTORY_FILE.exists():
        try:
            return json.loads(SHARP_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_history(history: dict) -> None:
    SHARP_HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _extract_odds(ev: dict) -> dict:
    keys = ("odds_over", "odds_under", "odds_home", "odds_away", "odds_draw")
    return {k: ev[k] for k in keys if k in ev}


def _update_history(history: dict, events: list[dict]) -> dict:
    for ev in events:
        key = f"{ev['id']}_{ev['market']}"
        if key not in history:
            history[key] = {
                "league":    ev["league"],
                "home":      ev["home"],
                "away":      ev["away"],
                "date":      ev["date"],
                "market":    ev["market"],
                "opening":   _extract_odds(ev),
                "snapshots": [],
            }
        history[key]["snapshots"].append({
            "at":   ev["fetched_at"],
            "odds": _extract_odds(ev),
        })
        history[key]["snapshots"] = history[key]["snapshots"][-50:]
    return history


# ── Drift calculation ─────────────────────────────────────────────────────────

def _drift_pct(opening: float, current: float) -> float:
    if not opening or not current:
        return 0.0
    return (current - opening) / opening


def _signal_label(pct: float) -> str:
    if pct <= -DRIFT_STRONG_PCT:
        return "STRONG"
    if pct <= -DRIFT_SHARP_PCT:
        return "SHARP"
    if pct >= DRIFT_STRONG_PCT:
        return "FADING"
    return "NEUTRAL"


def _build_tips(history: dict) -> pd.DataFrame:
    rows = []
    now  = datetime.utcnow()

    for key, rec in history.items():
        if not rec.get("snapshots"):
            continue
        match_dt = pd.to_datetime(rec["date"], errors="coerce")
        if pd.isna(match_dt) or match_dt < now:
            continue

        opening = rec["opening"]
        current = rec["snapshots"][-1]["odds"]
        n_snaps = len(rec["snapshots"])

        if rec["market"] == "totals":
            for side, ok in [("UNDER", "odds_under"), ("OVER", "odds_over")]:
                op = opening.get(ok)
                cu = current.get(ok)
                if not op or not cu:
                    continue
                d   = _drift_pct(op, cu)
                sig = _signal_label(d)
                if sig == "NEUTRAL":
                    continue
                rows.append({
                    "date":         rec["date"][:10],
                    "league":       rec.get("league", ""),
                    "match":        f"{rec['home']} vs {rec['away']}",
                    "market":       f"O/U 2.5 {side}",
                    "opening_odds": round(op, 3),
                    "current_odds": round(cu, 3),
                    "drift_pct":    round(d * 100, 1),
                    "signal":       sig,
                    "snapshots":    n_snaps,
                    "updated_at":   rec["snapshots"][-1]["at"],
                })

        elif rec["market"] == "h2h":
            for side, ok, name in [
                ("HOME", "odds_home", rec["home"]),
                ("AWAY", "odds_away", rec["away"]),
                ("DRAW", "odds_draw", "Draw"),
            ]:
                op = opening.get(ok)
                cu = current.get(ok)
                if not op or not cu:
                    continue
                d   = _drift_pct(op, cu)
                sig = _signal_label(d)
                if sig == "NEUTRAL":
                    continue
                rows.append({
                    "date":         rec["date"][:10],
                    "league":       rec.get("league", ""),
                    "match":        f"{rec['home']} vs {rec['away']}",
                    "market":       f"1X2 {side} ({name})",
                    "opening_odds": round(op, 3),
                    "current_odds": round(cu, 3),
                    "drift_pct":    round(d * 100, 1),
                    "signal":       sig,
                    "snapshots":    n_snaps,
                    "updated_at":   rec["snapshots"][-1]["at"],
                })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "league", "match", "market", "opening_odds", "current_odds",
            "drift_pct", "signal", "snapshots", "updated_at"
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(["signal", "drift_pct"], ascending=[True, True]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("Sharp Money Tracker — all enabled leagues")
    history = _load_history()
    total_events = 0

    for league in config.ENABLED_LEAGUES:
        sport_key = config.ODDS_API_SPORT_KEYS.get(league)
        if not sport_key:
            continue
        events = _fetch_odds_for_league(sport_key, league)
        if events:
            history = _update_history(history, events)
            total_events += len(events)
            log.info(f"  {league}: {len(events)} snapshots")

    _save_history(history)

    tips = _build_tips(history)
    tips.to_csv(SHARP_TIPS_FILE, index=False)

    strong = tips[tips["signal"] == "STRONG"]
    sharp  = tips[tips["signal"] == "SHARP"]
    fading = tips[tips["signal"] == "FADING"]

    log.info(f"\n{'='*55}")
    log.info(f"  SHARP MONEY SIGNALS — {total_events} snapshots")
    log.info(f"{'='*55}")
    log.info(f"  STRONG (>10%): {len(strong)}  |  SHARP (5-10%): {len(sharp)}  |  FADING: {len(fading)}")

    if not strong.empty:
        log.info("  STRONG signals:")
        for _, r in strong.iterrows():
            log.info(f"    {r['date']} | {r['league']} | {r['match']} | "
                     f"{r['market']} | {r['opening_odds']} → {r['current_odds']} ({r['drift_pct']:+.1f}%)")

    log.info(f"  Tips saved → {SHARP_TIPS_FILE}")
    return tips


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
