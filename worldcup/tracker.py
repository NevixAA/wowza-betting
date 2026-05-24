"""
World Cup 2026 — Drift Tracker
==============================
No ML. Pure odds movement analysis.

Fetches O/U 2.5 + 1X2 odds every few hours and tracks how they move.
"Sharp money" = odds shortening significantly = someone knows something.

Output:
  output/worldcup_history.json  — opening + all snapshots per match
  output/worldcup_tips.csv      — matches with significant drift flagged
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ── Config ────────────────────────────────────────────────────────────────────
WC_SPORT_KEY      = "soccer_fifa_world_cup"
WC_HISTORY_FILE   = config.OUTPUT_DIR / "worldcup_history.json"
WC_TIPS_FILE      = config.OUTPUT_DIR / "worldcup_tips.csv"

DRIFT_SHARP_PCT   = 0.05   # 5% move = sharp signal
DRIFT_STRONG_PCT  = 0.10   # 10% move = strong signal
DAYS_AHEAD        = 30     # look 30 days forward (group stage + knockouts)


# ── OddsAPI fetch ─────────────────────────────────────────────────────────────

def _fetch_wc_odds() -> list[dict]:
    """Fetch all WC fixtures with O/U 2.5 + 1X2 from OddsAPI."""
    results = []
    cutoff  = datetime.utcnow() + timedelta(days=DAYS_AHEAD)

    for market in ("totals", "h2h"):
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{WC_SPORT_KEY}/odds",
                params={
                    "apiKey": config.ODDS_API_KEY,
                    "regions": "eu",
                    "markets": market,
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
        except Exception as e:
            log.warning(f"OddsAPI error ({market}): {e}")
            continue

        if r.status_code == 404:
            log.info("World Cup not yet available on OddsAPI (404) — check back closer to June 11.")
            continue
        if r.status_code != 200:
            log.warning(f"OddsAPI {market}: HTTP {r.status_code} — {r.text[:200]}")
            continue

        for event in r.json():
            dt = pd.to_datetime(event.get("commence_time", ""), errors="coerce", utc=True)
            if pd.isna(dt) or dt.tz_localize(None) > cutoff:
                continue

            entry = {
                "id":        event["id"],
                "home":      event["home_team"],
                "away":      event["away_team"],
                "date":      str(dt.tz_localize(None))[:16],
                "market":    market,
                "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            }

            # Parse best available odds (first bookmaker that has the market)
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
                    break  # got what we need from first valid bookmaker

            results.append(entry)

    return results


# ── History management ────────────────────────────────────────────────────────

def _load_history() -> dict:
    if WC_HISTORY_FILE.exists():
        try:
            return json.loads(WC_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_history(history: dict) -> None:
    WC_HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_history(history: dict, events: list[dict]) -> dict:
    """Merge new snapshots into history. First snapshot = opening odds."""
    for ev in events:
        key = f"{ev['id']}_{ev['market']}"
        if key not in history:
            history[key] = {
                "home":     ev["home"],
                "away":     ev["away"],
                "date":     ev["date"],
                "market":   ev["market"],
                "opening":  _extract_odds(ev),
                "snapshots": [],
            }
        history[key]["snapshots"].append({
            "at":   ev["fetched_at"],
            "odds": _extract_odds(ev),
        })
        # Keep only last 50 snapshots
        history[key]["snapshots"] = history[key]["snapshots"][-50:]
    return history


def _extract_odds(ev: dict) -> dict:
    keys = ("odds_over", "odds_under", "odds_home", "odds_away", "odds_draw")
    return {k: ev[k] for k in keys if k in ev}


# ── Drift calculation ─────────────────────────────────────────────────────────

def _drift_pct(opening: float, current: float) -> float:
    """Positive = odds lengthened (money moved away). Negative = odds shortened (sharp money)."""
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
        if not rec["snapshots"]:
            continue

        match_dt = pd.to_datetime(rec["date"], errors="coerce")
        if pd.isna(match_dt) or match_dt < now:
            continue  # skip past matches

        opening = rec["opening"]
        current = rec["snapshots"][-1]["odds"]
        n_snaps = len(rec["snapshots"])

        if rec["market"] == "totals":
            for side, ok, ck in [
                ("UNDER", "odds_under", "odds_under"),
                ("OVER",  "odds_over",  "odds_over"),
            ]:
                op = opening.get(ok)
                cu = current.get(ck)
                if not op or not cu:
                    continue
                d = _drift_pct(op, cu)
                sig = _signal_label(d)
                if sig == "NEUTRAL":
                    continue
                rows.append({
                    "date":         rec["date"][:10],
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
            labels = [
                ("HOME", "odds_home", rec["home"]),
                ("AWAY", "odds_away", rec["away"]),
                ("DRAW", "odds_draw", "Draw"),
            ]
            for side, ok, name in labels:
                op = opening.get(ok)
                cu = current.get(ok)
                if not op or not cu:
                    continue
                d = _drift_pct(op, cu)
                sig = _signal_label(d)
                if sig == "NEUTRAL":
                    continue
                rows.append({
                    "date":         rec["date"][:10],
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
            "date", "match", "market", "opening_odds", "current_odds",
            "drift_pct", "signal", "snapshots", "updated_at"
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values(["signal", "drift_pct"], ascending=[True, True])
    return df.reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("World Cup 2026 Drift Tracker")
    log.info("Fetching odds from OddsAPI...")

    events = _fetch_wc_odds()
    if not events:
        log.info("No World Cup fixtures found yet. Tournament starts June 11, 2026.")
        log.info("Run this script again closer to the start date.")
        return

    log.info(f"  {len(events)} market snapshots fetched")

    history = _load_history()
    history = _update_history(history, events)
    _save_history(history)

    tips = _build_tips(history)
    tips.to_csv(WC_TIPS_FILE, index=False)

    strong = tips[tips["signal"] == "STRONG"]
    sharp  = tips[tips["signal"] == "SHARP"]
    fading = tips[tips["signal"] == "FADING"]

    log.info(f"\n{'='*60}")
    log.info(f"  WORLD CUP DRIFT SIGNALS")
    log.info(f"{'='*60}")
    log.info(f"  STRONG (>10% move):  {len(strong)}")
    log.info(f"  SHARP  (5-10% move): {len(sharp)}")
    log.info(f"  FADING (odds drift up): {len(fading)}")

    if not strong.empty:
        log.info(f"\n  STRONG signals:")
        for _, r in strong.iterrows():
            log.info(f"    {r['date']} | {r['match']} | {r['market']} | "
                     f"{r['opening_odds']} -> {r['current_odds']} ({r['drift_pct']:+.1f}%)")

    log.info(f"\n  Tips saved -> {WC_TIPS_FILE}")


if __name__ == "__main__":
    run()
