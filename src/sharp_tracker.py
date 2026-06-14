"""
Sharp Money Tracker v2
======================
Tracks odds drift across all enabled leagues.

Improvements over v1:
  - Volume-weighted odds: median across ALL bookmakers (not just first)
  - Steam detection: flags moves >3% in a single 2h window (fast money)
  - Consensus score: % of books agreeing on direction (coordinated = sharper)
  - Upgraded signal labels: STEAM_STRONG, STEAM_SHARP beat regular signals

Signals (ranked by strength):
  STEAM_STRONG — steam + >10% total drift  (highest confidence)
  STEAM_SHARP  — steam + 5–10% total drift
  STRONG       — >10% drift, no steam
  SHARP        — 5–10% drift
  FADING       — odds lengthening (square money, avoid)

Output:
  output/sharp_history.json  — full snapshot history per match/market
  output/sharp_tips.csv      — flagged signals with consensus scores
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from player_model.ledger import append_sharp_signals

log = logging.getLogger(__name__)

SHARP_HISTORY_FILE = config.OUTPUT_DIR / "sharp_history.json"
SHARP_TIPS_FILE    = config.OUTPUT_DIR / "sharp_tips.csv"

DRIFT_SHARP_PCT  = 0.05   # 5%  = sharp signal
DRIFT_STRONG_PCT = 0.10   # 10% = strong signal
STEAM_PCT        = 0.03   # 3%  move in one 2h window = steam
DAYS_AHEAD       = 14


# ── OddsAPI fetch — median across all bookmakers ──────────────────────────────

def _fetch_odds_for_league(sport_key: str, league: str) -> list[dict]:
    """Fetch totals + h2h odds — median across ALL bookmakers + consensus count."""
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
                continue

            # Collect odds from ALL bookmakers
            book_odds: dict[str, list[float]] = {
                "odds_over": [], "odds_under": [],
                "odds_home": [], "odds_away": [], "odds_draw": [],
            }

            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if market == "totals" and mkt["key"] == "totals":
                        for o in mkt["outcomes"]:
                            if o.get("point") == 2.5:
                                if o["name"] == "Over":
                                    book_odds["odds_over"].append(o["price"])
                                elif o["name"] == "Under":
                                    book_odds["odds_under"].append(o["price"])
                    elif market == "h2h" and mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == event["home_team"]:
                                book_odds["odds_home"].append(o["price"])
                            elif o["name"] == event["away_team"]:
                                book_odds["odds_away"].append(o["price"])
                            elif o["name"] == "Draw":
                                book_odds["odds_draw"].append(o["price"])

            # Build entry with median odds + book count
            entry = {
                "id":         event["id"],
                "league":     league,
                "home":       event["home_team"],
                "away":       event["away_team"],
                "date":       str(dt_naive)[:16],
                "market":     market,
                "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            }

            n_books = 0
            for k, vals in book_odds.items():
                if vals:
                    entry[k] = round(statistics.median(vals), 3)
                    n_books = max(n_books, len(vals))

            entry["n_books"] = n_books

            if any(k in entry for k in ("odds_over", "odds_home")):
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
            "at":      ev["fetched_at"],
            "odds":    _extract_odds(ev),
            "n_books": ev.get("n_books", 1),
        })
        history[key]["snapshots"] = history[key]["snapshots"][-50:]
    return history


# ── Drift + steam calculation ─────────────────────────────────────────────────

def _drift_pct(opening: float, current: float) -> float:
    if not opening or not current:
        return 0.0
    return (current - opening) / opening


def _signal_label(total_drift: float, is_steam: bool) -> str:
    if total_drift <= -DRIFT_STRONG_PCT:
        return "STEAM_STRONG" if is_steam else "STRONG"
    if total_drift <= -DRIFT_SHARP_PCT:
        return "STEAM_SHARP" if is_steam else "SHARP"
    if total_drift >= DRIFT_STRONG_PCT:
        return "FADING"
    return "NEUTRAL"


def _detect_steam(snapshots: list[dict], odds_key: str) -> bool:
    """True if odds moved >STEAM_PCT in the most recent single window."""
    if len(snapshots) < 2:
        return False
    prev = snapshots[-2]["odds"].get(odds_key)
    curr = snapshots[-1]["odds"].get(odds_key)
    if not prev or not curr:
        return False
    return abs(_drift_pct(prev, curr)) >= STEAM_PCT


def _consensus_pct(snapshots: list[dict], odds_key: str, direction: float) -> int:
    """
    % of snapshots where odds moved in same direction as total drift.
    direction: negative = shortening, positive = lengthening.
    """
    if len(snapshots) < 2:
        return 0
    moves = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i-1]["odds"].get(odds_key)
        curr = snapshots[i]["odds"].get(odds_key)
        if prev and curr:
            moves.append(curr - prev)
    if not moves:
        return 0
    agreeing = sum(1 for m in moves if (m < 0) == (direction < 0))
    return round(agreeing / len(moves) * 100)


def _build_tips(history: dict) -> pd.DataFrame:
    rows = []
    now  = datetime.utcnow()

    for key, rec in history.items():
        if not rec.get("snapshots") or len(rec["snapshots"]) < 2:
            continue
        match_dt = pd.to_datetime(rec["date"], errors="coerce")
        if pd.isna(match_dt) or match_dt < now:
            continue

        opening  = rec["opening"]
        current  = rec["snapshots"][-1]["odds"]
        n_snaps  = len(rec["snapshots"])
        n_books  = rec["snapshots"][-1].get("n_books", 1)
        snaps    = rec["snapshots"]

        if rec["market"] == "totals":
            sides = [("UNDER", "odds_under"), ("OVER", "odds_over")]
        else:
            sides = [
                ("HOME", "odds_home"),
                ("AWAY", "odds_away"),
                ("DRAW", "odds_draw"),
            ]

        for side, ok in sides:
            op = opening.get(ok)
            cu = current.get(ok)
            if not op or not cu:
                continue

            # Sanity: reject cross-market contamination and impossible values
            # Draw odds < 1.3 or > 20 are physically impossible
            if side == "DRAW" and not (1.3 <= op <= 20.0 and 1.3 <= cu <= 20.0):
                continue
            # Over/Under odds < 1.05 or > 15 are garbage data
            if side in ("OVER", "UNDER") and not (1.05 <= op <= 15.0 and 1.05 <= cu <= 15.0):
                continue
            # >70% single-step drift is almost always a market-mix bug
            if abs(op - cu) / op > 0.70:
                continue

            d       = _drift_pct(op, cu)
            steam   = _detect_steam(snaps, ok)
            sig     = _signal_label(d, steam)
            if sig == "NEUTRAL":
                continue

            consensus = _consensus_pct(snaps, ok, d)
            label = (f"O/U 2.5 {side}" if rec["market"] == "totals"
                     else f"1X2 {side} ({rec['home'] if side=='HOME' else rec['away'] if side=='AWAY' else 'Draw'})")

            rows.append({
                "date":         rec["date"][:10],
                "league":       rec.get("league", ""),
                "match":        f"{rec['home']} vs {rec['away']}",
                "market":       label,
                "opening_odds": round(op, 3),
                "current_odds": round(cu, 3),
                "drift_pct":    round(d * 100, 1),
                "signal":       sig,
                "steam":        steam,
                "consensus_pct": consensus,
                "n_books":      n_books,
                "snapshots":    n_snaps,
                "updated_at":   rec["snapshots"][-1]["at"],
            })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "league", "match", "market", "opening_odds", "current_odds",
            "drift_pct", "signal", "steam", "consensus_pct", "n_books", "snapshots", "updated_at"
        ])

    df = pd.DataFrame(rows)
    signal_order = {"STEAM_STRONG": 0, "STEAM_SHARP": 1, "STRONG": 2, "SHARP": 3, "FADING": 4}
    df["_order"] = df["signal"].map(signal_order)
    df = df.sort_values(["_order", "drift_pct"], ascending=[True, True])
    return df.drop("_order", axis=1).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("Sharp Money Tracker v2 — volume-weighted + steam detection")
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
    append_sharp_signals(tips)

    steam_strong = tips[tips["signal"] == "STEAM_STRONG"]
    steam_sharp  = tips[tips["signal"] == "STEAM_SHARP"]
    strong       = tips[tips["signal"] == "STRONG"]
    sharp        = tips[tips["signal"] == "SHARP"]
    fading       = tips[tips["signal"] == "FADING"]

    log.info(f"\n{'='*55}")
    log.info(f"  SHARP MONEY v2 — {total_events} snapshots processed")
    log.info(f"{'='*55}")
    log.info(f"  STEAM_STRONG : {len(steam_strong)}")
    log.info(f"  STEAM_SHARP  : {len(steam_sharp)}")
    log.info(f"  STRONG       : {len(strong)}")
    log.info(f"  SHARP        : {len(sharp)}")
    log.info(f"  FADING       : {len(fading)}")

    top = tips[tips["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"])]
    if not top.empty:
        log.info("  Top signals:")
        for _, r in top.iterrows():
            steam_tag = " 🔥STEAM" if r["steam"] else ""
            log.info(f"    [{r['signal']}]{steam_tag} {r['date']} | {r['league']} | "
                     f"{r['match']} | {r['market']} | "
                     f"{r['opening_odds']} → {r['current_odds']} "
                     f"({r['drift_pct']:+.1f}%) | {r['consensus_pct']}% consensus | {r['n_books']} books")

    log.info(f"  Tips saved → {SHARP_TIPS_FILE}")
    return tips


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
