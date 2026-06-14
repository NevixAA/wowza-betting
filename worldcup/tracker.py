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
    """Fetch all WC fixtures with O/U 2.5, O/U 1.5, O/U 3.5 + 1X2 from OddsAPI."""
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

            # Collect odds from ALL bookmakers, then take median (outlier-resistant)
            book_odds: dict[str, list[float]] = {
                "odds_over": [], "odds_under": [],
                "odds_over15": [], "odds_under15": [],
                "odds_over35": [], "odds_under35": [],
                "odds_home": [], "odds_away": [], "odds_draw": [],
            }
            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if market == "totals" and mkt["key"] == "totals":
                        for o in mkt["outcomes"]:
                            pt = o.get("point")
                            if pt == 2.5:
                                if o["name"] == "Over":  book_odds["odds_over"].append(o["price"])
                                else:                    book_odds["odds_under"].append(o["price"])
                            elif pt == 1.5:
                                if o["name"] == "Over":  book_odds["odds_over15"].append(o["price"])
                                else:                    book_odds["odds_under15"].append(o["price"])
                            elif pt == 3.5:
                                if o["name"] == "Over":  book_odds["odds_over35"].append(o["price"])
                                else:                    book_odds["odds_under35"].append(o["price"])
                    elif market == "h2h" and mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == event["home_team"]:
                                book_odds["odds_home"].append(o["price"])
                            elif o["name"] == event["away_team"]:
                                book_odds["odds_away"].append(o["price"])
                            elif o["name"] == "Draw":
                                book_odds["odds_draw"].append(o["price"])

            n_books = 0
            for k, vals in book_odds.items():
                if vals:
                    entry[k] = round(statistics.median(vals), 3)
                    n_books = max(n_books, len(vals))
            entry["n_books"] = n_books

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
            "at":      ev["fetched_at"],
            "odds":    _extract_odds(ev),
            "n_books": ev.get("n_books", 1),
        })
        # Keep only last 50 snapshots
        history[key]["snapshots"] = history[key]["snapshots"][-50:]
    return history


def _extract_odds(ev: dict) -> dict:
    keys = ("odds_over", "odds_under", "odds_over15", "odds_under15",
            "odds_over35", "odds_under35", "odds_home", "odds_away", "odds_draw")
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
        # Need at least 2 snapshots — a single data point can't prove movement
        if len(rec.get("snapshots", [])) < 2:
            continue

        match_dt = pd.to_datetime(rec["date"], errors="coerce")
        if pd.isna(match_dt) or match_dt < now:
            continue  # skip past matches

        opening = rec["opening"]
        current = rec["snapshots"][-1]["odds"]
        n_snaps = len(rec["snapshots"])
        n_books = rec["snapshots"][-1].get("n_books", 1)

        if rec["market"] == "totals":
            for side, ok, label in [
                ("UNDER", "odds_under",   "O/U 2.5 UNDER"),
                ("OVER",  "odds_over",    "O/U 2.5 OVER"),
                ("UNDER", "odds_under15", "O/U 1.5 UNDER"),
                ("OVER",  "odds_over15",  "O/U 1.5 OVER"),
                ("UNDER", "odds_under35", "O/U 3.5 UNDER"),
                ("OVER",  "odds_over35",  "O/U 3.5 OVER"),
            ]:
                op = opening.get(ok)
                cu = current.get(ok)
                if not op or not cu:
                    continue
                # Plausibility guard: O/U outside 1.05-15 is garbage data
                if not (1.05 <= op <= 15.0 and 1.05 <= cu <= 15.0):
                    continue
                # >70% drift in one window = almost certainly a data error
                if abs(op - cu) / op > 0.70:
                    continue
                d = _drift_pct(op, cu)
                sig = _signal_label(d)
                if sig == "NEUTRAL":
                    continue
                rows.append({
                    "date":         rec["date"][:10],
                    "match":        f"{rec['home']} vs {rec['away']}",
                    "market":       label,
                    "opening_odds": round(op, 3),
                    "current_odds": round(cu, 3),
                    "drift_pct":    round(d * 100, 1),
                    "signal":       sig,
                    "snapshots":    n_snaps,
                    "n_books":      n_books,
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
                # Plausibility guard: draw 1.3-20, home/away 1.05-40
                if side == "DRAW" and not (1.30 <= op <= 20.0 and 1.30 <= cu <= 20.0):
                    continue
                if side in ("HOME", "AWAY") and not (1.05 <= op <= 40.0 and 1.05 <= cu <= 40.0):
                    continue
                # >50% drift in one window from opening is almost certainly a data error
                if abs(op - cu) / op > 0.50:
                    log.warning(f"Implausible drift skipped: {rec['home']} vs {rec['away']} "
                                f"{side} {op:.2f}→{cu:.2f} ({abs(op-cu)/op:.0%})")
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
                    "n_books":      n_books,
                    "updated_at":   rec["snapshots"][-1]["at"],
                })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "match", "market", "opening_odds", "current_odds",
            "drift_pct", "signal", "snapshots", "n_books", "updated_at"
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values(["signal", "drift_pct"], ascending=[True, True])
    return df.reset_index(drop=True)


# ── ML model value detection ──────────────────────────────────────────────────

WC_MODEL_FILE = config.OUTPUT_DIR / "worldcup_model_tips.csv"

def _add_model_value(history: dict) -> pd.DataFrame:
    """
    Apply our FT + HT models to WC fixtures and compare vs market odds.

    ⚠️  IMPORTANT DISCLAIMER — ML output is INFORMATIONAL ONLY:
    The standard model was trained on European domestic leagues (League One,
    Bundesliga 2, La Liga 2, Ligue 2, League Two). International tournament
    football has fundamentally different dynamics (motivation, squad rotation,
    tactical approach, rest periods). Features like attack_str and defense_str
    will be NaN for WC fixtures (league not in historical data) and will be
    imputed to league averages — effectively making ML output equal to market
    price. Do NOT treat WC ML values as genuine independent edge signals.
    Use the drift tracker (sharp money) as the primary WC signal instead.
    """
    try:
        from src.model import load_models, predict_proba
        from src.feature_engineering import build_upcoming_features
        from src.data_loader import load_all_matches
    except Exception as e:
        log.warning(f"Model import failed: {e}")
        return pd.DataFrame()

    rows = []
    now  = datetime.utcnow()

    # Build a minimal upcoming DataFrame from WC history
    fixtures = []
    seen = set()
    for key, rec in history.items():
        if rec["market"] != "totals":
            continue
        match_dt = pd.to_datetime(rec["date"], errors="coerce")
        if pd.isna(match_dt) or match_dt < now:
            continue
        match_key = f"{rec['home']}|{rec['away']}"
        if match_key in seen:
            continue
        seen.add(match_key)
        current = rec["snapshots"][-1]["odds"] if rec["snapshots"] else rec["opening"]
        fixtures.append({
            "date":        match_dt,
            "league":      "World Cup 2026",
            "home_team":   rec["home"],
            "away_team":   rec["away"],
            "odds_over25": current.get("odds_over"),
            "odds_under25":current.get("odds_under"),
        })

    if not fixtures:
        return pd.DataFrame()

    upcoming = pd.DataFrame(fixtures)
    upcoming = upcoming.dropna(subset=["odds_over25", "odds_under25"])

    try:
        historical = load_all_matches()
        feat = build_upcoming_features(upcoming, historical)
        payload = load_models(model_file=config.MODEL_FILE_STANDARD)
        feat["p_over25"] = predict_proba(feat, payload=payload).values

        # HT model if available
        if config.HT_MODEL_FILE_05.exists():
            ht_payload = load_models(model_file=config.HT_MODEL_FILE_05)
            feat["p_ht_over05"] = predict_proba(feat, payload=ht_payload).values
        if config.HT_MODEL_FILE_15.exists():
            ht_payload15 = load_models(model_file=config.HT_MODEL_FILE_15)
            feat["p_ht_over15"] = predict_proba(feat, payload=ht_payload15).values
    except Exception as e:
        log.warning(f"WC model prediction failed: {e}")
        return pd.DataFrame()

    for _, row in feat.iterrows():
        p_over = float(row.get("p_over25", 0.5))
        market_over  = float(row["odds_over25"])
        market_under = float(row["odds_under25"])
        fair_over  = round(1 / max(p_over, 0.01), 2)
        fair_under = round(1 / max(1 - p_over, 0.01), 2)

        # Value = our fair price is lower than market (market is paying more than fair)
        ft_over_value  = round((market_over  / fair_over  - 1) * 100, 1) if fair_over  > 0 else 0
        ft_under_value = round((market_under / fair_under - 1) * 100, 1) if fair_under > 0 else 0

        entry = {
            "date":          str(row["date"])[:10],
            "match":         f"{row['home_team']} vs {row['away_team']}",
            "p_ft_over25":   round(p_over, 3),
            "fair_ft_over":  fair_over,
            "fair_ft_under": fair_under,
            "market_over25": market_over,
            "market_under25":market_under,
            "ft_over_value":  ft_over_value,
            "ft_under_value": ft_under_value,
        }

        if "p_ht_over05" in row and pd.notna(row["p_ht_over05"]):
            p_ht05 = float(row["p_ht_over05"])
            entry["p_ht_over05"]   = round(p_ht05, 3)
            entry["fair_ht_over05"]  = round(1 / max(p_ht05, 0.01), 2)
            entry["fair_ht_under05"] = round(1 / max(1 - p_ht05, 0.01), 2)

        if "p_ht_over15" in row and pd.notna(row.get("p_ht_over15")):
            p_ht15 = float(row["p_ht_over15"])
            entry["p_ht_over15"]   = round(p_ht15, 3)
            entry["fair_ht_over15"]  = round(1 / max(p_ht15, 0.01), 2)
            entry["fair_ht_under15"] = round(1 / max(1 - p_ht15, 0.01), 2)

        rows.append(entry)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(WC_MODEL_FILE, index=False)
    log.info(f"  WC model tips saved → {WC_MODEL_FILE}")
    return df


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
    tips["league"] = "World Cup"
    append_sharp_signals(tips)

    # ML model value analysis
    log.info("Running ML model on WC fixtures...")
    _add_model_value(history)

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
