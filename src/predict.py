"""
Upcoming Fixture Predictor — v9
================================
v8 predict.py + three additions from v7:

1. Odds drift snapshot + enrichment
   Every run snapshots current odds to odds_history_v9.json.
   Drift columns (over_drift, under_drift, drift_signal) are added to
   predictions and used by betting.py to adjust signal_tier.

2. Postponement detection
   After fetching live fixtures, compare against the previous predictions.csv.
   Any fixture that was predicted but has disappeared from OddsAPI is flagged
   as POSTPONED in the output.

3. Both-losing guard
   Handled in betting.py via v7's best_params.json — predict.py just passes
   the league column through so betting.evaluate_value() can apply the guard.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)


# ── Fixture + odds fetching ───────────────────────────────────────────────────

def _fetch_odds_api(days_ahead: int = 7) -> pd.DataFrame:
    """Fetch upcoming fixtures + O/U 2.5 odds from OddsAPI for all enabled leagues."""
    try:
        import requests
    except ImportError:
        log.warning("requests not installed")
        return pd.DataFrame()

    rows = []
    for league, sport_key in config.ODDS_API_SPORT_KEYS.items():
        if league not in config.ENABLED_LEAGUES:
            continue
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey": config.ODDS_API_KEY,
                    "regions": "eu",
                    "markets": "totals,btts",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            if r.status_code != 200:
                log.debug(f"OddsAPI {league}: HTTP {r.status_code}")
                continue

            for event in r.json():
                dt = pd.to_datetime(event.get("commence_time", ""), errors="coerce", utc=True)
                if pd.isna(dt):
                    continue
                dt = dt.tz_localize(None)
                cutoff = datetime.utcnow() + timedelta(days=days_ahead)
                if dt > cutoff:
                    continue

                ov25, un25 = None, None
                ov15, ov35 = None, None
                btts_yes    = None
                for bm in event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "totals":
                            for outcome in mkt["outcomes"]:
                                pt = outcome.get("point")
                                nm = outcome.get("name")
                                pr = outcome.get("price")
                                if pt == 2.5 and nm == "Over"  and not ov25:  ov25 = pr
                                if pt == 2.5 and nm == "Under" and not un25:  un25 = pr
                                if pt == 1.5 and nm == "Over"  and not ov15:  ov15 = pr
                                if pt == 3.5 and nm == "Over"  and not ov35:  ov35 = pr
                        elif mkt["key"] == "btts":
                            for outcome in mkt["outcomes"]:
                                if outcome.get("name") == "Yes" and not btts_yes:
                                    btts_yes = outcome.get("price")
                    if ov25:
                        break

                if ov25 and un25:
                    rows.append({
                        "league":       league,
                        "date":         dt.normalize(),
                        "home_team":    event.get("home_team", ""),
                        "away_team":    event.get("away_team", ""),
                        "odds_over25":  float(ov25),
                        "odds_under25": float(un25),
                        "odds_over15":  float(ov15) if ov15 else None,
                        "odds_over35":  float(ov35) if ov35 else None,
                        "odds_btts":    float(btts_yes) if btts_yes else None,
                    })
        except Exception as e:
            log.debug(f"OddsAPI {league} exception: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["league", "home_team", "away_team"])


# ── Postponement detection ────────────────────────────────────────────────────

def _detect_postponed(current_df: pd.DataFrame) -> list[dict]:
    """
    Compare current live fixtures against the previous predictions.csv.
    Returns list of dicts for fixtures that were predicted but are no longer live.
    Only checks leagues covered by OddsAPI (reliable source-of-truth).
    """
    prev_file = config.OUTPUT_DIR / "predictions.csv"
    if not prev_file.exists():
        return []

    try:
        prev = pd.read_csv(prev_file, parse_dates=["date"])
    except Exception:
        return []

    # Only check upcoming (future) matches from previous run
    today = pd.Timestamp(datetime.utcnow().date())
    prev_upcoming = prev[prev["date"] >= today].copy()
    if prev_upcoming.empty:
        return []

    # Build set of current fixture keys
    current_keys = set(
        f"{r.home_team}|{r.away_team}|{str(r.date)[:10]}"
        for r in current_df.itertuples()
    )

    postponed = []
    for _, row in prev_upcoming.iterrows():
        k = f"{row['home_team']}|{row['away_team']}|{str(row['date'])[:10]}"
        if k not in current_keys:
            postponed.append({
                "date":       row["date"],
                "league":     row.get("league", ""),
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "status":     "POSTPONED",
            })

    return postponed


# ── Main predict function ─────────────────────────────────────────────────────

def predict_upcoming(
    historical:        pd.DataFrame,
    payload:           dict,
    payload_newformat: dict | None = None,
    side_payloads:     dict | None = None,
    days_ahead:        int = 7,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Full predict pipeline for upcoming fixtures.

    Parameters
    ----------
    historical : raw historical DataFrame from data_loader.load_all_matches()
    payload    : trained model payload from model.load_models()
    days_ahead : how far ahead to look for fixtures

    Returns
    -------
    (predictions_df, postponed_list)
      predictions_df : all fixtures with model score + signal_tier + drift columns
      postponed_list : fixtures that disappeared since last run
    """
    from src.feature_engineering import build_upcoming_features
    from src.model import predict_proba
    from src.betting import evaluate_value
    from src.drift import snapshot, enrich, purge_old

    # 1. Fetch live fixtures
    log.info("Fetching upcoming fixtures from OddsAPI...")
    upcoming = _fetch_odds_api(days_ahead)
    log.info(f"  OddsAPI: {len(upcoming)} fixtures found")

    if upcoming.empty:
        log.warning("No upcoming fixtures found")
        return pd.DataFrame(), []

    # 2. Detect postponements (before we snapshot — compare prev vs current)
    postponed = _detect_postponed(upcoming)
    if postponed:
        log.info(f"  Postponed/disappeared since last run: {len(postponed)}")
        for p in postponed:
            log.info(f"    [PST] {p['home_team']} vs {p['away_team']} ({str(p['date'])[:10]})")

    # 3. Snapshot current odds for drift tracking
    snapshot(upcoming)
    purge_old(cutoff_days=10)

    # 4. Build features
    log.info(f"Building features for {len(upcoming)} upcoming fixtures...")
    feat = build_upcoming_features(upcoming, historical)

    # 5. Model prediction — route each fixture to the correct model
    log.info("Running ensemble model(s)...")
    probas = np.full(len(feat), np.nan)

    std_mask = feat["league"].isin(config.STANDARD_FORMAT_LEAGUES)
    nf_mask  = feat["league"].isin(config.NEW_FORMAT_LEAGUES)

    if std_mask.any():
        p_std = predict_proba(feat[std_mask], payload=payload)
        probas[std_mask.values] = p_std.values
        log.info(f"  Standard model: {std_mask.sum()} fixtures")

    if nf_mask.any():
        nf_payload = payload_newformat if payload_newformat is not None else payload
        p_nf = predict_proba(feat[nf_mask], payload=nf_payload)
        probas[nf_mask.values] = p_nf.values
        log.info(f"  New-format model: {nf_mask.sum()} fixtures"
                 + (" (using standard model as fallback)" if payload_newformat is None else ""))

    # Any remaining unclassified fixtures (unknown league) use standard model as fallback
    unknown_mask = ~std_mask & ~nf_mask
    if unknown_mask.any():
        p_unk = predict_proba(feat[unknown_mask], payload=payload)
        probas[unknown_mask.values] = p_unk.values
        log.info(f"  Fallback (unknown league): {unknown_mask.sum()} fixtures")

    feat["p_over25"]   = probas
    feat["model_type"] = "unknown"
    feat.loc[std_mask, "model_type"] = "standard"
    feat.loc[nf_mask,  "model_type"] = "new_format"

    # Side-market predictions (BTTS / O1.5 / O3.5) — all known leagues
    # New-format leagues have odds_btts/over15/over35 after af_odds_history backfill.
    if side_payloads:
        side_mask = std_mask | nf_mask
        for target, sp in side_payloads.items():
            col = f"p_{target}"
            feat[col] = np.nan
            if side_mask.any():
                try:
                    feat.loc[side_mask, col] = predict_proba(feat[side_mask], payload=sp).values
                    log.info(f"  Side-market {target}: {side_mask.sum()} fixtures "
                             f"({std_mask.sum()} std + {nf_mask.sum()} nf)")
                except Exception as e:
                    log.warning(f"  Side-market {target} prediction failed: {e}")

    # 5b. HT model predictions (standard-format leagues only)
    from src.model import load_models as _load_models
    feat["p_ht_over05"] = np.nan
    feat["p_ht_over15"] = np.nan
    if config.HT_MODEL_FILE_05.exists() and std_mask.any():
        try:
            ht_payload_05 = _load_models(model_file=config.HT_MODEL_FILE_05)
            ht_payload_15 = _load_models(model_file=config.HT_MODEL_FILE_15) \
                            if config.HT_MODEL_FILE_15.exists() else None
            feat.loc[std_mask, "p_ht_over05"] = predict_proba(feat[std_mask], payload=ht_payload_05).values
            if ht_payload_15:
                feat.loc[std_mask, "p_ht_over15"] = predict_proba(feat[std_mask], payload=ht_payload_15).values
            log.info(f"  HT model: {std_mask.sum()} fixtures scored")
        except Exception as e:
            log.warning(f"  HT model prediction failed: {e}")

    # 6. Value betting evaluation (includes both-losing guard)
    log.info("Applying value betting logic...")
    feat = evaluate_value(feat)

    # 7. Enrich with drift columns (uses history just written in step 3)
    log.info("Enriching with odds drift signal...")
    feat = enrich(feat)

    # 8. Re-run evaluate_value now that drift columns exist (single pass)
    feat = evaluate_value(feat)

    return feat, postponed
