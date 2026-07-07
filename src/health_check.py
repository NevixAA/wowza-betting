"""
Predict health state
=====================
predict.py records the outcome of each OddsAPI fetch here; the Telegram notifier
reads it and alerts on an outage. Split this way because the CI predict step has
no Telegram token (only the notifier step does), and because health state must
persist across runs (output/predict_health.json is committed by predict.yml).

The outage this guards against: v9.2 requested an unsupported market -> every
league 422'd -> 0 fixtures -> 0 tips -> notifier sent nothing -> DOWN for ~2
weeks with no signal. Recording *why* the fetch was empty (all-leagues-error vs
genuinely-no-games) lets the notifier tell an outage apart from a quiet night.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import config

HEALTH_FILE = config.OUTPUT_DIR / "predict_health.json"


def _load() -> dict:
    if HEALTH_FILE.exists():
        try:
            return json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(state: dict) -> None:
    HEALTH_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_fetch(n_fixtures: int, stats: dict) -> None:
    """Update health state after a predict fetch. Does NOT send anything.

    stats = {"queried", "ok", "err", "kept"} from predict._fetch_odds_api.
    Bumps last_fixtures_iso only when fixtures were actually found, so the
    notifier can measure how long the system has been dry.
    """
    now = datetime.now(timezone.utc).isoformat()
    state = _load()
    state["last_run_iso"]     = now
    state["last_n_fixtures"]  = int(n_fixtures)
    state["last_stats"]       = dict(stats or {})
    if n_fixtures > 0:
        state["last_fixtures_iso"] = now
    _save(state)
