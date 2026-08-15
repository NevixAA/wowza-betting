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
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger(__name__)

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


FEATURE_HEALTH_FILE = config.OUTPUT_DIR / "feature_health.json"


def record_feature_health(feat, cols_by_model: dict) -> dict:
    """Record how much of each model's input ACTUALLY arrived at predict time.

    A feature that is real during training but arrives blank or constant at predict time is
    silently destructive: _prep() imputes it, StandardScaler maps the constant to a fixed
    z-score, and the logistic member's output shifts. That is exactly the mechanism that
    crushed new-format P(over) to ~0.36 and produced a season of one-sided UNDER tips.

    Nothing warned about it, so it went unnoticed for months (audit 2026-08-15: 28 of the
    standard model's 65 features — 43% — were degenerate, 8 fully blank). This writes
    output/feature_health.json and logs a warning so the next occurrence is visible on the
    predict run that causes it. Diagnostic only: it never changes a prediction.
    """
    import pandas as pd

    state = {"ts": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model_type, cols in (cols_by_model or {}).items():
        sub = feat[feat["model_type"] == model_type] if "model_type" in feat.columns else feat
        if sub.empty or not cols:
            continue
        blank, const = [], []
        for c in cols:
            if c not in sub.columns:
                blank.append(c)
                continue
            s = pd.to_numeric(sub[c], errors="coerce")
            if s.isna().all():
                blank.append(c)
            elif s.nunique() <= 1:
                const.append(c)
        n = len(cols)
        deg = len(blank) + len(const)
        state["models"][model_type] = {
            "fixtures": int(len(sub)), "n_features": n,
            "blank": sorted(blank), "constant": sorted(const),
            "degenerate": deg, "degenerate_pct": round(deg / n * 100, 1) if n else 0.0,
        }
        if deg:
            log.warning(
                f"[feature-health] {model_type}: {deg}/{n} features degenerate "
                f"({deg / n * 100:.0f}%) — {len(blank)} blank, {len(const)} constant. "
                f"Blank: {', '.join(sorted(blank)[:6])}{' …' if len(blank) > 6 else ''}"
            )
    try:
        FEATURE_HEALTH_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass
    return state


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
