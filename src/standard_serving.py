"""
Standard O/U 2.5 serving switch - ONE recipe since 2026-11-15.
===============================================================

    config.STANDARD_OU25_SERVING_RECIPE   "legacy_v9" (the only servable value)

"legacy_v9" serves models/model_v9_standard.pkl, the clean-retrain Standard champion (std_v9_repaired,
owner clean-data switch 2026-11-09), scored by src/predict.py on build_upcoming_features exactly like
New Format, the side markets and HT. apply() leaves those probabilities untouched.

RETIRED (owner 2026-11-15, "fix what you need"): "v9_repaired", the 2026-10-28 switch to
models/model_v9_standard_repaired.pkl. Its train-only 1.0 imputer filled xG and inside-box shots -
features with NO training values - with 0.0 on every training row, so the model learned an invented
"xG = 0" (reports/REAL_DATA_FINAL_AUDIT.md T5). The serving branch is gone, the value is refused
like any unknown one (check_config -> ServingConfigError before predict fetches odds), the artifact
stays on disk for audit and the registry pointer records the retirement
(src.promotion_gate.retire_model_type). docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md.

What stays here, for audit and research scripts only (scripts/r2b_serving_parity.py,
scripts/audit/au2_*): validate_payload / load_payload / serving_history / score_fixtures - how the
retired recipe scored a fixture (frozen identity alias map, src/features_repaired, persisted imputer,
predict_proba(serving=True), which refuses the retired payload). Readers of old boards keep the
stamps: "v9_repaired" rows were scored by the standard_repaired model, "v9_repaired_failed" rows
failed closed with no probability, "legacy_v9_fallback" rows are pre-R2.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

log = logging.getLogger(__name__)

LEGACY, REPAIRED, FALLBACK = "legacy_v9", "v9_repaired", "legacy_v9_fallback"
#: stamp for Standard rows whose repaired scoring failed (fail closed: no probability, no bet).
#: FALLBACK ("legacy_v9_fallback") is no longer produced; kept so old boards / readers still parse.
FAILED = "v9_repaired_failed"
HEALTH_FILE = Path(__file__).resolve().parents[1] / "output" / "standard_serving_health.json"
_LAST_HEALTH: dict = {}


class ServingConfigError(RuntimeError):
    """STANDARD_OU25_SERVING_RECIPE holds a value other than legacy_v9 (unknown, or RETIRED)."""
MODEL_TYPE = "standard_repaired"
CORE_FORM = ("home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5")
#: what the training workflow writes into the payload; serving refuses a payload without it
REQUIRED_KEYS = ("models", "feature_cols", "imputer", "recipe", "research", "identity_alias_map")


def check_config() -> str:
    """Raise ServingConfigError unless the configured recipe is the one servable recipe (legacy_v9);
    return it.

    RETIREMENT (owner 2026-11-15, "fix what you need"): v9_repaired - models/model_v9_standard_repaired.pkl,
    whose 1.0 imputer filled xG / inside-box shots with 0.0 on every training row - is no longer a
    serving recipe. It is refused like any unknown value (config.RETIRED_STANDARD_OU25_SERVING_RECIPES
    names it); the file is kept for audit only.

    Reads config.STANDARD_OU25_SERVING_RECIPE_RAW (the env value before config.py normalised it)
    so a typo cannot be laundered into legacy_v9 on the way in."""
    norm = getattr(config, "STANDARD_OU25_SERVING_RECIPE", LEGACY)
    raw = getattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", norm)
    raw = (str(raw).strip() if raw is not None else "") or LEGACY
    retired = getattr(config, "RETIRED_STANDARD_OU25_SERVING_RECIPES", {})
    for v in (raw, norm):
        if v in retired:
            r = retired[v]
            raise ServingConfigError(
                f"STANDARD_OU25_SERVING_RECIPE={v!r} was RETIRED on {r['retired_on']} by {r['approved_by']} "
                f"(\"{r['owner_words']}\"): {r['why']}. The only serving recipe is {LEGACY!r}. Fix the "
                f"value in .github/workflows/predict.yml (docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md).")
    if raw != LEGACY:
        raise ServingConfigError(
            f"STANDARD_OU25_SERVING_RECIPE={raw!r} is not a valid serving recipe; the only one is "
            f"{LEGACY!r}. Fix the value in .github/workflows/predict.yml. Predict refuses to start "
            f"rather than guessing.")
    if norm != LEGACY:
        raise ServingConfigError(f"config.STANDARD_OU25_SERVING_RECIPE={norm!r} is not a valid serving recipe")
    return norm


def serving_recipe() -> str:
    """The configured recipe. Raises ServingConfigError on an invalid value (never maps it to
    legacy_v9 silently)."""
    return check_config()


def validate_payload(payload: dict) -> None:
    """Raise ValueError unless `payload` is a repaired Standard OU25 payload this module can serve."""
    if not isinstance(payload, dict):
        raise ValueError("payload is not a dict")
    missing = [k for k in REQUIRED_KEYS if k not in payload or payload.get(k) in (None, "")]
    if missing:
        raise ValueError(f"payload missing {missing}")
    rs = payload.get("research") or {}
    # Every fact must be STATED by the payload. A missing target / price_mode / team_scope used to
    # pass as if it held the servable value.
    checks = {"recipe": payload.get("recipe") == REPAIRED,
              "target": payload.get("target") == "over25" and rs.get("target") == "over25",
              "track": rs.get("track") == "standard",
              "price_mode": rs.get("price_mode") == "none",
              "team_scope": rs.get("team_scope") == "league"}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"not a servable repaired Standard OU25 payload ({bad})")
    if not isinstance(payload.get("identity_alias_map"), dict):
        raise ValueError("identity_alias_map is not a {league: {alias: canonical}} dict")


def load_payload(path: Path | None = None) -> dict:
    p = Path(path or config.MODEL_FILE_STANDARD_REPAIRED)
    with open(p, "rb") as f:
        payload = pickle.load(f)
    validate_payload(payload)
    return payload


def serving_history(historical: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """Standard-league history, COVID-filtered as training was, canonicalised with the FROZEN
    alias map (derive=False). Every operation is league-local, so restricting to Standard leagues
    first is exact, not an approximation."""
    from src.identity_repair import canonicalise_and_dedupe
    h = historical[historical["league"].astype(str).isin(config.STANDARD_FORMAT_LEAGUES)]
    if config.EXCLUDE_COVID_SEASONS:
        h = h[~config.covid_season_mask(h)]          # same per-league rule as pipeline.mode_train
    d, _rep = canonicalise_and_dedupe(h, payload["identity_alias_map"], derive=False)
    return d


def score_fixtures(fixtures: pd.DataFrame, history: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """Repaired p_over25 for Standard fixtures. `fixtures` needs date, league, home_team,
    away_team; `history` = serving_history(). Returns a frame aligned to `fixtures.index` with
    p_over25_repaired and serving_no_form. Raises on any misalignment."""
    from src.features_repaired import build_upcoming_features_repaired
    from src.model import predict_proba
    if fixtures.empty:
        return pd.DataFrame({"p_over25_repaired": pd.Series(dtype=float),
                             "serving_no_form": pd.Series(dtype=bool)}, index=fixtures.index)
    bad = set(fixtures["league"].astype(str)) - set(config.STANDARD_FORMAT_LEAGUES)
    if bad:
        raise ValueError(f"non-Standard leagues passed to the Standard repaired scorer: {sorted(bad)}")
    ident = fixtures[["date", "league", "home_team", "away_team"]].reset_index(drop=True)
    # the FROZEN map that canonicalised `history` (serving_history) also places the board's
    # spelling of a renamed club ('Stockport County FC' -> 'Stockport'): the same evidence as train
    feats = build_upcoming_features_repaired(ident, history, price_mode="none", team_scope="league",
                                             identity_aliases=payload.get("identity_alias_map"))
    if len(feats) != len(ident) or not (
            feats["home_team"].astype(str).to_numpy() == ident["home_team"].astype(str).to_numpy()).all():
        raise ValueError("repaired feature rows do not line up with the fixtures")
    # serving=True (real-data guard, 2026-11-15): a payload that fills a feature with no training
    # values refuses to score (raises -> fail closed below), and a fixture without its core form
    # gets NaN, which is exactly the serving_no_form rows.
    p = np.asarray(predict_proba(feats, payload=payload, serving=True), dtype=float)
    no_form = feats[list(CORE_FORM)].isna().any(axis=1).to_numpy()
    if not np.isfinite(p[~no_form]).all():
        raise ValueError("repaired payload produced a non-finite probability")
    return pd.DataFrame({"p_over25_repaired": p, "serving_no_form": no_form}, index=fixtures.index)


def apply(feat: pd.DataFrame, std_mask: pd.Series, probas: np.ndarray, historical: pd.DataFrame,
          payload: dict | None = None, model_file: Path | None = None) -> np.ndarray:
    """Called by src/predict.predict_upcoming AFTER the Standard scores are in `probas`.

    The only servable recipe is legacy_v9 (models/model_v9_standard.pkl): `probas` comes back untouched
    and no column is added. The v9_repaired branch that re-scored Standard rows with
    models/model_v9_standard_repaired.pkl was removed when the owner RETIRED that payload
    (2026-11-15, "fix what you need"); an invalid or retired recipe raises ServingConfigError.
    `payload` / `model_file` / `historical` are accepted for call compatibility and unused."""
    global _LAST_HEALTH
    recipe = serving_recipe()
    _LAST_HEALTH = _health_record(recipe, "PASS", int(std_mask.sum()), 0, 0)
    return probas


def _health_record(recipe: str, status: str, n_std: int, n_repaired: int, n_unscored: int,
                   error: str = "") -> dict:
    from datetime import datetime, timezone
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "serving_recipe": recipe, "status": status, "standard_fixtures": n_std,
            "served_by_repaired": n_repaired, "unscored_fail_closed": n_unscored, "error": error,
            "policy": "legacy_v9 is the only serving recipe; v9_repaired RETIRED 2026-11-15 (owner); an "
                      "invalid or retired recipe -> ServingConfigError before odds are fetched "
                      "(docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md)"}


def last_health() -> dict:
    return dict(_LAST_HEALTH)


def write_health(path: Path | None = None) -> dict:
    """Write this process's last serving outcome. No-op (returns {}) when apply() never ran."""
    import json
    if not _LAST_HEALTH:
        return {}
    p = Path(path or HEALTH_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_LAST_HEALTH, indent=1) + "\n", encoding="utf-8")
    return dict(_LAST_HEALTH)
