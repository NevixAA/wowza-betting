"""
Standard O/U 2.5 SERVING on the repaired recipe (owner 2026-10-28: "switch Standard now").
==========================================================================================

    config.STANDARD_OU25_SERVING_RECIPE   "legacy_v9" (default, rollback) | "v9_repaired"

SCOPE — deliberately the narrowest possible serving change:
  * ONLY the `p_over25` of STANDARD-format fixtures. New Format (legacy NF model), the BTTS /
    O1.5 / O3.5 side models and the HT models keep scoring the legacy feature frame exactly as
    before: this module never touches the frame they read.
  * Everything downstream (edge, tiers, predict.yml thresholds + best_params, direction guard,
    ledgers, Telegram, stake plan) is unchanged in logic; it simply reads a different number.

HOW A STANDARD FIXTURE IS SCORED WHEN THE SWITCH IS "v9_repaired"
  1. history  = v9's own loaded history (the frame predict already has), Standard leagues only,
                COVID seasons excluded exactly as training excluded them;
  2. identity = src/identity_repair.canonicalise_and_dedupe with the FROZEN alias map persisted in
                the payload at training time (`identity_alias_map`), derive=False — serving never
                derives a new alias, so train and serve canonicalise identically;
  3. features = src/features_repaired.build_upcoming_features_repaired (the SAME function training
                ran; board fixtures appended as unplayed rows, league-scoped, as-of) — a fixture's
                features depend only on history, never on the other fixtures on the board;
  4. score    = src/model.predict_proba with the payload's persisted train-only imputer (no batch
                medians), i.e. the served meta-blend probability.
  Invariant 8 (no rolling form -> no bet) is evaluated on the REPAIRED form features: a fixture
  whose repaired form is missing is flagged `serving_no_form` and betting.evaluate_value forces it
  to AVOID, exactly as the legacy guard does for legacy features.

FAILURE POLICY - FAIL CLOSED (R2 finding 17, 2026-11-01; owner doc
docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md). Any failure here (file missing, payload not a valid
repaired Standard OU25 payload, feature build error, row misalignment, non-finite probability)
leaves EVERY Standard row WITHOUT an O/U 2.5 probability (p_over25 = NaN -> no edge -> AVOID -> no
tip), stamps those rows `serving_recipe = "v9_repaired_failed"`, logs an ERROR and records
status FAIL in output/standard_serving_health.json (system health RED). It never raises, so New
Format, side and HT tips still go out. There is NO row-level legacy fallback any more: the old
policy re-scored with the legacy model (stamped "legacy_v9_fallback") and kept sending its tips,
i.e. silently served a model the owner had moved away from.

CONFIGURATION. STANDARD_OU25_SERVING_RECIPE must be exactly "legacy_v9" or "v9_repaired" (unset /
empty = legacy_v9, the documented rollback). Any other value raises ServingConfigError from
check_config(), which predict_upcoming calls before fetching odds: a typo is a red workflow run,
never a silent legacy run. The explicit owner rollback "legacy_v9" works exactly as before.

Every predictions row carries `serving_recipe` when the switch is on (column absent when off, so
the default path is byte-identical) and `model_version` names the artifact that produced its
p_over25 (`standard_repaired_<trained>_<sha8>` for repaired rows).
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
    """STANDARD_OU25_SERVING_RECIPE holds a value that is neither legacy_v9 nor v9_repaired."""
MODEL_TYPE = "standard_repaired"
CORE_FORM = ("home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5")
#: what the training workflow writes into the payload; serving refuses a payload without it
REQUIRED_KEYS = ("models", "feature_cols", "imputer", "recipe", "research", "identity_alias_map")


def check_config() -> str:
    """Raise ServingConfigError unless the configured recipe is valid; return it.

    Reads config.STANDARD_OU25_SERVING_RECIPE_RAW (the env value before config.py normalised it)
    so a typo cannot be laundered into legacy_v9 on the way in. A test or preview that sets
    config.STANDARD_OU25_SERVING_RECIPE directly is honoured when the two disagree only because
    the normalised value was overridden."""
    norm = getattr(config, "STANDARD_OU25_SERVING_RECIPE", LEGACY)
    raw = getattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", norm)
    raw = (str(raw).strip() if raw is not None else "") or LEGACY
    if raw not in (LEGACY, REPAIRED):
        raise ServingConfigError(
            f"STANDARD_OU25_SERVING_RECIPE={raw!r} is not a valid serving recipe; expected one of "
            f"{(LEGACY, REPAIRED)}. Fix the value in .github/workflows/predict.yml "
            f"(rollback doc: docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md). Predict refuses to start "
            f"rather than silently serving the legacy model.")
    if norm not in (LEGACY, REPAIRED):
        raise ServingConfigError(f"config.STANDARD_OU25_SERVING_RECIPE={norm!r} is not a valid serving recipe")
    if not getattr(config, "STANDARD_OU25_SERVING_RECIPE_EXPLICIT", True) and norm == LEGACY:
        log.warning("STANDARD_OU25_SERVING_RECIPE is not set: Standard O/U 2.5 is served by legacy_v9 "
                    "(the documented rollback). Production sets v9_repaired in predict.yml.")
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
    checks = {"recipe": payload.get("recipe") == REPAIRED,
              "target": payload.get("target", "over25") == "over25" and rs.get("target", "over25") == "over25",
              "track": rs.get("track") == "standard",
              "price_mode": rs.get("price_mode", "none") == "none",
              "team_scope": rs.get("team_scope", "league") == "league"}
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
    if getattr(config, "EXCLUDE_COVID_SEASONS", False) and "season" in h.columns:
        h = h[~h["season"].isin(config.COVID_SEASONS)]
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
    feats = build_upcoming_features_repaired(ident, history, price_mode="none", team_scope="league")
    if len(feats) != len(ident) or not (
            feats["home_team"].astype(str).to_numpy() == ident["home_team"].astype(str).to_numpy()).all():
        raise ValueError("repaired feature rows do not line up with the fixtures")
    p = np.asarray(predict_proba(feats, payload=payload), dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("repaired payload produced a non-finite probability")
    no_form = feats[list(CORE_FORM)].isna().any(axis=1).to_numpy()
    return pd.DataFrame({"p_over25_repaired": p, "serving_no_form": no_form}, index=fixtures.index)


def apply(feat: pd.DataFrame, std_mask: pd.Series, probas: np.ndarray, historical: pd.DataFrame,
          payload: dict | None = None, model_file: Path | None = None) -> np.ndarray:
    """Called by src/predict.predict_upcoming AFTER the legacy scores are in `probas`.

    legacy_v9   -> returns `probas` untouched and adds NO column (byte-identical default path).
    v9_repaired -> overwrites `probas` on Standard rows with the repaired probability and adds
                   `serving_recipe` (+ `serving_no_form`) to `feat`. On ANY scoring failure the
                   Standard rows get NaN (no probability, no bet), serving_recipe=v9_repaired_failed.
                   Never raises for a scoring failure (an invalid recipe raises ServingConfigError).
    """
    global _LAST_HEALTH
    recipe = serving_recipe()
    if recipe != REPAIRED:
        _LAST_HEALTH = _health_record(recipe, "PASS", int(std_mask.sum()), 0, 0)
        return probas
    out = np.array(probas, dtype=float, copy=True)
    feat["serving_recipe"] = LEGACY                       # NF / unknown rows: the legacy models
    feat["serving_no_form"] = False
    if not std_mask.any():
        _LAST_HEALTH = _health_record(recipe, "PASS", 0, 0, 0)
        return out
    try:
        pay = payload if payload is not None else load_payload(model_file)
        validate_payload(pay)
        hist = serving_history(historical, pay)
        res = score_fixtures(feat.loc[std_mask], hist, pay)
        out[std_mask.to_numpy()] = res["p_over25_repaired"].to_numpy()
        feat.loc[std_mask, "serving_recipe"] = REPAIRED
        feat.loc[std_mask, "serving_no_form"] = res["serving_no_form"].to_numpy()
        log.info(f"  Standard O/U 2.5 served by the REPAIRED recipe: {int(std_mask.sum())} fixtures "
                 f"({int(res['serving_no_form'].sum())} without repaired rolling form -> AVOID)")
        _LAST_HEALTH = _health_record(recipe, "PASS", int(std_mask.sum()), int(std_mask.sum()), 0)
    except Exception as e:                                             # noqa: BLE001
        out[std_mask.to_numpy()] = np.nan                              # FAIL CLOSED: no probability
        feat.loc[std_mask, "serving_recipe"] = FAILED
        feat.loc[std_mask, "serving_no_form"] = False
        log.error(f"  STANDARD_OU25_SERVING_RECIPE=v9_repaired but repaired scoring FAILED "
                  f"({type(e).__name__}: {e}) - {int(std_mask.sum())} Standard fixture(s) get NO "
                  f"probability, NO bet (fail closed; stamped serving_recipe={FAILED}; no legacy fallback)")
        _LAST_HEALTH = _health_record(recipe, "FAIL", int(std_mask.sum()), 0, int(std_mask.sum()),
                                      error=f"{type(e).__name__}: {e}")
    return out


def _health_record(recipe: str, status: str, n_std: int, n_repaired: int, n_unscored: int,
                   error: str = "") -> dict:
    from datetime import datetime, timezone
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "serving_recipe": recipe, "status": status, "standard_fixtures": n_std,
            "served_by_repaired": n_repaired, "unscored_fail_closed": n_unscored, "error": error,
            "policy": "v9_repaired failure -> no probability, no bet, status FAIL (no legacy fallback); "
                      "invalid recipe -> ServingConfigError before odds are fetched; "
                      "rollback: docs/STANDARD_REPAIRED_FAILCLOSED_ROLLBACK.md"}


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
