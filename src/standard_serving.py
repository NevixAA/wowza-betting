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

FAILURE POLICY. A crashed predict sends zero tips, which is worse than serving the approved
rollback. Any failure here (file missing, payload not a valid repaired Standard OU25 payload,
feature build error, row misalignment) keeps the LEGACY probability for every Standard row and
stamps those rows `serving_recipe = "legacy_v9_fallback"` with the reason logged as an ERROR, so
the fallback is visible in predictions.csv, the ledgers' model_version and system health.

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
MODEL_TYPE = "standard_repaired"
CORE_FORM = ("home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5")
#: what the training workflow writes into the payload; serving refuses a payload without it
REQUIRED_KEYS = ("models", "feature_cols", "imputer", "recipe", "research", "identity_alias_map")


def serving_recipe() -> str:
    r = getattr(config, "STANDARD_OU25_SERVING_RECIPE", LEGACY)
    return r if r in (LEGACY, REPAIRED) else LEGACY


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
                   `serving_recipe` (+ `serving_no_form`) to `feat`. Never raises.
    """
    if serving_recipe() != REPAIRED:
        return probas
    out = np.array(probas, dtype=float, copy=True)
    feat["serving_recipe"] = LEGACY                       # NF / unknown rows: the legacy models
    feat["serving_no_form"] = False
    if not std_mask.any():
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
    except Exception as e:                                             # noqa: BLE001
        feat.loc[std_mask, "serving_recipe"] = FALLBACK
        log.error(f"  STANDARD_OU25_SERVING_RECIPE=v9_repaired but repaired scoring FAILED "
                  f"({type(e).__name__}: {e}) — {int(std_mask.sum())} Standard fixture(s) served by "
                  f"the LEGACY model and stamped serving_recipe={FALLBACK}")
    return out
