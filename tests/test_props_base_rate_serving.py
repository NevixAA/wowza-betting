"""Owner decision 2026-11-18 for the rare props markets goals2 / goals3 / sot3 / sot4: "serve the clean
player_base_rate model - 100% real data, markets stay on. The clean ML candidates keep predicting as
challengers."

Committed state (run 43035218594, the run the five other clean champions come from):
  * every served props payload (config.MODEL_FILES, 9 markets) is a clean-harness payload with no
    withheld typed formula and no removed set-piece proxy; the registry pointer holds each file's sha256;
  * the four rare markets serve player_base_rate (reads the player id only) and park the ML candidate
    the clean-retrain head-to-head picks; the legacy payloads are kept only as registry rollback copies;
  * the withheld formulas are no longer built (feature_engineering._add_composites).
Synthetic: the serving path gives exactly the harness probability; the gate refuses an ML candidate
through the owner base-rate path and a base rate as a challenger.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config as pconfig  # noqa: E402
from player_model import props_contract as PC  # noqa: E402
from player_model import props_gate as PPG  # noqa: E402
from player_model import props_harness as H  # noqa: E402
from player_model.model import predict_proba, predict_with_diagnostics  # noqa: E402

RARE = ("goals2", "goals3", "sot3", "sot4")
FORBIDDEN = set(pconfig.TYPED_FORMULA_FEATURES_WITHHELD) | set(pconfig.SETPIECE_PROXIES_REMOVED)
T0 = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)


def _load(path: Path) -> dict:
    return PPG._HarnessUnpickler(open(path, "rb")).load()


# ── committed state ──────────────────────────────────────────────────────────────────────────────

def test_served_markets_are_the_nine_and_rare_markets_serve_the_base_rate():
    assert set(pconfig.SERVED_MARKETS) == set(pconfig.MODEL_FILES)
    assert pconfig.BASE_RATE_SERVED_MARKETS == set(RARE) == pconfig.HIDDEN_PENDING_CLEAN_RETRAIN
    assert "goals3" not in pconfig.MARKETS        # no retrain loop / promotion-gate thresholds


@pytest.mark.parametrize("market", sorted(pconfig.MODEL_FILES))
def test_every_served_payload_is_clean_and_registered(market):
    path = pconfig.MODEL_FILES[market]
    raw = path.read_bytes()
    pl = _load(path)
    assert pl["recipe"] == PC.RECIPE_CLEAN_HARNESS, market
    assert not (set(pl["feature_cols"]) & FORBIDDEN), market
    assert pl["provenance"]["run_id"] == "43035218594"
    ptr = json.loads(PPG.PROPS_POINTER_FILE.read_text(encoding="utf-8"))[market]
    assert ptr["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert ptr["promotion_source"] == PPG.CLEAN_RETRAIN_SOURCE
    assert ptr["training_cutoff"] == "2026-08-31"
    if market in RARE:
        assert pl["candidate"] == "player_base_rate" and pl["feature_cols"] == []
        assert pl["inputs"] == ["player_id"]


@pytest.mark.parametrize("market", RARE)
def test_rare_market_parks_its_clean_ml_challenger(market):
    meta = json.loads(PPG.challenger_meta_path(market).read_text(encoding="utf-8"))
    raw = gzip.decompress(PPG.challenger_path(market).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta["artifact_sha256"]
    pl = pickle.loads(raw)
    assert pl["recipe"] == PC.RECIPE_CLEAN_HARNESS
    assert pl["candidate"] in ("hgb_nan_native", "recipe_retrained")
    assert not (set(pl["feature_cols"]) & FORBIDDEN)
    assert meta["gate"]["qualified"] is False             # research shadow: never promoted from the slot
    ptr = json.loads(PPG.PROPS_POINTER_FILE.read_text(encoding="utf-8"))[market]
    assert meta["parent_champion_version"] == ptr["model_version"]


def test_no_parked_challenger_or_served_payload_reads_a_withheld_feature():
    for p in sorted(PPG.REGISTRY_DIR.glob("*/challenger.pkl.gz")):
        pl = pickle.loads(gzip.decompress(p.read_bytes()))
        assert not (set(pl.get("feature_cols") or []) & FORBIDDEN), p


def test_composites_no_longer_build_the_withheld_formulas():
    import inspect
    from player_model import feature_engineering as FE
    src = inspect.getsource(FE._add_composites)
    for f in pconfig.TYPED_FORMULA_FEATURES_WITHHELD:
        assert f'f["{f}"]' not in src, f


# ── synthetic: serving path, gate ────────────────────────────────────────────────────────────────

def _frame():
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.date_range("2026-01-01", periods=60, freq="D"):
        for p in range(30):
            if rng.random() < 0.5:
                continue
            rows.append({"date": d, "player_id": p + 1, "f1": rng.normal(),
                         "y": float(rng.random() < (0.02 + 0.01 * (p % 5)))})
    return pd.DataFrame(rows)


def _base_rate_artifact(market="sot4"):
    f = _frame()
    cut = pd.Timestamp("2026-02-15")
    tr, cal = f[f["date"] < cut], f[f["date"] >= cut]
    m = H.PlayerBaseRate().fit(tr[[]], tr["y"], cal[[]], cal["y"],
                               players_fit=tr["player_id"].to_numpy(), players_cal=cal["player_id"].to_numpy())
    art = {"market": market, "candidate": "player_base_rate", "harness_version": H.HARNESS_VERSION,
           "hand_adjustment": "none", "model": m, "state": m.state(),
           "window": {"fit_dates": ["2026-01-01", "2026-02-14"], "calibration_dates": ["2026-02-15", "2026-03-01"],
                      "fit_rows": len(tr), "calibration_rows": len(cal)}, "owner_decisions": {}}
    return pickle.dumps(art), m


def _ml_artifact(market="sot4"):
    f = _frame()
    cut = pd.Timestamp("2026-02-15")
    tr, cal = f[f["date"] < cut], f[f["date"] >= cut]
    m = H.HGBNaNNative().fit(tr[["f1"]], tr["y"], cal[["f1"]], cal["y"])
    art = {"market": market, "candidate": "hgb_nan_native", "harness_version": H.HARNESS_VERSION,
           "hand_adjustment": "none", "model": m, "state": m.state(),
           "window": {"fit_dates": ["2026-01-01", "2026-02-14"], "calibration_dates": ["2026-02-15", "2026-03-01"],
                      "fit_rows": len(tr), "calibration_rows": len(cal)}, "owner_decisions": {}}
    return pickle.dumps(art), m


@pytest.fixture()
def rtmp(tmp_path, monkeypatch):
    models = tmp_path / "models"
    lin = tmp_path / "out" / "model_lineage"
    files = {m: models / f"model_player_{m}.pkl" for m in pconfig.SERVED_MARKETS}
    monkeypatch.setattr(pconfig, "MODEL_FILES", files)
    monkeypatch.setattr(PPG, "REGISTRY_DIR", models / "registry" / "props")
    monkeypatch.setattr(PPG, "PROPS_POINTER_FILE", models / "registry" / "props" / "props_champions.json")
    monkeypatch.setattr(PPG, "LINEAGE_DIR", lin)
    monkeypatch.setattr(PPG, "DECISIONS_FILE", lin / "props_promotion_decisions.csv")
    monkeypatch.setattr(PPG, "CHAMPION_HISTORY", lin / "props_champion_history.csv")
    models.mkdir(parents=True)
    return files


def _kw(raw):
    return dict(expected_sha256=hashlib.sha256(raw).hexdigest(), run_id="1", code_sha="abc",
                holdout_metrics={"n": 10, "events": 1}, approval="serve the clean player_base_rate",
                approved_on="2026-11-18", now=T0)


def test_base_rate_serves_exactly_the_harness_probability(rtmp):
    raw, model = _base_rate_artifact()
    PPG.promote_clean_retrain("sot4", raw, owner_base_rate=True, **_kw(raw))
    pl = pickle.loads(rtmp["sot4"].read_bytes())
    df = pd.DataFrame({"player_id": [1, "2", 3.0, 999, None], "f1": [0.1, 0.2, 0.3, 0.4, 0.5]})
    served = predict_proba(df, pl).to_numpy()
    want = model.predict(None, players=np.array([1, 2, 3, 999, np.nan]))
    np.testing.assert_array_equal(served, want)
    assert served[3] == model.market_rate_ and served[4] == model.market_rate_   # unseen / no id
    assert predict_with_diagnostics(df, pl)["n_missing"].eq(0).all()              # it reads no feature


def test_gate_refuses_an_ml_candidate_on_the_base_rate_path_and_a_base_rate_challenger(rtmp):
    raw_ml, _ = _ml_artifact()
    with pytest.raises(PermissionError, match="player_base_rate only"):
        PPG.promote_clean_retrain("sot4", raw_ml, owner_base_rate=True, **_kw(raw_ml))
    raw_br, _ = _base_rate_artifact()
    with pytest.raises(PermissionError, match="hidden"):
        PPG.promote_clean_retrain("sot4", raw_br, **_kw(raw_br))           # no owner flag
    with pytest.raises(ValueError, match="champion recipe"):
        PPG.park_clean_challenger("sot4", raw_br, **_kw(raw_br))
    raw_g, _ = _base_rate_artifact(market="goals")
    with pytest.raises(PermissionError, match="BASE_RATE_SERVED_MARKETS"):   # not a rare market
        PPG.promote_clean_retrain("goals", raw_g, owner_base_rate=True, **_kw(raw_g))


def test_parked_ml_challenger_is_scored_first_sight_beside_the_base_rate(rtmp):
    raw_br, _ = _base_rate_artifact()
    rec = PPG.promote_clean_retrain("sot4", raw_br, owner_base_rate=True, **_kw(raw_br))
    raw_ml, ml = _ml_artifact()
    park = PPG.park_clean_challenger("sot4", raw_ml, **_kw(raw_ml))
    payload, meta = PPG.load_parked("sot4")
    assert meta["parent_champion_version"] == rec["model_version"] == park["champion_version"]
    df = pd.DataFrame({"player_id": [1, 2], "f1": [0.3, -0.2]})
    np.testing.assert_array_equal(predict_proba(df, payload).to_numpy(), ml.predict(df[["f1"]]))
    r = PPG.review_parked("sot4", now=T0)
    assert r["verdict"] == PPG.SHADOW and not r["promoted"]                # research shadow, kept
