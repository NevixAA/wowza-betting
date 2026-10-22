"""P0 2026-10-22 — the prospective champion-vs-challenger experiment must actually log a challenger.

Before this, output/model_lineage/shadow_h2h.csv had p_challenger empty on 1,388/1,388 rows and
output/v91_shadow.csv had p_pro_hgb empty on 669/669: H-MODEL-01's prospective window was open and
collecting nothing. These tests pin the whole path: prediction -> champion raw -> challenger ->
first-sight append -> rerun cannot revise -> settlement adds results only -> paired evaluation;
plus the guard that fails loudly when a registered experiment logs nothing.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import config
import pipeline
from src import model_registry as reg
from src import shadow_compare as sc
from src import shadow_experiments as SE

FEATS = ["home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5"]
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _payload(seed: int, name: str) -> dict:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(1.3, 0.5, (400, len(FEATS))), columns=FEATS)
    y = (X.sum(axis=1) + rng.normal(0, 1, 400) > 5.2).astype(int)
    return {"target": "over25", "feature_cols": FEATS,
            "models": {name: LogisticRegression(C=0.5 if seed else 2.0).fit(X, y)}}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Registry, ledger, shadow and h2h files all redirected into tmp_path."""
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(SE, "EXPERIMENTS_DIR", tmp_path / "models/registry/experiments")
    monkeypatch.setattr(SE, "EXPERIMENTS_FILE", tmp_path / "models/registry/experiments.json")
    monkeypatch.setattr(SE, "LEDGER_FILE", tmp_path / "shadow_experiments.csv")
    monkeypatch.setattr(SE, "V91_SHADOW_FILE", tmp_path / "v91_shadow.csv")
    monkeypatch.setattr(sc, "SHADOW_FILE", tmp_path / "v91_shadow.csv")
    monkeypatch.setattr(pipeline, "H2H_FILE", tmp_path / "h2h.csv")
    champ_file = tmp_path / "model_v9_standard.pkl"
    import pickle
    champ = _payload(0, "logistic")
    champ_file.write_bytes(pickle.dumps(champ))
    monkeypatch.setitem(reg.MODEL_FILES, "standard", champ_file)
    monkeypatch.setattr(reg, "stamp_for",
                        lambda mt, path=None: {"model_version": f"{mt}_CHAMP_v1"})
    monkeypatch.setattr(reg, "load_challenger", lambda mt: None)       # retrain slot empty
    SE._payload_cache.clear()
    exp = SE.register({
        "experiment_id": "H-MODEL-01__standard", "hypothesis_id": "H-MODEL-01",
        "status": "ACTIVE", "model_type": "standard", "market": "over25",
        "pred_column": "p_over25", "v91_shadow_column": "p_pro_hgb",
        "challenger_family": "pro_hgb", "trained_at": "2026-10-20T00:00:00Z",
        "registered_at": "2026-10-20T00:00:00Z", "training_cutoff": "2026-09-28",
        "feature_schema": "abc"}, _payload(1, "pro_hgb"))
    return {"champ": champ, "exp": exp, "tmp": tmp_path}


def _preds(lab, n=6, shift=0.0, kickoff="2026-10-25T15:00:00Z"):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(1.3, 0.5, (n, len(FEATS))) + shift, columns=FEATS)
    from src.model import predict_proba
    d = pd.DataFrame({
        "_oa_event_id": [f"evt{i}" for i in range(n)], "date": "2026-10-25",
        "kickoff_utc": kickoff, "league": "Championship",
        "home_team": [f"H{i}" for i in range(n)], "away_team": [f"A{i}" for i in range(n)],
        "model_type": "standard", "impl_prob_over": 0.5, "best_side": "OVER",
        "signal_tier": "AVOID", "best_edge": 0.0, "odds_over25": 1.9, "odds_under25": 1.9,
        "git_sha": "data123", **{c: X[c] for c in FEATS}})
    d["p_over25"] = predict_proba(d, payload=lab["champ"]).values
    return d


def _run(lab, preds, now):
    """Exactly mode_predict's recorder sequence."""
    rows = SE.score(preds, {"standard": lab["champ"]}, now=now)
    SE.append_first_sight(rows)
    pipeline._log_shadow(preds, rows)
    pipeline._log_h2h(preds, rows)
    return rows


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_end_to_end_prediction_to_paired_evaluation(lab):
    tmp = lab["tmp"]
    preds = _preds(lab)
    snapshot = preds.copy()
    rows = _run(lab, preds, NOW)

    # served probabilities are untouched — the recorder is read-only on preds
    pd.testing.assert_frame_equal(preds, snapshot)

    led = pd.read_csv(SE.LEDGER_FILE)
    assert len(led) == 6
    assert led["challenger_version"].eq(lab["exp"]["challenger_version"]).all()
    assert led["p_challenger"].notna().all() and led["p_champion_raw"].notna().all()
    assert led["champion_version"].eq("standard_CHAMP_v1").all()
    # champion raw on the identical rows reproduces the served number exactly here
    assert np.allclose(led["p_champion_raw"], led["p_served"], atol=1e-6)
    assert not np.allclose(led["p_challenger"], led["p_champion_raw"])
    for c in ("fixture_id", "kickoff_utc", "prediction_ts", "p_market", "dataset_id",
              "challenger_sha256", "champion_sha256", "challenger_training_cutoff"):
        assert led[c].notna().all(), c

    sh = pd.read_csv(sc.SHADOW_FILE)
    assert sh["p_pro_hgb"].notna().all()
    assert np.allclose(sh.sort_values("fixture_id")["p_pro_hgb"],
                       led.sort_values("fixture_id")["p_challenger"])
    assert sh["p_v91"].isna().all(), "no v91 challenger is registered — must stay empty"

    h = pd.read_csv(pipeline.H2H_FILE)
    std = h[h.model_type == "standard"]
    assert len(std) == 6 and std["p_challenger"].notna().all()
    assert std["challenger_version"].eq(lab["exp"]["challenger_version"]).all()

    # a later run with a different board and a later clock must not revise one byte
    before = {p: _sha(p) for p in (SE.LEDGER_FILE, sc.SHADOW_FILE, pipeline.H2H_FILE)}
    later = _preds(lab, shift=0.4)
    _run(lab, later, NOW + timedelta(hours=2))
    assert {p: _sha(p) for p in before} == before

    # settlement adds results only
    v91 = pd.read_csv(sc.SHADOW_FILE, dtype=str, keep_default_na=False)
    goals = [(3, 1), (0, 0), (1, 1), (2, 2), (1, 0), (4, 0)]
    v91["home_goals"] = [str(g[0]) for g in goals]
    v91["away_goals"] = [str(g[1]) for g in goals]
    v91["result_source"] = "football-data"
    v91.to_csv(sc.SHADOW_FILE, index=False)
    old = pd.read_csv(SE.LEDGER_FILE, dtype=str, keep_default_na=False)
    assert SE.settle_file() == 6
    new = pd.read_csv(SE.LEDGER_FILE, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(old[SE.FIRST_SIGHT_COLS], new[SE.FIRST_SIGHT_COLS])
    assert new["y"].tolist() == ["1", "0", "0", "1", "0", "1"]
    assert SE.settle_file() == 0, "settlement must be idempotent"

    # the prospective evaluator computes paired log loss / Brier on the settled rows
    res = SE.evaluate(n_boot=200)["H-MODEL-01__standard|standard"]
    y = new["y"].astype(float).to_numpy()
    a = new["p_champion_raw"].astype(float).to_numpy()
    b = new["p_challenger"].astype(float).to_numpy()
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))          # noqa: E731
    assert res["settled_paired"] == 6
    assert res["delta_log_loss"] == pytest.approx((ll(b) - ll(a)).mean(), abs=1e-9)
    assert res["delta_brier"] == pytest.approx((((b - y) ** 2) - ((a - y) ** 2)).mean(),
                                               abs=1e-9)


def test_first_sight_holds_when_the_ledger_already_has_the_fixture(lab):
    _run(lab, _preds(lab), NOW)
    first = pd.read_csv(SE.LEDGER_FILE)["p_challenger"].tolist()
    _run(lab, _preds(lab, shift=0.8), NOW + timedelta(hours=5))
    led = pd.read_csv(SE.LEDGER_FILE)
    assert len(led) == 6 and led["p_challenger"].tolist() == first


def test_kicked_off_fixture_is_never_logged(lab):
    rows = SE.score(_preds(lab, kickoff="2026-10-22T11:00:00Z"), {"standard": lab["champ"]},
                    now=NOW)
    assert rows.empty


def test_other_tracks_are_not_scored_by_a_standard_experiment(lab):
    p = _preds(lab)
    p.loc[:2, "model_type"] = "new_format"
    p.loc[3, "model_type"] = "unknown"           # standard model fallback, not the standard track
    rows = SE.score(p, {"standard": lab["champ"]}, now=NOW)
    assert sorted(rows["fixture_id"]) == ["evt4", "evt5"]


def test_tampered_artifact_is_not_scored(lab):
    f = SE.artifact_path(lab["exp"])
    import gzip
    f.write_bytes(gzip.compress(b"not the registered model"))
    SE._payload_cache.clear()
    assert SE.score(_preds(lab), {"standard": lab["champ"]}, now=NOW).empty


def test_settlement_refuses_a_prediction_written_after_kickoff(lab):
    rows = SE.score(_preds(lab), {"standard": lab["champ"]}, now=NOW)
    rows["prediction_ts"] = "2026-10-25T16:00:00+00:00"                 # after kickoff
    SE.append_first_sight(rows)
    oc = pd.DataFrame({"fixture_id": rows["fixture_id"], "match_date": "2026-10-25",
                       "home_goals": "2", "away_goals": "1"})
    _, n = SE.settle(pd.read_csv(SE.LEDGER_FILE, dtype=str, keep_default_na=False), oc)
    assert n == 0


def test_recorder_failure_cannot_reach_predict(lab, monkeypatch):
    monkeypatch.setattr(SE, "load_challenger", lambda e: (_ for _ in ()).throw(RuntimeError()))
    assert SE.score(_preds(lab), {"standard": lab["champ"]}, now=NOW).empty


# ── guard ─────────────────────────────────────────────────────────────────────────────────────
def _h2h_rows(n, at="2026-10-21T10:00:00+00:00"):
    return pd.DataFrame({"logged_at": [at] * n, "model_type": ["standard"] * n})


def test_guard_fails_loudly_when_a_registered_experiment_logs_nothing(lab):
    fails = SE.guard(now=NOW + timedelta(days=1), ledger=pd.DataFrame(), h2h=_h2h_rows(40))
    assert len(fails) == 1 and "ZERO challenger probabilities" in fails[0]


def test_guard_passes_once_challenger_probabilities_are_logged(lab):
    _run(lab, _preds(lab), NOW)
    assert SE.guard(now=NOW + timedelta(days=1), h2h=_h2h_rows(40)) == []


def test_guard_catches_a_path_that_worked_once_and_then_died(lab):
    _run(lab, _preds(lab), NOW)                                     # logged on 10-22
    fails = SE.guard(now=NOW + timedelta(days=5),                   # 10-27: window from 10-24
                     h2h=_h2h_rows(30, at="2026-10-26T09:00:00+00:00"))
    assert len(fails) == 1 and "ZERO challenger probabilities" in fails[0]


def test_guard_respects_the_grace_period(lab):
    assert SE.guard(now=datetime(2026, 10, 20, 12, tzinfo=timezone.utc),
                    ledger=pd.DataFrame(), h2h=_h2h_rows(40)) == []


def test_guard_fails_when_nothing_at_all_is_logged(lab):
    fails = SE.guard(now=NOW + timedelta(days=3), ledger=pd.DataFrame(), h2h=pd.DataFrame())
    assert len(fails) == 1 and "not running" in fails[0]


def test_guard_cli_exit_code(lab, monkeypatch):
    monkeypatch.setattr(SE, "guard", lambda **k: ["x: broken"])
    assert SE._main(["guard"]) == 1
    monkeypatch.setattr(SE, "guard", lambda **k: [])
    assert SE._main(["guard"]) == 0


# ── the real registered artifacts ────────────────────────────────────────────────────────────
REAL = Path(__file__).resolve().parents[1]


def test_registered_h_model_01_artifacts_verify_and_score_real_predictions():
    """The committed challengers load, match their registered sha256, read the champion's own
    feature columns, and produce probabilities on the committed predictions snapshot."""
    reg_file = REAL / "models/registry/experiments.json"
    if not reg_file.exists():
        pytest.skip("no experiment registry")
    exps = SE.load_registry(reg_file)
    assert {e["experiment_id"] for e in exps} >= {"H-MODEL-01__standard",
                                                  "H-MODEL-01__newformat"}
    preds_f = REAL / "output/predictions.csv"
    for e in exps:
        assert e["hypothesis_id"] == "H-MODEL-01" and e["training_cutoff"] == "2026-09-28"
        SE._payload_cache.pop(e["experiment_id"], None)
        payload = SE.load_challenger(e)
        assert payload is not None, f"{e['experiment_id']} sha mismatch"
        champ = __import__("pickle").loads(reg.MODEL_FILES[e["model_type"]].read_bytes())
        assert payload["feature_cols"] == champ["feature_cols"]
    if preds_f.exists():
        preds = pd.read_csv(preds_f)
        rows = SE.score(preds, now=datetime(2000, 1, 1, tzinfo=timezone.utc))
        if not rows.empty:
            assert rows["p_challenger"].between(0.01, 0.99).all()
            assert rows["p_champion_raw"].notna().all()
