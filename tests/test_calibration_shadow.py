"""Calibration SHADOW (2026-10-14): the log-odds Platt must never change a production number.

Pins: (1) production probabilities are byte-identical to the pre-shadow calibrator; (2) payload
structure is unchanged (no new keys in payload["models"]); (3) a pickle from before the shadow
returns None instead of failing; (4) the shadow log keeps FIRST sight only.
"""
import numpy as np
import pandas as pd
import pytest


def _team_frame(n=900, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "home_scored_last5": rng.normal(1.4, 0.5, n), "away_scored_last5": rng.normal(1.1, 0.5, n),
        "home_conceded_last5": rng.normal(1.2, 0.5, n), "away_conceded_last5": rng.normal(1.3, 0.5, n),
        "home_attack_str": rng.normal(1, 0.2, n),
    })
    z = 0.8 * (d.home_scored_last5 + d.away_scored_last5 - 2.5) + rng.normal(0, 1, n)
    d["over25"] = (z > 0).astype(int)
    return d


def test_team_production_unchanged_and_shadow_present():
    from src import model as M
    res = M.train(_team_frame(), "over25")
    X, _ = M._prep(_team_frame(seed=1), res["logistic"]["feature_cols"])
    for name, r in res.items():
        if name == "__meta__":
            continue
        m = r["model"]
        old = M._CalibratedModel(m._pipe, m._calibrator)          # exactly the pre-shadow wrapper
        assert np.array_equal(m.predict_proba(X), old.predict_proba(X))
        assert m.predict_proba_logit_shadow(X) is not None
    assert all(not k.startswith("__shadow") for k in res)           # no new results keys
    payload = {"models": {k: v["model"] for k, v in res.items()}, "feature_cols": res["logistic"]["feature_cols"]}
    df = _team_frame(seed=2)
    p = M.predict_proba(df, payload)
    s = M.predict_proba_logit_shadow(df, payload)
    assert s is not None and s.between(0, 1).all() and len(s) == len(p)


def test_team_old_pickle_returns_none():
    from src import model as M
    res = M.train(_team_frame(), "over25")
    for r in res.values():
        m = r["model"]
        if hasattr(m, "_calibrator_logit"):
            del m._calibrator_logit                                  # as a pre-shadow pickle
    payload = {"models": {k: v["model"] for k, v in res.items()}, "feature_cols": res["logistic"]["feature_cols"]}
    assert M.predict_proba_logit_shadow(_team_frame(seed=3), payload) is None
    M.predict_proba(_team_frame(seed=3), payload)                    # production still works


def test_player_production_unchanged():
    pm = pytest.importorskip("player_model.model")
    from player_model import config as pcfg
    rng = np.random.default_rng(0)
    n = 1200
    cols = list(pcfg.PLAYER_FEATURE_COLS)[:6]
    d = pd.DataFrame({c: rng.normal(0, 1, n) for c in cols})
    d["date"] = pd.date_range("2024-01-01", periods=n, freq="h")
    d["n_prev_games"] = 5
    d[pm.TARGETS["sot"]] = (d[cols[0]] + rng.normal(0, 1, n) > 0.3).astype(int)
    res = pm.train(d, "sot")
    X, _ = pm._prep(d)
    for r in res.values():
        m = r["model"]
        old = pm._PlattCalibratedModel(m._pipe, m._platt)
        assert np.array_equal(m.predict_proba(X), old.predict_proba(X))
        assert m.predict_proba_logit_shadow(X) is not None


def test_first_sight_only(tmp_path, monkeypatch):
    from src import calibration_shadow as cs
    monkeypatch.setattr(cs, "SHADOW_DIR", tmp_path)
    row = pd.DataFrame([{"date": "2026-10-20", "home_team": "A", "away_team": "B", "market": "ou25",
                         "p_prod": 0.51, "p_logit_shadow": 0.49}])
    assert cs._append_first_sight("team", row, cs.TEAM_KEY) == 1
    later = row.assign(p_prod=0.60)                                   # same fixture, later run
    assert cs._append_first_sight("team", later, cs.TEAM_KEY) == 0
    out = pd.read_csv(next(tmp_path.glob("team_*.csv")))
    assert len(out) == 1 and out["p_prod"].iloc[0] == 0.51
