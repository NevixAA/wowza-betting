"""Calibration space research option (2026-11-01): "prob" (legacy) vs "logit".

    python -m pytest tests/test_logit_calibration.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import model as M  # noqa: E402

FEATS = ["home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _multi_league(n_per=2500, seed=0):
    """Six leagues whose Over rate differs by league (logit level -0.9..+0.9). The features carry
    the level (team scoring rates scale with it) plus noise - the shape of the real data."""
    rng = np.random.default_rng(seed)
    rows, start = [], pd.Timestamp("2021-07-01")
    for i, lvl in enumerate(np.linspace(-0.9, 0.9, 6)):
        for k in range(n_per):
            team = rng.normal(0, 0.5)
            eta = lvl + team
            row = {"date": start + pd.Timedelta(days=int(k * 0.6)), "league": f"L{i}",
                   "over25": float(rng.random() < _sigmoid(eta))}
            for c in FEATS:
                row[c] = 1.35 + 0.45 * eta + rng.normal(0, 0.35)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _spread(df, p):
    g = pd.DataFrame({"league": df["league"].to_numpy(), "y": df["over25"].to_numpy(), "p": p})
    m = g.groupby("league")[["y", "p"]].mean()
    return m["y"].std(), m["p"].std()


@pytest.fixture(scope="module")
def trained():
    df = _multi_league()
    out = {}
    for space in M.CALIBRATION_SPACES:
        res = M.train_repaired(df, target="over25", feature_cols=FEATS, calibration_space=space)
        out[space] = M.build_payload(res, "over25")
    hold = M.four_block_split(df, "over25")["holdout"]
    return df, hold, out


def test_legacy_payload_without_flag_is_scored_exactly_as_before():
    """A committed serving payload has no calibration_space and no _space attributes, and
    predict_proba reproduces the pre-change arithmetic (Platt and meta on probabilities)."""
    p = pickle.load(open(config.MODEL_FILE_NEWFORMAT, "rb"))
    assert "calibration_space" not in p
    assert all(not hasattr(m, "_space") for k, m in p["models"].items() if k != "__meta__")
    bt = pd.read_csv(config.OUTPUT_DIR / "backtest_results_newformat.csv", low_memory=False).tail(400)
    got = M.predict_proba(bt, payload=p).to_numpy()
    X, _ = M._prep(bt, p["feature_cols"], imputer=p.get("imputer"))
    probas = [m._calibrator.predict_proba(m._pipe.predict_proba(X)[:, 1].reshape(-1, 1))[:, 1].clip(0.001, 0.999)
              for k, m in p["models"].items() if k != "__meta__"]
    want = p["models"]["__meta__"].predict_proba(np.column_stack(probas))[:, 1]
    np.testing.assert_array_equal(got, want)


def test_default_training_returns_the_legacy_payload_shape():
    df = _multi_league(n_per=300)
    res = M.train(df, target="over25", feature_cols=FEATS)
    assert type(res) is dict, "legacy prob training must stay a plain dict"
    pay = M.build_payload(res, "over25")
    assert "calibration_space" not in pay


def test_logit_platt_is_identity_on_calibrated_input():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 40000)
    y = (rng.random(len(p)) < p).astype(int)
    lg = LogisticRegression(C=1.0, max_iter=1000).fit(M._to_space(p.reshape(-1, 1), "logit"), y)
    assert lg.coef_[0][0] == pytest.approx(1.0, abs=0.06) and lg.intercept_[0] == pytest.approx(0.0, abs=0.04)


def test_logit_space_is_stored_and_honoured(trained):
    _, hold, pays = trained
    assert pays["logit"]["calibration_space"] == "logit" and "calibration_space" not in pays["prob"]
    assert all(getattr(m, "_space", "prob") == "logit" for k, m in pays["logit"]["models"].items() if k != "__meta__")
    a = M.predict_proba(hold, payload=pays["logit"]).to_numpy()
    flat = dict(pays["logit"]); flat.pop("calibration_space")
    b = M.predict_proba(hold, payload=flat).to_numpy()
    assert not np.allclose(a, b), "the meta blend must read the stored space"


def test_logit_keeps_the_league_spread_prob_squashes(trained):
    _, hold, pays = trained
    actual, sp_prob = _spread(hold, M.predict_proba(hold, payload=pays["prob"]).to_numpy())
    _, sp_logit = _spread(hold, M.predict_proba(hold, payload=pays["logit"]).to_numpy())
    assert sp_logit > sp_prob, (sp_prob, sp_logit)
    assert sp_logit >= 0.75 * actual, (actual, sp_logit)
