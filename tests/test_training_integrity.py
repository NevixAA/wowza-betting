"""Estate scan 2026-11-02, training findings (a)-(d). Synthetic data only; nothing touches the
network or writes a serving model file.

(a) sample weights travel with their rows in the legacy trainer (was: positional after sort);
(b) side models trained per league track and served per track (was: Standard + NF pooled);
(c) the scheduled retrain's meta-logistic is fitted on its own block, never on the gate holdout;
(d) the scheduled retrain produces payloads with a persisted train-fitted imputer, so a
    fixture's probability does not depend on the batch it is scored with; legacy payloads still
    serve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import model as M

STD = "Championship"
NF = "Brazil Serie A"
FEATS = ["f1", "f2", "f3", "f4"]


def _league_rows(league, n, seed, sign=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n) // 3, unit="D")
    X = rng.normal(size=(n, 4))
    X[rng.uniform(size=(n, 4)) < 0.1] = np.nan
    z = sign * (1.2 * np.nan_to_num(X[:, 0]) - 0.6 * np.nan_to_num(X[:, 1]))
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(float)
    df = pd.DataFrame(X, columns=FEATS)
    df.insert(0, "date", dates)
    df.insert(1, "league", league)
    df.insert(2, "home_team", [f"{league[:2]}H{i}" for i in range(n)])
    df.insert(3, "away_team", [f"{league[:2]}A{i}" for i in range(n)])
    df["over25"] = y
    df["btts"] = y
    return df


@pytest.fixture(scope="module")
def mixed():
    return pd.concat([_league_rows(STD, 600, 1), _league_rows(NF, 600, 2, sign=-1.0)],
                     ignore_index=True)


# ── (a) ─────────────────────────────────────────────────────────────────────────────────────

def test_legacy_sample_weights_travel_with_their_rows(monkeypatch):
    """Each row's weight is ALSO stored as feature `f_w`; every fit must see X.f_w == weight.
    Rows arrive in REVERSE date order, so a positional read after the sort misaligns them."""
    df = _league_rows(STD, 300, 3)
    rng = np.random.default_rng(9)
    df["f_w"] = rng.uniform(0.5, 4.0, len(df))
    df = df.iloc[::-1].reset_index(drop=True)
    seen = []

    class SpyPipe(M.Pipeline):
        def fit(self, X, y=None, **kw):
            seen.append((X["f_w"].to_numpy().copy(), np.asarray(kw.get("clf__sample_weight"))))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(M, "Pipeline", SpyPipe)
    M.train(df, "over25", sample_weight=df["f_w"].to_numpy(), feature_cols=["f_w", "f1"])
    assert seen
    for fw, w in seen:
        np.testing.assert_allclose(fw, w)


# ── (c) + (d) through the real retrain entry point ─────────────────────────────────────────

def _run_train_one(monkeypatch, tmp_path, valid, label, target, weights=None, feature_cols=None):
    import pipeline as P
    from src import promotion_gate as PG
    got = {}

    def fake_submit(mt, model_file, old_payload, results, test_df, target_, basis, v, **k):
        got.update(results=results, test_df=test_df)
        return ({"verdict": PG.KEEP, "reason": "test", "qualified": False, "parked": False},
                {})

    monkeypatch.setattr(PG, "review_parked", lambda *a, **k: {})
    monkeypatch.setattr(PG, "submit_candidate", fake_submit)
    monkeypatch.setattr(P, "_record_retrain", lambda *a, **k: None)
    P._train_one(valid, label, tmp_path / "never_serving.pkl", target=target, weights=weights,
                 feature_cols=feature_cols)
    return got["results"], got["test_df"]


def test_retrain_meta_not_fitted_on_gate_holdout(monkeypatch, tmp_path, mixed):
    std = mixed[mixed["league"] == STD].reset_index(drop=True)
    res, test_df = _run_train_one(monkeypatch, tmp_path, std, "standard", "over25",
                                  weights=np.linspace(1, 3, len(std)), feature_cols=FEATS)
    meta = getattr(res, "meta", {})
    assert meta.get("holdout_independence", {}).get("meta_fit") == "meta"
    assert meta["holdout_independence"]["holdout_labels_used_for_fitting"] is False
    # every row the gate scores is in the candidate's untouched holdout
    gate_first = pd.to_datetime(test_df["date"]).min()
    assert pd.Timestamp(meta["blocks"]["holdout"]["dates"][0]) <= gate_first
    assert pd.Timestamp(meta["training_end"]) < gate_first


def test_retrain_payload_is_batch_invariant(monkeypatch, tmp_path, mixed):
    std = mixed[mixed["league"] == STD].reset_index(drop=True)
    res, _ = _run_train_one(monkeypatch, tmp_path, std, "standard", "over25", feature_cols=FEATS)
    pay = M.build_payload(res, "over25")
    assert pay.get("imputer"), "scheduled retrain must persist a train-fitted imputer"
    ev = std.tail(60).copy()
    ev.iloc[0, ev.columns.get_loc("f1")] = np.nan                    # needs imputing
    p_one = M.predict_proba(ev.iloc[[0]], payload=pay).iloc[0]
    other = ev.copy()
    other.iloc[1:, other.columns.get_loc("f1")] = 25.0               # batch median moves a lot
    p_batch = M.predict_proba(ev, payload=pay).iloc[0]
    p_other = M.predict_proba(other, payload=pay).iloc[0]
    assert p_one == pytest.approx(p_batch, abs=1e-12)
    assert p_one == pytest.approx(p_other, abs=1e-12)


def test_legacy_payload_still_serves(mixed):
    """Payloads trained before this change (no imputer, no tracks) keep serving every row."""
    legacy = M.build_payload(M.train(mixed, "btts", feature_cols=FEATS), "btts")
    assert "imputer" not in legacy and "tracks" not in legacy
    p = M.predict_proba(mixed.tail(50), payload=legacy)
    assert np.isfinite(p.to_numpy()).all()


# ── (b) ─────────────────────────────────────────────────────────────────────────────────────

def test_side_model_trained_and_served_per_track(monkeypatch, tmp_path, mixed):
    res, _ = _run_train_one(monkeypatch, tmp_path, mixed, "btts", "btts", feature_cols=FEATS)
    pay = M.build_payload(res, "btts")
    assert pay["recipe"] == "side_v9_repaired"
    assert pay["provenance"]["track"] == "standard"
    nf_pay = pay["tracks"]["new_format"]
    assert nf_pay is not None and nf_pay["provenance"]["track"] == "new_format"
    assert pay["track_split"]["rows"] == {"standard": 600, "new_format": 600}
    # the Standard model never saw a New Format row and vice versa
    assert pay["training_rows"] < 600 and nf_pay["training_rows"] < 600

    board = pd.concat([mixed[mixed["league"] == STD].tail(20), mixed[mixed["league"] == NF].tail(20)])
    unknown = board.head(1).copy()
    unknown["league"] = "Not A Registered League"
    board = pd.concat([board, unknown], ignore_index=True)
    p = M.predict_proba(board, payload=pay)
    is_std = (board["league"] == STD).to_numpy()
    is_nf = (board["league"] == NF).to_numpy()
    np.testing.assert_allclose(p[is_std], M.predict_proba(board[is_std], payload={
        k: v for k, v in pay.items() if k != "tracks"}).to_numpy())
    np.testing.assert_allclose(p[is_nf], M.predict_proba(board[is_nf], payload=nf_pay).to_numpy())
    assert np.isnan(p.iloc[-1]), "a league in no track gets no probability"


def test_side_track_without_enough_rows_gets_no_probability(mixed):
    thin = pd.concat([mixed[mixed["league"] == STD], mixed[mixed["league"] == NF].head(30)],
                     ignore_index=True)
    res = M.train_for_retrain(thin, "over15", target="btts", feature_cols=FEATS)
    pay = M.build_payload(res, "btts")
    assert pay["tracks"]["new_format"] is None
    assert pay["track_split"]["status"]["new_format"].startswith("not trained")
    p = M.predict_proba(thin[thin["league"] == NF], payload=pay)
    assert p.isna().all(), "no NF model -> no NF probability, never the Standard model's"


def test_every_retrain_label_has_a_repaired_recipe():
    import config
    labels = {"standard", "newformat", "ht_over05", "ht_over15", *config.SIDE_MARKETS}
    for lb in labels:
        assert M.retrain_recipe(lb) in M.REPAIRED_RECIPES
    with pytest.raises(ValueError):
        M.retrain_recipe("no_such_label")
