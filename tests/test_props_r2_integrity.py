"""R2-D props preprocessing integrity (findings 20, 21).

Legacy serving payloads must score EXACTLY as before R2; repaired challenger payloads must be
batch-invariant, use only fit-block statistics, and never turn a missing column into 0 unless
the feature contract says SEMANTIC_ZERO."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import config
from player_model import model as M
from player_model import props_contract as PC

ROOT = Path(__file__).resolve().parents[1]
FEATS = list(config.PLAYER_FEATURE_COLS)


# ---------------------------------------------------------------------------- helpers
def _legacy_reference(df: pd.DataFrame, payload: dict) -> np.ndarray:
    """Frozen copy of the pre-R2 scoring code (model._prep + predict_proba as of b0a1204)."""
    cols = payload["feature_cols"]
    df = df.copy()
    for c in [c for c in cols if c not in df.columns]:
        df[c] = 0.0
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median().fillna(0.0))
    return np.mean([m.predict_proba(X)[:, 1] for m in payload["models"].values()], axis=0)


def _synthetic(n=1600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.gamma(2.0, 0.5, size=(n, len(FEATS))), columns=FEATS)
    for c in ("is_home", "pos_forward", "pos_midfielder", "pos_defender", "return_from_injury_flag"):
        df[c] = rng.integers(0, 2, n).astype(float)
    df["n_prev_games"] = rng.integers(1, 30, n).astype(float)
    # Missingness in IMPUTABLE / DEGRADED columns so imputation actually matters.
    for c in ("days_since_last_injury", "referee_yellows_pg", "season_goals_pg", "kp_per90"):
        df.loc[rng.random(n) < 0.25, c] = np.nan
    logit = -1.2 + 1.5 * df["goals_pg"] - 0.4 * df["is_home"] + 0.3 * df["kp_per90"].fillna(0)
    df["target_goals"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["date"] = pd.date_range("2024-01-01", periods=n, freq="6h")
    return df


@pytest.fixture(scope="module")
def repaired():
    df = _synthetic()
    res = M.train(df, "goals", recipe=PC.RECIPE_REPAIRED)
    return df, M.build_payload(res, "goals")


# ---------------------------------------------------------------------------- contract
def test_contract_file_covers_exactly_the_feature_list_and_is_hashed():
    c = PC.load_contract()
    assert c["contract_version"] == PC.CONTRACT_VERSION
    assert set(c["features"]) == set(FEATS)
    assert {v["class"] for v in c["features"].values()} <= set(PC.CLASSES)
    assert c["contract_sha256"] == PC.build_contract(FEATS)["contract_sha256"], \
        "contract drifted from the generator: regenerate and review"


def test_unknown_feature_is_refused_by_the_imputer():
    df = _synthetic(50)
    df["brand_new_feature"] = 1.0
    with pytest.raises(ValueError):
        PC.PropsImputer().fit(df, FEATS + ["brand_new_feature"])


# ---------------------------------------------------------------------------- legacy parity
def _real_rows(n=400, absent=("rating_pg",)) -> pd.DataFrame:
    parts = sorted((ROOT / "player_history.parquet").glob("month_*.parquet"))
    if not parts:
        pytest.skip("player_history.parquet not checked out")
    df = pd.read_parquet(parts[-2]).head(n).copy()
    rng = np.random.default_rng(1)
    for c in FEATS[:40]:                              # inject NaNs -> batch medians fire
        df.loc[rng.random(len(df)) < 0.1, c] = np.nan
    return df.drop(columns=list(absent))             # an ABSENT column (legacy used to fill 0.0)


def _hgb_only(df: pd.DataFrame, payload: dict) -> np.ndarray:
    cols = payload["feature_cols"]
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return np.mean([m.predict_proba(X)[:, 1] for m in payload["models"].values() if M._nan_native(m)],
                   axis=0)


@pytest.mark.parametrize("market", config.MARKETS)
def test_serving_legacy_payload_scores_exactly_as_before_r2(market):
    """Data Truth (dt/props, 2026-11-01): wherever every input is measured (or batch-median filled,
    the legacy recipe), the legacy payload scores EXACTLY as before R2. An ABSENT feature column is
    no longer a typed 0.0: those rows are scored by the NaN-native estimator alone."""
    path = config.MODEL_FILES[market]
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    raw = path.read_bytes()
    payload = pickle.loads(raw)
    if M.payload_recipe(payload) == PC.RECIPE_CLEAN_HARNESS:
        # owner 2026-11-08: sot/sot2/goals/assists/cards serve the props clean-retrain models
        # (tests/test_props_owner_rules_2026_11_08.py); this test covers the markets still on legacy
        assert market in {"sot", "sot2", "goals", "assists", "cards"}, market
        pytest.skip(f"{market} serves a promoted clean-retrain model")
    assert "recipe" not in payload and M.payload_recipe(payload) == PC.RECIPE_LEGACY
    full = _real_rows(absent=())
    got = M.predict_proba(full, payload).to_numpy()
    np.testing.assert_array_equal(got, _legacy_reference(full, payload))
    np.testing.assert_array_equal(M.predict_with_diagnostics(full, payload)["p"].to_numpy(), got)
    df = _real_rows()
    got = M.predict_proba(df, payload).to_numpy()
    np.testing.assert_allclose(got, _hgb_only(df, payload), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(M.predict_with_diagnostics(df, payload)["p"].to_numpy(), got)


def test_serving_files_are_the_registered_champions():
    reg = json.loads((ROOT / "models/registry/props/props_champions.json").read_text(encoding="utf-8"))
    for market, rec in reg.items():
        p = config.MODEL_FILES.get(market)
        if p is None or not p.exists():
            continue
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["artifact_sha256"], market


def test_legacy_payload_bytes_unchanged_shape():
    """build_payload for a legacy result adds no R2 keys (props_gate hashes these bytes)."""
    df = _synthetic(600)
    res = M.train(df, "goals")
    assert set(M.build_payload(res, "goals")) == {"market", "models", "feature_cols", "metrics"}


def test_legacy_is_batch_dependent_characterisation():
    """Documents the defect R2 repairs: the legacy path gives a NaN row a different value
    depending on the rest of the batch. If this ever starts failing, legacy behaviour changed."""
    df = _synthetic(600)
    payload = M.build_payload(M.train(df, "goals"), "goals")
    row = df.iloc[[5]].copy()
    row["kp_per90"] = np.nan
    other_a = df.iloc[100:130].copy()
    other_b = df.iloc[100:130].copy()
    other_b["kp_per90"] = 9.0
    pa = M.predict_proba(pd.concat([row, other_a]), payload).iloc[0]
    pb = M.predict_proba(pd.concat([row, other_b]), payload).iloc[0]
    assert pa != pb


# ---------------------------------------------------------------------------- repaired path
def _same(a, b):
    # Identical inputs to the model; the only allowed difference is BLAS summation order in a
    # matrix product (1 ulp, ~1e-16). Anything that depends on the batch moves p by >>1e-12.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_repaired_payload_carries_versions_and_blocks(repaired):
    _, payload = repaired
    assert payload["recipe"] == PC.RECIPE_REPAIRED
    assert payload["imputer_version"] == PC.IMPUTER_VERSION
    assert payload["contract_version"] == PC.CONTRACT_VERSION
    b = payload["blocks"]
    assert b["fit"][1] == b["calibration"][0] and b["calibration"][1] == b["holdout"][0]
    assert b["fit_dates"][1] < b["calibration_dates"][0] < b["holdout_dates"][0]


def test_imputer_uses_fit_block_statistics_only(repaired):
    df, payload = repaired
    _, cal = M.split_bounds(len(df))
    fill = payload["imputer"]["fill_values"]
    for c in ("kp_per90", "days_since_last_injury", "season_goals_pg"):
        assert fill[c] == pytest.approx(float(df[c].iloc[:cal].median()))
    # perturbing calibration + holdout rows must not move a single fill value
    df2 = df.copy()
    df2.loc[cal:, FEATS] = df2.loc[cal:, FEATS] * 50 + 7
    p2 = M.build_payload(M.train(df2, "goals", recipe=PC.RECIPE_REPAIRED), "goals")
    assert p2["imputer"]["fill_values"] == fill


def test_single_row_equals_same_row_in_batch_reordered_and_with_unrelated_rows(repaired):
    df, payload = repaired
    batch = df.iloc[1200:1260].copy()
    batch.loc[batch.index[:20], "kp_per90"] = np.nan          # rows that need imputation
    full = M.predict_proba(batch, payload)
    singles = pd.concat([M.predict_proba(batch.loc[[i]], payload) for i in batch.index])
    _same(full.to_numpy(), singles.to_numpy())
    rev = M.predict_proba(batch.iloc[::-1], payload)
    _same(rev.loc[batch.index].to_numpy(), full.to_numpy())
    junk = df.iloc[:40].copy()
    junk[FEATS[:60]] = 999.0
    junk.index = junk.index + 10_000
    mixed = M.predict_proba(pd.concat([junk, batch]), payload)
    _same(mixed.loc[batch.index].to_numpy(), full.to_numpy())


def test_absent_column_is_not_silently_zero(repaired):
    df, payload = repaired
    rows = df.iloc[1300:1310].drop(columns=["kp_per90", "red_cards_pg", "season_goals_pg"])
    imp = PC.PropsImputer.from_state(payload["imputer"])
    X, diag = imp.transform(rows)
    fill = payload["imputer"]["fill_values"]
    assert (X["kp_per90"] == fill["kp_per90"]).all() and fill["kp_per90"] != 0.0
    assert (X["season_goals_pg"] == fill["season_goals_pg"]).all()
    assert (X["red_cards_pg"] == 0.0).all()                    # documented SEMANTIC_ZERO
    assert (diag["n_missing"] >= 3).all()
    assert (diag["n_degraded"] >= 1).all()                     # season_goals_pg is DEGRADED
    out = M.predict_with_diagnostics(rows, payload)
    assert {"n_missing", "n_degraded", "imputer_version", "contract_version"} <= set(out.columns)
    assert (out["imputer_version"] == PC.IMPUTER_VERSION).all()


def test_required_feature_missing_gives_no_probability(repaired):
    df, payload = repaired
    rows = df.iloc[1400:1405].copy()
    rows.loc[rows.index[0], "minutes_pg"] = np.nan
    rows = rows.drop(columns=[])
    out = M.predict_with_diagnostics(rows, payload)
    assert np.isnan(out["p"].iloc[0]) and bool(out["required_fail"].iloc[0])
    assert out["p"].iloc[1:].notna().all()
    no_home = df.iloc[1400:1402].drop(columns=["is_home"])
    assert M.predict_proba(no_home, payload).isna().all()


def test_repaired_payload_roundtrips_through_pickle(repaired):
    df, payload = repaired
    p2 = pickle.loads(pickle.dumps(payload))
    rows = df.iloc[1500:1550]
    np.testing.assert_array_equal(M.predict_proba(rows, payload).to_numpy(),
                                  M.predict_proba(rows, p2).to_numpy())


# ---------------------------------------------------------------------------- fantasy defaults
def test_fantasy_defaults_register_is_classified_and_descriptive():
    from player_model import fantasy_defaults as FD
    classes = {"OBSERVED", "TRAIN_FITTED_PRIOR", "DOMAIN_PRIOR", "MISSING_DATA_FALLBACK"}
    assert all(d["class"] in classes for d in FD.DEFAULTS)
    ids = [d["id"] for d in FD.DEFAULTS]
    assert len(ids) == len(set(ids))
    for must in ("minutes_default_60", "p_start_default_0_6", "position_default_MID",
                 "goal_points_fallback_5", "referee_cards_3_5", "fdr_heuristic_xg"):
        assert must in ids
    s = FD.summary()
    assert s["behaviour_changed"] is False and s["register_version"] == FD.REGISTER_VERSION
