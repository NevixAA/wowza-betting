"""R2-B repaired challengers (findings 1, 2, 3, 4, 5, 13, 16, 19) — required automated tests.

Each defect is pinned twice where it makes sense: the R2 repaired recipe must NOT have it, and a
"documents legacy" test shows the legacy recipe still does (fail-before / pass-after, inside one
suite). Synthetic data only, except the serving-payload parity test, which loads the committed
serving payloads read-only. Nothing touches the network or writes a serving file.
"""
from __future__ import annotations

import json
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import model as M
from src import recency as R
from src.features_repaired import build_features_repaired, build_upcoming_features_repaired
from src.imputer import IMPUTER_VERSION, IMPUTER_VERSION_R2, apply_imputer, fit_imputer

from test_standard_forensic_repair import FEATS, _frame, _season

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "r2b_training_golden_main_4e9c933.json"
NF = "Brazil Serie A"


def _nf_frame():
    """Standard frame + a New Format league, so pooled / track-specific paths are exercised."""
    std = _frame()
    teams = [f"N{i}" for i in range(10)]
    nf = pd.concat([_season(NF, teams, s, 40 + k) for k, s in enumerate(("2023-02-01", "2024-02-01"))])
    return pd.concat([std, nf], ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def feat():
    return build_features_repaired(_nf_frame())


@pytest.fixture(scope="module")
def rows(feat):
    r = feat.dropna(subset=["over25", "home_scored_last5"]).copy()
    r["home_xg_last5"] = np.nan              # a feature with NO training values (Standard xG)
    return r


@pytest.fixture(scope="module")
def hs(rows):
    d = pd.to_datetime(rows["date"]).sort_values().reset_index(drop=True)
    return pd.Timestamp(d.iloc[int(len(d) * 0.8)])


R2_COLS = FEATS + ["home_xg_last5"]


@pytest.fixture(scope="module")
def r2(rows, hs):
    w = pd.Series(np.linspace(1.0, 2.0, len(rows)), index=rows.index)
    return M.train(rows, "over25", sample_weight=w, feature_cols=R2_COLS, recipe="side_v9_repaired",
                   holdout_start=hs, provenance={"feature_recipe_version": "t", "identity_version": "t"})


def _pay(res):
    return M.build_payload(res, "over25")


# ── finding 2: batch invariance ─────────────────────────────────────────────────────────────

def test_batch_invariance_single_reorder_and_unrelated_rows(rows, r2):
    pay = _pay(r2)
    ev = rows.tail(80).copy()
    ev.iloc[0, ev.columns.get_loc("home_corners_pg_roll")] = np.nan       # needs imputing
    ev.iloc[0, ev.columns.get_loc("away_scored_last5")] = np.nan
    p_one = M.predict_proba(ev.iloc[[0]], payload=pay).iloc[0]
    p_batch = M.predict_proba(ev, payload=pay)
    p_rev = M.predict_proba(ev.iloc[::-1], payload=pay)
    junk = ev.head(30).copy()
    for c in FEATS:
        junk[c] = np.where(np.arange(len(junk)) % 2, np.nan, 99.0)        # unrelated, extreme rows
    junk.index = junk.index + 10_000_000
    p_mixed = M.predict_proba(pd.concat([junk, ev]), payload=pay)
    assert p_one == pytest.approx(p_batch.iloc[0], abs=1e-12)
    assert np.allclose(p_batch.to_numpy(), p_rev.loc[p_batch.index].to_numpy(), atol=1e-12)
    assert np.allclose(p_batch.to_numpy(), p_mixed.loc[p_batch.index].to_numpy(), atol=1e-12)


def test_documents_legacy_batch_dependence(rows):
    leg = M.train(rows, "over25", feature_cols=FEATS)
    pay = _pay(leg)
    ev = rows.tail(80).copy()
    ev.iloc[0, ev.columns.get_loc("home_scored_last5")] = np.nan
    assert abs(M.predict_proba(ev.iloc[[0]], payload=pay).iloc[0]
               - M.predict_proba(ev, payload=pay).iloc[0]) > 1e-6


def test_upcoming_features_do_not_depend_on_the_board(feat):
    hist = _nf_frame()
    last = pd.Timestamp(hist["date"].max()) + pd.Timedelta(days=7)
    board = pd.DataFrame({"date": [last] * 3, "league": [NF, NF, "Championship"],
                          "home_team": ["N1", "N3", "A1"], "away_team": ["N2", "N4", "A2"]})
    one = build_upcoming_features_repaired(board.iloc[[0]], hist)
    all3 = build_upcoming_features_repaired(board, hist)
    cols = [c for c in FEATS if c in one.columns]
    pd.testing.assert_frame_equal(one[cols].reset_index(drop=True), all3[cols].iloc[[0]].reset_index(drop=True))


def test_repaired_payload_without_imputer_is_refused(r2):
    pay = dict(_pay(r2))
    pay.pop("imputer")
    with pytest.raises(ValueError, match="no persisted imputer"):
        M.predict_proba(pd.DataFrame({c: [1.0] for c in pay["feature_cols"]}), payload=pay)


# ── finding 2: no fully-missing column silently becomes 0 ───────────────────────────────────

def test_fully_missing_feature_is_dropped_and_recorded(r2):
    pay = _pay(r2)
    assert pay["imputer_version"] == IMPUTER_VERSION_R2
    assert "home_xg_last5" in pay["dropped_all_missing_features"]
    assert "home_xg_last5" not in pay["feature_cols"]


def test_r2_imputer_refuses_to_invent_a_value_for_a_no_training_feature():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan] * 3})
    with pytest.raises(ValueError):
        fit_imputer(tr, ["a", "b"], all_missing="error")
    m = fit_imputer(tr, ["a", "b"], all_missing="drop")
    assert m.all_missing_features == ("b",)
    with pytest.raises(ValueError, match="refusing to invent"):
        apply_imputer(pd.DataFrame({"a": [np.nan], "b": [np.nan]}), m, ["a", "b"])
    z = fit_imputer(tr, ["a", "b"], all_missing="drop", semantic_zero=["b"])   # documented zero
    X, _, _ = apply_imputer(pd.DataFrame({"a": [np.nan], "b": [np.nan]}), z, ["a", "b"])
    assert X.loc[0, "b"] == 0.0 and X.loc[0, "a"] == 2.0


def test_v1_manifest_unchanged_documents_legacy_zero_fill():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan] * 3})
    m = fit_imputer(tr, ["a", "b"])
    assert m.version == IMPUTER_VERSION
    d = m.to_dict()
    assert set(d) == {"version", "fitted_on_rows", "fitted_on_dates", "fill_all_missing",
                      "clip_to_training_range", "features"}
    X, _, _ = apply_imputer(pd.DataFrame({"a": [np.nan], "b": [np.nan]}), m, ["a", "b"])
    assert X.loc[0, "b"] == 0.0                      # the 1.0 behaviour the serving payload has


def test_league_prior_and_missing_indicator_options():
    tr = pd.DataFrame({"league": ["X", "X", "Y", "Y"], "a": [1.0, 1.0, 5.0, np.nan]})
    m = fit_imputer(tr, ["a", "a__missing"], all_missing="drop", group_col="league")
    X, _, _ = apply_imputer(pd.DataFrame({"league": ["X", "Y", "Z"], "a": [np.nan] * 3}), m,
                            ["a", "a__missing"])
    assert list(X["a"]) == [1.0, 5.0, 1.0]           # own league's TRAIN median, else global
    assert list(X["a__missing"]) == [1.0, 1.0, 1.0]


# ── finding 3: four chronological blocks, holdout labels never read ─────────────────────────

def test_blocks_are_chronological_disjoint_and_persisted(r2, hs):
    b = r2.meta["blocks"]
    order = ["base", "calibration", "meta", "holdout"]
    for x, y in zip(order, order[1:]):
        assert b[x]["dates"][1] < b[y]["dates"][0]
    assert b["holdout"]["dates"][0] >= str(hs.date())
    assert r2.meta["block_definition"]["holdout_start"] == str(hs.date())
    assert r2.meta["holdout_independence"]["holdout_labels_used_for_fitting"] is False


def _coefs(res):
    out = {}
    for k, v in res.items():
        m = v["model"]
        if k == "__meta__":
            out[k] = np.r_[m.coef_.ravel(), m.intercept_]
        else:
            out[k] = np.r_[m._calibrator.coef_.ravel(), m._calibrator.intercept_]
    return out


def test_calibrator_and_meta_never_read_holdout_labels(rows, hs):
    a = M.train(rows, "over25", feature_cols=FEATS, recipe="nf_v9_repaired", holdout_start=hs)
    flipped = rows.copy()
    m = pd.to_datetime(flipped["date"]) >= hs
    flipped.loc[m, "over25"] = 1.0 - flipped.loc[m, "over25"]
    b = M.train(flipped, "over25", feature_cols=FEATS, recipe="nf_v9_repaired", holdout_start=hs)
    ca, cb = _coefs(a), _coefs(b)
    for k in ca:
        assert np.allclose(ca[k], cb[k], atol=0), k
    assert a.meta["imputer"] == b.meta["imputer"]


def test_documents_legacy_meta_reads_its_holdout_labels(rows):
    a = M.train(rows, "over25", feature_cols=FEATS)
    d = rows.sort_values("date", kind="mergesort").copy()
    newest = d.index[int(len(d) * 0.85):]                 # inside legacy's 20% test block
    d.loc[newest, "over25"] = 1.0 - d.loc[newest, "over25"]
    b = M.train(d, "over25", feature_cols=FEATS)
    assert not np.allclose(_coefs(a)["__meta__"], _coefs(b)["__meta__"])


# ── finding 4: weights travel with their rows ───────────────────────────────────────────────

def test_weights_stay_with_fixture_after_shuffle_and_sort(rows, hs, monkeypatch):
    sh = rows.sample(frac=1.0, random_state=11).assign(rid=np.arange(len(rows), dtype=float))
    w = pd.Series(sh["rid"].to_numpy() + 1.0, index=sh.index)            # unique per row
    blocks = M.four_block_split(sh, "over25", w, holdout_start=hs)
    for b in blocks.values():
        assert (b[M.WEIGHT_COL] == b["rid"] + 1.0).all()
    seen = []
    from sklearn.pipeline import Pipeline
    orig = Pipeline.fit

    def spy(self, X, y=None, **kw):
        if "clf__sample_weight" in kw and not seen:
            seen.append((X["rid"].to_numpy().copy(), np.asarray(kw["clf__sample_weight"]).copy()))
        return orig(self, X, y, **kw)
    monkeypatch.setattr(Pipeline, "fit", spy)
    M.train(sh, "over25", sample_weight=w.to_numpy(), feature_cols=["rid"] + FEATS[:4],
            recipe="side_v9_repaired", holdout_start=hs)
    rid, ww = seen[0]
    assert np.allclose(ww, rid + 1.0)


def test_weight_length_mismatch_is_an_error(rows):
    with pytest.raises(ValueError):
        M._rows_with_weights(rows, "over25", np.ones(len(rows) - 1))


# ── finding 5: derived recency, frozen candidates, persisted ────────────────────────────────

def test_recency_candidates_are_frozen_and_persisted(r2, rows, hs):
    assert set(R.CANDIDATES) == {"season_age", "exp_half_life", "bounded_linear", "uniform"}
    assert R.PRIMARY == "season_age"
    pre = rows[pd.to_datetime(rows["date"]) < hs]
    as_of = pd.to_datetime(pre["date"]).max()
    for fn in R.CANDIDATES:
        w = R.weights(pre, fn, as_of)
        assert (w >= 0).all() and len(w) == len(pre)
        s = R.summary(pre, w, fn, as_of)
        assert s["function"] == fn and s["params"] == R.CANDIDATES[fn] and s["ess"] > 0
        assert abs(sum(v["weight_share"] for v in s["per_season"].values()) - 1.0) < 1e-3
    with pytest.raises(ValueError):
        R.weights(rows, "season_age", as_of)          # rows after as_of (= holdout) refused


def test_season_age_matches_the_standard_derived_weights(rows):
    from src.standard_repair import derived_weights
    as_of = pd.to_datetime(rows["date"]).max()
    assert np.allclose(R.weights(rows, "season_age", as_of), derived_weights(rows, as_of=as_of))


# ── finding 1: full payload metadata ────────────────────────────────────────────────────────

def test_r2_payload_metadata_is_complete(r2):
    pay = _pay(r2)
    for k in ("recipe", "model_recipe_version", "feature_recipe_version", "identity_version",
              "imputer_version", "training_cutoff", "training_start", "training_end",
              "training_rows", "frame_hash", "feature_cols", "blocks", "block_definition",
              "code_sha", "served_holdout_metrics", "metrics", "provenance", "weights", "imputer"):
        assert k in pay and pay[k] not in (None, ""), k
    assert pay["recipe"] == "side_v9_repaired"
    assert pay["model_recipe_version"] == M.R2_RECIPE_VERSION
    assert "ece" in pay["served_holdout_metrics"]


def test_unknown_recipe_and_legacy_options_refused(rows):
    with pytest.raises(ValueError):
        M.train(rows, "over25", feature_cols=FEATS, recipe="nf_v9_repairedX")
    with pytest.raises(TypeError):
        M.train(rows, "over25", feature_cols=FEATS, holdout_start="2024-01-01")


# ── legacy + Standard repaired training unchanged (golden from main 4e9c933) ────────────────

def test_legacy_and_v9_repaired_training_match_main_golden():
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    raw = _frame()
    f = build_features_repaired(raw)
    rr = f.dropna(subset=["over25", "home_scored_last5"])
    rep = M.train_repaired(rr, "over25", feature_cols=FEATS,
                           sample_weight=pd.Series(np.linspace(1, 3, len(rr)), index=rr.index))
    leg = M.train(rr, "over25", feature_cols=FEATS, sample_weight=np.linspace(1, 3, len(rr)))
    ev = rr.tail(60).copy()
    ev.iloc[3, ev.columns.get_loc("home_corners_pg_roll")] = np.nan
    # The golden was recorded on Windows from main 4e9c933. Probabilities are compared EXACTLY there
    # (same machine: proves training is unchanged). Elsewhere GradientBoosting/LightGBM float paths differ
    # (Linux CI: up to ~5e-4 repaired, >1e-3 legacy, whose batch-median imputation amplifies noise), so a
    # cross-machine probability compare proves nothing; the block and metadata checks below still run
    # everywhere, and real-data main-vs-branch serving equality is the `parity` job.
    if sys.platform == "win32":
        assert np.allclose(M.predict_proba(ev, payload=_pay(rep)), g["v9_repaired"], atol=1e-6)
        assert np.allclose(M.predict_proba(ev, payload=_pay(leg)), g["legacy_v9"], atol=1e-6)
    assert rep.meta["blocks"] == g["v9_repaired_blocks"]
    assert set(rep.meta) == {"recipe", "model_recipe_version", "imputer", "imputer_version",
                             "blocks", "served_holdout_metrics", "target"}


# ── every committed SERVING payload scores exactly as main's code did ───────────────────────

def _reference_predict(df, payload):
    """Verbatim main (4e9c933) _prep + predict_proba, the code every serving payload ran."""
    from src.imputer import apply_imputer as _ai
    feat_cols = payload["feature_cols"]
    imp = payload.get("imputer")
    if imp is not None:
        X, _, _ = _ai(df, imp, list(feat_cols))
    else:
        cols = [c for c in feat_cols if c in df.columns]
        X = df[cols].copy().apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median().fillna(0.0))
    base = {k: v for k, v in payload["models"].items() if k != "__meta__"}
    meta = payload["models"].get("__meta__")
    pr = [m.predict_proba(X)[:, 1] for m in base.values()]
    if meta is not None and len(pr) >= 2:
        try:
            return meta.predict_proba(np.column_stack(pr))[:, 1]
        except Exception:                                              # noqa: BLE001
            return np.mean(pr, axis=0)
    return np.mean(pr, axis=0)


SERVING = ["MODEL_FILE_STANDARD", "MODEL_FILE_STANDARD_REPAIRED", "MODEL_FILE_NEWFORMAT",
           "MODEL_FILE_BTTS", "MODEL_FILE_OVER15", "MODEL_FILE_OVER35", "HT_MODEL_FILE_05",
           "HT_MODEL_FILE_15"]


@pytest.mark.parametrize("name", SERVING)
def test_serving_payload_scores_identically_to_main(name):
    path = Path(getattr(config, name))
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout")
    with open(path, "rb") as f:
        pay = pickle.load(f)
    rng = np.random.default_rng(7)
    cols = list(pay["feature_cols"])
    df = pd.DataFrame(rng.normal(1.0, 0.6, (64, len(cols))), columns=cols)
    df = df.mask(rng.random(df.shape) < 0.15)                            # NaNs to impute
    new = M.predict_proba(df, payload=pay).to_numpy()
    ref = _reference_predict(df, pay)
    assert np.array_equal(new, ref), f"{path.name}: serving probability changed"


# ── finding 16: cold-start option leaves the default untouched ──────────────────────────────

def test_cold_start_nan_only_changes_the_cold_start_columns():
    raw = _nf_frame()
    a = build_features_repaired(raw)
    b = build_features_repaired(raw, cold_start="nan")
    cold = {"home_advantage", "away_home_adv_factor", "combined_sot_ratio", "combined_ht_goals_avg"}
    same = [c for c in a.columns if c not in cold]
    pd.testing.assert_frame_equal(a[same], b[same])
    for c in cold:
        assert b[c].isna().any()                                       # NaN where no history
    for c in ("combined_sot_ratio", "combined_ht_goals_avg"):
        known = b[c].notna()
        assert np.allclose(a.loc[known, c], b.loc[known, c])          # identical where history
    # home flags: "nan" never uses the typed 1.10 league-away prior, so only windows that reach
    # back to a league's very first matchday can differ; later rows are identical
    late = pd.to_datetime(a["date"]) >= pd.to_datetime(a["date"]).min() + pd.Timedelta(days=400)
    for c in ("home_advantage", "away_home_adv_factor"):
        k = late & b[c].notna()
        assert k.sum() > 50 and np.allclose(a.loc[k, c], b.loc[k, c])
    with pytest.raises(ValueError):
        build_features_repaired(raw, cold_start="zero")


# ── finding 19: track purity of side challengers ────────────────────────────────────────────

def test_track_specific_side_rows_never_mix():
    from src.standard_repair import track_rows
    f = build_features_repaired(_nf_frame())
    s, n = track_rows(f, "standard", "btts"), track_rows(f, "newformat", "btts")
    assert set(s["league"]) <= set(config.STANDARD_FORMAT_LEAGUES)
    assert set(n["league"]) == {NF}


# ── identity: NF duplicate identities are resolved on evidence ──────────────────────────────

def test_nf_identity_overlap_audit_china_duplicates():
    from src.identity_repair import canonicalise_and_dedupe, load_alias_map, overlap_audit
    lg = "China Super League"
    teams = ["Shandong Taishan", "Beijing Guoan", "Shanghai Port", "Wuhan Three Towns"]
    base = _season(lg, teams, "2024-03-01", 5)
    dup = base.copy()
    dup.loc[dup["home_team"] == "Shandong Taishan", "home_team"] = "Shandong Luneng"
    dup.loc[dup["away_team"] == "Shandong Taishan", "away_team"] = "Shandong Luneng"
    dup.loc[dup["home_team"] == "Shanghai Port", "home_team"] = "SHANGHAI SIPG"
    dup.loc[dup["away_team"] == "Shanghai Port", "away_team"] = "SHANGHAI SIPG"
    both = pd.concat([base, dup], ignore_index=True)
    amap = load_alias_map(ROOT / "config" / "standard_ou25_identity_aliases.json")
    assert amap[lg]["Shandong Luneng"] == "Shandong Taishan"
    assert amap[lg]["Henan Jianye"] == "Henan Songshan Longmen"
    aud = overlap_audit(both, amap)
    row = aud[(aud["a"].isin(["Shandong Luneng", "Shandong Taishan"]))
              & (aud["b"].isin(["Shandong Luneng", "Shandong Taishan"]))]
    assert not row.empty and (row["decision"] == "ALIASED_EXPLICIT").all()
    assert (row["overlap_days"] >= 0).all()
    fixed, rep = canonicalise_and_dedupe(both, amap, derive=False)
    assert "Shandong Luneng" not in set(fixed["home_team"]) | set(fixed["away_team"])
    assert len(fixed) == len(base)
