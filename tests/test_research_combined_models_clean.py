"""COMBINED-MODELS-DEV-2 research module (scripts/research_combined_models_clean.py): leak rules,
verification refusals and the statistics the report rests on. Synthetic data only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_combined_models as CM  # noqa: E402
import research_combined_models_clean as R  # noqa: E402


def test_protocol_is_registered_and_names_the_rules():
    p = json.loads(R.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert p["protocol"] == "COMBINED-MODELS-DEV-2"
    assert "2026-04-01..2026-10-31" in p["windows"]["scored"]
    assert "Holm" in p["family"]["correction"] and "WHOLE family" in p["family"]["correction"]
    assert "EXCLUDED" in p["matrix"]["helpers"]["for_team_target"]["props"]
    assert "PENDING" in p["inputs"]["side_models"]


def test_frame_hash_is_order_and_value_sensitive_but_not_index_sensitive():
    a = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": ["a", "b", "c"]})
    b = a.copy()
    b.index = [10, 11, 12]
    assert R.frame_hash(a) == R.frame_hash(b)
    c = a.copy()
    c.loc[1, "x"] = 0.0                       # a NaN read as 0 changes the hash
    assert R.frame_hash(a) != R.frame_hash(c)
    assert len(R.frame_hash(a)) == 16


def test_implied_from_total_matches_dev1_and_keeps_missing():
    r = R.implied_from_total([0.75, np.nan, 0.8], [0.5, 0.5, 0.6], [0.28, 0.3, np.nan])
    assert r["mu"][0] == pytest.approx(CM.poisson_total_from_over(0.75, 0.5, 0.28))
    assert np.isnan(r["mu"][1]) and np.isnan(r["ge6"][1]) and np.isnan(r["tot_le1"][1])
    assert np.isnan(r["mu"][2]) and np.isnan(r["nil_nil"][2])     # one input missing -> missing, never filled
    assert r["tot_le1"][0] == pytest.approx(0.25)
    assert r["nil_nil"][0] == pytest.approx(np.exp(-r["mu"][0]))


def test_pro_same_market_from_the_score_matrix():
    lam_h, lam_a = 1.4, 1.1
    from scipy.stats import poisson
    cs = {f"p_cs_{i}_{j}": poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) for i in range(7) for j in range(7)}
    tot = lambda lo, hi: sum(v for k, v in cs.items() if lo <= int(k.split("_")[2]) + int(k.split("_")[3]) <= hi)  # noqa: E731
    dc = pd.DataFrame([{**cs, "p_tot_0_1": tot(0, 1), "p_tot_4_5": tot(4, 5), "p_tot_6p": tot(6, 99)}])
    o = R.pro_same_market(dc).iloc[0]
    assert o["pro_over25"] == pytest.approx(1 - tot(0, 2))
    assert o["pro_over15"] == pytest.approx(1 - tot(0, 1))
    assert o["pro_over35"] == pytest.approx(tot(4, 99))
    assert o["pro_btts"] == pytest.approx((1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a)), abs=1e-3)


def test_tercile_groups_use_earlier_months_only():
    rng = np.random.default_rng(0)
    months = np.repeat(["2025-01", "2025-02", "2025-03"], 300)
    v = rng.normal(size=900)
    v[600:] += 100                                       # month 3 shifted: its own values must not set its cutoffs
    g = R.tercile_groups(pd.Series(v), pd.Series(["L"] * 900), pd.Series(months), min_prior=200)
    assert np.isnan(g[:300]).all()                       # no earlier month -> no group
    assert (g[600:] == 1).all()                          # cutoffs from months 1-2 only
    m2 = g[300:600]
    assert set(np.unique(m2)) <= {0.0, 1.0, -1.0}


def test_stack_is_fitted_on_earlier_months_only(monkeypatch):
    seen = []

    def spy(X, y):
        seen.append(len(X))
        return lambda Z: np.full(len(Z), 0.5)
    monkeypatch.setattr(R, "_fit_lr", spy)
    months = ["2026-02", "2026-03"] + R.eval_months()
    df = pd.DataFrame({"month": np.repeat(months, 10), "y": np.tile([0, 1], 5 * len(months)).astype(float),
                       "cluster": np.arange(10 * len(months)), "p_target": 0.5, "x_t": 0.0, "x_h": 1.0})
    sc = R.stack_predictions(df, ["x_t"], ["x_h"])
    # two fits per scored month (baseline, stack); the training rows grow by one month each time
    assert seen[0::2] == [10 * (2 + i) for i in range(len(R.eval_months()))]
    assert set(sc["month"]) == set(R.eval_months())
    assert (sc["train_fix"].to_numpy() == np.repeat([10 * (2 + i) for i in range(len(R.eval_months()))], 10)).all()


def test_stack_cell_gain_positive_when_helper_carries_the_signal():
    rng = np.random.default_rng(1)
    months = [str(m) for m in pd.period_range("2025-01", "2026-10", freq="M")]
    n = 120
    rows = []
    for m in months:
        h = rng.normal(size=n)
        t = rng.normal(size=n)
        y = rng.random(n) < 1 / (1 + np.exp(-(0.6 * t + 1.2 * h)))
        rows.append(pd.DataFrame({"month": m, "y": y.astype(float), "x_t": t, "x_h": h,
                                  "p_target": 1 / (1 + np.exp(-0.6 * t))}))
    df = pd.concat(rows, ignore_index=True)
    df["cluster"] = np.arange(len(df))
    rec, sc = R.stack_cell(df, ["x_t"], ["x_h"], reps=200, seed=7)
    assert rec["readable"] and rec["effect"] > 0.05 and rec["ci_lo"] > 0 and rec["p"] < 1e-6
    # a pure-noise helper gives no gain
    df["x_h"] = rng.normal(size=len(df))
    rec0, _ = R.stack_cell(df, ["x_t"], ["x_h"], reps=200, seed=7)
    assert rec0["readable"] and abs(rec0["effect"]) < 0.01


def test_stack_cell_unreadable_below_minimum_sample():
    months = [str(m) for m in pd.period_range("2026-01", "2026-10", freq="M")]
    df = pd.DataFrame({"month": np.repeat(months, 5), "y": np.tile([0.0, 1.0, 0.0, 1.0, 0.0], len(months)),
                       "x_t": 0.0, "x_h": 0.0, "p_target": 0.4})
    df["cluster"] = np.arange(len(df))
    rec, _ = R.stack_cell(df, ["x_t"], ["x_h"], reps=50, seed=7)
    assert not rec["readable"] and np.isnan(rec["p"])


def test_residual_cell_clusters_on_the_fixture_key_not_the_provider_id():
    df = pd.DataFrame({"fixture_id": [1] * 80, "cluster": np.repeat(np.arange(80), 1),
                       "res": np.r_[np.full(40, 0.1), np.full(40, -0.1)], "g": np.r_[np.ones(40), np.zeros(40)]})
    rec = R.residual_cell(df, "g", reps=200, seed=7)
    assert rec["fix_1"] == 40 and rec["fix_0"] == 40 and rec["readable"]
    assert rec["effect"] == pytest.approx(0.2)


def test_feature_classes_flags_ineligible_and_unregistered():
    reg = {"eligible": ["SOURCE_VERIFIED", "DERIVED_VERIFIED", "VALID_MODEL_ESTIMATE"],
           "rows": [{"feature": "a", "status": "active", "classification": "DERIVED_VERIFIED"},
                    {"feature": "b", "status": "active", "classification": "FABRICATED_VALUE"},
                    {"feature": "b", "status": "active", "classification": "DERIVED_VERIFIED"},
                    {"feature": "c", "status": "retired (kept for audit)", "classification": "FABRICATED_VALUE"}]}
    fc = R.feature_classes(["a", "a__missing", "b", "c", "d"], reg)
    assert fc["classes"]["a"] == "DERIVED_VERIFIED" and fc["classes"]["a__missing"] == "DERIVED_VERIFIED"
    assert fc["ineligible"] == {"b": "FABRICATED_VALUE", "c": "NOT_REGISTERED", "d": "NOT_REGISTERED"}


def test_load_team_track_keeps_earlier_fold_and_splits_stages(tmp_path):
    rid = "2025-03-01|L|A|B"
    rid2 = "2026-05-01|L|C|D"
    rows = []
    for fold, p in ((1, 0.6), (0, 0.4)):
        rows.append({"stage": "walk_forward", "fold": fold, "market": "over25", "candidate": "champion_recipe",
                     "row_id": rid, "date": pd.Timestamp("2025-03-01"), "league": "L", "y": 1.0, "p": p})
    rows.append({"stage": "walk_forward", "fold": 0, "market": "over25", "candidate": "market_bet",
                 "row_id": rid, "date": pd.Timestamp("2025-03-01"), "league": "L", "y": 1.0, "p": np.nan})
    rows.append({"stage": "holdout", "fold": 0, "market": "over25", "candidate": "champion_recipe",
                 "row_id": rid2, "date": pd.Timestamp("2026-05-01"), "league": "L", "y": 0.0, "p": 0.3})
    rows.append({"stage": "walk_forward", "fold": 3, "market": "over25", "candidate": "champion_recipe",
                 "row_id": rid2, "date": pd.Timestamp("2026-05-01"), "league": "L", "y": 0.0, "p": 0.9})
    pth = tmp_path / "p.parquet"
    pd.DataFrame(rows).to_parquet(pth)
    w = R.load_team_track(pth, "standard").set_index("row_id")
    assert w.loc[rid, "p_over25"] == pytest.approx(0.4)          # earlier fold kept
    assert np.isnan(w.loc[rid, "m_over25"])                       # no two-sided price -> missing
    assert w.loc[rid2, "p_over25"] == pytest.approx(0.3)          # holdout stage on/after 2026-04-01
    assert w.loc[rid2, "team_stage"] == "holdout" and w.loc[rid, "home_team"] == "A"


def test_props_wf_refuses_a_different_frame(tmp_path):
    f = pd.DataFrame({"date": ["2025-01-01"], "x": [1.0]})
    f.to_parquet(tmp_path / "f.parquet")
    with pytest.raises(ValueError, match="refused"):
        R.props_wf_clean(tmp_path / "f.parquet", tmp_path)


def test_no_price_value_column_in_report_outputs():
    d = R.REPORT_DATA
    if not d.exists():
        pytest.skip("report data not written yet")
    for f in d.glob("*.csv"):
        cols = pd.read_csv(f, nrows=0).columns
        assert not any(c.startswith(("m_over", "m_btts", "odds", "median_yes", "median_no")) for c in cols), f
