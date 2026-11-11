"""Phase 2 part 4 - the clean retrain harness (src/retrain_harness), on SMALL SYNTHETIC data only.

Owner rule: no model is trained and no backtest is run on real data until the data is 100% real and
the owner approves. Every fit below is on a generated frame of a few hundred matches; the real-data
check is the dry run, which only counts rows (scripts/phase2_retrain.py --dry-run).

    python -m pytest tests/test_phase2_retrain_harness.py -q
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.retrain_harness import candidates as C
from src.retrain_harness import evaluate as E
from src.retrain_harness import protocol as P
from src.retrain_harness import runner as R

ROOT = Path(__file__).resolve().parents[1]
STD = ("Championship", "League One")
NF = ("Sweden Allsvenskan",)
FEATS = ["home_scored_last5", "away_scored_last5", "home_xg_last5", "home_rest_days"]
ENV_ON = {P.ENABLE_VAR: "true"}


# ── synthetic world ───────────────────────────────────────────────────────────────────────────

def _frame(seed: int = 7, days: int = 150, teams: int = 8) -> pd.DataFrame:
    """Matches every 2nd day per league; goals Poisson from fixed club strengths; features are
    noisy pre-match signals of those strengths (with gaps); a two-sided O/U 2.5 price with a margin."""
    rng = np.random.default_rng(seed)
    rows = []
    for lg in STD + NF:
        att = rng.uniform(0.7, 1.4, teams)
        dfn = rng.uniform(0.7, 1.4, teams)
        clubs = [f"{lg[:3]} Club {i}" for i in range(teams)]
        for k in range(days):
            d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=2 * k)
            perm = rng.permutation(teams)
            for a, b in zip(perm[::2], perm[1::2]):
                lh, la = 1.45 * att[a] * dfn[b], 1.1 * att[b] * dfn[a]
                hg, ag = rng.poisson(lh), rng.poisson(la)
                hth, hta = rng.binomial(hg, 0.45), rng.binomial(ag, 0.45)
                p_over = 1 - sum(np.exp(-(lh + la)) * (lh + la) ** g / math.factorial(g) for g in range(3))
                rows.append({
                    "date": d, "league": lg, "home_team": clubs[a], "away_team": clubs[b],
                    "home_goals": hg, "away_goals": ag, "ht_home_goals": hth, "ht_away_goals": hta,
                    "home_scored_last5": lh + rng.normal(0, 0.3), "away_scored_last5": la + rng.normal(0, 0.3),
                    "home_xg_last5": (lh + rng.normal(0, 0.2)) if rng.random() > 0.3 else np.nan,
                    "home_ht_scored_last5": (0.45 * lh + rng.normal(0, 0.2)) if rng.random() > 0.2 else np.nan,
                    "home_rest_days": float(rng.integers(2, 8)),
                    "bookmaker_overround": 1.05,
                    "odds_over25": 1 / (p_over * 1.03), "odds_under25": 1 / ((1 - p_over) * 1.03),
                    "ou25_pair_is_price": True, "odds_btts": 1.9})
    f = pd.DataFrame(rows)
    tg = f["home_goals"] + f["away_goals"]
    f["over25"], f["over15"], f["over35"] = (tg > 2.5) * 1.0, (tg > 1.5) * 1.0, (tg > 3.5) * 1.0
    f["btts"] = ((f["home_goals"] > 0) & (f["away_goals"] > 0)) * 1.0
    ht = f["ht_home_goals"] + f["ht_away_goals"]
    f["ht_over05"], f["ht_over15"] = (ht >= 1) * 1.0, (ht >= 2) * 1.0
    f["home_has_ht_history"] = f["home_ht_scored_last5"].notna().astype(float)
    f["away_has_ht_history"] = 1.0
    return f


def _cfg(approved: bool = False, ht_variant: bool = False, league_variant: bool = False) -> dict:
    """A FIXTURE copy of the committed config on synthetic dates. approved=False empties
    approved_by/approved_at (the committed config is approved since 2026-11-07); the HT variant is
    off unless asked for. new_format excludes xG (stays out) and one HT column (the HT block)."""
    cfg = copy.deepcopy(P.load_config())
    cfg["holdout"].update(start="2024-09-01", end="2024-10-15")
    if approved:
        cfg["holdout"].update(approved_by="synthetic test", approved_at="2026-11-06")
    else:
        cfg["holdout"].update(approved_by=None, approved_at=None)
    cfg["owner_decisions"]["new_format_ht_variant"] = ht_variant
    cfg["owner_decisions"]["league_level_variant"] = league_variant
    cfg["walk_forward"].update(min_train_matchdays=50, block_matchdays=30, validation_matchdays=20)
    cfg["features"]["groups"] = {"form": {"include": True, "parity": "test",
                                          "columns": FEATS + ["bookmaker_overround", "home_ht_scored_last5",
                                                              "home_has_ht_history", "away_has_ht_history"]}}
    cfg["features"]["track_exclude"] = {"standard": [], "new_format": ["home_xg_last5", "home_ht_scored_last5"]}
    cfg["candidates"]["hgb"].update(max_iter=40)
    cfg["comparison"].update(n_boot=300, min_n_rows=50, min_matchdays=5)
    P.validate_config(cfg)
    return cfg


@pytest.fixture(scope="module")
def feat():
    return _frame()


def _frames(feat, cfg, track="standard", market="over25"):
    rows = P.eligible_rows(feat, cfg, track, market)
    pre, hold, _ = P.split_holdout(rows, cfg)
    return P.final_frames(pre, hold, cfg)


# ── config and guards ─────────────────────────────────────────────────────────────────────────

def test_committed_config_holds_the_owner_decisions_of_2026_11_07():
    """The COMMITTED config: holdout 2026-04-01..2026-11-08 approved by the owner, the comparison
    thresholds still the PG-1.0 values (reused, not changed), the new_format HT variant on."""
    cfg = P.load_config()
    h = cfg["holdout"]
    assert (h["start"], h["end"]) == ("2026-04-01", "2026-11-08")
    assert P.holdout_approved(cfg)
    assert str(h["approved_by"]).strip() and str(h["approved_at"]).startswith("2026-11-07")
    od = cfg["owner_decisions"]
    assert od["holdout_start"] == h["start"]
    assert od["new_format_ht_variant"] is True and P.ht_variant_on(cfg)
    assert "PG-1.0" in od["comparison_rules"]
    # the committed config's own HT block can run: no config-level blocker
    assert P.ht_variant_blockers(cfg) == []
    assert set(P.ht_block(cfg)) == {
        "home_ht_scored_last5", "away_ht_scored_last5", "home_ht_conceded_last5", "away_ht_conceded_last5",
        "home_ht_attack_str", "away_ht_attack_str", "combined_ht_goals_avg", "home_ht_over05_rate",
        "away_ht_over05_rate", "home_ht_over15_rate", "away_ht_over15_rate",
        "home_has_ht_history", "away_has_ht_history"}
    # with approval and the variant ready, only the repository flag still blocks fitting
    assert P.fit_blockers(cfg, {}) == [
        f"{P.ENABLE_VAR} is not 'true' (repository variable; set only after owner approval)"]
    assert P.fit_blockers(cfg, ENV_ON) == []


def test_committed_config_reuses_the_gate_thresholds_unchanged():
    cfg = P.load_config()
    gate_doc = json.loads((ROOT / "config" / "promotion_gate.json").read_text(encoding="utf-8"))
    assert gate_doc["gate_version"] == "PG-1.0"
    gate = gate_doc["holdout"]
    c = cfg["comparison"]
    assert c["min_effect_logloss"] == gate["material_ll_improvement"]
    assert c["min_n_rows"] == gate["min_n_per_track"]
    assert c["max_brier_worsening"] == gate["max_brier_worsening"]
    assert c["min_matchdays"] == gate["min_bootstrap_blocks"]
    assert c["min_league_n"] == gate["min_league_n"]
    assert (c["n_boot"], c["seed"], c["ece_bins"]) == (gate["n_boot"], gate["bootstrap_seed"], gate["ece_bins"])
    assert abs(c["family_alpha"] - (1 - gate["ci_level"])) < 1e-12
    assert c["correction"] == "holm" and c["reference"] == "champion_recipe"
    assert set(cfg["candidates"]) == set(P.CANDIDATES)


def test_fitting_is_refused_without_owner_approval_or_the_repository_flag(feat, tmp_path):
    with pytest.raises(PermissionError, match="not approved"):
        R.run(feat, _cfg(approved=False), tmp_path, env=ENV_ON)
    with pytest.raises(PermissionError, match=P.ENABLE_VAR):
        R.run(feat, _cfg(approved=True), tmp_path, env={})
    with pytest.raises(PermissionError, match=P.ENABLE_VAR):
        R.run(feat, _cfg(approved=True), tmp_path, env={P.ENABLE_VAR: "false"})
    assert not any(tmp_path.iterdir()), "a refused run must not create anything"


@pytest.mark.parametrize("field", ["approved_by", "approved_at"])
def test_a_copy_of_the_committed_config_with_the_approval_emptied_refuses_fitting(field, feat, tmp_path):
    cfg = copy.deepcopy(P.load_config())
    cfg["holdout"][field] = None
    assert not P.holdout_approved(cfg)
    assert any("not approved" in b for b in P.fit_blockers(cfg, ENV_ON))
    with pytest.raises(PermissionError, match="not approved"):
        R.run(feat, cfg, tmp_path, env=ENV_ON)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("bad", ["models", "models/registry", "output", "output/x", "config", "src"])
def test_run_folder_can_never_be_a_production_folder(bad):
    with pytest.raises(PermissionError):
        P.assert_run_folder(ROOT / bad)


def test_run_folder_outside_production_is_accepted(tmp_path):
    assert P.assert_run_folder(tmp_path / "runs") == (tmp_path / "runs").resolve()


def test_feature_list_never_creates_a_column_and_never_uses_a_price():
    cfg = _cfg()
    cols = ["home_scored_last5", "away_scored_last5", "home_xg_last5", "home_rest_days", "bookmaker_overround"]
    feats, ex = P.feature_list(cfg, "standard", cols)
    assert feats == FEATS
    assert ex["bookmaker_overround"].startswith("price-derived")
    assert ex["home_ht_scored_last5"] == "not in the feature frame"
    feats_nf, ex_nf = P.feature_list(cfg, "new_format", cols)
    assert "home_xg_last5" not in feats_nf and ex_nf["home_xg_last5"] == "excluded for track new_format"


# ── splits ────────────────────────────────────────────────────────────────────────────────────

def test_walk_forward_is_whole_date_expanding_and_never_touches_the_holdout(feat):
    cfg = _cfg()
    hs, he = P.holdout_window(cfg)
    rows = P.eligible_rows(feat, cfg, "standard", "over25")
    pre, hold, after = P.split_holdout(rows, cfg)
    assert len(pre) + len(hold) + len(after) == len(rows)
    assert pd.to_datetime(pre["date"]).max() < hs and pd.to_datetime(hold["date"]).min() >= hs
    folds = P.walk_forward_folds(pre, cfg)
    assert len(folds) >= 2
    tested = set()
    prev_end = None
    for f in folds:
        fr = P.fold_frames(pre, f)
        P.assert_disjoint(fr)
        assert pd.to_datetime(fr["validation"]["date"]).nunique() == cfg["walk_forward"]["validation_matchdays"]
        assert len(fr["fit"]) + len(fr["validation"]) == len(fr["train"])
        assert pd.to_datetime(fr["train"]["date"]).min() == pd.to_datetime(pre["date"]).min()   # expanding
        assert pd.to_datetime(fr["test"]["date"]).max() < hs
        ids = set(R._row_ids(fr["test"]))
        assert not ids & tested, "a row was tested twice"
        tested |= ids
        if prev_end is not None:
            assert f["test_start"] > prev_end
        prev_end = f["test_end"]
        # whole dates: no test date also appears in training
        assert not set(pd.to_datetime(fr["test"]["date"])) & set(pd.to_datetime(fr["train"]["date"]))
    assert not tested & set(R._row_ids(hold))


def test_final_holdout_frames_train_only_before_the_holdout(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    P.assert_disjoint(fr)
    assert pd.to_datetime(fr["train"]["date"]).max() < P.holdout_window(cfg)[0]
    assert pd.to_datetime(fr["test"]["date"]).min() >= P.holdout_window(cfg)[0]


def test_rows_without_form_or_label_are_not_eligible(feat):
    cfg = _cfg()
    f = feat.copy()
    f.loc[f.index[:5], "home_scored_last5"] = np.nan
    f.loc[f.index[5:9], "over25"] = np.nan
    rows = P.eligible_rows(f, cfg, "standard", "over25")
    std = f[f["league"].isin(STD)]
    assert len(rows) == int((std["home_scored_last5"].notna() & std["over25"].notna()).sum())


def test_dry_run_plan_counts_rows_and_fits_nothing(feat, monkeypatch):
    def _no_fit(*a, **k):
        raise AssertionError("the plan must never fit")
    monkeypatch.setattr(C.Candidate, "fit", _no_fit)
    for cls in (C.ChampionRecipe, C.Logistic, C.HGB, C.DixonColesCandidate, C.MarketOnly, C.ModelMarket):
        monkeypatch.setattr(cls, "fit", _no_fit)
    cfg = _cfg()
    plan = P.build_plan(feat, cfg)
    s = plan["tracks"]["standard"]["markets"]["over25"]
    assert s["eligible_rows"] == int(feat["league"].isin(STD).sum())
    assert s["walk_forward_region"] + s["holdout"] + s["after_holdout_unused"] == s["eligible_rows"]
    assert sum(v["walk_forward_region"] + v["holdout"] + v["after_holdout_unused"]
               for v in s["per_league"].values()) == s["eligible_rows"]
    assert s["price"]["walk_forward_region"]["two_sided"] == s["walk_forward_region"]
    assert plan["tracks"]["standard"]["markets"]["btts"]["price"]["holdout"]["two_sided"] == 0
    assert plan["tracks"]["standard"]["markets"]["ht_over05"]["price"]["holdout"]["note"].startswith("no price")
    assert plan["holdout"]["approved"] is False


# ── candidates ────────────────────────────────────────────────────────────────────────────────

def _cand(name, cfg, market="over25", track="standard"):
    feats, _ = P.feature_list(cfg, track, _frame(days=10).columns)
    return C.build(name, cfg, track, market, feats)


@pytest.mark.parametrize("name", ["logistic", "hgb", "champion_recipe"])
def test_model_probability_of_a_row_does_not_depend_on_its_batch(feat, name):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    c = _cand(name, cfg).fit(fr)
    test = fr["test"].copy()
    test.loc[test.index[::3], "home_xg_last5"] = np.nan
    full = c.predict(test)
    alone = np.array([c.predict(test.iloc[[i]])[0] for i in range(0, len(test), 7)])
    assert np.isfinite(full).all()
    assert np.allclose(full[::7], alone, atol=1e-12)


def test_logistic_drops_a_feature_with_no_training_value_instead_of_filling_it(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    for k in ("train", "fit"):
        fr[k] = fr[k].assign(home_xg_last5=np.nan)
    c = _cand("logistic", cfg).fit(fr)
    assert "home_xg_last5" not in c.used and any("home_xg_last5" in n for n in c.notes)


def test_hgb_is_nan_native(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    c = _cand("hgb", cfg).fit(fr)
    t = fr["test"].assign(home_xg_last5=np.nan, home_rest_days=np.nan)
    assert np.isfinite(c.predict(t)).all()


def test_champion_recipe_never_fits_on_the_test_labels(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    a = _cand("champion_recipe", cfg).fit(fr).predict(fr["test"])
    flipped = dict(fr, test=fr["test"].assign(over25=1.0 - fr["test"]["over25"]))
    b = _cand("champion_recipe", cfg).fit(flipped).predict(fr["test"])
    assert np.allclose(a, b, atol=1e-12)


def test_champion_recipe_uses_missing_indicators_and_a_persisted_train_only_imputer(feat):
    cfg = _cfg()
    c = _cand("champion_recipe", cfg).fit(_frames(feat, cfg))
    assert c.payload["imputer_version"] == "v9-train-only-imputer-1.1"
    assert any(f.endswith("__missing") for f in c.payload["feature_cols"])
    assert c.payload["holdout_independence"]["holdout_labels_used_for_fitting"] is False


def test_dixon_coles_is_coherent_and_gives_no_probability_to_an_unseen_club(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    p = {m: _cand("dixon_coles", cfg, market=m).fit(fr).predict(fr["test"]) for m in ("over15", "over25", "over35")}
    assert np.isfinite(p["over25"]).all()
    assert (p["over15"] >= p["over25"]).all() and (p["over25"] >= p["over35"]).all()
    c = _cand("dixon_coles", cfg).fit(fr)
    lo, hi = cfg["candidates"]["dixon_coles"]["rho_bounds"]
    assert lo <= c.dc.rho <= hi
    new = fr["test"].iloc[:2].copy()
    new.loc[new.index[0], "home_team"] = "Promoted FC"
    new.loc[new.index[1], "league"] = "Unseen League"
    assert np.isnan(c.predict(new)).all()


def test_dixon_coles_ht_market_reads_half_time_goals(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg, market="ht_over05")
    c = _cand("dixon_coles", cfg, market="ht_over05").fit(fr)
    ht_mu = np.mean([m[0] + m[1] for m in c.dc.league_mu.values()])
    ft = (fr["train"]["home_goals"] + fr["train"]["away_goals"]).mean()
    assert ht_mu < 0.7 * ft


def test_market_only_devigs_two_sided_pairs_only():
    cfg = _cfg()
    df = pd.DataFrame({"odds_over25": [1.9, 2.1, 1.8, np.nan], "odds_under25": [1.95, 1.75, 2.0, 1.9],
                       "ou25_pair_is_price": [True, True, False, True], "odds_btts": [1.8, 1.9, 2.0, 2.1]})
    p = _cand("market_only", cfg).predict(df)
    assert np.isfinite(p[:2]).all() and np.isnan(p[2:]).all()
    q = 1 - p[0]
    k = np.log(p[0]) / np.log(1 / 1.9)                      # power de-vig: p_i = (1/o_i)^k
    assert abs((1 / 1.95) ** k - q) < 1e-6
    assert np.isnan(_cand("market_only", cfg, market="btts").predict(df)).all()
    assert _cand("market_only", cfg).deployable is False    # closing-preferred price class


def test_model_market_blend_is_fitted_on_validation_only(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    a = _cand("model_market", cfg).fit(fr).predict(fr["test"])
    assert np.isfinite(a).all()
    flipped_test = dict(fr, test=fr["test"].assign(over25=1.0 - fr["test"]["over25"]))
    assert np.allclose(a, _cand("model_market", cfg).fit(flipped_test).predict(fr["test"]))
    v = fr["validation"].assign(over25=1.0 - fr["validation"]["over25"])
    flipped_val = dict(fr, validation=v)
    assert not np.allclose(a, _cand("model_market", cfg).fit(flipped_val).predict(fr["test"]))


def test_model_market_gives_no_probability_without_a_price(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    c = _cand("model_market", cfg).fit(fr)
    assert np.isnan(c.predict(fr["test"].assign(odds_under25=np.nan))).all()
    assert np.isnan(_cand("model_market", cfg, market="btts").fit(fr).predict(fr["test"])).all()


# ── evaluation ────────────────────────────────────────────────────────────────────────────────

def test_metrics_known_values():
    y = np.array([1, 0, 1, 0] * 25, float)
    m = E.metrics(y, np.full(100, 0.5), 10)
    assert abs(m["log_loss"] - np.log(2)) < 1e-12 and abs(m["brier"] - 0.25) < 1e-12
    assert m["ece"] == 0.0 and m["n"] == 100
    t = E.calibration_table(y, np.full(100, 0.5), 10)
    assert t == [{"bin": "0.5-0.6", "n": 100, "mean_p": 0.5, "observed": 0.5}]
    assert E.metrics([1, np.nan], [np.nan, 0.3], 10) == {"n": 0}


def test_calibration_slope_recovers_a_known_line():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 20000)
    p_true = 1 / (1 + np.exp(-(0.2 + 0.5 * x)))
    y = (rng.random(20000) < p_true).astype(float)
    a, b = E.calibration_line(y, 1 / (1 + np.exp(-x)))
    assert abs(a - 0.2) < 0.05 and abs(b - 0.5) < 0.05


def test_clv_needs_real_entry_and_closing_prices():
    mk = P.load_config()["markets"]["over25"]
    df = pd.DataFrame({"odds_over25": [2.0, 2.0]})
    assert E.clv(df, [0.6, 0.6], mk, "over25")["over"]["n"] == 0
    df = pd.DataFrame({"odds_over25_over_entry": [2.0, 2.0, 2.0, np.nan],
                       "odds_over25_over_close": [1.8, 2.2, 1.9, 1.8],
                       "odds_over25_under_entry": [1.9, 1.9, 1.9, 1.9],
                       "odds_over25_under_close": [2.0, 1.7, 1.9, 2.0]})
    r = E.clv(df, [0.6, 0.6, 0.4, 0.6], mk, "over25")
    assert r["over"]["n"] == 2 and r["over"]["eligible_rows"] == 3
    assert abs(r["over"]["mean"] - np.mean([2.0 / 1.8 - 1, 2.0 / 2.2 - 1])) < 1e-12
    assert r["under"]["n"] == 1 and abs(r["under"]["mean"] - 0.0) < 1e-12        # 0.6 x 1.9 > 1 only on row 3


def test_holm_adjustment():
    adj = E.holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def _pred(n_days, per_day, p_ref, p_cand, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for k in range(per_day):
            truth = 0.3 if (d + k) % 2 else 0.7
            y = float(rng.random() < truth)
            for cand, p in (("champion_recipe", p_ref(truth)), ("hgb", p_cand(truth))):
                rows.append({"stage": "walk_forward", "fold": 0, "track": "standard", "market": "over25",
                             "candidate": cand, "row_id": f"{d}-{k}", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                             "league": "Championship", "y": y, "p": p})
    return pd.DataFrame(rows)


def test_comparison_verdicts():
    cfg = _cfg()
    good = E.compare(_pred(60, 30, lambda t: 0.5, lambda t: t), cfg)
    assert good["verdict"].tolist() == [E.BETTER] and good["delta_log_loss"].iloc[0] < 0
    bad = E.compare(_pred(60, 30, lambda t: t, lambda t: 0.5), cfg)
    assert bad["verdict"].tolist() == [E.NOT_SHOWN]
    small = E.compare(_pred(3, 5, lambda t: 0.5, lambda t: t), cfg)
    assert small["verdict"].tolist() == [E.INSUFFICIENT]
    none = E.compare(_pred(60, 30, lambda t: 0.5, lambda t: np.nan), cfg)
    assert none["verdict"].tolist() == [E.NOT_COMPARABLE]


# ── end to end (synthetic) ────────────────────────────────────────────────────────────────────

def test_end_to_end_run_writes_only_its_run_folder(feat, tmp_path):
    cfg = _cfg(approved=True)
    for k in ("over15", "over35", "ht_over15"):
        cfg["markets"].pop(k)
    serving = sorted(p.name for p in (ROOT / "models").glob("*.pkl"))
    folder = R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="synthetic")
    for f in ("manifest.json", "predictions.parquet", "metrics.json", "per_league.csv", "comparison.csv",
              "holdout_opened.json"):
        assert (folder / f).exists(), f
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert man["config_sha256"] == P.config_sha(cfg) and "promotion" in man
    assert not [e for e in man["errors"] if "error" in e], man["errors"]
    pred = pd.read_parquet(folder / "predictions.parquet")
    assert set(pred["stage"]) == {"walk_forward", "holdout"}
    assert set(pred["candidate"]) == set(P.CANDIDATES)
    hs = P.holdout_window(cfg)[0]
    assert (pd.to_datetime(pred.loc[pred["stage"] == "walk_forward", "date"]) < hs).all()
    assert (pd.to_datetime(pred.loc[pred["stage"] == "holdout", "date"]) >= hs).all()
    models = sorted((folder / "models").rglob("*.pkl"))
    assert len(models) == 2 * 3 * len(P.CANDIDATES)                     # tracks x markets x candidates
    cmp_ = pd.read_csv(folder / "comparison.csv")
    assert set(cmp_["verdict"]) <= {E.BETTER, E.NOT_SHOWN, E.INSUFFICIENT, E.NOT_COMPARABLE}
    assert (cmp_["family_size"] == np.isfinite(cmp_["p_one_sided"]).sum()).all()
    text = " ".join(p.read_text(encoding="utf-8", errors="replace").lower()
                    for p in folder.iterdir() if p.suffix in (".json", ".csv"))
    assert "edge found" not in text and "no edge" not in text
    assert sorted(p.name for p in (ROOT / "models").glob("*.pkl")) == serving
    with pytest.raises(PermissionError, match="already scored"):
        R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="synthetic", stages=("holdout",))


# ── the new_format HT variant (owner_decisions.new_format_ht_variant, synthetic) ──────────────────

def test_ht_variant_feature_sets_restore_the_ht_block_only(feat):
    cfg = _cfg(ht_variant=True)
    sets = P.feature_sets(cfg, "new_format", feat.columns)
    assert set(sets) == {"", P.HT_VARIANT_SUFFIX}
    base, ht = sets[""][0], sets[P.HT_VARIANT_SUFFIX][0]
    assert "home_ht_scored_last5" not in base and "home_ht_scored_last5" in ht
    assert "home_xg_last5" not in base and "home_xg_last5" not in ht            # xG stays excluded
    assert sets[P.HT_VARIANT_SUFFIX][1]["home_xg_last5"] == "excluded for track new_format"
    assert set(ht) - set(base) == {"home_ht_scored_last5"}
    assert "bookmaker_overround" not in ht                                       # never a price
    assert set(P.feature_sets(cfg, "standard", feat.columns)) == {""}
    assert set(P.feature_sets(_cfg(ht_variant=False), "new_format", feat.columns)) == {""}


def test_ht_variant_candidate_names_feature_candidates_twice_market_only_once():
    cfg = _cfg(ht_variant=True)
    labels = [r[0] for r in P.candidate_runs(cfg, "new_format", list(P.CANDIDATES))]
    for n in P.FEATURE_CANDIDATES:
        assert labels.count(n) == 1 and labels.count(n + "+ht") == 1
    assert labels.count("market_only") == 1 and "market_only+ht" not in labels
    assert labels.count("dixon_coles") == 1 and "dixon_coles+ht" not in labels
    assert len(labels) == len(set(labels)) == len(P.CANDIDATES) + len(P.FEATURE_CANDIDATES)
    assert [r[0] for r in P.candidate_runs(cfg, "standard", list(P.CANDIDATES))] == list(P.CANDIDATES)
    off = _cfg(ht_variant=False)
    assert [r[0] for r in P.candidate_runs(off, "new_format", list(P.CANDIDATES))] == list(P.CANDIDATES)


def test_ht_variant_dry_run_lists_both_feature_lists_and_row_counts(feat, capsys):
    import scripts.phase2_retrain as cli
    cfg = _cfg(ht_variant=True)
    plan = P.build_plan(feat, cfg)
    assert plan["ht_variant"]["on"] and plan["ht_variant"]["blockers"] == []
    t = plan["tracks"]["new_format"]
    v = t["variants"]["+ht"]
    assert v["restored"] == ["home_ht_scored_last5"]
    assert set(v["candidates"]) == {n + "+ht" for n in P.FEATURE_CANDIDATES}
    assert "hgb" in t["candidates"] and "hgb+ht" in t["candidates"] and "market_only+ht" not in t["candidates"]
    for m, s in t["markets"].items():
        vs = v["markets"][m]
        assert (vs["eligible_rows"], vs["walk_forward_region"], vs["holdout"]) == \
            (s["eligible_rows"], s["walk_forward_region"], s["holdout"])
    rows = P.eligible_rows(feat, cfg, "new_format", "over25")
    pre, hold, _ = P.split_holdout(rows, cfg)
    hc = v["markets"]["over25"]["ht_block_complete"]
    assert hc == {"walk_forward_region": int(pre["home_ht_scored_last5"].notna().sum()),
                  "holdout": int(hold["home_ht_scored_last5"].notna().sum())}
    assert 0 < hc["holdout"] < v["markets"]["over25"]["holdout"]
    assert plan["tracks"]["standard"]["variants"] == {}
    cli.print_plan(plan, {})
    out = capsys.readouterr().out
    assert "variant '+ht'" in out and "home_ht_scored_last5" in out and "hgb+ht" in out


def test_ht_variant_that_cannot_run_refuses_fitting_never_skips(feat, tmp_path):
    # (1) the frame lacks an HT block column
    cfg = _cfg(approved=True, ht_variant=True)
    with pytest.raises(PermissionError, match="HT variant.*not in the feature frame"):
        R.run(feat.drop(columns=["home_ht_scored_last5"]), cfg, tmp_path, env=ENV_ON)
    # (2) the exclusion list holds no HT column: the variant would equal the champion list
    cfg2 = _cfg(approved=True, ht_variant=True)
    cfg2["features"]["track_exclude"]["new_format"] = ["home_xg_last5"]
    assert any("holds no HT column" in b for b in P.fit_blockers(cfg2, ENV_ON))
    with pytest.raises(PermissionError, match="HT variant"):
        R.run(feat, cfg2, tmp_path, env=ENV_ON)
    # (3) an HT column outside the included feature groups would be dropped silently
    cfg3 = _cfg(approved=True, ht_variant=True)
    cfg3["features"]["groups"]["form"]["columns"].remove("away_has_ht_history")
    assert any("not in an included feature group" in b for b in P.fit_blockers(cfg3, ENV_ON))
    with pytest.raises(PermissionError, match="HT variant"):
        R.run(feat, cfg3, tmp_path, env=ENV_ON)
    # (4) the flag is on but the new_format track is not configured
    cfg4 = _cfg(approved=True, ht_variant=True)
    cfg4["tracks"] = ["standard"]
    assert any("not configured" in b for b in P.fit_blockers(cfg4, ENV_ON))
    assert not any(tmp_path.iterdir()), "a refused run must not create anything"
    # the flag must be a boolean
    bad = _cfg()
    bad["owner_decisions"]["new_format_ht_variant"] = "yes"
    with pytest.raises(ValueError, match="new_format_ht_variant"):
        P.validate_config(bad)


def test_ht_variant_end_to_end_scores_both_variants_in_one_holm_family(feat, tmp_path):
    cfg = _cfg(approved=True, ht_variant=True)
    for k in [m for m in cfg["markets"] if m != "over25"]:
        cfg["markets"].pop(k)
    folder = R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="synthetic_ht")
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert not [e for e in man["errors"] if "error" in e], man["errors"]
    nf_labels = set(man["candidates_by_track"]["new_format"])
    assert nf_labels == set(P.CANDIDATES) | {n + "+ht" for n in P.FEATURE_CANDIDATES}
    assert set(man["candidates_by_track"]["standard"]) == set(P.CANDIDATES)
    assert "home_ht_scored_last5" in man["features"]["new_format"]["variants"]["+ht"]["features"]
    assert "home_ht_scored_last5" not in man["features"]["new_format"]["features"]
    pred = pd.read_parquet(folder / "predictions.parquet")
    for stage in ("walk_forward", "holdout"):
        g = pred[(pred["stage"] == stage) & (pred["track"] == "new_format")]
        assert set(g["candidate"]) == nf_labels
        assert not (pred[(pred["stage"] == stage) & (pred["track"] == "standard")]["candidate"]
                    .str.endswith("+ht").any())
    h = pred[(pred["stage"] == "holdout") & (pred["track"] == "new_format")].pivot_table(
        index="row_id", columns="candidate", values="p")
    assert not np.allclose(h["hgb"], h["hgb+ht"]), "the variant must train on the HT block"
    cmp_ = pd.read_csv(folder / "comparison.csv")
    nf = cmp_[cmp_["track"] == "new_format"]
    for stage in ("walk_forward", "holdout"):
        assert set(nf[nf["stage"] == stage]["candidate"]) == nf_labels - {"champion_recipe"}
    assert (cmp_["family_size"] == np.isfinite(cmp_["p_one_sided"]).sum()).all()
    assert int(cmp_["family_size"].iloc[0]) >= 2 * (len(nf_labels) - 1)
    models = sorted(p.name for p in (folder / "models" / "new_format" / "over25").glob("*.pkl"))
    assert models == sorted(f"{n}.pkl" for n in nf_labels)


def _no_load(monkeypatch):
    def _boom(cfg):
        raise AssertionError("data must not be loaded when fitting is refused")
    monkeypatch.setattr(P, "load_committed_frame", _boom)


def test_cli_refuses_an_unapproved_config_before_loading_any_data(monkeypatch, tmp_path):
    import scripts.phase2_retrain as cli
    _no_load(monkeypatch)
    cfg = copy.deepcopy(P.load_config())
    cfg["holdout"].update(approved_by=None, approved_at=None)
    path = tmp_path / "phase2_retrain_unapproved.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv(P.ENABLE_VAR, "true")
    with pytest.raises(PermissionError, match="not approved"):
        cli.main(["--config", str(path), "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_cli_refuses_the_committed_config_without_the_flag_before_loading_any_data(monkeypatch, tmp_path):
    import scripts.phase2_retrain as cli
    _no_load(monkeypatch)
    monkeypatch.delenv(P.ENABLE_VAR, raising=False)
    with pytest.raises(PermissionError, match=P.ENABLE_VAR):
        cli.main(["--out", str(tmp_path / "runs")])
    monkeypatch.setenv(P.ENABLE_VAR, "false")
    with pytest.raises(PermissionError, match=P.ENABLE_VAR):
        cli.main(["--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_cli_refuses_a_variant_that_cannot_run_before_loading_any_data(monkeypatch, tmp_path):
    import scripts.phase2_retrain as cli
    _no_load(monkeypatch)
    cfg = copy.deepcopy(P.load_config())
    cfg["features"]["track_exclude"]["new_format"] = ["home_xg_last5", "away_xg_last5"]   # no HT column
    path = tmp_path / "phase2_retrain_no_ht.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv(P.ENABLE_VAR, "true")
    with pytest.raises(PermissionError, match="HT variant"):
        cli.main(["--config", str(path), "--out", str(tmp_path / "runs")])


# ── workflow and code hygiene ─────────────────────────────────────────────────────────────────

def test_workflow_is_dispatch_only_gated_and_read_only():
    text = (ROOT / ".github" / "workflows" / "phase2_retrain.yml").read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    on = wf.get(True, wf.get("on"))                       # yaml reads the bare key `on` as True
    assert set(on) == {"workflow_dispatch"}
    jobs = wf["jobs"]
    assert jobs and all(j.get("if") == "vars.PHASE2_RETRAIN_ENABLED == 'true'" for j in jobs.values())
    assert all(j["permissions"] == {"contents": "read"} for j in jobs.values())
    for bad in ("git push", "git commit", "schedule:", "promotion_gate", "models/model_"):
        assert bad not in text


def test_harness_code_has_no_invented_defaults_and_no_forbidden_claims():
    from src.integrity.data_truth import scan_defaults
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "retrain_harness").glob("*.py"))
    files.append("scripts/phase2_retrain.py")
    assert scan_defaults(files) == []
    for f in files:
        t = (ROOT / f).read_text(encoding="utf-8").lower()
        for phrase in ("edge found", "no edge", "all fixed"):
            assert phrase not in t, (f, phrase)


# ── NOT_FITTED, never a crash (coordinator 2026-11-09: the 11 early-fold failures were a bug) ─────

def test_hgb_fits_when_a_feature_has_no_training_value_and_records_it(feat):
    """Reproduces run 42512834177: early Standard folds (2021-2022) have NO xG / HT history at all.
    sklearn 1.9's HistGradientBoosting binning raises 'window shape cannot be larger than input array
    shape' on any all-missing column. Now the column is dropped and recorded (the
    champion recipe's rule), and predict scores on exactly the fitted columns."""
    cfg = _cfg()
    fr = _frames(feat, cfg)
    for k in ("train", "fit", "validation"):
        fr[k] = fr[k].copy()
        fr[k]["home_xg_last5"] = np.nan                      # no training value in this window
    for name in ("hgb", "model_market"):
        c = _cand(name, cfg)
        c.fit(fr)
        p = np.asarray(c.predict(fr["test"]), float)
        assert np.isfinite(p).any(), name
        assert any("home_xg_last5" in n and "no training value" in n for n in c.notes), (name, c.notes)


def test_hgb_without_any_usable_column_or_with_one_class_is_not_fitted(feat):
    cfg = _cfg()
    fr = _frames(feat, cfg)
    c = _cand("hgb", cfg)
    allnan = {k: v.assign(**{f: np.nan for f in c.features}) for k, v in fr.items()}
    with pytest.raises(C.NotFitted, match="no feature column"):
        c.fit(allnan)
    one = {k: v.copy() for k, v in fr.items()}
    one["train"][c.target] = 1.0
    with pytest.raises(C.NotFitted, match="one outcome class"):
        _cand("hgb", cfg).fit(one)


def test_runner_records_not_fitted_and_fails_the_run_on_a_real_error(feat, tmp_path, monkeypatch):
    cfg = _cfg(approved=True)
    for k in ("over15", "over35", "ht_over15", "btts", "ht_over05"):
        cfg["markets"].pop(k, None)
    real_build = C.build

    class _NF(C.Candidate):
        name = "x"

        def fit(self, frames):
            raise C.NotFitted("synthetic reason")

    def build_nf(name, *a, **k):
        if name == "hgb":
            return _NF({}, cfg["markets"]["over25"], [], "standard", "over25")
        return real_build(name, *a, **k)
    monkeypatch.setattr(C, "build", build_nf)
    folder = R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="nf")
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    nf = [e for e in man["errors"] if e.get("status") == R.NOT_FITTED]
    assert nf and all(e["candidate"] == "hgb" and e["reason"] == "synthetic reason" for e in nf)
    cov = pd.read_csv(folder / "coverage.csv")
    h = cov[cov["candidate"] == "hgb"]
    assert (h["folds_not_fitted"] > 0).all() and (h["folds_scored"] == 0).all()

    class _Boom(C.Candidate):
        name = "y"

        def fit(self, frames):
            raise ZeroDivisionError("a bug")

    def build_boom(name, *a, **k):
        if name == "hgb":
            return _Boom({}, cfg["markets"]["over25"], [], "standard", "over25")
        return real_build(name, *a, **k)
    monkeypatch.setattr(C, "build", build_boom)
    with pytest.raises(R.RunErrors, match="a bug"):
        R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="boom")
    assert (tmp_path / "boom" / "manifest.json").exists()           # reports written first


# ── the league-level variant (owner_decisions.league_level_variant, owner 2026-11-11, synthetic) ──

LEAGUE_COLS = ["league_asof_goals_pg", "league_asof_btts_rate"]


def _league_cfg(**k):
    cfg = _cfg(league_variant=True, **k)
    cfg["features"]["league_level"] = {
        "columns": LEAGUE_COLS + ["league_asof_ht_over05_rate"],
        "param_runs": [{"suffix": "+league+logitcal", "candidate": "champion_recipe", "league": True,
                        "params": {"calibration_input": "logit"}}]}
    cfg["features"]["track_exclude"]["new_format"].append("league_asof_ht_over05_rate")
    return cfg


@pytest.fixture(scope="module")
def lfeat(feat):
    from src.feature_engineering import _league_prior_dates_mean
    f = feat.copy()
    f["total_goals"] = f["home_goals"] + f["away_goals"]
    f["league_asof_goals_pg"] = _league_prior_dates_mean(f, "total_goals").to_numpy()
    f["league_asof_btts_rate"] = _league_prior_dates_mean(f, "btts").to_numpy()
    f["league_asof_ht_over05_rate"] = _league_prior_dates_mean(f, "ht_over05").to_numpy()
    return f


def test_committed_config_runs_the_league_variant_and_never_changes_the_champion_list():
    cfg = P.load_config()
    assert P.league_variant_on(cfg) and P.league_variant_blockers(cfg) == []
    cols = P.league_level_columns(cfg)
    assert cols and all(c.startswith("league_asof_") for c in cols)
    grouped = {c for g in cfg["features"]["groups"].values() for c in g["columns"]}
    assert not set(cols) & grouped                                 # the champion list never reads them
    assert [r["suffix"] for r in P.league_param_runs(cfg)] == ["+league+logitcal", "+logitcal"]
    assert P.run_params(cfg, "champion_recipe+logitcal") == {"calibration_input": "logit"}
    assert P.run_params(cfg, "champion_recipe") == {} and P.run_params(cfg, "hgb+league") == {}
    # the HT block of the new_format variant never takes a league-level column
    assert not [c for c in P.ht_block(cfg) if c.startswith("league_asof_")]


def test_league_variant_feature_sets_add_the_block_only(lfeat):
    cfg = _league_cfg()
    for track in ("standard", "new_format"):
        sets = P.feature_sets(cfg, track, lfeat.columns)
        base, lg = sets[""][0], sets["+league"][0]
        assert not [c for c in base if c.startswith("league_asof_")]
        added = set(lg) - set(base)
        assert added == (set(LEAGUE_COLS) | ({"league_asof_ht_over05_rate"} if track == "standard" else set()))
        assert sets["+league+logitcal"][0] == lg
    assert set(P.feature_sets(_cfg(), "standard", lfeat.columns)) == {""}


def test_league_variant_candidate_runs_and_blockers(lfeat):
    cfg = _league_cfg()
    labels = [r[0] for r in P.candidate_runs(cfg, "standard", list(P.CANDIDATES))]
    for n in P.FEATURE_CANDIDATES:
        assert labels.count(n) == 1 and labels.count(n + "+league") == 1
    assert "champion_recipe+league+logitcal" in labels and "market_only+league" not in labels
    assert len(labels) == len(set(labels))
    assert P.league_variant_blockers(cfg, lfeat.columns) == []
    assert P.league_variant_blockers(cfg, [c for c in lfeat.columns if c != "league_asof_btts_rate"])
    bad = _league_cfg()
    bad["features"]["groups"]["form"]["columns"].append("league_asof_btts_rate")
    assert P.league_variant_blockers(bad)


def test_param_run_fits_the_champion_on_logit_calibration_and_serves_the_same_scale(lfeat):
    from src.model import predict_proba
    cfg = _league_cfg()
    fr = _frames(lfeat, cfg)
    feats = P.feature_sets(cfg, "standard", lfeat.columns)["+league"][0]
    a = C.build("champion_recipe", cfg, "standard", "over25", feats).fit(fr)
    b = C.build("champion_recipe", cfg, "standard", "over25", feats,
                overrides=P.run_params(cfg, "champion_recipe+league+logitcal")).fit(fr)
    assert "calibration_input" not in a.payload and b.payload["calibration_input"] == "logit"
    assert b.meta()["calibration_input"] == "logit"
    pb = b.predict(fr["test"])
    assert np.isfinite(pb).all() and not np.allclose(pb, a.predict(fr["test"]))
    # serving = the same payload through src.model.predict_proba, row by row = in a batch
    one = np.array([float(predict_proba(fr["test"].iloc[[i]], payload=b.payload).iloc[0]) for i in range(5)])
    assert np.allclose(one, pb[:5], atol=1e-12)


def test_league_variant_that_cannot_run_refuses_fitting(feat, tmp_path):
    cfg = _league_cfg(approved=True)
    with pytest.raises(PermissionError, match="league-level"):
        R.run(feat, cfg, tmp_path, env=ENV_ON, run_id="no_league_cols")


def test_league_variant_end_to_end_scores_every_label_in_one_holm_family(lfeat, tmp_path):
    cfg = _league_cfg(approved=True)
    for k in [m for m in cfg["markets"] if m != "over25"]:
        cfg["markets"].pop(k)
    folder = R.run(lfeat, cfg, tmp_path, env=ENV_ON, run_id="synthetic_league")
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert not [e for e in man["errors"] if "error" in e], man["errors"]
    want = (set(P.CANDIDATES) | {n + "+league" for n in P.FEATURE_CANDIDATES}
            | {"champion_recipe+league+logitcal"})
    assert set(man["candidates_by_track"]["standard"]) == want
    pred = pd.read_parquet(folder / "predictions.parquet")
    h = pred[(pred["stage"] == "holdout") & (pred["track"] == "standard")].pivot_table(
        index="row_id", columns="candidate", values="p")
    assert not np.allclose(h["hgb"], h["hgb+league"]), "the variant must train on the league block"
    cmp_ = pd.read_csv(folder / "comparison.csv")
    std = cmp_[(cmp_["track"] == "standard") & (cmp_["stage"] == "holdout")]
    assert set(std["candidate"]) == want - {"champion_recipe"}
    assert (cmp_["reference"] == "champion_recipe").all()
