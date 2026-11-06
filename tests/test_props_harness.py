"""Phase 2 props clean retrain harness (key propsharness, 2026-11-06). SYNTHETIC DATA ONLY.

The harness is BUILD ONLY: nothing here reads the committed player history, trains on real data or
prices a real quote. Covered: the owner gate (plan decisions + PHASE2_RETRAIN_ENABLED + the
dispatch-only workflow), verified-row selection under the three feature rules, the walk-forward and
holdout splits, the three candidates (train-fitted imputer, NaN-native, per-player base rate), the
metrics, the 'no hand adjustment' rule, the dry-run counts (today vs after the re-fetch) and CLV
against two-sided prices with the assumed margin kept apart.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import props_harness as H  # noqa: E402
from player_model import props_harness_clv as CLV  # noqa: E402
from player_model.player_stat_nulls import PARSED_WITH_RULE, REFETCHED, SOURCE_COL, UNVERIFIED  # noqa: E402


# ── synthetic data ───────────────────────────────────────────────────────────────────────────────

def _frame(n_players=60, n_days=90, seed=0, start="2026-01-01"):
    """A model-ready synthetic frame: one skill per player drives the label; f2 is noise with holes."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    skill = rng.normal(0, 1, n_players)
    rows = []
    fid = 1000
    for d in days:
        play = rng.random(n_players) < 0.5
        for p in np.flatnonzero(play):
            fid += 1
            f1 = skill[p] + rng.normal(0, 0.3)
            f2 = rng.normal() if rng.random() > 0.2 else np.nan
            y = float(rng.random() < 1 / (1 + np.exp(-(skill[p] - 0.3))))
            rows.append({"date": d, "player_id": int(p + 1), "fixture_id": fid, "f1": f1, "f2": f2,
                         "target_sot": y, "label_verified": True, "window_verified": True,
                         "all_prior_verified": True})
    return pd.DataFrame(rows)


def _raw(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _decisions(**kw):
    d = {"holdout_start": "2026-03-01", "holdout_end": "2026-03-31", "min_verified_share": 0.5,
         "markets": ["sot"], "feature_rule": "label_only"}
    d.update(kw)
    return d


def _plan(**kw):
    p = H.load_plan()
    p = json.loads(json.dumps(p))
    p["owner_decisions"] = _decisions(**kw)
    p["walk_forward"] = {"step_matchdays": 5, "n_folds": 2}
    return p


# ── the owner gate ───────────────────────────────────────────────────────────────────────────────

def test_committed_plan_waits_for_the_owner():
    plan = H.load_plan()
    assert plan["hand_adjustment"] == "none"
    assert H.missing_owner_decisions(plan) == list(H.OWNER_KEYS)
    with pytest.raises(PermissionError):
        H.owner_decisions(plan)
    assert set(plan["candidates"]) == set(H.CANDIDATES)
    assert set(plan["proposed"]["markets"]) <= set(H.MARKETS_IN_SCOPE)


def test_run_refuses_without_the_enable_variable(monkeypatch, tmp_path):
    monkeypatch.delenv(H.ENABLE_ENV, raising=False)
    with pytest.raises(PermissionError, match=H.ENABLE_ENV):
        H.run(_plan(), raw=pd.DataFrame(), out_dir=tmp_path, model_dir=tmp_path)


def test_run_refuses_while_a_decision_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(H.ENABLE_ENV, "true")
    plan = _plan()
    plan["owner_decisions"]["min_verified_share"] = None
    with pytest.raises(PermissionError, match="min_verified_share"):
        H.run(plan, raw=pd.DataFrame(), out_dir=tmp_path, model_dir=tmp_path)


@pytest.mark.parametrize("bad", [{"markets": ["sot3"]}, {"feature_rule": "anything"},
                                 {"min_verified_share": 1.5}, {"holdout_end": "2026-02-01"}])
def test_unusable_decisions_are_refused(bad):
    with pytest.raises(ValueError):
        H.validate_decisions(_decisions(**bad))


def test_workflow_is_dispatch_only_and_gated():
    import yaml
    text = (ROOT / ".github" / "workflows" / "props_clean_retrain.yml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    on = doc.get(True) or doc.get("on")
    assert set(on) == {"workflow_dispatch"}
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "schedule" not in code and "push:" not in code
    for job in doc["jobs"].values():
        assert job["if"] == "vars.PHASE2_RETRAIN_ENABLED == 'true'"
        assert job["permissions"] == {"contents": "read"}          # commits nothing
    assert "model_player_" not in code and "git push" not in code


def test_harness_never_applies_the_hand_adjustment():
    for mod in (H, CLV):
        src = inspect.getsource(mod)
        code = src.split('"""', 2)[2]                           # past the module docstring
        assert "hand_adjusted_prob(" not in code
        assert "prob_adjustment" not in code
    f = _frame(n_days=40)
    fitted = H.fit_window(f, ["f1", "f2"], "target_sot")
    ps = H.score(fitted, f.tail(50), ["f1", "f2"])
    for n, p in ps.items():
        own = fitted[n].predict(f.tail(50)[["f1", "f2"]], players=f.tail(50)["player_id"].to_numpy())
        np.testing.assert_array_equal(p, own)                  # the candidate's own probability


def test_harness_never_writes_a_serving_file():
    src = inspect.getsource(H)
    assert "model_player_" not in src.split('"""', 2)[2]
    assert "save_model(" not in src and "props_gate" not in src.split('"""', 2)[2]


# ── verified rows ────────────────────────────────────────────────────────────────────────────────

def test_source_frame_applies_the_refetch_and_marks_every_row():
    hist = _raw([
        {"date": "2026-01-01", "fixture_id": 1, "player_id": 7, "minutes": 90, "shots_on_target": 0},
        {"date": "2026-01-08", "fixture_id": 2, "player_id": 7, "minutes": 90, "shots_on_target": 0},
    ])
    cups = pd.DataFrame([{"date": "2026-01-05", "fixture_id": 9, "player_id": 7, "minutes": 90,
                          "shots_on_target": 2, SOURCE_COL: PARSED_WITH_RULE, "fixture_status": "FT"}])
    refetch = pd.DataFrame([{"fixture_id": 2, "player_id": 7, "shots_on_target": np.nan}])
    raw = H.source_frame(history=hist, cups=cups, refetch=refetch)
    by = raw.set_index("fixture_id")
    assert by.loc[1, SOURCE_COL] == UNVERIFIED
    assert by.loc[2, SOURCE_COL] == REFETCHED and np.isnan(by.loc[2, "shots_on_target"])
    assert by.loc[9, SOURCE_COL] == PARSED_WITH_RULE and by.loc[9, "harness_source"] == "cup_rows"
    assert H.label(raw, "sot").isna().sum() == 1                # a missing count is a missing label


def _player_rows(sources, minutes=None):
    minutes = minutes or [90] * len(sources)
    return _raw([{"date": f"2026-01-{i + 1:02d}", "fixture_id": 100 + i, "player_id": 5, "minutes": m,
                  SOURCE_COL: s, "shots_on_target": 1} for i, (s, m) in enumerate(zip(sources, minutes))])


def test_population_mirrors_build_features_and_the_three_rules():
    srcs = [UNVERIFIED, REFETCHED, REFETCHED, UNVERIFIED, REFETCHED, REFETCHED, REFETCHED,
            REFETCHED, REFETCHED, REFETCHED]
    pop = H.population(_player_rows(srcs)).set_index("fixture_id")
    assert 100 not in pop.index                                 # first appearance: no prior match
    assert pop.loc[101, "label_verified"] and not pop.loc[101, "window_verified"]
    assert not pop.loc[103, "label_verified"]
    assert not pop.loc[104, "all_prior_verified"]               # earlier unverified rows
    n = config.ROLLING_N
    # the first row whose previous ROLLING_N rows are all verified
    first_clean = 103 + n + 1
    if first_clean in pop.index:
        assert pop.loc[first_clean, "window_verified"]
        assert not pop.loc[first_clean - 1, "window_verified"]
    assert not pop["all_prior_verified"].any()


def test_population_drops_bench_rows_and_extra_time_cup_rows():
    raw = _player_rows([REFETCHED] * 4, minutes=[90, 5, 90, 90])
    raw["fixture_status"] = [None, None, None, "AET"]
    pop = H.population(raw)
    assert list(pop["fixture_id"]) == [102]                    # 101 bench, 103 extra time, 100 first
    assert pop["all_prior_verified"].all()


def test_after_refetch_view_counts_proven_fixtures_as_verified():
    raw = _player_rows([UNVERIFIED] * 4)
    extra = pd.to_numeric(raw["fixture_id"]).isin([100, 101, 102, 103])
    assert H.population(raw)["label_verified"].sum() == 0
    assert H.population(raw, verified_extra=extra)["all_prior_verified"].all()


# ── splits ───────────────────────────────────────────────────────────────────────────────────────

def test_walk_forward_folds_are_before_the_holdout_and_do_not_overlap():
    days = list(pd.date_range("2026-01-01", periods=60, freq="D"))
    folds = H.walk_forward_folds(days, "2026-02-20", step=7, n_folds=3)
    assert [f["fold"] for f in folds] == [0, 1, 2]
    assert folds[-1]["test_end"] == pd.Timestamp("2026-02-19")
    for a, b in zip(folds, folds[1:]):
        assert a["test_end"] < b["test_start"]
    assert all(f["n_matchdays"] == 7 for f in folds)
    assert len(H.walk_forward_folds(days[:10], "2026-01-09", step=7, n_folds=5)) == 1


def test_calibration_block_is_cut_on_a_date():
    d = pd.Series(pd.to_datetime(["2026-01-01"] * 50 + ["2026-01-02"] * 30 + ["2026-01-03"] * 20))
    assert H.calibration_cut(d) == pd.Timestamp("2026-01-03")
    d2 = pd.Series(pd.to_datetime(["2026-01-01"] * 90 + ["2026-01-02"] * 10))
    assert H.calibration_cut(d2) == pd.Timestamp("2026-01-02")    # the fit block keeps one matchday
    with pytest.raises(ValueError):
        H.calibration_cut(pd.Series(pd.to_datetime(["2026-01-01"] * 10)))


def test_split_of_uses_the_fixed_dates():
    s = H.split_of(pd.Series(pd.to_datetime(["2026-02-28", "2026-03-01", "2026-03-31", "2026-04-01"])),
                   "2026-03-01", "2026-03-31")
    assert list(s) == ["pre_holdout", "holdout", "holdout", "after_holdout"]


def test_profile_features_are_readmitted():
    cols = list(config.PLAYER_FEATURE_COLS) + sorted(config.PROFILE_FEATURES_WITHHELD)
    feats = H.harness_features(pd.DataFrame(columns=cols))
    assert config.PROFILE_FEATURES_WITHHELD <= set(feats)
    assert len(feats) == len(set(feats))


# ── candidates ───────────────────────────────────────────────────────────────────────────────────

def test_recipe_imputer_is_fitted_on_the_fit_block_only_and_saved():
    f = _frame(n_days=40)
    cut = H.calibration_cut(f["date"])
    fit = f[f["date"] < cut]
    m = H.fit_window(f, ["f1", "f2"], "target_sot")["recipe_retrained"]
    assert m.fill_["f2"] == pytest.approx(float(fit["f2"].median()))
    assert m.state()["fill_values"] == m.fill_


def test_recipe_drops_a_feature_with_no_training_value():
    f = _frame(n_days=40).assign(f3=np.nan)
    m = H.fit_window(f, ["f1", "f2", "f3"], "target_sot")
    assert m["recipe_retrained"].dropped_unfittable_ == ["f3"]
    assert "f3" not in m["recipe_retrained"].fill_


def test_every_candidate_is_batch_invariant():
    f = _frame(n_days=40)
    fitted = H.fit_window(f, ["f1", "f2"], "target_sot")
    test = _frame(n_days=5, seed=3, start="2026-03-01")
    full = H.score(fitted, test, ["f1", "f2"])
    for i in (0, 3, 7):
        one = H.score(fitted, test.iloc[[i]], ["f1", "f2"])
        for n in H.CANDIDATES:
            assert one[n][0] == pytest.approx(full[n][i], abs=1e-12)


def test_nan_native_scores_missing_values_without_a_fill():
    f = _frame(n_days=40)
    m = H.fit_window(f, ["f1", "f2"], "target_sot")["hgb_nan_native"]
    x = pd.DataFrame({"f1": [0.5], "f2": [np.nan]})
    p = m.predict(x)
    assert np.isfinite(p).all() and 0 < p[0] < 1
    assert "fill" not in json.dumps(m.state())


def test_base_rate_shrinks_to_the_market_rate_and_handles_new_players():
    rng = np.random.default_rng(1)
    rows = [{"player_id": p, "y": float(rng.random() < (0.1 if p < 20 else 0.6))}
            for p in range(40) for _ in range(30)]
    df = pd.DataFrame(rows)
    m = H.PlayerBaseRate().fit(None, df["y"][:600], None, df["y"][600:],
                               players_fit=df["player_id"][:600], players_cal=df["player_id"][600:])
    assert np.isfinite(m.prior_strength_) and m.prior_strength_ > 0
    p = m.predict(None, players=np.array([0, 39, 999]))
    assert p[0] < p[1]
    assert p[2] == pytest.approx(df["y"].mean())                 # unseen player -> market rate
    flat = pd.Series([0.0, 1.0] * 200)
    pl = pd.Series(np.repeat(np.arange(4), 100))
    m2 = H.PlayerBaseRate().fit(None, flat[:200], None, flat[200:], players_fit=pl[:200], players_cal=pl[200:])
    assert m2.prior_strength_ == float("inf")
    assert all(v == pytest.approx(0.5) for v in m2.rates_.values())


def test_window_too_small_is_refused():
    with pytest.raises(ValueError):
        H.fit_window(_frame(n_players=4, n_days=10), ["f1", "f2"], "target_sot")


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────

def test_metrics_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    m = H.metrics(y, p)
    assert m["n"] == 4 and m["events"] == 2
    assert m["brier"] == pytest.approx(np.mean([0.04, 0.04, 0.16, 0.16]))
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.8, 0.6, 0.6])))
    assert sum(r["n"] for r in m["reliability"]) == 4
    m2 = H.metrics(np.array([1, 0]), np.array([np.nan, 0.3]))
    assert m2["n"] == 1 and m2["unscored"] == 1


def test_block_ci_brackets_the_mean():
    r = H.block_ci(np.arange(100, dtype=float), np.repeat(np.arange(10), 10))
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1] and r["week_blocks"] == 10


# ── evaluation and the gated run (synthetic) ─────────────────────────────────────────────────────

def test_evaluate_market_walk_forward_and_holdout():
    f = _frame(n_days=90)
    dec = _decisions()
    folds = H.walk_forward_folds(H.matchdays(f["date"]), dec["holdout_start"], 5, 2)
    rep, scored, fitted = H.evaluate_market(f, "sot", ["f1", "f2"], dec, folds)
    assert [w["status"] for w in rep["walk_forward"]] == ["OK", "OK"]
    assert set(rep["walk_forward_pooled"]) == set(H.CANDIDATES)
    assert rep["holdout"]["status"] == "OK"
    assert rep["holdout"]["window"]["calibration_dates"][1] < dec["holdout_start"]
    assert scored["date"].between(dec["holdout_start"], dec["holdout_end"]).all()
    assert set(rep["holdout"]["log_loss_minus_base_rate"]) == {"recipe_retrained", "hgb_nan_native"}
    for n in H.CANDIDATES:
        assert rep["holdout"]["metrics"][n]["n"] == len(scored)


def _run_frame():
    f = _frame(n_days=90)
    f["player_name"] = "P " + f["player_id"].astype(str)
    f["home_team"], f["away_team"] = "Home FC", "Away FC"
    return f


def test_run_stops_under_the_owner_minimum(monkeypatch, tmp_path):
    monkeypatch.setenv(H.ENABLE_ENV, "true")
    f = _run_frame()
    f["label_verified"] = np.arange(len(f)) % 4 == 0             # 25% verified
    monkeypatch.setattr(H, "build_frame", lambda raw: f)
    rep = H.run(_plan(min_verified_share=0.9), raw=pd.DataFrame(), out_dir=tmp_path / "o",
                model_dir=tmp_path / "m", quotes=pd.DataFrame())
    assert rep["status"].startswith("STOPPED") and not (tmp_path / "m").exists()


def test_run_end_to_end_on_synthetic_rows(monkeypatch, tmp_path):
    monkeypatch.setenv(H.ENABLE_ENV, "true")
    f = _run_frame()
    unver = f.index % 10 == 0
    f.loc[unver, "label_verified"] = False                      # excluded from training
    monkeypatch.setattr(H, "build_frame", lambda raw: f)
    monkeypatch.setattr(H, "harness_features", lambda frame: ["f1", "f2"])
    rep = H.run(_plan(), raw=pd.DataFrame(), out_dir=tmp_path / "o", model_dir=tmp_path / "m",
                quotes=CLV.load_quotes(tmp_path / "none"))
    assert rep["status"] == "OK" and rep["hand_adjustment"] == "none"
    assert rep["verified_share"] == pytest.approx(1 - unver[f["date"] <= "2026-03-31"].mean())
    holdout_rows = int(((f["date"] >= "2026-03-01") & (f["date"] <= "2026-03-31") & ~unver).sum())
    assert rep["markets"]["sot"]["holdout"]["test_rows"] == holdout_rows
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == sorted(
        f"props_clean_sot_{n}.pkl" for n in H.CANDIDATES)
    assert (tmp_path / "o" / "props_harness_report.json").exists()


# ── dry run (synthetic) ──────────────────────────────────────────────────────────────────────────

def test_dry_run_counts_today_and_after_the_refetch():
    rows = []
    for p in (1, 2):
        for i in range(6):
            src = PARSED_WITH_RULE if p == 1 else UNVERIFIED
            rows.append({"date": f"2026-0{1 + i // 3}-{10 + i:02d}", "fixture_id": 10 * p + i, "player_id": p,
                         "minutes": 90, "goals": i % 2, "assists": 0, "shots_on_target": 1, "yellow_cards": 0,
                         SOURCE_COL: src, "harness_source": "history"})
    raw = _raw(rows)
    rep = H.dry_run(raw, "2026-02-01", "2026-02-28", ["sot", "goals"], {"step_matchdays": 1, "n_folds": 2},
                    proven_fixtures={20, 21, 22, 23, 24, 25}, profiles=pd.DataFrame())
    assert rep["data"]["population_rows"] == 10
    lo = rep["verified_today"]["label_only"]["sot"]
    assert lo == {"pre_holdout": 2, "holdout": 3, "after_holdout": 0}
    up = rep["verified_after_refetch_upper_bound"]["label_only"]["sot"]
    assert up == {"pre_holdout": 4, "holdout": 6, "after_holdout": 0}
    assert rep["verified_share"]["label_only"]["today"] == pytest.approx(0.5)
    assert rep["verified_share"]["label_only"]["after_refetch_upper_bound"] == pytest.approx(1.0)
    assert rep["counts_all_rows"]["goals"]["holdout"] == 6
    assert rep["walk_forward"]["n_folds"] == 1                 # the first matchday has nothing before it
    assert "nothing fitted" in rep["mode"]


# ── CLV (synthetic quotes) ───────────────────────────────────────────────────────────────────────

def _q(ts, player, market, side, odds, book="Pinnacle", match="Home FC vs Away FC", day="2026-03-10", point=None):
    return {"snapshot_ts": ts, "match_date": day, "league": "L", "match": match, "player": player,
            "market": market, "side": side, "point": point, "bookmaker": book, "odds": odds, "source": "oddsapi"}


def _quotes():
    return pd.DataFrame([
        _q("2026-03-08T10:00:00Z", "victor torp", "goals", "yes", 4.0),
        _q("2026-03-09T10:00:00Z", "victor torp", "goals", "yes", 3.0),
        _q("2026-03-09T10:00:00Z", "victor torp", "goals", "no", 1.4),
        _q("2026-03-08T10:00:00Z", "ellis simms", "goals", "yes", 2.5),
        _q("2026-03-11T10:00:00Z", "ellis simms", "goals", "no", 1.2),     # after the match day: ignored
    ])


def test_clv_uses_the_two_sided_close_and_flags_the_assumed_margin():
    from src.market_edge import power_devig
    from player_model.predict import _devig_fair_prob
    scored = pd.DataFrame([
        {"market": "goals", "date": "2026-03-10", "player_name": "V. Torp", "home_team": "Home FC",
         "away_team": "Away FC", "p": 0.4, "y": 1.0},
        {"market": "goals", "date": "2026-03-10", "player_name": "Ellis Simms", "home_team": "Home FC",
         "away_team": "Away FC", "p": 0.3, "y": 0.0},
        {"market": "goals", "date": "2026-03-10", "player_name": "Nobody Here", "home_team": "Home FC",
         "away_team": "Away FC", "p": 0.9, "y": 0.0},
    ])
    rows = CLV.clv_rows(scored, _quotes(), "p").set_index("player_name")
    assert list(rows.index) == ["V. Torp", "Ellis Simms"]
    t = rows.loc["V. Torp"]
    assert t["fair_basis"] == "two_sided_devig" and not t["assumed_margin"]
    assert t["price_open"] == 4.0
    assert t["p_fair_close"] == pytest.approx(power_devig(3.0, 1.4))
    assert t["selected"] and t["clv"] == pytest.approx(4.0 * power_devig(3.0, 1.4) - 1)
    s = rows.loc["Ellis Simms"]
    assert s["fair_basis"] == "assumed_overround" and s["assumed_margin"]
    assert s["p_fair_close"] == pytest.approx(_devig_fair_prob(2.5))
    assert not s["selected"]                                     # 0.3 x 2.5 < 1
    summ = CLV.summarise(rows.reset_index())
    assert set(summ["goals"]) == {"two_sided_devig", "assumed_overround"}
    assert summ["goals"]["assumed_overround"]["assumed_margin"] is True
    assert summ["goals"]["two_sided_devig"]["benchmark_rows"] == 1


def test_pair_counts_are_counts_only():
    pc = CLV.pair_counts(_quotes())
    assert pc["player_markets_two_sided"] == 1 and pc["player_markets_priced"] == 2
    assert pc["by_market"]["goals"] == {"priced": 2, "two_sided": 1}
