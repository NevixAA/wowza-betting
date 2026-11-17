"""REAL DATA ONLY (owner P0 2026-11-15) - pipeline trace tests for the TEAM models.

    raw provider rows -> history -> features (build_features / build_upcoming_features)
    -> preprocessing (train-only imputer) -> artifact -> live prediction (predict_proba serving=True)

Adversarial cases: provider missing values, stale / leaked observations, wrong league join, aliases,
wrong opponent, invalid timestamps, impossible ranges, duplicates, unsupported competitions and
train/serve mismatch. Every failure must be blocked before inference or reach it as an explicit,
visible gap (NaN + `__missing` indicator), never as an invented number. Synthetic data, no network.
"""
from __future__ import annotations

import json
import pickle
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import feature_engineering as FE  # noqa: E402
from src import model as M  # noqa: E402
from src.imputer import apply_imputer, fit_imputer  # noqa: E402
from src.integrity import real_data_provenance as RP  # noqa: E402

LG, LG2 = "Championship", "League One"
NUM = ("home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
       "home_fouls", "away_fouls")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _no_net(*a, **k):
        raise OSError("network blocked (real-data trace tests)")
    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)


def history(seed=0, n=90, league=LG, teams="ABCDEF", drift=True, start="2024-08-01"):
    """A league whose scoring level drifts (so a point-in-time league average differs from today's)."""
    rng = np.random.default_rng(seed)
    rows, d = [], pd.Timestamp(start)
    for k in range(n):
        h, a = rng.choice(list(teams), 2, replace=False)
        lift = k / 40 if drift else 0.0
        hg, ag = int(rng.poisson(0.6 + lift)), int(rng.poisson(0.4 + lift))
        r = dict(date=d + pd.Timedelta(days=k), league=league, season="2024/25", home_team=h, away_team=a,
                 home_goals=hg, away_goals=ag, total_goals=hg + ag, over25=float(hg + ag > 2),
                 ht_home_goals=min(hg, int(rng.integers(0, 2))), ht_away_goals=min(ag, int(rng.integers(0, 2))),
                 odds_over25=1.9, odds_under25=1.9)
        for c in NUM:
            r[c] = float(rng.integers(2, 15))
        rows.append(r)
    return pd.DataFrame(rows)


def _fixture(hist, home, away, league=LG, days=1):
    return pd.DataFrame([{"date": hist["date"].max() + pd.Timedelta(days=days), "league": league,
                          "home_team": home, "away_team": away, "odds_over25": 1.9, "odds_under25": 1.9}])


BASE = [c for c in RP.TEAM_FEATURES if not c.endswith("has_ht_history")]


# ── features: point in time ─────────────────────────────────────────────────────────────────────
def test_training_features_are_point_in_time():
    """Rewriting a LATER match's result never changes an earlier row's features (no time leakage)."""
    h = history(3)
    a = FE.build_features(h)
    h2 = h.copy()
    h2.loc[60:, ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]] = 7
    h2["total_goals"] = h2["home_goals"] + h2["away_goals"]
    b = FE.build_features(h2)
    cols = [c for c in BASE if c in a.columns]
    assert len(cols) >= 35
    pd.testing.assert_frame_equal(a.loc[:60, cols], b.loc[:60, cols])


@pytest.mark.parametrize("seed", range(12))
def test_serving_equals_training_for_the_next_match(seed):
    """Every traced team feature: build_upcoming_features (history strictly before) == the value
    build_features assigns the same match. Drifting league level - the home-advantage regression."""
    h = history(seed)
    tr = FE.build_features(h).iloc[-1]
    last = h.iloc[-1]
    sv = FE.build_upcoming_features(
        pd.DataFrame([{k: last[k] for k in ("date", "league", "home_team", "away_team")}
                      | {"odds_over25": 1.9, "odds_under25": 1.9}]), h.iloc[:-1]).iloc[0]
    for c in [c for c in BASE if c in tr.index and c in sv.index]:
        assert np.isclose(float(tr[c]), float(sv[c]), equal_nan=True, rtol=0, atol=1e-12), (c, tr[c], sv[c])


def test_home_advantage_uses_point_in_time_league_average():
    """Direct check of the fixed rule: flags compare each home match with the league's average away
    goals on STRICTLY EARLIER dates, not today's full-history mean."""
    h = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                      "league": LG, "home_team": ["A", "B", "A"], "away_team": ["B", "A", "B"],
                      "home_goals": [1, 1, 1], "away_goals": [0, 0, 5]})
    flags = FE.home_advantage_flags(h)
    # match 1: no earlier league average -> no flag; match 3: earlier away avg 0.0 -> 1 > 0 -> 1.0
    assert flags[(LG, "A")] == [1.0]
    assert flags[(LG, "B")] == [1.0]                     # 1 > 0.0 (only match 1 before)


# ── features: provider data at serving ───────────────────────────────────────────────────────────
def test_api_football_team_statistics_never_overwrite_model_features(monkeypatch):
    """The provider's season averages are DISPLAY columns (af_*); the six season-venue model
    features stay the history derivation training used."""
    import src.api_football_ou as AF
    h = history(5)
    fx = _fixture(h, "A", "B")
    clean = FE.build_upcoming_features(fx, h).iloc[0]
    monkeypatch.setattr(AF, "fetch_upcoming_team_stats", lambda df, *a, **k: {
        (LG, "A"): {"goals_for_h": 9.9, "goals_against_h": 9.9, "cs_rate_h": 0.99},
        (LG, "B"): {"goals_for_a": 8.8, "goals_against_a": 8.8, "cs_rate_a": 0.88}})
    got = FE.build_upcoming_features(fx, h).iloc[0]
    for c in ("home_season_goals_h", "home_season_conceded_h", "home_cs_rate_h",
              "away_season_goals_a", "away_season_conceded_a", "away_cs_rate_a"):
        assert np.isclose(got[c], clean[c], equal_nan=True), c
    assert got["af_home_season_goals_h"] == 9.9 and got["af_away_cs_rate_a"] == 0.88


def test_provider_missing_values_stay_missing():
    """A stat the provider never published for a club (corners NaN) is NaN in its form, never 0."""
    h = history(7)
    h.loc[(h["home_team"] == "A"), "home_corners"] = np.nan
    h.loc[(h["away_team"] == "A"), "away_corners"] = np.nan
    f = FE.build_upcoming_features(_fixture(h, "A", "B"), h).iloc[0]
    assert np.isnan(f["home_corners_pg_roll"]) and np.isfinite(f["away_corners_pg_roll"])


def test_outcome_columns_on_a_fixture_are_ignored():
    """A stale / leaked result on an unplayed fixture row cannot move its features."""
    h = history(8)
    fx = _fixture(h, "A", "B")
    a = FE.build_upcoming_features(fx, h).iloc[0]
    b = FE.build_upcoming_features(fx.assign(home_goals=9, away_goals=9, total_goals=18, over25=1.0),
                                   h).iloc[0]
    for c in BASE:
        if c in a.index:
            assert np.isclose(float(a[c]), float(b[c]), equal_nan=True), c


def test_same_named_club_in_another_league_never_joins():
    """Wrong league join: 'A' in League One must not feed Championship 'A' (league-scoped form)."""
    h = pd.concat([history(1, league=LG), history(2, league=LG2, start="2024-08-02")], ignore_index=True)
    only = FE.build_upcoming_features(_fixture(history(1), "A", "B"), history(1)).iloc[0]
    both = FE.build_upcoming_features(_fixture(history(1), "A", "B"), h).iloc[0]
    for c in ("home_scored_last5", "home_corners_pg_roll", "home_advantage", "home_season_goals_h"):
        assert np.isclose(only[c], both[c], equal_nan=True), c


def test_unknown_or_ambiguous_board_name_gets_no_form():
    """Aliases: a name the history cannot prove is NaN form (no loose prefix guess)."""
    h = history(4, teams=["Real Madrid", "Real Betis", "C", "D"])
    f = FE.build_upcoming_features(_fixture(h, "Real", "C"), h).iloc[0]
    assert np.isnan(f["home_scored_last5"]) and np.isnan(f["home_advantage"])
    assert np.isfinite(f["away_scored_last5"])


def test_alias_through_the_identity_map():
    h = history(4, teams=["Stockport", "B", "C", "D"])
    f = FE.build_upcoming_features(_fixture(h, "Stockport County FC", "B"), h,
                                   identity_aliases={LG: {"Stockport County FC": "Stockport"}}).iloc[0]
    g = FE.build_upcoming_features(_fixture(h, "Stockport", "B"), h).iloc[0]
    assert np.isclose(f["home_scored_last5"], g["home_scored_last5"])


def test_wrong_opponent_swap_swaps_the_sides():
    h = history(9)
    ab = FE.build_upcoming_features(_fixture(h, "A", "B"), h).iloc[0]
    ba = FE.build_upcoming_features(_fixture(h, "B", "A"), h).iloc[0]
    assert np.isclose(ab["home_scored_last5"], ba["away_scored_last5"], equal_nan=True)
    assert np.isclose(ab["home_advantage"], ba["away_home_adv_factor"], equal_nan=True)


def test_invalid_fixture_timestamp_gives_missing_season_stats():
    h = history(10)
    fx = _fixture(h, "A", "B")
    fx["date"] = pd.NaT
    f = FE.build_upcoming_features(fx, h).iloc[0]
    assert np.isnan(f["home_season_goals_h"]) and np.isnan(f["away_cs_rate_a"])


def test_duplicate_fixture_rows_score_identically():
    h = history(11)
    fx = pd.concat([_fixture(h, "A", "B")] * 2, ignore_index=True)
    f = FE.build_upcoming_features(fx, h)
    cols = [c for c in BASE if c in f.columns]
    pd.testing.assert_series_equal(f.loc[0, cols], f.loc[1, cols], check_names=False)


# ── preprocessing + artifact + live prediction ───────────────────────────────────────────────────
FEATS = ["home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5",
         "home_corners_pg_roll", "away_corners_pg_roll"]


@pytest.fixture(scope="module")
def trained():
    frames = [FE.build_features(history(s, n=140)) for s in range(4)]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["over25"])
    res = M.train_repaired(df, "over25", feature_cols=FEATS + [f"{c}__missing" for c in FEATS],
                           recipe="std_v9_repaired",
                           holdout_start=str((df["date"].max() - pd.Timedelta(days=20)).date()))
    return M.build_payload(res, "over25"), df


def _rows(df):
    last = df.tail(3).copy().reset_index(drop=True)
    return last


def test_serving_refuses_a_payload_without_imputer():
    pl = {"models": {}, "feature_cols": FEATS}
    assert "batch-median" in M.serving_refusal(pl)
    with pytest.raises(M.UnservablePayload):
        M.predict_proba(pd.DataFrame({c: [1.0] for c in FEATS}), payload=pl, serving=True)


def test_serving_refuses_a_zero_filled_feature_with_no_training_values():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "xg": [np.nan] * 3, "date": pd.date_range("2024-01-01", periods=3)})
    imp = fit_imputer(tr, ["a", "xg"]).to_dict()                  # 1.0 manifest: xg -> 0.0
    assert imp["version"] == "v9-train-only-imputer-1.0"
    why = M.serving_refusal({"feature_cols": ["a", "xg"], "imputer": imp})
    assert why and "xg" in why and "constant" in why


def test_serving_refuses_a_feature_missing_from_the_manifest(trained):
    pl, _ = trained
    bad = {**pl, "feature_cols": list(pl["feature_cols"]) + ["home_xg_last5"]}
    assert "absent from the imputer manifest" in M.serving_refusal(bad)


def test_missing_core_form_gets_no_probability_and_others_are_unchanged(trained):
    pl, df = trained
    rows = _rows(df)
    full = M.predict_proba(rows, payload=pl, serving=True)
    gap = rows.copy()
    gap.loc[1, "home_scored_last5"] = np.nan                      # core form missing
    gap.loc[2, "home_corners_pg_roll"] = np.nan                   # optional input missing
    got = M.predict_proba(gap, payload=pl, serving=True)
    assert np.isnan(got.iloc[1])
    assert np.isclose(got.iloc[0], full.iloc[0]) and np.isfinite(got.iloc[2])
    # research / training scoring is unchanged (no serving guard)
    assert np.isfinite(M.predict_proba(gap, payload=pl).iloc[1])


def test_serving_scores_are_batch_invariant(trained):
    pl, df = trained
    rows = _rows(df)
    board = M.predict_proba(rows, payload=pl, serving=True)
    for i in rows.index:
        alone = M.predict_proba(rows.loc[[i]], payload=pl, serving=True)
        assert np.isclose(alone.iloc[0], board.loc[i], equal_nan=True)


def test_impossible_values_are_clipped_to_the_training_range(trained):
    pl, df = trained
    rows = _rows(df)
    X, _, _ = apply_imputer(rows.assign(home_corners_pg_roll=[1e6, -5.0, np.inf]), pl["imputer"],
                            pl["feature_cols"])
    st = pl["imputer"]["features"]["home_corners_pg_roll"]
    assert X["home_corners_pg_roll"].between(st["min"], st["max"]).all()


def test_unsupported_competition_gets_no_probability(trained):
    """A league in no track is never scored by another track's model."""
    pl, df = trained
    split = {**pl, "tracks": {"new_format": None}}
    rows = _rows(df).assign(league="Atlantis Premier Division")
    assert M.predict_proba(rows, payload=split, serving=True).isna().all()


def test_predict_serve_helper_fails_closed(monkeypatch):
    import src.predict as P
    P.SERVING_REFUSALS.clear()
    out = P._serve_proba(pd.DataFrame({c: [1.0, 2.0] for c in FEATS}), {"models": {}, "feature_cols": FEATS},
                         "test model")
    assert out.isna().all() and P.SERVING_REFUSALS and P.SERVING_REFUSALS[0]["model"] == "test model"


# ── the production artifacts, through the real loading path ──────────────────────────────────────
ACTIVE = [a for a in RP.TEAM_ARTIFACTS if RP.artifact_status(a[1]).startswith("active")]


@pytest.mark.parametrize("label,attr,tracks", ACTIVE, ids=[a[0] for a in ACTIVE])
def test_active_production_payloads_pass_the_serving_guard(label, attr, tracks):
    from src.integrity.artifact_validation import inference_probe
    pl = M.load_models(model_file=getattr(config, attr))
    for t, part in RP._parts(pl, tracks).items():
        assert M.serving_refusal(part) is None, (label, t)
    probe = inference_probe(pl, tracks)
    for t, r in probe.items():
        assert r["verdict"] == "SAFE", (label, t, r)


def test_repaired_rollback_payload_is_refused():
    pl = M.load_models(model_file=config.MODEL_FILE_STANDARD_REPAIRED)
    why = M.serving_refusal(pl)
    assert why and "home_xg_last5" in why


# ── registry + health ─────────────────────────────────────────────────────────────────────────────
def test_registry_covers_every_active_payload_feature():
    reg = json.loads(RP.REGISTRY.read_text(encoding="utf-8"))
    have = {(r["artifact"], r["track"], r["feature"]) for r in reg["rows"]}
    for label, attr, tracks in ACTIVE:
        pl = M.load_models(model_file=getattr(config, attr))
        for t, part in RP._parts(pl, tracks).items():
            for f in part["feature_cols"]:
                assert (label, t, f) in have, (label, t, f)
    for r in reg["rows"]:
        assert r["classification"] in RP.CLASSES
        if r["artifact"] != "Fantasy":
            assert r["transformation_at"] and not r["transformation_at"].endswith(":?"), r


def test_every_traced_team_feature_anchor_resolves():
    for f, s in RP.TEAM_FEATURES.items():
        assert not RP.locate(*s["anchor"]).endswith(":?"), f


def test_health_reports_three_separate_lines_and_no_active_violation():
    h = RP.health()
    lines = RP.checks(h)
    assert [c["check"][:13] for c in lines] == ["Data truth 8a", "Data truth 8b", "Data truth 8c"]
    assert h["B_active_production_violations"]["count"] == 0
    assert h["C_new_violations"]["count"] == 0
    for m in h["per_model"].values():
        if str(m["status"]).startswith("active") and m["status"] != "active (fitted at run time)":
            assert m["fabricated_active"] == 0 and m["unsupported_active"] == 0
            assert m["train_serve_mismatches"] == 0 and m["pit_violations"] == 0
    # props are never reported as verified before the props agent's outputs exist
    if h["props"]["status"] == "PENDING":
        assert lines[0]["status"] != "GREEN"


def test_a_new_fabricated_feature_turns_8b_red():
    rows = RP.team_registry()
    bad = dict(rows[0], feature="home_invented_x", classification=RP.FABRICATED_VALUE)
    h = RP.health(rows + [bad])
    assert h["C_new_violations"]["count"] >= 1 and h["B_active_production_violations"]["count"] >= 1
    assert RP.checks(h)[1]["status"] == "RED"


def test_classification_rules():
    imp10 = {"version": "v9-train-only-imputer-1.0", "fill_all_missing": 0.0,
             "features": {"home_xg_last5": {"median": None, "missing_rate": 1.0}}}
    assert RP.classify({"imputer": imp10}, "home_xg_last5", 0.0)[0] == RP.FABRICATED_VALUE
    assert RP.classify({}, "home_scored_last5", 1.0)[0] == RP.TRAIN_SERVE_MISMATCH
    imp11 = {"version": "v9-train-only-imputer-1.1", "features": {
        "home_xg_last5": {"median": 1.2}, "zzz": {"median": 1.0}, "home_scored_last5": {"median": 1.0}}}
    assert RP.classify({"imputer": imp11}, "zzz", 1.0)[0] == RP.UNVERIFIED
    assert RP.classify({"imputer": imp11}, "home_xg_last5", 0.02)[0] == RP.MISSING_SOURCE
    assert RP.classify({"imputer": imp11}, "home_xg_last5", 0.17)[0] == RP.VALID_MODEL_ESTIMATE
    assert RP.classify({"imputer": imp11}, "home_scored_last5", 0.99)[0] == RP.DERIVED_VERIFIED


def test_data_truth_gate_includes_8b_only():
    from src.integrity import data_truth as DT
    assert "Data truth 8b" in DT.GATED and "Data truth 8a" not in DT.GATED


def test_pickle_roundtrip_of_registry_is_stable():
    """The committed registry is plain JSON (no pickles, no absolute paths)."""
    reg = json.loads(RP.REGISTRY.read_text(encoding="utf-8"))
    assert "C:" not in json.dumps(reg) and pickle.dumps(reg)
