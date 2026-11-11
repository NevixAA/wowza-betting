"""Tests for scripts/research_combined_models.py (research only, nothing served)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_combined_models as R  # noqa: E402


def test_holm_matches_textbook():
    adj = R.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_holm_nan_enters_as_one():
    adj = R.holm([0.001, np.nan])
    assert adj[1] == 1.0 and np.isclose(adj[0], 0.002)


def test_cluster_bootstrap_finds_a_real_effect_and_not_a_null():
    rng = np.random.default_rng(1)
    n_fix = 400
    fix = np.repeat(np.arange(n_fix), 10)
    grp = (fix % 2).astype(int)
    shared = np.repeat(rng.normal(0, 0.2, n_fix), 10)          # fixture-level noise
    v = shared + rng.normal(0, 0.3, len(fix)) + 0.1 * grp
    b = R.cluster_bootstrap_diff(v, grp, fix, reps=500)
    assert b["lo"] > 0 and abs(b["est"] - 0.1) < 0.06
    null = R.cluster_bootstrap_diff(shared + rng.normal(0, 0.3, len(fix)), grp, fix, reps=500)
    assert null["lo"] < 0 < null["hi"]


def test_cluster_ci_is_wider_than_row_ci_when_rows_share_a_fixture():
    rng = np.random.default_rng(2)
    fix = np.repeat(np.arange(200), 20)
    grp = (fix % 2).astype(int)
    v = np.repeat(rng.normal(0, 1, 200), 20)
    clus = R.cluster_bootstrap_diff(v, grp, fix, reps=400)
    rows = R.cluster_bootstrap_diff(v, grp, np.arange(len(v)), reps=400)
    assert (clus["hi"] - clus["lo"]) > 2 * (rows["hi"] - rows["lo"])


def test_prior_cutoff_never_uses_the_rows_own_month():
    df = pd.DataFrame({"league": ["A"] * 6, "month": ["2025-01"] * 3 + ["2025-02"] * 3,
                       "v": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]})
    cut = R.prior_cutoff(df["v"], df["league"], df["month"], 0.5, 3)
    assert np.isnan(cut[:3]).all()                    # no earlier month
    assert np.allclose(cut[3:], 2.0)                  # median of January only


def test_prior_cutoff_is_per_league():
    df = pd.DataFrame({"league": ["A", "A", "B", "B"], "month": ["1", "2", "1", "2"], "v": [1.0, 0, 50.0, 0]})
    cut = R.prior_cutoff(df["v"], df["league"], df["month"], 0.5, 1)
    assert cut[1] == 1.0 and cut[3] == 50.0


def test_norm_name_drops_generic_tokens_and_accents():
    assert R.norm_name("FC Nordsjælland") == R.norm_name("Nordsjaelland")
    assert R.norm_name("Atlético-MG") == "atletico mg"


def _fx(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"]) \
        .assign(date=lambda d: pd.to_datetime(d["date"]))


def test_crosswalk_learns_a_renamed_team_from_scores_and_refuses_thin_evidence():
    v9 = _fx([["L", f"2025-01-0{i}", "Alpha", "Beta Town", i % 3, 1] for i in range(1, 6)]
             + [["L", "2025-02-01", "Gamma", "Alpha", 0, 0]])
    af = _fx([["L", f"2025-01-0{i}", "Alpha", "Beta T.", i % 3, 1] for i in range(1, 6)]
             + [["L", "2025-02-01", "Gamma Utd", "Alpha", 0, 0]])
    af["fixture_id"] = np.arange(len(af))
    xw = R.learn_crosswalk(v9, af)
    m = dict(zip(xw["v9_name"], xw["af_name"]))
    assert m["Beta Town"] == "Beta T."
    assert "Gamma" not in m                           # one vote only: refused, never guessed


def test_map_fixtures_drops_a_score_mismatch():
    v9 = _fx([["L", "2025-01-01", "A", "B", 2, 1], ["L", "2025-01-08", "B", "A", 0, 0]])
    af = _fx([["L", "2025-01-01", "a", "b", 2, 1], ["L", "2025-01-09", "b", "a", 1, 0]])
    af["fixture_id"] = [10, 11]
    xw = pd.DataFrame({"league": ["L", "L"], "v9_name": ["A", "B"], "af_name": ["a", "b"], "votes": 3, "share": 1.0})
    out = R.map_fixtures(v9, af, xw)
    assert out["map_status"].tolist() == ["ok", "score_mismatch"]
    assert out["fixture_id"].iloc[0] == 10 and np.isnan(out["fixture_id"].iloc[1])


def test_map_fixtures_missing_score_is_not_a_mismatch_and_plus_minus_one_day_is_used():
    v9 = _fx([["L", "2025-01-08", "B", "A", 0, 0]])
    af = _fx([["L", "2025-01-09", "b", "a", np.nan, np.nan]])
    af["fixture_id"] = [11]
    xw = pd.DataFrame({"league": ["L", "L"], "v9_name": ["A", "B"], "af_name": ["a", "b"], "votes": 3, "share": 1.0})
    assert R.map_fixtures(v9, af, xw)["map_status"].tolist() == ["ok"]


def test_poisson_total_recovers_the_mean():
    from scipy.stats import poisson
    lam = 2.7
    got = R.poisson_total_from_over(*(poisson.sf(k, lam) for k in (1, 2, 3)))
    assert abs(got - lam) < 0.02
    assert np.isnan(R.poisson_total_from_over(0.7, np.nan, 0.3))


def test_load_team_keeps_walk_forward_champion_only_and_the_earlier_fold(tmp_path):
    base = dict(track="new_format", league="L", deployable=True)
    rows = []
    for stage, fold, cand, p in [("walk_forward", 1, "champion_recipe", 0.6), ("walk_forward", 2, "champion_recipe", 0.9),
                                 ("walk_forward", 1, "hgb", 0.1), ("holdout", -1, "champion_recipe", 0.2)]:
        for mk in R.TEAM_MARKETS:
            rows.append(dict(base, stage=stage, fold=fold, candidate=cand, market=mk, row_id="2025-01-01|L|A|B",
                             date=pd.Timestamp("2025-01-01"), y=1.0, p=p))
    pq = tmp_path / "p.parquet"
    pd.DataFrame(rows).to_parquet(pq)
    t = R.load_team([pq])
    assert len(t) == 1 and t["p_over25"].iloc[0] == 0.6 and t["home_team"].iloc[0] == "A"


def test_load_team_refuses_holdout_dates(tmp_path):
    rows = [dict(stage="walk_forward", fold=0, candidate="champion_recipe", market=mk, track="t", league="L",
                 row_id="2026-04-02|L|A|B", date=pd.Timestamp("2026-04-02"), y=1.0, p=0.5, deployable=True)
            for mk in R.TEAM_MARKETS]
    pq = tmp_path / "p.parquet"
    pd.DataFrame(rows).to_parquet(pq)
    with pytest.raises(ValueError):
        R.load_team([pq])


def test_holm_family_counts_unreadable_cells():
    j = pd.DataFrame({"market": [], "league": []})
    # the family rule itself: an unreadable cell enters with p = 1 and still raises m
    p = np.array([0.001, np.nan, np.nan, np.nan])
    adj = R.holm(np.where(np.isfinite(p), p, 1.0))
    assert np.isclose(adj[0], 0.004)
    assert j.empty


def test_protocol_is_registered_and_names_one_family():
    proto = json.loads((ROOT / "config" / "experiments" / "combined_models" / "protocol.json").read_text(encoding="utf-8"))
    assert proto["protocol"] == "COMBINED-MODELS-DEV-1"
    assert "ONE family" in proto["holm_family"]
    assert "2026-04-01" in proto["data"]["development_months"]


def test_dev_window_ends_before_both_spent_windows():
    assert R.TEAM_HOLDOUT_START <= R.PROPS_SPENT_START
    assert R.DEV_START < R.TEAM_HOLDOUT_START


def test_crosswalk_accepts_two_season_separated_af_spellings_but_not_overlapping_ones():
    rows_v, rows_a = [], []
    for i in range(1, 5):
        rows_v.append(["L", f"2023-0{i}-01", "Nurnberg", "Other", 1, 0]); rows_a.append(["L", f"2023-0{i}-01", "FC Nurnberg", "Other", 1, 0])
        rows_v.append(["L", f"2024-0{i}-01", "Nurnberg", "Other", 2, 0]); rows_a.append(["L", f"2024-0{i}-01", "1. FC Nürnberg x", "Other", 2, 0])
    v9, af = _fx(rows_v), _fx(rows_a)
    af["fixture_id"] = np.arange(len(af))
    xw = R.learn_crosswalk(v9, af)
    got = sorted(xw.loc[xw["v9_name"] == "Nurnberg", "af_name"])
    assert got == ["1. FC Nürnberg x", "FC Nurnberg"]
    mp = R.map_fixtures(v9, af, xw)
    assert (mp["map_status"] == "ok").all()


def test_gk_score_defect_is_kept_only_when_player_goals_fit_the_real_score():
    v9 = _fx([["L", "2025-01-01", "A", "B", 2, 1], ["L", "2025-01-08", "B", "A", 0, 0]])
    af = _fx([["L", "2025-01-01", "a", "b", 0, 1], ["L", "2025-01-08", "b", "a", 0, 0]])
    af["fixture_id"] = [10, 11]
    af["home_player_goals"] = [2, 1]          # second: a player goal the real 0-0 cannot hold
    af["away_player_goals"] = [1, 0]
    xw = pd.DataFrame({"league": ["L", "L"], "v9_name": ["A", "B"], "af_name": ["a", "b"], "votes": 3, "share": 1.0})
    assert R.map_fixtures(v9, af, xw)["map_status"].tolist() == ["ok_gk_score_defect", "score_mismatch"]


def test_p_value_can_pass_a_large_holm_family_when_the_effect_is_strong():
    rng = np.random.default_rng(3)
    fix = np.repeat(np.arange(2000), 5)
    grp = (fix % 2).astype(int)
    v = rng.normal(0, 0.3, len(fix)) + 0.1 * grp
    b = R.cluster_bootstrap_diff(v, grp, fix, reps=500)
    assert b["p_two"] * 340 < 0.05                    # the empirical floor 1/501 could never do this
    assert b["p_emp_one"] >= 1 / 501


def test_logistic_slope_recovers_a_planted_signal_and_ignores_noise():
    rng = np.random.default_rng(4)
    n = 20000
    df = pd.DataFrame({"league": rng.choice(["A", "B"], n), "fixture_id": rng.integers(0, 2000, n),
                       "p_props": rng.uniform(0.1, 0.5, n), "c": rng.normal(0, 1, n)})
    lo = np.log(df.p_props / (1 - df.p_props)) + 0.3 * df.c
    df["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-lo))).astype(float)
    r = R.logistic_slope(df, "c")
    assert abs(r["b"] - 0.3) < 0.06 and r["p_two"] < 1e-6
    df["noise"] = rng.normal(0, 1, n)
    assert R.logistic_slope(df, "noise")["p_two"] > 0.001
