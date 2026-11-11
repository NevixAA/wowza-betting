"""Synthetic tests for scripts/research_props_matchup.py (research only, no real data read)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_props_matchup as R  # noqa: E402


def _fixtures(n=8, team=10, opp0=100):
    ko = pd.date_range("2026-09-01 18:00", periods=n, freq="7D", tz="UTC")
    return R.fixture_table(pd.DataFrame({
        "fixture_id": np.arange(1, n + 1), "league": "L", "kickoff": ko.astype(str),
        "home_id": team, "away_id": opp0 + np.arange(n)}))


def _starts(fixtures, gk_by_fixture, defs_by_fixture, team=10):
    rows = []
    for fid in fixtures["fixture_id"]:
        rows.append({"fixture_id": fid, "team_id": team, "role": "start", "player_id": gk_by_fixture[fid], "pos": "G"})
        for d in defs_by_fixture[fid]:
            rows.append({"fixture_id": fid, "team_id": team, "role": "start", "player_id": d, "pos": "D"})
    return R.team_starts(pd.DataFrame(rows), fixtures)


def test_protocol_hash_is_verified(tmp_path):
    d = R.load_protocol()
    assert d["protocol_sha256"] == R.protocol_sha({k: v for k, v in d.items() if k != "protocol_sha256"})
    bad = dict(d)
    bad["regulars"] = {**d["regulars"], "regular_share": 0.5}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        R.load_protocol(p)


def test_regulars_use_only_earlier_fixtures_and_missing_stays_missing():
    fx = _fixtures(8)
    gk = {f: 1 for f in fx["fixture_id"]}
    gk[8] = 2                                   # target day's own lineup must not count
    defs = {f: (11, 12) for f in fx["fixture_id"]}
    st = _starts(fx, gk, defs)
    tg = fx[["fixture_id", "date"]].assign(team_id=10)
    reg = R.regulars(st, tg, lookback=10, min_obs=5, share=0.6).set_index("fixture_id")
    assert not reg.loc[5, "reg_known"] and pd.isna(reg.loc[5, "reg_gk"])      # 4 earlier fixtures < 5
    assert reg.loc[6, "reg_known"] and reg.loc[6, "reg_gk"] == 1
    assert reg.loc[8, "reg_gk"] == 1 and reg.loc[8, "reg_def"] == (11, 12)
    assert reg.loc[8, "n_obs"] == 7


def test_regular_share_threshold():
    fx = _fixtures(8)
    gk = {f: (1 if f % 2 else 2) for f in fx["fixture_id"]}   # 50/50 rotation -> no regular GK
    st = _starts(fx, gk, {f: (11,) for f in fx["fixture_id"]})
    tg = fx[fx["fixture_id"] == 8][["fixture_id", "date"]].assign(team_id=10)
    reg = R.regulars(st, tg, 10, 5, 0.6).iloc[0]
    assert reg["reg_known"] and pd.isna(reg["reg_gk"]) and reg["reg_def"] == (11,)


def test_strict_lists_drop_captures_at_or_after_final_kickoff_and_keep_latest():
    fx = _fixtures(2)
    ko = fx.set_index("fixture_id")["kickoff"]
    pro = pd.DataFrame({
        "captured_at": [ko[1] - pd.Timedelta(hours=30), ko[1] - pd.Timedelta(hours=3), ko[1] + pd.Timedelta(hours=1),
                        ko[2] + pd.Timedelta(minutes=5)],
        "af_fixture_id": [1, 1, 1, 2], "fixture_ts": [ko[1]] * 3 + [ko[2] + pd.Timedelta(hours=2)],
        "team_id": [100, 100, 100, 101], "player_id": [7, 8, 9, 5],
        "type": ["Missing Fixture"] * 4, "pre_kickoff": [True, True, False, True]})
    a, obs = R.strict_lists(pro, fx)
    assert obs == {1}                        # fixture 2: Pro thought pre-kickoff, final kickoff says no
    assert set(a["player_id"]) == {8}        # only the latest pre-kickoff list


def test_exposures_gk_out_def_out_and_unobserved_is_nan():
    fx = _fixtures(8)
    st = _starts(fx, {f: 1 for f in fx["fixture_id"]}, {f: (11, 12) for f in fx["fixture_id"]})
    tg = fx[fx["fixture_id"].isin([7, 8])][["fixture_id", "date"]].assign(team_id=10)
    reg = R.regulars(st, tg, 10, 5, 0.6)
    absent = pd.DataFrame({"fixture_id": [8, 8], "team_id": [10, 10], "player_id": [1, 99]})
    e = R.exposures(tg, reg, absent, observed={8}, clock="STRICT").set_index("fixture_id")
    assert e.loc[8, "gk_out"] == 1.0 and e.loc[8, "def_out"] == 0.0
    assert np.isnan(e.loc[7, "gk_out"]) and np.isnan(e.loc[7, "def_out"])     # list not observed


def test_exposures_retro_xi_uses_the_fixture_starters():
    fx = _fixtures(8)
    gks = {f: 1 for f in fx["fixture_id"]}
    gks[8] = 2                                   # backup keeper started fixture 8
    st = _starts(fx, gks, {f: (11, 12) for f in fx["fixture_id"]})
    tg = fx[fx["fixture_id"] == 8][["fixture_id", "date"]].assign(team_id=10)
    reg = R.regulars(st, tg, 10, 5, 0.6)
    e = R.exposures(tg, reg, pd.DataFrame(columns=["fixture_id", "team_id", "player_id"]), set(),
                    "RETRO_XI", xi=st[["fixture_id", "team_id", "player_id"]]).iloc[0]
    assert e["gk_out"] == 1.0 and e["def_out"] == 0.0


def _resid_frame(effect, n_fix=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fix):
        exp = float(f % 4 == 0)
        for _ in range(10):
            p = rng.uniform(0.1, 0.5)
            y = float(rng.uniform() < min(0.99, p + effect * exp))
            rows.append({"fixture_id": f, "gk_out": exp, "y": y, "p": p})
    return pd.DataFrame(rows)


def test_residual_diff_finds_a_real_effect_and_not_a_null_one():
    pos = R.residual_diff(_resid_frame(0.12), "gk_out", 2000, 1)
    assert pos["ci_lo"] > 0 and pos["p_one_sided"] < 0.01
    null = R.residual_diff(_resid_frame(0.0, seed=3), "gk_out", 2000, 1)
    assert null["ci_lo"] < 0 < null["ci_hi"]


def test_residual_diff_with_an_empty_group_returns_nan():
    d = _resid_frame(0.0).assign(gk_out=0.0)
    r = R.residual_diff(d, "gk_out", 100, 1)
    assert np.isnan(r["diff"]) and r["n_rows_exposed"] == 0


def test_holm_matches_definition_and_skips_missing():
    p = pd.Series([0.01, 0.04, np.nan, 0.03])
    h = R.holm(p)
    assert h[0] == pytest.approx(0.03) and h[3] == pytest.approx(0.06) and h[1] == pytest.approx(0.06)
    assert np.isnan(h[2])


def test_price_must_be_before_kickoff_and_equal_to_the_ledger_price():
    fx = _fixtures(1)
    ko = fx["kickoff"].iloc[0]
    L = pd.DataFrame({"match_date": [ko.strftime("%Y-%m-%d")] * 3, "player_name": ["Ünal A.", "B", "C"],
                      "market": ["sot"] * 3, "market_odds": [2.0, 3.0, 4.0], "fixture_id": [1, 1, np.nan]})
    pr = pd.DataFrame({"snapshot_ts": [ko - pd.Timedelta(hours=5), ko + pd.Timedelta(hours=1), ko - pd.Timedelta(days=2)],
                       "snapshot_date": [(ko - pd.Timedelta(hours=5)).strftime("%Y-%m-%d"), ko.strftime("%Y-%m-%d"),
                                         (ko - pd.Timedelta(days=2)).strftime("%Y-%m-%d")],
                       "match_date": [ko.strftime("%Y-%m-%d")] * 3, "player": ["unal a", "B", "C"],
                       "market": ["sot"] * 3, "odds": [2.0, 3.0, 4.0], "source": ["api_football"] * 3})
    pr["pkey"] = pr["player"].map(R.norm_name)
    out = R.corroborated_price(L, pr, fx)
    assert out["price_real"].tolist() == [True, False, True]   # B priced only after kickoff
    pr2 = pr.assign(odds=[2.1, 3.0, 4.0])
    assert not R.corroborated_price(L, pr2, fx)["price_real"].iloc[0]   # a different number is not the bet price


def test_opponent_side_needs_a_proven_side():
    fx = _fixtures(1)
    L = pd.DataFrame({"fixture_id": [1, 1, np.nan], "team": ["H", "Elsewhere", "H"],
                      "home_team": ["H", "H", "H"], "away_team": ["A", "A", "A"]})
    o = R.opponent_team_id(L, fx)
    assert o.iloc[0] == 100 and np.isnan(o.iloc[1]) and np.isnan(o.iloc[2])


def test_project_date():
    t = pd.Timestamp("2026-11-11")
    assert R.project_date(150, 0, 150, t) == "reached"
    assert R.project_date(10, 0.0, 150, t) == "never at the current rate"
    assert R.project_date(50, 10.0, 150, t) == "2027-01-20"
