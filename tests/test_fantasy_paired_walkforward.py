"""P1 Fantasy validation (2026-11-22): every arm on identical player-gameweeks, in time order.

Pins the contract of player_model.fantasy_paired_eval and the walk-forward it scores:
  * the primary population is the SAME rows for FPL, Wowza V1, LINEAR, RIDGE and HGB;
  * a row whose training window is empty or reaches its own / a later gameweek is never scored;
  * changing a later gameweek never changes an earlier gameweek's out-of-sample forecast;
  * captaincy regret and top-10 overlap are computed as documented;
  * the residual stays shadow-only (the workflow pin and the module both leave it alone).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import fantasy_paired_eval as pe
from player_model import fantasy_panel as fp
from player_model import fantasy_residual as R

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def _panel(n_gw=5, n=240, seed=7, wowza_share=0.5):
    """Synthetic FINAL panel; Wowza V1 covers only part of each gameweek (as in production)."""
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gw + 1):
        ml = rng.uniform(0, 1, n)
        ep = rng.uniform(0, 6, n)
        y = ep + 1.5 * (ml - 0.5) + rng.normal(0, 1.0, n)
        wz = np.where(rng.uniform(0, 1, n) < wowza_share, ep + rng.normal(0, 0.5, n), np.nan)
        rows.append(pd.DataFrame({
            "gw": gw, "element": np.arange(n), "status": "FINAL", "minutes": 100.0,
            "player_name": [f"p{i}" for i in range(n)], "team": "T", "price": 6.0,
            "ep_next": ep, "actual_points": y, "actual_minutes": 90.0, "bucket": "m60",
            "min_last": ml * 90, "min_per_team_game": 45.0, "start_rate": 0.5, "started_last": 1,
            "chance_next_f": 1.0, "is_out": 0, "is_doubt": 0, "has_news": 0, "wowza_p_start": np.nan,
            "xg90": 0.2, "xa90": 0.1, "goal_pts": 5.0, "pen_first": 0, "set_piece": 0, "bps90": 15,
            "threat_pg": 10, "creativity_pg": 10, "n_fix": 1, "home": 1.0, "fdr": 3, "lam_for": 1.4,
            "lam_against": 1.3, "cs_exp": 0.3, "attack_fixture": 1.0, "ownership": 5.0,
            "net_transfers_k": 0.0, "ppm": 0.5, "pts_per_team_game": 3.0, "form": 3.0, "pts_last": 3.0,
            "wowza_uncond": wz, "in_ledger": (~np.isnan(wz)).astype(int), "position": "MID",
            **{f"pos_{p}": int(p == "MID") for p in fp.POSITIONS}}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def oos():
    return R.walk_forward(_panel())


def test_primary_population_is_identical_for_every_arm(oos):
    res = pe.evaluate(oos)
    p = res["primary"]
    ns = {arm: m["n"] for arm, m in p["pooled"].items()}
    assert set(ns) == set(pe.ARMS)
    assert len(set(ns.values())) == 1 and p["rows"] == next(iter(ns.values()))
    for g in p["by_gw"]:
        assert {g[a]["n"] for a in pe.ARMS} == {g["n"]}
    # Wowza covers ~half the rows, so the paired set is strictly smaller than the HGB/FPL set
    assert p["rows"] < res["secondary"]["rows"]
    assert p["exclusions"]["dropped_any_arm_missing"] == p["exclusions"]["missing_WOWZA"]


def test_scored_gameweeks_are_out_of_sample_and_ordered(oos):
    p = pe.evaluate(oos)["primary"]
    assert p["gameweeks"] == sorted(p["gameweeks"]) and 1 not in p["gameweeks"]
    assert p["exclusions"]["chronology_violation"] == 0


def test_chronology_violation_is_excluded_not_scored(oos):
    bad = oos.copy()
    gw = int(bad["gw"].max())
    bad.loc[bad["gw"] == gw, "train_gws"] = f"1,{gw}"           # leak: own gameweek in training
    bad.loc[bad["gw"] == gw - 1, "train_gws"] = ""               # no training window at all
    pop, excl = pe.paired_population(bad)
    assert excl["chronology_violation"] == int((bad["gw"] >= gw - 1).sum())
    assert not set(pop["gw"]) & {gw, gw - 1}


def test_future_gameweek_cannot_change_past_forecasts():
    panel = _panel(seed=11)
    base = R.walk_forward(panel)
    late = panel.copy()
    last = late["gw"].max()
    late.loc[late["gw"] == last, "actual_points"] += 25.0        # rewrite the future
    moved = R.walk_forward(late)
    for arm in R.RESIDUAL_ARMS:
        a = base.loc[base["gw"] < last, arm].to_numpy(float)
        b = moved.loc[moved["gw"] < last, arm].to_numpy(float)
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_duplicate_player_gameweek_is_refused(oos):
    with pytest.raises(ValueError):
        pe.paired_population(pd.concat([oos, oos.head(1)], ignore_index=True))


def test_provisional_rows_are_not_scored(oos):
    d = oos.copy()
    gw = int(d["gw"].max())
    d.loc[d["gw"] == gw, "status"] = "PROVISIONAL"
    res = pe.evaluate(d)
    assert gw not in res["primary"]["gameweeks"]
    assert res["primary"]["exclusions"]["not_final"] == int((d["gw"] == gw).sum())


def test_captain_regret_and_top10_on_known_values():
    g = pd.DataFrame({"element": [1, 2, 3, 4], "player_name": list("abcd"),
                      "actual_points": [2.0, 15.0, 6.0, 1.0],
                      "A": [9.0, 3.0, 9.0, 1.0]})                # tie 9.0: lower element id wins
    m = pe.gw_metrics(g, "A", k=2)
    assert m["captain_element"] == 1
    assert m["captain_regret"] == pytest.approx(13.0)          # best 15 - captain's 2
    assert m["top10_overlap"] == 1                              # top-2 forecast {1,3} vs actual {2,3}
    assert m["mae"] == pytest.approx(np.mean([7, 12, 3, 0]))
    assert m["bias"] == pytest.approx(np.mean([7, -12, 3, 0]))


def test_paired_delta_is_zero_against_itself(oos):
    pop, _ = pe.paired_population(oos)
    pop["COPY"] = pop["FPL"]
    d = pe.paired_delta(pop, "COPY", n=50)
    assert d["mae_minus_fpl"] == 0 and d["mae_ci90"] == [0.0, 0.0]


def test_module_leaves_the_residual_in_shadow():
    src = (ROOT / "player_model" / "fantasy_paired_eval.py").read_text(encoding="utf-8")
    assert "os.environ" not in src and "residual_mode(" not in src
    wf = (ROOT / ".github" / "workflows" / "fantasy_refresh.yml").read_text(encoding="utf-8")
    assert "FANTASY_V2_RESIDUAL_MODE: shadow" in wf
    assert "python -m player_model.fantasy_paired_eval" in wf
    assert "\\n " not in wf, "a literal backslash-n inside the commit loop"


# ── real committed data ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not (OUT / "fantasy_v2_oos.csv").exists(), reason="no committed OOS frame")
def test_committed_oos_is_paired_and_chronological():
    o = pd.read_csv(OUT / "fantasy_v2_oos.csv", low_memory=False)
    assert int(pe.chronology_violations(o).sum()) == 0
    res = pe.evaluate(o)
    p = res["primary"]
    assert p["rows"] > 0
    assert len({m["n"] for m in p["pooled"].values()}) == 1
    # the secondary (FPL vs HGB) is every FINAL row: both arms cover all of them
    fin = int((o["status"] == "FINAL").sum())
    assert res["secondary"]["rows"] == fin


@pytest.mark.skipif(not (OUT / "fantasy_v2_paired_walkforward.json").exists(),
                    reason="no committed paired report")
def test_committed_paired_report_is_well_formed():
    j = json.loads((OUT / "fantasy_v2_paired_walkforward.json").read_text(encoding="utf-8"))
    p = j["primary"]
    assert p["population"] == "ALL_ARMS_PAIRED" and set(p["pooled"]) == set(pe.ARMS)
    assert len({m["n"] for m in p["pooled"].values()}) == 1
    assert "shadow" in j["residual_mode_note"]


# ── dashboard: paired table and player-level missingness ────────────────────────────────────
def test_paired_table_puts_fpl_first_and_shares_one_n(oos):
    from dashboard_data import fantasy as fdata
    rep = pe.evaluate(oos)
    t = fdata.v2_paired_table(rep, "primary")
    assert list(t["arm"]) == list(fdata.PAIRED_ARMS)
    assert t["n"].nunique() == 1
    assert fdata.v2_paired_table({}, "primary").empty
    g = fdata.v2_paired_by_gw(rep, "primary")
    assert set(g["arm"]) == set(pe.ARMS) and g.groupby("gw")["n"].nunique().max() == 1


def test_missingness_names_blanks_and_never_fills():
    from dashboard_data import fantasy_view as fv
    proj = pd.DataFrame({
        "web_name": ["A", "B", "C"], "team": ["X", "Y", "Z"], "position": ["FWD", "DEF", "MID"],
        "price": [8.0, 5.0, 6.0], "final_xpts": [5.0, 3.0, 4.0], "fpl_ep_next": [5.0, 3.0, 4.0],
        "wowza_residual": [0.2, np.nan, 0.1], "p_start": [0.9, 0.8, 0.7], "exp_minutes": [80, 70, 60],
        "min_last": [90, np.nan, 45], "xg90": [0.5, 0.05, 0.2], "xa90": [0.1, 0.1, 0.2],
        "goal_prob": [0.4, 0.05, 0.2], "assist_prob": [0.1, 0.1, 0.2],
        "cs_prob": [np.nan, 0.3, np.nan], "next_fixture": ["a", "b", "c"], "fdr": [2, 3, 4],
        "xpts_5gw": [20, 15, 18], "pen_order": [1, np.nan, np.nan], "set_piece": [0, 0, 1]})
    t = fv.player_data_table(proj).set_index("web_name")
    assert t.loc["A", "missing"] == "none"                      # forward: P(CS) not applicable
    assert t.loc["B", "n_missing"] == 2 and "Wowza adj (shadow)" in t.loc["B", "missing"]
    assert t.loc["C", "missing"] == "P(CS)"
    assert pd.isna(t.loc["B", "min_last"])                      # blank stays blank
    assert t.loc["A", "set_piece_role"] == "pens #1"
    assert t.loc["B", "set_piece_role"] == "not listed by FPL"
    assert t.loc["C", "set_piece_role"] == "corners/FKs (top 2)"
    s = fv.missingness_summary(proj).set_index("Field")
    assert s.loc["P(CS)", "Missing"] == 1 and s.loc["P(CS)", "Of"] == 2


def test_v1_note_flags_the_uncertified_props_frame():
    from dashboard_data import fantasy_view as fv
    assert "not certified" in fv.V1_NOTE and "Every V1 points figure" in fv.V1_NOTE
