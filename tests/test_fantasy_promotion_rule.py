"""Fantasy V2 promotion rule (owner decision 2026-11-23) and the V1-free shadow arm.

Pins:
  * the committed rule config (owner's words and date, 10 completed gameweeks, 90% gameweek-block
    interval of MAE - FPL entirely below 0, captain regret no worse than FPL, never auto-promote);
  * the evaluator on fixtures: eligible, too few gameweeks, interval touching 0, worse captaincy,
    paired population, missing arm;
  * that eligibility can never flip FANTASY_V2_RESIDUAL_MODE or the served headline by itself;
  * that the HGB_V1FREE shadow arm uses no Wowza V1 input and covers every row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import fantasy_paired_eval as pe
from player_model import fantasy_promotion_rule as pr
from player_model import fantasy_residual as R

ROOT = Path(__file__).resolve().parents[1]
RULE = pr.load_rule()


def _oos(n_gw=10, n=60, seed=3, cand_noise=0.3, fpl_noise=1.5, cand_picks_best=True,
         cand_equals_fpl=False):
    """Synthetic walk-forward frame: gameweeks 2..n_gw+1, each trained on the earlier ones."""
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(2, n_gw + 2):
        y = rng.uniform(0, 10, n)
        y[0] = 20.0                                       # player 0 is the gameweek's best scorer
        fpl = y + rng.normal(0, fpl_noise, n)
        fpl[0] = 1.0 if cand_picks_best else 30.0         # who captains the best scorer
        cand = fpl.copy() if cand_equals_fpl else y + rng.normal(0, cand_noise, n)
        if not cand_equals_fpl:
            cand[0] = 30.0 if cand_picks_best else 0.0    # the candidate's captain choice
        rows.append(pd.DataFrame({
            "gw": gw, "element": np.arange(n), "status": "FINAL", "player_name": [f"p{i}" for i in range(n)],
            "actual_points": y, "ep_next": fpl, "LINEAR": cand, "HGB_V1FREE": cand,
            "train_gws": ",".join(str(g) for g in range(1, gw))}))
    return pd.concat(rows, ignore_index=True)


# ── the committed rule ──────────────────────────────────────────────────────────────────────
def test_rule_config_is_pre_registered_as_approved():
    assert RULE["owner_decision"]["date"] == "2026-11-23"
    assert RULE["owner_decision"]["words"] == "approve all"
    assert RULE["baseline_arm"] == "FPL"
    assert set(RULE["candidate_arms"]) == {"LINEAR", "HGB_V1FREE"}
    c = RULE["conditions"]
    assert c["min_completed_gameweeks"] == 10
    assert c["mae_diff_interval"]["interval_level"] == 0.90
    assert "gameweek-block" in c["mae_diff_interval"]["resampling"]
    assert c["mae_diff_interval"]["pass_if"].startswith("upper bound < 0")
    assert c["captain_regret"]["pass_if"] == "candidate total <= FPL total"
    assert RULE["outcome"]["auto_promote"] is False


# ── evaluator on fixtures ───────────────────────────────────────────────────────────────────
def test_clear_winner_over_ten_gameweeks_is_eligible_not_promoted():
    res = pr.evaluate(_oos())
    for arm in ("LINEAR", "HGB_V1FREE"):
        r = res["arms"][arm]
        assert r["status"] == pr.ELIGIBLE and r["eligible"] is True
        assert r["auto_promote"] is False and r["reasons"] == []
        assert r["checks"]["completed_gameweeks"]["value"] == 10
        assert r["checks"]["mae_diff_ci90_below_zero"]["ci90"][1] < 0
        assert r["checks"]["captain_regret_no_worse"]["candidate"] == 0
    assert res["auto_promote"] is False


def test_nine_gameweeks_is_not_enough():
    r = pr.evaluate(_oos(n_gw=9))["arms"]["LINEAR"]
    assert r["status"] == pr.NOT_ELIGIBLE
    assert r["checks"]["completed_gameweeks"]["pass"] is False
    assert r["checks"]["mae_diff_ci90_below_zero"]["pass"] is True     # only the count fails
    assert any("9 of 10" in s for s in r["reasons"])


def test_interval_touching_zero_is_not_eligible():
    r = pr.evaluate(_oos(cand_equals_fpl=True))["arms"]["LINEAR"]
    assert r["checks"]["mae_diff_ci90_below_zero"]["ci90"] == [0.0, 0.0]
    assert r["checks"]["mae_diff_ci90_below_zero"]["pass"] is False
    assert r["status"] == pr.NOT_ELIGIBLE


def test_worse_captaincy_blocks_even_a_better_mae():
    r = pr.evaluate(_oos(cand_picks_best=False))["arms"]["LINEAR"]
    ch = r["checks"]
    assert ch["mae_diff_ci90_below_zero"]["pass"] is True
    assert ch["captain_regret_no_worse"]["candidate"] > ch["captain_regret_no_worse"]["baseline"]
    assert r["status"] == pr.NOT_ELIGIBLE


def test_unfinished_gameweeks_do_not_count():
    o = _oos()
    o.loc[o["gw"] == o["gw"].max(), "status"] = "PROVISIONAL"
    r = pr.evaluate(o)["arms"]["LINEAR"]
    assert r["checks"]["completed_gameweeks"]["value"] == 9 and r["status"] == pr.NOT_ELIGIBLE


def test_both_arms_are_scored_on_the_same_rows():
    o = _oos()
    o.loc[o["element"] % 3 == 1, "LINEAR"] = np.nan             # LINEAR misses a third of the rows
    res = pr.evaluate(o)["arms"]
    assert res["LINEAR"]["rows"] == int(o["LINEAR"].notna().sum())
    assert res["HGB_V1FREE"]["rows"] == len(o)
    assert res["LINEAR"]["exclusions"]["dropped_any_arm_missing"] == int(o["LINEAR"].isna().sum())


def test_chronology_leak_is_excluded_from_the_rule():
    o = _oos()
    last = o["gw"].max()
    o.loc[o["gw"] == last, "train_gws"] = f"1,{last}"
    r = pr.evaluate(o)["arms"]["LINEAR"]
    assert last not in r["gameweeks"] and r["checks"]["completed_gameweeks"]["value"] == 9


def test_missing_arm_is_no_data_not_eligible():
    r = pr.evaluate(_oos().drop(columns=["HGB_V1FREE"]))["arms"]["HGB_V1FREE"]
    assert r["status"] == pr.NO_DATA and r["eligible"] is False


# ── it can never flip the mode by itself ────────────────────────────────────────────────────
def test_eligibility_never_flips_the_residual_mode(monkeypatch):
    from player_model import fantasy_v2 as v2
    monkeypatch.delenv("FANTASY_V2_RESIDUAL_MODE", raising=False)
    res = pe.evaluate(_oos())                                    # rule attached to the paired report
    assert res["promotion_rule"]["arms"]["LINEAR"]["status"] == pr.ELIGIBLE
    assert v2.residual_mode() == "shadow"                         # unchanged after an ELIGIBLE result
    import os
    assert "FANTASY_V2_RESIDUAL_MODE" not in os.environ


def test_rule_code_has_no_path_to_the_mode_or_served_files():
    src = (ROOT / "player_model" / "fantasy_promotion_rule.py").read_text(encoding="utf-8")
    for banned in ("os.environ", "putenv", "residual_mode(", "write_text", "to_csv", "GITHUB_ENV",
                   ".github/workflows"):
        assert banned not in src, banned
    # the served decision reads only the env var and its own gate, never this rule
    v2src = (ROOT / "player_model" / "fantasy_v2.py").read_text(encoding="utf-8")
    assert "fantasy_promotion_rule" not in v2src
    wf = (ROOT / ".github" / "workflows" / "fantasy_refresh.yml").read_text(encoding="utf-8")
    assert "FANTASY_V2_RESIDUAL_MODE: shadow" in wf
    assert "FANTASY_V2_RESIDUAL_MODE: auto" not in wf


# ── the V1-free shadow arm ──────────────────────────────────────────────────────────────────
def test_v1free_arm_has_no_wowza_v1_input():
    assert not set(R.V1_FEATURES) & set(R.V1FREE_FEATURES)
    assert set(R.V1FREE_FEATURES) | set(R.V1_FEATURES) == set(R.FEATURES)
    assert R.make("HGB_V1FREE").cols == R.V1FREE_FEATURES
    assert "HGB_V1FREE" not in R.RESIDUAL_ARMS                    # never in the acceptance gate


def test_v1free_arm_covers_every_row_and_ignores_v1():
    from tests.test_fantasy_paired_walkforward import _panel
    panel = _panel(n_gw=4, n=200, seed=5)
    a = R.walk_forward(panel)
    assert a["HGB_V1FREE"].notna().all()
    moved = panel.copy()
    moved["wowza_uncond"] = moved["wowza_uncond"] * 3 + 1.0      # rewrite every V1 projection
    moved["wowza_p_start"] = 0.1
    b = R.walk_forward(moved)
    np.testing.assert_allclose(a["HGB_V1FREE"], b["HGB_V1FREE"])


# ── dashboard ───────────────────────────────────────────────────────────────────────────────
def test_promotion_table_shows_status_per_arm():
    from dashboard_data import fantasy as fdata
    t = fdata.v2_promotion_table(pe.evaluate(_oos(n_gw=9)))
    assert list(t["arm"]) == ["LINEAR", "HGB_V1FREE"]
    assert set(t["status"]) == {"Not eligible"}
    assert (t["completed GWs"] == "9 of 10").all()
    assert fdata.v2_promotion_table({}).empty


@pytest.mark.skipif(not (ROOT / "output" / "fantasy_v2_paired_walkforward.json").exists(),
                    reason="no committed paired report")
def test_committed_report_carries_the_rule_status():
    j = json.loads((ROOT / "output" / "fantasy_v2_paired_walkforward.json").read_text(encoding="utf-8"))
    st = j.get("promotion_rule")
    assert st and st["auto_promote"] is False
    assert set(st["arms"]) == set(RULE["candidate_arms"])
    for r in st["arms"].values():
        assert r["status"] in {pr.ELIGIBLE, pr.NOT_ELIGIBLE, pr.NO_DATA}
