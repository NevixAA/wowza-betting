"""Bars study live adapter and monthly proposal job (src/bars_opt/live.py) on synthetic data only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.bars_opt import live as L

ROOT = Path(__file__).resolve().parents[1]


def _cands():
    base = {"prediction_engine_version": "clean_v1", "league": "L", "model_type": "standard",
            "match_date": "2026-11-21", "home_team": "H", "away_team": "A", "fixture_id": "fx1"}
    return pd.DataFrame([
        base | {"market": "OU25", "side": "UNDER", "p_model_raw": 0.6, "odds": 1.9, "opposite_side_odds": 2.0},
        base | {"market": "BTTS", "side": "YES", "p_model_raw": 0.55, "odds": 1.8, "opposite_side_odds": 2.1},
        base | {"market": "OVER15", "side": "OVER", "p_model_raw": 0.7, "odds": float("nan"), "opposite_side_odds": 3.0},
        base | {"market": "OVER35", "side": "OVER", "p_model_raw": 0.3, "odds": 3.2, "opposite_side_odds": 1.4,
                "prediction_engine_version": "old_v0"},
    ])


def test_live_rows_maps_sides_prices_and_results_and_counts_exclusions():
    res = pd.DataFrame({"fixture_id": ["fx1"], "home_goals": [2], "away_goals": [1]})
    rows, counts = L.live_rows(_cands(), res, ["clean_v1"])
    assert counts["clean_engine"] == 3 and counts["no_two_sided_price_or_p"] == 1
    ou = rows[rows["market"] == "over25"].iloc[0]
    assert (ou["yes"], ou["no"]) == (2.0, 1.9)                  # the Under row: yes = the opposite price
    assert abs(ou["p"] - 0.4) < 1e-12                           # p is the Over/Yes probability
    assert ou["y"] == 1.0                                       # 3 goals > 2.5
    bt = rows[rows["market"] == "btts"].iloc[0]
    assert (bt["yes"], bt["no"], bt["y"]) == (1.8, 2.1, 1.0)
    assert rows["yes_c"].isna().all()                           # no closing pair in this source: CLV missing


def test_live_rows_without_a_result_are_excluded_not_filled():
    rows, counts = L.live_rows(_cands(), pd.DataFrame({"fixture_id": ["other"], "home_goals": [0], "away_goals": [0]}),
                               ["clean_v1"])
    assert rows.empty and counts["no_result"] == 2


def test_no_clean_engine_listed_means_no_live_evidence():
    cfg = json.loads((ROOT / "config" / "bars_opt.json").read_text(encoding="utf-8"))
    assert cfg["clean_engine_versions"] == []                   # set by the owner at go-live
    rows, counts = L.live_rows(_cands(), pd.DataFrame({"fixture_id": ["fx1"], "home_goals": [1], "away_goals": [1]}), [])
    assert rows.empty and counts["clean_engine"] == 0


def test_production_candidates_never_push_a_league_onto_the_global_bar():
    entries = [{"month": "2026-12", "mside": "over25_over", "league": "L", "config": "c", "valuable_th": 0.02,
                "marksman_th": 0.03, "sniper_th": 0.04, "wf_roi": 0.1, "wf_bets": 50, "mode": "MODEL"},
               {"month": "2026-12", "mside": "btts_yes", "league": "L", "config": "c", "valuable_th": 0.02,
                "marksman_th": 0.03, "sniper_th": 0.04, "wf_roi": 0.1, "wf_bets": 50, "mode": "LEAGUE_WIDE",
                "tier2": False},
               {"month": "2026-12", "mside": "over15_over", "league": "M", "off": "OWN_DATA_TOO_THIN"},
               {"month": "2026-12", "mside": "over35_over", "league": "N", "off": "NO_POSITIVE_BAR"},
               {"month": "2026-12", "mside": "btts_yes", "league": "K", "config": L.FAITHFUL_CONFIG,
                "valuable_th": 0.03, "marksman_th": 0.05, "sniper_th": 0.07, "wf_roi": 0.08, "wf_bets": 60,
                "mode": "MODEL", "tier2": False},
               {"month": "2026-12", "mside": "btts_yes", "league": "J", "config": "RAW|EV|max2.5|min-",
                "valuable_th": 0.03, "marksman_th": 0.05, "sniper_th": 0.07, "wf_roi": 0.08, "wf_bets": 60,
                "mode": "MODEL", "tier2": False}]
    current = {"over15": {"M": {"sniper_th": 0.07, "marksman_th": 0.05, "approved": True}}}
    c, off = L.production_candidates(entries, current)
    assert set(c) == {"btts", "over15"}                         # O/U 2.5 needs an owner serving decision
    assert c["btts"]["K"]["approved"] is True and c["btts"]["K"]["valuable_th"] == 0.03   # faithful: own bars
    assert "L" not in c["btts"] and "J" not in c["btts"]        # LEAGUE_WIDE / EV bars: production would misread
    assert c["over15"]["M"] == current["over15"]["M"]           # OFF: current entry carried, never revoked
    assert "over35" not in c                                    # OFF and absent: nothing written (no global push)
    reasons = {(o["market"], o["league"]): o["reason"] for o in off}
    assert reasons == {("over15", "M"): "OWN_DATA_TOO_THIN", ("over35", "N"): "NO_POSITIVE_BAR",
                       ("btts", "L"): "NEEDS_BARS_OPT_SERVING", ("btts", "J"): "NEEDS_BARS_OPT_SERVING"}


def test_monthly_workflow_is_gated_and_commits_only_proposals():
    text = (ROOT / ".github" / "workflows" / "bars_monthly.yml").read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    job = wf["jobs"]["reselect"]
    assert job["if"] == "vars.BARS_MONTHLY_ENABLED == 'true'"
    assert "git add output/threshold_proposals/" in text
    assert "models/" not in text.replace("no models/", "") and "threshold_approvals.json" not in \
        text.replace("config/threshold_approvals.json and src/threshold_apply", "")
