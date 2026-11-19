"""REAL DATA ONLY (owner P0 2026-11-15) - fantasy projection inputs never invent an observation.

Regression tests for the fantasy audit of 2026-11-15 (reports/REAL_DATA_FINAL_AUDIT.md):
  F1  no FPL match at all -> no board (players whose availability is unknown are never published fit)
  F2  a missing defensive-action component makes the figure missing, not 0
  F4  a fixture without an official FDR has unknown fixture-adjusted points, not x1.0
  F6  a team WITH games but no measured xG stays unknown (no league-mean fill)
  F7  a club with no fixture rows has unknown 3/5-GW points, not a run of blanks
  F8  the served residual fit trains on FINAL gameweeks only (= what the walk-forward validates)
  F11 a negative ep_next is never extrapolated into later gameweeks
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import fantasy as F  # noqa: E402
from player_model import fantasy_panel as FP  # noqa: E402
from player_model import fantasy_v2 as V2  # noqa: E402


def test_f2_missing_defensive_action_component_is_missing():
    rows = []
    for d in range(5):
        rows.append({"player_id": 1, "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=d),
                     "tackles_total": 2.0, "interceptions": 1.0, "duels_won": 3.0})
        rows.append({"player_id": 2, "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=d),
                     "tackles_total": 2.0, "interceptions": np.nan, "duels_won": 3.0})
    da = F._recent_def_actions(pd.DataFrame(rows))
    assert da[1] == 6.0
    assert 2 not in da                       # was 5.0: the unknown interceptions counted as 0


def test_f1_zero_fpl_matches_publish_nothing():
    src = inspect.getsource(F.build_fantasy_projections)
    assert "keeping all to avoid" not in src
    assert "pl = pl.iloc[0:0].copy()" in src


def test_f4_unknown_fdr_is_unknown_not_neutral():
    src = inspect.getsource(F)
    i = src.index("def _fx_mult")
    body = src[i:src.index("def _fixture_list", i)]
    assert 'return float("nan")' in body and "return 1.0" not in body


def _tfl():
    rows = []
    for k, (h, a) in enumerate([(1, 2), (2, 3), (3, 1), (1, 3)]):
        ko = pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(days=7 * k)
        rows.append({"fixture": k, "team_id": h, "kickoff": ko, "finished": True, "gf": 2, "ga": 1, "home": True})
        rows.append({"fixture": k, "team_id": a, "kickoff": ko, "finished": True, "gf": 1, "ga": 2, "home": False})
    return pd.DataFrame(rows)


def test_f6_team_with_games_but_no_xg_stays_unknown():
    snap = pd.DataFrame({"team_id": [1, 1, 2, 2, 3, 3, 4],
                         "position": ["GKP", "FWD", "GKP", "FWD", "GKP", "FWD", "GKP"],
                         "xg": [0.0, 2.0, 0.0, np.nan, 0.0, 3.0, 0.0],
                         "minutes": [270, 270, 270, 270, 270, 270, 0],
                         "xgc": [3.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0]})
    t = FP.team_strength_asof(snap, _tfl(), pd.Timestamp("2026-12-01", tz="UTC")).set_index("team_id")
    assert math.isnan(t.loc[2, "xg_for_pg"])           # games played, xG unknown -> unknown
    assert np.isfinite(t.loc[1, "xg_for_pg"]) and np.isfinite(t.loc[3, "xg_for_pg"])
    assert np.isfinite(t.loc[4, "xg_for_pg"])           # no game yet -> the prior (explicit estimate)


def test_f8_served_fit_trains_on_final_gameweeks_only():
    src = inspect.getsource(V2.project)
    assert '(prep["status"] == "FINAL")' in src and '"PROVISIONAL"' not in src


def _multi_gw_world(monkeypatch, fut):
    monkeypatch.setattr(FP, "team_fixture_long", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(FP, "load_fixtures", lambda *_a, **_k: [])
    monkeypatch.setattr(FP, "team_strength_asof", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(V2, "_future_context", lambda *_a, **_k: fut)
    monkeypatch.setattr(V2, "_team_meta", lambda: {1: {"short": "AAA"}, 2: {"short": "BBB"}})


def test_f7_f11_unknown_fixtures_and_negative_ep_are_not_extrapolated(monkeypatch):
    gws = list(range(10, 10 + V2.HORIZON))
    fut = pd.DataFrame([{"gw": g, "team_id": 1, "opp_id": 2, "home": True, "fdr": 3, "lam_for": 1.4,
                         "lam_against": 1.2, "cs_prob": 0.3} for g in gws])
    _multi_gw_world(monkeypatch, fut)
    pend = pd.DataFrame({"team_id": [1, 1, 9], "position": ["MID", "MID", "MID"], "p_play": [0.9, 0.9, 0.9],
                         "start_rate": [0.8, 0.8, 0.8], "minutes": [900, 900, 900],
                         "final_xpts": [5.0, -1.0, 5.0], "ppg": [5.0, 1.0, 5.0]})
    panel = pd.DataFrame({"status": ["PENDING"], "deadline": [pd.Timestamp("2026-11-20", tz="UTC")]})
    out = V2._multi_gw(pend, panel, 10)
    assert np.isfinite(out["xpts_5gw"].iloc[0]) and out["xpts_5gw"].iloc[0] > 5.0
    assert math.isnan(out["xpts_5gw"].iloc[1]) and math.isnan(out["xpts_3gw"].iloc[1])   # F11
    assert math.isnan(out["xpts_5gw"].iloc[2]) and "BLANK" not in out["fixture_run"].iloc[2]  # F7
