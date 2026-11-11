"""fix/props-feature-bugs (2026-11-11): keeper goals_conceded and the team_goals_pg_roll side.

1. API-Football sends null goals.conceded for keepers who conceded (measured on the re-fetch): the null
   rule must not read it as 0, the reader must not take a stored null-read-as-0, and a keeper value
   that contradicts the 90-minute score (only keeper, full match, FT) is missing.
2. team_goals_pg_roll is the player's OWN team's rolling goals (contract, name, composite), merged on
   the own key; every opp_* feature on the opponent key. SYNTHETIC DATA ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import feature_engineering as FE  # noqa: E402
from player_model import history_refetch as HR  # noqa: E402
from player_model import player_stat_nulls as PSN  # noqa: E402
from src.integrity import data_truth as DT  # noqa: E402
from src.integrity import props_feature_truth as PFT  # noqa: E402


def _stats(**groups):
    base = {"games": {"minutes": 90, "position": "G"}, "goals": {"total": None, "conceded": None,
                                                                 "assists": None, "saves": None}}
    for k, v in groups.items():
        base[k] = {**base.get(k, {}), **v}
    return base


# ── 1. the null rule ─────────────────────────────────────────────────────────────────────────────

def test_null_goals_conceded_stays_missing_for_a_player_who_played():
    c = PSN.player_counts(_stats(), answered=True)
    assert c["goals_conceded"] is None
    assert c["saves"] == 0 and c["goals"] == 0          # the ordinary counts keep the owner's rule


def test_sent_goals_conceded_is_kept_as_sent():
    assert PSN.player_counts(_stats(goals={"conceded": 0}), answered=True)["goals_conceded"] == 0
    assert PSN.player_counts(_stats(goals={"conceded": 3}), answered=True)["goals_conceded"] == 3


def test_reader_blanks_a_stored_null_read_as_zero_goals_conceded_and_keeps_the_rest():
    hist = pd.DataFrame({"fixture_id": [1, 1], "player_id": [10, 11], "goals_conceded": [2, 0],
                         "saves": [3, 0], "is_home": [1, 0]})
    rf = pd.DataFrame({"fixture_id": [1, 1], "player_id": [10, 11], "goals_conceded": [0, 0],
                       "saves": [4, 0], "filled_cols": ["goals_conceded", "saves"]})
    out = HR.with_refetched(hist, rf).set_index("player_id")
    assert np.isnan(out.loc[10, "goals_conceded"])       # provider null: missing, not 0
    assert out.loc[10, "saves"] == 4
    assert out.loc[11, "goals_conceded"] == 0            # a sent 0 stays 0
    assert out.loc[11, "saves"] == 0                     # null-and-played saves: the owner's 0
    assert (out["count_source"] == "refetched").all()


# ── 1b. keeper vs the match score ───────────────────────────────────────────────────────────────

def _fx():
    return pd.DataFrame({"fixture_id": [1, 2, 3], "ft_home": [2, 1, 1], "ft_away": [1, 0, 1],
                         "status": ["FT", "FT", "AET"]})


def _keepers():
    return pd.DataFrame({
        "fixture_id": [1, 1, 2, 2, 2, 3, 1],
        "player_id":  [1, 2, 3, 4, 5, 6, 7],
        "position":   ["G", "G", "G", "G", "G", "G", "D"],
        "minutes":    [90, 90, 90, 60, 30, 120, 90],
        "is_home":    [1, 0, 1, 0, 0, 1, 1],
        # fx1 home GK conceded 1 (agrees) / away GK says 0 but conceded 2 (contradicts)
        # fx2 home GK 0 (agrees) / away side had two keepers (not checkable); fx3 AET (not checkable)
        "goals_conceded": [1, 0, 0, 0, 1, 0, 0]})


def test_contradicting_keeper_value_is_missing_and_kept_beside():
    out = HR.with_gk_conceded_checked(_keepers(), _fx()).set_index("player_id")
    assert out.loc[1, "goals_conceded"] == 1 and out.loc[1, HR.GK_CHECK_COL] == HR.GK_AGREES
    assert np.isnan(out.loc[2, "goals_conceded"]) and out.loc[2, HR.GK_CHECK_COL] == HR.GK_CONTRADICTS
    assert out.loc[2, HR.GK_ORIG_COL] == 0                  # the provider value stays beside
    for pid in (4, 5, 6):                                   # two keepers / extra time: untouched
        assert out.loc[pid, HR.GK_CHECK_COL] == HR.GK_NOT_CHECKABLE
    assert out.loc[4, "goals_conceded"] == 0 and out.loc[6, "goals_conceded"] == 0
    assert out.loc[7, HR.GK_CHECK_COL] is None              # outfield row: not a keeper check
    assert out.loc[7, "goals_conceded"] == 0


def test_gk_truth_check_is_red_when_the_frame_still_holds_a_contradiction():
    checked = HR.with_gk_conceded_checked(_keepers(), _fx())
    assert PFT.gk_conceded(checked.assign(date="2026-11-01"), _fx())["status"] == PFT.GREEN
    broken = checked.assign(date="2026-11-01")
    broken.loc[broken["player_id"] == 2, "goals_conceded"] = 0.0      # a reader that forgot the check
    assert PFT.gk_conceded(broken, _fx())["status"] == PFT.RED
    assert PFT.gk_conceded(_keepers().assign(date="2026-11-01"), _fx())["status"] == PFT.RED


# ── 2. the team_goals_pg_roll side ──────────────────────────────────────────────────────────────

def test_team_goals_pg_roll_is_own_side_and_every_opp_feature_is_opponent_side():
    assert FE._TEAM_ROLL["goals_scored"] == ("team_goals_pg_roll", "MIN_OPP_MATCHES", "own")
    for feat, _m, side in FE._TEAM_ROLL.values():
        assert (side == "opp") == feat.startswith("opp_"), feat
    assert FE.roll_features("own") == ["team_goals_pg_roll"]


def test_attach_team_rolls_gives_own_goals_to_the_player_and_opponent_conceded_from_the_opponent():
    t, const = PFT._synthetic_team_table()
    got = FE.attach_team_rolls(t[["fixture_id", "_tkey", "_okey", "date"]].copy(), FE._team_rolling(t, 5))
    m = got["team_goals_pg_roll"].notna()
    assert m.any()
    assert (got.loc[m, "team_goals_pg_roll"] == got.loc[m, "_tkey"].map(lambda c: const[c]["goals_scored"])).all()
    m = got["opp_goals_conceded_pg"].notna()
    assert (got.loc[m, "opp_goals_conceded_pg"]
            == got.loc[m, "_okey"].map(lambda c: const[c]["goals_conceded"])).all()


def test_side_check_green_now_and_red_on_the_old_opponent_keying(monkeypatch):
    assert PFT.side_keys()["status"] == PFT.GREEN
    roll = dict(FE._TEAM_ROLL)
    roll["goals_scored"] = ("team_goals_pg_roll", "MIN_OPP_MATCHES", "opp")
    monkeypatch.setattr(FE, "_TEAM_ROLL", roll)
    assert PFT.side_keys()["status"] == PFT.RED


def test_side_check_catches_a_merge_that_ignores_the_declared_side(monkeypatch):
    def all_on_opponent(df, roll):
        feats = [f for f, _, _ in FE._TEAM_ROLL.values()]
        return df.merge(roll[["fixture_id", "_tkey", *feats]].rename(columns={"_tkey": "_okey"}),
                        on=["fixture_id", "_okey"], how="left")
    monkeypatch.setattr(FE, "attach_team_rolls", all_on_opponent)
    c = PFT.side_keys()
    assert c["status"] == PFT.RED and "team_goals_pg_roll" in c["detail"]


def test_props_truth_is_a_gated_stoplight_rule():
    assert any(g == "Data truth 6" for g in DT.GATED)
    red = [PFT._c("Data truth 6a: keeper goals_conceded = match score", PFT.RED, "x", "")]
    assert any(f.startswith("Data truth 6") for f in DT.gate(checks=red))
