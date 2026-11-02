"""Fantasy data truth (2026-11-01): no invented values, FPL joins on element ids, league facts from
the registry. Each test pins one behaviour that used to be a typed default.

    python -m pytest tests/test_fantasy_data_truth.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from player_model import fantasy as F
from player_model import fantasy_chips as FC
from player_model import fantasy_distribution as FD
from player_model import fantasy_optimizer as FO
from player_model import fantasy_panel as fp
from player_model import fantasy_residual as R
from player_model import fantasy_settle as FS
from player_model import fantasy_transfers as FT
from player_model import fantasy_v2 as v2


# ── league facts come from the registry ──────────────────────────────────────────────────────────
def test_fantasy_league_is_the_registry_fpl_league():
    from src import league_registry as reg
    assert F.FANTASY_LEAGUE == reg.fpl_league()
    assert F._fantasy_league_id() == reg.entry(reg.fpl_league())["api_football_id"]


def test_registry_loads_under_a_foreign_src_package(tmp_path):
    # wowzaV9-Pro imports player_model with its OWN `src`; the registry must still be v9's, and
    # Pro's `src` must be left in place.
    import sys
    import types
    fake = types.ModuleType("src")
    fake.__path__ = [str(tmp_path)]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    sys.modules["src"] = fake
    try:
        F._V9_SRC.clear()
        assert F._registry().fpl_league() == F.FANTASY_LEAGUE
        assert F._resolve_club("Man City", ["Manchester City", "Manchester United"]) == "Manchester City"
        assert sys.modules["src"] is fake
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
        F._V9_SRC.clear()


def test_league_rows_resolve_through_the_registry_not_raw_equality():
    df = pd.DataFrame({"league": [F.FANTASY_LEAGUE, "Not A League", None], "x": [1, 2, 3]})
    assert F._fantasy_league_rows(df)["x"].tolist() == [1]


# ── ranking / captaincy ──────────────────────────────────────────────────────────────────────────
def test_unknown_p_start_is_labelled_unknown_ranked_last_and_never_captain():
    df = pd.DataFrame({"player_name": ["A", "B", "C"], "position": ["FWD", "MID", "DEF"],
                       "fantasy_pts": [9.0, 4.0, 3.0], "p_start": [np.nan, 0.9, 0.95],
                       "availability": ["available"] * 3})
    out = F.rank_and_captains(df, n_captains=3)
    a = out.set_index("player_name").loc["A"]
    assert a["start_confidence"] == "Start probability unknown"
    assert math.isnan(a["xpts_uncond"]) and a["overall_rank"] == 3 and not a["captain_pick"]
    assert set(out.loc[out["captain_pick"], "player_name"]) == {"B", "C"}


def test_captain_flags_follow_the_sorted_rows():
    # A doubtful player sorted FIRST must not borrow the eligibility of the row that sat first
    # before sorting (the mask used to be aligned on the pre-sort index).
    df = pd.DataFrame({"player_name": ["fit", "doubt"], "position": ["FWD", "FWD"],
                       "fantasy_pts": [3.0, 9.0], "p_start": [0.9, 0.9],
                       "availability": ["available", "doubtful"]})
    out = F.rank_and_captains(df, n_captains=1)
    assert out.loc[out["captain_pick"], "player_name"].tolist() == ["fit"]


def test_unknown_bonus_driver_gives_unknown_bonus():
    assert math.isnan(F._expected_bonus(0.3, 0.2, 0.4, float("nan"), 0.3))
    assert F._expected_bonus(0.3, 0.2, 0.4, 6.0, 0.3) > 0


def test_missing_fdr_is_skipped_not_read_as_3():
    assert math.isnan(F._cs_prob([{"fdr": None}]))
    assert F._cs_prob([{"fdr": None}, {"fdr": 2}]) == pytest.approx(math.exp(-(0.5 + 0.84)))


# ── API-Football -> FPL bridge ───────────────────────────────────────────────────────────────────
def test_crosswalk_joins_on_ids_never_on_a_shared_name():
    # Round 2 (2026-11-02): the bridge is the stored id crosswalk, not a name. A name two FPL
    # elements share no longer matters; only the API-Football player_id does.
    fpl = pd.DataFrame({"fpl_id": [1, 2, 3], "match_key": ["johnsmith", "johnsmith", "aliceb"],
                        "web_key": ["smith", "smithj", "alice"]})
    pl = pd.DataFrame({"player_id": [10, 20, 30], "player_name": ["John Smith", "Alice B", "Nobody"]})
    cw = pd.DataFrame({"fpl_id": [2, 3], "af_player_id": [10, 20]})
    ids = F.fpl_crosswalk(pl, fpl, crosswalk=cw)
    assert ids.iloc[0] == 2 and ids.iloc[1] == 3 and math.isnan(ids.iloc[2])


# ── optimiser / transfers / chips ────────────────────────────────────────────────────────────────
def _pool(n_per=6):
    rows, i = [], 0
    for pos in ("GKP", "DEF", "MID", "FWD"):
        for k in range(n_per):
            i += 1
            rows.append({"fpl_id": i, "player_name": f"{pos}{k}", "position": pos,
                         "team": f"T{i % 10}", "price": 5.0, "xpts_uncond": 2.0 + k * 0.1})
    return pd.DataFrame(rows)


def test_optimizer_never_prices_an_unpriced_player_at_zero():
    p = _pool()
    p.loc[p["player_name"] == "FWD5", ["price", "xpts_uncond"]] = [np.nan, 50.0]
    sq = FO.optimal_squad(p)
    assert sq["feasible"] and "FWD5" not in set(sq["squad"]["player_name"])


def test_optimizer_refuses_to_assume_everyone_starts():
    p = _pool().drop(columns="xpts_uncond").assign(fantasy_pts=3.0)
    assert FO.optimal_squad(p) == {"feasible": False, "reason": "no unconditional points column"}


def test_transfers_join_on_ids_and_never_call_an_unpriced_player_affordable():
    proj = _pool()
    proj["total_xpts_next"] = proj["xpts_uncond"] * 5
    proj["n_fixtures_next"] = 5
    proj.loc[proj["player_name"] == "MID5", "price"] = np.nan
    squad = proj[proj["player_name"].isin(["MID0"])][["fpl_id", "player_name", "position", "price"]]
    squad = pd.concat([squad, pd.DataFrame([{"fpl_id": 999, "player_name": "Ghost",
                                             "position": "MID", "price": 5.0}])])
    out = FT.transfer_options(squad, proj, bank=10.0)
    assert {m["out"] for m in out["moves"]} == {"MID0"}            # Ghost is unknown, not 0 pts
    assert any("Ghost" in w for w in out["warnings"])
    mid5 = [m for m in out["moves"] if m["in"] == "MID5"]
    assert all(m["affordable"] is False for m in mid5)


def test_transfers_without_a_bank_mark_dearer_moves_unknown_not_unaffordable():
    proj = _pool().assign(n_fixtures_next=1)
    proj.loc[proj["player_name"] == "MID5", "price"] = 9.0
    squad = proj[proj["player_name"] == "MID0"][["fpl_id", "player_name", "position", "price"]]
    out = FT.transfer_options(squad, proj, n_in=6)
    m = {x["in"]: x for x in out["moves"]}
    assert m["MID5"]["affordable"] is None and m["MID5"]["verdict"].startswith("affordability unknown")
    assert m["MID1"]["affordable"] is True


def test_transfers_refuse_a_name_only_join():
    proj = _pool().drop(columns="fpl_id")
    squad = pd.DataFrame([{"player_name": "MID0", "position": "MID", "price": 5.0}])
    assert "error" in FT.transfer_options(squad, proj)


def test_chips_withheld_when_an_owned_player_has_no_projection():
    proj = _pool()
    squad = pd.concat([proj.head(14), pd.DataFrame([{"fpl_id": 999, "player_name": "Ghost",
                                                     "position": "MID"}])])
    out = FC.chip_values(proj, squad=squad)
    for k in ("bench_boost", "free_hit", "wildcard"):
        assert out[k]["available"] is False and "Ghost" in out[k]["why"]


# ── distribution ────────────────────────────────────────────────────────────────────────────────
def test_simulate_skips_a_player_with_an_unknown_input():
    base = {"position": "FWD", "p_start": 0.9, "minutes_pg": 85, "p_goal": 0.4, "p_assist": 0.2,
            "cs_pts": 0.0, "dc_pts": 0.1, "bonus_pts": 0.5}
    d = pd.DataFrame([base, {**base, "position": "XYZ"}, {**base, "p_start": np.nan}])
    out = FD.simulate(d, n_sims=2000)
    assert out["sim_mean"].notna().tolist() == [True, False, False]


# ── V2 panel / residual / multi-GW ──────────────────────────────────────────────────────────────
def _ts():
    return pd.DataFrame({"team_id": [1, 2], "games": [5, 5], "gf_pg": [1.5, 1.0],
                         "ga_pg": [1.0, 1.4], "xg_for_pg": [1.6, 1.1], "xga_pg": [1.1, 1.3],
                         "ha_home": [1.12, 1.12], "ha_away": [0.88, 0.88]})


def test_fixture_lambdas_are_unknown_for_an_unknown_club_not_league_average():
    assert all(math.isnan(v) for v in fp.fixture_lambdas(1, 77, True, _ts()))
    lf, la = fp.fixture_lambdas(1, 2, True, _ts())
    assert lf > la > 0


def test_home_advantage_is_measured_from_finished_fixtures():
    tfl = pd.DataFrame({"team_id": [1, 2, 1, 2], "opp_id": [2, 1, 2, 1], "event": [1, 1, 2, 2],
                        "kickoff": pd.to_datetime(["2026-08-01"] * 2 + ["2026-08-08"] * 2, utc=True),
                        "home": [True, False, False, True], "gf": [3, 1, 2, 2], "ga": [1, 3, 2, 2],
                        "fdr": [3] * 4, "finished": [True] * 4, "fixture": [1, 1, 2, 2]})
    snap = pd.DataFrame({"team_id": [1, 2], "position": ["GKP", "GKP"], "xg": [1.0, 1.0],
                         "xgc": [1.0, 1.0], "minutes": [180, 180]})
    ts = fp.team_strength_asof(snap, tfl, pd.Timestamp("2026-09-01", tz="UTC"))
    # home goals 3+2 over 2 games = 2.5, all goals 8 over 4 = 2.0 -> 1.25; away 1.5 / 2.0 = 0.75
    assert ts["ha_home"].iloc[0] == pytest.approx(1.25)
    assert ts["ha_away"].iloc[0] == pytest.approx(0.75)


def test_event_probs_never_invent_minutes_or_neutral_fixtures():
    x = pd.DataFrame({"attack_fixture": [np.nan, 1.0, np.nan], "n_fix": [1, 1, 0],
                      "xg90": [0.5, 0.5, 0.5], "xa90": [0.2, 0.2, 0.2], "cs_prob": [np.nan, 0.3, np.nan],
                      "position": ["FWD", "DEF", "DEF"], "bonus_per_start": [0.3, np.nan, 0.3],
                      "lam_for": [1.2, 1.2, np.nan]})
    P = pd.DataFrame({"m0": [0.1, 0.1, 1.0], "m1_59": [0.2, 0.2, 0.0], "m60": [0.7, 0.7, 0.0]})
    ev = R.event_probs(x, P, {"m1_59": 30.0, "m60": 89.0})
    assert math.isnan(ev["goal_prob"].iloc[0])                    # unknown fixture factor
    assert ev["goal_prob"].iloc[1] > 0 and math.isnan(ev["exp_bonus"].iloc[1])
    assert ev["goal_prob"].iloc[2] == 0 and ev["team_cs_prob"].iloc[2] == 0   # blank: real zero
    ev2 = R.event_probs(x, P, {"m60": 89.0})                      # bucket never seen in training
    assert ev2["exp_minutes"].isna().all()


def test_point_distribution_without_a_fitted_dispersion_is_unknown():
    mm = R.MinutesModel()
    mm.sub_pmf_, mm.size_ = {}, {}
    pmf = R.point_distribution(4.0, {"m0": 0.1, "m1_59": 0.1, "m60": 0.8}, "MID", mm)
    assert np.isnan(pmf).all() and math.isnan(R.dist_summary(pmf)["median"])


def test_v2_fixture_multiplier_has_no_share_for_an_unknown_position():
    assert math.isnan(v2._fixture_mult("XYZ", 1.4, 0.3, 1.4, 0.3))
    assert v2._fixture_mult("FWD", 1.4, 0.3, 1.4, 0.3) == pytest.approx(1.0)


def test_panel_joins_wowza_projections_on_the_fpl_element_id():
    # Two FPL players share a name: the id-stamped projection goes to its own element only.
    d = pd.DataFrame({"element": [10, 11], "player_name": ["Same Name", "Same Name"],
                      "web_name": ["Same", "Same"]})
    wz = pd.DataFrame({"key": ["samename"], "fpl_id": [11.0], "team_log": ["X"],
                       "wowza_cond": [5.0], "wowza_p_start": [0.8], "wowza_uncond": [4.0]})
    out = fp.attach_wowza(d, wz)
    assert math.isnan(out["wowza_uncond"].iloc[0]) and out["wowza_uncond"].iloc[1] == 4.0
    # a LEGACY (no-id) row under that shared name is refused rather than given to either
    out2 = fp.attach_wowza(d, wz.assign(fpl_id=np.nan))
    assert out2["wowza_uncond"].isna().all()


# ── settlement ─────────────────────────────────────────────────────────────────────────────────
def test_settlement_key_lends_an_id_to_legacy_rows_of_the_same_name_only():
    lg = pd.DataFrame({"player_name": ["A", "A", "B", "B", "B"],
                       "fpl_id": [np.nan, 7, np.nan, 1, 2]})
    k = FS._player_key(lg).tolist()
    assert k[0] == k[1] == "id:7"                                   # legacy A pairs with id 7
    assert k[2] == "name:B" and k[3] == "id:1" and k[4] == "id:2"   # namesakes: never guessed
