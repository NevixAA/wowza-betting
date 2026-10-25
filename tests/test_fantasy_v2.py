"""Regression tests for the Fantasy V2 projection / benchmark pipeline.

Synthetic tests pin the leakage rules (pre-deadline only, walk-forward only, unplayed != zero);
real-data tests pin the published benchmark so a silent change in scoring shows up as a failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import fantasy_panel as fp
from player_model import fantasy_residual as R
from player_model import fantasy_snapshots as fs
from player_model import fantasy_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


# ── synthetic FPL world ──────────────────────────────────────────────────────────────────────
DEADLINES = {1: "2026-08-21T17:30:00Z", 2: "2026-08-28T17:30:00Z", 3: "2026-09-04T17:30:00Z",
             4: "2026-09-11T17:30:00Z", 5: "2026-09-18T17:30:00Z", 6: "2026-09-25T17:30:00Z"}


def _boot(next_gw: int, elements: list[dict], finished_upto: int) -> dict:
    return {"events": [{"id": g, "deadline_time": d, "is_next": g == next_gw,
                        "finished": g <= finished_upto, "data_checked": g <= finished_upto}
                       for g, d in DEADLINES.items()],
            "teams": [{"id": 1, "name": "Alpha", "short_name": "ALP"},
                      {"id": 2, "name": "Beta", "short_name": "BET"}],
            "elements": elements}


def _el(i, team, etype, total, minutes, starts, ep, status="a", xg=0.5, xa=0.2, cost=60):
    return {"id": i, "web_name": f"P{i}", "first_name": "Player", "second_name": str(i),
            "team": team, "element_type": etype, "now_cost": cost, "status": status,
            "chance_of_playing_next_round": None if status == "a" else 0, "news": "",
            "ep_next": ep, "ep_this": ep, "form": ep, "points_per_game": ep,
            "selected_by_percent": "5.0", "transfers_in_event": 10, "transfers_out_event": 5,
            "cost_change_event": 0, "penalties_order": 1 if i == 1 else None,
            "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": None,
            "total_points": total, "minutes": minutes, "starts": starts, "goals_scored": 0,
            "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "bonus": 0, "bps": 0,
            "saves": 0, "yellow_cards": 0, "red_cards": 0, "expected_goals": xg,
            "expected_assists": xa, "expected_goals_conceded": 1.0, "threat": 10,
            "creativity": 10, "influence": 10, "defensive_contribution": 5}


def _ts(s):
    return pd.Timestamp(s).timestamp()


def _fixtures(played_upto: int, partial_gw: int | None = None) -> pd.DataFrame:
    rows = []
    for g in DEADLINES:
        ko = pd.Timestamp(DEADLINES[g]) + pd.Timedelta(days=1)
        fin = g <= played_upto
        if partial_gw == g:
            fin = False
        rows.append({"fixture": g, "event": g, "kickoff": ko, "team_h": 1 if g % 2 else 2,
                     "team_a": 2 if g % 2 else 1, "h_score": 1 if fin else None,
                     "a_score": 0 if fin else None, "h_fdr": 3, "a_fdr": 3, "finished": fin})
    return pd.DataFrame(rows)


def _archive(tmp_path: Path) -> Path:
    """GW2..GW5 pre-deadline snapshots; cumulative counters rise by known amounts."""
    path = tmp_path / "snap.csv"
    gains = {2: (6, 90, 1), 3: (2, 30, 0), 4: (0, 0, 0)}     # player 1 per-gameweek truth
    tot, mins, starts = 10, 180, 2
    for gw in (2, 3, 4, 5):
        els = [_el(1, 1, 3, tot, mins, starts, ep=5.0),
               _el(2, 2, 2, 4, 90, 1, ep=1.0),
               _el(3, 2, 4, 0, 0, 0, ep=0.0, status="i")]
        fetched = _ts(DEADLINES[gw]) - 3600
        fs.update_archive(_boot(gw, els, gw - 1), fetched, path=path)
        if gw in gains:
            g = gains[gw]
            tot, mins, starts = tot + g[0], mins + g[1], starts + g[2]
    return path


# ── snapshot archive ─────────────────────────────────────────────────────────────────────────
def test_snapshot_refuses_post_deadline_and_unknown_time(tmp_path):
    els = [_el(1, 1, 3, 10, 90, 1, 5.0)]
    late = _ts(DEADLINES[2]) + 60
    assert fs.rows_from_bootstrap(_boot(2, els, 1), late).empty
    assert fs.update_archive(_boot(2, els, 1), None, path=tmp_path / "a.csv") == 0
    assert not (tmp_path / "a.csv").exists()


def test_snapshot_replaces_gameweek_but_never_with_older(tmp_path):
    p = tmp_path / "a.csv"
    els = [_el(1, 1, 3, 10, 90, 1, 5.0)]
    fs.update_archive(_boot(2, els, 1), _ts(DEADLINES[2]) - 7200, path=p)
    newer = [_el(1, 1, 3, 10, 90, 1, 6.5)]
    fs.update_archive(_boot(2, newer, 1), _ts(DEADLINES[2]) - 3600, path=p)
    older = [_el(1, 1, 3, 10, 90, 1, 1.0)]
    fs.update_archive(_boot(2, older, 1), _ts(DEADLINES[2]) - 9000, path=p)
    a = fs.load(p)
    assert len(a) == 1 and float(a["ep_next"].iloc[0]) == 6.5


# ── panel: targets, leakage, provisional handling ────────────────────────────────────────────
def test_panel_actuals_are_cumulative_differences(tmp_path):
    arch = fs.load(_archive(tmp_path))
    ev = pd.DataFrame({"gw": list(DEADLINES), "deadline": pd.to_datetime(list(DEADLINES.values()), utc=True),
                       "finished": [g <= 4 for g in DEADLINES], "checked": [g <= 4 for g in DEADLINES]})
    p = fp.build_panel(arch, _fixtures(4), ev, log=pd.DataFrame())
    one = p[p["element"] == 1].set_index("gw")
    assert one.loc[2, "actual_points"] == 6 and one.loc[2, "actual_minutes"] == 90
    assert one.loc[3, "actual_points"] == 2 and one.loc[3, "bucket"] == "m1_59"
    assert one.loc[4, "bucket"] == "m0"
    assert one.loc[5, "status"] == "PENDING" and pd.isna(one.loc[5].get("actual_points"))
    # last-gameweek deltas come from the PREVIOUS snapshot only
    assert one.loc[3, "min_last"] == 90 and pd.isna(one.loc[2, "min_last"])


def test_team_strength_uses_only_fixtures_before_deadline():
    tfl = fp.team_fixture_long(_fixtures(5))
    snap = pd.DataFrame({"team_id": [1, 2], "position": ["GKP", "GKP"], "xg": [1.0, 1.0],
                         "minutes": [90, 90], "xgc": [1.0, 1.0]})
    early = fp.team_strength_asof(snap, tfl, pd.Timestamp(DEADLINES[2]))
    late = fp.team_strength_asof(snap, tfl, pd.Timestamp(DEADLINES[5]))
    assert early["games"].sum() == 2          # only GW1's fixture (both sides) had been played
    assert late["games"].sum() == 8


def test_provisional_gameweek_drops_unfinished_teams(tmp_path):
    arch = fs.load(_archive(tmp_path))
    ev = pd.DataFrame({"gw": list(DEADLINES), "deadline": pd.to_datetime(list(DEADLINES.values()), utc=True),
                       "finished": [g <= 3 for g in DEADLINES], "checked": [g <= 3 for g in DEADLINES]})
    # GW4 is not finished and its single fixture is unplayed -> nobody may be scored for GW4
    p = fp.build_panel(arch, _fixtures(3, partial_gw=4), ev, log=pd.DataFrame())
    assert p[(p["gw"] == 4) & (p["status"] == "PROVISIONAL")].empty


def test_wowza_projection_is_as_of_the_deadline():
    log = pd.DataFrame({"snapshot_ts": pd.to_datetime(["2026-09-10T10:00Z", "2026-09-12T10:00Z"], utc=True),
                        "player_name": ["Player 1", "Player 1"], "team": ["Alpha", "Alpha"],
                        "proj_fixture_adj": [4.0, 99.0], "p_start": [0.5, 1.0]})
    w = fp.wowza_asof(log, pd.Timestamp(DEADLINES[4]))       # deadline 2026-09-11
    assert float(w["wowza_uncond"].iloc[0]) == pytest.approx(2.0)


# ── residual models and the walk-forward ─────────────────────────────────────────────────────
def _synthetic_panel(n_gw=5, n=300, signal=1.5, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gw + 1):
        ml = rng.uniform(0, 1, n)
        ep = rng.uniform(0, 6, n)
        y = ep + signal * (ml - 0.5) + rng.normal(0, 1.0, n)
        rows.append(pd.DataFrame({
            "gw": gw, "element": np.arange(n), "status": "FINAL", "minutes": 100.0,
            "ep_next": ep, "actual_points": y, "actual_minutes": 90.0, "bucket": "m60",
            "min_last": ml * 90, "min_per_team_game": 45.0, "start_rate": 0.5, "started_last": 1,
            "chance_next_f": 1.0, "is_out": 0, "is_doubt": 0, "has_news": 0, "wowza_p_start": np.nan,
            "xg90": 0.2, "xa90": 0.1, "goal_pts": 5.0, "pen_first": 0, "set_piece": 0, "bps90": 15,
            "threat_pg": 10, "creativity_pg": 10, "n_fix": 1, "home": 1.0, "fdr": 3, "lam_for": 1.4,
            "lam_against": 1.3, "cs_exp": 0.3, "attack_fixture": 1.0, "price": 6.0, "ownership": 5.0,
            "net_transfers_k": 0.0, "ppm": 0.5, "pts_per_team_game": 3.0, "form": 3.0, "pts_last": 3.0,
            "wowza_uncond": np.nan, "in_ledger": 0, "position": "MID",
            **{f"pos_{p}": int(p == "MID") for p in fp.POSITIONS}}))
    return pd.concat(rows, ignore_index=True)


def test_walk_forward_never_trains_on_test_or_future():
    oos = R.walk_forward(_synthetic_panel())
    for gw, g in oos.groupby("gw"):
        used = [int(x) for x in str(g["train_gws"].iloc[0]).split(",")]
        assert max(used) < gw


def test_linear_contributions_add_up_exactly():
    p = R.prepare(_synthetic_panel(n_gw=2))
    r = (p["actual_points"] - p["ep_next"]).to_numpy()
    m = R.LinearResidual().fit(p, r)
    c = m.contributions(p)
    assert np.allclose(c.sum(axis=1).to_numpy(), m.predict(p))


def test_acceptance_needs_three_final_gameweeks():
    oos2 = R.walk_forward(_synthetic_panel(n_gw=3, signal=3.0))      # 2 OOS gameweeks
    assert R.acceptance(R.score_arms(oos2), "LINEAR")["decision"] == "NO-GO"
    oos4 = R.walk_forward(_synthetic_panel(n_gw=5, signal=3.0))      # 4 OOS gameweeks
    acc = R.acceptance(R.score_arms(oos4), "LINEAR")
    assert acc["decision"] == "GO" and acc["final_oos_gws"] == 4


def test_residual_finds_nothing_when_fpl_is_already_right():
    """ep_next is the exact conditional mean here, so no arm may claim a material gain."""
    oos = R.walk_forward(_synthetic_panel(n_gw=5, signal=0.0, seed=4))
    v = R.score_arms(oos)["vs_fpl_final_only"]
    for arm in R.RESIDUAL_ARMS:
        assert v[arm]["mae_minus_fpl"] > -0.03, arm


def test_metrics_on_known_values():
    m = R.metrics(np.arange(10.0), np.arange(10.0) + 1)
    assert m["mae"] == 1 and m["rmse"] == 1 and m["bias"] == 1 and m["spearman"] == 1


# ── minutes mixture and distribution ─────────────────────────────────────────────────────────
class _MM:
    sub_pmf_ = {"MID": pd.Series({1: 0.7, 2: 0.2, 6: 0.1})}
    size_ = {"MID": 3.0}


def test_distribution_is_a_pmf_with_the_headline_mean():
    probs = {"m0": 0.2, "m1_59": 0.2, "m60": 0.6}
    pmf = R.point_distribution(4.0, probs, "MID", _MM())
    assert pmf.sum() == pytest.approx(1.0)
    s = R.dist_summary(pmf)
    assert s["dist_mean"] == pytest.approx(4.0, abs=0.05)
    assert s["p25"] <= s["median"] <= s["p75"] <= s["p90"]
    assert pmf[2] >= 0.2 - 1e-9                    # index 2 == 0 points: at least P(0 minutes)


def test_blank_gameweek_means_zero_minutes():
    x = R.prepare(_synthetic_panel(n_gw=1, n=50))
    x["bucket"] = np.where(np.arange(len(x)) % 3 == 0, "m0", np.where(np.arange(len(x)) % 3 == 1, "m1_59", "m60"))
    x["actual_minutes"] = np.where(x["bucket"] == "m0", 0, np.where(x["bucket"] == "m1_59", 20, 90))
    mm = R.MinutesModel().fit(x)
    x2 = x.copy()
    x2["n_fix"] = 0
    P = mm.predict_proba(x2)
    assert (P["m0"] == 1.0).all() and (P["m60"] == 0).all()


# ── decisions: captaincy and transfers ───────────────────────────────────────────────────────
def _board():
    rng = np.random.default_rng(5)
    n = 40
    d = pd.DataFrame({"element": range(n), "player_name": [f"Player {i}" for i in range(n)],
                      "web_name": [f"P{i}" for i in range(n)], "team": ["Alpha", "Beta"] * 20,
                      "position": ["MID", "FWD", "DEF", "GKP"] * 10,
                      "price": np.round(rng.uniform(4, 12, n), 1), "final_xpts": rng.uniform(1, 8, n),
                      "fpl_ep_next": rng.uniform(1, 8, n), "wowza_residual": 0.0,
                      "residual_applied": False, "p_start": rng.uniform(0.5, 1, n),
                      "exp_minutes": 80.0, "p6": rng.uniform(0, 0.5, n), "p10": 0.1, "p15": rng.uniform(0, 0.1, n),
                      "p_blank": rng.uniform(0.1, 0.6, n), "p90": 10.0, "ceiling": 10.0, "sd": 3.0,
                      "next_fixture": "BET(H)", "team_goal_exp": 1.5, "pen_first": 0, "set_piece": 0,
                      "risk": "Low", "ownership": rng.uniform(0, 50, n), "xpts_3gw": rng.uniform(3, 20, n),
                      "xpts_5gw": rng.uniform(5, 35, n), "value_per_m": 0.5, "fixture_run": "", })
    return d


def test_captaincy_gives_three_distinct_starters():
    cap = v2.captaincy(_board())
    picks = [cap[k]["player"] for k in ("SAFE", "BALANCED", "HIGH_UPSIDE")]
    assert len(set(picks)) == 3
    assert all(cap[k]["p_start"] >= 0.75 for k in ("BALANCED", "HIGH_UPSIDE"))
    assert cap["SAFE"]["p_start"] >= 0.85


def test_transfers_are_same_position_affordable_and_positive():
    b = _board()
    t = v2.transfers(b)
    assert len(t)
    assert (t["gain_5gw"] > 0).all()
    assert (t["in_price"] <= t["out_price"] + v2.BUDGET_SLACK + 1e-9).all()
    pos = b.set_index("player_name")["position"]
    assert (t["in_player"].map(pos) == t["position"]).all()


def test_residual_stays_in_shadow_by_default(monkeypatch):
    monkeypatch.delenv("FANTASY_V2_RESIDUAL_MODE", raising=False)
    assert v2.residual_mode() == "shadow"
    monkeypatch.setenv("FANTASY_V2_RESIDUAL_MODE", "nonsense")
    assert v2.residual_mode() == "shadow"


# ── real committed data: the published benchmark must reproduce ──────────────────────────────
@pytest.mark.skipif(not (OUT / "fantasy_ledger.csv").exists(), reason="no committed ledger")
def test_published_ledger_benchmark_reproduces():
    """Pro's fantasy_benchmark (2026-10-20) scored GW4-6: FPL MAE 1.357 / Wowza 2.0375, n=724.
    The ledger is rewritten by settlement, so only pin it while it still covers exactly GW4-6."""
    b = v2.published_ledger_benchmark()
    if b["gameweeks"] != [4, 5, 6]:
        pytest.skip("ledger has grown past GW6; numbers legitimately move")
    assert b["FPL"]["n"] == 724
    assert b["FPL"]["mae"] == pytest.approx(1.357, abs=1e-3)
    assert b["WOWZA"]["mae"] == pytest.approx(2.0375, abs=1e-3)
    assert b["fpl_won_every_gw"] is True


@pytest.mark.skipif(not (OUT / "fantasy_fpl_snapshots.csv").exists(), reason="no archive")
def test_archive_actuals_match_the_settled_ledger():
    """Two independent settlements (projection-log cumulative diff vs snapshot-archive diff) must
    agree to the point for every player both can see."""
    led = pd.read_csv(OUT / "fantasy_ledger.csv")
    p = fp.build_panel()
    led["key"] = led["player_name"].map(fp.norm_name)
    m = led.merge(p[["gw", "key", "actual_points"]], on=["gw", "key"], suffixes=("", "_arch"))
    assert len(m) > 500
    assert (m["actual_points"] - m["actual_points_arch"]).abs().max() == 0


@pytest.mark.skipif(not (OUT / "fantasy_v2_benchmark.json").exists(), reason="no V2 benchmark")
def test_committed_benchmark_shows_fpl_and_a_decision():
    b = json.loads((OUT / "fantasy_v2_benchmark.json").read_text(encoding="utf-8"))
    assert b["decision"]["apply_residual_to_mean"] in {"GO", "NO-GO"}
    assert "FPL" in b["scores"]["ACTIVE"]["pooled"]
    from dashboard_data import fantasy as fdata
    t = fdata.v2_benchmark_table(b, "ACTIVE")
    assert t.iloc[0]["arm"] == "FPL"


@pytest.mark.skipif(not (OUT / "fantasy_v2_projections.csv").exists(), reason="no V2 board")
def test_fantasy_page_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    page = next((ROOT / "pages").glob("13_*_Fantasy.py"))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [s.value for s in at.subheader]
    assert any("benchmark" in h.lower() for h in heads)
    # 2026-10-25 redesign: the command-center block is the page header metric strip (GW, deadline,
    # projection health) above the task tabs, no longer a subheader.
    assert {"Gameweek", "Projection health"} <= {m.label for m in at.metric}
    assert "Overview" in [t.label for t in at.tabs]


# ── gameweek-by-gameweek scoreboard ──────────────────────────────────────────────────────────
def _panel_with_minutes(n_gw=5, n=120, seed=11):
    p = _synthetic_panel(n_gw=n_gw, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, len(p))
    p["bucket"] = np.where(u < 0.3, "m0", np.where(u < 0.5, "m1_59", "m60"))
    p["actual_minutes"] = np.where(p["bucket"] == "m0", 0, np.where(p["bucket"] == "m1_59", 25, 90))
    p.loc[p["element"] % 3 == 0, "wowza_uncond"] = p["ep_next"] + 1.0     # the old projection
    p.loc[p["element"] % 3 == 0, "wowza_p_start"] = 0.8
    p["web_name"] = "P" + p["element"].astype(str)
    return p


def _board_for(panel):
    oos = R.walk_forward(panel)
    return R.gw_scoreboard(oos, R.minutes_models_for(panel, oos["gw"].unique()))


def test_scoreboard_is_chronological():
    """Every GW is scored only by models trained on earlier FINAL GWs, so rewriting the LAST
    gameweek's outcomes must leave every earlier scoreboard row bit-for-bit unchanged."""
    p = _panel_with_minutes()
    board, rel = _board_for(p)
    assert sorted(board["gw"].unique()) == [2, 3, 4, 5]
    assert set(board["arm"]) == set(R.ARMS)
    for r in board.itertuples():
        assert max(int(x) for x in str(r.train_gws).split(",")) < r.gw
    p2 = p.copy()
    last = p2["gw"] == 5
    p2.loc[last, "actual_points"] = p2.loc[last, "actual_points"] + 7.0
    p2.loc[last, "bucket"] = "m60"
    board2, rel2 = _board_for(p2)
    early = lambda b: b[b["gw"] < 5].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(board), early(board2))
    pd.testing.assert_frame_equal(rel[rel["gw"] < 5].reset_index(drop=True),
                                  rel2[rel2["gw"] < 5].reset_index(drop=True))
    assert not board[board["gw"] == 5].reset_index(drop=True).equals(
        board2[board2["gw"] == 5].reset_index(drop=True))
    # the three residual arms share one minutes forecast; reliability is written once per source
    g5 = board[board["gw"] == 5].set_index("arm")
    assert g5.loc["LINEAR", "pred_p60"] == g5.loc["HGB", "pred_p60"] == g5.loc["RIDGE", "pred_p60"]
    assert rel.groupby(["gw", "source"]).size().eq(len(R.P60_BINS) - 1).all()


def test_scoreboard_metrics_on_a_known_gameweek():
    n = 12
    y = np.arange(n, dtype=float)
    d = pd.DataFrame({"gw": 9, "status": "PROVISIONAL", "train_gws": "7,8", "element": range(n),
                      "web_name": [f"P{i}" for i in range(n)], "actual_points": y,
                      "FPL": y + 1.0, "LINEAR": y[::-1].copy(), "RIDGE": y, "HGB": y + 0.5,
                      "WOWZA": np.where(np.arange(n) < 10, y + 2.0, np.nan),
                      "bucket": ["m0", "m1_59", "m60", "m60"] * 3,
                      "chance_next_f": 1.0, "min_per_team_game": 60.0, "start_rate": 0.5,
                      "n_fix": 1, "wowza_p_start": 0.75})
    board, rel = R.gw_scoreboard(d)
    b = board.set_index("arm")
    assert (b["status"] == "PROVISIONAL").all() and (b["gw"] == 9).all()
    f = b.loc["FPL"]
    assert f["mae"] == 1 and f["rmse"] == 1 and f["bias"] == 1 and f["spearman"] == 1
    assert f["top10_overlap"] == 10 and f["top10_actual_mean"] == pytest.approx(6.5)
    assert f["top15_overlap"] == 12 and f["top15_actual_mean"] == pytest.approx(5.5)
    assert f["top10_best_mean"] == pytest.approx(6.5)
    assert f["captain"] == "P11" and f["captain_actual"] == 11 and f["captain_shortfall"] == 0
    lin = b.loc["LINEAR"]
    assert lin["spearman"] == pytest.approx(-1.0)
    assert lin["top10_overlap"] == 8                        # predicted top-10 = actual 0..9
    assert lin["captain"] == "P0" and lin["captain_actual"] == 0 and lin["captain_shortfall"] == 11
    assert lin["mae_minus_fpl"] == pytest.approx(lin["mae"] - 1.0)
    assert b.loc["RIDGE", "mae"] == 0 and b.loc["HGB", "bias"] == pytest.approx(0.5)
    w = b.loc["WOWZA"]                                     # scored only on its 10 rows
    assert w["n"] == 10 and w["mae"] == 2 and w["fpl_mae_same_rows"] == 1
    assert w["mae_minus_fpl"] == pytest.approx(1.0) and w["best_captain_actual"] == 11
    # minutes: WOWZA carries only P(60+) = 0.75 on rows 0..9, of which 4 were 60+
    assert pd.isna(w["pred_p0"]) and w["pred_p60"] == pytest.approx(0.75)
    assert w["obs_p60"] == pytest.approx(0.4)
    assert w["p60_brier"] == pytest.approx(0.4 * 0.25 ** 2 + 0.6 * 0.75 ** 2)
    assert w["p60_ece"] == pytest.approx(0.35)
    # FPL heuristic: play = 1 * 1 * 0.5, P(60+) = 1 * 0.5 * 0.92 -> 0.46
    assert f["pred_p0"] == pytest.approx(0.5) and f["pred_p60"] == pytest.approx(0.46)
    assert f["obs_p0"] == pytest.approx(3 / 12) and f["obs_p60"] == pytest.approx(6 / 12)
    # no minutes model passed: the residual arms report it as unavailable, not as zero
    assert b.loc["LINEAR", "minutes_source"] == "not available"
    assert pd.isna(b.loc["LINEAR", "p60_brier"])
    wr = rel[rel["source"] == R.MINUTES_SOURCE["WOWZA"]].set_index("bin")
    assert wr.loc[4, "n"] == 10 and wr["n"].sum() == 10        # 0.75 falls in [0.7, 0.9)
    from dashboard_data import fantasy as fdata
    pooled = fdata.v2_pooled_reliability(pd.concat([rel, rel.assign(gw=10)]))
    assert pooled.loc[pooled["source"] == R.MINUTES_SOURCE["WOWZA"], "n"].sum() == 20


@pytest.mark.skipif(not (OUT / "fantasy_v2_scoreboard.csv").exists(), reason="no scoreboard")
def test_committed_scoreboard_matches_the_benchmark():
    """The scoreboard and the benchmark JSON are two views of one walk-forward; they must agree."""
    sb = pd.read_csv(OUT / "fantasy_v2_scoreboard.csv")
    b = json.loads((OUT / "fantasy_v2_benchmark.json").read_text(encoding="utf-8"))
    by = {r["gw"]: r for r in b["scores"]["ACTIVE"]["by_gw"]}
    assert set(sb["gw"]) == set(by)
    for r in sb.itertuples():
        arm = by[r.gw][r.arm]
        assert r.n == arm["n"]
        if "mae" not in arm:                     # too few rows to score (e.g. old Wowza, 9 rows in a PROVISIONAL GW):
            assert pd.isna(r.mae)                # the benchmark omits the metric, the scoreboard leaves it blank
            continue
        assert r.mae == pytest.approx(arm["mae"], abs=1e-4)
