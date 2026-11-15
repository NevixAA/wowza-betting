"""AU5 adversarial tests for the independent backtest (scripts/audit/independent_backtest).

Each test plants ONE deliberately broken row and asserts the independent implementation REJECTS it
(raises AuditViolation, or refuses to select / settle). A test here passing means the guard fired; if a
guard is removed the test fails. No production module is imported.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.audit.independent_backtest import validate as V
from scripts.audit.independent_backtest.regimes import Regime
from scripts.audit.independent_backtest.results import ResultIndex, outcome, pnl
from scripts.audit.independent_backtest.selection import decide_ou25, decide_side

T = pd.Timestamp("2026-10-20T10:00:00Z")


def _reg(**env):
    base = {"VALUABLE_THRESHOLD": "0.03", "MARKSMAN_THRESHOLD": "0.08", "LEAGUE_SNIPER_CAP": "0.12",
            "REQUIRE_FORM_DATA": "1"}
    base.update(env)
    return Regime("x", "2026-10-01T00:00:00Z",
                  values={"SNIPER_THRESHOLD": 0.12, "EDGE_CEILING": 0.19, "MIN_OVER_ODDS": 1.75,
                          "MIN_UNDER_ODDS": 1.75, "NF_MAX_OU_ODDS": 3.0, "league_sniper": {},
                          "league_marksman": {}, "league_edge_ceiling": {}},
                  env=base, std_params={}, side_params={})


def _row(**kw):
    r = {"league": "Championship", "model_type": "standard", "p_over25": 0.40, "odds_over25": 2.0,
         "odds_under25": 1.95, "kickoff_utc": "2026-10-21T19:00:00Z", "home_scored_last5": 1.0,
         "away_scored_last5": 1.0, "home_conceded_last5": 1.0, "away_conceded_last5": 1.0}
    r.update(kw)
    return r


# ── sanity: the clean row IS selected (so every rejection below is caused by the planted defect) ──
def test_clean_row_selected_under():
    d = decide_ou25(_row(), _reg(), T)
    assert d["selected"] and d["side"] == "UNDER"
    assert math.isclose(d["edge"], 0.60 - 1 / 1.95)


def test_missing_odds_rejected():
    d = decide_ou25(_row(odds_under25=float("nan"), odds_over25=float("nan")), _reg(), T)
    assert not d["selected"] and d["reason"] == "unpriced"
    with pytest.raises(V.AuditViolation):
        V.check_priced({"odds": None})
    with pytest.raises(V.AuditViolation):
        V.check_priced({"odds": 1.40, "price_source": "synthetic"})


def test_unpriced_side_market_prediction_not_selected():
    d = decide_side({"league": "Brazil Serie A", "p_btts": 0.70, "odds_btts": None,
                     "kickoff_utc": "2026-10-21T19:00:00Z"}, "btts", _reg(), T)
    assert not d["selected"] and d["reason"] == "unpriced"


def test_swapped_teams_flagged_not_silently_settled():
    res = pd.DataFrame([{"league": "Championship", "date": pd.Timestamp("2026-10-21"), "home_team": "Hull City",
                         "away_team": "Leeds United", "hg": 3, "ag": 0, "src": "fd"}])
    f = ResultIndex(res).find("Championship", "2026-10-21", "Leeds United", "Hull City")
    assert f["status"] == "SWAPPED" and (f["hg"], f["ag"]) == (0, 3)
    with pytest.raises(V.AuditViolation):
        V.check_result_orientation(f["status"])


def test_wrong_team_never_matched():
    res = pd.DataFrame([{"league": "Championship", "date": pd.Timestamp("2026-10-21"),
                         "home_team": "Manchester City", "away_team": "Leeds United", "hg": 1, "ag": 1, "src": "fd"}])
    f = ResultIndex(res).find("Championship", "2026-10-21", "Manchester United", "Leeds United")
    assert f["status"] == "NOT_FOUND"


def test_abbreviations_match_but_namesakes_do_not():
    res = pd.DataFrame([
        {"league": "Championship", "date": pd.Timestamp("2026-09-02"), "home_team": "QPR", "away_team": "Cardiff",
         "hg": 2, "ag": 1, "src": "fd"},
        {"league": "Championship", "date": pd.Timestamp("2026-09-06"), "home_team": "Birmingham",
         "away_team": "Wolves", "hg": 0, "ag": 0, "src": "fd"}])
    ix = ResultIndex(res)
    assert ix.find("Championship", "2026-09-02", "Queens Park Rangers", "Cardiff City")["status"] == "MATCHED"
    assert ix.find("Championship", "2026-09-06", "Birmingham City", "Wolverhampton Wanderers")["status"] == "MATCHED"
    assert ix.find("Championship", "2026-09-06", "Birmingham City", "Cardiff City")["status"] == "NOT_FOUND"


def test_inverted_under_probability_detected():
    with pytest.raises(V.AuditViolation):
        V.check_under_probability(0.40, 0.40)          # p_under planted as p_over
    V.check_under_probability(0.40, 0.60)
    # and the selector itself derives p_under = 1 - p_over: p_over 0.62 must never yield UNDER @1.95
    d = decide_ou25(_row(p_over25=0.62), _reg(), T)
    assert d["side"] != "UNDER"


def test_wrong_market_threshold_detected():
    with pytest.raises(V.AuditViolation):
        V.check_market_line("ou25", 1.5)
    with pytest.raises(V.AuditViolation):
        V.check_market_line("over35", 2.5)
    # settlement uses the market's own line: 2 goals
    assert outcome("over15", "OVER", 1, 1) == "WIN"
    assert outcome("ou25", "OVER", 1, 1) == "LOSS"
    assert outcome("ou25", "UNDER", 1, 1) == "WIN"
    assert outcome("over35", "OVER", 2, 1) == "LOSS"
    with pytest.raises(ValueError):
        outcome("ou25", "YES", 1, 1)                   # health-contract "YES/OVER" style side
    with pytest.raises(V.AuditViolation):
        V.check_side("ou25", "AVOID")                  # the 122 historical side=AVOID rows


def test_valuable_threshold_is_the_deployed_env_not_config_default():
    r = _row(p_over25=0.465, odds_under25=1.95)       # under edge = 0.535 - 0.5128 = 0.022
    assert decide_ou25(r, _reg(VALUABLE_THRESHOLD="0.02"), T)["selected"]
    assert decide_ou25(r, _reg(VALUABLE_THRESHOLD="0.03"), T)["reason"] == "below_threshold"


def test_duplicate_selections_rejected():
    bets = pd.DataFrame([{"fixture_id": "f1", "market": "ou25", "side": "UNDER"},
                         {"fixture_id": "f1", "market": "ou25", "side": "UNDER"}])
    with pytest.raises(V.AuditViolation):
        V.check_no_duplicates(bets)
    opp = pd.DataFrame([{"fixture_id": "f1", "market": "ou25", "side": "UNDER"},
                        {"fixture_id": "f1", "market": "over35", "side": "OVER"}])
    with pytest.raises(V.AuditViolation):
        V.check_one_direction_per_fixture(opp)


def test_replay_keeps_one_first_sight_decision(monkeypatch):
    from scripts.audit.independent_backtest import engine

    class TL:
        def at(self, t):
            return _reg()
    rows = []
    for i, (ts, ou) in enumerate([("2026-10-20T08:00:00Z", 1.95), ("2026-10-20T09:00:00Z", 2.10),
                                  ("2026-10-20T10:00:00Z", 2.20)]):
        r = _row(odds_under25=ou)
        r.update(date="2026-10-21", home_team="A", away_team="B", generated_at=ts, snap_commit_ts=ts,
                 snap_sha=f"s{i}")
        rows.append(r)
    fs, *_ = engine.replay(pd.DataFrame(rows), TL())
    ou = fs[fs["market"] == "ou25"]
    assert len(ou) == 1 and ou.iloc[0]["odds"] == 1.95 and ou.iloc[0]["n_boards_selected"] == 3


def test_same_player_name_different_fixtures():
    stats = pd.DataFrame([
        {"player_id": 1, "player_name": "Danilo", "fixture_id": "F1", "date": "2026-10-20", "minutes": 90, "goals": 1},
        {"player_id": 2, "player_name": "Danilo", "fixture_id": "F2", "date": "2026-10-20", "minutes": 90, "goals": 0}])
    with pytest.raises(V.AuditViolation):
        V.check_player_identity({"player": "Danilo", "date": "2026-10-20"}, stats)
    with pytest.raises(V.AuditViolation):
        V.settle_player_prop({"player": "Danilo", "stat": "goals", "line": 0.5}, stats)
    assert V.settle_player_prop({"player_id": 2, "fixture_id": "F2", "stat": "goals", "line": 0.5}, stats) == "LOSS"
    assert V.settle_player_prop({"player_id": 1, "fixture_id": "F1", "stat": "goals", "line": 0.5}, stats) == "WIN"


def test_missing_player_stats_is_unsettled_not_loss():
    stats = pd.DataFrame([{"player_id": 9, "player_name": "X", "fixture_id": "F9", "date": "2026-10-20",
                           "minutes": float("nan"), "goals": float("nan")}])
    assert V.settle_player_prop({"player_id": 9, "fixture_id": "F9", "stat": "goals", "line": 0.5}, stats) is None
    assert V.settle_player_prop({"player_id": 7, "fixture_id": "F9", "stat": "goals", "line": 0.5}, stats) is None
    assert pnl(None, 2.0) is None                     # unsettled never becomes -1 or 0


def test_post_kickoff_quote_rejected():
    d = decide_ou25(_row(kickoff_utc="2026-10-20T09:30:00Z"), _reg(), T)
    assert not d["selected"] and d["reason"] == "post_kickoff_quote"
    with pytest.raises(V.AuditViolation):
        V.check_pre_kickoff("2026-10-20T19:05:00Z", "2026-10-20T19:00:00Z")
    with pytest.raises(V.AuditViolation):
        V.check_pre_kickoff("2026-10-20T10:00:00Z", None)


def test_missing_model_artifact(tmp_path):
    with pytest.raises(V.AuditViolation):
        V.check_model_artifact(tmp_path / "model_v9_standard.pkl")
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    with pytest.raises(V.AuditViolation):
        V.check_model_artifact(empty)


def test_wrong_feature_order():
    feats = ["home_scored_last5", "away_scored_last5", "league_avg_goals"]
    V.check_feature_order(feats, ["x", *feats, "y"])
    with pytest.raises(V.AuditViolation):
        V.check_feature_order(feats, ["away_scored_last5", "home_scored_last5", "league_avg_goals"])
    with pytest.raises(V.AuditViolation):
        V.check_feature_order(feats, ["home_scored_last5", "away_scored_last5"])


def test_blind_fixture_guard_respects_regime():
    r = _row(home_scored_last5=float("nan"))
    assert decide_ou25(r, _reg(REQUIRE_FORM_DATA="1"), T)["reason"] == "blind_fixture_guard"
    # guard env-disabled but expired (UNTIL in the past) -> still guarded
    assert decide_ou25(r, _reg(REQUIRE_FORM_DATA="0", REQUIRE_FORM_DATA_UNTIL="2026-09-15"), T)["reason"] \
        == "blind_fixture_guard"
    assert decide_ou25(r, _reg(REQUIRE_FORM_DATA="0", REQUIRE_FORM_DATA_UNTIL="2026-12-01"), T)["selected"]


def test_nf_price_cap_and_min_odds():
    nf = _row(model_type="new_format", p_over25=0.55, odds_over25=3.2, odds_under25=1.30)
    after_cap = pd.Timestamp("2026-10-23T10:00:00Z")
    nf["kickoff_utc"] = "2026-10-24T19:00:00Z"
    assert decide_ou25(nf, _reg(), after_cap)["reason"] == "price_bound_blocked"
    assert decide_ou25(nf, _reg(), T)["selected"]      # cap is dated: not applied before 2026-10-22
    lo = _row(p_over25=0.30, odds_under25=1.70)
    assert decide_ou25(lo, _reg(), T)["reason"] == "price_bound_blocked"


def test_void_and_pnl_rules():
    assert pnl("WIN", 2.5) == 1.5 and pnl("LOSS", 2.5) == -1.0 and pnl("VOID", 2.5) == 0.0


def test_real_production_artifacts_present():
    """Real data: the artifacts the deployed predict path loads exist at the audited ref."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "models"
    for f in ["model_v9_standard.pkl", "model_v9_standard_repaired.pkl", "model_v9_newformat.pkl",
              "model_v9_btts.pkl", "model_v9_over15.pkl", "model_v9_over35.pkl"]:
        V.check_model_artifact(root / f)
