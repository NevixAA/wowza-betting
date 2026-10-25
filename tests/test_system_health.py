"""System-health rules are explicit and fail loudly on the regressions the 2026-10-22 audit found."""
from datetime import datetime, timezone

import pandas as pd

from src import system_health as S

POLICY = {"tracks": {"new_format|BTTS": {"state": "ACTIVE"}, "standard|BTTS": {"state": "PAPER"}}}


def _plan(track, state, model_type="new_format", odds=2.0, market="BTTS"):
    return pd.DataFrame([{"market": market, "model_type": model_type, "track": track, "deployment_state": state,
                          "odds": odds, "generated_at": "2026-10-22 10:00:00"}])


def test_active_track_fully_unsupported_is_red():
    assert S.side_routing(_plan("unknown|BTTS", "UNSUPPORTED", "unknown"), POLICY)["status"] == "RED"
    assert S.side_routing(_plan("new_format|BTTS", "ACTIVE"), POLICY)["status"] == "GREEN"


def test_lost_model_type_or_odds_turns_mapping_red():
    assert S.stake_mapping(_plan("unknown|BTTS", "UNSUPPORTED", "unknown"))["status"] == "RED"
    assert S.stake_mapping(_plan("new_format|BTTS", "ACTIVE", odds=None))["status"] == "RED"
    assert S.stake_mapping(_plan("new_format|BTTS", "ACTIVE"))["status"] == "GREEN"


def test_registered_challenger_with_no_probabilities_is_red():
    now = datetime(2026, 10, 22, tzinfo=timezone.utc)
    sh = pd.DataFrame({"logged_at": ["2026-10-21T10:00:00Z"] * 3, "p_pro_hgb": [None, None, None]})
    assert S.challenger_coverage(sh, now)["status"] == "RED"
    sh["p_pro_hgb"] = 0.55
    assert S.challenger_coverage(sh, now)["status"] == "GREEN"


def test_settlement_and_clv_thresholds():
    assert S.shadow_settlement({"settlement_pct_of_finished": 79.0})["status"] == "RED"
    assert S.shadow_settlement({"settlement_pct_of_finished": 96.4})["status"] == "GREEN"
    sh = pd.DataFrame({"result": ["WIN"] * 10, "close_status": ["OK"] + ["STALE_CLOSE_GT_30MIN"] * 9})
    assert S.certified_clv(sh, {})["status"] == "RED"


def test_stale_manifest_and_workflows():
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    assert S.stale_artifacts({"generated_at": now, "n_stale_decision_facing": 3})["status"] == "RED"
    assert S.stale_artifacts({"generated_at": now, "n_stale_decision_facing": 0})["status"] == "GREEN"
    assert S.workflow_health({"generated_at": now, "failures_24h": 0, "runs_24h": 90})["status"] == "GREEN"
    assert S.workflow_health({"generated_at": now, "failures_24h": 4, "runs_24h": 90})["status"] == "RED"


def test_weekly_research_panel_is_red_when_protected_tracks_move():
    reg = {"tracked_leagues": ["A", "B"], "regime_counts": {"PROTECTED": 1, "REPAIR": 0, "LEARNING": 9}}
    ok = {"protected_tracks_modified": False, "changes": [], "unapproved_new_protected_tracks": []}
    assert S.weekly_research_panel(reg, ok)["status"] == "GREEN"
    moved = dict(ok, protected_tracks_modified=True, changes=["deployed_env.MARKSMAN_THRESHOLD: 0.08 -> 0.09"])
    c = S.weekly_research_panel(reg, moved)
    assert c["status"] == "RED" and "MARKSMAN_THRESHOLD" in c["detail"]
    assert S.weekly_research_panel(reg, {"error": "boom"})["status"] == "RED"          # unverifiable = fail closed
    assert S.weekly_research_panel(reg, dict(ok, unapproved_new_protected_tracks=["x"]))["status"] == "AMBER"
    assert "NO_DATA" in S.weekly_research_panel(reg, ok)["value"]


def test_every_check_states_its_rule():
    for c in S.evaluate():
        assert c["status"] in (S.GREEN, S.AMBER, S.RED, S.GREY) and c["rule"]
