"""ops6 item 5: the OddsAPI credit breakdown is computed by a script from a committed inputs file
(no number typed into the report), and the arithmetic is checked on a hand-sized example."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "reports" / "oddsapi_credit_inputs_2026-11-02.json"


def _toy():
    return {
        "as_of_utc": "2026-11-02T00:00",
        "account": {
            "remaining_latest": {"value": 900, "ts": "2026-11-02T00:00"},
            "remaining_steady_start": {"value": 1000, "ts": "2026-11-01T00:00"},
            "monthly_allowance": {"value": 2000},
            "reset_utc": {"value": "2026-11-11T00:00"},
            "health_json": {"burn_per_day": 0, "logged_per_day": 0},
        },
        "steady_window": {"start": "2026-11-01T00:00", "end": "2026-11-02T00:00"},
        "cost_rules": {"sharp_credits_per_league": {"value": 2}},
        "schedules_requested_per_day": {"sharp_tracker.yml": {"guard_max": 4}},
        "runs": [
            {"workflow": "predict.yml", "started_utc": "2026-11-01T01:00", "btts_alt_credits": 40,
             "alt_totals_calls": 5, "predict_totals_credits": 10},
            {"workflow": "predict.yml", "started_utc": "2026-11-01T13:00", "btts_alt_credits": 40,
             "alt_totals_calls": 5, "predict_totals_credits": 10},
            {"workflow": "sharp_tracker.yml", "started_utc": "2026-11-01T05:00", "sharp_credits": 4},
            {"workflow": "update_results.yml", "started_utc": "2026-10-31T05:00", "scores_credits": 999},
        ],
    }


def test_math_on_a_hand_sized_example():
    from scripts.report_oddsapi_credits import compute
    c = compute(_toy())
    assert c["per_day"]["BTTS + alternate totals (predict)"] == pytest.approx(80)
    assert c["per_day"]["Totals board (predict)"] == pytest.approx(20)
    assert c["per_day"]["Sharp snapshots (sharp_tracker)"] == pytest.approx(4)
    assert c["per_day"]["Scores (update_results, daily_summary)"] == 0      # outside the window
    assert c["attributed_per_day"] == pytest.approx(104)
    assert c["burn_per_day"] == pytest.approx(100)                         # 1000 -> 900 in 24h
    assert c["days_left"] == pytest.approx(9.0)
    assert c["days_to_reset"] == pytest.approx(9.0)
    assert c["affordable_per_day"] == pytest.approx(100)
    assert c["predict_cost_per_run"] == pytest.approx(50)
    # break-even predict runs/day = (affordable - non-predict uses) / cost per run
    assert c["breakeven_predict_runs_per_day"] == pytest.approx((100 - 4) / 50)
    assert c["sharp_projected_per_day"] == pytest.approx(4 * 4)


def test_report_renders_from_the_committed_inputs():
    import json
    from scripts.report_oddsapi_credits import compute, render
    md = render(compute(json.loads(INPUTS.read_text(encoding="utf-8"))))
    for s in ("Credits per day by feature", "Days left", "Options", "Break-even", "Cross-check",
              "No schedule was changed"):
        assert s in md
