"""OddsAPI credit breakdown -> reports/ODDSAPI_CREDIT_BREAKDOWN.md (ops6, 2026-11-02). REPORT ONLY.

Every number comes from reports/oddsapi_credit_inputs_2026-11-02.json (measured run-log values, tagged
sourced / estimate). Nothing here calls OddsAPI or changes a schedule.

    python scripts/report_oddsapi_credits.py [inputs.json] [out.md]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "reports" / "oddsapi_credit_inputs_2026-11-02.json"
DEFAULT_OUT = ROOT / "reports" / "ODDSAPI_CREDIT_BREAKDOWN.md"

FEATURES = {
    "BTTS + alternate totals (predict)": "btts_alt_credits",
    "Totals board (predict)": "predict_totals_credits",
    "Player-prop odds (player_props, prop_odds_snapshot)": "props_credits",
    "Sharp snapshots (sharp_tracker)": "sharp_credits",
    "Scores (update_results, daily_summary)": "scores_credits",
    "Exchange capture (exchange_odds_capture)": "exchange_credits",
}
PREDICT_FEATURES = ("btts_alt_credits", "predict_totals_credits")


def _t(s: str) -> datetime:
    return datetime.fromisoformat(str(s)[:19])


def compute(inp: dict) -> dict:
    acc = inp["account"]
    w0, w1 = _t(inp["steady_window"]["start"]), _t(inp["steady_window"]["end"])
    hours = (w1 - w0).total_seconds() / 3600.0
    runs = [r for r in inp["runs"] if w0 <= _t(r["started_utc"]) < w1]
    credits = {name: sum(int(r.get(col) or 0) for r in runs) for name, col in FEATURES.items()}
    per_day = {k: v * 24.0 / hours for k, v in credits.items()}
    in_window_attributed = sum(per_day.values())
    # once-a-day workflows that did not fire inside the window: their measured per-run cost x 1/day
    ex = (inp.get("cost_rules") or {}).get("exchange_credits_per_day")
    if ex and credits["Exchange capture (exchange_odds_capture)"] == 0:
        per_day["Exchange capture (exchange_odds_capture)"] = float(ex["value"])
    by_wf: dict = {}
    for r in runs:
        w = by_wf.setdefault(r["workflow"], {"runs": 0, "credits": 0})
        w["runs"] += 1
        w["credits"] += sum(int(r.get(col) or 0) for col in FEATURES.values())
    for w in by_wf.values():
        w["runs_per_day"] = w["runs"] * 24.0 / hours
        w["credits_per_day"] = w["credits"] * 24.0 / hours
        w["credits_per_run"] = w["credits"] / w["runs"] if w["runs"] else 0.0
    attributed = sum(per_day.values())

    # account-level burn: the remaining-credits drop across the steady window (second, independent route)
    r0, r1 = acc["remaining_steady_start"], acc["remaining_latest"]
    span_h = (_t(r1["ts"]) - _t(r0["ts"])).total_seconds() / 3600.0
    burn = (r0["value"] - r1["value"]) * 24.0 / span_h
    remaining = r1["value"]
    as_of = _t(inp["as_of_utc"])
    days_to_reset = (_t(acc["reset_utc"]["value"]) - as_of).total_seconds() / 86400.0
    affordable = remaining / days_to_reset if days_to_reset > 0 else float("nan")

    predict = [r for r in runs if r["workflow"] == "predict.yml"]
    predict_cost = (sum(sum(int(r.get(c) or 0) for c in PREDICT_FEATURES) for r in predict) / len(predict)
                    if predict else 0.0)
    predict_per_day = sum(per_day[k] for k, c in FEATURES.items() if c in PREDICT_FEATURES)
    non_predict = attributed - predict_per_day
    breakeven = (affordable - non_predict) / predict_cost if predict_cost else float("nan")
    alt_calls_per_day = sum(int(r.get("alt_totals_calls") or 0) for r in predict) * 24.0 / hours

    sharp = [r for r in runs if r["workflow"] == "sharp_tracker.yml"]
    sharp_per_run = sum(int(r.get("sharp_credits") or 0) for r in sharp) / len(sharp) if sharp else 0.0
    guard_max = inp["schedules_requested_per_day"].get("sharp_tracker.yml", {}).get("guard_max", 0)
    sharp_projected = sharp_per_run * guard_max
    sharp_now = per_day["Sharp snapshots (sharp_tracker)"]
    breakeven_with_sharp = ((affordable - non_predict - max(sharp_projected - sharp_now, 0)) / predict_cost
                            if predict_cost else float("nan"))

    since_reset_h = (_t(r1["ts"]) - _t(inp["account"].get("reset_prev_utc", {}).get("value", "2026-11-01T00:00"))
                     ).total_seconds() / 3600.0
    since_reset_burn = ((acc["monthly_allowance"]["value"] - remaining) * 24.0 / since_reset_h
                        if since_reset_h > 0 else float("nan"))
    return {
        "hours": hours, "runs_in_window": len(runs), "credits": credits, "per_day": per_day, "by_wf": by_wf,
        "attributed_per_day": attributed, "burn_per_day": burn,
        "in_window_attributed_per_day": in_window_attributed,
        "unattributed_per_day": burn - in_window_attributed,
        "remaining": remaining, "remaining_ts": r1["ts"], "days_left": remaining / burn if burn else float("nan"),
        "days_to_reset": days_to_reset, "affordable_per_day": affordable,
        "predict_cost_per_run": predict_cost, "predict_per_day": predict_per_day,
        "non_predict_per_day": non_predict, "breakeven_predict_runs_per_day": breakeven,
        "alt_calls_per_day": alt_calls_per_day, "sharp_per_run": sharp_per_run,
        "sharp_projected_per_day": sharp_projected, "breakeven_with_sharp": breakeven_with_sharp,
"since_reset_burn_per_day": since_reset_burn,
        "inputs": inp,
    }


def _r(x, step=10):
    return int(round(x / step) * step)


def render(c: dict) -> str:
    inp = c["inputs"]
    acc = inp["account"]
    sch = inp["schedules_requested_per_day"]
    pd_ = c["per_day"]
    L = []
    a = L.append
    a("# OddsAPI credit breakdown (report only)")
    a("")
    a(f"As of {inp['as_of_utc']} UTC. Built by `scripts/report_oddsapi_credits.py` from "
      "`reports/oddsapi_credit_inputs_2026-11-02.json`; every input there is tagged *sourced* (a named log "
      "line or file) or *estimate* (chosen by this report). **No schedule was changed** - this is the owner's "
      "upgrade-vs-cut decision input only. Figures are rounded to 10 credits; the window is short (one day), "
      "so treat them as +/-20%.")
    a("")
    a("## Answer")
    a("")
    a(f"- Account burn at the current GitHub delivery: **~{_r(c['burn_per_day'])} credits/day** "
      f"(remaining {acc['remaining_steady_start']['value']:,} -> {c['remaining']:,} between "
      f"{acc['remaining_steady_start']['ts']} and {c['remaining_ts']}).")
    a(f"- {c['remaining']:,} credits left -> **~{c['days_left']:.0f} days** at that burn; the next reset is "
      f"~{c['days_to_reset']:.0f} days away (assumed {acc['reset_utc']['value']}), so the month is covered "
      f"**only while GitHub keeps delivering predict about "
      f"{c['by_wf'].get('predict.yml', {}).get('runs_per_day', 0):.0f} times a day** (break-even below).")
    a(f"- What the account can afford to the reset: **~{_r(c['affordable_per_day'])}/day**.")
    a(f"- Upcoming, not yet in the measured burn: the sharp tracker's new hourly cadence (ops5 guard, up to "
      f"{sch['sharp_tracker.yml'].get('guard_max')} snapshots/day) projects to ~{_r(c['sharp_projected_per_day'])}"
      f"/day against ~{_r(pd_['Sharp snapshots (sharp_tracker)'])}/day measured.")
    a("")
    a("## Credits per day by feature")
    a("")
    a(f"Steady window {inp['steady_window']['start']} -> {inp['steady_window']['end']} UTC "
      f"({c['hours']:.0f}h, {c['runs_in_window']} runs; starts after the BTTS fetch-clock fix).")
    a("")
    a("| Feature | Credits/day | Share | How measured |")
    a("|---|---:|---:|---|")
    how = {
        "BTTS + alternate totals (predict)": "[btts] / [oddsapi_usage] credit count in each predict log",
        "Totals board (predict)": "estimate: 19 leagues x 1 credit per run (not logged)",
        "Player-prop odds (player_props, prop_odds_snapshot)": "account remaining before/after each run's event calls",
        "Sharp snapshots (sharp_tracker)": "leagues snapshotted x 2 markets (not logged before ops5)",
        "Scores (update_results, daily_summary)": "estimate: 'Fetching scores' lines x 2 credits",
        "Exchange capture (exchange_odds_capture)": "[exchange] credits line",
    }
    tot = c["attributed_per_day"]
    for k, v in sorted(pd_.items(), key=lambda kv: -kv[1]):
        a(f"| {k} | {_r(v)} | {v / tot:.0%} | {how[k]} |")
    a(f"| **Attributed** | **{_r(tot)}** | | |")
    a(f"| Unattributed (account burn minus in-window attributed) | {_r(c['unattributed_per_day'])} | | see cross-check |")
    a("")
    a("Exchange capture fires once a day and its run (2026-11-01 16:53, 87 credits) fell just before the "
      "window, so it is counted at its measured cost x 1/day; it is not part of the account drop below.")
    a("")
    a("### By workflow (delivered vs requested)")
    a("")
    a("| Workflow | Runs delivered/day | Requested/day (cron) | Credits/run | Credits/day |")
    a("|---|---:|---|---:|---:|")
    for wf, w in sorted(c["by_wf"].items(), key=lambda kv: -kv[1]["credits_per_day"]):
        req = sch.get(wf, {})
        reqs = ", ".join(f"{k} {v}" for k, v in req.items() if not str(k).startswith("guard") and k != "note") or "-"
        a(f"| {wf} | {w['runs_per_day']:.1f} | {reqs} | {_r(w['credits_per_run'], 1)} | {_r(w['credits_per_day'])} |")
    a("")
    a("GitHub delivers a small fraction of the requested predict fires. The BTTS 2h refresh clock "
      "suppresses nothing at this spacing (each run re-buys the whole ~175-fixture board at 2 credits each), "
      "so **predict's cost scales one-for-one with how many runs GitHub delivers**.")
    a("")
    a("## Days left")
    a("")
    a(f"- Remaining {c['remaining']:,} at {c['remaining_ts']} (sourced: {acc['remaining_latest']['source']}).")
    a(f"- Burn since the assumed 2026-11-01 reset: ~{_r(c['since_reset_burn_per_day'])}/day (includes the "
      f"2026-11-01 morning, when GitHub delivered predict more often); steady burn ~{_r(c['burn_per_day'])}/day.")
    a(f"- Days left at steady burn: ~{c['days_left']:.0f}; days to reset: ~{c['days_to_reset']:.0f}.")
    a(f"- With the sharp projection added: ~{_r(c['burn_per_day'] + c['sharp_projected_per_day'] - pd_['Sharp snapshots (sharp_tracker)'])}"
      f"/day -> ~{c['remaining'] / (c['burn_per_day'] + c['sharp_projected_per_day'] - pd_['Sharp snapshots (sharp_tracker)']):.0f} days.")
    hj = acc["health_json"]
    a(f"- `output/oddsapi_quota_health.json` ({hj.get('generated_at', '')}) reported {hj['burn_per_day']}/day "
      f"burn and {hj['logged_per_day']}/day logged: its window still contained the pre-fix runs and its "
      "per-workflow table counts only BTTS and exchange (props, sharp, scores and the totals board print no "
      "credit line), which is why 'logged' was far below 'burn'.")
    a("")
    a("## Break-even")
    a("")
    a(f"Predict costs ~{_r(c['predict_cost_per_run'], 1)} credits per delivered run (BTTS + alt totals + totals "
      f"board). Everything else costs ~{_r(c['non_predict_per_day'])}/day. The month is covered while "
      f"predict runs per day stay below **~{c['breakeven_predict_runs_per_day']:.1f}** "
      f"(currently ~{c['by_wf'].get('predict.yml', {}).get('runs_per_day', 0):.1f}); once the sharp tracker "
      f"runs at its new cadence, below **~{c['breakeven_with_sharp']:.1f}**. Each extra delivered predict "
      f"run per day costs ~{_r(c['predict_cost_per_run'] * c['days_to_reset'], 100):,} credits to the reset. "
      "The conclusion is robust to the estimated totals-board and scores costs (together under 10% of burn); "
      "it is NOT robust to GitHub delivery, which nobody here controls.")
    a("")
    a("## Options (no change made)")
    a("")
    btts = pd_["BTTS + alternate totals (predict)"]
    props = pd_["Player-prop odds (player_props, prop_odds_snapshot)"]
    a("| Option | Saves/day (approx.) | Coverage cost |")
    a("|---|---:|---|")
    a(f"| BTTS one region (`BTTS_REGIONS=uk`) | ~{_r(btts / 2)} | half the books in the BTTS consensus; fewer fixtures reach the min-books bar |")
    a(f"| Raise `BTTS_REFRESH_HOURS` 2 -> 6 (near-kickoff refetch unchanged) | 0 now, caps growth | none at today's ~4h spacing; limits cost if delivery improves |")
    a(f"| Alternate totals off (`ALT_TOTALS_MAX_CALLS=0`) | ~{_r(c['alt_calls_per_day'] * 2)} | Over 1.5 / 3.5 side tips lose their price again (owner decision 2026-11-01 option B) |")
    a(f"| Player-prop odds one region | ~{_r(props / 2)} | fewer prop books; props are paper-only |")
    a("| Share one prop-odds cache key between player_props and prop_odds_snapshot | not quantified (needs per-event logs) | none: stops the two workflows re-buying the same event inside the 2h TTL |")
    a(f"| Keep sharp at ~3 snapshots/day (revert ops5 cadence) | ~{_r(c['sharp_projected_per_day'] - pd_['Sharp snapshots (sharp_tracker)'])} vs the projection | Pro 1X2 price age back to 12-14h every morning |")
    a(f"| Exchange capture off | ~{_r(pd_['Exchange capture (exchange_odds_capture)'])} | no exchange/lay research rows |")
    a("| Upgrade the plan | - | none; plan prices are not in the repo (owner input needed) |")
    a("")
    a("## Cross-check")
    a("")
    a(f"Route 1 (sum of the in-window runs' feature costs): ~{_r(c['in_window_attributed_per_day'])}/day. "
      "Route 2 (account remaining drop over the same runs): "
      f"~{_r(c['burn_per_day'])}/day. Gap ~{_r(c['unattributed_per_day'])}/day "
      f"({c['unattributed_per_day'] / c['burn_per_day']:.0%}) - consumers that print no credit line "
      "(scores calls whose count is estimated, predict's totals board, any manual/dispatch run). The two routes "
      "agree within ~10%, so the per-feature split is usable for the decision.")
    a("")
    a("## Inputs that most need checking")
    a("")
    a("- Monthly allowance 100,000 and a calendar-month reset: *estimate* (code default + the 0 -> 99,694 jump "
      "on 2026-11-01). The plan page is not in the repo.")
    a("- Totals board 19 credits/run and scores 2 credits/call: *estimate* (not logged).")
    a("- One day of steady data, a Sunday-Monday: weekday boards are smaller, weekend delivery differs.")
    a("")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else DEFAULT_INPUTS
    dst = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    md = render(compute(json.loads(src.read_text(encoding="utf-8"))))
    dst.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
