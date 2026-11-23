"""Fantasy V2 promotion rule: pre-registered eligibility check, never a promotion.

    python -m player_model.fantasy_promotion_rule      # prints the status from output/fantasy_v2_oos.csv

THE RULE (owner decision 2026-11-23, committed in config/fantasy_v2_promotion_rule.json before the
evidence exists). A candidate residual arm is ELIGIBLE only when, on its identical paired population
with the FPL baseline (completed gameweeks, both arms forecast the same rows, chronology checked):
  1. at least 10 completed gameweeks are scored;
  2. the 90% gameweek-block bootstrap interval of MAE(candidate) - MAE(FPL) lies entirely below 0;
  3. the candidate's total captain regret is no worse than FPL's.

ELIGIBLE IS NOT PROMOTED. This module only reads files and returns a status. It never reads or
writes FANTASY_V2_RESIDUAL_MODE, the workflow, or any served artifact. Promotion needs an explicit
owner approval and a separate reviewed change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from player_model import fantasy_paired_eval as pe

ROOT = Path(__file__).resolve().parents[1]
RULE_PATH = ROOT / "config" / "fantasy_v2_promotion_rule.json"
ELIGIBLE = "ELIGIBLE_AWAITING_OWNER_APPROVAL"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
NO_DATA = "NO_DATA"


def load_rule(path: Path | None = None) -> dict:
    return json.loads((path or RULE_PATH).read_text(encoding="utf-8"))


def evaluate_arm(oos: pd.DataFrame, arm: str, rule: dict) -> dict:
    """Eligibility of one candidate arm against the FPL baseline on their paired rows."""
    base = rule["baseline_arm"]
    cond = rule["conditions"]
    need_gws = int(cond["min_completed_gameweeks"])
    ci_cfg = cond["mae_diff_interval"]
    out = {"arm": arm, "baseline": base, "status": NO_DATA, "eligible": False,
           "auto_promote": False, "min_completed_gameweeks": need_gws}
    if arm not in oos.columns:
        out["reasons"] = [f"no {arm} forecasts in the walk-forward file yet"]
        return out
    pop, excl = pe.paired_population(oos, [base, arm])
    out["exclusions"] = excl
    if pop.empty:
        out["reasons"] = ["no paired completed rows"]
        return out
    sc = pe.score(pop, [base, arm])
    gws = sc["gameweeks"]
    dlt = pe.paired_delta(pop, arm, base=base, n=int(ci_cfg["resamples"]), seed=int(ci_cfg["seed"]))
    lo, hi = dlt["mae_ci90"]
    reg_arm = sc["pooled"][arm]["captain_regret_total"]
    reg_base = sc["pooled"][base]["captain_regret_total"]
    checks = {
        "completed_gameweeks": {"value": len(gws), "need": need_gws, "pass": len(gws) >= need_gws},
        "mae_diff_ci90_below_zero": {"point": dlt["mae_minus_fpl"], "ci90": [lo, hi],
                                     "pass": bool(hi < 0)},
        "captain_regret_no_worse": {"candidate": reg_arm, "baseline": reg_base,
                                    "pass": bool(reg_arm <= reg_base)},
    }
    ok = all(c["pass"] for c in checks.values())
    reasons = []
    if not checks["completed_gameweeks"]["pass"]:
        reasons.append(f"{len(gws)} of {need_gws} completed gameweeks")
    if not checks["mae_diff_ci90_below_zero"]["pass"]:
        reasons.append(f"MAE - FPL 90% interval {lo:+.3f} to {hi:+.3f} is not entirely below 0")
    if not checks["captain_regret_no_worse"]["pass"]:
        reasons.append(f"captain regret {reg_arm:.0f} vs FPL {reg_base:.0f}")
    out.update({"status": ELIGIBLE if ok else NOT_ELIGIBLE, "eligible": ok,
                "rows": int(len(pop)), "gameweeks": gws, "checks": checks, "reasons": reasons})
    return out


def evaluate(oos: pd.DataFrame, rule: dict | None = None) -> dict:
    """Status of every candidate arm named in the committed rule. Decides nothing."""
    rule = rule or load_rule()
    return pe._round({
        "rule_id": rule["rule_id"], "rule_version": rule["version"],
        "pre_registered_on": rule["pre_registered_on"],
        "owner_decision_date": rule["owner_decision"]["date"],
        "auto_promote": False,
        "note": rule["outcome"]["note"],
        "arms": {arm: evaluate_arm(oos, arm, rule) for arm in rule["candidate_arms"]},
    })


def main() -> int:
    if not pe.OOS.exists():
        print(f"no {pe.OOS.name}; nothing to evaluate")
        return 0
    res = evaluate(pd.read_csv(pe.OOS, low_memory=False))
    for arm, r in res["arms"].items():
        print(f"  {arm:11s} {r['status']}: {'; '.join(r.get('reasons') or []) or 'all conditions pass'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
