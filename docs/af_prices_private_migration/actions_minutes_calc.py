"""Option 3 cost: what moving the public repos to PRIVATE does to GitHub Actions billing.
Reads every number from actions_minutes_assumptions.json; nothing numeric is hard-coded here."""
import json, pathlib
A = json.loads((pathlib.Path(__file__).parent / "actions_minutes_assumptions.json").read_text())
v = lambda k: A[k]["value"]
run_min = {r: d["value"] for r, d in A["measured_run_minutes_30d"].items()}
ratio = {r: d["value"] for r, d in A["job_sample_ratio_billable_over_run_estimate"].items()}

billable_min_per_month = {r: run_min[r] * ratio[r] for r in run_min}
existing_private_min_per_month = v("private_minutes_used_nov_1_to_16") * 30 / v("days_elapsed_in_november")
included_min = v("included_private_minutes_per_month")
budget_usd = v("budget_usd_per_month")

def scenario(repos, rate_usd_per_min):
    public_min = sum(billable_min_per_month[r] for r in repos)
    total_min = public_min + existing_private_min_per_month
    overage_min = max(0.0, total_min - included_min)
    cost_usd = overage_min * rate_usd_per_min
    cap_min = included_min + budget_usd / rate_usd_per_min           # what $15 buys on top
    return public_min, total_min, cost_usd, cap_min

main3 = ["wowza-betting", "wowzaV9-Pro", "wowza-dashboard"]
all4 = main3 + ["wowza_v11"]
out = {}
for label, repos in (("betting+Pro+dashboard", main3), ("all four public repos", all4)):
    for rate_key in ("linux_rate_usd_per_min", "linux_rate_usd_per_min_alt"):
        rate = v(rate_key)
        pub, tot, cost, cap = scenario(repos, rate)
        out[f"{label} @ ${rate}/min"] = dict(public_min=round(pub), total_private_min=round(tot),
            overage_cost_usd_per_month=round(cost), minutes_affordable_within_budget=round(cap),
            cut_needed_pct=round(100 * max(0.0, 1 - (cap - existing_private_min_per_month) / pub)))

# Cross-check by a second route: raw run-estimate minutes (no job-sample correction) must land
# within ~15% of the job-corrected figure, otherwise one of the two methods is wrong.
raw = sum(run_min[r] for r in main3); corr = sum(billable_min_per_month[r] for r in main3)
out["cross_check"] = dict(raw_run_min=raw, job_corrected_min=round(corr), diff_pct=round(100 * (corr - raw) / raw, 1),
                          passed=abs(corr - raw) / raw < 0.15)
# Break-even: the per-minute price at which the current usage would fit in 3,000 min + $15.
pub = sum(billable_min_per_month[r] for r in main3)
over = pub + existing_private_min_per_month - included_min
out["break_even_rate_usd_per_min"] = round(budget_usd / over, 5)
out["existing_private_min_per_month_extrapolated"] = round(existing_private_min_per_month)
print(json.dumps(out, indent=1))
