"""Sample-size arithmetic for the NF Under release protocol (reports/P1_UNDER25_RELEASE_GATE.md).

Inputs come from reports/p2_under_data/under_analysis.json (committed); nothing is typed except the
target CI half-widths, which are protocol choices stated in the report. Two methods each: closed form
and simulation, so the figures are cross-checked.
    python scripts/p2_under_sample_size.py
"""
import json
import math
from pathlib import Path

import numpy as np

R = json.loads((Path(__file__).resolve().parents[1] / "reports" / "p2_under_data" / "under_analysis.json")
               .read_text(encoding="utf-8"))
p_over = R["shadow"]["all_settled"]["over_rate"]            # 0.5845 observed NF over rate
u = R["ledger"]["under_all_tiers"]
odds, hit = u["mean_odds"], u["breakeven_hit"]              # a break-even Under tip at the mean price
Z = 1.959964
rng = np.random.default_rng(20261103)
out = {}

# 1. fixtures needed so the 95% CI of the calibration bias (model - actual over rate) is +-4pp
sd_y = math.sqrt(p_over * (1 - p_over))
n_bias = math.ceil((Z * sd_y / 0.04) ** 2)
sim = [np.mean(rng.random(n_bias) < p_over) for _ in range(4000)]
out["bias_ci_pm4pp"] = {"n_closed_form": n_bias, "sim_halfwidth": round(1.959964 * float(np.std(sim)), 4)}

# 2. ROI 95% CI half-width at n Under tips, flat 1u, at the observed mean price and break-even hit
sd_r = odds * math.sqrt(hit * (1 - hit))
for n in (150, 300, 600, 1000):
    sims = [np.mean(np.where(rng.random(n) < hit, odds - 1, -1.0)) for _ in range(4000)]
    out[f"roi_halfwidth_n{n}"] = {"closed_form": round(Z * sd_r / math.sqrt(n), 4),
                                  "simulated": round(1.959964 * float(np.std(sims)), 4)}
# 3. tips needed for the ROI CI half-width to be 8pp
out["n_for_roi_halfwidth_8pp"] = math.ceil((Z * sd_r / 0.08) ** 2)
out["inputs"] = {"p_over": p_over, "mean_odds": odds, "breakeven_hit": hit}
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parents[1] / "reports" / "p2_under_data" / "sample_size.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
