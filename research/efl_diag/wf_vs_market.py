"""P11 champion walk-forward + holdout OOS predictions vs the de-vigged bet-time market and a trailing
league base rate, per Standard league x market. Metrics only (no prices).
    python research/efl_diag/wf_vs_market.py <priced_universe.parquet> <out_csv>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.efl_diag.experiments import metrics, boot_diff, ll_rows  # noqa
from src.market_anchor import power_devig

u = pd.read_parquet(sys.argv[1])
def dv(a, b):
    out = np.full(len(a), np.nan)
    for i in np.flatnonzero((a > 1) & (b > 1)):
        r = power_devig([a[i], b[i]]); out[i] = r[0] if len(r) == 2 else np.nan
    return out
u = u.sort_values("date").reset_index(drop=True)
u["pm"] = dv(u["yes"].to_numpy(float), u["no"].to_numpy(float))
# trailing league base rate: expanding mean of y over strictly earlier dates, per league x market
u["day"] = pd.to_datetime(u["date"]).dt.normalize()
d = u.groupby(["market", "league", "day"])["y"].agg(["sum", "count"])
pr = d.groupby(level=[0, 1]).cumsum() - d
u["pb"] = (pr["sum"] / pr["count"]).reindex(pd.MultiIndex.from_frame(u[["market", "league", "day"]])).to_numpy()
rows = []
for (st, m, lg), g in u.groupby(["stage", "market", "league"]):
    y = g.y.to_numpy(float)
    for name, col in (("champion", "p"), ("market_bettime", "pm"), ("base_trailing", "pb")):
        rows.append({"stage": st, "market": m, "league": lg, "model": name, **metrics(y, g[col].to_numpy(float))})
    priced = np.isfinite(g.pm.to_numpy())
    rows.append({"stage": st, "market": m, "league": lg, "model": "champion_on_priced_rows", **metrics(y[priced], g.p.to_numpy()[priced])})
    for ref in ("pm", "pb"):
        b = boot_diff(y, g.p.to_numpy(float), g[ref].to_numpy(float), g.date.to_numpy(), B=1000)
        if b:
            rows.append({"stage": st, "market": m, "league": lg, "model": f"champion_minus_{ref}", **b})
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print("written")
