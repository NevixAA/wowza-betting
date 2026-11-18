"""Independent forward ROI recompute (flat, per market x track, staked tiers SNIPER/MARKSMAN) from ledgers.
usage: python roi.py <worktree> [as_of]  -- read only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
wt = Path(sys.argv[1]); out = wt / "output"
CUT = pd.Timestamp("2026-08-10")
fr = []
m = pd.read_csv(out / "bets_ledger.csv", low_memory=False); m["market"] = "ou25"; fr.append(m)
s = pd.read_csv(out / "side_bets_ledger.csv", low_memory=False); fr.append(s)
d = pd.concat(fr, ignore_index=True)
res = {"rows_main_ledger": len(m), "rows_side_ledger": len(s)}
d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
res["results"] = d["result"].astype(str).str.upper().value_counts().to_dict()
st = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"]) & d["pnl"].notna() & (d["match_date"] >= CUT)]
res["settled_since_cutoff"] = len(st)
res["max_match_date"] = str(st["match_date"].max().date())
stk = st[st["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))]
cells = {}
for (mk, mt), g in stk.groupby([stk["market"].astype(str), stk["model_type"].astype(str)]):
    p = g["pnl"].to_numpy(float); n = len(p)
    se = p.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    # second method: unit-stake ROI from WIN/LOSS and odds (flat 1u), independent of the pnl column
    o = pd.to_numeric(g.get("odds"), errors="coerce").to_numpy(float)
    win = (g["result"].astype(str).str.upper() == "WIN").to_numpy()
    flat = np.where(win, o - 1, -1.0)
    cells[f"{mk}|{mt}"] = {"n": n, "pnl": round(p.sum(), 2), "roi": round(p.mean(), 4),
                           "iid_ci95": [round(p.mean() - 1.96 * se, 4), round(p.mean() + 1.96 * se, 4)] if n > 1 else None,
                           "flat_from_odds_roi": round(np.nanmean(flat), 4), "odds_missing": int(np.isnan(o).sum()),
                           "pnl_vs_flat_maxdiff": round(float(np.nanmax(np.abs(p - flat))) if n else 0, 4)}
res["staked_cells"] = cells
# real-money (ACTIVE) tracks from policy, if stake columns exist
res["has_stake_cols"] = [c for c in d.columns if "stake" in c.lower()][:10]
print(json.dumps(res, indent=1, default=str))
