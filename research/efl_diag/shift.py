"""Train vs current-board distribution of EFL Standard features: training window (< 2026-04-01)
vs the newest weeks (2026-09-01..2026-11-08, the rows the board's features look like).
Missing rate, mean and standardized mean difference. No prices."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrain_harness import protocol as P
out = Path(sys.argv[1])
feat = pd.read_parquet(out / "feat.parquet"); cfg = P.load_config()
feats, _ = P.feature_list(cfg, "standard", feat.columns)
d = pd.to_datetime(feat.date)
E = ["Championship", "League One", "League Two"]
res = []
for lg in E:
    tr = feat[(feat.league == lg) & (d < "2026-04-01")]
    nw = feat[(feat.league == lg) & (d >= "2026-09-01") & (d <= "2026-11-08")]
    for c in feats:
        a, b = pd.to_numeric(tr[c], errors="coerce"), pd.to_numeric(nw[c], errors="coerce")
        sd = a.std()
        res.append({"league": lg, "feature": c, "train_missing": round(a.isna().mean(), 3), "now_missing": round(b.isna().mean(), 3),
                    "train_mean": round(a.mean(), 3), "now_mean": round(b.mean(), 3),
                    "smd": round((b.mean() - a.mean()) / sd, 2) if sd and sd > 0 else None, "n_train": len(a), "n_now": len(b)})
r = pd.DataFrame(res)
r.to_csv(out / "exp" / "train_vs_now_shift.csv", index=False)
flag = r[(r.smd.abs() >= 0.3) | ((r.train_missing - r.now_missing).abs() >= 0.3)]
print(flag.to_string(index=False))
