"""HT-score coverage in the committed history banks, by track, overall and since 2026-07-01. Read only, column-projected.
usage: python ht_banks.py <worktree>"""
import sys, json
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
wt = Path(sys.argv[1]); sys.path.insert(0, str(wt))
res = {}
try:
    import config
    nf = set(getattr(config, "NEW_FORMAT_LEAGUES", []) or [])
except Exception as e:
    nf = set(); res["cfg_err"] = str(e)
res["n_nf_leagues_cfg"] = len(nf)
for f in ("af_history.parquet", "fd_history.parquet"):
    p = wt / "output" / f
    cols = pq.ParquetFile(p).schema.names
    ht = [c for c in cols if "ht" in c.lower() and ("goal" in c.lower() or "score" in c.lower() or c.lower() in ("hthg", "htag"))]
    use = [c for c in ("date", "league") if c in cols] + ht[:2]
    d = pd.read_parquet(p, columns=use)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    r = {"rows": len(d), "ht_cols": ht, "max_date": str(d["date"].max())}
    if ht:
        d["has_ht"] = d[ht[0]].notna()
        d["track"] = d["league"].astype(str).map(lambda x: "new_format" if x in nf else "standard")
        for tr, g in d.groupby("track"):
            rec = g[g["date"] >= "2026-07-01"]
            r[tr] = {"n": len(g), "ht_pct": round(g["has_ht"].mean() * 100, 1),
                     "n_since_jul": len(rec), "ht_pct_since_jul": round(rec["has_ht"].mean() * 100, 1) if len(rec) else None}
    res[f] = r
print(json.dumps(res, indent=1, default=str))
