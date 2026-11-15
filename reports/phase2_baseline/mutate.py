"""Can the GREEN checks fail? In-memory mutations only (no file written). usage: python mutate.py <worktree>"""
import os, sys, json
wt = sys.argv[1]; os.chdir(wt); sys.path.insert(0, wt)
from src.integrity import data_truth as DT
out = {}
# 1 a new invented default appears in production code
orig = DT.scan_defaults
DT.scan_defaults = lambda: [{"file": "src/predict.py", "kind": "fillna_literal", "code": "x = y.fillna(1.5)"}]
c = DT.defaults(); out["defaults_with_injected_hit"] = c["status"]
fails = DT.gate(checks=[c]); out["gate_with_injected_hit"] = fails[:2]
DT.scan_defaults = orig
# 2 ratchet raised
out["gate_ceiling_raised"] = DT.gate(previous_ceiling=-1)[:1]
# 3 one-match: inject a duplicate
import pandas as pd
fd = DT._bank_matches("fd_history.parquet").head(2000)
dup = fd.iloc[[0]].copy(); dup["date"] = dup["date"] + pd.Timedelta(days=1)
c5 = DT.one_match({"fd_cache": pd.concat([fd, dup], ignore_index=True)})
out["one_match_with_dup"] = [c5["status"], c5["value"][:120]]
print(json.dumps(out, indent=1, default=str))
