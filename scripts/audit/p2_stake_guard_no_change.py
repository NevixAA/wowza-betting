"""Phase 2 check: the 2026-11-03 stake-contract guards change no stake on real tips.

Re-sizes the selections in output/stake_plan.csv @ --ref (the tips production sized last) with
src/bankroll_policy.py as it was at --base (before the guards) and as it is in the working tree, under
the committed policy, and compares final_fraction / reason row by row. Pure, no network.

    python scripts/audit/p2_stake_guard_no_change.py --ref origin/main --base origin/data-truth-100
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GIT = r"C:\Program Files\Git\cmd\git.exe"


def show(ref, path) -> bytes:
    return subprocess.run([GIT, "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def main(ref: str, base: str, file: str = "output/stake_plan.csv") -> dict:
    plan = pd.read_csv(io.BytesIO(show(ref, file)))
    sel_cols = ["date", "fixture", "league", "model_type", "market", "side", "signal_tier", "odds"]
    sel = plan[sel_cols].copy()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bankroll_policy_base.py"
        p.write_bytes(show(base, "src/bankroll_policy.py"))
        spec = importlib.util.spec_from_file_location("bankroll_policy_base", p)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    import src.bankroll_policy as new
    pol = new.load_policy()
    res = {}
    for name, mod in (("base", old), ("new", new)):
        res[name] = pd.concat([mod.size_day(g, pol, bankroll=100.0, peak=100.0) for _, g in sel.groupby("date")],
                              ignore_index=True)
    a, b = res["base"], res["new"]
    diff = (a["final_fraction"].round(9) != b["final_fraction"].round(9)) | (a["reason"] != b["reason"])
    out = {"ref": ref, "base": base, "file": file, "rows": int(len(sel)), "staked_base": int((a["final_fraction"] > 0).sum()),
           "staked_new": int((b["final_fraction"] > 0).sum()),
           "sum_fraction_base": round(float(a["final_fraction"].sum()), 6),
           "sum_fraction_new": round(float(b["final_fraction"].sum()), 6),
           "rows_differing": int(diff.sum()),
           "differences": pd.concat([a.loc[diff, ["fixture", "market", "side", "final_fraction", "reason"]]
                                     .add_suffix("_base"), b.loc[diff, ["final_fraction", "reason"]]
                                     .add_suffix("_new")], axis=1).to_dict("records")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--base", default="origin/data-truth-100")
    a = ap.parse_args()
    r = {f: main(a.ref, a.base, f) for f in ("output/stake_plan.csv", "output/stake_audit.csv")}
    p = REPO / "reports" / "p2_data" / "stake_guard_no_change.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=1, default=str))
