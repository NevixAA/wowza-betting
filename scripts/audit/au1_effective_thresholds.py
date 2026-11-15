"""AU1 Phase 7 - EFFECTIVE O/U 2.5 tier floors as production computes them (READ-ONLY).

Never reads a threshold from config.py text. It imports config + src.betting in a CHILD process whose
environment is exactly a given regime's predict.yml env, then probes betting._base_tier on an edge grid
(0.000 .. 0.400 step 0.0005) for every enabled league x side. The lowest edge that yields each tier is
the effective floor; a non-monotone ladder (e.g. a ceiling) shows up as a tier change above SNIPER.

Regimes (from reports/au1_data/config_timeline.csv):
  R_pre0821   : no env overrides (config defaults)        2026-06-17 .. 2026-08-21 19:06 +03
  R_post0821  : LEAGUE_SNIPER_CAP=0.12 MARKSMAN=0.08 VALUABLE=0.03   2026-08-21 .. now
Both use the CURRENT models/best_params_standard.json (its own history is in artifact_epochs.csv).

Writes reports/au1_data/effective_thresholds.csv
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "au1_data"
REGIMES = {
    "R_pre0821_defaults": {},
    "R_post0821_predict_yml": {"LEAGUE_SNIPER_CAP": "0.12", "MARKSMAN_THRESHOLD": "0.08", "VALUABLE_THRESHOLD": "0.03"},
}
CHILD = r"""
import json, sys
sys.path.insert(0, %r)
import numpy as np, config
from src import betting as B
grid = [round(x, 4) for x in np.arange(0, 0.4001, 0.0005)]
rows = []
for lg in sorted(config.ENABLED_LEAGUES):
    for side in ("OVER", "UNDER"):
        ladder = [B._base_tier(e, side, lg) for e in grid]
        first = {}
        for e, t in zip(grid, ladder):
            first.setdefault(t, e)
        changes = [(grid[i], ladder[i]) for i in range(1, len(grid)) if ladder[i] != ladder[i-1]]
        rows.append({"league": lg, "side": side, "model_type": config.model_type_for_league(lg),
                     "valuable_floor": first.get("VALUABLE"), "marksman_floor": first.get("MARKSMAN"),
                     "sniper_floor": first.get("SNIPER"),
                     "ladder_changes": ";".join(f"{e}:{t}" for e, t in changes),
                     "non_monotone": any(t != "SNIPER" for e, t in changes if first.get("SNIPER") is not None and e > first["SNIPER"])})
print(json.dumps(rows))
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, env in REGIMES.items():
        e = {k: v for k, v in os.environ.items()
             if k not in ("LEAGUE_SNIPER_CAP", "MARKSMAN_THRESHOLD", "VALUABLE_THRESHOLD", "SNIPER_THRESHOLD",
                          "SNIPER_THRESHOLD_OVER", "SNIPER_THRESHOLD_UNDER", "EDGE_CEILING")}
        e.update(env)
        e["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable, "-c", CHILD % str(ROOT)], env=e, cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            raise SystemExit(r.stderr[-3000:])
        d = pd.DataFrame(json.loads(r.stdout.strip().splitlines()[-1]))
        d.insert(0, "regime", name)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "effective_thresholds.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
