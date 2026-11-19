"""AU2 Phase 3 - verify the 2026-10-28 Standard OU25 serving switch from committed evidence. READ-ONLY.

Walks every committed version of output/predictions.csv on origin/main between 2026-10-25 and the
audited commit and records, per board: generated_at, git_sha, Standard row count, serving_recipe
values, model_version values on Standard rows, and the presence of any fallback / failure stamp.
Also reads predict.yml at each commit that changed it (the env actually deployed).

    python scripts/audit/au2_switch_verify.py  ->  reports/au2_data/switch_boards.csv
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT  # noqa: E402

import pandas as pd  # noqa: E402

SINCE = "2026-10-25"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def main() -> int:
    revs = git("log", f"--since={SINCE}", "--format=%H %cI", AUDITED_COMMIT, "--",
               "output/predictions.csv").split("\n")
    rows = []
    for line in filter(None, revs):
        sha, ts = line.split()
        try:
            p = pd.read_csv(io.StringIO(git("show", f"{sha}:output/predictions.csv")), low_memory=False)
        except Exception as e:                                         # noqa: BLE001
            rows.append({"commit": sha[:9], "commit_ts": ts, "error": str(e)[:100]})
            continue
        std = p[p.get("model_type", pd.Series(dtype=str)).astype(str) == "standard"]
        rec = std["serving_recipe"].astype(str).value_counts().to_dict() if "serving_recipe" in p.columns else {"<column absent>": len(std)}
        mv = std["model_version"].astype(str).value_counts().to_dict() if "model_version" in p.columns else {}
        rows.append({"commit": sha[:9], "commit_ts": ts,
                     "generated_at": str(p["generated_at"].iloc[0]) if "generated_at" in p.columns and len(p) else "",
                     "git_sha": str(p["git_sha"].iloc[0]) if "git_sha" in p.columns and len(p) else "",
                     "rows": len(p), "std_rows": len(std),
                     "std_serving_recipe": ";".join(f"{k}={v}" for k, v in rec.items()),
                     "std_model_version": ";".join(f"{k}={v}" for k, v in mv.items()),
                     "std_p_over25_nan": int(std["p_over25"].isna().sum()) if "p_over25" in std.columns else -1,
                     "any_fallback_or_failed": any(k in ("legacy_v9_fallback", "v9_repaired_failed") for k in rec),
                     "std_serving_no_form_true": int(std["serving_no_form"].astype(str).str.lower().eq("true").sum())
                     if "serving_no_form" in std.columns else -1})
    df = pd.DataFrame(rows).sort_values("commit_ts")
    df.to_csv(DATA / "switch_boards.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    yml = git("log", "--since=2026-10-20", "--format=%h %cI %s", AUDITED_COMMIT, "--",
              ".github/workflows/predict.yml")
    print("\npredict.yml commits since 2026-10-20:\n" + yml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
