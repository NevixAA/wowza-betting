"""OddsAPI quota health from GitHub Actions run logs (R2-C finding 23).

    GH_TOKEN=... python scripts/integrity_oddsapi_quota.py [--hours 24] [--max-runs 200]

Writes output/oddsapi_quota_health.json and prints its one-line health check. Needs a token with
actions:read on the repo (the workflow's GITHUB_TOKEN). Without one the artifact is written with
status UNKNOWN — an unmeasured quota is never reported healthy. The token is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--max-runs", type=int, default=200)
    ap.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "NevixAA/wowza-betting"))
    ap.add_argument("--allowance", type=int, default=int(os.getenv("ODDSAPI_MONTHLY_ALLOWANCE", "100000")))
    ap.add_argument("--out", default="output/oddsapi_quota_health.json")
    a = ap.parse_args(argv)
    from src.integrity import oddsapi_quota as Q
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    if not token:
        doc = {"artifact": "oddsapi_quota_health", "rule_version": Q.QUOTA_RULE_VERSION,
               "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "status": "UNKNOWN", "reasons": ["no GitHub token: run logs not readable"],
               "health_check_line": "OddsAPI quota UNKNOWN: logs not readable"}
    else:
        try:
            runs, listed, fails = Q.fetch_runs(a.repo, token, a.hours, max_runs=a.max_runs)
            doc = Q.build_health(runs, a.hours, monthly_allowance=a.allowance,
                                 runs_listed=listed, log_failures=fails)
        except Exception as e:                                     # noqa: BLE001
            doc = {"artifact": "oddsapi_quota_health", "rule_version": Q.QUOTA_RULE_VERSION,
                   "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "status": "UNKNOWN", "reasons": [f"INFRA_FAILURE reading logs: {type(e).__name__}"],
                   "health_check_line": "OddsAPI quota UNKNOWN: log read failed (infra)"}
    p = ROOT / a.out
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(doc["health_check_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
