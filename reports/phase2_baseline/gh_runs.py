"""Latest completed run per workflow per repo (main and data-truth-100), via gh CLI. Read only."""
import json, os, subprocess, sys
from collections import Counter
GH = os.path.expandvars(r"%LOCALAPPDATA%\Programs\gh\bin\gh.exe")
REPOS = ["wowza-betting", "wowzaV9-Pro", "wowza_v11", "wowza-dashboard", "wowza-exec"]
res = {}
for r in REPOS:
    res[r] = {}
    wf = json.loads(subprocess.run([GH, "api", f"repos/NevixAA/{r}/actions/workflows", "--paginate", "-q", ".workflows"],
                                   capture_output=True, text=True, encoding="utf-8").stdout or "[]")
    res[r]["_n_workflows"] = len(wf)
    for br in ("main", "data-truth-100"):
        p = subprocess.run([GH, "run", "list", "-R", f"NevixAA/{r}", "--branch", br, "--limit", "300",
                            "--json", "workflowName,conclusion,status,createdAt,headSha,event"],
                           capture_output=True, text=True, encoding="utf-8")
        runs = json.loads(p.stdout or "[]")
        latest = {}
        for x in runs:  # newest first
            if x["status"] != "completed":
                continue
            latest.setdefault(x["workflowName"], x)
        res[r][br] = {"runs_seen": len(runs), "oldest_seen": runs[-1]["createdAt"] if runs else None,
                      "latest_counts": dict(Counter(v["conclusion"] for v in latest.values())),
                      "not_success": {k: [v["conclusion"], v["createdAt"], v["headSha"][:8]] for k, v in latest.items()
                                      if v["conclusion"] not in ("success", "skipped")},
                      "workflows_with_runs": len(latest)}
        if p.returncode:
            res[r][br]["error"] = p.stderr[:200]
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
