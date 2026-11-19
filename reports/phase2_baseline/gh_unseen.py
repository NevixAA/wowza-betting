"""For each workflow, latest run on any branch (1 call per workflow). Read only."""
import json, os, subprocess
GH = os.path.expandvars(r"%LOCALAPPDATA%\Programs\gh\bin\gh.exe")
for r in ["wowza-betting", "wowzaV9-Pro", "wowza_v11", "wowza-dashboard", "wowza-exec"]:
    wf = json.loads(subprocess.run([GH, "api", f"repos/NevixAA/{r}/actions/workflows", "--paginate", "-q", ".workflows"],
                                   capture_output=True, text=True, encoding="utf-8").stdout or "[]")
    for w in wf:
        q = subprocess.run([GH, "api", f"repos/NevixAA/{r}/actions/workflows/{w['id']}/runs?branch=main&per_page=5",
                            "-q", "[.workflow_runs[] | {c: .conclusion, s: .status, t: .created_at}]"],
                           capture_output=True, text=True, encoding="utf-8").stdout or "[]"
        runs = json.loads(q)
        done = [x for x in runs if x["s"] == "completed"]
        last = done[0] if done else None
        flag = "" if last and last["c"] in ("success", "skipped") else "  <--"
        print(f"{r:16} {w['state']:20} {w['name'][:45]:45} {(last or {}).get('c')!s:10} {(last or {}).get('t')}{flag}")
