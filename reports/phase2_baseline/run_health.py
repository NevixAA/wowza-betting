"""Run system_health.build_artifact + data_truth offline in a worktree; write JSON to scratchpad only.
usage: python run_health.py <worktree> <pro_worktree> <out.json>"""
import json, os, sys, time
from datetime import datetime, timezone

wt, pro, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["PRO_LOCAL"] = pro
os.chdir(wt)
sys.path.insert(0, wt)
now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
t = time.time()
from src import system_health as SH
art = SH.build_artifact(now, prior=SH._json(SH.HEALTH_FILE))
res = {"overall": art["overall"], "counts": art["counts"],
       "checks": [{k: c.get(k) for k in ("check", "status", "value")} for c in art["checks"]],
       "contract_violations": art.get("contract_violations"),
       "protected_tracks_modified": art.get("protected_tracks_modified"),
       "versions": art.get("versions"), "secs": round(time.time() - t, 1)}
try:
    from src.integrity import data_truth as DT
    res["gate_failures"] = DT.gate()
    res["baseline_entries"] = len(DT._baseline().get("entries", []))
    res["ratchet_ceiling"] = DT._ratchet().get("max_known_defaults")
    hits = DT.scan_defaults()
    res["default_scan_hits"] = len(hits)
    known = {(e["file"], e["code"]) for e in DT._baseline().get("entries", [])}
    res["new_defaults"] = [h for h in hits if (h["file"], h["code"]) not in known][:20]
except Exception as e:
    res["dt_error"] = f"{type(e).__name__}: {e}"
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({"overall": res["overall"], "counts": res["counts"], "secs": res["secs"]}))
